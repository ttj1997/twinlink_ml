r"""Ein eigenständiges Netz je Zielgröße, statt eines gemeinsamen (Befund 26).

Warum, mit Verweis auf ``docs/04_stand_ap1.md``
------------------------------------------------
Befund 26 diagnostiziert negativen Transfer im Soft-Sensor-Betrieb:
``torch_sequence.py`` trainiert alle Zielgrößen mit einem geteilten rekurrenten Zustand
und einem einzigen Skalarverlust. Die beiden Qualitätsgrößen sind dort praktisch
unlösbar (Befund 19); ihr Gradientenrauschen verschlechtert über den geteilten Zustand
die sonst gute Lösung für ``energy_target`` -- gemessen als Faktor 14 im NRMSE
(0.3422 gemeinsam trainiert gegen 0.0238 einzeln, ``n_train = 32``, Median über drei
Seeds). Ein Fortsetzungsexperiment hat die Alternativursache -- das globale
``gradient_clip`` -- ausgeschlossen: mit einer 100-fach höheren Clip-Norm blieb der
Fehler bei 0.3574, keine Erholung Richtung 0.0238. Der geteilte Zustand selbst ist damit
die verbleibende, nicht widerlegte Ursache.

Drei Gründe für diese Umsetzung (Option 2 aus Befund 26) statt Verlustgewichtung
(Option 1) oder eines der Zwei-Phasen-Vorschläge:

1. **Verlustgewichtung dämpft nur die Gradientenmagnitude.** Eine Gewichtung ändert, wie
   stark eine Zielgröße im Skalarverlust zählt, aber nicht, dass ihr Gradient weiterhin
   durch denselben rekurrenten Zustand läuft wie der von ``energy_target``. Die
   Verschränkung in der geteilten Repräsentation -- der eigentliche Mechanismus aus
   Befund 26 -- bliebe bestehen, nur schwächer.
2. **Die klassischen Baselines sind bereits pro Zielgröße entkoppelt.** ``LinearARXModel``
   und ``RandomFeatureNARX`` lösen ``(X^T X + \lambda I)^{-1} X^T Y`` über
   :func:`~twinlink_ml.models.baselines._ridge_solve` mit einer mehrspaltigen
   Zielgrößenmatrix ``Y``. Diese Normalgleichung zerfällt exakt in unabhängige
   Regressionen je Spalte, weil ``gram = X^T X + \lambda I`` für alle Spalten identisch
   ist und der Lösungsvektor je Spalte nur von der zugehörigen Spalte der rechten Seite
   abhängt -- numerisch geprüft (Restdifferenz zur getrennten Lösung an
   Maschinenpräzision, 1.1e-16). Der bisherige Vergleich GRU/LSTM gegen ARX im
   Soft-Sensor-Betrieb war dadurch kein fairer Architekturvergleich: ARX war strukturell
   nie demselben Transferrisiko ausgesetzt wie das gemeinsame Netz. Mit dieser Klasse
   gilt dieselbe Entkopplung jetzt auch für GRU/LSTM.
3. **Keine neuen Hyperparameter.** Bei wenigen Trainingsläufen (die Lernkurve beginnt bei
   acht) ist jeder zusätzliche Hyperparameter selbst ein Freiheitsgrad, der kalibriert
   werden müsste, ohne dafür Daten übrig zu haben -- eine zentrale Projektanforderung.
   Verlustgewichtung nach Kendall et al. oder GradNorm hätten genau das gebraucht.

Was das kostet
---------------
:class:`CompositeSequenceModel` sieht pro Zielgröße nur die eigene Historie, nicht die
der anderen -- ein bewusster Kompromiss (siehe Klartext-Diskussion, die der
Implementierung voranging, in der Session-Historie zu Befund 26): eine Architektur, die
zusätzlich die Historie der jeweils anderen Zielgrößen einspeist, bräuchte entweder eine
neue Kanalart in :mod:`twinlink_ml.data.windows` mit einer Platzhalterpolitik für den
Rollout, oder eine gekoppelte Schritt-für-Schritt-Trainingsschleife, die
``torch_sequence.py`` außerhalb der Datei dupliziert. Beides vergrößert den Eingriff
erheblich und war nicht Teil dieser Entscheidung. Diese Klasse reproduziert stattdessen
exakt das im Diagnoseexperiment geprüfte Fall (b): ``target_names = (eine Zielgröße,)``
je Teilmodell.

Warum ``torch_sequence.py`` unverändert bleibt
-----------------------------------------------
Open-Closed, wie im Modul-Docstring von ``torch_sequence.py`` selbst begründet: eine
neue Kombination aus bestehenden Bausteinen darf den Trainer nicht anfassen müssen.
:class:`CompositeSequenceModel` hält ``n_targets`` unabhängige
:class:`~twinlink_ml.models.torch_sequence.TorchSequenceModel`-Instanzen und einen
generischen :class:`~twinlink_ml.data.windows.SequenceDataset`-Blick
(:class:`_SingleTargetView`), der einen mehrspaltigen Datensatz auf eine Zielgröße
reduziert. Die Reduktion ist keine Näherung: :meth:`Standardizer.fit
<twinlink_ml.data.windows.Standardizer.fit>` schätzt Mittelwert und Streuung je Kanal
unabhängig von den übrigen Kanälen, also ist ``scaler.select([i])`` bitgleich mit einer
Anpassung, die von Grund auf nur auf diesem einen Kanal geschätzt worden wäre.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch, WindowConfig
from ..simulation.backend import FloatArray
from ..utils.seeding import _derive
from .base import SequenceModel
from .torch_sequence import TorchSequenceConfig, TorchSequenceModel

__all__ = [
    "CompositeGruSequenceModel",
    "CompositeLstmSequenceModel",
    "CompositeMambaSequenceModel",
    "CompositeSequenceModel",
    "CompositeTransformerSequenceModel",
    "CompositeXLSTMSequenceModel",
]


def _submodel_seed(seed: int, target: str) -> int:
    """Teilmodell-Seed, SHA-256-abgeleitet wie :class:`~twinlink_ml.utils.seeding.SeedBundle`.

    **Nicht** ``seed + index``: ``utils/seeding.py`` begründet den Wechsel von einer
    linearen Ableitung zu SHA-256 ausdrücklich damit, dass benachbarte Master-Seeds sonst
    korrelierte Teilseeds erzeugen -- dasselbe Argument gilt hier zwischen den
    Zielgrößen-Teilmodellen. Der Zielgrößenname (nicht der Index) ist das Salt, damit der
    Seed stabil bleibt, wenn sich die Reihenfolge von ``target_names`` ändert.
    """
    return _derive(seed, "composite_submodel", target)


def _select_target(
    batch: WindowBatch, history_columns: Sequence[int], target_index: int
) -> WindowBatch:
    """``WindowBatch`` auf die Eingangskanäle und die eine Zielgröße eines Teilmodells kürzen."""
    return WindowBatch(
        history_inputs=batch.history_inputs[..., history_columns],
        future_exogenous=batch.future_exogenous,
        future_targets=batch.future_targets[..., [target_index]],
        future_targets_truth=(
            None
            if batch.future_targets_truth is None
            else batch.future_targets_truth[..., [target_index]]
        ),
        run_index=batch.run_index,
        window_start=batch.window_start,
    )


class _SingleTargetView(SequenceDataset):
    """Ein mehrspaltiger Datensatz, reduziert auf genau eine Zielgröße.

    Keine Neuberechnung aus der Trajektorie -- reine Kanalauswahl auf bereits
    standardisierten Arrays des zugrundeliegenden Datensatzes, siehe Moduldocstring.
    Funktioniert über die :class:`~twinlink_ml.data.windows.SequenceDataset`-Abstraktion,
    nicht über ``TrajectoryWindowDataset``-Interna, und bleibt damit gültig für jede
    künftige Implementierung dieser ABC.
    """

    def __init__(self, base: SequenceDataset, target_index: int) -> None:
        self._base = base
        self._target_index = target_index
        self._history_columns = [
            *range(len(base.exogenous_names)),
            base.autoregressive_index[target_index],
        ]

    @property
    def config(self) -> WindowConfig:
        name = self.target_names[0]
        observed = name in self._base.observed_target_names
        return dataclasses.replace(
            self._base.config,
            target_names=(name,),
            observed_target_names=(name,) if observed else (),
        )

    @property
    def n_windows(self) -> int:
        return self._base.n_windows

    @property
    def exogenous_names(self) -> tuple[str, ...]:
        return self._base.exogenous_names

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self._base.target_names[self._target_index],)

    @property
    def input_scaler(self) -> Any:
        return self._base.input_scaler.select(self._history_columns)

    @property
    def target_scaler(self) -> Any:
        return self._base.target_scaler.select([self._target_index])

    def batch(self, indices: Sequence[int]) -> WindowBatch:
        return _select_target(self._base.batch(indices), self._history_columns, self._target_index)


class CompositeSequenceModel(SequenceModel):
    r"""Ein :class:`TorchSequenceModel` je Zielgröße statt eines gemeinsamen Netzes.

    Siehe Moduldocstring für die Begründung (Befund 26, ``docs/04_stand_ap1.md``).

    Jedes Teilmodell bekommt einen über :func:`_submodel_seed` aus ``(seed, Zielgröße)``
    abgeleiteten Seed -- deterministisch, aber ohne die Korrelation zwischen benachbarten
    Teilseeds, die eine lineare Ableitung (``seed + i``) einführen würde (siehe
    ``utils/seeding._derive``). Wer dieselbe Vorhersage mit einzeln aufgerufenen
    ``TorchSequenceModel``-Instanzen reproduzieren will, verwendet dieselbe Funktion.

    Parameters
    ----------
    config
        Architektur und Optimierung, an jedes Teilmodell unverändert weitergereicht. Ein
        Objekt für alle Zielgrößen -- unterschiedliche Kapazität je Zielgröße wäre ein
        weiterer Freiheitsgrad, den dieses Modul bewusst nicht einführt (Grund 3 oben).
    """

    def __init__(self, config: TorchSequenceConfig | None = None) -> None:
        self._config = config if config is not None else TorchSequenceConfig()
        self._submodels: dict[str, TorchSequenceModel] = {}
        self._target_names: tuple[str, ...] = ()
        self._n_exogenous = 0

    # ------------------------------------------------------------------- Interface

    @property
    def name(self) -> str:
        """Kurzname, mit ``composite_``-Vorsatz gegen Verwechslung mit dem alten Einzelnetz."""
        return (
            f"composite_{self._config.core}_h{self._config.hidden_size}_l{self._config.num_layers}"
        )

    @property
    def config(self) -> TorchSequenceConfig:
        """Architektur- und Trainingsparameter, an jedes Teilmodell weitergereicht."""
        return self._config

    @property
    def n_parameters(self) -> int:
        """Summe der trainierbaren Parameter aller Teilmodelle."""
        return sum(model.n_parameters for model in self._submodels.values())

    @property
    def training_history(self) -> dict[str, tuple[dict[str, float], ...]]:
        """Trainingsverlauf je Zielgröße -- getrennte Netze, getrennte Kurven."""
        return {name: model.training_history for name, model in self._submodels.items()}

    def training_diagnostics(self) -> dict[str, Any]:
        """Epochen-/Gradientenschritt-Kennzahlen je Teilmodell, mit Zielgrößen-Suffix."""
        return {
            f"{key}/{target}": value
            for target, model in self._submodels.items()
            for key, value in model.training_diagnostics().items()
        }

    def describe(self) -> dict[str, Any]:
        """Konfiguration für das Tracking, dieselben ``net_*``-Schlüssel wie beim Einzelnetz.

        ``net_actual_device`` wird vom ersten Teilmodell übernommen (``dict``-Iterationsreihenfolge
        = Einfügereihenfolge, siehe :meth:`fit`) statt hier neu berechnet -- alle Teilmodelle
        teilen dieselbe ``TorchSequenceConfig`` und lösen ``device=None`` deshalb identisch auf
        (siehe :meth:`TorchSequenceModel.fit`), ein zweiter Blick auf jedes Teilnetz wäre
        redundant. ``None`` vor dem ersten :meth:`fit`-Aufruf (``self._submodels`` noch leer).
        """
        first_submodel = next(iter(self._submodels.values()), None)
        return {
            **super().describe(),
            **{f"net_{key}": value for key, value in asdict(self._config).items()},
            "composite/n_submodels": len(self._submodels),
            "net_actual_device": (
                None if first_submodel is None else first_submodel.describe()["net_actual_device"]
            ),
        }

    # --------------------------------------------------------------------- Training

    def fit(
        self,
        train: SequenceDataset,
        *,
        seed: int,
        validation: SequenceDataset | None = None,
        verbose: bool = False,
    ) -> None:
        """Ein Teilmodell je Zielgröße anpassen, unabhängig voneinander.

        Keine Fallunterscheidung nach Beobachtbarkeits-Betriebsart: ``_SingleTargetView``
        schneidet den bereits fertig gefensterten Datensatz zu, die Maskierung nicht
        beobachteter Zielgrößen ist darin schon enthalten (siehe
        :mod:`twinlink_ml.data.windows`). Dieselbe Schleife läuft also identisch in
        ``INLINE`` und ``SOFT_SENSOR``.

        ``verbose`` nur an :meth:`TorchSequenceModel.fit` durchgereicht (siehe dort) --
        kein Teil von ``TorchSequenceConfig``, ändert kein Trainingsverhalten.
        """
        self._target_names = train.target_names
        self._n_exogenous = len(train.exogenous_names)
        submodels: dict[str, TorchSequenceModel] = {}
        for index, target in enumerate(self._target_names):
            sub_train = _SingleTargetView(train, index)
            sub_validation = None if validation is None else _SingleTargetView(validation, index)
            submodel = TorchSequenceModel(self._config)
            if verbose:
                print(f"-- Teilmodell {target!r} --", flush=True)
            submodel.fit(
                sub_train,
                seed=_submodel_seed(seed, target),
                validation=sub_validation,
                verbose=verbose,
            )
            submodels[target] = submodel
        self._submodels = submodels

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Jedes Teilmodell auf seine eigene Spalte anwenden, Ergebnis in Zielreihenfolge."""
        if not self._submodels:
            raise RuntimeError("Modell ist nicht angepasst")
        predictions = []
        for index, target in enumerate(self._target_names):
            history_columns = [*range(self._n_exogenous), self._n_exogenous + index]
            sub_batch = _select_target(batch, history_columns, index)
            predictions.append(self._submodels[target].predict(sub_batch))
        return np.concatenate(predictions, axis=-1)

    # ------------------------------------------------------------------ Persistenz

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Ein State-Dict je Teilmodell -- Befund 26, ein Netz je Zielgröße."""
        if not self._submodels:
            raise RuntimeError("Modell ist nicht angepasst")
        return {target: model.state_dict() for target, model in self._submodels.items()}

    def load_checkpoint(
        self,
        *,
        submodel_state_dicts: Mapping[str, dict[str, Any]],
        target_names: Sequence[str],
        n_exogenous: int,
    ) -> None:
        """Alle Teilmodelle aus gespeicherten Zuständen rekonstruieren, ohne zu trainieren.

        Jedes Teilmodell bekommt dieselbe Architektur (``self._config``) wie beim
        ursprünglichen Training und genau ein Ziel -- derselbe Aufbau wie in
        :meth:`fit`, nur ohne den Trainingsschritt.
        """
        self._target_names = tuple(target_names)
        self._n_exogenous = n_exogenous
        submodels: dict[str, TorchSequenceModel] = {}
        for target in self._target_names:
            submodel = TorchSequenceModel(self._config)
            submodel.load_checkpoint(
                state_dict=submodel_state_dicts[target],
                target_names=(target,),
                n_exogenous=n_exogenous,
            )
            submodels[target] = submodel
        self._submodels = submodels


def CompositeGruSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """GRU-Variante -- Fabrik für die Modellregistrierung."""
    return CompositeSequenceModel(config if config is not None else TorchSequenceConfig(core="gru"))


def CompositeLstmSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """LSTM-Variante -- Fabrik für die Modellregistrierung."""
    return CompositeSequenceModel(
        config if config is not None else TorchSequenceConfig(core="lstm")
    )


def CompositeXLSTMSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """mLSTM-Variante (kein sLSTM) -- Fabrik für die Modellregistrierung.

    Derselbe Grund für ein Verbundmodell wie bei GRU/LSTM (Befund 26): der geteilte
    rekurrente Zustand schadet ``energy_target`` im Soft-Sensor-Betrieb, unabhängig vom
    Kern. Siehe :mod:`twinlink_ml.models.xlstm` für die mLSTM-Gleichungen selbst.
    """
    return CompositeSequenceModel(
        config if config is not None else TorchSequenceConfig(core="xlstm")
    )


def CompositeTransformerSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """Transformer-Variante -- Fabrik für die Modellregistrierung.

    Anders als bei GRU/LSTM/xLSTM ist ``hidden_size=64`` (Dataclass-Standardwert) für
    diesen Kern **kein** angemessener Standard: ein einzelner Encoder-Layer bei
    ``d_model=64`` hat allein rund 50000 Parameter (vier Attention-Projektionen plus
    FFN-Block, ``torch_sequence._build_transformer_network``), mehr als das gesamte
    Drei-Ziele-GRU-Verbundmodell aus Befund 28 zusammen (42819, ``docs/04_stand_ap1.md``).
    Deshalb hier ein bewusst kleiner Standard (``hidden_size=16``, ``num_heads=2``,
    ``num_layers=1``) statt des GRU/LSTM/xLSTM-Standards -- Kapazitäts-Matching gegen GRU
    ist erst nach einer echten Kostenmessung eine Option, nicht per Formel (derselbe
    Fallstrick, der beim capacity-matched xLSTM-Versuch bereits einmal >12h/Trainingspunkt
    gekostet hat).
    """
    return CompositeSequenceModel(
        config
        if config is not None
        else TorchSequenceConfig(core="transformer", hidden_size=16, num_heads=2, num_layers=1)
    )


def CompositeMambaSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> CompositeSequenceModel:
    """Mamba-Variante (selektiver State-Space-Kern) -- Fabrik für die Modellregistrierung.

    Anders als beim Transformer braucht Mamba kein reduziertes ``hidden_size`` gegen den
    Dataclass-Standard: die Zelle (:mod:`twinlink_ml.models.mamba`) hat pro Block nur eine
    Handvoll linearer Projektionen plus eine ``(d_inner, d_state)``-Matrix, keine
    quadratisch in ``hidden_size`` wachsende Attention-Projektion. Trotzdem hier bewusst
    klein gehalten (``hidden_size=16``, ``state_size=16``, ``num_layers=1``) --
    Vergleichbarkeit mit dem Transformer-Standard (Befund 37) und derselbe Grundsatz wie
    dort: Kapazität an kleinen Lernkurven-Stützstellen ist kein Startvorteil, den man sich
    ungeprüft nimmt.
    """
    return CompositeSequenceModel(
        config
        if config is not None
        else TorchSequenceConfig(core="mamba", hidden_size=16, state_size=16, num_layers=1)
    )
