r"""Rekurrente Architekturen (GRU, LSTM) hinter einem gemeinsamen Trainer.

Arbeitsplan Schritt 6. Bewusst **nach** den vier rechenfreien Modellen: das Interface ist
von Persistenz, Mittelwertvorhersage, statischer Regression, linearem ARX und dem
Zufallsmerkmal-NARX validiert, bevor es sich um ein neuronales Netz formen kann.

Warum ein Modul und nicht ein Modul je Architektur
--------------------------------------------------
Der Arbeitsplan verlangt für Schritt 6, dass eine neue Architektur **keine Änderung an
Trainings- oder Evaluationscode** erfordert (Open-Closed). Das ist nur einzuhalten, wenn
Trainingsschleife, Rollout-Semantik, Residual-Kopf und Early Stopping *einmal* existieren
und die Architektur ein austauschbarer Kern ist. ``core="gru"`` und ``core="lstm"``
unterscheiden sich hier in vier Zeilen; xLSTM, Mamba und Transformer kommen als weitere
Kerne dazu, ohne dass der Trainer angefasst wird. Wären es getrennte Module, unterschieden
sich beim dritten Modell die Trainingsschleifen -- und ein Architekturvergleich, dessen
Kandidaten verschieden trainiert werden, vergleicht die Trainingsschleifen.

Drei Entwurfsentscheidungen mit Begründung
-----------------------------------------
**Residual-Parametrisierung.** Der Kopf sagt nicht :math:`\hat y(t+h)` vorher, sondern die
Änderung gegenüber dem letzten Wert:

.. math:: \hat y(t+h) = \hat y(t+h-1) + \Delta_\theta\bigl(h_{t+h}\bigr)

Damit ist ein Netz mit Nullgewichten **exakt** die Persistenz-Baseline. Das ist bei diesem
Prozess kein kosmetischer Vorteil: Persistenz erreicht bei :math:`\Delta t = 25` s
laufweise NRMSE 0.17 auf der Produktfeuchte, ein Netz mit direkter Ausgabe muss diesen
Arbeitspunkt erst finden und verbraucht dafür an den kleinen Stützstellen der Lernkurve
genau die Daten, um die es geht.

**Kein Teacher Forcing (Voreinstellung 0.0).** Die Primärmetrik ist der
Mehrschritt-Rollout-Fehler. Wird mit den wahren Vergangenheitswerten trainiert und mit
den eigenen Vorhersagen ausgewertet, entsteht Exposure Bias -- das Modell hat den
Zustandsraum, in den es im Rollout gerät, nie gesehen. Der Preis ist teureres Training
(die Sequenz muss Schritt für Schritt durchlaufen werden) und ein schwierigeres
Optimierungsproblem. ``teacher_forcing`` ist als Parameter vorhanden, weil ein Curriculum
von 1.0 auf 0.0 bei langen Horizonten hilft, aber der Standard ist der ehrliche Fall.

**Ein Kern, kein Encoder-Decoder-Paar.** Historie und Horizont unterscheiden sich nur
darin, woher die Zielgrößen-Eingänge kommen: aus der Messung oder aus der eigenen
Vorhersage. Zwei getrennte Netze würden dieselbe Dynamik zweimal lernen müssen.

**Beobachtbarkeit braucht keinen Sonderpfad.** Im Soft-Sensor-Fall ist der Historienkanal
der nicht gemessenen Zielgröße durch den Trainingsmittelwert ersetzt (siehe
:mod:`twinlink_ml.data.windows`). Der Anker der Residual-Parametrisierung ist dann Null,
das Netz startet ohne Niveauinformation und schreibt seinen eigenen Zustand fort. Hier
liegt der einzige strukturelle Vorteil, den ein rollout-trainiertes Netz gegenüber dem
NARX hat: es sieht seine eigenen Vorhersagen im Training und kann die Rückführung
tatsächlich lernen. Ein auf Ein-Schritt-Fehler angepasstes Modell kann das nicht.

Status -- ausdrücklich
---------------------
Dieses Modul wurde inzwischen auf einer Maschine mit PyTorch ausgeführt (``04_stand_ap1.md``,
Befund 24). Die Bestandsprüfungen halten: nullinitialisierter Kopf reproduziert die
Persistenz exakt, GRU und LSTM teilen denselben Trainer, der Rollout blickt nicht in die
Zukunft. **Nicht** geprüft ist damit, ob GRU oder LSTM die Persistenz bei realistischem
Seed-Budget schlagen -- ein einzelner Trainingslauf mit einem Seed bei kleinem
``n_train`` beantwortet das nicht, siehe die Auflösungsschwelle aus Schritt 5. Diese Frage
ist Aufgabe der Lernkurve (``scripts/study_architectures.py``, mehrere Seeds), nicht eines
Unit-Tests. Bis diese Lernkurve vorliegt, ist der Architekturvergleich weiterhin auch von
``RandomFeatureNARX`` getragen, der ohne Torch läuft und über mehrere Seeds geprüft ist.
Auf Maschinen ohne PyTorch überspringen sich die sechs Tests dieses Moduls weiterhin --
das ist der vorgesehene Fallback, nicht ein Zeichen fehlender Reife.

``xLSTM`` (nur der mLSTM-Anteil, siehe :mod:`twinlink_ml.models.xlstm`) ist seit
Schritt 6, Teil 2 implementiert und hängt sich hier über genau eine zusätzliche Zeile im
``cell_type``-Dict von ``_build_network`` ein -- der Trainer selbst (``fit``, Rollout,
Verlust, Early Stopping) ist dafür unverändert geblieben, wie im Docstring oben für
jeden zusätzlichen Kern versprochen. ``Mamba`` (selektiver State-Space-Kern, siehe
:mod:`twinlink_ml.models.mamba`) ist auf demselben Weg implementiert -- ebenfalls genau
eine zusätzliche ``cell_type``-Zeile, kein Sonderzweig wie beim Transformer-Kern nötig,
weil die SSM-Zelle wie GRU/LSTM/xLSTM einen fortgeschriebenen Zustand fester Größe hat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..data.windows import SequenceDataset, WindowBatch
from ..simulation.backend import FloatArray
from .base import SequenceModel

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import Tensor, nn
else:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = None

__all__ = [
    "GruConfig",
    "GruSequenceModel",
    "LstmSequenceModel",
    "RecurrentCore",
    "TorchSequenceConfig",
    "TorchSequenceModel",
    "XLSTMSequenceModel",
]

RecurrentCore = Literal["gru", "lstm", "xlstm", "transformer", "mamba", "slstm", "xlstm_mixed"]
"""Austauschbare Kerne. Erweiterung um weitere Kerne ändert den Trainer nicht.

``"xlstm"`` löst auf :func:`twinlink_ml.models.xlstm.build_xlstm_core_class` auf --
**nur** der mLSTM-Anteil (kein sLSTM), siehe Modul-Docstring dort für die Gleichungen und
die dokumentierten Vereinfachungen.

``"slstm"`` (nur sLSTM-Schichten, rekurrente Speichermischung) und ``"xlstm_mixed"`` (mLSTM-
Schichten, zuletzt eine sLSTM-Schicht; ``num_layers >= 2``) lösen auf
:func:`twinlink_ml.models.slstm.build_slstm_core_class` bzw.
:func:`twinlink_ml.models.slstm.build_xlstm_mixed_core_class` auf; ``num_heads`` ist die
Kopfzahl der sLSTM-Schichten.

``"transformer"`` ist **keine** Rekurrenz im engeren Sinn -- der Name bleibt trotzdem
stehen (kein Alias eingeführt), weil er nur intern als Typ des ``core``-Felds verwendet
wird und eine Umbenennung keinen Vertrag ändert. Er löst auf :func:`_build_transformer_network`
auf, nicht auf den ``cell_type``-Dict-Eintrag in :func:`_build_network`, weil das dortige
``forward()`` fest an das ``(input, hidden)``-Aufrufmuster von ``nn.GRU``/``nn.LSTM``/dem
mLSTM-Kern gebunden ist -- ein Aufrufmuster, das ``nn.TransformerEncoder``/``-Decoder``
nicht kennen. Der Trainer selbst (``TorchSequenceModel.fit``/``predict``/``state_dict``/
``load_checkpoint``) bleibt unverändert, weil er nur den ``forward(history,
future_exogenous, teacher, teacher_forcing) -> (batch, horizon, n_targets)``-Vertrag
kennt, nicht die Interna von ``_build_network``.

``"mamba"`` löst auf :func:`twinlink_ml.models.mamba.build_mamba_core_class` auf -- anders
als ``"transformer"`` **über** den ``cell_type``-Dict-Eintrag in :func:`_build_network`,
weil eine selektive State-Space-Zelle wie GRU/LSTM/xLSTM einen fortgeschriebenen Zustand
fester Größe hat und damit exakt in das ``(input, hidden) -> (output, hidden)``-
Aufrufmuster passt -- kein Sonderzweig wie bei Transformer nötig. Siehe
:mod:`twinlink_ml.models.mamba` für die Rekursionsgleichungen und die O(1)-pro-Schritt-
Eigenschaft, die der Grund für diese Architektur ist (Befund 37, ``docs/04_stand_ap1.md``:
der Transformer-Decoder-Loop ist strukturell O(H²), Mamba soll das genau vermeiden).
"""


def _require_torch() -> Any:
    """PyTorch nachladen; verständlicher Fehler statt ``ModuleNotFoundError``."""
    try:
        import torch as torch_module
    except ImportError as error:  # pragma: no cover - abhängig von der Umgebung
        raise ImportError(
            "TorchSequenceModel braucht PyTorch: pip install 'twinlink-ml[torch]'"
        ) from error
    return torch_module


@dataclass(frozen=True, slots=True)
class TorchSequenceConfig:
    """Architektur- und Trainingsparameter.

    Attributes
    ----------
    core
        Kern: ``"gru"``, ``"lstm"``, ``"xlstm"`` (nur mLSTM, siehe
        :mod:`twinlink_ml.models.xlstm`) oder ``"transformer"``. Alles andere --
        Residual-Kopf, Rollout, Verlustgewichtung, Early Stopping -- ist kernunabhängig.
    hidden_size, num_layers
        Größe des GRU/LSTM/xLSTM bzw. ``d_model``/Layerzahl des Transformer-Encoders und
        -Decoders. An den kleinen Stützstellen der Lernkurve ist die Kapazität nicht der
        begrenzende Faktor; klein anfangen. **Bei ``core="transformer"`` ist der
        Dataclass-Standard 64 kein sinnvoller Ausgangspunkt** -- ein einzelner
        Encoder-Layer bei ``d_model=64`` hat allein rund 50000 Parameter (vier
        Attention-Projektionen plus FFN-Block), mehr als das gesamte
        Drei-Ziele-GRU-Verbundmodell aus Befund 28 zusammen (42819). Muss bei
        ``core="transformer"`` gerade sein (sinusoidale Positionskodierung).
    num_heads
        Nur bei ``core="transformer"`` (Attention-Köpfe) und ``core="slstm"``/
        ``"xlstm_mixed"`` (Köpfe der sLSTM-Schichten) verwendet. Muss ``hidden_size``
        teilen. Bei den übrigen Kernen wirkungslos.
    state_size
        Nur bei ``core="mamba"`` verwendet: Dimension des SSM-Zustands je Kanal
        (:math:`d_{state}` in :mod:`twinlink_ml.models.mamba`), unabhängig von
        ``hidden_size`` (:math:`d_{model}`). Referenz-Standard 16. Bei den übrigen Kernen
        wirkungslos -- exakt nach dem Muster von ``num_heads``.
    dropout
        Nur wirksam bei ``num_layers > 1`` (GRU/LSTM/xLSTM) bzw. immer wirksam bei
        ``core="transformer"`` (PyTorchs ``TransformerEncoderLayer``/``-DecoderLayer``
        wenden Dropout unabhängig von der Layerzahl an).
    learning_rate, weight_decay, max_epochs, batch_size
        Optimierung (AdamW). ``weight_decay`` gilt nicht ausnahmslos fuer jeden Parameter:
        ``TorchSequenceModel._param_groups`` setzt ihn auf 0 fuer Parameter mit dem
        Attribut ``_no_weight_decay=True`` -- aktuell nur ``A_log``/``D`` bei
        ``core="mamba"`` (siehe :mod:`twinlink_ml.models.mamba`, Referenzkonvention aus
        ``state-spaces/mamba``). Fuer alle anderen Kerne wirkungslos, da dort kein
        Parameter so markiert ist -- bitgleich zum vorherigen einheitlichen AdamW-Aufruf.
    patience
        Early Stopping auf dem Validierungs-Rollout-Verlust. ``0`` schaltet es ab und
        trainiert das feste Epochenbudget -- die Variante für die Kennzahl zum
        Trainingsdatenbedarf, die keine zusätzlichen Validierungsläufe verbrauchen darf.
    teacher_forcing
        Anteil der Horizontschritte, in denen der wahre statt des vorhergesagten Wertes
        eingespeist wird. Voreinstellung 0.0, siehe Modul-Docstring.
    horizon_weight_decay
        Exponentielle Gewichtung des Verlusts über den Horizont. ``1.0`` gewichtet alle
        Schritte gleich; Werte < 1 betonen kurze Horizonte. Bei 1.0 dominieren die späten
        Schritte den Gradienten, weil ihr Fehler größer ist.
    gradient_clip
        Norm-Clipping. Bei Rollout-Training über 24 Schritte ohne Teacher Forcing sind
        Gradientenexplosionen der Regelfall, nicht die Ausnahme.
    device
        ``"cpu"``, ``"cuda"`` oder ``None`` für automatische Wahl.
    """

    core: RecurrentCore = "gru"
    hidden_size: int = 64
    num_layers: int = 1
    num_heads: int = 2
    state_size: int = 16
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    batch_size: int = 256
    patience: int = 20
    teacher_forcing: float = 0.0
    horizon_weight_decay: float = 1.0
    gradient_clip: float = 1.0
    device: str | None = None

    def __post_init__(self) -> None:
        if self.core not in (
            "gru",
            "lstm",
            "xlstm",
            "transformer",
            "mamba",
            "slstm",
            "xlstm_mixed",
        ):
            raise ValueError(
                "unbekannter Kern "
                f"{self.core!r}, erlaubt: gru, lstm, xlstm, transformer, mamba, slstm, "
                "xlstm_mixed"
            )
        if self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("hidden_size und num_layers müssen >= 1 sein")
        if not 0.0 <= self.teacher_forcing <= 1.0:
            raise ValueError("teacher_forcing muss in [0, 1] liegen")
        if not 0.0 < self.horizon_weight_decay <= 1.0:
            raise ValueError("horizon_weight_decay muss in (0, 1] liegen")
        if self.core == "transformer":
            if self.num_heads < 1:
                raise ValueError("num_heads muss >= 1 sein")
            if self.hidden_size % 2 != 0:
                raise ValueError(
                    "hidden_size muss bei core='transformer' gerade sein "
                    "(sinusoidale Positionskodierung)"
                )
            if self.hidden_size % self.num_heads != 0:
                raise ValueError(
                    "hidden_size muss bei core='transformer' durch num_heads teilbar sein"
                )
        if self.core == "mamba" and self.state_size < 1:
            raise ValueError("state_size muss bei core='mamba' >= 1 sein")
        if self.core in ("slstm", "xlstm_mixed") and (
            self.num_heads < 1 or self.hidden_size % self.num_heads != 0
        ):
            raise ValueError(
                f"hidden_size muss bei core={self.core!r} durch num_heads teilbar sein"
            )
        if self.core == "xlstm_mixed" and self.num_layers < 2:
            raise ValueError("num_layers muss bei core='xlstm_mixed' >= 2 sein")


class TorchSequenceModel(SequenceModel):
    """Rekurrentes Netz mit autoregressivem Rollout und Residual-Kopf.

    Parameters
    ----------
    config
        Kern, Architektur und Optimierung.
    """

    def __init__(self, config: TorchSequenceConfig | None = None) -> None:
        self._config = config if config is not None else TorchSequenceConfig()
        self._torch = _require_torch()
        self._network: Any | None = None
        self._n_exogenous = 0
        self._n_targets = 0
        self._target_names: tuple[str, ...] = ()
        self._autoregressive_index: tuple[int, ...] = ()
        self._history: list[dict[str, float]] = []
        self._best_epoch: int | None = None
        self._n_train_windows = 0

    # ------------------------------------------------------------------- Interface

    @property
    def name(self) -> str:
        """Kurzname mit Kern und Kapazitätsangabe."""
        return f"{self._config.core}_h{self._config.hidden_size}_l{self._config.num_layers}"

    @property
    def config(self) -> TorchSequenceConfig:
        """Architektur- und Trainingsparameter, für Checkpoint-Persistenz."""
        return self._config

    @property
    def n_parameters(self) -> int:
        """Anzahl trainierbarer Parameter."""
        if self._network is None:
            return 0
        return sum(p.numel() for p in self._network.parameters() if p.requires_grad)

    @property
    def training_history(self) -> tuple[dict[str, float], ...]:
        """Verlust je Epoche, für die Diagnose der Lernkurven-Punkte."""
        return tuple(self._history)

    _FLATTEN_WINDOW = 5

    def training_diagnostics(self) -> dict[str, Any]:
        """Epochen-, Gradientenschritt- und Abflachungs-Kennzahl für das Tracking.

        ``gradient_steps`` verwendet dieselbe Aufrundung wie
        :meth:`~twinlink_ml.data.windows.SequenceDataset.iter_batches` (der letzte,
        kleinere Batch zählt mit) -- eine Floor-Rechnung würde die reale Schrittzahl
        systematisch unterschätzen.

        ``train_loss_ratio_last10`` vergleicht den Mittelwert der letzten 5 Epochen
        gegen den Mittelwert der 5 Epochen davor (Epochen 6-10 vor Schluss) -- **nicht**
        Einzelpunkt gegen Einzelpunkt. Ein Zwei-Punkt-Vergleich (frühere Fassung) erwies
        sich am realen `n_train=8`-LSTM-Lauf als irreführend: `product_moisture_kg_per_kg`
        pendelte über die letzten 20 Epochen ohne Trend zwischen 0.000075 und 0.000105
        (~±20 % relative Amplitude) -- der Zwei-Punkt-Quotient (1.074) hing dort vom
        Zufall ab, auf welche Stelle der Schwankung die verglichenen Epochen gerade
        fielen, nicht vom tatsächlichen Kurvenverlauf. Der gemittelte Fensterschnitt
        federt genau dieses Epoche-zu-Epoche-Rauschen ab, ohne die Komplexität einer
        Regressionssteigung einzuführen -- für eine Ja/Nein-Diagnose (abgeflacht oder
        nicht) ausreichend. Nahe 1.0 heißt abgeflacht (Plateau), deutlich darunter heißt
        der Verlust fiel an der Abbruchstelle noch -- dieselbe Unterscheidung wie in
        Befund 38(e). Mit ``patience=20`` sind mindestens 21 Epochen garantiert, das
        volle 10-Epochen-Fenster (zwei mal 5) ist also praktisch immer verfügbar; bei
        kürzeren Verläufen (nur bei ``patience=0`` oder sehr kleinem ``max_epochs``
        möglich) verkleinert sich das Fenster entsprechend, bei weniger als zwei Epochen
        bleibt die Kennzahl neutral bei 1.0 (keine Grundlage für eine Aussage).
        """
        epochs_run = len(self._history)
        if epochs_run == 0:
            return {}
        batches_per_epoch = -(-self._n_train_windows // self._config.batch_size)
        window = min(self._FLATTEN_WINDOW, epochs_run // 2)
        if window == 0:
            ratio = 1.0
        else:
            recent = self._history[-window:]
            prior = self._history[-2 * window : -window]
            recent_mean = sum(record["train_loss"] for record in recent) / window
            prior_mean = sum(record["train_loss"] for record in prior) / window
            ratio = float(recent_mean / prior_mean) if prior_mean != 0.0 else 1.0
        return {
            "epochs_run": epochs_run,
            "gradient_steps": epochs_run * batches_per_epoch,
            "train_loss_ratio_last10": ratio,
            **self._loss_curves(),
        }

    def _loss_curves(self) -> dict[str, Any]:
        """Per-epoch training/validation loss and the epoch whose weights were kept (VH1).

        Lists of floats rounded to 7 significant digits (compact; relative error 1e-7). The
        validation list is empty when the fit ran without validation or with ``patience=0``
        (no validation loss is computed then). ``best_epoch`` is the epoch whose weights were
        restored at the end of :meth:`fit` (the last epoch that improved the validation loss by
        more than the early-stopping tolerance); without validation the weights of the last
        epoch are kept, so ``best_epoch`` is the last epoch. With early stopping,
        ``epochs_run == best_epoch + 1 + patience``.
        """
        epochs_run = len(self._history)
        best = self._best_epoch if self._best_epoch is not None else epochs_run - 1
        return {
            "loss_history_train": [float(f"{r['train_loss']:.7g}") for r in self._history],
            "loss_history_val": [
                float(f"{r['val_loss']:.7g}") for r in self._history if "val_loss" in r
            ],
            "best_epoch": best,
        }

    def describe(self) -> dict[str, Any]:
        """Konfiguration für das Tracking.

        ``net_device`` ist der KONFIGURIERTE Wunsch (``TorchSequenceConfig.device``, kann
        ``None`` sein); ``net_actual_device`` ist das zur Fit-Zeit TATSÄCHLICH verwendete
        Gerät -- ``device=None`` löst in :meth:`fit` zu ``"cuda"`` auf, falls verfügbar,
        sonst zu ``"cpu"`` (siehe dort), ohne dieses Feld wäre das bei einem CUDA-fähigen
        Torch-Build nicht mehr rekonstruierbar (Anlass: ``befund42_tbed_exhaust_sweep.jsonl``,
        Torch-Build mit CUDA-Unterstützung, kein Geräte-Feld im Log). ``None`` vor dem ersten
        :meth:`fit`-Aufruf (``self._network`` noch nicht gebaut) -- alle aktuellen
        Aufrufstellen rufen ``describe()`` erst danach auf (nach der Auswertung, die ein
        trainiertes Modell voraussetzt); dieser Fall ist defensiv abgesichert, nicht
        beobachtet, analog zu :attr:`n_parameters`.
        """
        return {
            **super().describe(),
            **{f"net_{key}": value for key, value in asdict(self._config).items()},
            "net_actual_device": (
                None if self._network is None else str(next(self._network.parameters()).device)
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
        """Trainieren, optional mit Early Stopping auf dem Validierungs-Rollout.

        ``verbose`` ist **kein** Teil von ``TorchSequenceConfig`` (also auch nicht im
        Checkpoint-Schema) -- reiner Laufzeit-Diagnoseschalter für lange Läufe ohne
        sonstige Fortschrittsausgabe, standardmäßig aus, ändert kein Trainingsverhalten.
        """
        torch_module = self._torch
        config = self._config
        self._n_exogenous = len(train.exogenous_names)
        self._n_targets = len(train.target_names)
        self._target_names = train.target_names
        self._autoregressive_index = train.autoregressive_index
        self._n_train_windows = train.n_windows

        torch_module.manual_seed(seed)
        device = torch_module.device(
            config.device
            if config.device is not None
            else ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self._network = _build_network(
            torch_module,
            n_exogenous=self._n_exogenous,
            n_targets=self._n_targets,
            config=config,
        ).to(device)
        optimizer = torch_module.optim.AdamW(
            self._param_groups(config.weight_decay),
            lr=config.learning_rate,
        )
        weights = self._horizon_weights(train.config.horizon, device)
        generator = np.random.default_rng(seed)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0
        self._history = []
        self._best_epoch = None

        for epoch in range(config.max_epochs):
            self._network.train()
            total, count = 0.0, 0
            per_target_total = np.zeros(self._n_targets, dtype=np.float64)
            for batch in train.iter_batches(config.batch_size, rng=generator):
                tensors = self._to_tensors(batch, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self._network(
                    tensors["history"],
                    tensors["future"],
                    teacher=tensors["targets"] if config.teacher_forcing > 0.0 else None,
                    teacher_forcing=config.teacher_forcing,
                )
                loss = self._weighted_mse(prediction, tensors["targets"], weights)
                loss.backward()
                torch_module.nn.utils.clip_grad_norm_(
                    self._network.parameters(), config.gradient_clip
                )
                optimizer.step()
                total += float(loss.detach()) * batch.n_windows
                count += batch.n_windows
                # Rein diagnostisch (Befund 25, 04_stand_ap1.md): dieselbe Rechnung wie
                # oben, nur vor der Mittelung über Zielgrößen aufgeschlüsselt. Auf dem
                # bereits losgelösten `prediction`-Tensor berechnet, geht nicht in
                # `loss.backward()` ein und ändert daher weder Gradienten noch Optimizer-
                # Zustand -- reine Beobachtung, kein Eingriff.
                with torch_module.no_grad():
                    per_target = self._weighted_mse_per_target(
                        prediction, tensors["targets"], weights
                    )
                per_target_total += per_target.cpu().numpy().astype(np.float64) * batch.n_windows
            record = {"epoch": float(epoch), "train_loss": total / max(count, 1)}
            for index, name in enumerate(self._target_names):
                record[f"train_loss/{name}"] = float(per_target_total[index] / max(count, 1))

            if validation is not None and config.patience > 0:
                validation_loss = self._evaluate_loss(validation, device, weights)
                record["val_loss"] = validation_loss
                if validation_loss < best_loss - 1e-6:
                    best_loss, stale = validation_loss, 0
                    self._best_epoch = epoch
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self._network.state_dict().items()
                    }
                else:
                    stale += 1
            self._history.append(record)
            if verbose:
                message = f"    epoch {epoch}: train_loss={record['train_loss']:.5f}"
                if "val_loss" in record:
                    message += f" val_loss={record['val_loss']:.5f} stale={stale}"
                print(message, flush=True)
            if config.patience > 0 and stale >= config.patience:
                break

        if best_state is not None:
            self._network.load_state_dict(best_state)

    def predict(self, batch: WindowBatch) -> FloatArray:
        """Autoregressiver Rollout, Rückgabe standardisiert wie im Interface gefordert."""
        if self._network is None:
            raise RuntimeError("Modell ist nicht angepasst")
        torch_module = self._torch
        device = next(self._network.parameters()).device
        self._network.eval()
        with torch_module.no_grad():
            tensors = self._to_tensors(batch, device)
            prediction = self._network(tensors["history"], tensors["future"])
        return np.asarray(prediction.cpu().numpy(), dtype=np.float64)

    # ------------------------------------------------------------------ Persistenz

    def state_dict(self) -> dict[str, Any]:
        """Netzgewichte, losgelöst vom Rechengraphen -- für Checkpoint-Persistenz.

        CPU-Kopien, nicht die Live-Tensoren: ein gespeichertes Dict darf sich nicht mehr
        ändern, wenn danach weitertrainiert würde (hier nicht der Fall, aber die Kopie
        macht die Methode auch dann noch korrekt, wenn sich das ändert).
        """
        if self._network is None:
            raise RuntimeError("Modell ist nicht angepasst")
        return {
            key: value.detach().cpu().clone() for key, value in self._network.state_dict().items()
        }

    def load_checkpoint(
        self, *, state_dict: dict[str, Any], target_names: Sequence[str], n_exogenous: int
    ) -> None:
        """Netz aus einem gespeicherten Zustand rekonstruieren, ohne zu trainieren.

        Baut dieselbe Architektur wie :meth:`fit` (aus ``self._config``), dann werden
        die Gewichte geladen -- kein Trainingsschritt, kein Seed nötig, die Gewichte
        stehen bereits fest. ``target_names``/``n_exogenous`` kommen vom
        Trainings-Datensatz des ursprünglichen Laufs, nicht aus dem Checkpoint-Schema
        selbst (siehe :mod:`twinlink_ml.utils.checkpoints`).
        """
        torch_module = self._torch
        self._target_names = tuple(target_names)
        self._n_targets = len(self._target_names)
        self._n_exogenous = n_exogenous
        self._autoregressive_index = tuple(n_exogenous + i for i in range(self._n_targets))
        device = torch_module.device(
            self._config.device
            if self._config.device is not None
            else ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        self._network = _build_network(
            torch_module,
            n_exogenous=self._n_exogenous,
            n_targets=self._n_targets,
            config=self._config,
        ).to(device)
        self._network.load_state_dict(state_dict)
        self._network.eval()

    # ----------------------------------------------------------------------- intern

    def _param_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Zwei Parametergruppen: ``weight_decay=0`` fuer als sensibel markierte Parameter.

        Generisch, kein Kern-Sonderfall hier: ein Parameter zaehlt als sensibel, wenn er
        ein Attribut ``_no_weight_decay=True`` traegt -- exakt die Markierung, die die
        Mamba-Referenzimplementierung (``state-spaces/mamba``, ``mamba_simple.py``) fuer
        ``A_log``/``D`` setzt (siehe :mod:`twinlink_ml.models.mamba`). GRU/LSTM/xLSTM/
        Transformer markieren nichts, fallen also vollstaendig in die zweite Gruppe --
        bitgleiches Verhalten zum vorherigen ``AdamW(self._network.parameters(), ...)``
        fuer jeden Kern ausser Mamba (siehe Tests in ``test_composite.py``, die das nach
        diesem Umbau erneut bestaetigen).
        """
        assert self._network is not None  # nur intern, direkt nach dem Netzaufbau in fit() gerufen
        no_decay = [p for p in self._network.parameters() if getattr(p, "_no_weight_decay", False)]
        decay = [p for p in self._network.parameters() if not getattr(p, "_no_weight_decay", False)]
        groups = [{"params": decay, "weight_decay": weight_decay}]
        if no_decay:  # leere params-Liste liesse AdamW mit ValueError abbrechen
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def _horizon_weights(self, horizon: int, device: Any) -> Any:
        decay = self._config.horizon_weight_decay
        weights = np.power(decay, np.arange(horizon, dtype=np.float64))
        weights = weights / weights.mean()
        return self._torch.tensor(weights, dtype=self._torch.float32, device=device)

    def _to_tensors(self, batch: WindowBatch, device: Any) -> dict[str, Any]:
        torch_module = self._torch

        def convert(array: FloatArray) -> Any:
            return torch_module.tensor(
                np.ascontiguousarray(array), dtype=torch_module.float32, device=device
            )

        return {
            "history": convert(batch.history_inputs),
            "future": convert(batch.future_exogenous),
            "targets": convert(batch.future_targets),
        }

    def _weighted_mse(self, prediction: Any, target: Any, weights: Any) -> Any:
        squared = (prediction - target) ** 2
        return (squared.mean(dim=-1) * weights[None, :]).mean()

    def _weighted_mse_per_target(self, prediction: Any, target: Any, weights: Any) -> Any:
        """Wie ``_weighted_mse``, nur vor der Mittelung über Zielgrößen aufgeschlüsselt.

        Rein diagnostisch (Befund 25, ``04_stand_ap1.md``): Rückgabe hat Form
        ``(n_targets,)`` und mittelt sich zu ``_weighted_mse`` weg, weil dieselbe
        Rechnung nur an anderer Stelle summiert. Nirgends im Rückgabepfad des Trainings
        verwendet, ausschließlich fürs Logging.
        """
        squared = (prediction - target) ** 2
        return (squared * weights[None, :, None]).mean(dim=(0, 1))

    def _evaluate_loss(self, dataset: SequenceDataset, device: Any, weights: Any) -> float:
        torch_module = self._torch
        assert self._network is not None
        self._network.eval()
        total, count = 0.0, 0
        with torch_module.no_grad():
            for batch in dataset.iter_batches(self._config.batch_size):
                tensors = self._to_tensors(batch, device)
                prediction = self._network(tensors["history"], tensors["future"])
                loss = self._weighted_mse(prediction, tensors["targets"], weights)
                total += float(loss) * batch.n_windows
                count += batch.n_windows
        return total / max(count, 1)


def _build_network(
    torch_module: Any, *, n_exogenous: int, n_targets: int, config: TorchSequenceConfig
) -> Any:
    """Erzeugt das ``nn.Module``.

    Als Fabrik und nicht als Klasse auf Modulebene, weil ``nn.Module`` zur Definitionszeit
    verfügbar sein müsste -- das Paket soll ohne PyTorch importierbar bleiben.

    ``core="transformer"`` zweigt vor dem ``cell_type``-Dict komplett ab
    (:func:`_build_transformer_network`): das ``Network.forward()`` unten ist fest an das
    ``(input, hidden)``-Aufrufmuster von ``nn.GRU``/``nn.LSTM``/dem mLSTM-Kern gebunden --
    ``nn.TransformerEncoder``/``-Decoder`` haben dieses Aufrufmuster nicht, ein einzelner
    zusätzlicher ``cell_type``-Eintrag wie bei xLSTM würde hier also nicht funktionieren.
    Der Trainer (``TorchSequenceModel.fit``/``predict``/Persistenz) bleibt trotzdem
    unverändert, weil er nur den ``forward(history, future_exogenous, teacher,
    teacher_forcing) -> (batch, horizon, n_targets)``-Vertrag kennt, nicht diese Interna.
    """
    if config.core == "transformer":
        return _build_transformer_network(
            torch_module, n_exogenous=n_exogenous, n_targets=n_targets, config=config
        )

    nn_module = torch_module.nn

    # Lokaler Import, nicht auf Modulebene: dieselbe Begründung wie bei
    # `_resolve_torch_factory` in `registry.py`, damit dieses Modul ohne Torch
    # importierbar bleibt. Nur der Dict-Eintrag ist neu -- der Rest von
    # `_build_network` und der gesamte Trainer bleiben unverändert (siehe Modul-
    # Docstring, Abschnitt zu xLSTM/Mamba).
    from .mamba import build_mamba_core_class
    from .slstm import build_slstm_core_class, build_xlstm_mixed_core_class
    from .xlstm import build_xlstm_core_class

    cell_type = {
        "gru": nn_module.GRU,
        "lstm": nn_module.LSTM,
        "xlstm": build_xlstm_core_class(torch_module),
        "mamba": build_mamba_core_class(torch_module, config.state_size),
        "slstm": build_slstm_core_class(torch_module, config.num_heads),
        "xlstm_mixed": build_xlstm_mixed_core_class(torch_module, config.num_heads),
    }[config.core]

    # Die Basisklasse ist eine Laufzeit-Attributzugriffskette (`torch_module.nn.Module`)
    # und keine statisch auflösbare Typreferenz -- genau deshalb ist dieser ganze Block
    # eine Fabrikfunktion und keine Modulebene-Klasse (siehe Docstring oben). Erst über
    # eine explizit als `type` annotierte Zwischenvariable wird `class Network(base):`
    # für mypy überhaupt als Klassenausdruck lesbar; die verbleibende Meldung (Subklasse
    # von `Any`) ist bei einer zur Laufzeit geladenen Basisklasse unvermeidbar und der
    # dafür vorgesehene Weg laut mypy-Dokumentation. Zur Laufzeit ist `base` exakt
    # `torch.nn.Module`, ausgeführt und getestet auf einer Maschine mit PyTorch (siehe
    # tests/test_experiments.py, HAS_TORCH-Gate).
    base: type = nn_module.Module

    class Network(base):  # type: ignore[misc]
        """Rekurrenter Kern über Historie und Horizont mit gemeinsamer Zelle."""

        def __init__(self) -> None:
            super().__init__()
            self.core = cell_type(
                input_size=n_exogenous + n_targets,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn_module.Linear(config.hidden_size, n_targets)
            # Nullinitialisierung des Kopfes: das untrainierte Netz ist damit exakt
            # die Persistenz-Baseline, nicht eine zufällige Abbildung.
            nn_module.init.zeros_(self.head.weight)
            nn_module.init.zeros_(self.head.bias)

        def forward(
            self,
            history: Any,
            future_exogenous: Any,
            teacher: Any | None = None,
            teacher_forcing: float = 0.0,
        ) -> Any:
            """Historie kodieren, dann Schritt für Schritt fortschreiben.

            ``hidden`` ist beim GRU ein Tensor und beim LSTM ein Paar ``(h, c)``. Es wird
            deshalb nirgends ausgepackt, sondern nur weitergegeben -- damit ist der Kern
            wirklich austauschbar und nicht nur nominell.
            """
            _, hidden = self.core(history)
            last = history[:, -1, n_exogenous:]
            horizon = future_exogenous.shape[1]
            outputs = []
            for step in range(horizon):
                step_input = torch_module.cat(
                    [future_exogenous[:, step, :], last], dim=-1
                ).unsqueeze(1)
                output, hidden = self.core(step_input, hidden)
                last = last + self.head(output[:, 0, :])
                outputs.append(last)
                if teacher is not None and teacher_forcing > 0.0 and self.training:
                    mask = torch_module.rand(last.shape[0], 1, device=last.device) < teacher_forcing
                    last = torch_module.where(mask, teacher[:, step, :], last)
            return torch_module.stack(outputs, dim=1)

    return Network()


def _sinusoidal_position_encoding(torch_module: Any, positions: Any, d_model: int) -> Any:
    """Feste (nicht gelernte) Positionskodierung nach Vaswani et al. (2017).

    Keine Lernparameter -- passt sich ohne Neukalibrierung an eine spätere Änderung von
    ``history_length``/``horizon`` an (Schritt-0-Diskussion, Punkt c: Grund 3 aus dem
    ``composite.py``-Moduldocstring -- keine neuen Hyperparameter bei kleinen
    Lernkurven-Stützstellen). ``positions`` ein 1D-Tensor beliebiger Länge; ``d_model``
    muss gerade sein (validiert in ``TorchSequenceConfig.__post_init__``).
    """
    device = positions.device
    half = d_model // 2
    index = torch_module.arange(half, device=device, dtype=torch_module.float32)
    inverse_freq = torch_module.pow(10000.0, -2.0 * index / d_model)
    angles = positions.to(torch_module.float32).unsqueeze(-1) * inverse_freq.unsqueeze(0)
    encoding = torch_module.zeros(
        positions.shape[0], d_model, device=device, dtype=torch_module.float32
    )
    encoding[:, 0::2] = torch_module.sin(angles)
    encoding[:, 1::2] = torch_module.cos(angles)
    return encoding


def _build_transformer_network(
    torch_module: Any, *, n_exogenous: int, n_targets: int, config: TorchSequenceConfig
) -> Any:
    r"""Encoder-Decoder-Transformer als Kern, mit demselben Rückgabevertrag wie die RNN-Kerne.

    **Sequenzielle Erzeugung, keine explizite kausale Maske** (Klartext-Diskussion vor der
    Implementierung): bei ``teacher_forcing=0.0`` -- der Voreinstellung, siehe Moduldocstring
    -- ist die Decoder-Eingabe an Schritt :math:`t>0` ausschließlich die eigene Vorhersage
    aus Schritt :math:`t-1`, nie der wahre Zielwert; das gilt strukturell und nicht nur durch
    eine Laufzeitprüfung, weil ``TorchSequenceModel.fit`` bei ``config.teacher_forcing == 0.0``
    bereits ``teacher=None`` übergibt (kein Tensor mit Zukunftswerten liegt dann überhaupt
    vor). Der Decoder attendiert bei Schritt ``t`` nur über die bislang tatsächlich erzeugte
    Sequenz (Länge ``t + 1``) -- es gibt zu diesem Zeitpunkt keine "zukünftigen"
    Decoder-Token, über die attendiert werden könnte, also ist die Kausalität durch die
    wachsende Sequenz selbst gegeben, nicht durch eine Dreiecksmaske. ``H = 36`` sequenzielle
    Teilaufrufe sind bei diesem ``d_model`` trivial billig (bestätigt in Schritt 1 durch die
    reale Zeitmessung, nicht durch eine Kostenformel).

    Der Encoder sieht die volle Historie **ungemaskiert** (zum Vorhersagezeitpunkt bereits
    vollständig bekannt, siehe Schritt-0-Antwort zu Punkt b) -- nur der Decoder-Anteil ist
    autoregressiv.

    Positionskodierung: sinusoidal, mit Segment-Offset (Historie: Positionen
    ``0 .. history_length - 1``; Horizont: Positionen ``history_length ..
    history_length + horizon - 1``), wie in Schritt 0 vorgeschlagen und bestätigt.
    """
    nn_module = torch_module.nn
    d_model = config.hidden_size
    n_features = n_exogenous + n_targets
    feedforward_size = 4 * d_model

    base: type = nn_module.Module

    class Network(base):  # type: ignore[misc]
        """Encoder über die Historie, sequenzieller Decoder über den Horizont."""

        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn_module.Linear(n_features, d_model)
            encoder_layer = nn_module.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dim_feedforward=feedforward_size,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn_module.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            decoder_layer = nn_module.TransformerDecoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dim_feedforward=feedforward_size,
                dropout=config.dropout,
                batch_first=True,
            )
            self.decoder = nn_module.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
            self.head = nn_module.Linear(d_model, n_targets)
            # Nullinitialisierung des Kopfes: dieselbe Residual-Äquivalenz wie bei
            # GRU/LSTM/xLSTM -- das untrainierte Netz ist exakt die Persistenz-Baseline,
            # unabhängig von der (zufälligen) Encoder-/Decoder-Initialisierung.
            nn_module.init.zeros_(self.head.weight)
            nn_module.init.zeros_(self.head.bias)

        def forward(
            self,
            history: Any,
            future_exogenous: Any,
            teacher: Any | None = None,
            teacher_forcing: float = 0.0,
        ) -> Any:
            """Historie ungemaskiert kodieren, dann Schritt für Schritt autoregressiv decodieren."""
            device = history.device
            history_length = history.shape[1]
            horizon = future_exogenous.shape[1]

            history_positions = torch_module.arange(history_length, device=device)
            encoder_embed = self.input_proj(history) + _sinusoidal_position_encoding(
                torch_module, history_positions, d_model
            )
            memory = self.encoder(encoder_embed)

            last = history[:, -1, n_exogenous:]
            decoder_embeds: list[Any] = []
            outputs = []
            for step in range(horizon):
                step_input = torch_module.cat(
                    [future_exogenous[:, step, :], last], dim=-1
                ).unsqueeze(1)
                step_position = torch_module.tensor([history_length + step], device=device)
                step_embed = self.input_proj(step_input) + _sinusoidal_position_encoding(
                    torch_module, step_position, d_model
                ).unsqueeze(0)
                decoder_embeds.append(step_embed)
                # Nur die bislang erzeugte Sequenz (Länge step + 1) -- siehe Docstring:
                # keine "zukünftigen" Token existieren an dieser Stelle, also keine
                # explizite kausale Maske nötig.
                decoder_sequence = torch_module.cat(decoder_embeds, dim=1)
                decoded = self.decoder(decoder_sequence, memory)
                last = last + self.head(decoded[:, -1, :])
                outputs.append(last)
                if teacher is not None and teacher_forcing > 0.0 and self.training:
                    mask = torch_module.rand(last.shape[0], 1, device=device) < teacher_forcing
                    last = torch_module.where(mask, teacher[:, step, :], last)
            return torch_module.stack(outputs, dim=1)

    return Network()


GruConfig = TorchSequenceConfig
"""Alter Name aus Schritt 3. Bleibt als Alias, damit vorhandene Skripte laufen."""


def GruSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> TorchSequenceModel:
    """GRU-Variante -- Fabrik für die Modellregistrierung."""
    return TorchSequenceModel(config if config is not None else TorchSequenceConfig(core="gru"))


def LstmSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> TorchSequenceModel:
    """LSTM-Variante.

    Der Vergleich GRU gegen LSTM ist bei wenigen Trainingsläufen der interessantere Teil
    des Paares: das LSTM hat mit dem Zellzustand einen zweiten, additiv fortgeschriebenen
    Speicher und damit rund ein Drittel mehr Parameter bei gleicher Breite. Bei acht
    Läufen ist Kapazität keine Ressource, sondern ein Risiko.
    """
    return TorchSequenceModel(config if config is not None else TorchSequenceConfig(core="lstm"))


def XLSTMSequenceModel(  # noqa: N802 - Fabrikname folgt dem Modell-Label
    config: TorchSequenceConfig | None = None,
) -> TorchSequenceModel:
    """mLSTM-Variante (kein sLSTM) -- Fabrik für die Modellregistrierung.

    Siehe :mod:`twinlink_ml.models.xlstm` für Gleichungen und dokumentierte
    Vereinfachungen. Anders als bei GRU/LSTM ist die Kapazität bei gleicher
    ``hidden_size`` hier **nicht** automatisch größer -- die mLSTM-Zelle hat keine
    Zustand-zu-Zustand-Gewichtsmatrix (das ist gerade der Grund, warum sie
    parallelisierbar ist), und bei der kleinen Eingangsdimension dieses Projekts
    dominiert bei GRU/LSTM genau diese Matrix den Parameterhaushalt. Siehe
    ``docs/04_stand_ap1.md`` für die gemessenen Zahlen.
    """
    return TorchSequenceModel(config if config is not None else TorchSequenceConfig(core="xlstm"))
