r"""Experimentkern: Datensatz erzeugen, Δt-Vorstudie, Lernkurve über mehrere Seeds.

Dieses Modul enthält die Orchestrierung, die in Schritt 5 zum vollen
Evaluationsprotokoll ausgebaut wird. Hier steht der minimale Kern, der Schritt 3 und 4
abschließt: eine Funktion, die aus (Prozesskonfiguration, Datensatzspezifikation, Seed)
einen Datensatz erzeugt, und eine, die daraus eine Lernkurve macht.

Validierungsbudget -- getroffene Entscheidung
--------------------------------------------
Die Frage war, ob :math:`n` auf der x-Achse "Trainingsläufe" oder "Gesamtbudget" heißt.
Umgesetzt ist :class:`ValidationPolicy` mit zwei Betriebsarten und der Voreinstellung
``FIXED_POOL``:

* ``FIXED_POOL`` -- fester Validierungs-Pool, über alle Stützstellen identisch. Alle
  Architekturen bekommen dieselbe Behandlung, der Vergleich ist fair. Protokolliert wird
  zusätzlich ``n_budget = n_train + n_val``, damit die Aussage "so viele Läufe wurden
  verbraucht" nicht verloren geht.
* ``NONE`` -- kein Validierungssplit, festes Epochenbudget statt Early Stopping. Diese
  Variante liefert die Zahl, die im M2-Bericht als Trainingsdatenbedarf stehen darf: mit
  15 zusätzlichen Validierungsläufen ist eine Aussage über 8 Trainingsläufe irreführend.

Beide Betriebsarten sind zu fahren, ``FIXED_POOL`` für den Architekturvergleich und
``NONE`` für die Datenbedarfs-Kennzahl. Die Differenz zwischen beiden Kurven ist selbst
ein Ergebnis -- sie quantifiziert, wie viel Early Stopping an den kleinen Stützstellen
wert ist.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..data.splits import (
    DEFAULT_LEARNING_CURVE_SIZES,
    IndexBlockSplitter,
    RunSplit,
    SplitName,
    split_schedule,
)
from ..data.windows import FutureActions, TrajectoryWindowDataset, WindowConfig
from ..models.base import SequenceModel
from ..simulation.base import Trajectory
from ..simulation.config import (
    MeasurementConfig,
    PlaceholderConfigWarning,
    ProcessConfig,
)
from ..simulation.excitation import ExcitationConfig, ExcitationSchedule, MixedExcitationDesign
from ..simulation.greybox import GreyBoxFluidBedGranulator
from ..utils.seeding import SeedBundle, VarianceSource
from ..utils.tracking import ExperimentLogger, RunRecord, environment_fingerprint
from .evaluator import EvaluationResult, RolloutEvaluator

__all__ = [
    "DataSpec",
    "GeneratedData",
    "ModelFactory",
    "ValidationPolicy",
    "build_datasets",
    "evaluate_point",
    "generate_data",
    "run_learning_curve",
    "run_sample_time_study",
    "summarise_curve",
]

ModelFactory = Callable[[], SequenceModel]


def _measurement_digest(measurement: MeasurementConfig) -> str:
    """Kurzer Fingerprint des Sensormodells."""
    payload = ":".join(
        f"{key}={value}" for key, value in sorted(dataclasses.asdict(measurement).items())
    )
    return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()


class ValidationPolicy(str, Enum):
    """Umgang mit dem Validierungssplit, siehe Modul-Docstring."""

    FIXED_POOL = "fixed_pool"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Spezifikation eines Datensatzes.

    Attributes
    ----------
    sample_time_s
        Abtastzeit der ML-Pipeline. Versuchsparameter erster Ordnung, kein Detail.
    run_duration_s
        Dauer je Lauf. 28 800 s entsprechen den 8 h aus ``02_prozess_setup.md``.
    n_train_max, n_val, n_test
        Blockgrößen. Zweierpotenzen, damit Trainings-Präfix und Blöcke auf
        Sobol-Balance-Grenzen liegen.
    history_length, horizon, stride, target_names
        Fensterparameter. ``target_names`` default reduziert auf zwei Zielgrößen -- siehe
        Docstring des ``target_names``-Feldes.
    constant_run_fraction
        Anteil Läufe mit konstanten Stellgrößen im erzeugten Pool, siehe
        ``ExcitationConfig.constant_run_fraction``. Default deckt sich mit dem dortigen
        Default (0.10) -- unverändertes Verhalten, bis ein Aufrufer bewusst abweicht.
        Bei 1.0 besteht der gesamte Pool aus Konstantläufen: die reale Betriebsweise der
        Referenzanlage, als eigene Auswertungsmenge, nicht zum Trainieren gedacht.
    """

    sample_time_s: float = 25.0
    run_duration_s: float = 28_800.0
    n_train_max: int = 64
    n_val: int = 32
    n_test: int = 32
    history_length: int = 32
    horizon: int = 24
    stride: int = 8
    target_names: tuple[str, ...] = (
        "energy_target",
        "product_moisture_kg_per_kg",
    )
    """``d50_um`` als Trainingsziel entfernt (Beobachtbarkeit; nicht reproduzierbar erklärte
    frühere Begründung, siehe unten). Bleibt Zustand im Simulator (``GreyBoxFluidBedGranulator``
    führt es weiter als vierte Größe in ``outputs``/``diagnostics['d50_m']``) und Eingang der
    Fluidisierungsmargen (``fluidization_margin_lower/upper_m_per_s``) -- dort unverändert.

    Grund, mit an dieser Sitzung reproduzierten Zahlen (nicht an der früheren "84 %
    Rauschanteil"-Kennzahl oder an R²_stat=0.72 ohne Streuung der Anfangszustände -- beide
    ausdrücklich als nicht reproduzierbar zu kennzeichnen, nicht zu löschen, siehe
    ``docs/04_stand_ap1.md``):

    * FIR-R²_holdout (Stellgrößen-Historie, kein Zielgrößen-Lag) ist für ``d50_um``
      durchgehend negativ und fällt mit wachsendem Lag: -0.042 bei k=1 bis -0.123 bei
      k=43 (``scripts/probe_e2_e3_e4_lag_depth.py``,
      ``artifacts/runs/probe_e2_e3_e4_lag_depth.json``) -- schlechter als die eigene
      Mittelwertvorhersage, und die Richtung ist falsch (mehr Historie hilft nicht).
    * Über zehn Trainings-Seeds liegt die Seed-Spannweite von LSTM bei h=1 auf dem
      Konstantlauf-Holdout bei ≈0 (0.1045 bis 0.1045,
      ``artifacts/runs/probe_f3_f4_arch_holdout_inference.json``) -- das Modell
      kollabiert auf den (seed-unabhängigen) Mittelwertsprädiktor, nicht auf eine
      gelernte Abbildung.
    * Der frühere, scheinbare Maskierungseffekt bei ``d50_um`` ist überwiegend ein
      Nennerartefakt der laufweisen NRMSE-Normierung, kein Fehlereffekt
      (``scripts/probe_e0_normalisation_artifact.py``): der Nenner streut zwischen
      verletzenden und sauberen Läufen um Faktor ~2 (Median 49.96 vs. 26.36 µm, GRU h1),
      während der unnormierte RMSE zwischen den Gruppen ähnlich bleibt oder für
      verletzende Läufe sogar kleiner ist.
    """
    observed_target_names: tuple[str, ...] | None = None
    """Inline gemessene Zielgrößen; ``None`` heißt alle. Siehe ``WindowConfig``."""
    constant_run_fraction: float = 0.10
    future_actions: FutureActions = "realised"
    """Which action series enters ``future_exogenous``, see ``WindowConfig.future_actions``
    (review D1). ``"realised"`` is the behaviour up to Befund 81 and stays the default so that
    existing scripts reproduce their data; **new data sets are to be generated with
    ``"planned"``** -- the planned set points, not the ones limited by the fluidisation
    bounds. Only acts when the process config has ``numerics.enforce_gas_velocity_projection``
    on; otherwise both series are identical. Enters the data fingerprints only when not
    ``"realised"``."""

    @property
    def pool_size(self) -> int:
        """Anzahl zu simulierender Läufe."""
        return self.n_train_max + self.n_val + self.n_test

    @property
    def window(self) -> WindowConfig:
        """Fensterkonfiguration."""
        return WindowConfig(
            history_length=self.history_length,
            horizon=self.horizon,
            stride=self.stride,
            target_names=self.target_names,
            observed_target_names=self.observed_target_names,
            future_actions=self.future_actions,
        )

    def as_parameters(self) -> dict[str, Any]:
        """Für das Run-Log."""
        return {f"data/{key}": value for key, value in dataclasses.asdict(self).items()}


@dataclass(frozen=True, slots=True)
class GeneratedData:
    """Erzeugter Datensatz samt allem, was für die Rekonstruktion nötig ist."""

    trajectory: Trajectory
    schedule: ExcitationSchedule
    split: RunSplit
    process_config: ProcessConfig
    spec: DataSpec
    seeds: SeedBundle

    @property
    def fingerprints(self) -> dict[str, str]:
        """Die Fingerprints der Datenquelle.

        ``measurement/digest`` ist in Schritt 5 dazugekommen: sobald das Rauschniveau ein
        Versuchsfaktor ist, unterscheiden sich zwei Datensätze mit identischem
        Prozess-Fingerprint. Ohne diesen Eintrag wären die Läufe im Log nicht
        unterscheidbar.
        """
        fingerprints = {
            "process/fingerprint": self.process_config.fingerprint,
            "excitation/fingerprint": str(self.schedule.metadata.get("fingerprint", "unknown")),
            "split/digest": self.split.digest,
            "measurement/digest": _measurement_digest(self.process_config.measurement),
        }
        if self.spec.future_actions != "realised":
            # Only when set (review D1): records of data sets built before the option keep
            # exactly their keys and values.
            fingerprints["window/future_actions"] = self.spec.future_actions
        return fingerprints

    @property
    def relative_noise_std(self) -> float:
        """Rauschniveau des Sensormodells dieses Datensatzes."""
        measurement = self.process_config.measurement
        return measurement.relative_noise_std if measurement.enabled else 0.0

    def with_measurements(
        self,
        *,
        noise_seed: int | None = None,
        measurement: MeasurementConfig | None = None,
    ) -> GeneratedData:
        """Denselben Lauf-Pool mit neu gezogenem Sensorrauschen.

        **Ohne neue Integration.** Das Grey-Box-Modell ist deterministisch; ``noise_seed``
        und ``MeasurementConfig`` wirken ausschließlich im Sensormodell. Für Schritt 5
        heißt das: die Rausch-Achse der Varianzzerlegung und die Rauschniveau-Studie
        kosten je Variante nur die Filter- und Zufallsschleife, nicht den Integrator.

        Der Fallstrick, den das vermeidet: variiert man den Rausch-Seed über einen neuen
        Master-Seed, ändert sich mit ihm auch der Anregungs-Seed. Dann streut nicht das
        Rauschen, sondern der Datensatz -- und die Zerlegung misst das Gegenteil dessen,
        was sie behauptet.
        """
        seeds = (
            self.seeds
            if noise_seed is None
            else dataclasses.replace(self.seeds, noise_seed=noise_seed)
        )
        config = (
            self.process_config
            if measurement is None
            else dataclasses.replace(self.process_config, measurement=measurement)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            simulator = GreyBoxFluidBedGranulator(config)
        measurements = simulator.remeasure(self.trajectory.outputs, seed=seeds.noise_seed)
        trajectory = dataclasses.replace(
            self.trajectory,
            measurements=measurements,
            metadata={**self.trajectory.metadata, "seed": seeds.noise_seed},
        )
        return dataclasses.replace(self, trajectory=trajectory, process_config=config, seeds=seeds)


def generate_data(
    process_config: ProcessConfig,
    spec: DataSpec,
    *,
    master_seed: int | None = None,
    seeds: SeedBundle | None = None,
) -> GeneratedData:
    """Anregung bauen, simulieren, Split bilden.

    Die Anfangszustände kommen aus dem Anregungsdesign und werden über
    ``state_from_physical`` in den Zustandsraum übersetzt -- das Design muss die
    Zustandskodierung nicht kennen. Der Aufruf setzt außerdem das Sensormodell zurück.

    Genau eines von ``master_seed`` und ``seeds`` angeben. Der Master-Seed ist die
    bequeme, konservative Variante (alle drei Quellen streuen gemeinsam); ``seeds``
    erlaubt die Zerlegung nach Quelle, für die Schritt 5 gebaut ist. Ein Vorgabewert
    wäre hier falsch: er würde stillschweigend eine der beiden Betriebsarten wählen.
    """
    if (master_seed is None) == (seeds is None):
        raise ValueError("genau eines von master_seed und seeds angeben")
    if seeds is None:
        # Die Prüfung oben stellt sicher, dass master_seed hier gesetzt ist -- die
        # XNOR-Form (master_seed is None) == (seeds is None) narrowt für mypy aber nicht,
        # deshalb die explizite Fallunterscheidung statt eines Ternarys.
        assert master_seed is not None
        seeds = SeedBundle.from_master(master_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = dataclasses.replace(
            process_config,
            numerics=dataclasses.replace(process_config.numerics, sample_time_s=spec.sample_time_s),
        )
        simulator = GreyBoxFluidBedGranulator(config)

    design = MixedExcitationDesign(
        ExcitationConfig(
            run_duration_s=spec.run_duration_s,
            constant_run_fraction=spec.constant_run_fraction,
        ),
        config.actions,
        config.disturbances,
        sample_time_s=config.numerics.sample_time_s,
        geometry=config.geometry,
        material=config.material,
        operation=config.operation,
    )
    schedule = design.build(spec.pool_size, seed=seeds.data_seed)

    initial = None
    if schedule.initial_states is not None:
        columns = schedule.initial_states
        initial = simulator.state_from_physical(
            columns[:, 0], columns[:, 1], columns[:, 2], columns[:, 3]
        )
    trajectory = simulator.rollout(
        schedule.actions,
        schedule.disturbances,
        run_id=schedule.run_id,
        seed=seeds.noise_seed,
        initial_state=initial,
    )
    splitter = IndexBlockSplitter(n_train=spec.n_train_max, n_val=spec.n_val, n_test=spec.n_test)
    return GeneratedData(
        trajectory=trajectory,
        schedule=schedule,
        split=split_schedule(schedule, splitter),
        process_config=config,
        spec=spec,
        seeds=seeds,
    )


def build_datasets(
    data: GeneratedData,
    n_train: int,
    *,
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset | None, TrajectoryWindowDataset]:
    """Datensätze für eine Lernkurven-Stützstelle.

    Die Normalisierungsstatistiken werden aus dem Trainings-Split **dieser** Stützstelle
    geschätzt und an ``val`` und ``test`` weitergegeben. Werden sie einmal über den vollen
    Pool berechnet, hat das Modell an der kleinsten Stützstelle Information aus allen
    Läufen -- das hebt genau das linke Ende der Lernkurve.
    """
    split = data.split.subset(n_train)
    window = data.spec.window
    train = TrajectoryWindowDataset.from_trajectory(data.trajectory, split, SplitName.TRAIN, window)
    test = TrajectoryWindowDataset.from_trajectory(
        data.trajectory, split, SplitName.TEST, window, scalers=train.scalers
    )
    validation = (
        TrajectoryWindowDataset.from_trajectory(
            data.trajectory, split, SplitName.VAL, window, scalers=train.scalers
        )
        if policy is ValidationPolicy.FIXED_POOL and split.val
        else None
    )
    return train, validation, test


def _record(
    group: str,
    data: GeneratedData,
    model: SequenceModel,
    result: EvaluationResult,
    *,
    n_train: int,
    n_val: int,
    policy: ValidationPolicy,
    scaler_digest: str,
    training_elapsed_s: float,
    extra: Mapping[str, Any] | None = None,
) -> RunRecord:
    metrics = result.as_flat_metrics()
    parameters: dict[str, Any] = {
        **data.spec.as_parameters(),
        **data.fingerprints,
        "scaler/digest": scaler_digest,
        **data.seeds.as_parameters(),
        "measurement/relative_noise_std": data.relative_noise_std,
        "n_train": n_train,
        "n_val": n_val,
        "n_budget": n_train + n_val,
        "validation_policy": policy.value,
        **{f"model/{key}": value for key, value in model.describe().items()},
        **{f"training/{key}": value for key, value in model.training_diagnostics().items()},
        "training/elapsed_s": training_elapsed_s,
        **environment_fingerprint(),
    }
    if extra is not None:
        parameters.update(extra)
    return RunRecord(
        group=group,
        params=parameters,
        metrics=metrics,
        tags={"model": model.name, "group": group},
    )


def evaluate_point(
    data: GeneratedData,
    n_train: int,
    factories: Mapping[str, ModelFactory],
    *,
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
    evaluator: RolloutEvaluator | None = None,
    logger: ExperimentLogger | None = None,
    group: str = "learning_curve",
    extra: Mapping[str, Any] | None = None,
) -> list[RunRecord]:
    """Eine Stützstelle: Datensätze bauen, alle Modelle anpassen, auswerten, loggen.

    Der innerste Block aller Studien von Schritt 5. Bewusst herausgezogen und nicht in
    jeder Studie neu geschrieben: Fensterung, Skalierer-Weitergabe und die Zuordnung
    ``model_seed`` → ``fit`` sind genau die Stellen, an denen sich ein stiller Unterschied
    zwischen zwei Treibern einschleicht -- und ein Vergleich zwischen zwei Studien wäre
    dann nicht mehr zulässig.
    """
    protocol = evaluator if evaluator is not None else RolloutEvaluator()
    train, validation, test = build_datasets(data, n_train, policy=policy)
    n_val = 0 if validation is None else len(data.split.val)
    records: list[RunRecord] = []
    for label, factory in factories.items():
        model = factory()
        fit_start = time.perf_counter()
        model.fit(train, seed=data.seeds.model_seed, validation=validation)
        training_elapsed_s = time.perf_counter() - fit_start
        result = protocol.evaluate(model, test)
        record = _record(
            group,
            data,
            model,
            result,
            n_train=n_train,
            n_val=n_val,
            policy=policy,
            scaler_digest=train.target_scaler.digest,
            training_elapsed_s=training_elapsed_s,
            extra={"model/label": label, **(dict(extra) if extra else {})},
        )
        records.append(record)
        if logger is not None:
            logger.log(record)
    return records


def run_learning_curve(
    process_config: ProcessConfig,
    factories: Mapping[str, ModelFactory],
    *,
    spec: DataSpec | None = None,
    sizes: Sequence[int] = DEFAULT_LEARNING_CURVE_SIZES,
    master_seeds: Sequence[int] = (1, 2, 3),
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
    evaluator: RolloutEvaluator | None = None,
    logger: ExperimentLogger | None = None,
    group: str = "learning_curve",
) -> list[RunRecord]:
    """Lernkurve: Fehler über Anzahl Trainingsläufe, mehrere Seeds, mehrere Modelle.

    Ein Master-Seed variiert alle drei Zufallsquellen gemeinsam (Datensatz-Realisierung,
    Sensorrauschen, Modellinitialisierung). Das ist die konservative Wahl: sie erzeugt die
    größte, also ehrlichste Streuung, und bleibt die Voreinstellung für die berichtete
    Kurve. Die Zerlegung, *welche* Quelle streut, steht in
    :func:`twinlink_ml.evaluation.protocol.run_variance_decomposition`.
    """
    specification = spec if spec is not None else DataSpec()
    records: list[RunRecord] = []

    for master_seed in master_seeds:
        data = generate_data(process_config, specification, master_seed=master_seed)
        for n_train in sizes:
            if n_train > specification.n_train_max:
                continue
            records.extend(
                evaluate_point(
                    data,
                    n_train,
                    factories,
                    policy=policy,
                    evaluator=evaluator,
                    logger=logger,
                    group=group,
                    extra={
                        "master_seed": master_seed,
                        "variance/source": VarianceSource.JOINT.value,
                        "variance/replicate": int(master_seed),
                    },
                )
            )
    return records


def run_sample_time_study(
    process_config: ProcessConfig,
    factories: Mapping[str, ModelFactory],
    *,
    sample_times_s: Sequence[float] = (5.0, 25.0, 60.0),
    base_spec: DataSpec | None = None,
    master_seed: int = 1,
    logger: ExperimentLogger | None = None,
    group: str = "sample_time_study",
) -> list[RunRecord]:
    r"""Δt-Vorstudie mit **konstantem Zeithorizont** je Fenster.

    Der Fallstrick: hält man ``history_length`` und ``horizon`` in Abtastschritten fest,
    ändert sich mit :math:`\Delta t` gleichzeitig der abgedeckte Zeitraum. Bei
    :math:`\Delta t = 5` s sieht ein Fenster mit 32 Schritten 160 s, bei 60 s dagegen
    1920 s -- ein Vergleich der NRMSE-Werte mischt dann Abtastzeit und Modellhorizont.
    Diese Funktion hält deshalb die **physikalischen** Fenster- und Horizontlängen
    konstant und rechnet die Schrittzahlen um.

    Referenz ist ``base_spec`` bei :math:`\Delta t = 25` s: 800 s Historie, 600 s Horizont.
    """
    reference = base_spec if base_spec is not None else DataSpec()
    history_s = reference.history_length * reference.sample_time_s
    horizon_s = reference.horizon * reference.sample_time_s
    stride_s = reference.stride * reference.sample_time_s
    records: list[RunRecord] = []

    for sample_time in sample_times_s:
        spec = dataclasses.replace(
            reference,
            sample_time_s=sample_time,
            history_length=max(2, round(history_s / sample_time)),
            horizon=max(2, round(horizon_s / sample_time)),
            stride=max(1, round(stride_s / sample_time)),
        )
        data = generate_data(process_config, spec, master_seed=master_seed)
        records.extend(
            evaluate_point(
                data,
                spec.n_train_max,
                factories,
                evaluator=RolloutEvaluator(horizons=(1, spec.horizon)),
                logger=logger,
                group=group,
                extra={
                    "master_seed": master_seed,
                    "study/history_s": history_s,
                    "study/horizon_s": horizon_s,
                },
            )
        )
    return records


def summarise_curve(
    records: Sequence[RunRecord], *, target: str, horizon: int, metric: str = "nrmse_run_median"
) -> dict[str, dict[int, tuple[float, float, float]]]:
    """Median und Spannweite über die Seeds, je Modell und Stützstelle.

    Rückgabe: ``{Modell: {n_train: (Median, Minimum, Maximum)}}``. Median und Spannweite
    statt Mittelwert und Standardabweichung, weil bei drei bis fünf Seeds die
    Normalverteilungsannahme nicht zu prüfen ist.
    """
    key = f"{metric}/{target}/h{horizon}"
    grouped: dict[tuple[str, int], list[float]] = {}
    for record in records:
        if key not in record.metrics:
            continue
        label = str(record.params.get("model/label", record.tags.get("model", "?")))
        n_train = int(record.params["n_train"])
        grouped.setdefault((label, n_train), []).append(float(record.metrics[key]))
    result: dict[str, dict[int, tuple[float, float, float]]] = {}
    for (label, n_train), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        result.setdefault(label, {})[n_train] = (
            float(np.median(array)),
            float(np.min(array)),
            float(np.max(array)),
        )
    return result
