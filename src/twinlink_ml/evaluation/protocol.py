r"""Evaluationsprotokoll (Arbeitsplan Schritt 5): die vier Studien.

Was hier orchestriert wird
--------------------------
1. :func:`run_variance_decomposition` -- Streuung nach Quelle zerlegen
   (``data`` / ``noise`` / ``model`` / gemeinsam). Auswertung in
   :mod:`twinlink_ml.evaluation.resolution`.
2. Das Auflösungsvermögen fällt als Nebenprodukt aus (1) ab, sobald eine parameterfreie
   Baseline mitläuft. Es gibt deshalb **keine** eigene Studie dafür -- eine separate
   würde mit eigenen Seeds laufen und eine andere Schwelle liefern als die Kurve, auf
   die sie angewendet wird.
3. :func:`run_noise_study` -- Rauschniveau als expliziter Versuchsfaktor, zusammen mit
   :func:`estimate_noise_floor` für die **Sättigungshöhe**.
4. :func:`run_validation_policy_comparison` -- ``FIXED_POOL`` gegen ``NONE``.

Der Kostenhebel: eine Simulation je Datensatz-Realisierung
---------------------------------------------------------
Naiv gerechnet kostet Schritt 5 ein Vielfaches von Schritt 4: drei Achsen mal drei
Replikate mal drei Rauschniveaus mal zwei Validierungs-Betriebsarten. Tatsächlich ist das
Grey-Box-Modell deterministisch, und ``noise_seed`` wie Rauschniveau wirken
**ausschließlich** im Sensormodell. :class:`DataPoolCache` nutzt das aus: integriert wird
einmal je ``data_seed``, alles Weitere entsteht über
:meth:`~twinlink_ml.evaluation.experiments.GeneratedData.with_measurements` aus den
gespeicherten rauschfreien Ausgangsgrößen. Der Cache ist damit kein
Performance-Detail, sondern die Voraussetzung dafür, dass die Zerlegung sauber ist: er
erzwingt, dass zwei Läufe, die sich nur im Rausch-Seed unterscheiden sollen, wirklich
denselben Anregungs- und Zustandsverlauf haben.

Fallstrick, der genau dadurch vermieden wird
--------------------------------------------
Variiert man den Rausch-Seed über einen neuen Master-Seed, ändert sich mit ihm der
Anregungs-Seed. Die Achse "Rauschen" misst dann die Datensatz-Realisierung mit, und die
Zerlegung behauptet das Gegenteil dessen, was sie zeigt. Deshalb ist
:class:`~twinlink_ml.utils.seeding.SeedPlan` und nicht die Master-Seed-Liste die
Schnittstelle dieser Studien.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..data.splits import DEFAULT_LEARNING_CURVE_SIZES, SplitName
from ..data.windows import TrajectoryWindowDataset
from ..simulation.config import MeasurementConfig, ProcessConfig
from ..utils.seeding import SeedBundle, SeedPlan, VarianceSource
from ..utils.tracking import ExperimentLogger, RunRecord
from .evaluator import RolloutEvaluator, per_run_nrmse, pooled_nrmse
from .experiments import (
    DataSpec,
    GeneratedData,
    ModelFactory,
    ValidationPolicy,
    build_datasets,
    evaluate_point,
    generate_data,
)

__all__ = [
    "DataPoolCache",
    "NoiseFloor",
    "ObservationMode",
    "estimate_noise_floor",
    "measurement_at_level",
    "primary_metric",
    "run_architecture_comparison",
    "run_noise_study",
    "run_validation_policy_comparison",
    "run_variance_decomposition",
    "spec_for_mode",
    "summarise_policy_gap",
]

DEFAULT_VARIANCE_SOURCES: tuple[VarianceSource, ...] = (
    VarianceSource.DATA,
    VarianceSource.NOISE,
    VarianceSource.MODEL,
    VarianceSource.JOINT,
)

DEFAULT_NOISE_LEVELS: tuple[float, ...] = (0.0, 0.005, 0.02)
r"""Rauschniveaus der Studie: rauschfrei, Platzhalter (0.5 %), pessimistisch (2 %).

Die 0.5 % sind der Platzhalter aus ``conf/process/greybox_placeholder.yaml``; der wahre
Wert kommt aus den Anwendungspartner-Zeitreihen und ist offene Frage 5 in ``04_stand_ap1.md``. Die
Studie ersetzt die Zahl nicht, sie macht die **Empfindlichkeit** der Lernkurven gegen sie
sichtbar -- und damit die Antwort auf die Frage, wie dringend die Messung ist.
"""


def _spec_digest(spec: DataSpec) -> str:
    """Kurzer Fingerprint eines ``DataSpec``, wertbasiert.

    Dasselbe Muster wie ``_measurement_digest`` in ``experiments.py``: Feldnamen und
    -werte sortiert verketten, dann hashen. Gebraucht für die Cache-Invalidierung in
    :func:`_run_grid` -- ``spec_for_mode`` baut bei jedem Aufruf ein neues Objekt, ein
    Vergleich auf Objektidentität würde dort bei *jedem* Punkt einen neuen Cache
    auslösen, nicht nur bei echten Moduswechseln. Wertegleichheit über einen Fingerprint
    ist deshalb keine Kosmetik, sondern die Voraussetzung für korrekte Wiederverwendung.
    """
    payload = ":".join(f"{key}={value}" for key, value in sorted(dataclasses.asdict(spec).items()))
    return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()


@dataclass(slots=True)
class DataPoolCache:
    """Lauf-Pools nach ``data_seed`` zwischenspeichern, Messungen billig neu ziehen.

    Nicht als globaler Zustand, sondern als injizierbares Objekt: eine Studie bekommt
    ihren Cache übergeben und kann ihn mit anderen teilen oder nicht. Der Cache hält
    vollständige Trajektorien, also mehrere hundert Megabyte bei großen Pools --
    ``clear`` ist deshalb Teil der Schnittstelle und nicht optional.
    """

    process_config: ProcessConfig
    spec: DataSpec
    _pools: dict[int, GeneratedData] = field(default_factory=dict, init=False)
    n_simulations: int = field(default=0, init=False)
    n_remeasurements: int = field(default=0, init=False)

    def base(self, data_seed: int) -> GeneratedData:
        """Integrierter Pool zu einem ``data_seed``.

        Rausch- und Modell-Seed sind hier ohne Bedeutung: :meth:`variant` ersetzt beide.
        Sie werden auf den ``data_seed`` gesetzt, damit der Basis-Pool nicht versehentlich
        als fertiger Datensatz benutzt wird und dann für alle ``data_seed`` dasselbe
        Rauschen trägt.
        """
        if data_seed not in self._pools:
            seeds = SeedBundle(data_seed=data_seed, noise_seed=data_seed, model_seed=data_seed)
            self._pools[data_seed] = generate_data(self.process_config, self.spec, seeds=seeds)
            self.n_simulations += 1
        return self._pools[data_seed]

    def variant(
        self,
        *,
        data_seed: int,
        noise_seed: int,
        model_seed: int,
        measurement: MeasurementConfig | None = None,
    ) -> GeneratedData:
        """Pool mit vorgegebenen Seeds und Sensorkonfiguration.

        Die Integration passiert nur beim ersten Aufruf je ``data_seed``. ``noise_seed``,
        ``model_seed`` und ``measurement`` sind danach reine Nachbearbeitung.
        """
        pool = self.base(data_seed)
        variant = pool.with_measurements(noise_seed=noise_seed, measurement=measurement)
        self.n_remeasurements += 1
        return dataclasses.replace(
            variant, seeds=dataclasses.replace(variant.seeds, model_seed=model_seed)
        )

    def clear(self) -> None:
        """Speicher freigeben."""
        self._pools.clear()

    def as_metrics(self) -> dict[str, float]:
        """Aufwandskennzahlen -- gehören in den Bericht, nicht nur in die Logdatei."""
        return {
            "cost/n_simulations": float(self.n_simulations),
            "cost/n_remeasurements": float(self.n_remeasurements),
        }


@dataclass(frozen=True, slots=True)
class _StudyPoint:
    """Ein Punkt auf der variierten Achse einer Schritt-5-Studie.

    Alles, worin sich die vier ``run_*``-Funktionen unterscheiden, steckt in einem
    ``_StudyPoint`` -- der Rest ist :func:`_run_grid`.

    Attributes
    ----------
    label
        Für Logs und Fortschrittsausgaben, z. B. ``"joint"``, ``"0.02"``, ``"fixed_pool"``.
    axis_value
        Der rohe Achsenwert: ``VarianceSource``, Rauschniveau als ``float``,
        ``ValidationPolicy``, oder ein ``(ObservationMode, VarianceSource)``-Tupel, je
        nach Studie. Hooks lesen **hieraus**, nicht aus dem string-keyed ``extra`` -- ein
        Rauschniveau ist eine Zahl, kein zurückgeparster String.
    variance_source
        Welche Seed-Achse dieser Punkt für :meth:`SeedPlan.bundle` benutzt.
    spec
        Override der Basis-``DataSpec``; ``None`` heißt unverändert.
    measurement
        Override der ``MeasurementConfig``; ``None`` heißt unverändert.
    policy
        Override der ``ValidationPolicy`` für ``evaluate_point``; ``None`` heißt, der
        Funktions-Default von :func:`_run_grid` greift.
    extra
        Zusätzliche, **punktfeste** Record-Parameter -- nicht replikatabhängig, dafür
        gibt es ``extra_fn`` in :func:`_run_grid`.
    """

    label: str
    axis_value: object
    variance_source: VarianceSource = VarianceSource.JOINT
    spec: DataSpec | None = None
    measurement: MeasurementConfig | None = None
    policy: ValidationPolicy | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def _run_grid(
    process_config: ProcessConfig,
    factories: Mapping[str, ModelFactory],
    points: Sequence[_StudyPoint],
    *,
    spec: DataSpec,
    plan: SeedPlan,
    sizes: Sequence[int],
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
    evaluator: RolloutEvaluator | None = None,
    logger: ExperimentLogger | None = None,
    group: str,
    cache: DataPoolCache | None = None,
    extra_fn: Callable[[_StudyPoint, int], Mapping[str, Any]] | None = None,
    on_variant: Callable[[_StudyPoint, int, GeneratedData], None] | None = None,
    on_point: Callable[[_StudyPoint, int, int], None] | None = None,
) -> list[RunRecord]:
    r"""Kern aller vier Schritt-5-Studien: (Achsenpunkt, Replikat, ``n_train``)-Gitter.

    Gemeinsames Gerüst der vier ``run_*``-Funktionen, einmal statt vier fast identischer
    Kopien (``04_stand_ap1.md``, Befund zur Kern-Konsolidierung). Was sich zwischen ihnen
    unterscheidet, steckt in ``points`` (siehe :class:`_StudyPoint`) und drei Hooks:

    - ``extra_fn`` für replikatabhängige, aber seiteneffektfreie Record-Parameter
      (``run_variance_decomposition``\ s ``variance/master_seed``).
    - ``on_variant`` für Seiteneffekte nach dem Bau einer Datenvariante, vor der
      ``n_train``-Schleife (``run_noise_study``\ s Rauschgrenze bei Replikat 0).
    - ``on_point`` ausschließlich für Fortschrittsereignisse, unmittelbar vor
      ``evaluate_point`` (``run_architecture_comparison``\ s ``on_point``-Parameter).

    Cache-Lebenszyklus
    -------------------
    Ohne injizierten ``cache`` baut diese Funktion selbst einen -- **neu, sobald sich der
    Fingerprint der pro Punkt aufgelösten Spec ändert** (:func:`_spec_digest`, nicht
    Objektidentität oder ``!=`` auf dem ``DataSpec`` selbst: ``spec_for_mode`` baut bei
    jedem Aufruf ein neues Objekt, ein Identitätsvergleich würde bei *jedem* Punkt einen
    neuen Cache auslösen, nicht nur bei echten Moduswechseln). Bei injiziertem Cache wird
    nie selbst aufgeräumt -- das bleibt Sache des Aufrufers, wie bisher bei den drei
    Studien, die einen Cache entgegennehmen.
    """
    records: list[RunRecord] = []
    owns_cache = cache is None
    active_cache: DataPoolCache | None = cache
    active_digest: str | None = None

    for point in points:
        point_spec = point.spec if point.spec is not None else spec
        if owns_cache:
            digest = _spec_digest(point_spec)
            if digest != active_digest:
                if active_cache is not None:
                    active_cache.clear()
                active_cache = DataPoolCache(process_config, point_spec)
                active_digest = digest
        # Für mypy: durch den cache-Parameter oder den Zweig oben immer gesetzt.
        assert active_cache is not None

        for replicate in range(plan.n_replicates):
            bundle = plan.bundle(point.variance_source, replicate)
            data = active_cache.variant(
                data_seed=bundle.data_seed,
                noise_seed=bundle.noise_seed,
                model_seed=bundle.model_seed,
                measurement=point.measurement,
            )
            if on_variant is not None:
                on_variant(point, replicate, data)

            for n_train in sizes:
                if n_train > point_spec.n_train_max:
                    continue
                if on_point is not None:
                    on_point(point, replicate, n_train)
                records.extend(
                    evaluate_point(
                        data,
                        n_train,
                        factories,
                        policy=point.policy if point.policy is not None else policy,
                        evaluator=evaluator,
                        logger=logger,
                        group=group,
                        extra={
                            "variance/replicate": replicate,
                            **point.extra,
                            **(extra_fn(point, replicate) if extra_fn is not None else {}),
                        },
                    )
                )

    if owns_cache and active_cache is not None:
        active_cache.clear()
    return records


def run_variance_decomposition(
    process_config: ProcessConfig,
    factories: Mapping[str, ModelFactory],
    *,
    spec: DataSpec | None = None,
    plan: SeedPlan | None = None,
    sizes: Sequence[int] = DEFAULT_LEARNING_CURVE_SIZES,
    sources: Sequence[VarianceSource] = DEFAULT_VARIANCE_SOURCES,
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
    evaluator: RolloutEvaluator | None = None,
    logger: ExperimentLogger | None = None,
    cache: DataPoolCache | None = None,
    group: str = "variance_decomposition",
) -> list[RunRecord]:
    """Lernkurve, aber je Zufallsquelle einzeln variiert.

    Ergebnis sind Records mit den Parametern ``variance/source`` und
    ``variance/replicate``; die Auswertung macht
    :func:`twinlink_ml.evaluation.resolution.decompose_variance`.

    Replikat 0 ist auf allen Achsen dasselbe Referenz-Bundle. Es wird trotzdem je Achse
    geloggt -- die Streuung einer Achse braucht ihren Referenzpunkt, und ein
    Deduplizieren im Log würde die Auswertung von der Reihenfolge der Achsen abhängig
    machen. Gerechnet wird es nur einmal, weil der Cache greift.
    """
    specification = spec if spec is not None else DataSpec()
    seed_plan = plan if plan is not None else SeedPlan()
    points = [
        _StudyPoint(
            label=source.value,
            axis_value=source,
            variance_source=source,
            extra={"variance/source": source.value},
        )
        for source in sources
    ]

    def extra_fn(point: _StudyPoint, replicate: int) -> Mapping[str, Any]:  # noqa: ARG001
        # point ungenutzt -- gemeinsame Hook-Signatur mit on_variant/on_point.
        return {
            "variance/master_seed": seed_plan.master_seeds[replicate],
            **seed_plan.as_parameters(),
        }

    return _run_grid(
        process_config,
        factories,
        points,
        spec=specification,
        plan=seed_plan,
        sizes=sizes,
        policy=policy,
        evaluator=evaluator,
        logger=logger,
        group=group,
        cache=cache,
        extra_fn=extra_fn,
    )


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """Sättigungshöhe der Lernkurve: der Fehler eines perfekten Modells.

    Berechnet **ohne Modell**. Als Vorhersage wird der deterministische Anteil der
    Messung eingesetzt -- also die Ausgangsgröße nach der Sensordynamik, aber ohne
    Rauschen, gewonnen aus derselben Trajektorie über
    :meth:`~twinlink_ml.evaluation.experiments.GeneratedData.with_measurements` mit
    Rauschniveau Null. Besser als das kann kein Modell werden, das auf die Messung
    trainiert und gegen die Messung geprüft wird.

    Warum das nötig ist: eine Lernkurve, die bei 0.12 abflacht, ist ohne diese Zahl nicht
    als Modellgrenze *oder* Rauschgrenze lesbar. Mit ihr ist die Frage entschieden, und
    zwar vor dem ersten Architekturvergleich statt nach dem zehnten.

    Der zweite, oft übersehene Anteil: ``against_truth`` misst denselben perfekten
    Prädiktor gegen die **rauschfreie und ungefilterte** Ausgangsgröße. Diese Zahl ist
    nicht Null, weil das Sensor-PT1 (Platzhalter 10 s) eine echte Phasenverschiebung
    einbringt. Wer ``nrmse_truth`` als "Fehler gegen die Wahrheit" liest, ohne diesen
    Sockel zu kennen, hält Sensordynamik für Modellfehler.
    """

    target_names: tuple[str, ...]
    horizons: tuple[int, ...]
    relative_noise_std: float
    pooled: dict[str, dict[int, float]]
    per_run: dict[str, dict[int, tuple[float, float, float]]]
    against_truth: dict[str, dict[int, tuple[float, float, float]]]

    def as_metrics(self) -> dict[str, float]:
        """Flache Kennzahlen für das Log.

        Drei eigene Variablennamen für die drei Schleifen, absichtlich: ``pooled`` liefert
        einen einzelnen Wert je Horizont, ``per_run`` und ``against_truth`` liefern
        Median/Minimum/Maximum-Tripel. Denselben Namen wiederzuverwenden würde mypy als
        inkompatible Neuzuweisung ablehnen -- zu Recht, es sind unterschiedliche Formen.
        """
        metrics: dict[str, float] = {"measurement/relative_noise_std": self.relative_noise_std}
        for target, pooled_by_horizon in self.pooled.items():
            for horizon, value in pooled_by_horizon.items():
                metrics[f"floor_pooled/{target}/h{horizon}"] = value
        for target, run_by_horizon in self.per_run.items():
            for horizon, (median, minimum, maximum) in run_by_horizon.items():
                metrics[f"floor_run_median/{target}/h{horizon}"] = median
                metrics[f"floor_run_min/{target}/h{horizon}"] = minimum
                metrics[f"floor_run_max/{target}/h{horizon}"] = maximum
        for target, truth_by_horizon in self.against_truth.items():
            for horizon, (median, _, _) in truth_by_horizon.items():
                metrics[f"floor_lag_run_median/{target}/h{horizon}"] = median
        return metrics

    def format(self) -> str:
        """Tabelle für Run-Logs."""
        columns = "".join(f"{f'h={h}':>10}" for h in self.horizons)
        lines = [
            f"Rauschgrenze bei relative_noise_std = {self.relative_noise_std:.4f}",
            f"{'Zielgröße':<30}{columns}",
        ]
        for target in self.target_names:
            run = "".join(f"{self.per_run[target][h][0]:>10.4f}" for h in self.horizons)
            lines.append(f"{target:<30}{run}")
            lag = "".join(f"{self.against_truth[target][h][0]:>10.4f}" for h in self.horizons)
            lines.append(f"{'  davon Sensor-PT1 (vs. Truth)':<30}{lag}")
        return "\n".join(lines)


def _clean_measurement(measurement: MeasurementConfig) -> MeasurementConfig:
    """Sensormodell ohne Rauschen, Dynamik unverändert."""
    return dataclasses.replace(
        measurement, relative_noise_std=0.0, absolute_noise_std_temperature_k=0.0
    )


def measurement_at_level(base: MeasurementConfig, level: float) -> MeasurementConfig:
    """Sensorkonfiguration zu einem Rauschniveau -- **ein** Faktor, nicht zwei.

    Das Sensormodell hat zwei Rauschterme: relativ auf allen Kanälen und absolut auf der
    Temperatur. Variiert man nur den relativen, ist ein "rauschfreier" Datensatz auf dem
    Temperaturkanal weiter verrauscht, und der Versuchsfaktor wäre nicht der, der im Log
    steht. Der absolute Term wird deshalb im gleichen Verhältnis mitgezogen.
    """
    reference = base.relative_noise_std
    scale = 0.0 if reference <= 0.0 else level / reference
    return dataclasses.replace(
        base,
        relative_noise_std=level,
        absolute_noise_std_temperature_k=base.absolute_noise_std_temperature_k * scale,
    )


def estimate_noise_floor(
    data: GeneratedData,
    n_train: int,
    *,
    horizons: Sequence[int] | None = None,
    which: SplitName = SplitName.TEST,
) -> NoiseFloor:
    """Rauschgrenze auf demselben Testsplit und in derselben Normierung wie die Kurve.

    Bewusst über die Fensterung und nicht direkt auf der Zeitreihe: die Primärmetrik ist
    laufweise über **Fenster** normiert, und überlappende Fenster gewichten die Zeitachse
    anders als die Rohdaten. Eine Rauschgrenze in einer anderen Normierung als die Kurve
    wäre nicht vergleichbar -- und genau der Vergleich ist der Zweck.
    """
    train, _, test = build_datasets(data, n_train)
    clean = data.with_measurements(measurement=_clean_measurement(data.process_config.measurement))
    clean_test = TrajectoryWindowDataset.from_trajectory(
        clean.trajectory,
        data.split.subset(n_train),
        which,
        data.spec.window,
        scalers=train.scalers,
    )
    if clean_test.n_windows != test.n_windows:
        raise ValueError("Fensteranzahl weicht ab -- Split oder Fensterkonfiguration passt nicht")

    horizon_list = tuple(horizons) if horizons is not None else (1, data.spec.horizon)
    scaler = train.target_scaler
    noisy_batch = test.full_batch()
    clean_batch = clean_test.full_batch()

    prediction = scaler.inverse(clean_batch.future_targets)
    reference = scaler.inverse(noisy_batch.future_targets)
    if noisy_batch.future_targets_truth is None:
        # Sollte für einen von generate_data() erzeugten Datensatz nie eintreten -- die
        # Wahrheit wird dort immer mitgeführt. Ein None hier wäre ein struktureller
        # Fehler in der Datensatzerzeugung, kein erwarteter Fall zum Abfangen.
        raise ValueError(
            "Testdaten enthalten keine rauschfreie Referenz (future_targets_truth); "
            "die Rauschgrenze kann ohne sie nicht berechnet werden"
        )
    truth = scaler.inverse(noisy_batch.future_targets_truth)
    runs = noisy_batch.run_index
    names = test.target_names

    return NoiseFloor(
        target_names=names,
        horizons=horizon_list,
        relative_noise_std=data.relative_noise_std,
        pooled=pooled_nrmse(prediction, reference, horizon_list, names),
        per_run=per_run_nrmse(prediction, reference, runs, horizon_list, names),
        against_truth=per_run_nrmse(prediction, truth, runs, horizon_list, names),
    )


def run_noise_study(
    process_config: ProcessConfig,
    factories: Mapping[str, ModelFactory],
    *,
    spec: DataSpec | None = None,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    plan: SeedPlan | None = None,
    sizes: Sequence[int] = DEFAULT_LEARNING_CURVE_SIZES,
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
    evaluator: RolloutEvaluator | None = None,
    logger: ExperimentLogger | None = None,
    cache: DataPoolCache | None = None,
    group: str = "noise_study",
) -> tuple[list[RunRecord], dict[float, NoiseFloor]]:
    """Lernkurven über mehrere Rauschniveaus, plus die Rauschgrenze je Niveau.

    Die Seeds variieren gemeinsam (Achse ``JOINT``) -- innerhalb einer Rauschstudie ist
    die Zerlegung nach Quelle nicht die Frage, und die konservative Streuung ist die
    richtige Angabe für die Höhe der Sättigung.

    Rückgabe ist ein Paar: die Records der Modelle und die modellfreien Rauschgrenzen je
    Niveau. Getrennt, weil die Grenze **keine** Modelleigenschaft ist und in einem
    Records-Log unter einem Modellnamen falsch einsortiert wäre.
    """
    specification = spec if spec is not None else DataSpec()
    seed_plan = plan if plan is not None else SeedPlan()
    points = [
        _StudyPoint(
            label=f"{level}",
            axis_value=level,
            measurement=measurement_at_level(process_config.measurement, level),
            extra={"variance/source": VarianceSource.JOINT.value, "noise/relative_std": level},
        )
        for level in noise_levels
    ]
    floors: dict[float, NoiseFloor] = {}

    def on_variant(point: _StudyPoint, replicate: int, data: GeneratedData) -> None:
        if replicate == 0:
            # axis_value ist bewusst `object` (siehe _StudyPoint) -- hier ist es immer
            # der float aus noise_levels, siehe Punktbau oben.
            assert isinstance(point.axis_value, float)
            floors[point.axis_value] = estimate_noise_floor(
                data, min(sizes), horizons=(1, specification.horizon)
            )

    records = _run_grid(
        process_config,
        factories,
        points,
        spec=specification,
        plan=seed_plan,
        sizes=sizes,
        policy=policy,
        evaluator=evaluator,
        logger=logger,
        group=group,
        cache=cache,
        on_variant=on_variant,
    )
    return records, floors


def run_validation_policy_comparison(
    process_config: ProcessConfig,
    factories: Mapping[str, ModelFactory],
    *,
    spec: DataSpec | None = None,
    plan: SeedPlan | None = None,
    sizes: Sequence[int] = DEFAULT_LEARNING_CURVE_SIZES,
    policies: Sequence[ValidationPolicy] = (
        ValidationPolicy.FIXED_POOL,
        ValidationPolicy.NONE,
    ),
    evaluator: RolloutEvaluator | None = None,
    logger: ExperimentLogger | None = None,
    cache: DataPoolCache | None = None,
    group: str = "validation_policy",
) -> list[RunRecord]:
    """Beide Validierungs-Betriebsarten auf identischen Daten.

    Identische Daten heißt hier wirklich identisch: derselbe Pool aus dem Cache, derselbe
    Trainings-Split, dieselben Skalierer. Nur die Frage, ob ein Validierungssplit
    existiert, unterscheidet sich. Die Differenz der beiden Kurven ist dann der Wert von
    Early Stopping und nichts sonst.

    **Was diese Studie mit den Baselines noch nicht zeigen kann:** Persistenz, statische
    Regression und ARX haben keinen Trainingsverlauf und ignorieren ``validation``. Die
    Differenz ist für sie exakt Null. Das ist kein Leerlauf, sondern eine
    Regressionsprüfung -- eine Differenz ungleich Null wäre der Beweis, dass
    Validierungsläufe irgendwo ins Training sickern. Die inhaltliche Zahl entsteht mit
    dem ersten Modell, das früh abbricht (Schritt 6).
    """
    specification = spec if spec is not None else DataSpec()
    seed_plan = plan if plan is not None else SeedPlan()
    points = [
        _StudyPoint(
            label=policy.value,
            axis_value=policy,
            policy=policy,
            extra={"variance/source": VarianceSource.JOINT.value},
        )
        for policy in policies
    ]
    return _run_grid(
        process_config,
        factories,
        points,
        spec=specification,
        plan=seed_plan,
        sizes=sizes,
        evaluator=evaluator,
        logger=logger,
        group=group,
        cache=cache,
    )


def summarise_policy_gap(
    records: Sequence[RunRecord],
    *,
    target: str,
    horizon: int,
    metric: str = "nrmse_run_median",
) -> dict[tuple[str, int], tuple[float, float, float]]:
    r"""Differenz zwischen den Validierungs-Betriebsarten je Modell und Stützstelle.

    Rückgabe: ``{(Modell, n_train): (Median FIXED_POOL, Median NONE, Differenz)}``, wobei
    die Differenz positiv ist, wenn ``FIXED_POOL`` besser (kleiner) ist -- also wenn
    Early Stopping etwas wert war.

    Der Median läuft über die Replikate. Ein Vergleich einzelner Seeds wäre hier
    besonders irreführend, weil die Differenz klein gegen die Seed-Streuung ist: sie ist
    an der Auflösungsschwelle aus
    :func:`twinlink_ml.evaluation.resolution.resolution_thresholds` zu messen, nicht an
    Null.
    """
    key = f"{metric}/{target}/h{horizon}"
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for record in records:
        if key not in record.metrics:
            continue
        label = str(record.params.get("model/label", record.tags.get("model", "?")))
        policy = str(record.params.get("validation_policy", "?"))
        n_train = int(record.params["n_train"])
        grouped.setdefault((label, n_train, policy), []).append(float(record.metrics[key]))

    result: dict[tuple[str, int], tuple[float, float, float]] = {}
    for (label, n_train, policy), values in grouped.items():
        if policy != ValidationPolicy.FIXED_POOL.value:
            continue
        other = grouped.get((label, n_train, ValidationPolicy.NONE.value))
        if other is None:
            continue
        fixed_median = float(np.median(values))
        none_median = float(np.median(other))
        result[(label, n_train)] = (fixed_median, none_median, none_median - fixed_median)
    return dict(sorted(result.items()))


class ObservationMode(str, Enum):
    """Betriebsart des Vergleichs bezüglich der Inline-Messbarkeit der Zielgröße.

    Der Grund, beide zu fahren: ob die Produktfeuchte an der Zielanlage inline gemessen
    wird, ist offen (siehe ``02_prozess_setup.md``). Die Antwort ändert die Aufgabe
    grundlegend -- mit Messung ist es eine Vorhersageaufgabe mit Anker, ohne Messung eine
    Soft-Sensor-Aufgabe ohne Anker -- und sie ändert, welche Baseline überhaupt definiert
    ist. Eine Architekturauswahl, die nur für einen der beiden Fälle gilt, ist wertlos,
    wenn sich der Fall später als der andere herausstellt.
    """

    INLINE = "inline"
    """Alle Zielgrößen sind inline gemessen und stehen in der Fensterhistorie."""

    SOFT_SENSOR = "soft_sensor"
    """Die Qualitätsgrößen sind nicht gemessen; nur Stell-, Stör- und Energiegrößen."""


def spec_for_mode(spec: DataSpec, mode: ObservationMode) -> DataSpec:
    """Datensatzspezifikation für eine Betriebsart.

    Im Soft-Sensor-Fall bleiben genau die Kanäle beobachtet, die an einer Wirbelschicht
    ohnehin verdrahtet sind: Temperaturen, Drücke, Volumenströme und der Energieverbrauch.
    Nicht beobachtet sind die Qualitätsgrößen -- Produktfeuchte und mittlere Korngröße.
    Das entspricht der Ausgangslage der Arbeit von Kräuter et al. am SPE, in der die
    Feuchte über Mikrowellenresonanz zwar *im Labor* gemessen wurde, aber genau als die
    Größe, die ein Soft Sensor ersetzen soll.
    """
    if mode is ObservationMode.INLINE:
        return dataclasses.replace(spec, observed_target_names=None)
    quality = ("product_moisture_kg_per_kg", "d50_um")
    observed = tuple(name for name in spec.target_names if name not in quality)
    return dataclasses.replace(spec, observed_target_names=observed)


def run_architecture_comparison(
    process_config: ProcessConfig,
    factories: Mapping[str, ModelFactory],
    *,
    spec: DataSpec | None = None,
    plan: SeedPlan | None = None,
    sizes: Sequence[int] = DEFAULT_LEARNING_CURVE_SIZES,
    modes: Sequence[ObservationMode] = (ObservationMode.INLINE,),
    sources: Sequence[VarianceSource] = (VarianceSource.JOINT, VarianceSource.MODEL),
    policy: ValidationPolicy = ValidationPolicy.FIXED_POOL,
    evaluator: RolloutEvaluator | None = None,
    logger: ExperimentLogger | None = None,
    group: str = "architecture_comparison",
    on_point: Callable[..., None] | None = None,
) -> list[RunRecord]:
    """Architekturvergleich über Stützstellen, Seed-Achsen und Beobachtbarkeit.

    Parameters
    ----------
    on_point
        Optionaler Rückruf ``(source: str, replicate: int, n_train: int) -> None``,
        aufgerufen **vor** jedem Datenpunkt -- nicht danach, damit ein Abbruch mitten in
        einem Punkt (z. B. durch Ctrl-C) noch anzeigt, welcher Punkt gerade lief.
        Gedacht für Fortschrittsausgaben in Treiberskripten: die Funktion selbst gibt
        nichts aus, ein stiller Bibliotheksaufruf soll nicht plötzlich auf stdout
        schreiben, nur weil ein Skript ihn benutzt.

    Die Voreinstellung fährt zwei Seed-Achsen: ``JOINT`` als berichtete, konservative
    Streuung und ``MODEL`` als die Achse, die bei den Baselines exakt Null war. Erst mit
    einem Modell, dessen Initialisierung wirkt, ist die Zerlegung von Schritt 5 an einem
    Fall geprüft, in dem sie etwas zu zeigen hat -- und erst dann ist die Aussage "diese
    Architektur ist besser" von "dieser Seed war besser" zu trennen.

    Der Cache wird **je Betriebsart neu** aufgebaut. Er ist nach ``data_seed`` verschlüsselt,
    und die Fensterung -- nicht die Simulation -- unterscheidet die Betriebsarten; ein
    gemeinsamer Cache wäre korrekt, aber die Spezifikation gehört zu seinem Zustand, und
    ein Cache, dessen Schlüssel nicht alles enthält, was das Ergebnis bestimmt, ist genau
    die Fehlerquelle, gegen die Schritt 5 die Fingerprints eingeführt hat.
    """
    base = spec if spec is not None else DataSpec()
    seed_plan = plan if plan is not None else SeedPlan()

    points: list[_StudyPoint] = []
    for mode in modes:
        mode_spec = spec_for_mode(base, mode)
        unobserved = list(mode_spec.window.unobserved_targets)
        for source in sources:
            points.append(
                _StudyPoint(
                    label=f"{mode.value}/{source.value}",
                    axis_value=(mode, source),
                    variance_source=source,
                    spec=mode_spec,
                    extra={
                        "variance/source": source.value,
                        "observation/mode": mode.value,
                        "observation/unobserved": unobserved,
                    },
                )
            )

    def on_point_adapter(point: _StudyPoint, replicate: int, n_train: int) -> None:
        if on_point is not None:
            on_point(source=point.extra["variance/source"], replicate=replicate, n_train=n_train)

    return _run_grid(
        process_config,
        factories,
        points,
        spec=base,
        plan=seed_plan,
        sizes=sizes,
        policy=policy,
        evaluator=evaluator,
        logger=logger,
        group=group,
        cache=None,
        on_point=on_point_adapter,
    )


def primary_metric(mode: ObservationMode) -> str:
    r"""Primärmetrik der Betriebsart -- und warum sie **nicht** dieselbe ist.

    Der Befund aus Schritt 3 lautete: die gepoolte Normierung ist systematisch zu klein,
    weil die gepoolte Streuung :math:`\sigma` die Niveauunterschiede **zwischen** den
    Läufen enthält, "und die muss kein Modell vorhersagen, sie stehen in der
    Fensterhistorie". Der zweite Halbsatz ist eine **Voraussetzung**, nicht eine
    Tatsache: er gilt genau dann, wenn die Zielgröße inline gemessen wird.

    Ohne Inline-Messung steht das Niveau nicht in der Historie. Es ist dann Teil der
    Aufgabe, und der gepoolte Nenner ist der richtige. Die laufweise Normierung wird dort
    zusätzlich instabil: bei einem Lauf mit kleiner Eigenstreuung geht
    :math:`\sigma_{\text{run}}` gegen Null und der laufweise NRMSE gegen unendlich --
    gemessen wurden Mediane über 3 und Seed-Spannweiten über 19 bei einer
    parameterfreien Mittelwertvorhersage.

    Deshalb:

    * ``INLINE`` → ``nrmse_run_median`` (laufweiser Median, unverändert)
    * ``SOFT_SENSOR`` → ``nrmse`` (gepoolt)

    Beide werden in jedem Fall protokolliert; die Wahl betrifft, welche berichtet wird.
    Eine Kurve über beide Betriebsarten hinweg darf **nicht** in einem Diagramm stehen:
    die y-Achsen bedeuten Verschiedenes.
    """
    return "nrmse_run_median" if mode is ObservationMode.INLINE else "nrmse"
