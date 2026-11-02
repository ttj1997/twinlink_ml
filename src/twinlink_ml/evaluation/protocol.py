r"""Evaluation protocol (work plan step 5): the four studies.

What is orchestrated here
--------------------------
1. :func:`run_variance_decomposition` -- decompose the spread by source
   (``data`` / ``noise`` / ``model`` / joint). Analysis in
   :mod:`twinlink_ml.evaluation.resolution`.
2. The resolving power comes out of (1) as a by-product as soon as a parameter-free
   baseline runs along. There is therefore **no** separate study for it -- a separate
   one would run with its own seeds and provide a different threshold than the curve to
   which it is applied.
3. :func:`run_noise_study` -- noise level as an explicit experimental factor, together
   with :func:`estimate_noise_floor` for the **saturation level**.
4. :func:`run_validation_policy_comparison` -- ``FIXED_POOL`` against ``NONE``.

The cost lever: one simulation per data set realisation
---------------------------------------------------------
Naively counted, step 5 costs a multiple of step 4: three axes times three replicates
times three noise levels times two validation modes. In fact the grey-box model is
deterministic, and ``noise_seed`` as well as the noise level act **exclusively** in the
sensor model. :class:`DataPoolCache` exploits this: integration happens once per
``data_seed``, everything else is derived via
:meth:`~twinlink_ml.evaluation.experiments.GeneratedData.with_measurements` from the
stored noise-free outputs. The cache is therefore not a performance detail but the
precondition for a clean decomposition: it enforces that two runs that are supposed to
differ only in the noise seed really have the same excitation and state trajectory.

Pitfall that is avoided precisely by this
--------------------------------------------
If the noise seed is varied via a new master seed, the excitation seed changes with it.
The "noise" axis then also measures the data set realisation, and the decomposition
claims the opposite of what it shows. That is why
:class:`~twinlink_ml.utils.seeding.SeedPlan` and not the list of master seeds is the
interface of these studies.
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
    """Short fingerprint of a ``DataSpec``, value-based.

    The same pattern as ``_measurement_digest`` in ``experiments.py``: concatenate field
    names and values sorted, then hash. Needed for the cache invalidation in
    :func:`_run_grid` -- ``spec_for_mode`` builds a new object on every call, so a
    comparison on object identity would trigger a new cache at *every* point there, not
    only on genuine mode changes. Value equality via a fingerprint is therefore not
    cosmetics but the precondition for correct reuse.
    """
    payload = ":".join(f"{key}={value}" for key, value in sorted(dataclasses.asdict(spec).items()))
    return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()


@dataclass(slots=True)
class DataPoolCache:
    """Cache run pools by ``data_seed``, redraw measurements cheaply.

    Not as global state but as an injectable object: a study is handed its cache and can
    share it with others or not. The cache holds complete trajectories, i.e. several
    hundred megabytes for large pools -- ``clear`` is therefore part of the interface and
    not optional.
    """

    process_config: ProcessConfig
    spec: DataSpec
    _pools: dict[int, GeneratedData] = field(default_factory=dict, init=False)
    n_simulations: int = field(default=0, init=False)
    n_remeasurements: int = field(default=0, init=False)

    def base(self, data_seed: int) -> GeneratedData:
        """Integrated pool for a ``data_seed``.

        Noise and model seed have no meaning here: :meth:`variant` replaces both. They are
        set to the ``data_seed`` so that the base pool is not accidentally used as a
        finished data set and then carries the same noise for all ``data_seed``.
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
        """Pool with given seeds and sensor configuration.

        The integration only happens on the first call per ``data_seed``. ``noise_seed``,
        ``model_seed`` and ``measurement`` are pure post-processing after that.
        """
        pool = self.base(data_seed)
        variant = pool.with_measurements(noise_seed=noise_seed, measurement=measurement)
        self.n_remeasurements += 1
        return dataclasses.replace(
            variant, seeds=dataclasses.replace(variant.seeds, model_seed=model_seed)
        )

    def clear(self) -> None:
        """Release memory."""
        self._pools.clear()

    def as_metrics(self) -> dict[str, float]:
        """Cost figures -- they belong in the report, not only in the log file."""
        return {
            "cost/n_simulations": float(self.n_simulations),
            "cost/n_remeasurements": float(self.n_remeasurements),
        }


@dataclass(frozen=True, slots=True)
class _StudyPoint:
    """One point on the varied axis of a step-5 study.

    Everything in which the four ``run_*`` functions differ is contained in a
    ``_StudyPoint`` -- the rest is :func:`_run_grid`.

    Attributes
    ----------
    label
        For logs and progress output, e.g. ``"joint"``, ``"0.02"``, ``"fixed_pool"``.
    axis_value
        The raw axis value: ``VarianceSource``, noise level as ``float``,
        ``ValidationPolicy``, or a ``(ObservationMode, VarianceSource)`` tuple, depending
        on the study. Hooks read **from here**, not from the string-keyed ``extra`` -- a
        noise level is a number, not a string parsed back.
    variance_source
        Which seed axis this point uses for :meth:`SeedPlan.bundle`.
    spec
        Override of the base ``DataSpec``; ``None`` means unchanged.
    measurement
        Override of the ``MeasurementConfig``; ``None`` means unchanged.
    policy
        Override of the ``ValidationPolicy`` for ``evaluate_point``; ``None`` means the
        function default of :func:`_run_grid` applies.
    extra
        Additional, **point-fixed** record parameters -- not replicate-dependent; for that
        there is ``extra_fn`` in :func:`_run_grid`.
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
    r"""Core of all four step-5 studies: (axis point, replicate, ``n_train``) grid.

    Common scaffold of the four ``run_*`` functions, once instead of four almost identical
    copies (``04_stand_ap1.md``, finding on the core consolidation). What differs between
    them is contained in ``points`` (see :class:`_StudyPoint`) and three hooks:

    - ``extra_fn`` for replicate-dependent but side-effect-free record parameters
      (``run_variance_decomposition``\ 's ``variance/master_seed``).
    - ``on_variant`` for side effects after building a data variant, before the
      ``n_train`` loop (``run_noise_study``\ 's noise limit at replicate 0).
    - ``on_point`` exclusively for progress events, immediately before
      ``evaluate_point`` (``run_architecture_comparison``\ 's ``on_point`` parameter).

    Cache life cycle
    -------------------
    Without an injected ``cache`` this function builds one itself -- **anew as soon as the
    fingerprint of the spec resolved per point changes** (:func:`_spec_digest`, not object
    identity or ``!=`` on the ``DataSpec`` itself: ``spec_for_mode`` builds a new object on
    every call, an identity comparison would trigger a new cache at *every* point, not
    only on genuine mode changes). With an injected cache it never cleans up itself --
    that remains the caller's business, as before for the three studies that accept a
    cache.
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
        # For mypy: always set by the cache parameter or the branch above.
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
    """Learning curve, but varied separately per random source.

    The result is records with the parameters ``variance/source`` and
    ``variance/replicate``; the analysis is done by
    :func:`twinlink_ml.evaluation.resolution.decompose_variance`.

    Replicate 0 is the same reference bundle on all axes. It is nevertheless logged per
    axis -- the spread of an axis needs its reference point, and deduplicating in the log
    would make the analysis depend on the order of the axes. It is computed only once,
    because the cache applies.
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
        # point unused -- common hook signature with on_variant/on_point.
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
    """Saturation level of the learning curve: the error of a perfect model.

    Computed **without a model**. The deterministic part of the measurement is used as the
    prediction -- i.e. the output after the sensor dynamics but without noise, obtained
    from the same trajectory via
    :meth:`~twinlink_ml.evaluation.experiments.GeneratedData.with_measurements` with
    noise level zero. No model that is trained on the measurement and checked against the
    measurement can become better than that.

    Why this is necessary: without this number, a learning curve that flattens at 0.12
    cannot be read as a model limit *or* a noise limit. With it the question is settled,
    and that before the first architecture comparison instead of after the tenth.

    The second, often overlooked part: ``against_truth`` measures the same perfect
    predictor against the **noise-free and unfiltered** output. This number is not zero,
    because the sensor PT1 (placeholder 10 s) introduces a real phase shift. Anyone who
    reads ``nrmse_truth`` as "error against the truth" without knowing this base level
    mistakes sensor dynamics for model error.
    """

    target_names: tuple[str, ...]
    horizons: tuple[int, ...]
    relative_noise_std: float
    pooled: dict[str, dict[int, float]]
    per_run: dict[str, dict[int, tuple[float, float, float]]]
    against_truth: dict[str, dict[int, tuple[float, float, float]]]

    def as_metrics(self) -> dict[str, float]:
        """Flat figures for the log.

        Three separate variable names for the three loops, on purpose: ``pooled`` provides
        a single value per horizon, ``per_run`` and ``against_truth`` provide
        median/minimum/maximum triples. Reusing the same name would be rejected by mypy as
        an incompatible reassignment -- rightly so, they are different shapes.
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
        """Table for run logs."""
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
    """Sensor model without noise, dynamics unchanged."""
    return dataclasses.replace(
        measurement, relative_noise_std=0.0, absolute_noise_std_temperature_k=0.0
    )


def measurement_at_level(base: MeasurementConfig, level: float) -> MeasurementConfig:
    """Sensor configuration for a noise level -- **one** factor, not two.

    The sensor model has two noise terms: relative on all channels and absolute on the
    temperature. If only the relative one is varied, a "noise-free" data set is still
    noisy on the temperature channel, and the experimental factor would not be the one
    recorded in the log. The absolute term is therefore scaled along in the same ratio.
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
    """Noise limit on the same test split and in the same normalisation as the curve.

    Deliberately via the windowing and not directly on the time series: the primary
    metric is normalised run-level over **windows**, and overlapping windows weight the
    time axis differently than the raw data. A noise limit in a different normalisation
    than the curve would not be comparable -- and exactly that comparison is the purpose.
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
        # Should never happen for a data set generated by generate_data() -- the truth is
        # always carried along there. A None here would be a structural error in the
        # data set generation, not an expected case to be caught.
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
    """Learning curves over several noise levels, plus the noise limit per level.

    The seeds vary jointly (axis ``JOINT``) -- within a noise study the decomposition by
    source is not the question, and the conservative spread is the right figure for the
    saturation level.

    The return value is a pair: the records of the models and the model-free noise limits
    per level. Separate, because the limit is **not** a model property and would be filed
    wrongly under a model name in a records log.
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
            # axis_value is deliberately `object` (see _StudyPoint) -- here it is always
            # the float from noise_levels, see the point construction above.
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
    """Both validation modes on identical data.

    Identical data really means identical here: the same pool from the cache, the same
    training split, the same scalers. Only the question whether a validation split exists
    differs. The difference between the two curves is then the value of early stopping
    and nothing else.

    **What this study cannot yet show with the baselines:** persistence, static
    regression and ARX have no training history and ignore ``validation``. For them the
    difference is exactly zero. That is not idle running but a regression check -- a
    non-zero difference would be proof that validation runs leak into training somewhere.
    The substantive number arises with the first model that stops early (step 6).
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
    r"""Difference between the validation modes per model and learning-curve point.

    Returns: ``{(Modell, n_train): (Median FIXED_POOL, Median NONE, Differenz)}``, where
    the difference is positive if ``FIXED_POOL`` is better (smaller) -- i.e. if early
    stopping was worth something.

    The median runs over the replicates. A comparison of individual seeds would be
    particularly misleading here, because the difference is small compared with the seed
    spread: it is to be measured against the resolution threshold from
    :func:`twinlink_ml.evaluation.resolution.resolution_thresholds`, not against zero.
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
    """Operating mode of the comparison with respect to inline measurability of the target.

    The reason to run both: whether the product moisture is measured inline at the target
    plant is open (see ``02_prozess_setup.md``). The answer changes the task
    fundamentally -- with measurement it is a prediction task with an anchor, without
    measurement a soft-sensor task without an anchor -- and it changes which baseline is
    defined at all. An architecture selection that holds only for one of the two cases is
    worthless if the case later turns out to be the other one.
    """

    INLINE = "inline"
    """Alle Zielgrößen sind inline gemessen und stehen in der Fensterhistorie."""

    SOFT_SENSOR = "soft_sensor"
    """Die Qualitätsgrößen sind nicht gemessen; nur Stell-, Stör- und Energiegrößen."""


def spec_for_mode(spec: DataSpec, mode: ObservationMode) -> DataSpec:
    """Data set specification for an operating mode.

    In the soft-sensor case exactly those channels remain observed that are wired at a
    fluidized bed anyway: temperatures, pressures, volume flows and the energy
    consumption. Not observed are the quality variables -- product moisture and mean
    granule size. This corresponds to the starting point of the work of Kräuter et al. at
    the SPE, in which the moisture was indeed measured *in the lab* via microwave
    resonance, but precisely as the quantity that a soft sensor is meant to replace.
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
    """Architecture comparison over learning-curve points, seed axes and observability.

    Parameters
    ----------
    on_point
        Optional callback ``(source: str, replicate: int, n_train: int) -> None``,
        called **before** every data point -- not after, so that an abort in the middle of
        a point (e.g. by Ctrl-C) still shows which point was running. Intended for
        progress output in driver scripts: the function itself prints nothing, a silent
        library call should not suddenly write to stdout just because a script uses it.

    The default runs two seed axes: ``JOINT`` as the reported, conservative spread and
    ``MODEL`` as the axis that was exactly zero for the baselines. Only with a model whose
    initialisation has an effect is the decomposition of step 5 checked on a case in which
    it has something to show -- and only then can the statement "this architecture is
    better" be separated from "this seed was better".

    The cache is rebuilt **per operating mode**. It is keyed by ``data_seed``, and the
    windowing -- not the simulation -- distinguishes the operating modes; a shared cache
    would be correct, but the specification belongs to its state, and a cache whose key
    does not contain everything that determines the result is exactly the source of error
    against which step 5 introduced the fingerprints.
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
    r"""Primary metric of the operating mode -- and why it is **not** the same.

    The finding from step 3 was: the pooled normalisation is systematically too small,
    because the pooled spread :math:`\sigma` contains the level differences **between**
    the runs, "and no model has to predict them, they are in the window history". The
    second half of the sentence is a **precondition**, not a fact: it holds exactly when
    the target is measured inline.

    Without inline measurement the level is not in the history. It is then part of the
    task, and the pooled denominator is the right one. The run-level normalisation also
    becomes unstable there: for a run with small intrinsic spread,
    :math:`\sigma_{\text{run}}` goes to zero and the run-level NRMSE to infinity --
    medians above 3 and seed ranges above 19 were measured for a parameter-free mean
    prediction.

    Therefore:

    * ``INLINE`` → ``nrmse_run_median`` (run-level median, unchanged)
    * ``SOFT_SENSOR`` → ``nrmse`` (pooled)

    Both are logged in every case; the choice concerns which one is reported. A curve
    across both operating modes must **not** appear in one chart: the y-axes mean
    different things.
    """
    return "nrmse_run_median" if mode is ObservationMode.INLINE else "nrmse"
