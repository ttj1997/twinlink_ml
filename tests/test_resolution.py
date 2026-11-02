"""Tests for resolving power and variance decomposition (work plan step 5).

Pure statistics, no simulation: the statements of this module decide whether an
architecture difference may be reported. They must therefore be checkable without a
process model -- otherwise an error here would only be visible in a learning-curve run and
there indistinguishable from a real effect.
"""

from __future__ import annotations

import math

import pytest

from twinlink_ml.evaluation.resolution import (
    PointKey,
    SeedSpread,
    collect_spreads,
    decompose_variance,
    resolution_report,
    resolution_thresholds,
    select,
)
from twinlink_ml.utils.seeding import SeedBundle, SeedPlan, VarianceSource
from twinlink_ml.utils.tracking import RunRecord


def make_record(
    *,
    label: str,
    source: VarianceSource,
    replicate: int,
    value: float,
    n_train: int = 8,
    target: str = "energy_target",
    horizon: int = 24,
    extra: dict | None = None,
) -> RunRecord:
    """Minimal record, as ``run_variance_decomposition`` writes it."""
    params = {
        "model/label": label,
        "n_train": n_train,
        "variance/source": source.value,
        "variance/replicate": replicate,
    }
    params.update(extra or {})
    return RunRecord(
        group="variance_decomposition",
        params=params,
        metrics={f"nrmse_run_median/{target}/h{horizon}": value},
        tags={"model": label},
    )


# --------------------------------------------------------------------- SeedPlan


def test_seed_plan_varies_exactly_one_axis() -> None:
    plan = SeedPlan(master_seeds=(1, 2, 3))
    reference = plan.reference
    for source in (VarianceSource.DATA, VarianceSource.NOISE, VarianceSource.MODEL):
        for replicate in range(1, plan.n_replicates):
            bundle = plan.bundle(source, replicate)
            differing = [
                field
                for field in ("data_seed", "noise_seed", "model_seed")
                if getattr(bundle, field) != getattr(reference, field)
            ]
            assert differing == [f"{source.value}_seed"], (source, replicate, differing)


def test_replicate_zero_is_the_reference_on_every_axis() -> None:
    """Otherwise each axis would have its own reference point and the spans would not be.

    comparable.
    """
    plan = SeedPlan()
    for source in VarianceSource:
        assert plan.bundle(source, 0) == plan.reference


def test_data_axis_shares_seeds_with_joint_axis() -> None:
    """Precondition for the pool cache taking effect across the axes."""
    plan = SeedPlan()
    data_seeds = {bundle.data_seed for bundle in plan.bundles(VarianceSource.DATA)}
    joint_seeds = {bundle.data_seed for bundle in plan.bundles(VarianceSource.JOINT)}
    assert data_seeds == joint_seeds
    assert len(plan.data_seeds(tuple(VarianceSource))) == plan.n_replicates


def test_seed_plan_rejects_degenerate_configuration() -> None:
    with pytest.raises(ValueError):
        SeedPlan(master_seeds=(1,))
    with pytest.raises(ValueError):
        SeedPlan(master_seeds=(1, 1, 2))


def test_bundle_replace_rejects_joint() -> None:
    with pytest.raises(ValueError):
        SeedBundle.from_master(1).replace(VarianceSource.JOINT, 5)


# ------------------------------------------------------------------- SeedSpread


def test_spread_statistics() -> None:
    spread = SeedSpread(source=VarianceSource.DATA, values=(0.90, 1.00, 1.10))
    assert spread.median == pytest.approx(1.0)
    assert spread.span == pytest.approx(0.2)
    assert spread.relative_span == pytest.approx(0.2)
    # Hartley d2(3) = 1.693
    assert spread.sigma_from_span == pytest.approx(0.2 / 1.693)


def test_minimum_detectable_difference_falls_with_replicates() -> None:
    r"""The threshold falls with :math:`\sqrt{2/n}` and is smaller than the span."""
    spread = SeedSpread(source=VarianceSource.JOINT, values=(0.90, 1.00, 1.10))
    three = spread.minimum_detectable_difference(n_replicates=3)
    twelve = spread.minimum_detectable_difference(n_replicates=12)
    assert three > twelve
    assert three / twelve == pytest.approx(math.sqrt(12 / 3))
    assert three < spread.span


def test_required_replicates_is_monotone_and_capped() -> None:
    spread = SeedSpread(source=VarianceSource.JOINT, values=(0.90, 1.00, 1.10))
    assert spread.required_replicates(0.20) < spread.required_replicates(0.05)
    assert spread.required_replicates(1e-6) == 200
    with pytest.raises(ValueError):
        spread.required_replicates(0.0)


def test_parameter_free_baseline_has_zero_model_spread() -> None:
    """A model without parameters cannot scatter over the model seed -- exactly for that reason.

    it is suitable as a reference for the resolving power.
    """
    spread = SeedSpread(source=VarianceSource.MODEL, values=(0.794, 0.794, 0.794))
    assert spread.sigma_from_span == 0.0
    assert spread.required_replicates(0.05) == 1


# --------------------------------------------------------------- Aggregation


def test_collect_and_decompose() -> None:
    values = {
        VarianceSource.DATA: (1.00, 1.08, 0.94),
        VarianceSource.NOISE: (1.00, 1.01, 0.99),
        VarianceSource.MODEL: (1.00, 1.00, 1.00),
        VarianceSource.JOINT: (1.00, 1.09, 0.93),
    }
    records = [
        make_record(label="arx", source=source, replicate=k, value=value)
        for source, series in values.items()
        for k, value in enumerate(series)
    ]
    decomposition = decompose_variance(records, target="energy_target", horizon=24)
    item = decomposition[PointKey("arx", 8)]
    assert item.dominant_source is VarianceSource.DATA
    assert item.additivity_ratio == pytest.approx(math.sqrt(0.14**2 + 0.02**2) / 0.16, rel=1e-6)


def test_mixed_factor_log_is_rejected() -> None:
    """Two noise levels in the same log would yield a 'seed scatter' that in truth is the.

    effect of the noise level. The duplicate entry must be noticed.
    """
    records = [
        make_record(
            label="arx",
            source=VarianceSource.JOINT,
            replicate=0,
            value=0.5,
            extra={"noise/relative_std": 0.005},
        ),
        make_record(
            label="arx",
            source=VarianceSource.JOINT,
            replicate=0,
            value=0.9,
            extra={"noise/relative_std": 0.02},
        ),
    ]
    with pytest.raises(ValueError):
        collect_spreads(records, target="energy_target", horizon=24)

    filtered = select(records, **{"noise/relative_std": 0.005})
    assert len(filtered) == 1
    assert collect_spreads(filtered, target="energy_target", horizon=24)


def test_resolution_threshold_requires_parameter_free_reference() -> None:
    records = [
        make_record(label="arx", source=VarianceSource.JOINT, replicate=k, value=value)
        for k, value in enumerate((1.0, 1.1, 0.9))
    ]
    with pytest.raises(ValueError):
        resolution_thresholds(records, target="energy_target", horizon=24)

    records += [
        make_record(label="persistence", source=VarianceSource.JOINT, replicate=k, value=value)
        for k, value in enumerate((0.80, 0.84, 0.79))
    ]
    thresholds = resolution_thresholds(records, target="energy_target", horizon=24)
    assert set(thresholds) == {8}
    assert 0.0 < thresholds[8] < 0.5
    assert "persistence" in resolution_report(records, target="energy_target", horizon=24)
