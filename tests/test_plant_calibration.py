"""Tests of the plant calibration study (docs/40_plant_calibration.md). Synthetic data only."""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.plant_calibration import (
    ModelContext,
    apply_bias,
    bias_offsets,
    collinearity_index,
    delta_msqr,
    determinant_measure,
    error_metrics,
    evaluate,
    evaluate_rows,
    fd_steps,
    finite_difference_thetas,
    fit_parameters,
    identifiable_subsets,
    load_plant_calibration_config,
    offset_for,
    profile_verdict,
    reading,
    select_subset,
    sensitivity_from_outputs,
    svd_summary,
)
from twinlink_ml.plant_scale import (
    Freeboard,
    PlantParameters,
    PlantPoint,
    load_plant_scale_config,
    steady_plant,
)
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

ROOT = Path(__file__).resolve().parents[1]

# Synthetic operating points (invented round numbers of plant-like size, not plant data).
POINT_A = PlantPoint("syn_a", 10.0, 30000.0, 600.0, 85.0, 1500.0, 55.0, 47.0)
POINT_B = PlantPoint("syn_b", 15.0, 28000.0, 600.0, 82.0, 900.0, 62.0, 56.0)


@pytest.fixture(scope="module")
def cfg():
    return load_plant_calibration_config()


@pytest.fixture(scope="module")
def ctx(cfg):
    ps = load_plant_scale_config(cfg.plant_scale_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(ps.process_config)
    kwargs = {
        "normal_density_kg_per_m3": ps.data.normal_density_dry_air_kg_per_m3,
        "inlet_humidity_kg_per_kg": ps.placeholders.inlet_absolute_humidity_kg_per_kg,
        "feed_celsius": ps.placeholders.feed_temperature_celsius,
        "iterations": ps.calibration.velocity_iterations,
        "relative_tolerance": ps.calibration.velocity_relative_tolerance,
    }
    return ModelContext(base, ps.placeholders.solid_fraction_reference, kwargs)


# ------------------------------------------------------------------------------------ config


def test_config_parameters(cfg):
    names = cfg.names
    assert names == (
        "ua_bed_w_per_k",
        "offset_product_1_k",
        "offset_product_2_k",
        "freeboard_ua_w_per_k",
        "cp_solid_j_per_kg_k",
        "critical_moisture_kg_per_kg",
        "max_saturation_degree",
    )
    kinds = {p.name: p.kind for p in cfg.parameters}
    assert kinds["critical_moisture_kg_per_kg"] == "kinetics"
    assert kinds["max_saturation_degree"] == "kinetics"
    assert kinds["cp_solid_j_per_kg_k"] == "dynamic"
    assert cfg.sensitivity.max_parameters == 3
    assert cfg.workers <= 8
    assert cfg.sigma_bed_k == pytest.approx(0.2 / math.sqrt(6))
    # switch off at nominal
    nom = cfg.nominal()
    assert nom["offset_product_1_k"] == nom["offset_product_2_k"] == 0.0
    assert nom["freeboard_ua_w_per_k"] == 0.0


def test_offset_for():
    theta = {"offset_product_1_k": 2.0, "offset_product_2_k": 5.0}
    assert offset_for(theta, "product_1") == 2.0
    assert offset_for(theta | {"offset_shared_k": 1.0}, "product_2") == 6.0
    assert offset_for({}, "product_1") == 0.0


# ------------------------------------------------------------------------------------ model


def test_switch_off_is_bit_identical(cfg, ctx):
    """Nominal parameters: exactly the unchanged plant-near model (Freeboard none)."""
    ref = steady_plant(
        ctx.base, POINT_A, PlantParameters(0.0, ctx.solid_fraction), **ctx.steady_kwargs
    )
    pred = evaluate(ctx, POINT_A, "product_1", cfg.nominal())
    assert pred.bed_celsius == ref.bed_temperature_celsius
    assert pred.exhaust_celsius == ref.exhaust_temperature_celsius
    assert pred.exhaust_celsius == pred.bed_celsius
    # offset: exactly subtracted; heat-loss term: the Befund 117 formula (C1)
    theta = cfg.nominal() | {"offset_product_1_k": 3.0}
    shifted = evaluate(ctx, POINT_A, "product_1", theta)
    assert shifted.exhaust_celsius == ref.exhaust_temperature_celsius - 3.0
    other = evaluate(ctx, POINT_A, "product_2", theta)
    assert other.exhaust_celsius == ref.exhaust_temperature_celsius
    fb = evaluate(ctx, POINT_A, "product_1", cfg.nominal() | {"freeboard_ua_w_per_k": 1500.0})
    c1 = steady_plant(
        ctx.base,
        POINT_A,
        PlantParameters(0.0, ctx.solid_fraction, Freeboard("heat_loss", 1500.0)),
        **ctx.steady_kwargs,
    )
    assert fb.exhaust_celsius == pytest.approx(c1.exhaust_temperature_celsius, abs=1e-9)
    assert fb.bed_celsius == ref.bed_temperature_celsius


def test_heat_capacity_has_no_steady_effect(cfg, ctx):
    a = evaluate(ctx, POINT_B, "product_2", cfg.nominal())
    b = evaluate(ctx, POINT_B, "product_2", cfg.nominal() | {"cp_solid_j_per_kg_k": 1800.0})
    assert b.bed_celsius == pytest.approx(a.bed_celsius, abs=1e-6)
    assert b.exhaust_celsius == pytest.approx(a.exhaust_celsius, abs=1e-6)


def test_synthetic_recovery(cfg, ctx):
    """Synthetic 'measurements' from the model with known UA and distance are recovered."""
    truth = cfg.nominal() | {"ua_bed_w_per_k": 1500.0, "offset_product_1_k": 6.0}
    rows = [(POINT_A, "product_1")]
    measured = evaluate_rows(ctx, rows, truth)
    sigma = np.array([cfg.sigma_bed_k, cfg.sigma_exhaust_k])
    guesses: dict[str, float] = {}
    fit = fit_parameters(
        lambda t: evaluate_rows(ctx, rows, t, guesses),
        measured,
        sigma,
        cfg.nominal(),
        ["ua_bed_w_per_k", "offset_product_1_k"],
        scales={p.name: p.scale for p in cfg.parameters},
        bounds={p.name: p.bounds for p in cfg.parameters},
        start={"ua_bed_w_per_k": 1000.0, "offset_product_1_k": 4.0},
        max_nfev=30,
    )
    assert fit.theta["ua_bed_w_per_k"] == pytest.approx(1500.0, rel=1e-3)
    assert fit.theta["offset_product_1_k"] == pytest.approx(6.0, abs=1e-3)
    assert fit.ssr < 1e-4


# ------------------------------------------------------------------------- identifiability


def test_sensitivity_of_linear_model():
    a = np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 0.0], [3.0, 0.0, 0.0]])  # rows x params

    def model(theta):
        x = np.array([theta["p"], theta["q"], theta["r"]])
        return a @ x

    theta0 = {"p": 1.0, "q": 2.0, "r": 0.0}
    scales = {"p": 2.0, "q": 0.5, "r": 1.0}
    thetas = finite_difference_thetas(theta0, ["p", "q", "r"], scales, 0.01)
    assert len(thetas) == 7
    outputs = np.vstack([model(t) for t in thetas])
    sigma = np.array([0.1, 0.2, 0.5])
    s = sensitivity_from_outputs(outputs, sigma, fd_steps(thetas, ["p", "q", "r"], scales))
    expected = a * np.array([2.0, 0.5, 1.0])[None, :] / sigma[:, None]
    np.testing.assert_allclose(s, expected, rtol=1e-9, atol=1e-12)
    assert delta_msqr(s)[2] == 0.0
    summary = svd_summary(s, ["p", "q", "r"])
    assert summary["numerical_rank"] == 2
    assert math.isinf(summary["condition_number"])


def test_collinearity_and_selection():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(8, 2)) * 5.0
    s = np.column_stack([base[:, 0], base[:, 1], 2.0 * base[:, 0], np.zeros(8)])
    names = ["a", "b", "c", "d"]
    assert math.isinf(collinearity_index(s[:, [0, 2]]))
    assert collinearity_index(s[:, [0, 1]]) < 10.0
    assert math.isinf(collinearity_index(s[:, [3]]))
    assert determinant_measure(s[:, [0, 2]]) == pytest.approx(0.0, abs=1e-6)
    scores = identifiable_subsets(s, names, names, delta_min=1.0, gamma_max=10.0, max_size=3)
    sel = select_subset(scores, names)
    assert len(sel) == 2 and "d" not in sel and set(sel) != {"a", "c"}
    # eligibility restricts the choice
    scores_b = identifiable_subsets(s, names, ["b", "d"], delta_min=1.0, gamma_max=10.0, max_size=3)
    assert select_subset(scores_b, names) == ("b",)
    # nothing identifiable
    assert (
        select_subset(
            identifiable_subsets(s, names, ["d"], delta_min=1.0, gamma_max=10.0, max_size=1), names
        )
        == ()
    )


def test_profile_verdict_quadratic_and_flat():
    grid = np.linspace(-5.0, 5.0, 11)
    ssr = 10.0 + grid**2  # weighted SSR; chi2 threshold 3.84 -> |x| < 1.96
    v = profile_verdict(
        grid,
        ssr,
        x_opt=0.0,
        ssr_opt=10.0,
        bounds=(-5.0, 5.0),
        n_residuals=12,
        n_free=2,
        threshold_chi2=3.84,
    )
    assert v["sensor"]["practically_identifiable"]
    assert -2.5 < v["sensor"]["lower"] < -1.9 and 1.9 < v["sensor"]["upper"] < 2.5
    flat = profile_verdict(
        grid,
        np.full(11, 10.0) + 0.01 * grid**2,
        x_opt=0.0,
        ssr_opt=10.0,
        bounds=(-5.0, 5.0),
        n_residuals=12,
        n_free=2,
        threshold_chi2=3.84,
    )
    assert not flat["sensor"]["practically_identifiable"]
    # optimum on the lower bound: only the upper side must be crossed
    onesided = profile_verdict(
        grid + 5.0,
        10.0 + (grid + 5.0) ** 2,
        x_opt=0.0,
        ssr_opt=10.0,
        bounds=(0.0, 10.0),
        n_residuals=12,
        n_free=2,
        threshold_chi2=3.84,
    )
    assert onesided["optimum_at_bound"] and onesided["sensor"]["practically_identifiable"]
    # rescaled variant: s^2 = 10 / 10 = 1 -> same as sensor here
    assert v["rescaled"]["practically_identifiable"]


# ------------------------------------------------------------------------------------ calibration


def test_bias_and_metrics():
    model = np.array([50.0, 50.0, 52.0, 52.0])
    measured = np.array([51.0, 45.0, 53.0, 47.0])
    b = bias_offsets(model, measured)
    assert b == {"bed": 1.0, "exhaust": -5.0}
    corrected = apply_bias(model, b)
    np.testing.assert_allclose(corrected, measured)
    m = error_metrics(model, measured)
    assert m["bed"]["max_abs_k"] == 1.0 and m["exhaust"]["max_abs_k"] == 5.0
    assert m["max_abs_k"] == 5.0
    failed = error_metrics(np.array([np.nan, 1.0]), np.array([0.0, 0.0]))
    assert math.isinf(failed["max_abs_k"])


def _m(bed: float, exh: float) -> dict:
    return error_metrics(np.array([bed, exh]), np.zeros(2))


def test_reading_cases():
    good = {"d1": _m(1.0, 1.5), "d2": _m(-1.9, 0.5)}
    bias = {"d1": _m(3.0, 4.0), "d2": _m(2.0, 2.5)}
    assert reading(good, bias, 2.0)["reading"] == "K0"
    worse = {"d1": _m(1.0, 4.5), "d2": _m(0.5, 0.5)}
    assert reading(worse, bias, 2.0)["reading"] == "K1"
    equal = {"d1": _m(1.0, 4.0), "d2": _m(0.5, 0.5)}
    assert reading(equal, bias, 2.0)["reading"] == "K1"
    between = {"d1": _m(1.0, 3.0), "d2": _m(0.5, 0.5)}
    assert reading(between, bias, 2.0)["reading"] == "between"
    good_but_bias_better = {"d1": _m(1.0, 1.5), "d2": _m(0.5, 0.5)}
    tiny_bias = {"d1": _m(0.1, 0.1), "d2": _m(0.1, 0.1)}
    assert reading(good_but_bias_better, tiny_bias, 2.0)["reading"] == "K1"


def test_fit_without_free_parameters():
    fit = fit_parameters(
        lambda t: np.array([t["a"], t["a"]]),
        np.array([1.0, 3.0]),
        np.array([1.0, 1.0]),
        {"a": 2.0},
        [],
        scales={"a": 1.0},
        bounds={"a": (0.0, 5.0)},
        start={"a": 1.0},
        max_nfev=5,
    )
    assert fit.theta == {"a": 2.0} and fit.ssr == pytest.approx(2.0)


def test_fit_respects_bounds_and_failure_residual():
    def outputs(theta):
        if theta["a"] > 4.0:
            return np.array([np.nan])
        return np.array([theta["a"]])

    fit = fit_parameters(
        outputs,
        np.array([10.0]),
        np.array([1.0]),
        {"a": 1.0},
        ["a"],
        scales={"a": 1.0},
        bounds={"a": (0.0, 3.0)},
        start={"a": 1.0},
        max_nfev=50,
    )
    assert fit.theta["a"] == pytest.approx(3.0, abs=1e-6)
    assert "a" in fit.at_bound


# ------------------------------------------------------------------------------------ driver


def _driver():
    sys.path.insert(0, str(ROOT / "scripts" / "verify"))
    spec = importlib.util.spec_from_file_location(
        "run_plant_calibration", ROOT / "scripts" / "run_plant_calibration.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_matrix_is_sum_of_product_columns():
    drv = _driver()
    names = ["ua_bed_w_per_k", "offset_product_1_k", "offset_product_2_k", "freeboard_ua_w_per_k"]
    s = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, -25.0, 0.0, -3.0],
            [2.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, -25.0, -4.0],
        ]
    )
    shared, new_names = drv.shared_matrix(s, names)
    assert new_names == ["ua_bed_w_per_k", "offset_shared_k", "freeboard_ua_w_per_k"]
    np.testing.assert_allclose(shared[:, 1], [0.0, -25.0, 0.0, -25.0])
    np.testing.assert_allclose(shared[:, [0, 2]], s[:, [0, 3]])


def test_analysis_reading_on_synthetic_record(tmp_path):
    """The analysis applies the reading to a synthetic record (no model runs)."""
    sys.path.insert(0, str(ROOT / "scripts" / "verify"))
    spec = importlib.util.spec_from_file_location(
        "analyze_plant_calibration", ROOT / "scripts" / "analyze_plant_calibration.py"
    )
    assert spec is not None and spec.loader is not None
    ana = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ana)

    def block(bed: float, exh: float) -> dict:
        return error_metrics(np.array([bed, exh]), np.zeros(2))

    variant = {
        d: {
            "selected": ["ua_bed_w_per_k"],
            "fit": {"theta": {}, "at_bound": []},
            "own_day": block(0.1, 0.1),
            "other_day": block(1.0, 1.0),
        }
        for d in ("day_1", "day_2")
    }
    record = {
        "calibration": {
            "variants": {"main": variant, "shared": variant},
            "bias": {
                d: {"other_day": block(3.0, 3.0), "offsets_k": {"bed": 0.0, "exhaust": 0.0}}
                for d in ("day_1", "day_2")
            },
            "uncalibrated": {d: block(5.0, 5.0) for d in ("day_1", "day_2")},
        }
    }
    rows = ana.cross_table(record, "main")
    assert set(rows) == {"day_1->day_2", "day_2->day_1"}
    assert ana.apply_reading(rows, 2.0)["reading"] == "K0"
