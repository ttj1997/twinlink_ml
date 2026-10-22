"""Sobol study package (docs/25).

Estimator on the Ishigami function, counting of invalid samples, verdict and ranking rules,
configuration, and the two model wrappers at their nominal points.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("SALib")

from twinlink_ml.sensitivity import (
    InputSpec,
    SobolIndices,
    block_valid,
    load_sensitivity_config,
    rank_change,
    saltelli_unit,
    sobol_indices,
    verdict,
)

# Ishigami, a = 7, b = 0.1 (Saltelli et al. 2008): analytic indices.
ISHIGAMI_S1 = np.array([0.3139, 0.4424, 0.0])
ISHIGAMI_ST = np.array([0.5576, 0.4424, 0.2437])
NAMES = ("x1", "x2", "x3")


def _ishigami(unit: np.ndarray) -> np.ndarray:
    x = -math.pi + 2.0 * math.pi * unit
    return np.sin(x[:, 0]) + 7.0 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])


def test_ishigami_indices_match_the_analytic_values() -> None:
    unit = saltelli_unit(3, 13, seed=1)
    y = _ishigami(unit)
    ix = sobol_indices(y, np.ones(y.size, dtype=bool), NAMES, resamples=200, seed=2)
    np.testing.assert_allclose(ix.s1, ISHIGAMI_S1, atol=0.02)
    np.testing.assert_allclose(ix.st, ISHIGAMI_ST, atol=0.02)
    assert ix.n_blocks == ix.n_blocks_valid == 2**13
    assert ix.n_rows_invalid == 0
    # the analytic values lie within the bootstrap intervals (x3 first order: estimator noise)
    assert np.all(np.abs(ix.st - ISHIGAMI_ST) <= 2.0 * ix.st_conf + 1e-3)


def test_design_prefix_is_the_smaller_design() -> None:
    big = saltelli_unit(4, 11, seed=5)
    small = saltelli_unit(4, 9, seed=5)
    np.testing.assert_array_equal(big[: small.shape[0]], small)


def test_invalid_samples_are_counted_and_whole_blocks_dropped() -> None:
    # a fourth, inert input decides validity (about 3 % of the rows): dropping its blocks does
    # not condition the Ishigami inputs, so their indices must stay
    names = (*NAMES, "x4")
    unit = saltelli_unit(4, 12, seed=3)
    y = _ishigami(unit[:, :3])
    invalid = unit[:, 3] > 0.97
    y_bad = y.copy()
    y_bad[invalid] = np.nan  # invalid rows may hold anything, NaN included
    valid = ~invalid
    blocks = block_valid(valid, 4)
    ix = sobol_indices(y_bad, valid, names, resamples=100, seed=4)
    assert ix.n_rows_invalid == int(invalid.sum()) > 0
    assert ix.n_blocks_valid == int(blocks.sum()) < ix.n_blocks
    assert ix.valid_share == pytest.approx(blocks.mean())
    np.testing.assert_allclose(ix.st[:3], ISHIGAMI_ST, atol=0.03)
    assert abs(ix.st[3]) < 0.01
    # a NaN output in a row marked valid counts as invalid as well
    y_nan = y.copy()
    y_nan[0] = np.nan
    ix_nan = sobol_indices(y_nan, np.ones(y.size, dtype=bool), names, resamples=50)
    assert ix_nan.n_rows_invalid == 1
    assert ix_nan.n_blocks_valid == ix_nan.n_blocks - 1


def test_prefix_and_degenerate_outputs() -> None:
    unit = saltelli_unit(3, 10, seed=6)
    y = _ishigami(unit)
    ix = sobol_indices(y, np.ones(y.size, dtype=bool), NAMES, n_blocks=256, resamples=50)
    assert ix.n_blocks == 256
    flat = sobol_indices(np.ones(y.size), np.ones(y.size, dtype=bool), NAMES, resamples=50)
    assert flat.std == 0.0 and np.all(flat.st == 0.0)
    none = sobol_indices(y, np.zeros(y.size, dtype=bool), NAMES, resamples=50)
    assert none.n_blocks_valid == 0 and np.all(np.isnan(none.st))
    with pytest.raises(ValueError):
        block_valid(np.ones(7, dtype=bool), 3)


def _indices(st: list[float], conf: float = 0.01) -> SobolIndices:
    n = len(st)
    arr = np.asarray(st)
    return SobolIndices(
        names=tuple(f"p{i}" for i in range(n)),
        s1=arr * 0.8,
        s1_conf=np.full(n, conf),
        st=arr,
        st_conf=np.full(n, conf),
        n_blocks=100,
        n_blocks_valid=100,
        n_rows_invalid=0,
        mean=0.0,
        std=1.0,
    )


def test_verdict_dominance_and_convergence() -> None:
    v = verdict(_indices([0.5, 0.12, 0.05]), dominant_total_index=0.1, max_ci_width=0.05)
    assert v["dominant"] == ["p0", "p1"]
    assert v["converged"] is True  # full widths 0.02
    wide = verdict(_indices([0.5, 0.12], conf=0.03), dominant_total_index=0.1, max_ci_width=0.05)
    assert wide["converged"] is False  # full width 0.06
    assert (
        verdict(_indices([0.01]), dominant_total_index=0.1, max_ci_width=0.05)["converged"] is None
    )


def test_rank_change_rule() -> None:
    ref = _indices([0.6, 0.3, 0.2, 0.01])
    same = _indices([0.5, 0.25, 0.15, 0.02])
    r = rank_change(ref, same, ref.names, dominant_total_index=0.1, top_k=3)
    assert r["k"] == 3 and not r["changed"] and r["affected"] == []
    swapped = _indices([0.6, 0.2, 0.3, 0.01])
    r = rank_change(ref, swapped, ref.names, dominant_total_index=0.1, top_k=3)
    assert r["changed"] and r["affected"] == ["p1", "p2"]
    assert r["swaps"] == [{"pair": ["p1", "p2"], "resolved": True}]
    close = _indices([0.6, 0.2, 0.205, 0.01], conf=0.02)
    r = rank_change(ref, close, ref.names, dominant_total_index=0.1, top_k=3)
    assert r["changed"] and not r["any_resolved_swap"]


def test_input_mapping_and_widening() -> None:
    lin = InputSpec("a", "epistemic", 1.0, 0.0, 2.0, "-", "test")
    np.testing.assert_allclose(lin.value(np.array([0.0, 0.5, 1.0])), [0.0, 1.0, 2.0])
    log = InputSpec("c", "epistemic", 25.0, 12.5, 50.0, "-", "test", scale="log")
    np.testing.assert_allclose(log.value(np.array([0.0, 0.5, 1.0])), [12.5, 25.0, 50.0])
    assert lin.widened(2.0).low == -1.0 and lin.widened(2.0).high == 3.0
    clipped = InputSpec("a", "epistemic", 1.0, 0.0, 2.0, "-", "test", hard=(0.0, 2.5))
    assert clipped.widened(2.0).low == 0.0 and clipped.widened(2.0).high == 2.5
    wide_log = log.widened(2.0)
    assert wide_log.low == pytest.approx(6.25) and wide_log.high == pytest.approx(100.0)
    with pytest.raises(ValueError):
        InputSpec("b", "other", 1.0, 0.0, 2.0, "-", "test")  # type: ignore[arg-type]


def test_config_loads_with_nominals_inside_the_ranges() -> None:
    cfg = load_sensitivity_config()
    assert set(cfg.inputs) == {"p1", "p2"}
    for specs in cfg.inputs.values():
        for s in specs:
            assert s.rationale
            if not math.isnan(s.nominal):
                assert s.low <= s.nominal <= s.high, s.name
            if s.group == "epistemic":
                assert s.question in cfg.questions, s.name
    assert cfg.fingerprint == load_sensitivity_config().fingerprint
    assert cfg.designs["J"].base_log2_for("p2") < cfg.designs["J"].base_log2_for("p1")


def test_line_model_reproduces_data_set_5_physical_mode() -> None:
    """docs/11 12.6, Combined data set 5: 2.267 / 4.103 kW recovered, 0.7100 / 0.5383 kg/h."""
    from twinlink_ml.sensitivity.models import LineModel

    cfg = load_sensitivity_config()
    expected = {
        "physical_unpressurised_95": (2.267, 0.7100),
        "physical_pressurised": (4.103, 0.5383),
    }
    for mode, (recovered, fuel) in expected.items():
        model = LineModel(cfg.settings["p1"], cfg.inputs["p1"], mode)
        e = model.evaluate({})
        assert e.reasons == ()
        assert all(e.valid.values())
        assert e.outputs["recovered_kw"] == pytest.approx(recovered, abs=5e-4)
        assert e.outputs["fuel_kg_per_h"] == pytest.approx(fuel, abs=5e-5)
    # the heat loss reaches the dryer: a warmer loss lowers the exhaust and the recovery
    model = LineModel(cfg.settings["p1"], cfg.inputs["p1"], "physical_pressurised")
    lossy = model.evaluate({"heat_loss_ua_w_per_k": 3.0})
    assert lossy.outputs["recovered_kw"] < model.evaluate({}).outputs["recovered_kw"]


def test_grey_box_model_nominal_point_and_invalid_counting() -> None:
    pytest.importorskip("casadi")
    from twinlink_ml.sensitivity.models import GreyBoxModel

    cfg = load_sensitivity_config()
    model = GreyBoxModel(cfg.settings["p2"], cfg.inputs["p2"])
    e = model.evaluate({})
    assert e.reasons == () and all(e.valid.values())
    # at the nominal point the fixed command sits at limit - R0 margin
    limit = model.limit
    assert e.outputs["moisture_fixed_kg_per_kg"] == pytest.approx(limit - 0.04355, abs=1e-5)
    assert e.outputs["fixed_violates_limit"] == 0.0
    assert e.outputs["savings_all"] > 0.0
    # a critical moisture below the isotherm maximum is rejected and counted, not dropped
    bad = model.evaluate({"critical_moisture_kg_per_kg": 0.05})
    assert bad.reasons == ("config_rejected:isotherm_or_material_check",)
    assert not any(bad.valid.values())
    # a fixed point that exceeds the limit stays valid for (a) and (d, all), not for (d, feasible)
    wet = model.evaluate({"spray_rate_kg_per_h": 1.8})
    assert wet.outputs["fixed_violates_limit"] == 1.0
    assert wet.valid["savings_all"] and not wet.valid["savings_fixed_feasible"]
