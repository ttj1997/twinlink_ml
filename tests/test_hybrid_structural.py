"""Tests of the study docs/41_hybrid_structural.md (structurally deviating truth, model M2b).

The truth: switched off it is the grey-box model bit for bit; switched on, simulator, steady
config and the exhaust balance of Befund 117 agree. The models and the drivers of Befund 122 are
tested in ``tests/test_hybrid_model.py``; here only what this study adds.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation.experiments import DataSpec, generate_data
from twinlink_ml.hybrid_model.maps import GreyBoxMap
from twinlink_ml.hybrid_model.refit import refit_config
from twinlink_ml.hybrid_model.residual import replay
from twinlink_ml.hybrid_model.study import load_hybrid_config, widen
from twinlink_ml.hybrid_structural.bias import BiasedMap, training_bias
from twinlink_ml.hybrid_structural.study import (
    DEFAULT_CONFIG_PATH,
    MODELS,
    StructuralSetup,
    build_structural_setup,
    freeboard_share,
    load_structural_config,
)
from twinlink_ml.hybrid_structural.truth import (
    FreeboardEvaporationGranulator,
    bed_feed,
    exhaust_temperature_celsius,
    steady_config,
)
from twinlink_ml.rl.env import FIXED_NOZZLE_PRESSURE_BAR
from twinlink_ml.sensor_value.draws import truth_config
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
    load_process_config,
)
from twinlink_ml.simulation.steady_state import solve_steady_state

ROOT = Path(__file__).resolve().parents[1]
SHARE = 0.3
TARGETS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
TRUE_VALUES = {"critical_moisture_kg_per_kg": 0.24, "max_saturation_degree": 0.48}


@pytest.fixture(scope="module")
def nominal():  # type: ignore[no-untyped-def]
    cfg = load_structural_config(DEFAULT_CONFIG_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(cfg.process_config)
    return dataclasses.replace(
        config, numerics=dataclasses.replace(config.numerics, enforce_gas_velocity_projection=True)
    )


def _spec() -> DataSpec:
    return DataSpec.new_study(
        sample_time_s=25.0, run_duration_s=6000.0, n_train_max=4, n_val=2, n_test=2,
        history_length=8, horizon=6, stride=4, target_names=TARGETS,
    )  # fmt: skip


def _pool(config, share: float | None):  # type: ignore[no-untyped-def]
    factory = None if share is None else (lambda c: FreeboardEvaporationGranulator(c, share))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return generate_data(config, _spec(), master_seed=1, simulator_factory=factory)


@pytest.fixture(scope="module")
def pools(nominal):  # type: ignore[no-untyped-def]
    truth = truth_config(nominal, TRUE_VALUES)
    return _pool(truth, None), _pool(truth, 0.0), _pool(truth, SHARE)


def test_bed_feed_keeps_the_solid_feed_and_removes_the_share_of_the_water() -> None:
    spray, x_s = np.array([1.5, 0.4, 3.0]), np.array([0.30, 0.20, 0.40])
    bed_spray, bed_x_s, water = bed_feed(spray, x_s, SHARE)
    assert np.allclose(bed_spray * bed_x_s, spray * x_s, rtol=1e-15)
    assert np.allclose(bed_spray * (1.0 - bed_x_s), (1.0 - SHARE) * spray * (1.0 - x_s))
    assert np.allclose(water, SHARE * spray * (1.0 - x_s))
    assert bed_feed(1.5, 0.3, 0.0) == (1.5, 0.3, 0.0)


def test_share_must_lie_in_the_unit_interval(nominal) -> None:  # type: ignore[no-untyped-def]
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="share"):
            steady_config(nominal, bad)
        with pytest.raises(ValueError, match="share"), warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            FreeboardEvaporationGranulator(nominal, bad)


def test_switched_off_the_truth_is_the_grey_box_model_bit_for_bit(nominal, pools) -> None:  # type: ignore[no-untyped-def]
    plain, off, on = pools
    for name in ("outputs", "measurements", "actions", "disturbances", "states"):
        a, b = getattr(plain.trajectory, name, None), getattr(off.trajectory, name, None)
        if a is not None:
            assert np.array_equal(a, b), name
    assert plain.fingerprints == off.fingerprints
    assert steady_config(nominal, 0.0) is nominal
    # switched on it is another truth (otherwise the test would be empty), on the same excitation
    assert np.array_equal(plain.trajectory.disturbances, on.trajectory.disturbances)
    assert plain.split.digest == on.split.digest
    assert not np.array_equal(plain.trajectory.outputs, on.trajectory.outputs)


def test_the_truth_is_drier_and_warmer_than_the_model_without_the_term(pools) -> None:  # type: ignore[no-untyped-def]
    plain, _, on = pools
    names = list(plain.trajectory.metadata["output_names"])
    moisture = names.index("product_moisture_kg_per_kg")
    bed = names.index("bed_temperature_celsius")
    late = slice(plain.trajectory.outputs.shape[1] // 2, None)
    assert (
        on.trajectory.outputs[:, late, moisture].mean()
        < plain.trajectory.outputs[:, late, moisture].mean()
    )
    assert (
        on.trajectory.outputs[:, late, bed].mean() > plain.trajectory.outputs[:, late, bed].mean()
    )


def test_steady_config_is_the_steady_state_of_the_simulator(nominal) -> None:  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        by_config = solve_steady_state(GreyBoxFluidBedGranulator(steady_config(nominal, SHARE)))
        simulator = FreeboardEvaporationGranulator(nominal, SHARE)
        by_simulator = solve_steady_state(simulator)
        plain = solve_steady_state(GreyBoxFluidBedGranulator(nominal))
    assert by_config.particle_moisture_kg_per_kg == pytest.approx(
        by_simulator.particle_moisture_kg_per_kg, rel=1e-12
    )
    assert by_config.bed_temperature_celsius == pytest.approx(
        by_simulator.bed_temperature_celsius, rel=1e-12
    )
    assert by_config.particle_moisture_kg_per_kg < plain.particle_moisture_kg_per_kg
    # the commanded inputs at the steady state of the config: the simulator is at rest there
    derivative, _ = simulator.evaluate_derivative(
        np.asarray(by_config.state, dtype=np.float64).reshape(1, -1),
        simulator.nominal_action(1),
        simulator.nominal_disturbance(1),
    )
    assert by_config.residual_inf_norm < 1e-5
    reference, _ = GreyBoxFluidBedGranulator(steady_config(nominal, SHARE)).evaluate_derivative(
        np.asarray(by_config.state, dtype=np.float64).reshape(1, -1),
        GreyBoxFluidBedGranulator(steady_config(nominal, SHARE)).nominal_action(1),
        GreyBoxFluidBedGranulator(steady_config(nominal, SHARE)).nominal_disturbance(1),
    )
    assert np.allclose(np.asarray(derivative), np.asarray(reference), rtol=1e-9, atol=1e-15)


def test_steady_config_changes_nothing_but_the_two_nominals(nominal) -> None:  # type: ignore[no-untyped-def]
    changed = steady_config(nominal, SHARE)
    spray, x_s, _ = bed_feed(1.5, 0.30, SHARE)
    assert changed.actions.spray_rate_kg_per_h.nominal == pytest.approx(spray)
    assert changed.disturbances.suspension_solid_fraction.nominal == pytest.approx(x_s)
    back = dataclasses.replace(
        changed,
        actions=dataclasses.replace(
            changed.actions, spray_rate_kg_per_h=nominal.actions.spray_rate_kg_per_h
        ),
        disturbances=dataclasses.replace(
            changed.disturbances,
            suspension_solid_fraction=nominal.disturbances.suspension_solid_fraction,
        ),
    )
    assert back == nominal


def test_exhaust_balance_is_the_one_of_befund_117(nominal) -> None:  # type: ignore[no-untyped-def]
    from twinlink_ml.observability.core import steady_measurements
    from twinlink_ml.plant_scale.model import Freeboard, exhaust_after_freeboard

    steady = steady_measurements(steady_config(nominal, SHARE))
    _, _, water_kg_h = bed_feed(1.5, 0.30, SHARE)
    expected, _, _ = exhaust_after_freeboard(
        nominal,
        steady,
        Freeboard("evaporation", SHARE),
        ambient_celsius=20.0,
        freeboard_water_kg_per_s=water_kg_h / 3600.0,
        feed_celsius=nominal.operation.suspension_temperature_celsius,
    )
    got = exhaust_temperature_celsius(
        nominal,
        bed_temperature_celsius=steady.bed_temperature_celsius,
        dry_air_flow_kg_per_s=steady.dry_air_flow_kg_per_s,
        outlet_humidity_kg_per_kg=steady.outlet_humidity_balance_kg_per_kg,
        freeboard_water_kg_per_s=water_kg_h / 3600.0,
    )
    assert got == pytest.approx(expected, rel=1e-12)
    assert 5.0 <= steady.bed_temperature_celsius - got <= 10.0


# --- model M2b ----------------------------------------------------------------------------------


def _map(config) -> GreyBoxMap:  # type: ignore[no-untyped-def]
    return GreyBoxMap(
        config,
        widen(config, (0.0, 40.0), (0.10, 0.95)),
        nozzle_pressure_bar=FIXED_NOZZLE_PRESSURE_BAR,
        quality_limit_kg_per_kg=0.15,
    )


def test_m2b_with_zero_bias_equals_m2_and_shifts_only_the_moisture(nominal) -> None:  # type: ignore[no-untyped-def]
    base = _map(nominal)
    actions = np.array([[90.0, 1.6], [75.0, 1.1], [119.0, 2.4]])
    contexts = np.array([[20.0, 0.5], [8.0, 0.8], [33.0, 0.3]])
    expected = base.predict(actions, contexts, wide=False)
    zero = BiasedMap(base, 0.0).predict(actions, contexts, wide=False)
    shifted = BiasedMap(base, -0.01).predict(actions, contexts, wide=False)
    for field in dataclasses.fields(expected):
        assert np.array_equal(
            getattr(expected, field.name), getattr(zero, field.name), equal_nan=True
        ), field.name
        if field.name != "moisture_kg_per_kg":
            assert np.array_equal(
                getattr(expected, field.name), getattr(shifted, field.name), equal_nan=True
            ), field.name
    assert np.allclose(shifted.moisture_kg_per_kg, expected.moisture_kg_per_kg - 0.01)
    with pytest.raises(ValueError, match="finite"):
        BiasedMap(base, float("nan"))


def test_training_bias_is_the_mean_residual_of_the_replay(nominal, pools) -> None:  # type: ignore[no-untyped-def]
    plain, _, on = pools
    rows = [0, 1, 2, 3]
    m2 = refit_config(nominal, list(TRUE_VALUES), list(TRUE_VALUES.values()))
    # without the term the true parameters leave only the sensor noise
    without = training_bias(plain, m2, rows, TARGETS)
    assert abs(without["product_moisture_kg_per_kg"]) < 2e-3
    # with the term the refitted structure is too wet: the bias of the moisture is negative
    with_term = training_bias(on, m2, rows, TARGETS)
    names = list(on.trajectory.metadata["output_names"])
    col = names.index("product_moisture_kg_per_kg")
    expected = on.trajectory.measurements[rows][..., col] - replay(on, m2, rows).outputs[..., col]
    assert with_term["product_moisture_kg_per_kg"] == pytest.approx(float(expected.mean()))
    assert with_term["product_moisture_kg_per_kg"] < -1e-3
    assert set(with_term) == set(TARGETS)


# --- config and set-up --------------------------------------------------------------------------


def test_committed_config_differs_from_befund_122_only_where_stated() -> None:
    new = load_structural_config(DEFAULT_CONFIG_PATH)
    old = load_hybrid_config()
    assert new.master_seeds == (1, 2, 3, 4, 5) and new.n_train == (4, 8, 16)
    assert MODELS == ("M0", "M1", "M2", "M2b", "M3") and new.ensemble_members == 1
    same = set(old.raw) - {"data", "models"}
    assert {k: new.raw[k] for k in same} == {k: old.raw[k] for k in same}
    assert new.raw["data"]["max_epochs"] == old.raw["data"]["max_epochs"]
    assert set(new.raw) - set(old.raw) == {"structure"}


def test_share_of_the_config_is_the_one_selected_by_the_committed_rule() -> None:
    artifact = ROOT / "artifacts" / "runs" / "hybrid_structural_share_v1.json"
    config = load_structural_config(DEFAULT_CONFIG_PATH)
    structure = config.raw["structure"]
    if not artifact.exists():
        assert structure["share"] is None  # not fixed yet: no set-up, no run
        with pytest.raises(ValueError, match="probe_hybrid_structural_share"):
            freeboard_share(config)
        return
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["provenance"]["git/dirty"] is False
    assert result["rule"]["candidates"] == structure["share_candidates"]
    assert result["rule"]["target_gap_k"] == structure["target_gap_k"]
    assert freeboard_share(config) == result["selected_share"]
    low, high = structure["accepted_gap_k"]
    assert low <= result["selected_gap_k"] <= high and result["selected_within_accepted_band"]
    gaps = {c["share"]: abs(c["gap_k"] - structure["target_gap_k"]) for c in result["candidates"]}
    assert min(gaps, key=lambda s: (gaps[s], s)) == result["selected_share"]


def test_setup_has_the_truth_with_the_term_and_the_prior_without(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    config = load_structural_config(DEFAULT_CONFIG_PATH)
    if config.raw["structure"]["share"] is None:
        raw = {**config.raw, "structure": {**config.raw["structure"], "share": SHARE}}
        config = dataclasses.replace(config, raw=raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        setup = build_structural_setup(config)
    assert isinstance(setup, StructuralSetup) and setup.share == freeboard_share(config)
    assert len(setup.draws) == 20 and len(setup.cases) == 140
    truth = setup.steady_truth(3)
    material = dataclasses.replace(setup.steady_nominal.material, **setup.draw_values(3))
    assert truth.material == material
    spray, x_s, _ = bed_feed(1.5, 0.30, setup.share)
    assert truth.actions.spray_rate_kg_per_h.nominal == pytest.approx(spray)
    assert truth.disturbances.suspension_solid_fraction.nominal == pytest.approx(x_s)
    # prior knowledge and the data config (excitation design) are those of Befund 122
    assert setup.steady_nominal.actions.spray_rate_kg_per_h.nominal == 1.5
    assert setup.data_truth(3).actions.spray_rate_kg_per_h.nominal == 1.5


# --- packed records, drivers, reading -----------------------------------------------------------


def _script(name: str):  # type: ignore[no-untyped-def]
    import importlib

    for extra in (ROOT / "scripts" / "verify", ROOT / "scripts"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    return importlib.import_module(name)


def test_packed_records_append_read_and_repair(tmp_path: Path) -> None:
    from twinlink_ml.hybrid_structural.records import (
        append_record,
        complete_length,
        read_records,
        repair_tail,
    )

    path = tmp_path / "out.jsonl.gz"
    assert read_records(path) == [] and repair_tail(path) == 0
    records = [
        {"kind": "task", "key": f"k{i}", "x": np.float64(i), "v": np.arange(3)} for i in range(5)
    ]
    for r in records:
        append_record(path, r)
    got = read_records(path)
    assert [r["key"] for r in got] == [f"k{i}" for i in range(5)] and got[2]["v"] == [0, 1, 2]
    size = path.stat().st_size
    assert complete_length(path) == size and repair_tail(path) == 0
    # a process killed in the middle of a write leaves an incomplete last member
    whole = path.read_bytes()
    extra = gzip.compress(json.dumps({"kind": "task", "key": "k5"}).encode() + b"\n")
    path.write_bytes(whole + extra[: len(extra) // 2])
    assert complete_length(path) == size
    with pytest.raises((EOFError, OSError)):
        read_records(path)
    assert repair_tail(path) == len(extra) // 2
    assert path.read_bytes() == whole
    append_record(path, {"kind": "task", "key": "k5"})
    assert [r["key"] for r in read_records(path)][-1] == "k5"
    with pytest.raises(ValueError, match="gz"):
        append_record(tmp_path / "plain.jsonl", {})


def test_training_units_put_n_train_4_last_and_cover_every_unit() -> None:
    train = _script("train_hybrid_structural")
    config = load_structural_config(DEFAULT_CONFIG_PATH)
    units = train.unit_list(list(range(20)), list(config.n_train), list(config.master_seeds))
    assert len(units) == len(set(units)) == 20 * 5 * 3 * 3
    sizes = [u[2] for u in units]
    assert sizes == sorted(sizes, reverse=True)  # the budget rule cuts n_train 4 from the tail
    assert {u[3] for u in units} == {"M1", "M3", "M2"}
    assert train.unit_key(3, 2, 8, "M2") == "d03_s2_n8_M2"
    assert Path("C:/twinlink_runs/hybrid-structural/checkpoints") == train.CHECKPOINT_ROOT


def test_evaluation_driver_uses_the_tasks_of_befund_122(tmp_path: Path) -> None:
    ev = _script("run_hybrid_structural_evaluation")
    base = ev.base
    assert ev.output_name("margins", "p", None) == "p_margins_per_draw"
    assert ev.output_name("errors", "p", None) == "p_errors"
    assert ev.output_name("oracle", "p", "n") == "n"
    # the scheduler is the one of Befund 122; only the record form differs
    from twinlink_ml.hybrid_model.grid_rto import BisectionSettings, GridSettings
    from twinlink_ml.hybrid_structural.records import read_records

    run = base.MarginRun(
        "o", "t", GridSettings(), BisectionSettings(0.12, 1e-4), BisectionSettings(10.0, 0.01)
    )
    out = tmp_path / "m.jsonl.gz"
    sched = ev.PackedScheduler(out, run, 0.95)
    for m in MODELS:
        sched.add_cell(base.Cell(m, 1, 4, (0,)), {})
        sched.add_cell(base.Cell(m, 1, 16, (0,)), {})
    order = []
    while (task := sched.pop()) is not None:
        key, fn, _ = task
        assert fn is base.margin_search_task
        order.append(key)
    # most expensive first; M0 and, within a model, n_train 4 last
    assert order[0] == base.search_key("M3", 1, 16, 0)
    assert order[-1] == base.search_key("M0", 1, 4, 0)
    assert order.index(base.search_key("M2b", 1, 16, 0)) > order.index(
        base.search_key("M2", 1, 16, 0)
    )
    sched.write({"kind": "footer", "n": 1})
    assert read_records(out) == [{"kind": "footer", "n": 1}]


def test_set_up_is_installed_for_the_tasks_of_befund_122(monkeypatch: pytest.MonkeyPatch) -> None:
    ev = _script("run_hybrid_structural_evaluation")
    base = ev.base
    monkeypatch.setattr(base, "_SETUP", None)
    monkeypatch.setattr(base, "_SETUP_DIGEST", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        installed = ev.setup()
        assert isinstance(installed, StructuralSetup) and base.setup() is installed
        base.reset_task_state()  # the digest matches
        truth = base.truth_map(0).config
        prior = base.nominal_map().config
    spray, _, _ = bed_feed(1.5, 0.30, installed.share)
    assert truth.actions.spray_rate_kg_per_h.nominal == pytest.approx(spray)
    assert prior.actions.spray_rate_kg_per_h.nominal == 1.5
    # another set-up cached in the process is an error, not silently used
    monkeypatch.setattr(base, "_SETUP", object())
    with pytest.raises(RuntimeError, match="another study set-up"):
        ev.setup()


def test_m2b_map_of_the_driver_is_m2_plus_the_recorded_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ev = _script("run_hybrid_structural_evaluation")
    base = ev.base
    monkeypatch.setattr(base, "_SETUP", None)
    monkeypatch.setattr(base, "_SETUP_DIGEST", None)
    records = {
        "d00_s1_n4_M2": {
            "refit": {
                "names": ["critical_moisture_kg_per_kg", "max_saturation_degree"],
                "theta": [0.24, 0.48],
            },
            "moisture_bias_kg_per_kg": -0.004,
        }
    }
    actions, contexts = np.array([[90.0, 1.6], [70.0, 1.0]]), np.array([[20.0, 0.5], [10.0, 0.7]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        ev.setup()
        m2 = base.build_map("M2", 0, 1, 4, records).predict(actions, contexts, wide=False)
        m2b = base.build_map("M2b", 0, 1, 4, records).predict(actions, contexts, wide=False)
    assert np.allclose(m2b.moisture_kg_per_kg, m2.moisture_kg_per_kg - 0.004, rtol=0, atol=1e-15)
    assert np.array_equal(m2b.energy_kw, m2.energy_kw) and np.array_equal(m2b.valid, m2.valid)


def _rows(margins: dict[str, dict[int, list[float]]]) -> dict[tuple[str, int], list[dict]]:  # type: ignore[type-arg]
    table: dict[tuple[str, int], list[dict]] = {}  # type: ignore[type-arg]
    for model, per_n in margins.items():
        for n, values in per_n.items():
            table[(model, n)] = [
                {"seed": s + 1, "determinable": True, "margin_censored": v}
                for s, v in enumerate(values)
            ]
    return table


FLAT = [0.050, 0.052, 0.049, 0.051, 0.050]
FLAT_B = [0.049, 0.053, 0.050, 0.050, 0.051]
FLAT_C = [0.052, 0.049, 0.051, 0.050, 0.050]
SWING = [0.2, 0.01, 0.2, 0.01, 0.2]


def test_holm_adjusted_p_values() -> None:
    an = _script("analyze_hybrid_structural")
    assert an.holm_adjusted([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    got = an.holm_adjusted([0.02, float("nan"), 0.5])
    assert got[0] == pytest.approx(0.04) and np.isnan(got[1]) and got[2] == pytest.approx(0.5)


def test_reading_z0_needs_both_references_at_the_same_n_train() -> None:
    an = _script("analyze_hybrid_structural")
    small = [0.010, 0.011, 0.009, 0.012, 0.010]
    m3 = {4: small, 8: FLAT, 16: FLAT}
    m2 = {4: FLAT, 8: FLAT_B, 16: FLAT_C}
    both = an.reading(
        _rows(
            {
                "M3": m3,
                "M2": m2,
                "M2b": {4: [0.040, 0.043, 0.041, 0.042, 0.040], 8: FLAT_C, 16: FLAT_B},
            }
        )
    )
    assert both["verdict"] == "Z0" and both["z0_n_train"] == [4]
    t = both["families"]["M3_vs_M2"][4]
    assert t["resolved"] and t["candidate_smaller"] and t["n_consistent"] == 5
    assert t["family_size"] == 3 and t["p_holm"] >= t["p_value"] and t["threshold"] > 0
    # resolved against M2 only: not Z0
    one = an.reading(
        _rows(
            {
                "M3": m3,
                "M2": m2,
                "M2b": {4: [0.011, 0.009, 0.010, 0.012, 0.011], 8: FLAT_C, 16: FLAT_B},
            }
        )
    )
    assert one["verdict"] != "Z0" and one["z0_n_train"] == []


def test_reading_z1_z2_and_mixed() -> None:
    an = _script("analyze_hybrid_structural")
    small = [0.004, 0.005, 0.003, 0.004, 0.006]
    noisy = [0.050, 0.020, 0.080, 0.035, 0.065]
    flat = {4: FLAT, 8: FLAT, 16: FLAT}
    # M2 resolved smaller than M3 everywhere: Z1 for M2
    z1 = an.reading(
        _rows(
            {
                "M3": flat,
                "M2": {4: small, 8: small, 16: small},
                "M2b": {4: SWING, 8: SWING, 16: SWING},
            }
        )
    )
    assert z1["verdict"] == "Z1" and z1["z1_models"] == ["M2"]
    assert all(z1["families"]["M3_vs_M2"][n]["candidate_larger"] for n in (4, 8, 16))
    # nothing resolved: Z2, whatever the medians say
    z2 = an.reading(
        _rows(
            {
                "M3": {4: noisy, 8: noisy, 16: noisy},
                "M2": {4: noisy[::-1], 8: noisy[1:] + noisy[:1], 16: noisy[::-1]},
                "M2b": {4: noisy[2:] + noisy[:2], 8: noisy[::-1], 16: noisy[1:] + noisy[:1]},
            }
        )
    )
    assert z2["verdict"] == "Z2" and not z2["any_resolved"]
    assert all(np.isfinite(t["threshold"]) for f in z2["families"].values() for t in f.values())
    # M3 resolved smaller than M2 at 4, M2 smaller at 16, M2b unresolved with larger medians
    worse = [0.090, 0.092, 0.089, 0.091, 0.090]
    mixed = an.reading(
        _rows(
            {
                "M3": flat,
                "M2": {4: worse, 8: FLAT_B, 16: small},
                "M2b": {4: SWING, 8: SWING, 16: SWING},
            }
        )
    )
    assert mixed["verdict"] == "mixed (none of Z0, Z1, Z2)"
    assert mixed["families"]["M3_vs_M2"][4]["candidate_smaller"]
    assert mixed["families"]["M3_vs_M2"][16]["candidate_larger"]


def test_sign_consistency_is_required_and_needs_four_seeds() -> None:
    an = _script("analyze_hybrid_structural")
    cand = dict(enumerate([0.010, 0.010, 0.010, 0.010, 0.010], start=1))
    ref = dict(enumerate([0.013, 0.013, 0.013, 0.009, 0.009], start=1))
    t = an.resolve_family({4: an.paired_test(cand, ref)})[4]
    assert t["n_consistent"] == 3 and not t["sign_consistent"] and not t["resolved"]
    three = an.resolve_family(
        {4: an.paired_test({1: 0.01, 2: 0.011, 3: 0.009}, {1: 0.05, 2: 0.052, 3: 0.049})}
    )[4]
    assert three["holm_rejects"] and not three["resolved"]  # fewer than 4 seeds
    same = an.resolve_family({4: an.paired_test({1: 0.25, 2: 0.5}, {1: 0.5, 2: 0.75})})[4]
    assert same["degenerate"] and not same["resolved"]


def test_m2_against_befund_122_is_paired_on_the_common_seeds() -> None:
    an = _script("analyze_hybrid_structural")
    table = _rows({"M2": {4: [0.010, 0.020, 0.030, 0.040, 0.050]}})
    reference = {
        "curves": {
            "M2/n4": {
                "margin_censored": {
                    "median": 0.002,
                    "min": 0.001,
                    "max": 0.05,
                    "n_seeds": 3,
                    "per_seed": {"1": 0.001, "2": 0.05, "3": 0.002},
                }
            }
        }
    }
    got = an.compare_with_befund_122(table, reference)
    assert got is not None and set(got) == {"n4"}
    assert got["n4"]["median_here"] == pytest.approx(0.030)
    assert got["n4"]["difference_of_medians"] == pytest.approx(0.028)
    assert got["n4"]["common_seeds"] == [1, 2, 3]
    assert got["n4"]["paired_differences_here_minus_122"] == pytest.approx([0.009, -0.03, 0.028])
    assert an.compare_with_befund_122(table, None) is None
