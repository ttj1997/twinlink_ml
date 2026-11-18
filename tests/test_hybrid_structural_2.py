"""Tests of the study docs/42_hybrid_structural_2.md (structural error to the unsafe side).

The truth is the grey-box model with its existing recirculation switch; the switch itself is tested
in ``tests/test_exhaust_recirculation.py`` (off = bit-identical). Here: the set-up of the study, the
rule of the strength, the cost measure and the pre-registered reading E0-E3. Models and drivers
are tested in ``tests/test_hybrid_model.py`` and ``tests/test_hybrid_structural.py``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twinlink_ml.hybrid_structural.study import (
    DEFAULT_CONFIG_PATH as BEFUND_124_CONFIG_PATH,
)
from twinlink_ml.hybrid_structural.study import (
    MODELS,
    StructuralSetup,
    build_structural_setup,
    load_structural_config,
)
from twinlink_ml.hybrid_structural_2.study import (
    DEFAULT_CONFIG_PATH,
    RecirculationSetup,
    load_recirculation_config,
    recirculation_fraction,
    with_recirculation,
)
from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    PlaceholderConfigWarning,
)
from twinlink_ml.simulation.config import config_fingerprint
from twinlink_ml.simulation.steady_state import solve_steady_state

ROOT = Path(__file__).resolve().parents[1]
FRACTION = 0.4
ARTIFACT = ROOT / "artifacts" / "runs" / "hybrid_structural_2_strength_v1.json"


def _script(name: str):  # type: ignore[no-untyped-def]
    import importlib

    for extra in (ROOT / "scripts" / "verify", ROOT / "scripts"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    return importlib.import_module(name)


def _config():  # type: ignore[no-untyped-def]
    config = load_recirculation_config(DEFAULT_CONFIG_PATH)
    if config.raw["structure"]["fraction"] is None:
        raw = {**config.raw, "structure": {**config.raw["structure"], "fraction": FRACTION}}
        config = dataclasses.replace(config, raw=raw)
    return config


@pytest.fixture(scope="module")
def setup():  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return build_structural_setup(_config())


# --- truth and set-up ---------------------------------------------------------------------------


def test_committed_config_differs_from_befund_124_only_in_the_structure() -> None:
    new = load_recirculation_config(DEFAULT_CONFIG_PATH)
    old = load_structural_config(BEFUND_124_CONFIG_PATH)
    assert set(new.raw) == set(old.raw)
    same = set(old.raw) - {"structure"}
    assert {k: new.raw[k] for k in same} == {k: old.raw[k] for k in same}
    assert new.raw["structure"]["mode"] == "recirculation"
    assert old.raw["structure"]["mode"] == "evaporation"
    assert new.master_seeds == (1, 2, 3, 4, 5) and new.n_train == (4, 8, 16)
    assert MODELS == ("M0", "M1", "M2", "M2b", "M3")


def test_fraction_of_the_config_is_the_one_selected_by_the_committed_rule() -> None:
    config = load_recirculation_config(DEFAULT_CONFIG_PATH)
    structure = config.raw["structure"]
    assert structure["target_bias_kg_per_kg"] == -0.05
    assert structure["accepted_bias_kg_per_kg"] == [-0.07, -0.03]
    if not ARTIFACT.exists():
        assert structure["fraction"] is None  # not fixed yet: no set-up, no run
        with pytest.raises(ValueError, match="probe_hybrid_structural_2_strength"):
            recirculation_fraction(config)
        return
    result = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert result["provenance"]["git/dirty"] is False
    assert result["rule"]["candidates"] == structure["fraction_candidates"]
    assert result["rule"]["target_bias_kg_per_kg"] == structure["target_bias_kg_per_kg"]
    assert recirculation_fraction(config) == result["selected_fraction"]
    low, high = structure["accepted_bias_kg_per_kg"]
    assert low <= result["selected_median_bias_kg_per_kg"] <= high
    assert result["selected_within_accepted_band"]
    probe = _script("probe_hybrid_structural_2_strength")
    again = probe.select(result["candidates"], structure["target_bias_kg_per_kg"])
    assert again["fraction"] == result["selected_fraction"]
    # only the moisture of M0 against the truth: no energy in the artifact
    assert "energy" not in json.dumps(result["candidates"]) + json.dumps(result["without_term"])


def test_with_recirculation_sets_the_switch_and_nothing_else(setup) -> None:  # type: ignore[no-untyped-def]
    nominal = setup.steady_nominal
    assert nominal.operation.exhaust_recirculation_fraction is None
    changed = with_recirculation(nominal, FRACTION)
    assert changed.operation.exhaust_recirculation_fraction == FRACTION
    back = dataclasses.replace(
        changed,
        operation=dataclasses.replace(changed.operation, exhaust_recirculation_fraction=None),
    )
    assert back == nominal and config_fingerprint(back) == config_fingerprint(nominal)
    assert config_fingerprint(changed) != config_fingerprint(nominal)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="recirculation fraction"):
            with_recirculation(nominal, bad)


def test_setup_has_the_truth_with_the_term_and_the_prior_without(setup) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(setup, RecirculationSetup) and not isinstance(setup, StructuralSetup)
    fraction = recirculation_fraction(setup.config)
    assert setup.fraction == fraction
    assert setup.structure_record() == {"mode": "recirculation", "fraction": fraction}
    assert "42_hybrid_structural_2" in setup.study_label
    assert len(setup.draws) == 20 and len(setup.cases) == 140
    for truth, prior in (
        (setup.steady_truth(3), setup.steady_nominal),
        (setup.data_truth(3), setup.data_nominal),
    ):
        assert truth.operation.exhaust_recirculation_fraction == fraction
        assert prior.operation.exhaust_recirculation_fraction is None
        material = dataclasses.replace(prior.material, **setup.draw_values(3))
        expected = with_recirculation(dataclasses.replace(prior, material=material), fraction)
        assert truth == expected  # draw parameters plus the switch, nothing else


def test_set_up_of_befund_124_is_unchanged() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        old = build_structural_setup()
    assert isinstance(old, StructuralSetup)
    assert old.structure_record() == {"mode": "evaporation", "share": old.share}
    assert old.study_label == (
        "hybrid against GRU against refit, structurally deviating truth "
        "(docs/41_hybrid_structural.md)"
    )
    assert old.steady_nominal.operation.exhaust_recirculation_fraction is None


def test_the_truth_is_wetter_than_the_model_without_the_term(setup) -> None:  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        without = solve_steady_state(GreyBoxFluidBedGranulator(setup.steady_nominal))
        with_term = solve_steady_state(
            GreyBoxFluidBedGranulator(with_recirculation(setup.steady_nominal, setup.fraction))
        )
    assert with_term.particle_moisture_kg_per_kg > without.particle_moisture_kg_per_kg
    assert float(with_term.diagnostics["inlet_humidity_kg_per_kg"]) > float(
        without.diagnostics["inlet_humidity_kg_per_kg"]
    )


def test_drivers_take_the_config_and_refuse_a_second_set_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _script("train_hybrid_structural")
    monkeypatch.setattr(train, "_SETUP", None)
    monkeypatch.setattr(train, "_CONFIG_PATH", None)
    train.use_config(str(DEFAULT_CONFIG_PATH))
    assert str(DEFAULT_CONFIG_PATH) == train._CONFIG_PATH
    monkeypatch.setattr(train, "_SETUP", object())
    train.use_config(str(DEFAULT_CONFIG_PATH))  # the same config again: fine
    with pytest.raises(RuntimeError, match="another study set-up"):
        train.use_config(None)
    ev = _script("run_hybrid_structural_evaluation")
    monkeypatch.setattr(ev.base, "_SETUP", object())
    monkeypatch.setattr(ev, "_CONFIG_PATH", None)
    with pytest.raises(RuntimeError, match="another study set-up"):
        ev.use_config(str(DEFAULT_CONFIG_PATH))
    with pytest.raises(RuntimeError, match="another study set-up"):
        ev.setup()  # a cached set-up that is not one of the structural studies


# --- rule of the strength -----------------------------------------------------------------------


def test_strength_rule_takes_the_median_of_the_steady_cases_closest_to_the_target() -> None:
    probe = _script("probe_hybrid_structural_2_strength")

    def row(fraction: float, median: float, missing: int = 0) -> dict[str, Any]:
        return {"fraction": fraction, "bias": {"median": median, "n_without_steady_truth": missing}}

    rows = [row(0.2, -0.01), row(0.3, -0.04), row(0.4, -0.055), row(0.5, -0.051, missing=2)]
    # points without a steady truth are counted and reported, they do not exclude a candidate
    assert probe.select(rows, -0.05)["fraction"] == 0.5
    tie = [row(0.3, -0.03125), row(0.4, -0.09375)]  # binary fractions: an exact tie
    assert probe.select(tie, -0.0625)["fraction"] == 0.3  # tie: the smaller fraction
    with pytest.raises(SystemExit):
        probe.select([row(0.5, float("nan"), missing=100)], -0.05)
    s = probe.summary(np.array([-0.03, -0.05, np.nan, -0.07]))
    assert s["median"] == pytest.approx(-0.05) and s["n_without_steady_truth"] == 1
    assert s["n_cases"] == 4 and s["min"] == pytest.approx(-0.07)
    assert s["median_missing_as_wettest"] == pytest.approx(-0.06)  # nan counted as the wettest


# --- cost measure -------------------------------------------------------------------------------


def _job(
    holds: list[bool],
    excess: list[float],
    *,
    reached: bool = True,
    needed: list[float | None] | None = None,
) -> dict[str, Any]:
    cases = [
        {
            "inside": True,
            "kind": "interior",
            "holds": h,
            "predicted": {"valid": True},
            "truth": {"energy_kw": 10.0 * (1.0 + e)},
            "oracle_energy_kw": 10.0,
            "energy_excess_relative": e,
            "bed_violated": False,
            "moisture_bias_at_action": -0.01,
            "effective_margin": 0.02,
        }
        for h, e in zip(holds, excess, strict=True)
    ]
    return {
        "model": "M3",
        "master_seed": 1,
        "n_train": 16,
        "parameter_kind": "margin",
        "search": {"upper": 0.12, "tolerance": 1e-4},
        "n_inside": len(cases),
        "parameter": 0.02 if reached else None,
        "reached": reached,
        "raised": 0,
        "cases": cases,
        "coarse_bias": [],
        "needed": [{"needed": v} for v in (needed if needed is not None else [0.0] * len(cases))],
    }


def test_cost_is_the_median_energy_ratio_of_an_admissible_cell() -> None:
    an = _script("analyze_hybrid_structural_2")
    excess = [0.10] * 10 + [0.20] * 10
    row = an.cost_row(_job([True] * 19 + [False], excess, needed=[0.0] * 15 + [0.01] * 4 + [None]))
    assert row["admissible"] and row["not_admissible_reason"] is None
    assert row["energy_cost"] == pytest.approx(1.15)
    assert row["violations_at_margin_0"] == [5, 20]  # a margin that is never reached counts too


def test_cells_that_do_not_reach_the_margin_are_not_admissible_and_get_no_value() -> None:
    an = _script("analyze_hybrid_structural_2")
    excess = [0.10] * 20
    not_reached = an.cost_row(_job([True] * 20, excess, reached=False))
    assert not not_reached["admissible"] and not_reached["energy_cost"] is None
    assert "not reached" in not_reached["not_admissible_reason"]
    below = an.cost_row(_job([True] * 18 + [False] * 2, excess))
    assert not below["admissible"] and below["energy_cost"] is None
    assert "fewer than 95 %" in below["not_admissible_reason"]
    few = an.cost_row(_job([True] * 19, excess[:19]))
    assert not few["admissible"] and few["not_admissible_reason"] == "not determinable"
    table = an.tabulate([not_reached, below])
    counts = an.admissibility(table)["M3/n16"]
    assert counts["n_cells"] == 2 and counts["n_admissible"] == 0


# --- reading ------------------------------------------------------------------------------------

A = np.array([0.00, 0.01, -0.01, 0.02, 0.00])
Z = np.array([0.01, -0.01, 0.01, -0.01, 0.00])  # zero mean: not resolved against the same level


def _rows(values: dict[str, dict[int, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    table = {}
    for model, by_n in values.items():
        for n in (4, 8, 16):
            costs = by_n.get(n, [1.1] * 5)  # the same flat level everywhere: degenerate tests
            table[(model, n)] = [
                {
                    "model": model,
                    "n_train": n,
                    "seed": seed,
                    "admissible": cost is not None,
                    "energy_cost": None if cost is None else float(cost),
                }
                for seed, cost in enumerate(costs, start=1)
            ]
    return table


def test_reading_e0_calibration_with_bias_first() -> None:
    an = _script("analyze_hybrid_structural_2")
    m3 = 1.20 + A
    got = an.reading(
        _rows({"M3": {16: m3}, "M2b": {16: 1.08 + 0.5 * A}, "M2": {16: m3 - Z}, "M1": {16: m3 + Z}})
    )
    assert got["verdict"] == "E0" and got["criteria"] == {
        "E0": True,
        "E1": False,
        "E2": False,
        "E3": False,
    }
    t = got["families"]["M3_vs_M2b"][16]
    assert t["resolved"] and t["candidate_larger"] and t["n_consistent"] == 5
    assert t["family_size"] == 1 and t["threshold"] > 0  # the flat n_train are degenerate
    assert got["e0_m2b_resolved_lower_than_m3_at"] == [16]
    # M3 resolved lower than M2b at another n_train: E0 no longer holds
    both = an.reading(
        _rows(
            {
                "M3": {16: m3, 8: 1.05 + A},
                "M2b": {16: 1.08 + 0.5 * A, 8: 1.20 + 0.5 * A},
                "M2": {16: m3 - Z, 8: 1.05 + A - Z},
                "M1": {16: m3 + Z, 8: 1.05 + A + Z},
            }
        )
    )
    assert not both["criteria"]["E0"] and both["verdict"].startswith("mixed")


def test_reading_e1_e2_e3_and_mixed() -> None:
    an = _script("analyze_hybrid_structural_2")
    high, low = 1.20 + A, 1.08 + 0.5 * A
    e1 = an.reading(
        _rows({"M3": {8: low}, "M2b": {8: high}, "M2": {8: high + Z}, "M1": {8: low + Z}})
    )
    assert e1["verdict"] == "E1" and e1["e1_n_train"] == [8]
    # resolved against M2b only: not E1, and nothing else holds -> mixed (none)
    one = an.reading(
        _rows({"M3": {8: low}, "M2b": {8: high}, "M2": {8: low - Z}, "M1": {8: low + Z}})
    )
    assert one["verdict"] == "mixed (none of E0, E1, E2, E3)"
    e2 = an.reading(_rows({"M1": {4: low}, "M2b": {4: high}, "M3": {4: high + Z}, "M2": {4: high}}))
    assert e2["verdict"] == "E2" and e2["e2_n_train"] == [4]
    both = an.reading(
        _rows(
            {
                "M1": {4: 1.02 + 0.2 * A},
                "M3": {4: 1.10 + A},
                "M2b": {4: 1.50 + 0.5 * A},
                "M2": {4: 1.30 + 0.5 * A},
            }
        )
    )
    assert both["criteria"]["E1"] and both["criteria"]["E2"]
    assert both["verdict"] == "mixed (E1 and E2 hold)"
    e3 = an.reading(
        _rows(
            {
                "M3": {16: high},
                "M2b": {16: high + Z},
                "M2": {16: high - Z},
                "M1": {16: high + Z[::-1]},
            }
        )
    )
    assert e3["verdict"] == "E3" and not e3["any_resolved"]


def test_not_admissible_cells_drop_out_without_a_substitute() -> None:
    an = _script("analyze_hybrid_structural_2")
    m3 = 1.20 + A
    m2b: list[Any] = list(1.08 + 0.5 * A)
    m2b[1] = m2b[3] = None  # two cells of M2b not admissible: three pairs left
    got = an.reading(
        _rows({"M3": {16: m3}, "M2b": {16: m2b}, "M2": {16: m3 - Z}, "M1": {16: m3 + Z}})
    )
    t = got["families"]["M3_vs_M2b"][16]
    assert t["n_pairs"] == 3 and t["seeds"] == [1, 3, 5] and t["seeds_dropped_reference"] == [2, 4]
    assert not t["resolved"] and not t["resolvable"]  # fewer than 4 pairs: never resolved
    assert "M3_vs_M2b/n16" in got["tests_not_resolvable_fewer_than_4_pairs"]
    assert got["verdict"] == "E3"


def test_refit_at_bounds_counts_per_parameter() -> None:
    an = _script("analyze_hybrid_structural_2")
    header = {
        "config": {
            "refit": {
                "parameters": ["critical_moisture_kg_per_kg", "max_saturation_degree"],
                "lower": [0.15, 0.20],
                "upper": [0.45, 1.00],
            }
        }
    }
    names = header["config"]["refit"]["parameters"]

    def record(theta: list[float], role: str = "M2") -> dict[str, Any]:
        return {
            "role": role,
            "status": "ok",
            "n_train": 8,
            "refit": {"names": names, "theta": theta},
        }

    got = an.refit_at_bounds(
        [
            record([0.15, 1.0]),
            record([0.30, 1.0]),
            record([0.30, 0.5]),
            {"role": "M1", "status": "ok"},
        ],
        header,
    )["n8"]
    assert got["n_refits"] == 3 and got["at_any_bound"] == 2
    assert got["at_a_bound_of_every_parameter"] == 1
    assert got["per_parameter"]["max_saturation_degree"] == {"at_lower": 0, "at_upper": 2}
