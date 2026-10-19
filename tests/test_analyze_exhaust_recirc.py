"""Analysis script of the exhaust-recirculation study (docs/22): transfer grid and statistics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module() -> ModuleType:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify"))
    spec = importlib.util.spec_from_file_location(
        "analyze_exhaust_recirc", REPO_ROOT / "scripts" / "analyze_exhaust_recirc.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_transfer_placeholder_has_value_band_source_and_status() -> None:
    module = _module()
    for name, entry in module.TRANSFER.items():
        assert set(entry) == {"value", "band", "source", "status"}, name
        assert entry["source"] and entry["status"], name


def test_transfer_scales_linearly_and_the_two_readings_differ_by_the_factor() -> None:
    module = _module()
    config = module._config()
    out = module.transfer_tables(config, {"a": 0.1, "b": 0.2})
    rows = {(r["plant"], r["saving_fraction_label"]): r for r in out["rows"]}
    assert rows[("A", "b")]["fuel_saving_mwh_per_year"] == pytest.approx(
        2.0 * rows[("A", "a")]["fuel_saving_mwh_per_year"]
    )
    volume = rows[("C_volume_reading", "a")]["fuel_saving_mwh_per_year"]
    annual = rows[("C_annual_reading", "a")]["fuel_saving_mwh_per_year"]
    assert volume / annual == pytest.approx(4.9)
    # plant A heater duty at the placeholders is of the order of the measured process air
    # (Befund 99: 798-957 kW on the two days)
    assert 600.0 < out["plants"]["A"]["heater_duty_kw"] < 1000.0
    co2 = rows[("A", "a")]
    assert co2["co2_saving_t_per_year"] == pytest.approx(co2["fuel_saving_mwh_per_year"] * 0.201)
    assert co2["value_keur_per_year"]["300"] == pytest.approx(co2["co2_saving_t_per_year"] * 0.3)


def test_mixing_bound_and_headroom_behave() -> None:
    module = _module()
    out = module.transfer_tables(module._config(), {"a": 0.1})
    bound = [r for r in out["mixing_upper_bound"] if r["inlet"] == "A" and r["r"] == 0.4]
    bounds = sorted(bound, key=lambda r: r["exhaust_celsius"])
    assert [r["duty_saving_pct_upper_bound"] for r in bounds] == sorted(
        r["duty_saving_pct_upper_bound"] for r in bounds
    )
    for row in out["humidity_headroom"]:
        assert 0.0 <= row["r_max"] < 1.0
    saturated = [
        r
        for r in out["humidity_headroom"]
        if r["exhaust_celsius"] == 34.4
        and r["humidity_pickup_kg_per_kg"] == 0.020
        and r["exhaust_rh_limit"] == 0.6
    ]
    assert saturated
    assert all(r["r_max"] == 0.0 for r in saturated)


def test_stats_of_an_empty_list() -> None:
    module = _module()
    assert module._stats([]) == {"n": 0, "median": None, "min": None, "max": None}
    assert module._stats([1.0, 3.0])["median"] == 2.0
