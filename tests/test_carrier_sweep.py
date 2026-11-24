"""Cost-optimal medium flow of one line: settings, pump law, fresh gas, cost curve, optimum, driver.

Every test uses laboratory-scale inputs and the property values of the materials database
shipped with Dyssol; the delivered sheet and the local inputs of the line are never read here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from twinlink_ml.dyssol_steady import carrier_sweep as cs
from twinlink_ml.dyssol_steady.materials import DEFAULT_HEAT_CAPACITY, DyssolProperties
from twinlink_ml.dyssol_steady.reference import LineInputs
from twinlink_ml.evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH
from twinlink_ml.flowsheet.auxiliary import AuxiliaryConfig, pump_power_w
from twinlink_ml.heatpump.config import Economics

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "conf" / "dyssol_steady" / "carrier_sweep_local.yaml.example"

SHOMATE = (1670.36, 0.379262, 0.000377093, -1.40685e-07, 4.5594e06, -1.3926e07, -1.34234e07)
PROPS = DyssolProperties(
    cp_gas=1006.0,
    cp_vapour=1906.0,
    cp_liquid=4185.0,
    cp_solid=DEFAULT_HEAT_CAPACITY,
    heat_of_vaporization_j_per_kg=40660.5 / 0.0180154,
    boiling_point_k=373.15,
    gas_enthalpy_linear=(1005.0, 0.0),
    vapour_enthalpy_shomate=SHOMATE,
    gas_enthalpy_range_k=(200.0, 2773.0),
    vapour_enthalpy_range_k=(290.0, 1700.0),
)
DMDB = "\n".join(
    [
        "DyssolMaterialsDatabase 3",
        "",
        "21 KEYAIR",
        "22 Air",
        "27 207 2 10 10000 1000 1e+09 1006",
        "27 208 3 200 2773 1000 1e+09 1005 0 0",
        "",
        "21 KEYLIQ",
        "22 H2O",
        "26 115 40660.5",
        "27 207 2 10 10000 1000 1e+09 4185",
        "",
        "21 KEYVAP",
        "22 H2O_gas",
        "27 207 2 10 10000 1000 1e+09 1906",
        "27 208 8 290 1700 1000 1e+09 " + " ".join(f"{v:.9g}" for v in SHOMATE),
        "",
        "21 KEYSOL",
        "22 Al2O3",
        "",
    ]
)

HOT: dict[str, Any] = {
    "gas_kg_per_s": 100.0 / 3600.0,
    "gas_moisture_kg_per_s": 0.6 / 3600.0,
    "gas_celsius": 10.0,
    "dryer_inlet_celsius": 150.0,
    "spray_kg_per_s": 3.0 / 3600.0,
    "spray_solid_kg_per_s": 1.5 / 3600.0,
    "spray_celsius": 30.0,
    "heater_efficiency_percent": 80.0,
    "calorific_value_kj_per_kg": 45000.0,
    "equilibrium_moisture_g_per_kg": 30.0,
}
"""A laboratory-scale line whose exhaust stays far from saturation."""
WET = {
    **HOT,
    "dryer_inlet_celsius": 60.0,
    "spray_kg_per_s": 6.0 / 3600.0,
    "spray_solid_kg_per_s": 3.0 / 3600.0,
}
"""A cold line with much spray: over-saturated at the dryer outlet."""

CONFIG = cs.load_config()
LAB_PUMP = cs.PumpLaw(1.0e5, 40.0, 0.40, 1000.0)
LAB_FLOWS = np.linspace(0.0, 200.0, 401)


def lab_curve(inputs: LineInputs, pump: cs.PumpLaw = LAB_PUMP) -> cs.CostCurve:
    return cs.cost_curve(PROPS, inputs, LAB_FLOWS, pump, CONFIG.prices)


# ----------------------------------------------------------------------------- settings
def test_config_takes_the_placeholders_of_the_network_studies_and_of_befund_115() -> None:
    aux, economics = AuxiliaryConfig(), Economics()
    assert CONFIG.pump.efficiency == aux.pump_efficiency == 0.40
    assert CONFIG.pump.density_kg_per_m3 == aux.htm_density_kg_per_m3
    assert CONFIG.pump.reference_pressure_drop_pa == aux.loop_pressure_drop_pa
    assert CONFIG.prices.gas_eur_per_kwh == economics.gas_eur_per_kwh
    assert CONFIG.prices.co2_eur_per_t == economics.co2_eur_per_t
    assert CONFIG.prices.electricity_eur_per_kwh == economics.electricity_eur_per_kwh
    assert CONFIG.prices.gas_co2_kg_per_kwh == GAS_CO2_KG_PER_KWH
    assert CONFIG.prices.electricity_co2_kg_per_kwh == ELECTRICITY_CO2_KG_PER_KWH
    storage = yaml.safe_load(
        (ROOT / "conf" / "storage_price" / "storage_price.yaml").read_text(encoding="utf-8")
    )["economics"]
    for key in ("gas_eur_per_kwh", "co2_eur_per_t", "gas_co2_kg_per_kwh"):
        assert getattr(CONFIG.prices, key) == storage[key]
    assert CONFIG.prices.fuel_eur_per_kwh == pytest.approx(0.051055)


def test_config_scan_is_the_decided_one() -> None:
    flows = CONFIG.flows()
    assert flows[0] == 0.0 and flows[-1] == 15000.0 and len(flows) == 1501
    assert CONFIG.ambient_celsius == (-10.0, 0.0, 10.0, 20.0)
    assert CONFIG.neighbour_factors == (0.8, 1.2)
    assert CONFIG.pressure_factors == (0.5, 1.0, 2.0)
    assert len(CONFIG.fingerprint) == 64


def test_template_names_the_fields_of_the_input_block_and_holds_no_value() -> None:
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    assert set(template["line"]) == set(cs.SHEET_FIELDS)
    assert set(template["line"].values()) == {"VALUE"}
    assert {"workbook", "workbook_sha256", "process"} == set(template["source"])
    with pytest.raises(ValueError, match="expected a number"):
        cs.line_inputs(template["line"])


# ----------------------------------------------------------------------------- line
def test_line_inputs_read_the_gas_flow_as_dry_gas() -> None:
    inputs = cs.line_inputs(HOT, 40.0)
    assert inputs.dry_gas_kg_per_h == pytest.approx(100.0)
    assert inputs.gas_humidity_g_per_kg == pytest.approx(6.0)
    assert inputs.solid_fraction == pytest.approx(0.5)
    assert inputs.spray_kg_per_h == pytest.approx(3.0)
    assert (inputs.gas_celsius, inputs.dryer_inlet_celsius, inputs.spray_celsius) == (
        10.0,
        150.0,
        30.0,
    )
    assert inputs.medium_kg_per_h == 40.0 and cs.line_inputs(HOT).medium_kg_per_h == 0.0
    with pytest.raises(ValueError, match="exactly the fields"):
        cs.line_inputs({k: v for k, v in HOT.items() if k != "spray_celsius"})


def test_saturation_is_checked_at_the_dryer_outlet_and_at_the_exhaust() -> None:
    for medium in (0.0, 20.0, 80.0):
        dryer, exhaust = cs.saturation_ratios(cs.recompute(PROPS, cs.line_inputs(HOT, medium)))
        assert exhaust >= dryer - 1e-12 and (exhaust > dryer) == (medium > 0.0)
        assert exhaust < 1.0
    wet = cs.saturation_ratios(cs.recompute(PROPS, cs.line_inputs(WET, 20.0)))
    assert wet[0] > 1.0
    damp = {
        **HOT,
        "dryer_inlet_celsius": 95.0,
        "spray_kg_per_s": 5.0 / 3600.0,
        "spray_solid_kg_per_s": 2.5 / 3600.0,
        "gas_celsius": 5.0,
    }
    dryer, exhaust = cs.saturation_ratios(cs.recompute(PROPS, cs.line_inputs(damp, 400.0)))
    assert dryer < 1.0 < exhaust
    curve = cs.cost_curve(
        PROPS, cs.line_inputs(damp), np.asarray([0.0, 400.0]), LAB_PUMP, CONFIG.prices
    )
    assert curve.oversaturated.tolist() == [False, True]


def test_fresh_gas_keeps_its_humidity_up_to_saturation() -> None:
    base = cs.line_inputs(HOT)
    warm, capped_warm = cs.fresh_gas_at(base, 20.0)
    cold, capped_cold = cs.fresh_gas_at(base, -10.0)
    assert not capped_warm and warm.gas_humidity_g_per_kg == base.gas_humidity_g_per_kg
    assert warm.gas_celsius == 20.0 and cold.gas_celsius == -10.0
    assert capped_cold and cold.gas_humidity_g_per_kg == pytest.approx(1.60, abs=0.02)
    assert replace(cold, gas_celsius=20.0, gas_humidity_g_per_kg=6.0) == warm
    assert cs.fresh_gas_saturation_g_per_kg(-1e-9) == pytest.approx(
        cs.fresh_gas_saturation_g_per_kg(0.0), rel=1e-4
    )
    assert cs.fresh_gas_saturation_g_per_kg(20.0) == pytest.approx(14.7, abs=0.2)


# ----------------------------------------------------------------------------- pump and cost
def test_pump_law_is_cubic_and_meets_the_network_placeholder_at_the_reference_flow() -> None:
    pump, aux = CONFIG.pump, AuxiliaryConfig()
    at_reference = pump.power_kw(pump.reference_flow_kg_per_h)
    assert at_reference == pytest.approx(
        pump_power_w(
            pump.reference_flow_kg_per_h / 3600.0,
            aux.loop_pressure_drop_pa,
            aux.htm_density_kg_per_m3,
            aux.pump_efficiency,
        )
        / 1000.0
    )
    assert at_reference == pytest.approx(0.694, abs=5e-4)
    assert pump.power_kw(0.0) == 0.0
    assert pump.power_kw(2.0 * pump.reference_flow_kg_per_h) == pytest.approx(8.0 * at_reference)
    assert pump.power_kw(5000.0, 2.0) == pytest.approx(2.0 * pump.power_kw(5000.0))
    with pytest.raises(ValueError):
        replace(pump, efficiency=0.0)
    with pytest.raises(ValueError):
        replace(pump, reference_flow_kg_per_h=0.0)


def test_cost_curve_matches_the_rebuild_and_the_prices() -> None:
    inputs = cs.line_inputs(HOT)
    curve = lab_curve(inputs)
    out = cs.recompute(PROPS, replace(inputs, medium_kg_per_h=float(LAB_FLOWS[100])))
    assert curve.fuel_kw[100] == pytest.approx(out.fuel_kg_per_h * 45000.0 / 3600.0)
    assert curve.heat_recovered_kw[100] == pytest.approx(out.heat_recovered_kw)
    assert curve.heat_recovered_kw[0] == 0.0 and np.all(np.diff(curve.heat_recovered_kw) > 0.0)
    assert not curve.oversaturated.any()
    cost = curve.costs_eur_per_h(2.0)[100]
    assert cost == pytest.approx(
        CONFIG.prices.fuel_eur_per_kwh * curve.fuel_kw[100]
        + 0.16 * LAB_PUMP.power_kw(float(LAB_FLOWS[100]), 2.0)
    )
    assert curve.cost_eur_per_h(float(LAB_FLOWS[100]), 2.0) == pytest.approx(cost)
    assert curve.co2_kg_per_h(0.0) == pytest.approx(0.201 * curve.fuel_kw[0])
    assert curve.co2_kg_per_h(50.0) == pytest.approx(
        0.201 * curve.fuel_kw_at(50.0) + 0.366 * LAB_PUMP.power_kw(50.0)
    )


def test_the_fuel_saved_is_the_recovered_heat_over_the_heater_efficiency() -> None:
    """Above the lower bound of the vapour correlation the heater duty is linear in the inlet."""
    inputs, _ = cs.fresh_gas_at(cs.line_inputs(HOT), 20.0)
    curve = lab_curve(inputs)
    saved = curve.fuel_kw[0] - curve.fuel_kw
    assert saved[1:] == pytest.approx(curve.heat_recovered_kw[1:] / 0.8, rel=0.02)


def test_optimum_is_interior_with_a_costly_pump_and_at_the_bound_with_a_cheap_one() -> None:
    inputs = cs.line_inputs(HOT)
    costly = lab_curve(inputs)
    result = cs.optimum(costly, 1.0, (0.8, 1.2))
    costs = costly.costs_eur_per_h(1.0)
    assert not result["at_upper_bound"] and not result["at_zero"]
    assert result["cost_eur_per_h"] <= costs.min()
    assert result["cost_eur_per_h"] == pytest.approx(costs.min(), rel=1e-6)
    assert abs(result["flow_kg_per_h"] - LAB_FLOWS[int(np.argmin(costs))]) <= 0.5
    assert result["flow_share_of_scan"] == pytest.approx(result["flow_kg_per_h"] / 200.0)
    assert 0.0 < result["saving_percent_of_no_flow_cost"] < 100.0
    assert 0.0 < result["heat_recovered_share_of_scan_max"] < 1.0
    assert not result["oversaturated_at_optimum"]
    for neighbour in result["neighbours"]:
        assert neighbour["increase_percent_of_cost"] >= 0.0
        assert neighbour["increase_percent_of_saving"] >= 0.0
        assert neighbour["flow_kg_per_h"] == pytest.approx(
            neighbour["factor"] * result["flow_kg_per_h"]
        )
        assert neighbour["cost_eur_per_h"] == pytest.approx(
            costly.cost_eur_per_h(neighbour["flow_kg_per_h"], 1.0)
        )
    higher = cs.optimum(costly, 2.0, (0.8, 1.2))
    lower = cs.optimum(costly, 0.5, (0.8, 1.2))
    assert higher["flow_kg_per_h"] < result["flow_kg_per_h"] < lower["flow_kg_per_h"]
    cheap = cs.optimum(
        lab_curve(inputs, replace(LAB_PUMP, reference_pressure_drop_pa=1.0)), 1.0, (0.8, 1.2)
    )
    assert cheap["at_upper_bound"] and cheap["flow_share_of_scan"] == 1.0
    assert cheap["neighbours"][1]["above_scanned_range"]
    assert not cheap["neighbours"][0]["above_scanned_range"]
    assert cheap["co2_change_percent_of_no_flow"] < 0.0


def test_optimum_is_no_flow_when_the_exhaust_is_colder_than_the_fresh_gas() -> None:
    inputs, _ = cs.fresh_gas_at(cs.line_inputs(WET), 40.0)
    curve = lab_curve(inputs)
    assert curve.heat_recovered_kw[-1] < 0.0
    result = cs.optimum(curve, 1.0, (0.8, 1.2))
    assert result["at_zero"] and result["flow_kg_per_h"] == 0.0
    assert result["saving_percent_of_no_flow_cost"] == 0.0
    assert result["neighbours"][0]["increase_percent_of_saving"] is None
    assert result["oversaturated_at_optimum"]


# ----------------------------------------------------------------------------- driver
def test_dry_run_of_the_driver_writes_no_input_flow_or_cost_into_the_record(tmp_path: Path) -> None:
    local = tmp_path / "local"
    (local / "dyssol_steady").mkdir(parents=True)
    (local / "dyssol_steady" / "carrier_sweep_local.yaml").write_text(
        yaml.safe_dump(
            {
                "source": {"workbook": "none.xlsx", "workbook_sha256": "0" * 64, "process": "lab"},
                "line": {
                    **HOT,
                    "gas_kg_per_s": 10.0,
                    "gas_moisture_kg_per_s": 0.06,
                    "spray_kg_per_s": 0.3,
                    "spray_solid_kg_per_s": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dyssol").mkdir()
    (tmp_path / "dyssol" / "Materials.dmdb").write_text(DMDB, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "TWINLINK_LOCAL_DIR": str(local),
        "TWINLINK_CARRIER_OUT": str(tmp_path / "out"),
        "TWINLINK_RUNS_DIR": str(tmp_path / "runs"),
        "DYSSOL_HOME": str(tmp_path / "dyssol"),
    }
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_carrier_sweep.py"),
            "--name",
            "t",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    record = json.loads((tmp_path / "out" / "t.json").read_text(encoding="utf-8"))
    local_record = json.loads((tmp_path / "out" / "t_local.json").read_text(encoding="utf-8"))
    assert record["provenance"]["code/root_checked"] is True
    assert record["header"]["dry_run"] is True and record["header"]["grid_points"] == 1501
    assert [cell["ambient_celsius"] for cell in record["scan"]] == [-10.0, 0.0, 10.0, 20.0]
    assert all(len(cell["optima"]) == 3 for cell in record["scan"])
    assert all(cell["heat_recovered_monotonic"] for cell in record["scan"])
    assert [cell["humidity_capped_at_saturation"] for cell in record["scan"]] == [
        True,
        True,
        False,
        False,
    ]
    assert record["scan"][0]["optimum_without_humidity_cap"] is not None
    assert record["scan"][2]["optimum_without_humidity_cap"] is None

    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    forbidden = {
        "inputs",
        "sheet_inputs",
        "source",
        "curve",
        "flow_kg_per_h",
        "cost_eur_per_h",
        "cost_no_flow_eur_per_h",
        "heat_recovered_kw",
        "pump_kw",
        "fuel_kw",
    }
    assert not forbidden & keys({k: v for k, v in record.items() if k != "config"})
    assert {"sheet_inputs", "curve", "flow_kg_per_h", "cost_eur_per_h"} <= keys(local_record)
    assert local_record["sheet_inputs"]["dry_gas_kg_per_h"] == pytest.approx(36000.0)
