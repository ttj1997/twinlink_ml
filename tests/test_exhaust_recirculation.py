"""Exhaust recirculation (``OperationConfig.exhaust_recirculation_fraction``, study docs/22).

Unset, the field is NOT part of the fingerprint and the fresh-air path is bit-identical to the code
before the field existed (reference numbers computed with commit c3f13a8). Set, the inlet air is the
mixture of fresh air and bed exhaust; the tests check the mixing balances (water, enthalpy), the
water balance of the whole loop at steady state, the heater booking against the mixing
temperature, the convergence of the fixed iteration and the cube law of the recirculation fan.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.simulation import (
    GreyBoxFluidBedGranulator,
    OperationConfig,
    PlaceholderConfigWarning,
    ProcessConfig,
    dump_process_config,
    load_process_config,
    thermo,
)
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import EnergyConfig
from twinlink_ml.simulation.oracle_search import build_variant_config, evaluate_point
from twinlink_ml.simulation.sorption import isotherm_from_material
from twinlink_ml.simulation.steady_state import solve_steady_state

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf" / "process"
GAB = "greybox_gab_placeholder.yaml"

# Same table as tests/test_quality_moisture_limit.py (fingerprints of the configs as committed
# before any optional field existed): the new fields must not move them.
FINGERPRINTS_BEFORE: dict[tuple[str, bool | None], str] = {
    ("greybox_gab_placeholder.yaml", None): "2acd06f38963c3af",
    ("greybox_gab_placeholder.yaml", False): "6ecedc6bd993fb3d",
    ("greybox_gab_placeholder.yaml", True): "bddca37f716f5cfc",
    ("greybox_lab_moisture.yaml", None): "ebbc22316fe248c0",
    ("greybox_lab_moisture.yaml", False): "f9a4455aa9539a91",
    ("greybox_lab_moisture.yaml", True): "ba85b873034ff983",
    ("greybox_placeholder.yaml", None): "034ee446f64b5159",
    ("greybox_placeholder.yaml", False): "eed1084074c7180c",
    ("greybox_placeholder.yaml", True): "a87281ef999aa37a",
}

# Fresh-air path at (T_in 90, u 1.26, p 1.0, T_amb 20, phi 0.5), search precision 30 / 1e-7,
# computed with commit c3f13a8 (before the field): float.hex of the reported numbers.
REFERENCE_FRESH_AIR = {
    "power_heater_kw": "0x1.b7dc1704ab5b7p+0",
    "power_fan_kw": "0x1.e49ef1b21f18fp-5",
    "outlet_humidity_kg_per_kg": "0x1.3d141e6ed6ee3p-6",
    "evaporation_rate_kg_per_s": "0x1.2b5ba6e08d35ep-12",
    "power_total_kw": "0x1.c713acd3663f7p+0",
}
REFERENCE_BED = "0x1.c51a38b400000p+5"
REFERENCE_MOISTURE = "0x1.94ccaf633815cp-5"
REFERENCE_EXHAUST = ("0x1.cd303d9a83e26p-1", "0x1.666b1a6344952p-1")
REFERENCE_DIAGNOSTIC_KEYS = {
    "bed_pressure_drop_pa",
    "d50_m",
    "dry_air_mass_flow_kg_per_s",
    "energy_target",
    "evaporation_rate_kg_per_s",
    "fluidization_margin_lower_m_per_s",
    "fluidization_margin_upper_m_per_s",
    "inlet_humidity_kg_per_kg",
    "minimum_fluidization_velocity_m_per_s",
    "outlet_humidity_kg_per_kg",
    "particle_moisture_kg_per_kg",
    "power_atomiser_kw",
    "power_fan_kw",
    "power_heater_kw",
    "power_pump_kw",
    "power_total_kw",
    "product_mass_flow_kg_per_s",
    "saturation_degree",
    "saturation_humidity_kg_per_kg",
    "terminal_velocity_m_per_s",
}

_BACKEND = NumpyBackend()


def _load(name: str = GAB, projection: bool | None = None) -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_DIR / name)
    if projection is None:
        return config
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(
            config.numerics, sample_time_s=25.0, enforce_gas_velocity_projection=projection
        ),
    )


def _with_r(config: ProcessConfig, r: float | None, zeta: float | None = None) -> ProcessConfig:
    return dataclasses.replace(
        config,
        operation=dataclasses.replace(config.operation, exhaust_recirculation_fraction=r),
        energy=dataclasses.replace(config.energy, recirculation_duct_loss_coefficient=zeta),
    )


def _solve(
    config: ProcessConfig,
    t_in: float = 90.0,
    u: float = 1.26,
    t_amb: float = 20.0,
    phi: float = 0.5,
):
    return evaluate_point(
        t_in, u, 1.0, t_amb, phi, max_iterations=30, tolerance=1e-7, base_config=config
    )


# ----------------------------------------------------------------------------- fingerprint


@pytest.mark.parametrize(("name", "projection"), sorted(FINGERPRINTS_BEFORE, key=str))
def test_existing_configs_keep_their_fingerprints(name: str, projection: bool | None) -> None:
    config = _load(name, projection)
    assert config.operation.exhaust_recirculation_fraction is None
    assert config.energy.recirculation_duct_loss_coefficient is None
    assert config.fingerprint == FINGERPRINTS_BEFORE[(name, projection)]


def test_set_fraction_enters_the_fingerprint_and_only_the_operation_section() -> None:
    base = _load()
    variant = _with_r(base, 0.0)
    assert variant.fingerprint != base.fingerprint
    changed = {
        key
        for key, value in variant.section_fingerprints.items()
        if value != base.section_fingerprints[key]
    }
    assert changed == {"operation"}
    with_fan = _with_r(base, 0.3, 350.0)
    changed = {
        key
        for key, value in with_fan.section_fingerprints.items()
        if value != base.section_fingerprints[key]
    }
    assert changed == {"operation", "energy"}


@pytest.mark.parametrize(("r", "zeta"), [(None, None), (0.4, 350.0)])
def test_dump_and_load_round_trip(tmp_path: Path, r: float | None, zeta: float | None) -> None:
    config = _with_r(_load(), r, zeta)
    path = tmp_path / "config.yaml"
    dump_process_config(config, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        loaded = load_process_config(path)
    assert loaded.operation.exhaust_recirculation_fraction == r
    assert loaded.energy.recirculation_duct_loss_coefficient == zeta
    assert loaded.fingerprint == config.fingerprint


@pytest.mark.parametrize("r", [-0.01, 0.951, 1.0])
def test_fraction_outside_the_range_is_rejected(r: float) -> None:
    with pytest.raises(ValueError, match="exhaust_recirculation_fraction"):
        OperationConfig(exhaust_recirculation_fraction=r)


def test_negative_duct_coefficient_is_rejected() -> None:
    with pytest.raises(ValueError, match="recirculation_duct_loss_coefficient"):
        EnergyConfig(recirculation_duct_loss_coefficient=-1.0)


# ------------------------------------------------------------------------ fresh-air path


def test_unset_fraction_reproduces_the_code_before_the_field_bit_for_bit() -> None:
    solution = _solve(_load())
    for key, value in REFERENCE_FRESH_AIR.items():
        assert solution.diagnostics[key].hex() == value, key
    assert solution.bed_temperature_celsius.hex() == REFERENCE_BED
    assert solution.particle_moisture_kg_per_kg.hex() == REFERENCE_MOISTURE
    assert (solution.exhaust_sensible_kw.hex(), solution.exhaust_latent_kw.hex()) == (
        REFERENCE_EXHAUST
    )
    assert set(solution.diagnostics) == REFERENCE_DIAGNOSTIC_KEYS
    assert solution.exhaust_dry_air_mass_flow_kg_per_s == solution.dry_air_mass_flow_kg_per_s


def test_zero_fraction_agrees_with_fresh_air_to_the_picard_residual() -> None:
    """Zero fraction agrees with the fresh-air path.

    r = 0 runs the bracketed recirculation search instead of the 5 damped
    Picard steps of the fresh-air path; both converge to the same fixed point.
    """
    fresh = _solve(_load())
    zero = _solve(_with_r(_load(), 0.0))
    assert (
        zero.diagnostics["inlet_humidity_kg_per_kg"]
        == fresh.diagnostics["inlet_humidity_kg_per_kg"]
    )
    assert zero.diagnostics["mixing_temperature_celsius"] == 20.0
    for key in ("power_heater_kw", "power_total_kw", "outlet_humidity_kg_per_kg"):
        assert zero.diagnostics[key] == pytest.approx(fresh.diagnostics[key], rel=1e-3)
    assert zero.particle_moisture_kg_per_kg == pytest.approx(
        fresh.particle_moisture_kg_per_kg, rel=1e-3
    )
    assert zero.bed_temperature_celsius == pytest.approx(fresh.bed_temperature_celsius, abs=1e-3)


# ------------------------------------------------------------------------------ balances


@pytest.mark.parametrize("r", [0.0, 0.2, 0.5, 0.8])
@pytest.mark.parametrize(("t_amb", "phi"), [(5.0, 0.25), (20.0, 0.5), (35.0, 0.85)])
def test_mixing_balances_of_water_and_enthalpy_close(r: float, t_amb: float, phi: float) -> None:
    solution = _solve(_with_r(_load(), r), t_amb=t_amb, phi=phi)
    d = solution.diagnostics
    gas = _load().gas
    y_a = d["fresh_air_humidity_kg_per_kg"]
    y_in = d["inlet_humidity_kg_per_kg"]
    y_out = d["outlet_humidity_kg_per_kg"]
    t_out = solution.bed_temperature_celsius
    t_mix = d["mixing_temperature_celsius"]
    assert y_in == pytest.approx((1.0 - r) * y_a + r * y_out, rel=1e-12, abs=1e-15)

    def enthalpy(t: float, y: float) -> float:
        # J/kg dry air, liquid water at 0 degC as reference
        return gas.cp_dry_air_j_per_kg_k * t + y * (2.5008e6 + gas.cp_vapour_j_per_kg_k * t)

    mixed = (1.0 - r) * enthalpy(t_amb, y_a) + r * enthalpy(t_out, y_out)
    assert enthalpy(t_mix, y_in) == pytest.approx(mixed, rel=1e-12)
    assert t_amb - 1e-12 <= t_mix <= t_out + 1e-12


@pytest.mark.parametrize("r", [0.0, 0.3, 0.6, 0.8])
def test_water_balance_of_the_loop_closes_at_steady_state(r: float) -> None:
    config = _with_r(_load(), r)
    solution = _solve(config)
    d = solution.diagnostics
    m_da = solution.dry_air_mass_flow_kg_per_s
    evaporation = d["evaporation_rate_kg_per_s"]
    # loop: the water leaves only with the purge (1 - r) m_da
    purge = solution.exhaust_dry_air_mass_flow_kg_per_s
    assert purge == pytest.approx((1.0 - r) * m_da, rel=1e-15)
    assert purge * (solution.outlet_humidity_kg_per_kg - d["fresh_air_humidity_kg_per_kg"]) == (
        pytest.approx(evaporation, rel=1e-12)
    )
    # bed: spray water in = evaporation + water with the product (steady state)
    spray = config.actions.spray_rate_kg_per_h.nominal / 3600.0
    w_solid = config.disturbances.suspension_solid_fraction.nominal
    product_water = d["product_mass_flow_kg_per_s"] * solution.particle_moisture_kg_per_kg
    assert spray * (1.0 - w_solid) == pytest.approx(evaporation + product_water, rel=1e-5)
    assert solution.residual_inf_norm < 1e-5
    # latent exhaust heat refers to the purge and the fresh-air humidity
    assert solution.exhaust_latent_kw > 0.0


@pytest.mark.parametrize("r", [0.0, 0.4, 0.8])
def test_heater_books_the_duty_from_the_mixing_temperature(r: float) -> None:
    config = _with_r(_load(), r)
    solution = _solve(config, t_in=95.0)
    d = solution.diagnostics
    gas = config.gas
    duty_kw = (
        solution.dry_air_mass_flow_kg_per_s
        * (gas.cp_dry_air_j_per_kg_k + d["inlet_humidity_kg_per_kg"] * gas.cp_vapour_j_per_kg_k)
        * (95.0 - d["mixing_temperature_celsius"])
        / 1000.0
    )
    assert d["power_heater_kw"] == pytest.approx(duty_kw / config.energy.heater_efficiency)


def test_recirculation_lowers_the_heater_power_and_raises_the_humidity() -> None:
    base = _load()
    solutions = [_solve(_with_r(base, r)) for r in (0.0, 0.2, 0.4, 0.6, 0.8)]
    heater = [s.diagnostics["power_heater_kw"] for s in solutions]
    y_in = [s.diagnostics["inlet_humidity_kg_per_kg"] for s in solutions]
    moisture = [s.particle_moisture_kg_per_kg for s in solutions]
    assert all(a > b for a, b in itertools.pairwise(heater))
    assert all(a < b for a, b in itertools.pairwise(y_in))
    assert all(a < b for a, b in itertools.pairwise(moisture))


def test_recirculation_fan_follows_the_cube_of_the_recirculated_flow() -> None:
    base = _load()
    fan = {}
    for r in (0.2, 0.4):
        on = _solve(_with_r(base, r, 350.0)).diagnostics
        off = _solve(_with_r(base, r, None)).diagnostics
        assert off["power_recirculation_fan_kw"] == 0.0
        assert on["power_fan_kw"] == pytest.approx(
            off["power_fan_kw"] + on["power_recirculation_fan_kw"], rel=1e-9
        )
        fan[r] = on["power_recirculation_fan_kw"]
    # density at the two operating points differs slightly (bed temperature, humidity)
    assert fan[0.4] / fan[0.2] == pytest.approx(8.0, rel=0.02)


# ------------------------------------------------------------------------- numerics


@pytest.mark.parametrize("r", [0.3, 0.8, 0.9])
@pytest.mark.parametrize(
    ("t_bed", "moisture", "t_amb", "phi"),
    [
        (30.0, 0.02, 5.0, 0.25),
        (45.0, 0.12, 35.0, 0.85),
        (60.0, 0.15, 20.0, 0.5),
        (75.0, 0.06, 35.0, 0.25),
        (40.0, 0.19, 25.0, 0.85),
    ],
)
def test_fixed_iteration_is_converged(
    monkeypatch: pytest.MonkeyPatch,
    r: float,
    t_bed: float,
    moisture: float,
    t_amb: float,
    phi: float,
) -> None:
    config = _load()
    isotherm = isotherm_from_material(config.material)
    y_a = thermo.absolute_humidity_kg_per_kg(
        _BACKEND, config.gas, np.array([t_amb]), np.array([phi])
    )
    args = (
        _BACKEND,
        config.gas,
        config.material,
        isotherm,
        y_a,
        r,
        np.array([t_bed]),
        np.array([moisture]),
    )
    y_in, y_out, _ = thermo.recirculated_inlet_humidity(*args)
    monkeypatch.setattr(thermo, "_RECIRCULATION_ITERATIONS", 200)
    y_in_ref, y_out_ref, _ = thermo.recirculated_inlet_humidity(*args)
    assert y_in == pytest.approx(y_in_ref, rel=1e-6)
    assert y_out == pytest.approx(y_out_ref, rel=1e-6)


def test_rollout_from_the_steady_state_stays_there() -> None:
    config = build_variant_config(
        _with_r(_load(), 0.6), t_in=90.0, u_gas=1.26, p_nozzle=1.0, t_amb=20.0, phi_amb=0.5
    )
    simulator = GreyBoxFluidBedGranulator(config)
    solution = solve_steady_state(simulator, max_iterations=60, tolerance=1e-12)
    state = solution.state[None, :]
    action = simulator.nominal_action(1)
    disturbance = simulator.nominal_disturbance(1)
    for _ in range(20):
        state = simulator.step(state, action, disturbance).state
    assert np.allclose(state[0], solution.state, rtol=1e-6)


def test_condensing_branch_mixes_fresh_air_with_saturated_exhaust() -> None:
    config = _load()
    isotherm = isotherm_from_material(config.material)
    y_s = thermo.saturation_humidity_kg_per_kg(_BACKEND, config.gas, np.array([20.0]))
    y_a = 1.2 * y_s  # wetter than saturation at bed temperature
    y_in, y_out, _ = thermo.recirculated_inlet_humidity(
        _BACKEND,
        config.gas,
        config.material,
        isotherm,
        y_a,
        0.5,
        np.array([20.0]),
        np.array([0.1]),
    )
    assert y_out == pytest.approx(y_s)
    assert y_in == pytest.approx(0.5 * y_a + 0.5 * y_s)


def test_steady_state_does_not_exist_when_the_drying_capacity_collapses() -> None:
    """No steady state where the drying capacity collapses.

    At r = 0.95 the humidity rise per kg of air shrinks to (1 - r) / (1 - r + r nu) of the
    fresh-air value; at a cold ambient point the spray water then cannot leave even at the
    largest moisture of the bracket, and the solver reports it (an infeasible set point, which
    the oracle counts as a solve failure).
    """
    with pytest.raises(RuntimeError, match="Wasserbilanz"):
        _solve(_with_r(_load(), 0.95), t_amb=5.0, phi=0.25)
