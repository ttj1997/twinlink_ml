"""Per-carrier booking of the process power (D5, M1)."""

from __future__ import annotations

import pytest

from twinlink_ml.evaluation.carrier_energy import (
    CarrierEnergy,
    carrier_energy,
    carrier_energy_from_diagnostics,
)
from twinlink_ml.evaluation.net_energy import ELECTRICITY_CO2_KG_PER_KWH, GAS_CO2_KG_PER_KWH

POWERS = {
    "power_heater_kw": 1.0,
    "power_fan_kw": 0.05,
    "power_pump_kw": 0.01,
    "power_atomiser_kw": 0.04,
}


def test_gas_heater_books_the_duty_on_gas_and_the_rest_on_electricity() -> None:
    result = carrier_energy_from_diagnostics(
        POWERS, heater="gas", process_heater_efficiency=0.98, gas_heater_efficiency=0.80
    )
    assert result.electric_kw == pytest.approx(0.10)
    assert result.gas_kw == pytest.approx(1.0 * 0.98 / 0.80)
    assert result.kwh_sum_kw == pytest.approx(result.electric_kw + result.gas_kw)
    assert result.co2_kg_per_h == pytest.approx(
        ELECTRICITY_CO2_KG_PER_KWH * result.electric_kw + GAS_CO2_KG_PER_KWH * result.gas_kw
    )


def test_electric_heater_reproduces_the_gross_power_of_befund_52() -> None:
    result = carrier_energy_from_diagnostics(
        POWERS, heater="electric", process_heater_efficiency=0.98, gas_heater_efficiency=0.80
    )
    assert result.gas_kw == 0.0
    assert result.electric_kw == pytest.approx(sum(POWERS.values()))
    assert result.co2_kg_per_h == pytest.approx(ELECTRICITY_CO2_KG_PER_KWH * sum(POWERS.values()))


def test_recovered_heat_reduces_the_heater_term_only_and_never_below_zero() -> None:
    kwargs = {
        "power_heater_kw": 1.0,
        "power_fan_kw": 0.05,
        "power_pump_kw": 0.01,
        "power_atomiser_kw": 0.04,
        "process_heater_efficiency": 0.98,
        "gas_heater_efficiency": 0.80,
    }
    base = carrier_energy(heater="gas", **kwargs)
    part = carrier_energy(heater="gas", recovered_heat_kw=0.48, **kwargs)
    assert part.electric_kw == base.electric_kw
    assert part.gas_kw == pytest.approx((0.98 - 0.48) / 0.80)
    full = carrier_energy(heater="gas", recovered_heat_kw=5.0, **kwargs)
    assert full.gas_kw == 0.0
    electric = carrier_energy(heater="electric", recovered_heat_kw=5.0, **kwargs)
    assert electric.electric_kw == pytest.approx(0.10)


def test_efficiency_enters_the_gas_booking_exactly_once() -> None:
    kwargs = {
        "power_heater_kw": 2.0,
        "power_fan_kw": 0.0,
        "power_pump_kw": 0.0,
        "power_atomiser_kw": 0.0,
        "process_heater_efficiency": 0.5,
        "heater": "gas",
    }
    high = carrier_energy(gas_heater_efficiency=1.0, **kwargs)
    low = carrier_energy(gas_heater_efficiency=0.5, **kwargs)
    assert high.gas_kw == pytest.approx(1.0)
    assert low.gas_kw == pytest.approx(2.0)


def test_invalid_inputs_raise() -> None:
    kwargs = {
        "power_heater_kw": 1.0,
        "power_fan_kw": 0.0,
        "power_pump_kw": 0.0,
        "power_atomiser_kw": 0.0,
        "process_heater_efficiency": 0.98,
        "gas_heater_efficiency": 0.8,
    }
    with pytest.raises(ValueError):
        carrier_energy(heater="steam", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        carrier_energy(heater="gas", recovered_heat_kw=-1.0, **kwargs)
    with pytest.raises(ValueError):
        carrier_energy(heater="gas", **{**kwargs, "gas_heater_efficiency": 1.5})


def test_carrier_energy_is_a_plain_value_object() -> None:
    assert CarrierEnergy(1.0, 2.0).kwh_sum_kw == 3.0
