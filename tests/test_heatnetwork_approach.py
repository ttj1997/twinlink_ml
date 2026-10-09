"""Minimum approach temperature along the exchanger (Befund 79, review finding A1).

``approach_check="along_exchanger"`` (default) limits every stage so that the approach is at least
``dT_min`` at every point of the counter-current exchanger; ``"warm_end"`` is the cascade of
Befunde 57-77, kept to reproduce old artifacts.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from twinlink_ml.evaluation import joint_search as js
from twinlink_ml.evaluation import site_network as sn
from twinlink_ml.evaluation.site_oracle import SiteOracleSpec
from twinlink_ml.heatnetwork.approach import build_source_profile
from twinlink_ml.heatnetwork.balance import SinkState, SourceState, evaluate_cascade
from twinlink_ml.heatnetwork.config import HeatNetworkConfig, HeatSinkConfig, HeatSourceConfig
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.simulation.thermo import dew_point_celsius

REPO_ROOT = Path(__file__).resolve().parents[1]
GAS = GasConfig()
BACKEND = NumpyBackend()
DISCRETISATION_RELATIVE_BOUND = 1.0e-5
"""Bound on the relative error of the stage duty at 400 profile nodes against 40000 nodes.

Measured on 2026-10-09 over 375 random condensing stages (seed 20261009; source 0.5-20 kg/s,
35-130 degC, Y 0.01-0.09; sink 0.5-40 kg/s, 5-35 -> 60-300 degC; ``T_kalt`` 10-32 degC;
``dT_min`` 10 or 20 K): largest relative error 2.9e-6, largest absolute error 0.46 W. The test
below repeats the measurement on 120 of these stages and fails above the bound."""


def _capacity(mass_flow: float, humidity: float) -> float:
    return mass_flow * (GAS.cp_dry_air_j_per_kg_k + humidity * GAS.cp_vapour_j_per_kg_k)


def _dew_point(humidity: float) -> float:
    return float(dew_point_celsius(BACKEND, GAS, BACKEND.asarray(humidity)))


def _run(network: HeatNetworkConfig, stages: tuple[tuple[str, str], ...]):
    sources = tuple(SourceState.from_config(c) for c in network.sources)
    sinks = tuple(SinkState.from_config(c) for c in network.sinks)
    return evaluate_cascade(network, stages, sources, sinks)


def test_default_is_the_check_along_the_exchanger() -> None:
    network = HeatNetworkConfig(
        sources=(HeatSourceConfig("s", 1.0, 80.0, 0.01),),
        sinks=(HeatSinkConfig("k", 1.0, 0.005, 20.0, 100.0),),
    )
    assert network.approach_check == "along_exchanger"
    assert network.approach_profile_nodes == 400
    with pytest.raises(ValueError, match="approach_check"):
        dataclasses.replace(network, approach_check="cold_end")


def test_cold_end_binds_when_the_sink_has_the_larger_capacity_rate() -> None:
    """Dry source (dew point below the floor, no condensation), ``C_sink = 2 C_source``.

    Expected duty ``C_source (T_out - T_sink,in - dT_min)``: the source leaves the stage at
    ``T_sink,in + dT_min``; the warm end keeps more than ``dT_min``. The warm-end check admits
    ``min(C_sink (T_out - dT_min - T_sink,in), C_source (T_out - T_kalt))``, which is more.
    """
    humidity = 0.005
    assert _dew_point(humidity) < 15.0
    sink_flow = 2.0 * _capacity(1.0, humidity) / _capacity(1.0, 0.005)
    network = HeatNetworkConfig(
        sources=(HeatSourceConfig("s", 1.0, 80.0, humidity),),
        sinks=(HeatSinkConfig("k", sink_flow, 0.005, 20.0, 200.0),),
        minimum_pinch_k=10.0,
        recovery_mode="condensing",
        cold_temperature_celsius=15.0,
    )
    result = _run(network, (("s", "k"),))
    c_source = _capacity(1.0, humidity)
    duty_w = result.stage_recovered_power_kw[0, 0] * 1000.0
    assert duty_w == pytest.approx(c_source * (80.0 - 20.0 - 10.0), rel=1e-9)
    assert result.stage_min_approach_k[0, 0] == pytest.approx(10.0, abs=1e-9)
    assert result.stage_pinch_position[0, 0] == pytest.approx(1.0)
    sink_out = result.stage_sink_outlet_temperature_celsius[0, 0]
    assert 80.0 - sink_out > 10.0 + 1.0

    legacy = _run(dataclasses.replace(network, approach_check="warm_end"), (("s", "k"),))
    assert legacy.stage_recovered_power_kw[0, 0] * 1000.0 == pytest.approx(
        c_source * (80.0 - 15.0), rel=1e-9
    )
    assert np.isnan(legacy.stage_min_approach_k[0, 0])


def test_pinch_inside_the_exchanger_at_the_dew_point() -> None:
    """Condensing source, ``C_source < C_sink <`` the effective capacity rate below the dew point.

    The tightest point is the dew point of the source (the kink of its profile, an exact node):
    expected duty ``C_source (T_out - T_dp) + C_sink (T_dp - dT_min - T_sink,in)``; position
    strictly inside the stage, both ends above ``dT_min``.
    """
    humidity, t_out, sink_in, dt_min = 0.05, 70.0, 10.0, 10.0
    t_dp = _dew_point(humidity)
    assert 35.0 < t_dp < 45.0
    c_source = _capacity(1.0, humidity)
    sink_flow = 2.0 * c_source / _capacity(1.0, 0.005)
    c_sink = _capacity(sink_flow, 0.005)
    network = HeatNetworkConfig(
        sources=(HeatSourceConfig("s", 1.0, t_out, humidity),),
        sinks=(HeatSinkConfig("k", sink_flow, 0.005, sink_in, 200.0),),
        minimum_pinch_k=dt_min,
        recovery_mode="condensing",
        cold_temperature_celsius=20.0,
    )
    result = _run(network, (("s", "k"),))
    expected = c_source * (t_out - t_dp) + c_sink * (t_dp - dt_min - sink_in)
    assert result.stage_recovered_power_kw[0, 0] * 1000.0 == pytest.approx(expected, rel=1e-9)
    assert result.stage_min_approach_k[0, 0] == pytest.approx(dt_min, abs=1e-9)
    assert 0.0 < result.stage_pinch_position[0, 0] < 1.0
    sink_out = result.stage_sink_outlet_temperature_celsius[0, 0]
    source_out = result.stage_source_outlet_temperature_celsius[0, 0]
    assert t_out - sink_out > dt_min + 1.0
    assert source_out - sink_in > dt_min + 1.0
    assert source_out < t_dp  # the stage condenses below the dew point ...
    assert result.source_condensate_kg_per_s[0, 0] > 0.0  # ... before reaching T_kalt


def _random_stage(rng: np.random.Generator) -> HeatNetworkConfig:
    return HeatNetworkConfig(
        sources=(
            HeatSourceConfig(
                "s",
                float(rng.uniform(0.5, 20.0)),
                float(rng.uniform(35.0, 130.0)),
                float(rng.uniform(0.01, 0.09)),
            ),
        ),
        sinks=(
            HeatSinkConfig(
                "k",
                float(rng.uniform(0.5, 40.0)),
                0.005,
                float(rng.uniform(5.0, 35.0)),
                float(rng.uniform(60.0, 300.0)),
            ),
        ),
        minimum_pinch_k=float(rng.choice([10.0, 20.0])),
        recovery_mode="condensing",
        cold_temperature_celsius=float(rng.uniform(10.0, 32.0)),
    )


def test_discretisation_error_of_the_stage_duty_is_below_its_stated_bound() -> None:
    """400 profile nodes against 40000; see :data:`DISCRETISATION_RELATIVE_BOUND` for the number."""
    rng = np.random.default_rng(20261009)
    worst = 0.0
    for _ in range(120):
        network = _random_stage(rng)
        coarse = _run(network, (("s", "k"),)).stage_recovered_power_kw[0, 0]
        fine = _run(
            dataclasses.replace(network, approach_profile_nodes=40000), (("s", "k"),)
        ).stage_recovered_power_kw[0, 0]
        if fine > 0.0:
            worst = max(worst, abs(coarse - fine) / fine)
    assert worst <= DISCRETISATION_RELATIVE_BOUND


def test_minimum_approach_holds_along_every_stage_on_a_fine_grid() -> None:
    """At every point of every stage the approach is at least ``dT_min`` within 1e-4 K.

    Two sources, two sinks, random order; the source temperature on a 40000-node profile at
    20001 evaluation points per stage.
    """
    rng = np.random.default_rng(7)
    for _ in range(40):
        sources = tuple(
            HeatSourceConfig(
                name,
                float(rng.uniform(1.0, 20.0)),
                float(rng.uniform(40.0, 130.0)),
                float(rng.uniform(0.01, 0.08)),
            )
            for name in ("a", "b")
        )
        sinks = tuple(
            HeatSinkConfig(
                name,
                float(rng.uniform(1.0, 30.0)),
                0.005,
                float(rng.uniform(5.0, 30.0)),
                float(rng.uniform(60.0, 300.0)),
            )
            for name in ("x", "y")
        )
        network = HeatNetworkConfig(
            sources=sources,
            sinks=sinks,
            minimum_pinch_k=float(rng.choice([10.0, 20.0])),
            recovery_mode="condensing",
            cold_temperature_celsius=float(rng.uniform(10.0, 30.0)),
        )
        pairs = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]
        rng.shuffle(pairs)
        stages = tuple(pairs)
        result = _run(network, stages)
        fine = {
            s.name: build_source_profile(
                dataclasses.replace(network, approach_profile_nodes=40000),
                dry_air_mass_flow_kg_per_s=np.array([s.dry_air_mass_flow_kg_per_s]),
                outlet_temperature_celsius=np.array([s.outlet_temperature_celsius]),
                outlet_humidity_kg_per_kg=np.array([s.outlet_humidity_kg_per_kg]),
            )
            for s in sources
        }
        released = dict.fromkeys(fine, 0.0)
        sink_temperature = {s.name: s.intake_temperature_celsius for s in sinks}
        for index, (source_name, sink_name) in enumerate(stages):
            duty = result.stage_recovered_power_kw[index, 0] * 1000.0
            sink = next(s for s in sinks if s.name == sink_name)
            c_sink = _capacity(sink.dry_air_mass_flow_kg_per_s, sink.humidity_kg_per_kg)
            if duty > 0.0:
                q = released[source_name] + np.linspace(0.0, duty, 20001)
                profile = fine[source_name]
                t_source = np.interp(q, profile.q_w[:, 0], profile.t_celsius[:, 0])
                t_sink = sink_temperature[sink_name] + (released[source_name] + duty - q) / c_sink
                assert np.min(t_source - t_sink) >= network.minimum_pinch_k - 1e-4
                assert result.stage_min_approach_k[index, 0] >= network.minimum_pinch_k - 1e-9
            released[source_name] += duty
            sink_temperature[sink_name] += duty / c_sink


def test_one_stage_never_recovers_more_than_the_warm_end_check() -> None:
    rng = np.random.default_rng(11)
    for _ in range(100):
        network = _random_stage(rng)
        along = _run(network, (("s", "k"),)).stage_recovered_power_kw[0, 0]
        warm = _run(dataclasses.replace(network, approach_check="warm_end"), (("s", "k"),))
        assert along <= warm.stage_recovered_power_kw[0, 0] + 1e-9


def test_available_heat_is_the_same_in_both_modes() -> None:
    """The profile changes where the heat is released, not how much (MA1 totals)."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        network = _random_stage(rng)
        along = _run(network, (("s", "k"),))
        warm = _run(dataclasses.replace(network, approach_check="warm_end"), (("s", "k"),))
        assert along.source_available_power_kw[0, 0] == pytest.approx(
            warm.source_available_power_kw[0, 0], rel=1e-12
        )
        assert along.source_used_power_kw[0, 0] + along.source_discarded_power_kw[
            0, 0
        ] == pytest.approx(along.source_available_power_kw[0, 0], rel=1e-12)


def test_site_spec_with_cap_keeps_the_approach_check() -> None:
    site = sn.SiteSpec(271.8, approach_check="warm_end")
    assert js.site_spec_with_cap(site, 50.0).approach_check == "warm_end"
    assert js.site_spec_with_cap(site, 50.0).shared_cap_kw == 50.0
    assert sn.SiteSpec(271.8).fingerprint != site.fingerprint


def test_warm_end_mode_reproduces_stored_m4c_values_bit_for_bit() -> None:
    """Legacy: two records of ``m4c_confirm_lab_umin126`` (Befund 77) with ``==`` on the value."""
    artifact = json.loads(
        (REPO_ROOT / "artifacts" / "runs" / "m4c_confirm_lab_umin126.json").read_text("utf-8")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml")
    configs = {"A": sn.with_spray_rate(base, 1.5), "B": sn.with_spray_rate(base, 2.0)}
    spec = SiteOracleSpec(
        0.15 - 0.005961819615175404,
        sn.SiteSpec(artifact["scale_factor_k"], approach_check="warm_end"),
    )
    scenario = artifact["scenarios"]["2"]
    for record in (scenario["best_s2"], scenario["per_seed_m3"]["0"]):
        allocation = tuple(tuple(c) for c in record["allocation"])
        result = js.evaluate_true(
            record["set_points"], allocation, configs, scenario["t_amb"], scenario["phi"], spec
        )
        assert result["value"] == record["value"]
