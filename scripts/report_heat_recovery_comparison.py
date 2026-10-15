r"""Heat recovery of the pre-heating stages: partner model v4.1 against an independent calculation.

Question: how much heat do the pre-heating stages of the partner model recover, compared with an
independent calculation with a realistic exchanger model, and which assumption makes the
difference? Docs: ``docs/11_partner_model_v4.md``, section 10. No simulation, no optimisation; the
only solve of the laboratory model is the one steady state of source variant (ii).

Work units
----------
WU0  Mapping of every stage of the six scenarios onto the heat-network module
     (:mod:`twinlink_ml.heatnetwork`). The module has no intermediate circuit (a stage is a direct
     source -> sink pair); the circuit (two exchangers in series over a heat transfer medium) is
     therefore rebuilt here (:func:`solve_stage`), ``src/`` is not changed. The rebuild is tested
     against ``evaluate_cascade`` (direct exchange) and against
     :mod:`twinlink_ml.partner_v4_1.loops` (equal outlet, counter-current bound).
WU0b Consistency check of the partner model: energy balance per exchanger and circuit, second
     law per stage, total recovery against the pinch target of the data case.
WU1  Per scenario and stage: (a) partner model as delivered, (b) same wiring, counter-current,
     approach as the minimum along the exchanger, no condensation, (c) as b with condensation,
     (d) upper bound (pinch target). Sources twice: (i) exhaust states of the partner model for
     both dryers, (ii) dryer 2 from the laboratory grey-box model at the partner's inputs (an
     extrapolation), dryer 1 from the partner model.
WU2  Decomposition of (a) against (b)/(c): co-current against counter-current, ``dT_min``,
     the dew-point limit, condensation, ``c_p`` of the gas (two orders of the steps).
WU3  Sensitivity to the mass flow of the heat transfer medium (20-150 kg/h per circuit).

Conventions (all placeholders or decisions, named in the report)
------------------------------------------------------------------
* ``dT_min`` is the minimum approach between exhaust and process gas **through the circuit**
  (the convention of the heat-network module, whose 20 K stands for two exchangers in series):
  each of the two exchangers of the circuit gets ``dT_min / 2``. The reading "``dT_min`` per
  exchanger" (chain effective ``2 dT_min``) is computed as a sensitivity (``split="each"``).
* Dew-point margin 3 K (module default) in the no-condensation limit; ``T_kalt`` 15 degC, the
  floor of the condensing profile (module default; not binding in the circuit, whose cold end is
  at least ``15 + dT_min / 2``).
* Gas formula corrected: ``m_gas = C (T_target - T_in) / (eta H_u)``, ``eta`` 0.8, ``H_u`` 45 MJ/kg
  of the partner workbook.
* Partner properties (``cp_gas`` 1.2, ``cp_vapour`` 2.0 kJ/(kg K)) in (a); the repository's
  (:class:`~twinlink_ml.simulation.config.GasConfig`: 1.006, 1.86) in (b)-(d).
* Condensation profile of the exhaust as in :mod:`twinlink_ml.heatnetwork.approach` (latent heat at
  ``T_kalt``); its difference to a complete enthalpy balance is reported (``profile_convention``).

The artifact is never overwritten (new name per run). Run from a clean working tree, started with
``tools/run_here.sh``.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from twinlink_ml.heatnetwork import (
    HeatNetworkConfig,
    HeatSinkConfig,
    HeatSourceConfig,
    SinkState,
    SourceState,
    StreamSegment,
    cold_segment_for_sink,
    compute_pinch_target,
    dew_point_of_source_celsius,
    evaluate_cascade,
    hot_segments_for_source,
)
from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4_1 import as_built as ab
from twinlink_ml.partner_v4_1 import layout as lay
from twinlink_ml.partner_v4_1.inputs import (
    ExcelInputsV41,
    SiInputsV41,
    load_excel_inputs_v41,
)
from twinlink_ml.simulation.backend import NumpyBackend
from twinlink_ml.simulation.config import GasConfig
from twinlink_ml.simulation.thermo import (
    dew_point_celsius,
    latent_heat_j_per_kg,
    saturation_humidity_kg_per_kg,
)
from twinlink_ml.utils.code_location import require_code_location

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", ROOT / "local"))

CASES = lay.CASES
SCENARIOS = (1, 2, 3, 4, 5, 6)
DT_MIN_K = (20.0, 10.0, 5.0)
DEW_POINT_MARGIN_K = 3.0
"""Module default ``DEFAULT_DEW_POINT_APPROACH_K`` (placeholder)."""
T_KALT_CELSIUS = 15.0
"""Module default ``DEFAULT_COLD_TEMPERATURE_CELSIUS`` (placeholder)."""
PROFILE_NODES = 400
"""Module default ``DEFAULT_APPROACH_PROFILE_NODES``."""
HTM_FLOWS_KG_PER_H = tuple(float(f) for f in range(20, 151, 5))
"""Scan range of the task (20-150 kg/h per circuit), step 5 kg/h."""
EQUAL_FLOW_CURVE_KG_PER_H = tuple(float(f) for f in range(5, 151, 5))
"""Both circuits equal; goes below the range of the task to show where an optimum at its
lower edge really lies."""
HTM_BOILING_CELSIUS = 100.0
SOLID_FRACTION = 0.40
"""Suspension solid fraction of this comparison (run parameter; the config default stays 0.3)."""

_BACKEND = NumpyBackend()
_GAS = GasConfig()

ProfileKind = Literal["linear", "sensible", "condensing"]
StageKind = Literal["equal_outlet", "counter_current", "direct"]


# --------------------------------------------------------------------------- properties, plant
@dataclass(frozen=True)
class Props:
    """Gas properties of a model, J/(kg K)."""

    name: str
    cp_air: float
    cp_vapour: float


REPO = Props("repository", _GAS.cp_dry_air_j_per_kg_k, _GAS.cp_vapour_j_per_kg_k)


def partner_props(inputs: SiInputsV41) -> Props:
    m = inputs.plant.material
    return Props("partner", m.cp_gas_j_per_kg_k, m.cp_vapour_j_per_kg_k)


@dataclass(frozen=True)
class Plant:
    """The data case: exhausts (sources), inlet air of the dryers (sinks), heater, circuits."""

    exhausts: dict[int, c.Stream]
    sink_air: dict[int, c.Stream]
    targets_celsius: dict[int, float]
    htm_flow_kg_per_s: dict[int, float]
    htm_cp_j_per_kg_k: float
    efficiency: float
    heating_value_j_per_kg: float
    ambient_celsius: float
    partner: Props

    def with_htm_flows_kg_per_h(self, flows: tuple[float, float]) -> Plant:
        return dataclasses.replace(
            self, htm_flow_kg_per_s={1: flows[0] / 3600.0, 2: flows[1] / 3600.0}
        )


def plant_from_inputs(
    inputs: SiInputsV41, exhaust_override: dict[int, tuple[float, float]] | None = None
) -> Plant:
    """Plant of the partner workbook; ``exhaust_override[i] = (T degC, Y kg/kg)`` replaces source i."""
    p = inputs.plant
    m = p.material
    dryers = {1: p.dryer_1, 2: p.dryer_2}
    exhausts = {i: c.exhaust_stream(d, c.dryer_balance(d, m)) for i, d in dryers.items()}
    for i, (t, y) in (exhaust_override or {}).items():
        exhausts[i] = c.Stream(exhausts[i].dry_air_kg_per_s, y, t)
    return Plant(
        exhausts=exhausts,
        sink_air={
            i: c.Stream(
                d.air_mass_flow_kg_per_s, d.air_humidity_kg_per_kg, p.ambient.temperature_celsius
            )
            for i, d in dryers.items()
        },
        targets_celsius={i: d.air_temperature_celsius for i, d in dryers.items()},
        htm_flow_kg_per_s={1: inputs.htm_flow_kg_per_s[0], 2: inputs.htm_flow_kg_per_s[1]},
        htm_cp_j_per_kg_k=inputs.cp_htm_j_per_kg_k,
        efficiency=m.heater_efficiency,
        heating_value_j_per_kg=m.lower_heating_value_j_per_kg,
        ambient_celsius=p.ambient.temperature_celsius,
        partner=partner_props(inputs),
    )


def capacity(stream: c.Stream, props: Props) -> float:
    """``M (cp_air + Y cp_vapour)`` [W/K]."""
    return stream.dry_air_kg_per_s * (props.cp_air + stream.humidity_kg_per_kg * props.cp_vapour)


# --------------------------------------------------------------------------- thermodynamics
def dew_point(y: float) -> float:
    return float(dew_point_celsius(_BACKEND, _GAS, _BACKEND.asarray(np.asarray(y))))


def saturation_humidity(t: np.ndarray) -> np.ndarray:
    return np.asarray(
        saturation_humidity_kg_per_kg(_BACKEND, _GAS, _BACKEND.asarray(np.asarray(t))),
        dtype=np.float64,
    )


def latent_heat(t: float) -> float:
    return float(latent_heat_j_per_kg(_BACKEND, _BACKEND.asarray(np.asarray(t))))


@dataclass(frozen=True)
class Profile:
    """Heat released by the exhaust ``q_w`` [W] against its temperature ``t_c`` [degC]."""

    q_w: np.ndarray
    t_c: np.ndarray

    @property
    def total_w(self) -> float:
        return float(self.q_w[-1])

    def temperature_at(self, q_w: float) -> float:
        return float(np.interp(q_w, self.q_w, self.t_c))


def exhaust_profile(
    stream: c.Stream, props: Props, kind: ProfileKind, *, nodes: int = PROFILE_NODES
) -> Profile:
    """Release curve of an exhaust.

    ``linear``: sensible heat down to ``T_kalt`` (the dew point is not a limit: what the partner
    model assumes); ``sensible``: sensible heat down to dew point + margin (no condensation);
    ``condensing``: as :mod:`twinlink_ml.heatnetwork.approach`, condensation from the dew point on,
    latent heat at ``T_kalt``, down to ``T_kalt``.
    """
    m, y, t_out = stream.dry_air_kg_per_s, stream.humidity_kg_per_kg, stream.temperature_celsius
    cap = capacity(stream, props)
    dew = dew_point(y)
    floor = dew + DEW_POINT_MARGIN_K if kind == "sensible" else T_KALT_CELSIUS
    if t_out <= floor:
        return Profile(np.zeros(2), np.array([t_out, t_out]))
    if kind != "condensing":
        return Profile(np.array([0.0, cap * (t_out - floor)]), np.array([t_out, floor]))
    knee = float(np.clip(dew, floor, t_out))
    fractions = np.linspace(0.0, 1.0, nodes + 1)[1:]
    t_nodes = np.concatenate([[t_out, knee], knee + (floor - knee) * fractions])
    y_s = saturation_humidity(t_nodes)
    q_nodes = cap * (t_out - t_nodes) + m * np.maximum(y - y_s, 0.0) * latent_heat(T_KALT_CELSIUS)
    return Profile(np.maximum.accumulate(np.maximum(q_nodes, 0.0)), t_nodes)


def condensate_kg_per_h(stream: c.Stream, t_exhaust_out: float) -> float:
    """Water that would leave the exhaust below its saturation humidity [kg/h]."""
    y_s = float(saturation_humidity(np.asarray(t_exhaust_out)))
    return stream.dry_air_kg_per_s * max(stream.humidity_kg_per_kg - y_s, 0.0) * 3600.0


# --------------------------------------------------------------------------- stages
@dataclass(frozen=True)
class StageOut:
    """One pre-heating stage (temperatures degC, ``q_w`` W)."""

    source: int
    sink: int
    q_w: float
    gas_in_celsius: float
    gas_out_celsius: float
    exhaust_in_celsius: float
    exhaust_out_celsius: float
    htm_cold_celsius: float | None
    """``T1``: medium leaving the pre-heating exchanger (``None``: no circuit, direct exchange)."""
    htm_hot_celsius: float | None
    """``T2``: medium leaving the recovery exchanger."""
    binding: str
    min_approach_recovery_k: float
    """Smallest approach in the recovery exchanger (exhaust against medium; direct: against gas)."""
    min_approach_preheater_k: float | None
    exhaust_dew_point_celsius: float
    condensate_modelled_kg_per_h: float
    """Condensate the model takes out (0 unless the profile is condensing)."""
    condensate_unmodelled_kg_per_h: float
    """Water the exhaust would drop below saturation at its outlet temperature, if the model
    ignores condensation (the partner's (4.1) does)."""

    @property
    def cools_gas(self) -> bool:
        return self.q_w < -1e-9


def _min_g(profile: Profile, inverse_capacity: float, q: float) -> float:
    mask = profile.q_w <= q
    g_nodes = profile.t_c[mask] + profile.q_w[mask] * inverse_capacity
    return float(min(g_nodes.min(), profile.temperature_at(q) + q * inverse_capacity))


def solve_stage(
    profile: Profile,
    gas_capacity_w_per_k: float,
    gas_in_celsius: float,
    gas_target_celsius: float,
    htm_capacity_w_per_k: float | None,
    dt_recovery_k: float,
    dt_preheater_k: float,
) -> tuple[float, str]:
    """Largest duty [W] of one stage whose approaches stay >= the given minima; and what binds.

    Counter-current, unlimited area. ``htm_capacity_w_per_k=None``: direct exchange exhaust -> gas
    with ``dt_recovery_k`` as the minimum along the exchanger (the module's stage). Otherwise the
    circuit: recovery exchanger exhaust -> medium (``dt_recovery_k``) and pre-heating exchanger
    medium -> gas (``dt_preheater_k``); the medium returns to the recovery exchanger at ``T1`` and
    leaves it at ``T2 = T1 + Q / B``. The most favourable ``T1`` is the lowest the pre-heating
    exchanger admits, ``T1 = T_h,in + dt + max(0, Q (1/C - 1/B))`` (its cold end, or its warm end
    for a medium with more capacity than the gas), so that the condition on the recovery exchanger,
    ``min over q <= Q of (T_src(q) + q/B) - Q/B - T1 >= dt_recovery``, is monotone in ``Q``.
    """
    cg = gas_capacity_w_per_k
    demand = max(cg * (gas_target_celsius - gas_in_celsius), 0.0)
    cap_q = min(profile.total_w, demand)

    def margin(q: float) -> float:
        if htm_capacity_w_per_k is None:
            return _min_g(profile, 1.0 / cg, q) - q / cg - gas_in_celsius - dt_recovery_k
        b = htm_capacity_w_per_k
        t1 = gas_in_celsius + dt_preheater_k + max(0.0, q * (1.0 / cg - 1.0 / b))
        return _min_g(profile, 1.0 / b, q) - q / b - t1 - dt_recovery_k

    if cap_q <= 0.0 or margin(0.0) < 0.0:
        return 0.0, "no_driving_force"
    if margin(cap_q) >= 0.0:
        return cap_q, "sink_target" if demand <= profile.total_w else "exhaust_floor"
    lo, hi = 0.0, cap_q
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if margin(mid) >= 0.0 else (lo, mid)
        if hi - lo <= 1e-13 * cap_q:
            break
    return lo, "approach"


@dataclass(frozen=True)
class Model:
    """One way of computing a stage."""

    name: str
    props: Props
    kind: StageKind
    profile: ProfileKind = "linear"
    dt_min_k: float = 0.0
    split: Literal["half", "each"] = "half"
    """``half``: ``dt_min_k`` is the approach through the circuit, ``dt_min_k / 2`` per exchanger;
    ``each``: ``dt_min_k`` per exchanger."""
    nodes: int = PROFILE_NODES


def _approach_diagnostics(
    profile: Profile,
    q: float,
    gas_in: float,
    cg: float,
    b: float | None,
    t1: float | None,
) -> tuple[float, float | None]:
    """Smallest approach in the recovery exchanger and (circuit) in the pre-heating exchanger."""
    if q <= 0.0:
        return math.nan, None
    mask = profile.q_w <= q
    qs = np.concatenate([profile.q_w[mask], [q]])
    ts = np.concatenate([profile.t_c[mask], [profile.temperature_at(q)]])
    if b is None or t1 is None:
        return float(np.min(ts - (gas_in + (q - qs) / cg))), None
    recovery = float(np.min(ts - (t1 + (q - qs) / b)))
    t2 = t1 + q / b
    preheater = min(t1 - gas_in, t2 - (gas_in + q / cg))
    return recovery, float(preheater)


def run_stage(
    plant: Plant, model: Model, source: int, sink: int, gas: c.Stream, htm_kg_per_s: float
) -> StageOut:
    exhaust = plant.exhausts[source]
    props = model.props
    cg = capacity(gas, props)
    a = capacity(exhaust, props)
    b = htm_kg_per_s * plant.htm_cp_j_per_kg_k
    t_c, t_h = exhaust.temperature_celsius, gas.temperature_celsius
    dew = dew_point(exhaust.humidity_kg_per_kg)
    if model.kind == "equal_outlet":
        q = (t_c - t_h) / (1.0 / a + 1.0 / b + 1.0 / cg)
        t1 = t_h + q / cg
        t2 = t1 + q / b
        t_ex_out = t_c - q / a
        return StageOut(
            source, sink, q, t_h, t1, t_c, t_ex_out, t1, t2, "equal_outlet", 0.0, 0.0, dew, 0.0,
            condensate_kg_per_h(exhaust, t_ex_out),
        )  # fmt: skip
    dt_total = model.dt_min_k
    profile = exhaust_profile(exhaust, props, model.profile, nodes=model.nodes)
    target = plant.targets_celsius[sink]
    if model.kind == "direct":
        q, binding = solve_stage(profile, cg, t_h, target, None, dt_total, 0.0)
        t1 = t2 = None
        b_arg = None
    else:
        dt = dt_total / 2.0 if model.split == "half" else dt_total
        q, binding = solve_stage(profile, cg, t_h, target, b, dt, dt)
        t1 = t_h + dt + max(0.0, q * (1.0 / cg - 1.0 / b))
        t2 = t1 + q / b
        b_arg = b
    t_ex_out = profile.temperature_at(q) if q > 0.0 else t_c
    rec, pre = _approach_diagnostics(profile, q, t_h, cg, b_arg, t1)
    modelled = condensate_kg_per_h(exhaust, t_ex_out) if model.profile == "condensing" else 0.0
    unmodelled = condensate_kg_per_h(exhaust, t_ex_out) if model.profile != "condensing" else 0.0
    return StageOut(
        source, sink, q, t_h, t_h + q / cg, t_c, t_ex_out, t1, t2, binding, rec, pre, dew,
        modelled, unmodelled,
    )  # fmt: skip


@dataclass(frozen=True)
class ScenarioOut:
    """One scenario: stages, heaters, totals."""

    case: int
    stages: tuple[StageOut, ...]
    heater_in_celsius: dict[int, float]
    heater_kw: dict[int, float]
    gas_kg_per_h: float
    recovered_kw: float
    demand_kw: float


def run_scenario(
    plant: Plant, model: Model, case: int, flows_kg_per_h: tuple[float, float] | None = None
) -> ScenarioOut:
    """One scenario (``case`` 0: no recovery). ``flows_kg_per_h``: circuit 1, 2 (default: plant)."""
    flows = (
        {1: flows_kg_per_h[0] / 3600.0, 2: flows_kg_per_h[1] / 3600.0}
        if flows_kg_per_h is not None
        else plant.htm_flow_kg_per_s
    )
    orders = ((), ()) if case == 0 else CASES[case]
    stages: list[StageOut] = []
    heater_in: dict[int, float] = {}
    heater_kw: dict[int, float] = {}
    demand = 0.0
    for sink, sources in ((1, orders[0]), (2, orders[1])):
        gas = plant.sink_air[sink]
        for source in sources:
            stage = run_stage(plant, model, source, sink, gas, flows[source])
            stages.append(stage)
            gas = c.Stream(gas.dry_air_kg_per_s, gas.humidity_kg_per_kg, stage.gas_out_celsius)
        cg = capacity(gas, model.props)
        heater_in[sink] = gas.temperature_celsius
        heater_kw[sink] = (
            max(cg * (plant.targets_celsius[sink] - gas.temperature_celsius), 0.0) / 1e3
        )
        demand += (
            capacity(plant.sink_air[sink], model.props)
            * (plant.targets_celsius[sink] - plant.ambient_celsius)
            / 1e3
        )
    gas_kg_per_s = sum(heater_kw.values()) * 1e3 / (plant.efficiency * plant.heating_value_j_per_kg)
    return ScenarioOut(
        case, tuple(stages), heater_in, heater_kw, gas_kg_per_s * 3600.0,
        sum(s.q_w for s in stages) / 1e3, demand,
    )  # fmt: skip


# --------------------------------------------------------------------------- upper bound (d)
USED_SINKS = {0: (), 1: (1, 2), 2: (1, 2), 3: (1,), 4: (2,), 5: (1,), 6: (2,)}


def pinch_bound(
    plant: Plant, case: int, props: Props, dt_min_k: float, kind: Literal["sensible", "condensing"]
) -> dict[str, float]:
    """Pinch target of the data case: the most that any wiring of this case can recover.

    Sources: both exhausts; sinks: those that the case connects (the others keep their full
    heater). ``dt_min_k`` is the approach through any chain (0: the thermodynamic limit).
    """
    gas = GasConfig(cp_dry_air_j_per_kg_k=props.cp_air, cp_vapour_j_per_kg_k=props.cp_vapour)
    sources = [
        HeatSourceConfig(f"S{i}", e.dry_air_kg_per_s, e.temperature_celsius, e.humidity_kg_per_kg)
        for i, e in plant.exhausts.items()
    ]
    used = USED_SINKS[case]
    sinks = [
        HeatSinkConfig(
            f"K{i}",
            plant.sink_air[i].dry_air_kg_per_s,
            plant.sink_air[i].humidity_kg_per_kg,
            plant.ambient_celsius,
            plant.targets_celsius[i],
        )
        for i in (1, 2)
    ]
    demand_all = sum(cold_segment_for_sink(s, gas).duty_kw() for s in sinks)
    if not used:
        return {
            "recovered_kw": 0.0,
            "hot_utility_kw": demand_all,
            "gas_kg_per_h": _gas(plant, demand_all),
        }
    hot: list[StreamSegment] = []
    for s in sources:
        if kind == "condensing":
            hot.extend(
                hot_segments_for_source(s, gas, T_KALT_CELSIUS, n_segments_below_dew_point=200)
            )
        else:
            floor = dew_point_of_source_celsius(s, gas) + DEW_POINT_MARGIN_K
            if s.outlet_temperature_celsius > floor:
                cap_kw = (
                    s.dry_air_mass_flow_kg_per_s
                    * (
                        gas.cp_dry_air_j_per_kg_k
                        + s.outlet_humidity_kg_per_kg * gas.cp_vapour_j_per_kg_k
                    )
                    / 1e3
                )
                hot.append(StreamSegment(s.name, True, s.outlet_temperature_celsius, floor, cap_kw))
    cold = [cold_segment_for_sink(s, gas) for s in sinks if int(s.name[1:]) in used]
    result = compute_pinch_target(hot, cold, minimum_pinch_k=dt_min_k)
    demand_used = sum(seg.duty_kw() for seg in cold)
    unused = demand_all - demand_used
    return {
        "recovered_kw": demand_used - result.minimum_hot_utility_kw,
        "hot_utility_kw": result.minimum_hot_utility_kw + unused,
        "gas_kg_per_h": _gas(plant, result.minimum_hot_utility_kw + unused),
        "pinch_hot_side_celsius": result.pinch_temperature_hot_side_celsius,
    }


def _gas(plant: Plant, heater_kw: float) -> float:
    return heater_kw * 1e3 / (plant.efficiency * plant.heating_value_j_per_kg) * 3600.0


# --------------------------------------------------------------------------- the module (reference)
def module_direct(
    plant: Plant, case: int, dt_min_k: float, mode: Literal["sensible", "condensing"]
) -> dict[str, Any]:
    """The same scenario with the heat-network module: direct exchange, no circuit."""
    config = HeatNetworkConfig(
        sources=tuple(
            HeatSourceConfig(
                f"S{i}", e.dry_air_kg_per_s, e.temperature_celsius, e.humidity_kg_per_kg
            )
            for i, e in plant.exhausts.items()
        ),
        sinks=tuple(
            HeatSinkConfig(
                f"K{i}",
                plant.sink_air[i].dry_air_kg_per_s,
                plant.sink_air[i].humidity_kg_per_kg,
                plant.ambient_celsius,
                plant.targets_celsius[i],
            )
            for i in (1, 2)
        ),
        minimum_pinch_k=dt_min_k,
        dew_point_approach_k=DEW_POINT_MARGIN_K,
        fuel_lower_heating_value_j_per_kg=plant.heating_value_j_per_kg,
        heater_efficiency=plant.efficiency,
        recovery_mode=mode,
        cold_temperature_celsius=T_KALT_CELSIUS,
        approach_check="along_exchanger",
        approach_profile_nodes=PROFILE_NODES,
    )
    orders = ((), ()) if case == 0 else CASES[case]
    stages = tuple(
        (f"S{s}", f"K{sink}") for sink, srcs in ((1, orders[0]), (2, orders[1])) for s in srcs
    )
    result = evaluate_cascade(
        config,
        stages,
        tuple(SourceState.from_config(s) for s in config.sources),
        tuple(SinkState.from_config(s) for s in config.sinks),
    )
    return {
        "stage_q_kw": [float(x[0]) for x in result.stage_recovered_power_kw],
        "gas_kg_per_h": float(np.sum(result.fuel_mass_flow_kg_per_s)) * 3600.0,
    }


# --------------------------------------------------------------------------- laboratory model (ii)
def bed_exhaust_at_partner_inputs(solid_fraction: float) -> dict[str, Any]:
    """Dryer 2 of the partner workbook in the laboratory grey-box model (an extrapolation).

    Inputs of dryer 2 (300 degC, 12 kg/h spray, 100 kg/h dry air of 6 g/kg at 15 degC); the gas
    velocity is the root of the dry-air flow. The bounds of the process configuration are widened
    in memory for this run only (the files stay untouched), as in ``report_dryer_block_vs_bed.py``
    (Befund 75, 9.6), where the same state was computed with the solid fraction 0.4.
    """
    from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
    from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.operating_point import (
        DEFAULT_SEARCH_MAX_ITERATIONS,
        DEFAULT_SEARCH_TOLERANCE,
        build_variant_config,
    )
    from twinlink_ml.simulation.steady_state import solve_steady_state
    from twinlink_ml.simulation.thermo import relative_humidity_from_absolute_humidity

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml")
    t_in, spray, air_kg_h, y_in, t_s, t_amb, nozzle_bar = 300.0, 12.0, 100.0, 0.006, 15.0, 15.0, 1.0
    phi = float(
        relative_humidity_from_absolute_humidity(
            _BACKEND, config.gas, np.array(y_in), np.array(t_amb)
        )
    )

    def variant_for(u: float) -> Any:
        actions = dataclasses.replace(
            config.actions,
            inlet_temperature_celsius=dataclasses.replace(
                config.actions.inlet_temperature_celsius, maximum=t_in, nominal=t_in
            ),
            gas_velocity_m_per_s=dataclasses.replace(
                config.actions.gas_velocity_m_per_s, nominal=u
            ),
            spray_rate_kg_per_h=dataclasses.replace(
                config.actions.spray_rate_kg_per_h, maximum=spray, nominal=spray
            ),
            nozzle_pressure_bar=dataclasses.replace(
                config.actions.nozzle_pressure_bar, nominal=nozzle_bar
            ),
        )
        disturbances = dataclasses.replace(
            config.disturbances,
            ambient_temperature_celsius=dataclasses.replace(
                config.disturbances.ambient_temperature_celsius, nominal=t_amb
            ),
            ambient_relative_humidity=dataclasses.replace(
                config.disturbances.ambient_relative_humidity, nominal=phi
            ),
            suspension_solid_fraction=dataclasses.replace(
                config.disturbances.suspension_solid_fraction, nominal=solid_fraction
            ),
        )
        operation = dataclasses.replace(config.operation, suspension_temperature_celsius=t_s)
        return dataclasses.replace(
            config, actions=actions, disturbances=disturbances, operation=operation
        )

    def solve(u: float) -> Any:
        variant = build_variant_config(
            variant_for(u), t_in=t_in, u_gas=u, p_nozzle=nozzle_bar, t_amb=t_amb, phi_amb=phi
        )
        return solve_steady_state(
            GreyBoxFluidBedGranulator(variant),
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
        )

    target = air_kg_h / 3600.0
    u_star = brentq(lambda u: solve(u).dry_air_mass_flow_kg_per_s - target, 1.3, 1.9, xtol=1e-6)
    solution = solve(u_star)
    return {
        "label": "extrapolation: every input of dryer 2 is outside the laboratory model's range",
        "solid_fraction": solid_fraction,
        "gas_velocity_m_per_s": u_star,
        "dry_air_kg_per_h": solution.dry_air_mass_flow_kg_per_s * 3600.0,
        "exhaust_temperature_celsius": solution.bed_temperature_celsius,
        "exhaust_humidity_kg_per_kg": solution.outlet_humidity_kg_per_kg,
        "product_moisture_kg_per_kg": solution.particle_moisture_kg_per_kg,
        "feasible": bool(solution.feasible),
        "residual_inf_norm": solution.residual_inf_norm,
        "config_fingerprint_base": config.fingerprint,
    }


# --------------------------------------------------------------------------- serialisation
def stage_dict(s: StageOut) -> dict[str, Any]:
    return {
        "source": s.source,
        "sink": s.sink,
        "recovered_kw": s.q_w / 1e3,
        "gas_in_celsius": s.gas_in_celsius,
        "gas_out_celsius": s.gas_out_celsius,
        "exhaust_in_celsius": s.exhaust_in_celsius,
        "exhaust_out_celsius": s.exhaust_out_celsius,
        "htm_cold_celsius": s.htm_cold_celsius,
        "htm_hot_celsius": s.htm_hot_celsius,
        "binding": s.binding,
        "cools_gas": s.cools_gas,
        "min_approach_recovery_k": s.min_approach_recovery_k,
        "min_approach_preheater_k": s.min_approach_preheater_k,
        "exhaust_dew_point_celsius": s.exhaust_dew_point_celsius,
        "condensate_modelled_kg_per_h": s.condensate_modelled_kg_per_h,
        "condensate_unmodelled_kg_per_h": s.condensate_unmodelled_kg_per_h,
        "htm_above_boiling_at_1_atm": bool(
            s.htm_hot_celsius is not None and s.htm_hot_celsius > HTM_BOILING_CELSIUS
        ),
    }


def scenario_dict(r: ScenarioOut) -> dict[str, Any]:
    return {
        "stages": [stage_dict(s) for s in r.stages],
        "heater_in_celsius": {str(k_): v for k_, v in r.heater_in_celsius.items()},
        "heater_kw": {str(k_): v for k_, v in r.heater_kw.items()},
        "gas_kg_per_h": r.gas_kg_per_h,
        "recovered_kw": r.recovered_kw,
        "demand_kw": r.demand_kw,
    }


# --------------------------------------------------------------------------- the model families
def models_for(
    partner: Props, dt_min_k: float, split: Literal["half", "each"] = "half"
) -> dict[str, Model]:
    return {
        "a": Model("a: partner model as delivered", partner, "equal_outlet"),
        "b": Model(
            "b: counter-current, no condensation",
            REPO,
            "counter_current",
            "sensible",
            dt_min_k,
            split,
        ),
        "c": Model(
            "c: counter-current, condensation",
            REPO,
            "counter_current",
            "condensing",
            dt_min_k,
            split,
        ),
    }


def decomposition_steps(
    partner: Props, dt_min_k: float, order: Literal["props_first", "props_last"]
) -> list[Model]:
    """Chain a -> c: each entry adds one assumption of the independent calculation."""

    def cc(props: Props, kind: ProfileKind, dt: float, name: str) -> Model:
        return Model(name, props, "counter_current", kind, dt)

    steps = [Model("a: partner model as delivered", partner, "equal_outlet")]
    if order == "props_first":
        steps.append(Model("c_p of the gas (partner -> repository)", REPO, "equal_outlet"))
        p = REPO
    else:
        p = partner
    steps += [
        cc(p, "linear", 0.0, "counter-current, unlimited area (no dew-point limit)"),
        cc(p, "linear", dt_min_k, "approach dT_min along the exchangers"),
        cc(p, "sensible", dt_min_k, "dew-point limit (no condensation)"),
        cc(p, "condensing", dt_min_k, "condensation below the dew point"),
    ]
    if order == "props_last":
        steps.append(cc(REPO, "condensing", dt_min_k, "c_p of the gas (partner -> repository)"))
    return steps


def decompose(plant: Plant, dt_min_k: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for order in ("props_first", "props_last"):
        steps = decomposition_steps(plant.partner, dt_min_k, order)
        per_case = {}
        for case in SCENARIOS:
            values = [run_scenario(plant, m, case) for m in steps]
            per_case[str(case)] = [
                {
                    "step": m.name,
                    "recovered_kw": v.recovered_kw,
                    "gas_kg_per_h": v.gas_kg_per_h,
                    "delta_recovered_kw": (v.recovered_kw - values[i - 1].recovered_kw)
                    if i
                    else None,
                    "delta_gas_kg_per_h": (v.gas_kg_per_h - values[i - 1].gas_kg_per_h)
                    if i
                    else None,
                }
                for i, (m, v) in enumerate(zip(steps, values, strict=True))
            ]
        out[order] = per_case
    return out


# --------------------------------------------------------------------------- WU3: circuit flows
def flow_scan(plant: Plant, model: Model) -> dict[str, Any]:
    """Gas and recovered heat on the grid of circuit flows, the optimum and the plateau.

    ``argmin``: smallest total flow among the grid points whose gas is within 1e-9 (relative) of
    the minimum. ``plateau_equal``: flows (both circuits equal) whose gain over the sheet's
    20 / 20 kg/h is at least 99 % of the best equal-flow gain. ``equal_flow_curve`` (5-150 kg/h)
    also lists the hottest medium temperature, whether a stage cools the gas, and the water the
    exhaust would drop below saturation if condensation is not modelled.
    """
    flows = HTM_FLOWS_KG_PER_H
    cases: dict[str, Any] = {}
    for case in SCENARIOS:
        results = [[run_scenario(plant, model, case, (f1, f2)) for f2 in flows] for f1 in flows]
        arr = np.array([[r.gas_kg_per_h for r in row] for row in results])
        best = float(arr.min())
        tol = 1e-9 * max(abs(best), 1.0)
        candidates = [
            (i, j)
            for i, j in itertools.product(range(len(flows)), repeat=2)
            if arr[i, j] <= best + tol
        ]
        i, j = min(candidates, key=lambda ij: (flows[ij[0]] + flows[ij[1]], ij))
        equal = np.diag(arr)
        best_equal = float(equal.min())
        gain = float(arr[0, 0]) - best_equal
        plateau = [f for f, g in zip(flows, equal, strict=True) if g <= best_equal + 0.01 * gain]
        curve = []
        for f in EQUAL_FLOW_CURVE_KG_PER_H:
            r = run_scenario(plant, model, case, (f, f))
            curve.append(
                {
                    "flow_kg_per_h": f,
                    "gas_kg_per_h": r.gas_kg_per_h,
                    "recovered_kw": r.recovered_kw,
                    "stage_recovered_kw": [s.q_w / 1e3 for s in r.stages],
                    "htm_hot_max_celsius": max(s.htm_hot_celsius or math.nan for s in r.stages),
                    "any_stage_cools_gas": any(s.cools_gas for s in r.stages),
                    "unmodelled_condensate_kg_per_h": sum(
                        s.condensate_unmodelled_kg_per_h for s in r.stages
                    ),
                }
            )
        cases[str(case)] = {
            "gas_kg_per_h": arr.tolist(),
            "recovered_kw": [[r.recovered_kw for r in row] for row in results],
            "argmin_flows_kg_per_h": [flows[i], flows[j]],
            "argmin_gas_kg_per_h": best,
            "argmin_on_the_edge_of_the_range": bool(
                flows[i] in (flows[0], flows[-1]) or flows[j] in (flows[0], flows[-1])
            ),
            "gas_at_20_20": float(arr[0, 0]),
            "gas_at_150_150": float(arr[-1, -1]),
            "best_equal_flow_kg_per_h": flows[int(np.argmin(equal))],
            "plateau_equal_flows_kg_per_h": [min(plateau), max(plateau)],
            "equal_flow_curve": curve,
        }
    return {"flows_kg_per_h": list(flows), "cases": cases}


# --------------------------------------------------------------------------- WU0: mapping
def mapping_table(plant: Plant) -> list[dict[str, Any]]:
    rows = []
    for case in SCENARIOS:
        orders = CASES[case]
        for sink, sources in ((1, orders[0]), (2, orders[1])):
            for position, source in enumerate(sources, start=1):
                e = plant.exhausts[source]
                g = plant.sink_air[sink]
                rows.append(
                    {
                        "case": case,
                        "sink": sink,
                        "stage_position": position,
                        "stage_label_in_sheet": "low" if position == 1 else "high",
                        "source": source,
                        "source_is": f"exhaust of dryer {source}",
                        "sink_is": f"inlet air of dryer {sink}",
                        "circuit": source,
                        "htm_flow_kg_per_h": plant.htm_flow_kg_per_s[source] * 3600.0,
                        "exhaust_dry_air_kg_per_h": e.dry_air_kg_per_s * 3600.0,
                        "exhaust_temperature_celsius": e.temperature_celsius,
                        "exhaust_humidity_g_per_kg": e.humidity_kg_per_kg * 1e3,
                        "gas_dry_air_kg_per_h": g.dry_air_kg_per_s * 3600.0,
                        "gas_humidity_g_per_kg": g.humidity_kg_per_kg * 1e3,
                        "gas_intake_celsius": g.temperature_celsius,
                        "gas_target_celsius": plant.targets_celsius[sink],
                        "module": {
                            "source": f"HeatSourceConfig(name='S{source}')",
                            "sink": f"HeatSinkConfig(name='K{sink}')",
                            "stage": f"('S{source}', 'K{sink}') as position {position} of the stage list",
                            "circuit": "no counterpart (see module_capability)",
                        },
                    }
                )
    return rows


def module_capability() -> dict[str, Any]:
    fields = [f.name for f in dataclasses.fields(HeatNetworkConfig)]
    return {
        "can_represent_intermediate_circuit": False,
        "reasons": [
            "a stage of evaluate_cascade is a pair (source, sink); heat goes directly from the "
            "exhaust to the process gas, the medium has no state",
            "HeatNetworkConfig has no field for a medium or its mass flow: " + ", ".join(fields),
            "source and sink capacity rates are m (cp_air + Y cp_vapour) of moist air; the medium "
            "(liquid, B = m cp = 4.2 kJ/(kg K)) has no counterpart, and its temperature level is a "
            "result of the circuit, not a configurable value",
            "the dew-point floor, the profile and the approach check are defined for the exhaust "
            "stream only; a second exchanger with the medium as the hot stream would need a "
            "second pass with a state that evaluate_cascade does not carry",
        ],
        "rebuild": "solve_stage in this script (circuit: recovery exchanger + pre-heating "
        "exchanger, counter-current, minimum approach along each); tested against "
        "evaluate_cascade (direct exchange) and partner_v4_1.loops (equal outlet, counter-current bound)",
        "module_fields": fields,
    }


# --------------------------------------------------------------------------- WU0b: consistency
def _entropy_w_per_k(
    cap_hot: float,
    t_hot_in: float,
    t_hot_out: float,
    cap_cold: float,
    t_cold_in: float,
    t_cold_out: float,
) -> float:
    k_ = 273.15
    return cap_hot * math.log((t_hot_out + k_) / (t_hot_in + k_)) + cap_cold * math.log(
        (t_cold_out + k_) / (t_cold_in + k_)
    )


def consistency_check(
    plant: Plant, partner: Props, label: str, flows_kg_per_h: tuple[float, float] | None = None
) -> dict[str, Any]:
    """Energy balance per exchanger and circuit, second law per stage, total against the pinch."""
    model = Model("a", partner, "equal_outlet")
    stage_rows = []
    violations: list[dict[str, Any]] = []
    scenario_rows = []
    for case in SCENARIOS:
        result = run_scenario(plant, model, case, flows_kg_per_h)
        flows = flows_kg_per_h or tuple(plant.htm_flow_kg_per_s[i] * 3600.0 for i in (1, 2))
        for st in result.stages:
            a = capacity(plant.exhausts[st.source], partner)
            gas_before = st.gas_in_celsius
            sink_air = plant.sink_air[st.sink]
            cg = capacity(sink_air, partner)
            b = flows[st.source - 1] / 3600.0 * plant.htm_cp_j_per_kg_k
            t_c, t_h = st.exhaust_in_celsius, gas_before
            t1, t2, t_ex = st.htm_cold_celsius, st.htm_hot_celsius, st.exhaust_out_celsius
            assert t1 is not None and t2 is not None
            res_ex1 = a * (t_c - t_ex) - b * (t2 - t1)
            res_ex2 = b * (t2 - t1) - cg * (st.gas_out_celsius - t_h)
            s1 = _entropy_w_per_k(a, t_c, t_ex, b, t1, t2)
            s2 = _entropy_w_per_k(b, t2, t1, cg, t_h, st.gas_out_celsius)
            row = {
                "case": case,
                "sink": st.sink,
                "source": st.source,
                "recovered_kw": st.q_w / 1e3,
                "balance_residual_recovery_exchanger_w": res_ex1,
                "balance_residual_preheating_exchanger_w": res_ex2,
                "entropy_production_recovery_exchanger_w_per_k": s1,
                "entropy_production_preheating_exchanger_w_per_k": s2,
                "gas_enters_warmer_than_exhaust": t_h > t_c,
                "htm_hot_celsius": t2,
                "gas_leaves_above_target": st.gas_out_celsius
                > plant.targets_celsius[st.sink] + 1e-9,
                "exhaust_leaves_below_dew_point": st.exhaust_out_celsius
                < st.exhaust_dew_point_celsius,
                "exhaust_dew_point_celsius": st.exhaust_dew_point_celsius,
                "exhaust_out_celsius": st.exhaust_out_celsius,
            }
            stage_rows.append(row)
            where = {
                "case": case,
                "sink": st.sink,
                "source": st.source,
                "stage_gas_in_celsius": t_h,
            }
            tol = 1e-9 * max(abs(st.q_w), 1.0)
            if abs(res_ex1) > tol or abs(res_ex2) > tol:
                violations.append(
                    {
                        **where,
                        "check": "energy balance of an exchanger or the circuit",
                        "value_w": [res_ex1, res_ex2],
                    }
                )
            if min(s1, s2) < -1e-12 or (st.q_w > 1e-9 and t_h > t_c):
                violations.append(
                    {
                        **where,
                        "check": "second law: heat transferred against the temperature gradient",
                        "recovered_kw": st.q_w / 1e3,
                        "entropy_w_per_k": [s1, s2],
                    }
                )
            if st.q_w < -1e-9:
                violations.append(
                    {
                        **where,
                        "check": "stage cools the process gas (it enters warmer than the exhaust; "
                        "the entropy production is positive, so not a second-law violation)",
                        "recovered_kw": st.q_w / 1e3,
                        "entropy_w_per_k": [s1, s2],
                    }
                )
            if st.exhaust_out_celsius < st.exhaust_dew_point_celsius:
                violations.append(
                    {
                        **where,
                        "check": "exhaust leaves below its dew point (condensation neglected by (4.1))",
                        "exhaust_out_celsius": st.exhaust_out_celsius,
                        "dew_point_celsius": st.exhaust_dew_point_celsius,
                    }
                )
            if st.gas_out_celsius > plant.targets_celsius[st.sink] + 1e-9:
                violations.append(
                    {
                        **where,
                        "check": "gas leaves above the target of its sink",
                        "gas_out_celsius": st.gas_out_celsius,
                    }
                )
            if t2 > HTM_BOILING_CELSIUS:
                violations.append(
                    {
                        **where,
                        "check": "heat transfer medium above 100 degC at 1 atm (water, cp 4.2 kJ/(kg K) in the workbook)",
                        "htm_hot_celsius": t2,
                    }
                )
        bound = {
            "condensing_dt0": pinch_bound(plant, case, partner, 0.0, "condensing"),
            "sensible_dt0": pinch_bound(plant, case, partner, 0.0, "sensible"),
        }
        over = result.recovered_kw - bound["condensing_dt0"]["recovered_kw"]
        scenario_rows.append(
            {
                "case": case,
                "recovered_kw": result.recovered_kw,
                "pinch_bound_condensing_dt0_kw": bound["condensing_dt0"]["recovered_kw"],
                "pinch_bound_sensible_dt0_kw": bound["sensible_dt0"]["recovered_kw"],
                "share_of_bound_condensing": result.recovered_kw
                / bound["condensing_dt0"]["recovered_kw"],
                "exceeds_pinch_target": over > 1e-9,
            }
        )
        if over > 1e-9:
            violations.append(
                {
                    "case": case,
                    "check": "total recovery above the pinch target",
                    "recovered_kw": result.recovered_kw,
                    "bound_kw": bound["condensing_dt0"]["recovered_kw"],
                }
            )
    return {
        "label": label,
        "stages": stage_rows,
        "scenarios": scenario_rows,
        "violations": violations,
    }


def workbook_stage_closure(inputs: ExcelInputsV41) -> list[dict[str, Any]]:
    """Closure of the stage energies on Excel's own values (the as-built reproduction, which
    agrees with every stored number to 2e-16): ``E`` of the stage cell against
    ``A (T_c - T2)`` and ``C (T1 - T_h)`` recomputed from the temperatures of the same sheet."""
    cells = ab.scenario_cells_as_built(inputs)
    si = inputs.to_si()
    plant = plant_from_inputs(si)
    partner = plant.partner
    rows = []
    for case in SCENARIOS:
        ci, co = lay.CASE_COLUMNS[case]
        for sink, sources in ((1, CASES[case][0]), (2, CASES[case][1])):
            sink_rows = lay._SINK_ROWS[sink]
            t_in = plant.ambient_celsius
            for position, source in enumerate(sources):
                r_t, r_cold = lay._SOURCE_ROWS[source][2], lay._SOURCE_ROWS[source][4]
                t_c = cells[f"{ci}{r_t}"]
                t2 = cells[f"{co}{r_t}"]
                t1 = cells[f"{ci}{r_cold}"]
                energy_cell = sink_rows["low"][2] if position == 0 else sink_rows["high"][3]
                e_kw = cells[f"{ci}{energy_cell}"]
                a = capacity(plant.exhausts[source], partner)
                cg = capacity(plant.sink_air[sink], partner)
                rows.append(
                    {
                        "case": case,
                        "sink": sink,
                        "source": source,
                        "stage_position": position + 1,
                        "cell_energy_kw": e_kw,
                        "exhaust_side_kw": a * (t_c - t2) / 1e3,
                        "gas_side_kw": cg * (t1 - t_in) / 1e3,
                        "largest_relative_residual": max(
                            abs(e_kw - a * (t_c - t2) / 1e3), abs(e_kw - cg * (t1 - t_in) / 1e3)
                        )
                        / max(abs(e_kw), 1e-12),
                    }
                )
                t_in = t1
    gas_rows = []
    for case in SCENARIOS:
        ci, _ = lay.CASE_COLUMNS[case]
        sheet = cells[f"{ci}41"]
        document = run_scenario(plant, Model("a", partner, "equal_outlet"), case).gas_kg_per_h
        gas_rows.append(
            {
                "case": case,
                "sheet_total_gas_kg_per_h": sheet,
                "balanced_total_gas_kg_per_h": document,
                "ratio": sheet / document,
            }
        )
    return {"stage_energy_closure": rows, "heater_gas": gas_rows}


# --------------------------------------------------------------------------- checks of this script
def validation(plant: Plant) -> dict[str, Any]:
    """The rebuild against the module and against partner_v4_1.loops; numerical floors."""
    out: dict[str, Any] = {}
    # (1) direct exchange against evaluate_cascade
    worst_q = worst_gas = 0.0
    for case in (0, *SCENARIOS):
        for dt in (20.0, 10.0, 5.0, 0.0001):
            for mode in ("sensible", "condensing"):
                ref = module_direct(plant, case, dt, mode)
                mine = run_scenario(plant, Model("d", REPO, "direct", mode, dt), case)
                worst_q = max(
                    [
                        worst_q,
                        *(
                            abs(r - s.q_w / 1e3)
                            for r, s in zip(ref["stage_q_kw"], mine.stages, strict=True)
                        ),
                    ]
                )
                worst_gas = max(worst_gas, abs(ref["gas_kg_per_h"] - mine.gas_kg_per_h))
    out["direct_exchange_against_evaluate_cascade"] = {
        "largest_abs_difference_stage_kw": worst_q,
        "largest_abs_difference_gas_kg_per_h": worst_gas,
    }
    # (2) node count of the condensing profile
    model = Model("c", REPO, "counter_current", "condensing", 5.0)
    out["profile_nodes_400_against_4000"] = {
        "largest_abs_difference_recovered_kw": max(
            abs(
                run_scenario(plant, model, case).recovered_kw
                - run_scenario(plant, dataclasses.replace(model, nodes=4000), case).recovered_kw
            )
            for case in SCENARIOS
        )
    }
    return out


def _stream_enthalpy_w(m: float, y_total: float, t: float, y_gas: float) -> float:
    """Moist air of humidity ``y_gas`` at ``t`` plus its condensate (``y_total - y_gas``), W
    against 0 degC liquid water."""
    return (
        m * (REPO.cp_air * t + y_gas * (2.5008e6 + REPO.cp_vapour * t))
        + m * (y_total - y_gas) * 4180.0 * t
    )


def profile_convention(plant: Plant) -> dict[str, Any]:
    """The module's condensing profile (latent heat at ``T_kalt``, sensible part with the capacity
    of the entering exhaust) against a complete enthalpy balance of moist air with the condensate
    leaving at the exhaust temperature."""
    rows = {}
    for i, e in plant.exhausts.items():
        profile = exhaust_profile(e, REPO, "condensing")
        m, y = e.dry_air_kg_per_s, e.humidity_kg_per_kg

        out = []
        for t in (60.0, 45.0, 35.0, 25.0):
            if t >= e.temperature_celsius:
                continue
            y_s = float(saturation_humidity(np.asarray(t)))
            y_gas = min(y, y_s)
            exact = _stream_enthalpy_w(m, y, e.temperature_celsius, y) - _stream_enthalpy_w(
                m, y, t, y_gas
            )
            module = float(np.interp(t, profile.t_c[::-1], profile.q_w[::-1]))
            out.append(
                {
                    "temperature_celsius": t,
                    "module_profile_kw": module / 1e3,
                    "enthalpy_balance_kw": exact / 1e3,
                    "relative_difference": (module - exact) / exact,
                }
            )
        rows[str(i)] = out
    return rows


# --------------------------------------------------------------------------- main
def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def _find(*names: str) -> Path:
    for name in names:
        if (LOCAL / name).exists():
            return LOCAL / name
    raise FileNotFoundError(f"none of {names} in {LOCAL}")


def intermediate_inputs() -> ExcelInputsV41:
    """Inputs of the intermediate version as communicated (docs/11, section 8): no file."""
    from twinlink_ml.partner_v4.inputs import ExcelDryerInputs, ExcelInputs, ExcelMaterialInputs

    plant = ExcelInputs(
        dryer_1=ExcelDryerInputs(100.0, 6.0, 300.0, 12.0, 0.4, 15.0),
        dryer_2=ExcelDryerInputs(50.0, 6.0, 150.0, 3.0, 0.4, 15.0),
        material=ExcelMaterialInputs(1.2, 45000.0, 80.0, 4.2, 1.0, 2.0, 2.0, 2500.0, 4.2),
        ambient_temperature_celsius=15.0,
        ambient_pressure_atm=1.0,
    )
    return ExcelInputsV41(plant, (40.0, 20.0))


def evaluate_source_variant(plant: Plant) -> dict[str, Any]:
    """WU1 (+ the sensitivities in the convention of ``dT_min``) for one source variant."""
    partner = plant.partner
    out: dict[str, Any] = {
        "exhausts": {
            str(i): {
                "dry_air_kg_per_h": e.dry_air_kg_per_s * 3600.0,
                "temperature_celsius": e.temperature_celsius,
                "humidity_g_per_kg": e.humidity_kg_per_kg * 1e3,
                "dew_point_celsius": dew_point(e.humidity_kg_per_kg),
            }
            for i, e in plant.exhausts.items()
        },
        "no_recovery": {
            "partner_properties": scenario_dict(
                run_scenario(plant, Model("a", partner, "equal_outlet"), 0)
            ),
            "repository_properties": scenario_dict(
                run_scenario(plant, Model("a", REPO, "equal_outlet"), 0)
            ),
        },
        "wu1": {},
        "wu2": {},
        "wu3": {},
    }
    a = models_for(partner, 20.0)["a"]
    out["wu1"]["a"] = {str(case): scenario_dict(run_scenario(plant, a, case)) for case in SCENARIOS}
    for dt in DT_MIN_K:
        for split in ("half", "each"):
            ms = models_for(partner, dt, split)
            key = f"dt{dt:g}" + ("" if split == "half" else "_per_exchanger")
            out["wu1"][key] = {
                "b": {
                    str(case): scenario_dict(run_scenario(plant, ms["b"], case))
                    for case in SCENARIOS
                },
                "c": {
                    str(case): scenario_dict(run_scenario(plant, ms["c"], case))
                    for case in SCENARIOS
                },
                "direct_module_like_sensible": {
                    str(case): scenario_dict(
                        run_scenario(plant, Model("direct", REPO, "direct", "sensible", dt), case)
                    )
                    for case in SCENARIOS
                },
                "direct_module_like_condensing": {
                    str(case): scenario_dict(
                        run_scenario(plant, Model("direct", REPO, "direct", "condensing", dt), case)
                    )
                    for case in SCENARIOS
                },
            }
    for dt in (0.0, *DT_MIN_K):
        out["wu1"][f"d_dt{dt:g}"] = {
            kind: {str(case): pinch_bound(plant, case, REPO, dt, kind) for case in (0, *SCENARIOS)}
            for kind in ("sensible", "condensing")
        }
    out["wu1"]["d_partner_properties_dt0"] = {
        kind: {str(case): pinch_bound(plant, case, partner, 0.0, kind) for case in (0, *SCENARIOS)}
        for kind in ("sensible", "condensing")
    }
    for dt in DT_MIN_K:
        out["wu2"][f"dt{dt:g}"] = decompose(plant, dt)
    out["wu3"]["a"] = flow_scan(plant, a)
    for dt in DT_MIN_K:
        out["wu3"][f"c_dt{dt:g}"] = flow_scan(plant, models_for(partner, dt)["c"])
    out["wu3"]["b_dt10"] = flow_scan(plant, models_for(partner, 10.0)["b"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workbook", type=Path, default=None)
    parser.add_argument("--name", default="heat_recovery_comparison")
    parser.add_argument("--solid-fraction", type=float, default=SOLID_FRACTION)
    args = parser.parse_args()
    target = ROOT / "artifacts" / "runs" / f"{args.name}.json"
    if target.exists():
        print(f"{target} exists; choose a new --name (artifacts are never overwritten)")
        return 1
    workbook = args.workbook or _find("Model v4.1.xlsx", "Model_v4_1.xlsx")
    inputs = load_excel_inputs_v41(workbook)
    si = inputs.to_si()
    plant_i = plant_from_inputs(si)
    bed = bed_exhaust_at_partner_inputs(args.solid_fraction)
    plant_ii = plant_from_inputs(
        si, {2: (bed["exhaust_temperature_celsius"], bed["exhaust_humidity_kg_per_kg"])}
    )
    partner = plant_i.partner
    intermediate = plant_from_inputs(intermediate_inputs().to_si())

    record: dict[str, Any] = {
        "provenance": {
            "git/commit": _git("rev-parse", "HEAD").strip(),
            "git/dirty": bool(_git("status", "--porcelain").strip()),
            "python": sys.version.split()[0],
            **require_code_location(ROOT),
        },
        "parameters": {
            "dt_min_k_through_the_circuit": list(DT_MIN_K),
            "dt_min_convention": "half: dT_min / 2 per exchanger of the circuit (chain effective = dT_min); "
            "each: dT_min per exchanger (chain effective 2 dT_min), keys *_per_exchanger",
            "dew_point_margin_k": DEW_POINT_MARGIN_K,
            "t_kalt_celsius": T_KALT_CELSIUS,
            "profile_nodes": PROFILE_NODES,
            "htm_flows_kg_per_h": list(HTM_FLOWS_KG_PER_H),
            "htm_flow_sheet_kg_per_h": [
                plant_i.htm_flow_kg_per_s[1] * 3600.0,
                plant_i.htm_flow_kg_per_s[2] * 3600.0,
            ],
            "suspension_solid_fraction_of_the_laboratory_model_run": args.solid_fraction,
            "partner_properties": dataclasses.asdict(partner),
            "repository_properties": dataclasses.asdict(REPO),
            "heater_efficiency": plant_i.efficiency,
            "heating_value_j_per_kg": plant_i.heating_value_j_per_kg,
            "htm_cp_j_per_kg_k": plant_i.htm_cp_j_per_kg_k,
            "workbook": workbook.name,
        },
        "inputs_sheet_units": {
            "plant": dataclasses.asdict(inputs.plant),
            "htm_flow_kg_per_h": list(inputs.htm_flow_kg_per_h),
        },
        "wu0": {
            "mapping": mapping_table(plant_i),
            "module_capability": module_capability(),
            "validation_of_the_rebuild": validation(plant_i),
        },
        "wu0b": {
            "v4_1": consistency_check(
                plant_i, partner, "Model v4.1, circuit flows as in the sheet"
            ),
            "v4_1_workbook_cells": workbook_stage_closure(inputs),
            "v4_1_flows_150": consistency_check(
                plant_i, partner, "Model v4.1, circuits 150 kg/h", (150.0, 150.0)
            ),
            "intermediate_version_positive_control": consistency_check(
                intermediate,
                partner,
                "intermediate version (inputs as communicated), circuits 40 / 20 kg/h",
            ),
        },
        "profile_convention": profile_convention(plant_i),
        "laboratory_model_dryer_2": bed,
        "source_variants": {
            "i_partner_exhausts": evaluate_source_variant(plant_i),
            "ii_laboratory_model_dryer_2": evaluate_source_variant(plant_ii),
        },
    }
    target.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
