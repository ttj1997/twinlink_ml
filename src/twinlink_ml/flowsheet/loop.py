"""The heat transfer loop: recovery exchanger -> medium -> pre-heating exchanger.

One pre-heating stage of the partner model is a circuit of a heat transfer medium between an
exhaust and the process gas of a sink (equations 4.1/4.2 of the equation document). The loop
couples two exchangers through the medium temperature, so it is solved as a whole, in one of
three ways (:class:`StageModel`):

``equal_outlet``
    The partner's assumption: both fluids of each exchanger leave at the same temperature. Closed
    form ``Q = (T_c,in - T_h,in) / (1/A + 1/B + 1/C)`` (series of three conductances). With a
    condensing release curve of the exhaust (``profile="condensing"``) the exhaust outlet is the
    point of the curve where it equals the medium temperature (bisection).
``counter_current``
    Two counter-current exchangers of unlimited area, the approach held along both
    (:func:`~twinlink_ml.flowsheet.exchanger.solve_stage`).
``direct``
    No circuit: the exhaust exchanges with the gas directly (the heat-network module's stage; a
    reference, not a scenario of the partner model).

The numerics are those of ``scripts/report_heat_recovery_comparison.py`` (Befund 90), lifted
unchanged; ``tests/test_flowsheet_units.py`` checks the unit against that script's artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from twinlink_ml.partner_v4 import corrected as c

from .base import Assumption, Unit
from .exchanger import approach_diagnostics, solve_stage
from .thermo import (
    PROFILE_NODES,
    GasProps,
    ProfileKind,
    capacity,
    condensate_kg_per_h,
    dew_point,
    exhaust_profile,
)

__all__ = [
    "HeatTransferLoop",
    "LoopInputs",
    "StageKind",
    "StageModel",
    "StageOut",
    "run_stage",
]

StageKind = Literal["equal_outlet", "counter_current", "direct"]


@dataclass(frozen=True)
class StageModel:
    """One way of computing a stage."""

    name: str
    props: GasProps
    kind: StageKind
    profile: ProfileKind = "linear"
    dt_min_k: float = 0.0
    split: Literal["half", "each"] = "half"
    """``half``: ``dt_min_k`` is the approach through the circuit, ``dt_min_k / 2`` per exchanger;
    ``each``: ``dt_min_k`` per exchanger."""
    nodes: int = PROFILE_NODES
    htm_max_celsius: float | None = None
    """Counter-current circuit only: highest temperature the medium may reach (an unpressurised
    liquid circuit); ``None``: no limit. The medium leaves the recovery exchanger at
    ``T2 = T_h,in + dt + Q max(1/B, 1/C)``, so the limit caps ``Q`` in closed form (binding
    ``"medium_limit"``)."""


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

    def above_boiling(self, limit_celsius: float) -> bool:
        """The medium leaves the recovery exchanger hotter than ``limit_celsius``."""
        return self.htm_hot_celsius is not None and self.htm_hot_celsius > limit_celsius


@dataclass(frozen=True)
class LoopInputs:
    """What a stage needs: the two streams, the medium and the sink's set point."""

    source: int
    sink: int
    exhaust: c.Stream
    gas: c.Stream
    htm_kg_per_s: float
    htm_cp_j_per_kg_k: float
    target_celsius: float


def _equal_outlet_q(
    exhaust: c.Stream, props: GasProps, kind: ProfileKind, nodes: int, b: float, cg: float,
    t_h: float,
) -> float:  # fmt: skip
    """Duty of the equal-outlet circuit; closed form unless the release curve condenses."""
    a = capacity(exhaust, props)
    t_c = exhaust.temperature_celsius
    closed = (t_c - t_h) / (1.0 / a + 1.0 / b + 1.0 / cg)
    if kind != "condensing" or closed <= 0.0:
        return closed
    profile = exhaust_profile(exhaust, props, kind, nodes=nodes)
    inv = 1.0 / b + 1.0 / cg

    def f(q: float) -> float:
        return profile.temperature_at(q) - t_h - q * inv

    if f(profile.total_w) >= 0.0:
        return profile.total_w
    lo, hi = 0.0, profile.total_w
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0.0 else (lo, mid)
        if hi - lo <= 1e-13 * profile.total_w:
            break
    return 0.5 * (lo + hi)


def run_stage(model: StageModel, inp: LoopInputs) -> StageOut:
    """One stage in the way ``model`` prescribes (the one place where the three kinds are told)."""
    exhaust, gas = inp.exhaust, inp.gas
    props = model.props
    cg = capacity(gas, props)
    a = capacity(exhaust, props)
    b = inp.htm_kg_per_s * inp.htm_cp_j_per_kg_k
    t_c, t_h = exhaust.temperature_celsius, gas.temperature_celsius
    dew = dew_point(exhaust.humidity_kg_per_kg)
    source, sink = inp.source, inp.sink
    if model.kind == "equal_outlet":
        q = _equal_outlet_q(exhaust, props, model.profile, model.nodes, b, cg, t_h)
        t1 = t_h + q / cg
        t2 = t1 + q / b
        if model.profile == "condensing":
            t_ex_out = t2
            modelled, unmodelled = condensate_kg_per_h(exhaust, t_ex_out), 0.0
        else:
            t_ex_out = t_c - q / a
            modelled, unmodelled = 0.0, condensate_kg_per_h(exhaust, t_ex_out)
        return StageOut(
            source, sink, q, t_h, t1, t_c, t_ex_out, t1, t2, "equal_outlet", 0.0, 0.0, dew,
            modelled, unmodelled,
        )  # fmt: skip
    dt_total = model.dt_min_k
    profile = exhaust_profile(exhaust, props, model.profile, nodes=model.nodes)
    target = inp.target_celsius
    htm_cold: float | None
    htm_hot: float | None
    b_arg: float | None
    if model.kind == "direct":
        q, binding = solve_stage(profile, cg, t_h, target, None, dt_total, 0.0)
        htm_cold = htm_hot = b_arg = None
    else:
        dt = dt_total / 2.0 if model.split == "half" else dt_total
        q, binding = solve_stage(profile, cg, t_h, target, b, dt, dt)
        if model.htm_max_celsius is not None:
            q_limit = max(model.htm_max_celsius - t_h - dt, 0.0) / max(1.0 / b, 1.0 / cg)
            if q_limit < q:
                q, binding = q_limit, "medium_limit"
        htm_cold = t_h + dt + max(0.0, q * (1.0 / cg - 1.0 / b))
        htm_hot = htm_cold + q / b
        b_arg = b
    t_ex_out = profile.temperature_at(q) if q > 0.0 else t_c
    rec, pre = approach_diagnostics(profile, q, t_h, cg, b_arg, htm_cold)
    modelled = condensate_kg_per_h(exhaust, t_ex_out) if model.profile == "condensing" else 0.0
    unmodelled = condensate_kg_per_h(exhaust, t_ex_out) if model.profile != "condensing" else 0.0
    return StageOut(
        source, sink, q, t_h, t_h + q / cg, t_c, t_ex_out, htm_cold, htm_hot, binding, rec, pre,
        dew, modelled, unmodelled,
    )  # fmt: skip


class HeatTransferLoop(Unit[LoopInputs, StageOut]):
    """The circuit of a heat transfer medium between one exhaust and one process gas."""

    def __init__(self, model: StageModel, *, assumption: Assumption = "partner") -> None:
        if model.kind == "direct":
            raise ValueError("a loop has a circuit; use HeatExchanger for direct exchange")
        self.model = model
        self.name = model.name
        self._assumption = assumption

    def assumptions(self) -> dict[str, Assumption]:
        return {"exchanger": self._assumption}

    def solve(self, inputs: LoopInputs) -> StageOut:
        return run_stage(self.model, inputs)
