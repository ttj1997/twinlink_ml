"""The equations of the partner's equation document for the exchangers and the heater, in SI.

Units: kg/s, kg/kg dry air, W, W/K, Pa; temperatures in degC and named so. The numbers of the
equations are those of ``Flow diagram and balance equations v4.docx``; two typos of the
document are corrected here (and listed in :mod:`.deviations`): in (2.1) the right-hand side
carries ``T_w,out`` (the document has ``T_w,in`` twice), in (3.1) the vapour term on the left
carries ``T_g,in`` (the document has ``T_g,out``). With these corrections (2.3), (2.4), (3.2)
follow from (2.1), (3.1) -- the tests close each balance independently.

**The assumption of the document, stated as such.** Every exchanger is taken as unlimited:
both fluids leave at the *same* temperature (2.3/2.4, 4.1/4.2: "the exit temperature of the
heat transfer medium is equal to the exit temperature of the exhaust gas"). For one exchanger
this is the limit of **parallel flow (co-current) with unlimited area**: both streams
approach their mixing temperature. It is not the best an exchanger can do -- a counter-current
exchanger with unlimited area lets the stream with the smaller capacity rate leave at the
other's inlet temperature -- so the assumption *underestimates* the recoverable heat.
:func:`counter_current_infinite_area_loop` gives that bound for comparison.

Notation for a pre-heating stage: exhaust (dry-air flow ``Mg,c``, humidity ``Y_out`` of the
dryer, inlet ``T_g,c,in``), heat transfer medium ``Mw`` (``cp_w``), process gas (``Mg,h``,
``Y_in``, inlet ``T_g,h,in``); ``T1`` leaves the pre-heating exchanger (gas and medium),
``T2`` leaves the recovery exchanger (exhaust and medium). With the capacity rates
``A = Mg,c (cp_g + Y_out cp_v)``, ``B = Mw cp_w``, ``C = Mg,h (cp_g + Y_in cp_v)``:

    (4.1)  A (T_c,in - T2) = B (T2 - T1)        (4.2)  C (T1 - T_h,in) = B (T2 - T1)

and the closed solution ``Q = (T_c,in - T_h,in) / (1/A + 1/B + 1/C)`` (a series of three
conductances; negative when the gas enters the stage warmer than the exhaust).
"""

from __future__ import annotations

from dataclasses import dataclass

from twinlink_ml.partner_v4 import corrected as c
from twinlink_ml.partner_v4.inputs import SiMaterial

__all__ = [
    "ExchangerResult",
    "LoopResult",
    "counter_current_infinite_area_loop",
    "equal_outlet_exchanger",
    "equal_outlet_loop",
    "equation_2_3_celsius",
    "equation_2_4_celsius",
    "exhaust_dew_point_celsius",
    "heater_enthalpy_residual_w",
    "heater_gas_kg_per_s",
    "loop_capacity_rates",
]


@dataclass(frozen=True)
class ExchangerResult:
    """One exchanger with equal outlet temperatures (equations 2.1-2.5)."""

    outlet_celsius: float
    """``T_g,out = T_c,out = T_w,out``."""
    recovered_w: float
    """(2.5): ``Q = M_w cp_w (T_w,out - T_w,in)``."""
    condensate_kg_per_s: float
    """(2.2): ``M_c = M_g (Y_in - Y_out)``."""
    outlet_humidity_kg_per_kg: float


@dataclass(frozen=True)
class LoopResult:
    """One pre-heating stage: recovery exchanger -> circuit -> pre-heating exchanger."""

    sink_out_celsius: float
    """``T1``: process gas and (cold) heat transfer medium leaving the pre-heating exchanger."""
    exhaust_out_celsius: float
    """``T2``: exhaust and (hot) heat transfer medium leaving the recovery exchanger."""
    recovered_w: float
    """``Q = M_w cp_w (T2 - T1)``; negative if the circuit would run backwards."""


def equation_2_4_celsius(
    exhaust: c.Stream, htm_capacity_w_per_k: float, t_w_in_celsius: float, m: SiMaterial
) -> float:
    """(2.4), no condensation (``Y_out = Y_in``).

    ``T = (Mg (cp_g + Y_in cp_v) T_in + Mw cp_w T_w,in) / (Mg (cp_g + Y_out cp_v) + Mw cp_w)``.
    """
    a = c.heat_capacity_rate_w_per_k(exhaust, m)
    return (a * exhaust.temperature_celsius + htm_capacity_w_per_k * t_w_in_celsius) / (
        a + htm_capacity_w_per_k
    )


def equation_2_3_celsius(
    exhaust: c.Stream,
    outlet_humidity_kg_per_kg: float,
    htm_capacity_w_per_k: float,
    t_w_in_celsius: float,
    m: SiMaterial,
) -> float:
    """(2.3), with condensation, for a given outlet humidity ``Y_out`` of the exhaust.

    ``T = (Mg [(cp_g + Y_in cp_v) T_in + (Y_in - Y_out) dh] + Mw cp_w T_w,in)
    / (Mg [cp_g + Y_out cp_v + (Y_in - Y_out) cp_l] + Mw cp_w)``.
    """
    y_in, y_out = exhaust.humidity_kg_per_kg, outlet_humidity_kg_per_kg
    mg = exhaust.dry_air_kg_per_s
    numerator = (
        mg
        * (
            (m.cp_gas_j_per_kg_k + y_in * m.cp_vapour_j_per_kg_k) * exhaust.temperature_celsius
            + (y_in - y_out) * m.enthalpy_of_evaporation_j_per_kg
        )
        + htm_capacity_w_per_k * t_w_in_celsius
    )
    denominator = (
        mg
        * (
            m.cp_gas_j_per_kg_k
            + y_out * m.cp_vapour_j_per_kg_k
            + (y_in - y_out) * m.cp_liquid_j_per_kg_k
        )
        + htm_capacity_w_per_k
    )
    return numerator / denominator


def equal_outlet_exchanger(
    exhaust: c.Stream,
    htm_flow_kg_per_s: float,
    cp_htm_j_per_kg_k: float,
    t_w_in_celsius: float,
    pressure_pa: float,
    m: SiMaterial,
) -> ExchangerResult:
    """One exchanger exhaust -> heat transfer medium, equal outlet temperatures (2.1-2.5).

    Uses (2.4) if the exhaust stays above its dew point; otherwise the outlet humidity is the
    saturation humidity at the outlet temperature (the document leaves ``Y_out`` implicit) and
    the enthalpy balance (2.1, condensate included) is solved by bisection; (2.3) then holds
    for the resulting ``Y_out`` (the tests check it).
    """
    b = htm_flow_kg_per_s * cp_htm_j_per_kg_k
    t_dry = equation_2_4_celsius(exhaust, b, t_w_in_celsius, m)
    y_in = exhaust.humidity_kg_per_kg
    if y_in <= 0.0 or t_dry >= c.dew_point_celsius(y_in, pressure_pa):
        t_out, y_out = t_dry, y_in
    else:
        h_in = c.stream_enthalpy_j_per_kg_dry(y_in, exhaust.temperature_celsius, pressure_pa, m)

        def residual(t: float) -> float:
            h_out = c.stream_enthalpy_j_per_kg_dry(y_in, t, pressure_pa, m)
            return exhaust.dry_air_kg_per_s * (h_in - h_out) - b * (t - t_w_in_celsius)

        lo, hi = min(t_w_in_celsius, t_dry), exhaust.temperature_celsius
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if residual(mid) > 0.0 else (lo, mid)
        t_out = 0.5 * (lo + hi)
        y_out = min(y_in, c.saturation_humidity_kg_per_kg(t_out, pressure_pa))
    return ExchangerResult(
        outlet_celsius=t_out,
        recovered_w=b * (t_out - t_w_in_celsius),
        condensate_kg_per_s=exhaust.dry_air_kg_per_s * (y_in - y_out),
        outlet_humidity_kg_per_kg=y_out,
    )


def loop_capacity_rates(
    exhaust: c.Stream,
    sink: c.Stream,
    htm_flow_kg_per_s: float,
    cp_htm_j_per_kg_k: float,
    m: SiMaterial,
) -> tuple[float, float, float]:
    """``(A, B, C)`` in W/K: exhaust, heat transfer medium, process gas (see module docstring)."""
    return (
        c.heat_capacity_rate_w_per_k(exhaust, m),
        htm_flow_kg_per_s * cp_htm_j_per_kg_k,
        c.heat_capacity_rate_w_per_k(sink, m),
    )


def equal_outlet_loop(
    exhaust: c.Stream,
    sink: c.Stream,
    htm_flow_kg_per_s: float,
    cp_htm_j_per_kg_k: float,
    m: SiMaterial,
) -> LoopResult:
    """Equations (4.1) and (4.2) solved for ``T1`` and ``T2`` (no condensation, as the document).

    The 2x2 linear system ``(A+B) T2 - B T1 = A T_c,in`` and ``-B T2 + (B+C) T1 = C T_h,in`` is
    solved in closed form; ``Q = B (T2 - T1) = (T_c,in - T_h,in) / (1/A + 1/B + 1/C)``.
    """
    a, b, cc = loop_capacity_rates(exhaust, sink, htm_flow_kg_per_s, cp_htm_j_per_kg_k, m)
    t_c, t_h = exhaust.temperature_celsius, sink.temperature_celsius
    det = a * cc + a * b + b * cc
    t2 = (a * t_c * (cc + b) + b * cc * t_h) / det
    t1 = ((a + b) * cc * t_h + b * a * t_c) / det
    return LoopResult(sink_out_celsius=t1, exhaust_out_celsius=t2, recovered_w=b * (t2 - t1))


def counter_current_infinite_area_loop(
    exhaust: c.Stream,
    sink: c.Stream,
    htm_flow_kg_per_s: float,
    cp_htm_j_per_kg_k: float,
    m: SiMaterial,
) -> LoopResult:
    """Same stage with two counter-current exchangers of unlimited area (a bound, not the model).

    One counter-current exchanger of unlimited area transfers ``min(C_hot, C_cold) (T_hot,in -
    T_cold,in)``. The circuit closes when the medium leaves the pre-heating exchanger at the
    temperature it enters the recovery exchanger with (bisection; the map is monotone). If the
    gas enters warmer than the exhaust nothing is transferred (a real circuit would be bypassed).
    """
    a, b, cc = loop_capacity_rates(exhaust, sink, htm_flow_kg_per_s, cp_htm_j_per_kg_k, m)
    t_c, t_h = exhaust.temperature_celsius, sink.temperature_celsius
    if t_c <= t_h:
        return LoopResult(t_h, t_c, 0.0)

    def exchanger(c_hot: float, t_hot: float, c_cold: float, t_cold: float) -> tuple[float, float]:
        q = min(c_hot, c_cold) * (t_hot - t_cold)
        return t_hot - q / c_hot, t_cold + q / c_cold

    def closure(t1: float) -> tuple[float, float]:
        _, t2 = exchanger(a, t_c, b, t1)
        t1_back, _ = exchanger(b, t2, cc, t_h)
        return t1_back, t2

    lo, hi = t_h, t_c
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if closure(mid)[0] - mid > 0.0 else (lo, mid)
    t1 = 0.5 * (lo + hi)
    t2 = closure(t1)[1]
    exhaust_out, _ = exchanger(a, t_c, b, t1)
    _, gas_out = exchanger(b, t2, cc, t_h)
    return LoopResult(
        sink_out_celsius=gas_out, exhaust_out_celsius=exhaust_out, recovered_w=cc * (gas_out - t_h)
    )


def exhaust_dew_point_celsius(exhaust: c.Stream, pressure_pa: float) -> float:
    """Dew point of the exhaust (Antoine); below it (4.1) would have to include condensation."""
    return c.dew_point_celsius(exhaust.humidity_kg_per_kg, pressure_pa)


def heater_gas_kg_per_s(stream: c.Stream, t_out_celsius: float, m: SiMaterial) -> float:
    """(3.2): ``M_f = M_g (cp_g + Y cp_v) (T_out - T_in) / (LHV eta_th)`` in kg/s."""
    power = c.heater_power_w(stream, stream.temperature_celsius, t_out_celsius, m)
    return c.gas_mass_flow_kg_per_s(power, m)


def heater_enthalpy_residual_w(stream: c.Stream, t_out_celsius: float, m: SiMaterial) -> float:
    """Residual of (3.1) with the typo corrected, written out independently of (3.2) [W].

    ``Mg [cp_g T_in + Y (dh + cp_v T_in)] + Mf LHV eta - Mg [cp_g T_out + Y (dh + cp_v T_out)]``.
    """
    mg, y = stream.dry_air_kg_per_s, stream.humidity_kg_per_kg
    t_in = stream.temperature_celsius
    mf = heater_gas_kg_per_s(stream, t_out_celsius, m)
    enthalpy_in = mg * (
        m.cp_gas_j_per_kg_k * t_in
        + y * (m.enthalpy_of_evaporation_j_per_kg + m.cp_vapour_j_per_kg_k * t_in)
    )
    enthalpy_out = mg * (
        m.cp_gas_j_per_kg_k * t_out_celsius
        + y * (m.enthalpy_of_evaporation_j_per_kg + m.cp_vapour_j_per_kg_k * t_out_celsius)
    )
    return enthalpy_in + mf * m.lower_heating_value_j_per_kg * m.heater_efficiency - enthalpy_out
