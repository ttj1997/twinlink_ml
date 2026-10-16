r"""Storage test 1: time-slice pinch -- upper bound for the benefit of a heat storage.

Question: how much fossil residual heat can an IDEAL heat storage save at most, compared with
using heat only at the moment it occurs, and does a simple rule without foresight reach it?
This is the gate for any RL or MPC set-up on the heat network. No reinforcement learning here.

THIS DOCSTRING IS THE PRE-REGISTRATION. It was committed before any number of this study was
computed; nothing in it may be changed after seeing results (corrections only as dated addenda in
the lab book). Every number below that is not a measurement is a PLACEHOLDER (source and band are
stated) and every result carries its scenario and says so. Nothing here is a plant statement.

Method (code: ``twinlink_ml.storage_timeslice``; reuses ``heatnetwork.pinch`` for the stream
segments and the condensation curve of the exhaust)
    Time grid 15 min (block means of the one-minute plant data); horizons 1 day and 1 week.
    Per step t the problem-table algorithm (``heatnetwork.pinch``, minimum approach ``dT_min``)
    gives the minimum hot utility U_t of all hot streams (sources, cooled down to ``T_kalt``,
    condensation allowed) and cold streams (sinks) that exist at that step:
    E_direct = sum_t U_t dt, "fossil residual heat" without storage (heat usable only when it
    occurs). U is the useful heat of the heater, not fuel; fuel = U / efficiency, which does not
    change any ratio.
    Ideal storage of capacity C [kWh] at ONE temperature level T_S: charging is an additional
    isothermal cold stream at T_S, discharging an additional isothermal hot stream at T_S in the
    same problem table, i.e. charging takes heat only from source parts above T_S + a and
    discharging serves sink parts only below T_S - a, ``a`` = approach on the storage side
    (default ``a = dT_min``, as specified; the variant ``a = dT_min/2`` is the sensitivity that
    treats the storage as part of the existing water circuit). Because the stream temperatures
    are fixed, U_t is piecewise linear and convex in (q_charge, q_discharge) with only three
    active pieces per step (see the module); E_ideal(C) is therefore a linear programme over the
    horizon: minimise sum_t U_t dt with s_{t+1} = (1 - lambda dt) s_t + (q_c - q_d) dt,
    0 <= s_t <= C, no power limits, CYCLIC (s at the end = s at the start, so the storage
    cannot be emptied for free). E_ideal(C) = the best T_S of the grid (30 .. 80 degC, 5 K).
    As a check of the upper-bound character the same programme is solved with ALL levels of the
    grid at once sharing the capacity (multi-level ideal storage) on the reference scenarios.
    Rule E_rule(C): no foresight, same storage, same T_S grid (best T_S reported), per step:
    first discharge as much as reduces the hot utility (not more), then charge with exactly the
    surplus that does not raise it ("charge surplus, discharge on demand", greedy), run cyclically
    to its periodic steady state. This is the natural rule; no deliberately weak rule is chosen.
    Checks (tests and recorded in the artifact): C = 0 gives E_direct exactly; storage energy
    balance closed (without losses sum q_c = sum q_d; with losses the difference equals the
    loss); E_ideal non-increasing in C; E_rule >= E_ideal; U of the programme equals the
    problem table with the isothermal streams added explicitly.
    Losses: fraction of the stored energy per hour, 0 and the placeholder 0.5 %/h.

Scenarios (all placeholders as grids; band = what counts as REALISTIC)
    Sources (hot): compressor heat of plant A: level 400 / 800 / 1200 kW (partner: about 1200 kW
    in total, about 400 kW used today; band 400-1200), as a stream from 85 to 45 degC (water
    loop; placeholder, band top 70-90, bottom 40-55). Compressor heat of plant C: mean 2.4 GWh/a /
    8760 h = 274 kW (partner annual figure, placeholder), same temperatures. Exhaust of plant A:
    the two measured days (product 1 / 2): measured outlet temperature, humidity RECONSTRUCTED
    (not measured) from the spray rate, solid fraction 0.3 (placeholder, band 0.2-0.5) and the
    intake humidity (relative humidity 0.75, placeholder, band 0.5-0.9), capped at saturation
    at the outlet temperature; condensation down to ``T_kalt`` = 15 degC (repository default,
    band 10-25). Exhaust of plant C, lines C1 and C2: about 20 000 Nm3/h per line (partner,
    band 15 000-25 000), state of the measured exhaust of plant A (placeholder: same process);
    variant: scaled so that the flat-profile heat down to ``T_kalt`` equals the partner annual
    figure 1.8 GWh/a (reported as a check, not forced).
    Sinks (cold): process air of plant A (measured days: flow, intake temperature, supply
    temperature as logged; no credit for fan heat); process air of plant C, lines C1/C2 (20 000
    Nm3/h each, supply temperature profile of the measured day, outdoor-air profile of the
    measured day); hall heating up to 200 kW (band 0-200) as a stream from 35 to 50 degC with a
    demand proportional to the heating degree (15 degC heating limit, -10 degC design point)
    times a night factor (1.0 from 18:00 to 06:00, 0.4 otherwise; placeholder); process water
    0 / 200 kW (band 0-200) from 15 to 60 degC (placeholder), constant or tied to the shift.
    ``dT_min`` 20 K (repository default, band 10-30).
    Operating modes (profiles; plant A runs 24/7 in every mode, as the partner states):
    ``flat``    compressors constant, C1 and C2 24/7, process water constant;
    ``shift``   compressors 100 % from 06:00 to 22:00 and 30 % otherwise (band 10-50 %), C1 from
                06:00 to 22:00, C2 from 06:00 to 14:00, process water in 06:00-22:00 only;
    ``weekly``  (week horizon only) compressors 100 % Monday-Friday and 30 % Saturday-Sunday,
                C1 and C2 Monday-Friday 24 h and off at the weekend, process water Monday-Friday.
    Day horizon: modes flat and shift; week horizon: flat, shift, weekly (the measured day is
    repeated seven times; weather constant over the week).
    Weather: both measured days, ambient temperature shifted by -10 / -5 / 0 / +5 K (the
    exhaust state is NOT shifted). Capacity C: 0, 0.5, 1, 2, 5, 10, 20 MWh (for comparison:
    storage A1 holds about 0.7 MWh).
    Main grid: site {A, C, A+C} x product day {1, 2} x weather {4 values} x mode x compressor
    level {400, 800, 1200 kW; plant C alone: n/a} x hall {0, 200 kW} x process water {0,
    200 kW} x loss {0, 0.5 %/h} x horizon {day, week}. Plus the sensitivity variants below (one
    placeholder at a time around the two reference scenarios); all of them are inside the
    bands and enter the gate. C = 20 MWh is outside the realistic range and is reported only.

Gate (pre-registered). A CELL is (scenario, capacity C <= 10 MWh); the saving is
    G = (E_direct - E_ideal) / E_direct and the rule gap D = (E_rule - E_ideal) / E_direct
    (both on the same best-T_S convention, per day: week values divided by 7 for kWh/d; the
    fractions are unaffected). Scenarios with E_direct < 100 kWh/d are excluded from the
    gate (the percentages have no meaning there; their number is reported).
    L1  RL / MPC set-up justified:  some cell has G >= 5 % AND D >= 2 percentage points.
    L2  storage pays, a rule suffices:  some cell has G >= 5 %, but no cell has both.
    L3  no relevant potential:  no cell has G >= 5 %.
    The letter reading is a statement about the BEST case over the placeholder bands and must
    not be quoted alone. Always reported with it: the number and share of cells with G >= 5 %
    and with both conditions, the readings per site subset, per horizon and per product day, and
    the reading restricted to the conditions closest to the measured days (site A, mode flat,
    no hall heating, no process water).

Evaluation (all labelled with the scenario)
    Per scenario: saving over C in kWh/d and in % of E_direct, C at 90 % of the maximum saving
    (over the C grid up to 20 MWh), gap rule - ideal.
    Decomposition on three scenarios fixed by rule (the reference scenario; the realistic cell
    with the largest saving fraction; the one with the largest saving in kWh/d): leave one source
    or one sink out at a time (saving recomputed), and the matrix of single source x single
    sink pairs (all other streams removed). The split is order-dependent and says so.
    Sensitivity: reference scenarios (site A+C, product 1, 0 K, compressor 800 kW, hall 200 kW,
    process water 200 kW, loss 0): (R1) day horizon, mode shift; (R2) week horizon, mode weekly.
    One placeholder at a time over its band (``dT_min`` 10/30, ``T_kalt`` 10/25, compressor
    level 400/1200, compressor temperatures 75-45 and 90-50, low fraction 0.1/0.5, hall 0/100 kW,
    hall temperatures, process water 0/100 kW, line flow 15 000/25 000, exhaust C by annual figure,
    plant C compressor 1.2/3.6 GWh/a, solid fraction 0.2/0.5, humidity 0.5/0.9, loss 0.5/1 %/h,
    storage approach dT_min/2, shift window 07-19 / 05-23, weather -10/+5 K, product day 2),
    ranked by the change of the saving at C = 5 MWh in kWh/d and in percentage points of
    E_direct. The ranking becomes the list of questions to the partner (which time series are
    really needed).

Call::

    TWINLINK_PLANT_DATA_ACE=... TWINLINK_PLANT_DATA_RID=... TWINLINK_LOCAL_DIR=... \
        tools/run_here.sh scripts/gate_storage_timeslice.py --name gate_storage_timeslice
"""

from __future__ import annotations

import sys


def main() -> int:
    raise SystemExit("not implemented yet: pre-registration commit only")


if __name__ == "__main__":
    sys.exit(main())
