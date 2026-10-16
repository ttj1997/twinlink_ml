r"""Heat-pump study: Stage A (heat pump on waste heat), Stage B (plus storage) and the gate.

Question: does a heat pump on low-temperature waste heat pay, by CO2 and by cost, today and 2030
(Stage A), and does heat pump plus storage under a time-varying power signal need foresight
(Stage B: the gate for any RL or MPC set-up)?

THIS DOCSTRING IS THE PRE-REGISTRATION. It was committed before any number of this study was
computed on the plant data (the package ``twinlink_ml.heatpump`` and its tests had been written and
run on synthetic streams only); nothing in it may be changed after seeing results (corrections
only as dated addenda in the lab book). Every number below that is not a measurement or a partner
statement is a PLACEHOLDER (central value, band, source status in the docstrings of
``twinlink_ml.heatpump.config`` and ``twinlink_ml.storage_timeslice.config``); every result
carries its scenario and says so. Nothing here is a plant statement. No reinforcement learning.

Streams and site (unchanged from storage test 1, Befund 99; ``twinlink_ml.storage_timeslice``)
    Time grid 15 min, the two measured days of plant A (product 1 / product 2, block means),
    horizons 1 day and 1 week (the day repeated). Sources: compressor heat of plant A (level
    400 / 800 / 1200 kW, 85 -> 45 degC; the user's statement "cooling water 50-80 degC" is the
    sensitivity variant 80 -> 50), compressor heat of plant C (2.4 GWh/a), exhaust of plant A
    (measured outlet temperature, humidity reconstructed, condensation down to ``T_kalt`` =
    15 degC), exhaust of plant C (two lines of 20 000 Nm3/h, state of the exhaust of plant A).
    Sinks: process air of plant A (measured), process air of plant C (lines C1/C2), hall heating
    0 / 200 kW, process water 0 / 200 kW. Operating modes flat / shift / weekly and weather
    shifts as in Befund 99. ``dT_min`` 20 K. The CONTRADICTION of Befund 99 (exhaust of plant C by
    volume flow = 4.9 x the partner's annual figure) is carried as two variants: ``volume`` and
    ``annual`` (scaled to 1.8 GWh/a) for every scenario that contains plant C.

Heat-pump model (``twinlink_ml.heatpump``)
    COP = min(quality x COP_Carnot(T_e, T_k), cop_max), COP_Carnot = T_k / (T_k - T_e) in K;
    quality 0.50 (grid 0.45 / 0.50 / 0.55, band = user 0.45-0.55); admissible designs: T_e in
    5 ... 70 degC, T_k in 60 ... 90 degC (5 K grid), 20 K <= lift <= 60 K, T_k <= 90 degC, cop_max
    8. Evaporator and condenser are ISOTHERMAL PSEUDO-STREAMS in the same problem table as all
    process streams (approach dT_min/2 on the process side, refrigerant contribution 0): the
    evaporator is a cold stream at T_e (takes heat from hot parts above T_e + 10 K; condensation
    of the exhaust is allowed because the exhaust stream is sampled with its condensation curve),
    the condenser a hot stream at T_k (serves cold parts below T_k - 10 K). The hot utility per
    step is then exact and piecewise linear in the condenser heat Q (Q_e = Q (1 - 1/COP),
    W = Q / COP), so the PINCH RULE is not imposed but follows from the table: across the pinch
    dU = Q, entirely above dU = W, entirely below dU = 0.
    BASELINE: the pinch target of the site with maximum DIRECT heat recovery (as in Befund 99);
    the heat pump is credited only with what it saves on top of direct recovery.
    Gas heat: heater efficiency 0.85 (band 0.75-0.95; steam heater of plant A and direct burner
    of plant C, one value). Criteria (value of a kWh of gas heat c_U and of a kWh of electricity
    c_el): CO2 with gas 0.201 kg/kWh and electricity 0.366 (repository factors, "today") or 0.20
    (variant "2030", placeholder); cost with gas price 0.030 / 0.040 / 0.060 EUR/kWh (without
    CO2), electricity 0.12 / 0.16 / 0.20 EUR/kWh, CO2 price 55 / 150 / 300 EUR/t on gas
    (placeholders; central: 0.040 / 0.16; "today" = 55, "2030" = 150 and 300). Capital (only for
    the payback column): 600 EUR/kW (band 400-1000), 15 a, 6 %, 2 %/a maintenance.

Stage A (stationary: the heat pump has no storage; it runs at the best output per step)
    Per step and criterion the operator chooses 0 <= Q <= Q_max minimising c_U U(Q) + c_el Q/COP
    (exact: ends and kinks of a convex piecewise-linear function). Per scenario, size and
    criterion the best admissible design pair (T_e, T_k) is reported with its heat, the gas heat
    replaced, the work, the saving of the criterion, the share of its heat delivered in steps in
    which the pair straddles the pinch, and annual figures (day x 365, a scaling of the measured
    day, not an annual simulation). Sizes Q_max: 0, 100, 250, 500, 750, 1000, 1250, 1500 kW.
    Main grid A: site {A, C, A+C} x day {1, 2} x mode {flat, shift} (day horizon) x weather
    {-10, 0, +5 K} x compressor level of plant A {400, 800, 1200 kW; not for site C} x hall
    {0, 200 kW} x process water {0, 200 kW} x exhaust-C basis {volume, annual; not for site A}
    = 528 scenarios, each for the three qualities and the 29 criteria (2 CO2 + 27 cost).
    Reference scenarios: RA1 = A+C, product 1, flat, 0 K, compressor 800, hall 200, water 200,
    exhaust C by volume; RA2 = RA1 with exhaust C by annual figure; RA3 = A, product 1, flat,
    0 K, compressor 800, hall 0, water 0 (closest to the measured days); RA4 = RA3 on product 2.
    For them: own pinch calculation (utilities, pinch temperatures, mean grand composite curve on
    -10 ... 100 degC shifted), the pinch-rule check (every admissible pair at every size: placement
    per step and the utility saving at forced full output against Q, W and 0), break-even sweeps
    at 500 and 1000 kW, quality 0.50: electricity CO2 factor 0 ... 1 kg/kWh (step 0.02) and
    electricity value / gas-heat value 0.5 ... 8 (step 0.1); quality sweep 0.30 ... 0.70 (step
    0.05) under CO2 today and cost today ("from which COP does it pay").
    Sensitivity (one placeholder at a time around RA1 and RA3, size 500 kW, criteria CO2 today,
    CO2 2030, cost today, cost 300): quality 0.45 / 0.55; max lift 50 / 70 K; min lift 15 / 30 K;
    max condensing 80 / 100 degC; cop_max 6 / 10; heat-pump approach 5 / 20 K; dT_min 10 / 30 K;
    T_kalt 10 / 25 degC; compressor level 400 / 1200 kW; compressor temperatures 80-50 / 75-45 /
    90-50; solid fraction 0.2 / 0.5; intake humidity 0.5 / 0.9; exhaust C by annual figure; line
    flow 15 000 / 25 000 Nm3/h; compressor C 1.2 / 3.6 GWh/a; hall 0 / 100 kW; water 0 / 100 kW;
    weather -10 / +5 K; product day 2; mode shift; heater efficiency 0.75 / 0.95; electricity
    price 0.12 / 0.20; gas price 0.030 / 0.060 (cost criteria only). Ranked by the absolute change
    of the annual saving (t CO2/a, kEUR/a); the ranking becomes the list of questions to the
    partner. No gate in Stage A: it answers the question with grids.

Stage B (heat pump plus storage plus time-varying signal; the GATE)
    Storage as in storage test 1 (one level T_S in {40, 50, 60, 70 degC}, approach dT_min, i.e.
    charging from hot parts above T_S + 20 K -- among them the condenser -- and discharging to
    cold parts below T_S - 20 K; capacity C in {0, 0.5, 1, 2, 5 MWh}; loss 0; no power limit;
    cyclic). Electricity value per step c_el,t = mean x (1 + A shape(h)) x week(d), shape = daily
    pattern of mean 0 and minimum -1 (morning peak, solar trough at midday, evening peak;
    placeholder), A = depth of the trough in fractions of the mean, weekend 10 % below weekdays
    (week horizon; band 0-20 %). Realistic amplitude bands: price 0-0.6, CO2 factor 0-0.4.
    Objective J = sum dt (c_U U_t + c_el,t Q_t / COP); J_none = without pump and storage;
    J_pump = pump alone (C = 0, best output per step).
    Methods: FORESIGHT = linear programme with perfect foresight over the whole horizon (24 h for
    the day horizon, 1 week for the week horizon), cyclic (HiGHS); in the week horizon also the
    24 h foresight variant "daily state-neutral" (content equal at every midnight). RULES without
    foresight (same storage, same levels, cyclic steady state, at most 30 cycles): per step the
    exact optimum of c_U U + p_t Q - v_t (q_c - q_d) with a value v_t of stored heat from the
    present and past only: ``myopic`` (v = 0+, stores only free heat, the rule of Befund 99),
    ``gas_value`` (v = c_U-, fills whenever pump heat is cheaper than gas heat), ``trailing``
    (v_t = min(c_U, mean c_el of the past 24 h / COP)). The RULE value of a cell is the best of
    the three (the most favourable no-foresight reading, conservative against L1). Best level per
    method and capacity.
    Design pair: the Stage-A best pair for (scenario, size, criterion) at quality 0.50 under the
    constant (mean) signal; if no pair saves anything there, the Stage-A best pair at the trough
    value mean x (1 - A); if none either, the run is recorded with the pump idle.
    Main grid B: site {A, A+C} x day {1, 2} x (horizon, mode) {(day, flat), (day, shift),
    (week, flat), (week, shift), (week, weekly)} x size {500, 1000 kW} x criterion {CO2 today,
    CO2 2030, cost today (0.16 / 0.040 / 55), cost 150, cost 300} x amplitude A {0, 0.1, 0.2,
    0.3, 0.4, 0.6, 0.8, 1.0}; weather 0 K, compressor 800 kW, hall 200 kW, water 200 kW, exhaust C
    by volume = 1600 runs.
    GATE (pre-registered). A CELL is (run, capacity C in {0.5, 1, 2, 5 MWh}); the rule gap
    D = (J_rule - J_foresight) / J_none and the storage gain G = (J_pump - J_foresight) / J_none.
    Only REALISTIC runs count: all stream placeholders inside the bands of Befund 99, all
    heat-pump, price and factor placeholders inside the bands of ``heatpump.config`` and the
    amplitude inside its band (price <= 0.6, CO2 <= 0.4); runs with E_direct < 100 kWh/d are
    excluded (their number is reported).
    L1  RL / MPC set-up justified: some cell has D >= 2 percentage points (CO2 or cost).
    L2  storage with the pump pays, a rule suffices: no cell with D >= 2 points, but some cell
        has G >= 5 %.
    L3  no relevant potential: no cell with D >= 2 points and no cell with G >= 5 %.
    The letter is a statement about the BEST case over the bands and must not be quoted alone.
    Always reported with it: numbers and shares of cells with D >= 2 points and with G >= 5 %;
    readings per criterion, per horizon, per site and per product day; the map max D over
    (amplitude, capacity) for every amplitude of the grid (the ones outside the band marked) and
    the smallest amplitude at which D reaches 2 points per capacity ("from which signal swing
    and storage size foresight becomes valuable"); the value of weekly over daily foresight
    (daily state-neutral minus week programme); the share of non-converged rule runs.
    Sensitivity (one at a time around RB1 = A+C, product 1, day, shift, 1000 kW, cost today,
    A = 0.4 and RB2 = A+C, product 1, week, weekly, 1000 kW, cost today, A = 0.4): loss 0.5 %/h;
    storage approach dT_min/2; quality 0.45 / 0.55; size 500 / 1500 kW; electricity price 0.12 /
    0.20; gas price 0.030 / 0.060; heater efficiency 0.75 / 0.95; product day 2; amplitude 0.2 /
    0.6; weekend reduction 0 / 0.2 (RB2 only); criterion cost 300; site A; ranked by the change of
    D and G at C = 2 MWh (points).

Checks (tests and recorded): the pseudo-streams equal the problem table with explicit isothermal
streams; the pinch rule dU = Q / W / 0 on a hand case; the step dispatch equals a brute-force
search; the exact step optimum of the rules equals HiGHS; the programme with C = 0 equals the step
dispatch; J_foresight non-increasing in C; rules never below the programme; daily state-neutral
never below the week programme; storage balance closed.

Call::

    TWINLINK_PLANT_DATA_ACE=... TWINLINK_PLANT_DATA_RID=... \
        tools/run_here.sh scripts/gate_heat_pump.py --name gate_heat_pump
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("pre-registration only; the study body follows in the next commit")
