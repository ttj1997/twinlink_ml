# Storage test 1: time-slice pinch -- upper bound for the benefit of a heat storage

Status: complete (SCENARIO BAND, not a plant statement). Lab book: Befund 99 in
`docs/04_stand_ap1.md` (prepared as "Befund 98"; 98 went to the Holm addendum to Befund 94; scripts and artifacts keep the working label). Code:
`src/twinlink_ml/storage_timeslice/`. Scripts: `scripts/gate_storage_timeslice.py` (pre-registered
gate, grids, decomposition, sensitivity), `scripts/analyze_storage_timeslice_sensitivity.py`
(declared post-hoc). Artifacts: `artifacts/runs/gate_storage_timeslice.json` (script commit
`db8a646`, clean tree, `git/dirty=False`) and
`artifacts/runs/gate_storage_timeslice_posthoc_sensitivity.json` (clean tree). Tests:
`tests/test_storage_timeslice.py`, `tests/test_storage_timeslice_gate.py`.

Confidentiality: roles only (plant A with storage A1, plant C with lines C1/C2, "the partner");
channel tags only in the git-ignored `conf/plant/plant_local.yaml` (template with placeholder tags
in `conf/plant/plant_local.yaml.example`, key `timeslice_roles`); raw exports only through
`TWINLINK_PLANT_DATA_PROD1` / `TWINLINK_PLANT_DATA_PROD2`.

## 1. Question and answer in one paragraph

How much fossil residual heat can an IDEAL storage save at most compared with using heat only when
it occurs, and does a rule without foresight reach it? **Gate reading (pre-registered): L2** -- some
scenarios inside the placeholder bands save more than 5 % and no cell has a rule gap of 2
percentage points; a rule suffices and **no RL/MPC set-up is justified by this test**. The reading
is the best case over the bands and must not be quoted alone: in the main grid 339 of 2240
scenarios (15.1 %) have a cell above 5 %, **all of them on the product-2 day**; on the product-1
day the maximum of the whole grid is 2.6 %; with constant operation (mode flat) the maximum is
3.0 % (L3 for that mode), and under the conditions closest to the measured days (plant A, flat, no
hall heating, no process water) it is 0.0 %. The saving exists only where an ASSUMED compressor
load profile (shift mode, placeholder) makes compressor heat exceed the temperature-compatible
demand of the process air during the day.

## 2. T1.0 What was reused

| module | reused for |
|---|---|
| `heatnetwork/pinch.py` | stream segments (`StreamSegment`, `hot_segments_for_source` with the existing condensation sampling, `cold_segment_for_sink`); the problem-table result is the reference of the tests (`compute_pinch_target`) |
| `heatnetwork/balance.py` | indirectly (the condensation curve is sampled from `evaluate_cascade`); the cascade/exchanger checks are not needed for a target-level bound |
| `simulation/thermo.py`, `GasConfig` | humidity, saturation, specific heats (no second set of constants) |
| `storage_a1/` (Befunde 91/92) | conventions (normal density 1.293 kg/Nm3, intake humidity 0.75; no fan credit, see section 4); the A1 model itself is not used -- this test bounds ANY storage by a mixed ideal one |
| `flowsheet/` | not needed (no unit models; streams are given by the measured days and placeholders) |
| `plant_data/loader.py` | reading the exports; roles through the git-ignored role file |

New: `storage_timeslice` (vectorised problem table that keeps the cascade, ideal-storage LP, greedy
rule, streams, gate logic).

## 3. T1.1 Method

**Time grid** 15 min (block means of the one-minute data), horizons 1 day and 1 week (the measured
day repeated; weather constant over the week).

**E_direct.** At each step the problem-table algorithm (`dT_min`, sources cooled down to `T_kalt`
with condensation) gives the minimum hot utility `U_t` of all streams that exist at that step;
`E_direct = sum_t U_t dt`, the fossil residual heat (useful heat of the heater; fuel = `U` /
efficiency, which changes no ratio). Heat is usable only when it occurs.

**E_ideal(C).** Ideal storage of capacity `C` kWh at one temperature level `T_S`: charging is an
isothermal cold stream and discharging an isothermal hot stream in the same problem table, i.e.
charging takes heat only from source parts above `T_S + a` and discharging serves sink parts only
below `T_S - a`, `a` = `dT_min` (as specified; the variant `a = dT_min/2` treats the storage as
part of the existing water circuit). Because the stream temperatures are fixed, `U_t` is the
maximum of `2K + 1` linear pieces in the charge and discharge duties (`K` levels) -- exact, no
solver inside the problem table -- and `E_ideal(C)` is a **linear programme** over the horizon
(HiGHS): minimise `sum U_t dt` with `s_{t+1} = (1 - lambda dt) s_t + (q_c - q_d) dt`,
`0 <= s <= C`, no power limits, cyclic (end = start). `E_ideal(C)` is the best level of the grid
30 ... 80 degC (5 K). One detail that mattered and is covered by a test: heat of the discharging
stream enters only BELOW its temperature, so the cascade value arriving at that boundary is
unchanged (strict inequality there); an earlier version with an inclusive boundary overstated the
saving and was caught by the comparison with the problem table with explicit isothermal streams.
Upper-bound check: the same programme with all levels at once sharing the capacity (multi-level,
`ideal` in the sense of a perfectly stratified storage), solved on the four decomposition
scenarios.

**E_rule(C).** No foresight, same storage and levels (best level reported): per step first
discharge exactly as much as lowers the hot utility, then charge exactly the surplus that does not
raise it ("charge surplus, discharge on demand", greedy), run cyclically to its steady state. This
is the natural rule; no weak rule was chosen.

**Checks** (tests, and recorded per scenario in the artifact): `C = 0` equals `E_direct` (relative
1e-12, float summation order); the storage energy balance closes (largest residual over all runs
1.1e-10 kWh; with losses the difference charged - discharged equals the loss); `E_ideal` is
non-increasing in `C`; `E_rule >= E_ideal`; `E_ideal <= E_direct`; the LP's `U` equals the problem
table with the isothermal streams added explicitly (tolerance 0.05 kW, narrow-band stand-in); the
vectorised table equals `compute_pinch_target`; exhaust scaling equals direct sampling; the air
sink equals `cold_segment_for_sink`; a hand-solved toy case (100 kW for 12 h, then 100 kW demand)
gives `1200 - min(C, 1200)` for programme and rule. All pass in all 2428 runs except one flag:
`rule_converged` is false in 139 results (day horizon, `C >= 2` MWh): the content at the start of
the last cycle differs from its end after 60 cycles (very slow filling); in those cells the gap to
the programme is below 1.1e-15 anyway.

## 4. T1.2 Scenarios (every value is a placeholder unless marked; band = realistic)

| item | value / grid | status and band |
|---|---|---|
| compressor heat plant A | level 400 / 800 / 1200 kW as a stream 85 -> 45 degC | partner: about 1200 kW in total, about 400 kW used today (order of magnitude); temperatures placeholder (top 70-90, bottom 40-55) |
| compressor heat plant C | 2.4 GWh/a / 8760 h = 274 kW mean, same temperatures | partner annual figure, placeholder (band 1.2-3.6 GWh/a) |
| exhaust plant A | measured outlet temperature of both days; humidity RECONSTRUCTED from spray rate, solid fraction 0.3, intake humidity 0.75, dry air = process + atomising air; capped at saturation | solid fraction band 0.2-0.5, humidity 0.5-0.9; product 2 is capped at saturation in every step (outlet 34.4 degC), product 1 never (60.7 degC) |
| exhaust plant C (C1, C2) | 20 000 Nm3/h per line, state of the measured exhaust of plant A (same process assumed) | band 15 000-25 000; variant: scaled to 1.8 GWh/a |
| process air A | measured days (flow, intake temperature, supply temperature) | measurement; mean 957 / 798 kW without fan credit -- minus 0.9 x measured fan power this is 852 / 697 kW, the heating demand of Befund 91 (consistency check) |
| process air C | 20 000 Nm3/h per line, supply-temperature profile and outdoor profile of the measured day | placeholder |
| hall heating | 0 / 200 kW peak, 35 -> 50 degC, heating degree (15 degC limit, -10 degC design), factor 1 from 18:00 to 06:00 and 0.4 otherwise | partner "up to about 200 kW"; profile placeholder |
| process water | 0 / 200 kW, 15 -> 60 degC, constant or tied to the shift | placeholder |
| `dT_min`, `T_kalt` | 20 K, 15 degC | repository defaults (bands 10-30 K, 10-25 degC) |
| weather | both days, ambient -10 / -5 / 0 / +5 K | exhaust state not shifted |
| capacity | 0, 0.5, 1, 2, 5, 10, 20 MWh | storage A1 holds about 0.7 MWh; 20 MWh outside the realistic range (reported only) |
| loss | 0 and 0.5 %/h of the content (band 0-1 %/h) | placeholder |
| modes | flat; shift (compressors 100 % 06-22 h, 30 % otherwise; C1 06-22, C2 06-14; process water 06-22); weekly (week horizon: compressors 100 % Mon-Fri, 30 % weekend; C1/C2 and process water Mon-Fri) | placeholders; plant A runs 24/7 in every mode (partner) |

Main grid: site {A, C, A+C} x product day x 4 weather shifts x 5 (horizon, mode) x compressor
level (A and A+C) x hall x water x loss = **2240 scenarios**, plus 68 reference and sensitivity
runs (2308 runs; labels of variants are not unique, so "distinct scenarios" are counted on the
grid only). All are inside the bands.

## 5. T1.3 Gate (fixed in the script docstring before any computation)

Cell = (scenario, `0 < C <= 10` MWh), `G = (E_direct - E_ideal) / E_direct`,
`D = (E_rule - E_ideal) / E_direct`. L1: some cell with `G >= 5 %` and `D >= 2` points; L2:
`G >= 5 %` but no cell with both; L3: no cell with `G >= 5 %`. Scenarios with `E_direct < 100
kWh/d` excluded (none occurred; none was unrealistic).

**Result: L2.** 11 540 cells; 1420 with `G >= 5 %`; 0 with both; largest rule gap `D` 3.9e-15 (i.e.
zero). Per group (artifact `gate.per_group`): site A, A+C, C all L2; horizon day and week L2;
mode flat L3 (max 3.0 %), shift L2, weekly L2; product day 1 L2 only through one sensitivity
variant (see below), product day 2 L2; conditions closest to the measured days L3 (60 cells, max
0.0 %).

## 6. T1.4 Results

**What carries it (main grid, 2240 scenarios, C <= 10 MWh).** 339 scenarios (15.1 %) have a cell
with `G >= 5 %`, **all product 2** (339 of 1120; product 1: 0 of 1120, maximum 2.6 %); by site
A 174, A+C 121, C 44; by mode shift 256 (day and week alike), weekly 83, flat 0; with compressor
level 1200 kW 185, 800 kW 110, 400 kW 0; both hall and process-water values, all weather shifts
and both loss values occur. Day and week horizons give identical savings in mode shift (the
storage is used as a daily buffer; nothing is shifted across days). Losses reduce the saving (day horizon, 5 MWh) by
2 kWh/d on average (largest 88 kWh/d): the surplus heat is free, so the programme charges
extra to cover the loss.

**Table of the main scenarios (saving of the ideal storage; kWh/d and % of `E_direct`; every row
is a scenario with placeholders as listed in section 4).**

| scenario | `E_direct` kWh/d | 0.5 MWh | 1 MWh | 2 MWh | 5-20 MWh | C at 90 % of max | rule gap |
|---|---|---|---|---|---|---|---|
| A, product 2, day, shift, -10 K, compressor 1200, hall 200, water 0 (largest fraction) | 7211 | 500 (6.9 %) | 1000 (13.9 %) | 1459 (20.2 %) | 1459 (20.2 %) | 2 MWh | 0 |
| A+C, product 2, week, shift, -10 K, compressor 800, hall 0, water 0 (largest kWh/d) | 10 825 | 500 (4.6 %) | 1000 (9.2 %) | 1696 (15.7 %) | 1696 (15.7 %) | 2 MWh | 0 |
| A+C, product 1, day, shift, 0 K, compressor 800, hall 200, water 200 (R1) | 10 563 | 0 | 0 | 0 | 0 | -- | 0 |
| A+C, product 1, week, weekly, 0 K, same (R2) | 12 644 | 0 | 0 | 0 | 0 | -- | 0 |
| A, product 1 or 2, day, flat, 0 K, no hall, no water | 5643 / 4838 | 0 | 0 | 0 | 0 | -- | 0 |

The saving saturates at about 1.5-2 MWh (one charge-discharge cycle per day), about twice to three
times storage A1 (about 0.7 MWh) and far below 10 MWh. The best level in the two largest scenarios is 45-60 degC.

**Decomposition (artifact `decomposition`, C = 5 MWh; the split is order-dependent and says so;
leave-one-out and single pairs are two readings, not parts of one sum).**

* Single source x sink pair: the saving sits in **compressor heat -> process air** (plant A:
  1416 kWh/d in R1's site and 2146 kWh/d in the largest-fraction scenario; plant C: compressor ->
  process air of C1/C2, 585-754 kWh/d); exhaust-based pairs are at most 28 kWh/d (in the four decomposition scenarios).
* In the full network of R1 and R2 the saving is 0 although single pairs have potential: the
  process air of plant A (957 kW) absorbs all compressor heat directly, **network integration
  replaces the storage**. Leaving the exhaust of plant A out (so that compressor heat is the main
  source) opens 277 kWh/d in R1; leaving out the process air of A removes every saving in the two largest scenarios.
* Largest-fraction scenario: leaving out hall heating lowers the saving to 1197 kWh/d, process
  water changes nothing (it is 0 there); largest-kWh scenario: leaving out C1/C2 air sinks lowers
  it to 1138 kWh/d.
* Multi-level (perfectly stratified) ideal storage at 5 MWh: no gain in R1, R2 and the
  largest-fraction scenario; 8968 against 9130 kWh/d in the largest-kWh scenario (162 kWh/d, 1.5 %
  of `E_direct`) -- the single-level bound is close to the stratified one here, and the rule would
  still be within 2 points of it in that scenario.

**Sensitivity (artifacts; change of the saving at C = 5 MWh, kWh/d and points of `E_direct`).**
The two pre-registered references (A+C, product 1: R1 day/shift, R2 week/weekly) have a saving of
0, so a variant can only open a potential: opened by storage approach `dT_min/2` (+781 kWh/d,
7.4 points; **the only product-1 cell above 5 %**), exhaust of C scaled to the annual figure (+398,
+3.6), compressor temperatures 90-50 (+385, +4.7), `dT_min` 10 K (+249, +4.9) in R1; exhaust C by
annual figure (+512) and `dT_min` 10 K (+118) in R2; everything else 0. Because of that, two
further references with a non-zero saving were fixed AFTER seeing it (declared post-hoc, script
committed before its run): R3 = A+C, product 2, week, shift, 0 K, compressor 800, hall 200, water
200 (saving 29 kWh/d, at the edge); R4 = A, product 2, day, shift, 0 K, compressor 1200, hall 200,
water 0 (saving 1245 kWh/d, 18.8 points).

| rank | placeholder | R4 effect (kWh/d) | R3 effect (kWh/d) |
|---|---|---|---|
| 1 | compressor level (400 / 1200) and process-air demand (product day) | -1245 / 0 ; product 1: -1245 | -29 (400) / +840 (1200); product 1: -29 |
| 2 | compressor load contrast (`low_fraction` 0.5 / 0.1) and shift window (07-19 / 05-23) | -1212 / +345; +414 / -322 | -26 / +180; +298 / -29 |
| 3 | `dT_min` (10 / 30 K) | +264 / -954 | +247 / +361 |
| 4 | compressor temperatures (75-45 / 90-50) | -442 / +360 | +1133 / -29 |
| 5 | process-water demand (0 / 100 kW), plant-C compressor (1.2 / 3.6 GWh/a), plant-C line flow | 0 (not in site) | +1508 / +708; -29 / +1337; +644 (15 000) / -29 |
| 6 | storage approach `dT_min/2` | +357 | 0 |
| 7 | weather (+5 / -10 K) | -355 / +214 | +117 / -29 |
| 8 | hall demand and temperatures | -49 ... +98 | 0 ... +193 |
| -- | `T_kalt`, solid fraction, intake humidity, loss rate | 0 to +-10 | 0 to +-13 |

Direct reading: the potential is created by the **time profile and temperature level of the
compressor heat against the process-air demand**, not by the exhaust humidity, `T_kalt` or the
loss rate. The sign of several effects flips with the base (a placeholder that adds a source or
removes a sink opens or closes it): the ranking is local to the reference and is the list of what
to ask, not a statement of size.

## 7. Caveats

* **Scenario band, not a plant statement.** The reading is the best case over the bands; the main
  driver, the compressor load profile (shift / weekly), is itself a placeholder; with constant
  operation (the only profile for which plant data exist in the sense of Befunde 91/92) there is
  nothing to shift.
* Product 2 carries every above-5 % cell of the main grid: its lower process-air demand (798
  against 957 kW) and warmer intake leave compressor heat unused during the day. Two days only.
* The exhaust humidity is reconstructed (solid fraction placeholder; product 2 capped at
  saturation in every step, product 1 never) and plays no role in the ranking.
* The exhaust of plant C by volume flow (about 999 kW mean down to `T_kalt` for both lines) is
  4.9 times the partner's annual figure (1.8 GWh/a = 205 kW mean) -- the figures are not
  consistent as read here; the annual variant is in the sensitivity.
* Ideal storage: no power limits, no stratification loss, one level per run (multi-level checked
  on four scenarios), perfect exchangers; the upper bound is not a design. Nothing in this test
  says that a storage of any size is or is not worth its cost.
* A greedy rule equals the programme by structure (one reservoir, surplus and deficit as a queue,
  no foresight needed): a rule gap would need power limits, several competing storages, forecast
  of sources, or a stratification trade-off; none is in this model. That is why L1 cannot occur
  here; it does not say that an RL/MPC agent has nothing to do at a heat network with other
  limits (not tested).
* 139 rule results are flagged not converged (day horizon, `C >= 2` MWh); their gap is below
  1.1e-15 anyway. Labels of sensitivity variants coincide with grid labels (placeholders are not in
  the label); cell counts are counted over runs.

## 8. Questions for the partner (from the ranking: which time series are really needed)

1. **Compressor heat as a time series** (or the compressor load profile and the water temperatures
   at which it is available), separately for plant A and, planned, plant C; how much of it is
   unused today and when (decides everything: with a constant profile the saving is 0).
2. **Process-air demand of plant A over the day, per product**, and the real supply temperature
   (product 2 uses less heat than product 1 and that alone opens the potential).
3. **Temperature levels**: the highest and lowest temperature of the compressor cooling water
   (75-45 / 85-45 / 90-50 matters) and the admissible approach per exchanger (`dT_min`; is the
   storage inside the existing circuit).
4. **Operating schedule of plant C** (lines C1/C2 per shift and weekday, how long a line stands) and
   its compressor heat; **the volume flow and the annual waste-heat figures read together** (20 000
   Nm3/h per line does not match 1.8 GWh/a).
5. Process-water consumers (demand, temperature, time of day) and hall heating (demand at night,
   temperatures) -- second-order in this test.
6. Not needed for this question: intake humidity, solid fraction of the sprayed suspension, `T_kalt`.

## 9. Reproducibility

`scripts/gate_storage_timeslice.py --name gate_storage_timeslice` (script committed before the run,
clean tree; artifact with config fingerprints per scenario, hashes of the raw files, provenance);
`scripts/analyze_storage_timeslice_sensitivity.py --name gate_storage_timeslice_posthoc_sensitivity`.
Both need `TWINLINK_PLANT_DATA_PROD1` / `TWINLINK_PLANT_DATA_PROD2` and the git-ignored role file;
runtime about 8 minutes and 3 minutes on 14 processes.
