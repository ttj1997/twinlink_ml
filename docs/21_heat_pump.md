# Heat pump on low-temperature waste heat: Stage A (does it pay?) and Stage B (storage, gate)

Status: complete (SCENARIO BAND, not a plant statement). Lab book: Befund 101 in
`docs/04_stand_ap1.md`. Code:
`src/twinlink_ml/heatpump/`. Scripts: `scripts/gate_heat_pump.py` (pre-registered: method, grids,
gate, sensitivity lists in the docstring, commit `37fa2e9`, before any computation on plant data),
`scripts/analyze_heat_pump_rule_tuning.py` (DECLARED POST-HOC), `scripts/analyze_heat_pump_tables.py`
and `scripts/analyze_heat_pump_stage_b_maps.py` (pure aggregations). Artifacts (all clean tree,
`git/dirty=False`): `artifacts/runs/gate_heat_pump.json` (script `dad1d80`),
`gate_heat_pump_rule_tuning_posthoc.json` (script `87bc625`), `gate_heat_pump_tables.json`,
`gate_heat_pump_stage_b_maps.json`. Tests: `tests/test_heatpump.py`, `tests/test_heatpump_gate.py`.

Confidentiality: roles only (plant A with storage A1, plant C with lines C1/C2, "the partner"); raw
exports only through `TWINLINK_PLANT_DATA_ACE` / `TWINLINK_PLANT_DATA_RID`; channel tags only in
the git-ignored role file of storage test 1.

**Solid fraction (added 2026-10-16).** Sections 1-8 were computed with the solid fraction 0.3 of
Befund 99 (the "Befund-99 assumption"). The parallel study 4 (branch `plant-a-operating-point`,
`docs/23` section 5) shows from the energy balance of the two measured days that it must be larger
(product-specific minima kept locally); its reference 0.634 (band 0.60-0.70, PLACEHOLDER) was
rerun with new artifact names (`*_sf0634`). **Section 9 gives the study-4 results side by side
with the Befund-99 assumption and the difference; the readings and conclusions do not change.**

**Every number below is a scenario result on placeholders and carries its scenario.** Annual
figures are day values x 365 of the measured day (a scaling, not an annual simulation).

## 1. Question and answer in one paragraph

Does a heat pump that lifts low-temperature waste heat (humid exhaust 30-60 degC, compressor
cooling water) to 60-90 degC and replaces gas or steam heat pay, by CO2 and by cost, today and
2030? **Stage A: yes, in every scenario of the grid, but only where it works across the pinch and
only up to the hot-utility deficit that direct recovery leaves.** On top of maximum direct heat
recovery (pinch target, the baseline), the best design saves 7-893 t CO2/a (median 265 t/a; 0.06-7.8 %
of the site's 11 500 t/a) today and 52-1334 t/a (median 356 t/a) with the 2030 electricity factor;
by cost 0.4-130 kEUR/a (median 37 kEUR/a) at a CO2 price of 55 EUR/t and 4-504 kEUR/a (median 141)
at 300 EUR/t (528 scenarios, 1000 kW, quality 0.50, central prices). The saving saturates at about
250-1000 kW heating capacity (grid median: from 500 kW on). The designs chosen in the four reference
scenarios reach COP 2.9-6.1, far above the break-even COPs
(CO2: 1.55 today, 0.85 in 2030; cost: 2.66 today, 1.94 at 150 EUR/t, 1.36 at 300 EUR/t).
**Stage B, gate reading (pre-registered): L1** -- 114 of 4480 realistic cells (2.5 %) show a gap of at
least 2 points between the programme with perfect foresight and the best of three rules without
foresight. The reading is the best case over the bands and must not be quoted alone: a DECLARED
POST-HOC check shows that a once-tuned constant value of stored heat closes most of it (18 cells
remain, at most 2.7 points, all in the assumed shift mode, where compressor heat is scarce at night);
under the cost criterion a gap from the power-price swing alone (constant operation) never
reaches 2 points.

## 2. What was reused

| module | reused for |
|---|---|
| `storage_timeslice` (Befund 99) | all streams and the site (measured days of plant A, reconstructed exhaust humidity with condensation, compressor heat, plant C, hall heating, process water, modes, weather), the vectorised problem table that keeps the cascade, the piecewise-linear "level groups", the storage convention (one level, approach `dT_min`, cyclic, no power limit) |
| `heatnetwork/pinch.py` | reference of the tests (`compute_pinch_target`, explicit isothermal streams) |
| `evaluation/net_energy.py` | CO2 factors electricity 0.366 and gas 0.201 kg/kWh |

New: `heatpump` (COP model and design grid, heat pump as two pseudo-streams, own small site pinch
calculation and placement classification, exact step dispatch, foresight LP and rules for Stage B,
signal profile, gate). The pinch module is deliberately small; the parallel site pinch study (study
1) is the place to merge it.

## 3. Method

**Heat pump in the problem table.** `COP = min(quality x T_k/(T_k - T_e), cop_max)` (kelvin; heating
COP). The evaporator is an isothermal COLD pseudo-stream at `T_e`, the condenser an isothermal HOT
pseudo-stream at `T_k`, both with approach `dT_min/2` on the process side (refrigerant contribution
about 0): the evaporator takes heat only from hot parts above `T_e + 10 K` (the exhaust is sampled
with its condensation curve down to `T_kalt`, so latent heat is available), the condenser serves
cold parts only below `T_k - 10 K`. With `Q` the condenser heat, `Q_e = Q (1 - 1/COP)`, `W = Q/COP`,
the hot utility of a step is the maximum of a few linear pieces in `Q` -- exact, no solver. **The
pinch rule is therefore not imposed but follows from the table**: across the pinch each kWh of `Q`
replaces a kWh of hot utility; entirely above it only `W` is replaced (an electric heater with
extra steps); entirely below nothing (test `test_placement_rule_across_above_below`; in the
artifact the forced full-output saving of above-pinch pairs equals `W`, of below-pinch pairs 0).
**Baseline**: the pinch target with maximum DIRECT recovery (as in Befund 99); the heat pump is
credited only with what it saves on top.

**Stage A (stationary).** Per 15-minute step and criterion the operator chooses `0 <= Q <= Q_max`
minimising `c_U U(Q) + c_el Q/COP` (`c_U` value of a kWh of gas heat, `c_el` of a kWh of
electricity); the minimum of a convex piecewise-linear function lies at an end or a kink, so it is
exact. Per scenario, size and criterion the best admissible design `(T_e, T_k)` of the grid
(`T_e` 5-70, `T_k` 60-90 degC, 5 K; lift 20-60 K; `T_k <= 90 degC`) is reported.

**Stage B.** Heat pump plus one storage (level `T_S` 40-70 degC, approach `dT_min` as in Befund 99,
capacity 0.5-5 MWh, cyclic, no power limit) under a time-varying electricity value
`c_el,t = mean x (1 + A shape(h)) x week(d)` (shape: morning peak, solar trough at midday, evening
peak, mean 0, minimum -1; `A` = depth of the trough; weekend 10 % lower). FORESIGHT: one linear
programme over the horizon (24 h or 1 week, cyclic, HiGHS); in the week horizon also a "daily
state-neutral" programme (content equal at every midnight = 24 h foresight). RULES without
foresight: per step the exact optimum of `c_U U + p_t Q - v_t (q_c - q_d)` with a value of stored heat
from present and past only -- `myopic` (only free heat is stored, the rule of Befund 99),
`gas_value` (fill whenever pump heat is cheaper than gas heat), `trailing` (24 h mean of the
signal / COP) -- run to the cyclic steady state; the rule value of a cell is the best of the
three. The step optimum is solved by vertex enumeration and equals HiGHS (test).

**Checks** (tests and artifact): pseudo-streams equal the problem table with explicit isothermal
streams (30 random cases, 0.05 kW); site pinch equals `compute_pinch_target`; step dispatch equals
a brute-force search; step optimum equals HiGHS (60 random cases); programme with `C = 0` equals the
step dispatch (all 1600 runs); programme non-increasing in `C`; daily state-neutral never below the
week programme; storage balance closed (largest residual 5.6e-10 kWh). Flags: 48 runs have a
non-converged rule (30 cycles; 31 realistic cells), none of them an L1 cell; in 2 runs a
non-converged `trailing` rule lies 0.07 points below the programme (its last cycle draws on stored
heat; the gap is understated there, never overstated in the L1 cells).

## 4. Placeholders (every value is a placeholder unless marked; band = realistic)

| item | central / grid | band, source status |
|---|---|---|
| streams and site | as Befund 99 (`docs/19`, section 4); compressor heat 85 -> 45 degC | the user's "cooling water 50-80 degC" is the variant 80 -> 50 |
| exhaust of plant C | by volume flow (2 x 20 000 Nm3/h) AND by annual figure (1.8 GWh/a) | the factor-4.9 contradiction of Befund 99, carried as two variants |
| quality (Guetegrad) | 0.50; grid 0.45 / 0.50 / 0.55 | user band 0.45-0.55; reviews of industrial heat pumps about 0.4-0.6 |
| lift, condensing limit, COP cap | 20-60 K, `T_k <= 90 degC`, `cop_max` 8 | bands 15-30 / 40-80 K, 80-100 degC, 6-10; data-sheet orders of magnitude |
| heat-pump approach | `dT_min/2` = 10 K | 5-20 K |
| heater efficiency | 0.85 (steam heater A and direct burner C, one value) | 0.75-0.95 (partner data case of Befund 60: 0.80) |
| CO2 factors | electricity 0.366 (repository), 2030 variant 0.20; gas 0.201 | 0.10-0.40 for electricity |
| prices | gas 0.040 EUR/kWh without CO2 (grid 0.030/0.040/0.060), electricity 0.16 EUR/kWh (0.12/0.16/0.20), CO2 price 55 / 150 / 300 EUR/t on gas | gas 0.025-0.060, electricity 0.10-0.25 (order of magnitude of German industrial prices 2025/26, not checked against one table) |
| capital (payback only) | 600 EUR/kW, 15 a, 6 %, 2 %/a maintenance | 400-1000 EUR/kW (technology-catalogue order of magnitude) |
| signal (Stage B) | amplitude 0-1 (grid), weekend 0.1 | price amplitude 0-0.6, CO2 amplitude 0-0.4, weekend 0-0.2 |

Value of a kWh (artifact `criteria_values`): gas heat 0.2365 kg CO2; 0.0601 / 0.0825 / 0.118 EUR at
55 / 150 / 300 EUR/t. **Break-even COP** (pump heat replacing gas heat 1:1): CO2 1.55 today, 0.85 in
2030 (every heat pump, even direct electric heating, beats gas on CO2 with 0.20 kg/kWh); cost 2.66
today, 1.94 at 150 EUR/t, 1.36 at 300 EUR/t. Equivalently, for a pump of COP 3 the electricity CO2
factor may be up to 0.71 kg/kWh (COP 2: 0.47; COP 4: 0.95).

## 5. Stage A: results (Befund-99 assumption, solid fraction 0.3; study 4 in section 9)

**Own pinch calculation (references, artifact `stage_a.references.*.pinch`).**

| reference | hot utility (mean) | cold utility (mean) | pinch (shifted) | reading |
|---|---|---|---|---|
| RA1: A+C, product 1, flat, 0 K, compressor 800, hall 200, water 200, exhaust C by volume | 637 kW | 1157 kW | 50.5-50.8 degC | large exhaust surplus below the pinch |
| RA2: RA1 with exhaust C by annual figure | 739 kW | 466 kW | 13.1-17.5 degC | little surplus; evaporator must sit at 10 degC |
| RA3: A only, product 1, flat, no hall, no water (closest to the measured day) | 235 kW | 903 kW | 75.0 degC | pinch at the top of the compressor heat (85 - 10) |
| RA4: RA3 on product 2 | 202 kW | 1120 kW | 75.0 degC | same |

Direct reading: plant A alone is pinched by the compressor heat -- the heat pump works as a
**booster that lifts compressor heat from about 65 degC to 90 degC** (design 55 -> 90, COP 5.2) for the
part of the process air above 65 degC that the cooling water cannot reach with `dT_min` 20 K. With
plant C by volume flow the exhaust surplus moves the pinch to about 50 degC; with plant C by annual
figure there is little surplus and the design falls to 10 -> 60/70 degC, COP 2.9-3.3.

**Reference table (quality 0.50, artifact `gate_heat_pump_tables`, 500 kW; 1000 kW changes only
where marked).**

| ref | criterion | design, COP | pump heat / gas heat replaced / work kWh/d | saving per year | payback at 400/600/1000 EUR/kW |
|---|---|---|---|---|---|
| RA1 | CO2 today | 35->90, 3.30 | 10 842 / 10 842 / 3284 | 497 t (4.3 % of site) | -- |
| RA1 | CO2 2030 | 30->90, 3.03 | 11 997 / 11 522 / 3964 (1000 kW: 14 548 / 12 365) | 705 t (1000 kW: 716 t) | -- |
| RA1 | cost today | 40->85, 3.98 | 7351 / 7351 / 1847 | 53 kEUR | 3.8 / 5.6 / 9.4 a (1000 kW: 7.5 / 11.3 / 18.8) |
| RA1 | cost 300 | 35->90, 3.30 | 10 842 / 10 842 / 3284 | 275 kEUR | 0.7 / 1.1 / 1.8 a |
| RA2 | CO2 today | 10->70, 2.86 | 5081 / 5081 / 1777 | 201 t (1.7 %) | -- |
| RA2 | cost today | 10->60, 3.33 | 4168 / 4168 / 1251 | 18 kEUR | 10.9 / 16.4 / 27.3 a |
| RA2 | cost 300 | 10->70, 2.86 | 5081 / 5081 / 1777 | 115 kEUR | 1.7 / 2.6 / 4.3 a |
| RA3 | CO2 today / 2030 | 55->90, 5.19 | 4325 / 4325 / 834 | 262 / 312 t (2.3 / 2.7 %) | -- |
| RA3 | cost today / 300 | 55->90, 5.19 | same | 46 / 138 kEUR | 4.3 / 6.5 / 10.8 a ; 1.5 / 2.2 / 3.6 a |
| RA4 | CO2 today | 60->90, 6.05 | 3816 / 3816 / 630 | 245 t (2.1 %) | -- |
| RA4 | cost today | 60->90, 6.05 | same | 47 kEUR | 4.3 / 6.4 / 10.7 a |

In every reference the saving no longer grows above 500 kW (RA3, RA4: 250 kW): the pump has
replaced all hot utility it can reach below `T_k - 10 K` = 80 degC; the rest (air above 80 degC)
stays with the gas/steam heater. Under CO2 2030 the pump also runs where it only replaces `W`
(direct electric heating beats gas on CO2), so pump heat exceeds gas heat replaced; under every
other criterion all pump heat is delivered in steps where the design straddles the pinch (smallest
share over the whole main grid 1.0 for CO2 today and all cost criteria; down to 0.0 only for CO2
2030, artifact `gate_heat_pump_stage_b_maps.across_share_min_per_criterion_stage_a`).

**Main grid (528 scenarios: site x day x mode x weather x compressor x hall x water x exhaust-C
basis; 1000 kW, quality 0.50; annual saving median [min - max], artifact `gate_heat_pump_tables`).**

| site, exhaust C, day | CO2 today t/a | CO2 2030 t/a | cost today kEUR/a | cost 300 kEUR/a |
|---|---|---|---|---|
| A, product 1 | 200 [118 - 295] | 295 [199 - 355] | 31 [14 - 60] | 111 [67 - 152] |
| A, product 2 | 240 [58 - 466] | 291 [89 - 681] | 37 [7 - 60] | 128 [33 - 261] |
| C, volume, product 1 | 152 [75 - 262] | 258 [116 - 405] | 14 [8 - 36] | 86 [43 - 150] |
| C, volume, product 2 | 393 [150 - 730] | 573 [225 - 1083] | 50 [16 - 87] | 216 [85 - 409] |
| C, annual, product 1 | 73 [7 - 165] | 144 [52 - 340] | 5 [0.4 - 10] | 41 [4 - 93] |
| C, annual, product 2 | 117 [41 - 301] | 224 [88 - 517] | 9 [2.5 - 24] | 66 [23 - 170] |
| A+C, volume, product 1 | 342 [231 - 528] | 493 [393 - 740] | 51 [34 - 78] | 187 [125 - 292] |
| A+C, volume, product 2 | 336 [138 - 893] | 506 [207 - 1334] | 51 [7.5 - 130] | 191 [78 - 504] |
| A+C, annual, product 1 | 251 [41 - 511] | 386 [125 - 722] | 22 [5.7 - 75] | 142 [23 - 274] |
| A+C, annual, product 2 | 326 [138 - 816] | 473 [207 - 1234] | 51 [7.5 - 99] | 177 [78 - 459] |

By mode (A+C, 1000 kW): flat 444 [41 - 893] t/a against shift 288 [114 - 587] t/a (CO2 today);
plant C alone in shift mode is the weakest case (CO2 today 103 [7 - 318] t/a, cost today 9.5
[0.4 - 40] kEUR/a). Quality (grid 0.45 / 0.50 / 0.55): median CO2 today 246 / 265 / 279 t/a, cost
today 32 / 37 / 42 kEUR/a. Price grid (1000 kW, quality 0.50, 27 combinations): every scenario
saves something except at CO2 55 EUR/t with electricity / gas 0.16 / 0.030 (99 % of scenarios),
0.20 / 0.030 (94 %) and 0.20 / 0.040 (99 %); median from 7.6 kEUR/a (electricity 0.20, gas 0.030, CO2 55) to 213 kEUR/a (0.12 /
0.060 / 300) (artifact `gate_heat_pump.stage_a.summary.cost_grid_1000kW_quality_0.5`).

**From which COP / CO2 factor does it pay?** Analytically: when the COP of a design that works across
the pinch exceeds the break-even COP of section 4. Numerically (sweeps at 500 and 1000 kW, quality
0.50): in RA1, RA3 and RA4 some design still pays at an electricity CO2 factor of 1.0 kg/kWh and an
electricity-to-gas-heat value ratio of 7.9 (the ends of the sweep axes: small-lift boosters with
COP up to the cap 8); in RA2 the saving vanishes above 0.86 kg/kWh and above a value ratio of 3.7
(today's cost ratio is 2.66). Down to a quality of 0.30 (COP of the chosen designs 2.0-3.6) every
reference still saves under CO2 today; under cost today RA2 saves nothing at a quality of 0.30-0.35.

**Sensitivity (one placeholder at a time, 500 kW; change of the annual saving).**

| rank | placeholder | RA1 CO2 today (base 497 t/a) | RA1 cost today (base 53 kEUR/a) | RA3 CO2 today (base 262 t/a) |
|---|---|---|---|---|
| 1 | exhaust of plant C: annual figure instead of volume flow | -296 | -35 | (not in site) |
| 2 | heat-pump approach 20 K (5 K) | -245 | -34 (+25) | -175 (+80) |
| 3 | compressor heat: level 400 kW; temperatures 90-50 / 75-45 / 80-50 | -201; -201 / +92 / . | . ; . / +34 / +16 | . ; -66 / +138 / +87 |
| 4 | condensing limit 80 degC | -162 | . | -162 |
| 5 | mode shift (compressor 30 % at night) | -170 | -19 | -76 |
| 6 | `dT_min` 10 K (more direct recovery leaves less for the pump) | -151 | . | -75 |
| 7 | heater efficiency 0.75 / 0.95 | +125 / -99 | +25 / . | +50 / . |
| 8 | gas price 0.060 / 0.030; electricity price 0.12 / 0.20 | -- | +86 / -27; +41 / -21 | -- |
| 9 | plant C line flow 15 000, compressor C 1.2 GWh/a | -88, -71 | . | -- |
| -- | quality 0.45 / 0.55; max lift 50 | -26 / +20; -48 | -10 / +12 | -12 / +10 |
| -- | exhaust humidity, solid fraction, `T_kalt`, `cop_max`, min lift | 0 to -25 | about 0 | about 0 |

(`.` = not among the largest; full lists in `gate_heat_pump.stage_a.sensitivity`.) The variant
"condensing limit 100 degC" has no effect because the pre-registered design grid stops at 90 degC
(a limitation of the grid, not a statement that 100 degC adds nothing).

## 6. Stage B: storage and the gate (Befund-99 assumption; study 4 in section 9)

**Gate (fixed in the script docstring before any computation).** Cell = (run, `C` in {0.5, 1, 2,
5 MWh}); rule gap `D = (J_rule - J_foresight)/J_none`, storage gain `G = (J_pump - J_foresight)/J_none`.
L1: some realistic cell with `D >= 2` points; L2: none, but `G >= 5 %` somewhere; L3: neither.

**Result: L1** (artifact `gate_heat_pump.stage_b.gate`). 1600 runs, 1120 realistic (480 have an
amplitude outside its band), none below the 100 kWh/d floor; 4480 realistic cells; **114 cells (2.5 %)
in 62 runs with `D >= 2` points**; 700 cells (15.6 %) with `G >= 5 %`. Largest gap 9.2 points (A,
product 2, week, flat, 1000 kW, CO2 2030, amplitude 0.4, 5 MWh); largest storage gain 20.7 % (A,
product 1, day, shift, 500 kW, CO2 2030, amplitude 0.4, 2 MWh). Per criterion: CO2 today 14 cells (max
2.3 points), CO2 2030 42 (9.2), cost today 4 (2.4), cost 150 34 (3.3), cost 300 20 (2.6). Per site:
A 104 cells, A+C 10 (max 2.4 points). Per mode: shift 76, flat 36, weekly 2. Value of weekly over
daily foresight (week horizon): up to 10.3 points. The two pre-registered sensitivity references
(A+C, product 1, 1000 kW, cost today, amplitude 0.4; RB1 day/shift, RB2 week/weekly) have `D = 0`
and `G` 2.2-3.4 % (RB1) and 0.8 % (RB2).

**What the L1 cells are (two mechanisms).**

1. *Source scarcity at night (mode shift, assumed compressor profile).* Gaps of up to 4.0 points exist
   with a CONSTANT signal. Trace of that cell (A, product 1, day, shift, 500 kW, CO2 2030, 2 MWh,
   level 50 degC): at night the compressor heat drops to 30 %, the evaporator runs short of free
   source heat and the marginal effect of the pump falls to 1/COP; the programme pre-charges the
   storage with pump heat in the evening while the source is plentiful, `myopic` and `trailing` never
   do, and `gas_value` keeps running the pump at night.
2. *Signal arbitrage (any mode).* In mode flat the gap is about the whole storage gain: the named
   rules do not use the storage at all, because their value of stored heat is wrong for the case
   (e.g. `trailing` divides by the COP although, under CO2 2030, the pump also works where it only
   replaces `W`).

**Declared post-hoc check (artifact `gate_heat_pump_rule_tuning_posthoc`, script committed before
its run).** With the step optimum and a CONSTANT value of stored heat `v = kappa c_U`, `kappa` chosen
per cell in hindsight from 0.05-1 (an optimistic bound for a once-tuned constant), **18 of the 114
cells keep `D >= 2` points, all in mode shift** (amplitudes 0, 0.1, 0.4, 0.6; criteria CO2 today 4,
cost 150 8, cost 300 6); the largest gap falls from 9.2 to 2.7 points, the median from 2.4 to 0.8
points. Recomputation reproduces the artifact (programme 5e-7 relative, gap 3e-8). The remainder is
what a constant cannot do: a time-of-day (schedule) dependent value of stored heat.

**From which signal swing and storage size does foresight become valuable?** (artifact
`gate_heat_pump_stage_b_maps`, max over runs; named rules.)

| mode, criterion | smallest amplitude with `D >= 2` points at 0.5 / 1 / 2 / 5 MWh | max `D` inside the band |
|---|---|---|
| flat, CO2 | 0.2 / 0.1 / 0.1 / 0.1 | 9.2 points (amplitude 0.4, 5 MWh) |
| flat, cost | never (max 2.0 points at amplitude 1.0, outside the band) | 1.2 points (amplitude 0.6, 1 MWh) |
| shift, CO2 | 1.0 / 0.4 / 0 / 0 | 4.0 points (amplitude 0, 2 MWh) |
| shift, cost | 0.8 / 0.3 / 0 / 0 | 3.3 points (amplitude 0.6, 1 MWh) |
| weekly, cost | never / 1.0 / never / 0.6 | 2.1 points (amplitude 0.6, 5 MWh) |

Direct reading: under the **cost** criterion the power-price swing alone does not make foresight worth
2 points at any storage size inside the band; under the **CO2** criterion (largest gaps with the 2030 factor) a swing of
10-20 % of the grid CO2 factor is enough with 1-2 MWh, but a tuned constant rule takes almost all of it; a
storage of 0.5 MWh rarely matters (largest gain about 6.9 %). Where the assumed shift profile makes
the source scarce at night, a schedule-aware rule (or a planner) is needed from 2 MWh on.

**Sensitivity (RB1, RB2; points at 2 MWh).** RB1: storage approach `dT_min/2` raises `G` by 5.1
points (to 8.5 %) with `D` 0.1; product day 2 opens `D` 1.6; storage loss 0.5 %/h `D` 0.4; prices,
efficiency, quality, amplitude, size change `D` by at most 0.3. RB2: site A alone opens `D` 1.3 and `G`
+3.0; product day 2 `D` 0.5; cost 300 `D` 0.9; everything else at most 0.3.

## 7. Caveats

* **Scenario band, not a plant statement.** Two measured days; the annual figures are day x 365;
  the weather shifts stand in for seasons.
* The baseline is the pinch target with IDEAL direct recovery (all compressor heat usable, `dT_min`
  20 K). Against today's plant (about 400 kW of compressor heat used) the heat pump would look
  better, but part of that would be direct recovery, not the pump.
* COP model `quality x Carnot` with refrigerant temperatures; small-lift boosters (20-25 K, COP up to
  the cap 8) are optimistic; no part load, no defrosting, no exchanger costs; isothermal condenser
  and evaporator (a real condenser for a sensible sink gains from glide or subcooling -- not
  modelled). The design grid stops at `T_k` = 90 degC.
* One heater efficiency for steam (plant A) and direct burner (plant C). Prices and the 2030 factor
  are placeholders; the CO2 cost of electricity is assumed inside the price.
* Stage B: one storage level per run, no power limits, perfect exchangers, constant weekday
  weather; the signal shape is a placeholder and the same for price and CO2. The rules are
  heuristics; the gap measures these three rules, and the post-hoc check shows how much a tuned one
  recovers. L1 says that foresight beats THESE rules by 2 points somewhere inside the bands; it does
  not say that RL is the right tool (a schedule-aware rule or an MPC with a day-ahead forecast is
  the obvious next candidate).
* The contradiction of Befund 99 (exhaust of plant C by volume flow 4.9 times the annual figure) is
  the largest single sensitivity of Stage A.

## 8. Questions for the partner (from the ranking: which number moves the result most)

1. **Exhaust of plant C: volume flow or annual heat figure?** (Stage A: -296 t/a in RA1; it decides
   where the pinch lies, 50 or 15 degC, and thereby the COP.)
2. **Achievable exchanger approaches** of a heat pump on the process air and on the cooling water
   (10 K assumed; 20 K costs -245 / -175 t/a in RA1 / RA3), and the `dT_min` of the existing recovery.
3. **Compressor heat: amount and temperatures** of the cooling water as a time series (level 400 vs
   800 kW: -201 t/a; 90-50 vs 85-45 degC: -201 t/a; 75-45: +92 / +138 t/a), and the load profile at
   night and at weekends (shift mode: -170 t/a in Stage A; it is also the driver of every L1 cell
   that survives the post-hoc check).
4. **Required supply-air temperature and admissible condensing temperature** (limit 80 instead of 90
   degC: -162 t/a); whether a high-temperature machine above 90 degC is an option.
5. **Steam generation efficiency of plant A and burner efficiency of plant C** (0.75 / 0.95: +125 /
   -99 t/a).
6. **Prices**: gas and electricity price (tariff structure: fixed or spot-indexed; the gas price is the
   largest cost sensitivity, +86 kEUR/a at 0.060 EUR/kWh) and the expected CO2 price path.
7. For Stage B: whether a dynamic tariff or a CO2-optimised operation is intended at all, and the
   space and temperature level for a storage of 1-2 MWh.
8. Not needed for this question: intake humidity, solid fraction, `T_kalt` (effects 0 to -25 t/a).

## 9. Solid fraction 0.634 (study 4) against the Befund-99 assumption 0.3

Rerun of the unchanged pre-registered grids with `--solid-fraction 0.634 --solid-fraction-band
0.60 0.70` (script `3603fb3`; default behaviour of the script unchanged); realism of the solid
fraction checked against the band 0.60-0.70, every other placeholder against its old band; in the
Stage-A sensitivity the band ends 0.60 / 0.70 and the Befund-99 value 0.3 (marked unrealistic)
replace 0.2 / 0.5. Artifacts (clean tree): `gate_heat_pump_sf0634`, `gate_heat_pump_tables_sf0634`,
`gate_heat_pump_stage_b_maps_sf0634`, `gate_heat_pump_rule_tuning_posthoc_sf0634` (declared
post-hoc, as in section 6) and the side-by-side `gate_heat_pump_solid_fraction_comparison`.

**What changes physically.** Less water evaporates, so the reconstructed exhaust of plant A is
drier: its heat down to `T_kalt` falls from 825 to 604 kW mean (product 1) and from 917 to 642 kW
(product 2), and product 2 is no longer capped at saturation (share of capped steps 1.0 -> 0.0).
The surplus below the pinch shrinks (cold utility RA1 27 779 -> 16 052 kWh/d), the pinch
temperatures stay (RA1 50.7 degC, RA2 15.2 -> 14.9 degC, RA3/RA4 75.0 degC).

**Stage A (quality 0.50; annual saving; study 4 | Befund 99 | difference).**

| item | CO2 today t/a | CO2 2030 t/a | cost today kEUR/a | cost 300 kEUR/a |
|---|---|---|---|---|
| main grid median (1000 kW) | 244 / 265 / -20 | 328 / 356 / -28 | 34.3 / 37.1 / -2.8 | 130 / 141 / -11 |
| main grid range (1000 kW) | 2.3-813 / 7.0-893 | 55-1248 / 52-1334 | 0.5-89 / 0.4-130 | 1.2-465 / 3.9-504 |
| RA1 (500 kW, design 35->90 resp. 40->85 unchanged) | 483 / 497 / -14 | 686 / 705 / -19 | 52.4 / 53.3 / -0.9 | 267 / 275 / -8 |
| RA2 (500 kW, design 10->70 -> 5->65, COP 2.86 -> 2.82) | 154 / 201 / -48 | 240 / 311 / -71 | 9.6 / 18.3 / -8.7 | 88 / 115 / -27 |
| RA3, RA4 (plant A alone) | unchanged (262, 245) | unchanged | unchanged (46, 47) | unchanged |

Every main-grid scenario still saves under every central criterion; in the price grid three
combinations at 55 EUR/t leave 2-5 % of the scenarios without saving (electricity / gas 0.16 / 0.030
and 0.20 / 0.040: 98 %, 0.20 / 0.030: 95 %). Payback at 600 EUR/kW (500 kW, cost today): RA1 5.7 a
(5.6 a), RA2 31 a (16 a), RA3 and RA4 unchanged. Plant A alone is unaffected because its pinch is
set by the compressor heat, not by the exhaust; plant C by annual figure, where the surplus below
the pinch is already scarce, loses most. Around the new base the solid fraction is a small
sensitivity (RA1, CO2 today: 0.60 +1.5, 0.70 -2.9 t/a; RA3: 0); the sensitivity ranking is unchanged
in its order (exhaust-C reading -329, heat-pump approach 20 K -230, compressor temperatures 90-50
-201, compressor level 400 kW -191, mode shift -157 t/a in RA1).

**Stage B (study 4 | Befund 99).** Reading **L1 | L1**. Cells with `D >= 2` points 121 (58 runs) |
114 (62 runs); cells with `G >= 5 %` 821 | 700; largest `D` 9.2 | 9.2 points (same cell: A, product 2,
week, flat, 1000 kW, CO2 2030, amplitude 0.4); largest `G` 20.9 | 20.7 %. Per criterion: cost today 8
cells (max 2.9 points) | 4 (2.4); CO2 today 24 (2.4) | 14 (2.3); CO2 2030 43 | 42; cost 150 12 (2.5) |
34 (3.3); cost 300 34 (3.1) | 20 (2.6); mode shift 84 (max 4.2) | 76 (4.0), flat 36 | 36, weekly 1 | 2;
site A+C 19 (2.9) | 10 (2.4). Declared post-hoc tuned constant: **18 | 18 cells remain, all in mode
shift**, largest 2.7 | 2.7 points, median 0.70 | 0.79. Signal swing alone (mode flat, cost): largest
`D` inside the band 1.45 | 1.17 points; 2 points first at amplitude 0.8 (outside the band) | never.
Flags: 137 | 48 runs with a non-converged rule, none in an L1 cell; 9 | 2 runs with a rule below the
programme (at most 0.08 points, non-converged rules).

**Reading.** The study-4 value lowers the Stage-A savings by about 8 % in the median (more where
plant C is read by its annual figure, nothing for plant A alone) and leaves the Stage-B reading and
its interpretation unchanged (L1, mostly closed by a tuned constant; what survives rests on the
assumed night profile of the compressors). The partner question on the solid fraction (section 8,
item 8) is answered by study 4 for this purpose.

## 10. Reproducibility

`scripts/gate_heat_pump.py --name gate_heat_pump` (pre-registration commit `37fa2e9`, script
`dad1d80`, clean tree, about 12 minutes on 14 processes);
`scripts/analyze_heat_pump_rule_tuning.py --name gate_heat_pump_rule_tuning_posthoc` (script
`87bc625`); `scripts/analyze_heat_pump_tables.py --name gate_heat_pump_tables` and
`scripts/analyze_heat_pump_stage_b_maps.py --name gate_heat_pump_stage_b_maps` (aggregations, no
plant data). Study-4 variant: `scripts/gate_heat_pump.py --name gate_heat_pump_sf0634
--solid-fraction 0.634 --solid-fraction-band 0.60 0.70` (script `3603fb3`, about 43 minutes on a
shared, fully loaded machine), the follow-up scripts with `--source gate_heat_pump_sf0634` and
`scripts/analyze_heat_pump_solid_fraction_comparison.py --name
gate_heat_pump_solid_fraction_comparison`. The first two need `TWINLINK_PLANT_DATA_ACE` / `TWINLINK_PLANT_DATA_RID` and the
git-ignored role file of storage test 1.
