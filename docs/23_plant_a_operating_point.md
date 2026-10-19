# Operating point of plant A: CO2 lever of air flow and supply-air temperature (steam and fans)

Status: complete (UPPER BOUND under placeholders and proxy constraints, not an operating
recommendation). Lab book: Befund 103 in `docs/04_stand_ap1.md`. Code: `src/twinlink_ml/plant_a_op/`. Script:
`scripts/study_plant_a_operating_point.py` (its docstring is the pre-registration, commit
`2b41d9d`, committed before any optimisation). Artifact:
`artifacts/runs/study_plant_a_operating_point.json` (run on `2b41d9d`, clean tree,
`git/dirty=False`, committed as `7bc187a`). Tests: `tests/test_plant_a_op.py`.

Confidentiality: roles only (plant A with storage A1, "the partner", product 1 / product 2);
channel tags only in the git-ignored `conf/plant/plant_local.yaml` (template with placeholder tags
in `conf/plant/plant_local.yaml.example`, key `operating_point_roles`); raw exports only through
`TWINLINK_PLANT_DATA_ACE` / `TWINLINK_PLANT_DATA_RID`.

**The proxy constraints do not replace a product-moisture measurement. Every saving below is an
upper bound of the lever, not an operating recommendation. Every number that is not a measurement
carries placeholders (section 3); no number is quoted without its labels.**

## 1. Question and answer in one paragraph

What would another combination of process-air flow and supply-air temperature at plant A save in
CO2 against today's operation, counting the steam (gas boiler) and the electricity of both fans?
At the reference case (section 4) the lever is **758 kg CO2/d on product 1 (14.6 % of today's
5 206 kg/d of steam plus fans) and 90 kg CO2/d on product 2 (2.8 % of 3 196 kg/d)**, reached by
less air (the minimum air fraction 0.8 binds) and a hotter supply (90.4 / 94.7 degC) with the
exhaust held at its observed minimum. Over the placeholder grid at the measured weather the range
is 73-2 649 kg/d (product 1) and 0-2 109 kg/d (product 2). The result is decided by three
unknowns: the **minimum air flow** the fluidisation allows, the **fan law** (how much the fan power
falls with the flow) and **how the waste-heat side reacts** (does the storage loop deliver its heat
at a lower air flow). On product 2 a lower air flow takes less waste heat from the preheater (at
the measured water inlet temperature) and the steam saving is eaten up by that; only the fans
save. The dryer-balance placeholders (solid fraction, feed, room and atomising-air temperatures,
heat capacity of the solids) move the reference result by at most 11 kg/d.

## 2. Model

Stationary, at plant scale, on the day means of each measured day (within a day the air flow
moves by less than 0.3 % and the supply temperature by less than 0.15 K standard deviation: the
plant runs one operating point per product). Production (the spray rate) is held at the measured
day mean in every operating point.

* **Dryer balance** (`plant_a_op.dryer`): process air at `T_in`, atomising air at `T_atom`,
  water of the suspension `spray (1 - x_s)` evaporates completely (product moisture neglected),
  solids leave at the outlet temperature, wall loss `UA (T_out - T_room)`. The balance is linear in
  `T_out`, so the outlet temperature of every operating point is explicit. Exhaust humidity from
  the water balance (`Y_out = Y_in + m_w / (m_p + m_a)`), relative humidity at `T_out` with the
  Buck saturation pressure of the grey-box model, not clipped (over-saturation is rejected).
* **Steam via the air balance** (as Befund 91): `Q_steam = C_air (T_in - T_amb) - f_fan
  P_process - Q_waste`; only the process-fan heat enters the supply air. `Q_waste` is the preheater
  duty (water side of the preheater loop, measured), capped at what the air still needs.
* **Waste heat at another air flow** (`plant_a_op.waste_heat`), two variants: `fixed_water_inlet`
  (water inlet and loop flow as measured, duty from effectiveness-NTU with the conductance fitted
  on the day; less air means less duty) and `duty_held` (the measured duty is kept, the storage
  loop would run warmer, capped where the water inlet would exceed 80 degC; in the sensitivity runs
  70 / 80 / 90 degC give the same saving, so the cap does not bind there).
* **Fans** (`plant_a_op.fans`): `P = P_meas (V / V_meas)^n` per fan, anchored at the day mean.
  `data_fit`: n = 0.51, the two-point exponent of the process fan between the two days, applied to
  both fans; `affinity`: n = 3 (placeholder). The exhaust fan's own two-point exponent is 8.8,
  outside any physical range (its power halves between the days for an 8 % lower air flow; the
  difference is confounded with the product, e.g. exhaust temperature 61 / 34 degC and the filter
  state), so it is not used. Within a day the data cannot identify any exponent: log-log slopes
  -2.5 +- 2.9 and 1.2 +- 7.1 (process fan, product 1 / 2), 2.0 +- 1.9 and 1.5 +- 4.2 (exhaust
  fan), with a flow coefficient of variation of 0.05 % / 0.10 % on the 15-minute blocks.
  **The data contain almost no variation of the air flow; the fan law is a placeholder in every
  variant.**
* **CO2**: gas `= Q_steam / eta`, CO2 `= 0.201 gas + f_el (P_process + P_exhaust)` with
  `f_el` = 0.366 kg/kWh (project factor of `carrier_energy`) or 0.20 kg/kWh (2030 placeholder).
* **Constraints** (product moisture is not measured; proxies, each variant separately):
  (a) outlet temperature not below the observed minimum of the product's day (15-minute blocks:
  60.52 / 34.18 degC); (b) relative exhaust humidity not above the observed maximum (15-minute
  blocks, RECONSTRUCTED from the spray rate, so it moves with the solid fraction: 0.095 / 0.763 at
  `x_s` = 0.65); (a and b) both; plus a minimum air fraction 0.6 / 0.7 / 0.8 / 0.9 of today's
  (placeholder for the fluidisation limit). Thresholds never exclude today's operation (they are
  moved to the model baseline where it lies beyond them).
* **Search**: air factor 0.6-1.1, supply 70-100 degC, grid 0.005 x 0.1 K plus a 10x refinement;
  the measured point is always a candidate.

Consistency with Befund 91 (reference placeholders): heating demand with fan credit 852 / 697 kW,
waste-heat share 32.5 % / 55.6 %, demand without fan credit 957 / 798 kW (Befund 99) -- reproduced.

## 3. Placeholders (value, band, source)

| placeholder | reference | band / grid | source, status |
|---|---|---|---|
| solid fraction `x_s` of the suspension | 0.65 | 0.60 / 0.65 / 0.70 / 0.75 | not measured; lower end from the energy balance (see 5); a suspension density of 1339 kg/m3 (screenshot, third date, product unknown) means 0.42-0.76 for a solid density of 2500-1500 kg/m3; same value for both products (assumption) |
| intake relative humidity | 0.75 | 0.5-0.9 | not measured (as Befund 91) |
| feed temperature | 30 degC | 20-50 | not measured |
| atomising-air temperature | 25 degC | 15-40 | not measured |
| room temperature (wall loss) | 20 degC | 10-30 | not measured |
| cp of the solids | 1200 J/(kg K) | 800-1600 | default of `MaterialConfig`, no source |
| fan heat fraction | 0.90 | 0.8-1.0 | as `storage_a1` |
| boiler efficiency | 0.88 | 0.85 / 0.88 / 0.92 | band of the task, no plant value |
| electricity factor | 0.366 kg/kWh | 0.366 / 0.20 | project factor / 2030 placeholder without source |
| gas factor | 0.201 kg/kWh | -- | project factor |
| fan law | `data_fit` (n = 0.51) | `data_fit` / `affinity` (n = 3); n = 0 and 2 as sensitivity | section 2 |
| waste-heat variant | `fixed_water_inlet` | `fixed_water_inlet` / `duty_held` | the storage dynamics are not modelled |
| maximum preheater water (`duty_held`) | 80 degC | 70-90 | compressor heat at about 85 degC (Befund 99) minus an approach |
| minimum air fraction | 0.8 | 0.6 / 0.7 / 0.8 / 0.9 | placeholder for the fluidisation limit |
| calibration day of `UA` | own day | own / other day | the transfer check, section 5 |
| weather | measured | intake temperature -10 / -5 / 0 / +5 K | as Befund 99 |

## 4. Result

Reference case: measured weather, `x_s` 0.65, `UA` on the own day, `fixed_water_inlet`,
`data_fit`, eta 0.88, electricity 0.366, constraint (a), minimum air 0.8.

**Today's operation (model baseline, reference placeholders).** Product 1: 5 206 kg CO2/d, of
which steam 3 151 and fans 2 055 (fan share 39.5 %; 26.3 % with the 2030 electricity factor).
Product 2: 3 196 kg/d, steam 1 699, fans 1 498 (46.9 %; 32.5 % with 0.20 kg/kWh). The two fans
(234 / 171 kW electric) are a large part of the CO2 of this air path.

**a) Reference optimum.**

| | product 1 | product 2 |
|---|---|---|
| air factor / supply temperature | 0.800 / 90.4 degC (today 1 / 84.6) | 0.804 / 94.7 degC (today 1 / 83.1) |
| outlet temperature / reconstructed RH | 60.53 degC / 0.104 (today 60.67 / 0.090) | 34.18 degC / 0.838 (today 34.39 / 0.709) |
| binding | outlet minimum, minimum air | outlet minimum |
| CO2 today -> optimum | 5 206 -> 4 448 kg/d (216.9 -> 185.3 kg/h) | 3 196 -> 3 106 kg/d (133.2 -> 129.4 kg/h) |
| **saving** | **758 kg/d (14.6 %)** | **90 kg/d (2.8 %)** |
| of which steam / fan electricity | 535 / 223 kg/d (steam share 0.70) | -69 / 160 kg/d |
| steam part: air heating / lost fan heat / lost waste heat | +743 / -62 / -146 | +289 / -59 / -300 |
| fans: process / exhaust | 111 / 112 | 105 / 55 |

**b) Grid at the measured weather: saving in kg CO2/d (reference value; range over solid fraction,
calibration day, waste-heat variant, boiler efficiency and electricity factor in brackets, 84
admissible cells each, 12 inadmissible).**

| product, constraint | min air 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| 1, (a), data_fit | 1 449 (582-1 846) | 1 119 (582-1 395) | 758 (426-930) | 389 (218-469) |
| 1, (a), affinity | 2 263 (850-2 649) | 1 842 (850-2 109) | 1 320 (644-1 483) | 711 (343-787) |
| 1, (b), data_fit | 909 (125-1 231) | 894 (125-1 186) | 633 (125-832) | 367 (73-484) |
| 1, (b), affinity | 1 640 (322-1 960) | 1 618 (322-1 900) | 1 194 (322-1 386) | 689 (189-802) |
| 2, (a), data_fit | 100 (0-1 598) | 100 (0-1 396) | 90 (0-1 197) | 62 (0-1 002) |
| 2, (a), affinity | 527 (38-2 109) | 527 (38-1 848) | 449 (38-1 548) | 268 (38-1 204) |
| 2, (b), data_fit | 154 (42-1 291) | 154 (42-1 205) | 154 (42-1 108) | 154 (42-1 016) |
| 2, (b), affinity | 337 (1-1 796) | 337 (1-1 657) | 329 (1-1 460) | 252 (1-1 218) |

Relative to today's CO2 the reference values are 7.5-27.8 % (product 1, data_fit, constraint a)
and 1.9-3.1 % (product 2, data_fit, constraint a); (a and b) equals (b) on product 1 and lies
below both on product 2 (56 kg/d at data_fit, 1.8 %). Over all weathers and placeholders the
saving spans 73-2 891 kg/d (product 1) and -7-2 340 kg/d (product 2; the 52 negative cells are
product 2, +5 K, `x_s` 0.6 with the other day's `UA`, where the model predicts today's exhaust to
be over-saturated, i.e. a model that does not describe the day).

**c) Which constraint binds** (count over the grid, admissible cells). Product 1, (a): outlet
minimum with minimum air 1 920, with the supply maximum 528, outlet minimum alone 240; (b): RH
maximum with minimum air 1 560, with supply maximum 624, alone 504. Product 2, (a): outlet minimum
with minimum air 1 202, alone 1 002, with supply maximum 274; (b): RH maximum alone 1 094, with
minimum air 788, **with the air MAXIMUM 312**. The proxy for the product moisture binds in almost
every cell, and the second bound is nearly always the minimum air fraction or the 100 degC limit:
the lever is "as little air as allowed, as hot as needed to keep the exhaust where it was". On
product 2 under (b) at `data_fit` the optimum goes the other way, to MORE air (1.1) at a lower
supply (75.4 degC): the extra air takes more waste heat from the preheater and dilutes the exhaust
humidity, which the RH proxy rewards and the temperature proxy (a) forbids.

**d) Weather** (reference, product 1 / 2): -10 K 839 / 117, -5 K 798 / 103, measured 758 / 90,
+5 K 721 / 83 kg/d.

**e) Order of magnitude per year (kg/d x 365, 24/7 as stated by the partner; upper bound).**
Reference: 277 t/a if product 1 ran all year, 33 t/a for product 2, 155 t/a for a 50/50 mix
(placeholder), i.e. 1.3 % of the site's about 11 500 t/a; at 55 / 300 EUR/t about 8 500 / 46 500
EUR/a for the mix. Range at the measured weather: 27-967 t/a (product 1), 0-770 t/a (product 2).

## 5. Calibration on one day, check on the other

The dryer balance closes with a non-negative wall loss only if the solid fraction is large enough
(product-specific minima kept locally; reference placeholders): with less solids the
evaporation alone needs more heat than the process air releases. The value 0.3 used for the
exhaust humidity in Befund 99 (band 0.2-0.5) is therefore inadmissible by the energy balance of
the same days (section 8, item 1). With one shared `x_s` and one shared `UA` both days are
consistent only at **`x_s` = 0.634, `UA` = 0.93 kW/K** -- a calibration of `x_s` from the two
days, not a validation.

| `x_s` | `UA` product 1 / 2 (kW/K) | outlet error, UA from product 1 -> product 2 | product 2 -> product 1 |
|---|---|---|---|
| 0.60 | 0.39 / -2.49 (inadmissible on product 2) | -3.26 K | +11.59 K (UA < 0) |
| 0.65 | 1.18 / 2.49 | +1.40 K | -3.54 K |
| 0.70 | 1.96 / 7.46 | +5.57 K | -11.17 K |
| 0.75 | 2.74 / 12.44 | +9.31 K | -15.77 K |
| 0.634 | 0.93 / 0.93 | 0.00 K | 0.00 K |

Block RMSE equals the day-mean error to within 0.03 K (the within-day spread of the measured outlet
temperature is 0.08 / 0.19 K). By the pre-registered criterion (|error| <= 2 K both ways) the
calibration transfers only at the consistent value 0.634; at the reference 0.65 it fails in one
direction (-3.5 K). Preheater: conductance 17.4 / 33.3 kW/K from the day means (Befund 91: 17.3 /
18.9 kW/K on the minute data, ill-conditioned above effectiveness 0.85 on product 2); transferred
to the other day it predicts 312 against 387 kW (product 2) and 349 against 277 kW (product 1).
**Consequence:** the cells with the other day's `UA` (grid column "other day") are models that do
not reproduce the measured day of the product except near `x_s` 0.634; they widen the ranges of
section 4 and are reported, not discarded. With the own day's `UA` the solid fraction barely
matters (section 6), because the constraint keeps the outlet where it was and the calibration
closes the balance at today's point.

## 6. Sensitivity ranking (one placeholder at a time around the reference)

Largest absolute change of the saving over the two products, kg CO2/d (reference 758 / 90):

| rank | placeholder | values | saving product 1 / 2 (kg/d) | largest change |
|---|---|---|---|---|
| 1 | minimum air fraction | 0.6 / 0.7 / 0.9 | 1 449 / 100, 1 119 / 100, 389 / 62 | 690 |
| 2 | fan law | affinity / n = 0 / n = 2 | 1 320 / 449, 598 / 17, 1 130 / 329 | 561 |
| 3 | waste-heat variant | duty_held | 904 / 396 | 306 |
| (3) | maximum preheater water with duty_held | 70 / 80 / 90 degC | 904 / 396 each | 306 (the switch to duty_held; within 70-90 degC: 0) |
| 5 | constraint variant | (b) / (a and b) | 633 / 154, 633 / 56 | 125 |
| 6 | calibration day | other day | 703 / 193 | 103 |
| 7 | electricity factor | 0.20 | 657 / 26 | 101 |
| 8 | weather | -10 / -5 / +5 K | 839 / 117, 798 / 103, 721 / 83 | 80 |
| 9 | boiler efficiency | 0.85 / 0.92 | 777 / 88, 735 / 94 | 23 |
| 10 | solid fraction (own-day `UA`) | 0.60 / 0.70 / 0.75 / 0.634 | 758 / --, 760 / 95, 760 / 101, 758 / 89 | 11 (0.60 inadmissible on product 2) |
| 11 | fan heat fraction | 0.8 / 1.0 | 765 / 98, 752 / 84 | 7 |
| 12 | room temperature | 10 / 30 degC | 758 / 90, 759 / 96 | 6 |
| 13 | intake humidity | 0.5 / 0.9 | 756 / 87, 760 / 93 | 3 |
| 14 | feed temperature | 20 / 50 degC | 758 / 89, 759 / 91 | 1 |
| 15 | atomising-air temperature | 15 / 40 degC | 758 / 90, 759 / 90 | 0.8 |
| 16 | cp of the solids | 800 / 1600 | 758 / 90, 759 / 91 | 0.5 |

The solid fraction matters strongly only together with the calibration on the other day (grid
medians product 2: 114 / 291 / 510 / 765 kg/d for `x_s` 0.60-0.75), i.e. through the failed
transfer, not through the balance at today's point.

## 7. Questions to the partner (ordered by how much the answer moves the result)

1. **Minimum process-air flow** of plant A per product (fluidisation limit, nozzle and
   agglomeration constraints): is 80 % of today's possible, 70 %, 60 %? (rank 1)
2. **Fan control and fan curves**: are the process-air and exhaust fans speed-controlled
   (frequency converter) or throttled by dampers? Fan curves or a test with two air flows on one
   product; why does the exhaust-fan power halve between the two products? (rank 2)
3. **Waste-heat side at a lower air flow**: can the storage loop / compressor cooling deliver the
   same heat at a higher water temperature, and what limits it (return temperature to the
   compressors, maximum storage temperature)? Otherwise less air means less waste heat used.
   (rank 3)
4. **Product-moisture limit**: which quantity does the operator use (exhaust temperature, bed
   temperature, exhaust humidity, laboratory moisture)? A product-moisture measurement at two
   operating points would replace the proxies. Is a supply temperature of up to 95-100 degC
   allowed for the products? (ranks 5 and the supply bound)
5. **Solid fraction / suspension density per product** (and solid density): decides whether the
   dryer balance transfers between products (section 5) and the reconstructed exhaust humidity.
6. Steam flow or boiler gas and boiler efficiency; electricity factor the partner uses for 2030.
7. Exhaust humidity (one measurement) to check the reconstruction.

## 8. Notes, caveats and what was not done

1. **Report, not fixed:** the solid fraction 0.3 (band 0.2-0.5) of Befund 99 is inadmissible by
   the dryer energy balance of the same two days (needs a larger value; product-specific minima kept locally). It enters
   Befund 99 only through the reconstructed exhaust humidity of plant A (product 2 capped at
   saturation in every step there; with `x_s` 0.65 the reconstructed relative humidity is 0.71).
   Whether this moves a Befund 99 number is for the managing session to decide.
2. Stationary model on day means; no dynamics of the bed or the storage; the product moisture is
   neither modelled nor measured; the RH proxy rests on a reconstructed humidity.
3. Two measured days, one per product; product and day are confounded.
4. The compressed atomising air and its electricity are unchanged and not counted.
5. Plant C does not enter (its flow / exhaust-heat inconsistency, Befund 99, plays no role here).
6. Not done: dynamic or MPC operation, a moisture model, other products, the steam-heater capacity
   (the optimum needs less steam duty than today in every reference cell).
