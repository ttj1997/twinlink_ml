# Site pinch: largest usable waste heat, pinch, source/sink pairing, network level

Status: complete (SCENARIO GRID on placeholders, not a plant statement). Lab book: Befund 100 in
`docs/04_stand_ap1.md`. Code: `src/twinlink_ml/site/`.
Data set: `conf/site/site_placeholder.yaml`. Scripts: `scripts/run_site_pinch.py` (grid,
references, sensitivity, checks; `--variant-of` for the paired variant), `scripts/plot_site_pinch.py`
(figures from the artifact). Artifacts: `artifacts/runs/run_site_pinch_sf0634.json` (script commit
`ed0f277`, clean tree, `git/dirty=False`) and `artifacts/runs/run_site_pinch_sf0634_variant_befund99.json`
(script commit `1e2770c`, clean tree). Tests: `tests/test_site.py`.

Confidentiality: roles only (plant A with storage A1, plant C with lines C1/C2 (planned), "the
partner", compressor station, generic further lines); channel tags only in the git-ignored
`conf/plant/plant_local.yaml`; raw exports only through `TWINLINK_PLANT_DATA_ACE` /
`TWINLINK_PLANT_DATA_RID`.

Every number below is from `run_site_pinch_sf0634` unless marked; numbers marked "derived" are
arithmetic on artifact numbers and partner statements (stated in the text). Every result carries
its scenario; the placeholders and their bands are in section 3.

## 1. Question and answer in one paragraph

How much waste heat can the site use at most, where is the pinch, which sources and sinks belong
together, and at which temperature should the planned waste-heat network run? **In the reference
scenario R1** (plant A product 1, plant C as stated by volume flow, compressor heat 800 kW, hall
heating and process water 200/200 kW, `dT_min` 20 K, no further lines) the minimum fossil heat
(target, no storage) is 655 kW against 1092 kW today: **436 kW less, 3821 MWh/a, 853 t CO2/a**;
the site recovers 1573 kW in total at the target against 1137 kW today. Over the whole grid (2232
scenarios) the saving lies between 62 and 3564 kW (median 993 kW); the number of further lines
on the site, the reading of plant C and the temperature of the compressor cooling water move it
most. **The pinch sits at the inlet of the warmest waste-heat stream**: the exhaust of plant A
(about 61 degC on product 1, about 34 degC on product 2) in 1269 of 2232 scenarios, the exhaust of
the further lines in 439, the compressor water (80 degC) in 400. **Network level: a supply of
70-75 degC with a return of 40-45 degC** reaches at least 90 % of the target saving in 96 % of the
scenarios (direct recovery inside each plant kept, one loop between plants); a supply of at most
60 degC does so in 53 %, and is never better than the 70-90 degC levels by more than 5 points.
What remains above the pinch is the top of the process-air heating (about 60 -> 85 degC on the
cold side) -- the place for a heat pump (section 7, handover to study 2).

## 2. What was reused

| module | reused for |
|---|---|
| `heatnetwork/pinch.py` | condensing exhaust as piecewise-linear segments (`hot_segments_for_source`, sampled from `evaluate_cascade` in condensing mode); `compute_pinch_target` / `evaluate_pinch_target` as the reference of the tests (Befund 60 reproduced: 4048.99 kW, pinch 81.42 degC) |
| `heatnetwork/approach.py` | the minimum approach ALONG a counter-current exchanger (`max_duty_along`) as the cross-check of the loop-capacity formula |
| `storage_timeslice/` (Befund 99) | measured days of plant A (15-minute block means, roles), exhaust-humidity reconstruction, plant C, compressor heat, hall heating and process water (mode flat, day horizon), unchanged; Befund 99's `direct_kwh_per_day` is reproduced for all 896 day-horizon scenarios (largest relative deviation 3.1e-11) |
| `simulation/thermo.py`, `GasConfig` | humidity, saturation, specific heats |
| `evaluation/carrier_energy` | CO2 factors (electricity 0.366, gas 0.201 kg/kWh) |

New: `twinlink_ml.site` -- the site data set (`config.py`), streams of the whole site per
representative month (`streams.py`), an exact problem table on ramp sums with network-level and
heat-pump evaluation (`curves.py`), the year evaluation (`evaluate.py`).

## 3. Site data set (`conf/site/site_placeholder.yaml`; every value a placeholder)

| stream | kind | temperatures | capacity | operation |
|---|---|---|---|---|
| exhaust of plant A | hot | measured outlet (product 1 about 61, product 2 about 34 degC) down to `T_kalt` 15 degC, condensing | measured process + atomising air; humidity reconstructed from spray rate and solid fraction 0.634 (band 0.60-0.70, energy balance of plant A, study 4, `docs/23` section 5), capped at saturation | 24/7 |
| compressor station | hot | 80 -> 50 degC (bands 70-90 / 40-55) | 400 / 800 / 1200 kW (partner: up to about 1200 kW, about 400 kW used today) | 24/7, constant |
| compressor heat of plant C | hot | as above | 2.4 GWh/a = 274 kW (band 1.2-3.6 GWh/a) | as plant C |
| exhaust of plant C (C1, C2) | hot | state of the measured exhaust of plant A | reading "volume": 20 000 Nm3/h per line (band 15 000-25 000); reading "annual": scaled to the partner's 1.8 GWh/a (205 kW mean) | continuous (fraction 1, band 0.5-1) |
| further lines (0-5) | hot + cold | copies of plant A on product day 1 or 2 | 10 000 / 20 000 / 30 000 Nm3/h each | 24/7 |
| boiler flue gas | hot (variant) | 180 degC (band 130-250) down to `T_kalt`, condensing | fuel of the steam heater of plant A today, methane, air ratio 1.15 | variant only |
| process air of plant A | cold | measured intake (shifted to the month) -> measured supply (about 84 degC) | measured | 24/7 |
| process air of plant C | cold | intake -> supply profile of plant A | 20 000 Nm3/h per line | as plant C |
| hall heating | cold | 35 -> 50 degC | 0 / 200 kW at -10 degC, heating limit 15 degC, night factor 1, day 0.4 | heating season |
| process water | cold | 15 -> 60 degC | 0 / 200 kW, constant | 24/7 |

Further placeholders: `dT_min` 20 K (band 10-30; the repository's two exchanger stages in series),
`T_kalt` 15 degC (10-25), intake humidity 0.75 (0.5-0.9), useful heat per fuel 0.90 (0.80-1.0),
today's compressor use 400 kW (300-500), plant C's planned recovery to hall/process water 200 kW
(100-300), pump electricity 1.5 % of transported heat (0.5-3 %), heat-pump efficiency 0.5 of
Carnot (0.4-0.6), electricity CO2 0.366 kg/kWh with the variant "2030" 0.20 (placeholder), gas
0.201 kg/kWh. Climate: 12 monthly means typical of northern Germany (order of magnitude,
placeholder; annual mean 9.8 degC), uniform shift -3/+3 K. Sources and bands per value: YAML file
and the docstring of `twinlink_ml.site.config`.

Contradiction of the partner figures (Befund 99): 2 x 20 000 Nm3/h of plant C give 4.9 times the
partner's annual exhaust-heat figure; both readings are grid axes ("volume", "annual"), plus
"none" (plant C not built = the existing site).

## 4. Method

**Year.** 12 representative days: each measured day of plant A shifted so that its mean intake
temperature equals the monthly mean (the exhaust state is not shifted, Befund 99 convention),
96 steps of 15 minutes; annual means weighted by the days of the month. Per step the streams that
exist at that step; **no storage** (heat is usable only when it occurs, `E_direct` of Befund 99).

**Target.** Problem table of all site streams (`dT_min`, sources cooled down to `T_kalt` with
condensation): minimum hot utility `Q_H`, cold utility `Q_C`, pinch, and the stream boundary at the
pinch. Implemented on sums of ramp functions, exact (no temperature raster); tested equal to
`heatnetwork.pinch.compute_pinch_target`.

**Today.** Recovery only inside today's groups: compressor -> process air of plant A (capped at
400 kW); plant C's own exhaust and compressor heat -> its process air plus at most 200 kW to hall
heating and process water (its planned design); further lines nothing (variant: own exhaust -> own
air). Fossil today = demand - recovery today. **Saving = fossil today - target** (useful heat; fuel
= heat / 0.90; CO2 with 0.201 kg/kWh on the fuel).

**Network level** (the study's network model). One water loop at (supply, return) with approach
`dT_min/2` at every exchanger (consistent with the two-stage `dT_min`); each zone is connected on its
own branch between supply and return (parallel), heat moves directly inside a zone and between
zones only through the loop; a zone may reject unused heat (return cooler). Zones: compressor
station | plant A (own exhaust and process air) | plant C with hall heating and process water |
further lines | (without plant C) hall heating and process water | boiler flue gas. A small LP per
step and level (`zonal_network_target`); levels 40/20 ... 90/60 degC (13). Reported: fossil heat
and the share of the target saving the level reaches, `(fossil today - fossil with network) /
(fossil today - target)`. Diagnostic only: the composite loop (all streams through the loop in any
order, no direct recovery, no return cooler; closed form) and two loops at once.

**Heat pump hints.** From the grand composite curve of each step: the heat an evaporator at `T_e`
can take below the pinch and the heat a condenser at `T_c` can deliver above it without raising
the hot utility elsewhere; COP = 0.5 x Carnot; best pair by CO2 at both electricity factors.

**Checks** (all passed): energy balance per step `Q_H - Q_C = demand - supply` (largest residual
4.0e-8 kW); fossil today >= target and every network level >= target in every run; Befund 99
reproduced (896 scenarios, 3.1e-11); tests: Befund 60 (4048.99 kW), single line = heat-network
pinch per step, network <= direct target, LP = closed form, loop capacity = `max_duty_along` along
the exchanger, time average never needs more utility than the mean of the steps, heat-pump limits.

## 5. Results

### 5.1 Target against today (reference scenarios; `dT_min` 20 K, hall/water 200/200 kW, compressor 800 kW)

| scenario | demand kW | fossil today kW | target kW | saving kW | saving MWh/a | saving t CO2/a | recovered today / target kW | pinch (hot side) |
|---|---|---|---|---|---|---|---|---|
| R1 A product 1, C volume | 2228 | 1092 | 655 | **436** | 3821 | 853 | 1137 / 1573 | 80.0 degC, compressor water inlet |
| R2 A product 2, C volume | 2120 | 1365 | 904 | 462 | 4044 | 903 | 755 / 1216 | 34.5 degC, exhaust inlets (72 % of the time) |
| R3 A product 1, C annual | 2228 | 1421 | 688 | 733 | 6420 | 1434 | 807 / 1540 | 80.0 degC (55 %) / process-air intake (45 %) |
| R4 A product 1, without C | 1136 | 736 | 296 | 440 | 3854 | 861 | 400 / 840 | 80.0 degC, compressor water inlet |
| R5 R1 + 2 lines 20 000 Nm3/h, product 1 | 3320 | 2184 | 1229 | 955 | 8365 | 1868 | 1137 / 2091 | 60.7 degC, exhaust inlets |
| R6 R1 + 2 lines 20 000 Nm3/h, product 2 | 3298 | 2162 | 1295 | 867 | 7595 | 1696 | 1137 / 2003 | 34.6 degC, exhaust inlet of the further lines (64 %) |

**Grid (main slice: `dT_min` 20 K, hall/water 200/200, further lines of 20 000 Nm3/h), saving in
kW (MWh/a in brackets).**

| plant A day, plant C | compressor | 0 lines | 2 lines (product 1) | 2 lines (product 2) | 5 lines (product 1) | 5 lines (product 2) |
|---|---|---|---|---|---|---|
| product 1, volume | 400 | 247 (2163) | 697 (6107) | 468 (4096) | 1373 (12023) | 592 (5186) |
| product 1, volume | 800 | 436 (3821) | 955 (8365) | 867 (7595) | 1630 (14281) | 992 (8690) |
| product 1, volume | 1200 | 436 (3821) | 1169 (10238) | 1169 (10239) | 1888 (16539) | 1392 (12194) |
| product 1, annual | 800 | 733 (6420) | 1231 (10783) | 896 (7845) | 1921 (16831) | 1021 (8945) |
| product 1, none | 800 | 440 (3854) | 1065 (9326) | 867 (7596) | 1740 (15242) | 988 (8658) |
| product 2, volume | 400 | 62 (540) | 530 (4638) | 142 (1248) | 1226 (10739) | 264 (2310) |
| product 2, volume | 800 | 462 (4044) | 930 (8142) | 542 (4752) | 1626 (14243) | 664 (5814) |
| product 2, volume | 1200 | 770 (6746) | 1329 (11644) | 942 (8256) | 2026 (17747) | 1064 (9318) |
| product 2, none | 800 | 391 (3424) | 929 (8141) | 542 (4752) | 1621 (14196) | 664 (5814) |

Whole grid: saving 62-3564 kW (median 993 kW), CO2 121-6972 t/a (median 1943 t/a); `dT_min` 10
instead of 20 K adds 100-783 kW (median 303 kW).

Reading:

* **Compressor heat saturates without further sinks.** With plants A and C alone, 800 and 1200 kW
  of compressor heat give the same target on product 1 (R1: 436 kW both): at 80 degC the water
  can heat process air only up to about 60 degC (`dT_min` 20 K), and below that the demand is
  already covered. 1200 kW pay off only with further low-temperature sinks (further lines, more
  process water) or on product 2 (lower exhaust temperature, less exhaust heat at a useful level).
* **The contradiction of plant C moves the baseline more than the target.** R1 and R3 differ by
  33 kW in the target (655 / 688) but by 329 kW in today's fossil heat (1092 / 1421), because
  plant C's own recovery today is 737 kW by volume flow and 407 kW by the annual figure.
* **Further lines carry the largest potential**, and their exhaust temperature decides how much:
  two lines like product 1 (61 degC) add about 520 kW to R1's saving, two like product 2 (34 degC)
  about 430 kW; at five lines 1630 against 992 kW.
* **Roadmap 400 kW -> 1.2 MW** (derived comparison). In R1 the site recovers 1137 kW today
  (400 kW compressor -> plant A plus plant C's own 737 kW) and 1573 kW at the target, i.e. 436 kW
  more; the roadmap's step from 400 kW to 1.2 MW of used compressor heat (+800 kW) is larger than
  the whole remaining potential of R1 and is reached only with further sinks: the saving exceeds
  800 kW in 1456 of the 2232 grid scenarios (65 %), 1452 of them with further lines (the other four
  at `dT_min` 10 K: plant C by annual figure, or 1200 kW of compressor heat on product 2); the
  total recovered heat at the target exceeds 1.2 MW in 82 %.
* **CO2** (derived): R1's 853 t/a are 7.4 % of the site's 11 500 t/a and 9.3 % of the reduction
  of 9200 t/a the -80 % goal asks for; R5's 1868 t/a are 16 % and 20 %. At 55 / 300 EUR/t the
  853 t/a are worth about 47 000 / 256 000 EUR/a. Pump electricity (1.5 % of the extra heat) costs
  21 t/a in R1 at 0.366 kg/kWh, 11 t/a at 0.20.
* **Consistency with the site's 30 GWh/a gas** (derived from the "none" rows; plant C is planned,
  so not in today's figure): plant A with hall and process water implies 7.2 GWh/a (product 1,
  compressor 800 kW); every further line of 20 000 Nm3/h adds about 5.3 GWh/a; four lines reach
  28.4 GWh/a (95 %), five 33.7 GWh/a (112 %). **More than three to four further lines of that size
  are not compatible with the partner's gas figure** (less if gas has other uses).

### 5.2 Where the pinch is

The pinch is at the inlet of the warmest waste-heat stream that is still short: over the grid the
dominant pinch is the exhaust inlet of plant A in 1269 scenarios, of the further lines in 439, the
compressor water inlet (80 degC) in 400, the intake of the process air in 115 and the process-water
inlet in 9. Its temperature is therefore one of three values: about 61 degC (exhaust, product 1),
about 34 degC (exhaust, product 2), 80 degC (compressor water); with plant C by annual figure it
moves down to the process-air intake (29 degC) part of the year. Consequence: **everything below the
pinch is surplus** (in R1 837 kW go to cooling at the target), **everything above it is the top of
the process-air heating** (R1: 296 kW of plant A and 180 kW per line of plant C above 60 degC on the
cold side = the 655 kW fossil target). No source lies above the pinch in R1.

### 5.3 Which sources and sinks belong together

Single source -> sink pairs (R1, kW recoverable alone): compressor station -> process air A 605,
-> process air C1 / C2 366 each; exhaust A -> process air A 372; compressor C -> process air A /
C1 / C2 274 each; exhaust C1/C2 -> process air A 232 each. The pairing is therefore **compressor
water -> preheating of all process air up to about 60 degC, exhaust -> first preheating stage of
the process air (and process water)**, low-temperature sinks (hall heating, process water) at the
bottom of the cascade. In the network at 70/40 degC (R1) the connectable sources are the compressor
station (800 kW), compressor heat of C (274), exhaust A (132) and C (80 each); the connectable sinks
process air A (605), C1/C2 (366 each), process water (200), hall heating (35).

### 5.4 Network temperature level

Share of the target saving reached by one level (network + direct recovery inside each plant),
median over the 2232 scenarios with a saving above 1 kW (min-max in brackets):

| supply/return degC | 40/20 | 50/30 | 55/35 | 60/40 | 65/35 | 70/40 | 75/45 | 80/50 | 80/60 | 90/60 |
|---|---|---|---|---|---|---|---|---|---|---|
| median share | 0.57 | 0.76 | 0.84 | 0.91 | 0.93 | **0.97** | 0.94 | 0.84 | 0.60 | 0.59 |
| range | -2.85 ... 0.95 | -1.04 ... 1.0 | -0.14 ... 1.0 | 0.30 ... 1.0 | 0.59 ... 1.0 | 0.79 ... 1.0 | -0.08 ... 1.0 | -1.16 ... 1.0 | -3.33 ... 0.98 | -3.33 ... 0.98 |

(A negative share means the network at that level recovers less than today's in-plant recovery.)
Best level per scenario: 70/40 in 623, 75/45 in 522, 60/30 in 333, 50/30 in 244, 55/35 in 173,
65/35 in 170, 80/50 in 93 scenarios. Best level with supply 70-90 degC: median share 0.98 (min
0.82), at least 0.9 in 96 % of the scenarios; with supply at most 60 degC: median 0.91 (min 0.30),
at least 0.9 in 53 %; the low level is never better than the high one by more than 5 points.

Per reference: R1 70/40 and 75/45 reach 1.00, the best low level 60/40 only 0.50; R2 70/40 1.00
and 60/40 0.99; R3 70/40 0.93, 60/40 0.74; R4 70/40 1.00, 60/40 0.62; R5 70/40 1.00, 60/40 0.83;
R6 70/40 0.94, 60/40 0.82. **A low level (supply 40-60 degC) is best only when the pinch is low**: in all
806 scenarios where it is best, the median pinch lies below 40 degC (product 2 on plant A or on the
further lines in 790 of them; 443 at `dT_min` 10 K); **when the
compressor water or a 61 degC exhaust sets the pinch, the network must take the compressor heat at
its own level (supply 70-75, return 40-45 degC)**. Figure:
`docs/figures/fig_site_pinch_network_levels.png`; composite and grand composite curves:
`docs/figures/fig_site_pinch_composites.png`.

Diagnostic (composite loop, two loops at once, R1): one composite loop carries at most 1349 kW
(65/35), two loops 50/30 + 70/50 1520 kW; a second, low loop matters mainly where the composite
loop must replace in-plant recovery.

## 6. Paired variant "Befund 99 assumption" (solid fraction 0.3)

Main value 0.634 (band 0.60-0.70): the energy balance of the two measured days of plant A requires
a larger solid fraction (product-specific minima kept locally) (parallel study 4, branch
`plant-a-operating-point`, `docs/23` section 5); the 0.3 of Befund 99 is inadmissible by that
balance (on product 2 its reconstructed exhaust humidity was capped at saturation in every step).
The 0.3 is kept as a paired variant: 198 grid scenarios (`dT_min` 20 K, hall/water 200/200, further
lines of 20 000 Nm3/h) and R1-R6, same labels, artifact `run_site_pinch_sf0634_variant_befund99`
(differences variant minus main).

| quantity | difference 0.3 - 0.634 (min / median / max over 198 scenarios) |
|---|---|
| available heat of all sources (down to `T_kalt`) | +221 / +765 / +1535 kW |
| fossil target | -38 / -0.1 / +24 kW |
| saving (target vs today) | -15 / +4 / +36 kW (relative -1.9 % / +0.4 % / +13.3 %; the largest relative change is +8.6 kW on a saving of 64 kW) |
| fossil heat with the network at the main best level | -34 / 0.0 / +29 kW |
| best network level changed | 27 of 198, always between levels from 55/35 to 75/45 (e.g. 70/40 -> 65/35 in 5, 75/45 -> 70/40 in 4) |
| dominant pinch changed | 46 of 198; 44 of them the same exhaust stream (inside its condensing range -> its inlet), 2 from the process-air intake to the exhaust inlet of the further lines |

References: R1 -4.0, R2 0.0, R3 +19.2, R4 0.0, R5 -4.0, R6 +4.4 kW of saving. **Reading:** the
lower solid fraction puts much more water into the exhaust (latent heat up to 1.5 MW more), but
that heat lies below the pinch and goes to the cold utility; the site statements of this study
(saving, pinch, network level, heat-pump limits within a few kW) do not depend on the solid
fraction. Within the new band (0.60 / 0.70) the saving of R1 changes by -4 / +7 MWh/a.

## 7. Heat pump: handover to study 2

Heat below the pinch that is needed above it (annual means; evaporator limit `E(T_e)` = heat
available at or above `T_e` below the pinch; condenser limit `U(T_c)` = heat a condenser at `T_c`
can deliver above the pinch):

| scenario | surplus below pinch (cold utility) | `E` at 30 / 40 / 50 degC | fossil above pinch | `U` at 85 / 90 / 100 degC | best pair at 0.366 (COP, condenser kW, electric kW, t CO2/a) | best pair at 0.20 |
|---|---|---|---|---|---|---|
| R1 | 837 kW | 345 / 88 / 43 kW | 655 kW | 133 / 267 / 533 kW | 30 -> 100 degC, 2.67, 515, 193, 388 | same pair, 669 t/a |
| R2 | 1343 kW | 0 / 0 / 0 (51 at 10 degC) | 904 kW | 438 / 567 / 825 kW | 10 -> 45 degC, 4.54, 64, 14, 81 | 10 -> 100 degC, 109 t/a |
| R5 | 1050 kW | 287 / 20 / 0 kW | 1229 kW | 421 / 627 / 1040 kW | 30 -> 85 degC, 3.26, 401, 123, 390 | 10 -> 100 degC, 807 t/a |
| R6 | 1248 kW | 1 / 1 / 0 (95 at 10 degC) | 1295 kW | 509 / 715 / 1128 kW | 10 -> 85 degC, 2.39, 154, 65, 95 | 10 -> 100 degC, 204 t/a |

Handover: **the heat pump case is the product-1 situation** (pinch at the 61 degC exhaust or the
80 degC compressor water): about 300-350 kW of surplus at 30 degC and above, to be lifted to the
top of the process-air heating at 85-100 degC (condenser 400-500 kW, COP about 2.7-3.3 at 0.5 of
Carnot). On the product-2 day the surplus lies below 15 degC and the heat pump is not attractive at
the electricity factor 0.366. The condenser must reach at least 85 degC, the evaporator source is
the exhaust below its dew point or the compressor return water. Study 2 should use these limits
per step (artifact `references.*.heat_pump`), not the annual means alone.

## 8. Sensitivity (one at a time around R1 and R5; largest change of the saving, MWh/a)

| rank | placeholder | R1 | R5 |
|---|---|---|---|
| 1 | number of further lines (R1: 0 -> 2; R5: 2 -> 0 / 5) | +4544 | -4544 / +5916 |
| 2 | today's recovery of the further lines (none / own) | -- | -3944 |
| 3 | reading of plant C (volume / annual / none) | +2599 (annual) | +2418 (annual) |
| 4 | compressor water supply temperature (70 / 90 degC) | -2424 / +1426 | -1829 / +827 |
| 5 | `dT_min` (10 / 30 K) | +1038 / -1119 | +2892 / -2462 |
| 6 | compressor heat (400 / 1200 kW) | -1659 / 0 | -2258 / +1874 |
| 7 | flow of the further lines (10 000 / 30 000) | 0 | -1972 / +1972 |
| 8 | compressor water return (40 / 55 degC) | -641 / -10 | -1240 / +1203 |
| 9 | compressor heat of plant C (1.2 / 3.6 GWh/a) | +1016 / -927 | +417 / -154 |
| 10 | process-water demand (0 kW) | -907 | -154 |
| 11 | today's compressor use (300 / 500 kW; baseline only) | +876 / -867 | +876 / -867 |
| 12 | product day of the further lines | -- | -770 |
| 13 | climate (-3 / +3 K) | +430 / -406 | +737 / -730 |
| 14 | line flow of plant C (15 000 / 25 000) | -598 / +515 | +20 / -20 |
| 15 | operating fraction of plant C (0.5) | +16 | +481 |
| -- | hall heating, process-water temperatures, boiler flue gas (+225 / +268), product day of plant A | 1-310 | 13-476 |
| -- | `T_kalt`, intake humidity, solid fraction 0.60/0.70, useful heat per fuel (CO2 only) | 0-79 | 0-78 |

## 9. Caveats

* **Scenario grid on placeholders.** None of the numbers is a plant statement; the headline (R1)
  is one cell of a grid and must be quoted with its scenario.
* **Two measured days** of plant A; months are the same day shifted in intake temperature (the
  exhaust state is not shifted); the further lines are copies of plant A.
* **Target = lower bound of the fossil heat** (any number of ideal counter-current exchangers,
  `dT_min` everywhere; no storage). The network model is a bound for one loop at fixed temperatures
  with parallel branches and a return cooler; no pressure drops, distances, part load, control.
* **Today's baseline** is a model of the planned design of plant C (its own recovery as a pinch
  target inside the plant, plus 200 kW to hall/process water) and of the compressor use (400 kW).
* The exhaust of plant C copies the state of plant A's exhaust (no combustion products of the
  direct burners); the flue gas is approximated with the specific heats of air.
* Heat-pump figures are limits from the grand composite curve with an isothermal evaporator and
  condenser at `dT_min`; not a design.

## 10. Questions for the partner (from the ranking)

1. **Further lines of the site**: how many, air flow, exhaust temperature and humidity, and what do
   they recover today? (ranks 1, 2, 7, 12; decides most of the potential and whether a 1.2 MW
   network has consumers; the gas figure allows at most three to four lines of 20 000 Nm3/h).
2. **Plant C**: volume flow against the annual exhaust-heat figure, its planned in-plant recovery
   (how much of its own exhaust and compressor heat it uses, at which temperatures), its schedule.
3. **Compressor cooling water**: supply and return temperatures actually available (70-90 / 40-55
   degC matter), the heat as a function of load, how much is used today and when.
4. **Admissible approach** per exchanger and whether intermediate circuits are planned (`dT_min`).
5. **Process water and hall heating**: demand, temperatures, schedule.
6. Not needed for this question: `T_kalt`, intake humidity, solid fraction within 0.60-0.70, the
   efficiency of the heaters (CO2 only).

## 11. Reproducibility

`tools/run_here.sh scripts/run_site_pinch.py --name run_site_pinch_sf0634` (script committed before
the run, clean tree; needs `TWINLINK_PLANT_DATA_ACE` / `TWINLINK_PLANT_DATA_RID` and the git-ignored
role file; 2546 runs, 4.3 h on 8 processes on a shared machine);
`... --name run_site_pinch_sf0634_variant_befund99 --variant-of run_site_pinch_sf0634` (204 runs);
`tools/run_here.sh scripts/plot_site_pinch.py --artifact run_site_pinch_sf0634`. A first start
with solid fraction 0.3 as the main value was stopped after 100 of 2340 runs (no artifact, nothing
used). The field `variant_befund_99` of `run_site_pinch_sf0634` is invalid (it holds the string
"0.3": a shadowed loop variable in the script overwrote the paired result); the variant is in its
own artifact, the script is fixed (`1e2770c`).
