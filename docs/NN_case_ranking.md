# Ranking of the six heat-recovery cases of the partner workbook over the operating range

Branch `case-ranking` (worktree `C:/twinlink_work/case_ranking`, from `master` `a2f44b9`). Lab book: Befund NN
(number assigned at merge). Model against model; no plant data. Code `src/twinlink_ml/case_ranking/`, config
`conf/case_ranking/case_ranking.yaml`, driver `scripts/run_case_ranking.py`, analysis
`scripts/analyze_case_ranking.py` (readings in its docstring, committed with the design before the dry run and the
main run), tests `tests/test_case_ranking.py`.

## 1. Question

The workbook `TwinLink Excel.xlsx` (docs/11 section 13, Befund 110) has six cases of how the two exhausts (dryer 1,
spray tower; dryer 2, fluidised bed) pre-heat the supply air of two sinks over two circuits: cases 1 and 2 one-to-one
(straight / crossed), 3 to 6 cascades; a gas heater per sink. At the workbook point case 1 is best (least total fuel
of both heaters) in all three modes of docs/11 section 13.6, in the partner mode only 0.0053 kg/h ahead of case 2.

* **Q1** Is there, per mode, one case with the least total fuel in the whole admissible input space?
* **Q2** If not: where is the Excel-best case (case 1) also best here, and which simple conditions describe the
  boundary?
* **Q3** How large is the extra fuel (regret) of always choosing case 1 instead of the pointwise best case (median,
  95th percentile, maximum; kg/h and relative)?

## 2. Design (decisions of the project lead of 2026-11-06, F1-F6)

**Inputs (F1 (a), 8 dimensions):** shared ambient temperature and relative humidity of the outside air (the absolute
humidity of the supply air of both dryers and both sinks follows from it, so that no supersaturated corners arise);
per dryer the supply-air temperature, the spray load (kg spray per kg dry air) and the solid fraction. Fixed at the
workbook values of docs/11 section 13.1 (`case_ranking/workbook.py`; the workbook is not read): dry-air flows 120 /
100 kg/h, circuit flows 40 / 30 kg/h (circuit *i* with exhaust *i*), properties, calorific value 45 000 kJ/kg,
efficiency 80 %, spray at 20 degC, 1 atm.

**Modes (F2):** all three of docs/11 section 13.6, each read separately. For a recommendation to the application
partner the physical unpressurised mode is the main variant (as Befund 114); the partner mode shows whether the
workbook says the same.

**Variants (F6):** `fixed` (the workbook's circuit flows; main computation, "does the statement of the workbook
hold") and `optimised` (both circuit flows chosen per case and point, 20-150 kg/h, the range of docs/11 section 10.6;
5 x 5 grid plus the workbook flows, bounded Nelder-Mead from the three best, at most 60 evaluations each; the
workbook flows are a candidate, so the optimised fuel is never above the fixed one).

**Box (F3).** Workbook point in brackets; "lead" = decision of the project lead; every other bound is a placeholder
with the reason given.

| Input | Range | Workbook | Reason |
|---|---|---|---|
| ambient temperature, degC | -10 to 40 | 20 | docs/37 pretraining box |
| relative humidity of the outside air | 0.20 to 0.95 | 0.414 (6 g/kg at 20 degC) | lead |
| supply air dryer 1 (spray tower), degC | 60 to 300 | 300 | up to 300 degC only for the dryer modelled as a spray dryer (lead); lower bound 60 so that the plant-like 75-95 degC lie inside (docs/37 lower bound) |
| supply air dryer 2 (fluidised bed), degC | 60 to 200 | 150 | up to about 200 degC for a fluidised bed (lead); 60-200 = docs/37 pretraining box |
| spray load dryer 1, kg/kg | 0.05 to 0.15 | 0.10 | 0.5-1.5 times the workbook value (lead) |
| spray load dryer 2, kg/kg | 0.015 to 0.045 | 0.03 | 0.5-1.5 times the workbook value (lead) |
| solid fraction dryer 1 and 2 | 0.30 to 0.70 | 0.40 | lead |

The workbook point lies inside the box; for the supply air of dryer 1 it lies on the upper face (300 degC is the
lead's upper bound).

**Named regions (secondary):** *workbook neighbourhood* = workbook point +- 10 % of the box width per input, cut at
the box (one-sided in the supply air of dryer 1); *plant-like* = supply air of dryer 2 (fluidised bed, the type of
plant A) at 75-95 degC as in docs/37, every other input over the full box. Dryer 1 at 75-95 degC with the box's
spray loads saturates its exhaust in practice, so the region is defined on dryer 2.

**Designs:** scrambled Sobol streams (seed 20261106; one stream per set; nested prefixes). Variant `fixed`: 32 768
points over the box (primary), 8 192 in each region, 8 192 in an own stream over the box as the test set of the rule
of Q2. Variant `optimised`: the first 4 096 / 1 024 / 1 024 / 2 048 points of the same streams. Plus 1D cuts through
the workbook point (97 points per input; optimised 25) and three 2D cuts (49 x 49: supply air 1 x supply air 2,
supply air 1 x spray load 1, supply air 2 x spray load 2) for the figures.

**Admissibility (fixed before the run):** both exhausts at or below 90 % relative humidity (IAPWS-IF97) in every mode,
no failure at the point. Inadmissible points are counted (by dryer), not dropped silently. Counted and reported,
not excluded: stages without driving force, stages with negative recovery, and in the partner mode a medium above
99.974 degC (flagged as in docs/11 section 13.5).

**Ties (F4):** exact = less than 1e-9 kg/h (docs/11 section 13); practical = gap below 1 % of the saving of the
pointwise best case against case 0 (placeholder; sensitivity 0.5 % and 2 %; not derived from the rounding of the
workbook).

**Readings** (pre-registered in the docstring of `scripts/analyze_case_ranking.py`): per mode and variant on the
admissible points of the box, on the share of "case 1 strictly best" and, in addition, of "case 1 best or
practically equal": robust >= 0.99, conditional 0.50-0.99 (with the rule of Q2), not robust < 0.50; 95 %
Clopper-Pearson interval, flagged "at the threshold" when it contains 0.99 or 0.50. Q2: decision tree of depth <= 3
on the inputs and derived quantities (exhaust temperatures and their difference, heat-capacity flows of the
exhausts, demand per sink and its share, heat available in each exhaust above ambient + 20 K), tested on its own
stream; "describes the boundary" at a test accuracy >= 0.95 (placeholder). Q3: regret median, 95th percentile,
maximum, in kg/h, relative to the best fuel and relative to the best saving.

**Statistics:** the computation is deterministic and noise-free; `paired_resolution` does not apply. The only
uncertainty is the sample of the space: binomial intervals and the convergence over nested Sobol prefixes.

**Stage B (neural surrogate, F5):** not done (decision of the project lead: the rebuild is exact and practically
free; reference static-transfer, Befund 120).

**`DataSpec.new_study`:** does not apply (static, no process simulator); exception requested in docs/12 section L.

## 3. Reproduction

`tests/test_case_ranking.py` pins at the workbook point (built from the values of docs/11 section 13.1, no file of
`local/`) the total fuel of every case (docs/11 section 13.6, four decimals) and the ranking in all three modes
(partner `1 < 2 < 3 < 5 < 4 < 6`, unpressurised `1 < 2 < 3 = 5 < 4 = 6`, pressurised `1 < 2 < 3 < 5 < 4 < 6`) and the
gap 0.0053 kg/h between cases 1 and 2 in the partner mode.

## 4. Decisions taken without asking (confirmed by the project lead, 2026-11-09)

1. Supply air of dryer 1 up to 300 degC as stated by the lead; the workbook point therefore lies on the upper face of
   that input (not strictly interior); the neighbourhood is one-sided there.
2. Lower bound 60 degC of both supply-air temperatures (docs/37), so that 75-95 degC lie inside for both dryers; the
   plant-like region is defined on dryer 2 only (section 2).
3. Ambient -10 to 40 degC (docs/37). The relative humidity of the outside air uses the saturation pressure over
   water of Buck (1996) (as `simulation/thermo.py`; IAPWS-IF97 is valid from 0 degC only); the exhausts use IAPWS-IF97
   as docs/37.
4. Neighbourhood +- 10 % of the box width (placeholder).
5. Circuit flows of the optimised variant 20-150 kg/h per circuit (docs/11 section 10.6) and the multistart
   settings of section 2. In the optimisation the exhausts are pinned to their own computed values (they do not depend
   on the circuit flows; identical to the unpinned evaluation, tested).
6. Sample sizes: 32 768 points over the box (instead of 65 536 in the first draft) to keep the committed array
   artifact in the size of existing ones; the optimised variant on 4 096 points after its timing (about 2.5 s per
   point for the three modes on one core).
7. Rule: at least 50 points per leaf; reading threshold 0.95 (placeholder).
8. "Strictly best" requires a margin of at least 1e-9 kg/h to every other case; the practical threshold is never
   below the exact tie (relevant only where no case saves fuel).

## 5. Runs

Script commits `74a2c4b` (design, readings) and `7d3ff2b` (memory helper; full suite green on both). Dry runs
`case_ranking_dryrun_v1` (at `816e2d5`; 419 units, 0 failures; the peak memory was not measured: bug in the helper,
fixed in `7d3ff2b`) and `case_ranking_dryrun_v2` (at `7d3ff2b`; 419 units, 0 failures, 0 point errors, 126 MB per
worker, projected 76 min with 8 workers); end-to-end test of the analysis on the dry run
(`case_ranking_dryrun_v2_analysis`, no result). Main run `case_ranking_v1` (at `a9c16be`, clean tree, 419 units, 0
failures, 0 point errors, 4312 s with 8 workers; arrays `case_ranking_v1.npz`); analysis `case_ranking_v1_analysis`
(at `be37865`, clean tree) with the figures in `docs/figures/case_ranking/`.

## 6. Results

All numbers from `case_ranking_v1_analysis.json`. Shares with the 95 % Clopper-Pearson interval; fuel in kg/h.

### 6.1 Admissibility and counted flags

Box (`full`): 16 803 of 32 768 points admissible. Inadmissible 15 965: exhaust 1 above 90 % at 14 675 points, exhaust 2
at 2 344 (both possible). No point failed in any mode; the admissible sets of the three modes are identical (0
disagreements). The workbook neighbourhood is admissible throughout (8 192 of 8 192); plant-like 3 778 of 8 192; the
rule's test stream 4 185 of 8 192. The inadmissible part is mostly dryer 1 (spray tower) at low supply air and high
load (figure `..._cut2d_0_...`: below about 200 degC at the workbook load and solid fraction).

Counted on the admissible points of the box (fixed flows), not excluded:

* partner mode, medium above 99.974 degC: case 1 at 5 400 points, case 2 5 470, cases 3-6 6 236 to 7 298;
* partner mode, a stage with negative recovery: cases 1 and 2 at 19 points, cases 3, 4, 5, 6 at 703, 982, 2 530,
  3 048;
* physical modes, a stage without driving force: cases 1 and 2 at 1 081 points, the cascades at 6 028 to 7 376
  (unpressurised) and 6 725 to 8 975 (pressurised); no negative recovery.

### 6.2 Readings of case 1 (primary: admissible points of the box)

| Variant | Mode | case 1 strictly best | reading | best or practically equal | reading | 0.5 % / 2 % |
|---|---|---|---|---|---|---|
| fixed | partner | 0.740 [0.733, 0.747] | conditional | 0.854 [0.849, 0.859] | conditional | 0.801 / 0.934 |
| fixed | unpressurised 95 (main) | 0.471 [0.464, 0.479] | **not robust** | 0.666 [0.659, 0.673] | conditional | 0.651 / 0.705 |
| fixed | pressurised | 0.541 [0.533, 0.548] | conditional | 0.645 [0.637, 0.652] | conditional | 0.617 / 0.695 |
| optimised | partner | 0.693 [0.673, 0.713] | conditional | 0.793 [0.775, 0.810] | conditional | 0.744 / 0.875 |
| optimised | unpressurised 95 | 0.580 [0.559, 0.601] | conditional | 0.825 [0.808, 0.841] | conditional | 0.811 / 0.848 |
| optimised | pressurised | 0.914 [0.901, 0.925] | conditional | 0.977 [0.970, 0.983] | conditional | 0.972 / 0.988 |

No interval contains 0.99 or 0.50 (no reading "at the threshold"). The shares are stable over the nested prefixes: from
2 096 admissible points on every share is within 0.007 of its value at 16 803 (figure `..._convergence.png`).

Secondary sets (fixed flows):

| Set | Mode | strictly best | reading | best or practically equal | reading |
|---|---|---|---|---|---|
| workbook neighbourhood | partner | 1.000 [0.9995, 1] | robust | 1.000 [0.9995, 1] | robust |
| workbook neighbourhood | unpressurised 95 | 0.950 [0.945, 0.954] | conditional | 0.993 [0.991, 0.995] | robust |
| workbook neighbourhood | pressurised | 0.977 [0.973, 0.980] | conditional | 0.997 [0.996, 0.998] | robust |
| plant-like | partner | 0.976 [0.970, 0.980] | conditional | 0.994 [0.991, 0.996] | robust |
| plant-like | unpressurised 95 | 0.779 [0.765, 0.792] | conditional | 0.951 [0.944, 0.958] | conditional |
| plant-like | pressurised | 0.779 [0.765, 0.792] | conditional | 0.951 [0.944, 0.958] | conditional |

(In the plant-like set the two physical modes give identical shares: dryer 2's supply air at 75-95 degC keeps the
medium below the 95 degC limit wherever case 1 is decided.)

### 6.3 Q1: no case is best in the whole admissible space

In no mode and no variant is one case co-best at every admissible point of the box, neither with the exact nor with
the practical tie. Case 1 has the largest co-best share everywhere (fixed: 0.740 / 0.636 / 0.592 for partner /
unpressurised / pressurised; optimised: 0.693 / 0.798 / 0.965); its competitor is case 2 (fixed: 0.259 / 0.481 /
0.411). The cascades are almost never the sole best: case 3 alone at 19 of 16 803 points (partner); in the physical
modes they are co-best only in ties (`1=3=5` at 801 points, `2=3=5` at 233 / 223, all six equal at 47 points where no
case saves anything). Only in the workbook neighbourhood is case 1 best at every point (partner, both variants;
pressurised optimised).

Median saving against case 0 (fixed, box): case 1 19.5 / 29.2 / 34.7 %, case 2 19.1 / 28.8 / 32.7 %, cascades 14.8 to
16.6 / 16.7 to 20.2 / 20.9 to 26.1 % (partner / unpressurised / pressurised). Gap between places 1 and 2: median
0.0036 / 0.0107 / 0.0132, 95th percentile 0.0105 / 0.0316 / 0.0562 kg/h.

### 6.4 Q2: the boundary is the order of the two exhaust temperatures

Depth-3 trees on the inputs and derived quantities, target "case 1 strictly best" (fixed flows), tested on the own
stream (4 185 admissible points): test accuracy 0.978 (partner), 0.966 (unpressurised), 0.971 (pressurised); outside
the practical tie 0.998 / 0.992 / 0.995; reading "describes the boundary" (>= 0.95) in all three modes, and also for the
target "best or practically equal" (0.979 / 0.973 / 0.974) and in the optimised variant (0.955 to 0.986). The dominant
quantity is the difference of the exhaust temperatures `dT = T_exhaust,1 - T_exhaust,2` (importance 0.90 / 0.59 /
0.79). The effective leaves:

* **physical modes**: case 1 where `dT > -0.54 K` (exhaust 1 at least as warm as exhaust 2) and exhaust 2 has heat
  above ambient + 20 K; case 2 where exhaust 2 is the warmer one. Unpressurised, one small branch with exhaust 2
  above 118.5 degC depends on the supply humidity (21.1 g/kg);
* **partner mode**: case 1 where `dT > -15.5 K`; for `-27.1 K < dT <= -15.5 K` where exhaust 1 has more than 1.30 kW
  above ambient + 20 K; below `-27.1 K` only with more than 2.71 kW and `dT > -44.6 K`. The partner model keeps case 1
  even when exhaust 1 is 15 to 27 K colder than exhaust 2.

Case 1 connects exhaust 1 (circuit 40 kg/h) with sink 1, case 2 crosses them. The rule is a description of where
the rebuild ranks them, not physics. In the optimised variant the circuit flows absorb part of the temperature
order; the pressurised tree then splits on the heat available in the two exhausts (accuracy 0.976).

### 6.5 Q3: regret of always choosing case 1

`F_1 - min_k F_k` over the admissible points of the box (median, 95th percentile, maximum):

| Variant | Mode | kg/h | % of the best fuel | % of the best saving |
|---|---|---|---|---|
| fixed | partner | 0 / 0.0039 / 0.0082 | 0 / 0.51 / 1.54 | 0 / 2.3 / 5.9 |
| fixed | unpressurised 95 | 0 / 0.0269 / 0.0416 | 0 / 4.15 / 10.13 | 0 / 10.6 / 16.7 |
| fixed | pressurised | 0 / 0.0300 / 0.0495 | 0 / 4.94 / 10.50 | 0 / 9.8 / 14.9 |
| optimised | partner | 0 / 0.0076 / 0.0135 | 0 / 1.04 / 2.60 | 0 / 3.3 / 6.3 |
| optimised | unpressurised 95 | 0 / 0.0243 / 0.0390 | 0 / 3.42 / 9.13 | 0 / 8.4 / 16.7 |
| optimised | pressurised | 0 / 0 / 0.0098 | 0 / 0 / 1.68 | 0 / 0 / 5.2 |

(The relative-to-saving column leaves out the 47 points without any saving, 6 in the optimised set.) In the
workbook neighbourhood the regret is 0 (partner; pressurised optimised) or at most 0.0114 kg/h (unpressurised, fixed).

**Circuit flows against topology.** Case 1 at the workbook's flows against the best case at its optimised flows
(same 2 096 points): median / 95th percentile / maximum 0.058 / 0.090 / 0.120 kg/h (partner), 0.024 / 0.056 / 0.086
(unpressurised), 0.063 / 0.130 / 0.195 (pressurised). The median topology regret is 0 in every mode: choosing the
circuit flows is worth more than choosing among the six drawings (as docs/11 section 13.7 suggested at the workbook
point). In the partner mode 76 % of the optimised flows sit at the upper bound of 150 kg/h (its recovery grows with
the flow up to the bound), so the partner number depends on the bound.

### 6.6 Reading

1. **The statement of the workbook ("case 1 is best") holds near the workbook point, not over the operating range.**
   In the main variant (physical, unpressurised) case 1 is strictly best on 47 % of the admissible box (not robust)
   and best or practically equal on 67 % (conditional). Near the workbook point it is best or practically equal on
   99.3 % (robust in that reading).
2. **The partner mode overstates the robustness of case 1** (strictly best 74 % against 47 % / 54 %) and reports a
   gap between places 1 and 2 three to four times smaller than the physical modes.
3. **The choice is between cases 1 and 2 and follows the order of the exhaust temperatures** (physical modes: the
   warmer exhaust goes to sink 1); the cascades are not competitive in this box.
4. **Always choosing case 1 costs little in the median and up to about 10 % of the fuel at the worst point**
   (unpressurised: 95th percentile 0.027 kg/h, maximum 0.042 kg/h); the circuit flows matter more than the choice of
   the drawing.

### 6.7 Caveats

* Model against model; no plant data. The box is a placeholder box (section 2); the supply air of dryer 1 has the
  workbook point on its upper face.
* About half of the box is inadmissible, almost entirely through dryer 1; the shares are shares of the admissible
  part.
* The rule is fitted on the same kind of points it describes (own test stream, but same box); it describes the
  rebuild, not a plant.
* Optimised variant: multistart Nelder-Mead on a bounded box; flat objectives (a stage without driving force) can
  end at a bound without meaning. Partner mode: the optimum lies at the upper flow bound for most points.
* Partner mode: the medium exceeds the boiling point at about one third of the admissible points for case 1 (flagged
  as in docs/11 section 13.5).
* Deterministic computation; the intervals cover only the sampling of the space.
