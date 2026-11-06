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

## 4. Decisions taken without asking (to be confirmed)

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

## 5. Results

(after the main run)
