# NN -- Stationary Dyssol flowsheets as a data source: how many Dyssol runs does a surrogate need?

Status: **complete; pre-registered reading D2 in both cases (sections 11 to 13).** Sections 1 to 10
are the design and the pre-registered reading, committed BEFORE any Dyssol data set was generated
(cfefde2) and unchanged since; the results below take their numbers only from the artifacts named
there. Branch `dyssol-steady-state` (from `master` 61d5b18), worktree `C:/twinlink_work/dyssteady`.
The document number and the finding number are open (assigned at the merge). Code
`src/twinlink_ml/dyssol_steady/`, settings `conf/dyssol_steady/dyssol_steady.yaml`, drivers
`scripts/run_dyssol_steady.py`, `scripts/analyze_dyssol_steady.py`, `scripts/build_dyssol_units.py`.

## 1. Question

TUHH-A delivered stationary Dyssol units (dryer, heater, heat recovery) and flowsheets built from
them (2026-11-19). **How many Dyssol runs does a surrogate model of such a flowsheet need when no
rebuild of the flowsheet exists?**

The stationary case is the rehearsal, not the application: a stationary Dyssol run takes about a
second, so a surrogate saves nothing here. The chain design -> Dyssol script -> run -> data set ->
fit -> reading is what carries over to dynamic Dyssol models, where one run is expensive
(docs/02, section "Temporal resolution").

**Relation to Befund 120 (docs/37).** There the balance was the competitor of the learned models,
and calibrating a few physical parameters was best wherever the balance's structure fitted. Here
the balance itself is exact: a Python rebuild of the three units reproduces Dyssol (control of
section 6). This study therefore does NOT ask whether a learned model beats the balance; it
measures the data need of a surrogate for the case that a rebuild is not available. The rebuild
serves as the reference that checks every Dyssol result.

## 2. Decisions of the project lead (2026-11-19)

1. **Box.** Dry-gas flow and dryer inlet temperature after the ranges of the partner meeting of
   2026-11-19 (the gas flow widened downwards so that the stored point of line 1 lies inside);
   spray flow, solid fraction of the spray and medium flow +-25 % around the stored point.
   Over-saturated points are flagged and counted, not removed.
2. **Confidentiality.** Plant values, the box and every generated data set stay local
   (git-ignored; copy in `C:/twinlink_runs/<branch>/`). In the repository: code, a template with
   placeholders, derived numbers.
3. **Training** with this document and a pre-registered reading before the data set is generated.
   Models: MLP against linear and quadratic regression; learning curve over the number of runs;
   5 seeds; Holm per family. Reading on the smallest number of runs from which the MLP (or a
   baseline) meets 1 K on the exhaust temperature and 1 % on the fuel.
4. **Then the same for a flowsheet with two lines** (own section, same reading), without asking
   back.
5. **Compute.** Another study runs on the machine: until its probe has finished only code,
   adapter, design and single tests; then the full suite, the dry run and the data generation
   with at most 2 workers. Rules of "Long runs" apply. No merge, no push.

## 3. Source

**Delivery** (`<local>/partner/dyssol_2026-11-19/`, confidential, read through
`TWINLINK_LOCAL_DIR`, never committed): seven `.dflw` flowsheets with stored results, the C++
sources of four units, a comparison workbook. The delivery has no compiled units;
`scripts/build_dyssol_units.py` builds the three finished ones against the SDK shipped with
Dyssol and writes a manifest of digests. The fourth unit (a heat exchanger with area and
heat-transfer coefficient) is unfinished and used by no flowsheet.

**Materials database.** The dryer unit needs the normal boiling point of liquid water; the
database shipped with Dyssol has none, and the unit refuses to start. The unit's author set
373.15 K and changed nothing else (statement of 2026-11-19). The drivers apply exactly this change
to a copy (`materials.patch_materials_database`); the installed file is never written. Digests of
both files are in every record.

**The units** (closed formulas, `dyssol_steady.reference`):

* dryer: the feed is dried to a given equilibrium moisture; energy balance with constant heat
  capacities and liquid water at the boiling point as the reference; gas and product leave at one
  temperature. **No saturation limit**: the exhaust can come out over-saturated (the open question
  of Befund 106, docs/11 section 12.7 item 2);
* heat recovery: hot gas, a circulating medium and cold gas, ideal exchangers, no condensation;
* heater: heats to a set temperature; duty from Dyssol's tabulated mixture enthalpy; fuel from
  efficiency and calorific value.

Because every dryer inlet is held at its heater's set temperature, the flowsheets of this family
have no real recycle.

**Flowsheets** (keys of `cases` in the YAML):

* `one_line` -- `Case 1.1.dflw`: one line whose exhaust preheats its own fresh gas;
* `two_lines` -- `Case 2.dflw`: two lines, each exhaust preheats the fresh gas of the other line.

The topology is read from the ports of the file (`flowsheet.discover_topology`); unit and stream
names never appear in the code. The adapter handles the family "one heat recovery per exhaust,
each preheating exactly one fresh-gas feed" and rejects anything else.

**How Dyssol is run** (`flowsheet.DyssolFlowsheet`): one `DyssolC` process per point with a
generated script that keeps everything of the source file and sets, at the first and the last time
point, mass flow, temperature, phase and compound fractions of both feeds of every line and the
unit parameters. Results are read from the result file at the last time point with the existing
reader (`partner_v4_1.comparison.load_dyssol_steady_state`). Only the folder with the three
compiled units is on the model path.

## 4. Data

**Varied inputs, five per line:** dry-gas flow, dryer inlet temperature (the heater's set
temperature), spray flow, solid fraction of the spray, medium flow of the heat recovery that cools
the line's exhaust. Everything else (humidity and temperature of the fresh gas, spray temperature,
equilibrium moisture, heater efficiency, calorific value) stays at the value stored in the source
file. `one_line` has 5 inputs, `two_lines` 10 (the same rule for the second line, decision 4).

**Box:** `<local>/dyssol_steady/box_local.yaml` (template `conf/dyssol_steady/box_local.yaml.example`).
Records carry the SHA-256 of the resolved box only.

**Design:** training pool of 1024 scrambled Sobol points and an independent test set of 256
scrambled Sobol points in the same box (seeds in the YAML). Every point is one Dyssol run and one
unit of the data stage. **Test set name: in-box Sobol test set, independent of the training pool.**

**Learning curve:** N = 8, 16, 32, 64, 128, 256, 512 runs. Per seed (1-5) the pool is permuted
once; the training set of size N is the first N points of that permutation (nested over N,
different between seeds). The seed also initialises the MLP.

**Over-saturation:** a point is flagged when the dryer exhaust of any line holds more water than
saturated air at its temperature (1 atm; `flowsheet.validators.saturation_humidity_kg_per_kg`).
Flagged points stay in the pool and in the test set and are counted per set.

**Methodology note.** The preset `DataSpec.new_study` (CLAUDE.md) governs data generated with the
grey-box simulator; this study generates no grey-box data, so the preset does not apply.

## 5. Models

All models see the inputs scaled to [-1, 1] by the box and predict five targets per line at once:
exhaust temperature after the heat recovery, dryer outlet temperature, preheated-air temperature,
recovered heat, fuel consumption.

* `linear`, `quadratic`: ordinary least squares on all monomials up to degree 1 / 2 (6 / 21
  coefficients at 5 inputs, 11 / 66 at 10). No regularisation parameter; minimum-norm solution
  where N is below the number of coefficients.
* `mlp`: `d -> 64 -> 64 -> targets`, tanh, float64, full-batch Adam on the mean squared error of
  targets standardised with the training set, optional weight decay (architecture as in docs/37).

**Pilot of the MLP optimiser** (per case, before any Dyssol data): six settings (learning rate
0.001 / 0.003, weight decay 0 / 0.001 / 0.1, 5000 steps) at N = 16, 64, 256 and pilot seeds 11, 12,
on a pilot pool and pilot test set **computed with the Python rebuild, not with Dyssol**, with
their own Sobol seeds. Selection: lowest mean over pilot seeds and budgets of the criterion
(mean of both reading numbers, each divided by its threshold); ties to the setting listed first;
no extension of the grid. The selection is committed into `mlp.selected.<case>` before the data
stage, which refuses to start without it.

## 6. Pre-registered reading (the same for both cases)

Per cell (model, N, seed), on the test set:

* `exhaust_p95_k`: 95th percentile of the absolute error of the exhaust temperature after the heat
  recovery [K]; with two lines the larger of the two;
* `fuel_p95_percent`: 95th percentile of the relative error of the total fuel consumption [%].

**Thresholds: 1 K and 1 %** (decision 3). Justification: 1 K is of the size by which the partner's
own two implementations of the dryer differed in the delivery 4.1_2 (Dyssol against the Excel
equations: 0.46 K at identical inputs, +0.65 to +1.5 K in the tables; docs/11 section 12.7 item 4,
Befund 106); a surrogate error at or below it is not distinguishable from the choice between
those implementations. The 1 % on the fuel has no measured source and is a setting: the heater
efficiency that turns duty into fuel is itself a round placeholder of the flowsheet.

A model **meets the thresholds at N** if the median over the five seeds of both numbers is at or
below its threshold. **N\*** of a model: the smallest N of the grid from which on it meets them at
every larger N of the grid as well.

0. **Control (before any fit):** Dyssol against the rebuild at every point of the data set. Bounds
   in the YAML (1e-6 K, 1e-6 % of heat flows and fuel relative to max(|value|, 1)). Above them the
   data set is not used and the deviation is reported.
1. **Feasibility first:** per model and N the number of seeds (of five) meeting both thresholds.
2. **Verdict:**
   * **D1** -- the MLP has an N\* and no baseline has one at or below it: an MLP is needed; data
     need = N\* of the MLP.
   * **D2** -- a baseline has an N\* at or below that of the MLP, or the MLP has none: a regression
     suffices; data need = the smallest baseline N\* (the simpler model at equal N\*).
   * **D3** -- no model has an N\*: the data need exceeds 512 runs.
3. **Rank claims** (MLP against each baseline, per N and per number): paired t-test over the seeds
   (`evaluation/paired_resolution.py`; the seeds share the training subsets), Holm per family; a
   family is one baseline and one number over all seven N. A rank is claimed only where the
   Holm-adjusted test is resolved. The verdict does not depend on these tests. The test is on the
   mean of the paired differences; the reading uses medians.

**Reported without a reading:** the same reading on the admissible (not over-saturated) test
points; median and maximum error next to the 95th percentile; the other targets.

## 7. Stages and compute rules

`pilot` -> commit of the selection -> `data --dry-run` -> `training --dry-run` -> `data` ->
`training` -> `analyze`, per case; `two_lines` after `one_line`.

* Units of the data stage are the points; each finished point is appended to a cache at once,
  `--resume` continues, a failing point is recorded and does not stop the others. Units of the
  training stage are the cells (model x N x seed), cached the same way.
* **Dry run:** every point through the rebuild (extreme draws, over-saturation counts); Dyssol at
  the points with the extreme rebuild results and at the first points of pool and test set
  (time and peak memory per Dyssol process); then every cell of the training stage with 5
  optimiser steps on that data set. Its numbers are never results; its outputs stay local.
* At most 2 workers (decision 5). Main runs are started detached with output to a log file.
* Data sets and caches: `<root>/local/runs/dyssol_steady/` (git-ignored), copied to
  `C:/twinlink_runs/dyssol-steady-state/`. Committed artifacts hold errors in physical units,
  counts and digests, never an input value.

## 8. Seen before the data set (disclosure)

* The seven stored flowsheets of the delivery and the workbook.
* Single Dyssol runs while writing the adapter: re-runs of the stored flowsheets and four
  off-nominal points (two flowsheets with one line, one with two). Three stored flowsheets were
  reproduced to the last stored digit, four to about 0.01-0.03 K (probably saved with an older
  state of the units; open question to the author). These were probes without an artifact; the
  control of section 6 measures the agreement on every point.
* The rebuild agreed with those probe runs to rounding after one correction that the probes
  revealed: outside the temperature range of a compound's enthalpy correlation Dyssol evaluates
  the correlation at the nearest bound (visible in the heater duty when the heater inlet is
  colder than 290 K, the lower bound of the vapour correlation).
* Timing of single runs (about 1.4 s with only the three units on the model path).
* No learning curve, no fit on Dyssol data, no pilot result at the time of this commit.

## 9. Decisions taken without asking

* **Reading statistic.** Decision 3 fixes the thresholds "in the median"; this document reads that
  as the median over the seeds and takes the 95th percentile over the test set as the per-seed
  number (a surrogate that "meets 1 K" should do so at nearly all points; the maximum over 256
  points is reported next to it).
* **"Exhaust temperature"** is the exhaust after the heat recovery (the workbook's "exhaust gas
  temperature"), not the dryer outlet; the dryer outlet is a reported target.
* **"Fuel"** is the total fuel of the flowsheet (one heater per line).
* **Five varied inputs per line**, the ones decision 1 names; the remaining inputs stay at the
  stored values.
* **Two-line flowsheet:** `Case 2.dflw`. Two other two-line flowsheets of the delivery put two heat
  recoveries in series on one fresh-gas stream (outside the adapter's family, rejected by a test);
  the remaining ones still hold inputs that differ from the plant sheet.
* **"From which on":** N\* requires the thresholds at every larger N of the grid as well.
* **Pilot on the rebuild** instead of fixed optimiser settings: the selected setting of docs/37
  was chosen for noisy data at N <= 64 and carries a strong weight decay.
* **Control bounds** of 1e-6 instead of a tenth of the thresholds, since the probes agreed to
  rounding.

## 10. Confidentiality

Not in the repository: the delivery (sources carry a person's name), the plant values of the
workbook, the box, the data sets, the names of the plant lines. The committed template of the box
holds placeholders. `tests/test_denylist.py` is run before every commit of this branch.

## 11. Results: `one_line`

Artifacts: training `artifacts/runs/dyssol_steady_v1_one_line_training.json.gz` (driver aeedeec,
clean tree), analysis `artifacts/runs/dyssol_steady_v1_one_line_analysis.json`. The data set
`dyssol_steady_v1_one_line.json.gz` is local; its SHA-256 is in the training artifact. Sections
1-10 are unchanged since the commit before the data set (cfefde2); the pilot selection was added
to the YAML in aeedeec, before the data stage.

**Data.** 1024 + 256 Dyssol runs, none failed. Over-saturated: 9 of 1024 points of the pool, 3 of
256 of the test set (253 admissible test points). Dyssol per run: median 1.43 s, maximum 2.42 s,
1881 s in total, peak 54 MB per process.

**Step 0, control: held.** Dyssol against the rebuild at all 1280 points: largest temperature
deviation 1.7e-13 K, largest relative deviation of heat flows and fuel 6.3e-12 % (bounds 1e-6).

**Step 1, feasibility, and the reading numbers.** Median over the five seeds (range); bold = both
medians at or below their threshold; "seeds" = seeds meeting both thresholds. Test set: in-box
Sobol test set, independent of the training pool (all 256 points).

| N | linear: exhaust P95 [K] | fuel P95 [%] | seeds | quadratic: exhaust P95 [K] | fuel P95 [%] | seeds | MLP: exhaust P95 [K] | fuel P95 [%] | seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2.72 (2.21-6.26) | 12.71 (8.44-24.25) | 0/5 | 26.81 (19.66-35.07) | 44.28 (26.44-50.02) | 0/5 | 6.63 (3.83-8.83) | 9.80 (7.62-15.80) | 0/5 |
| 16 | 2.21 (1.90-2.35) | 9.07 (8.34-10.87) | 0/5 | 13.89 (7.47-14.09) | 19.65 (16.99-22.72) | 0/5 | 2.08 (1.90-2.76) | 5.12 (4.10-6.15) | 0/5 |
| 32 | 1.94 (1.90-2.14) | 8.23 (7.98-9.16) | 0/5 | **0.37** (0.26-0.40) | **0.41** (0.33-0.54) | 5/5 | 1.69 (1.09-2.19) | 1.97 (1.81-2.33) | 0/5 |
| 64 | 1.97 (1.83-2.00) | 7.40 (6.72-8.81) | 0/5 | **0.24** (0.22-0.25) | **0.32** (0.26-0.39) | 5/5 | **0.48** (0.37-0.76) | **0.84** (0.68-1.04) | 4/5 |
| 128 | 1.92 (1.77-1.99) | 7.42 (6.61-7.93) | 0/5 | **0.20** (0.19-0.23) | **0.28** (0.26-0.32) | 5/5 | **0.33** (0.26-0.48) | **0.60** (0.47-0.78) | 5/5 |
| 256 | 1.91 (1.77-1.97) | 7.22 (7.02-7.31) | 0/5 | **0.18** (0.18-0.20) | **0.24** (0.23-0.26) | 5/5 | **0.23** (0.20-1.01) | **0.41** (0.35-1.67) | 4/5 |
| 512 | 1.95 (1.82-2.00) | 7.14 (6.98-7.67) | 0/5 | **0.17** (0.17-0.17) | **0.22** (0.22-0.23) | 5/5 | **0.19** (0.18-0.19) | **0.38** (0.32-0.40) | 5/5 |

**Step 2, verdict: D2.** N\* of the quadratic regression is 32, of the MLP 64; the linear
regression has none. **Data need: 32 Dyssol runs, reached by the quadratic regression.**

**Step 3, rank claims** (22 of 28 tests resolved after Holm). MLP against quadratic: the MLP is
better at N = 8 and 16 on both numbers (resolved; there the quadratic regression has fewer runs
than its 21 coefficients); the quadratic regression is better at N = 32, 64, 128 and 512 on both
numbers (resolved); at N = 256 nothing is resolved (one MLP seed lies at 1.01 K and 1.67 %). MLP
against linear: the MLP is better on the exhaust from N = 64 and on the fuel from N = 16
(resolved); the remaining four tests (both numbers at N = 8, the exhaust at N = 16 and 32) are
not resolved.

**Without a reading.** On the 253 admissible test points the verdict and every N\* are the same.
Largest error over the test set (median over seeds; largest seed in brackets): quadratic at N = 32
0.82 K (0.90 K) and 1.45 % (1.59 %), at N = 512 0.42 K (0.46 K) and 0.91 % (0.95 %); MLP at N = 64
1.24 K (1.98 K) and 2.12 % (2.43 %), at N = 512 0.38 K (0.72 K) and 0.92 % (1.21 %). The
thresholds are therefore met at the 95th percentile, not at every test point. Other targets of the
quadratic regression at N = 32 (95th percentile, median over seeds): dryer outlet 0.47 K,
preheated air 0.31 K, recovered heat 2.61 kW.

## 12. Results: `two_lines`

Artifacts: training `artifacts/runs/dyssol_steady_v1_two_lines_training.json.gz` (driver aeedeec;
run on 0924f2a, which differs from aeedeec only by the two artifacts of section 11; clean tree),
analysis `artifacts/runs/dyssol_steady_v1_two_lines_analysis.json`. Data set
`dyssol_steady_v1_two_lines.json.gz` local, SHA-256 in the training artifact. Same reading, same
settings; 10 varied inputs, 10 targets.

**Data.** 1024 + 256 Dyssol runs, none failed. Over-saturated: **378 of 1024** points of the pool
and **99 of 256** of the test set (157 admissible test points). Dyssol per run: median 1.63 s,
maximum 2.31 s, 2135 s in total, peak 59 MB per process.

**Step 0, control: held.** 1280 points: largest temperature deviation 1.7e-13 K, largest relative
deviation 4.6e-11 %.

**Step 1, feasibility, and the reading numbers** (as in section 11; the exhaust number is the
larger one of the two lines, the fuel is the total of both heaters; all 256 test points).

| N | linear: exhaust P95 [K] | fuel P95 [%] | seeds | quadratic: exhaust P95 [K] | fuel P95 [%] | seeds | MLP: exhaust P95 [K] | fuel P95 [%] | seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 35.83 (22.77-50.79) | 39.09 (25.07-67.37) | 0/5 | 50.66 (42.67-62.72) | 58.03 (47.74-71.90) | 0/5 | 29.33 (16.30-41.03) | 19.32 (14.06-26.62) | 0/5 |
| 16 | 5.67 (5.14-6.47) | 9.05 (7.36-10.58) | 0/5 | 35.28 (30.42-40.83) | 43.47 (34.38-45.89) | 0/5 | 6.43 (5.07-7.10) | 7.87 (5.87-9.19) | 0/5 |
| 32 | 4.57 (4.41-5.95) | 6.29 (5.22-6.37) | 0/5 | 23.80 (21.86-30.43) | 34.80 (32.64-42.99) | 0/5 | 4.67 (4.17-5.61) | 5.16 (4.60-5.57) | 0/5 |
| 64 | 4.18 (3.81-4.60) | 5.35 (5.28-6.32) | 0/5 | 6.94 (5.24-10.94) | 10.40 (6.62-10.63) | 0/5 | 3.36 (2.77-3.98) | 2.85 (2.59-3.11) | 0/5 |
| 128 | 3.88 (3.71-4.03) | 5.32 (4.93-5.49) | 0/5 | **0.53** (0.46-0.54) | **0.23** (0.18-0.27) | 5/5 | 1.61 (1.34-1.79) | 1.65 (1.12-1.72) | 0/5 |
| 256 | 3.72 (3.55-4.06) | 5.09 (5.00-5.31) | 0/5 | **0.37** (0.35-0.39) | **0.18** (0.16-0.20) | 5/5 | **0.83** (0.76-1.09) | **0.83** (0.72-0.98) | 4/5 |
| 512 | 3.56 (3.43-3.67) | 5.17 (4.96-5.24) | 0/5 | **0.34** (0.30-0.35) | **0.16** (0.15-0.17) | 5/5 | **0.58** (0.51-0.92) | **0.57** (0.53-0.73) | 5/5 |

**Step 2, verdict: D2.** N\* of the quadratic regression is 128, of the MLP 256; the linear
regression has none. **Data need: 128 Dyssol runs, reached by the quadratic regression.**

**Step 3, rank claims** (21 of 28 tests resolved after Holm). MLP against quadratic: all 14 tests
are resolved; the MLP is better at N = 8 to 64 (the quadratic regression has fewer runs than its
66 coefficients), the quadratic regression is better at N = 128, 256 and 512, on both numbers. MLP
against linear: the MLP is better on the exhaust from N = 128 and on the fuel from N = 64
(resolved); the remaining seven tests are not resolved.

**Without a reading.** On the 157 admissible test points the verdict and every N\* are the same
(quadratic at N = 128: 0.45 K and 0.20 %). Largest error over the test set (median over seeds;
largest seed in brackets), line 1 / line 2 and total fuel: quadratic at N = 128 0.67 K (1.01 K) /
1.14 K (1.26 K) and 0.39 % (0.63 %), at N = 512 0.48 K / 0.87 K and 0.25 %; MLP at N = 256 1.31 K /
3.06 K and 1.66 %, at N = 512 1.30 K / 2.30 K and 1.42 %. Other targets of the quadratic regression
at N = 128 (95th percentile, median over seeds), line 1 / line 2: dryer outlet 0.46 / 0.79 K,
preheated air 0.33 / 0.29 K, recovered heat 2.70 / 2.10 kW.

## 13. Both cases together, and caveats

**Answer to the question of section 1.** Without a rebuild, a surrogate of these stationary
flowsheets needs 32 Dyssol runs with one line (5 inputs) and 128 with two lines (10 inputs), and a
quadratic regression reaches both; an MLP needs twice as many (64 and 256). A linear regression
does not reach the thresholds at any N: it settles near 1.9 K and 7.1 % (one line) and 3.6 K and
5.2 % (two lines) at N = 512. Both verdicts are D2.

**Caveats.**

1. **The grid sets the resolution of N\*.** In both cases N\* of the quadratic regression is the
   first N of the grid above its number of coefficients (21 and 66, section 5); below it the
   minimum-norm solution is the worst model of the study. The grid has no N between 16 and 32 and
   none between 64 and 128, so the study does not say how few runs above the number of
   coefficients suffice. The step from 32 to 128 follows the number of coefficients, not the
   number of inputs.
2. **A closed, smooth function.** The three units are closed formulas with constant heat
   capacities (section 3). That a low-order polynomial fits them says nothing about dynamic Dyssol
   models; what carries over is the chain (design, script, run, control, fit, reading).
3. **Over-saturation.** With two lines more than a third of the box (378 of 1024, 99 of 256) gives
   a dryer exhaust above saturation, because the dryer unit has no saturation limit. The surrogate
   reproduces the unit there, not a dryer. The reading on the admissible test points is the same,
   but the training pools contain the flagged points in every case.
4. **MLP optimiser.** The selected setting (learning rate 0.003, weight decay 0.1) lies at the
   edge of the pilot grid in both cases (criterion 1.81 against 1.83 for the next setting with one
   line, 4.01 against 4.37 with two); the grid was not extended, as fixed beforehand. "The MLP
   needs twice as many runs" holds for this grid. At its N\* the MLP meets the thresholds in 4 of
   5 seeds in both cases, and with one line one seed misses them again at N = 256.
5. **95th percentile.** The thresholds are met at the 95th percentile of the test set; the largest
   errors at N\* are above 1 K or 1 % for some models (sections 11 and 12).
6. **The fuel threshold is a setting** without a measured source (section 6).
7. **In-box test set.** No statement about points outside the box.
8. **Dyssol equals the rebuild** to rounding at every point (control). The learning curves are
   therefore the ones the rebuild would have given; the Dyssol runs demonstrate the chain and its
   cost (about 1.4 to 1.6 s and below 60 MB per run), not a different data source.
9. **Seen before the data set:** section 8. All inputs are stored values of the delivery varied in
   a local box; no statement about a plant.

**Open for the project lead.** Acceptance of D2 for both cases and of the decisions taken without
asking (section 9). Questions to the units' author: whether the dryer is to get a saturation
limit; with which state of the units four of the stored flowsheets were saved (section 8); whether
evaluating the vapour enthalpy at the bound of its correlation below 290 K is intended.

## 14. Course of the run (2026-11-19)

* Code, design and reading committed 16:06 (cfefde2), units built on that commit; pilots on the
  rebuild (1fa3b12, a96f84f); selection committed 16:17 (aeedeec).
* The probe of the other study on the machine had finished before (decision 5). Chain started
  detached 16:21 on aeedeec with 2 workers: full suite 1614 passed, 6 skipped (34 min, next to the
  main run of the other study).
* Per case: dry run of the data stage (every point through the rebuild; Dyssol at 21 points with
  one line and 32 with two, none failed), dry run of the training stage (105 cells at 5 optimiser
  steps), data stage (15.9 min and 18.3 min with 2 workers, chain log), copy of the data set to
  `C:/twinlink_runs/dyssol-steady-state/data` (verified), training stage (105 cells, 134 s and
  125 s, peak 364 MB and 365 MB per worker), analysis. No stage was retried, no unit failed,
  nothing was cut. Chain complete 17:44.
* Every artifact was committed with a clean tree after the denylist test (4de73d2, 0924f2a,
  cd2fdd2, f0aa215). No merge, no push.

## 15. Artifacts

| Artifact | Content | Driver |
|---|---|---|
| `artifacts/runs/dyssol_steady_pilot_v1_one_line.json.gz` | pilot of the MLP optimiser on the rebuild | cfefde2 |
| `artifacts/runs/dyssol_steady_pilot_v1_two_lines.json.gz` | the same, two lines | cfefde2 (run on 1fa3b12) |
| `artifacts/runs/dyssol_steady_v1_one_line_training.json.gz` | 105 cells, data summary, control | aeedeec |
| `artifacts/runs/dyssol_steady_v1_one_line_analysis.json` | reading, rank tests, numbers without a reading | run on 4de73d2 |
| `artifacts/runs/dyssol_steady_v1_two_lines_training.json.gz` | 105 cells, data summary, control | aeedeec (run on 0924f2a) |
| `artifacts/runs/dyssol_steady_v1_two_lines_analysis.json` | reading, rank tests, numbers without a reading | run on cd2fdd2 |

Local only (git-ignored; copy in `C:/twinlink_runs/dyssol-steady-state/`): the two data sets, the
box, the compiled units with their manifest, the materials database with the added constant, the
caches and the outputs of the dry runs.
