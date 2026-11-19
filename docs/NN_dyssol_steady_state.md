# NN -- Stationary Dyssol flowsheets as a data source: how many Dyssol runs does a surrogate need?

Status: **design and pre-registered reading, committed BEFORE any Dyssol data set was generated.**
Results are added below (sections 11 and 12) after the runs, numbers only from the artifacts named
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
