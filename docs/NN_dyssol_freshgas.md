# NN -- Dyssol flowsheet with the fresh-gas state as an input: data need of a surrogate with 7 inputs, and the partner sheet as a named test set

Status: **complete. Step 0 held at all 1291 points; pre-registered reading D1 (an MLP is needed,
256 Dyssol runs; the quadratic regression does not reach the thresholds up to 512 runs); the
prediction N\* = 64 is not confirmed; reading on the sheet B1 (sections 11 to 14).** Accepted by
the project lead on 2026-11-24 (section 16). **Add-on on the cause (sections 17 and 18, added
after the fact, rebuild only): its control is not met as committed, reading K is formally not
read; the quadratic regression shows the pattern of K0 (N\* 64 with the medium flow of
Befund 128, none with the full range), and N\* of the MLP is 128 instead of 256 when the targets
differ by rounding.**
Sections 1 to 10 are the design and the reading, committed BEFORE the pilot and before any Dyssol
run of this study (4639bef) and unchanged since; the results below them take their numbers only
from the artifacts named there. Branch `dyssol-freshgas` (from `master` 2e54dde), worktree
`C:/twinlink_work/freshgas`. Document number and finding number are open (assigned by the managing
session at merge). Code `src/twinlink_ml/dyssol_steady/` (extended, not copied), settings
`conf/dyssol_steady/dyssol_freshgas.yaml`, drivers `scripts/run_dyssol_steady.py --config
dyssol_freshgas.yaml` and `scripts/analyze_dyssol_freshgas.py`.

**Everything in this document stands under the reading "the gas mass flow of the sheet is the dry
gas". That reading is inferred from the agreement in Befund 130 and is not confirmed by the
sender.**

## 1. Occasion and questions

Befund 130 (docs/45, section 17) compared the partner sheet of 2026-11-24 (two one-line processes
over eleven flows of the heat transfer medium) with our Dyssol runs and the Python rebuild and
found that the sheet is **no valid test of the surrogates of Befund 128**: the fresh-gas state is
not one of their inputs, and 3 of the 22 points lie in their box (open point 3 of Befund 130).
This study generates a data set in which the fresh-gas state and the full range of the medium
flow are varied inputs.

1. **Does Dyssol keep up with the rebuild at cold fresh gas (step 0)?** Below the lower bound of
   its correlation Dyssol evaluates the vapour enthalpy at that bound (docs/45, section 8), and
   the carrier sweep (docs/49) computed its scans at -10, 0 and 20 degC with the rebuild only
   (caveat 5 there).
2. **How many Dyssol runs does a surrogate of the one-line flowsheet need with 7 instead of 5
   inputs?** The five inputs of docs/45, section 4, plus the fresh-gas temperature and the
   relative humidity of the fresh gas. With a prediction fixed beforehand (section 6).
3. **Does such a surrogate meet the points of the sheet for the admissible process 2?**

Only the case `one_line` (source flowsheet `Case 1.1.dflw`, as in Befund 128 and 130). Process 1
of the sheet is over-saturated at every point (saturation ratio 2.015, Befund 130) and is not part
of this study: **no statement about process 1**.

## 2. Task of the project lead (2026-11-24)

1. **Box:** a new local file that must contain all eleven points of process 2. Medium flow 0 to
   15 000 kg/h; fresh-gas temperature -10 to 30 degC; the humidity NOT as an absolute humidity
   with a cap but sampled as a relative humidity of 0.2 to 0.95 from which the humidity ratio is
   computed (as in `case_ranking/space.py`); the points of the sheet must lie inside in this
   parametrisation. The other inputs as in the box of Befund 128 as far as the points of the
   sheet lie in it.
2. **Design, models, thresholds and reading as docs/45, sections 4 to 6:** pool of 1024 and test
   set of 256 Sobol points with their own seeds, N = 8 ... 512, seeds 1-5, linear / quadratic /
   MLP, thresholds 1 K and 1 %, reading D1 / D2 / D3 with feasibility first, rank claims only
   with the paired resolution and Holm.
3. **MLP pilot on the rebuild before the data**, with a grid widened against Befund 128 (learning
   rate, width, steps: at least one more level on both sides each); if the selection lies at an
   edge, the grid is extended once automatically in that direction. Grid and rule committed
   before the pilot.
4. **Step 0** at every point with the bounds of docs/45; the deviation tabulated over the
   fresh-gas temperature. Above the bounds: the data set is not used, the deviation is reported,
   the study stops and asks.
5. **Second, named test set** from the sheet with its own reading (section 6), fixed in the
   script before the first run; disclosure that the sheet is not a blind test.
6. **Over-saturated points** flagged and counted; the reading also on the admissible points
   (without a reading of its own).
7. **Compute:** dry run over all units, probe with full-size fits, at most 4 workers, main run
   detached, saved per unit, resumable, outputs as `.json.gz`; **4 h wall clock for the main run**.
8. **Confidentiality:** sheet, box, input values and data sets stay local; committed artifacts
   hold errors in physical units, counts and digests.
9. A branch is delivered; no merge, no push.

## 3. Placing

The Python rebuild of the three units is exact and available (Befund 128: at most 1.7e-13 K at
1280 points per case; Befund 130: also at the fresh-gas state of the sheet and without medium
flow). **This study does not ask whether a learned model beats the balance** (Befund 120,
docs/37): wherever the rebuild exists it is the model to use. The study measures the data need of
a surrogate for the case that no rebuild exists, and it tests the rule of thumb that Befund 128
suggests (a quadratic regression needs about 1.5 to 2 times as many runs as it has coefficients).
The stationary case remains the rehearsal for dynamic Dyssol models (docs/45, section 1).

## 4. Data

**Varied inputs, seven:** dry-gas flow, dryer inlet temperature, spray flow, solid fraction of the
spray, medium flow of the heat recovery (the five of docs/45, section 4); **fresh-gas temperature
and relative humidity of the fresh gas** (new). The humidity ratio of a point follows from the
last two with Buck's saturation pressure over water at 1 atm
(`case_ranking.space.ambient_humidity_kg_per_kg`), so no point of the box holds fresh gas above
saturation. Spray temperature, equilibrium moisture, heater efficiency and calorific value stay
at the values stored in the source file. The surrogates see the seven inputs scaled to [-1, 1] by
the box; the relative humidity is one of them (not the humidity ratio).

**Box:** `<local>/dyssol_steady/box_freshgas_local.yaml` (template
`conf/dyssol_steady/box_freshgas_local.yaml.example`, placeholders only). Records carry the
SHA-256 of the resolved box. Set by the task and therefore stated here: medium flow 0 to
15 000 kg/h (the scan range of docs/49), fresh-gas temperature -10 to 30 degC, relative humidity
0.2 to 0.95. Dry-gas flow, dryer inlet temperature, spray flow and solid fraction are the
intervals of the box of Befund 128 (local).

**Design:** training pool of 1024 scrambled Sobol points and an independent test set of 256
scrambled Sobol points in the box (new seeds, in the YAML). **Test set name: in-box Sobol test
set, 7 inputs.** Learning curve N = 8, 16, 32, 64, 128, 256, 512; per seed (1-5) the first N
points of one permutation of the pool (nested over N); the seed also initialises the MLP.

**Second test set: "partner sheet 2026-11-24, process 2, 11 points".** The inputs of process 2 of
the sheet at its eleven medium flows, read with `dyssol_steady.sweep_sheet` (gas flow = dry gas)
and expressed as design rows (the humidity as a relative humidity at the sheet's fresh-gas
temperature). The drivers abort if a point lies outside the box, if an input that is not varied
differs from the stored flowsheet, or if a design point of the pool or of the Sobol test set
coincides with a point of the sheet. Every point of the sheet gets its own Dyssol run; **the truth
is our Dyssol run at that point**, not the sheet's value.

**Over-saturation.** A point is flagged as in docs/45 when the dryer exhaust holds more water than
saturated air at its temperature (1 atm); flagged points stay in every set and are counted. This
flag defines the "admissible test points". In addition the stricter rule of docs/49 is counted
(also the exhaust after the heat recovery, which gets colder with cold fresh gas and a large
medium flow).

**Methodology note.** As in docs/45: no grey-box data are generated, the preset
`DataSpec.new_study` does not apply.

## 5. Models and pilot

Models as docs/45, section 5: `linear` and `quadratic` (ordinary least squares on all monomials up
to degree 1 / 2; **8 / 36 coefficients at 7 inputs**; minimum-norm solution below that number of
runs) and `mlp` (two tanh layers, float64, full-batch Adam on standardised targets, optional
weight decay). Five targets: exhaust temperature after the heat recovery, dryer outlet
temperature, preheated-air temperature, recovered heat, fuel.

**Pilot of the MLP optimiser** on the Python rebuild (no Dyssol run), with its own pool, test
points and seeds (11, 12) at N = 16, 64, 256; the pilot stage does not load the sheet. Criterion
and selection as docs/45 (lowest mean over pilot seeds and budgets of the mean of both reading
numbers, each divided by its threshold; ties to the setting listed first).

**Grid (full factorial, 108 settings):**

| Axis | Befund 128 | This study | Level added if the selection lies at the lower / upper edge |
|---|---|---|---|
| learning rate | 0.001, 0.003 | 0.0003, 0.001, 0.003, 0.01 | 0.0001 / 0.03 |
| width of the two hidden layers | 64 | 32, 64, 128 | 16 / 256 |
| optimiser steps | 5000 | 2000, 5000, 10000 | 1000 / 20000 |
| weight decay | 0, 0.001, 0.1 | 0, 0.001, 0.1 | none (first level is zero) / 1.0 |

**Extension rule (fixed with this commit, before the pilot):** if the selected setting lies at the
first or the last level of an axis, the level of the last column is added to that axis, **once**,
and the full factorial of the extended axes is run (only the new settings need fits). The
selection is then made over all settings of the extended grid. There is no second extension; a
final selection at an edge is reported as a caveat. The selection is committed into
`mlp.selected.one_line` of the YAML before the data stage, which refuses to start without it.

## 6. Pre-registered reading

Also in the docstring of `scripts/analyze_dyssol_freshgas.py`, fixed before the first run.

**Step 0, control (before any fit).** Dyssol against the rebuild at EVERY point (pool, Sobol test
set, points of the sheet): 1e-6 K on temperatures, 1e-6 % on heat flows and fuel relative to
max(|value|, 1), as docs/45. Tabulated over the fresh-gas temperature in bins of 5 K, and given
for the points whose heater inlet is colder than the lower bound of the vapour-enthalpy
correlation. **Above the bounds the data set is not used, the deviation is reported and the study
stops with a question to the project lead** (the training stage refuses to fit).

**Reading D (data need), docs/45 section 6 unchanged.** Per cell (model, N, seed), on the in-box
Sobol test set, 7 inputs: the 95th percentile of the absolute error of the exhaust temperature
after the heat recovery [K] and of the relative error of the fuel [%]. A model meets the
thresholds (1 K, 1 %) at N if the median over the five seeds of both numbers is at or below its
threshold; N\* = the smallest N of the grid from which on it meets them at every larger N.

1. Feasibility first: per model and N the number of seeds (of five) meeting both thresholds.
2. **D1** the MLP has an N\* and no baseline has one at or below it; **D2** a baseline has an N\*
   at or below that of the MLP, or the MLP has none (data need = the smallest baseline N\*, the
   simpler model at equal N\*); **D3** no model has an N\*.
3. Rank claims (MLP against each baseline, per N and number): paired test over the seeds
   (`evaluation/paired_resolution.py`), Holm per family (one baseline, one number, all seven N).
   The verdict does not depend on them.

**Prediction (fixed before the data).** In Befund 128, N\* of the quadratic regression was 32 at
21 coefficients and 128 at 66, a data need of about 1.5 to 2 times the number of coefficients.
With seven inputs the quadratic regression has 36 coefficients. **Predicted: N\* = 64** (the first
step of the grid above 1.5 x 36 = 54). The result is reported as **"prediction confirmed"** if
N\* of the quadratic regression is 64 and as **"prediction not confirmed"** otherwise.

**Reading B (the sheet).** Test set: partner sheet 2026-11-24, process 2, 11 points; truth = our
Dyssol run at each point. For the model of reading D at its N\* (the model that reaches the data
need), per seed the LARGEST error over the eleven points of the exhaust temperature [K] and of the
fuel [%]:

* **B0** -- the median over the seeds of both largest errors is at or below its threshold (1 K
  and 1 %): the surrogate holds the thresholds at all eleven points;
* **B1** -- otherwise.

With D3 there is no model of the reading and B is "not applicable".

**Reported without a reading:** reading D on the admissible test points and on the points
admissible also after the heat recovery; counts of over-saturated points per set; median and
maximum error next to the 95th percentile; the other targets; the largest errors on the sheet for
every model and N; the surrogate of reading B against the sheet's own values (number of reported
values within half a unit / one unit of the last reported decimal, as
`sweep_sheet.compare` in Befund 130) next to the same counts for our Dyssol runs; where the points
of the sheet lie in the box.

## 7. Stages and compute rules

`pilot` -> commit of the selection -> full suite -> `data --dry-run` -> `training --dry-run` ->
`data` -> copy -> `training` -> `analyze`.

* Units of the data stage are the points (1024 + 256 + 11 = 1291 Dyssol runs), each appended to a
  cache at once; `--resume` continues; a failing point is recorded and does not stop the others.
  Units of the training stage are the 105 cells (model x N x seed), cached the same way.
* **Dry run:** every point through the rebuild; Dyssol at the points with the extreme rebuild
  results, at the extreme draws of every input, at the first points of pool and test set and at
  all points of the sheet (time and peak memory per Dyssol process); every cell of the training
  stage with 5 optimiser steps on that data set. **Probe with full-size fits:** in the dry run of
  the training stage the largest cell (N = 512) of every model with the full selected optimiser
  setting; its time and peak memory per worker are the basis of the worker count. Outputs of dry
  runs are never results and stay local.
* At most 4 workers (8 physical cores). Main run detached, output to log files.
* **Budget: 4 h wall clock for the main run.** Estimate before the main run from Befund 128
  (median 1.43 s per Dyssol run): about 10 minutes for the data stage with 4 workers and minutes
  for the training stage. No cut is planned; the dry run and the probe confirm or correct this
  before the main run.
* Data sets and caches: `<worktree>/local/runs/dyssol_steady/` (git-ignored), copied to
  `C:/twinlink_runs/dyssol-freshgas/`. Compiled units: those built for Befund 128
  (`C:/twinlink_runs/dyssol-steady-state/units`, checked against their manifest); the materials
  database is patched as in Befund 128 (digests in the records).

## 8. Seen before the data (disclosure)

* **The sheet is not a blind test.** The values of the sheet, our Dyssol runs and the rebuild at
  its 22 points, and their agreement (Befund 130) are known, as are the scans of the carrier
  sweep (docs/49) at the inputs of process 2. The points of the sheet enter neither the pool nor
  the Sobol test set nor the pilot.
* The results of Befund 128 with five inputs (D2, N\* of 32 and 64), from which the prediction is
  derived.
* Probes without an artifact in the session that wrote this study (2026-11-24): the box of
  Befund 128, the stored point and the inputs of both processes of the sheet (to set the box);
  the position of the eleven points in the new box (next section, item 1); the rebuild at 1024
  Sobol points of the new box with a seed that is no seed of the study (ranges of exhaust
  temperature and fuel; 5 points over-saturated by the flag of docs/45 and 25 by the stricter
  rule; 145 points with a heater inlet below 290 K); five MLP fits on these rebuild points for
  their run time only (errors not looked at).
* No Dyssol run in the new box, no pilot result, no learning curve at the time of this commit.

## 9. Decisions taken without asking

1. **Box of the other four inputs unchanged.** All eleven points of the sheet lie in the box of
   Befund 128 for dry-gas flow, dryer inlet temperature, spray flow and solid fraction, so it was
   not widened (rule of the task). **But they lie on its boundary:** all eleven on the face of
   the dryer inlet temperature (the sheet's value is a bound of the box), and one of them also on
   the face "no medium flow" (the task's lower bound). In the fresh-gas temperature and the
   relative humidity all eleven lie strictly inside, as the task demands. The second test set is
   therefore a test on a face of the box, where a regression is usually at its worst; widening
   the box is a decision of the project lead (caveat of the results).
2. **Humidity:** Buck over water at 1 atm, also below 0 degC (the functions of
   `case_ranking/space.py`, as the task names them), not the Magnus form over ice of the carrier
   sweep. The control and the surrogates do not depend on this choice; the position of the
   sheet's points in the humidity axis does.
3. **Weight decay** keeps the three levels of Befund 128 and gets an upper extension level only;
   the extension levels of the other axes continue their spacing.
4. **Pilot budgets, seeds and criterion** as docs/45; new Sobol seeds for all four designs.
5. **The prediction is checked on N\* of the quadratic regression** (not on the data need of
   reading D, which is reported next to it).
6. **Statistic of reading B:** per seed the largest error over the eleven points, median over the
   seeds (the task: "maximum, median over seeds"); with D3 "not applicable".
7. **"Admissible"** is the flag of docs/45 (dryer exhaust); the stricter rule of docs/49 is
   counted and reported next to it.
8. **Control table** in bins of 5 K, plus the points with a heater inlet below the lower bound of
   the vapour correlation.
9. **Dry run** with Dyssol at selected points as in docs/45 (extreme results, extreme inputs,
   first points, all points of the sheet), not at all 1291; the probe is the largest cell per
   model with the full optimiser setting.
10. **A new analysis script** instead of a change to `scripts/analyze_dyssol_steady.py`, whose
    docstring is the pre-registered reading of Befund 128.
11. **The three ranges named by the task are stated in this document** (the medium range is in
    docs/49; fresh-gas temperature and relative humidity are weather settings, no plant values);
    everything else of the box stays local.
12. **Truth at the points of the sheet:** Dyssol is run at the line inputs that follow from the
    design row (the humidity through the relative humidity and back; equal to the sheet's inputs
    to a relative 1e-12, checked by the driver), so that truth and surrogate see the same point.

## 10. Confidentiality

Not in the repository: the delivery, the sheet and its values, the box, the stored inputs, the
data sets, the names of the processes. The committed template of the box holds placeholders.
Committed artifacts hold errors in physical units, counts and digests, no input value and no
value of the sheet. `tests/test_denylist.py` is run before every commit of this branch.

## 11. Results

Artifacts: pilot `artifacts/runs/dyssol_freshgas_pilot_v1.json.gz` (driver 4639bef, clean tree),
training `artifacts/runs/dyssol_freshgas_v1_training.json.gz` (driver 6e0bfc2, clean tree),
analysis `artifacts/runs/dyssol_freshgas_v1_analysis.json` (run on 5f4af7e, which differs from
6e0bfc2 only by the training artifact). The data set `dyssol_freshgas_v1.json.gz` is local; its
SHA-256 is in the training artifact. **All of this section is conditional on the reading "gas mass
flow of the sheet = dry gas" (inferred, not confirmed by the sender).**

### 11.1 Pilot

1440 fits on the rebuild. Round 1 (108 settings) selected width 64, learning rate 0.0003, 10000
steps, weight decay 0.1: at the edge of three axes (learning rate lower, steps upper, weight decay
upper). By the committed rule the grid was extended once on these three axes (240 settings).
**Final selection: width 128, learning rate 0.0003, 20000 steps, weight decay 0.1** (criterion
4.09, against 4.12 for the selection of round 1 and 4.33 for the setting of Befund 128, which is
part of the grid). The final selection lies at the upper edge of the width and of the steps in the
extended grid; there is no second extension (caveat 4).

### 11.2 Data

1024 + 256 + 11 = 1291 Dyssol runs, none failed. Dyssol per run: median 1.33 s, maximum 1.84 s,
1728 s in total, peak 54 MB per process. Units, patched materials database and source flowsheet
have the digests of the training artifact of Befund 128 (`dyssol_steady_v1_one_line_training`).

Over-saturated (flag of docs/45, dryer exhaust): 9 of 1024 points of the pool, 0 of 256 of the
test set, 0 of 11 of the sheet. With the stricter rule of docs/49 (also after the heat recovery):
26 of 1024, 5 of 256, 0 of 11.

The eleven points of the sheet: all inside the box; all strictly inside in the fresh-gas
temperature and the relative humidity; **all eleven on the face of the dryer inlet temperature,
one of them also on the face of the medium flow** (no flow); none on a face of the other inputs.

### 11.3 Step 0, control: held

Dyssol against the rebuild at all 1291 points: largest temperature deviation 1.7e-13 K, largest
relative deviation of heat flows and fuel 1.0e-11 % (bounds 1e-6). Per set: pool 1.7e-13 K and
1.0e-11 %, Sobol test set 1.7e-13 K and 1.2e-12 %, sheet 5.7e-14 K and 2.2e-13 %.

| Fresh-gas temperature [degC] | Points | Largest temperature deviation [K] | Largest relative deviation [%] |
|---|---:|---:|---:|
| -10 to -5 | 160 | 1.7e-13 | 1.0e-12 |
| -5 to 0 | 160 | 1.7e-13 | 9.9e-13 |
| 0 to 5 | 160 | 1.7e-13 | 1.0e-12 |
| 5 to 10 | 160 | 1.1e-13 | 1.2e-12 |
| 10 to 15 | 171 | 1.7e-13 | 1.2e-12 |
| 15 to 20 | 160 | 1.7e-13 | 1.2e-12 |
| 20 to 25 | 160 | 1.1e-13 | 1.5e-12 |
| 25 to 30 | 160 | 1.1e-13 | 1.0e-11 |

At 177 of the 1291 points the heater inlet is colder than 290 K, the lower bound of the
vapour-enthalpy correlation; there the deviation is at most 1.7e-13 K and 9.9e-13 %.

**Answer to question 1: Dyssol keeps up with the rebuild at cold fresh gas.** The deviation does
not depend on the fresh-gas temperature; it is rounding in every bin. The rebuild's rule (the
vapour enthalpy evaluated at the bound of its correlation) reproduces Dyssol down to -10 degC.
This closes caveat 5 of docs/49 for the range of this box: the scans of the carrier sweep at -10
and 0 degC used a rebuild that agrees with Dyssol there. Whether evaluating the correlation at its
bound is intended remains the question to the units' author (docs/45, section 16, item 5).

### 11.4 Reading D: D1

Median over the five seeds (range); bold = both medians at or below their threshold; "seeds" =
seeds meeting both thresholds. Test set: in-box Sobol test set, 7 inputs (all 256 points).

| N | linear: exhaust P95 [K] | fuel P95 [%] | seeds | quadratic: exhaust P95 [K] | fuel P95 [%] | seeds | MLP: exhaust P95 [K] | fuel P95 [%] | seeds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 35.03 (17.02-328.78) | 41.61 (14.65-642.43) | 0/5 | 45.45 (26.59-66.94) | 53.34 (38.19-66.73) | 0/5 | 15.66 (14.45-34.66) | 20.21 (19.21-39.80) | 0/5 |
| 16 | 11.98 (8.66-14.51) | 20.04 (12.44-25.74) | 0/5 | 27.10 (18.41-42.93) | 48.14 (34.03-55.85) | 0/5 | 10.05 (7.59-12.96) | 12.10 (9.35-18.06) | 0/5 |
| 32 | 9.73 (7.16-11.22) | 12.57 (11.56-21.93) | 0/5 | 13.27 (11.54-22.02) | 19.50 (18.92-27.19) | 0/5 | 5.41 (3.54-6.15) | 5.67 (4.70-6.95) | 0/5 |
| 64 | 7.65 (7.08-8.33) | 12.25 (11.49-16.71) | 0/5 | 4.38 (3.87-5.48) | 5.53 (4.97-5.72) | 0/5 | 2.26 (1.76-3.20) | 2.51 (2.23-2.60) | 0/5 |
| 128 | 7.42 (7.11-8.44) | 11.85 (10.65-12.61) | 0/5 | 3.08 (2.96-3.17) | 3.98 (3.61-4.44) | 0/5 | 0.94 (0.80-1.03) | 1.04 (0.97-1.49) | 2/5 |
| 256 | 7.33 (7.21-7.89) | 11.35 (10.85-11.80) | 0/5 | 2.89 (2.69-2.92) | 3.76 (3.28-4.18) | 0/5 | **0.47** (0.38-0.58) | **0.64** (0.61-0.70) | 5/5 |
| 512 | 7.63 (7.19-7.66) | 10.97 (10.86-11.52) | 0/5 | 2.52 (2.45-2.74) | 4.05 (3.37-4.41) | 0/5 | **0.27** (0.19-0.35) | **0.41** (0.37-0.49) | 5/5 |

**Feasibility:** only the MLP meets the thresholds, from N = 256 (5 of 5 seeds at 256 and 512; at
N = 128 2 of 5 seeds, and the median of the fuel misses with 1.04 %). No seed of a regression
meets them at any N.

**Verdict: D1.** N\* of the MLP is 256; the quadratic and the linear regression have none.
**Data need: 256 Dyssol runs, reached by the MLP.** With five inputs and the narrow medium range
(Befund 128) the verdict was D2 with 32 runs.

**Rank claims** (24 of 28 tests resolved after Holm). MLP against quadratic: the MLP is better at
every N on both numbers, all 14 tests resolved. MLP against linear: the MLP is better on both
numbers from N = 32 (10 tests resolved); the four tests at N = 8 and 16 are not resolved.

### 11.5 Prediction: not confirmed

Predicted N\* of the quadratic regression: 64. Found: **none up to N = 512.** **Prediction not
confirmed.** The quadratic regression does not fail for lack of runs: from N = 128 to N = 512 its
numbers stay at about 2.5 to 3.1 K and 3.8 to 4.1 %, with a narrow range over the seeds. It has
reached the error of the best quadratic function of this box, and that error is above the
thresholds. The rule of thumb of Befund 128 (data need about 1.5 to 2 times the number of
coefficients) presupposes that a polynomial of degree 2 can represent the flowsheet over the box;
that held in the box of Befund 128 and does not hold in this one.

### 11.6 Reading B: B1

Test set: partner sheet 2026-11-24, process 2, 11 points; truth = our Dyssol run at each point.
Model of reading D: the MLP at N = 256. Largest error over the eleven points, median over the
seeds (range): **exhaust temperature 2.27 K (1.94-4.41), fuel 1.32 % (0.91-2.26)**; 0 of 5 seeds
meet both thresholds. **Verdict: B1** -- the surrogate does not hold 1 K and 1 % at all eleven
points.

**Answer to question 3: no, not at all points.** Not a blind test (section 8), and a test on a
face of the box (section 11.2, caveat 1).

### 11.7 Without a reading

* **Admissible test points.** No test point is over-saturated by the flag of docs/45, so the
  reading on the admissible points is reading D itself. With the stricter rule 251 of 256 test
  points are admissible; on them verdict and N\* are the same (MLP at N = 256: 0.47 K and
  0.64 %, 5 of 5 seeds; quadratic at N = 512: 2.51 K and 3.83 %).
* **Largest error over the Sobol test set** (median over seeds; largest seed in brackets): MLP at
  N = 256 4.74 K (5.18 K) and 2.70 % (3.31 %), at N = 512 3.29 K (4.01 K) and 1.67 % (2.64 %);
  quadratic at N = 512 9.53 K (9.83 K) and 5.81 % (6.16 %). Median error: MLP at N = 256 0.08 K
  and 0.12 %; quadratic at N = 512 0.88 K and 1.12 %. The thresholds are met at the 95th
  percentile, not at every test point.
* **Other targets** (95th percentile, median over seeds). MLP at N = 256: dryer outlet 0.32 K,
  preheated air 0.45 K, recovered heat 3.73 kW. Quadratic at N = 512: dryer outlet 0.33 K,
  preheated air 2.63 K, recovered heat 25.85 kW. The quadratic regression fits the dryer as well
  as the MLP and misses the two quantities of the heat recovery.
* **The sheet, every model and N.** No model meets the rule of reading B at any N. MLP at
  N = 512: 1.66 K (1.41-3.09) and 1.16 % (0.78-1.28), 0 of 5 seeds; quadratic at N = 512: 9.77 K
  and 5.09 %; linear at N = 512: 17.91 K and 11.15 %.
* **Where on the sheet** (MLP at N = 256; per point the median over the seeds, computed from the
  per-point errors stored in the training artifact). 8 of the 11 points are within both
  thresholds. The other three are the first three in the order of the sheet, the point without
  medium flow and the two smallest flows: 2.06, 1.16 and 1.06 K, and 0.63, 1.18 and 1.00 %. From
  the fifth point on the errors are at most 0.36 K and 0.29 %. In 4 of 5 seeds the largest
  exhaust error is at the point without medium flow, which lies on two faces of the box.
* **The surrogate against the sheet's own values** (MLP at N = 256; 64 reported values). Within
  half a unit of the last reported decimal: 2, 2, 11, 4 and 6 values in the five seeds; within
  one unit: 3, 6, 16, 9 and 18. Our Dyssol runs at the same points: 48 and 62 of 64 (the counts
  of the addendum to Befund 130 for process 2). A surrogate that is read against 1 K and 1 % does
  not reproduce a sheet reported to one decimal; Dyssol and the rebuild do.

## 12. Answers and caveats

**Question 1.** Dyssol and the rebuild agree to rounding at all 1291 points, at every fresh-gas
temperature from -10 to 30 degC and at the 177 points with a heater inlet below 290 K.

**Question 2.** With seven inputs and the full range of the medium flow a surrogate needs 256
Dyssol runs, and it has to be an MLP (D1). The quadratic regression, which sufficed with 32 runs
in Befund 128, settles at about 2.5 K and 4.0 % and never reaches the thresholds. The prediction
N\* = 64 is not confirmed.

**Question 3.** The MLP at its N\* holds the thresholds at 8 of the 11 points of the sheet in the
median over the seeds and misses them at the point without medium flow and at the two smallest
flows (B1).

**Placing.** The rebuild is exact at every point of this study, including the points of the
sheet. Nothing here is a reason to use a surrogate where the rebuild exists; the result is the
data need for the case without one, and the limit of the rule of thumb.

**Caveats.**

1. **The points of the sheet lie on the boundary of the box** (all on the face of the dryer inlet
   temperature, one also without medium flow; section 9, item 1). B1 is a statement about the
   surrogate at the edge of its box. The largest errors are at the corner. Whether the surrogate
   would hold the thresholds at these points inside a wider box is not tested.
2. **The sheet is not a blind test** (section 8).
3. **"Gas mass flow of the sheet = dry gas"** is inferred, not confirmed by the sender; reading B
   and every count against the sheet depend on it.
4. **MLP optimiser.** The final selection lies at the upper edge of width and steps of the
   extended grid (no second extension, as fixed beforehand); the criterion differs little between
   the best settings (4.09, 4.12, 4.13). What the study says about the MLP (N\* = 256, B1) holds
   for this grid.
5. **The grid of N sets the resolution of N\*.** At N = 128 the MLP misses the thresholds
   narrowly (0.94 K, 1.04 %; 2 of 5 seeds); the study has no N between 128 and 256.
6. **Why the quadratic regression fails is not separated.** Two things changed against Befund 128
   at once: the fresh-gas state became an input, and the medium flow went from +-25 % around the
   stored point to the full range from no flow. The other targets (section 11.7) point to the
   heat recovery, whose outlet temperatures saturate with the medium flow; this is a hypothesis,
   not a result of this study.
7. **95th percentile.** The largest errors of the MLP at N\* are 4.74 K and 2.70 %.
8. **Closed, smooth formulas** with constant heat capacities; no statement about dynamic Dyssol
   models. What carries over is the chain and the observation that a rule of thumb for the data
   need holds only as long as the model class fits the box.
9. **Over-saturation.** Few points (9 of 1024 in the pool, none in the test set; 26 and 5 with
   the stricter rule); the units have no saturation limit and no condensation.
10. **The 1 % on the fuel is a setting** (docs/45, section 6). **In-box test set:** no statement
    outside the box.
11. **Dyssol equals the rebuild** at every point: the learning curves are those of the rebuild.

**Relation to existing findings (reported, nothing changed).** No number of Befund 128 or 130
changes. The supplement to Befund 128 states that "the data need follows the number of
coefficients"; this study shows that this holds in the box of Befund 128 and not in a box with
the full range of the medium flow and the fresh-gas state. Whether Befund 128 gets a scope note is
a decision of the project lead. The Dyssol counts against the sheet (48 and 62 of 64) are those of
the addendum to Befund 130.

## 13. Open for the project lead

1. Acceptance of D1, of "prediction not confirmed" and of B1, and of the twelve decisions taken
   without asking (section 9).
2. Whether Befund 128 gets a scope note on the rule of thumb (section 12).
3. Whether the box is widened in the dryer inlet temperature so that the points of the sheet lie
   inside, and the study repeated under a new name (about 15 minutes of Dyssol runs). The point
   without medium flow stays on a face in any case.
4. Whether the cause of the quadratic regression's failure is to be separated (descriptive, with
   the rebuild alone: seven inputs with the narrow medium range; five inputs with the full one).
5. Whether the pilot grid is to be extended a second time (width and steps upwards).
6. Whether the three ranges of the task may stay in this document (section 9, item 11).
7. The sender's confirmation of the reading "dry gas" is still open (Befund 130).

## 14. Course of the run (2026-11-24)

* Code, design, reading, grid and extension rule committed 14:57 (4639bef). Pilot detached
  14:57 to 15:38 with 4 workers (1440 fits, none failed); next to it the full suite on 4639bef:
  1801 passed, 6 skipped (49 min). Selection and pilot artifact committed (6e0bfc2).
* Dry runs detached 15:49 to 15:53 on 6e0bfc2: every point through the rebuild; Dyssol at 49
  points (extreme results and inputs, first points, all points of the sheet), none failed,
  median 1.32 s and 53 MB per process; 105 cells with 5 optimiser steps; probe with full-size
  fits: the MLP at N = 512 takes 58 s and 366 MB per worker (4 workers: 1.5 GB of 31 GB).
  The dry run showed the control at its 49 points (at most 1.1e-13 K) before the main run.
* **Main run** detached 15:54 to 16:07 on 6e0bfc2 with 4 workers: **12 min 50 s of the 4 h
  budget.** Data stage 439 s (1291 runs), verified copy to `C:/twinlink_runs/dyssol-freshgas/data`,
  training stage 291 s (105 cells, peak 371 MB per worker). No stage was retried, no unit failed,
  nothing was cut.
* Artifacts committed with a clean tree after the denylist test (6e0bfc2, 5f4af7e, dbd7434).
  No merge, no push.

## 15. Artifacts

| Artifact | Content | Driver |
|---|---|---|
| `artifacts/runs/dyssol_freshgas_pilot_v1.json.gz` | pilot of the MLP optimiser on the rebuild, both rounds, extension record | 4639bef |
| `artifacts/runs/dyssol_freshgas_v1_training.json.gz` | 105 cells (Sobol test set and sheet), data summary, control with its table | 6e0bfc2 |
| `artifacts/runs/dyssol_freshgas_v1_analysis.json` | readings D and B, prediction, rank tests, numbers without a reading | run on 5f4af7e |

Local only (git-ignored): the data set and the caches (`<worktree>/local/runs/dyssol_steady/`,
copy in `C:/twinlink_runs/dyssol-freshgas/`), the box, the outputs of the dry runs, the report
with the values that must not enter the repository (`local/reports/dyssol_freshgas_bericht.md`).

## 16. Decisions of the project lead (2026-11-24; added after the fact, sections 1 to 15 are unchanged)

1. **D1, "prediction N\* = 64 not confirmed" and B1 are accepted.** The twelve decisions taken
   without asking (section 9) are confirmed. The three ranges of the task (medium flow, fresh-gas
   temperature, relative humidity) may stay in this document; they are placeholders of the
   project lead. **Restriction named with the verdict:** the selected MLP setting lies at the
   edge of the extended pilot grid (caveat 4 of section 12); what the study says about the MLP
   (N\* = 256, B1) holds for this grid. The grid is not extended a second time.
2. **No repetition with a widened box.** B1 stays a statement at the edge of the box (caveat 1).
3. **The cause is to be separated, with the rebuild only** (no Dyssol run: step 0 found the
   rebuild exact in this box), as an add-on marked as added after the fact, with its reading
   committed before the run: section 17.
4. **Scope note to Befund 128** (a dated addendum, append-only, entered by the managing session
   at the merge, not on this branch): "The rule of thumb (data need about 1.5 to 2 times the
   number of coefficients of the quadratic regression) holds in the box of Befund 128; with the
   full range of the medium flow the quadratic regression does not reach the thresholds up to
   N = 512 (new finding), the cause is clarified by the add-on." After the add-on the note is
   supplemented by its reading.

## 17. Add-on: medium-flow range or fresh-gas inputs? (added after the fact; design and reading committed before its run)

**Status of this section:** written after the results of sections 11 to 13 were known, on the
decision of section 16, item 3. The design and the reading below were committed before the first
run of `scripts/run_dyssol_freshgas_cause.py` (whose docstring holds the same reading); the
results follow in section 18.

**Question.** Against Befund 128 two things changed at once (caveat 6 of section 12): the
fresh-gas state became an input, and the medium flow went from the interval of Befund 128 to the
full range from no flow. Which of the two keeps the quadratic regression from the thresholds?

**No Dyssol run.** Every variant box lies inside the box of the main run, where step 0 found
Dyssol and the rebuild equal to rounding at all 1291 points. All data of the add-on come from the
rebuild.

**Variants**, each with the design of the main run (pool of 1024 and test set of 256 scrambled
Sobol points with the seeds of the main run, N = 8 ... 512, seeds 1-5, thresholds 1 K and 1 %);
only the quadratic regression and the MLP with the setting selected for the main run:

| Variant | Inputs | Medium flow | Coefficients of the quadratic regression |
|---|---|---|---:|
| V-a | 7, as in the main run | interval of the box of Befund 128 | 36 |
| V-b | 5, as in Befund 128 (fresh gas fixed at the stored state) | full range of the main run | 21 |
| V-c (control) | 7, as in the main run | full range of the main run | 36 |

Test set of a variant: "in-box Sobol test set, variant V-x".

**Reading (fixed before the run).** Per variant and model the rule of section 6; "reaches the
thresholds" = has an N\*.

* **Control first:** V-c must reproduce the main run (the same box and seeds; N\* of the quadratic
  regression none and of the MLP 256). Otherwise the add-on is not read.
* **K0** -- the quadratic regression reaches the thresholds in V-a and not in V-b: the cause is
  the range of the medium flow, not the additional inputs.
* **K1** -- the reverse: the cause is the fresh-gas inputs.
* **K2** -- otherwise.

**Hypothesis of the project lead:** the recovery of an ideal counter-current exchanger has a kink
where the capacity rates of both sides are equal (Q = min(C_hot, C_cold) x dT_max); with the full
range of the medium flow the kink lies in the box, in the box of Befund 128 it does not.

**Known before the run, from the equations of the unit** (no number of the add-on had been
computed): the heat recovery of the delivered units is not that exchanger. By equations
4.1 / 4.2 (`reference.heat_recovery`; section 3 of docs/45) the recovered heat is

    Q = (T_hot - T_cold) / (1/A + 1/B + 1/C)

with the capacity rates A of the dryer exhaust, B of the medium and C of the fresh gas: a smooth
saturation in the medium flow (Q goes to zero with B, and to a limit for large B), **without a
kink** (a test checks the formula against the unit and its strict concavity). Reading K does not
depend on the mechanism; it separates the range of the medium flow from the additional inputs. A
verdict K0 would confirm the cause that the hypothesis names (the range of the medium flow), not
its mechanism.

**Reported without a reading:** per variant the share of pool and test points on each side of
capacity ratio 1 (B / min(A, C): the medium below, or at or above, the smaller gas stream) and in
the bins 0-0.25, 0.25-0.5, 0.5-1, 1-2, above 2; the reading numbers of the quadratic regression
(and of the MLP) per side and per bin; N\* and learning curve of the MLP per variant;
over-saturated points per variant.

**Disclosure.** Known before the add-on: all results of the main run, the equations of the unit,
rough sizes of the capacity rates from the local box. No fit on a variant box had been made.

## 18. Add-on: results (added after the fact)

Artifact `artifacts/runs/dyssol_freshgas_cause_v1.json.gz` (driver a069204, clean tree; 210
cells; Python rebuild, no Dyssol run). Detached 2026-11-24 17:07 to 17:23 with 4 workers: dry run
over all 210 cells (5 optimiser steps), then the run (704 s, peak 370 MB per worker); no unit
failed, nothing was retried. The dry run already showed the numbers of the quadratic regression
(a regression has no optimiser steps); the reading had been committed before it.

### 18.1 Control: NOT met as committed -- reading K is formally not read

V-c has the box and the seeds of the main run. **The quadratic regression reproduces the main
run:** its reading numbers differ from those of `dyssol_freshgas_v1_training` by at most 1.4e-13 K
and 3.1e-13 % in all 35 cells, and it has no N\*, as in the main run. **The MLP does not:** its
reading numbers differ by up to 0.10 K and 0.13 %, and its N\* is 128 instead of 256. The rule
committed in section 17 asks for the N\* of both models; it is not met, so the artifact's verdict
is "not read".

**Why the MLP differs** (probe without an artifact, 2026-11-24): the fit is deterministic (two
fits on the same data are identical), and with the Dyssol results of the local data set it
reproduces the cell of the main run bit for bit (checked for one cell, N = 128, seed 1). The
targets of the rebuild and of Dyssol differ at the level of rounding (at most 3e-12 in the pool;
the control of step 0), and 20000 optimiser steps carry that difference into the second decimal
of the reading numbers. At N = 128 the MLP sits on the threshold: main run 0.94 K and 1.04 %,
V-c 0.94 K and 0.98 % (2 of 5 seeds in both).

**This touches a result of the main run (reported, sections 11 to 13 are unchanged):** N\* of the
MLP is 256 with the Dyssol targets and 128 with targets that differ from them by rounding. The
verdict D1 does not depend on it (no regression has an N\* in either case); the number "256
Dyssol runs" does. Caveat 5 of section 12 named the narrow miss at N = 128; the add-on shows
that the miss is within the effect of rounding in the targets. How the data need is to be stated
is a decision of the project lead (section 18.4).

### 18.2 The variants

Median over the five seeds (range); bold = both medians at or below their threshold; test sets:
in-box Sobol test set, variant V-a / V-b.

**V-a: 7 inputs, medium flow as in Befund 128** (36 coefficients). N\* of the quadratic regression
**64**, of the MLP 128.

| N | quadratic: exhaust P95 [K] | fuel P95 [%] | seeds | MLP: exhaust P95 [K] | fuel P95 [%] | seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 42.79 (28.39-59.71) | 52.96 (40.59-66.62) | 0/5 | 16.01 (6.28-25.69) | 20.78 (11.28-30.29) | 0/5 |
| 16 | 28.63 (20.65-32.14) | 47.36 (33.50-51.80) | 0/5 | 2.69 (2.46-2.95) | 6.72 (6.51-8.74) | 0/5 |
| 32 | 8.87 (6.45-15.01) | 17.15 (10.39-23.43) | 0/5 | 1.78 (1.27-2.61) | 3.02 (2.31-3.15) | 0/5 |
| 64 | **0.32** (0.25-0.34) | **0.50** (0.41-0.55) | 5/5 | 0.68 (0.60-1.18) | 1.27 (0.87-1.44) | 1/5 |
| 128 | **0.22** (0.21-0.25) | **0.37** (0.32-0.47) | 5/5 | **0.31** (0.28-0.36) | **0.54** (0.50-0.60) | 5/5 |
| 256 | **0.21** (0.20-0.22) | **0.36** (0.31-0.41) | 5/5 | **0.18** (0.16-0.23) | **0.36** (0.30-0.39) | 5/5 |
| 512 | **0.20** (0.19-0.20) | **0.32** (0.30-0.35) | 5/5 | **0.13** (0.12-0.16) | **0.26** (0.22-0.36) | 5/5 |

**V-b: 5 inputs, full range of the medium flow** (21 coefficients). The quadratic regression has
**no N\***; N\* of the MLP is 128.

| N | quadratic: exhaust P95 [K] | fuel P95 [%] | seeds | MLP: exhaust P95 [K] | fuel P95 [%] | seeds |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24.29 (14.16-55.21) | 29.63 (21.94-57.93) | 0/5 | 10.74 (7.59-11.97) | 14.47 (11.53-16.26) | 0/5 |
| 16 | 15.79 (8.84-36.62) | 24.21 (12.97-42.46) | 0/5 | 5.74 (4.04-11.81) | 9.60 (6.61-15.09) | 0/5 |
| 32 | 3.82 (3.32-5.18) | 4.95 (4.01-7.28) | 0/5 | 2.42 (2.13-5.61) | 3.51 (2.11-5.86) | 0/5 |
| 64 | 3.25 (2.57-3.83) | 3.64 (3.45-4.97) | 0/5 | 1.31 (0.90-2.16) | 1.53 (0.97-2.35) | 0/5 |
| 128 | 2.69 (2.17-2.99) | 3.69 (3.44-3.85) | 0/5 | **0.56** (0.36-0.91) | **0.76** (0.47-1.13) | 3/5 |
| 256 | 2.33 (2.13-2.47) | 3.22 (2.99-3.39) | 0/5 | **0.29** (0.24-0.36) | **0.41** (0.32-0.45) | 5/5 |
| 512 | 2.19 (2.03-2.36) | 3.00 (2.82-3.27) | 0/5 | **0.20** (0.18-0.32) | **0.33** (0.27-0.47) | 5/5 |

**V-c (control):** quadratic regression as in section 11.4 (2.52 K and 4.05 % at N = 512, no
N\*); MLP 0.94 K and 0.98 % at N = 128 (2 of 5 seeds), 0.46 K and 0.62 % at N = 256, 0.24 K and
0.41 % at N = 512, N\* 128.

Over-saturated (flag of docs/45), pool / test set: V-a 9 / 0, V-b 9 / 3, V-c 9 / 0.

**What the numbers of the quadratic regression say.** N\* 64 in V-a, none in V-b, none in V-c:
this is the pattern of **K0** (the range of the medium flow, not the additional inputs). With the
medium flow of Befund 128 the quadratic regression reaches the thresholds with seven inputs at
N = 64, the number the prediction of section 6 had named for 36 coefficients; with the full range
it does not reach them even with five inputs and 21 coefficients. **By the committed rule this is
not the reading of the add-on** (section 18.1); whether K is read on the quadratic regression
alone is a decision of the project lead.

### 18.3 Without a reading: the capacity ratio

Capacity ratio B / min(A, C) of the medium over the smaller gas stream; "below" = the medium is
the smallest stream. Points of the pool below / at or above 1: V-a 134 / 890, V-b 543 / 481,
V-c 539 / 485; of the test set: V-a 34 / 222, V-b 135 / 121, V-c 134 / 122.

Quadratic regression at N = 512, median over the seeds of the 95th percentile over the test
points of the group (exhaust [K] / fuel [%]; number of test points):

| Capacity ratio | V-a | V-b | V-c |
|---|---|---|---|
| below 1 | 0.18 / 0.36 (34) | 2.50 / 2.88 (135) | 2.78 / 3.02 (134) |
| at or above 1 | 0.20 / 0.32 (222) | 1.84 / 3.23 (121) | 2.02 / 4.14 (122) |
| 0 to 0.25 | no point | 4.17 / 3.92 (34) | 5.41 / 4.29 (33) |
| 0.25 to 0.5 | no point | 1.68 / 2.00 (33) | 1.95 / 2.00 (35) |
| 0.5 to 1 | 0.18 / 0.36 (34) | 1.98 / 2.30 (68) | 2.65 / 2.91 (66) |
| 1 to 2 | 0.20 / 0.31 (221) | 1.53 / 2.60 (111) | 1.88 / 3.46 (112) |
| above 2 | 0.11 / 0.68 (1) | 2.51 / 5.94 (10) | 2.32 / 5.46 (10) |

* **Ratio 1 lies inside the box of Befund 128 as well:** in V-a 13 % of the points have the
  medium below the smaller gas stream (134 of 1024, 34 of 256), and the quadratic regression
  meets the thresholds on both sides of it. The hypothesis placed the equality of the capacity
  rates outside that box.
* **With the full range the quadratic regression misses the thresholds on both sides of ratio 1
  and in every bin.** Its error is largest towards no medium flow (ratio below 0.25) and, for
  the fuel, at the largest ratios; it is not concentrated at ratio 1.
* The MLP has its largest errors in the same lowest bin (V-c at N = 512: 0.73 K and 0.76 %
  below 0.25, at most 0.28 K and 0.44 % in the other bins).

This fits the equation of the unit (section 17): a saturation that is steepest towards no medium
flow and that no polynomial of degree 2 follows over the full range, and no kink. The cause that
the hypothesis names (the range of the medium flow) is what the numbers show; its mechanism (a
kink at equal capacity rates) is not found in the units.

### 18.4 Open for the project lead (add-on)

1. **Whether reading K is read with the control on the quadratic regression alone** (which
   reproduces the main run to 3e-13; K is defined on it). Then the verdict is K0. By the rule as
   committed the add-on is not read.
2. **How the data need of the main run is stated.** N\* of the MLP is 256 with the Dyssol targets
   and 128 with the rebuild's, which differ by rounding; in V-a and V-b it is 128 as well. D1 is
   unaffected. Sections 11 to 13 and the lab-book entry state 256 and are unchanged.
3. **Scope note to Befund 128** (section 16, item 4): its supplement "by the reading of the
   add-on" depends on item 1.
4. The hypothesis of a kink does not apply to the delivered units (smooth formula); whether the
   units are meant to model an ideal counter-current exchanger is a question to their author.
