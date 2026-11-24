# NN -- Dyssol flowsheet with the fresh-gas state as an input: data need of a surrogate with 7 inputs, and the partner sheet as a named test set

Status: **design and pre-registered reading; no Dyssol data set of this study exists yet.**
Sections 1 to 10 are the design and the reading, committed BEFORE the pilot and before any Dyssol
run of this study; results are added below them and take their numbers only from the artifacts
named there. Branch `dyssol-freshgas` (from `master` 2e54dde), worktree
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
