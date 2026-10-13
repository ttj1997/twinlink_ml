# Draft: main run of study B1 -- information per run and active learning (NOT released, NOT computed)

Status: draft of 2026-10-12, second version after the decisions O1-O6 of the project lead. Committed
BEFORE any computation of the main run. On release the hypotheses, criteria and scripts are committed
again unchanged (or with the changes decided) before the first main-run job starts. Placeholders without
a source are marked PLACEHOLDER. **Renumber this file when the branch is merged: `docs/15` is taken in
master.** The branch is not rebased (provenance of the pilot artifacts); master is merged in when needed.

## 1. Question

Which excitation yields the most information per run, and does active learning save runs? "Information"
is measured by the error on named test sets as a function of the number of training runs N (learning
curve); the model-free measures of `simulation/excitation_information.py` are diagnostics, not the answer.

## 2. Design

| factor | levels |
|---|---|
| arm | B0 constant; B1 random steps (previous excitation, no constant runs); B2 two-scale multi-level PRBS; B3 multisine; B4 space-filling holds; B5 active learning (ensemble disagreement); B5r random selection from the same candidate pool as B5 |
| N (training runs; all runs count; no validation runs, fixed epoch budget, the same for every arm and model) | 2, 4, 8, 16, 32 |
| seed | **ten** master seeds 1-10, the same for every arm (paired) |
| model | persistence (parameter free, reference), ARX, LSTM (composite, one network per target, repository defaults, 200 epochs) |
| targets | `energy_target` [kW], `product_moisture_kg_per_kg`, `bed_temperature_celsius` [degC] |
| test set | T1 neutral mix (8 runs of each of B0-B4), T2 constant operation (32 runs), T3 deployment profiles (32 runs) |

* Data only with `DataSpec.new_study()`. Plans B0-B4 share run duration, amplitude limits, rate limits,
  disturbance process and initial conditions (bit-identical disturbances and initial states per seed;
  `check_equal_cost`, tests). **How often the plans are limited is not equal, see Section 7.**
* Time constants are the measured ones (Befund 3, M1 addendum): fast mode 30.2-38.8 s, slow coupled mode
  133-1195 s, solids loop 600 s. B2 clocks 75 s / 900 s, B3 band 1.3e-4-5.3e-3 Hz, B4 hold 2400 s.
  The task text assumed 15-80 s and about 600 s.
* **B5 / B5r** (`evaluation/active_learning.py`, `evaluation/b1_learning_curve.py`). Candidate pool of
  M = 128 plans per seed (decision O2: 32 each from B1-B4, no B0), built from inputs only before any
  simulation. Both arms start with the same N = 2 runs drawn at random from the pool and add runs up to
  the next N (4, 8, 16, 32). B5 trains an ensemble of five LSTMs (decision O3; seeds differ in
  initialisation and batch order; member 0 is the model evaluated at that N, identical to the model of the
  other arms) on the runs chosen so far and adds the candidates with the largest disagreement. B5r draws
  uniformly from the remaining candidates. The disagreement of a candidate uses its inputs only: the
  planned actions, the disturbances, and the targets at t = 0 from the output map at its physical initial
  state (held over the first history, decision 3 of the project lead); afterwards every member is run
  freely on its own predictions in non-overlapping windows of the horizon. Disagreement = standard
  deviation across members, averaged over windows, horizon steps and the three targets (standardised
  with the scalers of the current training set). Greedy top-k within a round, no diversity term.
  `GrowingPool` offers no outputs of an unselected candidate (structural); a test shows that scoring runs
  without any simulation.
* Test sets use seeds from a fixed namespace that no master seed enters and are checked against every
  training pool (ids and action arrays). Generated once and shared by all arms and seeds.
* **T3** (decision O1). Four ambient conditions (quartiles of the disturbance ranges); every run starts in
  the steady state of its anchor and its own initial disturbance, holds one hour, then makes four set-point
  changes around the anchor (not cumulative).
  Family 1 (16 runs): anchor = oracle optimum at lower gas-velocity bound 1.26 m/s and quality limit
  0.15 kg/kg (decoupled field), corrected solver (artifact `b1_t3_anchors`); two random channels change by
  5-10 % of their span.
  Family 2 (16 runs, replaces the old one): anchor just inside the limit (half the M1 margin below it);
  two or three channels change by 0.5-1.0 times a step scale taken from the spread (max - min) of the
  oracle set points over the 25 disturbance scenarios of Befund 52 at the same bound and limit (artifact
  `b1_t3_step_spread`: 11.28 K for the inlet temperature, 0.244 m/s for the gas velocity, 0.0132 bar for
  the nozzle pressure, which sits at its lower bound in almost all scenarios; the oracle does not
  optimise the spray rate, its scale is the largest relative spread, 0.188 of the span, times the span
  2.6 kg/h = 0.489 kg/h). The changes are drawn by rejection sampling so that the **steady-state** moisture
  of the new set point alternates over the quality limit (1st and 3rd change) and back inside (2nd and 4th);
  the count of changes for which the requirement could not be met is recorded (a test requires 0).
  Reason (project lead): in the pilot persistence beat ARX and LSTM on the old T3 because the changes were
  too small to do anything.

## 3. Metrics

Primary: run-median RMSE in physical units of the **product moisture at horizon 24** (600 s), per test
set (decision O5). Secondary: energy target and bed temperature, horizon 1; NRMSE against the measurement
and against the truth are recorded. **For every test set the value relative to persistence is reported**
(median of the model over the median of persistence on the same test set; below 1 the model beats
persistence); a factor against persistence is only quoted with the test set named. Comparisons over seeds
use `evaluation/paired_resolution.paired_comparison` (paired t, 9 degrees of freedom, alpha = 0.05,
two-sided); no correction for multiplicity except where stated; the number of tests is reported with every
table.

**Smallest resolvable difference, from the pilot spread, fixed before the run** (n = 10 paired seeds,
threshold = t(0.975; 9) x sd / sqrt(10) = 2.262 x sd / 3.162, sd = standard deviation of the paired
differences B1 - B2 in the pilot at N = 4, five pairs; artifact `pilot_b1_b2_n4`; horizon 24):

| model, test set (pilot) | target | pilot sd | threshold, ten seeds | relative to the B1 median | threshold with five seeds |
|---|---|---|---|---|---|
| LSTM, T1 | moisture | 5.5e-4 | 4.0e-4 kg/kg | 25 % | 6.9e-4 |
| LSTM, T2 | moisture | 1.1e-4 | 7.8e-5 kg/kg | 6 % | 1.4e-4 |
| LSTM, T3 (old) | moisture | 4.0e-3 | 2.9e-3 kg/kg | 58 % | 5.0e-3 |
| LSTM, T1 | energy | 0.0126 | 0.0090 kW | 13 % | 0.0156 |
| LSTM, T2 | energy | 0.0170 | 0.0122 kW | 22 % | 0.0211 |
| LSTM, T3 (old) | energy | 0.0876 | 0.0627 kW | 77 % | 0.1088 |
| ARX, T1 / T2 / T3 (old) | moisture | 1.1e-3 / 3.4e-3 / 6.2e-3 | 7.7e-4 / 2.5e-3 / 4.4e-3 kg/kg | 10 / 39 / 40 % | |

Reading: with ten seeds the main run resolves differences of about 6-25 % of the median on T1 and T2 and,
at the pilot's spread, about 60 % on the old T3 (dominated by single diverging seeds at N = 4; the new T3
and larger N will change this). With five seeds the thresholds are about 1.7 times larger, which is why the
number of seeds was raised. The pilot sd rests on five pairs and is itself uncertain (its 95 % interval
spans a factor of about 0.6-2.9); the thresholds are planning figures and are recomputed from the main
run's own spread in the analysis.

## 4. Hypotheses and criteria (fixed before any computation; wording of the analysis code)

**HB1 -- a planned excitation (B2, B3, B4) beats B1 at small N on T3.** Primary outcome: LSTM, moisture,
horizon 24, T3, N = 4; each of B2, B3, B4 against B1, paired over the ten seeds, Holm correction over the
three tests (alpha = 0.05). Supported for design X: X Holm-resolved better than B1 at N = 4 and the mean
difference better at N = 2 and N = 8. Refuted: for every X the mean difference is not better at any of
N = 2, 4, 8. Otherwise not established. Secondary (reported, not part of the verdict): energy target, ARX.

**HB2 -- active learning needs fewer runs than random selection for the same error.** Primary outcome:
LSTM, moisture, horizon 24, on T3 and on T1 separately; B5 against B5r. Supported: at N = 8 or 16 B5 is
resolved better, and the number of runs B5r needs to reach the median error of B5 at that N (log-log
interpolation along the median curve of B5r) is at least 1.5 times N. Refuted: no resolved difference in
favour of B5 at N = 4, 8, 16, 32 and the median of B5 is not below the median of B5r at N = 8 and 16.
Otherwise not established. Both curves start from the same N = 2 pool, so a difference can only arise from
N = 4 on. The compute cost of the ensemble training is reported next to the result.

**HB3 -- the ranking of the excitations depends on the test set.** B0-B4, LSTM, moisture, horizon 24, at
N = 4, 8, 16, pairwise paired comparisons on each of T1, T2, T3. Supported: at least one pair (X, Y) and
two test sets A, B at one N with X resolved better than Y on A and Y resolved better than X on B.
Refuted: no resolved reversal at any of the three N and the rank orders of the five arms by the median
differ by at most one position per arm between every two test sets. Otherwise not established. Kendall tau
between the rank orders is reported for every N. Tests per model: 3 N x 10 pairs x 3 test sets = 90 (not
corrected).

All three are implemented in `evaluation/b1_analysis.py` and tested on synthetic records for every verdict.

## 5. Cost (8 workers)

Calibration (artifacts `pilot_b1_b2_n4` and `calibrate_b1_fit_time`, one thread per process, 200 epochs, no
validation, **three targets**): serial LSTM fit 64 s (N = 4, 552 windows), 96 s (8), 195 s (16), 508 s (32);
eight concurrent fits at N = 32: 723-732 s, i.e. a contention factor of 1.43 against the serial time. The
fit time is **not** proportional to the number of windows (a fixed cost per epoch dominates at small N);
the table uses the measured points. The pilot's N = 4 fit with two targets took 70-79 s under eight
concurrent jobs.

| item | calculation | CPU time |
|---|---|---|
| LSTM fits of one learning curve | (64 for N = 2, upper bound from N = 4, + 64 + 96 + 195 + 508) s x 1.43 | 1 330 s |
| 7 arms x 10 seeds | 70 curves | 92 800 s |
| ensemble members of B5 (4 extra members at N = 2, 4, 8, 16) | 4 x (64 + 64 + 96 + 195) s x 1.43 x 10 seeds | 24 000 s |
| data generation | about 2.1 s per run (pilot: 7.7-8.7 s for four runs), 70 pools of 32 runs | 4 700 s |
| ARX, persistence, evaluation of about 350 fits on T1-T3, candidate scoring, information measures | assumption, about 5 s per fit (not measured for three targets) | 1 800 s |
| **total CPU** | | **about 123 000 s** |
| **wall time with 8 workers** | / 8 | **about 15 400 s = 4.3 h** |

Plausible range 4-6 h (the contention factor is measured for N = 32 only; the evaluation and scoring items
are assumptions). Levers: ensemble of 3 instead of 5 saves about 0.4 h; five instead of ten seeds halves
the time. Sweeps run resumably (`--resume`, one job = arm x seed, the records of a job are written when it
is done), on a local machine in a worktree on C:\ (decision of the project lead), at most 8 workers.
Smoke run of the driver (not an artifact, two jobs, N = 2 and 4): the B5 job needed 222 s against 114 s for
B2, as expected from the four extra members.

## 6. State of the build

Built and tested (branch `b1-excitation-active-learning`): plans B0-B4, equal-cost check, model-free
measures (spectral coverage at 1, 10 and 25 % thresholds, decision O4, diagnosis only; FIR and ARX Fisher
information), growing pool (full pool bit-identical to `generate_data`), test sets T1-T3 with family 2,
B5 acquisition, learning-curve driver `scripts/run_b1_main.py` (resumable, header with provenance and test-set
digests), analysis `scripts/analyze_b1_main.py`, calibration. Pilot and calibration are artifacts; the
error figures of the pilot are no result. Full suite on the commit before this document: **783 passed, 15 skipped in 1883 s (0:31:23), exit code 0** (the 15 skips include `test_denylist.py`, which needs `local/denylist.txt` of the main checkout; the log ends with the summary line and the exit code).
Not run: the main run, and the analysis (it needs the main run).

## 7. Finding to decide on: the plans are not limited equally often (report only, nothing changed)

Artifact `b1_limit_rates` (script committed before the run; plans B0-B4, pilot master seeds 1-5, four runs
each, `DataSpec.new_study()`; shares of steps; channels in the order inlet temperature, gas velocity, spray
rate, nozzle pressure).

*Simulator side.* The realised action equals the requested one in **all** steps of all plans and channels
(share 0.0 at every channel and both bounds): the box clip (`enforce_action_bounds`) never acts because
every plan is inside the box by construction, and the gas-velocity projection is **off** in the study
config. The requested actions are never limited by the simulator.

*Design side* (differs between plans, and so does what the model sees):
| plan | rate limiter changed the signal (T_in / u_gas / spray / p), % of steps | requested action on lower / upper box bound (same order), % of steps |
|---|---|---|
| B0 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 ; 0.0 / 0.0 / 0.0 / 0.0 |
| B1 | 8.0 / 0.0 / 10.0 / 1.9 | 3.1 / 0.0 / 2.7 / 1.9 ; 3.0 / 0.0 / 2.6 / 2.1 |
| B2 | 18.1 / 0.3 / 35.5 / 1.3 | 19.5 / 24.5 / 15.5 / 26.0 ; 21.9 / 24.8 / 16.2 / 24.6 |
| B3 | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 ; 0.0 / 0.0 / 0.0 / 0.0 |
| B4 | 1.1 / 0.1 / 1.4 / 0.4 | 0.0 / 0.0 / 0.0 / 0.0 ; 0.0 / 0.0 / 0.0 / 0.0 |

*Where the projection would act if it were on.* The share of steps with the realised gas velocity **above
the terminal velocity** of the current particle size (never below the minimum fluidisation velocity):
B0 3.6 %, B1 6.9 %, B2 7.4 %, B3 2.0 %, B4 4.7 %.

Consequence: "equal cost" holds for run duration, bounds, rate limits, disturbances and initial states, but
not for how often the rate limiter shapes the signal (B2 changes 35 % of the spray-rate steps, B3 none) nor for
how long the signal sits on a box bound (B2 15-26 % of the steps per channel and bound, B1 2-3 %, B3 and B4
about 0). These are properties of the plans; they are part of what a comparison of the plans measures. If the
safety projection were turned on in a later study the plans would be limited by different amounts. Open
decision O7.

## 8. Open decisions for the project lead

* **O7** Accept the unequal limiting as a property of the plans (reported with the result), or equalise (for
  example a common amplitude cap or slew shaping for all plans)? Nothing is changed before the decision.
* **O8** Fit times and the 4.3 h estimate: reduce the ensemble to 3 members or the seeds to 5, or run as
  planned?
* **O9** Family 2 of T3 uses steady-state crossing of the limit; the transient moisture crosses it only if the
  hold between changes is long enough (6300 s per change). Keep?
