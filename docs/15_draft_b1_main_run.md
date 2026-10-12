# Draft: main run of study B1 -- information per run and active learning (NOT released, NOT computed)

Status: draft of 2026-10-12 for the decision of the project lead. Committed BEFORE any computation of
the main run. Everything below the pilot (Section 6) is a plan; nothing of it has been run. On release the
hypotheses, criteria and scripts are committed again unchanged (or with the changes decided) before the
first main-run job starts. Placeholders without a source are marked PLACEHOLDER.

## 1. Question

Which excitation yields the most information per run, and does active learning save runs? "Information"
is measured by the error on named test sets as a function of the number of training runs N (learning
curve); the model-free measures of `simulation/excitation_information.py` are diagnostics, not the answer.

## 2. Design

| factor | levels |
|---|---|
| arm (excitation) | B0 constant; B1 random steps (previous excitation, no constant runs); B2 two-scale multi-level PRBS; B3 multisine; B4 space-filling holds; B5 active learning (ensemble disagreement); B5r random selection from the same candidate pool as B5 |
| N (training runs, all runs count; no validation runs, `ValidationPolicy.NONE`) | 2, 4, 8, 16, 32 |
| seed | master seeds 1-5, the same for every arm (paired) |
| model | persistence (parameter free, reference), ARX, LSTM (composite, one network per target, repository defaults) |
| test set | T1 neutral mix (8 runs of each of B0-B4), T2 constant operation (32 runs), T3 deployment profiles (32 runs) |

* Data only with `DataSpec.new_study()` (planned future set points, own solid-fraction process); plans
  B0-B4 share run duration, amplitude limits, rate limits, disturbance process and initial conditions
  (bit-identical disturbances and initial states per seed; checked by `check_equal_cost` and tests).
* Time constants are the measured ones (Befund 3, M1 addendum): fast mode 30.2-38.8 s, slow coupled mode
  133-1195 s, solids loop 600 s. B2 clocks 75 s / 900 s, B3 band 1.3e-4-5.3e-3 Hz, B4 hold 2400 s
  (12 holds per run). The task text assumed 15-80 s and about 600 s; the measured fast mode starts at 30 s
  and the coupled slow mode reaches 1195 s at the cold end.
* B5/B5r: candidate pool of M = 128 candidate plans (PLACEHOLDER size, 32 each from B1-B4, inputs only,
  per seed, built before any simulation); start with N = 2 runs drawn at random (identical for B5 and B5r);
  rounds add runs up to the next N. B5 picks the candidates with the largest ensemble disagreement
  (5 LSTMs, differing in initialisation and batch order, trained on the current pool; disagreement = mean
  over the horizon of the standard deviation across the members of the predicted targets, in standardised
  units, from the candidate's inputs and the initial history computed from the output map at the plan's
  physical initial state; the candidate's simulated truth is never available to the selection, structurally:
  `GrowingPool.candidate_inputs`). B5r picks uniformly at random from the remaining candidates. Selection is
  greedy within a round (top-k by disagreement; no diversity term in the first version).
* Test sets use seeds from a fixed namespace that no master seed enters and are checked against every
  training pool (ids and action arrays). They are generated once and shared by all arms and seeds.
* T3: start in the steady state of four ambient conditions' optimum (lower gas-velocity bound 1.26 m/s,
  quality limit 0.15 kg/kg of the decoupled field, corrected solver) or of a point on the spray-rate line
  with the steady-state moisture half the M1 margin (0.00298 kg/kg) below the limit; one hour hold, then
  four changes of two channels by 5-10 % of their span around the anchor. Anchors: artifact
  `b1_t3_anchors`. **Observation:** at 1.26 m/s the optimum sits on the lower bounds of inlet temperature
  and nozzle pressure and the moisture limit binds at three of the four ambient points (moisture 0.1500),
  so the two T3 families differ only by about 1 % of the spray rate at those points; they are not two
  distinct regimes. See open decision O1.

## 3. Metrics

Primary: run-median RMSE in physical units (kW for `energy_target`, kg/kg for the product moisture) at
horizon 24 (600 s), per test set; secondary: horizon 1; NRMSE against the measurement and against the truth
are recorded. Persistence is reported on every test set next to every model, and a factor against
persistence is only quoted with the test set named (CLAUDE.md). Comparisons over seeds use
`evaluation/paired_resolution.paired_comparison` (paired t, n - 1 = 4 degrees of freedom, alpha = 0.05,
two-sided); no correction for multiplicity except where stated below; the number of tests is reported with
every table.

## 4. Hypotheses and criteria (fixed before any computation)

**HB1 -- a planned excitation (B2, B3, B4) beats B1 at small N on T3.**
Primary outcome: LSTM, product moisture, horizon 24, T3, N = 4. Each of B2, B3, B4 against B1, paired
over the five seeds, Holm correction over the three tests (alpha = 0.05).
* Supported for design X: X resolved better than B1 in the primary outcome, and the mean difference has
  the same sign at N = 2 and N = 8.
* Refuted: for none of B2-B4 is the mean difference better than B1 at N = 2, 4 and 8.
* Otherwise: not established (reported with the differences and thresholds).
Secondary (same rule, reported, not part of the verdict): energy target; ARX; N = 2, 8.

**HB2 -- active learning needs fewer runs than random selection for the same error.**
Primary outcome: LSTM, product moisture, horizon 24, on T3 and on T1 separately; B5 against B5r.
* Supported: at N = 8 or N = 16 B5 is resolved better than B5r (paired over seeds), and the number of
  runs B5r needs to reach the median error of B5 at that N (log-log interpolation along the learning
  curve of B5r, five grid points) is at least 1.5 times that N.
* Refuted: no resolved difference in favour of B5 at N = 4, 8, 16, 32, and the median of B5 is not below
  the median of B5r at N = 8 and 16.
* Otherwise: not established.
Both curves start from the same N = 2 pool, so a difference can only arise from N = 4 on. The compute
cost of the ensemble training is reported next to the result; it is not part of "runs".

**HB3 -- the ranking of the excitations depends on the test set.**
Over the arms B0-B4, LSTM, product moisture, horizon 24, at N = 4, 8, 16, pairwise paired comparisons
on each of T1, T2, T3.
* Supported: at least one pair (X, Y) and two test sets A, B at one N with X resolved better than Y on A and
  Y resolved better than X on B (a resolved reversal).
* Refuted: no resolved reversal at any of the three N, and the rank orders of the five arms by the median
  differ by at most one rank position per arm between every two test sets.
* Otherwise: not established. The Kendall tau between the rank orders is reported for every N and model.
Number of tests per N and model: 3 test sets x 10 pairs = 30 (not corrected).

## 5. Cost (8 workers)

Calibration (artifact `pilot_b1_b2_n4`, 8 worker processes, one thread each, N = 4 pool of 552 windows,
200 epochs, no validation): LSTM fit 70-79 s per job while eight jobs ran concurrently (40-42 s when
only two ran); data generation 7.7-8.7 s for four runs; ARX fit 1.3 ms; evaluation of one fitted model
on T1-T3 about 1 s; build of the three test sets 50 s in one process.

Projection (the fit time is assumed proportional to the number of windows, 138 per run, at fixed epoch
count and batch size; to be checked by one N = 32 fit before release):

| item | calculation | CPU time | wall time, 8 workers |
|---|---|---|---|
| LSTM fits, one learning curve of one arm and seed | (2+4+8+16+32) runs x 138 windows x 78 s / 552 windows | 1209 s | |
| 7 arms x 5 seeds | 35 curves | 42 300 s | 5 300 s |
| ensemble rounds of B5 (5 members, N = 2, 4, 8, 16) | 5 x 30 runs x 138 windows x 78 / 552, 5 seeds | 14 600 s | 1 800 s |
| data generation | about 2.1 s per run, 32 runs, 35 pools | 2 350 s | 300 s |
| ARX, persistence, evaluation, test sets | about 200 evaluations at about 1 s plus 50 s | about 250 s | 60 s |
| **total** | | **about 59 500 s** | **about 7 500 s = 2.1 h** |

Plausible range 2-4 h (the contention factor of 1.4 against a single process is included; the linear
scaling in the window count and the cost of the candidate scoring are the uncertain parts). Sweeps run
resumably (`--resume`), on a local machine, in a worktree on C:\ (decision of the project lead), at most
8 workers.

## 6. State of the build, and what is still missing

Built and tested (branch `b1-excitation-active-learning`, commit `4eccb3e` and following): plans B0-B4,
equal-cost check, model-free measures, growing pool (full pool bit-identical to `generate_data`, test
`test_full_pool_equals_generate_data`), test sets T1-T3, T3 anchors, pilot driver.
Pilot (B1.4, artifact `pilot_b1_b2_n4`): run times only, error figures are no result.
**Not built yet:** the B5 acquisition (ensemble training, initial-history from the output map, scoring of
the candidate pool, selection), the learning-curve driver with `--resume`, and the analysis script with
the criteria of Section 4. They are built after release of this draft, with tests, and committed before the
first main-run job.

## 7. Open decisions for the project lead

* **O1** T3 families: at u_min = 1.26 m/s the optimum already binds the moisture limit at three of four
  ambient points. Keep both families (nearly identical operating points, different ambient conditions),
  replace the second family by a point clearly inside the limit (for example half the margin further
  away from it), or add a third family with larger steps across the limit?
* **O2** Candidate pool for B5: M = 128 from B1-B4 in equal parts (PLACEHOLDER) or from all of B0-B4?
* **O3** Ensemble size 5 for B5 (about 1 800 s wall). A smaller ensemble (3) saves about 700 s.
* **O4** The spectral-coverage measure (1 % of the white-signal density) is saturated at 1.0 for both B1 and
  B2 in the pilot and does not discriminate; the Fisher-information measures do. Tighten the threshold
  (for example 10 %) before the main run?
* **O5** Primary outcome for HB1/HB2 is the moisture at horizon 24; the energy target is secondary. Confirm?
* **O6** Validation policy NONE (fixed 200 epochs) makes N the full cost; the alternative FIXED_POOL
  would add validation runs to N. Confirm NONE.
