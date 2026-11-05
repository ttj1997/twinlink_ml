# NN -- Hybrid model (physics + learned residual) against a pure GRU with few runs: how much safety margin does it save?

Status: design fixed and committed before the runs (sections 1-7); runs and results in sections
8-10, added after the runs (numbers only from the artifacts named there). Branch `hybrid-model`
(from `origin/master` 0106563). Finding number: open (assigned by the managing session at merge;
lab book entry "Befund NN (number open)").

## 1. Question

Befund 113 (surrogate-env): learned sequence models (GRU of the Befund 94 pipeline) underestimate
the product moisture systematically; the margin they need costs about 4.6 % energy. Befund 107: a
physics-based surrogate with a margin needs fewer runs. Befund 117: the physics model does not
match the plant structurally. Here: with few runs (4-32), how much margin does a hybrid model --
grey-box prior knowledge plus a GRU on the residual -- save against a pure GRU, and does a simple
refit of the grey-box parameters do as well?

## 2. Truth, prior knowledge, data

* **Truth**: the grey-box model of the Befund 94 pipeline (`conf/process/greybox_gab_placeholder.yaml`)
  with the material parameters of one of the **20 draws of Befund 111** (critical moisture, nu_max,
  five GAB parameters), read from `conf/sensor_value/sensor_value.yaml` with the call of
  `scripts/run_sensor_value.py`; the 20 draws equal those of `sensor_value_v1.json.gz` exactly.
* **Prior knowledge**: the same model with the nominal parameters.
* **Data** per draw and master seed 1-3: the pool of the Befund 94 pipeline (arm N,
  `DataSpec.new_study()`, preset version 2; 64 / 32 / 32 runs of 8 h at 25 s) generated with the
  truth; run-level splits and the fixed validation pool as Befund 94; `n_train` in {4, 8, 16, 32}.
  The same master seed gives the same excitation for every draw.

## 3. Models

| | model | data |
|---|---|---|
| M0 | prior knowledge only (nominal grey-box) | none |
| M1 | GRU of the Befund 94 pipeline (`gru`, 200 epochs, training seed of the seed bundle) | truth pool |
| M2 | refit of critical moisture and nu_max (least squares on all three targets of the training runs, open-loop replay, as M4 of Befund 107; bounds [0.15, 0.45] and [0.2, 1.0]) | truth pool |
| M3 | prior knowledge + GRU on the residual `truth - nominal replay` (same architecture, budget and training seed as M1) | residual pool |
| M4 | as M3, ensemble of 5 (M3 + 4 training seeds `_derive(model_seed, "hybrid_model_ensemble", j)`), constraint `mean + k std` | residual pool |

Residual: the nominal model replays every run of the truth pool open loop from the run start
(realised actions, disturbances, physical initial states); the residual pool has outputs
`truth - replay` and measurements `measured truth - replay`, the same windows and splits. Steady
state of M3: nominal steady state + fixed point of the residual network (criterion of Befund 113);
with a residual of zero M3 is M0 exactly (test). Prediction error of M3 on a window = error of the
residual network on the residual data; of M0/M2 = minus the residual of their replay.

## 4. Task and optimiser

The one-step operating-point problem of Befund 113 (actions inlet temperature [60, 120] degC and gas
velocity [0.8, 2.5] m/s, context ambient temperature and humidity, nozzle pressure, spray rate and
solids fixed; limits moisture 0.15 kg/kg (placeholder), bed 80 degC; objective `energy_target`).
Steady state of a network: fixed point of its h36 map, criterion of Befund 113 unchanged.

**Optimiser (decision of the project lead, 2026-11-03)**: grid + zoom instead of the 8-start
Nelder-Mead of Befund 113 (that one costs about 1300 evaluations per solve, ~700 CPU-h with the
margin search). Coarse grid 21 x 21 over the action box (shared by every margin), then from each
of the 4 best coarse points three zoom levels of 9 x 9, on every level recentred on its best point
until it stays (final spacing 0.047 K, 0.0013 m/s; the best end point wins), on the penalised
objective of the oracle (`E + 50 kW x normalised violation`, invalid point 1e6 kW); the same for
every model and for the oracle (the truth, margin 0). The truth's steady state is computed by a
batched version of the steady solver (`hybrid_model/steady_batch.py`), bit-identical to the bandit
evaluation at the tested points.

**Change after the pilot, before the main run.** The first version (one start, two zoom levels,
no recentring) stayed up to 2.0 % energy above the 8-start Nelder-Mead in the pilot's
discretisation check (`_hybrid_pilot_oracle.jsonl`, gitignored): the optimum lies on the moisture
limit and moves along it to the edge of the box, where a fixed zoom window cannot follow, and in one
case a second local optimum along the limit is cheaper. With recentring, 4 starts and 3 levels a
scratch re-check of the same 10 cases gave at most 0.32 % (one case at the corner u = 0.8), the
others at most 0.11 %; 8 starts changed nothing. The check of the main run is the one reported.

**Discretisation check (addendum 1 of the request)**: at 10 interior cases the oracle once more by
the 8-start Nelder-Mead of Befund 113 on the same objective; energy and slack difference reported;
above 0.2 % energy the number stands next to every energy figure.

## 5. Cases, inside / outside

Per draw 5 interior contexts of the 25 Befund-52 grid points (draw d: temperature level j, humidity
level (j + d) mod 5; every grid point 4 times) and 2 extrapolation contexts **(a)** outside the
disturbance box of the data design (0 / 40 degC at 0.5; 20 degC at 0.10 / 0.95; one of each kind per
draw). 140 cases, the same for every model, seed and `n_train` (rotation committed in
`conf/hybrid_model/hybrid_model.yaml`). The steady evaluation of (a) uses the config with widened
ambient ranges (only the bracket of the inner bisection depends on them).

**(b)**: the excitation covers every channel over its full range already at `n_train` 16, so an
interior case is *inside* the training range of a (draw, seed, `n_train`) cell iff the oracle
operating point (T_in, u_gas, T_amb, phi) lies within 0.2 of the design range of some training
sample in all four channels (Chebyshev), else *outside (b)*. The threshold was chosen before the run
on input data only (seeds 1-2, nominal oracle points): at `n_train` 4 about a third, at 32 most
points inside; 0.1 would leave about ten cases for the quantile at `n_train` 4.

## 6. Metrics and pre-registered reading

Full text: docstring of `scripts/analyze_hybrid_model.py` (committed before the first run). In
short: **margin** = 95 % quantile over the INSIDE cases of the per-case needed margin (bisection;
M0-M3 constant m on [0, 0.12], tolerance 1e-4; M4 factor k on [0, 10], tolerance 0.01, effective
margin = median `k std` at the chosen points); not reached -> reported, not aborted; raise rule of
Befund 111. At exactly that margin every case is optimised and evaluated at the truth: energy
relative to the oracle (median over inside cases), violation share outside (b) and (a) (no own
margin search outside), moisture bias at the chosen points and box-wide, one-step / rollout errors.
Learning curves over `n_train`, 3 seeds, median and range; paired resolution over seeds for every
rank claim.

* **Y0**: at every `n_train` <= 16 the margin of M3 is at most half that of M1, resolved (paired) ->
  hybrid recommended for few data.
* **Y1**: at every `n_train` <= 16, M2 within 10 % of M3 in margin and in energy -> a simple
  parameter refit suffices, no residual network needed.
* **Y2**: at no `n_train` <= 16 is M3 resolvably better than M1 -> no advantage of the hybrid here.
* **Extrapolation**: per model and `n_train`, "margin holds outside" (<= 5 % violations, pooled over
  seeds), (a) and (b) separately.

**No valid point (added after the pilot, before the main run).** In the pilot one member of M4 at
`n_train` 32 had an unstable moisture fixed point at every point; under the rule of Befund 113
(valid only if every head of every member is valid) M4 then has no valid point at all. The rule is
kept. A case without a single valid grid point counts as not holding, its energy is left out of the
energy figures, and the number of such cases is reported.

## 7. Budget rule and runs

Wall-clock budget 30 h; above it, first `n_train` 32 is dropped, then M4 (the task order of the
training driver puts these last). Scripts: `scripts/train_hybrid_model.py`,
`scripts/run_hybrid_evaluation.py` (parts `oracle`, `margins`, `errors`),
`scripts/analyze_hybrid_model.py`; tests `tests/test_hybrid_model.py`. Checkpoints
`artifacts/checkpoints/hybrid_model/` (gitignored), copied to `C:/twinlink_runs/hybrid-model/`.

## 8. Decisions without consultation and decisions of the project lead

**Decisions of the project lead.**

* 2026-11-03: grid + zoom instead of the 8-start Nelder-Mead (section 4); 5 interior contexts per
  draw, rotated; extrapolation (a) and (b); M4 by bisection on k; the addendum of the request
  (discretisation check, paired cases, margin only from inside cases, k <= 10).
* 2026-11-04, **30-h budget rule**: the run is to finish provided it demonstrably works; no cut.
  The budget was exceeded: the study started 2026-11-03 12:24 (training of de3079f, 1680 tasks,
  done 22:32, log `hyb_main_train.log`: 36436 s wall); the oracle of a343840 ran 22:35-22:45;
  the margins run of a343840 ran from 22:45 until it failed on 2026-11-05 04:50 (30.1 h; 40.4 h
  since the start of the study). Nothing was cut. The run was continued on instruction; it failed
  for the reason below, not by the budget rule.
* 2026-11-05, **rerun of the margins** after that failure: (1) a draw without any inside case
  has no margin search, its cases are evaluated at the chosen parameter, with a test for exactly
  this case; (2) the driver split per draw (one task per draw, every record written at once,
  `--resume`, errors recorded per task, progress with time and memory peak in the log, the
  computation unchanged); (3) a dated addendum to the reading (committed in ac8173a before any
  margin result): a margin is determinable only from >= 20 pooled inside cases per cell, the
  number of inside cases reported per cell, Y0 on the determinable cells; (4) a dry run over all
  tasks at minimal cost before the main run; (5) the main run in full (`n_train` 32 included, no
  cut, running over the weekend approved), at most 8 workers (= physical cores), the sum of the
  memory peaks below 80 % of the RAM; (6) the main run is started by the project lead.

**The failed margins run of a343840.** For 8 (seed, `n_train`, draw) combinations (s1 n4: draws 9,
14; s2 n4: draws 3, 8, 13, 18; s2 n8: draws 8, 13) no case of the draw is inside the training
range; the search then called the evaluator with an empty case set (`IndexError`). The pilot (draw
0) has inside cases everywhere and did not show it. After the first failed job the driver left its
loop, while `ProcessPoolExecutor.shutdown(wait=True)` still ran every remaining job and discarded
the results; the four idle workers seen at 03:40 had no job left. No margin result was written;
the header of that run is kept (`hybrid_model_v1_margins.jsonl`, committed in 3207796, never to be
appended to).

**Decisions without consultation** (taken by the session, reported to the project lead).

* Coverage threshold 0.2 (section 5), the optimiser change after the pilot (section 4) and the
  rule for a model without a valid point (section 6), each before the main run.
* **Clean-tree exception** (a343840, kept in 3c00410): the training records (12.6 MB) are above
  the size rule of the request and stay uncommitted as JSONL; the evaluation exempts exactly its
  own output and two inputs -- the training records and the oracle -- from the clean-tree check
  (`clean_tree_exceptions` in `scripts/run_hybrid_evaluation.py`; test
  `test_clean_tree_exception_is_exactly_the_output_and_the_two_inputs`: any other untracked file
  and any modified tracked file still count) and pins their sha256 in its header; the analysis
  exempts only the training records. Reason: the evaluation needs the training records as input,
  and committing them needed the project lead's consent. The training records are committed as
  gzip (`hybrid_model_training_v1.jsonl.gz`, 3207796, with consent); the sha256 of the
  decompressed content equals the value pinned in the oracle header (e0cd7a0d...f2418b, checked
  2026-11-05). The pinned hashes refer to the files as written on Windows (CRLF line ends); git
  stores them with LF, so a checkout with other line-ending settings gives other bytes.
* **Bounded prediction batches** (3c00410): `CaseEvaluator(max_batch=256)` in the margins tasks.
  The coarse grid of five cases (2205 points) in one call of a GRU map needs about 9.6 GB peak
  commit (M1 and M4, d0 s1 n32, scratch check of 2026-11-05), 8 workers would need about 77 GB;
  in calls of 256 points 2.2 GB with bit-identical predictions in every field.
* **Worker processes reused** (7044da1): a fresh process per task costs 40-70 s of imports from
  the network drive (about 6-7 h over the main run); the memory peak in the records is the peak
  of the worker process so far. Checks requested by the project lead before the main run
  (ffa30a9): every task starts with `reset_task_state()` (random, numpy and torch set to a fixed
  seed, torch one thread, the cached study setup verified unchanged by the sha256 of its pickle);
  maps, networks and evaluators are built in every task. Audit: no task draws from a global random
  source, there is no module-level cache besides the setup, the networks predict in eval mode.
  Fresh process against reused worker at the grid of the main run (one bisection step; scratch
  check of 2026-11-05 on ffa30a9): search M1 s1 n4 d0, evaluation M3 s2 n8 d5, search M4 s3 n16
  d10 and errors d4 s2, the reused worker after 2 to 5 other tasks: all four records bit-identical
  (without elapsed time, memory and pid). The same comparison at the minimal grid is the test
  `test_tasks_in_a_reused_worker_equal_fresh_processes`. Leak check: in the minimal dry run (about
  300 tasks per worker) the running peak commit of every worker stays flat after the first quarter
  of its tasks (rise over the second half 0 MB); since ffa30a9 the records and the log also carry
  the current commit at the end of each task.

**Dry run** (2026-11-05, 7044da1, clean tree; outputs gitignored: `_hybrid_dryrun_margins`,
`_hybrid_memprobe_margins`, `_hybrid_dryrun_errors`). *Minimal* (1 bisection step, 3 x 3 grid, no
zoom) over every margins task: 2400 tasks, 0 errors, 60 of 60 cells; 40 search tasks without any
inside case (the 8 combinations above x 5 models) ran without a search and their cases were
evaluated; pooled inside cases per cell 29 (s2 n4) to 94 (s1 n32), so every cell is determinable
under the addendum; 10 cells took the "not reached" path; the raise over rounds did not occur (it is
covered by `test_scheduler_raises_over_rounds_and_resumes`). *Memory probe* (full coarse grid, all
models, s1 n32, draws 0-3): 40 tasks, 0 errors, peak commit 1246-1273 MB per worker, sum of the 8
workers 10.1 GB (32 % of 31.2 GB). *Errors part* (complete, it has no reduced mode): 60 tasks,
0 errors, 954-973 MB per worker, 43 min wall.

## 9. Runs and controls

(after the runs)

## 10. Results

(after the runs)

## 11. Caveats

(after the runs)
