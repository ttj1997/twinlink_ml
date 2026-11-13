# NN -- Hybrid model (physics + learned residual) against a pure GRU with few runs: how much safety margin does it save?

Status: design fixed and committed before the runs (sections 1-7); runs and results in sections
8-11, added after the runs (numbers only from the artifacts named there). Branch `hybrid-model`
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

**Course of the main run** (decisions of the project lead of 2026-11-06 and 2026-11-09).

* **Start** (2026-11-06, "Start" of the project lead with a condition: start as soon as no run or
  suite of the case-ranking study computes any more, at the latest at 19:30). A watcher
  (`C:/twinlink_work/hyb_start_watch.ps1`, outside the repository) sampled every 60 s the process
  tree below any process with `case_ranking` in its command line (busy above 5 CPU-s per minute)
  and every other Python process (busy above 30 CPU-s per minute) and started after 10 quiet minutes
  in a row. Start 2026-11-06 16:37:38 on 1e2a635, clean tree. Reason: no case-ranking run or suite
  and no other Python process computing for 10 minutes (case-ranking processes had kept it back
  until the sample of 16:27:38: 18 processes, 481 CPU-s per minute). Overlap at the start: only a `tail -f` on the case-ranking log (0 CPU-s per
  minute), no computing process. Logged in `C:/twinlink_work/hyb_start_decision.log` and
  `hyb_main_rerun.log`.
* **Margins** (`hybrid_model_v1_margins_per_draw`): finished 2026-11-08 19:07:05, 181751 s wall
  (50.5 h); 2400 tasks, 0 errors, 60 of 60 cells, no cell blocked. Per-worker memory: peak commit
  1246-1326 MB, flat after the first tasks; the commit at the end of each task shows no steady rise
  (slope per worker between -0.3 and +0.6 MB per task over about 292 tasks per worker). Integrity check of the output
  (2026-11-09): 2462 lines, every line valid JSON, last line (footer) complete, no duplicate key;
  nothing was cut.
* **Abort of the errors stage and interruption.** The errors stage started at 19:07:07 and stopped
  at 19:07:24 at the clean-tree check without writing anything: the margins output was untracked,
  and the exceptions of the errors stage cover only its own output, the training records and the
  oracle. The run chain ended there. The Claude Code session ended later through an expired
  sign-in; that stopped the session's watcher, but no process of the run was running any more.
  Interruption: 2026-11-08 19:07:24 to 2026-11-09 10:08:02 (about 15 h without computation); cause:
  the chain of the session, which did not foresee that the margins output would be untracked when
  the errors stage started (the dry runs wrote to gitignored names), not the end of the session.
* **Decision of the project lead (2026-11-09)**: no new exception in the clean-tree check. The
  margins file was committed packed (`hybrid_model_v1_margins_per_draw.jsonl.gz`, 3ad9ef6, 2.46 MB);
  sha256 of the raw file noted before packing and checked against the unpacked content after
  packing: d67336b5...2b7acff, identical. The raw file (15.4 MB) was copied to
  `C:/twinlink_runs/hybrid-model/` (same sha256) and then removed from the worktree. The analysis
  reads the packed file directly (`read_lines` and `records_path` in
  `scripts/analyze_hybrid_model.py`, aafc95f, test `test_analysis_reads_the_packed_margins_file`);
  the errors stage does not read the margins file. Full suite on aafc95f: 1417 passed, 30 skipped.
  The errors stage was restarted detached from the Claude Code session (`Win32_Process.Create`, so
  that an end of the session cannot end the run; script `C:/twinlink_work/hyb_errors_stage.sh`, log
  `hyb_rerun_errors2.log`): start 2026-11-09 10:08:02 on aafc95f, clean tree, training records
  matching the pinned sha256; finished 10:43:05, 2084 s, 60 tasks, 0 errors, peak commit
  904-970 MB per worker. `--resume`: a task is recorded only when it is complete and a resume skips
  exactly the keys with a complete `ok` record, so no finished task is computed again and a started
  task without a record is computed again in full (`test_scheduler_raises_over_rounds_and_resumes`).
  The margins needed no resume (complete); the errors stage started without any record.
* **Decision without consultation (aafc95f)**: in the reuse check `fresh_and_reused` the fresh
  processes now run while the reused worker is alive. Reason: the pid comparison of
  `test_tasks_in_a_reused_worker_equal_fresh_processes` failed once in the full suite (3ad9ef6 plus
  the reader change); a standalone rerun of the same check gave identical records, distinct pids
  and one reused worker. With a single-worker pool only pid reuse by Windows explains the failure;
  while the reused worker lives, its pid cannot be handed out again. The test now prints the
  per-task result on failure. Results of the run are not affected (the function is used only by the
  check and the test).

## 9. Runs and controls

| stage | script, commit | artifact | units, wall time |
|---|---|---|---|
| training | `train_hybrid_model.py`, de3079f | `hybrid_model_training_v1.jsonl.gz` (3207796) | 1680 tasks, 36438 s |
| oracle + discretisation check | `run_hybrid_evaluation.py --part oracle`, a343840 | `hybrid_model_v1_oracle.jsonl` (3207796) | 20 draws + 10 check cases, 580 s |
| margins | `--part margins`, 1e2a635 | `hybrid_model_v1_margins_per_draw.jsonl.gz` (3ad9ef6) | 2400 tasks, 0 errors, 60/60 cells, 181751 s |
| errors | `--part errors`, aafc95f | `hybrid_model_v1_errors.jsonl.gz` (e3ac41f) | 60 tasks, 0 errors, 2084 s |
| analysis | `analyze_hybrid_model.py`, f0d8120 | `hybrid_model_v1_analysis.json` (f7ad0be) | -- |

Every artifact records `git/dirty=False` and the code location of its worktree. The sha256 of the
training records and of the oracle are pinned in the headers of the margins and errors stages and
match. The two packed artifacts were checked after packing (sha256 of the unpacked content equal to
the raw file; raw files kept under `C:/twinlink_runs/hybrid-model/`). Checkpoints (gitignored):
1440 files of the main training and 12 of the pilot, copied to `C:/twinlink_runs/hybrid-model/`
(file names and total size, respectively sha256, checked).

**Controls.**

* Tests (`tests/test_hybrid_model.py`): M3 with a zero residual equals M0 exactly
  (`test_m3_with_zero_residual_equals_m0_exactly`); M2 with the true parameters hits the truth,
  steady and dynamic (`test_m2_with_true_parameters_hits_the_truth_steady`, `..._dynamic`); the
  batched steady solver equals the bandit evaluation; the refit recovers the true parameters from
  noise-free runs; the per-draw driver, the no-inside-case path, the raise rule, the resume and the
  worker reuse (section 8). Full suite on f0d8120: 1418 passed, 30 skipped.
* **Discretisation** (addendum 1): over the 10 check cases the oracle by grid + zoom differs from the
  8-start Nelder-Mead by at most **0.317 %** in energy (one case; the other nine at most 0.113 %) and
  at most 1.6e-4 kg/kg in slack. This is above 0.2 %, so every energy figure below is to be read with
  this discretisation of the oracle (marked "disc. 0.317 %").
* **Determinability** (addendum ac8173a): pooled inside cases per (seed, `n_train`) 29 (s2, n 4) to
  94 (s1, n 32); every one of the 60 cells has at least 20 and is determinable; no `n_train` drops out.
  The inside cases do not depend on the model.
* The raise rule of Befund 111 was not needed in any cell (`raised` 0 everywhere); every cell is
  decided at its first candidate or at the upper end ("not reached").

## 10. Results

All numbers from `artifacts/runs/hybrid_model_v1_analysis.json`. Median over the 3 master seeds,
range in brackets; margins in kg/kg; "not reached" counts as the upper end 0.12 (censored) as
pre-registered, the number of such seeds is given. Test sets named per table.

**Margin (95 % of the inside cases holding), M0-M3.**

| `n_train` | inside cases (s1/s2/s3) | M0 | M1 | M2 | M3 |
|---|---|---|---|---|---|
| 4 | 36 / 29 / 38 | 0.0617 [0.0440, 0.12] (1 n.r.) | 0.1173 [0.0616, 0.12] (1 n.r.) | 0.0019 [0.0012, 0.0534] | 0.0375 [0.0180, 0.1192] |
| 8 | 54 / 35 / 50 | 0.1071 [0.0440, 0.12] (1 n.r.) | 0.12 [0.0496, 0.12] (2 n.r.) | 0.0032 [0.0025, 0.0534] | 0.1089 [0.0111, 0.1195] |
| 16 | 68 / 63 / 77 | 0.0622 [0.0602, 0.0623] | 0.0456 [0.0225, 0.0630] | 0.0039 [0.0035, 0.0057] | 0.0374 [0.0111, 0.12] (1 n.r.) |
| 32 | 94 / 82 / 91 | 0.0622 [0.0444, 0.0622] | 0.0318 [0.0137, 0.0410] | 0.0032 [0.0031, 0.0034] | 0.0195 [0.0123, 0.12] (1 n.r.) |

**M4** (ensemble, k on [0, 10]): k reached in 1 / 2 / 1 / 2 of 3 seeds at `n_train` 4 / 8 / 16 / 32;
where reached, effective margin (median `k std` over the inside cases) 0.0125 (n 4), 0.0278
[0.0074, 0.0481] (n 8), 0.0170 (n 16), 0.0066 [0.0054, 0.0078] (n 32), with k 3.04, 7.65, 8.09,
3.89 (medians). Where k = 10 does not reach 95 %, the holding share at k = 10 is 0.734-0.944. M4 is
not part of the reading.

**Pre-registered reading (n_train 4, 8, 16; all cells determinable).**

| `n_train` | half criterion `m_M3 <= 0.5 m_M1` | paired M3 vs M1: mean difference (M1 - M3), threshold, p | resolved |
|---|---|---|---|
| 4 | yes (0.0375 vs 0.1173) | 0.0414, 0.1049, 0.23 | no |
| 8 | no (0.1089 vs 0.12) | 0.0167, 0.0487, 0.28 | no |
| 16 | no (0.0374 vs 0.0456) | -0.0125, 0.1344, 0.73 | no |

* **Y0 not met** (half criterion only at `n_train` 4, and not resolved there).
* **Y2 holds** (verdict of the analysis: "Y2"): at no `n_train` <= 16 is the margin of M3 resolvably
  smaller than that of M1. Pre-registered meaning: no advantage of the hybrid in this setting.
* **Y1 not met**: the margin of M2 is not within 10 % of M3 at any `n_train` (M2 0.0019 / 0.0032 /
  0.0039 against M3 0.0375 / 0.1089 / 0.0374); the energy part holds at `n_train` 4 and 16, not at 8
  (relative energy below). Paired M2 vs M3 (margin): differences 0.0394 / 0.0601 / 0.0518 against
  thresholds 0.0620 / 0.1212 / 0.1435, not resolved. Y1 is an equivalence statement; its failure
  here means M2 and M3 are not equivalent, not that one ranks above the other.

**Energy relative to the oracle** (median over the inside cases with a valid model point; disc.
0.317 %).

| `n_train` | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| 4 | 1.137 [1.098, 1.668] | 1.321 [1.059, 1.513] | 1.005 [1.003, 1.149] | 1.086 [1.035, 1.679] | 1.031 [1.027, 1.337] |
| 8 | 1.407 [1.123, 1.628] | 1.285 [1.095, 1.409] | 1.008 [1.007, 1.154] | 1.464 [1.026, 1.702] | 1.031 [1.026, 1.107] |
| 16 | 1.164 [1.153, 1.167] | 1.065 [1.033, 1.141] | 1.009 [1.008, 1.012] | 1.089 [1.022, 1.670] | 1.065 [1.040, 1.073] |
| 32 | 1.157 [1.101, 1.159] | 1.066 [1.023, 1.086] | 1.007 [1.007, 1.007] | 1.059 [1.034, 1.712] | 1.059 [1.021, 1.069] |

For cells that did not reach, the energy is the one at the upper end of the margin (or k = 10).

**No valid model point.** M0-M2: none. M3: 0 / 1 / 11 / 14 cases (of them inside 0 / 0 / 6 / 10) at
`n_train` 4 / 8 / 16 / 32, all in seed 1; M4: 0 / 2 / 33 / 30 (inside 0 / 0 / 17 / 19), seed 1 except 7
(inside 5) in seed 3 at `n_train` 16.
These cases count as not holding; they are the reason why M3 s1 does not reach at `n_train` 16
(holding share 0.882 at 0.12) and 32 (0.883). Bed above 80.05 degC at the chosen point: M1 2
(`n_train` 8), M3 5 / 2 / 1 / 3, M4 4 / 2 / 1 / 0 cases; M0 and M2 none.

**Outside: violation share at exactly the model's parameter** (pooled over the seeds; (a) 120 cases
per `n_train`, (b) 197 / 161 / 92 / 33 cases at `n_train` 4 / 8 / 16 / 32). "Holds" = share <= 5 %.

| `n_train` | M0 (a) / (b) | M1 (a) / (b) | M2 (a) / (b) | M3 (a) / (b) | M4 (a) / (b) |
|---|---|---|---|---|---|
| 4 | 7.5 / 6.1 % | 9.2 / 7.6 % | 12.5 / 9.1 % | 9.2 / 11.2 % | 10.0 / 6.1 % |
| 8 | 6.7 / 4.3 % (b holds) | 10.0 / 3.7 % (b holds) | 9.2 / 5.6 % | 10.8 / 4.3 % (b holds) | 12.5 / 6.2 % |
| 16 | 8.3 / 5.4 % | 11.7 / 5.4 % | 5.0 / 1.1 % (both hold) | 10.8 / 5.4 % | 15.8 / 12.0 % |
| 32 | 9.2 / 12.1 % | 25.0 / 18.2 % | 5.0 / 0.0 % (both hold) | 10.0 / 6.1 % | 21.7 / 18.2 % |

**Moisture bias at the chosen point, inside cases** (predicted - true, kg/kg, median over seeds of
the per-cell medians): M0 -0.0067 / -0.0026 / -0.0079 / -0.0079; M1 -0.0368 / -0.0222 / -0.0123 /
-0.0046; M2 +0.0001 / +0.0002 / -0.0006 / -0.0003; M3 -0.0019 / -0.0010 / -0.0033 / -0.0011; M4
-0.0019 / -0.0024 / -0.0025 / -0.0015 (`n_train` 4 / 8 / 16 / 32). Box-wide on the coarse grid,
inside: M1 -0.0208 / -0.0093 / -0.0080 / -0.0029, M3 -0.0009 / 0.0000 / -0.0005 / -0.0001.

**One-step and rollout error, product moisture** (RMSE kg/kg, median over 3 seeds x 20 draws, range
in brackets; M0/M2 open-loop replay, so h1 and h36 are the same quantity at different windows).

| `n_train` | excited test set, h36: M1 | M2 | M3 | M4 | constant-run holdout, h36: M1 | M2 | M3 |
|---|---|---|---|---|---|---|---|
| 4 | 0.0451 [0.0343, 0.0684] | 0.0097 [0.0004, 0.0694] | 0.0037 [0.0014, 0.0093] | 0.0034 | 0.0618 | 0.0161 | 0.0044 |
| 8 | 0.0259 [0.0105, 0.0546] | 0.0120 [0.0003, 0.0694] | 0.0033 [0.0014, 0.0082] | 0.0031 | 0.0362 | 0.0235 | 0.0042 |
| 16 | 0.0085 [0.0049, 0.0138] | 0.0013 [0.0003, 0.0261] | 0.0025 [0.0008, 0.0054] | 0.0025 | 0.0124 | 0.0016 | 0.0030 |
| 32 | 0.0041 [0.0027, 0.0121] | 0.0010 [0.0003, 0.0028] | 0.0020 [0.0004, 0.0044] | 0.0020 | 0.0080 | 0.0013 | 0.0026 |

M0 (no data): 0.0206 [0.0053, 0.0614] on the excited test set, 0.0284 on the constant-run holdout.
At h1 M1 is 0.0008-0.0019 and M3 0.0008 (excited). On the constant-run holdout with widened ambient
ranges (runs outside the data box, (a)) the figures differ from the plain holdout by at most 0.002
(M1 0.0598 / 0.0364 / 0.0141 / 0.0085, M3 0.0045 / 0.0043 / 0.0032 / 0.0028). The full table
(also inside / outside runs, bed temperature and energy target) is in the artifact under `errors`.

**Answer to the question.** In this setting the hybrid model saves no resolvable safety margin
against the pure GRU at `n_train` <= 16 (pre-registered reading Y2), although its rollout error of
the moisture (h36) is 8-12 times smaller than that of the GRU at `n_train` 4 and 8 (excited test
set).
The refit of critical moisture and nu_max (M2) needs margins of 0.0019-0.0039 kg/kg (medians) and lies within
1 % of the oracle energy in the median (disc. 0.317 %); it is not equivalent to the hybrid (Y1 not
met), and the paired comparison M2 vs M3 is not resolved either.

## 11. Caveats

* **Resolution.** Three master seeds give paired thresholds of 0.049-0.134 kg/kg for M3 vs M1, of the
  order of the margins themselves. Y2 is the pre-registered reading ("not resolvably better"); it is
  not a demonstration that M3 and M1 need the same margin.
* **Censoring.** "Not reached" counts as 0.12. At `n_train` 8 two of three M1 seeds do not reach, so
  the median of M1 there is the bound itself; the half criterion at 8 compares 0.1089 with this bound.
* **Invalid fixed points.** The residual networks of M3 (seed 1) and the ensemble M4 lose valid
  fixed points at `n_train` 16 and 32; those cases count as not holding and decide the "not reached"
  of these cells. The margin then measures fixed-point validity as much as the moisture error.
* **Truth family favours M2.** The truth is the same grey-box structure with other material
  parameters, and M2 refits two of the seven varied parameters (critical moisture and nu_max, the
  ones that dominate the mismatch per Befund 111). The small M2 margins are not to be expected at a
  plant with a structural mismatch (Befund 117).
* **Rollout error vs margin.** The much smaller rollout error of M3 does not show up as a smaller
  margin: the margin is set by the steady state near the moisture limit at the 95 % quantile of the
  inside cases, not by the mean trajectory error. This is a descriptive observation, not a
  resolved rank.
* Discretisation of the oracle 0.317 % in one of 10 check cases (stated with every energy figure).
  Moisture limit 0.15 kg/kg is a placeholder; one-step operating-point problem only; one lab line;
  20 draws; inside / outside by the Chebyshev threshold 0.2 chosen before the run.
* Course of the run (section 8): the margins ran at 1e2a635, the errors at aafc95f (only the
  analysis reader and the reuse check changed in between, no computation); the 15-h pause before the
  errors stage had no effect on any result.
