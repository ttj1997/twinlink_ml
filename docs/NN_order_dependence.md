# Order dependence of the steady-state optimiser

Status: design, reading and code committed before the first measurement run; results in
sections 9 to 13. Branch `order-dependence` (worktree `C:/twinlink_work/orderdep`, from
`origin/master` 61d5b18). Document number open (`NN`; assigned by the managing session at merge).
Lab-book entry: "Befund NN (number open)".

**Placeholders.** Every number below inherits the placeholders of Befund 105 and Befund 126 (lab
line, quality limit 0.15 kg/kg as a path condition, scenario amplitudes, tariff and price file of
Befund 115). The truth is the simulator. No statement about a plant.

## 1. Question

The control of the learned-setpoints study (branch `learned-setpoints`, section 15 of its
document) found that its run of the schedule C2 is not bit-equal to the records of the
load-following artifact, and named the cause:

* `headroom.controllers.SteadyStateOptimizer.solve` starts every solve from the solution of the
  previous one;
* the caches in front of it use rounded keys (`load_following.steady.SteadyMap`: 10 decimals;
  `QuasiStationaryController`: 12 decimals).

The steady optimum returned for a query therefore depends on the queries the worker process has
solved before. This document measures

1. how often and by how much the result of a query depends on the order, and how often the warm
   start returns a *worse* local optimum than a cold multi-start;
2. whether the key numbers that rest on this optimiser (Befund 105, Befund 126, the twin of the
   learned-setpoints study) change when the order dependence is removed;
3. a correction as a switch (default off, earlier results bit-equal).

Existing results and artifacts are not changed.

## 2. What rests on the steady-state optimiser

| use | where | finding |
|---|---|---|
| R0 (fixed point), R1 (quasi-stationary RTO) | `headroom.study` | Befund 105 |
| initial state of every run, start guess and safeguard of R2 | `headroom.study`, `PreviewController` | Befund 105 |
| steady-state map: C0, C1, C2, the tail of every run, start of C3 and C4 | `load_following.steady.SteadyMap` | Befund 126 |
| stage 0 of the load-following study (admissible range, bands, tuning of C1) | `load_following.stage0` | Befund 126 |
| twin C2 (and C0) of the learned-setpoints study | runner of that study, equal to `run_days` by its own test | learned set points |

Recomputed here: the reference controllers R0, R1, C0, C1, C2 only (task). Not recomputed: every
controller with a dynamic optimiser (R2, R3, C3, C4), the learned arms, the stress, product-change
and robustness stages of Befund 105, the variants G, box, weekly and constant price of Befund 126.
Stage 0 is checked descriptively (section 5, part `stage0_check`), not replaced.

## 3. The switch (proposal, default off)

`SteadyStateOptimizer(study, process, *, cold_starts=0, start_seed=COLD_START_SEED)`.

* `cold_starts = 0` (default): the code path of Befund 105 and Befund 126, unchanged.
* `cold_starts = n >= 1`: every solve is a function of its query alone. `n` solves from fixed
  start points: start 0 is the default guess of the existing code (`m_w` 0.25 kg, bed 45 degC,
  inlet 90 degC, gas velocity at its minimum); the others are a Latin hypercube with a fixed seed
  (20261120) over moisture 0.02 to 0.30 kg/kg, bed temperature 25 degC to the bed limit and the
  action box of inlet temperature and gas velocity. The start points are the same for every
  query. The first start is the incumbent; a later start replaces it only if its power is lower
  by more than 1e-6 relative (the solver tolerance), so that solver noise does not move the set
  points; a strictly converged solve (`Solve_Succeeded`) is preferred to one at the acceptable
  level. Nothing is remembered between solves.
* With an order-independent optimiser the two caches use the **unrounded** query as their key
  (`SteadyStateOptimizer.order_independent`); without the switch they keep their rounded keys.
* `headroom.study.Solvers` and `load_following.study.Solvers` pass the switch
  (`steady_cold_starts`, default 0); `SteadyMap` takes an optimiser (`optimizer=`, default: the
  one of Befund 126). `solve_from` (one solve from a given start, no retry, no memory) is public
  for the measurement.

Nothing is set as a default. The config dataclasses and their fingerprints are untouched.

**Evidence that the default is unchanged.**

* Test `test_default_is_the_warm_start_of_befund_105_bit_for_bit`: the default path against a
  copy of the old algorithm kept in the test file, 60 queries, bit equality of power, set points
  and state. Tests for the rounded keys and for the unchanged `Solvers`.
* One-off check before the commit (scratch, not an artifact): the same script run with the code
  of `origin/master` 61d5b18 (extracted with `git archive`) and with the worktree: a warm chain of
  400 queries, R1 / R0 / R1 on four scenarios of one hour, C0 / C2 / C1 / C2 on three days of two
  hours; the SHA-256 over all results (floats as hex) is the same for both
  (`4b8550a09fc2fef8...`).
* The config-fingerprint tests and the full suite (run by the chain on the commit of the run).

## 4. Pre-registered reading

Committed with the driver before the dry run, the probe and the main run: docstring of
`scripts/analyze_order_dependence.py`. In short:

| key number | finding | definition | printed |
|---|---|---|---|
| K1 | Befund 105 | median over the 105 observed scenarios of `(E_R0 - E_R1) / E_R0`, each at its own margin | 8.93 % |
| K2 | Befund 126 | `(J_C0 - J_C1) / J_C0` | 2.07 % |
| K3 | Befund 126 | `(J_C0 - J_C2) / J_C0` | 2.06 % |
| K4 | learned set points | cost of the twin, `J_C2` at the margin of C2, tail at the margin of C0 | 0.39794 EUR/kg |
| K5 | learned set points | smallest median gap of a learned arm against the twin (D-gru, largest budget), costs of the arm held | 2.98 % |

"Old" = computed from the records of the artifacts; "new" = the reference controllers with the
cold multi-start, in two variants: **held** (at the margins of the artifacts) and **searched**
(the margin searches of the findings repeated with the cold multi-start). `change = |new - old| /
|old|`.

* **O0**: no change above 10 % -> the effect is immaterial for the reported statements; noted in
  the lab book.
* **O1**: at least one change above 10 % -> a dated addendum to the finding concerned is due
  (append-only, by the managing session).

Invalid if a unit of the two parts ended with an error or a search did not converge. Descriptive
and without a reading: frequency and size of worse local optima (part `queries`), the second run
of the unchanged code, the repeat with other chunks, the stage-0 check.

## 5. Design of the measurement

Driver `scripts/run_order_dependence.py`, package `src/twinlink_ml/order_dependence/`, settings
`OrderDependenceSettings` (8 starts, reference 64 starts, three seeded permutations). Units: the
105 scenarios of the family "observed" and the 40 test days; band and rule of C1 from the stage-0
artifact; "artifact margins" from `dynamic_headroom_v2.json.gz` and `load_following_v1.json.gz`
(configs compared with the artifacts by their fingerprints before every run).

| part | what | for |
|---|---|---|
| `warm_rerun` | R0, R1, C0, C1, C2 at the artifact margins, unchanged code, fresh workers, chunks of this run | the order dependence between two runs of the same code |
| `cold_held` | the same with the cold multi-start | key numbers, variant held |
| `cold_search` | margin searches with the cold multi-start (same bisection, bracket, tolerances) | key numbers, variant searched |
| `cold_repeat` | `cold_held` again: reversed order, other chunk size, workers with another history | with the switch the records must be bit-equal |
| `queries` | per controller (R1, C0, C1, C2) and chunk: the queries recorded in lockstep on a fresh optimiser, solved again warm in five orders, cold with one start, with the multi-start and with every start of the dense reference | frequency and size of worse local optima |
| `stage0_check` | admissible range, bands and tuning of C1 on the map with the cold multi-start | whether stage 0 would have chosen differently |

Per query all relative differences are taken against the multi-start power `P_m`:
`(P - P_m) / P_m`; positive = worse than the multi-start. Thresholds 1e-9 (the "not equal" of the
learned-setpoints control), 1e-6 (solver tolerance), 1e-4, 1e-3, 1e-2.

## 6. Decisions taken without asking (to be confirmed by the project lead)

1. **K5 added.** The task names "the twin C2" as a key number. Its cost cannot change by 10 %;
   the statements of that study are gaps of a few per cent against the twin. K5 (the smallest
   reported gap, recomputed with the new twin and the costs of the arm held) makes the twin count
   in the reading; K4 (the cost itself) is kept as the literal reading of the task.
2. **Both variants count.** The task says "by recomputing the reference controllers"; whether at
   the old margins or with their margin searches is left open. Both are run; a change above 10 %
   in either gives O1.
3. **The measurement uses the proposed switch**, so that what is measured is what is proposed.
   The existing code is touched only by the default-off switch (section 3).
4. **Multi-start**: 8 starts as in the task; start 0 = the default guess; Latin hypercube with
   seed 20261120; incumbent rule with 1e-6; strict convergence preferred (section 3).
5. **Natural order** = the order in which one fresh optimiser receives the queries of a chunk run
   in lockstep (the tasks of the drivers of the findings). Other orders: reversed and three
   seeded permutations, each on a fresh optimiser.
6. **Dense reference**: 64 starts (seed 20261121), plain minimum over the strictly converged
   starts; local optima counted as clusters of the set points (0.05 K, 5e-4 m/s).
7. **Stage 0 held**: band and rule of C1 from the stage-0 artifact; `stage0_check` reports what
   the cold map would have given.
8. **Twin**: C2 with the code of master at the margin of the load-following artifact (the
   package of the learned-setpoints study is not on master; by its own test its runner equals
   `run_days`). Old values of K4 and K5 from the two artifacts of that branch (commit 2e630e7),
   read from copies under `local/` and recorded with their SHA-256.
9. **No budget** was set by the task; the estimate of the probe is logged before the main run.
10. **Smoke runs before the commit of the reading**: two runs of the driver on 3 and 5 units
    shortened to 2 h, into a scratch directory, to debug the driver and the analysis. No number
    of them is used.

## 7. Tests

`tests/test_order_dependence.py` (15): default path bit-equal to the algorithm of Befund 105;
`solve_from` leaves the warm start alone; rounded keys without the switch; `Solvers` unchanged by
default and passing the switch; with the switch: function of the query alone (other order, other
history), incumbent rule against every start, one start = the default guess, unrounded keys, R1
and C2 independent of what the solvers ran before and of the composition of the run; recording
and measuring of queries; statistics on a hand-made case; key numbers and the boundary of the
reading; chunks and record comparison.

## 8. Course of the run

(filled in after the run)
