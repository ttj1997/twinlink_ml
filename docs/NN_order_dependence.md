# Order dependence of the steady-state optimiser

Status: design, reading and code committed before the first measurement run; run of 2026-11-20
complete, pre-registered reading **O0** (valid; largest relative change of a key number 3.78 %,
threshold 10 %); results in sections 8 to 13. Not merged, not pushed. Branch `order-dependence` (worktree `C:/twinlink_work/orderdep`, from
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

All times 2026-11-20, local time. Chain `C:/twinlink_work/orderdep_chain.sh`, started detached
from the session (WMI process creation), 4 workers, log `C:/twinlink_work/orderdep_chain.log`.

| step | time | commit | result |
|---|---|---|---|
| code, reading, tests, design of this document | before the chain | 7be6144 | clean tree |
| full suite | 09:56 to 10:29 | on 7be6144 | 1580 passed, 32 skipped |
| dry run over all units (2 h per unit, margin tolerance 4e-3) | 10:29 to 10:49 (19.7 min) | on 7be6144, output f18b924 | 2672 units, 0 errors, peak 153 MB per worker, cold repeat bit-equal in every unit |
| probe at full length and full batch size | 10:50 to 11:27 (36.5 min) | on f18b924, output 9acd22c | 202 units, 0 errors, peak 271 MB per worker (4 x 271 MB = 1084 MB of 31912 MB), estimate of the main run 6.0 h |
| main run | 11:28 to 15:43 (255 min) | on 9acd22c, output 060c6bc | 3987 units, 0 errors, one run of the stage, nothing cut, peak 438 MB per worker |
| analysis | 15:44 to 15:45 | on 060c6bc, output f21929c | reading O0, valid |

Parts of the main run: `warm_rerun` 8.6 min, `cold_held` 14.6 min, `cold_search` 133.6 min,
`cold_repeat` 19.6 min, `queries` 66.9 min, `stage0_check` 11.8 min. All five margin searches
converged. The peak memory of the main run (438 MB per worker) is above that of the probe
(271 MB); four workers stay far below 80 % of the RAM in both.

Artifacts: `artifacts/runs/order_dependence_v1_dryrun.json.gz`, `..._probe.json.gz`,
`order_dependence_v1.json.gz`, `order_dependence_v1_analysis.json`. Every number of sections 9 to
12 is from `order_dependence_v1_analysis.json` unless another artifact is named.

## 9. Result: deviations per query

94211 queries from 16 chunks (R1 24115, C0 11534, C1 11542, C2 47020), each chunk recorded in
lockstep on a fresh optimiser. The warm pass in the natural order reproduced the recording bit
for bit in all 16 chunks. No solve failed in any pass (warm in five orders, cold single start,
multi-start, every start of the dense reference).

Shares in % of the queries; `tau` is the threshold of the relative power difference.

| source | n | tau | differing between orders | warm natural worse | warm natural better | any order worse | cold single worse | reference better |
|---|---|---|---|---|---|---|---|---|
| R1 | 24115 | 1e-09 | 11.79 | 3.30 | 0.98 | 7.05 | 1.03 | 20.66 |
| R1 | 24115 | 1e-06 | 5.39 | 1.78 | 0.00 | 5.58 | 1.03 | 0.11 |
| R1 | 24115 | 0.0001 | 5.37 | 1.77 | 0.00 | 5.56 | 1.02 | 0.00 |
| R1 | 24115 | 0.001 | 5.23 | 1.67 | 0.00 | 5.40 | 0.92 | 0.00 |
| R1 | 24115 | 0.01 | 0.10 | 0.05 | 0.00 | 0.10 | 0.00 | 0.00 |
| C0 | 11534 | 1e-09 | 6.75 | 3.15 | 1.18 | 5.50 | 2.10 | 20.34 |
| C0 | 11534 | 1e-06 | 2.79 | 1.65 | 0.00 | 3.26 | 2.10 | 0.23 |
| C0 | 11534 | 0.0001 | 2.71 | 1.52 | 0.00 | 3.14 | 2.00 | 0.00 |
| C0 | 11534 | 0.001 | 2.45 | 1.35 | 0.00 | 2.83 | 1.82 | 0.00 |
| C0 | 11534 | 0.01 | 0.17 | 0.06 | 0.00 | 0.18 | 0.05 | 0.00 |
| C1 | 11542 | 1e-09 | 13.80 | 4.87 | 3.38 | 9.69 | 1.44 | 26.42 |
| C1 | 11542 | 1e-06 | 3.64 | 1.89 | 0.02 | 3.84 | 1.44 | 0.36 |
| C1 | 11542 | 0.0001 | 3.54 | 1.77 | 0.00 | 3.73 | 1.39 | 0.00 |
| C1 | 11542 | 0.001 | 3.25 | 1.58 | 0.00 | 3.35 | 1.19 | 0.00 |
| C1 | 11542 | 0.01 | 1.07 | 0.68 | 0.00 | 1.10 | 0.14 | 0.00 |
| C2 | 47020 | 1e-09 | 11.36 | 4.23 | 1.98 | 8.00 | 1.76 | 21.39 |
| C2 | 47020 | 1e-06 | 3.97 | 1.86 | 0.01 | 4.02 | 1.76 | 0.28 |
| C2 | 47020 | 0.0001 | 3.90 | 1.74 | 0.00 | 3.95 | 1.71 | 0.00 |
| C2 | 47020 | 0.001 | 3.45 | 1.49 | 0.00 | 3.48 | 1.47 | 0.00 |
| C2 | 47020 | 0.01 | 0.55 | 0.22 | 0.00 | 0.55 | 0.09 | 0.00 |
| pooled | 94211 | 1e-09 | 11.21 | 3.94 | 1.80 | 7.66 | 1.58 | 21.69 |
| pooled | 94211 | 1e-06 | 4.15 | 1.82 | 0.01 | 4.31 | 1.58 | 0.24 |
| pooled | 94211 | 0.0001 | 4.09 | 1.72 | 0.00 | 4.24 | 1.53 | 0.00 |
| pooled | 94211 | 0.001 | 3.76 | 1.53 | 0.00 | 3.88 | 1.34 | 0.00 |
| pooled | 94211 | 0.01 | 0.45 | 0.21 | 0.00 | 0.46 | 0.07 | 0.00 |

Reading of the table (descriptive, no pre-registered reading):

* **How often.** No query is bit-equal over the five warm orders. 11.21 % of the queries differ
  between the orders by more than 1e-9 in the power (the "not equal" of the learned-setpoints
  control), 4.15 % by more than the solver tolerance 1e-6, 3.76 % by more than 0.1 %, 0.45 % by
  more than 1 %.
* **Two populations.** Between 1e-9 and 1e-6 the differences are solver noise around the same
  optimum: of the 3710 queries in which the warm start in the natural order is worse than the
  multi-start by more than 1e-9, the median excess is 7.6e-8 and the median set-point difference
  is 5.5e-6 K. Above 1e-6 they are other local optima: 1712 queries (1.82 %), median excess of
  the power 0.46 %, 90 % quantile 1.06 %, 99 % quantile 1.90 %, largest 2.58 %; inlet temperature
  different by 26.6 K in the median (largest 42.6 K), gas velocity by 0.54 m/s (largest 1.09 m/s).
* **Worse, not only different.** The warm start in the natural order returns a worse optimum than
  the cold multi-start in 1.82 % of the queries (above 1e-6) and a better one in 0.007 %; in at
  least one of the five orders it is worse in 4.31 %. By more than 1 % of the power: 0.21 % of
  the queries (201; C1 0.68 %, C2 0.22 %, C0 0.06 %, R1 0.05 %).
* **The warm start is not the only cause.** A single cold start from the default guess is worse
  than the multi-start in 1.58 % of the queries (above 1e-6). Removing the memory alone (cold
  single start, unrounded keys) would make the result a function of the query but would keep
  worse local optima at about the same frequency.
* **Local optima.** In the dense reference (64 starts) 6.91 % of the queries have several local
  optima (R1 9.51 %, C0 6.54 %, C1 5.39 %, C2 6.04 %), up to 7; where there are several, the
  power of the worst is above that of the best by 0.56 % in the median, 2.66 % at the 99 %
  quantile, 3.67 % at most.
* **Eight starts are enough at this resolution.** The dense reference is better than the
  multi-start of 8 starts by more than 1e-6 in 0.24 % of the queries and by more than 1e-4 in
  none.

## 10. Result: key numbers and reading

| key number | old | held | change % | searched | change % |
|---|---|---|---|---|---|
| K1 | 0.089320 | 0.089320 | 0.000 | 0.090967 | 1.844 |
| K2 | 0.020706 | 0.020832 | 0.610 | 0.020832 | 0.610 |
| K3 | 0.020622 | 0.020800 | 0.864 | 0.019842 | 3.782 |
| K4 | 0.397944 | 0.397855 | 0.022 | 0.398244 | 0.075 |
| K5 | 0.029775 | 0.030007 | 0.778 | 0.029000 | 2.603 |

**Reading: O0 (valid, with the twin).** The largest relative change is 3.78 % (K3, variant
searched); the threshold is 10 %. No unit of `cold_held` or `cold_search` ended with an error, all
five searches converged. By the pre-registered reading the effect is immaterial for the reported
statements and is noted in the lab book; no addendum to Befund 105, Befund 126 or the
learned-setpoints finding is due from this reading.

Margins and the limit (kg/kg; tolerance of the violation 1e-4, tolerance of the margin 2.5e-4,
both from the headroom config):

| controller | margin of the artifact | margin searched (cold multi-start) | largest violation at the artifact margin (cold multi-start) | holds |
|---|---|---|---|---|
| R0 | 0.043555 | 0.043555 | 0 | yes |
| R1 | 0.006641 | 0.005859 | 0 | yes |
| C0 | 0.005469 | 0.005469 | 0 | yes |
| C1 | 0.007422 | 0.007422 | 9.44e-05 | yes |
| C2 | 0.007422 | 0.007812 | 4.01e-04 | **no** |

Below the threshold, but named because they touch printed numbers:

* **C2 does not hold the limit at the margin of the artifact** when its set points come from the
  cold multi-start (largest violation 4.01e-4 kg/kg against the tolerance 1e-4). The values of
  K3, K4 and K5 in the variant held are therefore not costs at a held constraint; the variant
  searched is (C2 at 0.007812, largest violation 9.37e-5). There the gain of the schedule C2
  against C0 is 1.98 % (artifact 2.06 %), that of the rule C1 2.08 % (artifact 2.07 %); costs in
  EUR per kg: C0 0.406306 (artifact 0.406330), C1 0.397842 (0.397917), C2 0.398244 (0.397950).
* **R1 holds the limit with a smaller margin** (0.005859 instead of 0.006641); the median saving
  against R0 becomes 9.10 % (artifact 8.93 %). At the artifact margin the median is unchanged to
  ten digits although 17 of the 105 scenarios differ (section 11): the median scenario is not
  among them.
* **Twin.** Its cost changes by 0.02 % (held) and 0.08 % (searched); the smallest median gap of a
  learned arm (K5, costs of the arm held) is 3.00 % (held) and 2.90 % (searched) against 2.98 %.

## 11. Controls (descriptive)

**Second run of the unchanged code (`warm_rerun`), against the records of the artifacts.** Units
whose energy (R0, R1) or cost (C0, C1, C2) differs by more than 1e-9 relative, and the range of
the relative difference:

| controller | units differing | range |
|---|---|---|
| R0 | 0 of 105 | none |
| R1 | 4 of 105 | -0.81 % to +0.010 % |
| C0 | 4 of 40 | -0.017 % to +0.004 % |
| C1 | 4 of 40 | -0.035 % to +0.069 % |
| C2 | 6 of 40 | -0.053 % to +0.099 % |

The unchanged code with other chunks does not reproduce the artifacts bit for bit; this is the
order dependence at the level of the results. The key numbers of this second run differ from the
old ones by 0.00 % (K1), 0.10 % (K2), 0.17 % (K3), 0.005 % (K4) and 0.16 % (K5) relative. All
five controllers hold the limit in it (largest violation 9.29e-5 for C1, 7.67e-5 for C2).

**Cold multi-start at the artifact margins, against the records of the artifacts.** R0 0 of 105;
R1 17 of 105 (-0.81 % to +0.004 %); C0 15 of 40 (-0.10 % to +0.027 %); C1 17 of 40 (-0.14 % to
+0.041 %); C2 17 of 40 (-0.18 % to +0.046 %). Set points of R1 per interval, warm rerun against
cold multi-start: 1.42 % of the intervals differ (more than 0.05 K or 5e-4 m/s), by up to 31.8 K
and 0.856 m/s.

**Repeat with the switch (`cold_repeat`).** Reversed order of the units, another chunk size,
workers with another history: the records are bit-equal in every unit (R1 105 of 105, R0 105 of
105, C0 40 of 40, C1 40 of 40, C2 40 of 40). With the switch the result does not depend on the
order.

**Twin of the learned-setpoints study.** Its C2 records (artifact of that branch) differ from
those of the load-following artifact on 17 of 40 days (-0.13 % to +0.15 % of the cost per day);
this reproduces the observation that started this study. The cold multi-start at the same margin
differs from the twin records on 18 of 40 days (-0.15 % to +0.011 %).

**Stage 0 on the map with the cold multi-start (`stage0_check`).** Admissible range and bands
unchanged (main 1.2 to 1.8 kg/h, box 0.4 to 3.0 kg/h); the tuned rule of C1 unchanged for both
carriers (quantile 0.5, spread 0.6 kg/h; relative change of its cost on the training days 8.6e-5
for E, 7.4e-4 for G). Map: 1325 points, an optimum at every point in both, no feasibility flag
differs, no failure of the steady solve. Power of the artifact minus cold multi-start: up to
+2.41 % and down to -0.033 %; 1.74 % of the points above 1e-6, 1.51 % above 1e-3, 0.08 % below
-1e-6. Stage 0 would have chosen the same band and the same rule.

## 12. The switch as measured

* **Effect.** With `cold_starts = 8` every steady optimum is a function of its query; records are
  bit-equal over chunk layouts, orders and worker histories (section 11). Without the switch the
  default path is bit-equal to the code of `origin/master` 61d5b18 (section 3).
* **Quality.** Against the warm start it is better by more than 1e-6 in 1.82 % of the queries and
  worse in 0.007 %; a dense reference of 64 starts improves on it by more than 1e-4 in no query
  (section 9).
* **Cost.** The five reference controllers at the artifact margins took 515 s with the unchanged
  code and 873 s with the switch on the same 4 workers (`part_runtime_s` of
  `order_dependence_v1.json.gz`); the truth simulation dominates both.
* **Not done.** The switch is not a default anywhere; no config dataclass, fingerprint, artifact
  or finding was changed. Whether new studies set `steady_cold_starts` is a decision of the
  project lead (section 13).

## 13. Caveats and open decisions

Caveats:

1. **Scope of the recomputation.** Only R0, R1, C0, C1, C2. Every controller with a dynamic
   optimiser (R2, R3, C3, C4) takes its initial state, tail, start guess or safeguard from the
   same steady optimiser and was not recomputed; the same holds for the learned arms (trained on
   set points of the warm twin), the stress, product-change and robustness stages of Befund 105
   and the variants G, box, weekly and constant price of Befund 126. O0 is a statement about K1
   to K5 only.
2. **K5 is a mixed quantity**: new twin, costs of the learned arm held. An arm trained on the set
   points of the cold twin was not run.
3. **Margins move in steps of the bisection.** The searched margins of R1 and C2 differ from
   those of the artifacts by 0.00078 and 0.00039 kg/kg at a margin tolerance of 0.00025 kg/kg; a
   key number in the variant searched contains this step.
4. **K1 is a median** and does not see the 17 scenarios that change.
5. **The multi-start is not a proof of the global optimum**: 8 starts, one Latin hypercube, one
   seed; checked against 64 starts of another seed only.
6. **Base of the branch.** `origin/master` 61d5b18; master has moved since (Befund 127 and 128
   merged). The switch was not run against that code; the full suite of the merge decides.
7. **Placeholders** as in Befund 105 and Befund 126; the truth is the simulator.

For the project lead to decide:

1. The ten decisions of section 6 (in particular K5 and "both variants count").
2. Whether the two observations of section 10 that touch printed numbers (C2 does not hold the
   limit at its artifact margin with cold set points; margin of R1) are to be recorded with
   Befund 126 and Befund 105 beyond the lab-book entry of this study. By the pre-registered
   reading (O0) no addendum is due.
3. Whether new studies use `steady_cold_starts = 8` (reproducible over chunk layouts, factor
   about 1.7 in the time of the reference controllers) and whether the default changes at some
   point. Nothing of this was done here.
4. Whether the controllers with a dynamic optimiser are to be recomputed with the switch (hours
   per controller in the drivers of the findings; not estimated here).
