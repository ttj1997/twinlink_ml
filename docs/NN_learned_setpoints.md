# NN -- Learned prediction of optimised transient set points: learned controllers for load following of the granulator

Status: **design fixed and committed before any code and any run** (sections 1-12; decisions of
the project lead of 2026-11-18 in section 10, points settled without asking in section 12). No code
is written and no run is started before the branch `load-following` is merged. Runs and results
follow in sections 13 onwards, added after the runs (numbers only from the artifacts named there).
Document number and finding number are open (the managing session assigns them at merge). Branch
`learned-setpoints` (worktree, from `master` 230312c; to be rebased onto the current
`origin/master` once `load-following` is merged). **Scenario on placeholders, not a plant
statement.**

Precondition: the branch `load-following` (`docs/NN_load_following.md` there) is merged; this study
uses its package, its days and its reference controllers unchanged.

**Disclosure (state of knowledge when the design was drafted and fixed).** The main run of the
load-following study was in progress; its reading T0-T3 was not known and no result artifact of it
existed. Seen: its design, its stage-0 artifact (`load_following_v1_stage0.json.gz`) and its probe
artifact (`load_following_v1_probe.json.gz`), both on the branch `load-following`, and log values
of the margins of C0, C1, C2 and C4 from the running chain (not artifacts, not quoted here).

## 1. Why this study

The proposal promises for TUHH-B-AP 1 a comparison of ML methods "for the dynamic control of single
processes and in particular for the prediction of optimised transient manipulated variables"; AP 3
builds on it (models that "propose new transient operating procedures"; both quotations translated
from the German proposal). In the repository today:

* **Forward models** exist: the sequence models predict moisture, bed temperature and energy from
  planned future commands (Befund 94).
* **Optimised transient set points** exist, but every controller that computes them carries the
  truth model itself as its internal model: R1-R3 of Befund 105 and C0-C4 of the load-following
  study ("same model as the truth"; the `SteadyMap` and the programmes of
  `twinlink_ml.load_following`).
* **Control on a learned sequence model** exists only for the one-step problem (Befund 113: RL and
  an optimiser on the composite GRU; the margin 5 m carries the feasibility).
* **A model whose output is the command** does not exist: the amortised arm `w -> u*` of
  `docs/25_surrogate_rto.md` (section 5) was not run, and no sequential policy was trained.

The headroom gates (Befund 105, load following) ask whether advanced control pays **given the true
model**. At a plant, or with a Dyssol flowsheet as the truth, that model is not available for free;
what exists is a number of runs. This study asks the question the gates leave open: **what does it
cost to replace the known model by a learned one, as a function of the data, and in which form
(learned model plus optimiser, or direct prediction of the command)?** It is therefore run whatever
the reading of the load-following study is (decision 1); only its reference follows from that
reading, by a rule fixed here.

## 2. Questions

* **Q1 (indirect form).** With the internal model learned from `N` training runs, what cost per kg
  of product does the controller reach at the held moisture limit, against the same control law
  with the true model? Does a static learned map suffice, or does a sequence model add something?
* **Q2 (direct form).** Can a network that predicts the commands directly, trained on `K` solved
  days, replace the optimiser? How many solved days does it need, and what does a decision cost in
  computing time?
* **Q3 (feedback).** How much of the gap of Q1 is closed by an inline moisture measurement used as
  a bias correction (the mechanism of Befund 111), and, descriptively, by a laboratory sample every
  4 h (Befund 112)?

## 3. Frame taken over unchanged from the load-following study

Read from its config and package, not copied: the truth (grey-box model, continuous mode, 5 s
samples, three commands with their rate limits), the line settings of the headroom config (moisture
limit 0.15 kg/kg, bed temperature <= 80 degC, minimum gas velocity 0.80 m/s, nozzle pressure fixed,
solid fraction 0.30; all placeholders), the **main band** of the throughput (1.2-1.8 kg/h; stage-0
artifact), the price, the tariff floor, the production target per day, the tail of 1 h, the
ambient day profiles, the split into training months and test months, the **test days chosen
there** (the 40 stratified days; probe artifact), the cost per kg of dry product as the metric and
the margin search (smallest family-wide margin with which the path condition holds on all days,
tail included; bisection, tolerance 2.5e-4 kg/kg, upper end 0.12 kg/kg). Variant E only (electric
heater); no box sensitivity, no weekly target, no variant G.

Everything that is learned or tuned uses training runs and days of the **training months** only.

## 4. Reference ladder (controllers that know the true model)

| label | controller | source |
|---|---|---|
| Cfix | constant nominal throughput, **fixed** set points (steady optimum at the nominal disturbance, R0 of Befund 105), own margin | new, cheap |
| C0 | constant throughput, set points from the true steady optimum at the current disturbance | load following |
| C1 | price rule, set points as C0 | load following |
| C2 | schedule on the true steady-state map, set points as C0 | load following |
| C3, C4 | economic MPC, perfect preview | load following (reported, not re-run) |

**Twin.** Every learned arm is compared with the controller that has the same control law and the
true model (its *twin*); the gap between the two is the cost of learning the model and nothing
else. **Reference rule (fixed before the load-following reading is known):** T0 -> the learned arms
run at constant throughput and the twin is C0; T1, T2 or T3 -> the learned arms use the schedule
and the twin is C2. At T3 the gap of C2 to C3 is reported next to every result as the part that no
arm of this study addresses.

## 5. Learned arms

All arms command inlet temperature, gas velocity and spray rate on the 5 min grid and plan for
`X <= limit - m` with their own margin `m`. A guard that is the same for every arm keeps the box,
and meets the daily target as C2 does (the remainder in the last hour); its interventions are
counted per arm.

### Family M -- learned steady-state map inside the unchanged law of C2

The classes `Schedule` and `SetPointLaw` of the load-following package are used unchanged; only
their `SteadyMap` is replaced by a map computed from a learned model (optimum of the predicted
power over inlet temperature and gas velocity subject to the predicted moisture and bed
temperature).

| arm | internal model | role |
|---|---|---|
| M-quad | static quadratic regression of the Befund 94 pipeline (`static_quadratic`) | baseline |
| M-gru | composite GRU of the Befund 94 pipeline; steady map = fixed point of the h36 map (the mapping of Befund 113, `surrogate_env/fixed_point.py`) | sequence model |

Training data: `DataSpec.new_study(...)` as arm N of Befund 94 (preset version 2; sample time 25 s,
history 32, horizon 36, runs of 8 h, 32 validation runs), `n_train` in {8, 16, 32, 64}, master
seeds 1-5. The checkpoints of Befund 94 are not kept (`docs/30_surrogate_env.md`, section 6; those
retrained for Befund 113 cover master seeds 1-3 at `n_train` 16 and 64 and lie outside the
repository); the networks are retrained and checked against the records of Befund 94 as in
Befund 113.

**Factor feedback (Q3; decision 3).**

| level | arms | what it is | reading |
|---|---|---|---|
| *none* | M-quad, M-gru | the map is used as learned | yes |
| *fb* | M-quad, M-gru | the predicted moisture is corrected by the filtered difference between an inline moisture measurement and the prediction, filter 30 min, the scheme of Befund 111; measurement noise of the existing sensor model | yes |
| *lab* | M-gru only | a laboratory sample every 4 h as F4 of Befund 112 (`sensor_value.feedback.LabSample`, settings read from `conf/sensor_value/sensor_value_2.yaml`: analysis delay 30 min, Gaussian error sigma 0.003 kg/kg, gain 0.5, bias held between the samples; placeholders) | no, descriptive |

### Family D -- the sequence model in the loop (lower layer)

| arm | controller |
|---|---|
| D-gru | throughput from the schedule of M-gru; every 5 min inlet temperature and gas velocity from an optimisation through the GRU rollout over its horizon (900 s, three control intervals), from the measured history of the last 32 samples (inline operation) |

Its twin is C2; next to it a new reference **C3-15** is run on the true model (the lower layer of
C3 with its horizon cut from 2 h to 15 min), so that the cost of the short horizon and the cost of
the learned model are reported apart (decision 10). D-gru is the only arm in which the measured
history enters the decision; `n_train` in {16, 64}.

### Family P -- direct prediction of the commands

A network maps the information of the moment to the three commands; it is trained by imitation on
days of the training months solved by an *expert* that has the same information: the twin (C0 or
C2), at T3 additionally C3.

* Inputs: hour of the day, the 24 prices of the day and the price of the hour, ambient temperature
  and humidity, the remaining target, the last commands, the margin. **The margin is an input**:
  the expert is solved at three margins per day, so that the margin of a policy can be found by the
  same bisection as for every other controller (decision 9).
* Arms: **P-lin** (ridge regression on hand-made features: price quantile of the hour, ambient
  state, remaining target; baseline), **P-mlp**, **P-gru** (GRU over the hours of the day). No
  Transformer (decision 11).
* Budgets `K` in {5, 10, 20, 40} solved days, nested, drawn as the tuning days of the
  load-following study (stratified by the price spread); seeds 1-5 (draw of the days and
  initialisation).
* **Distribution shift (addition C), descriptive:** per arm, budget and seed the share of the
  decisions on the test days whose inputs lie outside the range of the training labels, and the
  interventions of the guard. No correction of the shift (no DAgger).

RL is not part of this study (decision 2); it stays tied to the gate of the load-following study
(T3 and a gap above 5 %).

## 6. Admissibility, margins, metric, statistics

* **Cell** = arm x feedback level x budget x seed.
* **Margin of the reading:** per cell the smallest family-wide margin on the test days, exactly as
  for C0-C4, so that every number is a cost at a held constraint and comparable with the ladder
  (`CLAUDE.md`, Methodology). The expensive searches use the critical subset of the load-following
  study (8 days, verification on all days).
* **Admissibility first (addition A; definition as in `docs/42_hybrid_structural_2.md`).** A cell
  is **admissible** iff its margin is reached within the upper end of the search (0.12 kg/kg) and
  the path condition holds at that margin on all test days, tail included. Otherwise it is **not
  admissible**: counted as such, with no substitute value, and it takes part in no comparison.
* **Deployable margin (only for the arms the reading names; decision 7):** the same search on days
  of the training months; reported: the violations on the test days at that margin and its cost.
  It answers whether the margin could have been set without the test days.
* **Metric:** `gap(L) = (J_L - J_twin) / J_twin`, cost per kg of dry product, and the position on
  the ladder: `(J_Cfix - J_L) / J_Cfix` and `(J_C0 - J_L) / J_C0`. Decomposition as in `docs/24`,
  section 6: the twin at the margin of the arm gives the margin share, the rest is the share of the
  wrong set points. Reported next to it: margin, largest violation, deviation from the daily
  target, guard interventions, map error at the limit (predicted minus true moisture and power at
  the chosen points, the exploitation gap of Befund 113), time per decision.
* **Statistics (addition B):** median and range over the admissible seeds; every rank statement
  between two learned arms with the paired resolution over seeds
  (`evaluation/paired_resolution.py`) on the seeds at which both cells are admissible, and **Holm
  per family (M, D and P separately)**, effect size and sign consistency next to the verdict. The
  reference controllers are deterministic.

## 7. Pre-registered reading (thresholds: decision 6)

To be copied verbatim into the docstring of the analysis script before the first run.

**Step 1, admissibility.** Per family the reading starts with the share of admissible cells per
arm and budget. An arm with fewer than 4 of 5 admissible seeds at the largest budget (`n_train` 64
or `K` 40) is **"not admissible"** for the reading and enters no class. A family in which no arm
is admissible has the reading "not admissible".

**Step 2, classes.** On the admissible arms only, at the largest budget, median over the admissible
seeds, first class that applies:

| class | condition | meaning |
|---|---|---|
| **U0** | no admissible arm of the family is cheaper than Cfix | the margin of the learned model eats the whole benefit; learned set points are not usable at this budget |
| **U1** | the baseline arm (M-quad, P-lin) is within 2 % of its twin | no network is needed for the set points |
| **U2** | the baseline arm is not, another arm of the family (M-gru; P-mlp or P-gru, named) is within 2 % | the network earns its place in control |
| **U3** | otherwise | better than a fixed point, but the gap to the known model remains; the decomposition says whether margin or set points carry it |

Family M is read once without feedback and once with the level *fb*; the level *lab* has no
reading. Family D has one arm and no baseline: U1 cannot occur there; its classes are U0, U2
(D-gru within 2 % of its twin) and U3. **Data need:** the smallest budget at which the arm named by
the reading is admissible (at least 4 of 5 seeds), within 2 % in the median and within 5 % in
every admissible seed; "not determined" if this is not reached at the largest budget. A margin
search that does not reach a margin is not an error but a not admissible cell (step 1); the reading
of a family is invalid, and reported as such, only if a unit of the family ended with an error.

Expectation, declared as such and not a result: Befund 113 found that the GRU underestimate the
moisture by 0.005-0.008 kg/kg and that only the margin 5 m keeps an optimiser on them feasible;
Befund 111 found a margin of 0.059 instead of 0.0045 kg/kg without feedback under model mismatch.
Without feedback U3 (or U0 at small budgets) is the likely reading of family M, with feedback U1.
On the training days the tuned price rule reaches the static cost of the schedule to within 0.1 %
(stage-0 artifact of the load-following study: 157.85 against 157.72, constant throughput 162.30;
steady-state map only), so the throughput layer is unlikely to separate the arms; the set-point
layer and the margin will.

## 8. What the study will not show (caveats fixed beforehand)

* Every caveat of the load-following study holds: product quality is the moisture only, continuous
  operation without standstill or batch, one year of prices, placeholders throughout.
* **The truth is the simulator without any parameter or structural deviation (addition D).** The
  twin therefore corresponds to a perfectly calibrated physics model, and the learned models learn
  the simulator that is also the truth. The study measures the cost of learning in the case that
  is most favourable for the physics; what a physics model with a parameter or structural error
  costs against learned models is the subject of Befunde 120, 122, 124 and 125, in their own
  frames.
* **`N` runs are `N` excitation experiments of 8 h each** (plus 32 validation runs), not operating
  records; in the simulator, models trained on constant operation alone stay far behind (Befund 55).
* **The feedback level *fb* assumes an inline moisture measurement.** It exists at the laboratory
  line; at the plant the product moisture is not measured (Befund 56). The level *lab* rests on the
  placeholder laboratory settings of Befund 112.
* **Family P needs an expert that knows the model.** It saves computing time at run time, not
  modelling effort; its learning curve counts solved days, not plant runs.
* The reading is a statement about the tested, untuned configuration of each model (as Befund 94).

## 9. Stages, cost, budget and cuts

Stage 0 (no truth run of a day): training of the models, tabulation of the learned maps, map error
at the limit (descriptive). Dry run over all units on days shortened to 1 h; probe on three
training days of full length under parallel load; then the main run, detached, resumable, one unit
= arm x feedback level x budget x seed x day x margin (rules "Long runs").

Indication of the cost from the probe of the load-following study (8 workers, three training
days): 35 s per day-run of a cheap controller, 234 s for C4, 1679 s for C3. With the critical
subset a margin search of a cheap arm takes `8 x 9 + 2 x 40 = 152` day-runs. Family M has
`(2 + 3) arm-level pairs x 4 budgets x 5 seeds = 100` cells; whether a day-run of a learned map
costs the 35 s of the true map is not known (placeholder; the own probe decides). Label generation
of family P with C2 as the expert: `40 days x 3 margins` cheap day-runs; with C3 as the expert the
same count at the cost of C3.

**Budget (decision 8).** 30 h wall-clock. If the extrapolation after the dry run and the probe is
above 30 h, the budget is first extended to 48 h; cuts are applied only if the extrapolation is
still above 48 h, in this order and only as far as needed: (1) the feedback level *lab*;
(2) `n_train` 32; (3) seeds 5 -> 3 for D-gru; (4) P-gru. **Never cut:** everything that carries
the reading of the families M, P and D at the largest budget. The extrapolation, the budget that
applies and every cut are recorded in the probe artifact and in the chain log before the main run.

## 10. Decisions of the project lead (2026-11-18)

| # | decision |
|---|---|
| 1 | The study is run whatever the reading T0-T3 of the load-following study, with the reference rule of section 4 |
| 2 | Families M and P in, D in, RL out (it stays tied to the gate of the load-following study: T3 and a gap above 5 %) |
| 3 | Feedback as a level of family M (inline moisture, bias correction of Befund 111). In addition a level "laboratory sample every 4 h" (error, delay and gain as F4 of Befund 112, placeholders) for M-gru only, without a reading of its own, descriptive |
| 4 | Training data: the data specification of arm N of Befund 94, `n_train` 8-64, five seeds, networks retrained and checked against the records |
| 5 | Exception from `DataSpec.new_study` approved for the labels of family P (solved days are not generated by the data pipeline; precedent `docs/12`, section L) |
| 6 | Thresholds 2 % (within the twin) and 5 % (every seed) |
| 7 | Deployable margin only for the arms named by the reading |
| 8 | Budget 30 h wall-clock; if the extrapolation is above it, first extend to 48 h, only then cut. Order of cuts: (1) the level "laboratory sample", (2) `n_train` 32, (3) seeds 5 -> 3 for D-gru, (4) P-gru. Never cut: everything that carries the reading of M, P and D at the largest budget |
| 9 | The margin is an input of the policies (expert solved at three margins per day) |
| 10 | Horizon of D-gru: the existing 900 s, with C3-15 on the true model as the control |
| 11 | Policy architectures P-lin, P-mlp, P-gru; no Transformer |
| A | Admissibility first: a margin search that does not reach a margin within the upper end makes the cell not admissible (no substitute value). The reading starts per family with the share of admissible cells per arm; U0-U3 on admissible cells only. An arm with fewer than 4 of 5 admissible seeds at the largest budget counts as "not admissible" for the reading |
| B | Statistics: Holm per family (M, D, P separately), not over all cells of the study |
| C | Family P: the distribution shift is measured (share of the decisions whose inputs lie outside the range of the training labels, guard interventions per arm); descriptive, no DAgger |
| D | Caveat in section 8: the truth is the simulator without a parameter or structural deviation; the twin corresponds to a perfectly calibrated physics model; the study measures the cost of learning in the case most favourable for the physics (reference Befunde 120, 122, 124, 125) |

Order of work after these decisions: wait until `load-following` is merged; bring the worktree to
the current `origin/master` (rebase of the documentation commits); code, tests, the reading into
the docstring; full suite; dry run; probe; main run detached from the session. **Computing load:**
the full suite, the dry run and the main run start only when no other run and no suite of the
managing session computes on the machine. No merge, no push by this session.

## 11. Implementation plan (nothing written yet)

* Package `src/twinlink_ml/learned_setpoints/`: `maps` (learned `SteadyMap` for the quadratic
  regression and for the GRU fixed point, tabulated), `feedback` (bias correction; the level *lab*
  through the unchanged `sensor_value.feedback.LabSample`), `gru_mpc` (lower layer of D-gru),
  `labels` (expert days), `policy` (P arms, the target guard and the range check of addition C),
  `study`; settings `conf/learned_setpoints/learned_setpoints.yaml`; scripts
  `run_learned_setpoints.py` (stages `stage0`, `dryrun`, `probe`, `main`; `--resume`) and
  `analyze_learned_setpoints.py`. The packages `load_following`, `headroom`, `surrogate_env` and
  `sensor_value` are imported, not changed.
* Tests to be written before the runs: with the true map behind the learned-map interface an M arm
  reproduces C2 on a day bit for bit; with the truth behind the GRU interface D-gru reproduces
  C3-15; a policy that looks its labels up reproduces its expert; with the margin input moved the
  policy's moisture moves in the right direction; training and test months are disjoint for models,
  labels and tuning; the guard meets the daily target; a search that does not reach a margin gives
  a not admissible cell and no number; the fingerprints of the load-following and headroom configs
  equal those of the load-following artifacts.
* Lab-book entry "Befund NN (number open)" and the `docs/12` row only with results.

## 12. Points settled without asking when the design was fixed (most conservative reading each time)

None of them is a result. **All seven points were confirmed by the project lead on 2026-11-18**,
point 7 with the caveat added there.

1. **Reading of family D.** The decisions name a reading of D; D has one arm and no baseline. Its
   classes are U0, U2 and U3 as defined in section 7 (U1 cannot occur). The comparison of D-gru
   with M-gru (same budget and seeds, each feedback level of M-gru with a reading) is a rank
   statement of family D and counts in its Holm family.
2. **Class U2 in family P.** The draft named only P-gru next to the baseline; P-mlp within 2 %
   without P-lin would have had no class. U2 now reads "another arm of the family, named".
3. **Budgets of D-gru:** `n_train` 16 and 64, as in the draft that was decided on.
4. **Admissibility after cut (3).** With three seeds an arm is admissible for the reading only if
   all three seeds are admissible (2 of 3 is below the 4 of 5 of addition A).
5. **Budget rule read as:** extrapolation at most 30 h -> no change; above 30 h and at most 48 h ->
   uncut with the budget 48 h; above 48 h -> cuts in the order of decision 8 until the
   extrapolation is at most 48 h.
6. **Level *lab*:** the settings of F4 are read unchanged from the config of Befund 112, with the
   interval 4 h only.
7. **Range of the training labels (addition C):** per input the minimum and maximum over the labels
   of the cell (budget and seed); a decision counts as outside if at least one input is outside.
   The check per input does not capture unfamiliar combinations of inputs; the quantity is
   descriptive and tends to underestimate the distribution shift.
