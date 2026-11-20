# NN -- Learned prediction of optimised transient set points: learned controllers for load following of the granulator

Status: **design fixed and committed before any code and any run** (sections 1-12; decisions of
the project lead of 2026-11-18 in section 10, points settled without asking in section 12). Code
committed before the dry run (section 13, with the points settled during the implementation).
Runs and results: sections 14-19, added on 2026-11-20 after the runs (numbers only from the
artifacts named there). **Pre-registered reading: U3 in every family (M without and with feedback,
D, P), valid, no errored unit; family P at the boundary to U0. Not yet accepted by the project
lead.** Document number and finding number are open (the managing session assigns
them at merge). Branch `learned-setpoints` (worktree; rebased onto `origin/master` 61d5b18 after
the merge of `load-following`). **Scenario on placeholders, not a plant statement.**

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

## 13. Implementation (code committed before the dry run; no result here)

State on 2026-11-19: `load-following` is merged (Befund 126, `docs/43_load_following.md`, reading
**T1**). By the reference rule of section 4 the learned arms use the schedule and **the twin is
C2**. The branch was rebased onto `origin/master` 61d5b18. Package
`src/twinlink_ml/learned_setpoints/`, settings `conf/learned_setpoints/learned_setpoints.yaml`,
driver `scripts/run_learned_setpoints.py`, analysis `scripts/analyze_learned_setpoints.py` (its
docstring holds section 7 verbatim), tests `tests/test_learned_setpoints.py`.

**Points settled without asking during the implementation** (most conservative reading each time;
none is a result; to be objected to before the main run is read):

1. **Tabulation and the control M-tab.** A query of the fixed point of the GRU costs seconds, a
   day needs about 1300 queries. The steady response of every model is therefore tabulated once
   (13 inlet temperatures x 12 gas velocities x 7 spray rates x 7 ambient temperatures x 5
   humidities, linear interpolation, the moisture in its logarithm) and the optimum is searched on
   the table. What the table costs is measured by a control that is not in the design: **M-tab**,
   the TRUE steady response behind the same table and the same law. It has no part in the
   reading; the gap of an M arm to its twin contains the gap of M-tab.
2. **Prediction of the feedback.** The bias is the filtered difference between the measured
   moisture and the STEADY prediction of the map at the inputs of the 25 s block (the map has no
   dynamics). Level *fb*: the existing sensor model without outliers. Levels *fb* and *lab* get
   the pre-roll of Befund 111 (6 h, first disturbance sample held, nominal throughput) in which
   the bias settles; cost and violations of the pre-roll do not count. The level without feedback
   has no pre-roll and starts as C0-C4 do.
3. **Margin search.** Critical subset = the 8 days with the largest violation of the cell itself
   at margin 0 (one pass over all days), then the bisection on them and the verification on all
   days, as in the load-following study (`bisect_margin_over_scenarios`). A cell that reaches no
   margin is not admissible. Cfix has no set point where the steady optimiser finds none at the
   margin; such a margin counts as violating.
4. **Family D.** Horizon problem: energy over the 36 blocks plus the slack weight of the headroom
   config per block, moisture and bed temperature at every block end, commands piecewise constant
   over the three intervals, solved by SLSQP with finite differences. The rate limits are not in
   the prediction; the spray rate is held at its current level over the horizon; the history is
   measured through the sensor model. C3-15 uses the schedule on the true map and the unchanged
   optimiser of the headroom study with three intervals. The test compares the first command of
   both on the true model (tolerance 0.5 K and 0.01 m/s); the two programmes discretise the model
   differently, bit equality is not possible.
5. **Family P.** Label margins 0, 0.02 and 0.04 kg/kg (stage 0 of the load-following study
   guarantees a steady optimum up to 0.02 at every weather grid point; a policy asked for a larger
   margin extrapolates, and the range check counts it). Label days of a seed: the 40 training
   days of the load-following study in five strata of the price spread, permuted per stratum, a
   budget takes the first days of every stratum; at 40 days the seeds differ in the
   initialisation only. P-mlp and P-gru stop on validation days of their cell (a fifth, at least
   one day). P-gru runs over the decisions of the day (5 min steps). Guard: slack 0.02 kg of
   suspension per day before the last hour, so that the schedule of C2 passes it untouched.
6. **Deployable margin.** Computed for every arm with a reading at the largest budget (a superset
   of the arms the reading names; the analysis reports the named ones), on 40 tuning days of the
   training months drawn as the training days of the load-following study, without its training
   days (the label days of family P).
7. **Decomposition.** The twin is run at the margin of every admissible cell of the largest
   budget; for the other cells its cost is interpolated on a grid of 13 margins (marked).
8. **Ladder.** C0 and C2 are run again at their margins of the load-following artifact (control:
   the records must equal the artifact); C3 and C4 are quoted from it. The tail uses the margin
   of C0 of the load-following study for every arm.
9. **Reproduction check.** GRU and static quadratic regression against the Befund 94 records on
   the excited test set (largest absolute difference of the metrics).
10. **Computing.** The truth simulation of a day costs about the same for one controller as for
    some hundred, so the searches of all cells run side by side and their days share batches. A
    controller that fails gives an error record for its day and does not touch the others.

**Tests** (green on the commit of the code): the runner equals `run_days` for C2 bit for bit, and
the true map behind family M is C2 bit for bit; lookup policy = expert; margin input moves the
moisture; months disjoint, fingerprints equal the load-following artifacts; guard meets the target
(to within its slack); no margin -> not admissible and no number; true steady response = steady
optimiser; table optimum of a known response; NumPy GRU = PyTorch model; horizon problem on the
truth = decision of C3-15 (tolerance above); reading and budget rule on synthetic inputs.

## 14. Course of the runs (2026-11-19 to 2026-11-20)

One chain, started detached from the session (`C:/twinlink_work/learnsp_chain.sh`, outside the repository; its log gives the times and the result of the full suite). Driver commit ad9eb7a for every stage; every stage ran on a clean tree and its artifact was committed (denylist test first) before the next stage started. No stage was repeated, nothing was resumed. Chain log: start 2026-11-19 11:56, 21 min of waiting for a free machine, main run 2026-11-19 15:57 to 2026-11-20 03:47, chain complete 03:49.

| step | ran on | artifact (committed as) | units | errors | runtime | peak memory per worker |
|---|---|---|---|---|---|---|
| full suite | ad9eb7a | chain log, no artifact | 1579 passed, 32 skipped | - | 33.6 min | - |
| dry run over all units (stage 0 with tiny models, days of 1 h plus tail) | ad9eb7a | `learned_setpoints_v1_dryrun.json.gz` (5e49f4c) | 16320 and 81 stage-0 units | 0 | 30.1 min | 575 MB (evaluation), 1711 MB (stage-0 units) |
| stage 0 (models, tables, labels, policies) | 5e49f4c | `learned_setpoints_v1_stage0.json.gz` (ad3053f) | 81 | 0 | 142.6 min | 1790 MB |
| probe (3 training days of full length, margin 0.01) | ad3053f | `learned_setpoints_v1_probe.json.gz` (d34825b) | 39 | 0 | 4.1 min | 375 MB |
| main run | d34825b | `learned_setpoints_v1.json.gz` (5238b5e) | 36982 in 125 rounds, 1086 batches of at most 300 rows | 0 | 11.82 h | 2198-2331 MB |
| analysis | 5238b5e | `learned_setpoints_v1_analysis.json` (de261bc) | - | 0 | seconds | - |

**Budget rule (decision 8), decided by the probe before the main run:** extrapolation 12.8 h (10.4 h for the main run with the safety factor 1.25, plus the 2.4 h stage 0 had taken) against the budget of 30 h: "within the budget", cuts: none. Every arm, level, budget and seed of the design was run. The main run took 11.82 h, 1.4 h more than the probe had extrapolated for it; stage 0 and the main run together stay far below the budget.

Runtime of the parts of the main run (h): reference 1.06, M 3.29, D 3.57, P 0.84, twin_curve 0.09, decomposition 0.18, deployable 2.79.

**Memory, a lesson for the next study.** The probe (375 MB) and the evaluation of the dry run (575 MB) underestimated the peak of the main run (2331 MB per worker): the probe ran batches of at most 6 rows and the dry run days of 1 h, the main run full days in batches of up to 300 rows. Eight workers at the peak are 18645 MB, below the limit of 80 % of the RAM (25529 MB of 31912 MB, gate line of the chain log); the rule held, but not by the measurement that was meant to guarantee it. A probe must contain one batch of the largest size on full-length days.

The dry run had 47 cells without a margin (tiny models, days of 1 h, coarse tolerance); this is no result, it shows that the path "no margin -> not admissible, no number" was exercised.

## 15. Controls

* **Reproduction of Befund 94.** All 20 model units (GRU and static quadratic regression, `n_train` 8-64, master seeds 1-5): largest absolute difference of the 56 metrics on the excited test set against the records of Befund 94: 0.0 (stage-0 artifact).
* **Table of the true steady response (behind M-tab).** 100.0 % of the nodes valid. At 75 optima at the limit (bound 0.15 kg/kg), table minus truth: moisture mean +0.00029, range -0.00001 to +0.00167 kg/kg; power mean -0.00007 kW; 0 points truly above the limit. The table errs to the safe side.
* **M-tab (true response behind the table and the law of family M; not in the design, no part in the reading).** Own margin 0.00657 kg/kg, cost 0.39744 EUR/kg, gap to the twin -0.13 %. Decomposition: margin share -0.21 % (its margin is smaller than the 0.00742 of the twin, taken from the load-following artifact), set-point share +0.08 %. **The tabulation costs about 0.08 % at an equal margin**; the gaps of family M contain this amount.
* **C2 against the records of the load-following artifact: NOT bit-equal.** Point 8 of section 13 expected equality. 40 days compared, largest absolute difference of cost, product or violation per day 0.005778 (field `control_c2_against_load_following`). Computed from the records of the main artifact and the records taken over from `load_following_v1.json.gz` (not a field of the analysis): the cost of the day differs by more than 1e-9 relative on 17 of 40 days, between -0.129 % and +0.154 %; on the other days by at most 5.0e-10. In the sum the effect is small: twin 0.397944 here against 0.397950 EUR/kg there (-0.0016 %), C0 0.406341 against 0.406330 (+0.0027 %). Every gap of this study uses the twin as run HERE.
  **Cause (read from the code, confirmed by a scratch check that is not an artifact and from which no number is quoted):** `headroom.controllers.SteadyStateOptimizer.solve` starts every solve from the solution of the previous one, and `load_following.steady.SteadyMap` caches by keys rounded to ten decimals. The steady optimum returned for a query therefore depends on the queries the worker process solved before it; the load-following run and this run gave the workers different sequences. The test "runner equals `run_days` bit for bit" compares two runs within one process, on one map and its cache. **This concerns existing code of Befund 105 and Befund 126, not code of this study; it is reported to the project lead and NOT changed here** (`CLAUDE.md`: report first, fix after confirmation). What it means for those findings is open (section 19). For this study the control M-tab bounds it: the table search has no warm start and no history, and it is not cheaper than the twin at an equal margin (+0.08 %).
* **Cfix.** Margin 0.03613 kg/kg, cost 0.43969 EUR/kg, +10.49 % above the twin (margin share +7.95 %, set-point share +2.54 %) and 8.21 % above C0.
* **C3-15.** Margin 0.00391 kg/kg, cost 0.39601 EUR/kg, gap to the twin -0.49 %; 119 ms per decision. Against C3 of the load-following artifact (0.39463 EUR/kg, horizon 2 h): +0.35 % (derived): the cost of the short horizon.

**Ladder** (cost per kg of dry product, 40 test days, variant E, main band; margins in kg/kg):

| controller | margin | cost EUR/kg | against the twin % | source |
|---|---|---|---|---|
| Cfix | 0.03613 | 0.43969 | +10.49 | this run |
| C0 | 0.00547 | 0.40634 | +2.11 | this run, margin of the load-following artifact |
| C2 (twin) | 0.00742 | 0.39794 | 0 | this run, margin of the load-following artifact |
| M-tab (control) | 0.00657 | 0.39744 | -0.13 | this run, own margin |
| C3-15 | 0.00391 | 0.39601 | -0.49 | this run, own margin |
| C3 | 0.00379 | 0.39463 | -0.83 | load-following artifact, not re-run |
| C4 | 0.00023 | 0.39053 | -1.86 | load-following artifact, not re-run |

**What a margin costs (twin curve; C2 run at 13 margins, cost computed from the records of the main artifact).** The reading threshold of 2 % corresponds to a margin of about 0.015 kg/kg.

| margin kg/kg | cost EUR/kg | against the twin at its own margin % | largest violation kg/kg |
|---|---|---|---|
| 0.0000 | 0.39087 | -1.78 | 0.00690 |
| 0.0025 | 0.39321 | -1.19 | 0.00449 |
| 0.0050 | 0.39559 | -0.59 | 0.00241 |
| 0.0075 | 0.39802 | +0.02 | 0.00000 |
| 0.0100 | 0.40050 | +0.64 | 0.00000 |
| 0.0150 | 0.40560 | +1.92 | 0.00000 |
| 0.0200 | 0.41091 | +3.26 | 0.00000 |
| 0.0300 | 0.42214 | +6.08 | 0.00000 |
| 0.0400 | 0.43450 | +9.19 | 0.00000 |
| 0.0600 | 0.46304 | +16.36 | 0.00000 |
| 0.0800 | 0.49898 | +25.39 | 0.00000 |
| 0.1000 | 0.55158 | +38.61 | 0.00000 |
| 0.1200 | 0.66023 | +65.91 | 0.00000 |

## 16. Results (40 test days, variant E, main band, true prices; twin C2)

All numbers from `learned_setpoints_v1_analysis.json` unless marked. `gap` = (cost of the arm - cost of the twin) / cost of the twin at the margin each of them needs; median, smallest and largest over the ADMISSIBLE seeds. "against Cfix" and "against C0": saving, positive = the arm is cheaper (median over the admissible seeds, computed from the per-cell values of the analysis).

### 16.1 Admissibility and cost

| arm | level | budget | admissible seeds | gap median (min .. max) % | margin median kg/kg | against Cfix % | against C0 % |
|---|---|---|---|---|---|---|---|
| M-quad | none | 8 | 1 of 5 | +137.04 (+137.04 .. +137.04) | 0.1164 | -114.54 | -132.14 |
| M-quad | none | 16 | 3 of 5 | +49.98 (+20.74 .. +52.32) | 0.1043 | -35.74 | -46.88 |
| M-quad | none | 32 | 2 of 5 | +45.77 (+42.87 .. +48.68) | 0.0742 | -31.93 | -42.76 |
| M-quad | none | 64 | 4 of 5 | +22.36 (+13.14 .. +74.42) | 0.0934 | -10.74 | -19.83 |
| M-quad | fb | 8 | 1 of 5 | +68.41 (+68.41 .. +68.41) | 0.1152 | -52.42 | -64.93 |
| M-quad | fb | 16 | 5 of 5 | +13.84 (+11.63 .. +25.42) | 0.0449 | -3.03 | -11.48 |
| M-quad | fb | 32 | 3 of 5 | +11.16 (+8.34 .. +17.63) | 0.0318 | -0.61 | -8.86 |
| M-quad | fb | 64 | 5 of 5 | +8.51 (+4.29 .. +11.59) | 0.0203 | +1.79 | -6.27 |
| M-gru | none | 8 | 5 of 5 | +11.28 (+6.55 .. +20.55) | 0.0711 | -0.72 | -8.98 |
| M-gru | none | 16 | 5 of 5 | +7.44 (+4.33 .. +9.82) | 0.0463 | +2.76 | -5.22 |
| M-gru | none | 32 | 5 of 5 | +7.04 (+2.38 .. +10.82) | 0.0416 | +3.13 | -4.82 |
| M-gru | none | 64 | 5 of 5 | +7.07 (+0.78 .. +10.11) | 0.0357 | +3.09 | -4.86 |
| M-gru | fb | 8 | 5 of 5 | +6.06 (+5.18 .. +11.44) | 0.0285 | +4.01 | -3.87 |
| M-gru | fb | 16 | 5 of 5 | +3.98 (+2.55 .. +6.51) | 0.0224 | +5.89 | -1.83 |
| M-gru | fb | 32 | 5 of 5 | +3.23 (+0.58 .. +4.85) | 0.0195 | +6.57 | -1.10 |
| M-gru | fb | 64 | 5 of 5 | +3.87 (+0.96 .. +6.37) | 0.0221 | +5.99 | -1.73 |
| M-gru | lab | 8 | 5 of 5 | +22.25 (+11.10 .. +28.42) | 0.0682 | -10.64 | -19.72 |
| M-gru | lab | 16 | 5 of 5 | +8.97 (+5.46 .. +18.19) | 0.0381 | +1.38 | -6.72 |
| M-gru | lab | 32 | 5 of 5 | +8.83 (+4.15 .. +30.75) | 0.0361 | +1.50 | -6.58 |
| M-gru | lab | 64 | 5 of 5 | +7.62 (+2.91 .. +41.62) | 0.0348 | +2.60 | -5.40 |
| D-gru | none | 16 | 5 of 5 | +5.44 (+4.40 .. +7.02) | 0.0155 | +4.57 | -3.26 |
| D-gru | none | 64 | 5 of 5 | +2.98 (+1.47 .. +3.37) | 0.0088 | +6.80 | -0.85 |
| P-lin | none | 5 | 5 of 5 | +17.01 (+13.38 .. +28.00) | 0.0413 | -5.90 | -14.59 |
| P-lin | none | 10 | 5 of 5 | +16.36 (+11.77 .. +30.81) | 0.0484 | -5.32 | -13.96 |
| P-lin | none | 20 | 5 of 5 | +13.76 (+11.15 .. +18.50) | 0.0344 | -2.96 | -11.41 |
| P-lin | none | 40 | 5 of 5 | +14.15 (+14.15 .. +14.15) | 0.0344 | -3.31 | -11.79 |
| P-mlp | none | 5 | 4 of 5 | +27.65 (+18.81 .. +35.54) | 0.0816 | -15.53 | -25.01 |
| P-mlp | none | 10 | 3 of 5 | +19.96 (+19.17 .. +23.28) | 0.0498 | -8.57 | -17.48 |
| P-mlp | none | 20 | 3 of 5 | +22.97 (+17.05 .. +31.47) | 0.0639 | -11.30 | -20.43 |
| P-mlp | none | 40 | 1 of 5 | +24.35 (+24.35 .. +24.35) | 0.0631 | -12.55 | -21.78 |
| P-gru | none | 5 | 3 of 5 | +25.27 (+22.84 .. +27.01) | 0.0748 | -13.38 | -22.68 |
| P-gru | none | 10 | 5 of 5 | +21.18 (+14.23 .. +22.67) | 0.0752 | -9.67 | -18.67 |
| P-gru | none | 20 | 5 of 5 | +12.05 (+7.83 .. +17.27) | 0.0330 | -1.41 | -9.74 |
| P-gru | none | 40 | 5 of 5 | +10.47 (+9.17 .. +11.59) | 0.0338 | +0.02 | -8.19 |

budget = `n_train` (training runs of 8 h) for the families M and D, `K` (solved days) for family P. P-lin at `K` 40 has one value: all seeds use the same 40 label days and the ridge regression has no initialisation.

Gap per seed at the largest budget (%):

* M-quad, level none, budget 64: seed 1: +74.42, seed 2: +13.14, seed 3: +18.84, seed 4: +25.87, seed 5: not admissible
* M-quad, level fb, budget 64: seed 1: +11.59, seed 2: +4.29, seed 3: +5.21, seed 4: +9.06, seed 5: +8.51
* M-gru, level none, budget 64: seed 1: +4.83, seed 2: +0.78, seed 3: +7.07, seed 4: +10.11, seed 5: +7.62
* M-gru, level fb, budget 64: seed 1: +1.71, seed 2: +0.96, seed 3: +3.87, seed 4: +6.37, seed 5: +4.76
* M-gru, level lab, budget 64: seed 1: +41.62, seed 2: +2.91, seed 3: +6.55, seed 4: +7.62, seed 5: +13.33
* D-gru, level none, budget 64: seed 1: +2.98, seed 2: +1.47, seed 3: +3.23, seed 4: +2.83, seed 5: +3.37
* P-lin, level none, budget 40: seed 1: +14.15, seed 2: +14.15, seed 3: +14.15, seed 4: +14.15, seed 5: +14.15
* P-mlp, level none, budget 40: seed 1: +24.35, seed 2: not admissible, seed 3: not admissible, seed 4: not admissible, seed 5: not admissible
* P-gru, level none, budget 40: seed 1: +9.17, seed 2: +11.59, seed 3: +10.75, seed 4: +10.47, seed 5: +9.95

27 of the 170 learned cells reach no margin within 0.12 kg/kg and are not admissible (no number, in no comparison): M-quad level none 10, M-quad level fb 6, P-mlp level none 9, P-gru level none 2. No unit ended with an error.

### 16.2 Pre-registered reading (section 7, verbatim in the docstring of the analysis script)

| family | step 1: arms admissible for the reading (at least 4 of 5 seeds at the largest budget) | median gap at the largest budget | reading | named arm | data need |
|---|---|---|---|---|---|
| M/none | M-quad, M-gru | M-quad +22.36 %, M-gru +7.07 % | **U3** (valid, 0 errored units) | none | none (no arm named) |
| M/fb | M-quad, M-gru | M-quad +8.51 %, M-gru +3.87 % | **U3** (valid, 0 errored units) | none | none (no arm named) |
| D | D-gru | D-gru +2.98 % | **U3** (valid, 0 errored units) | none | none (no arm named) |
| P | P-lin, P-gru; not admissible for the reading: P-mlp | P-lin +14.15 %, P-gru +10.47 % | **U3** (valid, 0 errored units) | none | none (no arm named) |

**U3 in every family: better than a fixed set point, but the gap to the known model remains; no arm is within 2 % of its twin, so no arm is named and no data need is determined.** In words:

* **Family M without feedback.** M-gru is admissible in 5 of 5 seeds at every budget, M-quad in 4 of 5 at `n_train` 64 and in at most 3 of 5 below. Median gap at `n_train` 64: M-gru +7.07 %, M-quad +22.36 %. Not U0: M-gru is cheaper than Cfix (+3.09 % in the median).
* **Family M with the level fb.** Both arms admissible in 5 of 5 seeds at `n_train` 64. Median gap: M-gru +3.87 %, M-quad +8.51 %. **The expectation declared in section 7 (U1 with feedback) is not met.**
* **Family D.** D-gru admissible in 5 of 5 seeds at both budgets; median gap +2.98 % at `n_train` 64 (+1.47 .. +3.37), above the 2 % of U2.
* **Family P: U3 by the letter, at the boundary to U0.** P-mlp is not admissible for the reading (1 of 5 seeds at `K` 40). P-lin is more expensive than Cfix. P-gru decides the class: its median cost 0.439619 EUR/kg is 0.015 % below Cfix (0.439686); 3 of 5 seeds are cheaper than Cfix. One seed the other way and the reading would be U0. **In substance the policies do not beat a fixed set point at 40 solved days.**

### 16.3 Where the gap comes from (decomposition; twin run at the margin of the cell)

Margin share = (twin at the margin of the cell - twin) / twin; set-point share = the rest. Largest budget only (there the twin was run at the margin of every admissible cell; for the smaller budgets the analysis interpolates on the twin curve, field `twin_at_margin`). Median over the admissible seeds, computed from the per-cell values of the analysis; the medians of the two shares need not add up to the median gap.

| arm | level | budget | gap median % | margin share median % | set-point share median % | mean moisture median kg/kg |
|---|---|---|---|---|---|---|
| M-quad | none | 64 | +22.36 | +33.90 | +0.46 | 0.0932 |
| M-quad | fb | 64 | +8.51 | +3.34 | +1.94 | 0.1296 |
| M-gru | none | 64 | +7.07 | +7.83 | -1.17 | 0.1174 |
| M-gru | fb | 64 | +3.87 | +3.82 | +0.05 | 0.1279 |
| M-gru | lab | 64 | +7.62 | +7.52 | +0.31 | 0.1153 |
| D-gru | none | 64 | +2.98 | +0.34 | +2.54 | 0.1349 |
| P-lin | none | 40 | +14.15 | +7.40 | +6.74 | 0.1018 |
| P-mlp | none | 40 | +24.35 | +17.58 | +6.77 | 0.0815 |
| P-gru | none | 40 | +10.47 | +7.22 | +3.31 | 0.1111 |

Mean moisture of the twin over the test days: 0.1424 kg/kg (limit 0.15; computed from the records of the main artifact).

* **Family M: the gap is the margin.** With feedback the set-point share of M-gru is about zero; the learned map chooses the set points of the twin and pays for the distance it must keep from the limit. Without feedback the set-point share of M-gru is negative: at an equal nominal margin the arm is cheaper than the twin because its map underestimates the moisture and it runs closer to the limit than it plans to; the larger margin pays for exactly that (the exploitation gap of Befund 113, section 16.7).
* **Family D: the nominal margin is small, the distance to the limit is not.** D-gru needs a margin close to that of the twin (0.0088 against 0.0074 kg/kg; the measured history corrects the level), but at that margin its commands cost more than the twin's, and its mean moisture is 0.1349 against 0.1424 kg/kg for the twin: the arm stays further from the limit than its nominal margin says. Whether this is the short horizon, the slack penalty or the model is not separated here (C3-15 on the true model does not show it).
* **Family P: both.** P-lin and P-gru need more than four times the margin of the twin and choose dearer set points on top.

**Decision 10, the two costs of D-gru apart (derived from the values above):** short horizon on the true model, C3-15 against C3: +0.35 %; learned model in the loop, D-gru (median cost 0.40979 EUR/kg at `n_train` 64) against C3-15: +3.48 %.

### 16.4 Position on the ladder

* **No learned arm reaches C0** (constant throughput, true steady optimum): the saving against C0 is negative in the median for every arm, level and budget. Closest (cells with at least 4 admissible seeds): D-gru level none budget 64 (-0.85 %), M-gru level fb budget 32 (-1.10 %), M-gru level fb budget 64 (-1.73 %). The schedule gains 2.07 % against C0 with the true model (this run); the margin a learned model needs costs more than that.
* **Against the fixed set point** the twin saves 9.49 % (derived). Cheaper than Cfix in the median, with at least 4 admissible seeds: M-quad fb 64, M-gru none 16, M-gru none 32, M-gru none 64, M-gru fb 8, M-gru fb 16, M-gru fb 32, M-gru fb 64, M-gru lab 16, M-gru lab 32, M-gru lab 64, D-gru none 16, D-gru none 64, P-gru none 40 (arm, level, budget).

### 16.5 Rank statements between learned arms (paired resolution over seeds, Holm per family)

**Family M:** 16 comparisons, 3 with fewer than two common admissible seeds (no test), 0 resolved after Holm.

| comparison (first vs second) | common seeds | cost of the second minus cost of the first, mean EUR/kg | relative to the twin % | resolution threshold EUR/kg | sign consistency | p | p Holm | resolved |
|---|---|---|---|---|---|---|---|---|
| M-gru: fb vs none, n_train 8 | 5 | +0.0205 | +5.16 | 0.0154 | 1.00 | 0.0209 | 0.2087 | no |
| M-gru vs M-quad, level none, n_train 16 | 3 | +0.1369 | +34.39 | 0.1542 | 1.00 | 0.0623 | 0.4359 | no |
| M-gru vs M-quad, level fb, n_train 16 | 5 | +0.0510 | +12.82 | 0.0296 | 1.00 | 0.0087 | 0.1130 | no |
| M-quad: fb vs none, n_train 16 | 3 | +0.0957 | +24.05 | 0.1671 | 1.00 | 0.1326 | 0.5096 | no |
| M-gru: fb vs none, n_train 16 | 5 | +0.0114 | +2.87 | 0.0072 | 1.00 | 0.0117 | 0.1290 | no |
| M-gru vs M-quad, level none, n_train 32 | 2 | +0.1466 | +36.84 | 0.2426 | 1.00 | 0.0824 | 0.4554 | no |
| M-gru vs M-quad, level fb, n_train 32 | 3 | +0.0341 | +8.58 | 0.0582 | 1.00 | 0.1274 | 0.5096 | no |
| M-quad: fb vs none, n_train 32 | 2 | +0.1434 | +36.02 | 0.2182 | 1.00 | 0.0759 | 0.4554 | no |
| M-gru: fb vs none, n_train 32 | 5 | +0.0153 | +3.84 | 0.0089 | 1.00 | 0.0088 | 0.1130 | no |
| M-gru vs M-quad, level none, n_train 64 | 4 | +0.1089 | +27.37 | 0.1786 | 1.00 | 0.1476 | 0.5096 | no |
| M-gru vs M-quad, level fb, n_train 64 | 5 | +0.0167 | +4.20 | 0.0163 | 1.00 | 0.0468 | 0.3747 | no |
| M-quad: fb vs none, n_train 64 | 4 | +0.1016 | +25.53 | 0.1588 | 1.00 | 0.1346 | 0.5096 | no |
| M-gru: fb vs none, n_train 64 | 5 | +0.0101 | +2.55 | 0.0077 | 0.80 | 0.0216 | 0.2087 | no |

**Family D:** 4 comparisons, 0 with fewer than two common admissible seeds (no test), 0 resolved after Holm.

| comparison (first vs second) | common seeds | cost of the second minus cost of the first, mean EUR/kg | relative to the twin % | resolution threshold EUR/kg | sign consistency | p | p Holm | resolved |
|---|---|---|---|---|---|---|---|---|
| D-gru vs M-gru level none, n_train 16 | 5 | +0.0066 | +1.66 | 0.0081 | 0.80 | 0.0870 | 0.2674 | no |
| D-gru vs M-gru level fb, n_train 16 | 5 | -0.0048 | -1.21 | 0.0066 | 0.80 | 0.1136 | 0.2674 | no |
| D-gru vs M-gru level none, n_train 64 | 5 | +0.0132 | +3.31 | 0.0146 | 0.80 | 0.0669 | 0.2674 | no |
| D-gru vs M-gru level fb, n_train 64 | 5 | +0.0030 | +0.76 | 0.0092 | 0.60 | 0.4137 | 0.4137 | no |

**Family P:** 12 comparisons, 2 with fewer than two common admissible seeds (no test), 1 resolved after Holm.

| comparison (first vs second) | common seeds | cost of the second minus cost of the first, mean EUR/kg | relative to the twin % | resolution threshold EUR/kg | sign consistency | p | p Holm | resolved |
|---|---|---|---|---|---|---|---|---|
| P-lin vs P-mlp, K 5 | 4 | +0.0356 | +8.94 | 0.0687 | 0.75 | 0.1979 | 1.0000 | no |
| P-lin vs P-gru, K 5 | 3 | +0.0254 | +6.38 | 0.0656 | 0.67 | 0.2380 | 1.0000 | no |
| P-mlp vs P-gru, K 5 | 3 | -0.0039 | -0.99 | 0.1016 | 0.33 | 0.8830 | 1.0000 | no |
| P-lin vs P-mlp, K 10 | 3 | -0.0009 | -0.23 | 0.0982 | 0.33 | 0.9720 | 1.0000 | no |
| P-lin vs P-gru, K 10 | 5 | +0.0047 | +1.19 | 0.0467 | 0.60 | 0.7926 | 1.0000 | no |
| P-mlp vs P-gru, K 10 | 3 | -0.0129 | -3.25 | 0.0520 | 0.67 | 0.3970 | 1.0000 | no |
| P-lin vs P-mlp, K 20 | 3 | +0.0320 | +8.03 | 0.0463 | 1.00 | 0.0971 | 0.8737 | no |
| P-lin vs P-gru, K 20 | 5 | -0.0094 | -2.37 | 0.0146 | 0.80 | 0.1471 | 1.0000 | no |
| P-mlp vs P-gru, K 20 | 3 | -0.0398 | -10.00 | 0.0798 | 1.00 | 0.1648 | 1.0000 | no |
| P-lin vs P-gru, K 40 | 5 | -0.0150 | -3.76 | 0.0045 | 1.00 | 0.0007 | 0.0074 | yes (second cheaper) |

Read with the table: M-gru is cheaper than M-quad in every comparison that could be tested, with sign consistency 1.00, and the feedback level fb is cheaper than none for M-gru at every budget (sign consistency 1.00 at `n_train` 8-32, 0.80 at 64); with five seeds and 16 tests in the family none of these passes Holm. The only resolved statement of the study is P-gru cheaper than P-lin at `K` 40 (P-lin has no seed variation there). D-gru against M-gru is not resolved at either budget or level.

### 16.6 Feedback (Q3)

| `n_train` | M-gru none: gap median %, margin | M-gru fb: gap median %, margin | M-gru lab (descriptive): gap median %, margin | M-quad none | M-quad fb |
|---|---|---|---|---|---|
| 8 | +11.28, 0.0711 (5 of 5) | +6.06, 0.0285 (5 of 5) | +22.25, 0.0682 (5 of 5) | +137.04, 0.1164 (1 of 5) | +68.41, 0.1152 (1 of 5) |
| 16 | +7.44, 0.0463 (5 of 5) | +3.98, 0.0224 (5 of 5) | +8.97, 0.0381 (5 of 5) | +49.98, 0.1043 (3 of 5) | +13.84, 0.0449 (5 of 5) |
| 32 | +7.04, 0.0416 (5 of 5) | +3.23, 0.0195 (5 of 5) | +8.83, 0.0361 (5 of 5) | +45.77, 0.0742 (2 of 5) | +11.16, 0.0318 (3 of 5) |
| 64 | +7.07, 0.0357 (5 of 5) | +3.87, 0.0221 (5 of 5) | +7.62, 0.0348 (5 of 5) | +22.36, 0.0934 (4 of 5) | +8.51, 0.0203 (5 of 5) |

The inline measurement roughly halves the margin and the gap of M-gru at every budget and makes M-quad admissible in 5 of 5 seeds at `n_train` 16 and 64; it does not bring either arm within 2 %. What remains with feedback is a margin of about 0.02 kg/kg against the 0.0074 of the twin. A plausible reading, not tested here: a slow bias correction removes the mean error of the map, not its variation over the day. With the laboratory sample every 4 h (level lab, placeholders of Befund 112) the median gap is not smaller than without feedback at any budget, although the median margin is slightly smaller.

### 16.7 Map error at the limit (stage 0, descriptive; the exploitation gap of Befund 113)

At 75 weather and throughput points the optimum of the learned map at the limit (bound 0.15 kg/kg) is evaluated on the truth: predicted minus true moisture at the chosen point, and how many of the chosen points lie truly above the limit. Per `n_train` the median over the five seeds of the mean error, the range of that mean, and the count per seed.

| model | `n_train` | moisture error, median of the seed means kg/kg | range of the seed means | points truly above the limit, per seed (of 75) | table nodes valid, min over seeds % |
|---|---|---|---|---|---|
| GRU fixed point (M-gru) | 8 | -0.0530 | -0.3172 .. -0.0187 | 40, 75, 75, 56, 75 | 96.9 |
| GRU fixed point (M-gru) | 16 | -0.0185 | -0.0545 .. -0.0086 | 60, 56, 75, 55, 75 | 97.1 |
| GRU fixed point (M-gru) | 32 | -0.0135 | -0.0205 .. -0.0039 | 49, 54, 72, 50, 74 | 97.4 |
| GRU fixed point (M-gru) | 64 | -0.0077 | -0.0147 .. -0.0021 | 48, 61, 75, 62, 41 | 97.0 |
| static quadratic (M-quad) | 8 | -0.7151 | -0.7567 .. +0.0415 | 15, 75, 75, 72, 75 | 100.0 |
| static quadratic (M-quad) | 16 | -0.7209 | -0.7326 .. +0.0603 | 7, 75, 75, 75, 75 | 100.0 |
| static quadratic (M-quad) | 32 | -0.5419 | -0.7137 .. +0.0021 | 25, 75, 75, 42, 75 | 100.0 |
| static quadratic (M-quad) | 64 | -0.5755 | -0.7048 .. -0.0588 | 44, 75, 75, 67, 75 | 100.0 |

Both learned maps err to the unsafe side at their own optimum: the optimiser goes where the model underestimates the moisture. The GRU map underestimates by about 0.008 kg/kg at `n_train` 64 (the order of Befund 113), the static quadratic regression by several tenths of a kg/kg: it has no usable steady response near the limit, which is why M-quad without feedback is admissible only with large margins (median per budget 0.07-0.12 kg/kg) or not at all.

### 16.8 Family P: distribution shift, guard, computing time (addition C; descriptive)

Share of the decisions on the test days whose inputs lie outside the range of the training labels (smallest and largest value over the admissible seeds), and the share of the decisions in which the guard changed the command (all admissible seeds together; 40 days x 288 decisions per cell, computed from the counts of the analysis).

| arm | `K` | admissible seeds | outside the label range, min .. max | guard: box | guard: band | guard: daily target | guard: last hour | largest deviation from the daily target kg |
|---|---|---|---|---|---|---|---|---|
| P-lin | 5 | 5 of 5 | 0.253 .. 1.000 | 0.461 | 0.313 | 0.004 | 0.042 | 0.0207 |
| P-lin | 10 | 5 of 5 | 0.003 .. 1.000 | 0.330 | 0.314 | 0.001 | 0.042 | 0.0212 |
| P-lin | 20 | 5 of 5 | 0.000 .. 1.000 | 0.339 | 0.322 | 0.000 | 0.042 | 0.0201 |
| P-lin | 40 | 5 of 5 | 0.000 .. 0.000 | 0.341 | 0.325 | 0.000 | 0.042 | 0.0006 |
| P-mlp | 5 | 4 of 5 | 1.000 .. 1.000 | 0.301 | 0.097 | 0.183 | 0.042 | 0.0218 |
| P-mlp | 10 | 3 of 5 | 0.806 .. 1.000 | 0.334 | 0.188 | 0.151 | 0.042 | 0.0220 |
| P-mlp | 20 | 3 of 5 | 1.000 .. 1.000 | 0.279 | 0.278 | 0.038 | 0.042 | 0.0216 |
| P-mlp | 40 | 1 of 5 | 1.000 .. 1.000 | 0.247 | 0.496 | 0.030 | 0.042 | 0.0221 |
| P-gru | 5 | 3 of 5 | 1.000 .. 1.000 | 0.278 | 0.133 | 0.165 | 0.042 | 0.0220 |
| P-gru | 10 | 5 of 5 | 0.832 .. 1.000 | 0.328 | 0.173 | 0.121 | 0.042 | 0.0218 |
| P-gru | 20 | 5 of 5 | 0.574 .. 1.000 | 0.426 | 0.307 | 0.088 | 0.042 | 0.0220 |
| P-gru | 40 | 5 of 5 | 0.461 .. 0.575 | 0.369 | 0.316 | 0.039 | 0.042 | 0.0221 |

* The margin is an input of the policies and the labels cover 0 to 0.04 kg/kg (section 13, point 5). A cell whose own margin is above 0.04 is outside the label range in every decision; that is the share 1.000 of P-mlp and of the small budgets. The check is per input (section 12, point 7) and tends to underestimate the shift.
* The guard is not a rare safety net here: it changes the command of a policy in a large part of the decisions (box and band each in about a third at `K` 40), and the last hour of every day is set by the guard (share 12 / 288 = 0.042). The twin runs at one of the two ends of the throughput band for 94 % of the day (mean over the test days, computed from the records of the main artifact), so a policy that misses an end of the band by any amount counts as a band intervention; which command the box interventions concern was not analysed. For the arms of the families M and D the guard never intervened (all four counts 0 in every admissible cell).

Computing time per decision (mean over the days of a cell, median over the admissible cells of the largest budget):

| controller | ms per decision |
|---|---|
| M-quad, level none | 5.750 |
| M-quad, level fb | 6.110 |
| M-gru, level none | 6.349 |
| M-gru, level fb | 6.145 |
| M-gru, level lab | 5.853 |
| D-gru, level none | 84.356 |
| P-lin, level none | 0.034 |
| P-mlp, level none | 0.042 |
| P-gru, level none | 0.054 |
| M-tab (control) | 5.134 |
| C3-15 (true model, 15 min horizon) | 119.160 |

The policies decide in less than a tenth of a millisecond, the table arms in about 6 ms, D-gru in about 84 ms. Computing time is no obstacle for any arm on the 5 min grid; the probe of the load-following study put C3 at 1679 s per day-run (section 9).

Horizon problems of D-gru that ended without a solution (the last command is held; counted from the records of the main artifact, five seeds x 40 days x 288 decisions): 90 at `n_train` 16, 105 at `n_train` 64.

### 16.9 Deployable margin (descriptive: the reading names no arm)

Decision 7 asks for the deployable margin of the arms the reading names; it names none. The search on 40 tuning days of the training months was run for every arm with a reading at the largest budget (section 13, point 6) and is reported as a description: could the margin have been set without the test days?

| arm | level | margin found on the tuning days (of 5 seeds) | margin on tuning days minus margin on test days, per seed kg/kg | test days of 40 violated at the tuning margin, per seed | largest violation on the test days kg/kg | gap at the tuning margin, median % |
|---|---|---|---|---|---|---|
| M-quad | none | 4 | -0.0027, +0.0002, +0.0001, -0.0021, - | 2, 0, 0, 1, - | 0.0052 | +21.58 |
| M-quad | fb | 5 | +0.0051, -0.0039, -0.0018, +0.0002, -0.0031 | 0, 1, 2, 0, 2 | 0.0069 | +7.55 |
| M-gru | none | 5 | +0.0035, +0.0000, -0.0012, -0.0008, -0.0008 | 0, 0, 1, 1, 1 | 0.0054 | +6.67 |
| M-gru | fb | 5 | +0.0021, +0.0002, +0.0010, -0.0053, +0.0006 | 0, 0, 0, 2, 0 | 0.0045 | +4.14 |
| D-gru | none | 5 | -0.0001, +0.0021, -0.0031, +0.0035, +0.0010 | 0, 0, 2, 0, 0 | 0.0032 | +2.96 |
| P-lin | none | 5 | -0.0047, -0.0047, -0.0047, -0.0047, -0.0047 | 2, 2, 2, 2, 2 | 0.0074 | +12.70 |
| P-mlp | none | 3 | -0.0086, no test margin, -, no test margin, - | 3, 5, -, 3, - | 0.0760 | +21.80 |
| P-gru | none | 5 | +0.0143, +0.0029, -0.0111, -0.0094, +0.0094 | 0, 0, 4, 1, 0 | 0.0159 | +12.41 |

Seeds (of 5) whose tuning margin holds the limit on all 40 test days within the violation tolerance: M-quad none 2, M-quad fb 2, M-gru none 2, M-gru fb 4, D-gru none 4, P-lin none 0, P-mlp none 0, P-gru none 3. "-" = no margin on the tuning days. For no arm does a margin found on 40 days of the training months hold on the test months in every seed. The margin of the reading is therefore an optimistic number for every learned arm: it was found on the days on which it is reported. The same question was not put to the twin (its margin, too, was found on the test days, in the load-following study).

## 17. Answers to the questions of section 2

* **Q1 (indirect form).** With the steady map learned from `n_train` runs and no feedback, the law of C2 costs 7.07 % more than with the true map at `n_train` 64 (7.44 % at 16, 11.28 % at 8; M-gru, median). From `n_train` 16 on, more runs do not close the gap. The static quadratic regression does not suffice: it is admissible in at most 4 of 5 seeds and costs 22.36 % more at `n_train` 64. The sequence model is cheaper than the static baseline in every comparison that could be tested (sign consistency 1.00), none of which passes Holm with five seeds. The gap of M-gru is the margin, not the set points.
* **Q2 (direct form).** No. At 40 solved days P-gru costs 10.47 % more than its expert and is level with the fixed set point; P-lin costs 14.15 % more and P-mlp is admissible in 1 of 5 seeds. The learning curve of P-gru falls from `K` 10 to 40 (21.18 %, 12.05 %, 10.47 %), slowly between 20 and 40; how many days it would need is not determined. A decision costs under 0.1 ms, but computing time is not the bottleneck of any arm.
* **Q3 (feedback).** The inline measurement closes about half of the gap of M-gru (7.07 % -> 3.87 % at `n_train` 64, 7.44 % -> 3.98 % at 16; not resolved after Holm) and makes the static baseline admissible in 5 of 5 seeds at `n_train` 16 and 64; it does not close the gap. The laboratory sample every 4 h does not help.
* **The smallest gap of the study** has the sequence model in the loop with the measured history: D-gru, 2.98 % at `n_train` 64 with a margin of 0.0088 kg/kg, close to the twin's. It is not resolved against M-gru with feedback.

**In short.** On this simulator, learning the model costs about 3 to 7 % of the cost per kg against the known model, which is more than the 2.07 % that load following gains with the known model: with a learned model, the benefit of the schedule is eaten by the margin, and no learned arm is cheaper than constant throughput with the true steady optimum (C0). "Prediction of optimised transient set points" by a network that outputs the commands (family P) is not competitive here; a learned model inside an optimiser with measured feedback (D-gru, M-gru with fb) is the form that comes closest, and its price is the margin.

## 18. Caveats of the results (in addition to section 8)

1. **The margin of the reading is found on the test days** (as for C0-C4, by design). Section 16.9 shows that a margin found on other days is violated on the test days in one or more seeds of every arm; the gaps above are therefore optimistic for a deployed learned controller.
2. **Five seeds.** The spread over seeds is of the size of the effects (M-gru without feedback at `n_train` 64: +0.78 to +10.11 %). One rank statement of the 27 tested is resolved after Holm. The classes U0-U3 are read on medians; family P sits on the boundary between U3 and U0.
3. **The twin is not reproducible bit for bit** (section 15): its cost differs by 0.0016 % between the load-following run and this run, single days by up to 0.15 %. This is far below every gap reported here and does not touch a class, but the cause sits in code that carries Befund 105 and Befund 126.
4. **The twin's margin is taken from the load-following artifact** (0.00742 kg/kg, found there on all 40 days); the control M-tab, searched here on the critical subset, needs 0.00657. If the twin could run at that margin, every gap would be larger by about 0.2 % (the margin share of M-tab).
5. **Untuned configurations** (section 8): one architecture and one training recipe per arm, the settings of Befund 94 for the models and one small set for the policies; no DAgger, no margin-aware training, no tuning of the feedback filter.
6. **The feedback of family M is a steady bias correction** on a map without dynamics (section 13, point 2); a dynamic observer could do better. The level fb assumes an inline moisture measurement that the plant does not have (Befund 56).
7. **The truth is the simulator** (section 8, addition D): the twin is a perfectly calibrated physics model. All values are placeholders or synthetic; no statement about a plant.

## 19. Open for the project lead

1. **Acceptance of the reading** U3 in the families M (both levels), D and P, with the note that family P is U3 by the letter and at the boundary to U0.
2. **The ten points settled without asking during the implementation** (section 13) are not yet confirmed; neither is the handling in point 4 below.
3. **Order dependence of the steady optimiser** (`SteadyStateOptimizer.solve` warm start, `SteadyMap` cache; section 15). Found by the control of this study, in code of the headroom and load-following packages. Not changed, not investigated beyond a scratch check. To decide: whether and how the effect on Befund 105 and Befund 126 is measured (for example the reference controllers with a cold start per query, or a multi-start, against the artifacts), and by which session.
4. **Register row and README paragraph.** Section 11 planned a `docs/12` row with the results. As for the load-following branch, this branch delivers the study document and the lab-book entry only; the row and the paragraph need the finding number and the decision of the project lead and are left to the managing session at merge.
5. **Possible follow-ups, none started:** a margin-aware training of the map or of the GRU (the gap is the margin); DAgger or more solved days for family P (its curve has not levelled off); a dynamic observer instead of the steady bias correction; the deployable margin as the pre-registered quantity of a next study (section 16.9).

