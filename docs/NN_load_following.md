# NN -- Throughput as a manipulated variable: load following of the granulator under a time-varying price

Status: **design fixed and committed before stage 0, the dry run and the main run** (sections 1-13;
decisions of the project lead of 2026-11-17 in section 10). Runs and results follow in sections 14
onwards, added after the runs (numbers only from the artifacts named there). Document number and
finding number are open (the managing session assigns them at merge). Branch `load-following`
(worktree, from `origin/master` ba02e0e). **Scenario on placeholders, not a plant statement.**

## 1. Why this study

The proposal motivates the dynamic control of the granulator with running the plant up and down
after the supply of renewable electricity. No study so far has tested that:

* In every control study the spray rate was fixed as the throughput (Befund 52: "otherwise
  'minimise energy' is degenerate"; `rl/env.py`, `ACTION_NAMES` = inlet temperature and gas
  velocity) or was a disturbance (Befund 105: levels within +-1.5 %, stress +-10 / +-20 %).
* The time-varying CO2 signal, the optional last variant of the headroom study, was not run
  (`docs/24_dynamic_headroom.md`, section 9).
* The price study (Befund 115) concerns heat pump and storage, not the process.

Befund 105 closed the RL line for the granulator in continuous operation at **fixed** throughput:
tracking the operating point is worth a median 8.93 % (R0 -> R1), the dynamics add 0.55 % (R1 -> R3,
upper bound 1.51 %). That reading says nothing about a case in which decisions carry over in time
because a production target may be met early or late. This is that case.

## 2. Question

With a production target per day and an hourly price, **does shifting the throughput in time lower
the cost per kg of product at a held moisture limit, against constant throughput with a
quasi-stationary optimiser?** If so: does a price rule suffice, a schedule on the steady-state map,
or does it take a dynamic optimisation (MPC), and is an RL-against-MPC study justified?

## 3. Line, truth model, placeholders

Truth = the grey-box model (`GreyBoxFluidBedGranulator`, continuous mode, RK4, sample time 5 s),
unchanged; no surrogate. Line settings, control grid, margin search and ambient profiles are READ
from the headroom config (`conf/headroom/dynamic_headroom.yaml`, unchanged; a test compares its
fingerprint with the artifact of Befund 105), so that Befund 105 is the reference at fixed
throughput:

* quality limit of the product moisture 0.15 kg/kg (placeholder) as a path condition at every truth
  sample, tolerance 1e-4 kg/kg; bed temperature <= 80 degC (placeholder) hard for the optimisers;
* minimum gas velocity 0.80 m/s (placeholder); nozzle pressure fixed at the Befund 52 fixed point;
* solid fraction at the lab line's nominal 0.30 (decision of the project lead of 2026-10-20 for
  `docs/24`, section 2: at 0.60-0.70 the limit does not bind on this line);
* rate limits of `ActionBound.max_rate_per_s`: 0.5 K/s inlet temperature, 0.05 m/s per s gas
  velocity, **0.02 (kg/h)/s spray rate**;
* **new: the spray rate is a command**, box 0.4-3.0 kg/h (`ActionBound`, placeholder; the
  laboratory range in `docs/02_prozess_setup.md` is 0.6-3.9 kg/h), nominal 1.5 kg/h.
* Product withdrawal by the unchanged level control (P controller on the solids mass, set point
  2.0 kg marked `TODO` in the config, time constant 600 s). **The bed-mass set point stays fixed**
  (decision 8). From these placeholders the mean residence time at the nominal point is about 4.4 h
  (2.0 kg / (1.5 kg/h x 0.30); derived here, not a finding).

**Throughput bands (decision 2).** *Main* band: +-20 % around 1.5 kg/h, cut with the admissible
range of stage 0; **the reading applies only here**. *Box*: the whole admissible part of the
ActionBound box, a sensitivity with C0, C2 and C4 only and without a reading. Reason for the narrow
main band: in the grey-box model the median particle diameter does not depend on the throughput, so
large throughput steps would be free of any product cost that a real line would have.

## 4. Signal, production target, day and tail, days

* **Price.** Day-ahead prices DE-LU 2024, hourly: the file, the adder (price + 0.08 EUR/kWh,
  placeholder) and the month split of Befund 115 are read from `conf/storage_price/storage_price.yaml`
  (raw file in `local/prices/`, SHA-256 checked). Training months Jan, Mar, Apr, Jun, Jul, Sep, Oct,
  Dec; **test months Feb, May, Aug, Nov.** Nothing of a test month is used for tuning.
* **Tariff floor** 0.01 EUR/kWh (placeholder; section 13, item 2).
* **Price information** as in Befund 115: the prices of the day are known at its start; the hour of
  the tail lies on the next day and is forecast by the same hour of the day itself. Costs are booked
  with the true tariff.
* **No CO2 signal** (decision 1).
* **Production target per day** (decision 3): the suspension sprayed in the 24 h equals that of the
  nominal point (1.5 kg/h x 24 h). A weekly target is reported descriptively only, with C2 on the
  steady-state map, without a reading.
* **Tail (decision 10).** Every day is followed by a tail of 1 h at the nominal throughput with the
  set points of C0. Its cost counts for every controller, the moisture limit holds there too, and
  MPC and preview plan to the end of the tail. A test shows that with a constant price the end of
  the day gives no advantage (`test_no_advantage_from_the_end_of_the_day_at_a_constant_price`).
* **Scenario = one day and its tail**, common initial state = steady optimum at the nominal
  throughput (bound `limit - start_margin` of the headroom config). Ambient air: the day profiles of
  the headroom family `ambient` (5 x 5 weather grid, `DiurnalDisturbanceDesign`), one grid point per
  day by a seeded permutation; weather and price are therefore independent.
* **Days (decision 5), rule fixed here before the dry run.** Candidate sets: *whole* = all days of
  the four test months (121); *stratified* = 10 days per test month, drawn by seed from the
  terciles of the daily price spread (3 low, 4 middle, 3 high; 40 days). **The whole test months
  are used if the wall-clock time extrapolated after the dry run and the probe is at most 30 h,
  otherwise the stratified days.** Extrapolation (`extrapolate_hours` in
  `scripts/run_load_following.py`): from the probe (full-length days, three training days, parallel
  load) the time per day-run of C3 and C4 and per day of a cheap run; counted day-runs with
  `S = 1 + ceil(log2(0.05 / 2.5e-4)) = 9` evaluations per bisection, `K = 8` critical days and `N`
  test days: C3 `K S + 3 N` (search on the subset, two verification passes, the constant-price
  control), C4 `K S + 3 N`, C4 of the box `K S + 2 N`, cheap runs `N (6 S + 6)`; wall-clock =
  1.25 x CPU time / workers + 0.3 h. If the stratified days are still above the budget, the weekly
  part is dropped, then the box part (decision 7; `choose_days_and_parts`); the parts that carry
  the reading are never cut. The choice is recorded in the probe artifact and in the chain log
  before the main run. Tuning of C1: 5 days per training month drawn the same way (2 / 1 / 2 per tercile; 40 days).
* **No standstill.** The spray rate stays inside its band; stopping and restarting the line is not
  in the model and not in this study.

## 5. Factor "carrier" (decision 1)

| variant | heater | priced by the hourly tariff | controllers | reading |
|---|---|---|---|---|
| **E** | electric, efficiency 0.98 (process config) | the whole power | C0-C4 | yes |
| **G** | gas, efficiency 0.80 (placeholder of `docs/24`), fuel 0.040 EUR/kWh + 55 EUR/t CO2 (the `Economics` of Befund 101) | fan, pump, atomiser only | C0, C1, C2 | no, reported next to E |

## 6. Controllers

All controllers command inlet temperature, gas velocity and spray rate, piecewise constant over
5 min, and plan for `X <= limit - m` with their own margin `m` (section 7). They decide during the
24 h of the day; the tail is the same law for all (section 13, item 1).

| label | controller |
|---|---|
| C0 | **reference:** constant nominal throughput; inlet temperature and gas velocity from the steady optimum at the current disturbance (R1 of Befund 105; reproduced by a test) |
| C1 | **price rule:** two throughput levels fixed at the start of the day; the hour is cheap when its price is at most the `q`-quantile of the prices of the day; cheap hours run at the upper level, the others at the lower one, placed so that the daily target is met; set points from the steady optimum at the current disturbance and throughput; `q` and the level spread tuned on the training days (steady-state map) |
| C2 | **schedule on the steady-state map:** at every full hour the exact programme over the hourly levels of the remaining day (41 levels across the band, dynamic programme over the level sum) on the steady power curve at the current disturbance (held), with the price information and the remaining target; no dynamics, no rate limits; set points as C1 |
| C3 | **economic MPC:** at every full hour the dynamic programme (same model as the truth, spray rate as third control, priced objective, remaining target as terminal condition) from the true state over the rest of the day and the tail, current disturbance held; it fixes the spray rate. Every 5 min the 2 h MPC of Befund 105 (R3, unchanged optimiser) sets inlet temperature and gas velocity from the true state with the planned spray rate as a known input (section 13, item 3) |
| C4 | **perfect preview:** one dynamic programme per day with the true prices and the true disturbance course, open loop; started at the schedule of C2 with the same information and never worse than it in its own model (upper bound, as R2 of Befund 105) |

**Negative control.** With a constant price (the reference tariff) C0, C2, C3 and C4 are run on the
same days at their margins of the main family: the gap to C0 then measures the curvature of the
steady-state map and the dynamics, not the signal.

RL (SAC, TD3) is **not** part of this study; it would be the follow-up if the reading is T3.

## 7. Stages, margins, metric

**Stage 0 -- steady-state map (minutes, no truth run).** For every weather grid point and a grid
of the spray rate over the box (step 0.05 kg/h): the steady optimum at the limit, its power and
heater share. *Admissible range:* the largest interval around the nominal rate in which a steady
optimum exists at every weather point with the bound `limit - 0.02` (placeholder). From it the two
bands. On the 40 training days (hourly mean of the true ambient air, margin 0): tuning of C1 and
the static gain of the schedule against constant throughput, per band and carrier.

**Dry run (rule "Long runs").** All units of the main run on days shortened to 1 h plus the tail,
on the whole test months, with a coarse margin tolerance (4e-3): surfaces failing units and
reports time and peak memory per worker. Gate of the chain: no errored unit, workers x peak memory
below 80 % of the RAM (also after the probe). **Probe:** C0-C4 on three training days of full length
(smallest, median and largest price spread of the tuning days) at the fixed margin 0.005 under
parallel load; gives the times of the extrapolation and the choice of the day set.

**Main run -- truth runs** on the chosen test days, 5 s samples, parts in this order: main family
(E, main band), decomposition (C0 at the margin of every other controller), constant-price control,
variant G, weekly target, box.

**Margins (decision 9).** As in `docs/24`, section 5: every controller gets the smallest
family-wide margin with which the path condition holds on all days of the family, tail included
(bisection, tolerance 2.5e-4 kg/kg, bracket [0, 0.05], doubled up to 0.12). C0, C1, C2 on all days;
C3 and C4 on a critical subset (the 8 days with the largest C2 violation at margin 0, bracket
[0, 2 x C2 margin]) and verified on all days; failing days join the subset. The margins of Befund
105 are not taken over. C1 and C2 ignore the dynamics of their own throughput steps; what that
costs appears as their margin. The reading is on the cost at the held constraint, not on the
margin (`CLAUDE.md`).

**Metric.** Cost per kg of dry product: sum over the days of the cost of the day and its tail
(hourly truth energy times the true tariff) divided by the sum of the dry product. Reported next
to it: energy, the margin, the largest violation (day and tail), the deviation from the daily
target, the share of the time at a throughput bound, the range of the median particle diameter and
the smallest fluidisation margins (diagnostic only), solver failures and fall-backs, time per
decision. `gap(X, Y) = (J_X - J_Y) / J_Y`.

## 8. Pre-registered reading (thresholds: decision 4)

Verbatim in the docstring of `scripts/analyze_load_following.py`, committed with this section. On
variant E, main band, test days, first class that applies:

| class | condition | consequence |
|---|---|---|
| **T0** | `(J_C0 - J_C4) / J_C0 < 2 %` | load following does not pay on this line, not even with perfect preview; the item is closed |
| **T1** | `gap(C1, C3) <= 2 %` | a price rule suffices; neither schedule, MPC nor RL justified |
| **T2** | `gap(C2, C3) <= 2 %` | a schedule on the steady-state map suffices; the dynamics add nothing |
| **T3** | `gap(C2, C3) > 2 %` | the dynamics pay: MPC justified; an RL-against-MPC study is justified only if the gap exceeds 5 % |

Every condition is also reported on its own, per test month and per tercile of the price spread.
`gap(C3, C4)` is reported as the value of perfect information. Reduced stage 1 (decided by stage 0
before any truth run): if the static gain on the training days is below 2 % **and** the admissible
range is narrower than +-10 % of the nominal value, only C0, C2 and C4 are run; the reading is then
T0 or "none (reduced)". The reading is invalid, and reported as such, if a margin search of the
family did not converge or a unit of the family ended with an error.

Expectation, declared as such and not a result: the spray rate crosses its whole box in about 130 s
(box width over rate limit) and the moisture and temperature modes are 15-80 s (`docs/10`, a figure
as supplied there, not matched to a finding), both short against an hourly price; T1 or T2 is the
likely reading.

## 9. What the study will not show (caveats fixed beforehand)

* **Product quality is the moisture only.** In the grey-box model the median particle diameter does
  not depend on the operating point; a real change of throughput changes growth and residence time.
  A positive reading is conditional on a size distribution that tolerates the throughput moves.
* **Continuous operation, no standstill, no batch.** The headroom test for batch operation
  (`docs/12`, section I, item 8) stays a follow-up of its own.
* **True state measured**, no estimator, no measurement noise (as Befund 105; `docs/12` I.7).
* **One year of prices, a constant tariff adder, a tariff floor, weather independent of price.**
* **Common start state.** Every run starts at the steady optimum with the start margin 0.01, above
  the margins the controllers end with; a dynamic programme uses this once at the start of the day
  (seen in the test of the tail on a 2 h day; small against a 24 h day).
* **Placeholders:** the throughput box, the moisture limit, the minimum gas velocity, the bed-mass
  set point, the heater efficiencies, the gas price, the whole process config. Whether the moisture
  limit binds at the plant is open (`docs/12` I.9).

## 10. Decisions of the project lead (2026-11-17)

1. Reading on variant E (electric heater, the whole power priced); G with C0, C1, C2 only, reported
   next to it. No CO2 signal.
2. Throughput: main variant +-20 % around 1.5 kg/h, cut with the admissible range of stage 0
   (reading only here). Sensitivity "whole admissible box" with C0, C2, C4 only, without a reading.
3. Production target per day. Weekly target descriptive only, C2 on the steady-state map.
4. Thresholds 2 % and 5 %.
5. Test days: whole test months if the extrapolation after the dry run gives at most 30 h, else the
   40 stratified days; the rule committed before the dry run (section 4), the choice recorded after
   the dry run and before the main run.
6. Exception from `DataSpec.new_study` approved (the study generates no training data; precedent
   `docs/12`, section L).
7. Budget 30 h wall-clock. Order of cuts fixed beforehand: whole months -> 40 days, then the weekly
   sensitivity, then the box sensitivity. Parts that carry the reading are not cut.
8. The bed-mass set point stays fixed.
9. Margins family-wide per controller.
10. Day boundary: a tail of 1 h with the set points of C0 at the nominal point (section 4).

Unattended run: no questions; where the instruction left room, the most conservative reading was
taken and is listed in section 13. No merge, no push.

## 11. Questions for the application partner

* Turn-down: between which throughputs can a line run without a product change, and how fast may
  the feed be changed?
* Over which window is the production fixed (shift, day, campaign)?
* Does a change of throughput shift the particle size or other release properties?
* Which carrier heats the process air of the line considered, and is an electric heater planned?
* Does the moisture limit bind in normal operation (`docs/12` I.9)?

## 12. Implementation and run plan

* Package `src/twinlink_ml/load_following/` (`config`, `market`, `steady`, `stage0`, `dynamic`,
  `truth`, `controllers`, `study`, `memory`), settings `conf/load_following/load_following.yaml`,
  tests `tests/test_load_following.py`. **The headroom package is not changed**: it is imported
  (`SteadyStateOptimizer`, `DynamicOptimizer`, `symbolic_derivative`, `ramp_mean`, the margin
  bisections, the weather grid); the truth with three commands and the programme with the spray
  rate as a control are new modules next to it.
* Tests fixed beforehand and written: with the spray rate held the interval function is the
  headroom one (1e-12) and the programme gives the headroom optimum; the truth with three commands
  is the headroom truth at the nominal spray rate (bit for bit); C0 reproduces the R1 commands on a
  headroom scenario; C3 with the spray rate pinned reproduces the energy of R3 on the same day;
  the headroom and process fingerprints equal those recorded in `dynamic_headroom_v2`; the schedule
  is exact against brute force and flat at a constant price where the cost is convex; the daily
  target is met; training and test days come from disjoint months and the tail is forecast from the
  day itself; the tail removes the end-of-day effect.
* Scripts `scripts/run_load_following.py` (stages `stage0`, `dryrun`, `probe`, `main`; `--resume`;
  one unit = controller x day x margin, saved as soon as it is finished; a failed unit is logged
  and recorded, it does not discard the others) and `scripts/analyze_load_following.py`; clean
  tree, code location checked by `provenance()`, outputs as `.json.gz`.
* Workers: 8 (physical cores of the run machine); the dry run and the probe report the peak
  memory per worker, workers x peak must stay below 80 % of the 32 GB.
* Order: commit (this document, code, tests, reading) -> stage 0 -> full suite -> dry run -> probe
  and choice of the day set -> main run detached from the session, in the worktree with
  `tools/run_here.sh`. Full suite, dry run, probe and main run start only when no run of another
  study computes on the machine (checked every 10 min; waiting and start times are logged). From
  the full suite on, the steps run as one chain outside the session
  (`C:/twinlink_work/loadf_chain.sh`, log `C:/twinlink_work/loadf_chain.log`), which commits the
  output of every stage on the branch before the next stage starts (denylist test first).

## 13. Decisions taken without asking (unattended run; the most conservative reading each time)

1. **Tail law.** "Set points of C0 at the nominal point" is implemented as the law of C0 itself:
   nominal spray rate, inlet temperature and gas velocity from the steady optimum at the current
   disturbance, **with the family-wide margin of C0** -- the same commands for every controller,
   applied by the runner and not by the controller. C4 plans with the tail commands of the true
   disturbance course, C3 with those at the current disturbance. (A fixed set point of the nominal
   disturbance would violate the limit on warm, humid days.)
2. **Tariff floor 0.01 EUR/kWh.** With a negative tariff the dynamic programmes would be paid for
   wasting energy, which C0-C2 never do; the gap would then measure an artefact of the tariff
   placeholder. The floor applies to every controller and to the booking; the number of hours it
   changes is recorded in the stage-0 artifact.
3. **C3 has two layers.** With an hourly programme applied open loop C3 would react to the
   disturbance twelve times more slowly than C0-C2 (steady optimum every 5 min) and pay for it in
   margin; the gap C2-C3 would then be biased towards "the dynamics add nothing". The unchanged R3
   optimiser of Befund 105 gives C3 the same 5 min feedback.
4. **C4 starts at the C2 schedule** computed with the same perfect information and never takes a
   plan its own model rates worse (non-convex problem; safeguard of R2 in Befund 105, counted).
5. **Production target in suspension mass**, measured on the realised (rate-limited) spray rate of
   the truth; C2 and C3 plan with the remaining target; C2 takes the remainder exactly in the last
   hour. The metric is normalised by the dry product, so small deviations do not move it.
6. **Programme numerics.** The target is a chain over the cumulative mass (a dense row made the
   Jacobian eight times as expensive at 60 intervals); only the two most recent horizon lengths are
   kept per worker (memory).
7. **Tuning of C1 on the steady-state map**, not on the truth: 40 training days, hourly mean of the
   true ambient air, margin 0; ties go to the smaller spread and quantile.
8. **Variant G: set points not re-optimised.** Inlet temperature and gas velocity stay those of
   minimum total power; only the throughput schedule uses the gas-electricity cost. Cheap, the same
   for C0-C2, and declared.
9. **Admissible range of stage 0** with the bound `limit - 0.02`, so that margins up to 0.02 keep a
   steady optimum at every weather point.
10. **Critical subset and bracket of the expensive searches** as in the headroom driver; an errored
    unit counts as holding in the search and makes the reading invalid.
11. **Constant-price control at the margins of the main family** (no own search); its violations
    are reported.
12. **Decomposition added** (C0 at the margin of every other controller), cheap, to separate the
    margin share from the schedule share as in `docs/24`, section 6.
13. **Weekly target on the first four whole weeks of every test month**, whichever day set is
    chosen (steady-state map only; with stratified days there are no whole weeks).
14. **Dry run and probe** as in section 7 (the rule "Long runs" asks for all units at minimal cost
    and forbids estimating from a single unit; shortened days cannot give the times of full-length
    programmes, so the times come from the probe).
15. **Disclosure (development order).** Before this commit, development checks produced numbers that
    are not artifacts: a smoke run of stage 0 (training days only: admissible range, the tuned rule,
    the static gain, which lies above the 2 % of the reduction rule in the main band), and smoke
    runs of the five controllers at the fixed margin 0.005 on two days shortened to 3 h, which lie
    in test months (days 40 and 131). The thresholds are those of the project lead; neither the
    config nor the reading was changed after these numbers were seen.
