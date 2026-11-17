# NN -- Throughput as a manipulated variable: load following of the granulator under a time-varying price

Status: **DRAFT for review by the project lead (2026-11-17). Nothing is committed, built or run.**
Every threshold, band and budget below is a proposal; section 10 lists the decisions that have to be
taken before the pre-registration is committed. Document number and finding number are open (the
managing session assigns them at merge). Proposed branch `load-following`. **Scenario on
placeholders, not a plant statement.**

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
unchanged; no surrogate. Process config and line settings as in the headroom study
(`conf/headroom/dynamic_headroom.yaml`), so that Befund 105 is the reference at fixed throughput:

* quality limit of the product moisture 0.15 kg/kg (placeholder) as a path condition at every truth
  sample, tolerance 1e-4 kg/kg; bed temperature <= 80 degC (placeholder) hard for the optimisers;
* minimum gas velocity 0.80 m/s (placeholder); nozzle pressure fixed at the Befund 52 fixed point;
* solid fraction at the lab line's nominal 0.30 (decision of the project lead of 2026-10-20 for
  `docs/24`, section 2: at 0.60-0.70 the limit does not bind on this line);
* rate limits of `ActionBound.max_rate_per_s`: 0.5 K/s inlet temperature, 0.05 m/s per s gas
  velocity, **0.02 (kg/h)/s spray rate**;
* **new: the spray rate is a command**, box 0.4-3.0 kg/h (`ActionBound`, placeholder; the
  laboratory range in `docs/02_prozess_setup.md` is 0.6-3.9 kg/h), nominal 1.5 kg/h. The admissible
  part of the box is a result of stage 0, not an input.
* Product withdrawal by the unchanged level control (P controller on the solids mass, set point
  2.0 kg marked `TODO` in the config, time constant 600 s). The bed-mass set point is **not**
  released (section 10, item 8). From these placeholders the mean residence time at the nominal
  point is about 4.4 h (2.0 kg / (1.5 kg/h x 0.30); derived here, not a finding).

**Carrier of the heat.** The process config books the air heater as electric (efficiency 0.98,
config comment: "at application partner possibly gas"). The headroom study booked CO2 with a
gas-fired heater (efficiency 0.80, placeholder). The carrier decides how much of the power sees an
electricity signal, so it is a factor of the design, not a setting (section 5).

## 4. Signal, production target, scenarios

* **Price.** Day-ahead prices DE-LU 2024, hourly: the file, the tariff (price + 0.08 EUR/kWh,
  placeholder) and the split of Befund 115 are reused unchanged (`docs/32_storage_price.md`,
  section 3; raw file in `local/prices/`, SHA-256 in `conf/storage_price/storage_price.yaml`).
  Training and tuning months Jan, Mar, Apr, Jun, Jul, Sep, Oct, Dec; **test months Feb, May, Aug,
  Nov.** Nothing of a test month is used for tuning or selection. The environment guard of Befund 115
  (no observation beyond the training months) is taken over together with its test.
* **Price information** as in Befund 115: at hour `t` of day `D` the prices of `D` are known to its
  end, later hours are forecast by the same hour of `D`.
* **Gas price** (variant G below): 0.040 EUR/kWh + 55 EUR/t CO2 (the `Economics` of Befund 101, as
  used in `docs/32`, section 2), constant.
* **Production target.** Per calendar day the dry product equals that of the nominal point
  (1.5 kg/h x 0.30 x 24 h of solids fed), a hard constraint of every optimiser and met by
  construction by the rule. Variant: target per week (more freedom).
* **Scenario = one day**, 24 h, common initial state = steady optimum at the nominal throughput.
  Ambient air: the day profiles of the headroom family `ambient` (5 x 5 weather grid,
  `DiurnalDisturbanceDesign`), assigned to the days by a seeded permutation; weather and price are
  therefore independent (caveat, section 9).
* **Days.** Proposal: 10 days per test month, drawn by seed and stratified by the terciles of the
  daily price spread (40 test days); tuning on 40 training days drawn the same way. Whole months
  are the alternative if the dry run shows that the budget allows it (section 10, item 5).
* **No standstill.** The spray rate stays inside its box; stopping and restarting the line (hot
  standby, start-up losses) is not in the model and not in this study.

## 5. Factor "carrier" (which power sees the signal)

| variant | heater | priced by the hourly signal | meaning |
|---|---|---|---|
| **E** | electric, efficiency 0.98 (process config) | the whole power | electrified line; upper estimate |
| **G** | gas, efficiency 0.80 (placeholder of `docs/24`) | fan, pump, atomiser only | plant-like today |

Proposal: the pre-registered reading is taken on **E**; **G** is run with the rule and the
schedule only and reported next to it as what is left with a gas-fired heater. A CO2 signal (hourly
emission factor of the grid) is a third candidate; no data source is named in the repository, so it
is not part of the draft (section 10, item 1).

## 6. Controllers

All controllers command inlet temperature, gas velocity and spray rate, piecewise constant over
5 min, and plan for `X <= limit - m` with their own margin `m` (section 7).

| label | controller |
|---|---|
| C0 | **reference:** constant nominal throughput; inlet temperature and gas velocity from the steady optimum at the current disturbance (R1 of Befund 105) |
| C1 | **price rule:** two throughput levels; the hour is cheap when its price is at most the `q`-quantile of the price information of the day; cheap hours run at the upper level, the others at the lower one, the levels chosen so that the daily target is met; set points from the steady optimum at the current throughput; `q` and the level spread tuned on the training days |
| C2 | **schedule on the steady-state map:** per day one programme over the hourly throughput that minimises the priced steady-state power `P*(m_spray, disturbance)` under the daily target; no dynamics, no rate limits; re-solved every hour with the price information |
| C3 | **economic MPC:** the dynamic model of the headroom study extended by the spray rate as a third control, priced objective, remaining daily target as a terminal constraint, horizon to the end of the day, re-solved every hour, true state measured, current disturbance held |
| C4 | **perfect preview:** one dynamic programme per day with the true prices and disturbances, open loop (upper bound, as R2 of Befund 105) |

**Negative control.** With a constant price C2, C3 and C4 are run on the same days: the gap to C0
then measures only the curvature of the steady-state map and the numerics, not the signal.

RL (SAC, TD3) is **not** part of this study; it would be the follow-up if the reading is T3
(section 8).

## 7. Stages, margins, metric

**Stage 0 -- steady-state map (minutes, no dynamics).** For every weather grid point and a grid of
the spray rate over its box: the steady optimum of inlet temperature and gas velocity at the limit,
its power by component, and the specific energy per kg of dry product. Reported: the admissible
throughput range (where a steady state within the limits exists), the shape of the specific energy
over the throughput, and the static gain `g_static = (J_C0 - J_C2) / J_C0` on the training days
with margin 0. Stage 0 fixes the box of the following stages and nothing else.

**Stage 1 -- truth runs.** C0-C4 on the test days against the truth, 5 s samples.

**Margins (held constraint).** As in `docs/24`, section 5: every controller gets the smallest
family-wide margin with which the path condition holds in all test days (bisection, tolerance
2.5e-4 kg/kg; for C3 and C4 on a critical subset, verified on all days). The margins of Befund 105
are not taken over: throughput moves are a disturbance the fixed-throughput study did not contain.
C1 and C2 ignore the dynamics of their own throughput steps; what that costs appears as their
margin. This follows the rule of `CLAUDE.md`: the reading is on the cost at the held constraint,
not on the margin.

**Metric.** Cost in EUR per kg of dry product per day at the held path condition and the met
target; reported next to it: energy, CO2 (repository factors 0.366 / 0.201 kg/kWh), the margin, the
largest violation, the share of the time at a throughput bound, the excursion of the median
particle diameter and of the fluidisation window (diagnostic only), solver failures, time per
decision. Median and range over the days; gaps on the summed test-day costs,
`gap(X, Y) = (J_X - J_Y) / J_Y`.

## 8. Pre-registered reading (proposal; thresholds of `docs/24` and `docs/32`)

To be written into the docstring of `scripts/analyze_load_following.py` and committed before the
first run of stage 1. On variant E, test days, first that applies:

| class | condition | consequence |
|---|---|---|
| **T0** | `(J_C0 - J_C4) / J_C0 < 2 %` | load following does not pay on this line, not even with perfect preview; the item is closed |
| **T1** | `gap(C1, C3) <= 2 %` | a price rule suffices; neither schedule, MPC nor RL justified |
| **T2** | `gap(C2, C3) <= 2 %` | a schedule on the steady-state map suffices; the dynamics add nothing |
| **T3** | `gap(C2, C3) > 2 %` | the dynamics pay: MPC justified; an RL-against-MPC study is justified only if the gap exceeds 5 % |

Every condition is also reported on its own, per test month and per tercile of the price spread.
`gap(C3, C4)` is reported as the value of perfect information. If stage 0 gives
`g_static < 2 %` **and** the admissible throughput range is narrower than +-10 % of the nominal
value, stage 1 is run with C0, C2 and C4 only (decision before stage 1, logged).

Expectation, declared as such and not a result: the spray rate crosses its whole box in about 130 s
(box width over rate limit) and the moisture and temperature modes are 15-80 s (`docs/10`, a figure
as supplied there, not matched to a finding), both short against an hourly price;
T1 or T2 is the likely reading. The study is still needed, because the size of the static gain and
the margin that the throughput steps cost are unknown.

## 9. What the study will not show (caveats fixed beforehand)

* **Product quality is the moisture only.** In the grey-box model the median particle diameter does
  not depend on the operating point; a real change of throughput changes growth and residence time.
  A positive reading is conditional on a size distribution that tolerates the throughput moves.
* **Continuous operation, no standstill, no batch.** The headroom test for batch operation
  (`docs/12`, section I, item 8) stays a follow-up of its own.
* **True state measured**, no estimator, no measurement noise (as Befund 105; `docs/12` I.7).
* **One year of prices, a constant tariff adder, weather independent of price.**
* **Placeholders:** the throughput box, the moisture limit, the minimum gas velocity, the bed-mass
  set point, the heater efficiencies, the whole process config. Whether the moisture limit binds at
  the plant is open (`docs/12` I.9).

## 10. Decisions needed from the project lead before the pre-registration

1. **Signal and carrier of the reading:** E (electric heater, whole power priced) as proposed, or G,
   or a CO2 signal (then: which data source).
2. **Throughput box:** the whole admissible part of 0.4-3.0 kg/h as proposed, or a narrower band
   (for example the +-20 % of the stress family of Befund 105).
3. **Production target:** per day as proposed; the weekly variant in or out.
4. **Thresholds:** 2 % and 5 % as in `docs/24` and `docs/32`.
5. **Test days:** 40 stratified days as proposed, or the four whole test months.
6. **`DataSpec.new_study`:** the study generates no training data; an exception as for the headroom
   and the static studies (`docs/12`, section L) has to be approved.
7. **Budget (wall-clock):** proposal 30 h for stage 1, as in Befund 115; cuts are decided after the
   dry run and before the main run.
8. **Bed-mass set point** as a further degree of freedom (holdup as a buffer): proposal no.
9. **Margins:** family-wide per controller as proposed (expensive for C3 and C4), or fixed
   multiples of the Befund 105 margins as a cheaper, weaker alternative.

## 11. Questions for the application partner

* Turn-down: between which throughputs can a line run without a product change, and how fast may
  the feed be changed?
* Over which window is the production fixed (shift, day, campaign)?
* Does a change of throughput shift the particle size or other release properties?
* Which carrier heats the process air of the line considered, and is an electric heater planned?
* Does the moisture limit bind in normal operation (`docs/12` I.9)?

## 12. Implementation and run plan

* Package `src/twinlink_ml/load_following/`, settings `conf/load_following/load_following.yaml`
  (frozen dataclasses, fingerprint), tests `tests/test_load_following.py`. The headroom package is
  reused (`TruthPlant`, `SteadyStateOptimizer`, `DynamicOptimizer`, `symbolic`, `margin`); the third
  control and the priced objective are added without changing the behaviour of the headroom study:
  its config fingerprint and the reproduction of `dynamic_headroom_v2` stay as they are.
* Tests fixed beforehand: with the spray rate pinned to the nominal value, C0 reproduces R1 and C3
  the R3 commands of Befund 105 on a headroom scenario; at a constant price the daily target is met
  and C2 equals C0 where the map is convex; the price guard of Befund 115.
* Scripts `scripts/run_load_following.py` (stages, `--resume`, one unit = controller x day, saved
  per unit, a failed unit is logged and does not discard the others) and
  `scripts/analyze_load_following.py`; both committed before the run, clean tree,
  `require_code_location(ROOT)`, outputs as `.json.gz`.
* Dry run over **all** units with days shortened to a few hours, reporting time and peak memory per
  worker (the first attempt of Befund 105 failed on memory with SX solvers: MX from the start);
  workers at most the number of physical cores, workers x peak below 80 % of RAM.
* Main run detached from the session, in a worktree with `tools/run_here.sh`.
* Lab-book entry, `docs/12` sections I and J and the README paragraph at merge, by the managing
  session.
