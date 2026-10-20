# 24 -- Dynamic headroom in continuous operation (headroom test before further RL work)

Status: design and code committed before the first run; results in section 8 (filled after the
run, numbers only from the artifacts named there). Branch `dynamic-headroom`; the finding number
is assigned on merge.

## 1. Question and decision rule

All RL tests so far were memoryless problems (bandit, heat network, single storage, heat pump);
there, RTO is superior. The plant runs continuously. Question: **when disturbances are ridden
out, is there an energy gain from dynamic operation over quasi-stationary RTO?** Only if so is an
RL-against-MPC study justified.

The reading is fixed in the docstring of `scripts/analyze_dynamic_headroom.py`, committed before
the first run. On the family "observed", pooled median over scenarios of the gap
`g13 = (E_R1 - E_R3) / E_R1`:

| class | median g13 | consequence |
|---|---|---|
| L0 | < 2 % | dynamic control does not pay in continuous operation; close the RL line for the granulator |
| L1 | 2-5 % | MPC suffices; RL only with a robustness argument |
| L2 | > 5 % | RL-against-MPC study justified |

Independently: robustness cost > 5 % (median, R1 or R3) -> candidate "robust control" open.
Median and range over scenarios, not only the mean.

## 2. Line, truth model, placeholders

Truth = the grey-box model (`GreyBoxFluidBedGranulator`, 4 states, RK4, sample time 5 s, five
internal substeps), unchanged; **no surrogate** in this study. Process config: the lab line of the
bandit studies (`conf/process/greybox_gab_placeholder.yaml`, GAB isotherm placeholder), with the
study settings of `conf/headroom/dynamic_headroom.yaml`:

* quality limit of the product moisture **0.15 kg/kg (placeholder)** as a **path condition at
  every truth sample** (continuous product stream: product moisture = bed moisture), violation
  tolerance 1e-4 kg/kg (`operating_point.TOLERANCE_MOISTURE_KG_PER_KG`);
* minimum gas velocity **0.80 m/s (placeholder)**, the unchanged `ActionBound` minimum;
* nozzle pressure fixed at the Befund 52 fixed point (as in the bandit env); bed temperature
  <= 80 degC (Befund 52 placeholder) as a hard constraint of the optimisers;
* rate limits of the commands as in the excitation design (`ActionBound.max_rate_per_s`:
  0.5 K/s inlet temperature, 0.05 m/s per s gas velocity), applied sample by sample in the truth
  (`apply_rate_limit` rule);
* throughput = spray rate (nominal 1.5 kg/h), a disturbance for the controllers.

**Solid fraction (decision of the project lead, 2026-10-20).** The request named the band
0.60-0.70 (plant-like). On this lab line the 0.15 limit does **not bind** at such a solid
fraction: the steady product moisture at the inlet-temperature minimum (60 degC, u = 0.80 m/s) is
0.07-0.11 kg/kg (probe before the study, steady-state solver at 0.60/0.65/0.70 and three weather
points; only 0.60 at 35 degC / 85 % RH exceeded 0.15). Every controller would sit at the box
corner and the gap would be zero by construction. The study therefore keeps the lab line's own
band around the nominal 0.30 and carries over only the **relative** amplitude of the plant band
(0.60-0.70 = +-7.7 %). **Transferability to the plant depends on whether the moisture limit binds
there -- an open question to the application partner, not a result of this study.**

## 3. Scenarios

Deterministic from the configured seed (`twinlink_ml.headroom.scenarios`), 24 h each (product
change: 6 h), at least 20 per type. Placeholders explicit in `conf/headroom/dynamic_headroom.yaml`;
values marked "derived from plant data, placeholder" are relative amplitudes only (no raw values,
channel names or apparatus numbers in the repository). The plant runs 24/7 with small variations
(air flow about +-0.5 to +-1.4 % within a day), hence two families reported separately:

**observed** (the reading applies here)

| type | n | disturbance |
|---|---|---|
| `ambient` | 25 | day profile around each point of the 5 x 5 bandit weather grid (`DiurnalDisturbanceDesign`, its defaults) |
| `throughput` | 20 | piecewise-constant levels within +-1.5 %, hold 1-4 h (derived from plant data, placeholder) |
| `solid_fraction` | 20 | Ornstein-Uhlenbeck drift in 0.30 +- 7.7 %, std half the half band, 3 h (placeholder) |
| `inlet_temperature_slow` | 20 | additive OU offset of the realised inlet temperature, sigma 0.35 K, 3 h (derived from plant data, placeholder) |
| `inlet_temperature_fast` | 20 | the same, sigma 1.5 K, 4 min -- below the 5-min command grid (derived from plant data, placeholder) |

**stress** (upper estimate only, no reading): throughput steps +-10 / +-20 %, solid fraction steps
across 0.20-0.40, inlet-temperature offsets with doubled sigma (20 each).

**product change**: one changeover after 2 h of 6 h, moisture target 0.15 <-> 0.12 or solid
fraction 0.30 <-> 0.25 (second product = placeholder), 5 per direction; reported **per
changeover** (energy, duration to specification), not extrapolated -- the frequency is unknown.

Outside `ambient` the ambient air is constant at a grid point (seeded permutation). Steps fall on
the 5-min control grid; the throughput is rate limited with the spray-rate limit. The optional
time-varying CO2 signal was not run (section 9).

## 4. Controllers and numerics

All controllers command inlet temperature and gas velocity, piecewise constant over 5 min, with a
moisture margin `m` (they plan for `X <= limit - m`):

* **R0** fixed operating point: steady optimum at the nominal disturbance, held everywhere.
* **R1** quasi-stationary RTO: at each interval start the steady optimum at the current
  disturbance (perfect knowledge, incl. the inlet-temperature offset) and the current limit.
* **R2** dynamic optimum with perfect preview: one multiple-shooting problem over the whole
  scenario with the true disturbance course; applied open loop.
* **R3** MPC without disturbance preview: 2 h horizon (24 intervals), receding, current
  disturbance and limit held constant over the horizon, true state measured.

Optimisers (`twinlink_ml.headroom.controllers`): CasADi 3.8.1 / IPOPT (optional dependency, extra
`optimal-control`). **Same model as the truth**: a CasADi backend of the existing `ArrayBackend`
protocol evaluates the unchanged thermo / sorption / energy functions symbolically; the
composition of `_rhs` is pinned against `evaluate_derivative` (test, relative difference
< 1e-10). The optimisers use the three states (m_s, m_w, T_bed); the particle count enters neither
these derivatives nor the power.

* RK4 with 30 s steps; the path condition at every step end (10 nodes per interval). Discretisation
  error against the truth under random 5-min command steps of up to 40 K: max |dX| = 2.3e-4 kg/kg
  (20 s: 1.7e-4; 60 s: 7e-4; probe before the study).
* **Ramp model.** Without the ramp of the rate limit the predicted moisture of a preview optimum
  was off by up to 9e-3 kg/kg (it swings the inlet temperature by tens of K; a 50 K change takes
  100 s of the 300 s interval). Each RK4 step therefore uses the mean of the ramped command over the
  step (`symbolic.ramp_mean`, C1 in the command change); the remaining node mismatch was 1.1e-4.
  The clipped ramp itself (non-smooth) made IPOPT erratic. The interval-level form
  `|u_k - u_(k-1)| <= rate * 300 s` is kept as a constraint.
* Path condition as a soft constraint with L1 slack (exact penalty, weight 10 kWh per node and
  kg/kg): R3 meets limits it cannot hold any more (e.g. right after a product change). The bed
  temperature limit is hard.
* **R2 safeguard.** The problem is non-convex (ramp model); on an undisturbed scenario a default
  IPOPT start ended in a chattering local optimum worse than the R1 schedule in the model itself.
  R2 starts near its guess (R1 schedule; small barrier parameter and bound push) and never takes a
  plan its own model rates worse than the R1 schedule (counted as "fallback" in the record).
* Finite-horizon end effect: in the last intervals a preview optimum lets the moisture rise
  towards the bound (water left in the bed at the end is free); about 0.006 kWh, 0.02 % of a 24 h
  scenario (test `test_r2_equals_r1_without_disturbance`).
* Common initial state: steady optimum at the first disturbance sample with bound
  `limit - 0.01` (placeholder start margin), so no controller starts violated.
* R3: warm start from the previous solution shifted by one interval and re-simulated under the
  new disturbance; about 1 s per solve (2 h horizon). A failed solve holds the previous command
  and is counted.

## 5. Margins

Every controller gets the smallest **family-wide** margin with which the path condition holds in
all scenarios of the family (truth, every sample, tolerance 1e-4 kg/kg). Bisection
(`twinlink_ml.headroom.margin`) with bracket [0, 0.05] (doubled up to 0.12 if needed) and
tolerance 2.5e-4 kg/kg; the returned margin was evaluated as feasible. For R2 and R3 the bisection
runs on a critical subset (three scenarios per type with the largest R1 violation at margin 0)
and is verified on all scenarios; failing scenarios join the subset and the search continues.
Their bracket starts at [0, 2 x R1 margin] (fewer of the expensive evaluations; doubled whenever
the upper end is infeasible, so the result is the same to within the tolerance). A smoke run with
shortened scenarios (4 h, two per type, 2 x 2 weather grid, 3 isotherm draws; scratch, not an
artifact) checked every stage end to end before the main run.

## 6. Metrics

Per scenario: energy (model total power integrated, kWh), CO2 (gas-fired heater, eta 0.80
placeholder, 0.366 / 0.201 kg/kWh), maximum limit violation, required margin. Gaps `g13`, `g12`,
`g10` and the decomposition into margin share `(E_R1(m1) - E_R1(m3)) / E_R1(m1)` and trajectory
share `(E_R1(m3) - E_R3(m3)) / E_R1(m1)` (R1 with R3's margin as an intermediate point, even if it
violates the path condition -- only for the decomposition); analogously for R2.

## 7. Robustness variant

The GAB parameters of the **truth** are spread (20 LHS draws; placeholder band: monolayer
x0.8-1.2, C x0.5-2 log-uniform, K 0.70-0.80, both enthalpies x0.5-1.5; draws violating the
isotherm check against the critical moisture are replaced and counted); the controllers keep the
nominal model. The true plant starts from 6 h at the initial command (warm-up). Subset: scenarios
0 and 12 of every observed type (10 scenarios). Reported: the margins of R1 and R3 with which the
path condition holds in 95 % of the draws (19 of 20; R1 by a per-draw bisection, R3 by a k-of-n
bisection on the draws ranked most demanding by R1, verified on all draws), and their energy cost
on the nominal truth against the nominal margins.

## 8. Results

Pending (filled from `artifacts/runs/dynamic_headroom_v1.json.gz` and
`artifacts/runs/dynamic_headroom_v1_analysis.json`).

## 9. Open placeholders and limitations

Placeholders: quality limit 0.15; u_min 0.80; start margin 0.01; second product (0.12 / 0.25);
all scenario amplitudes (observed ones derived from plant data, relative only); GAB band of the
robustness variant; gas-heater efficiency 0.80; soft-constraint weight; the whole process config
(placeholder status).

Limitations: one lab line with a 4-state model (no particle-size dynamics in the moisture or
energy balance); controllers see the true state (no estimator, no measurement noise); family-wide
margins (one margin per family, set by its most demanding scenario); the time-varying CO2 signal
(optional last variant of the request) was not run; whether the moisture limit binds on the plant
is open (section 2).
