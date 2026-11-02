# Process setup: fluidized-bed spray granulation

> **This file is the most important one in the project knowledge.** Without concrete value ranges and
> time constants, any modelling remains generic.
>
> Status: 13.08.2026 · version 3. **The previous addendum is incorporated into this file
> and thus obsolete** — a file with three appendices is not read but
> skimmed. The implementation status of the ML part is in `04_stand_ap1.md`.
>
> New in this version: the values of the SPE lab plant from Kräuter, Zhou & Heinrich
> (ABS23, TwinGuide) — the first source in the project that contains no placeholders.

## Status of the numerical values — legend

| Mark | Meaning |
|---|---|
| `MESS` | measured or confirmed by a partner |
| `LAB` | **new**: measured on the SPE lab plant in the previous project TwinGuide, but on the **dryer** and with MCC pellets, not on the granulation and not at application partner |
| `PLH` | placeholder, usable for the architecture comparison, **not** for absolute statements |
| `MODELL` | computed from the grey-box model — valid under the `PLH` assumptions |
| `TODO` | missing completely |

`LAB` ranks between `MESS` and `PLH` and is explicitly **not** the same as
`MESS`: the values come from a drying process without solids build-up, with
Cellets® 200 as inert bed material, in batch operation and at lab scale. For
orders of magnitude, value ranges and time grids they are robust; for the granulation at
application partner they are an analogy.

The authoritative, versioned version of all values is
`conf/process/greybox_placeholder.yaml`. This file is the technical rationale for it.

---

## Priority list of the open numbers

Order by effect on the experiment planning. Compared with version 2, three items are
done and two are new.

| # | Missing number | Who | Why it comes first |
|---|---|---|---|
| 1 | **Maximum rates of change of the actions** | Anwendungspartner, SPE | They fix the achievable excitation bandwidth. With the `PLH` values the rate limiting is permanently active: the signal design is replaced by the actuator dynamics. |
| 2 | **Inline measurability of the product moisture at application partner** | Anwendungspartner | **New at rank 2.** The architecture comparison centres on the moisture; whether it is measured decides whether it is a prediction task or a soft-sensor task. Both are implemented (see below), but they have different baselines, different primary metrics and different results. In the SPE lab the measurement exists (`LAB`, microwave resonance). |
| 3 | **Sorption isotherm of the actual material system** | SPE | **Upgraded.** In the previous project the wrong isotherm (Cellets® 500 instead of 200) was the *main cause* of the Dyssol prediction of the moisture being unusable. This is not fine-tuning but the difference between R² = 0.16 and usable. |
| 4 | **Time constant of the air heater** | Anwendungspartner, SPE | With it, the actuator dynamics would sit in the process model instead of in the excitation. Without it, `dE` is 97 % explained by a static mapping. |
| 5 | **Definition of `dE`** | Consortium | Instantaneous power [kW] or specific consumption [kWh/kg]? The monotonicity constraints hold universally only for the instantaneous power. |
| 6 | Noise level of the plant sensors | Anwendungspartner | **Downgraded.** The ranking of the models is stable over two orders of magnitude of the noise level (noise study from step 5, see `04_stand_ap1.md`, section "Das Sensormodell ist zu gutmütig"). It remains decisive only for `d_50`. |
| 7 | Operating mode and discharge intervals | Anwendungspartner, SPE | Continuous with level control or cyclic discharge? In the SPE lab: **batch operation** (`LAB`). Changes one time constant of the system. |
| 8 | Cyclone and recirculation behaviour | Anwendungspartner | In the standard excitation design `u_gas` lies above `u_t` of the mean grain in 13 % of the sampling steps. |
| 9 | Lower limit of the product moisture | Anwendungspartner | New (Befund 77, `09_entwurf_modellbasierte_optimierung.md`). The coupling gain of the two-line optimum requires one line to dry its product to 0.043-0.080 kg/kg against the upper limit 0.144 kg/kg. If a lower limit exists, the mechanism may fall away. |
| 10 | Condensing heat recovery from the humid exhaust | Anwendungspartner | New (Befund 77). The gain requires a condensing recovery; the apparatus (condensate drain, exchanger material, fouling, pressure loss) is not modelled. Is it feasible in the plant, and at what cost? |
| 11 | Achievable approach temperature (`dT_min`) of the heat exchangers at the site, including an intermediate circuit | Anwendungspartner | New (Befund 81). The coupling gain of the two lines depends on it: at `dT_min` 20 K none (0 of 6 warm, humid scenarios, the joint optimum equals the line optimum), at 10 K a small, scenario-dependent one (1 of 6 W scenarios above 1.0 kg CO2e/h; up to 3.6 kg CO2e/h at mild, humid air). 20 K is a placeholder of the network model. |
| 12 | Measured values of the real heat-recovery circuit: inlet and outlet temperatures and flow of the heat transfer medium, storage temperatures (and which medium, at what pressure) | Anwendungspartner | New (Befund NN, `11_partner_model_v4.md`, section 10). The only basis for a real validation of the recovery: the partner model and an independent counter-current calculation differ by a factor of about 1.5 to 2.2 in the recovered heat at the sheet's flows, and without measurements neither can be called right. The medium leaves the recovery exchanger above 100 degC in the partner model. |

**Done compared with version 2:** time step of the plant data (2 s, `LAB`),
computing time per Dyssol run (see below, `LAB`), question about the controlled quantity for
`u_gas` (volume flow, `LAB`).

---

## Variable structure

### Actions (manipulated by the controller)

| Symbol | Quantity | Unit | Min | Max | Operating point | Max. rate of change | Status |
|---|---|---|---|---|---|---|---|
| `T_in` | Inlet air temperature | °C | 60 | 120 | 90 | 0.5 K/s | `PLH`; lab 50…100 °C (`LAB`) / rate `TODO` |
| `u_gas` | Gas velocity | m/s | 0.8 | 2.5 | 1.6 | 0.05 m/s² | `PLH`; lab 1.26…1.88 m/s (`LAB`) / rate `TODO` |
| `m_spray` | Spray rate | kg/h | 0.4 | 3.0 | 1.5 | 0.02 (kg/h)/s | `PLH`; lab 0.6…3.9 kg/h (`LAB`) / rate `TODO` |
| `p_nozzle` | Nozzle pressure | bar | 1.0 | 4.0 | 2.5 | 0.05 bar/s | `PLH`; lab 1.0…1.8 bar (`LAB`) / rate `TODO` |

**The placeholder ranges are confirmed by the lab values** — with one exception: the
nozzle pressure is run in the lab only between 1.0 and 1.8 bar, not up to 4.0. This is
more than a range correction: since `p_nozzle` in the grey-box model has a pure cost effect without
process benefit, a range that is too wide artificially enlarges the leeway that an
RL agent believes it can exploit there.

**On `u_gas`: the plant controls the volume flow, not the velocity** (`LAB`).
This settles the open item from version 2, and the consequence is unpleasant:
at constant volume flow, the gas density changes with `T_in` and with it the
air **mass** flow — i.e. the coupling to `dE` and to the drying capacity. In the model,
`u_gas` is the superficial velocity at bed conditions; for the comparison with plant data
the conversion via the equation of state is necessary, and it is **not** a constant.
Conversion at lab scale (0.0177 m² assumed): 80 m³/h → 1.26 m/s, 120 m³/h →
1.88 m/s. The nominal value 1.6 m/s of the placeholder lies right in between.

**Rate limiting, unchanged:** from `t_switch = Amplitude / max. Änderungsrate` follows the
minimum achievable switching time per channel. Relative to the fastest process mode
(`tau_fast`, see section time constants), only `u_gas` can excite it at all. **The
effectively fastest observable time scale is set by the actuators, not by the
process.**

**Hard operating limits (safety constraints for RL)**

Fluidization window `u_mf < u_gas < u_t`, Wen-Yu and Haider-Levenspiel, respectively,
`rho_p = 1500 kg/m³` (`PLH`), sphericity 0.9, 60 °C — status `MODELL`:

| `d_50` | `u_mf` | `u_t` |
|---|---|---|
| 150 µm (nuclei) | 0.010 m/s | 0.62 m/s |
| 400 µm (start bed) | 0.070 m/s | 2.10 m/s |
| 750 µm (target grain) | 0.222 m/s | 3.57 m/s |
| 1000 µm | 0.350 m/s | 4.36 m/s |

**The window is state-dependent.** Between 400 and 750 µm, `u_mf` grows by a factor of 3;
a fixed lower limit leads to the collapse of fluidization at the end of the batch.
For RL: the action bounds are a function of the state.

Further limits: exhaust air humidity ≤ saturation humidity (structurally satisfied),
particle moisture ≥ equilibrium moisture, max. bed moisture before agglomeration risk
`TODO`. In the lab the over-wetting limit was **deliberately exceeded** (batch 4:
low bed temperature, low air flow, high spray rate → 22 % moisture, `LAB`) — so there
is a measured reference point for the collapse, if only one.

### Disturbances (exogenous, measurable but not controllable)

| Symbol | Quantity | Unit | Range | Nominal | Status |
|---|---|---|---|---|---|
| `T_amb` | Ambient temperature | °C | 5 … 35 | 20 | `PLH`; lab 20.5 … 24.5 (`LAB`) |
| `phi_amb` | Ambient humidity (rel.) | – | 0.25 … 0.85 | 0.50 | `PLH`; lab 0.08 … 0.70 (`LAB`) |
| `w_solid` | Suspension solid fraction | – | 0.20 … 0.40 | 0.30 | `PLH` (lab: pure water, so 0) |

**The lab range of the ambient humidity is wider than the placeholder** (down to 0.08).
This is relevant because `phi_amb` enters the drying capacity directly via the dryness of the
inlet air — the placeholder underestimates the scatter of the most important
disturbance.

**The day/night indicator stays out.** It is not a physical input quantity; it
acts exclusively via `T_amb` and `phi_amb`. As a model input it would be a channel without
causal effect that a network misuses as a shortcut.

**State variables, not disturbances:** `d_50`, `T_part`, `X_part`. A disturbance is given to the
model as an input; a state variable has to be propagated by it. Whoever feeds in state variables
builds a model that needs full state measurement at the plant.

| State variable | Placeholder range | Lab range (`LAB`) | Remark |
|---|---|---|---|
| `d_50` | 250 … 900 µm (initial scatter) | Cellets® 200, constant | in the lab without growth |
| `T_part` | 35 … 80 °C | 33.6 … 92.4 °C | placeholder too narrow |
| `X_part` | 0.003 … 0.020 kg/kg | **0.033 … 0.29 kg/kg** | **placeholder too dry by a factor of 10** |

### The moisture range is the most important correction

In the lab the particle moisture lies between 3.2 and 22.5 % relative to the **wet**
mass, i.e. 0.033 to 0.29 kg/kg dry (`LAB`). The placeholder in the model ranges from
0.003 to 0.020 kg/kg — **an order of magnitude too dry.** Cellets® are microcrystalline
cellulose and strongly sorptive; granules from the application partner's product range may be drier.
The number is therefore not to be adopted unchecked, but the direction is unambiguous.

**This is not a cosmetic correction; it shifts the time constants.** Recomputed
at the same operating point, with `X_cr` and `X_eq` raised proportionally (status `MODELL`):

| Case | `X_cr` / `X_eq` / `X_0` | τ fast | τ second mode | Stiffness |
|---|---|---|---|---|
| Placeholder | 0.020 / 0.003 / 0.005 | **13.7 s** | **85 s** | 1171 |
| Lab level | 0.20 / 0.03 / 0.05 | 36.8 s | 371 s | 435 |
| Lab level, high | 0.29 / 0.033 / 0.08 | 33.3 s | 600 s | 480 |

The fastest mode becomes **slower** by a factor of 2.5 and the second by a factor of 4 to 7. The
reason is physical and not numerical: at 0.5 % moisture the bed operates deep in the
second drying period (`nu` ≈ 0.06, the drying capacity of the air exceeds the
water input by more than an order of magnitude), at 5 % no longer. **The time constant of 9.5 s
reported in version 2 is thus an artifact of a placeholder that is too
dry**, and with it the statement "Δt = 60 s fails because of the moisture mode".

Consequences to be drawn before the next Δt decision:

- At a realistic moisture level, Δt = 60 s is **no longer** disqualified: 10 steps
  per τ ≈ 35 s are acceptable. This halves the sequence length compared with Δt = 25 s.
- The stiffness falls from 1171 to ~450. The gap between the fastest and the slowest
  mode remains three decades, but the spread within the fast modes shrinks.
- **The Δt pre-study is to be repeated with a corrected sorption isotherm.** It is
  cheap (three runs per Δt), and its result determines the sequence length of all
  architectures.

### Target(s)

| Symbol | Quantity | Unit | Status | Note |
|---|---|---|---|---|
| `dE` | Energy consumption | kW **or** kWh/kg | Definition `TODO` | primary target of the proposal |
| `X_prod` | Product moisture | kg/kg | Target of the architecture comparison | in the model = bed moisture |
| `PSD` | Particle size distribution | – | from AP 2; target 10 % | **cannot be addressed with the grey-box model** |
| `d_50` | Mean grain size | µm | **removed from the comparison** | early measurement gave ~84 % noise share of the scatter (**not reproducible**: the original number was in a since renumbered version of `04_stand_ap1.md`, numbering 1–15 no longer preserved there, see Befund 26 part 4, and can no longer be assigned unambiguously to one place). Independently confirmed with numbers reproduced in this session: FIR R²_holdout negative throughout (−0.042 at k=1 to −0.123 at k=43, `artifacts/runs/probe_e2_e3_e4_lag_depth.json`), LSTM seed range at h=1 ≈0 (collapse to a mean predictor, `artifacts/runs/probe_f3_f4_arch_holdout_inference.json`) -- see the `experiments.py::DataSpec.target_names` docstring |

**Prediction task:** target at `t+1` from actions, disturbances and targets of steps
`0..t` — **if** the target is measured. Otherwise from actions and disturbances
alone; that is the soft-sensor task, see below.

**Static variance share** — coefficient of determination of a quadratic regression on the
*current* actions and disturbances without history. The figure is **not a
process property** but a property of the process *and* the excitation design, and
is always to be reported with the design (status `MODELL`):

| Quantity | R² with scatter (standard design, placeholder config) | R² GAB, reproduced (fp `82f65f576a2e5253`, seeds 20260812/13/14, n_runs=32, median) |
|---|---|---|
| `dE` (instantaneous power) | 0.974 | 0.972 |
| Bed temperature | 0.843 | 0.768 |
| Product moisture | 0.274 | 0.508 |
| `d_50` | **0.100** | 0.144 |

Former column "R² without scatter of the initial states" removed -- cannot be produced with `report_baselines.py`
in its previous version (no switch for
`randomise_initial_conditions`, see the script docstring there) and the provenance of the
original numbers cannot be reconstructed. Historical values, not marked as wrong
but as not reproducible: `dE` 0.987, bed temperature 0.87, product moisture 0.43,
`d_50` 0.72.

**The dynamic quantities are the quality quantities, not the energy consumption.** The
heating power is algebraic in `(T_in, u_gas, T_amb)`; the process state acts only via
gas density and bed pressure drop. Consequences: the architecture comparison needs at least one
quality quantity, and the mandatory baselines need a **static regression** — for
`dE`, persistence is the wrong reference.

---

## Observability: two operating modes, not one

New in this version, because it changes the task and not just its implementation.

| | Measured inline | Soft sensor |
|---|---|---|
| History contains past target values | yes | no |
| Level of the run | is in the history | part of the task |
| Parameter-free baseline | persistence | mean prediction |
| Primary metric | NRMSE run-level median | **NRMSE pooled** |

**Why the primary metric changes.** The finding from step 3 was: the pooled NRMSE
is too small because the pooled scatter contains the level differences between the runs,
"and no model has to predict those, they are in the window history". The
second half-sentence is a **precondition**: it holds only with inline measurement. Without it,
the level is part of the task, and the run-level normalization additionally becomes unstable — for
a run with small intrinsic scatter the denominator goes to zero. Measured: a
parameter-free mean prediction reaches run-level medians above 3 with
seed ranges above 19, whereas pooled it is a clean 1.01.

**State of measurability.** In the SPE lab the particle moisture is measured inline via
**microwave resonance** (TEWS MW4200, sensor SPW-043-020-C), calibrated against
thermogravimetric reference samples (`LAB`). For the application partner's plant it is open — and
that is priority 2 of the overall list.

| Quantity | In Dyssol | SPE lab | Application partner's plant |
|---|---|---|---|
| Exhaust air humidity | yes | **yes** (`LAB`) | `TODO` |
| Exhaust air temperature | yes | **yes** (`LAB`) | `TODO` |
| Inlet air temperature after heater | yes | **yes** (`LAB`) | `TODO` |
| Particle temperature in the bed | yes | **yes** (`LAB`) | `TODO` |
| Particle moisture (inline) | yes | **yes, microwave resonance** (`LAB`) | `TODO` **urgent** |
| Pressure drop filter / expansion chamber | yes | **yes** (`LAB`) | `TODO` |
| Volume flows fluidization and atomization | yes | **yes** (`LAB`) | `TODO` |
| PSD (inline) | yes | no (offline CAMSIZER) | `TODO` |
| Power consumption per unit | yes | `TODO` | `TODO` |

Eleven inline channels in the lab (`LAB`) — that is more than the grey-box model currently offers as
input, and considerably more than the four actions plus three disturbances of the current
configuration. For AP 2 this means: the feature space at the real plant is richer, not
poorer.

---

## Temporal resolution

| Parameter | Value | Status |
|---|---|---|
| Time step of plant data (SPE lab) | **2 s** | `LAB` |
| Time step of plant data (application partner) | `TODO` | |
| Time step of Dyssol | `TODO` | to be asked from TUHH-A |
| Typical batch duration (SPE lab) | **1.5 … 3.9 h**, on average 1.8 h | `LAB` |
| Simulation horizon in the ML model | 8 h per run | `PLH` — **too long, see below** |
| Computing time Dyssol (dryer model, 10 batches ≈ 18 h process time) | **17,500 s**, i.e. ~0.27 s computing time per second of process time | `LAB` |
| Sampling time ML pipeline | 25 s | `MODELL`, to be checked |

**The assumed run duration is too long.** 8 h per run against 1.8 h in the lab is a factor of 4.4.
This is not only a question of cost: the slow mode (residence time of the solids,
~16,000 s ≈ 4.4 h) **does not fit into a lab batch at all**. In AP 2, grain growth over
a batch is thus a truncated transient and not a steady state — and
the scatter of the initial conditions between the runs, with which the slow mode is covered in the
excitation design, is not an optional extra there but the only source of this
information.

**The Dyssol computing time limits the end point of the learning curves.** From 0.27 s computing time per
second of process time follows, for 128 runs of 2 h process time each, about **69 h on one core** —
and that for a dryer model without solids build-up and without PSD. The granulation model
will be slower. For comparison: the grey-box model generates 128 runs in about 40 s,
i.e. roughly a factor of 6000 faster. This confirms the order of the work plan — the
methodology belongs on the grey-box model, and Dyssol runs are a scarce resource whose
sampling design is to be fixed **before** production.

**There is no "dominant time constant"** (status `MODELL`, placeholder moisture level —
see the correction above):

| Mode | τ | Physical cause |
|---|---|---|
| Particle moisture + bed temperature (coupled pair, see note) | 13.7 s (placeholder) / ~35 s (lab level) / 30.23 s & 359.93 s (GAB, eigenvalues) | Drying capacity of the air and heat capacity of the bed -- no longer cleanly separable under GAB |
| Mass control | 600 s | Level control (configuration value, not a process) |
| Particle number / grain growth | 16,000 s | Residence time of the solids |

**Note on GAB.** Under the GAB sorption isotherm (Befund 34) there is no
well-defined single thermal time constant of the bed: the closed form
τ_T = 84.75 s (`τ_T = C/(ṁ_L·c_pL + UA)`) lies between the two eigenvalues
τ₁ = 30.23 s and τ₂ = 359.93 s and matches neither of them -- moisture and
temperature dynamics are coupled under GAB. The split into "particle moisture" /
"bed temperature" as separate rows is therefore valid only for the placeholder/lab values
(constant equilibrium moisture), not for GAB
(`artifacts/runs/probe_a2_a3_static_variance_time_constants.json`).

**The sequence length is the decision which mode is modelled, not a
hyperparameter question.** The slow mode cannot be excited within a run by the
actions (monotonic growth without feedback); it is covered by the scatter of the
initial conditions.

**On the choice of Δt** (status `MODELL`, constant physical horizon of 600 s, window
800 s, linear ARX, run-level median):

| Δt | Steps per window | Persistence h=1 (`dE`) | Rollout 600 s `dE` | Rollout 600 s `X_prod` |
|---|---|---|---|---|
| 5 s | 160 | 0.069 | 0.177 | 0.727 |
| **25 s** | **32** | 0.259 | **0.192** | **0.688** |
| 60 s | 13 | 0.362 | 0.162 | 1.404 |

The **one-step** error rises monotonically with Δt; the **rollout** error over the same
physical horizon is almost invariant. **Δt is to be chosen by computational cost and sequence length,
not by accuracy.** The exclusion of Δt = 60 s was based on the
moisture mode of 9.5 s and is to be re-checked with the corrected moisture level.

Methodologically, for all future Δt comparisons: keep window and horizon length constant in **seconds**
and convert the step counts.

---

## Plant geometry

| Parameter | Value | Status |
|---|---|---|
| Plant in the SPE lab | ProCell® 5 LabSystem, process chamber GF3 (Glatt) | `LAB` |
| Cross-sectional area of the fluidized-bed chamber | 0.0177 m² (150 mm) assumed | `PLH` |
| Bed mass (solids, dry) | 2.0 kg | `PLH`; lab 1 or 2 kg (`LAB`) |
| Heat loss `UA` | 2.0 W/K | `PLH` |
| Operating mode | continuous with level control | `PLH`; lab **batch** (`LAB`) |
| Nozzle type | two-fluid nozzle, **top spray** | `LAB` (characteristic still `TODO`) |
| Bed material in the lab | Cellets® 200, MCC, inert and water-insoluble | `LAB` |
| Scale of the application partner's production plant | `TODO` | |

The assumed bed mass matches the lab value. The **operating mode does not match**: the model
runs a continuous level control with τ = 600 s, the lab runs batches. For
AP 2 the level control is to be switched off or parametrized as batch operation —
otherwise the model contains a time constant that does not exist in the validation experiment.

---

## Physical relations for the grey-box model

**State vector** — four states, ideally mixed bed, gas phase quasi-stationary:

    x = [m_s, m_w, N, T_bed]

`d_50` is **not** a state but follows algebraically from
`d_50 = (6 m_s / (pi rho_p N))^(1/3)`. Thus mass and number balances remain linear and are
conserved exactly in the integrator.

**Gas phase quasi-stationary:** residence time ≈ 0.2 s, three orders of magnitude below the
thermal time constant. The exhaust air humidity is determined algebraically — this removes the
stiffest mode and allows explicit RK4.

**Enthalpy balance** for the bed temperature, all outgoing streams referenced to `T_bed`:

    (m_s c_ps + m_w c_pw) dT_bed/dt
      = ṁ_L (c_pL + Y_in c_pD)(T_in - T_bed)
        + ṁ_spray c_psusp (T_susp - T_bed)
        - ṁ_evap Δh_v(T_bed)
        - UA (T_bed - T_amb)

**No additional lag element for the bed.** The thermal time constant already follows
from the balance as `tau_T = C / (ṁ_L c_pL + UA)`. A filter on top counts it
twice. The PT1 belongs in the **sensor model** (`PLH` 10 s).

**On the sensor model, new from the lab:** the raw data of the microwave resonance measurement
contained 17 to 34 % outliers per batch and were smoothed with a third-order Savitzky-Golay filter
over 100 sampling steps (`LAB`) — at Δt = 2 s, i.e. over 200 s.
Two conclusions, both unpleasant:

1. **The placeholder sensor model is too benign.** 0.5 % relative noise and a PT1
   with 10 s do not describe a signal from which a quarter of the points have to be removed as
   outliers. A realistic model needs an **outlier process**
   (heavy-tailed, not Gaussian) alongside the additive noise. This is a
   modelling task, not a parameter value.
2. **A centred smoothing window over 200 s is not causal.** It uses measured values
   from the future. For an offline analysis this is legitimate, for an online soft sensor
   or a controller it is not — and in a data set with randomly mixed observations
   it additionally creates information flow between training and test set. If smoothing
   is necessary, then **causally** (one-sided), and the filter length belongs in the protocol as an
   experimental factor, not in the preprocessing.

**Drying kinetics** via the normalized drying curve:

    ṁ_evap = nu_max * eta * ṁ_L * (Y_s(T_bed) - Y_in),    eta = clip((X_p - X_eq)/(X_cr - X_eq), 0, 1)

`nu_max` ≈ 0.6 (`PLH`, literature 0.4 … 0.8). **The condensation branch is not optional:**
if the inlet air is more humid than saturation at bed temperature, the mass transfer
reverses. Without it, the model silently delivers supersaturated exhaust air for a cold bed — exactly
the operating point into which an untrained RL agent runs first.

**Layering growth** — in the chosen state space a consequence, not a postulate:

    dd_50/dt = ṁ_spray w_solid d_50 / (3 m_s) = 2 V̇_fest / A_Partikel

**Steady-state diameter** `d_50,ss = d_nuc * b_nuc^(-1/3)`, with `d_nuc` = 150 µm and
`b_nuc` = 0.008 (`PLH`) → 750 µm. The nucleation share is thus **not a free
fit parameter**.

**Hard limits (implementable as output projection)**
- Exhaust air humidity ≤ saturation humidity — satisfied **structurally**, not by projection.
- Particle moisture ≥ equilibrium moisture — structurally via `eta → 0`, **but** with hot
  inlet air violable by the integrator; projection needed as a safety net.
- `dE` ≥ 0, physical value ranges of all quantities.

**Monotonicity (architecturally enforceable)**: ∂`dE`/∂`T_in` > 0, ∂`dE`/∂`u_gas` > 0. In the
grey-box model structurally satisfied. **Caveat:** holds universally only for the
*instantaneous power*; for the specific consumption it depends on the
product mass flow not depending on `T_in` — which no longer holds as soon as a quality limit
restricts the spray rate.

**Remarkable:** above `u_mf` the bed pressure drop is **independent** of the
gas velocity — it depends only on the bed mass. The dependence of the fan power
on `u_gas` comes from the volume flow and the distributor plate, not from the bed.

**Deliberately not in the model:** PSD (only two moments), agglomeration, attrition, breakage,
elutriation, droplet size distribution, spatial gradients, **outlier process of the
sensors**. Consequence for `p_nozzle`: pure cost effect without process benefit — fix it in RL
or attach a quality penalty.

---

## Excitation design for data generation

The proposal provides for TUHH-A-AP 3 to vary parameters "within their expected real
range of fluctuation". That is suitable for validation, but excites the dynamics **too weakly** for
learning tasks. Implemented:

- **Step sequences** with random amplitudes and dwell times
- **PRBS** as a true m-sequence via a shift register — not as a Bernoulli coin
- **Ramps** of different slopes — the tracking error is directly proportional
  to the time constant
- **Constant runs** as reference for steady states, **evenly distributed over the
  design order** (as a block at the beginning, `subset(8)` contained 37 % instead of 10 %)
- **Quasi-Monte Carlo over the disturbance space**, with daily cycle and OU drift
- **Scatter of the initial states** over the operating range

**Sobol instead of Latin hypercube**, and **support points only as powers of two**
(8/16/32/64/128): the balance property of a Sobol sequence holds only for `n = 2^k`, and
only nested designs separate data quantity from coverage quality.

**Rate limiting is a mandatory component.** As long as the heater dynamics are missing from the process model,
it is the only place where actuator dynamics are represented. Price: the spectrum
is no longer flat. Achieved bandwidth relative to the corner frequency of the fast mode:
`T_in` 0.16, `u_gas` 1.00, `m_spray` 0.23, `p_nozzle` 0.53.

**Initial states are scattered as well.** If every run starts at `d_50` = 400 µm and grows
monotonically, `d_50` within a run is practically a function of the time index — models
with position information then derive the slow mode from the step count.

**Collinearity:** signal type and amplitude level are drawn independently per run **and per channel**.
If all channels run the same step sequence, ∂`dE`/∂`T_in` and
∂`dE`/∂`u_gas` cannot be separated — exactly the derivatives that the monotonicity constraints
are meant to enforce. Achieved maximum correlation: 0.23.

**For comparison, the lab experimental design** (`LAB`): 10 batches, five parameters at two to
five levels, predominantly constant runs, one batch with spray rate in five levels. That is
a classical DoE for finding limits and **too weakly excited** for learning curves — the
actions take only a few different values over all batches, which makes them weak
predictors. For AP 2 this means: the experimental design of the lab validation is
to be extended relative to the previous project, not adopted. **This is the concrete,
cheap measure that follows from the comparison of the two designs.**

---

## Open questions to TUHH-A

1. **Variable list as interface contract**: exact names, units, classification
   (action/disturbance/state/output). The separation disturbance vs. state variable is
   important in substance.
2. **Time step and typical simulation horizon.** Reference point from the
   previous project: 2 s plant data, 1.8 h batch duration.
3. **Realistic action ranges incl. fluidization limits** and maximum
   rates of change.
4. **Computing time per simulation run.** Reference point from the previous project: 0.27 s computing time
   per second of process time for the dryer model. If this holds for the granulation even only
   approximately, a 128-run data set is a matter of days, not hours.
5. Which process units are static, which transient.
6. **Sorption isotherm in the material system of Dyssol.** In the previous project the wrong
   isotherm was the main cause of the deviation. Which one is stored in the new model, and
   for which material?
7. First (possibly unvalidated) flowsheet `.dflw` of the single granulator.
8. **Are time constants resolved in Dyssol such that a comparison with the
   grey-box values is possible?** That is the quantitative criterion for the calibration —
   if the time constants agree, the dynamics relevant for sequence models agree. To be
   compared are the values at the **realistic** moisture level, not those of the
   placeholders.
