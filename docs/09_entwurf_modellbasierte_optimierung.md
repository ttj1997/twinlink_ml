# Draft: model-based optimisation with a soft-sensor margin for two fluidised-bed lines

> Status: **DRAFT, 2026-10-02. Nothing is built, no run was made.** Written for the project
> lead as a basis for decisions (MO3). Numbers marked *(finding)* come from a lab-book entry
> and an artifact; numbers marked *(derived)* are my arithmetic from such numbers; numbers
> marked *(estimate)* have no measurement behind them. No number in this file comes from a
> conversation.
>
> Context: Befund 68 is closed (lab book, 2026-10-02): RL becomes the comparison, the main
> path for the conference is model-based optimisation. This draft is the design of that path.
>
> **Revision 2026-10-02 (second pass).** The stage plan RU1-RU3 was supplied by the project
> lead and is filed as `docs/10_bestandsaufnahme_rl_umgebung.md`, with a reconciliation
> against this plan (section "Reconciliation with MO2" there). Confirmed or fixed by the
> project lead since the first pass: D3 (limit 0.15 main case, 0.20 stress case, reported
> separately), D4 (lab lines, linear scaling, line B synthetic), D5 (per-carrier objective,
> gas-fired heaters main case, electric as sensitivity), a fixed check step of the table
> method against the direct method (1.6, M3), and a working assumption for the conference date
> (D1). RU1-RU3 are in the repository now; D2 is resolved by that file.

## MO1 Design

### 1.1 System

| Element | Content | Source / status |
|---|---|---|
| Line A | fluidised-bed line, steady-state grey-box **laboratory model** (`solve_steady_state`) | existing config |
| Line B | a defined, justified variation of line A, **synthetic** and labelled as such everywhere; **not** derived from the retracted parametrisation of Befund 56 | which parameters differ and why: to be fixed and justified in M0 (D4) |
| Scale | stationary mass and energy flows of each line scaled **linearly** to plant size; factor chosen so that the exhaust flow of one line equals that of the fluidised-bed source of the partner model (8 kg/s *(finding, Befund 60)*; the lab model runs at 0.0294 kg/s *(finding, Befund 57)*, factor about 272 *(derived)*) | a scaling computation, never a plant statement; the process model itself is not re-solved at plant size |
| External source | one fixed exhaust stream (partner model: spray-tower line Q1, 20 kg/s, 125.233 degC, Y = 0.06482) | Befund 60, data case PA2 *(finding)*; mapping to be confirmed (D4) |
| External sink | one fixed sink (partner model: S1, 20 kg/s, 15 -> 300 degC) | Befund 60 *(finding)* |
| Line sinks | the inlet-air heater of each line: intake ambient, set point = the line's `T_in` | follows from the partner model S2 (8 kg/s, 15 -> 120 degC) *(finding)*; `T_in` range 60-120 degC (action box) |
| Shared heat exchanger | cascade `heatnetwork.balance.evaluate_cascade`, `dT_min` 20 K, `T_kalt` placeholder | `docs/06_waermenetz.md`; both are placeholders |
| Disturbance | ambient temperature x relative humidity (shared by both lines), optionally external-source states | D10 |

The coupling is real: the exhaust state of line k (flow, temperature, humidity) is the
source of the network, and its `T_in` is the set point of a sink. Line k's process solution
depends only on its own set points and the ambient state, **not on the other line**. The
only coupling between the lines is the network. That is what makes the cost question (1.6)
tractable.

### 1.2 Decision variables

Four continuous: `T_in` and `u_gas` for each line. Spray rate fixed at nominal (as Befund
52: otherwise the minimum is zero production). Nozzle pressure: Befund 52 found it
practically invariant (0.4 % of its span, driven to its lower bound) *(finding)*; I propose
to fix it at that bound and to verify this once with a six-variable run on one scenario
(D6). Heat allocation (which source feeds which sink, in which cascade order): discrete,
enumerated over a pruned list of physically meaningful routings with `FixedRouting` /
`MatrixRouting`; no continuous split (the routing module states a split model is not part of
it, EA6). The count "four to six variables" is then: 4 process set points, +1 or +2 only if
nozzle pressure is released (D6).

### 1.3 Constraints

- Product moisture of each line: `X_k <= L_q - m` (quality limit tightened by a constant
  margin), with the **kinetics kept at the critical moisture 0.20 kg/kg** (decoupled
  mechanics of Befund 64; `MaterialConfig.product_moisture_limit_kg_per_kg` exists since
  S0.1). `m = 0.005961819615175404 kg/kg`, the median half-width of the calibrated ensemble
  band (Befund 65 FA2, used in S0.3) *(finding)*.
- Fluidisation window `u_mf(d50) <= u_gas <= u_terminal(d50)`, bed temperature <= 80 degC
  (placeholder, Befund 46; it never bound in Befund 52), solver validity
  (`SteadyStateSolution.feasible`), network grade `dT_min` (inside the cascade).
- **Distance to the validity boundary of the kinetics.** The solver marks a state as invalid
  above the critical moisture; it is a validity boundary of the normalised drying curve, not
  a numerical artefact (S0.4) *(finding)*. With the quality limit equal to 0.20 the tightened
  limit lies `m` (0.006 kg/kg) below that boundary: the optimiser would sit at the edge of
  the model. With `L_q = 0.15` the tightened limit is 9.4 m away (S0.4) *(finding)*. Rules for
  the design: (i) every optimum reports `d = 0.20 - (L_q - m)` in kg/kg and in units of `m`;
  (ii) at `L_q = 0.20` the case is reported separately as a stress case, not pooled; (iii) a
  solver failure or invalid state is **masked (NaN / excluded), never replaced by a large
  sentinel** inside any table or interpolation (the 1e6 pitfall of `CLAUDE.md`; for the
  ordinal DE a penalty is acceptable, for interpolation it is not).
- The margin is a coverage band, not a guarantee: Befund 65 FA2 gives 94.6 % / 96.1 %
  coverage on EV3 and finds the spread **not usable per operating point** (rank correlation
  -0.43 / -0.30) *(finding)*. The optimiser treats the margin as a deterministic
  tightening with the model as truth (as Befund 64); residual violations of the real plant
  are not modelled here and must be named as such. For line B no soft sensor exists:
  `m_B = m_A` is an assumption with a sensitivity (D8).

### 1.4 Objective: CO2-weighted, separated by energy carrier, kWh sum from the same run

Per scenario and per line the run stores two energy streams and derives everything from them:

- `E_el` [kW]: fan + pump + atomiser air (electric) *(`simulation.energy` components)*.
- `E_gas` [kW]: fuel of the line's air heater after recovery, `(Q_heater - Q_rec,k) / eta_H`
  with the cascade `Q_rec,k`; plus the same for the fixed external sink if it is in the
  system boundary (D4).
- `CO2 = 0.366 * E_el + 0.201 * E_gas` [kg CO2e/h] (factors of `net_energy.py`, source:
  full project description, UBA) *(finding: constants in code, not re-verified here)*.
- `kWh sum = E_el + E_gas`, reported from the **same solution**, not from a second run, so
  the two can be compared without a second optimisation.

**Decided (D5): the objective is rewritten per carrier.** The existing net objective treats
the whole gross power of the process, including the air heater, as electric
(`EnergyConfig.heater_efficiency` 0.98) and credits the recovered heat as gas fuel
`Q_rec / eta_H` (`net_energy.py` docstring). For a *gas-fired* heater that mixes carriers.
Main case: the air heaters of the lines are gas-fired (`E_gas` as above, `eta_H` from
`HeatNetworkConfig`, once); sensitivity: electric heaters (`E_el` carries the heater at
`EnergyConfig.heater_efficiency`). Hence the objective is a **new function**, not a call of
`net_objective_co2_kg_per_h`; the old function stays unchanged, because Befunde 57-68 rest on it. Befund 57 NB2 showed that the weighting, not the
formulation, decides the effect (the gas velocity leaves its lower bound only under the kW sum)
*(finding)*; the per-carrier output is what keeps that visible.

### 1.5 Comparison arms (test set to be named in every table)

Test set: the 25 Befund 52 ambient grid points for the single-line anchor, the scenario
set of D10 for the two-line system. Each arm is evaluated with the *same* two-line model and
the same objective.

| Arm | Set points | Heat allocation | Role |
|---|---|---|---|
| F | fixed per line (Befund 52 BB3(a) mechanics, searched per line over the scenario set, feasible in every scenario) | fixed routing (partner "Fall 1": each source to its own sink) | status quo |
| F+R | fixed per line | rule-based (e.g. highest exhaust temperature to highest sink demand) | isolates the allocation effect |
| S | per-scenario optimum of each line **ignoring the network** (as Befund 52/64), tightened limit | best routing at those set points | isolates the set-point effect |
| M | **joint optimum** (MO), tightened limit | optimised routing | the main arm |
| M0 | joint optimum with margin 0 | optimised | cost of the margin = M - M0 (Befund 64 mechanics) |
| P | joint optimum with the **pinch target** as network model | ideal network | lower bound of M (see below) |
| RL | existing single-line results (S0.3, S0.6), no new RL run | -- | comparison; single line only (D11) |

The pinch bound (`heatnetwork/pinch.py`): (a) evaluated at each arm's set points it bounds the
allocation only; (b) used *inside* the optimisation (arm P) it bounds the joint problem under
the same process constraints, for an ideal network with arbitrarily many exchangers. It is a
lower bound, not a construction rule (caveat of Befund 60, must be stated at every use). The gap
of 2.3-4.2 % of Befund 60 is a property of one data case *(finding)*; the regime-jump result of
Befund 61 is **suspended (R5)** and must not be cited. The pinch is non-smooth in the set
points, which is no problem for DE or for a table search.

Claims of ranking between arms are made only after the resolution check: the optimiser is
deterministic on a table, so the resolution is the table spacing and, for the direct DE, the
spread over three DE seeds on a subset (see 1.6). Report median and range over scenarios,
never a single scenario.

### 1.6 Computational cost: what does one optimum per disturbance scenario cost?

Measured basis *(finding)*: the 3-variable DE + Nelder-Mead of Befund 52 (`popsize` 10,
`maxiter` 50) uses `nfev = popsize * n * (maxiter + 1) = 1530` DE evaluations (checked against
the artifact `bb1_bb3_operating_point_regret.json`, all 25 points: median 1530) plus about 81
refinement iterations; 520-580 s per point on one core (Befund 65 FA3 text); 25 points on 16
workers in 17.0 min (S0.5 oracle, 1022 s). **Runtime difference, checked in the records
(2026-10-02, read-only):** the original Befund 52 artifact (`bb1_bb3_operating_point_regret.json`,
median DE phase 1765 s per point, refinement 205 s; the `_PREFIX` file 1734 s) is the
original *Linux* run (lab book Befund 52, addendum on the Windows reproduction) and carries no
environment, thread or load fields at all (keys: budget, fingerprint, grid results, fixed
point, regret, spread); whether it ran under parallel load cannot be read from it. The
Windows reproduction `study_operating_point_regret_win_repro.json` (clean tree, commit
`7103b73`) has median DE 503 s and refinement 51 s with the same budget (`nfev` 1530 in all
files, 16 workers in all); its environment record shows OMP/MKL thread counts unset and
`torch/num_threads` 8, no load information either. The later oracle runs show 527 s (S0.5)
and 399 s (`probe_margin_energy_cost.json`, 63 points). So the 3.5-fold difference is a
platform/run difference with the same algorithm and work; its cause (hardware, thread
oversubscription, load) is **not determinable from the records**. The estimates below use the
Windows figures. One solve costs about 0.3 s *(derived: 550 s / roughly 1700 solves, measured
with 16 workers busy)*; a single-thread measurement of 164 ms per environment step was
supplied in RU1 and is not in a lab-book entry; if it holds, the times below roughly halve.
M0 measures it.

Direct approach (DE + NM on the joint problem, one objective call = two solves + cascade):

| Variables | DE calls (`10 n 51`) | Solves incl. refinement *(estimate)* | One core | 25 scenarios on 16 cores | 100 scenarios |
|---|---|---|---|---|---|
| 4 | 2040 | about 4 600 | about 25 min | about 50 min (2 waves) | about 3 h |
| 6 | 3060 | about 6 800 | about 38 min | about 80 min | about 4.5 h |

*(estimate: assumes the generation budget stays at 50, so the DE is not converged by design as
in Befund 52, and that the cascade and the routing enumeration cost much less than a solve; the
cascade cost was not measured, and an unpruned 512-subset enumeration per call would dominate:
measure first, stage M0.)*

**Separable approach (recommended primary):** because each line depends only on its own set
points and the ambient state, precompute a response table per line and ambient point on the
13 x 9 reference grid (117 solves, about 35 s *(derived)* per line), optimise the joint
problem on the interpolated tables plus the exact cascade (negligible cost), then **polish**
with true solves (about 100-150 per line). About 2.5 min per scenario *(estimate)*, i.e. about
25 min for 150 scenarios on 16 cores, and external-source or sink variations (which do not
change a line's process solution) reuse the tables at no solve cost. The table gives a
near-global optimum within its spacing (a certificate the DE does not give); the direct DE on a
subset of three scenarios with three seeds is the cross-check and the source of the resolution.
Risk: the moisture constraint is a cliff; the tables need local refinement near
`X = L_q - m` and the polish must be verified to leave the table optimum's constraint side.

**Fixed check step (project lead's requirement, stage M3).** A sample of the table-based optima
is recomputed with the direct method (DE + NM on the joint problem). Sample design, fixed before
the run: at least 12 scenarios, of which at least half are scenarios where the moisture
constraint binds at the table optimum (binding = `X` within the tolerance of
`operating_point.TOLERANCE_MOISTURE_KG_PER_KG` of `L_q - m`), the rest chosen to cover the
non-binding regime and the regime transition of Befund 52 (c). Compared per scenario: objective
(CO2 and kWh sum), the four set points, `X` against `L_q - m`, constraint side, routing. Reading
fixed beforehand: the table method is accepted if its objective is not worse than the direct
one by more than the spread of the direct method over three DE seeds in that scenario, and no
table optimum violates the tightened limit by more than the tolerance; at a failing scenario the
direct result is reported and the table grid is refined there, not the criterion changed.

**Realistic number of scenarios:** the 5 x 5 ambient grid of Befund 52 resolved the regime
transition only to 7.5 K x 0.15 *(finding)*; I propose 7 x 7 ambient points (49) times three
external-source states (about 150 optima): about 25 min with tables *(estimate)*, about 4 h
direct *(estimate)*. A first useful number to expect from the direct route alone is 25
scenarios in one evening.

### 1.7 Reuse

| Building block | Where | Use | Gap / action |
|---|---|---|---|
| DE + NM mechanics, `OptimizationBudget`, `optimize_grid_point`, `find_fixed_operating_point`, `compute_regret`, `build_grid` | `scripts/study_operating_point_regret.py` | oracle, fixed point per line, regret | script with module globals (`_BASE_CONFIG`), not importable for two lines; move to `src/` as in S0.1, bit-identical (VA2 oracle test, config fingerprints) before any new code uses it |
| constraints, penalties, tolerances, `build_variant_config`, `MaterialConfig.product_moisture_limit_kg_per_kg` | `simulation/operating_point.py` | constraint check, decoupled limit | none; reuse as is |
| 13 x 9 reference grid | `rl/lagrange.py` (`REFERENCE_GRID_*`) | response tables | reuse the constants; `ReferenceTable` and the failure-value rule are the lesson (no sentinel) |
| cascade, routing, config | `heatnetwork/balance.py`, `routing.py`, `config.py` | network, `FixedRouting`, enumeration | `MatrixRouting` is binary/priority, no split; one source's `T_kalt` is network-wide (docs/06) |
| pinch target | `heatnetwork/pinch.py`, `scripts/run_pinch_benchmark.py` | arm P, bound at arm set points | wrap for use inside the objective; check cost per call |
| net objective | `evaluation/net_energy.py` | emission factors, `fuel_equivalent_credit_kw` | new per-carrier function (1.4); do not reuse `net_objective_co2_kg_per_h` as is |
| single-point `Q_rec` | `scripts/study_operating_point_regret_net.py::_q_rec_kw` | consistency check | replaced by the cascade; test that a one-line network reproduces it (sensible mode) |
| 25 test points, boundary test, closed-regret metrics | `rl/evaluation.py` | RL comparison, test set | reuse; single line only |
| SX1 resolution thresholds | `evaluation/resolution.py` | RL arm only | not needed for the deterministic optimiser |
| reference values at limit 0.15, oracle and fixed point | artifact `s05_limit015_oracle_and_fixed.json` | anchor for a single line | reuse; check the new code reproduces it before anything else |
| cost of the margin, decoupled | artifact `probe_margin_energy_cost_decoupled.json`, `probe_soft_sensor_margin_requirement.json` | anchor | same |
| Lagrange / dual ascent, SAC training | `rl/lagrange.py`, `scripts/run_rl_bandit_lagrange.py` | not used | RL arm uses committed artifacts |

New: two-line composite model, response-table builder, per-carrier objective, routing
enumeration (pruned), scenario definition, comparison driver, one result artifact per arm.

### 1.8 What can invalidate the result (state in every use)

1. **Scale.** The lab simulator runs at about 0.03 kg/s, the partner network at 8-20 kg/s
   (`docs/06_waermenetz.md` scale reservation). The lines stay laboratory models; their
   stationary flows are scaled linearly (1.1). Every kW from this coupling is a scaling
   computation, not a plant statement, and is labelled so in every table and caption. Linear
   scaling of the stationary flows is a definition here, not a tested property of the process.
2. **Plant parametrisation.** Befund 56 NB1/NB2 retracted the air-flow-based match and
   predicted a solid fraction of 71-73 % instead of the 30 % placeholder; Befund 61 is
   suspended on it *(finding)*. Line B must not be derived from the faulty NA5
   parametrisation.
3. **Placeholders without source:** critical moisture 0.20, `dT_min` 20 K, `T_kalt` (15 degC
   is the optimistic case, 29.1 degC the control-system hint), bed temperature 80 degC,
   `tau_4`. The headline is reported with the sensitivities of Befund 54 / docs/06 named.
4. **Steady state versus batch and asynchronous lines.** Befund 53 shows the process-level
   regret survives batch dynamics, but the batch-averaged `Q_rec` lies 36 % (median, up to
   76 %) above the fixed-point value *(finding)*: the network credit is the part that
   changes under transients. A transient check on a subset is part of the plan (M6).
5. **Margin transfer** to line B and to operating points other than EV3 (1.3).

## MO2 Stage plan to the conference

Working assumption (project lead): conference end of June 2027, about 39 weeks from
2026-10-01, until the date is confirmed (D1). The plan is built so that a **complete
contribution can be due several months earlier**: the core (M0-M4, with the check step) is
deliberately separated from the extensions (X1-X4), which only use time that exists.

Time is in working days (person time incl. waiting for machine time), *(estimate)*.
Long runs (hours) go to the local machine, sweeps with `--resume`; every script is
committed before it runs, tree clean, new artifact names; the three config fingerprints and
their regression tests stay green throughout.

| Stage | Content | Days | Cumulative | Output |
|---|---|---|---|---|
| M0 Preparation | confirm D6-D14; move the DE/NM helpers of the regret script to `src/` (bit-identical, VA2 oracle test, fingerprints); measure solve, cascade and pinch cost per call; define and justify line B | 3 | 3 | cost numbers replacing the estimates of 1.6; clean base |
| **M1 Earliest presentable stage** | single line, 25 grid points, `L_q = 0.15` (main) and 0.20 (stress, separate), oracle at `L_q - m`, `m = 0.00596`, arms F and M0/M; per-carrier objective, gas heater main; anchors first (`s05_limit015_oracle_and_fixed`, Befund 64) | 3 | 6 | cost of the soft-sensor margin in CO2 by carrier and kWh, distance to the validity boundary, the RL negative result (Befund 68) on the same test set |
| M2 One line plus external source | network with the fixed source and sink, scaled; arms F, F+R, S, M | 4 | 10 | allocation versus set-point effect |
| M3 Two lines and the check step | line B, response tables, joint optimum; **check step: at least 12 optima against the direct method, at least half at the binding moisture limit (1.6)**; DE seeds on 3 scenarios | 5 | 15 | arm M and the table-versus-direct verdict |
| M4 Comparison and bound | arms P, M0; resolution check; scenario set 7 x 7 x 3 | 4 | 19 | comparison table, bound gap |
| **Core complete** | M0-M4 | | about 19 days, about 4 weeks net; about 6 weeks with 50 % overhead *(estimate)* | a complete contribution |
| M5 Robustness | `dT_min`, `T_kalt` (per source 10/15 and 29.1), `L_q`, `m_B`, electric heaters, kinetics placeholder, nozzle pressure released | 4 | 23 | which statements survive |
| M6 Transient check and write-up | batch rollout on about 5 scenarios (Befund 53 pattern), lab-book entry, synthesis (numbers only from finding + artifact) | 5 | 28 | conference material |

Net 28 days (about 5.5 weeks), about 8-9 weeks with 50 % overhead. That leaves roughly 30
of the 39 weeks for the extensions below and for waiting on partner answers; none of them is
needed for a complete contribution:

| Extension | Content | Weeks *(estimate)* |
|---|---|---|
| X1 Storage | hot-water buffer module in `heatnetwork/` (none exists; a buffer is confirmed real, Befund 56); optimum with perfect foresight over the horizon as an upper bound for its benefit | 4-6 |
| X2 Sequential single-line environment | the RL environment of RU2 with rates, disturbance series, constant margin, as the comparison arm on a sequential problem; only meaningful with a positive bandit result or a decided remedy | 3-4 |
| X3 Split model | continuous allocation (EA6) replacing the enumeration | 2-3 |
| X4 Surrogate | only after validation against the grey box, error measured where an optimiser explores, not only at the oracle points | 3-4 |

**Earliest presentable stage: M1**, about 6 working days after M0 starts (about 2-3 weeks
with overhead and the D-decisions). It needs one line and existing mechanics, answers "what
does a realistic soft-sensor margin cost" in the units the partners care about, and contains
the model-based optimum next to the RL negative result on the same test set. For orientation
only (derived, not a finding): linear interpolation of the Befund 65 table between margins
0.0045 and 0.01 kg/kg gives about 94 % of the control benefit remaining at `m = 0.00596`; M1
replaces it by a run. If the deadline is earlier than the core allows, the cut order is: M5
reduced to `dT_min` x `T_kalt` and `L_q`, then M6's transient check to 3 scenarios, then M2
merged into M3; M1 alone remains presentable.

## MO3 Decisions

Status after the second pass. "Fixed" = decided by the project lead; "open" has a
recommendation.

| # | Decision | Status / recommendation |
|---|---|---|
| D1 | Conference date | **working assumption end of June 2027** (about 39 weeks from 2026-10-01) until confirmed; plan built for an earlier due date (MO2) |
| D2 | Stage plan RU1-RU3 | **resolved**: filed as `docs/10_bestandsaufnahme_rl_umgebung.md`, reconciled there |
| D3 | Quality limit | **fixed**: 0.15 main case, 0.20 separately as stress case |
| D4 | Mapping, scale, line B | **fixed**: lab lines; linear scaling of stationary flows, factor so that one line's exhaust flow equals that of the fluidised-bed source of the partner model; labelled scaling everywhere; line B a defined, justified, synthetic variation of line A, not derived from Befund 56. Open inside it: which parameters differ (recommendation: geometry and d50, not the critical moisture, so one validity boundary) |
| D5 | Heater carrier, objective | **fixed**: per-carrier objective rewritten; gas-fired heaters main, electric as sensitivity |
| D6 | Variables | open; recommend 4 process set points, nozzle pressure fixed at its lower bound after one 6-variable check on one scenario, spray rate fixed as in Befund 52 |
| D7 | Heat allocation | open; recommend enumerated discrete routings over a pruned list now; the continuous split is extension X3 |
| D8 | Margin for line B | open; recommend `m_B = m_A`, sensitivity `{0.5, 1, 2} m`; no soft sensor exists for line B; constant-trained band half-width 0.0082 kg/kg as an upper sensitivity |
| D9 | Weighting | open; recommend CO2 primary with the Befund 57 factors, kWh sum from the same run, no cost variant |
| D10 | Scenario set | open; recommend 7 x 7 ambient x 3 external-source states (about 150 optima); the external states need a definition from the partner model |
| D11 | RL comparison scope | open; recommend the existing single-line results only (S0.3, S0.6), compared on the Befund 52 grid; wording "did not reach the oracle under this budget", never "RL cannot" |
| D12 | Transient check | open; recommend yes, about 5 scenarios, Befund 53 pattern (batch-averaged `Q_rec` differs from the steady value, Befund 52 d) |
| D13 | Method | open, with the check step fixed: tables primary, direct DE as check on at least 12 optima, at least half binding |
| D14 | Pinch parameters for arm P | open; recommend `dT_min` 20 K, `T_kalt` 29.1 degC primary; {10, 20} K x {15, 29.1} degC sensitivity; per-source `T_kalt` (10 / 15 in the partner model) is not representable in the network-wide config and needs the probe-script route (docs/06) |
| D15 (new) | Storage and sequential RL environment (RU2/RU3) | open; recommend extensions X1/X2 after the core, only if time remains; not part of the core |

Blocking before M0: D4's open inside-point (line B parameters), D6-D8, D10. The rest has a
default above.
