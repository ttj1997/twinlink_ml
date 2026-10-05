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
>
> **Revision 2026-10-02 (third pass, M0).** All remaining decisions D1, D4 (line B), D6-D15 are
> confirmed by the project lead as recommended below (D1: working assumption stays, the date
> follows; D8: the transfer of the margin to line B is marked as an assumption in every finding;
> D10: the two additional external-source states are synthetic, e.g. part load of the spray
> tower, and marked as placeholders because the partner model supplies one data case only; D11: the
> result of the shield experiment (third addendum to Befund 68) and of SH3 (fourth addendum) is part of
> the RL comparison, see D11 below; the shield did not pass its gate).
> **Change of a criterion, with reason:** the first line-B criterion (median outlet-temperature
> difference of at least 10 K) was an unsuitable proxy and is replaced by LB1-LB3 (below): with
> condensing recovery (Befund 57) the moisture of the exhaust, hence its dew point, decides, not
> the dry temperature. The first criterion, its script and its artifact stay as they are; the
> result under it (no candidate qualifies) is in "M0 results". M0 results are at the end of this
> file.

## MO1 Design

### 1.1 System

| Element | Content | Source / status |
|---|---|---|
| Line A | fluidised-bed line, steady-state grey-box **laboratory model** (`solve_steady_state`) | existing config |
| Line B | line A with a **higher spray rate** (other product, higher water load); same geometry, same critical moisture; **synthetic**, labelled as such everywhere; qualitatively backed by the plant data, where two products had very different operating points, but **no number** from the retracted parametrisation of Befund 56 | **2.0 kg/h** (line A: 1.5), chosen by LB3, see "M0 results" below and the caveat there |
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
| D1 | Conference date | **working assumption end of June 2027** (about 39 weeks from 2026-10-01) until confirmed (confirmed to stay, the date follows); plan built for an earlier due date (MO2) |
| D2 | Stage plan RU1-RU3 | **resolved**: filed as `docs/10_bestandsaufnahme_rl_umgebung.md`, reconciled there |
| D3 | Quality limit | **fixed**: 0.15 main case, 0.20 separately as stress case |
| D4 | Mapping, scale, line B | **fixed**: lab lines; linear scaling of stationary flows, factor so that one line's exhaust flow equals that of the fluidised-bed source of the partner model; labelled scaling everywhere; line B a defined, justified, synthetic variation of line A, not derived from Befund 56. Line B differs in the water load only: a higher spray rate (other product), same geometry, same critical moisture; 2.0 kg/h by LB3 (see M0 results) |
| D5 | Heater carrier, objective | **fixed**: per-carrier objective rewritten; gas-fired heaters main, electric as sensitivity |
| D6 | Variables | **fixed (confirmed 2026-10-02)**:  4 process set points, nozzle pressure fixed at its lower bound after one 6-variable check on one scenario, spray rate fixed as in Befund 52 |
| D7 | Heat allocation | **fixed (confirmed 2026-10-02)**:  enumerated discrete routings over a pruned list now; the continuous split is extension X3 |
| D8 | Margin for line B | **fixed (confirmed 2026-10-02)**:  `m_B = m_A`, sensitivity `{0.5, 1, 2} m`; no soft sensor exists for line B; constant-trained band half-width 0.0082 kg/kg as an upper sensitivity |
| D9 | Weighting | **fixed (confirmed 2026-10-02)**:  CO2 primary with the Befund 57 factors, kWh sum from the same run, no cost variant |
| D10 | Scenario set | **fixed (confirmed 2026-10-02)**:  7 x 7 ambient x 3 external-source states (about 150 optima); the external states need a definition from the partner model |
| D11 | RL comparison scope | **fixed (confirmed 2026-10-02)**:  the existing single-line results only (S0.3, S0.6), compared on the Befund 52 grid; wording "did not reach the oracle under this budget", never "RL cannot" **Update 2026-10-05:** the shield experiment is recorded (third addendum to Befund 68): gate not met (G1, G2, G3 false), RL stays a diagnostic finding. SH3 (fourth addendum): about half of the distance of the agents to the fixed operating point (0.49-0.52 of the gap per arm) arises at the lower bound of the gas velocity, which the oracle reaches at 22 of 25 points and which a tanh-squashed policy does not reach exactly (no agent action within 0.01 m/s of 0.80 m/s); reading PARTLY. Wording for the comparison, unchanged: "did not reach the oracle under this budget", never "RL cannot". |
| D12 | Transient check | **fixed (confirmed 2026-10-02)**:  yes, about 5 scenarios, Befund 53 pattern (batch-averaged `Q_rec` differs from the steady value, Befund 52 d) |
| D13 | Method | **fixed (confirmed 2026-10-02)**, with the check step: tables primary, direct DE as check on at least 12 optima, at least half binding |
| D14 | Pinch parameters for arm P | **fixed (confirmed 2026-10-02)**:  `dT_min` 20 K, `T_kalt` 29.1 degC primary; {10, 20} K x {15, 29.1} degC sensitivity; per-source `T_kalt` (10 / 15 in the partner model) is not representable in the network-wide config and needs the probe-script route (docs/06) |
| D15 (new) | Storage and sequential RL environment (RU2/RU3) | **fixed (confirmed 2026-10-02)**:  extensions X1/X2 after the core, only if time remains; not part of the core |
| D16 | Lower bound of the gas velocity in M2 onwards | **changed (project lead, 2026-10-05; first decision of the same day superseded)**: **main case for M2 is the lower bound 1.26 m/s** (the laboratory minimum, the only value with measurement support; at 0.80 m/s the model extrapolates below the laboratory range, 1.26-1.88 m/s, and the 0.80 is a placeholder without derivation); **0.80 m/s runs along as the comparison** to Befund 52 to 69, which were computed at 0.80. Every heading, caption and table names the lower bound ("at lower bound 1.26 / 0.80 m/s"). The bound is a run parameter (`--gas-velocity-min`), the config default is not changed. Reason for both values: the optimum sits on the bound at (nearly) all points whatever its value (second addendum to Befund 69), so the bound decides the result and may not stay implicit. The third value (a multiple of `u_mf`) follows when the factor has been clarified with process expertise; nothing is computed for it until then. Stability rule for statements across bounds: same direction and a figure that changes by less than half; otherwise both columns are given |

All decisions are fixed; the blocking list of the second pass is empty. D8: the margin transfer to line B is marked as an assumption in every finding. D10: the two extra external-source states are synthetic placeholders.

## M0 results (2026-10-02)

Branch `m0` (fixed worktree `twinlink_m0`, merged fast-forward into `master`); every script was
committed before it ran, every run record has `git/dirty=False`. Artifacts: `m0_cost_per_call`,
`m0_line_b_spray_rate`, `m0_line_b_network`. The lab-book entry for M0 and M1 is Befund 69 (the
shield experiment is recorded as the third and fourth addendum to Befund 68, not as its own Befund).

**M0.1 Helpers moved to `src/`.** `evaluate_point`, `OptimizationBudget`, `run_de_nm` (was
`_optimize`), `build_grid`, `action_bounds`, `compute_regret` now live in
`src/twinlink_ml/simulation/oracle_search.py`; the script re-imports them; the function bodies were
checked equal to HEAD by AST comparison (docstrings aside); 3 new tests. Full suite: 561 passed,
1 skipped in 988.55 s; ruff clean.

**M0.2 Cost per call (measured, single process).** Steady-state solve at the search precision of
Befund 52: median of the three round medians 0.209 s (rounds 0.178 / 0.209 / 0.258 s; 23 of 24
points solved per round, a failed solve costs 0.013-0.020 s); at the solver defaults 0.49-0.67 s
(11 of 12 solved). Cascade on the partner data case: 1.30-1.40 ms per call in sensible mode,
0.056-0.081 ms in condensing mode (the difference was not investigated). Pinch target (50
segments): 4.1 ms. So the estimate of about 0.3 s per solve in 1.6 was somewhat high; the
solve dominates and the network and the pinch are negligible by two orders of magnitude, which
makes the enumeration of 343 allocations cheap (87 ms per scenario in the smoke test). The
increase from round 1 to round 3 is unexplained. Processor load, sampled at phase ends: 5 %
(start), 4 % (after the search-precision phase), **62 %** (after the solver-defaults phase), 7 %,
9 %: another session's `pytest -q -x` process (not started by this work) was running on the machine
from 13:51 and probably explains the 62 % reading; the search-precision phase itself ran at 4-5 %.
The 164 ms per environment step supplied in RU1 lies within the first round (178 ms median),
consistent as an order of magnitude, not reproduced exactly.

**M0.3 Line B under the first criterion (outlet temperature difference of at least 10 K).**
Spray rates 1.5 (line A) / 2.0 / 2.5 / 3.0 kg/h, 3 x 3 ambient grid, quality limit `0.15 - m`,
oracle of Befund 52 (artifact `m0_line_b_spray_rate`). All optima solver- and constraint-feasible
at all 9 scenarios. Median `|T_out(s) - T_out(1.5)|` at the optima: 4.6 K (2.0), 6.9 K (2.5),
6.9 K (3.0): no candidate meets 10 K. From 2.5 kg/h on the inlet-temperature optimum sits at the
upper bound of 120 degC in most scenarios. The criterion was an unsuitable proxy (see the
revision note at the top); the result stands as measured.

**M0.4 Line B under LB1-LB3 (artifact `m0_line_b_network`; evaluation only, the stored optima
re-solved at their set points; criteria committed before the run, script `f1c013d`).** Scale
factor `k = 271.797` (8 kg/s over the dry-air flow of line A at the nominal point, 0.029434 kg/s):
a scaling computation, not a plant statement. Network: Q1 and S1 from the partner data case, line
exhausts LA/LB as sources, line inlets SA/SB as sinks, condensing, `T_kalt` 29.1 degC, `dT_min` 20 K,
`eta_H` 0.80, `H_u` 45 MJ/kg, gas-fired heaters, CO2 factors 0.366 / 0.201 kg/kWh.

*LB1 (descriptive, lab scale).* Dew point of the exhaust over the 9 scenarios [degC] / `Y_out`
[kg/kg] / `T_out` [degC]: 1.5: 19.8-40.1 / 0.0146-0.0492 / 32.1-48.2; 2.0: 28.2-42.3 /
0.0244-0.0556 / 42.6-51.5; 2.5: 30.9-43.0 / 0.0288-0.0580 / 45.0-52.6; 3.0: 31.1-43.0 /
0.0290-0.0581 / 45.0-52.6. Recoverable down to `T_kalt + dT_min` = 49.1 degC: zero in all scenarios
for 1.5 and in 8 of 9 for 2.0 (maximum 0.039 kW sensible), in 7 of 9 for 2.5 and 3.0 (maximum
0.064 / 0.078 kW), latent part zero everywhere. Recoverable down to `T_kalt` = 29.1 degC
(sensible; latent): 1.5: 0.078-0.31 kW; latent only in 4 of 9 scenarios (maximum 0.85 kW); 2.0:
0.23-0.36 kW; latent in 7 of 9 (maximum 1.07 kW); 2.5: 0.30-0.44 kW; latent in 9 of 9
(0.12-1.31 kW); 3.0: 0.36-0.52 kW; latent in 9 of 9 (0.16-1.58 kW). So the water load raises the
dew point mainly at the dry end (minimum 19.8 -> 28.2-31.1 degC, maximum 40.1 -> 42.3-43.0 degC)
and makes condensation available in more scenarios, while the heat above the floor of 49.1 degC
is practically nil for all lines.

*LB2 (decisive criterion as fixed).* Median over the 9 scenarios of (CO2 of the worst allocation
minus CO2 of the best) / CO2 of the best, whole site: 0.358 (control, line B = line A at 1.5),
0.360 (2.0), 0.361 (2.5), 0.359 (3.0); **met for every candidate, and for the control.** The
criterion therefore does not discriminate between spray rates: the difference is the allocation of
the fixed partner source Q1 (125 degC) between S1 and the line inlets, independent of line B. In the
best allocation Q1 feeds S1, SA and SB; the line exhausts feed S1 without recovering anything; in
every scenario between 8 and 196 allocations tie with the best (extra connections of the line
exhausts add nothing). Number of distinct best allocations over the 9 scenarios / scenarios whose best differs
from the most frequent one: 3 / 5 (control), 2 / 2 (2.0), 2 / 1 (2.5), 3 / 2 (3.0). Descriptive
measure added before the run, which isolates what the lines themselves decide (allocations of the
two lines only, Q1 held at its best connections, relative to the best site CO2): median 0.56 %
(control), 0.65 % (2.0), 0.84 % (2.5), 1.03 % (3.0): all below 2 %, increasing with the spray
rate. Relative to the CO2 of the two line sinks plus line electricity only, the criterion
difference is a factor 2.3-3.0 (not part of the criterion).

*LB3 (choice).* Interior inlet temperature (below 120 degC minus 0.05 K) in 9 of 9 scenarios for
2.0, 3 of 9 for 2.5, 4 of 9 for 3.0. **The smallest rate that meets LB2 and has an interior optimum
in the majority is 2.0 kg/h; line B = 2.0 kg/h by the rule.**

*Caveats, to be stated in every use.* (1) LB2 as fixed does not select; the choice of 2.0 rests on
LB3 alone, i.e. on the interior optimum, not on the allocation being decisive. (2) What the
allocation decides in this system is mainly the distribution of the fixed source Q1; the lab
exhaust at 32-53 degC offers almost nothing to the sinks at `dT_min` 20 K (zero above 49.1 degC in
most scenarios), and the lines-only allocation effect is 0.6-1.0 % of the site CO2 for every
candidate. This bears on the design: the claim that heat allocation is a real decision between the
two lines is not supported by this pre-check at lab-scale exhaust temperatures; it holds for Q1.
(3) The lines are optimised individually, not jointly with the allocation. (4) All kW and kg CO2e/h
of the network evaluation are scaled by `k`, a scaling computation; the site is dominated by the
fixed sink S1 (20 kg/s to 300 degC). (5) `T_kalt` 29.1 degC, `dT_min` 20 K, `eta_H`, `H_u` are
placeholders. (6) Line B is synthetic.

## M1 results (2026-10-05) -- at the gas-velocity lower bound 0.80 m/s

**Every M1 number holds at the lower bound 0.80 m/s of the gas velocity in the config, a
placeholder that sets the optimum at 21-23 of 25 points (see "Gas-velocity bound" below).**

Single line A (spray rate 1.5 kg/h), no network (`Q_recovered` = 0), objective `co2_gas_heater`
(D5 main case: gas-fired air heater, `eta_gas` 0.80, factors 0.366 / 0.201 kg CO2e/kWh), margin
`m` = 0.005961819615175404 kg/kg, kinetics at the critical moisture 0.20 (decoupled mechanics of
Befund 64). **Test set (named):** the 25 ambient grid points of Befund 52 (temperature x relative
humidity). Arms: F fixed operating point against `L_q`, M0 oracle against `L_q`, M oracle against
`L_q - m`; the optimiser budget is that of Befund 52 (asserted against the committed artifact).
Code and settings were committed before any run (`851c23a`, settings and anchor rule in the script
docstring); all run records `git/dirty=False`. The artifacts and code are on the branch `m1` of the
fixed worktree `twinlink_m0` (commits `4faf95d`, `b65b3bb`, `e8e085b`), **not merged into master and
not pushed**. Full suite in the worktree before the code commit: 571 passed, 2 skipped in 1442.46 s
(the extra skip is the denylist test, which passed in the main checkout).

**Anchor (stop condition, answered first).** The legacy objective (Befund 52 `energy_target`,
threshold 0.15) of the new code path at grid points 0, 12, 24 reproduces the optimum of
`s05_limit015_oracle_and_fixed.json` bit for bit: set points and energy, all differences 0.0
(artifact `m1_anchor_s05`). The new objective function and penalty scaling (the Befund 52 constants
converted from kW to kg CO2e/h with the gas factor, a setting of this study) are therefore the only
difference to the earlier machinery.

**Result (artifacts `m1_margin_cost_l015`, `m1_margin_cost_l020`; per point and pooled; all 25
points feasible and converged in all arms, no solver failure).** `L_q` = 0.15 is the main case;
`L_q` = 0.20 is the stress case and is reported separately, not pooled.

| | `L_q` 0.15 (main) | `L_q` 0.20 (stress) |
|---|---|---|
| points where M0 binds the limit | 22 of 25 | 17 of 25 |
| remaining control benefit, CO2, median (all points) | 85.4 % | 93.8 % |
| remaining control benefit, CO2, pooled (all points) | 85.0 % | 94.2 % |
| remaining control benefit, CO2, median (binding points) | 84.9 % | 93.4 % |
| remaining control benefit, kWh sum, median / pooled (all points) | 85.5 % / 85.0 % | 93.8 % / 94.3 % |
| cost of the margin M - M0, CO2, median / maximum | +1.30 % / +1.81 % (0.0039 kg CO2e/h) | +1.06 % / +1.35 % (0.0030 kg CO2e/h) |
| cost of the margin, kWh sum, median | +1.34 % (0.0196 kW) | +1.10 % (0.0144 kW) |
| distance of M to the validity boundary 0.20, binding points | 0.0560 kg/kg = 9.4 m | 0.00596 kg/kg = 1.0 m |
| fixed operating point F [degC, m/s, bar] | 93.0, 0.800, 1.002 | 90.4, 0.800, 1.01 |

Remaining control benefit = `(F - M) / (F - M0)`, the definition of Befund 64 KA2. CO2 and kWh sum
come from the same solutions (the kWh sum is `E_el + E_gas` of the CO2-optimal point of the gas
booking), and they give nearly the same fractions, because the gas heater term dominates in this
configuration; the weighting matters again in the electric-heater sensitivity (M5).

**Reading.** With the margin that the calibrated ensemble band of Befund 65 FA2 gives, the CO2
cost is about 1.1-1.3 % (median, maximum 1.8 %), and 85 % (main case) to 94 % (stress case) of the
benefit of a per-instance optimum over a fixed operating point remains. The stress-case value
agrees in size with a linear interpolation of the Befund 65 table at this margin (about 94 %,
derived in 2026-10-02 above), but is not the same quantity (that table is for the electric gross
power); the agreement is a plausibility check, not a confirmation. In the main case the benefit is
smaller to begin with (the room for control is about 40 % smaller at `L_q` = 0.15, S0.5) and the
same margin takes a larger share of it. At `L_q` = 0.20 the tightened limit sits 1.0 m below the
validity boundary and the solver still returns valid, converged optima at all 25 points; that case
is the one where the model is closest to its edge, which is why it is reported separately.

**Caveats, to be stated in every use.** One laboratory line, no network, no scaling; the
steady state, not a batch (Befund 53); the margin is the constant median half-width of the band
and is a coverage band, not a guarantee (Befund 65 FA2: not usable per operating point); the
optimiser budget is exhausted by design as in Befund 52 (DE + Nelder-Mead, `converged` refers to
the refinement); `eta_gas` 0.80, the CO2 factors and the penalty scaling for the CO2 objective are
settings or placeholders; five-seed or resolution statements do not apply to the deterministic
optimiser, but the DE seed spread was not measured here (planned in M3 on a subset); the electric
heater is reported only through the second booking of each solution, not optimised.

**Gas-velocity bound (read-only check, 2026-10-05; from the committed artifacts, no new run).**
(a) *Origin of 0.80 m/s.* The bound is `actions.gas_velocity_m_per_s.minimum` in all three process
configs and the default of `ActionSpaceConfig`, with the config comment "the lower bound must lie
above u_mf(d_50,max): u_mf(750 um) about 0.24 m/s". It is part of the first commit of both
repositories (`313e455`, 2026-08-25 in the archive checkout; `7384455`, 2026-09-25 in this one,
which starts a fresh history), so there is no separate commit or reasoning for the number. In
`docs/02_prozess_setup.md` it is a placeholder (`PLH`, "Platzhalter") next to the laboratory range
1.26-1.88 m/s (80-120 m3/h at an assumed 0.0177 m2); the statement that the placeholder ranges are
confirmed by the laboratory values covers that the laboratory range lies inside [0.8, 2.5], not the
value 0.8. The laboratory never ran below 1.26 m/s. (b) *Physical lower bound over the contexts.*
In the 100 oracle solutions of M1 (25 contexts, both limits, M0 and M) the minimum fluidisation
velocity at the d50 placeholder (750 um, constant over contexts and actions) is 0.2258-0.2317 m/s
and the terminal velocity 3.46-3.55 m/s; the fluidisation check never binds (0 of 100 points at the
lower fluidisation bound), and at the optimum `u/u_mf` is at least 3.46. So 0.80 m/s is about 3.5
times the physical minimum and lies 0.57 m/s above it. (c) *Consequence.* The 0.80 m/s is a set
bound, not a physical one: at 21 (L_q 0.15, M0) / 22 (M) / 23 (L_q 0.20, both arms) of the 25
points the optimum sits on it (within 1e-3 m/s), the fixed operating point sits at 0.800-0.8005
m/s, and the regime "no constraint binds" of Befund 52 c is the regime in which this box bound
binds. Hence the optimum, the regret of Befund 52, the SH3 result (half of the agent's gap is the
approach to this bound) and every M1 number depend on this placeholder. Where the optimum would lie
without it is not known (the unconstrained optimum was never computed); at a physical bound near
0.23 m/s the optimum is a different problem, and bubbling, mixing and the ideal-mixing assumption of
the model would then be the limit, not the fluidisation window. The d50 placeholder itself has the
unresolved contradiction of Befund 62 (the plant grid failed at the same d50).
*Proposal for M2 onwards (D16):* state the bound as a placeholder, keep 0.80 as the labelled base
case so that M1 and M2 stay comparable, and vary it as a sensitivity over {0.46 m/s (= 2 u_mf at
0.23, a common engineering factor, not from this repository), 0.80, 1.26 m/s (the laboratory
minimum)}: a physical justification cannot fix a number without a fluidisation-quality limit from
the application partner (the factor 2 is an assumption). Cost, by the M1 measurements: the M1 arms
take about 1 h per limit and bound (oracle 17 min x 2, fixed-point search 18 min); two additional
bounds at the main limit 0.15 are about 2 h, at both limits about 4 h.

**Next.** M2 (one line plus the fixed external source and sink, scaled; arms F, F+R, S, M), after the
release of the project lead and after the decision on D16. The branch `m1` is merged into `master`
(merge commit); the lab-book entry is Befund 69.

**Decision on the gas-velocity bound (project lead, 2026-10-05).** The lower bound 0.80 m/s is a placeholder without derivation; the laboratory runs 1.26-1.88 m/s, the optimum sits on 0.80 at 22 of 25 points, i.e. below the laboratory range, where the model extrapolates. Regret, fixed operating point and M1 hold "at lower bound 0.80 (placeholder)" (addendum to Befund 52). M2 starts only after the sensitivity of M1 at 1.26 m/s has been reported and after the release of the project lead.

**M1 sensitivity to the gas-velocity lower bound (2026-10-05, D16 first part).** M1 (line A, gas heater, limit 0.15, `m` = 0.00596) re-run with the lower bound 1.26 m/s (laboratory minimum) as a run parameter; the config default 0.80 is unchanged. Artifacts `m1_margin_cost_l015_umin126`, `m1_gas_bound_sensitivity`; details and the table in the addendum to Befund 69. Result by the stability rule fixed beforehand: **partly stable**. At 1.26 the optimum sits on the bound at 25 of 25 points (21-22 at 0.80) and the moisture limit binds at 7 instead of 22 points; regret of the fixed point against the oracle (CO2, median) 19.8 % instead of 8.3 % (direction stable, size not), cost of the margin 0.0 % instead of 1.3 % in the median over all points (1.3-1.6 % where the limit binds), remaining control benefit 100 % / 96.9 % (median / pooled) instead of 85 % (stable). Consequence: every M1 and M2 table carries the bound in its heading; the saving of an adapted operating point and the cost of the margin are given for both bounds, not as one number. The third value (a multiple of `u_mf`) is not computed until the factor is clarified with process expertise. M2 only after the release of the project lead.

**Change of D16 (project lead, 2026-10-05).** M2 onwards: main case lower gas-velocity bound 1.26 m/s, 0.80 m/s as the comparison to Befund 52 to 69; both named in every heading. Reason: 1.26 m/s is the only value that rests on measurements; at 0.80 the model extrapolates below the laboratory range. The sensitivity of M1 (addendum to Befund 69) showed the result is partly stable against the bound (S3 stable; S1, S2 not), and the structural finding (second addendum) that the bound sets the optimum at (nearly) all points makes an implicit bound unacceptable. M2 is released for these two bounds (project lead, 2026-10-05); run-time estimate in the M2 section.

## M2 results (2026-10-05) -- at lower bound 1.26 m/s (main case) and 0.80 m/s (comparison)

Lab-book entry: Befund 70; artifacts `m2_site_network_l015_umin126`, `m2_site_network_l015_umin080`, `m2_gas_bound_comparison` (the failed first configuration is kept as `m2_attempt1_penalty_unscaled_l015_umin126`). Single line A, scaled site with the fixed external source Q1 and sink S1, arms F, F+R, S, M, 25 ambient points, limit 0.15, margin m. Direct DE instead of tables (single line; tables and the check step belong to M3), nozzle pressure still a search variable, F taken from M1 (flagged in Befund 70).
**Findings (median over the 25 points, relative to F, CO2 attributable to the line / of the site):** routing effect -8.4 % / -0.71 % at 1.26 m/s and -3.1 % / -0.23 % at 0.80 m/s; set-point effect -13.8 % / -1.07 % and -6.9 % / -0.48 %; joint optimum beyond the line optimum 0 within the optimiser resolution (|relative difference| below 2e-4); total -20.2 % / -1.71 % and -9.8 % / -0.69 %. The best allocation never uses line exhaust to line inlet (LA -> SA, 0 of 25 at both bounds): the line's exhaust goes to the external sink, the external exhaust preheats the line's inlet. By the stability rule: same direction, not stable in size (two to three times larger at 1.26 m/s), so the bound stands in every heading and both columns are given. For M3: the joint effect being zero at one line says the line's set points can be optimised without the network; whether that holds with two lines coupled through the same network is open and is what M3 tests; the check step (table against direct) is still to be run there.

