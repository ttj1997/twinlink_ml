# NN -- Operating point and heat recovery: jointly or one after the other?

*Number NN: assigned by the managing session on merge. Branch `joint-recovery` (from `origin/master`
a84b764). Sections 1-6 were written before the main run and are not changed afterwards; results
from section 7 on.*

## 1. Question and use

Befund 107 found on P1 (line heater -> dryer -> recovery) that the true energy optimum never uses
the heat recovery (0.0 kW recovered in all 50 test states): with the true exchanger approach of
20 K the dryer exhaust at dew point + 10 K is too cold for the circuit. Befund 108 showed that on
this line the recovered heat depends on the approach alone. Both results were obtained with the
operating point and the medium flow optimised together, at one approach.

Question: **is it worth optimising the operating point of the dryer and the recovery jointly,
compared with the order "first the dryer, then the recovery"?** If the sequential order loses
little, the dryer RTO (Befunde 105-107) can be designed without the recovery and the recovery
set afterwards; if it loses much, the recovery belongs in the RTO problem. Related, in a different
configuration (site coupling M4-R, not this line): Befund 81, where the joint optimum equalled the
sequential solution.

## 2. Model (read, not copied)

P1 of Befund 107: `twinlink_ml.surrogate_opt.line_problem.LineProblem` (read only), the line of
`twinlink_ml.flowsheet.line` in the physical mode: counter-current exchangers with unlimited area,
minimum approach `dT_min` per exchanger, saturation limit on, condensation of the exhaust in the
recovery exchanger allowed (condensing release curve; condensate reported). Operating ranges,
constraints, parameters (truth: heater efficiency 0.85, bed heat loss 2 W/K), material, the
optimiser settings of the true optimum and the **50 test states** (Sobol over the disturbance
box, fixed seed) are read from `conf/surrogate_opt/surrogate_rto.yaml` by the functions of
Befund 107 (`build_problems`, `make_test_states`); nothing is copied. All values there are
placeholders (lab-line ranges, synthetic material).

- Decisions: inlet temperature 60-120 degC, dry air 30-200 kg/h, medium 10-150 kg/h or 0 (bypass).
- Disturbances: ambient temperature 5-35 degC, relative humidity 0.25-0.85, spray rate
  1.2-1.8 kg/h, solid fraction 0.20-0.40.
- Constraints: dew-point margin >= 10 K at the dryer exhaust, fluidisation >= 51 m3/h at bed
  conditions, product (= exhaust) <= 80 degC, medium <= 95 degC (unpressurised only).
- Objective: final energy E (fuel + process fan + exhaust fan + pump), kW; CO2 reported.

Additions of this study (`src/twinlink_ml/joint_recovery/problem.py`, existing modules unchanged):

- **Bypass** (medium flow 0): pump off, no duty, the exhaust leaves the exchanger unchanged. The
  exchangers stay in the gas paths, so both fans see the same pressure drops as with recovery
  (same plant, circuit stagnant); the medium constraint of the bypass is `T_amb - 95 degC`. With a
  medium flow > 0 the line is P1 of Befund 107 bit for bit (tested).
- **Pressurised medium**: no upper limit of the medium temperature (`htm_max_celsius=None`), the
  medium constraint dropped; the saturation pressure of water at the hot medium temperature is
  reported.

## 3. Factors and strategies

Factors: `dT_min` in {5, 10, 20} K (20 K = the truth of Befund 107); medium (i) unpressurised
<= 95 degC (main), (ii) pressurised. Six combinations, each on the same 50 test states.

Strategies per test state (`src/twinlink_ml/joint_recovery/study.py`):

| | operating point (inlet temperature, air) | medium flow |
|---|---|---|
| S0 | optimal without recovery | 0 (bypass) |
| S1 | that of S0 (one after the other) | optimal over {0} and 10-150 kg/h |
| S2 | jointly optimal with the medium flow: `u*` of Befund 107's `reference` (multistart: 2048 candidates + 12 SLSQP polishes, independent check run 4096 + 16, the better of the two), or the bypass at S0's point, whichever is better | |
| S3 | the robust fixed point `u0` of Befund 107's `reference` (least energy at the nominal disturbance, feasible in every test state) | optimal over {0} and 10-150 kg/h |

S0 uses the same two optimiser runs as `u*` of Befund 107 over the two operating decisions;
S1 and S3 a one-dimensional search (256 candidates + 4 polishes, check 512 + 4, the better of the
two and the bypass). "Better" is always: feasible first, then lower energy. `u0` is computed per
combination, because the energy at the nominal disturbance depends on `dT_min`.

## 4. Metrics and pre-registered reading

Metrics per state: E, recovered heat, exhaust temperature before and after the recovery
exchanger, condensate, medium flow, active constraints (`g / scale >= -1e-3`), CO2. Value of the
joint optimisation `V_J = (E(S1) - E(S2)) / E(S1)`; median and range over the 50 states per
`dT_min` and medium variant.

Reading (fixed in the docstring of `scripts/analyze_joint_recovery.py`, committed before the main
run), per `dT_min` and medium variant on the median of `V_J`:
**J0** < 1 %: one after the other suffices; **J1** 1-5 %: the joint optimisation pays measurably;
**J2** > 5 %: operating point and recovery must be optimised jointly. Additionally: from which
`dT_min` (of the three levels) the recovery stays unused at S2 in every state, and in how many
states condensation occurs at S2.

Gates: **G1** the combination 20 K / unpressurised reproduces the P1 reference of
`surrogate_rto_v1` (test states, `u*`, `u0` within 1e-9); **G2** `S2 <= S1 <= S0` and
`S2 <= S3` in every state (relative tolerance 1e-6), if violated the reading is stated also for
`S2* = min(S2, S1, S3)`; **G3** every strategy feasible in every state.

Expectation stated before the run (not a reading): the medium leaves the recovery exchanger at
least `dT_min` below the dryer exhaust, which is limited to 80 degC, so the 95 degC limit cannot
bind on this line and both medium variants should coincide. Checked and reported.

## 5. Tests

`tests/test_joint_recovery.py`: medium flow > 0 equals P1 of Befund 107 bit for bit at three
`dT_min`; the bypass carries no heat, no pump, no condensate; the pressurised variant drops the
medium constraint; S0 does not depend on `dT_min` or the medium; the test states are those of
Befund 107; `S2 <= S1 <= S0` and `S2 <= S3` at a cold state for 5 and 10 K; for `dT_min` 200 K
`S1 = S2 = S0` exactly; the reading boundaries.

## 6. Run and budget

Driver `scripts/run_joint_recovery.py` (commit before the run, clean tree, at most 6 workers;
other studies run on the machine at the same time). Phase 1: six `reference` tasks (one per
combination) and 50 S0 tasks; phase 2: 300 tasks for S1 and S3. Estimate from the probe and the
smoke run (**measured under parallel load**): S0 about 6 s per state, the medium search 1-5 s,
`reference` about 15 min per combination (Befund 107: 936 s on P1); in total well under an hour
on 6 workers. Artifact `artifacts/runs/joint_recovery_v1.json.gz` (expected well below 5 MB).
