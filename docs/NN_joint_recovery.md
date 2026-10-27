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

## 7. Results

Run `joint_recovery_v1` (driver commit `798d61b`, clean tree, `git/dirty=False`, `code/package_dir`
`src/twinlink_ml`), artifact `artifacts/runs/joint_recovery_v1.json.gz` (0.59 MB, commit `a792cbd`);
analysis `artifacts/runs/joint_recovery_v1_analysis.json` (run at `a792cbd`, clean tree, commit
`e960286`). **Test set: the 50 test states of P1 (Befund 107).** All figures below are from the
analysis artifact; median [min, max] over the 50 states.

### 7.1 Gates

- **G1 passed.** The combination 20 K / unpressurised reproduces the P1 reference of
  `surrogate_rto_v1` exactly: largest differences of the test states, `u*`, `u0` and the energies
  at `u*` all 0.0.
- **G2 passed** in every state of every combination. Largest relative excess of `E(S2)` over
  `E(S1)`: 3.0e-10 (5 K), 7.5e-11 (10 K), 0 (20 K); `E(S1) <= E(S0)` exactly everywhere;
  `E(S2) < E(S3)` everywhere. The checks on the parts with the medium in its box
  (`S2_on <= S1_on`, `S2_on <= S3_on`) fail nowhere either.
- **G3:** 0 infeasible strategy results of 1200.

### 7.2 Value of the joint optimisation (pre-registered reading)

| dT_min | medium | V_J median | V_J range | reading |
|---|---|---|---|---|
| 5 K | unpressurised | 8.4e-9 | -3.0e-10 .. 1.4e-7 | **J0** |
| 5 K | pressurised | 4.0e-9 | -3.7e-10 .. 5.1e-8 | **J0** |
| 10 K | unpressurised | 2.9e-11 | -7.5e-11 .. 2.5e-7 | **J0** |
| 10 K | pressurised | 1.3e-11 | -4.1e-11 .. 2.6e-7 | **J0** |
| 20 K | unpressurised | 0 | 0 .. 0 | **J0** |
| 20 K | pressurised | 0 | 0 .. 0 | **J0** |

All 50 states are in class J0 in every combination; `V_J` on CO2 is of the same size (at most
2.9e-7). **Reading J0 at every `dT_min` and in both medium variants: optimising the dryer first
and the recovery afterwards suffices on this line.** The joint optimum S2 lies at the operating
point of S0 in every state (inlet temperature 89.08 [67.50, 111.99] degC, air 54.93
[51.88, 57.58] kg/h at S0, S1 and S2); the dew-point margin and the fluidisation are active in all
50 states at S0, S1 and S2. Differences of order 1e-7 and negative values of order 1e-10 are the
tolerance of the optimiser.

### 7.3 Use of the recovery and condensation

Unpressurised; the pressurised variant gives the same counts.

| dT_min | states with recovery used at S2 | recovered heat at S2, kW | exhaust before / after the exchanger at S2, degC | states with condensation at S2 | condensate at S2, kg/h | sequential value `(E(S0) - E(S1))/E(S0)` |
|---|---|---|---|---|---|---|
| 5 K | 47 | 0.163 [0, 0.337] | 40.47 [31.56, 49.39] / 31.35 [19.86, 44.92] | 25 | 0.0022 [0, 0.196] | 13.75 % [0, 24.30 %] |
| 10 K | 28 | 0.0064 [0, 0.180] | 40.47 [31.56, 49.39] / 37.92 [25.65, 49.39] | 3 | 0 [0, 0.021] | 0.48 % [0, 12.27 %] |
| 20 K | 0 | 0 | 40.47 [31.56, 49.39] / unchanged | 0 | 0 | 0 |

**The recovery stays unused at the optimum S2 from 20 K on** (both variants); at 10 K it is used in
28 of 50 states, at 5 K in 47. **Condensation occurs at the optimum S2** at 5 K in 25 states and at
10 K in 3, never at 20 K. Where the recovery is used the optimal medium flow is 13.3-13.9 kg/h
(median 13.42 at 5 K); otherwise the bypass is chosen (3 states at 5 K, 22 at 10 K, 50 at 20 K).
Energy at S2: 1.2095 [0.9692, 1.5102] kW (5 K), 1.3485 [1.0353, 1.6914] kW (10 K), 1.4089
[1.0353, 1.8921] kW (20 K, = S0); CO2 0.2629 / 0.2908 / 0.3026 kg/h (medians).

Descriptive (b'): the geometric bound `(T_exhaust(S0) - T_amb)/2` is 10.20 [4.27, 15.77] K; above it
no heat can pass at S0's operating point, consistent with use in about half the states at 10 K and
in none at 20 K.

### 7.4 Robust fixed point (S3)

`u0` (inlet temperature 109.18 degC, air 54.62 kg/h) is the same at every `dT_min`; its medium flow
is 13.31 kg/h at 5 and 10 K and 10 kg/h at 20 K (= Befund 107). At `u0` the exhaust is hotter
(56.83 [41.78, 68.72] degC) and the recovery carries more heat (5 K: 0.406 [0.108, 0.769] kW, used
in all 50 states; 10 K: 0.251 kW median, 49 states; 20 K: 0 [0, 0.309] kW, 19 states), and yet S3
needs more energy than S2 in every state. Price of the robust point `(E(S3) - E(S2))/E(S2)`:
4.09 % [0.12, 9.55 %] at 5 K, 4.20 % [0.08, 27.32 %] at 10 K, 21.01 % [0.60, 53.01 %] at 20 K. A
small approach lets the recovery win back most of what the robust margin costs; it does not make
the hotter operating point better than the optimal one.

### 7.5 Medium variants

As expected before the run, the variants coincide: the largest energy difference between
unpressurised and pressurised per strategy is 8.4e-7 kW (S3 at 5 K; optimiser tolerance), for S0
and S1 0 to 3e-11 kW. The hottest medium of any strategy is 63.72 degC (S3, 5 K); the 95 degC
limit is never approached. On this lab line the medium factor has no effect.

### 7.6 Run time (measured under parallel load)

Wall time 3309 s on 6 workers, sum of the task times 14 545 s, with other studies running on the
machine at the same time. `reference` per combination: 875-890 s at 20 K, 2020-2091 s at 10 K,
2658-2712 s at 5 K; S0 about 7 s per state; the medium search a few seconds per state and
strategy. The times were not re-measured on a free machine; they enter no result.

## 8. Reading

Pre-registered reading **J0 in all six combinations** (G1-G3 passed): on P1 of Befund 107 the
joint optimisation of operating point and medium flow gives nothing over the order "first the
dryer, then the recovery" (`V_J` at most 2.6e-7). The joint optimum never moves the dryer away
from its own optimum, where the dew-point margin and the fluidisation are active; the recovery
then uses what that exhaust offers. The minimum approach decides **how much** the recovery gives
(median 13.75 % at 5 K, 0.48 % at 10 K, 0 at 20 K), not **whether** the operating point must be
chosen with it. The side result of Befund 107 (the true optimum never recovers heat) is the 20 K
case of this picture; it is not caused by optimising the decisions together.

Interpretation (not a tested claim): moving the dryer to a hotter exhaust to feed the recovery
costs more fuel for the whole air flow than the recovery returns through the approach; S3 shows
this directly (hotter exhaust, more recovered heat, more energy).

## 9. Limitations and decisions taken without asking

**Limitations.** Model against model (P1, synthetic material, lab-line placeholder ranges and
limits, `UA` and heater efficiency of the P1 truth); one line with one exhaust and one sink;
unlimited exchanger area (only the approach limits the duty); three `dT_min` levels only (the
threshold between 10 and 20 K is not located; the geometric bound is descriptive). The dryer
constraints active here (dew-point margin 10 K, fluidisation 51 m3/h) are placeholders; with other
constraints or a high-temperature operating range (partner data sets, 200-300 degC) the medium
limit and the coupling may matter (Befund 108: there the 95 degC limit binds in every sample).
Steady state only.

**Decisions taken without asking.**

1. The "no recovery" option is a bypass of the same plant: pump off, exchangers stay in the gas
   paths with their pressure drops (fans unchanged). This makes S1 = S0 exact when the recovery
   carries nothing; a plant without exchangers would have slightly lower fan power in S0.
2. S1, S2 and S3 may choose the bypass (medium flow 0) besides the box 10-150 kg/h; without it a
   strategy with recovery could never equal S0 (the pump at the lower bound costs energy), and the
   requested relation `S1 = S2 = S0` at a very large `dT_min` could not hold.
3. S2 is the `u*` of Befund 107's `reference` (unchanged code), not a new optimiser; S3 uses its
   `u0`, recomputed per combination.
4. Thresholds: recovery "used" above 1 W, condensation above 1 mg/h; one-dimensional medium search
   with 256 + 4 and 512 + 4 starts.
5. The reading is applied to both medium variants separately (six verdicts), unpressurised as the
   main variant.
6. The medium constraint of the bypass is `T_amb - 95 degC` (stagnant medium at ambient); it never
   binds.
