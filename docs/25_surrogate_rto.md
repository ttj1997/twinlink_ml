# 25 -- Surrogate-based steady RTO with few true evaluations

Status: run and evaluated (sections 7-9; numbers only from the artifacts named there). Reading:
**S2 for both problems**, adaptive design decisive. Section 10: post-hoc declared variant M4b
(added 2026-10-22, not part of the reading). Branch `surrogate-rto`; the finding number is
assigned by the managing session (lab book: "Befund NN").

## 1. Question and decision rule

How many evaluations of the TRUE steady model does a surrogate need so that the optimisation on
it hits the true energy optimum **feasibly**? Result format: learning curve over the number of true
evaluations `N in {8, 16, 32, 64, 128, 256}`; metric: regret at the true model, not the prediction
error.

The reading is fixed in the docstring of `scripts/analyze_surrogate_rto.py`, committed before the
first run, separately per problem. A variant (model and kappa) succeeds at N when the median over
5 seeds of the captured share `eta >= 0.9` and the median infeasibility rate `<= 5 %`; `N_min` is
the smallest budget from which on it succeeds at every larger budget.

| case | condition | consequence |
|---|---|---|
| S1 | some variant has `N_min <= 32` | steady RTO with a surrogate is data-efficient |
| S2 | smallest `N_min` in 64-256 | feasible, but costly with an expensive Dyssol model; adaptive design decisive if it lowers N by at least a factor of 2 |
| S3 | no variant succeeds at N = 256 | physics-based RTO needed |

Additionally: from which N the grey-box fit M4 is overtaken by a black-box model (paired t-test
over seeds, `twinlink_ml.evaluation.paired_resolution`, metric `1 - eta_c`). **The error floor of
M4 and the overtake point hold for the shift of the hidden parameter of this study, not in
general.**

`eta >= 0.9` is exactly `pooled regret <= 0.1 H`, with `H` the headroom `E0 -> E*` (section 4);
the reading is also given in regret percent, and regret and `eta` always side by side (with a small
headroom `eta` reacts strongly to small regrets).

## 2. True models (read only)

| | P1 | P2 |
|---|---|---|
| model | line heater -> dryer -> recovery (`flowsheet.line.Line`), physical mode: counter-current, `dT_min` per exchanger, saturation limit on, unpressurised medium <= 95 degC | steady grey-box model of the lab line, solver of the RTO R1 (docs/24): unchanged physics via `headroom.symbolic.symbolic_derivative`, IPOPT |
| decisions `u` | inlet temperature 60-120 degC (upper bound = `T_max`), dry-air flow 30-200 kg/h, medium flow 10-150 kg/h | inlet temperature 60-120 degC, gas velocity 0.80-2.5 m/s (box of R1) |
| disturbances `w` | ambient temperature 5-35 degC, ambient RELATIVE humidity 0.25-0.85, spray rate 1.2-1.8 kg/h (1.5 +- 20 %), solid fraction 0.20-0.40 | same four, same ranges |
| energy `E` | final energy, kW: chemical power of the corrected fuel + fans + pump (`flowsheet.auxiliary` placeholders) | total power of the model, kW (the objective of R1) |
| constraints `g <= 0` | dew-point margin of the DRYER exhaust >= 10 K; fluidisation: bed volume flow >= 51 m3/h; product (= exhaust) temperature <= 80 degC; medium <= 95 degC | product moisture <= 0.15 kg/kg; bed temperature <= 80 degC |
| side metrics (reported, not optimised) | CO2 (0.366 electricity, 0.201 gas), condensate in the recovery exchanger | CO2 (gas-fired heater, booking of docs/24) |

Operating ranges are lab-line ranges, **not** the 200-300 degC of the partner data sets. All values
are placeholders of `conf/surrogate_opt/surrogate_rto.yaml` unless a source is named there:

* P1 material: the synthetic material of `tests/test_flowsheet_validators.py` (no source).
* P1 fluidisation minimum 51 m3/h = 0.8 m/s x 0.0177 m2, the lab `u_min` and cross-section (both
  placeholders); product limit 80 degC = the bed-temperature limit of the lab line (Befund 52
  placeholder); dew-point margin 10 K at the dryer exhaust (ducts and filter between dryer and
  recovery exchanger; condensation inside the recovery exchanger stays allowed and is reported).
* P1 medium limit: the physical mode caps the medium at 95 degC by construction, so this
  constraint cannot be violated by the truth; it is kept as a surrogate target as requested.
* P1 `T_max` is the upper bound of the decision box (known exactly, no surrogate target); P2's
  `u_min` 0.80 m/s likewise.
* P1 adds one thing to `Line`: the bed heat loss `UA (T_out - T_amb)` of the dryer, passed to the
  existing `DryerConfig` field (with `UA = 0` it is `Line` unchanged; tested).
* P2 true evaluation at fixed `(u, w)`: Newton on the two balances (several starts; a root counts
  only inside the bounds of the R1 NLP), else nested bisection (both balances monotone, the
  structure of `simulation.steady_state`); pinned against the exact NumPy solver (test).
  Newton fails in about a fifth of the joint box (hot inlet, high gas velocity, very dry bed);
  there the bisection takes about 0.3 s.

## 3. Uncertain parameters and the hidden shift (M4)

The truth uses shifted parameter values; M4 estimates two of them from the N runs, the third
(hidden) stays at its nominal value in M4. The hidden one is the parameter with the largest real
uncertainty (decision of the project lead, 2026-10-22), shifted to the edge of the band its
sensitivity study uses.

| problem | parameter | nominal | truth | role | source of nominal and shift |
|---|---|---|---|---|---|
| P1 | heater efficiency | 0.80 | 0.85 | fitted | placeholder (synthetic material; shift without source) |
| P1 | dryer heat loss `UA` | 0 W/K | 2 W/K | fitted | partner model none; lab placeholder 2 W/K (docs/11 9.6) |
| P1 | exchanger approach `dT_min` | 10 K | **20 K** | **hidden** | main convention (docs/11 section 11); upper edge of the Befund-90 band 5/10/20 K (`DT_MIN_K` of `scripts/report_heat_recovery_comparison.py`; a test checks the match) |
| P2 | `nu_max` (saturation degree) | 0.60 | 0.45 | fitted | x0.75, plant variant P1 of the A1 transfer study (`simulation.plant_variants`) |
| P2 | bed heat loss `UA` | 2.0 W/K | 2.6 W/K | fitted | x1.30, the same plant variant |
| P2 | GAB monolayer factor | 1.0 | **1.2** | **hidden** | upper edge of the robustness band 0.8-1.2 of docs/24, READ from `conf/headroom/dynamic_headroom.yaml` |

Why these hidden parameters:

* **P1, exchanger approach.** It stands for the unknown exchanger area (no exchanger data exist).
  In docs/11 the approach takes back 0.23 / 0.47 / 0.93 kW of recovery in scenario 1 at 5 / 10 /
  20 K (10.5), against 0.07-0.09 kW for the bed heat loss UA 0 -> 2 W/K (11.3). Its band (5-20 K) is the
  widest in relative terms. The upper edge was taken because it is the edge farther from the
  nominal value and the unsafe direction (the nominal model over-predicts the recovery).
* **P2, GAB monolayer factor.** All five GAB parameters are placeholders without a sorption
  measurement. At both edges of the docs/24 bands, at the nominal point and the 16 corners of the
  disturbance box, the monolayer factor moves the product moisture at the nominal-model optimum by
  up to +0.0023 kg/kg (factor 1.2) and the true optimal energy by up to +0.7 %; the other four by
  at most 0.0010 kg/kg (K constant 0.80) and 0.3 % (probe before the study, scratch). The upper edge
  is the unsafe direction (the nominal optimum becomes infeasible).

## 4. Reference per test state

50 test states: scrambled Sobol over the disturbance box with a fixed seed (20261022), an own
sequence disjoint from every training design (checked at run time and in a test).

* `u*`: P2 the R1 NLP from six starts, cross-checked by the candidate/SLSQP optimiser on the true
  model; P1 the candidate/SLSQP optimiser (2048 candidates, 12 polished starts), checked by an
  independent run (4096, 16, other scramble). `u*` is the better of the two on the truth;
  convergence (starts at the best value, difference of the two runs) is reported.
* `u0` (decision of the project lead, 2026-10-22): the **robust fixed point** -- least energy at
  the nominal disturbance subject to feasibility in all 50 test states (every constraint gets
  exactly the margin it needs; with one active constraint this is R0 of docs/24). **`u0` is
  determined on the 50 test states and therefore knows the disturbance envelope; this favours the
  baseline and makes `eta` conservative.** Secondary: the R0-style point (one normalised margin on
  the constraints active at the nominal optimum, bisected); on P1 it depends on the scales of the
  two active constraints (dew-point margin, fluidisation).

## 5. Models and arms

All models are trained on `(u, w)` jointly, with energy AND every constraint as its own target (no
feasibility classifier). Designs: scrambled Sobol over the joint box, nested in N, one scramble
per seed; seeds 0-4 (design scrambling and model initialisation).

* **M1** quadratic response surface (full quadratic, ridge with the penalty chosen by exact
  leave-one-out error per target). Conservative variant: fixed margin `kappa x` LOO-RMSE per
  constraint.
* **M2** Gaussian process per target, Matern 5/2, ARD, constant x kernel + small white noise
  (conditioning; the truth is noise-free), 2 optimiser restarts. Conservative variant: posterior
  standard deviation.
* **M3** ensemble of 5 MLPs (2 x 32 tanh, L-BFGS, L2 1e-3), multi-output on standardised targets,
  members differ in initialisation. Conservative variant: ensemble standard deviation.
* **M4** grey-box fit: least squares of the two fitted parameters on all outputs (each scaled by
  its spread in the data), from the nominal values; then the optimum of the fitted physics model
  (P2: R1 NLP; P1: candidate/SLSQP). kappa 0 only.
* **M2a** adaptive design (M2 only): the first 8 Sobol points of the seed, then sequential points
  at the predicted optimum (same kappa) of new training disturbance states (own Sobol sequence),
  GP refitted after batches of `min(8, N/2, next budget - N)`; every true evaluation counts
  (a hard budget counter, tested).
* Conservative variants: `g + kappa sigma <= 0`, kappa in {0, 1, 2} (M1-M3, M2a).
* Optimiser on a model at a test state: 512 scrambled Sobol candidates plus the centre, the 4
  best (feasible by energy, else least violating) polished by SLSQP with forward differences.
* Not run: the optional amortised arm `w -> u*`.

## 6. Metrics

Per (problem, model, kappa, N, seed) on the 50 test states at the true model: infeasibility rate
(any constraint beyond its tolerance; P2 moisture 1e-4 kg/kg as docs/24), maximum normalised
violation, pooled and mean regret over the feasible states, `eta` (headroom-weighted over the
feasible states), `eta_c` (an infeasible state counts as zero captured headroom), CO2 and P1
condensate at the points found, seconds per true evaluation against seconds per surrogate
prediction. Median and range over the 5 seeds. `DataSpec.new_study` does not apply (no time-series
data are generated).

## 7. Results

Run `surrogate_rto_v1` (scripts of commit `47114ad`, clean tree; 72 tasks, 4832 s on 14 workers;
artifact `artifacts/runs/surrogate_rto_v1.json.gz`, 4.98 MB). Analysis
`artifacts/runs/surrogate_rto_v1_analysis.json` (commit `60e35d3`, clean tree; a first analysis
run with the run artifact still untracked recorded `git/dirty=True` and was discarded unused).
Figures `docs/figures/surrogate_rto_v1_{P1,P2}_{eta,regret_mean,infeasible}.png`. **Test set: the
50 test states of each problem**; median [min, max] over 5 seeds.

### 7.1 Reference

| | P1 | P2 |
|---|---|---|
| `u*` feasible | 50/50 | 50/50 |
| convergence of `u*` | 12 of 12 polished starts at the best value in every state; second run within 1.2e-11 relative | at least 4 of 6 IPOPT starts at the best value (median 6); candidate optimiser within -1.9e-7 .. -8.7e-8 relative (the IPOPT tolerance; `u*` taken from it in all 50) |
| `u0` (robust) | 109.18 degC, 54.62 kg/h, 10 kg/h | 114.87 degC, 0.80 m/s |
| median `E*` / `E0` | 1.4096 / 1.7205 kW | 1.2123 / 1.4930 kW |
| **headroom `H`** (`E0 -> E*`) | **22.26 %** (states 0.60 .. 52.97 %) | **24.24 %** (states 0.00 .. 61.94 %) |
| `eta >= 0.9` equals pooled regret <= | 2.23 % | 2.42 % |
| R0-style `u0` | 65.54 degC, 137.61 kg/h, 10 kg/h (margin 1.52 on dew point and fluidisation), headroom 70.35 % | 114.85 degC, 0.80 m/s (margin 0.0753 kg/kg on the moisture), headroom 24.22 % |
| median CO2 at `u*` / `u0` | 0.3028 / 0.3651 kg/h | 0.3062 / 0.3717 kg/h |

`u0` is determined on the 50 test states and knows their disturbance envelope; this favours the
baseline, `eta` is conservative. On P2 the robust and the R0-style point coincide; on P1 the shared
margin of the R0-style point would triple the headroom.

**P1 structure at the optimum.** In all 50 states the dew-point margin at the dryer exhaust and the
fluidisation minimum are active, the product and medium limits are not. **The true optimum recovers
no heat** (0.0 kW in every state, medium flow at its lower bound 10 kg/h): with the true approach of
20 K per exchanger the exhaust at dew point + 10 K is too cold for the circuit. The condensate in the
recovery exchanger is therefore 0 at every `u*` and in the median of every variant; `u0` recovers up
to 0.23 kW. The third decision (medium flow) does not matter at the true optimum.

### 7.2 Learning curves and reading

Cells: `eta` median; infeasibility rate median; pooled regret median. Selected variants; the full
tables (with ranges) are in the analysis artifact and the figures.

P1:

| variant | N = 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| M2a, kappa 2 | 0.875; 0.04; 2.8 % | **0.985 [0.964, 0.987]; 0.02; 0.3 %** | 0.994; 0.00; 0.1 % | 0.998; 0.00; 0.1 % |
| M2a, kappa 1 | 0.950; 0.24; 1.1 % | 0.990; 0.12; 0.2 % | **0.997; 0.04; 0.1 %** | 0.999; 0.00; 0.0 % |
| M2, kappa 0 | 0.766; 0.88; 6.3 % | 0.811; 0.76; 6.4 % | 0.888; 0.70; 3.7 % | 0.922; 0.62; 2.2 % |
| M2, kappa 2 | 0.625; 0.26; 8.3 % | 0.724; 0.36; 7.6 % | 0.823; 0.16; 4.2 % | 0.859; 0.16; 3.3 % |
| M3, kappa 2 | 0.563; 0.32; 10.9 % | 0.636; 0.36; 8.7 % | 0.782; 0.14; 4.7 % | 0.858; 0.02; 3.2 % |
| M1, kappa 2 | -2.294; 0.06; 73 % | 0.183; 0.04; 18 % | 0.516; 0.06; 11 % | 0.531; 0.04; 11 % |
| M4 | 0.877; 0.00; 2.7 % | 0.864; 0.00; 3.0 % | 0.880; 0.00; 2.7 % | 0.878; 0.00; 2.7 % |

P2:

| variant | N = 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| M2a, kappa 2 | 0.158; 0.10; 23 % | 0.613; 0.02; 10 % | 0.844; 0.00; 3.8 % | **0.918 [0.639, 0.981]; 0.00; 2.0 %** |
| M2a, kappa 1 | 0.267; 0.12; 20 % | 0.660; 0.10; 9.1 % | 0.798; 0.08; 5.1 % | **0.931 [0.663, 0.980]; 0.04; 1.6 %** |
| M2, kappa 2 | 0.440; 0.22; 17 % | 0.370; 0.24; 15 % | 0.417; 0.08; 11 % | 0.640; 0.34; 9.4 % |
| M3, kappa 0 | 0.776; 0.70; 6.4 % | 0.895; 0.64; 2.4 % | 0.943; 0.88; 1.6 % | 0.959; 0.86; 1.0 % |
| M3, kappa 2 | 0.700; 0.50; 10 % | 0.837; 0.48; 4.3 % | 0.910; 0.46; 2.0 % | 0.928; 0.40; 1.6 % |
| M1, kappa 1 | -1.493; 0.02; 60 % | -1.136; 0.00; 52 % | -0.732; 0.00; 42 % | -0.824; 0.00; 44 % |
| M4 | --; 1.00; -- | --; 1.00; -- | --; 1.00; -- | --; 1.00; -- |

`eta` and regret are over the feasible states only; where almost every state is infeasible they rest
on a handful of states (e.g. M1 kappa 0 on P1, 96-100 % infeasible).

**Pre-registered reading: S2 for both problems.** P1: smallest `N_min` 64 (M2a, kappa 2), M2a kappa 1
from 128; P2: `N_min` 256 (M2a, kappa 1 and 2). No Sobol variant (M1-M4) succeeds at any N on either
problem, so **adaptive design is decisive** in the pre-registered sense. In regret percent: on P1 the
best variant is at 0.3 % pooled regret with 2 % infeasible states from N = 64 (gate 2.23 %), on P2 at
1.6-2.0 % with 0-4 % infeasible states only at N = 256 (gate 2.42 %).

Why the Sobol models fail the gate: they land on the wrong side of either energy or feasibility.
Without a margin (kappa 0) M2 and M3 reach small regrets on the feasible states but are infeasible in
48-86 % of the states even at N = 256 (P2 M3: 1.0 % regret, 86 % infeasible); with kappa 2 the
infeasibility falls (P1 M3: 2 %) but `eta` stays below 0.9 (0.858). M1's quadratic cannot represent
the switches of the active set; its LOO margin makes it feasible on P2 from N = 64 at an energy far
above `u0` (`eta` -0.82).

### 7.3 Adaptive against Sobol (paired t-test over the 5 seeds, metric `1 - eta_c`)

P1: M2a better than M2 at the same kappa, resolved, at N = 32-256 for kappa 2 (p 0.004, < 0.001,
0.001, < 0.001), at 32, 128, 256 for kappa 0 and at 32, 64 for kappa 1. P2: resolved only at
N = 256 (all kappas, p 0.028 / 0.001 / 0.001) and 128 (kappa 1, p 0.027); at N = 16-32 the adaptive
arm is worse in the mean (not resolved): with 8 start points the GP's predicted optima are poor and
the sequential points cluster where it is wrong. 18 comparisons per problem, no multiplicity
correction.

### 7.4 M4 against the black-box models

**This holds for the hidden shift of this study (P1 exchanger approach 10 -> 20 K; P2 GAB monolayer
factor 1.0 -> 1.2), not in general.**

* **P1:** `eta` 0.86-0.89 from N = 8 on, flat (error floor), feasible in every state from N = 16;
  pooled regret 2.5-3.2 %, just above the gate. The fit does not find the true parameters but
  compensates the hidden approach (seed 0: heater efficiency 0.72-0.74 instead of 0.85, heat loss
  3.5-4.2 instead of 2.0 W/K). **Overtaken first at N = 64, by M2a kappa 2 only** (stays overtaken
  up to 256); no Sobol black-box variant is resolved better than M4 up to N = 256.
* **P2:** the two fitted parameters are recovered almost exactly (seed 0: `nu_max` 0.444-0.450
  against 0.45, `UA` 2.60-2.66 against 2.6 W/K) and the energy is almost optimal (pooled regret
  0.1-0.2 % where feasible), but the hidden GAB shift leaves the product **0.0011-0.0013 kg/kg
  (median, seed 0) above the limit**, at most 0.0019 kg/kg: infeasible in every state of every seed
  from N = 32 (N = 16: median 1.0, one seed feasible; tolerance 1e-4 kg/kg). M4 has no kappa variant (as requested). Overtaken first at N = 32 (M2 and
  M3 at every kappa) and stays overtaken -- **by feasibility, not by energy** (`eta_c` of M4 is 0).
* 72 paired tests per problem, no multiplicity correction.

### 7.5 Effect of kappa

The margin buys feasibility with energy. P1, N = 256: M2 infeasible 0.62 / 0.34 / 0.16 at kappa
0 / 1 / 2, `eta` 0.922 / 0.899 / 0.859; M3 0.48 / 0.08 / 0.02, `eta` 0.907 / 0.898 / 0.858. M1's fixed
margin (LOO RMSE) is either too small (P1 kappa 1: 58 % infeasible at N = 256) or far too large (P2:
`eta` -0.82 at kappa 1, -3.55 at kappa 2). With adaptive design kappa 1-2 costs little energy once
the GP is accurate near the optimum (P1, N = 256: `eta` 0.999 / 0.998).

### 7.6 Maximum violation

P1: normalised by 10 K, the dew-point violation saturates at 1.0 (exhaust saturated, the dryer's
saturation limit binds): the typical failure of the black-box models is an inlet temperature too
low for the air flow. P2: maximum violations up to about 2 kg/kg of moisture (normalised 100-200)
where a model picks a cold, slow point at which the true model's bed floods (far above the critical
moisture 0.2 kg/kg, outside the model's plausible range); there the size of the violation is no
physical statement, the infeasibility rate is the measure.

### 7.7 Run time

Median seconds per true evaluation: P1 5.98e-3 (line with counter-current circuit), P2 2.17e-2
(Newton, bisection fallback in the hot corner). Surrogate prediction, one point / per point in a
batch of 512 (P1): M1 7.5e-5 / 3.2e-7 s, M2 3.7e-3 / 3.4e-5 s, M3 1.8e-3 / 5.6e-6 s. Fit (median):
M2 2.9 s, M3 2.5 s, M4 2.0 s (P1) and 75 s (P2, one CasADi build per parameter value). Both true
models here are cheap; what an expensive Dyssol model would cost is the number of true evaluations,
the axis of the learning curves.

## 8. Reading and consequences

* **S2 on both problems: steady RTO with a surrogate is feasible but not data-efficient** -- 64 (P1)
  and 256 (P2) true evaluations, and only with adaptive design and a GP margin of kappa 1-2. With an
  expensive Dyssol model this is costly; adaptive design is decisive (no space-filling design passes
  the gate up to 256).
* **M4** reaches a usable operating point from 8 runs on both problems but stops at an error floor
  set by the hidden parameter: on P1 just above the gate (2.5-3.2 % regret), on P2 a systematic
  moisture violation of about 0.001-0.002 kg/kg (at most 0.0019). A back-off of that size on M4 might pass; not
  tested here (no post-hoc variant).
* P1's true optimum uses no heat recovery at the true approach of 20 K: whether this line recovers
  heat at all hinges on the exchanger approach (docs/11), not on the surrogate.

## 9. Caveats

* Model against model, no plant data; all ranges and limits are placeholders (sections 2, 3).
* The hidden shifts are one value at the band edge per problem; error floor and overtake point of M4
  hold for them only.
* `u0` knows the test envelope (conservative `eta`); with a small headroom `eta` would be sensitive,
  hence regret and `eta` side by side.
* The P1 medium limit cannot bind (capped by the physical mode); P1's third decision is irrelevant
  at the true optimum.
* The optional amortised arm (`w -> u*`) was not run.

## 10. Post hoc, declared: M4b, M4 with a back-off from the training residuals (added 2026-10-22)

**Not part of the pre-registered reading; S2 (section 8) stays the main result.** Added after run v1
had shown M4 infeasible on P2. Rule (decision of the project lead, fixed in the docstring of
`scripts/analyze_surrogate_rto.py` and committed in `ba840dd` before the run): M4 unchanged, plus
per constraint a back-off `b_N = kappa * RMS(g_true - g_model)` over the N training runs at the
fitted parameters, `kappa in {1, 2}`; the fitted model is optimised with `g_model + b_N <= 0`. Only
the training runs enter `b_N`. Both problems, every constraint, same N, seeds, test states and the
same gate per kappa. Run `surrogate_rto_v2` (`--arms M4b`, commit `ba840dd`, clean tree, 10 tasks,
3377 s on 10 workers; own artifact `artifacts/runs/surrogate_rto_v2.json.gz`, 1.26 MB); analysis of
v1 and v2 together `artifacts/runs/surrogate_rto_v2_analysis.json` (commit `d85dbda`, clean tree;
the main reading, the overtake points and every v1 number are unchanged). Kappa 0 of the arm
reproduces M4 of v1 exactly (largest difference of `eta_c` 0.0). Figures with M4b:
`docs/figures/surrogate_rto_v2_{P1,P2}_{eta,regret_mean,infeasible}.png` (M4's `eta` line ends where
it becomes infeasible in every state). **Holds for the hidden shift of this study only.**

`eta`; infeasibility rate; pooled regret (median [min, max] over seeds where given):

| problem, variant | N = 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| P1 M4 | 0.855; 0.00; 3.2 % | 0.887; 0.00; 2.5 % | 0.877; 0.00; 2.8 % | 0.864; 0.00; 3.0 % | 0.880; 0.00; 2.7 % | 0.878; 0.00; 2.7 % |
| P1 M4b kappa 1 | 0.696; 0.00; 6.8 % | 0.735; 0.00; 5.9 % | 0.725; 0.00; 6.1 % | 0.711; 0.00; 6.4 % | 0.711; 0.00; 6.4 % | 0.738; 0.00; 5.8 % |
| P1 M4b kappa 2 | 0.526; 0.00; 10.5 % | 0.591; 0.00; 9.1 % | 0.570; 0.00; 9.6 % | 0.549; 0.00; 10.1 % | 0.581; 0.00; 9.3 % | 0.593; 0.00; 9.1 % |
| P2 M4 | 0.995; 0.10; 0.13 % | 0.993; 1.00; 0.17 % | --; 1.00; -- | --; 1.00; -- | --; 1.00; -- | --; 1.00; -- |
| P2 M4b kappa 1 | **0.977 [0.949, 0.995]; 0.00; 0.55 %** | 0.986; 0.00; 0.34 % | 0.989; 0.00; 0.27 % | 0.989; 0.00; 0.27 % | 0.989; 0.00; 0.26 % | 0.989 [0.989, 0.990]; 0.00; 0.26 % |
| P2 M4b kappa 2 | **0.959 [0.934, 0.976]; 0.00; 0.99 %** | 0.962; 0.00; 0.92 % | 0.965; 0.00; 0.86 % | 0.965; 0.00; 0.85 % | 0.966; 0.00; 0.83 % | 0.966 [0.965, 0.966]; 0.00; 0.83 % |

(P1 M4b kappa 1 at N = 8: one seed 26 % infeasible; every other M4b cell has 0 % in every seed.)

**Gate applied to M4b alone (post hoc):** P2: `N_min` 8 for kappa 1 and 2 -- what the reading would
call S1 for this variant. P1: no N passes (`eta` 0.53-0.74) -- S3 for this variant. Paired against
M4 (`1 - eta_c`): P2 better at N = 16-256 (p 0.019, 0.020 at N = 16; < 0.001 from 32), not resolved at
8; P1 worse at N = 16-256 (p <= 0.001), not resolved at 8.

**`b_N` against the actual deviation `g_true - g_model` at M4b's own decisions on the test states**
(descriptive; median over seeds of `b_N` and of the median / maximum over the 50 states):

* **P2, moisture (kg/kg):** kappa 1: `b_N` 0.0019 (N = 8), 0.0022 (16), 0.0023 (32-256); actual
  deviation median 0.0010-0.0012 from N = 16 (-0.0004 at N = 8), maximum 0.0019. `b_N` covers the
  deviation in every state (share above `b_N`: 0). Kappa 2: `b_N` 0.0038-0.0047, about twice what is
  needed. Without back-off (kappa 0) the deviation is the same size (median 0.0012, maximum 0.0018 from
  N = 16), i.e. the infeasibility of M4 in v1 is this model error of about 0.001-0.002 kg/kg at the
  limit, and the training RMS of kappa 1 happens to be of that size. Bed temperature: kappa 1 `b_N`
  0.018-0.068 K against a maximum deviation of 0.022-0.056 K (exceeded in up to 20 % of the states);
  the bed limit is inactive (about 30 K away), so this has no effect.
* **P1:** at the two active constraints the truth is on the SAFE side of the fitted model: the
  dew-point margin and the fluidisation deviate by -1.8 to -2.2 K and -0.29 to -0.36 m3/h (median
  over the states, kappa 1), while `b_N` (kappa 1) is +1.8 to +2.2 K and +0.62 to +0.74 m3/h. The RMS is blind to the sign: the
  back-off tightens constraints the model already gets wrong in the safe direction and costs only
  energy (regret 5.8-6.8 % at kappa 1 against 2.5-3.2 % for M4). The product-temperature deviation
  (+1.6 to +2.0 K) exceeds `b_N` in about half of the states, and the medium deviation (+10 K, the
  hidden approach) in 56 %, but both limits are inactive (about 30 K away).

**Reading of the addendum (post hoc, no claim beyond it):** a symmetric back-off from the training
RMS repairs M4 where its error at the limit is in the unsafe direction and of the size of the
training residual (P2, moisture), and harms it where the error is in the safe direction (P1). Whether
that holds for other hidden shifts is open. As declared, an underestimate or overestimate of `b_N`
is a result here and no reason for a further variant.
