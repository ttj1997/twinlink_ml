# 25 -- Surrogate-based steady RTO with few true evaluations

Status: design (sections 1-6); results in section 7 once run. Branch `surrogate-rto`; the finding
number is assigned by the managing session (lab book: "Befund NN").

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
