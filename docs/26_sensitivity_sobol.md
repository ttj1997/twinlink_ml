# 25 -- Global sensitivity analysis (Sobol indices) of the steady-state models

Status: run and evaluated. Design fixed before the run (sections 1-6, commit `ac7befe`); results in
sections 7-9 (numbers only from `artifacts/runs/sobol_steady_v1.json`). Branch `sensitivity`;
finding number open (lab book "Befund NN", assigned by the managing session).

## 1. Question and use

Which unknown model parameters decide the energy, the optimal operating point and the saving of
the RTO, and which of them are worth a data request to the application partner or a laboratory
measurement? Answer: a ranking of the epistemic parameters by their total Sobol index for the
energy at the steady optimum (b) and the optimal decision (c), mapped to the open questions
(table "which question is worth how much", section 8).

## 2. Models, inputs, ranges

Both models are only read (`src/twinlink_ml/sensitivity/models.py` wraps them):

* **P1** -- one line heater -> dryer -> heat recovery (`flowsheet.line.Line`, Befund 106) in the
  physical mode (counter-current exchangers with a minimum approach, saturation limit), at data set
  5 of delivery 4.1_2 (Combined; 150 kg/h dry gas at 250 degC, 10 kg/h spray at 0.5 solid fraction,
  30 kg/h medium; properties of 4.1_2). **Main variant: unpressurised medium (<= 95 degC)**, so that
  it matches the parallel surrogate study; the pressurised circuit is reported in addition
  (decision of 2026-10-22). The dryer's heat loss is fixed at 0 inside `Line`; the wrapper builds
  the same dryer unit with the sampled `UA`. **The fan is not modelled in P1**: the energy of P1 is
  heater duty and fuel only.
* **P2** -- the steady-state grey-box model of the lab line of docs/24 (GAB placeholder config,
  nozzle pressure at the Befund-52 fixed point, quality limit 0.15 kg/kg, lab band of the solid
  fraction around 0.30). Fixed operating point: exact steady state of the project's solver
  (`simulation.steady_state`) at a fixed command; optimum: the RTO solver of docs/24
  (`headroom.controllers.SteadyStateOptimizer`, CasADi/IPOPT), rebuilt per sample and checked
  with the exact solver at the optimal command (validity flags, moisture consistency 1e-4 kg/kg).

Inputs in two groups, independent and uniform on the ranges of
`conf/sensitivity/sobol_steady.yaml` (every range a placeholder with its reason there).
**Correlations are not modelled** (e.g. the GAB parameters of one material, ambient temperature
and humidity, throughput and solid fraction of one product are correlated in reality).

Epistemic (unknown model parameters):

| input | P1 | P2 | nominal | range | reason (short) |
|---|---|---|---|---|---|
| GAB monolayer moisture | - | x | 0.020 | 0.016-0.024 kg/kg | x0.8-1.2, robustness band of docs/24 |
| GAB C at 62 degC | - | x | 25 | 12.5-50 (log-uniform) | x0.5-2, as docs/24 |
| GAB K at 62 degC | - | x | 0.75 | 0.70-0.80 | as docs/24 |
| GAB enthalpy of C | - | x | 6000 | 3000-9000 J/mol | x0.5-1.5 |
| GAB enthalpy of K | - | x | 500 | 250-750 J/mol | x0.5-1.5 |
| nu_max (max. saturation degree) | - | x | 0.60 | 0.40-0.80 | literature band of fluidised beds |
| critical moisture | - | x | 0.200 | 0.150-0.300 kg/kg | from the quality limit to the wettest lab value |
| heater efficiency | x | x (gas, CO2 only) | 0.70 / 0.80 | 0.70-0.90 | partner data sets 70-90 % |
| minimum approach per exchanger | x | - | 10 K | 5-20 K | stands for the unknown exchanger characteristic |
| heat loss UA | x | x | 0 / 2 W/K | 0-3 / 0-4 W/K | P1: up to 5 % of the duty; P2: 0 to twice the placeholder |
| equilibrium moisture of the product | x | (GAB) | 0.002 | 0.002-0.030 kg/kg | partner sheet to lab level |
| cp_vapour | x | x | 1906 / 1860 | 1860-1906 J/(kg K) | repository against partner |

Operative (disturbances): ambient temperature 5-35 degC and relative humidity 0.25-0.85 (weather
box of the lab line; P1: fresh gas, converted with IAPWS-IF97), throughput +-20 % (P1 spray 8-12
kg/h, P2 1.2-1.8 kg/h), solid fraction (P1 0.40-0.60, P2 0.20-0.40). These are the **stress bands**
of docs/24, i.e. the operating envelope (part load, product variants), **not running operation**:
with the observed amplitudes of docs/24 (throughput +-1.5 %, solid fraction +-7.7 %) the two would
be nearly without effect (decision of 2026-10-22).

Model-specific notes. nu_max, the critical moisture and the GAB parameters exist only in P2 (the
dryer block of P1 has no kinetics); the minimum approach exists only in P1 (P2 has no recovery).
In P2 the equilibrium moisture is given by the GAB isotherm (the field `equilibrium_moisture` is
only its anchor), so it is represented by the GAB parameters. The heater efficiency of P2 is the
gas-heater efficiency of the CO2 booking (docs/24); the RTO objective is the model's total power
(electric heater 0.98, as in docs/24), so (b)-(d) cannot depend on it -- a built-in check of the
estimator (its total index must come out zero).

## 3. Outputs

* **(a)** energy and CO2 at the fixed operating point. P1: heater duty, fuel, CO2 (fuel x lower
  heating value x 0.201 kg/kWh: proportional to the fuel, same indices). P2: model total power,
  CO2 (gas-fired heater, carrier booking of docs/24), and the product moisture at the fixed point.
* **(b)** P2: energy (and CO2) at the steady optimum, re-optimised per sample with the RTO solver.
* **(c)** P2: the optimal decision, inlet temperature and gas velocity.
* **(d)** P2: saving fixed point -> optimum, `(E_fixed - E_optimum) / E_fixed`.
* **(e)** P1: recovered heat; dew-point distance of the exhaust after the recovery (its
  temperature minus its dew point).

**Fixed operating point of P2** (decision of 2026-10-22): the steady optimum at the nominal
disturbance and nominal parameters with the R0 margin of the family "observed" (0.04355 kg/kg,
Befund 105), held for every sample; the optimum per sample uses the R1 margin (0.00664). At the
nominal point this gives a steady saving of 9.7 % (probe before the study), the steady counterpart
of the 8.9 % of Befund 105 (median over dynamic scenarios). Samples in which the fixed point
exceeds the quality limit (0.15 + 1e-4 kg/kg) under the varied inputs are **counted and attributed
to the inputs** (share per fifth of each input's range, Sobol indices of the violation indicator
and of the moisture, each input alone at its range ends), and (d) is reported **separately**:
`savings_all` (every valid sample) and `savings_fixed_feasible` (only where the fixed point holds
the limit).

## 4. Method and pre-registered criteria

Saltelli design with SALib 1.6 (extra `sensitivity`; scrambled Sobol' sequence, no second-order
indices: `N (D + 2)` runs), first-order (Saltelli 2010) and total (Jansen) indices, 95 % bootstrap
intervals (1000 resamples). Fixed before the first run (YAML `criteria`, driver docstring):

* **dominant**: total index ST >= 0.1;
* **converged**: the FULL width of the 95 % interval < 0.05 for S1 and for ST of every dominant
  input, at the largest base sample; convergence read at N/4, N/2, N of one design (nested
  prefixes of the Sobol' sequence = doubling; tested);
* **ranking change** (designs J and R against E, epistemic inputs, per output): the ordered first
  K places by ST, K = number of inputs dominant in E, at least 3; reported with the affected inputs
  and whether a swap is resolved (intervals separated in both designs) or within the intervals;
* **range robustness**: the three epistemic inputs with the largest ST in E over the decision
  outputs (P2: (b) energy at the optimum, (c) optimal inlet temperature; P1: fuel, recovered heat,
  dew-point distance) get their range doubled about the centre (in the log for log inputs),
  clipped to physical bounds (efficiency and nu_max <= 1, approach >= 1 K, UA and moisture >= 0,
  K < 0.95, critical moisture >= the quality limit 0.15).

Estimator check: Ishigami function with its analytic indices (test), the prefix property of the
design (test), and the zero index of the gas-heater efficiency for (b)-(d) of P2.

## 5. Invalid samples

Counted per reason and never dropped silently. Reasons: the material check rejects the input
combination (critical moisture below the isotherm maximum on the safety grid); the exact steady
state is outside the model's validity (fluidisation, saturation degree, moisture outside
[X_eq, X_cr], no drying regime, bed temperature at the search bound) or its bisection fails; IPOPT
fails; IPOPT and the exact solver disagree by more than 1e-4 kg/kg at the optimum; P1: the line
raises, the dryer is saturation-limited, or a plausibility check reports a violation. The
Saltelli estimators need complete blocks: an output's indices use only blocks whose `D + 2` rows
are all valid for that output, and the valid share is given with every index. With a large invalid
share the indices describe the input distribution restricted to valid blocks (a conditional
analysis); the same holds for `savings_fixed_feasible` by construction.

**Solver setting found before the study.** In a smoke run (scratch, not an artifact) the exact
solver aborted in about 8 % of the operative samples, also at optimal commands with X = 0.143:
the inner moisture bracket ends at max(10 X_cr, 1) = 2 kg/kg, too narrow at the cold lower end of
the outer temperature bracket under wet operation. The wrapper passes the solver's own lower end
and an upper end of 20 kg/kg (YAML, with the reason); the model is unchanged. Afterwards one
sample of 192 remained invalid (fixed point with X above X_cr, a genuine model limit).

## 6. Designs and budget

| design | varied | P1 base sample | P2 base sample | purpose |
|---|---|---|---|---|
| E | epistemic (operative nominal) | 2^12 | 2^12 | ranking for the data request |
| O | operative (epistemic nominal) | 2^12 | 2^12 | operating envelope |
| J | all | 2^12 | 2^10 (prefix 2^9 reported) | check only: does the ranking of E change? |
| R | E with the top-3 ranges doubled | 2^12 | 2^12 | range robustness |

Cost per run (probe before the study): P1 0.8 ms; P2 about 0.35-0.5 s (rebuild of the IPOPT
problem 8 ms, IPOPT 3 ms, two exact steady states at search precision 30 steps / 1e-7, about
0.19 s each). (b)-(d) are therefore not the expensive part: the exact solver at the fixed and at
the optimal command is. With 14 workers about 25 runs/s, so 2^12 for E, O and R of P2 (about
49 000, 25 000, 49 000 runs) fits in about 1.5-2 h; J of P2 at 2^10 (16 000 runs) as a check
(the request allowed a smaller budget, e.g. 2^9).

Driver: `scripts/run_sensitivity.py` (committed before the run, clean tree, resumable cache).

## 7. Results

Artifact: `artifacts/runs/sobol_steady_v1.json` (driver commit `ac7befe`, clean tree,
`git/dirty=False`; per-run records in the gitignored cache `_sobol_steady_v1_cache.jsonl`). All
numbers below are from this file. "w" = full width of the 95 % interval. Indices of design E are
relative to the variance caused by the epistemic inputs at the nominal operating point.

### 7.1 Convergence (pre-registered criterion)

**The pre-registered criterion is not met at 2^12**: of 80 cases (output x design x variant with
at least one dominant input) 5 meet it, all in design J of P1. The S1 intervals are the binding
part; even an output that depends on one input only (P1 unpressurised, heater duty: S1 = ST =
1.000) has w(S1) = 0.073 and w(ST) = 0.061 at N = 4096. With the criterion on ST alone (not
pre-registered, for orientation only) 19 of 80 would pass. The point estimates do not move with
the doubling, and the widths fall like 1/sqrt(N), e.g. P2 E, (b), critical moisture:

| N | S1 (w) | ST (w) |
|---|---|---|
| 1024 | 0.515 (0.112) | 0.517 (0.085) |
| 2048 | 0.516 (0.077) | 0.518 (0.060) |
| 4096 | 0.516 (0.058) | 0.518 (0.042) |

Reading (post hoc, labelled as such): the ranking of the dominant inputs is resolved at 2^12
wherever it is stated below (intervals of ST separated); the absolute index values carry the
stated widths. Meeting the criterion for indices near 1 would need about 2^14. **Run afterwards
as a declared precision extension of design E (section 7.8): criterion met for 19 of 20 outputs at
2^14.**

### 7.2 P2 (lab line), design E: epistemic inputs at the nominal operating point

| output | mean (std) | dominant inputs, ST (w) |
|---|---|---|
| (a) energy at the fixed point | 1.239 (0.0055) kW | heat loss 0.987 (0.061) |
| (a) CO2 at the fixed point | 0.311 (0.022) kg/h | gas-heater efficiency 0.996 (0.060) |
| (a) moisture at the fixed point | 0.119 (0.026) kg/kg | critical moisture 0.497 (0.045), nu_max 0.422 (0.036) |
| (b) energy at the optimum | 1.156 (0.088) kW | critical moisture 0.518 (0.042), nu_max 0.388 (0.032); heat loss 0.097 (0.009) |
| (b) CO2 at the optimum | 0.290 (0.030) kg/h | gas-heater efficiency 0.467 (0.040), critical moisture 0.277 (0.025), nu_max 0.208 (0.018) |
| (c) optimal inlet temperature | 88.4 (7.67) degC | critical moisture 0.614 (0.075), nu_max 0.446 (0.061), heat loss 0.107 (0.022) |
| (c) optimal gas velocity | 0.808 (0.051) m/s | critical moisture 0.895 (0.236), nu_max 0.716 (0.223), heat loss 0.501 (0.196), GAB monolayer 0.120 (0.098) |
| (d) saving, all valid | 0.067 (0.070) | critical moisture 0.536 (0.043), nu_max 0.401 (0.033) |
| (d) saving, fixed point feasible | 0.093 (0.056) | critical moisture 0.681 (0.067), nu_max 0.415 (0.050) |

* The **drying kinetics decide** (b), (c) and (d): critical moisture and nu_max together carry
  at least 0.90 of the variance of (b) (sum of S1, a lower bound of the pair's closed index), 0.85
  of the inlet temperature (c), 0.93 of (d). Heat loss is third (ST 0.07-0.11).
* **The isotherm hardly matters here**: the five GAB parameters sum to ST <= 0.001 for (b) and
  <= 0.029 for the inlet temperature (c); only the gas velocity (c), which leaves its lower bound
  0.80 m/s in a few samples (std 0.051 m/s), shows the monolayer moisture at ST 0.120 (w 0.098),
  almost entirely as interaction (S1 0.004). At the RTO moisture (about 0.14 kg/kg) the
  equilibrium moisture (about 0.03) is small against the span X_cr - X_eq.
* **cp_vapour**: ST 0.000 for every output (range 1.86-1.906 kJ/(kg K)).
* **Built-in check**: the gas-heater efficiency, absent from the RTO objective, has ST 0.000 for
  (b) energy, (c) and (d); it dominates only the CO2 outputs.
* The optimal gas velocity is a threshold output (mostly at its bound): its indices are
  interaction-dominated (S1 << ST) and the widest of the study.

**(d) and the 8.9 %.** At the nominal point the steady saving is 9.67 %. Over the epistemic
ranges its mean is 6.7 % (std 7.0 %) on all valid samples, 9.3 % (std 5.6 %) where the fixed point
holds the limit; it depends on the drying kinetics (critical moisture ST 0.536, nu_max 0.401),
i.e. on how much moisture margin the R0 margin really buys. In design O (operative inputs, nominal
parameters) it depends on throughput (ST 0.561) and solid fraction (ST 0.411) -- the drying load
-- not on the weather (ST <= 0.081).

### 7.3 P2: where the fixed operating point exceeds the quality limit

In design E the fixed point (inlet temperature 94.97 degC, 0.80 m/s, planned for X = 0.106) exceeds
0.15 kg/kg in **13.1 % of the base samples** (8192 rows A and B; no fixed-point sample invalid in
E). Attribution:

| input | share of violations per fifth of its range (low -> high) | ST of the violation indicator (w) |
|---|---|---|
| nu_max | 0.430, 0.181, 0.046, 0.000, 0.000 | 0.748 (0.093) |
| critical moisture | 0.000, 0.007, 0.073, 0.211, 0.365 | 0.690 (0.097) |
| heat loss UA | 0.046, 0.092, 0.123, 0.165, 0.231 | 0.321 (0.069) |
| GAB, efficiency, cp_vapour | 0.12-0.14 in every fifth (flat) | <= 0.037 |

**No input alone** makes the fixed point infeasible: at its range end with the others nominal the
largest moisture is 0.146 kg/kg (critical moisture 0.30) and 0.141 (nu_max 0.40). Violations come
from combinations of slow drying (low nu_max, high critical moisture) and a large heat loss
(indicator: S1 0.245 / 0.190 / 0.041 against ST 0.748 / 0.690 / 0.321). Operatively (design O) the
throughput alone (1.8 kg/h: X = 0.161) and the solid fraction alone (0.20: X = 0.152) violate it;
there the violation indicator has the mean 0.129 on the valid blocks (A and B rows; valid block
share 0.861), and another 3.8 % of the rows (932) lie beyond the critical moisture (counted
invalid, see 7.6), which also exceeds the limit.

### 7.4 P1 (one line, data set 5), design E

| output | unpressurised (main), ST | pressurised (additional), ST |
|---|---|---|
| (a) heater duty | approach 1.000 | approach 0.925, heat loss 0.071 |
| (a) fuel (= CO2) | efficiency 0.932, approach 0.069 | efficiency 0.662, approach 0.314 |
| (e) recovered heat | approach 1.000 | approach 0.926, heat loss 0.071 |
| (e) dew-point distance | approach 0.644, heat loss 0.337 | approach 0.992 |

Unpressurised, the 95 degC medium limit binds in every sample (28 672 of 28 672): the recovered
heat is then fixed by the gas side, `C_gas (95 - dT_min - T_fresh)`, and depends on nothing but
the approach; the exhaust temperature (heat loss, product moisture) moves only the dew-point
distance. Pressurised, the approach binds in every sample. The equilibrium moisture of the product
and cp_vapour reach at most ST 0.02 (equilibrium moisture, dew-point distance, unpressurised). No invalid sample in P1 (all checks pass; no
saturation-limited dryer). The fan is not modelled in P1.

### 7.5 Ranking checks

**Design J (operative inputs varied as well): does the epistemic ranking of E change? No** for
(b), (c), (d) of P2 and for every P1 output -- also at the prefix 2^9 of P2. The only change: the
P2 violation indicator swaps critical moisture and nu_max (within the intervals). In J the weather
and the drying load dominate the absolute variance of P2 ((b): ambient temperature ST 0.384,
throughput 0.277, solid fraction 0.197; critical moisture 0.084, nu_max 0.066), so the epistemic
indices of J are smaller than in E; the order is the same.

**Design R (ranges of the three dominant epistemic inputs doubled).** P2: critical moisture
0.15-0.375 kg/kg (clipped at the quality limit), nu_max 0.2-1.0, heat loss 0-6 W/K. **The order of
the first two changes** for (b), (c) inlet temperature and (d, all): nu_max moves ahead of the
critical moisture ((b): nu_max 0.638 (w 0.053), critical moisture 0.281 (0.026); (c) inlet
temperature: 0.654 / 0.401), a resolved swap; heat loss stays third, the set of dominant inputs is
unchanged, the GAB parameters stay below 0.01 for (b). The order between the two drying-kinetics
parameters therefore depends on their placeholder ranges; that the two of them decide does not.
P1: approach 1-27.5 K, efficiency 0.6-1.0, heat loss 0-4.5 W/K; no change of ranking.

### 7.6 Invalid samples (counted, not dropped)

| variant / design | runs | rows with a reason | reasons |
|---|---|---|---|
| P1 (both variants, all designs) | 126 976 per variant | 0 | - |
| P2 E | 49 152 | 4 (0.01 %) | exact solver failed at the optimal command (4) |
| P2 R | 49 152 | 2835 (5.77 %) | fixed point beyond the critical moisture (2829), solver at the optimum (6) |
| P2 O | 24 576 | 1020 (4.15 %) | fixed point beyond the critical moisture (932), solver at the optimum (89) |
| P2 J (2^10) | 16 384 | 1033 (6.30 %) | fixed point beyond the critical moisture (865), solver at the optimum (168) |

The solver failures at the optimum in design O all lie at an optimal inlet temperature at its
lower bound 60 degC (hot ambient, gas velocity above 0.80 m/s; 89 of 89): there the exact
solver's bed-temperature bracket is the limit, not the optimum (not investigated further). Valid
block shares are stated per output in the artifact; for (d, fixed point feasible) they are 0.624
(E), 0.524 (O), 0.226 (R).

### 7.7 Run time

2.57 h on 14 workers: P2 E 61.7 min, R 45.4 min, O 22.6 min, J 20.1 min; P1 all designs of both
variants together 3.5 min. Per run (in the pool): P1 about 2 ms, P2 0.8-1.0 s (two exact steady
states, the IPOPT rebuild and solve).

### 7.8 Precision extension of design E to 2^14 (added 2026-10-22)

Declared before its run in the driver docstring (commit `834f1d8`): the same design E with base
sample 2^14 and the same seed, a precision extension of the same pre-registered test, **not a new
test**; 2^12 and 2^14 side by side; no further doubling. Artifact
`artifacts/runs/sobol_steady_e14.json` (driver `834f1d8`, clean tree, `git/dirty=False`), all three
variants, design E only. **Determinism:** its prefix N = 4096 reproduces every S1, ST and width of
`sobol_steady_v1` exactly (largest difference 0.0).

**Criterion at 2^14: met for 19 of 20 outputs** with a dominant input (all P1 outputs of both
variants, every P2 output except one). **Not met: P2 (c) optimal gas velocity** -- critical
moisture S1 w 0.067 and ST w 0.122, nu_max ST w 0.109, heat loss ST w 0.095 (all above 0.05); its
first-order intervals of nu_max and heat loss (w 0.049, 0.027) pass. This output sits at its bound
0.80 m/s in most samples (std 0.051 m/s) and is interaction-dominated; its indices are the least
precise of the study. Not doubled further (as declared).

**Dominance classification (ST >= 0.1): one change.** The GAB monolayer moisture for the optimal gas
velocity drops from ST 0.120 (w 0.098) at 2^12 to 0.095 (w 0.043) at 2^14 (2^13: 0.105) -- at
2^12 its interval already straddled 0.1. No other input changes class. Closest to the threshold:
heat loss for the optimal inlet temperature, 0.107 (w 0.022) -> 0.105 (w 0.011), interval
0.099-0.110, stays dominant; heat loss for (b), 0.097 -> 0.097 (interval 0.095-0.099 at 2^14),
stays below.

Side by side (P2, the decision outputs; ST with full width w):

| output | input | 2^12 ST (w) | 2^14 ST (w) | 2^12 S1 (w) | 2^14 S1 (w) |
|---|---|---|---|---|---|
| (b) energy at the optimum | critical moisture | 0.518 (0.042) | 0.519 (0.021) | 0.516 (0.058) | 0.517 (0.028) |
| | nu_max | 0.388 (0.032) | 0.387 (0.016) | 0.386 (0.051) | 0.385 (0.024) |
| (c) inlet temperature | critical moisture | 0.614 (0.075) | 0.616 (0.038) | 0.504 (0.076) | 0.502 (0.037) |
| | nu_max | 0.446 (0.061) | 0.446 (0.030) | 0.342 (0.056) | 0.350 (0.028) |
| | heat loss | 0.107 (0.022) | 0.105 (0.011) | 0.028 (0.031) | 0.025 (0.014) |
| (c) gas velocity | critical moisture | 0.895 (0.236) | 0.908 (0.122) | 0.136 (0.141) | 0.137 (0.067) |
| | nu_max | 0.716 (0.223) | 0.721 (0.109) | -0.003 (0.072) | 0.042 (0.049) |
| | heat loss | 0.501 (0.196) | 0.480 (0.095) | 0.024 (0.071) | 0.006 (0.027) |
| | GAB monolayer | 0.120 (0.098) | 0.095 (0.043) | 0.004 (0.070) | 0.004 (0.022) |
| (d) saving, all valid | critical moisture | 0.536 (0.043) | 0.536 (0.022) | 0.533 (0.058) | 0.534 (0.029) |
| | nu_max | 0.401 (0.033) | 0.401 (0.017) | 0.399 (0.052) | 0.399 (0.024) |
| (a) violation indicator | nu_max | 0.748 (0.093) | 0.761 (0.049) | 0.245 (0.084) | 0.246 (0.044) |
| | critical moisture | 0.690 (0.097) | 0.682 (0.047) | 0.190 (0.081) | 0.181 (0.038) |
| | heat loss | 0.321 (0.069) | 0.324 (0.037) | 0.041 (0.050) | 0.033 (0.025) |

P1: point estimates unchanged to the third decimal, widths halved (e.g. unpressurised fuel, heater
efficiency ST 0.932, w 0.060 -> 0.029; dew-point distance, approach 0.644, w 0.050 -> 0.024).
Every ranking and every reading of sections 7.2-7.5 and 8 stands; the only correction is the
monolayer moisture for the gas velocity (no longer dominant, and therefore no isotherm parameter is
dominant for any P2 output at 2^14).

Invalid at 2^14: P1 none; P2 21 of 196 608 runs (0.01 %, exact solver at the optimal command).
Run time 3.68 h on 14 workers (P2 218.2 min, P1 2.1 min).

## 8. Ranking for the data request ("which question is worth how much")

P2, design E, epistemic inputs by ST for (b) and (c), with the standard deviation of the output
that remains in expectation once the input is known exactly (`std x sqrt(1 - S1)`, independent
inputs). (b) std 0.0875 kW (mean 1.156 kW); (c) inlet temperature std 7.67 K.

| rank | input | ST (b) | ST (c) T_in | (b) std after | (c) std after | question |
|---|---|---|---|---|---|---|
| 1 | critical moisture | 0.518 | 0.614 | 0.061 kW | 5.40 K | lab: drying curve of the material |
| 2 | nu_max | 0.388 | 0.446 | 0.069 kW | 6.22 K | lab: exhaust humidity of the lab line (drying experiments) |
| 3 | heat loss UA | 0.097 | 0.107 | 0.083 kW | 7.56 K | partner: insulation / heat balance of the dryer |
| 4 | GAB (five parameters together) | <= 0.001 | <= 0.029 | 0.087 kW | 7.65 K (monolayer) | lab: sorption isotherm (Befund 34) |
| - | gas-heater efficiency | 0.000 | 0.000 | - | - | partner: burner data -- **decides CO2** (ST 0.467 of (b) CO2), not the operating point |
| - | cp_vapour | 0.000 | 0.000 | - | - | literature / partner item 8 of docs/11 12.7 |

Questions grouped: **drying kinetics** (critical moisture and nu_max, one series of drying
experiments on the lab line) -- sum of S1 0.901 for (b) and 0.846 for (c): knowing both would
leave at most about a third of the std of (b) and 40 % of that of (c) (from the lower bound of the
closed index). **Heat loss** -- about 0.1. **Isotherm** -- below 0.03 for (b) and (c); with the
RTO margin the sorption measurement is the least valuable of the three for the operating point (it
would matter for a product dried close to equilibrium, not studied). **Heater efficiency** --
worth asking for the CO2 booking only. Design R qualifies rank 1 against rank 2: their order
depends on the placeholder ranges, so both belong in the same request.

P1 (main: unpressurised; data set 5): **approach / exchanger characteristic** decides the recovered
heat (ST 1.000) and the dew-point distance (0.644); **heater efficiency** decides the fuel (0.932;
pressurised 0.662 next to the approach 0.314). Questions: exchanger design to the partner /
integration partner; burner efficiency to the partner. Heat loss matters for the dew-point
distance only (0.337).

## 9. Limitations and open points

* All ranges are placeholders; indices are range-dependent (design R: the order of the two
  drying-kinetics parameters swaps when their ranges are doubled).
* Independent uniform inputs; correlations not modelled.
* The pre-registered convergence criterion is not met at 2^12 (7.1); with the precision extension
  to 2^14 of design E it is met for 19 of 20 outputs, not for the optimal gas velocity (7.8).
  Designs O, J and R stay at 2^12 (P2 J 2^10) and carry the widths stated there.
* Indices on valid blocks only: up to 6 % invalid rows in P2 O, R, J; (d, fixed point feasible) is
  a conditional analysis by construction.
* Operative ranges are the operating envelope (stress bands), not running operation; with the
  observed amplitudes throughput and solid fraction would be nearly without effect.
* One operating point per model (P1 data set 5; P2 the lab line at its nominal point for E); the
  lab line's 4-state model and placeholder process config (docs/24 section 9) apply.
* P1 has no fan; P2's energy is the model's total power (electric heater 0.98) as in docs/24.
