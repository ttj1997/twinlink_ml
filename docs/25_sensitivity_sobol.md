# 25 -- Global sensitivity analysis (Sobol indices) of the steady-state models

Status: design fixed before the run (sections 1-6); results in sections 7-9 (written after the
run, numbers only from the artifact named there). Branch `sensitivity`; finding number open (lab
book, assigned by the managing session).

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
