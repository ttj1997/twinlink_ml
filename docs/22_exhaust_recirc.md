# Exhaust recirculation of one fluidised-bed line -- moisture limit, energy and CO2

Status: complete (LABORATORY LINE WITH PLACEHOLDER PHYSICS; no plant statement). Lab book:
Befund 102 in `docs/04_stand_ap1.md`. Branch
`exhaust-recirc`. Code: `src/twinlink_ml/simulation/` (optional recirculation field, commit
`e1eccbc`), `src/twinlink_ml/evaluation/exhaust_recirculation.py`. Scripts:
`scripts/run_exhaust_recirc.py` (settings and gate fixed in its docstring, commit `44a331c`, before
any run), `scripts/analyze_exhaust_recirc.py` (commit `6a9dca1`, before its run; one part declared
post hoc, section 5). Artifacts (all from a clean tree, `git/dirty=False`):
`exhaust_recirc_anchor` (`06d6356`), `exhaust_recirc_main` (`d665851`),
`exhaust_recirc_sensitivity` (`eaf105e`), `exhaust_recirc_analysis` (`d40613d`). Tests:
`tests/test_exhaust_recirculation.py` (57), `tests/test_exhaust_recirculation_study.py` (9),
`tests/test_analyze_exhaust_recirc.py` (4).

Scale labels used in every table: **lab** = laboratory line of M1 (0.0177 m2, 2 kg bed, spray rate
1.5 kg/h, `greybox_gab_placeholder.yaml`, placeholder physics); **lab x k** = lab flows scaled
linearly with the constant k = 271.797 of M2 (D4; a scaling computation, not a plant);
**transfer** = a grid of placeholders at the partner's orders of magnitude, no plant model.

Confidentiality: roles only (the application partner, "the partner"; plant A with storage A1; plant
C with lines C1/C2). No raw data are used; the two exhaust temperatures of plant A quoted in
section 6 are the figures already published in `docs/19`.

## 1. Question and answer in one paragraph

How much exhaust can a fluidised-bed line recirculate into its inlet air before the moisture limit
is violated, and what does that save? **At unchanged set points almost nothing:** at the per-context
optimum without recirculation the moisture limit (0.15 kg/kg with the soft-sensor margin m) already
binds at 17 of 25 contexts, and there the admissible recirculation fraction is 0.002 (within the
moisture tolerance); at the other 8 contexts it is 0.035-0.18 (lab). **With re-optimised set
points much more:** the optimiser raises the inlet temperature (median 60.3 -> 81.0 degC at
r = 0.8) and keeps the moisture exactly at the limit; every grid value up to r = 0.8 is feasible at
all 25 contexts, and the best point per context lies at a median r of 0.86 (10th percentile 0.1).
The CO2 of the line falls by **17.0 % in the median (10th percentile 0.4 %, maximum 29.1 %; lab,
gas-fired heater booking, fan power included)**, strongly dependent on the ambient temperature
(25-29 % at 5 degC, 14-20 % at 20 degC, 0-15 % at 27.5-35 degC; Spearman -0.89). The binding limit
is the product moisture (24 of 25 best points); the lower bound of the gas velocity is the second
active limit (17 of 25 within 0.01 m/s); bed temperature, inlet box and fog in the mixing chamber
do not bind in the main case. With the line's own heat recovery the recirculation saves **more,
not less** (20.4 % in the median, lab x k): at the optimum without recirculation the exhaust is too
cold to be recovered at `dT_min` 20 K (0 kW at 17 of 25 contexts); recirculation raises the inlet
and exhaust temperature and makes the smaller purge recoverable. The ranking of the placeholders by
their effect on the median saving: maximum saturation degree (gas-solid transfer proxy) and the
kinetics critical moisture (about +-4 to +7 percentage points each), then the quality limit and the
lower gas-velocity bound (1-2 points); the uncalibrated GAB parameters move it by at most 0.6 points.

## 2. Model extension (what was added, what stays)

**Field.** `OperationConfig.exhaust_recirculation_fraction` `r` (optional, default `None`) and
`EnergyConfig.recirculation_duct_loss_coefficient` `zeta_rec` (optional, default `None`), both in
`src/twinlink_ml/simulation/config.py`. Unset, neither enters the fingerprint
(`_OPTIONAL_UNSET_OMITTED_FIELDS`, the mechanism of `product_moisture_limit_kg_per_kg`); every
existing config keeps its fingerprint (`tests/test_exhaust_recirculation.py` checks the nine
fingerprints of the three committed configs) and the fresh-air path is bit-identical to the code
before the field (float-hex comparison of heater, fan, outlet humidity, evaporation, total power,
bed temperature, moisture and exhaust enthalpies with numbers computed on `c3f13a8`; the
diagnostics keep their keys). The existing fingerprint regression tests are green.

**Flowsheet.** A fraction `r` of the dry air through the bed is taken from the exhaust (bed outlet
state `T_bed`, `Y_out`; well-mixed bed) and mixed with fresh air (`T_amb`, `Y_a`) ahead of the
heater; the heater brings the mixture to `T_in`. Gas phase quasi-stationary as everywhere in the
model, so the loop is algebraic:

    Y_in = (1 - r) Y_a + r Y_out,      Y_out = Y_in + nu (Y_eff - Y_in)

with the saturation degree `nu = nu_max f(eta)` of the existing drying kinetics (same isotherm,
same cap `Y_eff`). For given `nu` the pair is linear (`Y_in = ((1-r) Y_a + r nu Y_eff) /
(1 - r + r nu)`); the remaining fixed point (`nu` depends on `phi_out` through the GAB isotherm)
is found by a bracketed root search on a monotone residual (`thermo.recirculated_inlet_humidity`,
15 steps, error below 2e-11 relative up to `r = 0.9` on a 40 x 40 x 3 x 3 grid of bed states and
ambient points). **A damped Picard iteration as in the fresh-air path does not converge here**
(relative errors above 1 at `r = 0.8` after 60 steps near the kink of the drying curve): the
recirculation multiplies the loop gain. Mixing temperature from the enthalpy balance
(`thermo.mixing_temperature_celsius`); heater duty `m_L (c_pL + Y_in c_pD)(T_in - T_mix)`; the
bed itself sees only `Y_in` and `T_in` (its balances are unchanged). Purge `(1 - r) m_L` leaves
the line; `SteadyStateSolution.exhaust_sensible_kw/exhaust_latent_kw` refer to it (latent part
against the fresh-air humidity). Recirculation fan: `r u A zeta_rec rho (r u)^2 / (2 eta_V)`,
i.e. proportional to the recirculated volume flow cubed, added to the fan power of the process
model (bed + distributor, unchanged).

**Tests** (`tests/test_exhaust_recirculation.py`, 57 cases): fingerprints, round trip,
range checks; bit-identity of the fresh-air path; `r = 0` against the fresh-air path (1e-3
relative, the iterations differ); water balance of the mixing (`Y_in = (1-r) Y_a + r Y_out`,
1e-12) and enthalpy balance of the mixing (1e-12) at 4 x 3 points; water balance of the whole loop
at steady state (`(1 - r) m_L (Y_out - Y_a) = m_evap`, 1e-12; spray water = evaporation + product
water, 1e-5); heater booking from `T_mix`; monotone heater power, inlet humidity and moisture in
`r`; cube law of the recirculation fan; convergence of the fixed point against 200 steps at 15
corners; a rollout from the steady state stays there (with recirculation); condensing branch;
**no steady state at `r = 0.95` at a cold point** (the drying capacity per kg of air shrinks to
`(1 - r) / (1 - r + r nu)` of the fresh-air value; the solver reports it, the oracle counts a
failure). The steady-state solver needs no change: it bisects the same right-hand side.

## 3. Method (fixed in the docstring of `scripts/run_exhaust_recirc.py` before any run)

| item | setting | status |
|---|---|---|
| line | laboratory line of M1, `greybox_gab_placeholder.yaml`, spray rate 1.5 kg/h | placeholder physics, lab scale |
| contexts | the 25 ambient points of Befund 52 (T_amb 5-35 degC x phi 0.25-0.85); sensitivity on the 3 x 3 sub-grid | as M1 |
| objective | CO2 with the gas-fired heater (`eta_gas` 0.80), 0.366 / 0.201 kg CO2e/kWh (`carrier_energy`), fan power included | as M1 (D5) |
| electricity "2030" | 0.20 kg CO2e/kWh, re-weighted at the same solutions (not re-optimised) | placeholder |
| quality limit | 0.15 kg/kg with margin m = 0.005961819615175404 (M arm of M1/M2); kinetics keep the critical moisture 0.20 | as M1 |
| other limits | fluidisation window, bed temperature 80 degC, solver validity (`check_constraints`), fog in the mixing chamber `Y_in <= Y_s(T_mix)` (new) | as M1 + fog |
| decision variables | `T_in` in 60-120 degC, `u` in [1.26, 2.5] m/s (0.80 as a sensitivity), `r` fixed on the grid or free in [0, 0.95]; nozzle pressure fixed at 1.0 bar (exact optimum: atomiser power zero up to ambient pressure) | M1 optimised the nozzle pressure and found 1.00-1.01 bar |
| recirculation fan | `zeta_rec` = 350 (equal to the distributor coefficient) | placeholder, band 100-1000, no source |
| optimiser | DE + Nelder-Mead of M1, seed 0, popsize 10, tolerance 1e-4, search precision 30 / 1e-7; `maxiter` 20 instead of 50 (run time) | checked by the anchor gate (section 4) |
| heat recovery | condensing cascade of `heatnetwork`, `dT_min` 20 K, `T_kalt` 29.1 degC, `eta_H` 0.80, `H_u` 45 MJ/kg, `dT_min` along the exchanger; the purge `(1 - r) k m_L` as source, the line's own inlet air as sink; k = 8 kg/s / line flow at the config nominal point (M2, D4) | placeholders of M2 |
| recovery layouts | `fresh_air`: exchanger before the mixing point (sink `(1 - r) k m_L` from `T_amb`, duty = heater duty exactly); `mixed_air`: exchanger after the mixing point (sink `k m_L` from `T_mix` to `T_in`) | both equal at r = 0 |

Arms: **G** -- `r` fixed on {0, 0.1, ..., 0.8} (requested grid) plus 0.9 (extension), `(T_in, u)`
optimised; **R** -- `r` as third variable; **J0 / J** -- `r = 0` / `r` free with the CO2 AFTER the
recovery cascade (`fresh_air` layout) as objective. Descriptive (analysis script, no new optimum):
the cascade at every grid solution in both layouts; the largest `r` at UNCHANGED set points (the
`r = 0` optimum) before the moisture exceeds `L - m` (and, labelled, `L`).

## 4. Anchor gate (artifact `exhaust_recirc_anchor`): passed

Contexts 0, 12, 24 (5 degC / 0.25, 20 degC / 0.55, 35 degC / 0.85). Gate fixed beforehand: every
relative CO2 difference below 0.5 %.

| check | context 0 | context 12 | context 24 |
|---|---|---|---|
| fresh-air path, reduced budget, against the M arm of `m1_margin_cost_l015_umin126_k1` (Befund 80) | -1.1e-16 | -3.9e-6 | -2.0e-5 |
| `r = 0.0` against the fresh-air path | +2.9e-7 | +7.6e-5 | +5.8e-5 |
| `r = 0`, maxiter 20 against 50 | 0 | +4.0e-7 | -3.3e-6 |
| `r` free, maxiter 20 against 50 | +4.0e-4 | +3.1e-7 | +6.5e-4 |

The reduced budget reproduces M1 and the full budget in CO2. **The optimal `r` itself is less well
determined than its CO2:** at context 24 the two budgets give r = 0.857 and 0.881 for a CO2
difference of 0.065 % -- the objective is flat in `r` near the optimum. Read `r*` to about +-0.03.

## 5. Results (lab, artifact `exhaust_recirc_analysis`)

### 5.1 Recirculation fraction fixed, set points re-optimised (arm G, 25 contexts)

CO2 change against `r = 0` at the same context (gas-fired heater booking; median and range over the
contexts). "moisture binds" / "u at box min" count the contexts whose optimum is within 1e-3 kg/kg
of `L - m` / within 0.01 m/s of 1.26 m/s.

| scale | r | feasible | CO2 saving, median (min..max) | heater gas saving, median | CO2 saving, electricity "2030" | inlet temperature, median | inlet humidity Y_in, median | exhaust dew point, median | moisture binds | u at box min |
|---|---|---|---|---|---|---|---|---|---|---|
| lab | 0 | 25 | 0 | 0 | 0 | 60.3 degC | 0.0077 | 23.6 degC | 17 | 21 |
| lab | 0.1 | 25 | 2.1 % (-1.2..4.6) | 2.3 % | 2.2 % | 61.5 | 0.0089 | 24.7 | 20 | 21 |
| lab | 0.2 | 25 | 4.2 % (-2.5..8.9) | 4.6 % | 4.4 % | 62.2 | 0.0104 | 26.0 | 25 | 18 |
| lab | 0.3 | 25 | 5.8 % (-4.3..12.1) | 6.4 % | 6.0 % | 63.4 | 0.0124 | 27.2 | 25 | 21 |
| lab | 0.4 | 25 | 7.3 % (-5.7..15.3) | 8.1 % | 7.6 % | 65.5 | 0.0150 | 28.6 | 25 | 21 |
| lab | 0.5 | 25 | 9.2 % (-7.2..18.5) | 10.2 % | 9.6 % | 67.8 | 0.0181 | 30.8 | 25 | 24 |
| lab | 0.6 | 25 | 11.1 % (-9.8..21.8) | 12.4 % | 11.7 % | 70.7 | 0.0247 | 34.8 | 25 | 23 |
| lab | 0.7 | 25 | 13.0 % (-11.8..25.2) | 14.7 % | 13.7 % | 74.3 | 0.0349 | 39.2 | 25 | 23 |
| lab | 0.8 | 25 | 15.0 % (-13.7..28.7) | 17.0 % | 15.9 % | 81.0 | 0.0567 | 46.0 | 25 | 21 |
| lab | 0.9 (extension) | 22 | 13.1 % (-15.7..22.2) | 16.8 % | 15.6 % | 94.2 | 0.1331 | 59.4 | 22 | 13 |

Reading. (a) Every requested grid value is feasible everywhere: the moisture limit is not a hard
ceiling on `r` once the set points can move; it is paid for with inlet temperature. (b) The
negative minima are the warm contexts (27.5 and 35 degC): there the exhaust is barely warmer than
the ambient air, the mixing gains little and the extra humidity costs inlet temperature;
recirculation is a cold-weather measure in this model. (c) At r = 0.9 there is no steady state at
three cold, humid contexts (5 degC, phi >= 0.55): the drying capacity per kg of air falls to
`(1 - r) / (1 - r + r nu)` of the fresh-air value and the spray water cannot leave (the solver
reports it; tested). (d) The exhaust dew point rises by 22 K between r = 0 and 0.8 in the median;
the inlet humidity reaches 0.057 kg/kg at r = 0.8 -- far above any ambient air, so the drying
kinetics and the isotherm are used outside the range in which they were set (section 7). (e) Fog in
the mixing chamber does not bind in the main case; the largest mixing saturation ratio at the best
points is 0.977 (cold contexts). (f) Electricity: the recirculation fan of the placeholder (cube
law, `zeta_rec` 350) adds at most 8.7 W against 0.7-1.5 kW of heater power at lab scale; the
electricity re-weighting "2030" (0.20 kg/kWh) raises the relative saving by 0.1-1.0 points (grid up
to r = 0.8 and envelope) because
the electric share of the CO2 shrinks.

### 5.2 How much before the moisture limit is violated -- at unchanged set points

Bisection on `r` at the set points of the `r = 0` optimum (no re-optimisation), largest `r` whose
steady state stays below the threshold:

| scale | threshold | contexts | median | range |
|---|---|---|---|---|
| lab | `L - m` = 0.144 kg/kg | 25 | 0.0024 | 0.0022..0.18 |
| lab | -- of these, where the r = 0 optimum binds the limit | 17 | 0.002 (the moisture tolerance 1e-4 kg/kg) | 0.0022..0.0026 |
| lab | -- where it does not bind | 8 | -- | 0.035..0.18 |
| lab | `L` = 0.15 kg/kg (consumes the soft-sensor margin m; labelled, not a design value) | 25 | 0.12 | 0.11..0.26 |

So the honest answer to "how much before the limit" depends on whether the set points follow: a
line run at its energy optimum has no humidity headroom (the optimum sits on the limit), and every
percent of recirculation must be paid for by a higher inlet temperature or gas velocity; a line run
with a dry product (the 8 non-binding contexts, mostly cold and dry air) can take 3.5-18 % without any
other change.

### 5.3 Recirculation as a decision variable (arm R) and the envelope

| scale | quantity | median | 10th percentile | max | count |
|---|---|---|---|---|---|
| lab | best `r` (envelope over R and G) | 0.86 | 0.10 | 0.92 | 2 of 25 at r = 0 |
| lab | CO2 saving against `r = 0` (envelope) | 17.0 % | 0.4 % | 29.1 % | |
| lab | heater gas saving (envelope) | 19.3 % | 0.5 % | 31.7 % | |
| lab | CO2 saving, electricity "2030" (envelope) | 18.0 % | 0.4 % | 30.3 % | |
| lab | exhaust dew point rise at R | 26.6 K | 3.4 K | 41.9 K | |
| lab | moisture binds / u within 0.01 m/s of 1.26 / inlet at box min (envelope) | | | | 24 / 17 / 3 of 25 |

By ambient temperature (envelope, the five humidities 0.25..0.85): 5 degC 29.1-25.1 %
(r 0.81-0.70); 12.5 degC 23.5-24.3 % (r 0.89-0.81); 20 degC 13.8-20.5 % (r 0.88-0.92); 27.5 degC
0.4-14.8 % (r 0.1-0.91); 35 degC 0-9.1 % (r 0-0.9). At 5 degC the best `r` falls with the ambient
humidity; at 20-35 degC the saving rises with it (the dry warm air is cheap to heat anyway). Over
all contexts `r*` does not correlate with the ambient absolute humidity (Spearman 0.01).

**Declared post hoc (written into the analysis script before it ran, after the main artifact was
seen):** the three-variable search with the reduced budget returned, at 9 of 25 contexts (all at
27.5 and 35 degC), a point worse than the best grid point, at 5 of them by more than 0.1 % of the
base CO2 (largest 1.8 %), and at 4 contexts worse than `r = 0` itself. The landscape is bimodal
there (little recirculation or a lot). The envelope -- the best feasible point per context over R
and all grid arms, same objective and constraints -- is therefore the figure reported; the raw R
values are in the artifact (median saving 17.0 % as well, minimum -0.8 %). The anchor gate checked
cold and medium contexts only and could not see this.

### 5.4 Heat recovery: less purge, but hotter and wetter (lab x k)

Condensing cascade (M2 placeholders), the line's purge `(1 - r) k m_L` into its own inlet air.

Joint optimum with the CO2 after recovery as the objective (arms J0 at r = 0, J with r free;
envelope = best of J and J0):

| scale | quantity | median | 10th percentile | max |
|---|---|---|---|---|
| lab x k | recovered power at the r = 0 optimum (J0) | 0 kW (17 of 25 contexts recover nothing) | 0 | 23.1 kW |
| lab x k | recovered power with recirculation (J) | 12.6 kW | 4.9 kW | 37.6 kW |
| lab x k | best `r` with recovery (J) | 0.87 | 0.71 | 0.90 |
| lab x k | CO2 saving of recovery alone (J0 against G0) | 0.0 % | 0.0 % | 5.5 % |
| lab x k | CO2 saving of recirculation with recovery (envelope J against J0) | 20.4 % | 1.0 % | 33.2 % |
| lab x k | CO2 saving of both against neither | 20.4 % | 1.0 % | 36.1 % |

Descriptive, cascade at the grid solutions (set points optimised without recovery), median over
25 contexts:

| scale | r | recovered, fresh-air layout | recovered, mixed-air layout | CO2 after recovery vs r = 0, fresh-air layout | idem, mixed-air layout |
|---|---|---|---|---|---|
| lab x k | 0 | 0 kW | 0 kW | 0 | 0 |
| lab x k | 0.4 | 0 kW | 0 kW | 7.3 % | 7.3 % |
| lab x k | 0.6 | 4.9 kW | 0 kW | 12.7 % | 11.1 % |
| lab x k | 0.7 | 10.1 kW | 0 kW | 16.2 % | 13.0 % |
| lab x k | 0.8 | 13.2 kW | 0 kW | 19.1 % | 15.0 % |

Reading. The expected conflict -- recirculation takes exhaust away from the recovery -- does not
appear in this model, because the lab optimum without recirculation runs the line cold (inlet
60 degC at the box minimum at 14 of 25 contexts, exhaust 32-44 degC), and its exhaust cannot heat
the inlet air across `dT_min` 20 K. Recirculation raises inlet and exhaust temperature; the purge
becomes warm enough to be recovered. **The layout matters:** an exchanger that preheats the fresh
air BEFORE the mixing point keeps the cold sink and recovers 13 kW at r = 0.8; one that heats the
mixture after it sees a sink already at `T_mix` (45 degC at r = 0.8) and recovers nothing at any
grid point. The dew point of the purge rises by 27 K in the median (R), so a condensing exchanger
would condense earlier; in the joint arm J the cascade stage is limited by `dT_min` (pinch) at all
25 contexts, with condensate of up to 8.5 g/s (lab x k); at the `r = 0` optimum (J0) the dew point
limits 5 of 25 stages. With the partner's plant C, which already has exhaust heat recovery, the
layout question is the first one to ask (section 8).

### 5.5 Sensitivity (3 x 3 sub-grid, envelope of R and r = 0 per variant)

Reference (main run on the same 9 contexts): median saving 16.96 %, median `r*` 0.81.

| rank | placeholder (status) | variant | median saving | change against reference | median `r*` | binding limit at `r*` |
|---|---|---|---|---|---|---|
| 1 | maximum saturation degree (literature 0.4-0.8, placeholder 0.60) | 0.45 | 23.5 % | +6.6 points | 0.87 | moisture 9/9 |
| 2 | kinetics critical moisture (placeholder 0.20) | 0.25 | 23.1 % | +6.1 points | 0.93 | moisture 9/9, r box 1/9 |
| 3 | kinetics critical moisture | 0.17 | 12.8 % | -4.2 points | 0.60 | moisture 9/9 |
| 4 | quality limit (0.15 main, M1 stress case) | 0.20 | 15.0 % | -2.0 points | 0.77 | moisture 4/9, **fog 5/9** |
| 5 | lower gas-velocity bound (lab 1.26 m/s) | 0.80 m/s | 15.7 % | -1.3 points | 0.80 | moisture 8/9, fog 1/9 |
| 6 | GAB monolayer moisture (uncalibrated, 0.020) | x0.7 | 16.3 % | -0.6 points | 0.81 | moisture 9/9 |
| 7 | GAB monolayer moisture | x1.3 | 17.5 % | +0.5 points | 0.81 | moisture 9/9 |
| 8 | GAB K (uncalibrated, 0.75) | 0.80 | 17.0 % | +0.1 points | 0.81 | moisture 9/9 |

Ambient humidity is an axis of the 25 contexts themselves (section 5.3): at 5 degC the saving falls
from 29.1 to 25.1 % between phi 0.25 and 0.85; at 20 degC it rises from 13.8 to 20.5 %. Reading:
the saving depends on how much drying capacity the line has to spare, i.e. on the kinetics
placeholders (saturation degree, critical moisture), not on the isotherm. A weaker gas-solid
transfer (0.45) makes the fresh-air operation more expensive and the recirculation relatively more
attractive. With the relaxed quality limit 0.20 the fog limit in the mixing chamber becomes the
binding one at 5 of 9 contexts. Every variant was feasible at all 9 contexts; the raw R arm was
beaten by `r = 0` at one context per variant (except 0.20), the same warm-context bimodality as in
the main run.

## 6. Transfer grid at the partner's orders of magnitude (scale: transfer, every number a placeholder)

Not a plant model. The relative heater-gas saving of the lab envelope (10th percentile 0.5 %,
median 19.3 %, 90th percentile 29.2 %) is applied to a heater duty `m_L c (T_in - T_amb)` built
from the partner's orders of magnitude. Placeholders (value; band; source; status) are listed in
`TRANSFER` of `scripts/analyze_exhaust_recirc.py` and in the artifact: annual mean ambient 10 degC
(8-12; long-term mean, placeholder), humidity 0.75; volumes read as Nm3/h (assumption); plant A
30 000 Nm3/h, inlet 82.5 degC (partner 80-85), steam heater with boiler and distribution
efficiency 0.85 (0.75-0.92, placeholder), 8760 h/a (partner, 24/7); plant C 2 x 20 000 Nm3/h,
inlet 80 degC (70-90, no partner figure), direct gas burner 0.95 (0.90-1.00, placeholder), volume
reading 8760 h/a, **annual reading = volume reading / 4.9** (Befund 99: the volume flow and the
annual exhaust figure of plant C disagree by this factor; both readings are carried). Gas factor
0.201 kg/kWh, CO2 price 55 and 300 EUR/t (partner).

| scale | plant | heater duty | fuel without recirculation | lab saving fraction | fuel saved | CO2 saved | share of site CO2 (11 500 t/a) | value at 55 / 300 EUR/t |
|---|---|---|---|---|---|---|---|---|
| transfer | A | 790 kW | 8140 MWh/a | median 19.3 % | 1570 MWh/a | 315 t/a | 2.7 % | 17 / 95 kEUR/a |
| transfer | A | | | p10 / p90 0.5 / 29.2 % | 42 / 2380 MWh/a | 8 / 477 t/a | 0.1 / 4.2 % | 55 EUR/t: 0.5 / 26; 300 EUR/t: 2.5 / 143 kEUR/a |
| transfer | C, volume reading | 1017 kW | 9370 MWh/a | median 19.3 % | 1810 MWh/a | 363 t/a | 3.2 % | 20 / 109 kEUR/a |
| transfer | C, volume reading | | | p10 / p90 | 49 / 2740 MWh/a | 10 / 550 t/a | 0.1 / 4.8 % | 55 EUR/t: 0.5 / 30; 300 EUR/t: 2.9 / 165 kEUR/a |
| transfer | C, annual reading | 207 kW (mean) | 1910 MWh/a | median 19.3 % | 369 MWh/a | 74 t/a | 0.6 % | 4 / 22 kEUR/a |
| transfer | C, annual reading | | | p10 / p90 | 10 / 558 MWh/a | 2 / 112 t/a | 0.0 / 1.0 % | 55 EUR/t: 0.1 / 6; 300 EUR/t: 0.6 / 34 kEUR/a |

The plant-A duty of the placeholder grid (790 kW) is of the order of the measured process-air
demand of plant A in Befund 99 (798 and 957 kW on the two days), a consistency check, not a
calibration. **What the transfer cannot carry:** the lab fraction comes from a line run at a
moisture limit with a placeholder kinetics; a plant's headroom is set by its own exhaust humidity.
Two partner-independent bounds instead:

Pure mixing (no moisture constraint, exhaust humidity = fresh air + 0.010 kg/kg, capped at
saturation), heater-duty saving `(T_mix - T_amb) / (T_in - T_amb)`, plant-A inlet 82.5 degC:

| scale | exhaust temperature | r = 0.2 | r = 0.4 | r = 0.6 |
|---|---|---|---|---|
| transfer | 34.4 degC (measured outlet of plant A, product-2 day, docs/19) | 6.8 % | 13.6 % | 20.3 % |
| transfer | 45 degC (placeholder) | 9.8 % | 19.5 % | 29.2 % |
| transfer | 60.7 degC (measured outlet of plant A, product-1 day, docs/19) | 14.2 % | 28.3 % | 42.3 % |

Humidity headroom of the exhaust (the largest `r` keeping the exhaust relative humidity below a
limit at the given exhaust temperature, for a humidity pickup `dY` per kg of dry air; ambient
10 degC / 0.75, `Y_a` 0.0057; `Y_out(r) = Y_a + dY / (1 - r)`):

| scale | exhaust | limit phi 0.6: dY 0.005 / 0.010 / 0.020 | limit phi 0.9: dY 0.005 / 0.010 / 0.020 |
|---|---|---|---|
| transfer | 34.4 degC | 0.67 / 0.33 / 0 | 0.81 / 0.61 / 0.23 |
| transfer | 45 degC | 0.84 / 0.69 / 0.37 | 0.90 / 0.81 / 0.62 |
| transfer | 60.7 degC | 0.94 / 0.88 / 0.75 | 0.96 / 0.93 / 0.85 |

A cool, humid exhaust (the product-2 day of plant A, whose reconstructed exhaust was at saturation
in docs/19) leaves little or no headroom; a warm exhaust leaves a lot. The pickup `dY` and the
exhaust humidity limit are the two numbers that decide it, and neither is known for the plants.

## 7. Caveats

1. **Placeholder physics, lab scale.** The kinetics (saturation degree, critical moisture), the GAB
   isotherm and the fan coefficients are uncalibrated; the line is the M1 lab line. The two
   kinetics placeholders move the median saving by 4-7 points (section 5.5).
2. **Humidity outside the range the kinetics were set for.** At the best points the inlet humidity
   reaches 0.009-0.18 kg/kg and the exhaust dew point 25-64 degC; the model has no limit on inlet
   humidity apart from the product moisture, fog in the mixing chamber and the bed temperature.
   Real constraints that are not modelled: condensation in ducts, filters and on cold walls,
   stickiness or agglomeration at high humidity, product quality other than moisture, corrosion.
   A recirculation of 0.8-0.9 is therefore a model optimum, not a recommendation.
3. **Optimiser.** Reduced DE budget (gate passed in CO2), `r*` determined to about +-0.03; at warm
   contexts the three-variable search is bimodal and the envelope (declared post hoc) is reported.
4. **Steady state only**, one spray rate, no dynamics of the recirculation loop (dead time, control
   of the damper), no start-up; the soft-sensor margin m is the one of M1.
5. **Recovery** with the M2 placeholders (`dT_min` 20 K, `T_kalt` 29.1 degC, `eta_H` 0.80,
   `H_u` 45 MJ/kg); only the line's own inlet air is a sink; no external source or sink, no storage.
6. **Direct gas burner not modelled:** a direct-fired heater (plant C) adds combustion water to the
   inlet air (of the order of 2-3 g per kg of air for 60 K of heating, a rough stoichiometric
   estimate, placeholder), which uses part of the humidity headroom that recirculation also needs.
7. **Transfer** (section 6) is a placeholder grid; the lab saving fraction is not a plant figure.
   Plant C's two readings differ by a factor 4.9 (Befund 99).
8. **Dust and hygiene.** Recirculated exhaust must be filtered before the heater; filter pressure
   loss is part of the placeholder `zeta_rec` only.

## 8. Questions for the partner (by how much the answer would move the result)

1. **Exhaust state of plant A and C** (temperature and humidity or dew point at the outlet, per
   product): decides the humidity headroom directly (section 6, `r_max` 0 to 0.96 across the
   grid). Most important.
2. **Water load per kg of process air** (spray rate, solid fraction of the feed per product): the
   pickup `dY`, second axis of the headroom.
3. **Product-moisture specification and its sensitivity to inlet humidity** (or a sorption isotherm
   and drying curve of the product): the kinetics placeholders are the top of the sensitivity
   ranking (critical moisture, saturation degree: 4-7 points of saving each).
4. **Any operational limit on exhaust humidity** (design relative humidity, condensation in the
   exhaust duct or filter, product stickiness) and whether recirculation is permitted for product
   or hygiene reasons.
5. **Heat recovery at plant C:** where the exchanger sits (before or after a possible mixing
   point), its cold side (`T_kalt`) and approach: the layout decides whether recirculation and
   recovery add up (fresh-air layout) or exclude each other (mixed-air layout).
6. **Plant C operating hours and flows** per line (resolves the factor 4.9 of Befund 99, a factor
   4.9 on every annual figure of plant C).
7. **Heater efficiencies:** steam generation and distribution for plant A, burner type for plant C
   (direct firing adds water to the inlet air).
8. **Fan reserve and duct layout** for a recirculation branch (pressure loss, damper): the
   recirculation fan is a placeholder with a negligible effect at lab scale; at plant scale its
   size is open.

## 9. Assumptions made instead of stopping (each documented where it is used)

| assumption | value | where |
|---|---|---|
| recirculation taken at the bed outlet state, no duct loss in the recirculation branch | `T_bed`, `Y_out` | model, section 2 |
| recirculation fan: pressure loss quadratic in the recirculated flow | `zeta_rec` 350 (= distributor coefficient), band 100-1000 | run script |
| nozzle pressure at the box minimum | 1.0 bar (exact optimum) | module |
| reduced DE budget | maxiter 20, checked by the anchor gate | run script |
| fog constraint | `Y_in <= Y_s(T_mix)`, tolerance 1e-4 | module |
| recovery placeholders | those of M2; k = 271.797 | run script |
| recovery layouts | fresh-air (main) and mixed-air (descriptive) | module |
| electricity factor "2030" | 0.20 kg CO2e/kWh, re-weighting at the same solutions | analysis script |
| transfer placeholders | table `TRANSFER` (value, band, source, status) | analysis script |
| volumes of the partner as Nm3/h | 1.293 kg/Nm3 | analysis script |
| plant C annual reading | volume reading / 4.9 | analysis script, Befund 99 |
| branch base | the local `master` at `c3f13a8` (contains `origin/master` and Befund 99, which the instruction refers to; same base as the parallel worktrees) | session |
