# 28 -- Energetic value of an inline moisture measurement under model mismatch

Status: run and evaluated. Design fixed and committed before the runs (sections 1-6); runs and
results in sections 7-9 (numbers only from the artifacts named there). Pre-registered reading:
**V2** (median value of the measurement 13.4 % of the energy). Branch `sensor-value` (from `origin/master`
cfb9314); the finding number is 111 (lab book).

## 1. Question

How much energy does it cost not to measure the product moisture, or to measure it badly, when
the controller's model deviates from the process? The headroom study (Befund 105) gave R1 perfect
knowledge of the state and the model; its robustness variant spread only the isotherm, which
Befund 108 found to matter least. Here the truth differs from the controller's model in the
parameters that matter most (critical moisture, nu_max) and in the isotherm, and the controller can
only correct its model through a feedback of the moisture.

## 2. Process, model mismatch, scenarios

* **Truth**: the dynamic grey-box model of Befund 105 (`GreyBoxFluidBedGranulator`, 4 states,
  5 s), lab line, solid fraction around 0.30, quality limit 0.15 kg/kg **(placeholder)** as a path
  condition at every truth sample, violation tolerance 1e-4 kg/kg (as there). Settings of the line
  and the numerics are read from `conf/headroom/dynamic_headroom.yaml`.
* **Model mismatch**: 20 parameter draws of the TRUTH (scrambled Sobol', fixed seed) over the
  critical moisture, nu_max (`max_saturation_degree`) and the five GAB parameters; bands and scales
  read from `conf/sensitivity/sobol_steady.yaml` (model p2; the C constant log-uniform). The
  controller keeps the nominal model. Draws that fail the isotherm check against the critical
  moisture would be replaced by the next points of the sequence and counted (none was).
* **Scenarios**: family "observed" of Befund 105, 4 of each of its five types (20 scenarios, 24 h
  each), indices `round(linspace(0, n_type - 1, 4))`: `ambient` 0, 8, 16, 24 (day profiles around the grid points
  (5 degC, 0.25), (12.5 degC, 0.70), (27.5 degC, 0.40), (35 degC, 0.85)); `throughput`, `solid_fraction`,
  `inlet_temperature_slow` (sigma 0.35 K, 3 h) and `inlet_temperature_fast` (1.5 K, 4 min, below
  the 5-min command grid) each 0, 6, 13, 19. 400 pairs (draw, scenario).

## 3. Controller and feedback

Quasi-stationary RTO as R1 of Befund 105 (nominal model, steady optimum at the current disturbance
with perfect knowledge of it, commands piecewise constant over 5 min, rate limits in the truth),
with a zeroth-order bias correction (`twinlink_ml.sensor_value`; existing models only read):

    X_model + b <= 0.15 - margin,     b = first-order filter of (y - x_model),  tau = 30 min

* `x_model` = the open-loop prediction of the nominal model: the nominal model simulated alongside
  the truth with the same realised commands and disturbances. Without mismatch and with an ideal
  measurement it equals the truth bit for bit, so b = 0 and the controller is R1 exactly (test).
* Filter: `b_j = a b_(j-1) + (1 - a) e_j`, `a = exp(-25 s / 1800 s)`, first value `b = e`; causal
  (test). The time constant **30 min is a placeholder, fixed before the run and not tuned**.
* Feedback y on a 25 s grid:
  * **F0** none (b = 0);
  * **F1** ideal inline measurement (true moisture);
  * **F2** realistic inline measurement: the existing sensor model (PT1 10 s, relative Gaussian
    noise 0.005, `MeasurementConfig` of `conf/process/greybox_gab_placeholder.yaml`) plus
    independent outliers as Befund 109 (`p = 0.25`, Student-t with 3 degrees of freedom, scale
    10 x the Gaussian std at that sample; `conf/sensor/outliers_placeholder.yaml`, placeholders),
    no Hampel filter (decision on Befund 109), no clipping;
  * **F3** soft sensor: the moisture head of a GRU trained with the existing pipeline of Befund 94
    (arm N, `DataSpec.new_study()` preset version 2, `n_train` 64, `max_epochs` 200, validation
    pool) in the operating mode `SOFT_SENSOR`, ONLY on data of the nominal model, three training
    seeds (1, 2, 3); `scripts/train_sensor_value_soft_sensor.py`.

## 4. Margins and metrics

For every variant (F3: every seed) and every pair the smallest margin with which the path condition
holds (bisection as Befund 105: bracket [0, 0.05] doubled up to 0.12, tolerance 2.5e-4 kg/kg; one
search per pair, `sensor_value.search.PairBisection`, the step form of `bisect_margin`, tested).
Variant margins: **(i)** all 400 pairs hold = the largest per-pair margin; **(ii)** at least 95 %
of the pairs (380) hold = the 380th smallest per-pair margin (**main quantity**). Both are evaluated
on all pairs (verification; a failed candidate is raised to the next per-pair margin).

Per pair at the margin of a criterion: energy and CO2 (gas-fired heater, 0.366 / 0.201 kg/kWh)
relative to F1 at its margin of the same criterion, median and range over the pairs; required
margin; maximum violation; for F3 the range over the training seeds. Value of the measurement
`V = (E_F0 - E_F2) / E_F1` and `W = (E_F3 - E_F2) / E_F1` per pair. Which draws drive the F0 margin:
per draw its own F0 margin (largest over its scenarios), Spearman correlation with the seven
parameters and the variance explained by the question groups of Befund 108 (drying kinetics:
critical moisture and nu_max; isotherm: the five GAB parameters); descriptive with 20 draws.

## 5. Pre-registered reading

Fixed in the docstring of `scripts/analyze_sensor_value.py`, committed before the first run
(before the soft-sensor training and the main run). Main: criterion (ii), median over the pairs of V:

| class | median V | consequence |
|---|---|---|
| V0 | < 1 % | the inline measurement has no notable energy value for this line; partial observability does not justify a learning control |
| V1 | 1-5 % | measurable value; the question to the application partner has priority |
| V2 | > 5 % | clear value; check a sensor investment and a control with state estimation |

If F2 meets no margin of (ii) up to the cap: no reading. If F0 meets none while F2 does:
"V2 (not quantified)". Reported separately: whether F3 lies between F0 and F2 (captured share
`s = median(E_F0 - E_F3) / median(E_F0 - E_F2)`, both relative to F1; between if 0 < s < 1) and how
close to F2 (1 - s, median W), per seed and over the seeds.

## 6. Decisions taken without asking (unattended run, most conservative reading)

1. **Base and number.** Branch from `origin/master` cfb9314 (equal to the local `master` HEAD; the
   uncommitted edits of `README.md` and `docs/12` in the main checkout belong to another session
   and are neither touched nor included). Study document number 28: the highest existing was 27.
2. **Which parameters.** Exactly the seven named (critical moisture, nu_max, five GAB). The other
   epistemic p2 inputs of the Sobol config (heat loss, gas-heater efficiency, cp_vapour) stay
   nominal: not named in the request.
3. **Draws.** Scrambled Sobol' with seed 20261024, a balanced sequence of 32 = 2^5 points, the
   first 20 valid ones in sequence order (a sequence of 20 points is not balanced). Validity = the
   isotherm check of Befunde 105/108; rejected points would be counted (none).
4. **Scenario subset.** Four per type at `round(linspace(0, n - 1, 4))`: deterministic, spread over
   the type, includes the slow and the fast inlet-temperature disturbance; no selection by outcome.
5. **Initial condition and pre-roll.** As Befund 105's robustness variant the truth first runs
   6 h at the nominal initial command (common to all variants, computed once per draw). Then a
   **pre-roll of 6 h** at the first disturbance sample with the controller and the bias filter in the
   loop; energy and violations count only over the 24 h scenario. Reason: with the mismatch the
   steady moisture of the truth at a command differs from the model's by about 0.01-0.02 kg/kg
   (probe before the study, scratch, not an artifact), so the truth can start above the limit and
   every margin would fail on the start alone, before any feedback acts; the plant
   runs continuously, so the controller has been operating before the evaluation window.
6. **Model prediction.** "Model prediction" is read as the open-loop output of the nominal model
   under the same inputs (simulated alongside), the standard form of an output-bias correction; it
   makes "ideal measurement without mismatch = R1 exactly" hold by construction.
7. **Feedback grid and filter details.** Measurements, soft-sensor estimates and filter updates on a
   25 s grid (the sample time of the sensor data of Befunde 94 and 109 and of the soft sensor). At
   5 s the filter would average five times as many independent outliers per time constant, i.e. a
   smaller noise of b: 25 s is the more conservative choice for F2. First filter value = first
   error; b is not clipped; the controller uses b at each interval start.
8. **Sensor model of F2.** The headroom process config switches the measurement off (controllers
   saw the true state); the existing `MeasurementConfig` is taken from the process file itself and
   run at 25 s. Outliers on every sample with probability 0.25, per-pair random streams derived from
   (noise seed, draw, scenario): a pair sees the same noise at every margin (common random numbers,
   needed for a meaningful bisection).
9. **Soft sensor F3.** "Existing pipeline" = Befund 94 arm N with its process config (lab GAB
   placeholder with the gas-velocity projection, fingerprint `bddca37f716f5cfc`); its material
   parameters are those of the controller's nominal model (checked by the training script), so the
   training uses only data of the nominal model. Only the moisture head is trained (the composite
   network trains one independent network per target; the outlier study did the same). In the loop
   its inputs are the 25 s block means of the realised commands (inlet temperature including the
   scenario's offset), spray rate, nozzle pressure (the fixed point of the line) and the
   disturbances; the estimate of block j is the h = 1 prediction from blocks j-32 ... j-1. **By the
   design of the composite network the moisture head sees only these exogenous channels**
   (commands and disturbances), not the measured bed temperature or energy (Befund 26, "Was das
   kostet"): it cannot observe the mismatch of the truth and therefore estimates the nominal model's
   moisture. Not changed (the request names the existing pipeline); consequence in section 9.
10. **Margins.** Never negative (lower end 0 as Befund 105). One bisection per pair; the variant
    margins as order statistics of the per-pair margins under the monotonicity assumption, each
    verified on all pairs. A pair that holds at no margin up to 0.12 counts as infinite.
11. **Normalisations.** Energies per pair relative to F1 at its margin of the SAME criterion; V and W
    divided by E_F1 (the reference of all energy figures of the request); V divided by E_F0 and by
    E_F2 reported next to it. Class boundaries: exactly 1 % and exactly 5 % belong to V1.
12. **Attribution of the F0 margin** (request: "as in Befund 108"): Befund 108 maps parameters to
    questions; with 20 draws no Sobol index can be estimated, so Spearman correlations and the
    variance explained by the two question groups (OLS on the unit-cube coordinates) are reported,
    descriptively.
13. **Solver failures** hold the previous command and are counted (as Befund 105), in the pre-roll
    and in the scenario separately.

## 7. Runs

Artifacts (all `git/dirty=False`):

* `artifacts/runs/sensor_value_soft_sensor_v1.json` -- training of the three soft sensors (script
  commit `e2f8ab1`; checkpoints in `artifacts/checkpoints/sensor_value/`, gitignored, SHA-256 in
  the record and checked by the main run). Training 3.9 min.
* `artifacts/runs/sensor_value_v1.json.gz` -- main run (driver commit `a0cb2b7`), 693.6 min wall
  (2026-10-23 14:04 to 2026-10-24 01:37), 14 workers: warm-up 1.7 min, per-pair searches 9.5 h,
  verification 2.0 h; 1476 jobs of 20 or fewer pairs (search and verification).
* `artifacts/runs/sensor_value_v1_analysis.json` -- pre-registered analysis (script committed in
  `e2f8ab1`, before every run).
* `artifacts/runs/sensor_value_v1_f0_candidate_check.json` -- check of the raise rule of section 8.2
  (script `scripts/check_sensor_value_f0_candidate.py`, committed in `c73b149` before its run, reads
  the verification runs kept in the gitignored cache of the main run).

**Budget (rule of the request).** Probe before the start (2026-10-23, scratch, not an artifact):
one full-length job (one draw, 20 pairs, 6 h pre-roll + 24 h) per variant at margin 0.01: F0
164 s, F1 308 s, F2 320 s, F3 257 s. Extrapolated with about ten bisection rounds per pair and two
verifications: about 150 core-hours, 11-12 h on 14 workers, below 30 h -- **no cut** (20 draws,
20 scenarios). Actual 11.6 h.

**Probe finding before the start (disclosed, design unchanged).** On the nominal truth (no
mismatch) the soft sensor's bias settled at -0.08 to -0.15 kg/kg in a short closed-loop probe
(three scenarios, margin 0.01, scratch): it underestimates the moisture at the operating point of
this line. The windows are those of the pipeline (test), so this is the estimator, not the
interface. The pre-specified design (h = 1 estimate, no tuning) was kept.

## 8. Results

### 8.1 Margins (kg/kg, path condition at every 5 s sample of 24 h, 400 pairs)

| variant | (ii) >= 95 % (main) | pairs holding | max violation at (ii) | (i) all | per-pair median |
|---|---|---|---|---|---|
| F0 none | 0.05938 | 382 | 0.0114 | 0.06641 | 0.01172 |
| F1 ideal | 0.00469 | 384 | 0.0032 | 0.00586 | 0.00039 |
| F2 inline sensor | 0.00449 | 382 | 0.0045 | 0.00664 | 0.00107 |
| F3 soft sensor, seed 1 | 0.07637 | 380 | 0.0542 | 0.08867 | 0.06133 |
| F3 seed 2 | 0.07578 | 380 | 0.0294 | 0.08242 | 0.05566 |
| F3 seed 3 | 0.08262 | 381 | 0.0294 | 0.08828 | 0.07227 |

Every pair converged below the cap 0.12; no solver failure in any run; no draw was rejected by the
isotherm check. F2's realised outlier share per pair 0.229-0.268. F1 and F2 need about the R1
margin of Befund 105 (0.0066) or less; F2's (ii) margin lies below F1's by less than the bisection
tolerance. F0 needs about thirteen times the F1 margin, F3 more than F0.

### 8.2 Energy relative to F1 (per pair, median and range over the 400 pairs)

| variant | criterion (ii) energy | (ii) CO2 | criterion (i) energy |
|---|---|---|---|
| F0 | +13.36 % (-2.19 .. +41.88 %) | +12.99 % | +15.85 % (+0.11 .. +45.29 %) |
| F2 | -0.05 % (-0.35 .. +0.04 %) | -0.04 % | +0.18 % (+0.11 .. +0.38 %) |
| F3 seed 1 | +17.62 % (-8.39 .. +44.88 %) | +16.64 % | +33.65 % (+1.07 .. +68.53 %) |
| F3 seed 2 | +12.58 % (-3.84 .. +43.81 %) | +11.73 % | +17.62 % (-0.33 .. +49.93 %) |
| F3 seed 3 | +12.07 % (-3.09 .. +41.01 %) | +11.51 % | +18.64 % (+1.34 .. +50.40 %) |

Median energy per 24 h scenario at (ii): F1 26.22 kWh, F2 26.21, F0 29.77, F3 29.54-31.48 kWh.

**Value of the measurement** `V = (E_F0 - E_F2) / E_F1`, criterion (ii): **median 13.40 %**
(-2.10 .. +41.93 %); CO2 13.04 %; normalised by E_F0 11.82 %, by E_F2 13.41 %. Criterion (i):
15.67 % (-0.02 .. +45.02 %). The same in every scenario type (medians 13.35-13.45 %): the value
does not come from the disturbances but from the unknown parameters. Per draw (median over its
scenarios) it ranges from -1.53 % (draw 3, the wettest truth, whose pairs F0 partly leaves violated)
to 27.39 % (draw 6, a drier truth: F0 must keep the family-wide margin, F1/F2 relax the bound by the
measured negative bias).

**Pre-registered reading: V2** (median V = 13.40 % > 5 %).

**Check of the raise rule.** At its first (ii) candidate 0.04785 F0 held in 379 pairs, one short of
380 (a non-monotone pair); the driver raised it to the next per-pair margin 0.05938 (382 hold). This
raises F0's energy by 3.69 % in the median (2.78-5.01 %) and therefore V. At the first candidate V
would be **9.47 %** in the median (-6.73 .. +37.38 %): still V2. (`..._f0_candidate_check.json`.)

### 8.3 Soft sensor (F3) against F0 and F2, criterion (ii)

| seed | W = (E_F3 - E_F2)/E_F1, median (range) | (E_F0 - E_F3)/E_F1, median | captured share s | position |
|---|---|---|---|---|
| 1 | 17.72 % (-8.35 .. 44.94 %) | -2.68 % | -0.20 | at F0 or worse |
| 2 | 12.65 % (-3.81 .. 43.79 %) | +1.67 % | 0.12 | between F0 and F2 |
| 3 | 12.10 % (-3.12 .. 41.00 %) | +1.62 % | 0.12 | between F0 and F2 |

Over the seeds: median W 12.65 % (12.10-17.72 %), captured share 0.12 (-0.20 .. 0.12). Two of three
seeds lie formally between F0 and F2, but at 12 % of the way from F0; one is worse than F0; under
criterion (i) all three are worse than F0 (s -0.94, -0.12, -0.18). Feedback error of F3 against the
true moisture (RMSE over the 24 h per pair) median 0.027 kg/kg (0.0006-0.123). The soft sensor does
not replace the measurement here: it saves at most about an eighth of what the inline sensor saves,
and only with two of three training seeds.

### 8.4 Which draws drive the F0 margin

Per-draw F0 margin (largest over its 20 scenarios): 0 in five draws, up to 0.0664 (draw 3:
critical moisture 0.278, nu_max 0.413). Spearman correlation with the parameters: critical moisture
+0.72, nu_max -0.60; GAB parameters between -0.16 and +0.36. Variance explained (OLS on the unit
coordinates): **drying kinetics (critical moisture, nu_max) R^2 0.91** (adjusted 0.90); isotherm
(five GAB parameters) 0.24 (adjusted -0.03); all seven 0.93 (adjusted 0.89). The five draws with
the largest margins (3, 11, 15, 4, 19) all have a critical moisture of 0.23-0.30 and nu_max of
0.41-0.62. The steady model error (mean F1 bias at the end of the pre-roll per draw) spans -0.046 to
+0.059 kg/kg. This is the ranking of Befund 108 (drying kinetics first, isotherm last), now for the
margin of a feedback-free controller.

## 9. Placeholders, caveats, open points

**Placeholders.** Quality limit 0.15; filter time constant 30 min; sensor model (PT1 10 s, 0.005)
and outlier settings (p = 0.25, t3, scale 10); parameter bands of the Sobol config (in particular
critical moisture 0.15-0.30 and nu_max 0.4-0.8); scenario amplitudes of Befund 105; the whole process
config; gas-heater efficiency 0.80; start margin 0.01; pre-roll 6 h (study setting).

**Caveats.**

1. **The value is a value against an unknown parameter band.** F0 must hold the limit for the
   whole band with one margin; the size of V scales with the width of the band of the drying
   kinetics, which is a placeholder. A drying experiment that narrows critical moisture and nu_max
   (the data request of Befund 108) lowers V directly. V is the value of the measurement for a plant
   whose drying kinetics are known only to that band.
2. **F0 is a static correction only.** No integral action, no adaptation; a learning or adaptive
   controller without a moisture measurement has to infer the mismatch from other channels, which
   this study does not test.
3. **The soft sensor is structurally blind to the mismatch** (section 6, item 9): the moisture head
   of the composite network sees only commands and disturbances, and it underestimates the moisture
   of this line's operating point by several 0.01 kg/kg already on the nominal model. F3 therefore
   says nothing about a soft sensor that uses the measured bed temperature or exhaust state; such
   a variant (or an observer on the grey-box model) is the obvious next test, not run.
4. **Non-monotone pairs.** F0's (ii) candidate failed by one pair; the raise rule inflates V by
   about 4 points (section 8.2); both values are V2.
5. One lab line with the 4-state model; 20 draws and 20 scenarios; one realisation of the sensor
   noise per pair; F2's (ii) margin below F1's lies within the bisection tolerance; the per-pair
   margins assume monotonicity (checked by the verification at both criteria).

**Open.** Whether the moisture limit binds on the plant (Befund 105, section 2) and whether the
product moisture is measured inline there (question to the application partner); a soft-sensor or
observer variant with the measured bed temperature; V for a narrower band after drying experiments.
