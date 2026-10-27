# 29 -- Value of the moisture feedback: periodic laboratory samples and soft sensors with process responses

Status: design fixed and committed before the runs (sections 1-6); runs and results in sections
7-9, added after the runs (numbers only from the artifacts named there). Follow-up of Befund 111
(`docs/28_sensor_value.md`). Branch `sensor-value-2` (from `origin/master` 8f5db3e); the finding
number is 112 (lab book).

## 1. Question

Befund 111 measured the value of a moisture feedback against no feedback: without feedback the
bias-corrected RTO needs a margin of 0.059 instead of 0.0045 kg/kg and 13.4 % more energy (median
over the pairs); a GRU soft sensor from commands and disturbances only captured at most an eighth of
it. The decision of the project lead on Befund 111 named two open questions, answered here:

1. Does a **periodic laboratory sample** (hours apart, analysed with a delay) capture the value, or
   does it take an inline sensor?
2. Does a **soft sensor that sees measured process responses** (bed temperature, exhaust
   temperature, exhaust humidity) capture it, and does training it with **domain randomisation** of
   the uncertain parameters in the simulation help?

Measure: the share of the value each variant captures, `A(X) = (E_F0 - E_X) / (E_F0 - E_F2)`, per
pair (draw, scenario), with the energies of F0 (no feedback) and F2 (inline sensor) of Befund 111.

## 2. What is identical to the predecessor (read, not changed)

Truth (grey-box model of Befund 105, lab line, limit 0.15 kg/kg as path condition, tolerance 1e-4),
the 20 parameter draws of the truth (critical moisture, nu_max, five GAB parameters; Sobol' seed
20261024), the 20 scenarios (family "observed", four per type), the controller (R1 on the nominal
model with `X_model + b <= 0.15 - margin`), warm-up 6 h and pre-roll 6 h, the feedback grid of
25 s, the bias filter of F1-F3 (first order, tau 30 min), the per-pair bisection and the margins of
criteria (i) all pairs and (ii) at least 95 % of the pairs (main), the noise seed. All are read from
`conf/sensor_value/sensor_value.yaml` (unchanged; its fingerprint `734320f5c5f18c27` is checked
against the predecessor's artifact). The driver imports the search and verification stages of
`scripts/run_sensor_value.py`.

**Reference variants F0, F1, F2 are not recomputed:** their per-pair records at the margins of
both criteria are taken from `artifacts/runs/sensor_value_v1.json.gz` (SHA-256 recorded). Control:
10 pairs (`round(linspace(0, 399, 10))` of the draw-major order) are recomputed for F0, F1 and F2 at
the predecessor's criterion-(ii) margins and every field of the record is compared with the
artifact (per-interval arrays after the artifact's rounding).

## 3. New feedback variants

Settings in `conf/sensor_value/sensor_value_2.yaml`, fixed before the runs and not tuned.

* **F4 periodic laboratory sample** (`sensor_value.feedback.LabSample`), interval h in {1, 4, 8} h,
  analysis delay 30 min, Gaussian error sigma 0.003 kg/kg **(placeholder)**. The sample is the
  true moisture at the sample instant plus the error; when its result arrives the bias is updated
  `b <- b + 0.5 (y_sample - (x_model + b))` with the model prediction and the bias of the sample
  instant, and held in between (no filter). The bias starts at 0.
* **F5 soft sensor with process responses**: the moisture head of a GRU (pipeline of Befund 94 arm N
  and of F3: `DataSpec.new_study()` preset version 2, `n_train` 64, `max_epochs` 200, validation
  pool, operating mode `SOFT_SENSOR`) whose inputs are the commands and disturbances PLUS measured
  bed temperature, exhaust temperature and exhaust humidity, through the existing sensor model.
  Three training variants, three seeds each (master seeds 1, 2, 3):
  * **F5-nom**: training data of the nominal model only (the F3 pool of the same seed);
  * **F5-DR**: domain randomisation -- every run of the pool simulated with its own draw of the
    same seven parameters from the same bands, other Sobol' seed (20261026), disjoint from the 20
    test draws (test; smallest distance in the unit cube recorded);
  * **F5-DR-oY**: as F5-DR, without the exhaust humidity (for a plant that does not measure it).
  In the loop the bias filter is the one of F1-F3.

The existing pipeline is only extended: `WindowConfig.measured_input_names` (default empty) appends
measured output channels to the known inputs; `build_datasets(..., window=...)` takes such a window.
Every existing configuration behaves bit for bit as before (test against digests of the windows of
three data configurations and of the closed-loop records of F0, F1, F2, computed on 8f5db3e before
the change: `tests/test_sensor_value_2.py`).

## 4. Metrics

Per variant (F5: per seed) and criterion the margin and the pairs holding; per pair at that margin
`A_X(p)`, the energy and CO2 relative to F1 and the feedback error. Main statistic: median of
`A_X(p)` over the 400 pairs, with range. Because `E_F0 - E_F2` is small or negative in some pairs
(predecessor: down to -2.1 % of E_F1, 20 pairs <= 0), two sensitivity statistics are reported with
it (S1: median over the pairs with `E_F0 - E_F2 >= 1 %` of E_F1; S2: ratio of the medians, the
captured share of the predecessor).

## 5. Pre-registered reading

Fixed in the docstring of `scripts/analyze_sensor_value_2.py`, committed before the first run of the
study. Criterion (ii), main statistic, threshold 0.8 inclusive.

* **La**: if F4 with 4 h reaches A >= 0.8: "periodic laboratory samples suffice; an inline sensor
  has little added value for this line" (simulation, placeholder lab settings). Otherwise the
  longest tested interval with A >= 0.8, or "none".
* **Lb**: if the median over the seeds of the per-seed median A of F5-DR reaches 0.8: "a soft sensor
  learned with domain randomisation in the simulation replaces the measurement (in simulation)";
  whether every seed reaches it is reported. Also the value of the randomisation
  (`A(DR) - A(nom)`) and of the exhaust humidity (`A(DR) - A(DR-oY)`) per seed with median and range;
  a direction is claimed only if the paired t-test over the seeds
  (`evaluation.paired_resolution`, 95 %) resolves it.
* A reading whose main statistic and a sensitivity statistic lie on different sides of 0.8 is marked
  "sensitive to the handling of small denominators".

## 6. Decisions taken without asking (unattended run, most conservative reading)

1. **Base and number.** Branch from `origin/master` 8f5db3e (after the merge of Befund 111). Study
   document 29: the highest existing was 28.
2. **"Model prediction" of the F4 update** is read as the prediction of the CORRECTED model,
   `x_model + b`, at the sample instant. Read literally with the uncorrected open-loop model output,
   the rule `b <- b + 0.5 (y - x_model)` adds half the (constant) model error at every sample and
   grows without bound; with the corrected prediction it is exponential smoothing of the bias with
   gain 0.5, which converges to the model error.
3. **F4 sampling phase.** First sample at the first feedback instant of the closed loop (25 s after
   the start of the pre-roll), then every h hours; the result arrives 30 min later and updates the
   bias at the end of that 25 s block; the controller reads the bias at each 5-min interval start,
   as for every variant. With 6 h of pre-roll, 1 h sampling has had six results, 4 h two and 8 h one
   before the evaluation window starts. The lab error is drawn per pair from its own generator
   (common random numbers over the margins).
4. **Exhaust temperature.** In the grey-box model the bed is well mixed and the exhaust temperature
   is the bed temperature minus the duct loss (`exhaust.outlet_duct_heat_loss_k`, 0.0 in the
   config): the two channels carry the same quantity with independent sensor noise. Kept as named in
   the request; consequence in section 9.
5. **Sensor model of the responses.** "The existing sensor model" = `GreyBoxFluidBedGranulator.measure`
   with the `MeasurementConfig` of the process config (PT1 10 s, relative Gaussian noise 0.005,
   absolute temperature noise 0.2 K), at 25 s. Bed temperature as in the training data (output
   channel 1); exhaust temperature and humidity as the temperature and the moisture-type channel of a
   second, independent sensor instance (`sensor_value.responses`). In the training data the bed
   temperature is the pool's own measurement; exhaust temperature and humidity are measured with a
   seed derived from the master seed. In the loop, per-pair streams derived from (noise seed, draw,
   scenario), the same for every F5 variant (paired).
6. **Responses as inputs.** The measured responses enter history and future inputs of the window;
   in the future part they are realised measurements, not forecasts. The soft sensor uses h = 1
   (estimate of the current block from the current block's inputs, as F3); the training loss over
   the horizon of 36 steps is unchanged (each decoder step sees the responses up to that step).
7. **Domain randomisation.** One parameter draw per run of the pool (128 runs: 64 training, 32
   validation, 32 test); same excitation schedule, initial states, split and model seed as the
   nominal pool of the same master seed, so F5-nom, F5-DR and F5-DR-oY of one seed differ only in
   the plant and the inputs. "Kinetics and isotherm parameters" = exactly the seven parameters of the
   test draws (critical moisture, nu_max, five GAB). Draws: scrambled Sobol', seed 20261026, 256 =
   2^8 points, the first 128 valid in order (isotherm check as for the test draws). The DR pool is
   validated against the noise-free moisture of its own test runs and of the nominal pool.
8. **Seeds.** F5 master seeds 1, 2, 3 as F3, so F5-nom of seed s has exactly the F3 pool of seed s.
9. **A and its aggregation** as in sections 4 and 5; Lb aggregates over the seeds by the median of
   the per-seed medians.
10. **Artifact size.** The run artifact holds per-pair scalars only (no per-interval arrays), to stay
    below 5 MB; the per-interval arrays are kept only for the control pairs.
11. **Budget.** Probe of one full-length job per variant (draw 0, 20 scenarios, margin 0.01),
    extrapolated with the predecessor's ratio of actual run time to its probe; the cuts of the
    request (first F4 1 h, then F5 seeds to 2) are applied above 30 h and recorded in the artifact.
12. **Step 0** (before everything else): the checkpoints of the F3 soft sensors of Befund 111 were
    lost; they were retrained with the unchanged committed script (`sensor_value_soft_sensor_v1_retrain`)
    and compared with `sensor_value_soft_sensor_v1.json` field by field
    (`scripts/compare_sensor_value_soft_sensor_retrain.py`). All checkpoints of this study are also
    copied to `C:/twinlink_runs/sensor-value-2/`.

## 7. Runs

Artifacts (all `git/dirty=False`):

* `artifacts/runs/sensor_value_soft_sensor_v1_retrain.json` and
  `sensor_value_soft_sensor_v1_retrain_comparison.json` -- step 0 (section 6, item 12): the three F3
  soft sensors retrained at 8f5db3e (190 s); **every compared field of every seed (184, 206 and 212
  fields) equals the original record exactly**; only checkpoint path, SHA-256 and wall times differ,
  by construction. The training code is unchanged between e2f8ab1 and 8f5db3e (`git diff`: none).
  Lab book: dated addendum to Befund 111.
* `artifacts/runs/sensor_value_2_soft_sensor_v1.json` -- training of the nine F5 soft sensors (script
  `scripts/train_sensor_value_2_soft_sensor.py`, commit 6246d34), 25.4 min. DR draws: no draw
  rejected, 0 points in common with the 20 test draws, smallest distance in the unit cube 0.236.
  Checkpoints in `artifacts/checkpoints/sensor_value_2/` (gitignored, SHA-256 in the record and
  checked by the main run), copied to `C:/twinlink_runs/sensor-value-2/checkpoints/`.
* `artifacts/runs/sensor_value_2_v1.json.gz` -- main run (driver `scripts/run_sensor_value_2.py`,
  commit 78b90d1), 1.05 MB, 1034.2 min wall (2026-10-26 11:07 to 2026-10-27 04:21), 14 workers:
  warm-up 1.6 min, control 14.6 min, per-pair searches 14.2 h, verification 2.7 h.
* `artifacts/runs/sensor_value_2_v1_analysis.json` -- pre-registered analysis (script committed in
  091fcd4, before every run of the study).

**Control of the reference variants.** F0, F1 and F2 recomputed on the 10 control pairs at the
predecessor's criterion-(ii) margins: **10 of 10 pairs identical for each, largest absolute
difference 0** over every field of the record. The references taken from
`sensor_value_v1.json.gz` (fingerprints, draws and scenarios checked) are therefore reproducible
with the extended code.

**Budget (rule of the request).** Probe before the start (2026-10-26, scratch outside the repository,
`C:/twinlink_runs/sensor-value-2/probe/`, commit 78b90d1, clean tree): one full-length job (draw 0,
20 pairs, 6 h pre-roll + 24 h) per variant at margin 0.01: F4 384-386 s (each interval), F5 448-488 s.
Extrapolated with the predecessor's ratio of actual run time to its probe (11.6 h for a probe sum of
F0 + F1 + F2 + 3 F3, times 20 draws / 14 workers: factor 18.7): all twelve variants 39.3 h; without
F4 1 h 36.4 h; with the F5 seeds also cut to 2: 26.4 h. **Cuts (in this order, recorded in the
artifact as `budget_dropped`): F4 1 h; seed 3 of F5-nom, F5-DR and F5-DR-oY.** The run took 17.2 h:
the extrapolation overestimated it (reason not examined); the cuts were decided on the
extrapolation before the start and are kept.

**Probe finding before the start (disclosed, design unchanged).** At margin 0.01 on draw 0 every F4
interval held the limit in all 20 pairs; the F5 soft sensors violated it by up to 0.040-0.180 kg/kg
(eight of nine; F5-nom seed 3 none). No setting was changed.

## 8. Results

All figures for the variants of this study come from `sensor_value_2_v1_analysis.json`; F0, F1, F2
are the predecessor's records (section 2). Criterion (ii) is the main criterion.

### 8.1 Margins (kg/kg, path condition at every 5 s sample of 24 h, 400 pairs)

| variant | (ii) >= 95 % (main) | pairs holding | max violation at (ii) | (i) all |
|---|---|---|---|---|
| F0 none (predecessor) | 0.05938 | 382 | 0.0114 | 0.06641 |
| F1 ideal (predecessor) | 0.00469 | 384 | 0.0032 | 0.00586 |
| F2 inline sensor (predecessor) | 0.00449 | 382 | 0.0045 | 0.00664 |
| F4 lab sample, 4 h | 0.00859 | 382 | 0.0057 | 0.01387 |
| F4 lab sample, 8 h | 0.01660 | 382 | 0.0071 | 0.02188 |
| F5-nom seed 1 / 2 | 0.06953 / 0.06602 | 380 / 380 | 0.0345 / 0.0350 | 0.07773 / 0.07559 |
| F5-DR seed 1 / 2 | 0.06973 / 0.07051 | 383 / 380 | 0.0443 / 0.0397 | 0.07656 / 0.07676 |
| F5-DR-oY seed 1 / 2 | 0.07285 / 0.06367 | 381 / 380 | 0.0507 / 0.0637 | 0.08145 / 0.07539 |

Every pair converged; no solver failure in any variant. The F4 margins lie between F2 and F0
(4 h about twice the F2 margin, 8 h about four times). **Every F5 margin lies above F0's**: with the
soft sensor the controller needs a larger margin than without any feedback.

### 8.2 Energy and captured share, criterion (ii)

Energy and CO2 relative to F1 per pair, median (range) over the 400 pairs; A per pair, median
(range); S1 = median of A over the pairs with `E_F0 - E_F2 >= 1 %` of E_F1 (380 pairs; 20 pairs have
a denominator <= 0); S2 = ratio of the medians. Feedback error: RMSE of the feedback against the
true moisture over the 24 h per pair, median.

| variant | energy vs F1 | CO2 vs F1 | A median (range) | S1 | S2 | feedback error |
|---|---|---|---|---|---|---|
| F0 (predecessor) | +13.36 % (-2.19 .. +41.88 %) | +12.99 % | 0 by definition | | | |
| F2 (predecessor) | -0.05 % (-0.35 .. +0.04 %) | -0.04 % | 1 by definition | | | |
| F4 4 h | +0.88 % (+0.01 .. +3.46 %) | +0.90 % | **0.934** (0.709 .. 2.386) | 0.931 | 0.941 | 0.0028 |
| F4 8 h | +2.64 % (+0.93 .. +8.38 %) | +2.63 % | **0.805** (0.349 .. 3.975) | 0.803 | 0.813 | 0.0026 |
| F5-nom seed 1 | +14.01 % (-6.30 .. +43.37 %) | +13.33 % | -0.080 (-6.13 .. 1.00) | -0.080 | -0.060 | 0.0171 |
| F5-nom seed 2 | +10.51 % (-7.31 .. +38.14 %) | +9.83 % | 0.209 (-4.46 .. 1.10) | 0.217 | 0.247 | 0.0242 |
| F5-DR seed 1 | +17.84 % (-6.80 .. +61.45 %) | +16.98 % | -0.264 (-6.61 .. 6.01) | -0.275 | -0.332 | 0.0144 |
| F5-DR seed 2 | +14.60 % (-5.19 .. +52.57 %) | +13.99 % | -0.103 (-4.80 .. 2.73) | -0.064 | -0.043 | 0.0165 |
| F5-DR-oY seed 1 | +21.27 % (-8.85 .. +70.31 %) | +20.86 % | -0.666 (-8.97 .. 8.99) | -0.681 | -0.849 | 0.0164 |
| F5-DR-oY seed 2 | +12.77 % (-9.53 .. +49.54 %) | +12.33 % | 0.059 (-6.68 .. 2.05) | 0.068 | 0.096 | 0.0177 |

Median energy per 24 h scenario at (ii): F4 4 h 26.48 kWh, F4 8 h 26.99 kWh, F5 28.71-34.58 kWh
(predecessor: F1 26.22, F2 26.21, F0 29.77 kWh). A above 1 in single pairs is a pair whose
denominator `E_F0 - E_F2` is small; the medians and both sensitivity statistics lie on the same side
of 0.8 for every variant.

Criterion (i) (references F0 0.06641, F1 0.00586, F2 0.00664): A median F4 4 h 0.894, F4 8 h 0.775;
F5 between -0.951 and -0.048 (all six below 0). Energy vs F1: F4 4 h +1.89 %, F4 8 h +3.74 %,
F5 +17.50 to +30.40 %. Under (i) the 8 h interval falls below 0.8; the reading is fixed on (ii).

### 8.3 Pre-registered readings

**La: holds.** F4 with 4 h reaches A = 0.934 >= 0.8 (S1 0.931, S2 0.941, not sensitive): "periodic
laboratory samples suffice; an inline sensor has little added value for this line" (simulation,
placeholder lab settings). The 8 h interval also reaches it under (ii) (0.805; S1 0.803, S2 0.813),
narrowly; under (i) it does not (0.775). The 1 h interval was cut by the budget rule and not run.
In energy: the 4 h sample costs +0.88 % against the ideal feedback where the inline sensor costs
-0.05 % and no feedback +13.36 %.

**Lb: does not hold.** Median over the seeds of the per-seed median A of F5-DR: **-0.183**
(-0.264 .. -0.103); no seed reaches 0.8. "The domain-randomised soft sensor does not replace the
measurement." Not sensitive (S1 and S2 negative for both seeds). Both F5-DR seeds lie below F0:
the soft sensor costs more energy than no feedback.

* Value of the randomisation, `A(DR) - A(nom)` per seed: -0.184 (seed 1), -0.312 (seed 2); median
  -0.248. Paired t-test over the seeds: p = 0.16, **not resolved** (two seeds). No direction is
  claimed; both seeds point against the randomisation.
* Value of the exhaust humidity, `A(DR) - A(DR-oY)` per seed: +0.402 (seed 1), -0.161 (seed 2);
  median +0.120. p = 0.74, **not resolved**; the seeds disagree in sign.

### 8.4 Soft sensors outside the loop (training record)

Test error at h = 1 (median over the test runs of the per-run RMSE, kg/kg), seeds 1 / 2 / 3, from
`sensor_value_2_soft_sensor_v1.json`:

| variant | excited, nominal test runs | excited, randomised test runs | constant-run holdout, nominal |
|---|---|---|---|
| F5-nom | 0.0093 / 0.0153 / 0.0010 | 0.0135 / 0.0146 / 0.0070 | 0.0107 / 0.0190 / 0.0010 |
| F5-DR | 0.0178 / 0.0284 / 0.0071 | 0.0186 / 0.0254 / 0.0069 | 0.0245 / 0.0321 / 0.0072 |
| F5-DR-oY | 0.0181 / 0.0215 / 0.0070 | 0.0201 / 0.0203 / 0.0058 | 0.0234 / 0.0245 / 0.0048 |

The pooled RMSE over all test samples is 0.049-0.134 kg/kg for every variant and seed (dominated by
very wet runs, as for F3). Domain randomisation does not lower the error even on the randomised
test runs (seeds 1 and 2 higher than F5-nom, seed 3 equal). In the loop the feedback error of F5
(0.014-0.024 kg/kg, section 8.2) is five to nine times that of F4 (0.0026-0.0028 kg/kg); the
margin search has to cover it.

## 9. Placeholders, caveats, open points

**Placeholders.** Laboratory error sigma 0.003 kg/kg, delay 30 min, update gain 0.5 (fixed before
the run, not tuned); everything listed in `docs/28_sensor_value.md` section 9 (limit 0.15, filter
tau 30 min, sensor model, outlier settings, parameter bands, scenario amplitudes, process config,
gas-heater efficiency, start margin, pre-roll 6 h).

**Caveats.**

1. **Exhaust temperature = bed temperature in the model** (section 6, item 4). The two temperature
   inputs of F5 carry the same quantity with independent noise; F5 therefore tests bed temperature
   plus exhaust humidity, not three independent responses. On a plant with a real bed-to-exhaust
   difference the exhaust temperature could carry information the model cannot represent.
2. **Lb rests on two seeds per F5 variant** (budget cut). The F5-DR reading is clear on both seeds
   (A < 0 for both, far from 0.8); the two comparisons (randomisation, exhaust humidity) are not
   resolved.
3. **F5 is the existing pipeline extended by measured inputs, not tuned** (n_train 64, h = 1, GRU,
   training on the 36-step horizon). It says nothing about an observer that uses the grey-box model's
   mass balance with the measured exhaust humidity, nor about a soft sensor trained on more runs or
   on a loss at h = 1; neither was run.
4. **F4 assumes an unbiased laboratory analysis** with Gaussian error and a sample of the product
   stream at a known instant; sampling bias, a variable delay and missing samples are not modelled.
   F4's sampling phase and its two (4 h) and one (8 h) results before the evaluation window
   (section 6, item 3) favour the bias at the start of the 24 h.
5. The value of every feedback is measured against the placeholder band of the drying kinetics
   (Befund 111, caveat 1); a narrower band lowers both F0's margin and the share any feedback can
   capture in absolute terms.

**Open.** The laboratory practice at the plant (interval, delay, analysis error; question to the
application partner) to replace the placeholders; an observer on the grey-box model with the measured
exhaust humidity; F4 1 h and the third F5 seeds (cut by the budget rule; the actual run time left
room for them).
