# 29 -- Value of the moisture feedback: periodic laboratory samples and soft sensors with process responses

Status: design fixed and committed before the runs (sections 1-6); runs and results follow in
sections 7-9 (numbers only from the artifacts named there). Follow-up of Befund 111
(`docs/28_sensor_value.md`). Branch `sensor-value-2` (from `origin/master` 8f5db3e); the finding
number is assigned by the managing session (lab book entry "Befund NN").

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
