# 40 -- Calibration of the grey-box model at plant A: identifiability first

Status: design, code and reading committed before the run on plant data; results are added after
the run (numbers only from the artifacts). Branch `plant-calibration` (worktree
`C:/twinlink_work/pcal`, from `origin/master` c4ecbfe). Document number 40 and finding number 123
(lab book). Quick variant of the request: steps
1 (identifiability) and 2 (stationary calibration with cross-check and bias comparison) only.

Confidentiality as in Befund 117/118: roles only; raw exports only through
`TWINLINK_PLANT_DATA_PROD1` / `TWINLINK_PLANT_DATA_PROD2`; channel tags only in the git-ignored role
file; absolute per-segment values only in the git-ignored local report. Day 1 = product 1, day 2 =
product 2 (day and product are confounded).

## 1. Question

Which parameters of the plant-near grey-box model (Befund 117) can be determined from the
stationary bed and exhaust temperatures of the two measured days, and does a stationary
calibration of at most three of them reproduce the other day better than a constant bias?

## 2. Design

Package `src/twinlink_ml/plant_calibration/` (model wrapper, identifiability, calibration), settings
`conf/plant_calibration/plant_calibration.yaml`, driver `scripts/run_plant_calibration.py`, analysis
`scripts/analyze_plant_calibration.py` (reading in its docstring), tests
`tests/test_plant_calibration.py` (synthetic data only). Data, stationary segments (four 60-minute
windows per day), roles and the plant-near parametrisation are those of Befund 117
(`conf/plant_scale/plant_scale.yaml`, `src/twinlink_ml/plant_scale/`, only read). Measured inputs
(process-air flow, supply-air temperature, spray rate, intake temperature) drive the model
directly; the heater is not modelled. Reference: solid fraction 0.65 on both days, nominal
kinetics.

* **Candidates.** Bed heat-loss UA (`geometry.heat_loss_ua_w_per_k`); bed-to-exhaust distance per
  product (K; switch of Befund 117, default off); heat transfer gas -> wall between bed and exhaust
  sensor (W/K; the `heat_loss` term C1 of Befund 117 -- the model has no gas-particle heat transfer
  coefficient, the gas leaves the ideally mixed bed at the bed temperature; interpretation decided
  by the project lead, 2026-11-13); effective heat capacity (`material.cp_solid_j_per_kg_k`; no
  steady effect by construction); critical moisture and nu_max (kinetics, never calibrated). The
  two bed-to-exhaust terms are applied after the steady state, outside the grey-box code; with both
  at 0 the model is the unchanged one, bit for bit (test).
* **Step 1, identifiability.** Normalised sensitivity matrix `S_ij = dy_i/dtheta_j * scale_j /
  sigma_i` at the nominal parameters (uncalibrated model), rows bed and exhaust per stationary
  segment, sigma 0.2 K per sensor (bed: mean of six, 0.2/sqrt(6) K); central differences (one-sided
  at a bound, e.g. UA = 0). Per data set (day 1, day 2, both days): delta_msqr, SVD and condition
  number, collinearity index gamma and determinant measure rho of every subset of at most three
  candidates (Brun et al. 2001). A subset is identifiable if every delta_msqr >= 1 and gamma <= 10.
  Profile likelihood on both days for the three candidates with the largest delta_msqr (11 grid
  points over the bounds, the joint calibration set re-fitted). Repeated at the estimates.
* **Step 2, stationary calibration.** Calibrated set on the calibration day = largest identifiable
  subset (at most three) of UA, the distances and the freeboard heat transfer; ties by rho. All
  other parameters at nominal; the distance of the other product therefore at 0 (switch off) in the
  cross-check (decision of the project lead, 2026-11-13). Bounded weighted least squares (weights
  1/sigma) on the four segments of one day, prediction on the four segments of the other day, both
  directions. Baselines: the uncalibrated model and the constant bias (B1 of Befund 120: the
  uncalibrated model plus the calibration day's mean residual per quantity).
* **Descriptive variant without reading** (decision of the project lead, 2026-11-13): one shared
  bed-to-exhaust distance for both products (as C3 of Befund 117), same tables and same comparison
  with the bias.
* **Not part of this study:** step 3 of the request (transients: dead time, sensor lag, settling,
  one or two dynamic parameters, wall state as a switch). Open follow-up; it waits for transient
  data of the plant.

## 3. Pre-registered reading (docstring of `scripts/analyze_plant_calibration.py`)

* **K0** (stationary calibration usable): the calibrated model meets bed AND exhaust within 2 K on
  every segment of the other day in both directions, and its largest absolute cross error is
  smaller than that of the constant bias.
* **K1** (the data do not suffice for a physical calibration; the bias suffices until transient
  data are available): the largest absolute cross error is not smaller than the bias's.
* **between** (case not named in the request, added before the run): smaller than the bias but
  above 2 K somewhere; no reading of its own.
* Kinetics are not calibrated on temperatures; their identifiability from step 1 is reported.
* The cross-check between the days is a check across products; a check within one product needs
  several days per product.

## 4. Results

Run `plant_calibration_v1` (driver commit fb1c4f6, clean tree, `git/dirty=False`, 2069 s with 8
workers, peak 114 MB per worker, no failed unit); analysis `plant_calibration_v1_analysis.json`.
Dry run `plant_calibration_dry1` before it (all units at minimal cost, 466 s, no failure; record
only in the git-ignored `local/runs/`). The stationary segments are those of Befund 117 (four per day,
all admissible). Reference: solid fraction 0.65, nominal kinetics. Errors are model - measurement.

### 4.1 Identifiability (step 1; at the nominal parameters)

delta_msqr (RMS output change in sensor sigma per parameter scale):

| candidate (scale) | day 1 | day 2 | both days |
|---|---|---|---|
| bed UA (1000 W/K) | 45.63 | 19.74 | 35.15 |
| distance product 1 (5 K) | 17.68 | 0 | 12.50 |
| distance product 2 (5 K) | 0 | 17.68 | 12.50 |
| freeboard heat transfer (1000 W/K) | 18.07 | 10.13 | 14.65 |
| effective heat capacity (600 J/(kg K)) | 0 | 0 | 0 |
| critical moisture (0.075 kg/kg) | 1.23 | 14.52 | 10.31 |
| nu_max (0.2) | 1.04 | 11.50 | 8.16 |

* **Singular values of S** (all seven columns): day 1 132.8 / 64.3 / 1.18 / 0.24 / 0.001 / 0 / 0;
  day 2 81.7 / 49.9 / 2.98 / 1.67 / 0.001 / 0 / 0; both days 146.9 / 69.5 / 56.3 / 37.4 / 1.79 /
  0.008 / 0. The condition number is infinite in every data set (the heat capacity has no steady
  effect; on one day the distance of the other product has none either). **Within one day only two
  directions carry information** (bed level and bed-exhaust gap); the next singular value is 50-270
  times smaller.
* **Collinearity.** On one day the distance and the freeboard heat transfer are collinear (gamma
  42.8 day 1, 18.7 day 2): both move only the exhaust at almost the same operating point. Each of
  them is identifiable together with the bed UA (gamma 1.27). No set of three is identifiable on one
  day.
* **Selected sets** (rule of the YAML): day 1 {bed UA, freeboard heat transfer}, day 2 {bed UA,
  distance product 2}, both days {bed UA, distance product 2, freeboard heat transfer}. **On day 1
  the choice is a near tie:** rho 78.14 for the freeboard heat transfer against 77.30 for the
  distance (both gamma 1.27); evaluated at the estimates of day 1 or of both days, the rule picks the
  distance instead (rho 77.30 against 74.91 / 74.92). Shared variant: day 1 the same as the main
  variant, day 2 {bed UA, shared distance}, both {bed UA, shared distance, freeboard heat transfer}.
* **Kinetics (not calibrated, request).** Critical moisture and nu_max are never separable from
  each other (gamma 1726 day 1, 25 429 day 2, 245 both days); only one combination (weights about
  0.62-0.64 / 0.77-0.78 in the near-null singular vector) moves the temperatures. Within one day that
  combination is collinear with the bed UA (gamma 25.6 / 25.9 day 1, 19.0 day 2). Only with both
  days together is the pair bed UA + one kinetics parameter formally identifiable (gamma 1.38),
  i.e. the kinetics would be fitted to the difference between the two days -- which are two products.
  Temperature data do not determine the kinetics; a moisture measurement remains necessary.
* **Effective heat capacity:** delta_msqr exactly 0 on every data set: it is not identifiable from
  stationary data at all (a parameter of the dynamics, step 3).
* **Profiles** (both days; free: the joint set minus the profiled parameter; verdict with sigma
  rescaled by the fit, s^2 = 222.8):

  | parameter | optimum | 95 % range (rescaled) | practically identifiable |
  |---|---|---|---|
  | bed UA | 0 W/K (lower bound) | 0-24 W/K | yes (at the bound) |
  | freeboard heat transfer | 1022 W/K | 675-1155 W/K | yes |
  | distance product 1 (not in the joint set) | 3.0 K (grid point below the joint fit) | up to 7.5 K, lower side not crossed down to 0 | no |

  With the sensor sigma alone every interval is very narrow (the model error, not the sensor noise,
  dominates the residuals: s^2 = 222.8 means a residual RMS of about 15 sensor sigmas).

### 4.2 Stationary calibration and cross-check (step 2)

Fitted parameters (bounded WLS): day 1: bed UA 0 W/K (lower bound), freeboard heat transfer
1026 W/K; day 2: bed UA 0 W/K (lower bound), distance product 2 10.12 K. Both days jointly: bed UA 0
(bound), distance product 2 7.32 K, freeboard heat transfer 1022 W/K (largest error day 1 2.14 / 0.41
K, day 2 1.27 / 0.70 K, bed / exhaust).

Largest absolute error (K), bed / exhaust, over the four segments:

| | own day (cal 1 / cal 2) | day 1 -> day 2 | day 2 -> day 1 |
|---|---|---|---|
| uncalibrated model (switch off) | day 1 2.14 / 5.30, day 2 1.27 / 10.27 | 1.27 / 10.27 | 2.14 / 5.30 |
| constant bias (B1) | 0.11 / 0.42, 1.32 / 0.34 | 2.79 / 5.24 | 2.19 / 5.51 |
| **calibrated, main variant** | 2.14 / 0.43, 1.27 / 0.34 | **1.27 / 7.66** | **2.14 / 5.30** |
| calibrated, shared distance (**without reading**) | 2.14 / 0.43, 1.27 / 0.34 | 1.27 / 7.66 | 2.14 / 5.51 |

Bias offsets (measured - model, calibration day): day 1 bed +2.04 K, exhaust -5.03 K; day 2 bed
-0.05 K, exhaust -10.12 K. Largest cross error: **calibrated 7.66 K, bias 5.51 K.**

* Day 1 -> day 2: the freeboard heat transfer fitted on day 1 (1026 W/K) carries 7.45-7.66 K too
  little gap to day 2 (exhaust errors +6.61 to +7.66 K); the bed transfers within 1.27 K.
* Day 2 -> day 1: with the bed UA on its bound (= the nominal 0) and the distance of product 2 not
  acting on day 1, the calibrated model IS the uncalibrated one on day 1 (identical errors).
* Shared variant (without reading): day 2's shared distance (10.12 K) is twice day 1's gap; on
  day 1 the exhaust comes out 4.82-5.51 K too cold. Neither variant beats the bias.
* The numbers reproduce Befund 117 where the structure is the same (C1 day 1: 1026 W/K, cross error
  exhaust 7.66 K; C3 day 2: 10.12 K, 5.51 K; bed 1.27 / 2.14 K).

## 5. Reading

**K1: the data do not suffice for a physical calibration; the constant bias suffices until transient
data are available.** The calibrated model's largest cross error (7.66 K, exhaust, day 1 -> day 2) is
larger than the bias's (5.51 K); the bed alone stays within 2.14 K in every variant, but the exhaust
misses 2 K in every variant and baseline. The descriptive shared variant gives the same picture
(without reading). The near tie of the selection on day 1 does not change the reading: with the
distance instead of the freeboard heat transfer, the main variant would carry no gap term to day 2
at all (the distance of product 2 stays at 0).

**What the two days can determine:** per day two directions (bed level, bed-exhaust gap); over both
days in addition the difference of the gap between the products. Which physical term carries the
gap (distance or freeboard heat transfer) cannot be told from one day, and across the two days
neither term is constant. **The cross-check across days is a check across products** (day = product):
the gap is product-specific (10.12 K against 5.03 K in the bias offsets). A check within one product
needs several days per product (question to the application partner, docs/12).

## 6. Caveats

* Two days, one per product; four near-identical stationary segments per day: rank two per day.
* Solid fraction 0.65 for both products and dried supply air are placeholders (Befund 117); kinetics
  nominal. The sensitivity was evaluated at the nominal parameters (selection) and at the estimates
  (reported); the selection on day 1 is a near tie that depends on the evaluation point and on the
  parameter scales (rho depends on the scales).
* Sensor sigma 0.2 K per sensor is the sensor model of the process config, not a measured noise; the
  model error dominates the fit (s^2 = 222.8), so the profiles with the sensor sigma are not
  meaningful as confidence intervals.
* The profile of the distance of product 1 found a better point than the joint fit (which did not
  include that parameter); reported as the profile optimum.

## 7. Open follow-up (not part of this study)

Step 3 of the request (transients: dead-time and sensor-lag estimation, masking the settling time,
fitting one or two dynamic parameters such as the effective heat capacity -- not identifiable from
stationary data, section 4.1 -- and the wall state as a switch; tested on synthetic transients from
the model). It waits for transient data of the plant (start-up, load change, product change).
Data needed (question to the application partner, docs/12): transients with time stamps; laboratory
moisture with clock time; several days per product.

## 8. Decisions taken without asking, disclosures

1. **Base commit:** `origin/master` c4ecbfe as requested (local `master` had one unpushed merge,
   Befund 121, not needed here).
2. **Normalisation and thresholds** (fixed in the YAML before the run): sensor sigma 0.2 K (bed:
   mean of six), parameter scales from the orders of magnitude of Befund 117 and the Befund 108
   kinetics band; Brun et al. thresholds delta_msqr >= 1 and gamma <= 10; ties by rho.
3. **One-sided differences at a bound** (UA = 0): found in the synthetic smoke test before the commit
   (the central difference tried a negative UA, which the solver rejects); fixed before the commit.
4. **Profile optimum from the grid** when a grid point lies below the joint fit: added to the
   analysis before the commit (seen in the synthetic smoke test).
5. **Case "between"** (better than the bias, but above 2 K) added to the reading before the run; it
   did not occur.
6. **Seen before the run:** nothing of the plant data; the driver was tested before the commit only
   with invented points (synthetic smoke test). The export files were matched to day 1 / day 2 by
   their SHA-256 against the Befund 117 record (no content read).

## 9. Artifacts and provenance

* `artifacts/runs/plant_calibration_v1.json.gz`: run record (driver `scripts/run_plant_calibration.py`,
  commit fb1c4f6, clean tree, Python of the main checkout's `.venv` via `tools/run_here.sh`,
  `code/root_checked=True`); normalised sensitivities, SVD, subset scores, fitted parameters, model
  errors per segment, bias offsets, profiles. No absolute measured or modelled values.
* `artifacts/runs/plant_calibration_v1_analysis.json`: reading and tables
  (`scripts/analyze_plant_calibration.py`, reading committed in fb1c4f6 before the run).
* Git-ignored: `local/reports/plant_calibration_v1_local.json.gz` (segment points, model outputs),
  `local/runs/plant_calibration_v1/` (stage units), `local/runs/plant_calibration_dry1/` (dry run),
  `local/logs/` (suite and run logs), German report `local/reports/plant_calibration_bericht.md`.
* Tests: `tests/test_plant_calibration.py` (14 tests, synthetic data only); full suite green on
  fb1c4f6 before the run (log `local/logs/suite_pre_run.log`).
