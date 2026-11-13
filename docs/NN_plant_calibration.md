# NN -- Calibration of the grey-box model at plant A: identifiability first

Status: design, code and reading committed before the run on plant data; results are added after
the run (numbers only from the artifacts). Branch `plant-calibration` (worktree
`C:/twinlink_work/pcal`, from `origin/master` c4ecbfe). Document number and finding number are
open ("Befund NN"); the managing session assigns them at merge. Quick variant of the request: steps
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

(added after the run)
