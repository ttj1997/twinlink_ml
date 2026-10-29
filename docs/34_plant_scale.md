# 34 -- Plant-near model of plant A: does the picture of the laboratory studies hold at plant scale?

Status: design, code and reading committed before the run (commit 49ed3cf); results added after the
run (numbers only from the artifacts in section 9). Branch `plant-scale` (from local `master`
be39e65, see section 7, item 1). The document number is 34 and the finding number 117
(lab book). **The model is the unchanged grey-box model of the lab line, parametrised at plant
scale with placeholders; every model number below carries them (section 8).**

Confidentiality as in docs/18 and docs/23: roles only, channel tags in the git-ignored
`conf/plant/plant_local.yaml` (key `plant_scale_roles`, template with generic tags in
`conf/plant/plant_local.yaml.example`), raw exports only through `TWINLINK_PLANT_DATA_P1` /
`TWINLINK_PLANT_DATA_P2`; per-segment absolute values only in the git-ignored local report. Day 1 =
product 1, day 2 = product 2 (day and product are confounded, as in docs/23).

## 1. Question

The laboratory studies (Befund 105, 108, 116) ran on the lab line. Does their picture hold at the
scale of plant A? Three steps: (1) what do the plant data say about stationary operation, in
particular about the gap between bed and exhaust temperature; (2) can the grey-box model,
parametrised at plant scale and calibrated on at most three temperature-identifiable parameters,
reproduce bed and exhaust temperature of the other measured day; (3) with that model, (a) the
observability of the bed moisture (docs/33) and (b) the steady headroom of a tracked against a
fixed operating point (Befund 105, steady variant) at plant scale, next to the lab line.

## 2. Design

Package `src/twinlink_ml/plant_scale/` (data and stationary segments, balances, model wrapper and
calibration, steady search), study config `conf/plant_scale/plant_scale.yaml`, process config
`conf/process/greybox_plant_a_placeholder.yaml`, driver `scripts/run_plant_scale.py`, analysis
`scripts/analyze_plant_scale.py` (reading in its docstring), tests `tests/test_plant_scale.py`
(synthetic data only). Existing modules are only read.

* **Data.** Two exported days of plant A (one-minute samples, 24 h each). Roles: intake (ambient)
  temperature, process-air flow, supply-air temperature (three sensors), six bed-temperature
  sensors (two tag families of three; positions not confirmed), exhaust temperature at the
  granulator outlet, spray rate, atomising air (six nozzles, summed), two sensors further
  downstream (informative only).
* **Stationary segments** (criterion fixed in the config before the first look at segment
  statistics): 60-minute windows whose rolling standard deviations stay at or below 0.3 K (bed
  mean, exhaust, supply mean), 0.5 % (process air) and 3 % (spray rate); 2-4 non-overlapping
  windows per day at least 60 minutes apart, best score (largest std / threshold) first; if fewer
  than two are admissible, the best ones are taken anyway and flagged.
* **Balances** (no model, no fit): per segment the heat the gas gives up between bed and exhaust
  sensor, and the one number each candidate explanation would need to close it alone: heat-loss
  conductance (surroundings at the intake temperature or the room placeholder), cooling by the
  atomising air if it bypassed the bed (upper bound), false-air share, share of the spray water
  evaporating above the bed, spread of the bed sensors and the distance of the coldest one to the
  exhaust. Plus the docs/23 dryer balance with the outlet at the exhaust and at the bed temperature.
* **Plant-near model.** The grey-box model with plant-scale geometry and feed placeholders
  (section 8), the measured inlet temperature, spray rate and intake temperature per segment, the
  gas velocity iterated until the steady dry-air flow equals the measured process air, dried
  supply air. Kinetics and isotherm of the lab line, NOT calibrated (no moisture measurement; their
  band enters through the 20 parameter draws of Befund 111, drawn from the Befund 108 ranges).
* **Switch of a bed-to-exhaust term** (default off; outside the grey-box code, which is unchanged):
  `offset` (the existing `exhaust.outlet_duct_heat_loss_k`), `heat_loss` (loss of the gas to the
  surroundings, duct formula), `evaporation` (a share of the spray water evaporates above the bed;
  emulated by the bed's feed, the freeboard energy balance closed exactly). Switched off, the
  steady state is the unchanged model's, bit for bit (test).
* **Calibration** (at most two parameters, because a segment has two measured temperatures): the
  bed heat-loss UA (>= 0) in every candidate, plus the switch's one parameter. Candidates C0
  (switch off), C3 (offset), C1 (heat loss), C2 (evaporation). Bounded least squares over all
  stationary segments of one day; prediction on every segment of the other day; both directions.
  Reference: solid fraction 0.65 on both days; sensitivity over 0.60 / 0.634 / 0.70 and over the 20
  kinetics draws.
* **Observability (4a)** with the code and the SNR definition of docs/33 at each day's aggregate
  operating point, own-day calibration of the selected candidate; lab line at P1-P7 as in docs/33
  (reproduction check) and with the dried supply air of the plant ("supply air dry").
* **Steady headroom (4b)**: R0 = one fixed command (inlet temperature, gas velocity), the cheapest
  at the nominal disturbance that holds the moisture limit and the bed-temperature limit (80 degC)
  at all 8 corners of the disturbance box; R1 = the steady optimum at the actual disturbance
  (margin 0); saving at the 9 box points. Box from the plant data: spray rate +-2 x the relative std
  of the 15-minute block means, ambient +-half the block range (larger day each); solid fraction
  +-7.69 % relative (placeholder). Plant: limit = the model moisture at today's measured operating
  point (proxy "today sits at the quality limit"), commands 70-100 degC and 0.8-1.1 x today's gas
  velocity (docs/23 box); per day with the nominal kinetics and the 20 draws. Lab line with the same
  relative amplitudes: limit 0.15 kg/kg, 60-120 degC, 0.8-2.5 m/s. The Befund 108 definition of the
  9.67 % (lab nominal, margins 0.04355 / 0.00664) is recomputed as a check of the optimiser.

## 3. Pre-registered reading (docstring of `scripts/analyze_plant_scale.py`, commit 49ed3cf)

* **P1** if at least one candidate reproduces bed temperature (mean of the six sensors) AND exhaust
  temperature within 2 K at every stationary segment of the other day, in both directions, at the
  reference (solid fraction 0.65, nominal kinetics): the plant-near model is usable for energy
  questions; step 4 uses the passing candidate with the smallest largest error.
* **P2** otherwise: structural finding; step 4 uses the candidate with the smallest largest error
  and holds only under reservation.
* **Structural statement** (independent): "T_out = T_bed cannot represent the data" if C0 leaves more
  than 2 K on its own calibration day. Parameter-free bound with it: max(|e_bed|, |e_exh|) >= gap/2.
* Without reading: balance numbers, sensitivities, SNR plant-near against lab, steady headroom
  plant-near against lab.

## 4. Results

Run `plant_scale_v1` (driver commit b4ce48d, clean tree, `git/dirty=False`, 13 805 s with 14
workers); analysis `plant_scale_v1_analysis.json`. Day 1 / day 2 throughout. Absolute values per
segment and per role are only in the git-ignored local report; below: day aggregates (mean over the
four segments of a day, as docs/23 reports day means), differences, relative values and model
numbers.

### 4.1 Data survey (step 1)

* **Stationary segments.** Four admissible 60-minute windows on each day (24 h each); the
  criterion was met everywhere, no window had to be flagged. Score (largest std / threshold) 0.39-0.46
  / 0.30-0.37. Within the windows: standard deviation of the bed mean 0.05-0.13 / 0.08-0.11 K,
  exhaust 0.001-0.023 / 0.004-0.009 K, supply mean 0.12-0.14 / 0.08-0.10 K; process air 0.12-0.19 % /
  0.15-0.18 % and spray rate 1.2-1.3 % / 0.06-0.08 % relative.
* **Day aggregates against the request's placeholders.** Process air 1.10 / 1.02 x the 30 000 Nm3/h
  placeholder; spray rate 0.47 / 1.06 x the 2 t/h placeholder (day 1 sprays less than half of day
  2); supply air 84.55 / 83.10 degC (placeholder about 80 degC); atomising air 1.7 % / 2.7 % of the
  process air (volume); intake temperature 3.9 / 12.1 degC.
* **Bed and exhaust.** Bed (mean of six) 67.68 / 44.43 degC, exhaust at the granulator outlet 60.61 /
  34.36 degC. **Gap bed - exhaust per segment 6.71-7.27 K / 9.48-11.05 K.** Spread of the six bed
  sensors (max - min) 1.27-1.99 / 4.02-4.76 K; group b - group a 0.52-1.04 / 2.69-3.56 K; the
  coldest bed sensor lies 6.10-6.50 / 7.87-9.02 K above the exhaust. Exhaust minus the next sensor
  downstream 2.05-2.36 / 0.24-0.40 K.
* **Disturbance amplitudes** (15-minute blocks, larger day): spray rate +-2.59 % (2 x relative
  std, day 1), intake temperature +-3.64 K (half range, day 2).

### 4.2 Balances of the bed-exhaust gap (no model, no fit)

Heat the gas gives up between bed and exhaust sensor: 82-89 kW / 111-129 kW (solid fraction 0.65;
the band changes it by less than 1 %). The number each candidate would need to close the gap ALONE:

| candidate explanation | day 1 | day 2 | consistent across days? |
|---|---|---|---|
| heat loss bed -> exhaust sensor, surroundings at intake temperature | UA 1.36-1.53 kW/K | 4.04-4.76 kW/K | no: about 3x on the colder bed |
| same, surroundings at the room placeholder (20 degC) | 1.88-2.03 kW/K | 5.97-6.54 kW/K | no: about 3x |
| false air at intake / room temperature | 11.7-13.2 % / 16.5-17.9 % of the gas | 41-50 % / 67-75 % | no; day 2 implausibly large |
| atomising air bypassing the bed (upper bound) | 0.71-0.72 K = 9.9-10.6 % of the gap | 0.49-0.55 K = 4.9-5.2 % | explains at most a tenth |
| spray water evaporating above the bed | 36-40 % of the water (band 0.60-0.70: 31-46 %) | 22-26 % (19-30 %) | product-dependent, not excluded |
| bed sensor location (spread among the six) | spread 1.27-1.99 K, coldest sensor still 6.10-6.50 K above exhaust | 4.02-4.76 K, 7.87-9.02 K | no, as far as the six sensors reach |

A heat loss scales with (T - T_surroundings); day 2 has the colder bed (about 44 against 68 degC) and
the larger loss, so a fixed conductance cannot close both days. **Overall balance (docs/23 dryer
balance, per segment):** with the outlet at the exhaust temperature the wall conductance is positive,
1.00-1.23 / 1.75-2.26 kW/K at solid fraction 0.65; with the outlet at the BED temperature it is
negative on every segment and at every solid fraction of the band (day 1 -0.19 to -1.61, day 2 -0.51
to -7.00 kW/K): the measured energy balance cannot close if the gas leaves at the measured bed
temperature, unless heat enters the bed. The solid fraction consistent with both days (docs/23
method): 0.639 (UA 0.96 kW/K) with the outlet at the exhaust, 0.714 (UA -0.03 kW/K, outside the
band) with the outlet at the bed. A single cause is not identified; the balance numbers bound it.

### 4.3 Calibration and cross-check (step 3; reference: solid fraction 0.65, nominal kinetics)

Largest absolute error (model - measurement, K) over the segments; "cal" = on the calibration day,
"check" = on the other day.

| candidate | calibrated on | parameters | cal bed / exh | check bed / exh |
|---|---|---|---|---|
| C0 none | day 1 | UA 314 W/K | 3.65 / 3.74 | 2.00 / 9.67 |
| C0 none | day 2 | UA 2890 W/K | 6.92 / 5.58 | 13.84 / 6.80 |
| C3 offset | day 1 | UA 0 (bound), offset 5.03 K | 2.14 / 0.42 | 1.27 / 5.24 |
| C3 offset | day 2 | UA 0 (bound), offset 10.12 K | 1.27 / 0.34 | 2.14 / 5.51 |
| C1 heat loss | day 1 | UA 0 (bound), 1026 W/K | 2.14 / 0.43 | 1.27 / 7.66 |
| C1 heat loss | day 2 | UA 0.0001 W/K, 4206 W/K | 1.27 / 1.56 | 2.14 / 13.63 |
| C2 evaporation | day 1 | UA 856 W/K, share 0.376 | 0.20 / 0.46 | 9.83 / 2.56 |
| C2 evaporation | day 2 | UA 2962 W/K, share 0.225 | 1.96 / 0.98 | 11.10 / 8.27 |

Worst cross error per candidate: **C0 13.84, C3 5.51, C1 13.63, C2 11.10 K; none <= 2 K.** The bed
temperature transfers between the days within 1.27-2.14 K with C3 or C1 (the bed UA sits on its
lower bound 0: the model's bed wants no wall loss or a heat input, in line with the negative
conductance of 4.2); the exhaust does not: the fitted offset is 5.03 K on day 1 and 10.12 K on day 2.
The term between bed and exhaust is day-specific (day = product), not a constant of the plant.

### 4.4 Sensitivity of the calibration (no reading)

* **Solid fraction** (worst cross error, K, at 0.60 / 0.634 / 0.70): C0 6.84 / 11.83 / 18.88; C3
  5.46 / 4.93 / 11.77; C1 9.54 / 12.38 / 18.34; C2 6.52 / 9.41 / 15.52. No candidate reaches 2 K
  anywhere in the band.
* **Kinetics** (the 20 draws of Befund 111, calibration repeated per draw): worst cross error C0
  8.36-19.93, C3 3.99-12.16 (median 5.60 / 6.38 per direction), C1 6.47-18.56, C2 9.33-14.90 K; no
  draw passes. The fitted offset of C3 stays at 4.93-5.47 K (day 1) and 8.77-10.16 K (day 2).

### 4.5 Observability at plant scale (step 4a; C3, own-day calibration, under reservation)

SNR for Delta X = 0.01 kg/kg (definition and code of docs/33):

| case | Y_out | Y_out_balance | T_out | T_bed | T_bed_x6 | m_s / m_air | X nominal (kg/kg) |
|---|---|---|---|---|---|---|---|
| plant-near day 1 | 3.76 | 3.86 | 0.89 | 0.84 | 2.05 | 0.014 | 0.021 |
| plant-near day 2 | 4.59 | 4.56 | 2.89 | 2.55 | 6.24 | 0.035 | 0.095 |
| lab P1-P7 as docs/33 | 0.46-0.60 | 0.45-0.59 | 0.28-0.52 | 0.28-0.52 | 0.68-1.26 | 0.004-0.008 | 0.084-0.169 |
| lab P1-P7, dried supply air | 0.83-0.87 | 0.80-0.84 | 0.28-0.52 | 0.28-0.52 | 0.68-1.27 | 0.004-0.008 | 0.076-0.146 |

R^2 of the slope fits >= 0.996, no failed or infeasible draw at plant scale. The lab rows reproduce
docs/33 exactly (largest SNR difference 0.0 at P1-P7). Dried supply air raises the humidity SNR of
the lab line by a factor 1.4-1.9 (lower absolute humidity, smaller sensor sigma) and leaves the
temperatures unchanged. At plant scale a single humidity sensor exceeds SNR 1 on both days (3.76 /
4.59), the bed temperature with six sensors on both (2.05 / 6.24), a single temperature sensor only
on day 2. For comparison, docs/33's plant-like placeholder (ratio 0.0129 / 0.0258, lab air
humidity): Y_out 0.94-1.67 / 1.96-3.54. The day-1 point sits near the dry end (X 0.021 kg/kg).

### 4.6 Steady headroom fixed against tracked operating point (step 4b; C3, own day, under reservation)

Box: spray rate +-2.59 %, intake +-3.64 K, solid fraction +-7.69 % relative (placeholder). Plant
limit = today's model moisture: 0.0207 / 0.0947 kg/kg.

| | day 1 | day 2 | lab, same relative amplitudes |
|---|---|---|---|
| saving R1 vs R0 at the nominal disturbance | **6.89 %** | **15.91 %** | **5.47 %** |
| over the 9 box points (min-max) | 0-13.40 % | 0-28.97 % | 0.003-10.32 % |
| over the 20 kinetics draws (min / median / max, nominal disturbance) | 6.63 / 6.83 / 7.15 % | 15.28 / 16.10 / 17.47 % | 4.19 / 5.26 / 6.69 % |
| R0 (inlet, gas velocity / today's) | 100 degC (bound), 0.82 | 100 degC (bound), 0.94 | 89.46 degC, 0.8 m/s (bound) |
| R1 at nominal | 93.80 degC, 0.80 (bound) | 98.00 degC, 0.80 (bound) | 84.84 degC, 0.8 m/s (bound) |
| power R0 / R1 against today's model point | -6.4 % / -12.9 % | +12.1 % / -5.7 % | - |

Befund 108's definition (lab nominal, margins 0.04355 / 0.00664) recomputed: **9.665 % against 9.67 %**
(fixed 94.97 degC, tracked 86.15 degC, both 0.8 m/s): the optimiser reproduces the reference. The
lab line under the plant's amplitudes gives 5.47 %, less than the 9.7 % (steady) of Befund 108,
whose margins came from the docs/24 disturbance families. **The plant numbers rest on two bounds and
one placeholder:** R0 sits on the 100 degC inlet bound and R1 on the air factor 0.8 on both days
(the docs/23 box), and the savings at the box points split by the solid fraction (nominal
kinetics; corners at the low solid fraction 0-3.3 % / 0-4.7 % / lab 0-4.7 %, at the high one
10.8-13.4 % / 25.1-29.0 % / lab 6.3-10.3 %), whose amplitude is a placeholder, not a measurement. Day 2's R0 needs more power than today's model point: today's point is not robust
over the box at its own limit. Solver failures: none at plant scale; 314 in 4 of the 20 lab draws
(of about 5 800 solves each), counted as inadmissible.

## 5. Reading

**P2: structural finding.** No candidate reproduces bed and exhaust temperature of the other day
within 2 K (best C3 offset, 5.51 K, on the exhaust); this holds over the solid-fraction band and
over all 20 kinetics draws. **Structural statement: T_out = T_bed cannot represent the data** (C0
leaves 6.92 K on its own calibration day; the parameter-free bound gap / 2 is 3.35-5.52 K per
segment). The bed temperature alone transfers between the days within 2.14 K (C3); the bed-exhaust
gap of 6.7-11.0 K is day-specific (5.03 vs 10.12 K offset) and none of the four candidate terms
carries it from one day to the other. Step 4 used C3 calibrated on the same day and **holds only
under reservation**: SNR and headroom at plant scale are reported, not read. The selection of C3
among four candidates on the same two days is a selection, not a validation.

## 6. Caveats and open points

* Two days, one per product: day and product are confounded; every day-specific term (offset,
  evaporation share) may be a product property or a state of the plant on that day.
* The model is the lab line's at plant geometry: kinetics, isotherm and bed heat transfer of the lab,
  solid fraction 0.65 for both products (placeholder), dried supply air (placeholder), no moisture
  measurement. The plant limit "today's model moisture" is a proxy.
* The headroom depends on the docs/23 command box (both bounds bind) and on the placeholder
  amplitude of the solid fraction; the lab comparison uses the plant's relative amplitudes, not the
  docs/24 families behind Befund 108.
* Sensor positions of the two bed groups are not confirmed; a hot zone common to all six sensors
  cannot be excluded from the data.
* What would decide between the candidates: a temperature (or humidity) measurement directly above
  the bed, the atomising-air and leak-air balance of the plant, the solid fraction per product, a
  third day with one product at a different load.

## 7. Decisions taken without asking (unattended run), disclosures

1. **Base commit.** The request names "current origin/master" (9feec5c). Local `master` was three
   commits ahead (be39e65: the merge of the observability study, Befund 116, not yet pushed). Step
   4a needs that study's code ("same code"); branching from 9feec5c would have meant merging it in
   (not allowed). The branch starts at be39e65, a superset of origin/master.
2. **Model unchanged; the switch lives in the study package.** "Existing model only parametrise"
   and "optional term as a switch, default off, earlier results bit-identical": the term is applied
   outside `twinlink_ml.simulation` (wrapper), so no earlier result can change; the test checks that
   switched off the steady state equals the unchanged model's bit for bit. The `offset` candidate
   uses the model's existing `exhaust.outlet_duct_heat_loss_k`.
3. **At most two calibrated parameters, not three.** A segment carries two temperatures; with all
   segments of one day at almost the same operating point, a third parameter would not be
   identifiable. The heater efficiency does not enter (the supply temperature is measured at the
   granulator); an air-flow factor ("gas balance") was not calibrated for the same reason; the
   atomising air is not part of the model and its size is given as a balance number instead.
4. **Solid fraction not calibrated.** It is product-specific (day = product), so a value fitted on
   one day cannot be transferred to the other; reference 0.65 (docs/23), band as sensitivity. The
   value 0.634 of docs/23 was fitted on both days there and is not independent of the check.
5. **Kinetics not calibrated** (request): critical moisture and nu_max are not identifiable from
   temperatures without a moisture measurement; at plant scale they do move the bed temperature
   (the product carries water), which is why the calibration is repeated over the 20 draws.
6. **Bed sensor groups named neutrally** (`group_a`, `group_b`): the process screen suggests two
   positions, but mounting heights are not confirmed.
7. **Dried supply air:** placeholder inlet humidity 2.4e-5 kg/kg (dew point about -50 degC); the
   model's "ambient" is the measured intake temperature (it serves the heater and the wall loss);
   the room temperature placeholder enters only the balances.
8. **Plant moisture limit = today's model moisture** (proxy; no moisture measurement), as the proxy
   constraints of docs/23 treat today's operation as the boundary.
9. **Precision:** calibration and observability with the solver's default precision (as docs/33);
   the headroom search with the search precision of Befund 52 (30 iterations / 1e-7; about 3x
   faster, moisture to about 1e-8 kg/kg).
10. **Seen before the run** (disclosed): a channel listing of both exports (day means of every
    channel) to assign the roles, two process screens (sensor layout, kept local), and a smoke test
    of the calibration functions on rounded day means (before the reading was committed, after
    the candidate set was fixed in the config): all four candidates missed 2 K on day 2 when
    calibrated on day 1. The reading wording is the request's; the candidate set and its
    selection rule were not changed afterwards.
11. **Redaction of the run record before the commit** (after the run). The driver wrote two
    per-segment absolute role values into the balances of the repository record (segment mean of
    the intake temperature as `surroundings_celsius`, segment mean of the process-gas flow as
    `gas_flow_kg_per_s`), against the design (section 2: per-segment absolute values only local).
    Conservative reading: they do not enter the repository. The full record stays git-ignored as
    `local/reports/plant_scale_v1_full.json` (SHA-256 486b75ee...a4b3, recorded in the redacted
    record); `scripts/redact_plant_scale.py` (committed before it ran, test with synthetic data)
    removed exactly these fields (160 entries: 8 segments x 4 solid fractions x 5 occurrences;
    the room surroundings, a 20 degC placeholder, go with them because the name is the same) and wrote
    `artifacts/runs/plant_scale_v1.json`; the analysis ran on the redacted record. The derived
    balance numbers (heat, conductances, shares) stay, they are the requested balance numbers.
    Day aggregates of every role stay in the record, as in the committed docs/23 artifact
    (`study_plant_a_operating_point.json` carries the day means of process air and spray rate).
12. **Log written outside the worktree** (disclosed). The log of the first full suite was written
    to `C:/twinlink_work/plant_suite_1.log`, i.e. outside the worktree, against the rule of the
    request; it was moved to the worktree's `local/logs/` before the main run. No other file
    outside the worktree was written; the main checkout was only read (`local/`, the role file and
    the exports).
13. **Numerical warnings of the steady search** (disclosed): the log of the main run carries 124
    `RuntimeWarning`s of scipy's bounded Brent search ("invalid value encountered in scalar
    subtract"): the objective is `inf` at commands that violate a limit or whose solve fails, and
    Brent's parabola step then computes `inf - inf`. A Brent result is used only if it is finite
    and below the best grid point, so the result is at worst the grid point. Solver failures: none
    at plant scale, 314 in 4 of the 20 lab draws (62 / 28 / 144 / 80 of about 5 800 solves),
    counted as inadmissible.

## 8. Placeholders

Values of the study request (marked there as approximate or unconfirmed) and how they entered:

| placeholder of the request | used as | measured instead? |
|---|---|---|
| process air about 30 000 Nm3/h | not used in the model | yes: day aggregates 1.10 / 1.02 x the placeholder; the model matches the measured dry-air flow per segment |
| supply-air temperature about 80 degC | not used in the model | yes: 84.55 / 83.10 degC (day aggregates), measured value per segment |
| bed mass 600-800 kg | 700 kg | no; no effect on the steady state |
| suspension feed about 2 t/h (solid fraction from local/) | not used in the model | yes: 0.47 / 1.06 x the placeholder; solid fraction 0.65 for both products (docs/23 reference), band 0.60 / 0.634 / 0.70 |
| supply air dried (very low dew point, unconfirmed) | inlet humidity 2.4e-5 kg/kg (dew point about -50 degC) | no |
| exhaust temperature at the granulator outlet | the exhaust role of the calibration | sensor position as stated in the request |
| six bed-temperature sensors | bed temperature = mean of six; SNR variant `T_bed_x6` | yes (six channels); positions not confirmed |

Further placeholders of this study (`conf/process/greybox_plant_a_placeholder.yaml`,
`conf/plant_scale/plant_scale.yaml`): cross-section 4.9 m2 (2.5 m diameter, no source; does not
enter the steady temperatures at a given dry-air flow); kinetics and isotherm of the lab line
(critical moisture, nu_max, GAB; band through the 20 draws); cp of the suspension 2243 J/(kg K)
(mixture at 0.65); feed temperature 30 degC; atomising-air temperature 25 degC and room temperature
20 degC (balances only); bed-temperature limit 80 degC; plant moisture limit = today's model
moisture; solid-fraction amplitude +-7.69 % relative; command box 70-100 degC and 0.8-1.1 x today's
gas velocity (docs/23).

## 9. Artifacts and provenance

* `artifacts/runs/plant_scale_v1.json`: run record (driver `scripts/run_plant_scale.py`, commit
  b4ce48d, clean tree, `git/dirty=False`, 13 805 s, 14 workers, Python 3.14.2, `.venv` of the main
  checkout via `tools/run_here.sh`, `code/root_checked=True`), redacted by
  `scripts/redact_plant_scale.py` (commit 2564940; entry `redaction` with the SHA-256 of the
  full record). Raw exports recorded by SHA-256 and environment-variable name only.
* `artifacts/runs/plant_scale_v1_analysis.json`: analysis and reading (`scripts/analyze_plant_scale.py`,
  reading committed in 49ed3cf, run at commit 2564940).
* Git-ignored, in the worktree: `local/reports/plant_scale_v1_full.json` (unredacted record),
  `local/reports/plant_scale_v1_local.json` (per-segment absolute values, block statistics,
  minimum admissible solid fraction per product), `local/reports/plant_scale_v1_analysis_unredacted.json`
  (first analysis, on the unredacted record; identical apart from the removed fields),
  `local/logs/` (suite and run logs), `conf/plant/plant_local.yaml` (role file with the channel
  tags).
* Tests: `tests/test_plant_scale.py` (14 tests, synthetic data only); full suite green on b4ce48d
  before the run, and on the working tree committed as 2564940 (redaction script and its test)
  before that commit; logs in the git-ignored `local/logs/`.
