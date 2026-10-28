# NN -- Plant-near model of plant A: does the picture of the laboratory studies hold at plant scale?

Status: design, code and reading committed before the run (commit 49ed3cf); results added after the
run (numbers only from the artifacts in section 9). Branch `plant-scale` (from local `master`
be39e65, see section 7, item 1). Document and finding number open ("NN"); the managing session
assigns them. **The model is the unchanged grey-box model of the lab line, parametrised at plant
scale with placeholders; every model number below carries them (section 8).**

Confidentiality as in docs/18 and docs/23: roles only, channel tags in the git-ignored
`conf/plant/plant_local.yaml` (key `plant_scale_roles`, template with generic tags in
`conf/plant/plant_local.yaml.example`), raw exports only through `TWINLINK_PLANT_DATA_ACE` /
`TWINLINK_PLANT_DATA_RID`; per-segment absolute values only in the git-ignored local report. Day 1 =
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

RESULTS_PLACEHOLDER

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
