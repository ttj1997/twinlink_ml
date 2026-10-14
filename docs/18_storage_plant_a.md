# Storage A1 of plant A: model, identifiability, calibration, shifting potential

Status: S1 (model, identifiability, calibration) complete; S2 (shifting potential) see the last
section. Lab book: Befund NN (S1) and Befund NN+1 (S2) in `docs/04_stand_ap1.md` (numbers assigned by
the managing session). Code: `src/twinlink_ml/storage_a1/`. Scripts: `scripts/study_storage_a1.py`,
`scripts/gate_storage_shift.py`. Artifacts: `artifacts/runs/study_storage_a1.json`,
`artifacts/runs/gate_storage_shift.json`.

Confidentiality: this file, the code and the tests name roles only (`storage_temperature`,
`preheater_loop_flow`, ...), never measurement points, equipment numbers, companies or products. The
mapping role -> channel is in the git-ignored `conf/plant/plant_local.yaml` (template with
placeholder tags: `conf/plant/plant_local.yaml.example`); the raw exports are read from the
environment variables of the plant-validation scripts and never leave `local/`. "Product 1" and
"product 2" are the two plant days (24 h, one-minute data, 1441 samples each).

## 1. System and what is measured

```
compressor heat --> storage loop (storage A1 inside) --> exchanger A1 --> preheater loop --> preheater A1
air path:  outside air --> preheater A1 --> steam heater (gas boiler) --> fan --> supply air (about 80-85 degC)
```

Two water loops, separated by exchanger A1 (information from the control-system pictures, supplied
by the project lead): the storage loop (about 37 m3/h) and the preheater loop (about 18-20 m3/h).
The order inside the storage loop is NOT settled -- the two pictures disagree -- so both orders are
a switch of the model (`topology`):

* **B** `storage_then_exchanger` (default): compressors -> storage -> exchanger A1 -> compressors.
  The hot inlet of exchanger A1 is the storage temperature (measurable).
* **A** `exchanger_then_storage`: compressors -> exchanger A1 -> storage -> compressors. The hot
  inlet is the storage temperature plus `Q_in / C_1` (the compressor heat is in the stream).

Measured roles: storage temperature (ONE sensor exported of three), both loop flows, supply and
return temperature of the preheater loop (water side), supply-air temperature (mean of three
sensors), intake-air temperature, process-air flow (normal volume), steam-valve position (proxy
only: two valves in split range), fan power. Not measured: compressor heat taken into the storage
loop `Q_in`, steam flow, gas consumption, intake humidity. An auxiliary air heater with a 60 degC
set point (outlet temperature exported) is NOT part of the waste-heat circuit and is left out; what
heats it is an open question.

## 2. Model

* Storage: one mixed node, `(rho V c_p) dT_S/dt = Q_in - Q_out - UA_S (T_S - T_room)`. The
  balance is the same for A and B because the loop is closed. Justification of one node: the
  control-system pictures show three height sensors within 0.1 K on two days (57.3/57.4/57.3 and
  69.3/69.3/69.2 degC) -- weak evidence (two snapshots on other days, not the exports); it stays an
  ASSUMPTION. The exports contain one of the three sensors, so the claim cannot be tested on the
  two days. Stratification would be a second implementation of the same ABC (`StorageModel`).
* Exchangers: counter-current, effectiveness-NTU with one conductance each. Chosen over a fixed
  `dT_min` because the flows change over the day and one `UA` follows the part load. `dT_min`
  stays as a CHECK (review finding A1 of the heat-network module): the minimum difference along the
  exchanger, which for constant capacity rates is at one of the two ends; a state below the floor
  is flagged infeasible, never silently capped. Observed minima: 7.2-11.8 K (exchanger A1),
  4.9-10.6 K (preheater) -- the floor (2 K, placeholder) is never active.
* Chain in closed form: with `a = eps_1 C_min1`, `b = eps_2 C_min2`, `C_2` the preheater-loop
  capacity rate, `Q_out = (T_h - T_a) / (1/a + 1/b - 1/C_2)` (`T_h` = hot inlet of exchanger A1,
  `T_a` = intake air); no iteration. Duty of the preheater loop from the water side:
  `Q_out = rho c_p V_2 (T_sup - T_ret)`.
* Steam: `Q_air = C_air (T_supply - T_amb) - f_fan P_fan = Q_out + Q_steam`; the steam heat is a
  RECONSTRUCTION from the air balance (the fan sits behind the steam heater, so its heat is in the
  measured supply-air temperature). Gas = `Q_steam / eta_boiler`, CO2 with the project factor
  (`evaluation/net_energy.py`, 0.201 kg/kWh gas).

### Placeholders and unconfirmed values (all in the config and its fingerprint)

| value | status |
|---|---|
| storage volume `V_S` (default 10 m3) | placeholder; only bounded (section 3) |
| storage heat loss `UA_S` (0.02 kW/K), room temperature (20 degC) | placeholder, not identifiable |
| `T_S,max` (90 degC), `T_S,min` (40 degC, S2) | placeholders, no source |
| exchanger A1 and preheater `UA` | calibrated per day (section 4) |
| minimum approach floor (2 K) | placeholder, physical sanity bound |
| boiler efficiency (0.90), fan heat fraction (0.90) | placeholders |
| relative humidity of the intake air (0.75) | placeholder, not measured |
| normal density of air (1.293 kg/Nm3) | textbook value, moist-air correction neglected |
| water as medium in both loops | assumption |
| total compressor waste heat up to about 1200 kW; up to about 400 kW used | partner statement (order of magnitude), no measurement; presentation held locally |
| fraction rejected over the cooler | unknown (`None`) |
| partner data: 7 compressors, about 1300 kW electric; process air about 30 000 m3/h; supply air about 80 degC; heating power about 800 kW; 24/7 | partner statements; measured means are 852 kW (product 1) and 697 kW (product 2) of heating demand |

## 3. Identifiability (checked before calibrating)

`Q_in` is not measured. The storage balance gives `Q_in(t; V) = Q0(t) + V g(t)` with
`g = rho c_p dT_S/dt` and `Q0 = Q_out + loss` known: `Q_in(t)` is determined only up to `V`, and
`V` and `UA_S` cannot be fitted. This holds for both topologies (the balance is identical).
The bracket of `V` follows from `0 <= Q_in(t) <= 1200 kW` for all `t`; both conditions are UPPER
bounds (cooling steps `Q_in >= 0`, heating steps `Q_in <= Q_max`); the only lower bound is `V > 0`.

| smoothing of `dT_S/dt` | upper bound product 1 | upper bound product 2 |
|---|---|---|
| 1 min (no smoothing) | 29.4 m3 | 57.2 m3 |
| 15 min | 30.2 m3 | 57.7 m3 |
| 30 min | 31.6 m3 | 57.9 m3 |
| 60 min | 51.7 m3 | 97.3 m3 |
| 120 min | 74.7 m3 | 168.6 m3 |

(`UA_S = 0.02 kW/K`, strict; allowing 1 % of the samples to violate raises the 15-min values to
30.7 / 58.2 m3; `UA_S` between 0 and 2 kW/K moves them to 30.1-36.6 / 57.6-72.2 m3.) The 1200 kW
condition is never the binding one (it gives at least 466 m3); with 700 kW it would give 202 / 298 m3.
Smoothing loosens the bound because it hides the cooling steps, so the least smoothed values are the
tightest valid ones. **Result: `V_S <= about 30 m3` (both days together), no lower bound.** Reconstructed
`Q_in` over the whole admissible range of `V`: mean 271-278 kW (product 1), 375-388 kW (product 2),
maxima 322-375 / 460-466 kW -- far below 1200 kW; the product-2 maximum exceeds the partner's
"up to about 400 kW used".

Determinable and not (details in the artifact):

| quantity | status |
|---|---|
| `Q_out(t)` (duty given to the preheater loop) | measured (water side): mean 277 / 387 kW, maximum 321 / 459 kW |
| heating demand of the process air | derived; mean 852 / 697 kW (placeholders: fan fraction, humidity) |
| steam heat `Q_steam` and steam share | reconstructed, not measured: share 67.5 % / 44.4 % |
| conductance of exchanger A1 | determinable per day for topology B (implied: 19.5 / 25.2 kW/K, +29 % between the days); for topology A it depends on `V` (13.3 / 15.7 kW/K) |
| conductance of the preheater | determinable but ill-conditioned when the effectiveness exceeds 0.85 (product 2: coefficient of variation 0.36 within the day); 17.3 / 18.9 kW/K |
| `Q_in(t)`, `V_S`, `UA_S` | only as a family in `V` / a bracket (above) |
| topology A against B | not decided; weak lean to A: the implied exchanger conductance varies less within the day (CV 0.05-0.07 against 0.08-0.10 for `V <= 20 m3`) and less between the days (+18 % against +29 %). One sensor, no independent check |
| three-sensor stratification | not testable (one sensor exported) |
| boiler efficiency, `T_S,max`, compressor total and rejected share | not in the data |

## 4. Calibration and validation (split by day only)

Data repair: one sample of the preheater-loop flow of product 2 drops to half its value for one
minute while both temperatures stay smooth; replaced by interpolation (`repair_glitches`, rule:
deviation from the 9-sample running median above 20 %). It did not change the fits.

The extraction part (exchanger A1, preheater loop, preheater A1) is evaluated open loop over the
whole day from the measured storage temperature, flows and intake air (it has no state), the storage
temperature as a true 24 h rollout from the initial state. Topology B, calibration on one day,
validation on the other (topology A is within 1-5 % of these numbers, `study_storage_a1.json`):

| quantity (validation day) | calibrated on product 1, validated on product 2: RMSE / max | calibrated on product 2, validated on product 1: RMSE / max | persistence (1 -> 2, 2 -> 1) |
|---|---|---|---|
| preheater supply temperature | 0.86 / 3.07 K | 0.55 / 2.12 K | 3.29 / 3.21 K |
| `Q_out` | 96 / 153 kW | 63 / 89 kW | 80 / 24 kW |
| steam share | 0.14 / 0.23 | 0.07 / 0.10 | 0.11 / 0.03 |
| storage temperature, 24 h rollout, constant `Q_in` (V = 10 m3) | 5.5 / 8.7 K | 6.3 / 9.7 K | 4.2 / 3.5 K |

In-sample (same day): product 1 `Q_out` RMSE 16 kW (max 31), supply temperature 0.52 K; product 2
`Q_out` RMSE 62 kW (max 89), supply 0.85 K, return 3.4 K -- on product 2 one preheater conductance
does not describe the whole day. ARX (linear, same inputs) is mostly no better and, trained on
product 2, unstable in free run (hence not a useful reference there). The conductances
transfer between days only moderately: the model beats persistence for the supply temperature (the
storage temperature is an input), but NOT for `Q_out` and the steam share when the days are exchanged.
The storage-temperature rollout is worse than persistence: `Q_in` is the unmeasured disturbance that
dominates it. With `Q_in` reconstructed from the validation day itself (an ORACLE, not a prediction)
the rollout error is still 7-15 K, because the error of `Q_out` accumulates in the storage; this is
the quantitative content of "the model is not a predictor of the storage temperature".

Possible cause of the misfit (not tested, no data): the air flow through the preheater may differ
from the measured process-air flow (bypass, recirculation), or the effectiveness is near its ceiling
in the product-2 day; a question for the partner.

## 5. Daily balance (reconstruction; bands over fan fraction 0.5-1.0, humidity 0.3-1.0, boiler efficiency 0.8-1.0)

| | heating demand | waste-heat share | steam share | gas | CO2 (with / without recovery) |
|---|---|---|---|---|---|
| product 1 | 20 467 kWh/d (20 084-21 645) | 32.5 % (30.8-33.2) | 67.5 % (66.8-69.2) | 15 342 kWh/d (13 425-18 733) | 3 084 (2 698-3 765) / 4 571 kg/d |
| product 2 | 16 748 kWh/d (16 345-17 900) | 55.6 % (52.0-56.9) | 44.4 % (43.1-48.0) | 8 272 kWh/d (7 041-10 746) | 1 663 (1 415-2 160) / 3 740 kg/d |

CO2 uses the project factor (0.201 kg/kWh gas; "PLH" read as "placeholder"); steam is
reconstructed from the air balance, the values are placeholders as listed, not plant statements.

## 6. Missing data and open questions for the partner

1. Compressor heat into the storage loop (or compressor load profile); heat rejected over the cooler.
2. Storage volume (or the two other height sensors and a mass/volume figure).
3. Order in the storage loop (A or B).
4. Steam flow or gas consumption of the steam heater; boiler efficiency.
5. Air flow through the preheater (bypass/recirculation?); intake humidity.
6. What the auxiliary air heater is heated by and whether it touches the waste-heat circuit.

## 7. Recommendation for S2

The model is NOT a predictor: the volume is unidentified (bound only), the exchanger conductances
differ between the days, the storage rollout is worse than persistence, and the compressor heat that
drives everything is unmeasured. It is usable for S2 only as a scenario band (all placeholders
varied on a grid, every number labelled); it must not be read as a statement about the plant.

## 8. Reproducibility

`scripts/study_storage_a1.py --name study_storage_a1` (clean tree, script committed before the run;
artifact `study_storage_a1.json` with config fingerprint, hashes of the raw files, provenance).
The call needs the two environment variables of the plant-validation scripts and the git-ignored
role file; paths are in `local/` only.


## 9. S2: shifting potential of storage A1 (SCENARIO BAND, not a plant statement)

Pre-registered in the docstring of `scripts/gate_storage_shift.py` (commit `191a227`, before the
run); artifact `gate_storage_shift.json`; post-hoc analysis `gate_storage_shift_posthoc.json`
(`scripts/analyze_gate_storage_shift.py`, declared post-hoc). Question: how much steam / CO2 could
better operation of the storage save at all? Policies on one discrete model (5 min steps): R0 replay
of the logged preheater-loop flow, R1 rule (maximum flow while `T_S > T_S,min`, takes all discarded
compressor heat that fits below `T_S,max`), R* perfect-foresight optimum over the day (backward
dynamic programming on a 0.25 K grid; chosen over an LP because the duty is non-linear in state and
flow). Decisions: preheater-loop flow and the share of the discarded compressor heat `Q_dump(t)`
taken into the storage loop. Constraints: `T_S` in [40, 90] degC (placeholders), supply-air set point
kept (`Q_out <= Q_air`), exchanger approach floors, terminal `T_S >= T_S(end of R0)`; all steam
energies corrected to the same stored energy (credit factor 1, optimistic). Grid (all placeholders):
`Q_dump` none / 100 / 300 kW / day profile (500 kW 06-18 h, 100 kW otherwise) / 800 kW; volume
5 / 10 / 20 / 30 / 50 m3 (10 = reference V0, 5 and 30 = ends of the S1 band, 20 = 2 x V0, 50 = 5 x V0);
weather: both measured days, ambient temperature -5 / 0 / +5 K. `Q_in_base(t)` (the compressor
heat the storage loop takes today) is reconstructed once for V0.

**R0 check.** Product 1: steam energy of R0 deviates by -0.2 % from the air-balance reconstruction of
the measured day, storage temperature RMSE 1.8 K (max 3.2 K) -- reproduced. Product 2: steam
deviates by -1.0 %, but the storage-temperature RMSE is 7.8 K (max 11.0 K) -- NOT reproduced by the
pre-registered criterion (<= 2 K): one preheater conductance does not describe that day (section 4).
Savings on product 2 are read relative to the model's R0 and carry that misfit.

**Table (weather 0 K, volume 10 m3; every row is a scenario; steam and CO2 per day).**

| product | `Q_dump` scenario | steam kWh/d R0 / R1 / R* | CO2 kg/d R0 / R1 / R* | saving R* | gap R1 - R* | extra heat taken by R* (kWh/d) |
|---|---|---|---|---|---|---|
| 1 | none | 13763 / 13763 / 13763 | 3074 / 3074 / 3074 | 0.0 % | 0.0 % | 0 |
| 1 | const_100 | 13763 / 11371 / 11379 | 3074 / 2539 / 2541 | 17.3 % | -0.1 % | 2392 |
| 1 | const_300 | 13763 / 7848 / 7793 | 3074 / 1753 / 1740 | 43.4 % | 0.4 % | 5988 |
| 1 | day_profile | 13763 / 9811 / 9768 | 3074 / 2191 / 2182 | 29.0 % | 0.3 % | 4007 |
| 1 | const_800 | 13763 / 7608 / 7533 | 3074 / 1699 / 1682 | 45.3 % | 0.5 % | 6249 |
| 2 | none | 7359 / 7359 / 7359 | 1643 / 1644 / 1643 | 0.0 % | 0.0 % | 0 |
| 2 | const_100 | 7359 / 4966 / 4973 | 1643 / 1109 / 1111 | 32.4 % | -0.1 % | 2392 |
| 2 | const_300 | 7359 / 3116 / 2770 | 1643 / 696 / 619 | 62.4 % | 4.7 % | 4600 |
| 2 | day_profile | 7359 / 4107 / 3910 | 1643 / 917 / 873 | 46.9 % | 2.7 % | 3458 |
| 2 | const_800 | 7359 / 3077 / 2728 | 1643 / 687 / 609 | 62.9 % | 4.7 % | 4643 |

Volumes 5-50 m3 and weather shifts are in the artifact; they move the saving moderately (product 2,
300 kW: 60.5 % at 5 m3, 62.4 % at 10, 66.1 % at 20, 76.2 % at 50; weather -5 / +5 K: 63.1 / 61.0 %).
Without discarded heat there is nothing to save in any scenario: R0, R1 and R* agree within 1 kWh/d.

**Gate (pre-registered).** Realistic scenario = `Q_dump > 0` and volume <= 20 m3 (72 scenarios).
All 72 reach a saving of at least 5 %; the gap R1 - R* of at least 2 % occurs in 29 scenarios, all
on product 2 (none on product 1). By the letter: **L1** over both days (product 1: L2, product 2:
L1; the conservative subset `Q_dump <= 300 kW` also L1). **This reading must not be quoted alone.**
(i) The decomposition puts the whole saving on "extra heat used": exchanger use 0 kWh/d, timing and
temperature level 0 to 1 kWh/d, extra heat 100 % -- the saving divided by the extra heat taken is
0.997-0.998 in every scenario. Pure shifting of heat in time, with the heat flow the storage loop
has today, saves nothing in this model: the duty is nearly flat above 15 m3/h of preheater-loop
flow (peak at 15-18.5 m3/h, which is where the logged flow already is), so there is nothing for flow
decisions to gain. (ii) The gap of R1 on product 2 comes from the max-flow rule: at high storage
temperature a flow near 30 m3/h gives a slightly LOWER duty than 15 m3/h. Post-hoc (not
pre-registered): the plainest rule R1b = logged flow plus all discarded heat leaves a gap of at most
0.9 % on the whole grid (product 1: below 0.1 %) and the gate rule with R1b gives **L2 on both days**.
(iii) The product-2 day fails the R0 check. Substantive reading, post-hoc: no shifting potential in
the sense of memory-dependent decisions (L3 for pure shifting); the whole saving is energy that is
only added, and a fixed rule that takes it captures it. Whether that heat exists and whether the
storage loop can take it up to `T_S,max` is not a control question but a process-integration
question for the partner. The saving is the added heat: it is not a result about an RL set-up.

**Sensitivity (reference: 300 kW constant, 10 m3, no shift; change of the R* saving in kWh/d,
product 1 / product 2).**

| rank | placeholder | effect |
|---|---|---|
| 1 | `T_S,max` (75 / 100 instead of 90 degC) | -2112 / -2696 and +858 / +1461 kWh/d |
| 2 | exchanger conductances (other day's fit; x0.8; x1.25) | +1187 / -2367; -1572 / -1567; +986 / +1283 kWh/d |
| 3 | topology A instead of B | +73 / +167 kWh/d |
| 4 | intake humidity, reference volume of `Q_in_base`, `UA_S` | each below 50 kWh/d |
| 5 | maximum preheater-loop flow, `T_S,min` | below 10 kWh/d |
| -- | fan heat fraction | no change in kWh, changes the percentage (-3.3 / -8.0 points at 0.5) |
| -- | boiler efficiency | no change in kWh; CO2 saving +167 / +128 kg/d at 0.8, -133 / -102 at 1.0 |

Not on this list because they are scenario axes: the availability of discarded heat (decides
everything: 0 kW saves nothing) and the volume (moderate).

**Questions for the partner (from the ranking).** (1) Is there discarded compressor heat at all, how
much and when, and can the storage loop absorb it? (2) Maximum admissible storage temperature and
the cooling-water temperature the compressors need (decides `T_S,max`). (3) Design data or
measurements of the two exchangers (conductance, fouling), and the air flow through the preheater.
(4) Order in the storage loop. (5) Storage volume. (6) Boiler efficiency (CO2 only).
