# NN -- Static surrogate of the dryer balance: pretraining on the partner balance and fine-tuning on few plant points, against calibration and a residual MLP

Status: **run and evaluated.** Design and pre-registered reading (sections 1-12) were committed BEFORE the pilot
and the main run (commit `360f858`); results added after the run (sections 13-14), numbers only from the artifacts named there. Branch
`static-transfer` (from `master` cd712df), worktree `C:/twinlink_work/static_transfer`. The document number
NN and the finding number ("Befund NN") are assigned by the managing session at the merge. Unattended run
over night (decision of the project lead, 2026-11-05): no questions during the run; where the approval was
ambiguous the most conservative reading was taken and is listed in section 12.

## 1. Question

From how many operating points of a plant, and for which kind of deviation, does an MLP that was pretrained on
the partner's stationary dryer balance and then fine-tuned on the plant points beat the alternatives

* **L0** the balance alone,
* **L1** the balance with 2-4 calibrated physical parameters (here four: heater efficiency, heat loss, false-air
  share, constant product-moisture excess),
* **L2** the balance plus a residual MLP (learns plant - balance, starts at zero),
* **L3** the pretrained MLP with fine-tuning,
* **L4** the MLP from scratch?

Targets: exhaust temperature [K], exhaust humidity [g/kg], heater power [kW per kg/s of dry air] (= fuel x
calorific value / 3600 per unit of air flow; the calorific value cancels). The recovered heat is not a target
(decision E5).

The model study A1 (Befund 88, docs/17) found pretraining + fine-tuning ahead of "from scratch" at small N for an
LSTM on dynamic, synthetic plants. Here the source model is STATIC and a closed formula, so the competitors
L1 (physics with few parameters) and L2 (physics plus a learned residual) are much stronger than in A1.

## 2. Decisions of the project lead (2026-11-05)

* **F1 input set:** `TwinLink Excel.xlsx`, dryer 2 (fluidised bed), values as committed in docs/11 section 13.1
  (properties of 4.1_2: `cp` gas 1.006, vapour 1.906, liquid 4.185, solid 1.0 kJ/(kg K), evaporation enthalpy
  2500 kJ/kg, equilibrium moisture 2 g/kg; heater efficiency 80 %; dryer 2: 100 kg/h of air at 6 g/kg and
  150 degC, 3 kg/h of spray at solid fraction 0.4 and 20 degC; ambient 20 degC, 1 atm). `Model v4.1_2.xlsx`
  only as an L0 sensitivity (`L0_v412`: efficiency 70 %, spray at 15 degC, docs/11 sections 12.1 and 13.1).
  `Model v4.1.xlsx` (4.1_1) is not used (docs/11 section 7.3: `cp` gas 1.2, gas formula off by a factor of
  108.9). No value from `local/` is committed that is not already in docs/11. The workbook is not read at run
  time; a test (skipped without `local/`) compares the YAML with both workbooks.
* **F2 stage 2:** halted and reported (section 10), with the coverage finding as its own section.
* **F3:** exception from `DataSpec.new_study` approved (precedent docs/25 section 6): the study generates no
  time-series data. Recorded here and in one line of docs/12.
* **E1-E6:** the recommendations of the draft are accepted: E1 closed-form product-moisture law instead of
  `steady_state.py` (outside the range of the laboratory model at the partner inputs, `flowsheet/bed_state.py`);
  E2 L1 with four parameters and a weak prior; E3 a-priori scales (section 6.1); E4 dryer 2 only; E5 no
  recovered heat; E6 single nets as in A1.
* **Changes of the approval:** (1) Holm per hypothesis family as the main form, "resolved better" = Holm-
  significant AND the same sign in at least 4 of 5 seeds, effect size = median of the paired difference in
  physical units, uncorrected verdicts only beside it; (2) a second condition "input errors" (section 4.3);
  (3) the intensive formulation, pretraining box `T_in` 60-200 degC and `X_s` 0.35-0.70, two more test sets
  "plant-like" and "product change"; (4) families relative to L1 fixed before the run, and per test set the
  smallest N from which L2 (L3) beats L1 and the residual error of L1 at N = 64; (5) a plant below 2 sigma in
  the deviation table is NOT dropped and nothing is changed; it is marked in the report.
* **Budget:** no fixed limit; the run must be finished by Friday 2026-11-06, 16:00. If the extrapolation says
  otherwise: first drop the high noise level, then the L0 sensitivity v4.1_2. Dry run over all units, at most
  8 workers, saving per unit, `--resume`. Full suite before the run commit. A German report, unversioned, in
  `local/reports/`.

## 3. Source model and the intensive formulation

The source is the adiabatic block of the equation document (equations 1.1-1.5) as rebuilt in
`partner_v4_1/dryer_block.py`, called point by point (`static_transfer/balance.py`); the heater heats the
process air from ambient (`heater_power_w`, vectorised, pinned by a test). Product moisture = equilibrium
moisture (an input of the sheet, not a target), constant efficiency, no heat loss.

**Intensive formulation** (decision of the project lead): inputs `x` = (inlet temperature `T_in` [degC],
specific spray load `M_s / M_g` [kg/kg], ambient temperature `T_amb` [degC], inlet humidity `Y_in` [g/kg],
solid fraction `X_s`); the heater power is given per unit of dry-air flow. Without heat loss all three targets
depend on these intensive quantities only. The heat loss (`UA (T_out - T_amb)`) is the one extensive term; it
is given per kg/h of process air (`UA = u M_g`, `u` in W/K per kg/h) and the plant's dry-air flow is the
source's 100 kg/h at every point, so `M_g` is not an input (section 12, item 1). The reproduction is tested:
at the nominal point the source gives 102.86 degC and 23.98 g/kg, the values of docs/11 section 13.1.

## 4. Synthetic plants (stage 1)

### 4.1 Plants and their strengths (fixed before the run, `conf/static_transfer/static_transfer.yaml`)

| Plant | Deviation from the balance | Affected targets | Family relative to L1 |
|---|---|---|---|
| P0 | none (control) | none | - |
| P-loss | heat loss `u` = 0.02 W/K per kg/h (= 2.0 W/K at 100 kg/h, laboratory placeholder of docs/11 9.6) | `T_out` | within |
| P-air | atomising air 2.5 % of the process air into the bed at 25 degC (docs/34: 1.7-2.7 %, 25 degC placeholder) AND false air 5 % of the process air at the ambient state into the exhaust before the sensors (placeholder) | `T_out`, `Y_out` | partial (false air within, atomising air not) |
| P-eta | heater efficiency `eta(q) = 0.85 - 0.60 (q - 0.90)^2`, `q` = specific duty / nominal specific duty (`eta(0.4) = eta(1.4) = 0.70`; placeholder) | power | outside |
| P-moist | product moisture above equilibrium, `X_p = X_eq + 0.20 RH / (RH + K)` at the relative humidity of the bed exhaust, `K` so that the excess is 0.06 kg/kg at the nominal point (placeholder), solved implicitly | `T_out`, `Y_out` | outside (variable `Delta X_p`) |
| P-comb | all four together | all three | outside |
| PP-st | pilot only: half strengths, own efficiency curve (`0.83 - 0.40 (q - 1.0)^2`) and moisture law (0.03 / 0.10) | all three | - |

L1's parameters are the heater efficiency (constant), the heat loss `u`, the false-air share into the exhaust and
a constant moisture excess; this fixes the families above. The deviation of each plant from L0 on every test set
(RMSE and RMSE / sigma per target) is computed exactly in the dry run and in the main run (key
`deviation_table`).

### 4.2 Measurement noise (targets only; placeholders)

Primary: `T_out` sigma 0.5 K, `Y_out` 0.3 g/kg, power 2 % relative. High: 3 x primary (descriptive). One
standard-normal draw per seed is shared by both levels.

### 4.3 Input errors (second condition)

Unknown offsets of the inputs, fixed per seed and not known to any method, of the size of the "observed"
scenarios of Befund 116 (docs/33 section 2): inlet temperature 0.35 K, spray load 1.5 %, solid fraction 3.8 %
relative; inlet humidity 1 g/kg (set by the project lead, no source). Full magnitude, sign drawn per seed and
quantity (section 12, item 3). The methods see the measured inputs; the plant runs at `true = measured +
offset`; evaluation against the noise-free plant truth at the true inputs. Primary noise level. The hypotheses
are stated for error-free inputs; for this condition the study reports per plant whether the order L1 / L2 / B1
changes.

Conditions: `clean_low` (primary of every hypothesis), `clean_high`, `inputerr_low`.

## 5. Ranges, test sets, admissibility, coverage

Boxes (order `T_in` degC, load kg/kg, `T_amb` degC, `Y_in` g/kg, `X_s`; placeholders unless noted):

| Box | `T_in` | load | `T_amb` | `Y_in` | `X_s` |
|---|---|---|---|---|---|
| pretraining | 60-200 (lead) | 0.010-0.060 | -10-40 | 0.5-16 | 0.35-0.70 (lead) |
| interpolation (training designs, test set) | 130-170 | 0.024-0.036 | 5-30 | 3-10 | 0.38-0.42 |
| extrapolation (outer box minus interpolation box) | 110-190 | 0.015-0.045 | -5-35 | 2-14 | 0.38-0.42 |
| plant-like | 75-95 | 0.020-0.055 | 0-25 | 1-6 | 0.60-0.70 |
| product change | 130-170 | 0.024-0.036 | 5-30 | 3-10 | 0.58-0.62 |

Plant-like: generic placeholders after docs/34 (supply air 83-85 degC on the two days; the load range spans the
request's placeholders of 2 t/h of suspension at 30 000 Nm3/h and the day factors 0.47 / 1.06 and 1.10 / 1.02 of
docs/34 4.1; the solid-fraction band 0.60-0.70 of docs/34; dried or dry supply air). Product change: training at
`X_s` near 0.40, test at 0.58-0.62.

**Admissibility:** a point is kept only if the bed exhaust of the source AND of every plant (including the pilot
plant) stays at or below 90 % relative humidity (placeholder; IAPWS-IF97 saturation). Every set is the first
`count` admissible points of its own scrambled Sobol stream (test sets: seed 20261105 + 1..4; pilot validation
set: + 11; training design of seed `s`: 3000 + `s`, nested prefixes, shared by every plant and condition). The
pretraining points need admissibility for the source only (the plants are unknown to it). Candidates and
admissible counts are recorded. Test sets: 512 points each.

**Coverage** (recorded per set): share inside the pretraining box and the range per input. In stage 1 every set
lies inside the pretraining box by construction.

## 6. Methods

| Method | What | Setting |
|---|---|---|
| L0 | balance (`TwinLink Excel`) | - |
| L0_v412 | balance with the v4.1_2 input set | - (sensitivity) |
| L1 | four parameters, bounds efficiency 0.4-1.0, `u` 0-0.2, false air 0-0.5, excess 0-0.3; bounded MAP (`least_squares`, trf): residuals / primary noise, Gaussian prior around the nominal values (0.8, 0, 0, 0) with sigmas 0.10, 0.02, 0.05, 0.05 x `prior_scale` | `prior_scale` from the pilot |
| L2 | L0 + MLP on the residual / fixed residual scales (2 K, 1 g/kg, 5 kW/(kg/s)), last layer initialised at zero, weight decay | lr, steps, decay from the pilot |
| L3 | MLP pretrained on 8192 balance points of the pretraining box (1024 validation points, minibatch Adam lr 1e-3, batch 256, at most 500 epochs, patience 40, best kept), one per seed; fine-tuned on the plant points: all parameters, full-batch Adam, L2-SP towards the pretrained weights | lr, steps, L2-SP lambda from the pilot |
| L3_0 | the pretrained MLP without fine-tuning (zero shot) | - |
| L4 | the same MLP from scratch, full-batch Adam, weight decay | lr, steps, decay from the pilot |
| B1 | L0 + the mean training residual per target (residual of degree 0) | - |
| B2 | linear ridge on the plant points, penalty per target by exact leave-one-out (1e-6 ... 1e3) | - |

All MLPs: 5 -> 64 -> 64 -> 3, tanh, float64; one torch thread per worker; net seed from (seed, method, N).

### 6.1 Scaling (E3)

Inputs of every MLP and of B2 are mapped from the PRETRAINING box to [-1, 1]; MLP targets by fixed centres and
scales (80 degC / 40 K, 25 / 15 g/kg, 130 / 50 kW per kg/s), L2's residuals by fixed scales. These are a-priori
constants of the design, not a statistic of any data, so no information of a test point enters a fit (no leak),
and L2, L3 and L4 see the same scaling. A statistic from the N training points would be degenerate at N = 2. B2
centres its features and targets on the training points only.

### 6.2 Pilot (settings; rule as in A1)

Pilot plant PP-st only (never in the main run), pilot seeds 11-13 (own pretraining, own designs), N = 4 and 16,
condition `clean_low`, scored on the pilot's own validation set (512 points of the interpolation box, own
scramble; none of the four test sets is used for any selection). Criterion C = mean over seeds, N and targets of
RMSE / primary noise (power: 2 % of the mean true power of the set). One setting per method for every plant.
Grids: L1 `prior_scale` 0.25, 1, 4, 16; L2 lr 1e-3, 3e-3, 1e-2 x steps 300, 1000, 3000 x decay 1e-3, 0.1, 10; L3
lr 1e-4, 3e-4, 1e-3, 3e-3 x steps 100, 300, 1000, 3000 x L2-SP 0, 1e-3, 0.1; L4 lr 1e-3, 3e-3, 1e-2 x steps
1000, 3000, 10000 x decay 1e-5, 1e-3, 0.1. Selection = lowest C. A selected value at the edge of its dimension
extends that dimension by one geometric step and the selection is repeated (at most two extensions per
dimension; an edge at 0 is a natural bound). Whether a selection still lies at an edge after the extensions is
recorded and reported. The selection is copied into the `selected` block of the YAML in its own commit; the main
run refuses to start if the two differ or if anything else in the YAML changed.

## 7. Design

Seeds 1-5 (design scramble, noise draw, input-error signs, net initialisation, pretraining 101-105). N = 2, 4, 8,
16, 32, 64 (nested). Units = plant (6) x condition (3) x seed (5) = 90; every unit fits L1, L2, L3, L4, B1, B2 at
every N and scores L0, L0_v412, L3_0 once. Metrics per test set and target: RMSE, largest absolute error, mean
error, against the noise-free plant truth. Learning curves: median and range over the seeds.

## 8. Statistics

Paired t-test over the 5 seeds (`evaluation/paired_resolution.paired_comparison`, two-sided, 4 degrees of
freedom) on the RMSE in physical units; Holm (`evaluation/b1_analysis.holm`, alpha 0.05) over all tests of ONE
family. **Resolved better** = Holm-rejected AND the candidate better in at least 4 of 5 seeds; resolved worse
likewise. Uncorrected verdicts (p < 0.05 and the same sign rule) are reported beside it. Effect size = median
of the paired differences (reference - candidate) in the unit of the target. "L3 beats X in a cell (plant, N)"
= resolved better on every affected target of the plant (section 4.1). Rank statements only with this rule.

## 9. Hypotheses and pre-registered reading

The reading is fixed in the docstring of `scripts/analyze_static_transfer.py` (committed with this document);
primary condition `clean_low`. Family sizes in brackets.

* **H1 (control, P0, interpolation; 108):** none of L1, L2, L3, L4, B1, B2 is resolved better than L0 at any N on
  any target. On P0 the noise-free truth IS L0, so H1 can only fail through an error of the pipeline (section 12,
  item 8); the informative part is the reported cost of each method where there is nothing to learn and the
  zero-shot error of L3_0.
* **H2 (within, P-loss, all four test sets, `T_out`; 24):** L3 is resolved better than L1 in no cell.
* **H3 (outside: P-eta, P-moist, P-comb, interpolation; 36):** per plant N* = smallest N from which L3 beats L1
  at that and every larger N. Supported if N* exists for all three; partly supported for one or two (named);
  not supported for none.
* **H4 (interpolation, five deviating plants; 54):** L3 is resolved better than L2 in no cell on an affected
  target.
* **H5 (extrapolation; 108):** (a) L3 resolved better than L4 on every affected target of every deviating plant
  at every N <= 16; (b) on P-loss and P-air L1 resolved better than each of L0, L2, L3, L4, B1, B2 at every
  N <= 16. Supported if (a) and (b); both parts reported.

**Reading** (interpolation, `clean_low`; family of 270 tests: L3 against L0, L1, L2, L4, B1 on the five deviating
plants, affected targets, every N):

| Case | Condition | Consequence |
|---|---|---|
| R1 | some cell (plant, N) has L3 beating all five of L0, L1, L2, L4, B1 | pretraining is worth it there; the map of such cells is the result |
| R2 | otherwise some cell has L3 beating L0, L1 and L4 (whether L3 beats L2 anywhere is reported) | with the balance available at run time a residual model does the job; pretraining is not needed |
| R3 | otherwise some cell has L3 beating L4 | pretraining only carries the physics where the balance itself cannot be used |
| none | L3 beats not even L4 in any cell | |
| R4 (flag) | L3 beats L4 in no cell with N <= 8 | a contradiction of A1 (Befund 88) for the static case |

The same reading is computed for the other three test sets and for `clean_high` (descriptive, own families).

**Core figures per test set** (family of 108 per set: L2 and L3 against L1, five deviating plants, affected
targets, every N): per plant the smallest N from which L2 (resp. L3) beats L1, and the residual error of L1 at
N = 64 (median over seeds, every target).

**Descriptive:** the input-error condition (order of L1 / L2 / B1 by median RMSE against `clean_low`, number of
cells where it changes, pairwise verdicts under input errors with Holm per plant); `clean_high`; `L0_v412`; the
deviation table (marked where a plant is below 2 sigma on an affected target); learning curves; L1's fitted
parameters.

## 10. Stage 2 (real data of plant A): halted

Stage 2 was to be run only with an INDEPENDENTLY measured target. It is halted (decision F2):

* **Heater power:** the main heater of plant A is a steam heater fed by a gas boiler (docs/23). The exports carry
  no steam, gas or heat meter of it; `REQUIRED_ROLES` of `plant_a_op/data.py` has none, the role template has
  only `steam_valve_position` as a proxy, and docs/23 lists the steam flow or boiler gas as missing data. A heater
  power computed from the air balance (`C_air (T_in - T_amb)`) is the balance itself: testing a balance surrogate
  against it is circular.
* **Exhaust humidity:** not measured at plant A (`plant_a_op/dryer.observed_outlet_extremes` reconstructs it from
  the balance and placeholders).
* **Exhaust temperature:** measured, but two exported days with one product each, and within a day the operating
  point moves by less than 0.5 % (docs/23): a split by days leaves effectively one operating point for training
  and one for the test. docs/34 (Befund 117) already showed that the bed-exhaust gap is day-specific. The solid
  fraction and the intake humidity are placeholders (Befund 103); any plant-A test would inherit them.

### 10.1 Coverage of plant A by the source (own section, decision F2)

* **Scale.** The partner input sets are laboratory scale: 100-150 kg/h of air per dryer in all three sets
  (docs/11 sections 7.2, 12.1, 13.1). Plant A runs at about 30 000 Nm3/h of process air (request placeholder; day
  aggregates 1.10 / 1.02 x, docs/34 4.1), about 300 times more. In extensive inputs (kg/h) no partner set and no
  pretraining range built around them covers plant A; any learned surrogate would extrapolate by two orders of
  magnitude. The intensive formulation of this study removes the scale from `T_out`, `Y_out` and the specific
  power, except the heat loss, which is extensive (section 3).
* **Supply temperature.** Plant A's supply air is at 83-85 degC (docs/34 4.1). The partner sets put the inlet at
  150-300 degC in every dryer of all three sets (dryer 2 of TwinLink Excel: 150 degC). Plant A lies below every
  operating point of the source; a pretraining box built around the source's points would not cover
  it. Only the pretraining box fixed by the project lead (60-200 degC) covers it, together with the plant-like
  load and solid fraction (the "plant-like" test set of stage 1 measures what that coverage buys in a synthetic
  setting).
* **Solid fraction and load.** Plant A's solid fraction is a placeholder (0.60-0.70 band, docs/34); the partner
  sets use 0.4 (0.5 in dryer 1 of 4.1_2). The specific load of plant A spans roughly 0.02-0.05 kg/kg from the
  placeholders and day factors of docs/34 (derived, not measured), against 0.03 kg/kg at dryer 2 of TwinLink Excel.

## 11. Compute budget and operation

Measured before this commit on this machine (scratch, one process, mid settings; NOT a result): one pretraining of
8192 points 27 s; one unit 12-15 s. Planned: pilot (3 pretrainings, about 106 settings x 6 fits, plus extensions),
then the dry run over all 90 units (pretraining 2 epochs on 256 points, at most 5 optimiser steps per fit, L1 at
most 3 evaluations; it records the time and the peak memory per worker), then the main run with 8 workers (8
physical cores, 31 GB of RAM). Extrapolated: well under one hour of wall time for pilot and main run together
(placeholder; replaced by the measured times in section 13). Deadline Friday 2026-11-06 16:00; fallback order:
drop `clean_high`, then the L0 sensitivity.

## 12. Decisions without asking (unattended run; conservative reading, disclosed)

1. **Heat loss per kg/h of air, fixed air flow.** "M_g only where UA makes it extensive; UA then per kg/h of air":
   read as: the heat loss is given as a conductance per kg/h of process air and the plant always runs at the
   source's 100 kg/h, so every method sees intensive inputs only and `M_g` is no input. Reason: one input set for
   all methods and plants (the methods do not know which deviation a plant has); at fixed air flow the specific
   and the absolute conductance are the same number (2.0 W/K). Not covered: a plant whose air flow varies with a
   fixed absolute UA (then the loss per kg of air changes with the flow).
2. **Moisture law of P-moist.** The draft's power law `0.06 (RH / RH_nom)^1.5` would give an excess of several kg/kg
   in the plant-like set (exhaust RH up to about 15 times the nominal one), which did not exist when the draft was
   approved. Kept: the nominal strength (0.06 kg/kg) and "rises with the exhaust RH, nonlinear"; changed to a
   saturating (Langmuir-shaped) law with a maximum excess of 0.20 kg/kg. A placeholder without source.
3. **Input errors at full size.** "+-x" read as magnitude x with a random sign per seed and quantity (not uniform in
   [-x, x], which would halve the typical size); the inlet humidity is clipped at 0 after the offset.
4. **Pilot on its own validation set**, never on a test set of the main run (otherwise the settings would be tuned
   on the test points).
5. **Admissibility** (90 % RH of the bed exhaust for the source and every plant) applied to every set, so test
   sets are identical across plants and conditions (paired); the plant-like set loses its most humid corner
   (counted in the record).
6. **Plant-like and product-change boxes** chosen as generic placeholders: plant-like `T_amb` 0-25 degC and `Y_in`
   1-6 g/kg; product change 0.58-0.62 against the training band 0.38-0.42; the extrapolation set keeps `X_s` at
   0.38-0.42 so that extrapolation in the operating variables and the product change stay separate.
7. **Evaluation** against the noise-free plant truth (at the true inputs under input errors); the methods see the
   measured inputs and noisy targets.
8. **H1 kept as approved** although it cannot fail on a correct pipeline (L0 is the truth of P0); it is reported as
   a check, its cost table as the content.
9. **Operational definitions:** H3 "supported" only if N* exists for all three outside plants (partly otherwise);
   H5(b) "L1 is the best method" = resolved better than each of the six others; the reading R1-R3 as an exhaustive
   order with R4 as an independent flag; "L3 beats X" on every affected target. The reading and the core figures are
   their own Holm families (not named by the project lead; one family per test set and condition).
10. **L1 weights** its residuals by the primary noise (a sensor specification) in every condition, also under the
    high noise level (the methods are not told the noise is higher).
11. **Noise draws shared** by the two noise levels (the high level scales the same draw) and the training design
    shared by every plant and condition: the conditions differ only in what they are meant to vary.
12. **L0_v412** is scored in every unit (it needs no fit); it costs nothing measurable, so the fallback order of the
    budget would not save time by dropping it.
13. **Order pilot -> dry run -> main run.** The dry run needs the selected settings to exercise the main-run code
    path with the real configuration; the pilot itself is cheap and runs first.

14. **Supersaturated truth under input errors (found by the dry run, before the main run; nothing changed).**
    Admissibility is checked at the measured inputs (the test sets must be the same for every seed). With the
    offsets of seed 2 (`T_in` -0.35 K, load +1.5 %, `X_s` -3.8 %, `Y_in` +1 g/kg) the plant at the TRUE inputs
    exceeds 100 % RH of the bed exhaust in 9 / 12 / 7 / 9 of the 512 plant-like test points for P0 / P-loss /
    P-air / P-eta (0 for P-moist and P-comb; largest RH 1.22, artifact `static_transfer_dryrun_v1`); no other seed,
    test set, training point or condition is affected. The balance has no saturation limit, so the truth there is
    the unlimited balance. Kept as is (the condition is descriptive; changing the sets would break the pairing);
    to be read as a caveat of the plant-like cells of `inputerr_low`, seed 2.

## 13. Results

Numbers only from the artifacts of 13.1; test sets named everywhere; `clean_low` unless noted; median over 5 seeds
of the RMSE against the noise-free plant truth, units K, g/kg, kW per kg/s.

### 13.1 Runs and provenance

| Step | Artifact | Script commit, tree | Remarks |
|---|---|---|---|
| pilot | `artifacts/runs/static_transfer_pilot_v1.json` | `360f858`, clean | 868 s, 8 workers, 0 failures; three grid rounds |
| selection into the YAML | commit `72f11b0` | - | checked by the main run against the pilot artifact |
| dry run | `artifacts/runs/static_transfer_dryrun_v1.json` | `72f11b0`, clean | 90 of 90 units, 0 failures, peak memory 385 MB per worker; numbers no result (decision 14 comes from it) |
| main run | `artifacts/runs/static_transfer_v1.json.gz` | `def4860`, clean | 90 units, 0 failures, 604 s wall time on 8 workers, peak 386 MB per worker; `def4860` = `72f11b0` plus the dry-run artifact and docs text |
| analysis | `artifacts/runs/static_transfer_v1_analysis.json`, figures `docs/figures/static_transfer_v1_<plant>_<test set>.png` | reading of `360f858`, run at `e9e174f`, clean | |

Full suite green on `360f858` and on `72f11b0` (logs in the gitignored `local/logs/` of the worktree). `.venv` of the
main checkout via `tools/run_here.sh`; every record carries `code/root_checked`.

### 13.2 Pilot (PP-st, pilot validation set)

| Method | Selected setting | C | Remark |
|---|---|---|---|
| L1 | `prior_scale` 4 | 0.470 | interior in the first round |
| L2 | lr 1e-2, 90 steps, weight decay 10 | 0.803 | interior after extensions (steps down twice, decay up once, lr up once) |
| L3 | lr 3.3e-5, 300 steps, L2-SP 0 | 0.758 | interior after extensions (lr down twice); flat: the five best settings all have C 0.758 |
| L4 | lr 3.3e-4, 10 000 steps, weight decay 0.1 | 4.818 | interior after extensions |

No selection lies at an edge after the extensions. **The fine-tuning setting of L3 adapts very little** (300 steps at
lr 3.3e-5); as in the A1 pilot (docs/17) it was chosen on a pilot plant of half the strength of the main plants
(consequence on P-comb: 13.7).

### 13.3 Deviation table, coverage, pretraining

L0 against each plant, RMSE / primary sigma, interpolation set (`deviation_table`): P-loss `T_out` 11.0; P-air
`T_out` 11.3, `Y_out` 4.2; P-eta power 2.3; P-moist `T_out` 3.5, `Y_out` 2.7; P-comb 18.3 / 7.0 / 2.3. **No plant is
below 2 sigma on an affected target in any test set** (smallest: P-eta, power, plant-like set, 2.1). Coverage: all
four test sets and all designs lie 100 % inside the pretraining box; admissible candidates per set 1024/1024
(interpolation), 949/1024 (extrapolation), 1011/1024 (plant-like), 1024/1024 (product change). Pretraining (5 seeds):
274-458 epochs, validation RMSE against the balance 0.057-0.090 K, 0.022-0.037 g/kg, 0.061-0.109 kW/(kg/s); zero-shot
error of L3_0 on P0, interpolation: 0.041 K, 0.019 g/kg, 0.069 kW/(kg/s).

### 13.4 Hypotheses (Holm per family; counts better / worse / unresolved)

| | Family | Result | Counts (Holm) | Uncorrected |
|---|---|---|---|---|
| H1 | 108 | **supported** (as it must be, section 12 item 8) | 0 / 23 / 85 | 0 / 96 / 12 |
| H2 | 24 | **supported**: L3 never better than L1 on P-loss | 0 / 13 / 11 | 0 / 23 / 1 |
| H3 | 36 | **partly supported**: N* P-eta 16, P-moist 64, P-comb not reached | 9 / 6 / 21 | 15 / 7 / 14 |
| H4 | 54 | **not supported**: one cell with L3 better than L2 | 1 / 2 / 51 | 10 / 8 / 36 |
| H5 | 108 | **not supported**: neither (a) nor (b) | 38 / 0 / 70 | 101 / 1 / 6 |

* **H1 (P0):** resolved worse than L0 per method: L4 9 cells, B2 9, L3 3, L2 1, B1 1, L1 0. The cost of learning where
  there is nothing to learn (`T_out`, N = 2 / 8 / 64): L1 0.35 / 0.14 / 0.026 K, L2 0.66 / 0.47 / 0.072, L3 0.67 / 0.86
  / 0.12, B1 0.42 / 0.21 / 0.061. The fine-tuned net is worse than its own zero-shot state (0.041 K) at every N:
  fine-tuning on noisy points moves it away from the balance.
* **H2 (P-loss, `T_out`, four test sets):** L3 is resolved WORSE than L1 in 13 of 24 cells, in every test set (median
  of the paired difference -0.16 to -0.60 K in the interpolation set at N = 8-64, down to -1.56 K on the plant-like
  set at N = 4). L1 recovers the heat loss (0.0196-0.0203 W/K per kg/h at N = 64, true 0.02).
* **H3:** P-eta (power): L3 beats L1 from N = 16 (median 1.41-1.74 kW/(kg/s), 5 of 5 seeds), L1's constant efficiency
  keeps a floor of 3.71. P-moist: N* = 64 (`T_out` from N = 32 by 0.20-0.27 K, `Y_out` only at 64 by 0.14 g/kg).
  P-comb: not reached -- L3 is better on the power from N = 16 but resolved WORSE than L1 on `T_out` at every N
  (median -0.82 to -1.27 K).
* **H4:** one cell, P-air, N = 32, `Y_out`: L3 better than L2 by a median 0.108 g/kg (p = 0.0002, 5 of 5 seeds). L3 is
  resolved worse than L2 on P-comb `T_out` at N = 32 and 64; every other cell unresolved. On all affected targets
  jointly L3 beats L2 in no cell.
* **H5 (extrapolation):** (a) L3 better than L4 only at N = 2 (9 of 36 cells: all affected targets of every plant at
  N = 2; from N = 4 every cell unresolved). (b) L1 better in 29 of 72 cells (against L0 12 of 12, B1 8, L4 5, B2 4,
  L2 0, L3 0): L1 is not resolved better than L2 or L3 at N <= 16 on P-loss and P-air, although its medians are lower
  (13.7).

### 13.5 Reading

| Test set | `clean_low` | `clean_high` (descriptive) |
|---|---|---|
| **interpolation (primary)** | **R3**, R4 flag not set | R3 |
| extrapolation | R3 | R3 |
| plant-like | none (R4 flag set) | none (R4 flag set) |
| product change | R3 | R3 |

**Pre-registered reading: R3.** L3 beats L4 (all affected targets) at N = 2 on every deviating plant and at no larger
N; it never beats L0, L1 and L4 jointly in a cell, and it never beats L2 on all affected targets jointly (reading
family: 270 tests, 78 better / 6 worse / 186 unresolved). Pretraining therefore only carries the physics into a net at
the smallest N; where the balance itself can be used at run time, L1, L2 or B1 do the same or better. On the
plant-like set (supply air 75-95 degC, inside the pretraining box, outside every point the plant was trained on) L3
does not even beat L4 in any cell: the R4 flag is set there, a contradiction of A1 in this set.

### 13.6 Core figures per test set (family of 108 per set)

Smallest N from which L2 (resp. L3) beats L1 on all affected targets ("-" = not within 64):

| Test set | P-loss | P-air | P-eta | P-moist | P-comb |
|---|---|---|---|---|---|
| interpolation, L2 / L3 | - / - | - / - | 32 / 16 | 64 / - | - / - |
| extrapolation, L2 / L3 | - / - | - / - | 32 / 32 | 32 / 64 | - / - |
| plant-like, L2 / L3 | - / - | - / - | - / - | - / - | - / - |
| product change, L2 / L3 | - / - | - / - | - / - | - / - | - / - |

(H3 finds N* = 64 for P-moist, the core family does not: same cell, a family of 36 against 108 tests under Holm.)

Residual error of L1 at N = 64 (affected targets):

| Test set | P-loss `T` | P-air `T` / `Y` | P-eta power | P-moist `T` / `Y` | P-comb `T` / `Y` / power |
|---|---|---|---|---|---|
| interpolation | 0.047 | 0.167 / 0.019 | 3.71 | 0.45 / 0.235 | 0.351 / 0.224 / 3.71 |
| extrapolation | 0.050 | 0.271 / 0.019 | 14.58 | 1.11 / 0.519 | 0.740 / 0.498 / 14.58 |
| plant-like | 0.048 | 0.233 / 0.119 | 6.82 | 3.55 / 1.58 | 0.901 / 0.554 / 6.82 |
| product change | 0.055 | 0.204 / 0.065 | 3.71 | 1.13 / 0.635 | 1.61 / 1.25 / 3.71 |

L1 is exact where its structure contains the deviation (P-loss) and compensates where it does not (P-air at N = 64:
false-air share 0.065-0.073 for the true 0.05 of false air plus 0.025 of atomising air; P-comb: heat loss 0.035-0.043
and moisture excess 0.12-0.15 for the true 0.02 and 0.06); the floors are the costs of a wrong structure (P-eta power
3.71 in the interpolation set, 14.6 in the extrapolation set).

### 13.7 Learning curves (descriptive, interpolation, N = 2 / 8 / 64; figures in `docs/figures/`)

| Plant, target | L0 | L1 | L2 | L3 | L4 | B1 | B2 |
|---|---|---|---|---|---|---|---|
| P-loss `T` | 5.49 | 0.43 / 0.20 / 0.047 | 0.95 / 0.62 / 0.19 | 0.80 / 0.65 / 0.21 | 11.8 / 0.67 / 0.25 | 0.92 / 0.93 / 0.91 | 12.3 / 0.61 / 0.15 |
| P-air `Y` | 1.26 | 0.11 / 0.038 / 0.019 | 0.29 / 0.21 / 0.20 | 0.25 / 0.12 / 0.097 | 2.62 / 0.44 / 0.13 | 0.18 / 0.16 / 0.15 | 2.72 / 0.33 / 0.10 |
| P-eta power | 7.33 | 4.52 / 4.17 / 3.71 | 3.84 / 3.50 / 1.94 | 3.89 / 2.56 / 2.02 | 20.3 / 8.44 / 2.02 | 4.02 / 3.27 / 3.13 | 21.1 / 5.19 / 2.13 |
| P-moist `T` | 1.73 | 0.49 / 0.49 / 0.45 | 0.80 / 0.44 / 0.16 | 0.84 / 0.62 / 0.17 | 12.1 / 0.81 / 0.30 | 0.65 / 0.64 / 0.61 | 12.7 / 0.55 / 0.21 |
| P-comb `T` | 9.17 | 0.75 / 0.44 / 0.35 | 2.25 / 0.89 / 0.26 | 2.03 / 1.50 / 1.17 | 10.4 / 0.69 / 0.29 | 2.23 / 2.22 / 2.20 | 10.8 / 0.56 / 0.18 |

* At N = 2 the learned nets without physics (L4) and the linear ridge (B2) are useless (10-21 in every unit); from
  N = 8-16 they reach L2 and L3 in interpolation. B2 has the lowest median on several cells at N = 64 (P-comb `T`
  0.18 K) but falls apart outside the training range (plant-like P-comb `T` 8.8 K at N = 64).
* L3 keeps a floor on P-comb `T` (about 1.2 K): its fine-tuning setting adapts too little for the strongest plant
  (13.2).
* Plant-like set: every learned method is worse than L0 on the P-eta power at N = 64 (L2 16.1, L3 13.7, L4 19.4 against
  L0 4.03; L1 6.82).
* Product change: L1's constant moisture excess does not carry over to the other solid fraction (P-moist `T` at N = 64:
  L1 1.13 K, L2 0.23, L3 0.25; descriptive, no resolved core cell).

### 13.8 Input errors (descriptive)

Order of L1 / L2 / B1 by median RMSE, `inputerr_low` against `clean_low`, over 72 cells per plant (4 sets x 6 N x 3
targets): changes in 40 (P0), 32 (P-loss), 18 (P-air), 41 (P-eta), 21 (P-moist), 19 (P-comb) cells. Of 168 changed
pairwise verdicts (Holm per plant) 163 lose their resolution (135 better -> unresolved, 28 worse -> unresolved); 3
become better and 2 worse. The offsets change sign from seed to seed, so they add scatter between the seeds and cost
the paired test its resolution. In the medians L1 loses most where it was exact: P-loss `T` at N = 64 0.047 -> 0.173 K
(L2 0.188 -> 0.194, B1 0.912 -> 0.932) -- the calibration absorbs the input offsets into its physical parameters. L0
on P0: 0.847 K from the offsets alone. Caveat: decision 14 (seed 2, plant-like, 7-12 supersaturated points for four
plants).

### 13.9 L0 sensitivity to the input set (v4.1_2)

Efficiency 70 % instead of 80 % and spray at 15 degC: on P0, interpolation, power 24.2 kW/(kg/s), `T_out` 0.41 K,
`Y_out` 0. The input set moves the power by far more than any plant deviation of this study (P-eta 7.33); `T_out` and
`Y_out` hardly.

## 14. Reading and answer

**Answer to the question (synthetic static plants, this input set, these strengths and noise levels).** A pretrained
and fine-tuned MLP beats the balance-based alternatives jointly in no cell (reading R3):

* Against L4 (from scratch) it wins only at N = 2 -- and on the plant-like set not even there.
* Against L1 (calibration) it wins only where L1's structure lacks the deviation: the load-dependent heater efficiency
  (power, from N = 16) and the variable product moisture (from N = 32-64). Where the deviation lies within L1's
  parameters (heat loss, false air) it never wins and is often resolved worse.
* Against L2 (balance plus residual MLP) it is resolved better in a single target cell (P-air, `Y_out`, N = 32) and on
  all affected targets jointly never. L2 starts from the exact balance and does not need to learn what pretraining
  teaches.

In a static setting with the balance available at run time, pretraining is not the lever. The calibration of a few
physical parameters (L1) is the best choice wherever the structure fits, at every N; a bias correction (B1) is a strong
baseline at N <= 4; a residual model (L2) or the pretrained net (L3) adds value only for deviations outside L1's
structure, from N = 16-64. Outside the trained operating range (plant-like set) no learned method is reliable; L1 is the
only method that keeps the errors near those of the interpolation range for structures it contains.

**What is not concluded.** The fine-tuning setting of L3 was selected on a milder pilot plant and adapts little (a
floor of about 1.2 K on P-comb); a stronger fine-tuning might move L3 towards L2 but, by H4, hardly beyond it. All
plants are variants of the same balance with placeholder strengths and noise. Five seeds give four degrees of freedom:
"unresolved" is not "equal". The input-error condition varies the offset signs by seed and therefore mostly costs
resolution. Stage 2 was not run (section 10); nothing here is a statement about plant A.
