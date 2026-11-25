# NN -- Process 1 of the partner sweep sheet against the measured operation of plant A: are the inputs those of the plant, and is the over-saturation a matter of the inputs or of the unit?

Status: **design, code and reading committed before any run of the report script.** Sections 1
to 13 are the design; results are added below them after the run and take their numbers only from
the artifact named there. Branch `sheet-plant-a` (from `master` 24476e3), worktree
`C:/twinlink_work/sheetpa`. Document and finding number are open (the managing session assigns
them at the merge). Code `src/twinlink_ml/dyssol_steady/plant_check.py`, script
`scripts/report_sheet_plant_a.py` (the reading is in its docstring), tests
`tests/test_sheet_plant_a.py`.

**Everything here stands under two readings that the sender has not confirmed:** "the gas mass
flow of the sheet is the dry gas" (inferred in Befund 130), and "process 1 of the sheet is the
line called plant A in this repository" (from the label of the sheet alone; section 12).

## 1. Occasion and questions

Befund 130 found the dryer exhaust of process 1 of the partner sheet of 2026-11-24 over-saturated
at every point (saturation ratio 2.015, in Dyssol and in the rebuild alike). Since then process 1
is excluded: `docs/49` computes no optimum for it, Befund 131 makes no statement about it. All of
Befund 128, 130 and 131 is model against model.

By its label in the sheet, process 1 is the line of which the repository has two measured days
(plant A; Befund 117, `docs/34_plant_scale.md`). That makes three questions answerable without the
sender:

1. **Does the unit over-saturate at the measured operation of plant A?** If not, the
   over-saturation of process 1 is a matter of its inputs, not of the missing saturation limit.
2. **Does the dryer unit reproduce the measured exhaust temperature** when it is given the
   measured inputs? This is the first comparison of the delivered units with a measurement.
3. **Which input of process 1 carries the over-saturation?**

## 2. Task of the project lead (2026-11-25)

The draft of this document (written in `local/reports/` on 2026-11-25, before any code) was
released with all ten proposals of section 11 accepted, and with two additions:

1. **Reservation.** The assignment "process 1 = plant A" rests on the label of the sheet alone;
   the question "for which product or which operating point does process 1 stand?" goes onto the
   list of questions to the sender.
2. **A short recommendation for the sender from part B** in the local German report: which solid
   fraction or which spray rate makes process 1 admissible -- as a hint, not as a partner value
   in the repository.

Then as planned: branch, worktree, code and reading committed before the run, dry run, full
suite, run on the clean tree, artifact, lab-book entry "Befund NN (number open)". No merge, no
push.

## 3. Placing

* **Not a calibration and not a model comparison.** No parameter is fitted, no model is trained,
  no seed exists. The grey-box model of plant A (Befund 117, 123) is not re-run; its numbers are
  quoted where they help to read the result.
* **The unit is known to be simpler than the line:** no wall loss, no atomising air, gas and
  product leave at one temperature, no saturation limit (`docs/45`, section 3). Befund 117 found
  that one outlet temperature cannot represent plant A (bed minus exhaust 6.71-7.27 K /
  9.48-11.05 K). Reading T below is therefore expected to fail; the size and the sign of the miss
  are the result.
* **Relation to existing findings.** No number of Befund 117, 123, 128, 130 or 131 changes. A
  verdict F0 with A0 would add a scope note to Befund 130 ("the over-saturation of process 1
  follows from its inputs"); that is a decision of the project lead at the merge.

## 4. Data and mapping

**Test set: plant A, two measured days (day 1 = product 1, day 2 = product 2; day and product are
confounded), four stationary 60-minute segments per day -- the segments of Befund 117.** The
exports are read through `TWINLINK_PLANT_DATA_PROD1` / `TWINLINK_PLANT_DATA_PROD2` with
`plant_scale.data` and the unchanged settings of `conf/plant_scale/plant_scale.yaml` (stationarity
criterion, normal density of dry air). No new selection of windows.

**The sheet** is read with `dyssol_steady.sweep_sheet` (gas flow = dry gas), process 1, without a
medium flow. The heat recovery is not part of this study.

**Inputs of the unit and where they come from, per segment (segment means):**

| Input of the unit | Source at the plant |
|---|---|
| dry-gas flow | measured process-air flow, normal cubic metres of dry air times the normal density (as Befund 117; a mass unit) |
| dryer inlet temperature | measured supply-air temperature (mean of the sensors) |
| spray flow | measured spray rate |
| fresh-gas temperature | measured intake temperature (acts on the heater only) |
| solid fraction of the spray | **not measured.** Reference 0.65, band 0.60 / 0.634 / 0.70 (placeholders of Befund 117); in addition the value of the sheet |
| humidity of the gas | **not measured.** Reference: the dried-supply-air placeholder of Befund 117; in addition the humidity ratio of the sheet |
| spray temperature, equilibrium moisture, heater efficiency, calorific value | the values of the sheet |

The atomising air is measured but is no input of the unit; it is left out and named as a caveat.

**Methodology note.** As in `docs/45`: no grey-box data are generated, the preset
`DataSpec.new_study` does not apply (section 11, item 8).

## 5. Evaluated points

* **S:** the sheet's process 1 as delivered (1 point).
* **M:** the 8 segments x 5 solid fractions (0.65, 0.60, 0.634, 0.70, the sheet's) x 2 humidities
  = 80 points. **Reference variant: solid fraction 0.65, dried supply air.**
* **R:** single replacements for reading A. Starting from S, ONE input is replaced by the value of
  a day (day aggregate = mean over its four segments, as in `docs/34`): dry-gas flow, dryer inlet
  temperature, spray flow, solid fraction (0.65), gas humidity (dried supply air). 5 inputs x 2
  days = 10 points. For the solid fraction and the humidity the two days give the same point; it
  is counted for both.
* **B (distance to admissibility):** starting from S, the spray flow, the solid fraction and the
  dryer inlet temperature (each alone) at which the saturation ratio of the unit is 1, found by
  bisection with the rebuild (up to 3 points). The solid fraction was added by the task
  (section 2, item 2).

## 6. Pre-registered reading

Fixed in the docstring of `scripts/report_sheet_plant_a.py` before its first run.

**Step 0, controls (before any reading).**

* (a) The point S reproduces Befund 130: its saturation ratio equals the one stored in
  `dyssol_sweep_case1_comparison.json` to a relative 1e-12.
* (b) The segments are those of Befund 117: start and stop minute of all eight, the digests of
  both exports and of the role file equal those of `plant_scale_v1.json`, and the day aggregates
  of the six quantities used here equal the stored ones to a relative 1e-9.
* (c) Dyssol against the rebuild at every point of S, M, R and B: 1e-6 K on temperatures, 1e-6 %
  on heat flows and fuel relative to max(|value|, 1), as `docs/45`; no Dyssol run failed.

If a control fails, the readings are not read and the deviation is reported.

**Reading F, feasibility first** (reference variant, 8 segments): the number of segments at which
the dryer exhaust of the unit is over-saturated.

* **F0** -- none: the unit is admissible at the measured operation of plant A.
* **F2** -- all eight.
* **F1** -- otherwise (reported per day).

**Reading T, the unit against the measurement** (reference variant; only segments that are not
over-saturated): dryer outlet temperature of the unit minus the measured exhaust temperature at
the granulator outlet.

* **T0** -- within 2 K at every segment (the threshold of reading P1 of Befund 117).
* **T1** -- otherwise; reported with the sign and the range per day.

Without such a segment reading T is "not applicable".

**Reading A, attribution** (points R): a replacement "removes" the over-saturation if the
saturation ratio of the unit is at or below 1.

* **A0** -- exactly one input removes it on both days: that input is named.
* **A1** -- more than one input removes it on both days, or none does on both days but at least
  one does on one day.
* **A2** -- no single replacement removes it on either day.

No rank claim, no test: there is no seed and no model to rank.

## 7. Predictions (fixed with this design)

From the committed day aggregates of `docs/34` and arithmetic by hand (section 10; the numbers
are in the local draft): **F0; T1 with the unit warmer than the measured exhaust at every read
segment; A0 with the solid fraction.** Each is reported as "confirmed" or "not confirmed".
Expected in addition, not part of the check: the spray flow removes the over-saturation on day 1
only.

## 8. Reported without a reading

* Reading F and T in every variant of M (solid-fraction band, both humidities); the number of
  segments over-saturated with the measured inputs and the SHEET'S solid fraction.
* The unit's outlet temperature against the measured bed temperature (mean of six sensors).
* The water to evaporate per kg of dry gas: the sheet against each day (local only).
* Per segment the smallest solid fraction at which the unit is admissible (rebuild only; local
  values; committed: the number of segments at which it lies above the sheet's value).
* B: the relative change of the spray flow and the change of the dryer inlet temperature in K
  that bring process 1 to a saturation ratio of 1; the solid fraction of B stays local (committed:
  whether such a solid fraction exists below 0.99).

## 9. Stages and compute

Code and this document committed before the run -> dry run (every point through the rebuild;
Dyssol at S and at one measured point) -> full suite -> run on the clean tree -> artifact
committed after `tests/test_denylist.py`. About 94 Dyssol runs of about 1.3 s each with the units
built for Befund 128 (`C:/twinlink_runs/dyssol-steady-state/units`, checked against their
manifest), patched materials database as there, source flowsheet of process 1 by the rule of
Befund 130. Minutes of wall clock; no detached run needed. **The dry run shows the readings of
the rebuild** (a closed formula has no cheaper form); the reading is committed before it. A
branch is delivered; no merge, no push.

## 10. Seen before the run (disclosure)

* The sheet in full, Befund 130 and 131, `docs/49`.
* The day aggregates of plant A in `docs/34`, section 4.1 (process air, spray rate, supply-air,
  intake, bed and exhaust temperature), the balances of section 4.2 and the errors of the
  grey-box candidates of section 4.3; the same aggregates and the segment windows in the artifact
  `plant_scale_v1.json`.
* The local note on the minimum admissible solid fraction of the two days by the dryer balance of
  `docs/23`. Both minima lie above the solid fraction of the sheet's process 1.
* Arithmetic by hand in the session that drafted this (2026-11-25): water to evaporate per kg of
  dry gas for the sheet and for both days, an estimate of the adiabatic capacity of the sheet's
  inlet gas, the single replacements of reading A, and an estimate of the unit's outlet
  temperature on both days. The predictions of section 7 come from it.
* No run of the rebuild or of Dyssol at a measured point; no segment value was opened. The tests
  of this branch use invented segment means and an invented laboratory-scale line.

## 11. Decisions of the project lead (2026-11-25): the ten proposals of the draft, all accepted

1. **Threshold of reading T: 2 K**, taken from Befund 117.
2. **Reference humidity: dried supply air** (placeholder of Befund 117, "very low, unconfirmed"),
   not the humidity of the sheet. The sheet's humidity is a variant.
3. **Reference solid fraction 0.65**, band as Befund 117. Neither the reference nor the sheet's
   value is measured; 0.634 was fitted on both days.
4. **Dyssol control at every point**, not the rebuild alone.
5. **What is committed:** saturation ratios, counts, temperature differences in K, relative
   changes, flags, digests. **Not committed:** any input of the sheet, any ratio "sheet over day"
   of an input (with the day aggregates of `docs/34` it would give the sheet's value), the
   product-specific solid fractions. Implemented as a rule: whatever depends on an input of the
   sheet is reduced to counts, flags and reading codes, except the saturation ratio of S (already
   in the artifact of Befund 130) and the two relative distances of B.
6. **The distance to admissibility (B) is part of this study**, descriptive, instead of a study of
   its own.
7. **Reading A uses day aggregates**, not segments.
8. **`DataSpec.new_study` does not apply** (no grey-box data are generated; as `docs/45`).
9. **The sentence of section 10 on the minimum solid fractions** may stay: it states an
   inequality and no value.
10. **Reading T is kept as a reading** although its outcome is expected (section 3).

## 12. Caveats known beforehand

* **The assignment "process 1 = plant A" rests on the label of the sheet alone** (section 2,
  item 1). The sheet does not say which product or which operating point process 1 stands for.
  It may be a nominal point or a third product; the comparison then says that it is not one of
  the two measured ones. Question to the sender, added to the local list
  (`local/reports/carrier_sweep_rueckfragen.md`, question 12; nothing has been sent).
* Two days, two products; a difference between the days is not separable from the product.
* Solid fraction and gas humidity are not measured on either side.
* The unit has no wall loss, no atomising air and one outlet temperature.
* "Dry gas" is inferred (Befund 130).

## 13. Code and artifacts

`src/twinlink_ml/dyssol_steady/plant_check.py` (segment -> inputs of the unit, replacements, the
roots of B, the three readings, the two records), `scripts/report_sheet_plant_a.py` (reading in
its docstring), `tests/test_sheet_plant_a.py` (invented day and invented line only),
`artifacts/runs/sheet_plant_a_v1.json`; local record
`<root>/local/runs/sheet_plant_a/sheet_plant_a_v1_local.json`, copied to
`C:/twinlink_runs/sheet-plant-a/`; German report with the recommendation for the sender in
`local/reports/` of the main checkout (not versioned).
