# 52 -- The six heat-recovery cases at the plant inputs of the partner deliveries: is case 1 still the best case, and does the rule of Befund 121 hold outside its box?

Status: **complete. All four controls held; pre-registered readings F0 (all 54 cells
admissible), C1 (case 1 is the best case at none of nine admissible points, case 2 at all of them,
in the delivered units and in all three modes of the workbook rebuild) and A0 (exchanging the
sizes of the two lines turns the order); all predictions confirmed, not blind (sections 15 to
17). C1 holds at the circuit flows the delivery stores: with both circuits at the same small flow
(the six smaller of the ten flows of the sweep sheet) case 1 is best or practically equal in the
physical modes (section 15.6).** Accepted by the project lead on 2026-11-25 (section 20). Sections
1 to 14 are the design, committed before any
run of the report script (d256082) and unchanged since except this paragraph; the results below
them take their numbers only from the artifact named there. Branch `plant-cases` (from `master`
8ae68d4), worktree `C:/twinlink_work/plantcases`. Document number 52 and finding number 133 (lab book); merged into
`master` on 2026-11-25. Code `src/twinlink_ml/dyssol_steady/network.py`, `plant_cases.py`,
an extension of `flowsheet.py`; script `scripts/report_plant_cases.py` (the reading is in its
docstring); tests `tests/test_plant_cases.py`.

**Everything here stands under two readings that the sender has not confirmed:** "the gas mass
flow of the sweep sheet is the dry gas" (Befund 130) and "the line labelled as plant A in both
deliveries is the line called plant A in this repository" (Befund 132; from the labels alone).

## 1. Occasion and questions

Befund 110 found case 1 (each exhaust preheats the fresh gas of its own line) the best of the six
cases at the workbook point and noted that this holds for the inputs of the colleague, "not
plant-near". Befund 121 varied eight inputs around that point with the dry-air flows and the
circuit flows held at the workbook's values: no case is best everywhere, the choice is between
cases 1 and 2, and it follows the rule "case 1 if exhaust 1 is at least as warm as exhaust 2".

Two partner deliveries hold plant inputs:

* **2026-11-19:** seven Dyssol flowsheets with stored results and a comparison workbook. One of
  its sheets (the "plant sheet" of `docs/45`, section 9) lists the six cases at plant inputs, with
  Excel columns for all six and Dyssol columns for four. The repository used the delivery as a
  source of flowsheets (Befund 128); **the comparison of the cases in it was never evaluated.**
* **2026-11-24:** the sweep sheet (Befund 130, 131, 132), which varies the medium flow for case 1
  only, at another state of the fresh gas.

In both, the dryer of the line labelled as plant A is over-saturated as delivered (Befund 130).
Befund 132 showed that this line is admissible with the reference solid fraction and at the
measured operation. That makes four questions answerable:

1. **Is the delivery reproduced?** Stored results of the seven flowsheets and the plant sheet
   against the rebuild, including the four cascades (stages in series), which the rebuild and the
   adapter did not cover.
2. **At admissible plant-near points, is case 1 the best of the six cases?**
3. **Does the rule of Befund 121, as worded, hold there?**
4. **If not: is it the size of the two lines that decides?** In the box of Befund 121 line 1 is
   always the larger one; at the plant inputs it is the smaller one.

## 2. Task of the project lead (2026-11-25)

"Test C" of the session's list of 2026-11-25. The draft of this document (written in
`local/reports/` before any code) was released with all thirteen proposals of section 12
accepted. Then as for Befund 132: branch, worktree, code and reading committed before the run, dry
run, full suite, run on the clean tree, artifact, lab-book entry "Befund NN (number open)". No
merge, no push.

**Addition of the project lead (2026-11-25, after the release, before any run):** a generalised
rule is evaluated descriptively, without a reading, and marked as a post-hoc hypothesis (section
9a). The four questions for the sender go onto the local list.

## 3. Placing

* **Model against model.** Line 2 gets measured inputs at the points M, but no fuel and no
  recovered heat of a plant is measured anywhere; nothing here is validated against a plant.
* **Not a re-run of Befund 110 or 121;** none of their numbers changes. A verdict C1 (section 7)
  would add a scope note to both ("holds for the size ratio of the workbook"); that is a decision
  of the project lead at the merge.
* **Deterministic.** No seed, no model fit; `paired_resolution` does not apply (as Befund 121).
* `docs/49` (cost over the medium flow, case 1, one line) is not touched.

## 4. Lines, cases and numbering

**Line 1** is the line numbered 1 in the delivered case flowsheets (in the workbook the spray
tower; the line with the hotter dryer inlet). **Line 2** is the line that both deliveries label as
plant A (fluidised bed). Exhaust *i*, sink *i* (the fresh gas and heater of line *i*) and circuit
*i* (the medium circuit that cools exhaust *i*) follow. **The sweep sheet of 2026-11-24 numbers its
two processes the other way round:** its process 1 is line 2 here. The processes of a sheet are
assigned to the lines by the rule of Befund 130 (nearest stored dryer inlet temperature,
one-to-one).

The cases by their wiring (docs/11, section 13; read from the ports of the delivered flowsheets,
`dyssol_steady.network.CASES`):

| Case | Sink 1 is preheated by | Sink 2 is preheated by |
|---|---|---|
| 1 | exhaust 1 | exhaust 2 |
| 2 | exhaust 2 | exhaust 1 |
| 3 | exhaust 2, then exhaust 1 | nothing |
| 4 | nothing | exhaust 2, then exhaust 1 |
| 5 | exhaust 1, then exhaust 2 | nothing |
| 6 | nothing | exhaust 1, then exhaust 2 |

Case 0 is no recovery. "Fuel" is the total of both heaters; "saving" is case 0 minus the case.

## 5. Models

* **U, the delivered units** (`dyssol_steady.reference`; dryer without a saturation limit, heat
  recovery by equations 4.1 / 4.2 without a kink, heater with Dyssol's tabulated enthalpy). The
  rebuild covered cases 1 and 2; `dyssol_steady.network.evaluate_network` adds stages in series
  with the same unit functions. The existing `evaluate` is unchanged. Dyssol control with the
  seven delivered flowsheets; the adapter reads the family "two recoveries in series on one
  fresh-gas stream" with `discover_network` (the existing `discover_topology` stays the default).
* **W, the workbook rebuild** (`flowsheet`, all six cases, as Befund 110 and 121) in its three
  modes: partner, physical unpressurised (medium at most 95 degC), physical pressurised. W is the
  model in which the rule of Befund 121 was found; without it a failed rule could not be separated
  from the change of the model. Properties of the workbook (docs/11, section 13.1); equilibrium
  moisture, heater efficiency and calorific value of the lines; exchanger placeholders of docs/11,
  section 13.6. The workbook has one ambient temperature: both lines share the fresh-gas
  temperature at every point.

## 6. Evaluated points

Circuit flows at every point: **those stored in the delivered case flowsheets, circuit *i* with
exhaust *i*** (the analogue of the workbook's flows in Befund 121).

* **P0:** the plant sheet of 2026-11-19 as delivered (reproduction only). U takes the inputs the
  flowsheets store (they equal the sheet's block for Dyssol); W takes the sheet's inlet block.
* **S:** both lines as the sweep sheet of 2026-11-24 states them.
* **R:** S with the solid fraction of line 2 replaced by 0.65 (reference of Befund 117 and 132).
* **M1 to M8:** line 2 at the measured inputs of the eight stationary segments of Befund 117
  (mapping, reference solid fraction 0.65 and dried supply air exactly as Befund 132); line 1 as in
  S; the fresh-gas temperature of both lines is the measured intake temperature of the segment.
  **Test set: plant A, two measured days (day 1 = product 1, day 2 = product 2), four segments
  each.**
* **X:** R with the sizes of the two lines exchanged: dry gas and spray of a line are scaled by
  one factor per line so that the two dry-gas flows are exchanged. The specific operating points,
  and with them both dryer exhausts, stay as in R; the circuit flows stay with their exhaust.

**Read points: R and M1 to M8 (nine).** P0 and S are expected to be inadmissible and are reported
without a reading.

In addition, without a reading: S, R, M and X with both circuits at each non-zero flow of the
sweep sheet's grid; R and M with the solid fractions of the band (0.60, 0.634, 0.70); X at the M
points; X with the circuit flows scaled with their exhaust. In all 158 combinations of point and
flow setting.

## 7. Pre-registered reading

Fixed in the docstring of `scripts/report_plant_cases.py` before its first run.

**Step 0, controls.**

* (a) For cases 1 and 2 the extended rebuild equals `reference.evaluate` at every point (relative
  1e-12).
* (b) The segments are those of Befund 132: labels, digests of both exports and of the role file
  equal those of `sheet_plant_a_v1.json`, and the saturation ratio of line 2 at M1 to M8 equals the
  stored one (relative 1e-9).
* (c) Dyssol against the rebuild at every point, case and flow setting: 1e-6 K on temperatures,
  1e-6 % on heat flows and fuel relative to max(|value|, 1), as `docs/45`; no Dyssol run failed.
* (d) W in the partner mode at P0 reproduces every Excel total of the plant sheet within rounding
  (the total is a sum of two rounded values and gets the sum of the two halves, as docs/11,
  section 13.3), with the circuit flows the sheet states or with those the flowsheets store.

If (a), (b) or (c) fails, U is not read. If (d) fails, W is not read; U is read all the same.

**Reading F, feasibility first** (per model). A cell (read point, case) is admissible if both
dryer exhausts are admissible and, in U, no exhaust is over-saturated after its recovery stage.
U: saturation ratio at or below 1 (Befund 130), also at the outlet of the recovery (the unit
cannot condense; `docs/49`, decision 5). W: both exhausts at or below 90 % relative humidity (rule
of Befund 121). A stage with negative recovery is counted, not excluded (as Befund 121).

* **F0** -- all 54 cells of the nine read points are admissible.
* **F1** -- otherwise; reported as admissible cells per case.

**Reading C, case 1 against the best case** (per model; at each read point over the cases
admissible there). Practical tie: a gap below 1 % of the saving of the pointwise best case (Befund
121, decision F4; 0.5 % and 2 % are reported). A point at which case 1 is inadmissible counts as
"not best".

* **C0** -- case 1 is best or practically equal at all nine points: the statement of the workbook
  carries over to the plant inputs.
* **C1** -- case 1 is neither best nor practically equal at any of the nine points.
* **C2** -- otherwise; reported for R and per day.

The worded rule of Befund 121 is evaluated at the same points by its scale-free part only: the
first split on exhaust 1 minus exhaust 2 (-0.54 K in the physical modes, -15.5 K in the partner
mode and, having the same exchanger, in U; `docs/38`, section 6.4). The further leaves are in kW
at laboratory scale and do not apply. Where the rule predicts case 1, C1 means that it fails
there. No separate code.

C is read in U and in each mode of W separately. **Headline: U** (the units that computed both
sheets, with a Dyssol control). The physical unpressurised mode of W is named next to it, as the
main mode for a recommendation in Befund 121 (decision F2).

**Reading A, attribution** (point X; not applicable if case 1 is best or practically equal at R
in U or in the physical unpressurised mode of W):

* **A0** -- at X case 1 is best or practically equal, in U and in the physical unpressurised mode
  of W: the size ratio of the lines alone turns the order.
* **A1** -- otherwise.

No rank claim over seeds, no test: the computation is deterministic.

## 8. Predictions (fixed with this design)

From arithmetic by hand with the formula of the heat-recovery unit and with a counter-current
balance, on the inputs of the sweep sheet and the day aggregates of `docs/34` (section 11; the
numbers are in the local draft), and from the totals the delivery itself stores:

**All controls hold; F0; C1 in U and in all three modes of W; A0.** Each is reported as
"confirmed" or "not confirmed". Expected in addition, not part of the check: case 2 is the best
case at all nine points, case 4 follows in U; the regret of case 1 is of the order of a tenth of
the best saving in U and larger in the physical modes; exhaust 1 is the warmer one at all nine
points, so the worded rule predicts case 1 everywhere.

**The predictions are not blind.** The order of cases 1 and 2 in U is in substance already in the
delivery of 2026-11-19 (at the inadmissible point). What the run adds: the admissible points, the
second model family, the control of the cascades, and the attribution.

## 9. Reported without a reading

* P0 and S: saturation ratios, the ranking of the six cases with the note that it stands on an
  inadmissible dryer (numbers of the unit's formula, as `docs/49`, section 5).
* P0: the Dyssol columns and the Excel columns of the plant sheet against U and W, counted within
  rounding; the stored results of the seven flowsheets against the rebuild at the stored inputs
  (`docs/45`, section 8: four of them were saved with another state of the units).
* Ranking of all six cases, saving of each case in % of case 0, regret of case 1 in % of the best
  saving and of the best fuel, per read point and model.
* Stages with negative recovery per case (the heat-recovery formula has no kink; question 11 of
  the list to the sender).
* The read points with both circuits at each flow of the sweep sheet's grid: does the best case
  change with the flow?
* The band of the solid fraction of line 2 (0.60, 0.634, 0.70) at R and M.
* X also at the M points, and X with the circuit flows scaled with their exhaust.

## 9a. Generalised rule (addition of the project lead, 2026-11-25): descriptive, no reading, a post-hoc hypothesis

**"The warmer exhaust preheats the sink with the larger heat-capacity rate"** (case 1 or case 2,
depending on the assignment). The hypothesis was formulated after the stored totals of the
delivery and the hand estimates of section 11 had been seen; it is not pre-registered and gets no
code of a reading.

* **Here:** at the nine read points and at X, in U and in the three modes of W, the hit rate for
  the choice between cases 1 and 2 (the better of the two by total fuel). Capacity rate of a sink:
  dry gas times its heat capacity plus the moisture of the fresh gas times that of vapour, with
  the properties of the model.
* **At the points of Befund 121:** from the stored arrays of `case_ranking_v1.npz` (points,
  exhaust temperatures and fuel of every case in the three modes; variant with the workbook's
  circuit flows; admissible points of the box and of the rule's own test stream). **Nothing is
  recomputed.** The capacity rates follow from the stored points and the workbook's constants; in
  that box sink 1 is the larger one at every point, so the generalised rule reduces to "case 1 if
  exhaust 1 is the warmer one". It is compared with the worded rule of Befund 121 (first split,
  thresholds as in section 7) on the same target, the choice between cases 1 and 2; the test
  accuracy of the fitted tree for its own target ("case 1 strictly best" among six) is quoted next
  to it and is not the same quantity.
* **Use:** a proposal for a scope note or a more precise wording of the rule of Befund 121. The
  decision is the project lead's at the merge.

## 10. Stages and compute

Code and this document committed before the run -> dry run (every point, case and setting through
the rebuild and W; Dyssol with every flowsheet at R) -> full suite -> run on the clean tree ->
artifact committed after `tests/test_denylist.py`. 1106 Dyssol runs (158 combinations times seven
flowsheets) of about 1.3 s each with the units built for Befund 128, four at a time; W and the
rebuild are closed formulas. Under an hour of wall clock; no detached run needed. **The dry run
shows the readings of the rebuild and of W**; the reading is committed before it. A branch is
delivered; no merge, no push.

## 11. Seen before the run (disclosure)

* Both deliveries in full: the plant sheet of 2026-11-19 with its Excel and Dyssol totals of the
  six cases, the sweep sheet of 2026-11-24.
* The stored inputs and results of all seven flowsheets, read with the existing reader in the
  session that drafted this (2026-11-25), and their wiring (while the draft was written and again
  with `discover_network` while the code was written).
* Befund 110, 121, 130, 132; the day aggregates of plant A in `docs/34`.
* Arithmetic by hand in that session: the recovered heat of the six cases with the unit's formula
  at R, at both day aggregates and at X; a counter-current balance with 10 K per exchanger for
  cases 1, 2 and 4 at R and day 1. The predictions of section 8 and the hypothesis of section 9a
  come from it and from the stored totals.
* The structure of the stored arrays of Befund 121 (names and shapes) and the text of its fitted
  tree; no hit rate was computed before this commit.
* No run of the rebuild, of W or of Dyssol at a point of this study; no segment value was opened.
  The tests of this branch use two invented laboratory-scale lines, invented segment means and
  invented rows of a plant sheet.

## 12. Decisions of the project lead (2026-11-25): the thirteen proposals of the draft, all accepted

1. **Numbering by the case flowsheets** (section 4), cases by their wiring; the reversed numbering
   of the sweep sheet is stated wherever a sheet is cited.
2. **Two model families, headline U;** W in three modes beside it, the physical unpressurised mode
   named as in Befund 121.
3. **Circuit flows: those stored in the delivered flowsheets,** not optimised (with ideal
   exchangers the recovery only grows with the flow, `docs/49`); the grid of the sweep sheet in
   both circuits is descriptive.
4. **Admissible variants of line 2 as Befund 132:** solid fraction 0.65 (R), the eight segments
   with 0.65 and dried supply air (M). Neither is a measured solid fraction.
5. **Fresh gas at M:** the measured intake temperature of the segment for both lines (one outside
   air); line 1 keeps the humidity of the sweep sheet, line 2 the dried supply air.
6. **Practical tie 1 %** of the best saving (Befund 121, F4).
7. **Feasibility of a cell includes the exhaust after its recovery;** negative recovery is counted,
   not excluded.
8. **Rebuild and adapter extended to stages in series, Dyssol control at every point, case and
   setting.**
9. **W:** properties of the workbook, the three values of the lines named in section 5, exchanger
   placeholders of docs/11, section 13.6.
10. **Reading A by exchanging the sizes of the lines** (gas and spray of a line scaled together).
11. **What is committed:** ranks, savings and regrets in %, saturation ratios, counts, flags,
    reading codes, differences Dyssol minus rebuild, digests. **Not committed:** any input or
    result of a partner sheet or flowsheet, the stored circuit flows, fuel and heat in absolute
    units, temperatures, measured values, the labels of the lines.
12. **`DataSpec.new_study` does not apply** (no grey-box data are generated; as `docs/45`).
13. **Questions for the sender** go onto the local list (`local/reports/`, not versioned); nothing
    is sent from the branch.

Taken while writing the code, to be confirmed with the result: P0 uses the stored inputs of the
flowsheets for U and the inlet block of the sheet for W (section 6); control (b) compares labels,
digests and saturation ratios with the artifact of Befund 132 instead of repeating its window
check against `plant_scale_v1.json`; reading A is "not applicable" as soon as case 1 is best or
practically equal at R in one of its two models.

## 13. Caveats known beforehand

* The predictions are not blind (section 8); the generalised rule is a post-hoc hypothesis
  (section 9a).
* Line 1 has no measurement; it stays the nominal point of the sheet at every read point. The M
  points mix a measured line 2 with a nominal line 1.
* The solid fraction of line 2 is a placeholder (Befund 132, caveat 1); the gas humidity likewise.
* Two days, two products.
* Ideal exchangers in U (docs/12, item 41): the sizes of the recovered heats are those of the
  formula; with a finite area the gaps between the cases will differ.
* The circuit flows are those of the delivery, not a design.
* "Dry gas" and the assignment of line 2 to plant A are unconfirmed.

## 14. Code and artifacts

`src/twinlink_ml/dyssol_steady/network.py` (stages in series, the six cases as wirings),
`src/twinlink_ml/dyssol_steady/plant_cases.py` (points, the two model families, the readings, the
rules, the reader of the plant sheet, the two records), `src/twinlink_ml/dyssol_steady/flowsheet.py`
(`NetworkTopology`, `discover_network`, the argument `discover` of `DyssolFlowsheet`; the existing
family and its default unchanged, tested), `scripts/report_plant_cases.py` (reading in its
docstring), `tests/test_plant_cases.py` (invented lines, segments and sheet rows only),
`artifacts/runs/plant_cases_v1.json`; local record
`<root>/local/runs/plant_cases/plant_cases_v1_local.json`, copied to `C:/twinlink_runs/plant-cases/`;
German report in `local/reports/` of the main checkout (not versioned).

## 15. Results

All numbers from `artifacts/runs/plant_cases_v1.json` (written at d256082, `git/dirty` False,
`code/root_checked` True). Fuel and heat in absolute units, temperatures and flows are in the
local record only.

### 15.1 Step 0: all four controls held

* **(a)** At all 158 combinations of point and flow setting the extended rebuild equals
  `reference.evaluate` for cases 1 and 2 (largest relative difference 0.0).
* **(b)** The segments are those of Befund 132: same labels, same digests of both exports and of
  the role file, saturation ratio of line 2 identical at all eight segments (deviation 0.0).
* **(c)** 1106 Dyssol runs, none failed; 1896 line results compared; Dyssol against the rebuild at
  most 2.3e-13 K and 5.6e-11 % (bounds 1e-6). This is the first Dyssol control of the four
  cascades.
* **(d)** With the circuit flows the flowsheets store, W in the partner mode reproduces all 6
  Excel totals of the plant sheet within rounding (largest difference 0.90 units of the last
  decimal), and all 12 heater values and all 12 recovered heats within half a unit. With the
  circuit flows the sheet states it reproduces none of the totals (up to 193 units). The Excel
  columns were computed with the stored flows.

U and W are read.

### 15.2 Reading F: F0 in every model

All 54 cells of the nine read points are admissible in U and in each of the three modes of W (9
points for each of the six cases). No exhaust is over-saturated after its recovery stage in U.

### 15.3 Reading C: C1 in every model

Case 1 is best or practically equal at **0 of 9** read points in U and in each mode of W; the
verdict is the same with a tie of 0.5 % and of 2 %. **Case 2 is the best case at all nine points
in all four models.** The order of the six cases is the same at all nine points: 2, 4, 1, 6, 3, 5
in U and in the partner mode of W; 2, 4, 6, 1, 3, 5 in both physical modes.

Regret of case 1 (its extra fuel over case 2), per model:

| Model | % of the best saving: R | day 1 | day 2 | % of the best fuel: R | day 1 | day 2 |
|---|---|---|---|---|---|---|
| U (headline) | 9.51 | 8.10 to 8.28 | 9.62 to 10.05 | 2.44 | 2.39 to 2.40 | 2.45 to 2.49 |
| W, partner | 9.56 | 8.06 to 8.26 | 9.56 to 10.05 | 2.45 | 2.37 to 2.39 | 2.43 to 2.48 |
| W, physical unpressurised | 23.62 | 23.56 to 25.37 | 23.91 to 24.75 | 11.46 | 14.19 to 16.71 | 11.28 to 11.72 |
| W, physical pressurised | 23.62 | 23.56 to 25.37 | 23.91 to 24.75 | 11.46 | 14.19 to 16.71 | 11.28 to 11.72 |

Saving against case 0 in U: case 2 20.42 % (R), 22.49 to 22.74 % (day 1), 19.83 to 20.32 % (day 2);
case 1 18.48 %, 20.62 to 20.90 %, 17.83 to 18.36 %. In the physical unpressurised mode: case 2
32.67 %, 37.58 to 39.72 %, 31.30 to 32.90 %. The two physical modes give the same numbers: the
medium limit does not bind at these inputs.

**The worded rule of Befund 121 predicts case 1 at all nine points, in every model** (exhaust 1 is
the warmer one everywhere), and case 1 is best or practically equal at none of them. At the plant
inputs the rule fails at every read point.

### 15.4 Reading A: A0

At R case 1 is not best in U nor in the physical unpressurised mode of W; at X it is the best case
in both (order 1, 3, 2, 5, 4, 6 in U and 1, 3, 5, 2, 4, 6 in the physical unpressurised mode).
**Exchanging the sizes of the two lines alone turns the order.** The same holds at X with the
circuit flows scaled, and at all eight segments with exchanged sizes, in all four models.

### 15.5 Predictions: all confirmed

F0 in all four models, C1 in all four models, A0. Expected in addition and found: case 2 best at
all nine points, case 4 second in U; the regret of case 1 about a tenth of the best saving in U and
larger in the physical modes; exhaust 1 the warmer one everywhere. **The predictions were not
blind** (section 8), and the dry run showed the readings of the rebuild and of W before the main
run (section 10).

### 15.6 Without a reading

* **As delivered, no comparison of the cases is admissible.** Line 2 is over-saturated at S
  (saturation ratio 2.015, as Befund 130) and at P0 (2.309). The order of the unit's formula is
  2, 4, 1, 3, 6, 5 at both; at P0 the savings against case 0 are 15.29 % (case 1), 17.52 % (case 2),
  15.24 % (case 3) and 17.50 % (case 4). In the physical modes of W the saturation limit holds the
  exhaust of line 2 at saturation at S, which the rule of Befund 121 counts as inadmissible.
* **The delivery of 2026-11-19 against the models.** The Dyssol columns of the plant sheet agree
  with U at the stored inputs within rounding in 5 of 8 heater values, 3 of 4 totals and 0 of 8
  recovered heats. The stored results of three of the seven flowsheets (cases 2, 4, 6) are
  reproduced exactly; the other four deviate by 0.010 to 0.030 K (in the small recovered heat of
  exhaust 2 by up to 1.75 %). This is the picture of `docs/45`, section 8 (four flowsheets saved
  with another state of the units); the columns of the sheet were filled from yet another run.
* **The two sheets number the processes the other way round,** and the plant sheet states a
  circuit flow for line 2 that differs from the one its flowsheets store and its own Excel columns
  use (control (d)). Both go to the sender as questions.
* **Negative recovery.** In U and in the partner mode of W the second stage of cases 5 and 6
  cools the fresh gas at R and on day 2 (case 6 not at the last segment of day 2); not on day 1,
  and never in the physical modes.
* **Both circuits at one flow of the sweep sheet's grid.** In U and in the partner mode of W case
  2 stays the best case at every one of the ten flows and every read point. **In the physical
  modes the best case depends on the flow:** at the six smaller flows case 1 is best or
  practically equal at every read point (at one segment of day 1 also at the largest flow), at the
  larger flows case 2. With a small medium flow the circuit limits the recovery and the cases tie.
  C1 in the physical modes is therefore a statement at the stored circuit flows.
* **Band of the solid fraction of line 2.** 0.634 and 0.70: F0 and C1 in every model, best case 2.
  0.60: C1 in every model; in U the reading F gives F1 (cells inadmissible after the recovery
  stage) and the best admissible case is 2 or 6; in W F0 and best case 2.

### 15.7 Generalised rule (section 9a; descriptive, post hoc)

**Here.** "The warmer exhaust preheats the sink with the larger heat-capacity rate" names the
better of cases 1 and 2 at 10 of 10 points (the nine read points and X) in each of the four
models. At the nine read points it says case 2 (exhaust 1 warmer, sink 2 larger), at X case 1.

**At the points of Befund 121** (stored arrays, nothing recomputed; variant with the workbook's
circuit flows; sink 1 is the larger one at every point, so the generalised rule is "case 1 if
exhaust 1 is the warmer one"). Hit rate for the choice between cases 1 and 2 at the points where
the two fuels differ:

| Mode | Set | Points | Generalised rule | Worded rule of Befund 121 | Where they differ: generalised / worded right |
|---|---|---|---|---|---|
| physical unpressurised | box | 14 841 | 0.9725 | 0.9735 | 13 / 28 |
| physical unpressurised | test stream | 3 716 | 0.9701 | 0.9701 | 5 / 5 |
| physical pressurised | box | 16 756 | 0.9736 | 0.9743 | 21 / 32 |
| physical pressurised | test stream | 4 178 | 0.9706 | 0.9703 | 9 / 8 |
| partner | box | 16 803 | 0.8398 | 0.9311 | 54 / 1589 |
| partner | test stream | 4 185 | 0.8315 | 0.9259 | 18 / 413 |

(16 803 admissible points of the box, 4 185 of the test stream; outside the practical tie the
rates are 0.992 to 0.994 in the physical modes for both rules, and 0.982 against 0.9997 in the
partner mode on the box. The fitted tree of Befund 121 has a test accuracy of 0.966 to 0.978 for
its own target, "case 1 strictly best" among six cases, which is another quantity.)

**Answer to the question of the addition.** In the physical modes the generalised rule is as good
as the rule of Befund 121 to within 0.001 (equal or better on the test stream, 0.0007 to 0.0010
lower on the box): there the two are the same rule up to the threshold of -0.54 K. **In the
partner mode it is not:** the partner model keeps case 1 until exhaust 1 is about 15 K colder
than exhaust 2, and the generalised rule has no such shift (0.84 against 0.93).

## 16. Answers and caveats

**Question 1.** Yes for what is controlled: Dyssol and the rebuild agree at all 1106 runs,
cascades included, and the Excel columns of the plant sheet are reproduced by the partner mode of
W. The stored results and the Dyssol columns of the sheet differ from a re-run by 0.01 to 0.03 K,
as known from `docs/45`.

**Question 2.** No. At all nine admissible points case 2 is the best case and case 1 the third or
fourth, in the delivered units and in all three modes of the workbook rebuild. Case 1 needs 2.4
to 2.5 % more fuel than case 2 in U and 11 to 17 % more in the physical modes.

**Question 3.** No. The rule predicts case 1 at every one of the nine points.

**Question 4.** Yes. With the sizes of the two lines exchanged and nothing else changed, case 1 is
the best case.

**Caveats.**

1. **The predictions were not blind,** and the order of cases 1 and 2 in U was already in the
   delivery (at the inadmissible point). The result confirms an estimate; it was not a risky test.
2. **Line 1 is a nominal point without a measurement** at every read point; the M points combine a
   measured line 2 with it. Nothing here is validated against a plant.
3. **The solid fraction of line 2 is a placeholder.** With 0.60 the exhaust of line 2 is so near
   saturation that the unit over-saturates it after the recovery at one point.
4. **Circuit flows.** C1 holds at the flows the delivery stores. In the physical modes case 1 is
   practically equal when both circuits run at one of the six smaller flows of the sweep sheet
   (section 15.6). The flows are those of the delivery, not a design.
5. **Ideal exchangers in U** (docs/12, item 41): the sizes of the gaps are those of the formula.
6. **The generalised rule is a post-hoc hypothesis.** Its ten points here lie far from any
   boundary (exhaust 1 much warmer, one sink clearly larger), and only two size ratios exist: the
   workbook's and the plant's. Where a slightly warmer exhaust meets a much larger sink the rule is
   untested. It does not describe the partner mode at the points of Befund 121.
7. Two days, two products; "dry gas" and the assignment of line 2 to plant A are unconfirmed.
8. Deterministic computation: no confidence statement; ranks are orders of totals.

**Relation to existing findings (reported, nothing changed).** No number of Befund 110, 121, 128,
130, 131 or 132 changes. Befund 110 ("case 1 is best") and the rule of Befund 121 do not carry
over to the plant inputs; both were obtained with line 1 as the larger line.

## 17. Open for the project lead

1. Acceptance of F0, C1 (in U and in the three modes of W) and A0, and of "all predictions
   confirmed".
2. **Scope note to Befund 110** at the merge. Proposal: "Case 1 is the best case for the size
   ratio of the workbook (line 1 the larger line). At the plant inputs of the partner deliveries,
   where line 2 is the larger one, case 2 is the best case at all nine admissible points, in the
   delivered units and in all three modes of the rebuild (finding plant-cases)."
3. **Scope note or more precise rule for Befund 121** at the merge. Proposal: "The rule 'case 1 if
   exhaust 1 is at least as warm as exhaust 2' was found with the dry-air flows of the workbook,
   in which sink 1 is the larger one at every point. At the plant inputs (sink 2 the larger one)
   it predicts case 1 at all nine admissible points, and case 2 is best there. In the physical
   modes the wording 'the warmer exhaust preheats the sink with the larger heat-capacity rate'
   makes the same choice between cases 1 and 2 at the points of Befund 121 (hit rate 0.9725 and
   0.9736 against 0.9735 and 0.9743 on the box) and holds at the plant inputs (10 of 10; post
   hoc, not pre-registered). In the partner mode it does not replace the rule (0.840 against
   0.931) (finding plant-cases)."
4. The three decisions taken while writing the code (section 12, last paragraph).
5. Whether the four new questions (numbering of the processes, circuit flow of line 2, why the
   sweep varies case 1, cases 5 and 6) go out with the list; nothing was sent.
6. Document and finding number, the row of `docs/12`, the README sentence: the managing session
   at the merge.

## 18. Course of the run (2026-11-25)

Design, reading, code and tests committed at d256082 before any run of the script. Dry run on
that commit: all 158 combinations through the rebuild and W, Dyssol with every flowsheet at R (7
runs); all controls held on what was run, and the readings of the rebuild and of W were visible
(F0, C1, A0). Full suite on d256082: 1823 passed, 36 skipped (the worktree holds only the denylist
and the price file of `local/`). Main run on the clean tree at d256082: 1106 Dyssol runs, four at a
time, 1.06 to 1.75 s per run and at most 60 MB per process, 378 s in all; no run failed. Artifact
committed at 781a17a after `tests/test_denylist.py`. A scan of the artifact against the local
record found one shared number, the saturation ratio of line 2 at P0, which is committed by
decision 11 (as the ratio at S in Befund 130).

## 19. Artifacts

`artifacts/runs/plant_cases_v1.json` (ranks, savings and regrets in %, saturation ratios, counts,
flags, reading codes, the control, the hit rates of the two rules, digests);
`tests/test_plant_cases.py::test_artifact_holds_the_numbers_of_the_document_and_of_the_lab_book_entry`
checks the numbers of sections 15 and 17 and of the lab-book entry against it. Local record
`local/runs/plant_cases/plant_cases_v1_local.json` of the worktree, copied to
`C:/twinlink_runs/plant-cases/` (digest verified); German report
`local/reports/plant_cases_bericht.md` of the main checkout (not versioned).

## 20. Decisions of the project lead (2026-11-25; added after the fact, sections 1 to 19 are unchanged)

1. **F0, C1 and A0 are accepted,** and with them "all predictions confirmed (not blind)" (section
   17, item 1). The three decisions taken while writing the code (section 12, last paragraph) are
   confirmed.
2. **Befund 110 and Befund 121 get scope notes** in the wording of section 17, items 2 and 3: dated
   addenda, append-only, entered by the managing session at the merge, not on this branch. For
   Befund 121 in addition: the generalised rule (the warmer exhaust to the sink with the larger
   heat-capacity rate) hits 10 of 10 at the new points and, in the physical modes, as well on the
   box of Befund 121 (0.9725 against 0.9735); in the partner mode worse (0.840 against 0.931). It
   is a post-hoc hypothesis and no replacement of the rule.
3. **The restriction on the circuit flows belongs in the summary of this document,** not only
   among the caveats: with both circuits at the same small flow cases 1 and 2 are practically
   equal in the physical modes (section 15.6). It now stands in the status paragraph at the top.
4. **The script step that was not chained is noted openly in the lab book** (addendum to the
   entry of this finding): before the commit d256082 a scripted cosmetic edit of
   `tests/test_plant_cases.py` failed on its own assertion and was not chained to the commands
   after it. Without consequence: the script wrote nothing and the file was committed as tested.
5. **The questions 13 to 16 and the hint for the sender go out,** sent by the project lead, not
   from the branch.

Full suite on the commit with the results and the test of the artifact (44b291f): 1824 passed,
36 skipped. This section, the status paragraph and the addendum to the lab-book entry are
documentation only.

With these decisions the study is closed on this branch. Open at the merge (managing session):
document and finding number, the two scope notes (item 2), the row of `docs/12` with the questions
13 to 16, the README sentence. Open outside the repository: the sender's answers.
