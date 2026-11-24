# NN -- Cost-optimal flow of the heat-transfer medium in the one-line Dyssol flowsheet ("case 1"): fuel against pump electricity

Status: **design; committed before the run.** Sections 1 to 8 are the design. Results are added
below them after the run and take their numbers only from the artifact named there. Descriptive
study, **no pre-registered reading**. Branch `carrier-sweep` (from `origin/master` 1c22a81),
worktree `C:/twinlink_work/csweep`. Document number open (`NN`), lab-book entry "Befund NN (number
open)"; the managing session assigns both at the merge. No merge, no push. Code
`src/twinlink_ml/dyssol_steady/carrier_sweep.py`, settings `conf/dyssol_steady/carrier_sweep.yaml`,
driver `scripts/run_carrier_sweep.py`, tests `tests/test_carrier_sweep.py`.

**Placeholders.** The optimum of this study exists only through a pump law **without a source**
(section 4); prices are the placeholders of Befund 115. The model is the rebuild of the
stationary Dyssol units with ideal exchangers. No statement about a plant.

## 1. Question

TUHH-A delivered a sheet (2026-11-24) with results of the stationary Dyssol flowsheet "one line
with its own heat recovery" (case 1 of the partner workbook) over the flow of the heat-transfer
medium, for two processes. In the units the recovered heat only grows with the medium flow
(section 4). **Against which medium flow does the electricity of the circulation pump balance the
fuel saved, at outside temperatures of -10, 0, 10 and 20 degC, and how flat is the cost around
that flow?**

## 2. Decisions of the project lead (2026-11-24)

1. Total cost = fuel + pump electricity over the medium flow (0 to 15 000 kg/h, fine), for
   outside temperatures of -10, 0, 10 and 20 degC. Pump: pressure drop quadratic in the flow with
   a placeholder (justified in this document and marked as a placeholder), efficiency as in the
   network studies; prices and CO2 factors as in Befund 115. Result: the optimal flow and the cost
   at 80 % and 120 % of it; sensitivity to the pressure drop (factors 0.5 and 2). No reading;
   descriptive.
2. **Change of the task (same day, before any run).** The task first contained a part 1, a check
   of the rebuild against the sheet. It is dropped: that check exists as a finding on the branch
   `dyssol-sweep-case1` (with a Dyssol control) and is cited from there (section 3). The optimal
   flow is computed **only for the admissible process** (process 2 of the sheet). Process 1 is
   over-saturated at every point; for it this document only notes that no meaningful statement is
   possible without a saturation limit (section 5). Code written for part 1 is not committed.
3. Plant values and results in absolute units stay local; the repository gets code, a template
   and relative key numbers. Full suite before the commit of the run, and only after the suite
   of `dyssol-sweep-case1` has finished. No merge, no push.

## 3. Source and what is cited

**The sheet** (`<local>/partner/dyssol_2026-11-24/`, confidential, never committed; SHA-256 in
the record). Only the input block of process 2 is used here: gas flow and moisture flow of the
fresh gas, its temperature, dryer inlet temperature, spray flow, solid flow and temperature,
heater efficiency, calorific value, equilibrium moisture. It is copied into a local file
(`<local>/dyssol_steady/carrier_sweep_local.yaml`; template
`conf/dyssol_steady/carrier_sweep_local.yaml.example`), generated from the cells of the sheet.

**Cited from Befund 130** (lab book and `docs/45_dyssol_steady_state.md`, section 17, on the
branch `dyssol-sweep-case1`, commit 0fa581c; not merged into `master` when this was written, the
number was set on that branch; artifact `artifacts/runs/dyssol_sweep_case1_comparison.json`
there, same SHA-256 of the sheet as here):

* the sheet's gas mass flow is the dry gas (91 of 128 reported values within rounding with this
  reading, 13 with the humid total);
* Dyssol and the rebuild agree to rounding at all 22 points of the sheet (at most 5.7e-14 K and
  2.2e-13 %); against the sheet the fuel differs by at most 0.067 kg/h (0.08 % and 0.07 % in the
  two processes) and the recovered heat of process 2 by at most 0.081 kW (0.05 %);
* the dryer exhaust of process 1 holds 2.015 times the water of saturated air at its temperature
  at every point; process 2 is not over-saturated.

Nothing of this is re-computed here. Befund 130 reports the verdict S3 of its own reading for the
sheet as a whole (12 of 128 values more than one unit of the last decimal away); the two rows
this study builds on, fuel and recovered heat of process 2, are the ones quoted above.

**The model** is `dyssol_steady.reference.evaluate` with one line and `preheated_by = (0,)`, with
the properties of the materials database shipped with Dyssol plus the one added constant
(docs/45, section 3; digests in the record). No Dyssol run.

## 4. Cost of fuel and pump over the medium flow

Per outside temperature `T_a` in {-10, 0, 10, 20} degC, over the medium flow `m` from 0 to
15 000 kg/h in steps of 10 kg/h (the optimum of the grid is refined between its two neighbouring
grid points with the closed formula):

    cost(m) = c_fuel * fuel(m) * H_u + c_el * P_pump(m)          [EUR/h]

* `fuel(m)`: fuel of the heater from the rebuild, with the inputs of process 2 and the fresh gas
  at `T_a`; `H_u` is the calorific value of the sheet.
* **Prices and CO2 factors as in Befund 115** (`conf/storage_price/storage_price.yaml`, i.e. the
  defaults of `heatpump.config.Economics`, Befund 101; all placeholders except the CO2 factors):
  gas 0.040 EUR/kWh of fuel plus the CO2 price of 55 EUR/t at 0.201 kg/kWh, so
  `c_fuel` = 0.051055 EUR/kWh; electricity `c_el` = 0.16 EUR/kWh (Befund 115 uses an hourly price;
  this stationary study takes its time mean, which docs/32 gives as 0.159 EUR/kWh, "i.e. the 0.16
  EUR/kWh of Befund 101"); the CO2 of electricity (0.366 kg/kWh) is inside its price and enters
  only the reported CO2.
* **Pump -- PLACEHOLDER WITHOUT A SOURCE.** `P_pump = m dp(m) / (rho eta)` with
  `dp(m) = f * 1 bar * (m / 10 000 kg/h)^2`, so the pump power is cubic in the flow.
  Justification of each part: the quadratic law is the pressure drop of turbulent flow through a
  fixed geometry; 1 bar is the repository's placeholder for one medium circuit, both exchangers
  and the piping (`flowsheet.auxiliary.AuxiliaryConfig.loop_pressure_drop_pa`, docs/11), which the
  network studies charged at any flow; the reference flow of 10 000 kg/h is the largest flow of
  the delivered sheet and a setting, since no design flow of the circuit is known; `eta` = 0.40
  and `rho` = 1000 kg/m3 are the values of the network studies (`AuxiliaryConfig`: the laboratory
  default of `simulation.config.EnergyConfig`, water). `f` = 1 is the placeholder; **sensitivity
  `f` = 0.5 and 2** (decision 1). At the reference flow and `f` = 1 the pump takes 0.69 kW.
* **Reported** per outside temperature and `f`: the optimal flow (in the repository as a share of
  the upper bound of the scan, with a flag when it lies at that bound or at zero), the saving
  against no medium flow, the cost at 80 % and at 120 % of the optimal flow as an increase over
  the optimal cost (in percent of the cost and in percent of that saving; 120 % of an optimum
  above 12 500 kg/h lies above the scanned range, is evaluated with the closed formula and is
  flagged), the share of the pump in the cost, the change of CO2 against no flow.
* **Over-saturation** is checked at every scanned point: the gas must not hold more water than
  saturated air at 1 atm, neither at the dryer outlet nor at the exhaust after the recovery
  (which gets colder with more medium flow and with colder fresh gas). The share of
  over-saturated points of each scan is recorded.

**What the units cannot show.** The exchangers are ideal: the recovered heat is
`(T_hot - T_cold) / (1/A + 1/B + 1/C)` with the capacity rates `A`, `B`, `C` of exhaust, medium and
fresh gas. It grows monotonically with the medium flow `B` and saturates at
`(T_hot - T_cold) / (1/A + 1/C)`; there is no exchanger area and therefore no medium flow beyond
which a real pair of exchangers would lose effectiveness. An optimum of this study exists only
through the pump placeholder.

## 5. Process 1: no statement

The dryer exhaust of process 1 is over-saturated at every point of the sheet (Befund 130:
2.015 times the saturation humidity, in Dyssol and in the rebuild). The dryer unit has no
saturation limit (docs/45, section 3; open point of Befund 106): it evaporates the whole spray
down to the equilibrium moisture and returns an exhaust state that does not exist. Exhaust
temperature and recovered heat of that process are therefore numbers of the unit's formula, not
of a dryer, and a cost optimum built on them would have no meaning. **No optimal medium flow is
computed for process 1.** A statement needs a dryer unit with a saturation limit (question to the
units' author, already sent after Befund 128; docs/45, section 16 item 5).

## 6. Seen before the run (disclosure)

* The whole sheet, read cell by cell while the dropped part 1 was prepared; arithmetic by hand on
  its values (recovered heat = capacity rate of the medium times the difference of its two
  temperatures; fuel saved = recovered heat over efficiency and calorific value).
* An estimate by hand, made before the reference flow of the pump placeholder was written down:
  with 1 bar at 10 000 kg/h the pump cost at the upper bound of the scan is of the order of a
  tenth to a third of the fuel still saved per added flow, so the optimum probably lies at or
  near the upper bound of the scan. The placeholder was not tuned after that.
* Befund 130 on the branch `dyssol-sweep-case1` (section 3), read after the change of the task.
* **History of this branch.** A first commit of this branch (039769b: design, a reading V0/V1 and
  code for both parts) was made before the change of the task and removed from the branch after
  it (the branch was not pushed; a patch is kept in `C:/twinlink_runs/carrier-sweep/part1_dropped/`).
  A full suite started on it was stopped at about 4 %; **no run was made with it, and no point of
  the sheet was recomputed in this session**. The code of the present commit was tested on
  laboratory-scale inputs only.

## 7. Decisions taken without asking (to be confirmed by the project lead)

1. **Pump placeholder:** 1 bar at 10 000 kg/h, quadratic; efficiency 0.40 and density 1000 kg/m3
   from `AuxiliaryConfig` (section 4).
2. **Electricity price:** the time mean 0.16 EUR/kWh instead of the hourly price of Befund 115.
3. **Humidity of the fresh gas at other outside temperatures:** the sheet's value, capped at
   saturation at the outside temperature (a fresh gas above saturation is not air); the optimum
   without the cap is reported where the cap is active. Below 0 degC the repository's saturation
   function is held at its value of 0 degC, so the cap uses the Magnus form over ice there
   (`carrier_sweep.fresh_gas_saturation_g_per_kg`); the shared function is unchanged.
4. **Everything else constant over the outside temperature** (gas mass flow, spray flow and
   temperature, dryer inlet temperature).
5. **Over-saturation at two places**, dryer outlet (as docs/45) and exhaust after the recovery.
6. **Cost at 120 % of an optimum near the upper bound** is evaluated above the scanned range and
   flagged, instead of being left out.
7. **What is committed:** shares, percentages, ratios and flags. Optimal flows in kg/h, cost in
   EUR/h, recovered heat and the inputs of the line stay in the local record
   (`<root>/local/runs/carrier_sweep/`, copied to `C:/twinlink_runs/carrier-sweep/`).
8. **Removal of the first commit** from the unpushed branch (section 6), to follow "code written
   for part 1 is not committed".

## 8. Confidentiality and tests

Not in the repository: the sheet (plant values, names of the plant lines), the local inputs of
the line, the local record, the report in German (`local/reports/`). `tests/test_denylist.py` is
run before every commit of this branch.

`tests/test_carrier_sweep.py` (laboratory-scale inputs only): the settings equal the placeholders
of the network studies and of Befund 115; the template names the fields of the input block and
holds no value; the gas flow is read as dry gas; over-saturation only after the recovery is
flagged; the fresh gas keeps its humidity up to saturation, also below 0 degC; the pump law is
cubic and meets the placeholder of the network studies at the reference flow; the cost curve
equals rebuild times prices; the fuel saved is the recovered heat over the heater efficiency; the
optimum is interior with a costly pump, at the upper bound with a cheap one and at zero when the
exhaust is colder than the fresh gas; a dry run of the driver writes no input, flow, heat or cost
into the record meant for the repository.
