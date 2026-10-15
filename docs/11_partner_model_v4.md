# Partner model v4 (`Model_v4.xlsx`): one-to-one reconstruction, deviation list, proposed reference values

Merged to `master` as Befund 72 (`docs/04_stand_ap1.md`; 71 is already used by the unmerged branch `m3`, so 72 was taken to avoid a clash). No comparison with the lab model:
the inputs of the partner model (300 degC inlet air, product dried to equilibrium moisture)
are outside it.

Status of the workbook: draft. Parts are `#REF!` or empty (see 3). All statements below are about
this draft state. The workbook itself is confidential, lives in `local/` and is not in the
repository; tests that read it are skipped when it is missing.

Code: `src/twinlink_ml/partner_v4/` (own package; `heatnetwork` is unchanged). Tests:
`tests/test_partner_v4.py`. Artifact: `artifacts/runs/partner_v4_deviations.json` (commit
`a3c710c` of the script, run on a clean tree). Units: internal SI (kg/s, kg/kg dry air, J, W, Pa,
temperatures in degC and named so); the as-built module works in the sheet's units on purpose.

## 1. Structure of the workbook and of the rebuild (V4.1)

Inputs -> Balances -> Scenarios. Inputs: two dryers (spray tower, fluidised bed) with identical
values (50 kg/h air, 6 g/kg, 300 degC; 5 kg/h spray, solid fraction 0.4, 15 degC), material
properties (`cp_gas` 1.2, `H_u` 45000 kJ/kg, `eta` 80 %, `cp_liquid` 4.2, `cp_solid` 1, `cp_vapour` 2
kJ/(kg K), equilibrium moisture 2 g/kg, `h_fg` 2500 kJ/kg), ambient 15 degC, 1 atm. Balances: per
dryer outlet moisture (F4, F11) and outlet temperature (F5, F12) from a mass and an enthalpy
balance; Antoine coefficients (not yet used by any formula). Scenarios: six cases; per case
two exhaust streams ("heat recovery 1/2"), two pre-heaters per sink (low/high temperature),
a gas heater per sink (rows 18-21 and 31-34), total fuel (row 36).

| Module | Content |
|---|---|
| `inputs.py` | `ExcelInputs` (sheet units), `PartnerV4Inputs` (SI), conversion only in `to_si()`, reader of the sheet `Inputs` |
| `as_built.py` | the sheet's formulas as they are, bugs included; reproduces the stored values |
| `corrected.py` | same model in SI, units consistent; dryer balance, heater, gas, Antoine, condensing heat release |
| `cascade.py` | the six cases as a cascade (our reconstruction, assumptions A1-A5) |
| `deviations.py`, `workbook.py` | deviation list; reading cells (formula text and Excel's stored value) |

**Tests against Excel's stored values of the finished cells** (rel. 1e-12): Balances F4 = 65.92 g/kg,
F5 = 159.00293 degC, F11, F12 (same, dryers identical); Scenarios E21, G21, I21, K21, O21, I34, M34,
I36, M36. A further test fails if the partner fills one of the `#REF!` cells (then the
reconstruction needs a look). The corrected dryer model agrees with the sheet and closes an
independent enthalpy balance to 1e-12 W (the energy balance and the units of `h_fg` in
Balances are consistent; the `Model_v3` problem with `h_fg` does not occur here).

## 2. Deviation list (V4.2)

Workbook value vs. rebuilt value *with the same inputs the cell itself uses*. Hand calculations are
given so that each line can be redone with a calculator.

| Item | Cell | Workbook | Rebuilt | Cause |
|---|---|---|---|---|
| B1/B2 | Balances F4, F5, F11, F12 | 65.92 g/kg; 159.003 degC | same | **no deviation**. Hand: `Y = 6 + 1000*5*(1-0.4*1.002)/50 = 65.92`; `T = 10909 / 68.6088 = 159.003` |
| G | Scenarios E21, G21, K21, O21, I34, M34 (gas) | e.g. K21 5.225e-6 kg/h | 0.47975 kg/h | (a) moisture as 6 (g/kg) instead of 0.006 in c_p: 13.2 vs 1.212 kJ/(kg K), factor 10.89 too high; (b) `H_u` is already kJ/kg and is multiplied by 1e6. Row 21 is therefore 9.2e4 times too low (ratio 1.089e-5). Hand: `P = 50/3600*1.212*285 = 4.7975 kW`; `m = 4.7975 / (0.8*45000) = 1.3326e-4 kg/s = 0.47975 kg/h` |
| T | Scenarios I36, M36 (total) | 0.01881 | 0.47975 | additionally `*3600` on a quantity that is already kg/h. Total factor `10.891 * 1e-6 * 3600 = 0.0392`: **the sheet is 25.5 times too low** (expectation confirmed) |
| R | Scenarios F25, H25, L25, P25, L29, P29 (cells are `#REF!`; formula evaluated at Q = 1 kW) | dT = 1.515 K | dT = 59.41 K | `Q*1000` where kW to kJ/h needs `*3600` (3.6 too low) and moisture in g/kg in c_p (10.89): total 39.2 too low. Hand: `1*1000/(50*13.2) = 1.515`; `1*3600/(50*1.212) = 59.41` |
| R | Scenarios E26, G26, K26, O26, K30, O30 | returns the 1 kW | 0.0255 kW for the sheet's own dT | `/1000` and the same wrong c_p cancel the error of F25: the energy cell reports the input back, but the temperature that reaches the heater (E33 = F25) corresponds to 2.6 % of it. Energy bookkeeping and heater input disagree |
| P | Inputs I4 (c_p gas) | 1.2 | 1.0: gas 0.40058 vs 0.47975 kg/h (-16.5 %) | 1.2 is high for air (about 1.0 over 15-300 degC). **Question, not corrected** (the corrected model keeps 1.2) |
| A | Balances J4, K4 | 101962, 102651 | 10.1962, 10.2651 | decimal point lost (factor 1e4) |
| A | Balances J6, K6 | 233426, 244485 | 233.426, 244.485 | same (factor 1e3). Probable cause: decimal/thousands separator of the locale (to be confirmed) |
| A | Balances J5, K5 | text '1730.63', '1810.94' | numbers | stored as text; works in an operator, not in functions |
| A | use | `log10 p = 101962 - 1730.63/(233426+100) = 101962` | 5.008 (p about 1.02e5 Pa at 100 degC) | with the stored values `p = 10^101962` overflows. Check of the corrected set: 101.3 kPa at 99.99 degC |
| U | Inputs D7, D14 | label `gm/kg` | kg/kg | value 0.4 is a mass fraction; formulas use it as kg/kg |
| U | Scenarios D6, D7 | `kg/s`, `kg/kg` | kg/h, g/kg | values 50 and 65.92 are kg/h and g/kg (as D3, D4) |
| S | Scenarios E20, G20 | 0 degC | proposal 159.0 degC | refer to F12/H12 (outlet of pre-heater low 1), empty: heater computed from 0 degC (E21 = 0.505 kg/h for 0 to 300 degC) |
| S | Scenarios I20, J20, M18..M21 | empty / bare `=` | | case 3: heater 1 empty (I21 = 0); case 5: M18 holds `=`, M19:M21 missing. I36 and M36 count only heater 2 |
| S | Scenarios F25 vs H25, L25, P25 | F25 without `MIN` | | only F25 is not capped at the heater target |
| S | Scenarios E26; K26, O26, K30, O30 | `Inputs!E4`; `G11` | own block | moisture reference to another dryer / another case column; identical today (all 6 g/kg), wrong as soon as the dryers differ |
| Q | Inputs E3 / Balances F4 | 65.92 | 66.28 g/kg if the 50 kg/h include the 6 g/kg | balance treats the flow as dry air; question on the basis (0.5 %) |

Not a deviation but worth stating: both dryers have identical inputs, so sources 1 and 2 are
identical (159.003 degC, 65.92 g/kg), which makes cases 2 = 1, 4 = 6 and 3 = 5 pairwise equal in
the numbers; the order of the cascade is not tested by these inputs.

## 3. Reference values for the open cells (V4.3) -- OUR RECONSTRUCTION, not the partner's numbers

The `#REF!` cells most probably referred to the heat that an exhaust stream offers (deleted rows). The
proposal fills them under these assumptions (details in `cascade.py`):

* **A1 case structure** (from which blocks exist per column and from the six cases of `Model_v3`,
  Befund 57): 1 separate (source 1 -> sink 1, source 2 -> sink 2), 2 crossed, 3/5 both sources into
  sink 1 (order 2,1 / 1,2), 4/6 both into sink 2. Source i = exhaust of dryer i, sink i = inlet air of dryer i.
* **A2 offer of a source**: heat released down to `T_cold` including condensation (Antoine, dew
  point 45.30 degC at 65.92 g/kg). `T_cold` is not in the sheet: **placeholder = ambient, 15 degC**. Offer per source:
  4.559 kW.
* **A3 stage limits** as the sheet's own `MIN(.., target, #REF!)` and `IF(#REF! <= T_sink, ..)`, reading the
  `#REF!` as the source's inlet temperature: the sink leaves a stage at `min(T_in + Q/C, 300 degC, T_source,in)`.
* **A4 additional feasibility check** (not in the sheet): counter-current with minimum approach `dT_min`
  (0 and 20 K; 20 K is the repo's placeholder).
* **A5** stage 1 = "low", stage 2 = "high" pre-heater row.

Result (kg/h gas, total of both heaters; case 0 = no recovery):

| Variant | 0 | 1, 2 | 3, 4, 5, 6 |
|---|---|---|---|
| sheet structure (A3) | 0.9595 | 0.4747 | 0.7171 |
| counter-current, dT_min 0 K | 0.9595 | 0.4747 | 0.7171 |
| counter-current, dT_min 20 K | 0.9595 | 0.5420 | 0.7508 |

Proposed cell values (sheet structure; cases 1/2 per sink): pre-heater outlet (F25, H25, F12, H12) 159.003
degC, recovered energy (E26, G26, E13, G13) 2.424 kW, heater inlet (E33, G33, E20, G20) 159.003 degC, gas of the
heater 0.2373 kg/h, total (E36, G36) 0.4747 kg/h; cases 3-6: stage 1 2.424 kW, **stage 2 0 kW** (K30, O30,
I17), total 0.7171 kg/h. The complete table is in the artifact (`proposal`).

**Reading.** The energy offered (4.559 kW per source) is not what a sink can take: a sink with the same
mass flow can at most reach the exhaust inlet temperature (159.0 degC), which is 2.424 kW of its 4.798 kW
need (50.5 %). The sum "offer" and the usable heat differ by a factor of 1.9; the second stage of a cascade
adds nothing because the sink is already at the temperature of the second source. This is the same
pattern as in Befund 57 d) (case 5: second stage exactly 0 kW) and follows from the sheet's own structure,
not from our assumptions; the size of the numbers depends on A2-A3 and on the equal flows.

## 4. Gas consumption as a computation (V4.4)

`gas_mass_flow_kg_per_s(P_heat) = P_heat / (eta * H_u)`: a deterministic function of the heating power
(`heater_power_w`), not a quantity to be predicted or learned. The fuel power is `E_gas = m_gas * H_u = P_heat / eta`.

Connection to the per-carrier objective (D5, `docs/09`) -- described, nothing changed: the objective of the
gas-fired heater uses `E_gas = (Q_heater - Q_rec) / eta_H` and `CO2 = 0.201 * E_gas` (gas factor, kg
CO2e/kWh). The partner's gas flow is the same quantity expressed in mass: `E_gas = m_gas * H_u`. Hence (i)
the model predicts the process state, the heater power follows from it, gas and CO2 follow from the power;
(ii) `eta` and `H_u` are inputs of one function, kept in one place (the values in the repository are listed in section 6; nothing is unified here); (iii) the mass flow is for
plausibility against the partner's table; the objective needs `E_gas` in kW, not kg/h.

## 5. Caveats

* The reconstruction reproduces a draft. All finished cells agree with Excel; the open cells are
  our proposal and say nothing about what the partner will compute.
* `cp_gas`, the moisture basis and `T_cold` are questions, not corrections.
* The cascade is energy- and temperature-limited, not a heat-exchanger design (no area, no dT profile beyond the
  check A4).
* The efficiency and the calorific value occur with several values in the repository (section 6).
* The feedback to the partner is in `local/` (not in the repository).

## 6. Efficiency and calorific value in the repository (inventory only, nothing changed)

The unification is decided after the M3 run. State of `master` (commit `b582c21`) and of branch `m3`
(commit `54e4bbb`; `m3` differs from `master` in `site_network.py` and `heatnetwork` only by the shared cap
and `with_spray_rate`, not in the values below).

| Quantity | Value | Where | Status |
|---|---|---|---|
| efficiency, process air heater (electric) | **0.98** | `EnergyConfig.heater_efficiency` (`simulation/config.py`, the three `conf/process/*.yaml`) | placeholder of the lab simulator; converts electric heater power to thermal duty |
| efficiency, heat-network heater | **0.90** | `DEFAULT_HEATER_EFFICIENCY` in `heatnetwork/config.py` | placeholder default, no source; overridden in the site model (`SiteSpec`) and in the pinch/six-case scripts that were checked, **not verified for every older script** |
| efficiency, gas heater (partner case) | **0.80** | `SiteSpec.heater_efficiency` (`evaluation/site_network.py`), `OracleSpec.gas_heater_efficiency` (`evaluation/carrier_oracle.py`), `GAS_HEATER_EFFICIENCY` in `run_m1_margin_cost.py`, `HEATER_EFFICIENCY` in the pinch and six-case scripts | taken from the partner workbook (80 %), also `Inputs!I6` of v4 |
| calorific value | **50 MJ/kg** | `DEFAULT_FUEL_LOWER_HEATING_VALUE_J_PER_KG` in `heatnetwork/config.py` | placeholder default (natural gas H, rule of thumb); overridden in the same places as the 0.90 |
| calorific value | **45 MJ/kg** | `SiteSpec.fuel_lower_heating_value_j_per_kg`, pinch/six-case scripts | taken from the partner workbook (45000 kJ/kg), also `Inputs!I5` of v4 |

So there are three efficiencies (0.98, 0.90, 0.80) and two calorific values (50, 45 MJ/kg); the partner
workbook v4 agrees with 0.80 and 45 MJ/kg. The 0.98 and the 0.80 are not alternatives for one thing: they
belong to two different devices (electric lab heater vs. gas-fired heater of the network).

**Which value which stage used** (from code and docstrings; the artifacts were not re-opened):

| Stage | Efficiency | Calorific value | Mechanism |
|---|---|---|---|
| M1 (single line, `run_m1_margin_cost.py`, `OracleSpec` with `co2_gas_heater`) | 0.98 converts heater power to duty, then 0.80 for the gas: `E_gas = duty / eta_gas` (`carrier_energy.py`) | not used (the carrier energy is in kW, not kg) | the 0.90 and the 50 MJ/kg do not enter |
| M2 (site network, `run_m2_site_network.py`, `SiteSpec(k)`) | 0.80 set explicitly in the `HeatNetworkConfig` built by `site_network.py`; the line's inlet-air sink is fed from the cascade | 45 MJ/kg (`SiteSpec`); gas power is `fuel_mass_flow * H_u` | the 0.90 and the 50 MJ/kg are overridden; the docstring of `site_network.py` says the fuel is what the sink needs after recovery, so the 0.98 does not enter the heater of the site model |
| M3 joint stage (branch `m3`, `run_m3_joint.py`) | line tables via `OracleSpec(.., "co2_gas_heater", m1.GAS_HEATER_EFFICIENCY)` = 0.80 (with the 0.98 as in M1); the site model `SiteSpec(k)` with 0.80 | 45 MJ/kg (`SiteSpec`) | `joint_search.site_spec_with_cap` copies `heater_efficiency` from the given `SiteSpec` |
| Befund 57-68 (older network studies) | 0.90 via the default where not overridden; 0.80 in the six-case and pinch scripts | 50 MJ/kg by default; 45 MJ/kg in the pinch/six-case scripts | to be checked case by case if these numbers are reused |

Open point to settle with the unification: in M1 the same fuel quantity passes the 0.98 (lab heater, electric)
and then the 0.80; in M2 and M3 the site model books the sink demand directly. Whether the two routes give
the same gas power for one line is not shown here (no run).

### Planned follow-up (noted only; own task AFTER the end of the M3 run; nothing is changed now)

a) **Check the double efficiency in M1.** Compute the gas demand of one line at one operating point on both
   routes: the M1 route (heater power x 0.98 = duty, duty / 0.80 = gas) and the site model of M2
   (sink demand booked directly, 0.80 and 45 MJ/kg). Expectation if the efficiency is counted twice: M1 too
   high by the factor 1/0.98.

b) **Unification.** One central configuration for gas-fired heaters with `eta` = 0.80 and `H_u` = 45 MJ/kg
   (partner model); 0.98 only for the sensitivity with an electric heater. Remove the defaults 0.90 and
   50 MJ/kg from `HeatNetworkConfig`, so that the values are always passed explicitly.

c) **Regression test.** M2 and `m3` stay bit-identical (they already use 0.80 / 45). Recompute M1; the
   percentages must not change, only absolute figures.


## 7. Partner model v4.1 (`Model_v4.1.xlsx`) with the equation document

`Model_v4.xlsx` and its rebuild (sections 1-6, `partner_v4`, `tests/test_partner_v4.py`) are unchanged and stay valid for that
file. The new version is reconstructed in its own package, `src/twinlink_ml/partner_v4_1/` (tests `tests/test_partner_v4_1.py`,
script `scripts/report_partner_v4_1_comparison.py`, artifact `artifacts/runs/partner_v4_1_comparison.json`, script commit
`8f7f31a`, run on a clean tree). Lab book: Befund 74. The workbook and the equation document (`Flow diagram and balance equations v4.docx`) are in
`local/` and not in the repository; in the working copy the workbook is named `Model v4.1.xlsx` (the tests also accept
`Model_v4_1.xlsx`; `TWINLINK_LOCAL_DIR` names the directory when `local/` is not in the checkout, e.g. in a worktree).
No comparison with the lab model, as for v4.

### 7.1 What changed in the structure

The `#REF!` cells of v4 are gone: each exhaust now heats a circuit of a heat transfer medium (equations 4.1/4.2 of the
document); the rebuild of section 3 (offer of a source, `T_cold`, `dT_min`, assumptions A2-A4) is therefore not needed for v4.1.
The case structure (A1) is confirmed by the six case drawings of the document (case 3: low stage exhaust 2, high stage exhaust 1;
case 5 the other way round; cases 4 and 6 the same two orders into sink 2). The Antoine table is no longer in the workbook
(condensation is neglected, as the document states for 4.1).

**The two balance sheets.** `Balance` (singular) repeats `Balances!A1:F15` (the two dryer blocks) in formula text and in the stored
values and is referenced by no formula; it is a leftover. `Balances` is the one in use: `A:F` feed `Scenarios`
(`Scenarios!E3`, `E4`, `E5`, `E9`, `E10`, `E11` read `Balances!E3`, `F4`, `F5`, `E10`, `F11`, `F12`), and `H:T`, rows 3-8, hold
for each case and each of the two circuits the intermediate variables `K1`..`K6` (the coefficients of the 2x2 system of 4.1/4.2:
`K1 = K5 = -Mw cp_w`, `K2 = A + B`, `K3 = A T_c,in`, `K4 = B + C`, `K6 = C T_h,in`; `A`, `B`, `C` the capacity rates of exhaust,
medium and process gas). Which equation is realised by which cells (`EQUATION_TO_CELLS` in `deviations.py`): (1.3) `Balances` F4, F11;
(1.5) F5, F12; (4.1)/(4.2) `Balances` I3:T8 and the solution in `Scenarios` rows 5, 7, 11, 13; (2.5) rows 20, 24, 33, 37; (3.2)
rows 26, 39. **(2.1)-(2.4) are not used as such** (the sheet has no single-exchanger cell with or without condensation); they are
in the rebuild (`loops.py`) because (4.1) is the two-exchanger form of (2.4) and the circuit flows must stay adjustable.

### 7.2 Inputs against `Model_v4.xlsx` (artifact `workbook.inputs_diff_previous_to_new`)

| Cell | v4 | v4.1 |
|---|---|---|
| `E3` air dryer 1 (spray tower) | 50 kg/h | 100 kg/h |
| `E5` inlet temperature dryer 1 | 300 degC | 150 degC |
| `E6` spray dryer 1 | 5 kg/h | 6 kg/h |
| `E10` air dryer 2 (fluidised bed) | 50 kg/h | 100 kg/h |
| `E13` spray dryer 2 | 5 kg/h | 12 kg/h |
| `E20`, `E21` mass flow of the circuits (new) | | 20 kg/h each |
| `K3:K8` (new) | | 50, 6, 150, 3, 0.4, 15 |

Everything else in `Inputs` (material properties, ambient 15 degC, 1 atm, moisture 6 g/kg, solid fraction 0.4, spray 15 degC) is
unchanged; `cp_gas` is still 1.2 kJ/(kg K); `Inputs!I12` (`cp` of the medium, 4.2) is used now. **The dryers are not swapped against
v4**: in v4 both were identical (50 kg/h, 300 degC, 5 kg/h); in v4.1 dryer 1 has twice the air at 150 degC and dryer 2 twice the air at
300 degC with 2.4 times the spray. The exhausts differ (dryer 1: 72.28 degC, 41.95 g/kg; dryer 2: 134.66 degC, 77.90 g/kg), so the order
of the cascade is now visible in the numbers (the feedback on v4, section 9 item 4, had asked for such an input set).
**Column `K` is referenced by no formula** (checked in the formula text of all 334 formulas, artifact `workbook.references`); its values
have the layout of a dryer block (air 50 kg/h, 6 g/kg, 150 degC, spray 3 kg/h, 0.4, 15 degC): a second, smaller input set for dryer 1,
not wired in. A test fails if a formula starts to use it, or the sheet `Balance`.

### 7.3 Three-way comparison, cell by cell (artifact `cells`: 334 numeric formula cells of `Scenarios`, `Balances`, `Balance`)

Three values per cell: the equation document (typos corrected, circuit flows as in the sheet), Excel's stored value, the as-built
reproduction.

* **Reproduction.** The as-built rebuild reproduces **all 334 stored values**, largest relative error 1.9e-16 (the cell references
  of the sheet really realise the case structure; this is tested, not assumed).
* **Document against workbook.** All temperatures, flows and energies of the document model agree with the workbook to 1e-9. They
  differ in **18 cells: rows 26 and 39 (gas) and row 41 (total)**, all by the same factor **108.91** (`document_vs_excel_factors`).
  The `K` constants and the solution of the 2x2 system are exactly (4.1)/(4.2) (checked against the document-based `K`).

| Item | Cells | Workbook | Document / rebuild | Cause and hand calculation |
|---|---|---|---|---|
| **G** gas | `Scenarios` rows 26, 39 (e.g. `I39`; `E26`) | `I39` 104.5 kg/h; `E26` 43.30 | 0.9595; 0.3976 | `= air*(T_out-T_in)*(cp_g + Y*cp_v) / (H_u*eta/1000)` with `eta` = 80 (percent): (a) the 1000 is at the efficiency: 0.08 instead of 0.8, factor 10; (b) the moisture enters in g/kg (6 instead of 0.006): `cp` 13.2 instead of 1.212 kJ/(kg K), factor 10.891. Hand, 100 kg/h from 15 to 300 degC: sheet `100*285*13.2/(45000*80/1000) = 376200/3600 = 104.5`; document `100*285*1.212/(45000*0.8) = 34542/36000 = 0.9595` (ratio 108.91). The `1e6` and the `*3600` of v4 are gone, the unit errors are not |
| **G** total | row 41 | `E41` 134.65 | 1.2363 | sum of two gas cells, same factor |
| D1 | equation (2.1) | right-hand side `M_w cp_w T_w,in` | `T_w,out` | (2.3) and (2.4) follow from (2.1) only with `T_w,out`; closed in the tests |
| D2 | equation (3.1) | left `Y(dh + cp_v T_g,out)` | `T_g,in` | (3.2) follows from (3.1) only with `T_g,in`; closed in the tests |
| D3-D5 | text | "contains moisture of `Y_in`"; "Combining 3 and 4"; `M_gas` and `M_g` | `Y_out`; (1.1), (1.2); one symbol | wording and notation only |
| P | `Inputs!I4` | 1.2 | question, not corrected | for air about 1.0: the gas of the reference case falls by 16.5 % (tested) |

Not deviations (checked): dryer balances (1.3), (1.5) in `Balances` = document; the energy cells are `Mw cp_w (T2 - T1)/3600` kW
(2.5, unit correct); the moisture of the heater is the own block's (the cross-references of v4 are gone); labels `Scenarios`
D6/D7 are now correct; `Inputs!D7`, `D14` still say `gm/kg` for a mass fraction (as in v4); no stored number in `Scenarios` is
negative (tested). The condensation neglected by (4.1) does not occur at the sheet's flows: the exhausts leave at 56.31 and 102.61
degC (case 1) against dew points of 37.4 and 48.3 degC.

### 7.4 The six cases from the document's equations (artifact `cases`, circuit flows 20 kg/h)

Gas of both heaters in kg/h with the corrected gas formula (the workbook's own totals are the 108.91-fold); case 0 = no recovery.

| Case | Gas, document | Workbook row 41 | Recovered, kW | Heater inlet (sink 1, sink 2), degC |
|---|---|---|---|---|
| 0 | 1.4140 | | 0 | 15.00, 15.00 |
| 1, 2 | 1.2363 | 134.65 | 1.7767 | 31.91, 50.86 (case 2 swapped) |
| 3, 4 | 1.2720 | 138.53 | 1.4202 | 57.19 (one sink), 15.00 |
| 5, 6 | 1.2534 | 136.51 | 1.6061 | 62.71 (one sink), 15.00 |

Cases 3/4 and 5/6 are pairwise equal because the two dryers have the same air flow and inlet moisture, so the sink choice does not matter
for the totals; the stage order does (1.2720 against 1.2534).

### 7.5 The assumption of equal outlet temperatures; scenario 6 in reversed order (N4)

**The assumption, stated as such.** The document takes every exchanger as unlimited: both fluids leave with the same temperature
(2.3/2.4, 4.1/4.2). For one exchanger this is **parallel flow (co-current) with unlimited area** (both streams reach their mixing
temperature; (2.4) is `(C1 T1 + C2 T2)/(C1 + C2)`, tested). For the circuit it gives `Q = (T_c,in - T_h,in)/(1/A + 1/B + 1/C)`: three
capacity rates in series. Hand, case 1, stage 1: `A` = 128.39, `B` = 84, `C` = 121.2 kJ/(h K) (`K2 - B`, `K1`, `K4 - B` of `Balances!I`);
`Q = 57.279/0.027945 = 2049.7` kJ/h = 0.5694 kW. It is **not the best an exchanger can do**: with counter-current exchangers of unlimited
area the smaller capacity rate leaves at the other's inlet temperature. That bound (`loops.py`; a comparison, not the model) gives for the
first stages 1.3365 kW (source 1) and 2.7921 kW (source 2), i.e. **2.35 and 2.31 times** the equal-outlet heat (0.5694 and 1.2073 kW)
(artifact `equal_outlet.counter_current_bound`). In that bound exhaust 1 leaves at 34.80 degC, below its dew point (37.41 degC), so latent
heat would add to it. Second stages are not comparable one to one (the gas arrives warmer in the bound: in cases 3/4 at 97.94 degC, above
exhaust 1 at 72.28 degC, so the bound gives 0 kW there; in cases 5/6 54.70 against 31.91 degC, ratio 1.80). Conclusion: the equal-outlet
assumption is conservative; the size of every recovered heat and gas figure depends on it, and it is a question to the partner whether
finite areas (an effectiveness or NTU model) are planned.

**Scenario 6 in reversed order.** The premise of the task was a negative recovery in scenario 6; **it is not found in this workbook.**
Scenario 6 (sink 2: source 1 first, source 2 second) is already the order "cold exhaust first" (exhaust 1 72.28 degC, exhaust 2 134.66
degC). Nothing in the workbook is negative, and in the rebuild no stage is negative at the sheet's flows in any case, in either order.
Reversing case 6 (source 2 first, the order of case 4):

| Case 6 | Gas inlet of heater 2, degC | Gas, kg/h | Stage energies, kW (first, second) |
|---|---|---|---|
| as drawn, order 1, 2 (cold first) | 62.71 | 1.2534 | 0.5694, 1.0367 |
| reversed, order 2, 1 (hot first) | 57.19 | 1.2720 | 1.2073, 0.2129 |

The reversed order needs 0.0186 kg/h more gas and **is the order in which a negative recovery can appear**: with equal outlet temperatures the
second stage runs backwards as soon as the gas enters it warmer than the exhaust. Scan over the circuit flow (both circuits equal, artifact
`equal_outlet.scan_over_htm_flow` and `break_even_htm_flow_kg_per_h`): hot first (cases 3/4, reversed 5/6) turns negative above **147.8 kg/h**
per circuit (smallest stage -0.0012 kW at 150 and -0.0846 kW at 1000 kg/h); cold first (cases 5/6, reversed 3/4) never does (smallest stage
0.9775 kW at 1000 kg/h). So the negative recovery does not disappear by reversing scenario 6: the hot-first order brings it, and only at
flows well above the sheet's 20 kg/h. If the partner saw a negative value, it was with other inputs (flow above 147.8 kg/h) or in a
hot-first case.

### 7.6 Caveats and open points

* The document model is the equation document, not a plant model: no areas, no condensation in 4.x, equal outlet temperatures.
* Rows 26, 39, 41 of the workbook are 108.91 times too high as long as the gas formula is not changed; every other number agrees.
* `cp_gas` (1.2) and the moisture basis remain questions. The workbook file in `local/` is named `Model v4.1.xlsx`.
* Feedback to the partner (English, not in the repository): `local/feedback_model_v4.md`, section "Model v4.1".

## 8. Intermediate version, stage order, proposed rule (addendum of 2026-10-07 to section 7)

Script `scripts/report_partner_v4_1_stage_order.py`, artifact `artifacts/runs/partner_v4_1_stage_order.json` (script commit
`04cbbc7`, clean tree). The intermediate version of the partner workbook is not available as a file; its inputs were communicated and are
set directly in the rebuild (`intermediate_inputs()` in the test and the script): dryer 1 100 kg/h, 6 g/kg, 300 degC, spray 12 kg/h; dryer 2
50 kg/h, 6 g/kg, 150 degC, spray 3 kg/h; solid fraction 0.4, spray 15 degC; ambient 15 degC; circuits 40 kg/h (1) and 20 kg/h (2); material
properties as in v4.1. **Only the inputs are taken over; no cell of that file was seen**, so the only check against it is the one below.

### 8.1 Reproduction of the intermediate version, scenario 6 (the negative recovery)

The statement in 7.5 that a negative recovery in scenario 6 "is not found" holds for `Model_v4.1.xlsx`; it is **superseded for the
intermediate version**, where it exists. With the inputs above, in scenario 6 (sink 2: source 1 first, source 2 second) the rebuild gives
(both as-built and document model; the stage energy is the same formula):

| Quantity | Rebuild | Expected (partner's cells of that version) |
|---|---|---|
| stage 1 (circuit 1, 40 kg/h): gas, degC | 15 -> 81.20 | 15 -> 81.2 |
| stage 1: circuit, degC | 105.08 -> 81.20 | 105.1 -> 81.2 |
| stage 2: circuit leaves the recovery exchanger at, degC | 75.44 | 75.4 |
| stage 2: gas, degC | 81.20 -> 77.85 | -> 77.85 |
| stage 2: recovered heat (cell `Scenarios!O37` there), kW | -0.0563 | -0.056 |

All five agree with the expected values within their stated precision (tested in `test_intermediate_version_scenario_6_has_the_negative_recovery`).
The cause is the one found in 7.5: here exhaust 1 is the **hot** one (134.66 degC; dryer 1 at 300 degC) and exhaust 2 the cold one (72.28 degC), the
reverse of v4.1, so scenario 6 (source 1 first) is the hot-first order, and the second stage meets gas that is already warmer than its exhaust.

### 8.2 "Low/high temperature" does not fix the order of the circuit temperatures

The labels name the **position** (first and second stage), not the temperature of the circuit; which circuit is warmer depends on the dryer
inputs. In v4.1 source 2 is the hot one, in the intermediate version source 1. The cases of the drawings put the same source numbers in the same
positions in both, so the same case is cold-first in one version and hot-first in the other. Check per case and stage (`stage_order_check`;
a stage reverses exactly when the exhaust, and with it the circuit leaving the recovery exchanger, is colder than the gas entering the stage; the
test asserts the equivalence):

| Version | Stages that reverse (cool the gas) | Stages that come close |
|---|---|---|
| `Model_v4.1.xlsx` | none of the 12 | case 3/4 second stage 0.2129 kW (hot first) |
| intermediate | **scenario 6, second stage (source 2): -0.0563 kW**; none of the other 11 | scenario 5, second stage (source 2): +0.0896 kW (gas 60.75 degC against exhaust 2 at 72.28 degC) |

Effect of the stage order in the intermediate version (document model, gas of both heaters in kg/h; the corrected gas formula is used):
case 3 1.0052, case 4 1.0592, case 5 1.0238, case 6 1.0810 (case 6 is the worst of the six and the only one with a negative stage).

### 8.3 Proposed rule (a proposal; **not** built into the as-built reproduction of the workbook)

Two ways to keep a stage from cooling the gas, both implemented only as an option of the document model (`evaluate_case(..., rule=...)`,
default `"drawn"` = the workbook):

1. **Order the stages by the actual exhaust temperature, coldest first** (`by_temperature`). The position labels then follow the temperature.
2. **Bypass a stage whose exhaust is not warmer than the gas entering it** (`bypass`), so that its recovery is 0 and not negative.

Comparison (artifact `rule_comparison`; gas in kg/h, recovered heat in kW):

| Version, case | Drawn | By temperature | Bypass | Both |
|---|---|---|---|---|
| intermediate, 6 | 1.0810, 1.058 | 1.0592, 1.2759 | 1.0753, 1.1143 | 1.0592, 1.2759 |
| intermediate, 5 | 1.0238, 1.630 | 1.0052, 1.8154 | 1.0238, 1.630 | 1.0052, 1.8154 |
| intermediate, 3 / 4 | 1.0052, 1.8154 / 1.0592, 1.2759 | same | same | same |
| v4.1, 3 / 4 | 1.2720, 1.4202 | 1.2534, 1.6061 | 1.2720, 1.4202 | 1.2534, 1.6061 |
| v4.1, 5 / 6 | 1.2534, 1.6061 | same | same | same |

Reading: reordering is the stronger rule (it also improves the cases that do not reverse but are hot-first, e.g. v4.1 case 3/4: 1.2720 -> 1.2534,
intermediate case 5: 1.0238 -> 1.0052); the bypass only removes the negative stage (intermediate case 6: 1.0810 -> 1.0753) and leaves the order.
With the reordering, cases 3 and 5, and cases 4 and 6, become equal in the totals (the sink and the two sources are the same, only the order differed),
so the two stage orders are no longer alternatives: **if a rule by temperature is adopted, the scenarios 3 to 6 no longer compare the
orders**, and the order becomes a result, not a scenario. Bypass without reordering keeps the comparison of the orders meaningful. Which is
wanted is a question for the project and the partner. Neither rule is a statement about the plant (equal outlet temperatures, section 7.5).

## 9. Dryer block of the equation document against the fluidised-bed model (feedback to the partner)

Branch `partner-v4-1`, merged into master. Lab book: Befund 75. **Question:** how well does the simplified dryer block (equations 1.3 and 1.5 of the equation document) describe the exhaust and the heat demand of the fluidised bed, and which of its simplifications matters how much?

Code: `src/twinlink_ml/partner_v4_1/dryer_block.py` (the block, generalised so that its properties can be exchanged), `src/twinlink_ml/evaluation/dryer_block_comparison.py` (mapping, step chain, recovery to `T_cold`), tests `tests/test_dryer_block_comparison.py` (13), script `scripts/report_dryer_block_vs_bed.py` (commit `7768b6d`, run on a clean tree, one process), artifact `artifacts/runs/dryer_block_vs_bed.json` (commit `1a4b997`). This is a comparison of two models, no plant data; the bed model is the laboratory grey-box model with its placeholder parameters (`greybox_gab_placeholder.yaml`), in steady state (Befund 53 for the batch caveat). All powers are thermal (no efficiency; the bed model's 0.98 belongs to its electric heater and does not enter).

**Test sets (named).**

* **F** and **M0**: the 25 disturbance scenarios of Befund 52 (ambient temperature x relative humidity), at the fixed operating point F of M1 (74.25 degC, 1.2600 m/s) and at the per-scenario oracle M0 (without margin), both at the **gas-velocity lower bound 1.26 m/s** and the quality limit 0.15 (artifact `m1_margin_cost_l015_umin126`; the set points are read from it; spray 1.5 kg/h, solid fraction 0.3). The 50 steady states were solved again and agree with the stored ones to 0.0 (`reproduction_of_m1_steady_states`). Statements about F and M0 hold "at lower bound 1.26 m/s".
* **G**: 5 x 5 grid over inlet temperature 60-120 degC and gas velocity 1.26-1.88 m/s (the laboratory range), ambient state 20 degC / 0.5, spray 1.5 kg/h, solid fraction 0.3, nozzle 1.0 bar. Five of the 25 points (inlet 120 degC) have an exhaust above 80 degC, the marker above which the bed model's drying curve is outside its documented validity (`max_driving_humidity`; the saturation humidity exceeds its cap of 0.5 kg/kg from about 78 degC on); they are in the statistics.
* **C** (added, not asked): the 25 scenarios of Befund 52 at the corner of G (60 degC, 1.26 m/s), because the oracle sits on this corner wherever the moisture limit does not bind and G shows the corner at the quality limit. 15 of the 25 have a steady state (9.5).

`T_cold` is not in the partner's documents; both values used are placeholders: 29.1 degC (the site model of M2/M3) and 15 degC (the heat-network default, also the ambient of the partner workbook). Dew point and saturation humidity use the project's correlation (Buck) for both models; the workbook's Antoine table is referenced by no formula.

### 9.1 Result in short

1. **The structure of the block is not the problem.** With the bed model's properties, its heat loss and its own product moisture inserted, the block and the bed are the same balance. The remaining difference of the exhaust temperature is -0.017 / -0.012 / -0.021 / -0.012 K (median over F / M0 / G / C), and all of it is the slope of the bed model's evaporation-enthalpy correlation (with that replaced, the difference is at most 4.3e-5 K, the solver's numerical floor). The remaining difference of the exhaust humidity (0.015 / 0.018 / 0.002 / 0.019 g/kg) is the damped fixed-point iteration behind the bed model's *reported* exhaust humidity, which misses the bed model's own mass balance by exactly that amount. Nothing structural is left.
2. **What counts is one property, one input and one missing term.** The exhaust temperature of the block as delivered is **warmer than the bed by 5.5 / 4.2 / 6.5 / 4.2 K** (median; ranges in 9.3). Of that, `cp_gas` = 1.2 instead of 1.006 kJ/(kg K) (with the other gas properties of step b) accounts for -4.5 / -4.4 / -3.9 / -4.4 K, the missing heat loss for -1.9 / -1.0 / -2.8 / -1.5 K, the equilibrium moisture instead of the bed's product moisture for +1.0 / +1.6 / +0.3 / +1.6 K; the other material constants for 0.06-0.17 K (below the 0.2 K noise placeholder of the temperature sensor).
3. **Heating power: +19.1 %** in every set (range 18.7-19.3 %), caused by `cp_gas` alone. The question marked P in sections 2 and 7.3 (1.2 against about 1.0 for air) is the largest single item for the heat demand.
4. **Exhaust humidity** is a pure mass balance; the block is only off by the product moisture it assumes: +0.44 / +0.70 / +0.15 / +0.72 g/kg (2.4 / 4.0 / 0.8 / 4.7 % median). The equilibrium moisture of the partner sheet (2 g/kg) is 3.6 to 21 times below the equilibrium moisture of the bed model's own isotherm at its exhaust states (7-43 g/kg) and far below the product moisture the bed reaches (13-181 g/kg), because the bed does not dry to equilibrium (kinetically limited, saturation degree).
5. **Recoverable heat is the quantity that amplifies the errors**, because it depends on the difference `T_out - T_cold` (median exhaust 4.5-33 K above 29.1 degC in F, M0, G): the block's sensible heat to 29.1 degC is 63 / 118 / 43 / 179 % too high, and the total to 15 degC is 28 / 28 / 27 / 36 % too high. Both the capacity rate (`cp_gas`) and the exhaust temperature enter; `cp_gas` is the largest step. The latent heat to 15 degC is reproduced within 5.5 / 9.5 / 2.2 / 16 % (median relative difference), with the product moisture as the only cause.
6. **Both models react alike to the inlet temperature and the gas velocity** (same sign at 25 of 25 points for all quantities), with the size of the properties' step: the block as delivered overstates the sensitivities of heating power and recovery by 19-31 %, overstates `d(exhaust T)/dT_in` by 11 % and understates `d(exhaust T)/du` by 16 %; with properties and heat loss exchanged (and the product moisture frozen) the ratio is 1.00-1.02 in the median.
7. **The block shows normal operation at every point where the bed model reports a moisture-limit violation**: six of the 90 solved points (one in G, five in C) exceed the quality limit 0.15, by up to 0.031 kg/kg; the critical moisture 0.20 is never exceeded; at a further 10 points of C the bed model has no drying steady state. The block has no product moisture of its own, so where it matters (cold inlet, low gas velocity) it cannot tell.
8. **The partner's fluidised-bed inputs** (300 degC, 12 kg/h spray, 100 kg/h air) give a converged steady state in the bed model, as an extrapolation far outside its range: exhaust 103.5 degC against 134.7 degC of the block; 108.9 degC would remain after the gas properties (step b) alone (9.6).

### 9.2 VG0: mapping, properties, reference states

**Mapping** (block input <- quantity of the bed model; `block_inputs` in `dryer_block_comparison.py`):

| Block input (document) | Bed model | Note |
|---|---|---|
| `M_g`, dry process gas | `dry_air_mass_flow_kg_per_s` = `rho(T_bed, Y_in) u A / (1 + Y_in)`, `A` = 0.0177 m2 | the density is taken **at the bed temperature**, so `M_g` is not independent of the exhaust: for a given velocity it is a result of the steady state. A block fed with a fixed `M_g` has no such coupling |
| `Y_in` | `absolute_humidity(T_amb, phi_amb)` (Buck) | partner: 6 g/kg given |
| `T_g,in` | `inlet_temperature_celsius` (after the heater) | |
| `M_s` | `spray_rate_kg_per_h` / 3600 | 1.5 kg/h in all sets |
| `X_s` | `suspension_solid_fraction` | 0.3 (partner: 0.4) |
| `T_s` | `suspension_temperature_celsius` | 20 degC (partner: 15 degC) |
| `X_p` | `particle_moisture_kg_per_kg` of the steady state (dry basis, as in 1.2) | document: equilibrium moisture `X_EQ`; partner sheet 2 g/kg |
| `M_p` | `X_s M_s` (steady state: the bed's product flow equals the solid inflow) | identical |
| `T_p = T_g,out` | `bed_temperature_celsius` (well-mixed bed, exhaust at bed temperature) | identical; the duct loss of `ExhaustConfig` is 0 |
| heater start, `T_amb` | `ambient_temperature_celsius` | partner: 15 degC |
| atomisation air, dust | neglected in both | |
| heat loss | none in the block; `UA (T_bed - T_amb)` in the bed, `UA` = 2 W/K (laboratory placeholder) | step d |

**Properties** (artifact `vg0`):

| | partner block | bed model |
|---|---|---|
| `cp_gas`, J/(kg K) | 1200 | 1006 |
| `cp_vapour` | 2000 | 1860 |
| `cp_liquid` | 4200 | 4180 |
| `cp_solid` | 1000 | 1200 |
| evaporation enthalpy at 0 degC, kJ/kg | 2500 | 2500.8 |
| evaporation enthalpy at 40 degC, kJ/kg | 2412 (`dh_0 + (cp_v - cp_l) T`) | 2406.5 (`2500.8 - 2.3586 T`) |
| heat capacity of the suspension, J/(kg K) | mixture rule: 3240 (`X_s` = 0.3), 2920 (0.4) | fixed 3500 |
| equilibrium moisture | 2 g/kg (constant) | GAB isotherm, 7-43 g/kg at the exhaust states here |
| heat loss | none | 2 W/K |
| pressure | 1 atm | 101325 Pa |

The bed model's evaporation-enthalpy slope (-2358.6 J/(kg K)) implies `cp_vapour` = 1821 J/(kg K), 2 % below its own 1860; this internal inconsistency is the "rest" of 9.3.

**Reference states.** The block refers all enthalpies to 0 degC with liquid water at zero (equation 1.4: dry gas `cp_g T`, vapour `dh_0 + cp_v T`, liquid `cp_l T`, solid `cp_s T`). The bed model writes its balance in differences to the bed temperature (`greybox._rhs`: the air cools from `T_in` to `T_bed`, the suspension is heated from `T_s` to `T_bed`, evaporation at `dh_v(T_bed)`); `exhaust_sensible_kw` and `exhaust_latent_kw` of the solver refer to the ambient state; the recovery here refers to `T_cold`. The two balances are algebraically the same when `dh_v(T) = dh_0 + (cp_v - cp_l) T` and the suspension heat capacity follows the mixture rule: with `E = M_s (1 - X_s) - M_p X_p`, numerator and denominator of 1.5 are `M_g (cp_g + Y_in cp_v) T_in + M_s cp_susp T_s - E dh_0` and `M_g (cp_g + Y_in cp_v) + M_s cp_susp + E (cp_v - cp_l)`, and the denominator equals `M_g (cp_g + Y_out cp_v) + M_p (cp_s + X_p cp_l)` for the mixture rule. The block of this study is written in that second form so that the bed model's constant suspension heat capacity, its heat loss and its slope can be inserted (`dryer_block.py`); with the partner's defaults it equals the rebuild `partner_v4.corrected.dryer_balance` to rounding and reproduces the exhausts of Model_v4.1 (72.28 degC / 41.95 g/kg and 134.66 degC / 77.90 g/kg), both tested. **The reference state therefore creates no apparent difference as long as `h_fg` of the sheet is read as the value at 0 degC, as equation 1.4 does**; reading 2500 kJ/kg as the evaporation enthalpy *at the bed temperature* would be an error (about 94 kJ/kg, 4 %, at 40 degC, hand calculation from the table above). Both models use the same heater formula `M_g (cp_g + Y_in cp_v)(T_in - T_amb)` (the bed model's heater term is that formula; tested), and the same recovery formula: sensible heat plus latent heat of the condensate `M_g max(0, Y_out - Y_s(T_cold))` down to `T_cold`, each with the properties of the model compared.

### 9.3 VG1/VG2: results per set

Median [min, max] over the points of the set. Columns: the bed model's value; **a** the block as delivered (partner properties, `X_p` = 2 g/kg, no heat loss, inputs from the bed model) *minus* the bed; then the contribution of each step (change of the block's value when the step is applied, in the order b, b2, c, d); **e** the rest, block after b-d minus bed. For every point a + b + b2 + c + d + e = 0 by construction. Steps: **b** `cp_gas`, `cp_vapour`, `dh_0` of the bed model; **b2** (not in the task's list for b, kept apart) `cp_liquid`, `cp_solid` and the fixed suspension heat capacity; **c** the bed's product moisture instead of the equilibrium moisture; **d** the bed's heat loss; **e** the remainder. Recovery rows are to `T_cold` = 29.1 and 15 degC. The columns b-e are the contributions to (block minus bed), i.e. the numbers a step *removes* from a.

Set F (fixed operating point, 25 scenarios, lower bound 1.26 m/s):

| quantity | bed model | a: block as delivered minus bed | b: gas properties | b2: other properties | c: product moisture | d: heat loss | e: rest (structure) |
|---|---|---|---|---|---|---|---|
| exhaust temperature, K | 44 [42.9, 45.5] | 5.5 [3.92, 6.67] | -4.53 [-4.56, -4.44] | -0.115 [-0.115, -0.115] | 0.977 [0.909, 1.43] | -1.86 [-2.94, -0.775] | -0.0171 [-0.0176, -0.0167] |
| exhaust humidity, g/kg | 19.1 [12.7, 42.5] | 0.443 [0.407, 0.674] | 0 [0, 0] | 0 [0, 0] | -0.428 [-0.656, -0.395] | 0 [0, 0] | 0.0149 [0.0107, 0.0181] |
| dew point, K | 24.2 [17.7, 37.6] | 0.382 [0.272, 0.492] | 0 [0, 0] | 0 [0, 0] | -0.369 [-0.479, -0.264] | 0 [0, 0] | 0.0128 [0.00732, 0.014] |
| heating power, kW | 1.36 [0.979, 1.74] | 0.259 [0.183, 0.334] | -0.259 [-0.334, -0.183] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| sensible to 29.1, kW | 0.381 [0.354, 0.42] | 0.24 [0.196, 0.271] | -0.215 [-0.216, -0.209] | -0.00294 [-0.00295, -0.00293] | 0.0247 [0.0229, 0.0361] | -0.0475 [-0.0751, -0.0197] | -0.000428 [-0.000437, -0.000419] |
| latent to 29.1, kW | 0 [0, 0.959] | 0 [0, 0.0402] | 0 [-0.00134, 0] | 0 [0, 0.000239] | 0 [-0.0376, 0] | 0 [0, 0] | 0 [0, 0.00148] |
| total to 29.1, kW | 0.381 [0.354, 1.38] | 0.242 [0.212, 0.271] | -0.215 [-0.216, -0.211] | -0.00294 [-0.00295, -0.00271] | 0.0238 [-0.00146, 0.0269] | -0.0475 [-0.0751, -0.0197] | -0.000424 [-0.000437, 0.00104] |
| sensible to 15, kW | 0.741 [0.715, 0.78] | 0.308 [0.263, 0.34] | -0.283 [-0.285, -0.276] | -0.00294 [-0.00295, -0.00293] | 0.0244 [0.0227, 0.0357] | -0.0475 [-0.0751, -0.0197] | -0.000418 [-0.000428, -0.00041] |
| latent to 15, kW | 0.51 [0.123, 1.85] | 0.0272 [0.025, 0.0403] | -0.000283 [-0.000996, -7.82e-05] | 6.54e-05 [1.81e-05, 0.00023] | -0.026 [-0.0381, -0.0242] | 0 [0, 0] | 0.00103 [0.000687, 0.00148] |
| total to 15, kW | 1.24 [0.838, 2.63] | 0.335 [0.304, 0.365] | -0.283 [-0.285, -0.277] | -0.00288 [-0.00293, -0.00271] | -0.0016 [-0.00237, -0.00148] | -0.0475 [-0.0751, -0.0197] | 0.000604 [0.000272, 0.00106] |

Set M0 (oracle without margin, 25 scenarios, lower bound 1.26 m/s):

| quantity | bed model | a: block as delivered minus bed | b: gas properties | b2: other properties | c: product moisture | d: heat loss | e: rest (structure) |
|---|---|---|---|---|---|---|---|
| exhaust temperature, K | 33.6 [31.6, 45] | 4.18 [3.52, 4.89] | -4.36 [-4.54, -4.34] | -0.0615 [-0.115, -0.0561] | 1.57 [0.956, 1.7] | -1.02 [-2, -0.757] | -0.0123 [-0.0176, -0.0116] |
| exhaust humidity, g/kg | 18.4 [12, 42.5] | 0.698 [0.436, 0.777] | 0 [0, 0] | 0 [0, 0] | -0.68 [-0.758, -0.421] | 0 [0, 0] | 0.0183 [0.0149, 0.0194] |
| dew point, K | 23.6 [16.8, 37.6] | 0.634 [0.282, 0.829] | 0 [0, 0] | 0 [0, 0] | -0.617 [-0.807, -0.274] | 0 [0, 0] | 0.0164 [0.00745, 0.022] |
| heating power, kW | 1.06 [0.963, 1.43] | 0.203 [0.18, 0.275] | -0.203 [-0.275, -0.18] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| sensible to 29.1, kW | 0.119 [0.0673, 0.407] | 0.16 [0.135, 0.212] | -0.158 [-0.214, -0.152] | -0.00162 [-0.00293, -0.00148] | 0.0414 [0.024, 0.0442] | -0.0268 [-0.0529, -0.0197] | -0.000321 [-0.000438, -0.000306] |
| latent to 29.1, kW | 0 [0, 0.958] | 0 [0, 0.0467] | 0 [-0.00134, 0] | 0 [0, 0.000239] | 0 [-0.045, 0] | 0 [0, 0] | 0 [0, 0.00149] |
| total to 29.1, kW | 0.119 [0.0673, 1.36] | 0.163 [0.135, 0.235] | -0.158 [-0.214, -0.152] | -0.00162 [-0.00293, -0.00148] | 0.0405 [-0.00148, 0.0442] | -0.0268 [-0.0529, -0.0197] | -0.000311 [-0.000438, 0.00107] |
| sensible to 15, kW | 0.49 [0.441, 0.767] | 0.23 [0.206, 0.28] | -0.228 [-0.281, -0.223] | -0.00162 [-0.00293, -0.00148] | 0.0409 [0.0238, 0.0437] | -0.0268 [-0.0529, -0.0197] | -0.000309 [-0.000428, -0.000293] |
| latent to 15, kW | 0.485 [0.0854, 1.85] | 0.044 [0.0266, 0.0476] | -0.00028 [-0.000997, -6.71e-05] | 6.47e-05 [1.55e-05, 0.00023] | -0.0426 [-0.0456, -0.0253] | 0 [0, 0] | 0.00125 [0.00103, 0.0015] |
| total to 15, kW | 0.97 [0.526, 2.62] | 0.274 [0.252, 0.306] | -0.229 [-0.282, -0.223] | -0.00156 [-0.00286, -0.00143] | -0.00183 [-0.00243, -0.00155] | -0.0268 [-0.0529, -0.0197] | 0.000949 [0.000604, 0.00109] |

Set G (grid inlet temperature x gas velocity, ambient 20 degC / 0.5):

| quantity | bed model | a: block as delivered minus bed | b: gas properties | b2: other properties | c: product moisture | d: heat loss | e: rest (structure) |
|---|---|---|---|---|---|---|---|
| exhaust temperature, K | 62.1 [32.8, 91.9] | 6.53 [3.54, 10.5] | -3.91 [-5.18, -3.02] | -0.173 [-0.33, -0.0561] | 0.334 [0.0961, 1.71] | -2.78 [-5.24, -0.957] | -0.0214 [-0.0358, -0.0108] |
| exhaust humidity, g/kg | 17.1 [14.9, 20.4] | 0.152 [0.0441, 0.762] | 0 [0, 0] | 0 [0, 0] | -0.146 [-0.742, -0.0441] | 0 [0, 0] | 0.00231 [-1.37e-05, 0.0192] |
| dew point, K | 22.4 [20.2, 25.3] | 0.132 [0.0428, 0.669] | 0 [0, 0] | 0 [0, 0] | -0.13 [-0.652, -0.0428] | 0 [0, 0] | 0.00215 [-1.23e-05, 0.0172] |
| heating power, kW | 2 [1.04, 3.24] | 0.382 [0.198, 0.62] | -0.382 [-0.62, -0.198] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| sensible to 29.1, kW | 0.941 [0.0983, 2.07] | 0.421 [0.133, 0.71] | -0.335 [-0.562, -0.152] | -0.00518 [-0.00877, -0.00149] | 0.00916 [0.00299, 0.0452] | -0.0835 [-0.143, -0.0253] | -0.000637 [-0.000964, -0.000314] |
| latent to 29.1, kW | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| total to 29.1, kW | 0.941 [0.0983, 2.07] | 0.421 [0.133, 0.71] | -0.335 [-0.562, -0.152] | -0.00518 [-0.00877, -0.00149] | 0.00916 [0.00299, 0.0452] | -0.0835 [-0.143, -0.0253] | -0.000637 [-0.000964, -0.000314] |
| sensible to 15, kW | 1.4 [0.471, 2.53] | 0.492 [0.204, 0.798] | -0.416 [-0.65, -0.222] | -0.00518 [-0.00877, -0.00149] | 0.00905 [0.00296, 0.0447] | -0.0835 [-0.143, -0.0253] | -0.000635 [-0.000964, -0.000301] |
| latent to 15, kW | 0.46 [0.385, 0.53] | 0.0104 [0.00375, 0.0481] | -0.000252 [-0.000282, -0.000216] | 5.81e-05 [4.98e-05, 6.51e-05] | -0.00997 [-0.0466, -0.00346] | 0 [0, 0] | 0.000274 [0.000106, 0.00131] |
| total to 15, kW | 1.81 [0.929, 2.98] | 0.502 [0.252, 0.802] | -0.416 [-0.651, -0.223] | -0.00512 [-0.00872, -0.00142] | -0.000939 [-0.00189, -0.000499] | -0.0835 [-0.143, -0.0253] | -0.000361 [-0.000857, 0.00101] |

Set C (extra: 25 scenarios at 60 degC / 1.26 m/s; 15 with a steady state):

| quantity | bed model | a: block as delivered minus bed | b: gas properties | b2: other properties | c: product moisture | d: heat loss | e: rest (structure) |
|---|---|---|---|---|---|---|---|
| exhaust temperature, K | 32.3 [31.6, 33.1] | 4.24 [3.32, 4.89] | -4.35 [-4.36, -4.32] | -0.0562 [-0.0563, -0.0561] | 1.63 [1.5, 2.02] | -1.47 [-2, -0.945] | -0.0118 [-0.012, -0.0116] |
| exhaust humidity, g/kg | 15.2 [12, 23.1] | 0.723 [0.659, 0.909] | 0 [0, 0] | 0 [0, 0] | -0.704 [-0.886, -0.642] | 0 [0, 0] | 0.0186 [0.017, 0.0223] |
| dew point, K | 20.6 [16.8, 27.3] | 0.73 [0.636, 0.829] | 0 [0, 0] | 0 [0, 0] | -0.711 [-0.807, -0.62] | 0 [0, 0] | 0.0194 [0.0158, 0.022] |
| heating power, kW | 1.23 [1.04, 1.43] | 0.236 [0.197, 0.275] | -0.236 [-0.275, -0.197] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| sensible to 29.1, kW | 0.0835 [0.0673, 0.106] | 0.149 [0.125, 0.167] | -0.152 [-0.153, -0.151] | -0.00149 [-0.00149, -0.00148] | 0.0431 [0.0396, 0.0534] | -0.039 [-0.0529, -0.025] | -0.00031 [-0.000315, -0.000306] |
| latent to 29.1, kW | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| total to 29.1, kW | 0.0835 [0.0673, 0.106] | 0.149 [0.125, 0.167] | -0.152 [-0.153, -0.151] | -0.00149 [-0.00149, -0.00148] | 0.0431 [0.0396, 0.0534] | -0.039 [-0.0529, -0.025] | -0.00031 [-0.000315, -0.000306] |
| sensible to 15, kW | 0.457 [0.441, 0.479] | 0.221 [0.195, 0.239] | -0.223 [-0.224, -0.221] | -0.00149 [-0.00149, -0.00148] | 0.0426 [0.0392, 0.0528] | -0.039 [-0.0529, -0.025] | -0.000297 [-0.000303, -0.000293] |
| latent to 15, kW | 0.289 [0.0854, 0.774] | 0.0458 [0.042, 0.057] | -0.000177 [-0.000438, -6.71e-05] | 4.08e-05 [1.55e-05, 0.000101] | -0.0444 [-0.0551, -0.0408] | 0 [0, 0] | 0.00125 [0.0011, 0.00156] |
| total to 15, kW | 0.734 [0.526, 1.25] | 0.267 [0.252, 0.281] | -0.223 [-0.224, -0.221] | -0.00145 [-0.00148, -0.00138] | -0.0018 [-0.00227, -0.00165] | -0.039 [-0.0529, -0.025] | 0.000949 [0.000808, 0.00127] |

Product moisture of the bed (the input of step c): F 86 g/kg median (80-126), M0 140 (84-150), G 34 (13-153), C 146 (134-181); the block as delivered has 2 g/kg.

### 9.4 Reading of the decomposition, order, resolution

* **Order.** The decomposition is sequential; the contributions depend on the order. The reversed order (d, c, b2, b) gives (medians, forward / reverse; the rest is the same in both orders):

| set | quantity | b | b2 | c | d | rest, forward order | rest, reverse order |
|---|---|---|---|---|---|---|---|
| F | exhaust temperature, K | -4.53 / -4.38 | -0.115 / -0.103 | 0.977 / 0.775 | -1.86 / -1.82 | -0.0171 | -0.0171 |
| F | heating power, kW | -0.259 / -0.259 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 |
| F | total to 29.1, kW | -0.215 / -0.205 | -0.00294 / -0.0031 | 0.0238 / 0.0223 | -0.0475 / -0.0552 | -0.000424 | -0.000424 |
| M0 | exhaust temperature, K | -4.36 / -4.07 | -0.0615 / -0.0628 | 1.57 / 1.24 | -1.02 / -1.03 | -0.0123 | -0.0123 |
| M0 | heating power, kW | -0.203 / -0.203 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 |
| M0 | total to 29.1, kW | -0.158 / -0.149 | -0.00162 / -0.00197 | 0.0405 / 0.0381 | -0.0268 / -0.0325 | -0.000311 | -0.000311 |
| G | exhaust temperature, K | -3.91 / -4.09 | -0.173 / -0.142 | 0.334 / 0.27 | -2.78 / -2.57 | -0.0214 | -0.0214 |
| G | heating power, kW | -0.382 / -0.382 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 |
| G | total to 29.1, kW | -0.335 / -0.324 | -0.00518 / -0.00508 | 0.00916 / 0.00868 | -0.0835 / -0.0918 | -0.000637 | -0.000637 |
| C | exhaust temperature, K | -4.35 / -4.06 | -0.0562 / -0.0583 | 1.63 / 1.3 | -1.47 / -1.43 | -0.0118 | -0.0118 |
| C | heating power, kW | -0.236 / -0.236 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 |
| C | total to 29.1, kW | -0.152 / -0.144 | -0.00149 / -0.00184 | 0.0431 / 0.0406 | -0.039 / -0.0449 | -0.00031 | -0.00031 |

  The ranking of the temperature steps is stable: `cp_gas` (b) is the largest in every set and both orders (3.9-4.5 K); heat loss (d, 1.0-2.8 K) and product moisture (c, 0.3-1.6 K) follow, and their rank depends on the set (d > c in F and G, c > d in M0 and C); b2 is below 0.2 K in F, M0 and C. The size of c is known only to about 0.3 K because of the order (M0: 1.57 against 1.24 K), that of b likewise.
* **Resolution limit.** Temperature steps are ranked only above 0.2 K (the noise standard deviation of the temperature sensor in the config, `measurement.absolute_noise_std_temperature_k`, a placeholder); the numerical floor of the solver at the search precision (30 steps, 1e-7, the precision of Befund 52) is 4.3e-5 K (what is left after the slope is replaced). By that: b, d and c are above the sensor resolution; b2 is below it (median 0.06-0.17 K; the maximum in G, 0.33 K, is above); the rest (0.012-0.021 K) is below the sensor resolution, though well above the numerical floor.
* **The rest has two named causes** (tested): the latent-heat slope of the bed model (temperature; replaced in the artifact's `after_slope_check`: difference at most 4.3e-5 K) and the lag of the damped fixed-point iteration in the reported humidity (`own_outlet_humidity_minus_mass_closure`: -0.0149 g/kg in F, equal and opposite to the rest). The rest in the recovery rows (up to 0.001 kW) comes with them.
* **The heat loss is a laboratory placeholder** (`UA` = 2 W/K, a TODO in the config). Step d is the part of the result that does not transfer to another plant size: it scales with the plant, and the block has no counterpart. At a laboratory flow of about 0.025 kg/s (25 W/K of capacity rate) it is 1-3 K; for 100 kg/h (29 W/K, 9.6) the same 2 W/K would be a statement about the laboratory bed only.
* **Sensible recovery to 29.1 degC is small to begin with** (0.12-0.94 kW median in F, M0 and G; exhaust only 4.5-33 K above 29.1 degC at the median), so its relative error is large. The latent part at 29.1 degC is non-zero only where the exhaust dew point exceeds it (humid ambient states in F and M0; none in G and C).

### 9.5 VG3: sensitivities and the moisture limit

**Sensitivities** (set G; one-sided differences with 1 K and 0.02 m/s, backward at 120 degC, the same stencil for both models; at each shifted point the block gets the inputs of the bed model's steady state there, so `M_g` follows the bed temperature; ratios block / bed, median [min, max]). "Frozen": steps b-d applied and the product moisture held at its value at the base point, which separates the effect of properties and heat loss from the effect of the bed's moisture response.

| derivative | bed model | block as delivered / bed | block with b-d, X_p frozen / bed |
|---|---|---|---|
| d(exhaust T) / dT_in (per K), K | 0.838 [0.771, 0.863] | 1.11 [1.09, 1.19] (same sign 25/25) | 1.01 [1, 1.09] (same sign 25/25) |
| d(heating power) / dT_in (per K), kW | 0.0243 [0.0173, 0.0335] | 1.19 [1.19, 1.19] (same sign 25/25) | 1 [1, 1] (same sign 25/25) |
| d(recovery to 29.1) / dT_in (per K), kW | 0.0227 [0.0159, 0.0316] | 1.31 [1.28, 1.41] (same sign 25/25) | 1.01 [1, 1.09] (same sign 25/25) |
| d(recovery to 15) / dT_in (per K), kW | 0.0227 [0.0157, 0.032] | 1.28 [1.25, 1.32] (same sign 25/25) | 0.996 [0.988, 0.999] (same sign 25/25) |
| d(exhaust humidity) / dT_in (per K), g/kg | 0.0311 [0.0221, 0.0595] | 0.84 [0.486, 0.977] (same sign 25/25) | 0.828 [0.454, 0.973] (same sign 25/25) |
| d(exhaust T) / du (per m/s), K | 14.6 [8.76, 24.7] | 0.84 [0.762, 1.08] (same sign 25/25) | 1.02 [1.01, 1.15] (same sign 25/25) |
| d(heating power) / du (per m/s), kW | 1.22 [0.761, 1.62] | 1.19 [1.19, 1.19] (same sign 25/25) | 1 [1, 1] (same sign 25/25) |
| d(recovery to 29.1) / du (per m/s), kW | 1.02 [0.491, 1.43] | 1.24 [1.22, 1.44] (same sign 25/25) | 1.01 [1, 1.13] (same sign 25/25) |
| d(recovery to 15) / du (per m/s), kW | 1.14 [0.675, 1.53] | 1.25 [1.23, 1.29] (same sign 25/25) | 0.998 [0.987, 1] (same sign 25/25) |
| d(exhaust humidity) / du (per m/s), g/kg | -5.67 [-9.29, -3.58] | 1.04 [1.01, 1.25] (same sign 25/25) | 1.03 [1.01, 1.17] (same sign 25/25) |

Reading: all derivatives have the sign of the bed model at 25 of 25 points. The block as delivered is too steep in the heating power and the recovery by the factor of `cp_gas` (1.19) and, for the recovery, by a further 4-10 % through the exhaust temperature; its `d(exhaust T)/dT_in` is 11 % too high and `d(exhaust T)/du` 16 % too low. With properties and heat loss exchanged and the moisture frozen the block agrees to 1-2 % in the median; the largest deviations (9 % and 15 % for the exhaust temperature) are at the point of highest product moisture (60 degC, 1.26 m/s) and its neighbours, where the bed's moisture response is not small. The humidity derivative with respect to the inlet temperature is 16-17 % smaller in the block, also when frozen (up to 55 % smaller at some points): that is the share the bed's product-moisture response contributes (the block with frozen moisture has none). This was not tested as a statement about a plant.

**Moisture limit.** The block has no moisture limit and no flag; all it can show is an impossible exhaust (negative evaporation, supersaturation, dew point at or above the exhaust temperature). At all 90 solved points of F, M0, G and C it shows normal operation. The bed model reports (tolerance 1e-4 kg/kg, as in Befund 52):

| Set | Bed model reports | Block |
|---|---|---|
| F, M0 | no violation of the quality limit 0.15 or the critical moisture 0.20 (the largest product moisture of M0 is 0.14999996, at the limit by construction) | normal |
| G | **(60 degC, 1.26 m/s)**: product moisture 0.1530, 0.0030 above the quality limit 0.15; none above 0.20 | normal |
| C | **5 of 15**: 12.5 degC / 0.70 (0.1523), 12.5 / 0.85 (0.1589), 20 / 0.55 (0.1565), 20 / 0.70 (0.1679), 20 / 0.85 (0.1807); none above 0.20 | normal at all 15 |
| C | **10 of 25: no steady state** (ambient 27.5 and 35 degC, every humidity): the energy balance is negative at both ends of the solver's bracket `[T_amb + 10, ...]`, i.e. a steady state, if there is one, lies below `T_amb + 10`, the regime the solver excludes by design (`steady_state.py`) | not evaluated (its input `M_g` needs the bed temperature of the solution) |

So the points where the bed model reports a violated moisture limit while the block shows normal operation are six (one in G, five in C), all at the cold end of the laboratory range (inlet 60 degC, gas velocity 1.26 m/s), in G at ambient 20 degC / 0.5 and in C at ambient 12.5-20 degC with relative humidity 0.55 or more. The quality limit 0.15 is the one exceeded; the critical moisture 0.20 is not. These are the points that the oracle avoids and that a soft sensor has to detect.

### 9.6 VG4: the partner's fluidised-bed inputs in the bed model

Inputs of dryer 2 of Model_v4.1: inlet 300 degC, spray 12 kg/h (solid fraction 0.4, 15 degC), dry air 100 kg/h at 6 g/kg and 15 degC (ambient set to 15 degC and relative humidity 0.568 for that humidity), 1 atm. **Extrapolation: every one of these is outside the bed model's configured range** (inlet 60-120 degC, spray 0.4-3.0 kg/h; the bounds were widened for this run only, the config default is untouched). The solver was run at six gas velocities (1.0-2.5 m/s) and **converges at all of them**; the velocity that gives 100 kg/h of dry air is 1.6907 m/s (root of the dry-air flow). Result (artifact `vg4`): bed and exhaust temperature **103.5 degC**, exhaust humidity 75.2 g/kg, product moisture 0.0566 kg/kg (not at equilibrium), heating power 8.05 kW (thermal), solver residual 3.9e-9, feasibility flag true. The bed temperature lies above the 80 degC validity marker of the drying curve and above 100 degC (the saturation humidity is at its cap; the correlations are stated for 0-100 degC): the number is what the model computes outside its range, not a prediction.

Block with the same inputs (the partner's, equal to the document model): 134.66 degC, 77.90 g/kg, heating power 9.595 kW. The step chain applied to these inputs (forward order): 134.66 -> **108.89** degC (b, the gas properties) -> 103.62 (b2) -> 108.26 (c: the bed's 0.0566 instead of 0.002; exhaust humidity 77.90 -> 75.28 g/kg) -> 103.28 degC (d) against 103.49 degC of the bed model. Of the 31.2 K between the block as delivered and the bed, the gas properties are 25.8 K; the heat loss step (-5.0 K) uses the laboratory bed's 2 W/K for a 100 kg/h flow and is not transferable. The heating power differs by +19.1 % (`cp_gas`), as at every other point.

### 9.7 What this means for the Dyssol model (for the partner)

1. **`cp_gas` first.** It is the one simplification that matters for the heat demand (+19 % at every operating point) and for the exhaust temperature (4-5 K at laboratory flows, 26 K at the partner's inlet temperature of 300 degC), and through the exhaust temperature for every recovered heat. Whether 1.2 is intended (a value for moist air, a rule of thumb, another gas) is a question to the partner; for air the value is about 1.0 kJ/(kg K) (the bed model uses 1.006). It has to be the same in the dryer, the heater and the recovery, as it is in the workbook.
2. **The product moisture is an input of the dryer, not an equilibrium.** The equilibrium assumption costs little in the exhaust (0.15-0.7 g/kg of 15-19 g/kg at laboratory flows; 2.6 g/kg of 77.9 g/kg, 3.4 %, and 4.6 K at the partner's inlet if the product keeps 0.057), but the bed does not dry to equilibrium, and the block cannot report a product above its moisture limit. A Dyssol unit that computes the product moisture from the drying kinetics should use that value in the exhaust balance (the block takes `X_p` as an input for exactly this); the limit can then be checked there.
3. **Heat loss.** The block has none; the bed model's 2 W/K lowers the laboratory exhaust by 1-3 K (median). For the partner's plant it must come from the plant (insulation, surface), not from the laboratory value.
4. **The balance itself needs no change.** Equations 1.3 and 1.5 are the bed model's steady-state balance; a Dyssol unit that closes the same mass and enthalpy balance with the same properties, heat loss and product moisture gives the same exhaust (what is left in this study is below the sensor resolution). The simplicity of the block does not cost accuracy; what it costs is in the three items above.
5. **Recovery to `T_cold` amplifies everything upstream**: an exhaust 4-6 K too warm gives a sensible heat that is 40-180 % too high where the exhaust is 5-35 K above `T_cold`. `T_cold` and the dew-point approach are placeholders here and in the workbook. The latent part (condensate below the dew point) is robust against the block's simplifications (2-16 % relative in the median) and needs the exhaust humidity, not the temperature.
6. **The dry-air flow of the bed model depends on the bed temperature** (the density at `T_bed`): for a given gas velocity, `M_g` shifts with the operating point. If Dyssol takes the flow from a velocity and a cross section, the density must be taken at the same temperature, or the flow must be given as a mass flow, as the block does.

### 9.8 Caveats

* Model against model, no plant data. The bed model is a laboratory grey-box model with placeholder parameters (product moisture kinetics, `UA`, `cp` values, the isotherm are placeholders or laboratory assumptions); none of the absolute numbers is a plant statement.
* F, M0 and C hold "at lower bound 1.26 m/s" (the oracle's set points are on that bound, Befund 69); G covers the laboratory range; section 9.6 is an extrapolation, reported because the solver converges, not because the model is valid there.
* `T_cold` (29.1 and 15 degC) is a placeholder; the recovery rows depend on it strongly through `T_out - T_cold`.
* The decomposition is order-dependent (9.4) and step b2 is an addition to the task's b; the figures are medians over 25 points (15 in C) of one model pair, with the ranges given, not a confidence statement.
* Not done: a comparison with the partner's Dyssol model itself (not available), a plant-scale heat loss, a sensitivity to the bed model's `cp` and isotherm placeholders, and the case that the partner's `cp_gas` 1.2 is correct for their gas.


## 10. Heat recovery of the pre-heating stages: partner model v4.1 against an independent calculation (feedback to the partner)

> **Corrected 2026-10-15 (see Befund 90, addendum, `docs/04_stand_ap1.md`).** The first version of this section (artifact `heat_recovery_comparison_r2.json`) used the repository's gas properties in the exchangers of the independent calculation (b, c, d) but took the exhausts of the dryers from the partner's model, which is computed with `cp_gas` 1.2 kJ/(kg K): two property sets in one calculation. All figures of b, c, d below are recomputed with **one** property set (the exhausts of the dryers are computed with the repository's properties as well; artifact `heat_recovery_comparison_r4_consistent.json`). Column a (the partner model) is unchanged. What changed: the recovery of b, c, d is lower (the exhausts are colder: 58.9 and 108.9 degC instead of 72.3 and 134.7 degC in variant (i)); what did not: the ranking of the steps of the difference except for `cp_gas`, the position of the optimum flow of the medium. Superseded statements are marked "was" in the text; the old numbers stay in the old artifact.

Branch `waermerueckgewinnung-vergleich` (merged 2026-10-15). Lab book: Befund 90 (and its addendum). **Question:** how much heat do the pre-heating stages of the partner model recover, compared with an independent calculation with a realistic exchanger model, and which assumption makes the difference?

Code: `scripts/report_heat_recovery_comparison.py` (commit `d3cba2c`; option `--consistent-properties`, commit `d59b6bf`), tests `tests/test_heat_recovery_comparison.py` (41), artifacts `artifacts/runs/heat_recovery_comparison_r4_consistent.json` (run on a clean tree, `git/dirty=False`; this section) and `heat_recovery_comparison_r2.json` (first version, mixed property set, kept as the record). No plant data: the partner model is the equation document of section 7 (circuit flows 20 kg/h each unless stated), the independent calculation is a model, not a measurement. The gas flow is always computed with the corrected formula `m_gas = C (T_target - T_in) / (eta H_u)`, `eta` 0.8, `H_u` 45 MJ/kg of the workbook (the sheet's own gas rows are 108.91 times too high, section 7.3).

### 10.1 Answer in short

1. **At the sheet's circuit flow of 20 kg/h the partner model recovers 60 to 88 % of what an independent counter-current calculation recovers** (was 46 to 65 %): scenario 1 1.78 kW against 2.28 / 2.75 / 2.98 kW (`dT_min` 20 / 10 / 5 K through the circuit; was 3.20 / 3.66 / 3.90), scenario 3 1.42 against 1.73 / 1.96 / 2.08 kW (was 2.33 / 2.56 / 2.68), scenario 5 1.61 against 1.82 / 2.10 / 2.23 kW (was 2.48 / 2.75 / 2.89), i.e. 78 / 65 / 60 %, 82 / 73 / 68 % and 88 / 77 / 72 % (10.3). With the main convention of section 11 (`dT_min` per exchanger, 10 K, equal to 20 K through the circuit) the partner model recovers 78 / 82 / 88 %.
2. **The flow arrangement is still the largest single step, but `cp_gas` is now the second.** Equal outlet temperatures of both fluids (equations 2.3/2.4, 4.1/4.2) is parallel flow with unlimited area; counter-current with the same capacity rates and no approach temperature gives +1.97 kW in scenario 1 (+1.21 kW in scenario 3) when the properties are exchanged first, +2.35 kW (+1.37 kW) when they are exchanged last (was +2.52 / +1.53 kW, the figure of one order). The other steps: the minimum approach takes back 0.23 (5 K), 0.47 (10 K) or 0.93 kW (20 K) in scenario 1; **`cp_gas` 1.2 against 1.006 kJ/(kg K) now lowers the recovery by 0.44 to 0.53 kW (properties first) or 0.91 kW in scenario 1 (properties last)** (was 0 to 0.17 kW), because it also lowers the exhaust temperatures of the dryers, and it lowers the gas flow by 0.17 to 0.18 kg/h; condensation 0 at 20 K and +0.34 / +0.11 / +0.04 kW at 10 K (scenarios 1 / 3 / 5; was 0.14 / 0 / 0.03) (10.5).
3. **The partner model is far from the upper limit but violates nothing at 20 kg/h.** Energy balances close (4.5e-13 W; 2.7e-15 relative on Excel's own cells), no stage runs against the temperature gradient or cools the gas, the total is 29-40 % of the pinch target (10.2; unchanged, the partner model is column a). One finding: the heat transfer medium leaves the recovery exchanger at 102.6 degC (scenarios 1-4) and 107.1 degC (5, 6), above the boiling point of water at 1 atm, while the sheet takes `cp` = 4.2 kJ/(kg K). **The counter-current circuit is much less hot than first reported:** 103.9 degC at 10 K and 106.4 degC at 5 K in the stage with exhaust 2 (was 129.7 and 132.2 degC), and at 20 K no stage exceeds 100 degC (was 8 of 12 flagged at every approach).
4. **The medium flow of 20 kg/h limits both models, but in opposite ways, and the optimum is where it was.** In the partner model more flow is always better (the optimum is on the edge of the range, 150 kg/h: gas 1.2363 -> 1.1306 kg/h in scenario 1). With counter-current exchangers the optimum is still about 25 kg/h per circuit (the capacity rates of the medium, the gas and the exhausts do not depend on the exhaust temperature, so the position of the optimum does not move); at 150 kg/h the gas flow is 17 % higher than at the optimum (0.9950 against 0.8536 kg/h, scenario 1, 10 K; was 25 %: 0.9316 against 0.7433) (10.6).

### 10.2 WU0 and WU0b: mapping, rebuild of the circuit, consistency of the partner model

**Mapping.** Source *i* = exhaust of dryer *i* (100 kg/h dry air; dryer 1 72.28 degC, 41.95 g/kg, dryer 2 134.66 degC, 77.90 g/kg; source variant (i)), circuit *i* = heat transfer medium of that exhaust (20 kg/h), sink *j* = inlet air of dryer *j* (100 kg/h, 6 g/kg, 15 degC to 150 degC for sink 1 and 300 degC for sink 2). The six scenarios are the case drawings of section 7.4 (stage list per scenario in the artifact, `wu0.mapping`); scenarios 2, 4, 6 give the same numbers as 1, 3, 5 in every calculation of this section (the dryers have equal air flow and humidity). In the heat-network module (`heatnetwork`) a source is a `HeatSourceConfig`, a sink a `HeatSinkConfig`, a stage a pair `(source, sink)` in the stage list.

**The module cannot represent the intermediate circuit** (`wu0.module_capability`): a stage is a direct source -> sink pair, `HeatNetworkConfig` has no field for a medium or its flow, the capacity rates are those of moist air, and the temperature level of the medium is a result, not a value. The circuit is therefore rebuilt in the script (two counter-current exchangers in series over a medium of constant `B = m cp`; the medium returns to the recovery exchanger at `T1`, leaves it at `T2 = T1 + Q/B`; the largest duty with the approach held along both exchangers, found by bisection on a condition that is monotone in `Q`); `src/` is unchanged. **Checks of the rebuild:** its direct exchange (no circuit) equals `evaluate_cascade` (approach along the exchanger, both recovery modes, 20 / 10 / 5 / 0.0001 K, all scenarios) to 3.8e-13 kW and 5.4e-14 kg/h; its circuit at `dT_min` 0 equals the counter-current closure of `partner_v4_1.loops` to 2.5e-10 W (flows 5-600 kg/h, gas inlet 15-80 degC); its equal-outlet stage equals `loops.equal_outlet_loop` and the document model of section 7.4 (1.4140, 1.2363, 1.2720, 1.2534 kg/h) to 1e-12. The condensing release curve of the module (latent heat at `T_kalt`) differs from a complete enthalpy balance of moist air by at most 0.9 % (`profile_convention`); 400 nodes against 4000 changes no recovery (0.0 kW).

**Convention (a decision, to be confirmed).** `dT_min` is the minimum approach between exhaust and process gas *through the circuit*, the convention of the module, whose 20 K stands for two exchangers in series (`DEFAULT_MINIMUM_PINCH_K`): each of the two exchangers gets `dT_min / 2`. The reading "`dT_min` per exchanger" (chain effective `2 dT_min`) is in the artifact (`*_per_exchanger`); it shifts every counter-current number one step along 5 -> 10 -> 20 K (scenario 1, condensing: 2.284 kW at 10 K per exchanger, the value of 20 K through the circuit). Dew-point margin 3 K (module default) in the no-condensation limit, `T_kalt` 15 degC (not binding in the circuit: its cold end is at least `15 + dT_min/2`); both are placeholders.

**WU0b, consistency of the partner model** (`wu0b`; the stage values of the rebuild equal Excel's stored values to 1e-9, section 7.3):

| Check | Result at 20 kg/h (12 stages) |
|---|---|
| energy balance per exchanger and circuit, `A (T_c - T2) = B (T2 - T1) = C (T1 - T_h)` | closed: largest residual 4.5e-13 W; on Excel's own cells (`Scenarios` stage energies against both sides, recomputed from the sheet's temperatures) 2.7e-15 relative |
| second law per stage (sign of the heat, entropy production of each exchanger) | no violation; smallest entropy production 0.014 W/K |
| a stage cools the process gas (gas enters warmer than the exhaust) | none |
| exhaust leaves below its dew point (condensation neglected by 4.1) | none |
| total recovery against the pinch target of the data case (`dT_min` 0, condensing, `T_kalt` 15 degC, partner properties) | scenarios 1/2: 1.777 of 6.206 kW (28.6 %); 3/4: 1.420 of 4.029 (35.3 %); 5/6: 1.606 of 4.029 (39.9 %); no violation |
| heater gas rows 26, 39, 41 of the sheet | **violated** (known, section 7.3): sheet 134.65 / 138.53 / 136.51 kg/h against 1.2363 / 1.2720 / 1.2534, factor 108.911 in every scenario |
| medium above 100 degC at 1 atm | **6 stages**: scenarios 1-4 stage with exhaust 2, 102.61 degC; scenarios 5, 6, 107.14 degC |

Two further runs. At **150 kg/h per circuit** the stage that takes exhaust 1 behind exhaust 2 (scenarios 3, 4) cools the gas, -0.0012 kW (gas 72.36 degC against exhaust 72.28 degC; the entropy production stays positive, so this is a stage that works against its purpose, not a violation of the second law; consistent with the 147.8 kg/h of section 7.5); no other check fails. **Positive control:** the checks applied to the intermediate version (inputs of section 8) find the known cooling stage, scenario 6 stage 2, -0.0563 kW, and the medium above 100 degC at 105.08, 110.39 and 105.08 degC.

### 10.3 WU1: recovered heat and gas per scenario

Source variants. **(i)** exhaust states of the partner model for both dryers. **(ii)** dryer 2 from the laboratory grey-box model at the partner's inputs (300 degC, 12 kg/h spray, 100 kg/h air of 6 g/kg at 15 degC, **solid fraction 0.40 as a run parameter**, the default of the process configuration is untouched; gas velocity 1.6907 m/s for 100 kg/h): exhaust 103.49 degC, 75.23 g/kg, dew point 47.6 degC, product moisture 0.0566 kg/kg. **This is an extrapolation** (every input is outside the model's range, Befund 75, 9.6; the state equals the one stored there). Columns: **a** partner model as delivered (partner `cp`, the partner's exhausts), **b** counter-current circuit, approach along both exchangers, no condensation, **c** as b with condensation below the dew point, **d** pinch target of the data case (any wiring of the scenario, condensing, repository properties), at `dT_min` through the circuit. **b, c, d use the repository's gas properties (1.006 and 1.86 kJ/(kg K)) in the exchangers and in the dryers (variant (i): exhausts of the dryer block with these properties, 58.94 degC / 41.95 g/kg and 108.93 degC / 77.90 g/kg; variant (ii): dryer 1 58.94 degC, dryer 2 from the laboratory model for all columns)**; a keeps the partner's. In variant (ii) column a combines the partner's properties in the exchangers with the laboratory-model exhaust of dryer 2, so only variant (i) compares the partner model with the independent calculation without a second mixture. The no-recovery gas flow is 1.4140 kg/h with the partner's properties and 1.1867 kg/h with the repository's.

__Recovered heat__, kW (circuits 20 kg/h; scenarios 2, 4, 6 equal 1, 3, 5):

| Source | Sc. | a | b 20 K | b 10 K | b 5 K | c 20 K | c 10 K | c 5 K | d 20 K | d 10 K | d 5 K | d 0 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (i) | 1 | 1.777 | 2.284 | 2.406 | 2.406 | 2.284 | 2.750 | 2.984 | 2.951 | 3.516 | 3.799 | 4.081 |
| (i) | 3 | 1.420 | 1.725 | 1.846 | 1.846 | 1.725 | 1.958 | 2.075 | 2.089 | 2.371 | 2.513 | 2.654 |
| (i) | 5 | 1.606 | 1.822 | 2.056 | 2.173 | 1.822 | 2.096 | 2.233 | 2.089 | 2.371 | 2.513 | 2.654 |
| (ii) | 1 | 1.461 | 2.157 | 2.244 | 2.244 | 2.157 | 2.623 | 2.857 | 2.771 | 3.336 | 3.618 | 3.901 |
| (ii) | 3 | 1.198 | 1.598 | 1.684 | 1.684 | 1.598 | 1.831 | 1.948 | 1.935 | 2.218 | 2.359 | 2.500 |
| (ii) | 5 | 1.291 | 1.695 | 1.929 | 2.045 | 1.695 | 1.969 | 2.106 | 1.935 | 2.218 | 2.359 | 2.500 |

__Gas__ of both heaters, kg/h (corrected formula):

| Source | Sc. | a | b 20 K | b 10 K | b 5 K | c 20 K | c 10 K | c 5 K | d 20 K | d 10 K | d 5 K | d 0 K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (i) | 1 | 1.2363 | 0.9583 | 0.9461 | 0.9461 | 0.9583 | 0.9116 | 0.8883 | 0.8916 | 0.8351 | 0.8068 | 0.7786 |
| (i) | 3 | 1.2720 | 1.0142 | 1.0021 | 1.0021 | 1.0142 | 0.9908 | 0.9792 | 0.9778 | 0.9495 | 0.9354 | 0.9213 |
| (i) | 5 | 1.2534 | 1.0044 | 0.9811 | 0.9694 | 1.0044 | 0.9770 | 0.9633 | 0.9778 | 0.9495 | 0.9354 | 0.9213 |
| (ii) | 1 | 1.2679 | 0.9710 | 0.9623 | 0.9623 | 0.9710 | 0.9244 | 0.9010 | 0.9096 | 0.8531 | 0.8248 | 0.7966 |
| (ii) | 3 | 1.2942 | 1.0269 | 1.0182 | 1.0182 | 1.0269 | 1.0036 | 0.9919 | 0.9932 | 0.9649 | 0.9508 | 0.9367 |
| (ii) | 5 | 1.2849 | 1.0172 | 0.9938 | 0.9821 | 1.0172 | 0.9898 | 0.9761 | 0.9932 | 0.9649 | 0.9508 | 0.9367 |

Reading. (1) The partner model is below every counter-current figure, including b at 20 K. The ratio c / a is, for `dT_min` 20 / 10 / 5 K: scenario 1 1.29 / 1.55 / 1.68 (was 1.80 / 2.06 / 2.19); scenario 3 1.22 / 1.38 / 1.46 (was 1.64 / 1.80 / 1.88); scenario 5 1.13 / 1.30 / 1.39 (was 1.54 / 1.71 / 1.80); in variant (ii), scenario 1, 1.48 / 1.80 / 1.96 (was 1.69 / 2.01 / 2.17). (2) The circuit at 20 kg/h reaches 78 % of the pinch target at 10 K in scenario 1 (2.750 of 3.516 kW, variant (i); the share is unchanged, 3.662 of 4.666 kW before); the rest is the medium flow (10.6). (3) **At 20 K nothing condenses** (b = c everywhere: the cold end of the exhaust stays above its dew point, as found for the network model in Befund 78); at 10 K 0.175 kg/h of exhaust 1 condenses in scenario 1 (c 10 K, stage 1; was 0.064 kg/h). In variant (ii) the dew-point limit binds b at 10 and 5 K (both 2.244 kW in scenario 1), and condensation adds 0.38 and 0.61 kW there (was 0.29 and 0.52). (4) **No stage cools the gas** in b and c, and a stage that cannot recover is bypassed at 0 kW: in scenarios 3, 4 the second stage (exhaust 1 behind exhaust 2) recovers nothing in b and c, because the gas leaves the first stage warmer than exhaust 1 (84.3 degC against 58.9 degC at 10 K; was 105.6 against 72.3 degC). In the partner model no stage cools the gas at 20 kg/h.

Stage values, variant (i), scenarios 1, 3, 5 (Q in kW; temperatures in degC; medium `T1` is the cold return, `T2` the hot supply):

| Model | Sc. | Stage (source -> sink) | Q | Gas in -> out | Exhaust in -> out | Medium `T1` -> `T2` | Binding |
|---|---|---|---|---|---|---|---|
| a | 1 | 1 -> 1 | 0.569 | 15.0 -> 31.9 | 72.3 -> 56.3 | 31.9 -> 56.3 | equal outlet |
| a | 1 | 2 -> 2 | 1.207 | 15.0 -> 50.9 | 134.7 -> 102.6 | 50.9 -> 102.6 | equal outlet |
| a | 3 | 2 -> 1 | 1.207 | 15.0 -> 50.9 | 134.7 -> 102.6 | 50.9 -> 102.6 | equal outlet |
| a | 3 | 1 -> 1 | 0.213 | 50.9 -> 57.2 | 72.3 -> 66.3 | 57.2 -> 66.3 | equal outlet |
| a | 5 | 1 -> 1 | 0.569 | 15.0 -> 31.9 | 72.3 -> 56.3 | 31.9 -> 56.3 | equal outlet |
| a | 5 | 2 -> 1 | 1.037 | 31.9 -> 62.7 | 134.7 -> 107.1 | 62.7 -> 107.1 | equal outlet |
| c 10 K | 1 | 1 -> 1 | 0.792 | 15.0 -> 43.0 | 58.9 -> 36.6 | 20.0 -> 53.9 | approach |
| c 10 K | 1 | 2 -> 2 | 1.958 | 15.0 -> 84.3 | 108.9 -> 48.1 | 20.0 -> 103.9 | approach |
| c 10 K | 3 | 2 -> 1 | 1.958 | 15.0 -> 84.3 | 108.9 -> 48.1 | 20.0 -> 103.9 | approach |
| c 10 K | 3 | 1 -> 1 | 0.000 | 84.3 -> 84.3 | 58.9 -> 58.9 | 89.3 -> 89.3 | no driving force |
| c 10 K | 5 | 1 -> 1 | 0.792 | 15.0 -> 43.0 | 58.9 -> 36.6 | 20.0 -> 53.9 | approach |
| c 10 K | 5 | 2 -> 1 | 1.304 | 43.0 -> 89.2 | 108.9 -> 68.1 | 48.0 -> 103.9 | approach |

The partner model lets the exhaust leave at the temperature of the medium (56.3 and 102.6 degC), the counter-current circuit lets it fall to 36.6 and 48.1 degC (was 37.1 and 54.6 degC). **The medium reaches 103.9 degC in c** (106.4 degC at 5 K; was 129.7 and 132.2 degC): a liquid at this temperature needs a slightly pressurised circuit or another medium (every stage with `T2` above 100 degC is flagged in the artifact, `htm_above_boiling_at_1_atm`; 6 of the 12 stages at 10 K and at 5 K, none at 20 K). For orientation, the direct exchange without circuit (what the network model of Befunde 78-81 assumes) recovers 3.331 kW in scenario 1 at 10 K (condensing): the circuit at 20 kg/h reaches 83 % of it, at a flow of 30 kg/h 93 % (3.10 kW; was 4.434 kW, 83 % and 92 %).

### 10.4 Reading of the counter-current figures

The counter-current figures are **maxima for the stated approach** (the exchangers are assumed large enough to reach it; the partner's equal-outlet model also assumes unlimited area). They are not predictions of a plant with finite areas; whether finite areas (effectiveness or NTU) are planned is a question to the partner (already asked in section 7.5, still open).

### 10.5 WU2: decomposition of the difference between a and c

Sequential, so the contributions depend on the order; two orders are computed (`wu2`: `props_first` as below, `props_last`), variant (i), `dT_min` 10 K, recovered heat in kW (change of the gas flow in kg/h in brackets). In this version the step of the properties also changes the exhausts of the dryers (one property set), which is why it is no longer small:

| Step added | Scenario 1 | Scenario 3 | Scenario 5 |
|---|---|---|---|
| a: partner model as delivered | 1.777 (1.2363) | 1.420 (1.2720) | 1.606 (1.2534) |
| `c_p` of the gas, partner -> repository (exhausts of the dryers follow) | -0.525 (-0.1748) | -0.441 (-0.1832) | -0.482 (-0.1792) |
| counter-current, unlimited area, no approach, no dew-point limit | +1.965 (-0.1965) | +1.212 (-0.1212) | +1.246 (-0.1246) |
| approach `dT_min` along the exchangers | -0.467 (+0.0467) | -0.233 (+0.0233) | -0.274 (+0.0274) |
| dew-point limit (no condensation) | -0.344 (+0.0344) | -0.112 (+0.0112) | -0.040 (+0.0040) |
| condensation below the dew point | +0.344 (-0.0344) | +0.112 (-0.0112) | +0.040 (-0.0040) |
| c: result | 2.750 (0.9116) | 1.958 (0.9908) | 2.096 (0.9770) |

Reading. **The flow arrangement is the largest step**: +1.97 kW (properties first) or +2.35 kW (properties last) in scenario 1 against a net of the other steps of -0.99 or -1.38 kW; the total a to c is +0.97 kW (was +1.89 kW). The approach takes back 0.93 / 0.47 / 0.23 kW at 20 / 10 / 5 K in scenario 1 (same in both orders). **`cp_gas` is order-dependent and no longer negligible**: first it lowers the recovery by 0.53 kW (-30 %), last by 0.91 kW in scenario 1, because the colder exhausts of the dryers (58.9 and 108.9 degC instead of 72.3 and 134.7 degC) carry less heat; where it is exchanged last the step is the second largest of the chain. Its effect on the **gas flow** is order-dependent too (-0.17 to -0.18 kg/h first, -0.14 last in scenario 1), the no-recovery demand falls by 16 % (1.4140 -> 1.1867 kg/h). **Condensation also depends on the order**: at 10 K +0.344 / +0.112 / +0.040 kW (scenarios 1 / 3 / 5) with the repository's properties first, 0.000 with the partner's properties and exhausts (`props_last`, before the property step); at 20 K 0 in both. The dew-point limit and the condensation step cancel by construction, their net is the condensation share of c over b. **Ranking:** flow arrangement (1.2 to 2.4 kW) >> `cp_gas` (0.4 to 0.9 kW) >= approach (0.2 to 0.9 kW) > condensation (0 to 0.3 kW) (was: flow arrangement >> approach > `c_p` and condensation); the order changes the counter-current step by 0.38 kW (+1.965 against +2.352, scenario 1) and the position of the `cp_gas` step, not the first rank. The same ranking holds in variant (ii) (artifact `source_variants.ii_laboratory_model_dryer_2.wu2`).

### 10.6 WU3: mass flow of the heat transfer medium (20-150 kg/h per circuit, step 5 kg/h)

Optimum = smallest total flow among the grid points with the lowest gas flow (`wu3`; both circuits scanned independently, 27 x 27 points; "edge" = at the end of the range 20-150).

| Model, source (i) | Sc. | Optimum flows (circuit 1, 2), kg/h | Gas at the optimum | Gas at 20 / 20 | Gas at 150 / 150 |
|---|---|---|---|---|---|
| a (partner, as delivered) | 1 | 150, 150 (edge) | 1.1306 | 1.2363 | 1.1306 |
| a (partner, as delivered) | 3 | 20, 150 (edge) | 1.2210 | 1.2720 | 1.2210 |
| a (partner, as delivered) | 5 | 150, 150 (edge) | 1.1739 | 1.2534 | 1.1739 |
| c, 10 K (counter-current, condensing) | 1 | 25, 25 | 0.8536 | 0.9116 | 0.9950 |
| c, 10 K (counter-current, condensing) | 3 | 20, 25 | 0.9495 | 0.9908 | 1.0490 |
| c, 10 K (counter-current, condensing) | 5 | 20, 25 | 0.9495 | 0.9770 | 1.0264 |

Variant (ii) (c, 10 K): scenario 1 25 / 25 kg/h, gas 0.8690 against 0.9244 at 20 / 20 and 1.0042 at 150 / 150 (was 0.8313, 0.8932, 0.9830); scenarios 3, 5 20 (edge) / 25 kg/h, 0.9649; the partner model again at 150 / 150 kg/h (1.1812, 1.2479, 1.2244 for scenarios 1, 3, 5). The optimum at 20 and 5 K is also at 25 / 25 kg/h in scenario 1 (0.9101 and 0.8254 kg/h in variant (i)). The grid step is 5 kg/h; the equal-flow curve is flat within a few kg/h of the optimum (artifact `equal_flow_curve`, 5-150 kg/h; scenario 1, c, 10 K: 0.9116 at 20, 0.8536 at 25, 0.8770 at 30, 0.9269 at 40 kg/h; was 0.8205, 0.7433, 0.7782, 0.8441).

* **Partner model.** `Q = (T_c,in - T_h,in) / (1/A + 1/B + 1/C)` increases with `B`: the sheet's 20 kg/h is the worst point of the range and 150 kg/h the best (gas -8.5 % in scenario 1, -4.0 % in scenario 3). There is no optimum in the range. In scenarios 3, 4 the second stage cools the gas from 150 kg/h (equal flows) on and at no smaller flow of the scan (section 7.5). No stage of the partner model drops below its dew point at any flow of the scan.
* **Counter-current.** The medium flattens its own temperature profile: with a large flow it is nearly isothermal, and the approach through the circuit then costs temperature level (the exhaust leaves no colder than the medium plus the approach, the gas no warmer than the medium minus the approach). The best flow is where the capacity rate of the medium equals those of the gas and the exhausts: 25 kg/h is `B` = 29.2 W/K (hand: `25/3600 x 4200`), against 28.3 W/K of the process gas (100/3600 x (1006 + 0.006 x 1860)) and 30.1 and 32.0 W/K of the two exhausts (variant (i), repository properties; the capacity rates depend on the humidity, not on the temperature, so they and the optimum are the same in both versions of this section). Below it the medium limits the heat (`Q` about `B (T_c,in - T1 - dT_min/2)`), above it the circuit loses temperature level. In scenarios 3, 4 the flow of circuit 1 (second stage) is irrelevant, because that stage recovers nothing; the table reports the edge value of the tie-break. In scenario 5 the optimum of circuit 1 (first stage, exhaust 1) is at the lower edge of the range or below it; the scan below 20 kg/h is only along equal flows, whose lowest point is 25 kg/h.
* **The medium temperature** falls with the flow: 103.9 degC at 20 and 25 kg/h, 98.7 at 30, 85.7 at 40, 80.0 at 50 and 68.7 at 150 kg/h (scenario 1, c, 10 K; was 129.7, 129.7, 122.2, 105.8, 98.4, 83.7). **At the optimum of 25 kg/h the medium is only just above the boiling point (103.9 degC) and below it from about 30 kg/h on**, so the best flow is much less demanding for the circuit than the first version suggested (it said 129.7 degC and a pressurised circuit or another medium).

### 10.7 For the partner (feedback; the questions in the order of their weight)

1. **Equal outlet temperatures** (parallel flow, unlimited area) recover 60 to 88 % of what counter-current exchangers recover with the same capacity rates and an approach of 5-20 K at the sheet's flows (was 46 to 65 %); the recovery numbers of the workbook are a conservative bound by this assumption. Are finite areas (effectiveness, NTU) or counter-current exchangers planned in the Dyssol unit?
2. **Medium temperature.** The medium leaves the recovery exchanger above 100 degC (102.6 and 107.1 degC at 20 kg/h, `cp` 4.2); with counter-current exchangers it is 103.9 degC at 10 K and 106.4 degC at 5 K (was 129.7 and 132.2 degC), below 100 degC from about 30 kg/h. Which medium and which pressure?
3. **Flow of the medium.** With equal outlet temperatures the recovery rises with the flow of the medium, with counter-current exchangers it has an optimum near the capacity rate of the gas (25 kg/h here, unchanged). A study of the flow in the partner model alone points to the highest flow.
4. **`cp_gas`** 1.2 against about 1.0 kJ/(kg K): -16 % of the no-recovery heater demand and, because it lowers the exhaust temperatures of the dryers, -0.44 to -0.91 kW of recovery depending on the order of the steps (was up to -0.17 kW; section 9, section 11).
5. **Condensation** adds at most 0.58 kW (scenario 1, 5 K; was 0.27 kW) and nothing at 20 K; it needs a small approach.
6. **Measured values of the real recovery circuit** (inlet and outlet temperatures and flow of the medium, storage temperatures) are the only basis for a validation; both models here are models (`docs/02`, open question 12).

### 10.8 Caveats

* Model against model, no plant data; the independent calculation is a maximum under a stated approach (unlimited area); its absolute numbers rest on placeholders (`dT_min`, `T_kalt`, the dew-point margin, the repository's gas properties, the efficiency and the heating value of the workbook).
* One property set per column (b, c, d: the repository's in the dryers and the exchangers; a: the partner's). The dryers of b, c, d are the block of the equation document with the repository's gas properties, no heat loss and the equilibrium moisture of the sheet; the heat loss and the product moisture of the laboratory model are not in this section (they are in section 11).
* Variant (ii) is an extrapolation of the laboratory model (section 9.6); the bed temperature lies above its 80 degC validity marker. Its column a mixes the partner's exchanger properties with the laboratory-model exhaust of dryer 2 (10.3). The ranking of the models is the same in (i) and (ii); the absolute recovery of dryer 2 is not a plant statement.
* The result depends on the convention of `dT_min` (through the circuit; per exchanger it is one step along 5 -> 10 -> 20 K). To be confirmed.
* The pinch target bounds any wiring with the same approach; the circuit stays at 78 % of it at 10 K (scenario 1, 20 kg/h) because of the medium flow.
* All numbers are deterministic; no confidence statement is made. Numerical floors: rebuild against `evaluate_cascade` 3.8e-13 kW, against `loops` 2.5e-10 W, node count 0.0 kW, condensation profile against a complete enthalpy balance at most 0.9 %. Every difference reported above exceeds them by many orders of magnitude, except the optimum flow, which is resolved to 5 kg/h only.
* The full test suite was run for the branch `partner-units` (section 11), which contains this correction.

## 11. Flowsheet of the partner model from exchangeable units: which assumption moves which scenario (feedback to the partner)

Branch `partner-units`. Lab book: Befund 95. **Question:** the units of the partner workbook existed twice, as a faithful rebuild (section 7) and as physical alternatives (sections 9 and 10); how do the six scenarios change when the assumptions are replaced one at a time, and does the ranking of the scenarios survive every intermediate step?

Code: `src/twinlink_ml/flowsheet/` (units `DryerBlock`, `HeatExchanger`, `HeatTransferLoop`, `Preheater`, `Burner`, class `Flowsheet`, `decomposition.py`), tests `tests/test_flowsheet_units.py` and `tests/test_flowsheet_workbook.py`, script `scripts/report_partner_flowsheet.py` (commit `7f6b8b8`), artifact `artifacts/runs/partner_flowsheet_decomposition.json` (`git/dirty=False`). Model against model, no plant data; the partner model is the equation document of section 7 with the circuit flows of the sheet (20 kg/h each). `Preheater` is the pre-heating train of one sink (its stages in the drawn order); `HeatExchanger` is one exchanger; `HeatTransferLoop` couples the recovery exchanger and the pre-heating exchanger through the medium.

### 11.1 The switches

One switch per assumption, each `"partner"` (the workbook) or `"physical"` (the independent calculation):

| Switch | `partner` | `physical` |
|---|---|---|
| `exchanger` | equal outlet temperatures of both fluids (parallel flow, unlimited area) | counter-current, unlimited area, minimum approach `dT_min` checked **along** the exchanger; main convention: `dT_min` per exchanger (10 K); sensitivity: `dT_min` through the circuit (each of the two exchangers gets half, the convention of section 10) |
| `cp_gas` | 1.2 kJ/(kg K) (gas), 2.0 (vapour) | 1.006 and 1.86 kJ/(kg K) (gas and vapour; the property set of the step "c_p of the gas" of section 10), in the dryer, the exchangers and the burner |
| `condensation` | off (neglected, equation 4.1) | on (condensing release curve of the exhaust; with the equal-outlet exchanger the outlet is the point of the curve where it equals the medium temperature) |
| `heat_loss` | off | `UA (T - T_amb)` of the bed, `UA` = 2 W/K (laboratory placeholder, **not** a plant value, section 9.6); dryers only |
| `product_moisture` | equilibrium moisture 2 g/kg | product moisture of the fluidised-bed model, **dryer 2 only** (0.0566 kg/kg), **an extrapolation** of the laboratory model (every input of dryer 2 is outside its range, section 9.6); dryer 1 keeps 2 g/kg |
| `gas_formula` | `Scenarios` row 26/39 as built (108.91 times too high) | corrected (efficiency `/100`, humidity in kg/kg) |
| `boiling_check` | water in the circuit unlimited | flag when the medium leaves the recovery exchanger above 100 degC (1 atm); a flag only, nothing is capped |

Not switches but outputs: fan and pump power (placeholders without source: fan and pump efficiencies of the laboratory process, pressure drops 4 kPa per dryer line and 0.5 kPa per exchanger side, 1 bar per medium circuit), booked as electricity in the CO2 with the factors 0.366 (electricity) and 0.201 kg/kWh (gas) of `carrier_energy`. CO2 is always computed from the **corrected** fuel; the as-built fuel is reported but never ranked or converted.

### 11.2 Reproduction

* **Workbook (all switches `partner`).** Scenarios 1-6 against every numeric cell of the sheet `Scenarios` (230 cells, as built, gas rows included): largest relative difference 2.0e-15 (cell I24), largest absolute 5.7e-14; the tolerance of the workbook tests is `rel` 1e-12, `abs` 1e-12. With the gas formula on `physical` only rows 26, 39 and 41 differ from the sheet, by the factor 108.911 (section 7.3).
* **Befund 90 (all switches `physical`, and `partner`).** With the exhausts pinned to the two source variants of Befund 90 (11.5 explains why they have to be pinned), columns a, b and c at `dT_min` 20 / 10 / 5 K in both conventions: 156 scenario rows compared, largest difference 0.0 kW (recovery and stages) and 4.4e-16 kg/h (gas).
* **Script.** `scripts/report_heat_recovery_comparison.py` now uses the lifted units; its new artifact `heat_recovery_comparison_r3_units.json` equals `heat_recovery_comparison_r2.json` in all 113 524 numbers (largest difference 0.0). Its 39 tests are unchanged and pass.
* **Mirror pairs** 1/2, 3/4, 5/6 have the same total fuel, recovery and CO2 in every state of the study (test, relative 1e-9).

### 11.3 Decomposition (main convention: `dT_min` 10 K per exchanger)

The decomposition depends on the order of the switches, so two orders are given: the order of the table in 11.1 and its reverse. Step *k* has the first *k* switches of the order on `physical`. Fuel and CO2 are shown from the step where `gas_formula` is on `physical` (before it the sheet's formula would give figures 108.9 times too high; the ranking uses the corrected fuel in every step). Fuel in kg/h of the corrected formula; scenarios 2, 4, 6 equal 1, 3, 5.

Forward order:

| Step | Switch turned to `physical` | Recovery kW (sc. 1 / 3 / 5) | Fuel kg/h (1 / 3 / 5) | CO2 kg/h (1 / 3 / 5) | Ranking (CO2, lowest first) | Medium above 100 degC, stages in sc. 1 |
|---|---|---|---|---|---|---|
| 0 | none (partner model as built) | 1.777 / 1.420 / 1.606 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 1 | exchanger | 3.195 / 2.325 / 2.592 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 2 | cp_gas | 2.284 / 1.725 / 1.822 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 3 | condensation | 2.284 / 1.725 / 1.822 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 4 | heat_loss | 2.097 / 1.601 / 1.687 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 5 | product_moisture | 2.205 / 1.709 / 1.795 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 6 | gas_formula | 2.205 / 1.709 / 1.795 | 0.9662 / 1.0158 / 1.0071 | 2.584 / 2.709 / 2.687 | 1=2 < 5=6 < 3=4 | not checked |
| 7 | boiling_check | 2.205 / 1.709 / 1.795 | 0.9662 / 1.0158 / 1.0071 | 2.584 / 2.709 / 2.687 | 1=2 < 5=6 < 3=4 | 0 |

Reverse order:

| Step | Switch turned to `physical` | Recovery kW (sc. 1 / 3 / 5) | Fuel kg/h (1 / 3 / 5) | CO2 kg/h (1 / 3 / 5) | Ranking (CO2, lowest first) | Medium above 100 degC, stages in sc. 1 |
|---|---|---|---|---|---|---|
| 0 | none (partner model as built) | 1.777 / 1.420 / 1.606 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 1 | boiling_check | 1.777 / 1.420 / 1.606 | - | - | 1=2 < 5=6 < 3=4 | 1 |
| 2 | gas_formula | 1.777 / 1.420 / 1.606 | 1.2363 / 1.2720 / 1.2534 | 3.265 / 3.355 / 3.308 | 1=2 < 5=6 < 3=4 | 1 |
| 3 | product_moisture | 1.817 / 1.448 / 1.646 | 1.2323 / 1.2692 / 1.2494 | 3.255 / 3.347 / 3.298 | 1=2 < 5=6 < 3=4 | 1 |
| 4 | heat_loss | 1.726 / 1.376 / 1.565 | 1.2414 / 1.2764 / 1.2575 | 3.277 / 3.365 / 3.318 | 1=2 < 5=6 < 3=4 | 1 |
| 5 | condensation | 1.726 / 1.376 / 1.565 | 1.2414 / 1.2764 / 1.2575 | 3.277 / 3.365 / 3.318 | 1=2 < 5=6 < 3=4 | 1 |
| 6 | cp_gas | 1.220 / 0.950 / 1.101 | 1.0646 / 1.0916 / 1.0766 | 2.832 / 2.900 / 2.862 | 1=2 < 5=6 < 3=4 | 0 |
| 7 | exchanger | 2.205 / 1.709 / 1.795 | 0.9662 / 1.0158 / 1.0071 | 2.584 / 2.709 / 2.687 | 1=2 < 5=6 < 3=4 | 0 |

Each switch alone (order-free). From the partner model, one switch to `physical` (dQ = change of the recovered heat, dFuel and dCO2 = change of the corrected fuel and of the CO2):

| Switch | dQ kW (sc. 1 / 3 / 5) | dFuel kg/h (1 / 3 / 5) | dCO2 kg/h (1 / 3 / 5) |
|---|---|---|---|
| exchanger | +1.419 / +0.905 / +0.986 | -0.1419 / -0.0905 / -0.0986 | -0.3575 / -0.2281 / -0.2486 |
| cp_gas | -0.525 / -0.441 / -0.482 | -0.1748 / -0.1832 / -0.1792 | -0.4401 / -0.4613 / -0.4510 |
| condensation | -0.000 / +0.000 / -0.000 | +0.0000 / -0.0000 / +0.0000 | +0.0000 / -0.0000 / +0.0000 |
| heat_loss | -0.089 / -0.071 / -0.080 | +0.0089 / +0.0071 / +0.0080 | +0.0221 / +0.0177 / +0.0198 |
| product_moisture | +0.040 / +0.028 / +0.040 | -0.0040 / -0.0028 / -0.0040 | -0.0100 / -0.0070 / -0.0101 |
| gas_formula | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |
| boiling_check | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |

From the all-`physical` state, one switch back to `partner`:

| Switch | dQ kW (sc. 1 / 3 / 5) | dFuel kg/h (1 / 3 / 5) | dCO2 kg/h (1 / 3 / 5) |
|---|---|---|---|
| exchanger | -0.985 / -0.759 / -0.694 | +0.0985 / +0.0759 / +0.0694 | +0.2483 / +0.1913 / +0.1751 |
| cp_gas | +0.876 / +0.572 / +0.731 | +0.1397 / +0.1701 / +0.1542 | +0.3518 / +0.4284 / +0.3883 |
| condensation | -0.017 / +0.000 / -0.003 | +0.0017 / -0.0000 / +0.0003 | +0.0043 / -0.0000 / +0.0008 |
| heat_loss | +0.193 / +0.130 / +0.141 | -0.0193 / -0.0130 / -0.0141 | -0.0483 / -0.0325 / -0.0353 |
| product_moisture | -0.108 / -0.108 / -0.108 | +0.0108 / +0.0108 / +0.0108 | +0.0272 / +0.0272 / +0.0272 |
| gas_formula | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |
| boiling_check | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |

Sensitivity, `dT_min` **through the circuit** (the convention of section 10; 5 K per exchanger). Forward order:

| Step | Switch turned to `physical` | Recovery kW (sc. 1 / 3 / 5) | Fuel kg/h (1 / 3 / 5) | CO2 kg/h (1 / 3 / 5) | Ranking (CO2, lowest first) | Medium above 100 degC, stages in sc. 1 |
|---|---|---|---|---|---|---|
| 0 | none (partner model as built) | 1.777 / 1.420 / 1.606 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 1 | exchanger | 3.662 / 2.559 / 2.897 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 2 | cp_gas | 2.406 / 1.846 / 2.056 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 3 | condensation | 2.750 / 1.958 / 2.096 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 4 | heat_loss | 2.564 / 1.834 / 1.961 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 5 | product_moisture | 2.672 / 1.942 / 2.069 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 6 | gas_formula | 2.672 / 1.942 / 2.069 | 0.9195 / 0.9924 / 0.9797 | 2.466 / 2.650 / 2.618 | 1=2 < 5=6 < 3=4 | not checked |
| 7 | boiling_check | 2.672 / 1.942 / 2.069 | 0.9195 / 0.9924 / 0.9797 | 2.466 / 2.650 / 2.618 | 1=2 < 5=6 < 3=4 | 1 |

Reverse order:

| Step | Switch turned to `physical` | Recovery kW (sc. 1 / 3 / 5) | Fuel kg/h (1 / 3 / 5) | CO2 kg/h (1 / 3 / 5) | Ranking (CO2, lowest first) | Medium above 100 degC, stages in sc. 1 |
|---|---|---|---|---|---|---|
| 0 | none (partner model as built) | 1.777 / 1.420 / 1.606 | - | - | 1=2 < 5=6 < 3=4 | not checked |
| 1 | boiling_check | 1.777 / 1.420 / 1.606 | - | - | 1=2 < 5=6 < 3=4 | 1 |
| 2 | gas_formula | 1.777 / 1.420 / 1.606 | 1.2363 / 1.2720 / 1.2534 | 3.265 / 3.355 / 3.308 | 1=2 < 5=6 < 3=4 | 1 |
| 3 | product_moisture | 1.817 / 1.448 / 1.646 | 1.2323 / 1.2692 / 1.2494 | 3.255 / 3.347 / 3.298 | 1=2 < 5=6 < 3=4 | 1 |
| 4 | heat_loss | 1.726 / 1.376 / 1.565 | 1.2414 / 1.2764 / 1.2575 | 3.277 / 3.365 / 3.318 | 1=2 < 5=6 < 3=4 | 1 |
| 5 | condensation | 1.726 / 1.376 / 1.565 | 1.2414 / 1.2764 / 1.2575 | 3.277 / 3.365 / 3.318 | 1=2 < 5=6 < 3=4 | 1 |
| 6 | cp_gas | 1.220 / 0.950 / 1.101 | 1.0646 / 1.0916 / 1.0766 | 2.832 / 2.900 / 2.862 | 1=2 < 5=6 < 3=4 | 0 |
| 7 | exchanger | 2.672 / 1.942 / 2.069 | 0.9195 / 0.9924 / 0.9797 | 2.466 / 2.650 / 2.618 | 1=2 < 5=6 < 3=4 | 1 |

One switch to `physical` from the partner model:

| Switch | dQ kW (sc. 1 / 3 / 5) | dFuel kg/h (1 / 3 / 5) | dCO2 kg/h (1 / 3 / 5) |
|---|---|---|---|
| exchanger | +1.885 / +1.139 / +1.291 | -0.1885 / -0.1139 / -0.1291 | -0.4751 / -0.2870 / -0.3255 |
| cp_gas | -0.525 / -0.441 / -0.482 | -0.1748 / -0.1832 / -0.1792 | -0.4401 / -0.4613 / -0.4510 |
| condensation | -0.000 / +0.000 / -0.000 | +0.0000 / -0.0000 / +0.0000 | +0.0000 / -0.0000 / +0.0000 |
| heat_loss | -0.089 / -0.071 / -0.080 | +0.0089 / +0.0071 / +0.0080 | +0.0221 / +0.0177 / +0.0198 |
| product_moisture | +0.040 / +0.028 / +0.040 | -0.0040 / -0.0028 / -0.0040 | -0.0100 / -0.0070 / -0.0101 |
| gas_formula | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |
| boiling_check | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |

One switch back to `partner` from all-`physical`:

| Switch | dQ kW (sc. 1 / 3 / 5) | dFuel kg/h (1 / 3 / 5) | dCO2 kg/h (1 / 3 / 5) |
|---|---|---|---|
| exchanger | -1.451 / -0.992 / -0.968 | +0.1451 / +0.0992 / +0.0968 | +0.3659 / +0.2501 / +0.2441 |
| cp_gas | +0.876 / +0.572 / +0.762 | +0.1397 / +0.1701 / +0.1511 | +0.3517 / +0.4284 / +0.3804 |
| condensation | -0.357 / -0.107 / -0.044 | +0.0357 / +0.0107 / +0.0044 | +0.0899 / +0.0269 / +0.0109 |
| heat_loss | +0.193 / +0.130 / +0.141 | -0.0193 / -0.0130 / -0.0141 | -0.0483 / -0.0325 / -0.0353 |
| product_moisture | -0.108 / -0.108 / -0.108 | +0.0108 / +0.0108 / +0.0108 | +0.0272 / +0.0272 / +0.0272 |
| gas_formula | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |
| boiling_check | +0.000 / +0.000 / +0.000 | +0.0000 / +0.0000 / +0.0000 | +0.0000 / +0.0000 / +0.0000 |

### 11.4 Reading

1. **The ranking of the scenarios never changes.** By CO2, by corrected fuel and by recovered heat, in every one of the 4 x 8 steps (two orders, two conventions) the order is 1 = 2 < 5 = 6 < 3 = 4 (lowest CO2 first; the equal signs are the mirror pairs). The smallest distance between two neighbouring groups in any step is 0.022 kg CO2/h. Scenario 1/2 (each exhaust to the sink of its own dryer) is best in every state, 3/4 (the cold exhaust behind the hot one) worst: the second stage of 3/4 recovers 0 kW in the physical states because the gas leaves the first stage warmer than exhaust 1.
2. **The levels move much more than the order.** From the partner model to the all-`physical` state the recovery of scenario 1 goes from 1.777 to 2.205 kW and the corrected fuel of scenarios 1 / 3 / 5 from 1.2363 / 1.2720 / 1.2534 to 0.9662 / 1.0158 / 1.0071 kg/h (about -20 %); the no-recovery fuel falls from 1.4140 to 1.1867 kg/h. The recovered *heat* moves in both directions (the exchanger raises it by 1.42 kW in scenario 1, `cp_gas` lowers it by 0.53 kW, the heat loss by 0.09 kW, the wetter product raises it by 0.04 kW), the *fuel* does not follow it: `cp_gas` lowers the recovery and the fuel, because the heat demand falls more than the recovery. For the fuel `cp_gas` is the largest single switch (-0.17 to -0.18 kg/h), the exchanger the second (-0.09 to -0.14 kg/h).
3. **Order dependence.** The exchanger step is +1.42 kW in scenario 1 from the partner model but -0.99 kW when taken away from the all-`physical` state; `cp_gas` is -0.53 kW from the partner model and +0.88 kW back from all-`physical` (as in section 10.5, where the heat effect of `cp_gas` depended on the order).
4. **Condensation** adds nothing at 10 K per exchanger (the exhaust stays above its dew point) and 0.36 kW in scenario 1 at 10 K through the circuit (5 K per exchanger), 0.04 kW in scenario 5.
5. **Boiling flag.** In the partner model one stage per scenario (the stage with exhaust 2) has the medium above 100 degC (102.6 degC in scenarios 1-4, 107.1 degC in 5, 6; section 10.2). With the physical exchanger at 10 K per exchanger no stage does; at 5 K per exchanger (through the circuit) one does. A flag, never a cap.
6. **Electricity** (placeholders) is 0.43 kW in each of the scenarios 1-6 (two exchangers on the gas side and two circuits in all six; the scenarios differ by less than 1 W in the partner state). In the physical state of scenario 1 it is 0.157 of 2.584 kg CO2/h (the fuel power is 12.1 kW). It is visible in the CO2 but, by this structure, cannot reorder the scenarios; it would if the pressure drops depended on the scenario (a placeholder question, not a result).

### 11.5 What the units showed that was not visible before

* **Befund 90 mixed two property sets (corrected, see the note at the top of section 10 and the addendum to Befund 90).** Its columns b, c and d used the repository's gas properties in the exchangers but took the exhausts of the dryers from the partner's model with `cp_gas` 1.2 (variant (i): 72.28 and 134.66 degC). With one property set for the whole flowsheet the exhausts fall to 58.94 and 108.93 degC. Section 10 now uses one set; its column c (2.750 kW in scenario 1, 10 K through the circuit, variant (i)) differs from the all-`physical` flowsheet of this section (2.672 kW) by the heat loss and the product moisture of the dryers that section 10 does not contain (about -0.19 kW and +0.11 kW alone, -0.08 kW together; see the single-switch tables).
* **The dryer block is not the bed.** In the all-`physical` state the block gives 108.25 degC for dryer 2 against 103.49 degC of the laboratory model itself. The remaining 4.8 K come from constants that are not switches here and are not examined in this study. Candidates (the differences of section 9.2): the heat capacity of the suspension (bed: fixed 3500 J/(kg K); block: mixture rule, 2920 at solid fraction 0.4), the slope of the evaporation enthalpy (bed -2358.6 J/(kg K); block `cp_v - cp_l`), the heat capacities of liquid and solid (bed 4180 / 1200; block 4200 / 1000) and the evaporation enthalpy at 0 degC (bed 2500.8 kJ/kg; block 2500). Which of them accounts for how much is not known. With dryer 2 pinned to the bed state the recovery of scenarios 1 / 3 / 5 is 2.094 / 1.598 / 1.684 kW (10 K per exchanger) against 2.205 / 1.709 / 1.795 kW with the block; the ranking is the same (artifact `physical_with_dryer_2_pinned_to_the_bed`).

### 11.6 For the partner

1. The ranking 1 = 2 < 5 = 6 < 3 = 4 does not depend on any single assumption of the workbook, nor on the order in which they are replaced; the amount of fuel and CO2 does (about -20 % of fuel with all assumptions replaced).
2. The assumptions that matter for the fuel, in this order: `cp_gas` (1.2 or about 1.0 kJ/(kg K)), the exchanger arrangement, then the heat loss and the product moisture of the dryer (small), condensation (only with small approaches). The gas formula of `Scenarios` row 26/39 changes only the figures of the sheet, not the physics (factor 108.91).
3. The questions of section 10.7 stand (finite areas or counter-current exchangers in the Dyssol unit, medium and pressure, flow of the medium, measured values of the real circuit).

### 11.7 Caveats

* Model against model, no plant data. Placeholders: `dT_min` 10 K (and its convention), `UA` 2 W/K, dew-point margin 3 K, `T_kalt` 15 degC, the fan and pump numbers, the product moisture of dryer 2 (extrapolation).
* Product moisture `physical` applies to dryer 2 only; dryer 1 keeps the equilibrium moisture because the laboratory model was solved only for dryer 2.
* The `heat_loss` and `product_moisture` switches act on the dryer block of the equation document; the other constants of the bed model (11.5) are not switches.
* "Ranking" is the order of totals of one deterministic calculation; no confidence statement is made. The medium temperature is not capped anywhere.
