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
`8f7f31a`, run on a clean tree). Lab book: Befund NN (placeholder; the number is assigned at the merge, after 73, which is
reserved for branch `m3`). The workbook and the equation document (`Flow diagram and balance equations v4.docx`) are in
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
