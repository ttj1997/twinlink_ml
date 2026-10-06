# Partner model v4 (`Model_v4.xlsx`): one-to-one reconstruction, deviation list, proposed reference values

Branch `v4` (worktree), not merged. Lab book entry: placeholder "Befund NN" at the end of
`docs/04_stand_ap1.md`; the number is assigned at the merge. No comparison with the lab model:
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
(ii) `eta` and `H_u` are inputs of one function, kept in one place (partner: 80 %, 45 MJ/kg; repo
placeholders: `eta_gas` 0.80 in D5, `H_u` 50 MJ/kg and `eta_H` 0.90 in `HeatNetworkConfig`, i.e. three
different values for one quantity, to be reconciled once and named); (iii) the mass flow is for
plausibility against the partner's table; the objective needs `E_gas` in kW, not kg/h.

## 5. Caveats

* The reconstruction reproduces a draft. All finished cells agree with Excel; the open cells are
  our proposal and say nothing about what the partner will compute.
* `cp_gas`, the moisture basis and `T_cold` are questions, not corrections.
* The cascade is energy- and temperature-limited, not a heat-exchanger design (no area, no dT profile beyond the
  check A4).
* The feedback to the partner is in `local/` (not in the repository).
