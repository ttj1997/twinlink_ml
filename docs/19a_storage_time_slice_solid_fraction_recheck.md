# Storage test 1 (Befund 99): re-check with an energy-balance-admissible solid fraction

Status: report for the managing session. **Not a lab-book entry, and no change to Befund 99.**
Befund 99, `docs/19_storage_time_slice.md` and their artifacts are unchanged. Branch
`storage-timeslice-solid-fraction` (from master `c3f13a8`).

- **Script:** `scripts/recheck_storage_timeslice_solid_fraction.py`, committed in `ec64e0f` before
  any run.
- **Artifacts:** `artifacts/runs/recheck_storage_timeslice_solid_fraction_{060,0634,070}.json`.
  Each was run on a clean tree (`git/dirty=False`; recorded commits `ec64e0f` / `629c6b1` /
  `7c08870`; the commits after `ec64e0f` only add the previous artifact).
- **Comparison tables:** printed by `scripts/report_storage_timeslice_solid_fraction.py`, which
  reads the artifacts only.

Confidentiality: roles only, as in docs/19. Raw exports only through `TWINLINK_PLANT_DATA_ACE` /
`TWINLINK_PLANT_DATA_RID`; the raw-file hashes equal those of Befund 99, and the script checks this.

## 1. Question and answer

**Question.** Befund 99 reconstructs the exhaust humidity of plant A from the spray rate with the
solid fraction 0.3 (placeholder, band 0.2-0.5). The operating-point study (branch
`plant-a-operating-point`, `docs/23_plant_a_operating_point.md` section 5) finds this value
inadmissible by the dryer energy balance of the same two days: the balance needs a larger value
(product-specific minima kept locally), and 0.634 is consistent with both days. Does any Befund 99 number move?

**Answer.** The gate, every number of docs/19 sections 1, 5 and the scenario table, and the base
savings of R1-R4 are unchanged at 0.60, 0.634 and 0.70.

What moves:
- the exhaust state itself;
- three sensitivity entries and two decomposition entries that involve the exhaust (largest:
  R1 "exhaust C by annual figure" +398 -> +175 / +139 / +84 kWh/d; R3 `dT_min` 10 K
  +247 -> +114 / +88 / +51 kWh/d);
- the product-1 maximum of the grid (2.59 % -> 2.66-2.69 %);
- `E_direct` of up to 254 product-1 scenarios that are far from the gate threshold (largest
  saving change 81 kWh/d, 0.60 points).

The conclusions of Befund 99 (L2; the potential comes from the compressor load profile against
the process-air demand; the solid fraction is not needed for this question) stand. The caveat
"product 2 capped at saturation in every step" no longer holds with an admissible value.

## 2. What was run

The script repeats the pre-registered computation of `scripts/gate_storage_timeslice.py` and the
post-hoc references R3/R4 of `scripts/analyze_storage_timeslice_sensitivity.py`. It reuses their
code, main grid (2240 scenarios), capacities, levels, gate definitions, decomposition scenarios
and variants. Only the following differ:

* **Solid fraction:** set to `x_s` in every scenario, one artifact per value. 0.60 lies at the lower
  end of the band 0.60-0.70 and is not admissible for every product (product-specific minima kept
  locally); it is reported anyway, as asked.
* **Dropped variants:** `solid_fraction=0.2/0.5` (the three runs replace them). The gate therefore
  has 11 520 cells instead of 11 540.
* **Realism check of the gate:** Befund 99 bands for every placeholder except the solid fraction.
  This uses a new optional argument `gate_cells(..., realistic=...)`; its default is unchanged and
  it is tested.
* **Decomposition scenarios:** the four Befund 99 scenarios with the same labels. The
  pre-registered rule selects the same maximum scenarios in every run, so no extra decomposition
  was needed.
* **Control:** 116 Befund 99 runs (every 20th grid scenario and R1-R4) re-run at 0.3 reproduce
  `E_direct` and `E_ideal` of the Befund 99 artifacts exactly (difference 0.0 kWh/d). The
  headline numbers of docs/19 are recomputed from the Befund 99 grid by the same summary
  function that is applied to the new runs; they reproduce 339 / 174 / 121 / 44 / 256 / 83 /
  185 / 110, 2.6 %, 3.0 % and 0.0 %.
* **Checks:** as in Befund 99, all pass except the `rule_converged` flag (161-163 results; this
  count includes R3/R4 and their variants). The largest rule gap is 3.9e-15; the storage balance
  residual is at most 1.1e-10 kWh.

## 3. Exhaust state of plant A (reconstructed; day means)

| `x_s` | product 1: `Y_out` kg/kg, capped share, exhaust A kW* | product 2: `Y_out`, capped share, exhaust A kW* |
|---|---|---|
| 0.3 (Befund 99) | 0.0191, 0.00, 825 | 0.0353, **1.00**, 917 |
| 0.60 | 0.0126, 0.00, 626 | 0.0273, 0.00, 693 |
| 0.634 | 0.0119, 0.00, 604 | 0.0255, 0.00, 642 |
| 0.70 | 0.0105, 0.00, 569 | 0.0221, 0.00, 545 |

Which of these values the dryer energy balance admits per product is kept locally (band 0.60-0.70,
common value 0.634).

\* Mean heat of the exhaust of plant A down to `T_kalt` = 15 degC. The exhaust of plant C by volume
(two lines, flat profile, same state assumed) drops correspondingly: 999 / 1203 kW at 0.3 against
759-689 / 908-715 kW. The exhaust heat drops by 24-41 %, mostly as latent heat below the dew point.

## 4. Befund 99 numbers: what moves and by how much

Values at `x_s` = 0.3 (Befund 99) -> 0.60 / 0.634 / 0.70. Savings are at C = 5 MWh in kWh/d
(percentage points of `E_direct` in brackets).

| item (docs/19 section) | 0.3 | 0.60 | 0.634 | 0.70 |
|---|---|---|
| gate reading; cells with `G >= 5 %`; cells with both (5) | L2; 1420; 0 | L2; 1420; 0 | L2; 1420; 0 | L2; 1420; 0 |
| per-group readings and maxima (site, horizon, day, mode), closest (5) | as docs/19 | unchanged | unchanged | unchanged |
| main grid: scenarios with a cell >= 5 % (1, 6) | 339 (15.1 %), all product 2 | 339 | 339 | 339 |
| main-grid maximum, product 1 (1, 6) | 2.59 % | 2.66 % | 2.67 % | 2.69 % |
| main-grid maximum, mode flat; closest conditions (1) | 3.05 %; 0.0 % | unchanged | unchanged | unchanged |
| loss effect, day horizon (6): mean / max kWh/d | 2.15 / 88 | 2.14 / 88 | 2.14 / 88 | 2.15 / 88 |
| table of the main scenarios, incl. C at 90 %, best level, rule gap (6) | as docs/19 | unchanged | unchanged | unchanged |
| multi-level check (6) | 162 kWh/d in max-kWh scenario, else 0 | unchanged | unchanged | unchanged |
| decomposition R1: leave out exhaust of A opens (6) | 277 | 263 | 261 | 258 |
| decomposition R1: largest exhaust pair (exhaust C2 -> air C1) (6) | 28 | 21 | 20 | 18 |
| decomposition max-kWh scenario: leave out exhaust A / C1 / C2 (not quoted in docs/19) | 1858 / 1696 / 1696 | 1422 / 1591 / 1832 | 1322 / 1528 / 1858 | 1107 / 1393 / 1858 |
| sensitivity R1: exhaust C by annual figure (6) | +398 (+3.6) | +175 (+1.6) | +139 (+1.3) | +84 (+0.8) |
| sensitivity R1: `dT_min` 10 K (6) | +249 (+4.9) | +249 (+4.9) | +250 (+4.9) | +254 (+5.0) |
| sensitivity R1: `dT_min/2`, compressor 90-50 (6) | +781, +385 | unchanged | unchanged | unchanged |
| sensitivity R2: exhaust C by annual figure; `dT_min` 10 K (6) | +512; +118 | +502; +118 | +500; +118 | +499; +118 |
| sensitivity R3: `dT_min` 10 K (table rank 3) | +247 (+3.2) | +114 (+1.5) | +88 (+1.2) | +51 (+0.7) |
| sensitivity R3: exhaust C by annual figure | 0 | -29 | -29 | -29 |
| sensitivity R3, R4: every other variant; base savings 29 / 1245 | -- | unchanged (< 0.5) | unchanged | unchanged |
| R1 variants low fraction 0.1, weather -10 K, hall 40-30 (see section 6) | +114, +97, +19 | +119, +103, +22 | +120, +104, +23 | +121, +105, +23 |

Over the whole main grid, compared label by label:

| `x_s` | scenarios changed by > 0.5 kWh/d | largest change of `E_direct` | largest change of the saving |
|---|---|---|---|
| 0.60 | 242 | +119 kWh/d | 14 kWh/d (0.10 points) |
| 0.634 | 244 | +173 kWh/d | 41 kWh/d (0.31 points) |
| 0.70 | 254 | +319 kWh/d (in a scenario with `E_direct` 15 850 kWh/d) | 81 kWh/d (0.60 points) |

All changed scenarios are on the product-1 day; no product-2 scenario changes. The largest changes
are in site-C, flat-mode, cold-weather scenarios.

## 5. Reading

* **The drier exhaust has 24-41 % less heat, but almost none of it was usable.** The lost part is
  mostly latent heat below the dew point (product 2: about 34 degC at the outlet). With
  `dT_min` 20 K it could serve only sinks colder than about 14 degC.
* **Product 2: nothing changes.** Presumably compressor heat already covers the cold end of
  the process-air demand, so no `E_direct` changes (a reading; not checked separately).
* **Product 1 (larger air demand, colder intake): `E_direct` rises slightly.** In scenarios where
  exhaust heat is used at the cold end of the air sinks, the change is at most 2 % of `E_direct`.
  It always stays far below the 5 % threshold (product-1 grid maximum 2.69 %).
* **Exhaust-dependent variants move.** The R1 variant "exhaust C by annual figure" fixes the
  total exhaust duty of plant C, so a drier exhaust means a larger flow of mostly sensible heat.
  Its effect shrinks by a factor 2-5. The R3 variant `dT_min` 10 K lowers the threshold at which
  exhaust heat becomes usable, so it depends on the exhaust humidity. Neither changes the
  conclusion. In the R1 ranking the annual-figure variant drops from rank 2 to rank 4 (0.60,
  0.634) or rank 6 (0.70). Question 4 to the partner (volume flow and annual figure read
  together) keeps its place.
* The pre-registered gate rows and the post-hoc R3/R4 references show the same picture. The
  results do not depend on whether 0.60 (lower end of the band, not admissible for every product) is counted.

## 6. Side observation (Befund 99 text against its own artifact; not caused by this re-check)

docs/19 section 6 lists the R1 sensitivity as dT_min/2 +781, annual +398, compressor 90-50 +385,
`dT_min` 10 K +249, "everything else 0". The Befund 99 artifact (`gate_storage_timeslice`,
`sensitivity.R1.ranking`) has four more non-zero entries: low fraction 0.1 +114 (+1.0 points),
weather -10 K +97 (+0.9), product day 2 +29 (+0.2), hall temperatures 40-30 +19 (+0.2). None of
them changes a reading. It is reported here for a possible dated addendum; nothing was changed.

## 7. Suggestions for the managing session (not done here)

1. Befund 99 needs no correction of its readings. A dated addendum could note:
   - the admissible solid fraction (band 0.60-0.70, common value 0.634; product-specific minima kept locally; docs/23) and this re-check;
   - that product 2 is not capped at saturation with an admissible value;
   - the moved sensitivity numbers (R1 annual figure +84 to +175 instead of +398; R3 `dT_min`
     10 K +51 to +114 instead of +247);
   - the side observation of section 6.
2. The placeholder band of the solid fraction in `storage_timeslice.config.PLACEHOLDER_BANDS`
   (0.2-0.5) is left unchanged (CLAUDE.md: no unilateral change of defaults or bands). If it is
   changed later, the Befund 99 artifacts no longer reproduce from the default config without
   passing 0.3 explicitly.
