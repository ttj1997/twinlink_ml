# Design of the corrections of review findings A1 and K1 (draft, NOT implemented)

Status: draft of 2026-10-09 for the decision of the project lead. Nothing of this is implemented; `src/`
is unchanged. Findings: `docs/12_review_2026-10-09.md`; checks: Befund 78 (`docs/04_stand_ap1.md`);
the size of the effect on every Befund that uses the cascade: artifact `review_a1_inventory_2`.

## 1. A1: minimum approach temperature along the exchanger

**Today.** `heatnetwork/balance.py`, `evaluate_cascade`, per stage: the sink is heated to
`min(set point, T_source,current - dT_min)`, i.e. only the warm end of a counter-current stage is
checked. Condensation is a flash at `T_kalt`: the source cools sensibly to `T_kalt` and releases all
latent heat there. Befund 78: the stages line exhaust -> S1 of the reported M3' solutions have an
approach of 0.14-15.7 K along the exchanger at `dT_min` = 20 K.

**Proposal.**

1. **Config.** `HeatNetworkConfig.approach_check: Literal["along_exchanger", "warm_end"] =
   "along_exchanger"` (frozen dataclass, validated in `__post_init__`). `"warm_end"` is the legacy mode,
   kept only to reproduce old artifacts (regression test, see 4.). A second field
   `approach_profile_nodes: int` (proposal 400) sets the discretisation below the dew point.
2. **Source profile** (new, in `balance.py`, vectorised over the time-step axis): the heat released
   down to `T` is `C (T_out - T) + m max(0, Y - Y_s(T)) dh_v(T_kalt)` in the condensing mode (the totals
   are those of MA1, so `source_available_power_kw` does not change), linear down to
   `T_dp + dT_approach` in the sensible mode. Nodes: `T_out`, the dew point as an exact node (as
   `pinch.hot_segments_for_source` does), `approach_profile_nodes` nodes between the dew point and the
   floor; linear between the nodes. Built ONCE per source and call (`build_source_profiles`), passed to
   `evaluate_cascade` as an optional argument so that `site_network.evaluate_site` builds it once for
   all 441 interconnections.
3. **Stage duty.** Largest `Q <= min(set-point duty, remaining source heat, shared cap)` with
   `min over the stage of (T_source(q) - T_sink(q)) >= dT_min`, counter-current. With
   `g(q) = T_source(q) + q / C_sink` the condition is
   `min_{q in [q0, q0 + Q]} g(q) >= dT_min + T_sink,in + (q0 + Q) / C_sink`; the left side falls and the
   right side rises with `Q`, so the root is found in closed form from the running minimum of `g` over
   the nodes plus one linear segment (the function `max_duty_along` of
   `scripts/check_review_a1_cascade_approach.py`, checked analytically and against brute force),
   vectorised over the time steps.
4. **Condensation along the exchanger.** The latent heat is released along the profile from the dew
   point on, not as a flash at `T_kalt`; `source_condensate_kg_per_s` follows from the profile
   (`Y - Y_s(T_out,stage)`), the sensible-first rule of MA1 is replaced in the new mode (kept in legacy).
5. **Diagnostics** (new arrays of `HeatNetworkResult`): `stage_min_approach_k`, `stage_pinch_position`
   (0 = warm end, 1 = cold end). `BINDING_PINCH` then means "the minimum along the stage binds".
6. **Unchanged:** `pinch.py` (the problem table shifts by `dT_min / 2` and is correct; the inventory
   reproduces the pinch targets of Befund 60), `SharedCap`, routing, the fuel formula.

**Run-time.** The check script needs about 0.6 ms per cascade evaluation (Python loop per stage), against
0.06-0.08 ms of today's condensing cascade (Befund 69). With profiles built once per call, a vectorised
implementation is expected at 0.1-0.3 ms; the site evaluation of 441 interconnections per objective
call then costs about as much as the two line solves (2 x 0.2 s). To be measured before any
re-optimisation and written into `docs/09` as an estimate.

**Tests (new or changed).**

- *Cold end binds:* one dry source (dew point below the floor), `C_sink > C_source`, `dT_min` 10 K;
  expected duty `C_source (T_out - T_sink,in - dT_min)` exactly (cold end = 10 K, warm end > 10 K);
  the legacy mode gives more (warm-end limit), which the test also states.
- *Pinch inside the exchanger:* a condensing source and a sink with `C_source < C_sink <` the effective
  capacity rate below the dew point, so that the tightest point is the dew point; expected duty
  `C_source (T_out - T_dp) + C_sink (T_dp - dT_min - T_sink,in)` exactly, position strictly inside
  (0, 1), both ends above `dT_min`.
- *Property test:* random networks (sources, sinks, orders, caps): for every stage with duty the
  minimum approach on an independent fine grid (brute force) is at least `dT_min - 1e-9`.
- *Monotonicity:* the new check never recovers more than the warm-end check (per stage and in total).
- *Legacy:* `approach_check="warm_end"` reproduces stored figures bit for bit (rows of
  `m4c_confirm_lab_umin126`, `m2_site_network_l015_umin126`, `run_pinch_benchmark`).
- Existing tests that encode the warm-end or flash semantics
  (`test_no_stage_violates_the_minimum_pinch`,
  `test_condensate_released_only_after_sensible_reserve_exhausted`, the EA1 formula tests) run in the
  legacy mode; their new-mode counterparts check the cold end and the condensation from the dew point.

## 2. K1: lower bound of the bed-temperature search

**Today.** `simulation/steady_state.py`: outer bisection on `[T_amb + 10, max(T_in, T_susp) + 5]`.
Befund 78: 301 of 1327 reported line solutions lie on `T_amb + 10`; below it the default solver fails
although valid drying states exist (all 211 probes at -1 K, with brackets from `T_amb`).

**Physics of the lower bound.** The inlet air carries the ambient absolute humidity
`Y_in = Y(T_amb, phi) <= Y_s(T_amb) < Y_s(T_bed)` for every `T_bed > T_amb` and `phi <= 1`, so the
condensation branch of `evaporation_rate_kg_per_s` (`Y_in > Y_s(T_bed)`) cannot occur above the ambient
temperature. The reason given in the code comment and in `oracle_search.evaluate_point` holds below
`T_amb`, not in the band `T_amb ... T_amb + 10`. Consistent with the probes: the lower edges
`T_amb + 0, + 3, + 5` worked in every probe, `T_amb - 5` failed in 64 of 212.

**Proposal (fallback, keeps every old result bit for bit).**

1. Evaluate the outer residual at `T_amb + 10` as today. If the bracket `[T_amb + 10, high]` has a sign
   change, the bisection runs exactly as today (same evaluations, bit-identical result).
2. Only if the residual at `T_amb + 10` is already negative (the root lies below): bisection on
   `[T_amb + delta, T_amb + 10]` with `delta` = 0.5 K (named module constant, proposal; any value in
   (0, 10) is admissible by the argument above; 0.5 K keeps a distance from saturated inlet air at
   `phi` -> 1). Without a sign change there either, `RuntimeError` as today (a genuine failure: no
   drying state above the ambient temperature).
3. **Regime check after the solution** (new entries in `infeasibility_reasons`, so that the solution is
   flagged, not silently accepted): drying requires `Y_in < Y_out < Y_s(T_bed)`; a solution within
   1e-6 K of the lower edge of the bracket used is reported as "on the search bound". Both are new
   fields of the diagnostics (`drying_regime`, `lower_bracket_celsius`, `on_lower_bracket`).
4. `evaluate_point` keeps passing no bracket of its own; its docstring and the code comment are
   corrected (the reason above).

**Tests (new).** The K1 probe point (line A, 35 degC / 0.25, `T_in` 73.25 degC, 1.26 m/s, 1.0 bar):
today `RuntimeError`, new `T_bed - T_amb` = 9.20 K, drying, feasible. Bit-identity: a sample of stored
solutions that are NOT on the bound (from `review_k1_solver_bracket`, across the 17 artifacts)
reproduces bed temperature and moisture with `==`. On-bound solutions: unchanged within 1e-5 K. The
regime check flags a state with an explicit bracket below `T_amb`.

## 3. Fingerprints and comparability of old results

- **`ProcessConfig.fingerprint`** is NOT changed by either correction (K1 is solver code, A1 is in
  `HeatNetworkConfig`); the config-fingerprint regression tests stay green. The fingerprint therefore
  does NOT distinguish old from new results. **Proposal:** the provenance block of the scripts
  (`scripts/verify/_common.provenance`) gets two fields, `solver/bed_temperature_bracket`
  (`"T_amb+10"` / `"fallback_T_amb+0.5"`) and `heatnetwork/approach_check`; the site scripts
  additionally log the `HeatNetworkConfig.fingerprint` (no artifact logs it today, checked).
- **`HeatNetworkConfig.fingerprint`** changes for every network (a new field enters the hash, also
  with the legacy value). No test pins a value and no artifact stores one (checked), so nothing breaks;
  the change is semantic and should be visible.
- **K1 with the fallback:** every stored solution reproduces bit for bit (the old solver succeeded
  there); results change only where the old solver failed, i.e. through re-optimisation (the warm
  points of Befunde 69, 73, 76, 77; possibly also failed table nodes of the cold, slow corner, Befund
  73 b, to be checked). Old figures stay valid as evaluations; optima on the bound were constrained by
  a solver convention (Befund 78 g).
- **A1:** every figure computed with the cascade changes. Size at held set points
  (`review_a1_inventory_2`, 20 K / 10 K along the exchanger): Befunde 57 d and 60 unchanged in the best
  case and the pinch gap (only case 2 changes); 69 b LB2 stays met; 70 routing effect smaller, set-point
  effect larger; 71 the gate stays met; 73, 76, 77 the gains vanish at the reported solutions. Old and
  new figures are not mixed in one comparison; the legacy mode exists only for the reproduction test.
- **Lab book:** old entries stay as written (append-only); every Befund re-measured after the
  corrections gets a dated addendum that names the new artifact and the mode.

## 4. Order of work proposed (after release)

1. K1 fallback and its tests (small, bit-identical for old results), full suite.
2. A1 along the exchanger with the legacy mode and the tests above, full suite; run-time measured.
3. Provenance fields (3.), then re-optimisations in the order the project lead decides (the
   confirmatory test of Befund 77 first, with a criterion fixed beforehand).
