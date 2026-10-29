# Inventory and design draft for a multi-line RL environment (RU1-RU3)

> Status: **filed 2026-10-02 as supplied by the project lead; reconciled with
> `docs/09_entwurf_modellbasierte_optimierung.md` (MO1-MO3) at the end.** Written before the
> change of the main path (Befund 68 closure, 2026-10-02); since then RL is the comparison arm
> and this plan describes extension X2 of the new plan, not its core.
>
> Provenance of numbers: every number in RU1-RU3 below is **as supplied (RU)**. I have not
> re-derived them from findings and artifacts unless a line says "checked". Under the
> documentation rules of `CLAUDE.md` a number from this file may enter a synthesis only after
> it has been matched to a finding and an artifact. Time figures are estimates by the author
> of RU, not from artifacts. The text was translated and condensed from the German original
> supplied in the task; no content was added to RU1-RU3 themselves.

## Points that touch the task (as supplied)

1. There is no dryer model. The only plant model is the fluidised bed. The spray tower (Q1) is
   a fixed set of constants from the partner model in Befund 60/61 (20 kg/s, 125.233 degC,
   Y = 0.06482). "Several drying and granulation plants" today means: several fluidised-bed
   instances plus fixed external sources and sinks.
2. The ensemble band of Befund 65 FA2 carries a margin only as a global constant. Coverage is
   valid (94.6 % excited-trained, 96.1 % constant-trained on EV3), but the rank correlation
   between operating-point offset and spread is negative (-0.43 / -0.30) and criterion (iii)
   fails in both arms: the band is not usable per operating point, a state-dependent margin
   cannot be derived from it.
3. The result of the simplest stage is still negative: the SAC agent in the one-step bandit
   ends at step 10 000 with a pooled closed fraction of -2.80 (median over 3 seeds), worse
   than the fixed operating point. The four YA2 options against penalty dominance (factor 516)
   were neither chosen nor tested. Without a positive bandit result, multi-plant RL and
   storage are not viable.
4. Nothing can be built on Befund 61 (suspended, R5): every plant-scale calculation (QA1-QA4,
   RA1-RA3) lapses. Befund 60 stays valid, because it rests on the partner-model data case and
   not on the NA5 parametrisation.
5. The target "total energy demand" has a problem: Befund 57i shows that the kW sum is the only
   variant that drives the gas velocity to its upper bound; it treats electricity and gas alike
   without a substantive reason.

## RU1 Inventory

### Reusable

| Building block | State | Gap for the goal |
|---|---|---|
| Bandit env (`rl/env.py`) | bit-identical to Befund 52, 2 actions, 2 disturbances | one plant; loads `scripts/study_operating_point_regret.py` via importlib; uses the critical moisture both as kinetics and as quality limit (`env.py:473`) |
| Evaluation (`rl/evaluation.py`) | pooled closed fraction, feasibility, boundary/interior | tied to the 25 grid points and the 2-D case |
| Cascade (`balance.py`) | sensible and condensing, dew point per source, pinch approach | `T_kalt` is network-wide, the partner model has 10 and 15 degC per source; no storage |
| Routing (`MatrixRouting`) | intended as action space | assignment and priority only, no split model (EA6) |
| Pinch (`pinch.py`) | exact lower bound per operating point | static and timeless, contains neither storage nor process optimisation |
| Net objective (`net_energy.py`) | kW sum and CO2-weighted | electricity and gas not booked per carrier; the process heater is electric (eta 0.98), in the network it is gas (eta 0.80 or 0.90) |

### Placeholders that carry results (as supplied)

- **Critical moisture 0.20 kg/kg:** dual role, kinetics parameter (must exceed the GAB lower
  bound 0.0889) and quality limit. Befund 64f decouples this only in the probe scripts. All
  margin numbers are relative to this value.
- **d50 = 750 um:** a population-balance result (nucleus diameter 150 um, nucleus fraction
  0.008); constant over the whole action and disturbance space; fixes the fluidisation window.
  At laboratory scale it never binds; in the plant grid 15 of 27 points failed at the same d50
  (13 alone, 2 together with moisture). The contradiction is resolved nowhere; Befund 62 calls
  it "compatible".
- **`T_kalt` = 15 degC:** optimistic; 29.1 degC cooling water of the compressor station is a
  hint for the real cold side; from 25 degC on Q2 no longer condenses (dew point 24.92 degC).
  The minimum heating demand was independent of `T_kalt` in the cases checked, but only for
  this data case and two lower-regime points, and those depend on Befund 61.
- **Wall loss:** `heat_loss_ua_w_per_k` = 2.0 W/K is a lab placeholder;
  `outlet_duct_heat_loss_k` = 0.0, no value was requested. In the data case the pinch sits
  bit-identically at `T_out` of Q2, so a duct loss moves it directly.
- **Solid fraction:** a disturbance in the lab (0.2-0.4, fixed at 0.30 in the bandit). The plant
  value 30 % is not tenable according to Befund 56 NB2, which predicts 71-73 % (unconfirmed).
  With the corrected air flow the solver finds no steady state at product 2.
- Others: `dT_min` = 20 K (two-stage circuit partly confirmed), `eta_H`, `H_u`, bed limit
  80 degC, penalty weight 50 kW.

What the suspension means for plant scale: a plant-scale environment has no reliable
parametrisation; real solid fractions and air flows from the application partner are missing.
Until then the laboratory scale with a scaling computation (0.03 versus 8 kg/s, two orders of
magnitude) remains; every kW is scaling, not a plant statement. The pinch migration
44.94 <-> 73.50 degC is only a qualitative hypothesis now.

### Step speed (as supplied)

- Grey box, steady state: 164 ms per environment step, single thread, with search precision;
  10^4 steps about 0.46 h, 10^5 steps about 4.6 h. The centred reward roughly doubles that;
  several plants multiply it.
- Grey box, transient: not measured as an environment step. From EV3 (50 runs of 1152 steps in
  93 s) and the holdout (36 runs in 20 s) the order of magnitude is milliseconds per 25-s
  step; the worker count was not checked.
- Learned surrogate (TE3a), throughput at batch 4096 and one thread, latency at batch 1 not
  measured: LSTM 51.5 us per window, Mamba 322, xLSTM 901, Transformer 2394. The models deliver
  only energy, bed temperature and moisture; `Y_out` and `T_out` would have to be reconstructed
  via `reconstruct_exhaust_state`, not checked.
- Surrogate error: constant-trained NRMSE gap factor 2.2-3.1; reserve at the moisture limit
  point 0.957 as a single network, 1.93 as an ensemble.

## RU2 Design (RL environment)

**Time scale.** The bandit stays as test bench (Befund 62 FA4). The actual environment is
sequential, even without storage. `max_rate_per_s` limits the set points; d50 has tau of about
4 h, bed mass 600 s, moisture and temperature 15-80 s; Befund 53 fixes batches of 28 800 s.
Proposal: decision interval 15-60 min, episode 8 h, hence 36-144 simulator steps of 25 s per
decision.

**State.** Per plant: bed temperature, moisture (estimate), d50, current set points.
Environment: ambient temperature, relative humidity (a stochastic process in the simulator,
placeholder), time of day. Network: sink demand. With storage additionally level and
temperature.

**Actions.** Per plant `T_in` and `u_gas`; spray rate and nozzle pressure stay fixed (throughput
and clamp artefact, Befund 62). Heat distribution as an assignment; continuous splits would
need an extension (EA6). With storage: charge and discharge.

**Reward.** Negative net energy demand per interval, booked per carrier (electricity: fan,
pump, compressor; gas: air heater minus recovery). Reported: CO2-weighted and kWh sum from the
same run.

**Product-moisture constraint.**
- *Constant margin:* threshold `X_limit - m`, the mechanism of Befund 64/65 FA3, tested there
  with an ideal model. Two sources for `m` from FA2: the band half-width 2k * spread (median
  0.0060 excited-trained, 0.0082 kg/kg constant-trained), the coverage quantity; and the margin
  against offset plus 3 * spread (E4, 0.0056 and 0.0088 kg/kg), the offset quantity. Both lie
  below the 90 % mark value 0.00954 kg/kg of GA1. For comparison: at margin 0.0045, 95.4 % of
  the control benefit remains, at 0.01 it is 89.5 % (Befund 64 decoupled, GA1).
- *No penalty term for the edge behaviour:* a penalty-based agent keeps its own margin of
  0.14 kg/kg (Befund 62 GA1), a factor 31-86 above the 3 x RMSE margin, and this price explains
  most of the pooled fraction -2.2 to -2.8 (Befund 64 KA3). The band margin only acts if the
  constraint handling lets the agent move to the separate threshold: Lagrange (YA2 option 3) or
  projection. Open, belongs in stage 0.
- *Decoupling the quality limit from the kinetics:* changes the config fingerprint, hence
  report first and have it confirmed.

**Comparison quantities.** Fixed operating point: per plant one fixed point as Befund 52; for
the network one common point optimised over the disturbance distribution. Rule-based heat
distribution: the default cascade `resolve_cascade_order`, plus case 1 (separate, Q1 -> S1 and
Q2 -> S2). Optimum where computable: without storage an oracle per disturbance instance as
Befund 52 (DE + Nelder-Mead; the network is cheap, the process solver dominates, with 4-6
instead of 3 variables noticeably more expensive; Befund 52: 520-580 s per point); the pinch as
an additional lower bound per operating point (Befund 60). With storage: optimum with perfect
foresight over the horizon (DP or LP) as an upper bound of the benefit; not a construction rule.

**How much room there is.** Befund 60: the best fixed connection is only 2.3-4.2 % above the
pinch bound in the data case, so little is to be gained from heat distribution alone; the gain
must come from set points and time shift. The regret of the fixed point is 16 % (median) and
25.8 % (maximum) gross; net CO2-weighted (V3) 8.6 %, net kW sum (V5) 128 % with an unstable
denominator. Rank claims only after comparison with the resolution limit of the seeds; always
name the test set.

**Stage plan (RU, today = 2026-10-01, conference in about 39 weeks).**

| Stage | Content | Time |
|---|---|---|
| S0 (gate) | make the bandit work: Lagrange or projection, decouple `X_limit` (after confirmation); criterion in advance: feasibility 1.0 and closed fraction > 0 above chance, 3-5 seeds, learning curve | 2-3 weeks |
| S1 | one plant, sequential, transient, with rates, disturbance series, net target per carrier, constant margin; measure environment step cost first; baselines before RL | 3-4 weeks |
| S2 | two fluidised-bed lines with a shared exchanger, sinks fixed, split extension, `T_kalt` per source; comparison with pinch and oracle | 4-6 weeks |
| S3 | storage: new module in `heatnetwork/`, none exists so far | 4-6 weeks |
| S4 | optional surrogate (only after validation against the grey box), final evaluation, text | 3-4 weeks |
| Buffer | | 4-6 weeks |

With buffer this is tight. If S0 or S2 fails the deadline, the viable result is: sequential
single plant with constant margin plus Befunde 62/64/65.

## RU3 Open decisions (as supplied)

- **Target measure.** CO2-weighted as primary, booked per carrier, kWh sum from the same run.
  Reason: Befund 57i; a policy trained on the kW sum alone aims at a point that is worse under
  cost and CO2 evaluation. The conference commitment "total energy demand" stays satisfiable.
- **Number and kind of plants.** Two controllable fluidised-bed lines (different
  parametrisation) plus fixed external source and sinks from the partner model. A spray-dryer
  model would be new physics without a data basis beyond the constants, probably 6-10 more
  weeks; only if the commitment "dryer" means it literally.
- **Storage.** Yes, but as the last stage (S3) and only if S0-S2 are on plan; environment built
  so that storage can be switched off. A hot-water buffer in the intermediate circuit is
  confirmed real (Befund 56).
- **Grey box or surrogate.** Grey box for training and all reported numbers, surrogate only
  after S2 and only after a cross-check: an RL agent finds errors of a surrogate, and Befund 65
  measured the error only at the points chosen by the oracle, not where an agent explores.
- **Data from the partner heat model.** Use only the constants already recorded in the
  repository (Q1, Q2, S1, S2, `H_u`, `eta_H`, `T_kalt` 10/15 degC). To be asked: the real
  `T_kalt`, `dT_min`, storage geometry and temperatures, demand time series of the sinks.
  Neither the Excel file nor raw data enter the repository. Missing series are generated
  synthetically and marked as placeholders.
- **Decoupling kinetics and quality limit.** Yes, in stage 0; changes the config fingerprint and
  touches Befunde 52, 53, 54, 57 and 62; per `CLAUDE.md` report first and have it confirmed.
- **Clean-up before extending.** Move `check_constraints` and the penalty weight from
  `scripts/` to `src/` before a second environment builds on them; the VA2 oracle test must
  stay bit-identical.
- Provenance note as supplied: the git status showed about 45 modified, uncommitted scripts and
  `point_split.py`; before any later run that writes artifacts the tree must be clean.

## Checks of RU1-RU3 against the repository (2026-10-02)

| Statement in RU | Check | Result |
|---|---|---|
| env loads the regret script via importlib | `rl/env.py` has no importlib any more | **outdated**: moved to `simulation/operating_point.py` in S0.1 (Befund 68 b) |
| env uses the critical moisture as quality limit (`env.py:473`) | `BanditEnvConfig.product_moisture_limit_kg_per_kg` and `MaterialConfig.product_moisture_limit_kg_per_kg` exist; default `None` = coupled as before | **partly outdated**: the decoupling exists as an optional field, fingerprints of all three configs unchanged (compared before/after, fixed as regression test, Befund 68 b); the config files themselves still couple |
| decoupling "changes the config fingerprint" | the field enters the fingerprint only when set (Befund 68 b) | **outdated** for the default; a run that sets it carries a different fingerprint, as intended |
| move `check_constraints` and penalty weight to `src/` | done in S0.1 (`operating_point.py`) | **done**; the DE/NM helpers (`OptimizationBudget`, `_optimize`, `optimize_grid_point`, `find_fixed_operating_point`) are still in the script |
| about 45 modified uncommitted scripts and `point_split.py` | `git status` at the start of this session: clean; now only the documents of this task | **not reproducible** here; probably another working copy |
| pooled closed fraction -2.80, median of 3 seeds | Befund 62 FA3, step 10 000: median -2.80 (lab book, per-seed values -1.999 / -2.797 / -3.012) | **checked**; the later 5-seed arms of Befund 68 (e.g. penalty arm -2.136) are a different protocol |
| YA2 factor 516 | lab book Befund 62 m) "Verhaeltnis (c)/(a) = 516" | **checked** |
| S0 gate: Lagrange or projection, criterion in advance | done as S0.3-S0.9 (Befund 68); gate not met, multiplier never active | **superseded** by the closure of Befund 68 |
| 164 ms per environment step | not found in a lab-book entry | **unchecked** |

## Reconciliation with MO2 (`docs/09_entwurf_modellbasierte_optimierung.md`)

| Aspect | RU plan | MO plan | Difference / resolution |
|---|---|---|---|
| Main path | RL; S0 bandit gate decides | model-based optimisation; RL is comparison | RU S0 is **closed negative** (Befund 68); RU S1-S3 become extension X2 and are not in the core |
| Core system | two fluidised-bed lines, fixed external source and sinks | the same | **agreed** |
| Time resolution | sequential, transient, decision interval 15-60 min, episode 8 h | steady state; transient check on about 5 scenarios (M6) | **different**; MO is cheaper by orders of magnitude per optimum; the transient check covers the known deviation (batch-averaged `Q_rec`, Befund 52 d); the sequential problem is only needed for RL or storage |
| Storage | S3, 4-6 weeks | extension X1, not in the core | **different in priority**; recommended after the core, because a complete earlier contribution does not need it |
| Heat allocation | assignment plus split extension in S2 | enumerated discrete routings; split as X3 | **different**; MO avoids new code in the core |
| `T_kalt` per source | S2 | network-wide in the core; per-source via the probe-script route; sensitivity in M5 | **different**; `T_kalt` per source is needed to match the partner model (10/15 degC) and is flagged as D14 |
| Margin | constant; two candidates (0.0060 / 0.0082 band half-width, 0.0056 / 0.0088 offset + 3 spread) | 0.00596 (band half-width, excited-trained, as S0.3); 0.0082 as an upper sensitivity | **consistent**; the offset-quantity variant is not used as a main case |
| Constraint handling | Lagrange or projection (open) | hard constraint of the optimiser at `L_q - m` | **different by construction**: the optimiser does not need a multiplier; this is the reason for the switch |
| Quality limit | decouple in stage 0 | decoupled via the existing optional field; `L_q` = 0.15 main, 0.20 stress (D3) | **agreed**; no fingerprint change for the default |
| Objective | CO2 primary per carrier, kWh from the same run | the same; gas-fired heaters main, electric sensitivity (D5) | **agreed**; RU adds the heater mismatch, which D5 resolves |
| Surrogate | S4, optional | extension X4 | **agreed** (after validation, error measured where an optimiser explores) |
| Scale | lab scale with scaling computation | lab lines, linear scaling to the exhaust flow of the partner model's fluidised-bed source (D4) | **agreed**, made concrete |
| Total duration | 4 stages + buffer, 18-29 weeks plus 4-6 buffer, "tight" in 39 weeks | core 28 working days net (about 8-9 weeks with overhead), extensions use the remaining weeks | **different**; MO's core fits a deadline several months earlier; the RU sequence does not |
| Earliest presentable stage | none named | M1 (single line, cost of the margin in CO2 and kWh, with the RL negative result) | **new** |
| Clean-up before extending | move helpers to `src/` | M0 | **agreed**; part is done (S0.1), the DE/NM helpers remain |
| Checks of the optimiser | oracle per instance | table method checked against the direct method on at least 12 optima, at least half at the binding moisture limit (M3) | **new**, requested by the project lead |
| Test-set naming, resolution limit, seeds | stated as rules | deterministic optimiser: resolution = table spacing and DE seed spread on a subset; RL arm uses the SX1 thresholds of the committed runs | **consistent** with `CLAUDE.md` |
