# 30 -- Learned sequence models as the environment for control: how strongly do RL and an optimiser exploit the model errors?

Status: design fixed and committed before the runs (sections 1-6); runs and results in sections
7-9, added after the runs (numbers only from the artifacts named there). Branch `surrogate-env`
(from `origin/master` a84b764). Finding number: 113 (lab book).

## 1. Question

Every RL and RTO result up to here was trained or optimised on the grey-box model itself (step 0
of the request, section 6). Befund 107 optimised on static surrogates (quadratic surface, GP, MLP
ensemble, grey-box fit) of the steady model; Befunde 111/112 used a GRU only as a soft sensor
inside a loop whose optimiser ran on the nominal grey-box model. Here the learned **sequence**
models of the Befund 94 pipeline become the environment: does a controller trained or optimised on
them stay feasible and economical on the true (grey-box) model, and which of RL and RTO exploits
the model errors more?

## 2. Problem (unchanged from Befund 96)

The one-step operating-point problem of Befund 62/93/96: context = ambient temperature and relative
humidity (ranges of the process config), action = inlet temperature [60, 120] degC and gas velocity
[0.8, 2.5] m/s; nozzle pressure fixed at 1.002634656115166 bar, spray rate 1.5 kg/h and solid
fraction 0.30 nominal; constraints product moisture <= 0.15 kg/kg (placeholder) and bed temperature
<= 80 degC; objective `energy_target` (kW). Test set: the 25 Befund-52 grid points, always named.
Truth and evaluation: always the grey-box steady state against the TRUE limit 0.15.

## 3. Environments and the mapping of a learned model to an operating point

| | environment | data |
|---|---|---|
| U0 | grey-box steady state (reference) | -- |
| U0' | the mapping below applied to the grey-box model | -- |
| U1 | composite GRU of the Befund 94 pipeline (arm N, preset version 2) | `n_train` 16 |
| U2 | the same | `n_train` 64 |
| U3 | ensemble of 5 GRU, moisture of the constraint `mean + 2 std` | the 64 runs of U2 |

Three replicates: GRU master seeds 1, 2, 3 (data and split of Befund 94); RL seed = master seed - 1.
The U2/U1 networks are the networks of Befund 94 (same pipeline, same seeds), retrained because the
checkpoints of Befund 94 are not kept; they are checked against the Befund 94 records. The four
further members of U3 differ only in the training seed (`_derive(model_seed,
"surrogate_env_ensemble", j)`), so the ensemble sees 64 runs, not 5 x 64.

**Mapping (decision of the project lead, 2026-10-27).** The request named a rollout with constant
inputs over the horizon h36 of Befund 94. A probe before the design (scratch, not an artifact)
showed that the grey-box model has the time constants 43-55 s, 600 s, 1600-2900 s and 16 000 s
(`time_constants` at the Befund-52 grid points); after a step from the fixed operating point to the
oracle point the moisture covers only about 30 % of the step within 900 s (for example 0.123
instead of the steady 0.150 kg/kg at 5 degC / 0.25). A plain h36 rollout would therefore turn the
steady problem into a transient one and let every controller exploit the inertia instead of the
learning error. The mapping used is the **fixed point of the h36 map**: constant inputs over history
and horizon, constant target history `x`, and `x*` such that the prediction at h36 equals `x*`
(per head; the composite network has one independent network per target). For the grey-box model
that fixed point is the steady state of the solver, so U0' = U0; the control is that a 900 s
constant-input simulation started at the solver's steady state does not drift.

**Fixed-point criterion (before the runs; `src/twinlink_ml/surrogate_env/fixed_point.py`).** Per
head and operating point, in target-standardised units: a scan of `r(x) = f_h36(x) - x` at 65
points over [-5, 5] (the multi-start of the root search); valid iff exactly one sign change and it
is stable (`r` from >= 0 to < 0); refinement by a second 65-point scan in the bracket and linear
interpolation; converged iff `|r(x*)| <= 1e-3`. Statuses `no_root`, `multiple_roots`, `unstable`,
`not_converged` are counted per environment and reported, not discarded; for the controller such a
point is not admissible (RL: a solver failure of the environment, masked as in cell `ressep`; RTO:
the solver-failure value 1e6 kW). A point is valid only if all three heads (U3: all five members)
are valid.

## 4. Controllers

**C1 (RL)**, `scripts/run_surrogate_env_rl.py`: everything of Befund 96 (cell `ressep`: residual
action of radius 0.5 around the fixed point of Befund 93, separated moisture cost, Lagrange per
context cell 5 x 5, lambda_0 0.5 kW, eta 0.2 kW every 250 steps, margin `5 m` -- the margin
Befund 96 determined --, 30 000 steps, checkpoints 1000-30 000, one torch thread). SAC with the
stable-baselines3 defaults (as Befund 96); because Befund 96 used SAC only, TD3 in addition with the
stable-baselines3 defaults plus Gaussian exploration noise sigma 0.1 (the default of the TD3 paper;
the library default without noise collapsed to a box corner in Befund 68 S0.8 -- decision taken
without asking, stated here). SAC on U0 is run by the unchanged script of Befund 96, so its records
can be compared bit for bit.

**C2 (RTO)**, `scripts/run_surrogate_env_rto.py`: multistart Nelder-Mead (the fixed point and 7
scrambled Sobol starts over the action box; options of the oracle refinement) on the penalised
objective of the oracle (energy + 50 kW x normalised violation against `0.15 - k m`); the same
optimiser on every environment. Main `k = 5` (the margin of C1, decision of the project lead);
`k = 1` as a sensitivity without reading.

## 5. Metrics and pre-registered reading

Full text: docstring of `scripts/analyze_surrogate_env.py` (committed before the first run). In
short, per (environment, controller, replicate), on the grey-box model at the 25 points: energy over
the true optimum (oracle without margin at 0.15, `s05_limit015_oracle_and_fixed`; also over
Oracle(m) for the connection to Befund 96), share of violated points, largest moisture violation,
and the **exploitation gap** -- predicted minus true energy (kW, %) and moisture (kg/kg) at the
chosen actions. Median and range over three replicates.

Criterion of an environment for a controller: feasibility pooled over 3 x 25 cases >= 0.95 and the
median energy excess over the same controller on U0 <= 2 %.
**G0** U2 meets it -> learned models are usable as an environment.
**G1** U2 does not, U3 does -> only with an uncertainty measure.
**G2** U3 does not either -> learned models are not a sufficient environment for this task.
In addition: RL against RTO on the same learned environment (paired over the replicates, t with
2 degrees of freedom): which exploits more (energy gap, moisture gap, violation share).

## 6. Step 0 of the request (answered from docs/04 before the design)

(a) Algorithms of the RL findings: Befund 62 SAC (stable-baselines3 defaults, smoke run of the
pipeline); Befund 68 SAC (defaults; in the second addendum, S0.8, TD3 without and with action noise
0.1 and SAC with target entropy -6 as counter-tests); Befund 93 SAC (defaults); Befund 96 SAC
(defaults, cell `ressep`). (b) No RL or RTO was ever trained or optimised on a learned sequence
model (GRU/LSTM of Befund 94) and evaluated on the grey-box model. Closest: Befund 107 (RTO on
static surrogates of the steady model, GP / MLP ensemble / quadratic / grey-box fit, regret at the
true model) and Befunde 111/112 (GRU of the Befund 94 pipeline as a soft sensor in the loop of an
RTO on the nominal grey-box model). The study was therefore not stopped.

**Reproduction finding (recorded before the runs).** The SAC checkpoints of Befund 96
(`artifacts/checkpoints/rl_bandit_residual/*_step030000.zip`, named in the records) and the GRU
checkpoints of Befund 94 (`artifacts/checkpoints/rerun/N/`) are no longer on the machine (main
checkout, worktrees and `C:/twinlink_runs/` searched); the records keep no per-context actions.
SAC on U0 (seeds 0-2) and the GRU are therefore retrained; whether that reproduces the records
bit for bit is reported in section 7.

## 7. Runs and controls

**Commits and artifacts.** Code and pre-registration 883fb1a (before every run); GRU training
`artifacts/runs/surrogate_env_models_v1.json` (18 networks, 2538 s, commit 883fb1a); RL records
`artifacts/runs/surrogate_env_rl_{U0..U3}_{SAC,TD3}_seed{0,1,2}.jsonl` (U0 SAC:
`surrogate_env_u0_sac_k5_ressep_seed{0,1,2}.jsonl`, unchanged Befund-96 script) and RTO
`surrogate_env_rto_v1.json` (500 tasks, 23 732 s at 4 workers), committed in 5ac91c5; error map
`surrogate_env_error_map_v1.json` (573 s) and oracle / U0' control
`surrogate_env_oracle_check_v1.json` (1315 s), both run on 5ac91c5; analysis
`surrogate_env_v1_analysis.json` run on 759e4c5. Every artifact `git/dirty=False`. Two changes to
the analysis script after the runs, neither touching the reading: a descriptive summary of the
error map (842db81, labelled as added after the runs) and the key of the provenance block
(759e4c5; the first call aborted before writing anything). The U0 SAC traces (about 14.5 MB each)
and all checkpoints (186 files, 698 MB) are kept outside the repository in
`C:/twinlink_runs/surrogate-env/` (file counts and bytes checked after the copy).

**Run time and budget rule.** Pilot extrapolation before the runs: about 6 h wall time, below the
30 h of the rule, so no cut (U1 kept, three RL seeds). Actual: RL 42.3 CPU hours (U0 runs
11 579-11 664 s each, U1 2256-2415 s, U2 2106-2252 s, U3 8808-9396 s; up to 6 in parallel), RTO
6.6 h wall at 4 workers, the whole chain from the GRU training to the analysis about 9.5 h wall.
Counted in CPU hours (RL plus RTO at 4 workers, about 69 h) the budget would have been above 30 h;
the rule was applied to wall time. Open point for the project lead.

**Reproduction (step 0, the lost checkpoints).** (1) The six GRU of Befund 94 (master seeds 1-3,
n_train 16 and 64), retrained with the committed pipeline: every metric and every training entry
equals the Befund-94 records (`artifacts/runs/rerun_full_N.jsonl`), max abs difference 0.0 over 56
metrics and 18 training entries on each test set (excited and constant-run holdout); only the
timing metrics are excluded. (2) SAC on U0, seeds 0-2, retrained by the unchanged script: the
records equal those of Befund 96 (`rl_bandit_residual_b_k5_ressep_seed{0,1,2}.jsonl`) at all 7
checkpoints, 191 / 191 / 193 values, max abs difference 0.0; the step traces equal the committed
Befund-96 traces at all 30 000 steps of each seed. (3) The oracle of the margin 5 m
(`bres_margin_ref_oracle_k5`) recomputed at 10 of 25 points (grid indices 0, 2, 5, 7, 10, 12, 15,
17, 20, 24): identical at 10 of 10 (action, objective, energy, moisture). (4) U0' = U0: a 900 s
constant-input simulation started at the solver's steady state, at the true optimum and at the
fixed point of all 25 points, drifts at most 1.3e-8 kW, 2.4e-8 kg/kg and 3.7e-6 K. The losses
are therefore without consequence for these numbers; the weights themselves are lost, so the
equality is shown on the records, not on the weights.

**Fixed-point statuses (counted, not discarded).** RL training, invalid points (masked as solver
failures): U1 SAC seed 1: 7 of 30 000 (moisture `no_root`); U2 SAC seed 2: 3; U3 SAC seed 1: 2;
U3 with GRU master seed 3: SAC seed 2 7824 masked steps (bed temperature `multiple_roots` 7794
member evaluations, moisture `no_root` 69), TD3 seed 2 6705 (6604 / 286); every other run 0. At
the 25 test points the policies of master seed 3 on U3 meet an invalid prediction at 6 points
(counted as "invalid prediction"; the truth is still evaluated). RTO result points: all valid
(U1, U2: 150 of 150 per head; U3: 750 of 750). Error map (1225 grid points per model): U1 seed 2:
8 moisture `no_root`, U2 seed 2: 4, U2 seed 3: 8, U3 seed 2: 20, U3 seed 3: 55 moisture `no_root`
and 114 bed-temperature `multiple_roots`; others 0. No `unstable` or `not_converged` anywhere.

## 8. Results (test set: the 25 Befund-52 grid points; truth = grey-box steady state, limit 0.15)

Per (environment, controller): median (range) over the three replicates; U0 RTO is deterministic
(one replicate). "Energy gap" = predicted minus true energy at the chosen actions, pooled in %;
"moisture gap" = predicted minus true moisture (U3: ensemble mean, and the constraint moisture
`mean + 2 std`). RTO k=1 is the sensitivity without reading. Source: `table` of
`surrogate_env_v1_analysis`.

| env / controller | energy over optimum % | over Oracle(m) % | feasible share (pooled) | violated points | largest moisture violation kg/kg | energy gap % (pooled) | moisture gap kg/kg | moisture gap, constraint kg/kg | model-feasible, truly infeasible | invalid predictions |
|---|---|---|---|---|---|---|---|---|---|---|
| U0 / SAC | 6.82 (6.58 .. 8.05) | 6.91 (6.67 .. 8.15) | 1.000 | 0 of 75 | 0.0000 (0.0000 .. 0.0000) | -- | -- | -- | -- | -- |
| U0 / TD3 | 6.93 (6.92 .. 25.56) | 7.02 (7.02 .. 25.67) | 1.000 | 0 of 75 | 0.0000 (0.0000 .. 0.0000) | -- | -- | -- | -- | -- |
| U0 / RTO | 6.40 | 6.49 | 1.000 | 0 of 25 | 0.0000 | -- | -- | -- | -- | -- |
| U0 / RTO k=1 | -0.12 | -0.04 | 1.000 | 0 of 25 | 0.0000 | -- | -- | -- | -- | -- |
| U1 / SAC | 2.51 (1.86 .. 4.56) | 2.60 (1.95 .. 4.65) | 0.933 | 5 of 75 | 0.0041 (0.0000 .. 0.0139) | -1.61 (-1.95 .. -0.87) | -0.0150 (-0.0194 .. -0.0086) | -0.0150 (-0.0194 .. -0.0086) | 2 (0 .. 3) | 0 (0 .. 0) |
| U1 / TD3 | 4.60 (3.51 .. 25.56) | 4.69 (3.60 .. 25.67) | 0.987 | 1 of 75 | 0.0000 (0.0000 .. 0.0093) | -0.77 (-1.85 .. -0.04) | -0.0092 (-0.0146 .. -0.0082) | -0.0092 (-0.0146 .. -0.0082) | 0 (0 .. 1) | 0 (0 .. 0) |
| U1 / RTO | 3.07 (1.07 .. 4.28) | 3.16 (0.56 .. 4.37) | 0.933 | 5 of 75 | 0.0079 (0.0000 .. 0.0985) | -0.87 (-1.82 .. -0.46) | -0.0134 (-0.0252 .. -0.0078) | -0.0134 (-0.0252 .. -0.0078) | 1 (0 .. 3) | 0 (0 .. 0) |
| U1 / RTO k=1 | -3.30 (-5.41 .. -1.40) | -3.22 (-6.52 .. -1.89) | 0.200 | 60 of 75 | 0.0831 (0.0163 .. 0.3257) | -1.38 (-2.49 .. -0.96) | -0.0171 (-0.0478 .. -0.0082) | -0.0171 (-0.0478 .. -0.0082) | 20 (14 .. 22) | 0 (0 .. 0) |
| U2 / SAC | 4.61 (4.43 .. 5.82) | 4.70 (4.52 .. 5.91) | 1.000 | 0 of 75 | 0.0000 (0.0000 .. 0.0000) | -0.92 (-1.13 .. -0.55) | -0.0058 (-0.0062 .. -0.0051) | -0.0058 (-0.0062 .. -0.0051) | 0 (0 .. 0) | 0 (0 .. 0) |
| U2 / TD3 | 6.14 (5.79 .. 25.56) | 6.23 (5.88 .. 25.67) | 0.987 | 1 of 75 | 0.0000 (0.0000 .. 0.0147) | -0.50 (-1.09 .. -0.40) | -0.0049 (-0.0068 .. -0.0020) | -0.0049 (-0.0068 .. -0.0020) | 0 (0 .. 1) | 0 (0 .. 0) |
| U2 / RTO | 4.43 (4.27 .. 5.40) | 4.51 (4.36 .. 5.49) | 0.973 | 2 of 75 | 0.0116 (0.0000 .. 0.0118) | -0.89 (-1.05 .. -0.52) | -0.0080 (-0.0085 .. -0.0042) | -0.0080 (-0.0085 .. -0.0042) | 1 (0 .. 1) | 0 (0 .. 0) |
| U2 / RTO k=1 | -0.98 (-2.28 .. -0.77) | -1.47 (-2.77 .. -1.27) | 0.480 | 39 of 75 | 0.0498 (0.0054 .. 0.0507) | -1.09 (-1.20 .. -0.72) | -0.0074 (-0.0127 .. -0.0053) | -0.0074 (-0.0127 .. -0.0053) | 10 (7 .. 19) | 0 (0 .. 0) |
| U3 / SAC | 6.72 (5.16 .. 7.80) | 6.81 (5.25 .. 7.89) | 1.000 | 0 of 75 | 0.0000 (0.0000 .. 0.0000) | -1.01 (-1.07 .. -0.91) | -0.0038 (-0.0057 .. -0.0031) | 0.0005 (-0.0003 .. 0.0020) | 0 (0 .. 0) | 0 (0 .. 6) |
| U3 / TD3 | 7.39 (5.60 .. 7.71) | 7.48 (5.69 .. 7.80) | 0.973 | 2 of 75 | 0.0000 (0.0000 .. 0.0169) | -1.02 (-1.07 .. -0.85) | -0.0039 (-0.0062 .. -0.0031) | -0.0004 (-0.0007 .. 0.0020) | 0 (0 .. 0) | 0 (0 .. 6) |
| U3 / RTO | 6.46 (6.19 .. 7.51) | 6.55 (6.28 .. 7.60) | 1.000 | 0 of 75 | 0.0000 (0.0000 .. 0.0000) | -0.97 (-1.01 .. -0.78) | -0.0042 (-0.0070 .. -0.0025) | -0.0003 (-0.0011 .. 0.0024) | 0 (0 .. 0) | 0 (0 .. 0) |
| U3 / RTO k=1 | 0.69 (0.60 .. 1.55) | 0.19 (0.10 .. 1.04) | 0.880 | 9 of 75 | 0.0301 (0.0000 .. 0.0362) | -1.12 (-1.14 .. -0.95) | -0.0075 (-0.0086 .. -0.0039) | -0.0004 (-0.0019 .. 0.0028) | 3 (0 .. 3) | 0 (0 .. 0) |

**Pre-registered reading** (`reading`; criterion: pooled feasibility >= 0.95 and median energy
excess over the same controller on U0 <= 2 %):

| controller | class | U1 feasible / excess | U2 feasible / excess | U3 feasible / excess |
|---|---|---|---|---|
| SAC | **G0** | 0.933 / -4.04 % (fails) | 1.000 / -2.07 % (meets) | 1.000 / -0.10 % (meets) |
| TD3 | **G0** | 0.987 / -2.17 % (meets) | 0.987 / -0.74 % (meets) | 0.973 / +0.43 % (meets) |
| RTO | **G0** | 0.933 / -3.13 % (fails) | 0.973 / -1.85 % (meets) | 1.000 / +0.06 % (meets) |

**RL against RTO on the same learned environment** (`rl_vs_rto`, paired over the three GRU
replicates, t with 2 degrees of freedom; 18 tests, no multiplicity correction):

| env | RL | metric | mean difference RL - RTO | threshold | p | resolved |
|---|---|---|---|---|---|---|
| U1 | SAC | gap_energy_mean_kw | -0.004926 | 0.008884 | 0.140 | no |
| U1 | SAC | gap_moisture_mean_kg_per_kg | +0.001122 | 0.01018 | 0.682 | no |
| U1 | SAC | violation_share | +3.701e-17 | 0.1987 | 1.000 | no |
| U1 | TD3 | gap_energy_mean_kw | +0.001719 | 0.01678 | 0.702 | no |
| U1 | TD3 | gap_moisture_mean_kg_per_kg | +0.004801 | 0.0242 | 0.483 | no |
| U1 | TD3 | violation_share | +0.05333 | 0.2295 | 0.423 | no |
| U2 | SAC | gap_energy_mean_kw | -0.0005641 | 0.0006228 | 0.060 | no |
| U2 | SAC | gap_moisture_mean_kg_per_kg | +0.00117 | 0.004678 | 0.394 | no |
| U2 | SAC | violation_share | +0.02667 | 0.05737 | 0.184 | no |
| U2 | TD3 | gap_energy_mean_kw | +0.001506 | 0.007281 | 0.467 | no |
| U2 | TD3 | gap_moisture_mean_kg_per_kg | +0.002313 | 0.009229 | 0.394 | no |
| U2 | TD3 | violation_share | +0.01333 | 0.05737 | 0.423 | no |
| U3 | SAC | gap_energy_mean_kw | -0.00113 | 0.001001 | 0.040 | **yes, SAC stronger** |
| U3 | SAC | gap_moisture_mean_kg_per_kg | +0.0003914 | 0.002443 | 0.562 | no |
| U3 | SAC | violation_share | +0 | -- | -- | degenerate |
| U3 | TD3 | gap_energy_mean_kw | -0.0009899 | 0.000618 | 0.020 | **yes, TD3 stronger** |
| U3 | TD3 | gap_moisture_mean_kg_per_kg | +0.0001853 | 0.001887 | 0.714 | no |
| U3 | TD3 | violation_share | -0.02667 | 0.1147 | 0.423 | no |

**Error map** (descriptive, added after the runs; `error_map_descriptive`; 7 x 7 residual box x
25 points = 1225 grid points, prediction valid and truth solved; signed error = predicted - true):

| model | points used | bias moisture kg/kg | RMSE moisture | bias energy kW | RMSE energy | bias bed temp K | RMSE bed temp | k=5: model-feasible, truly infeasible | k=1: same |
|---|---|---|---|---|---|---|---|---|---|
| U1/seed1 | 1225 | -0.0116 | 0.0345 | -0.0156 | 0.0203 | +0.38 | 1.36 | 0.3 % of 926 | 5.7 % of 1056 |
| U1/seed2 | 1217 | -0.0063 | 0.0171 | +0.0015 | 0.0283 | +0.78 | 1.15 | 0.0 % of 885 | 1.9 % of 1017 |
| U1/seed3 | 1225 | -0.0155 | 0.0364 | -0.0136 | 0.0178 | +0.83 | 1.57 | 0.3 % of 963 | 7.6 % of 1088 |
| U2/seed1 | 1225 | -0.0042 | 0.0174 | -0.0033 | 0.0066 | -0.37 | 0.71 | 0.0 % of 885 | 0.3 % of 995 |
| U2/seed2 | 1221 | -0.0046 | 0.0146 | -0.0079 | 0.0104 | +0.47 | 0.71 | 0.0 % of 872 | 0.3 % of 1004 |
| U2/seed3 | 1217 | -0.0058 | 0.0183 | -0.0062 | 0.0092 | -0.06 | 0.56 | 0.0 % of 882 | 2.9 % of 1033 |
| U3/seed1 | 1225 | -0.0035 | 0.0157 | -0.0069 | 0.0095 | -0.23 | 0.52 | 0.0 % of 863 | 0.0 % of 973 |
| U3/seed2 | 1213 | -0.0031 | 0.0113 | -0.0085 | 0.0104 | +0.61 | 0.85 | 0.0 % of 854 | 0.0 % of 973 |
| U3/seed3 | 1095 | -0.0029 | 0.0123 | -0.0071 | 0.0098 | +0.02 | 0.50 | 0.0 % of 859 | 0.3 % of 943 |

**Learning curves** (`learning_curves`, median over seeds of the feasibility rate at the
checkpoints 1000 / 2500 / 5000 / 10 000 / 20 000 / 30 000): SAC U0 0.96 / 0.96 / 0.96 / 0.84 / 1.00
/ 1.00, U1 0.96 / 0.92 / 0.76 / 0.20 / 0.92 / 0.92, U2 0.96 / 0.96 / 0.88 / 0.52 / 1.00 / 1.00, U3
1.00 / 0.96 / 1.00 / 1.00 / 1.00 / 1.00; TD3 between 0.96 and 1.00 at every checkpoint except U1
at 1000 steps (0.76). The SAC dip at 10 000 steps is the transient of the Lagrange multipliers
already seen in Befund 96 (U0 0.84); it is deeper on the learned models and recovers by 20 000.

## 9. Reading and interpretation

**Pre-registered reading: G0 for all three controllers (SAC, TD3, RTO k = 5).** On U2 (composite
GRU, 64 runs) every controller keeps the true limit in >= 95 % of the 75 cases (SAC 1.000, TD3
0.987, RTO 0.973) and needs no more energy than on U0 (median excess -2.07 %, -0.74 %, -1.85 %).
U3 also meets the criterion for all three, U1 only for TD3.

**What G0 rests on: the margin 5 m absorbs the model error.** The learned models underestimate the
moisture: at the chosen actions by 0.0049-0.0080 kg/kg on U2 (median), 0.0092-0.0150 on U1, and
over the whole residual box by 0.0042-0.0058 (U2) and 0.0063-0.0155 (U1). The controllers turn
part of that error into apparent savings (the negative energy excess against U0; SAC on U2 is
4.61 % above the true optimum against 6.82 % on U0), but 5 m = 0.0298 kg/kg is larger than the
bias, so the true limit holds. With the margin m (RTO k = 1, sensitivity without reading) the
same optimiser violates the limit in 39 of 75 cases on U2, 60 of 75 on U1 and 9 of 75 on U3
(U0: 0 of 25), largest violation 0.050 / 0.083 / 0.030 kg/kg in the median. G0 is therefore a
statement about learned model plus margin 5 m, not about the learned model alone; on U0 the
margin 5 m costs 6.4 % energy (RTO k = 5 against the true optimum).

**The optimiser seeks the errors.** Over the residual box only 0.3-2.9 % of the grid points that
U2 calls feasible at k = 1 are truly infeasible (U1 1.9-7.6 %, U3 0-0.3 %), but among the 25 RTO
optima at k = 1 it is 7-19 points (U2), 14-22 (U1) and 0-3 (U3). The optimum lies on the
constraint boundary, where a small negative moisture error is enough to make the point
infeasible: the exploitation is concentrated exactly where the decision is taken, and the
box-wide error statistics understate it.

**The ensemble U3.** The pessimistic moisture (`mean + 2 std`) removes the bias at the chosen
actions (constraint gap -0.0004 .. +0.0005 kg/kg in the median against -0.0038 .. -0.0042 for the
ensemble mean) and brings the energy close to U0 (excess -0.10 %, +0.43 %, +0.06 %). At k = 1 it
cuts the violations from 39 to 9 of 75 cases but stays below 0.95 (0.880): the ensemble spread
does not replace the margin here. Cost: with master seed 3, 22-26 % of the RL training steps are
masked (bed-temperature `multiple_roots` in one member) and 6 test points have an invalid
prediction.

**RL against RTO.** No resolved difference in the moisture gap or the violation share on any
learned environment (12 tests). Resolved: the energy gap on U3, RL more negative than RTO by
0.0011 kW (SAC, p 0.040) and 0.0010 kW (TD3, p 0.020), about 0.1 % of the energy of a point;
2 of 18 tests at 95 % without correction, where about 0.9 false positives are expected. With
three replicates there is no evidence that RL exploits the model errors more than the optimiser;
both exploit them to a similar degree, and the margin decides the result.

**TD3.** At master seed 3 (RL seed 2) TD3 ends on U0, U1 and U2 in the same context-independent
action, 107.99 degC and 0.80 m/s at all 25 points (energy 25.56 % above the optimum, feasible);
on U2 seed 0 in another constant action (91.0 degC, 0.80 m/s). The collapse does not depend on the
environment (it happens on U0 as well) and is a property of TD3 in this cell, as in Befund 68
S0.8. TD3's G0 rests on the median over replicates.

**Caveats.** Three replicates; the mapping is the fixed point of the h36 map (decision of the
project lead), not a rollout; the problem is steady and one-step, and a transient or closed-loop
use of the GRU could be exploited differently; the energy criterion of the reading is one-sided
and is met here by savings that come out of the margin; U1 is not part of the reading. Every
number inherits the placeholders (limit 0.15 kg/kg, gas-velocity bound 0.80 m/s, margin m,
penalty weight 50 kW, process config).
