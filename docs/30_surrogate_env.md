# 30 -- Learned sequence models as the environment for control: how strongly do RL and an optimiser exploit the model errors?

Status: design fixed and committed before the runs (sections 1-6); runs and results in sections
7-9, added after the runs (numbers only from the artifacts named there). Branch `surrogate-env`
(from `origin/master` a84b764). Finding number: assigned by the managing session (lab book entry
"Befund NN").

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
