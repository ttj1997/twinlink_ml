# twinlink-ml

Machine-learning and control part of **TwinLink AP1**, a BMWE-funded research project on a
digital twin for fluidized-bed spray granulation (subproject TUHH-B, TU Hamburg).

The project compares **reinforcement learning** and **recurrent / attention-based sequence
models** (GRU, LSTM, xLSTM, Mamba, Transformer) for controlling a fluidized-bed spray
granulation process. At the core is a **grey-box simulator** of that process. It generates
training data and is used to evaluate control strategies before anything touches a real plant.

> Status (October 2026): research code under active development. Results live in the lab book
> ([`docs/04_stand_ap1.md`](docs/04_stand_ap1.md), numbered *Befunde*) and are summarised
> below. Absolute numbers rest on placeholder parameters (see [Caveats](#caveats)); the
> comparisons between methods are the point.

## What is in here

| Area | Where | What |
|---|---|---|
| Process simulator | `src/twinlink_ml/simulation/` | Grey-box model of the bed (solids, water, particle count, bed temperature), steady-state solver, excitation design, plant variants for transfer studies |
| Data and evaluation | `src/twinlink_ml/data/`, `evaluation/` | Run-level splits, learning curves, paired resolution over seeds, test-set handling |
| Models | `src/twinlink_ml/models/` | Persistence, static regression, ARX, GRU/LSTM, xLSTM, Mamba, Transformer, hybrid (simulator in the loop), transfer (pretrain + fine-tune) |
| Control | `src/twinlink_ml/rl/` | Bandit/RL environment on the steady-state model, Lagrangian and shield variants |
| Heat recovery | `src/twinlink_ml/heatnetwork/` | Cascaded exhaust-to-process-air heat recovery with minimum approach checked along the exchanger, pinch target |
| Partner model | `src/twinlink_ml/partner_v4/`, `partner_v4_1/` | One-to-one rebuilds of an application partner's Excel balance model, used to check and give feedback |
| Real plant data | `src/twinlink_ml/plant_data/` | Reader for plant exports. **Raw plant data is never in this repository.** |
| Documentation | `docs/` | Lab book, process set-up, work plan, syntheses, design drafts (mostly English; early lab-book entries are German) |

## Main results so far

Every number below is stated in a lab-book entry and backed by an artifact in
`artifacts/runs/`; follow the *Befund* number for the test set, the seeds and the caveats.

**Architecture comparison (Befunde 51, 55, 58, 83, 94).** Eight methods (persistence, static
regression, ARX, GRU, LSTM, Transformer, xLSTM, Mamba). **Every statement below holds for the tested,
untuned configuration** (one fixed protocol for all networks, no hyperparameter tuning; at most 200
epochs with early stopping), not for the architectures in general. The networks differ in size: LSTM
57,027 parameters, GRU 42,819, Transformer 23,523, xLSTM 8,649 (the mLSTM-only variant), Mamba 8,139
(`docs/08`, section 3, Befund 67); at equal capacity (Befund 67, old data) the sLSTM and the
mixed variant largely close the gap to the LSTM on energy and bed temperature but not on the moisture,
no capacity-matched variant is better than GRU or LSTM, and a larger Mamba stays behind. The fixed
training budget (200 epochs) is reached by part of the heads of every network (addendum to Befund 94). A convergence check with 600 instead of 200 epochs (Befund 104; GRU, LSTM and Transformer at 16
and 64 runs) lets every head stop regularly and leaves the order and the effect sizes unchanged; two
Holm verdicts at the threshold change (Transformer against LSTM at 16 runs). Verdicts are paired over five seeds
(Befund 83, 94), per cell and not corrected for multiplicity: after Holm only about a quarter of
the rank cells remain (old data 82 of 336, new data 70 of 336, Befund 94). The re-run on the
corrected data basis (preset version 2, Befund 94, gate passed) reproduces the ends of the order:
- **GRU against LSTM:** not distinguishable on moisture (15 of 16 cells unresolved, one with the
  GRU worse); on energy and bed temperature the GRU is resolvably better in part of the cells
  (5 and 9 of 16, uncorrected). *Precision (2026-10-16, Befund 98, addendum to Befund 94): after Holm
  correction no GRU-against-LSTM cell is resolved in any channel; the GRU is recommended as the
  default on practical grounds (never resolved worse, fewer parameters, lower single-window
  latency), not as a ranked winner.*
- **Transformer, xLSTM, Mamba, ARX, static regression, persistence** are resolvably worse than
  the LSTM in the large majority of cells (uncorrected); xLSTM on moisture stays unresolved in 7 of
  16 cells. *After Holm (2026-10-16): Mamba and the baselines are resolved behind the LSTM in part of the cells. The xLSTM deficit
  is large and consistent across all seeds (median ratio 3.3; 40 of 48 cells individually at least
  1.5; all seeds worse in all 48 cells), not resolved under Holm with five seeds and a family of
  336 tests (the Transformer deficit on moisture likewise); only small or inconsistent cells are called
  an uncorrected tendency (Befund 98, `docs/08` version 5b).*
- **The middle of the order** (Transformer, xLSTM) changes between "unresolved" and "worse" from
  cell to cell: not supported by five seeds.
- Plant operating data without excitation do not suffice in the tested budget range (Befund 55):
  an excited experiment is needed.
- The old numbers rest on the old data (`DataSpec()` defaults); 85 rank verdicts differ from them
  in the re-run. Each number needs its arm and test set named (Befund 94).

**Value of a simulator for a data-poor model (Befund 88).** Pretraining an LSTM on simulator
data and fine-tuning on plant runs beats training from scratch at small N on every tested plant
variant. The ratio of RMSEs falls from about 12.7 (identical plant) and 10.3 (changed
parameters) at N = 2 runs to about 1.3 at N = 64. Pretraining + fine-tuning was never worse than
from scratch. The simulator-in-the-loop hybrid was never better than pretraining + fine-tuning.

**Model-based energy optimisation (Befunde 52, 64, 69, 80).** The energy-optimal operating point
shifts with ambient conditions. After the solver correction (Befund 79/80), at a gas-velocity
lower bound of 1.26 m/s the regret of a fixed operating point is 16.5 % (median); a safety margin
at the moisture limit costs about 1.5 % and keeps 94 % of the control benefit (85 % at 0.80 m/s).

**Coupling of two dryer lines through shared heat recovery (Befunde 73-81).** A large coupling
gain reported earlier does not survive the corrected cascade (minimum approach checked along the
whole exchanger): 0 of 6 test scenarios at `dT_min` = 20 K (Befund 81).

**Partner model checks (Befunde 72, 74, 75).** One-to-one rebuilds of the application partner's
balance model reproduce all finished cells; the gas rows of v4.1 are 108.91 times too high, and
`cp_gas` = 1.2 instead of about 1.0 kJ/(kg K) raises the heating power by 19.1 % (Befund 75).
Feedback goes to the partner through `docs/11_partner_model_v4.md`.

**Heat recovery in the partner model (Befunde 90, 95; Befund 90 corrected 2026-10-15).** At the sheet's flows the
partner model v4.1 recovers 60-88 % of what an independent counter-current calculation recovers
(a first version said 46-65 %: it mixed two gas-property sets; see the addendum to Befund 90). The
largest single assumption is equal outlet temperatures (parallel flow), followed by `cp_gas`; the
best medium flow is about 25 kg/h with counter-current exchangers, while the partner model always
prefers the highest flow. The six scenarios keep their order (1/2 before 5/6 before 3/4) when the
assumptions of the partner model are replaced one by one (Befund 95). For this the units of the
model are rebuilt as a flowsheet (`src/twinlink_ml/flowsheet/`) with seven switches, one per
assumption; with all switches on `partner` it reproduces all 230 checked cells of the sheet
(largest relative difference 2.0e-15). Model against model, no plant data.

**Storage of plant A (Befunde 91, 92).** The storage model (S1) bounds the volume only (about
30 m3 at most) and leaves the compressor heat unmeasured; it serves as a scenario band, not as a
predictor. The pre-registered gate of the shifting potential (S2) reads L1 by the letter; a
post-hoc, marked reading with a plain rule gives L2. The saving comes entirely from using
discarded heat (a placeholder without a measured source); pure shifting in time saves about
nothing. The storage capacity corresponds to about one hour of heating demand. Decision: no S3.

**Storage test, time-slice pinch (Befund 99; scenario band, not a plant statement).** An ideal
storage is compared with using heat only when it occurs. The pre-registered gate reads **L2**: a
rule without foresight reaches the ideal storage (largest gap 3.9e-15). A greedy rule equals the
programme by structure, so L1 (RL or MPC justified) cannot occur in this model; this says nothing
about other limits. A saving above 5 % exists only through an assumed compressor load profile (a
placeholder; with constant operation there is nothing to shift), in 339 of 2240 scenarios, all on
the product-2 day. In the full network of the two pre-registered references the saving is 0: the
process air absorbs the compressor heat directly, the network replaces the storage. Every number
carries its scenario; none is a statement about the plant.

**Storage test, re-check with an admissible solid fraction (addendum to Befund 99).** The solid
fraction 0.3 of Befund 99 is inadmissible by the dryer energy balance of the two measured days
(operating-point study, Befund 103). Repeated at 0.60, 0.634 and 0.70 (band 0.60-0.70, common value
0.634; placeholders; product-specific minima kept locally): the gate still reads L2, 339 of 2240
main-grid scenarios still have a cell above 5 %, all on the product-2 day; the caveat that the
product-2 exhaust was capped at saturation in every step no longer applies. Two sensitivity
numbers move (exhaust of plant C by annual figure +398 -> +84 to +175 kWh/d; `dT_min` 10 K +247 ->
+51 to +114 kWh/d); no reading changes.

**Site pinch (Befund 100; scenario grid on placeholders, not a plant statement).** Largest usable
waste heat of the site: in the reference scenario R1 the minimum fossil heat is 655 kW against 1092
kW today, a saving of 436 kW (3821 MWh/a, 853 t CO2/a, 7.4 % of the site's 11 500 t/a); over the
grid 62-3564 kW (median 993 kW). The pinch lies at the inlet of the warmest waste-heat stream; a
network at 70/40 degC reaches a median 0.97 of the target saving. The solid fraction does not move
the site result (change of the saving -15 to +36 kW between 0.3 and 0.634).

**Heat pump on low-temperature waste heat (Befund 101; scenario band on placeholders).** Stage A:
every scenario of the main grid saves, by CO2 and by cost, as a booster across the pinch on top of
direct recovery (median 244 t CO2/a at 1000 kW with the solid fraction 0.634, 265 t/a with 0.3).
Stage B (heat pump plus storage under a time-varying signal): the gate reads **L1 formally** (114 of
4480 realistic cells with a gap of at least 2 points between the programme with foresight and the
best rule without, largest 9.2 points). A constant value of stored heat tuned per cell in hindsight
(declared post hoc) closes most of it: 18 of the 114 cells remain, all in the assumed shift mode
(largest gap 2.7 points). L1 says that foresight beats these three rules somewhere, **not that RL is
needed**: a tuned or schedule-aware rule or a day-ahead MPC closes most of it, and the surviving
cells rest on the assumed compressor night profile.

**Exhaust recirculation of one line (Befund 102; laboratory line with placeholder physics).** At
unchanged set points there is no headroom (admissible recirculation 0.002 at 17 of 25 contexts). With
re-optimised set points the moisture limit is paid for with inlet temperature: the best recirculation
per context (envelope, declared post hoc) saves 17.0 % CO2 in the median (maximum 29.1 %), and
recirculation makes the line's own heat recovery possible instead of competing with it (12.6 kW
median). The placeholder kinetics are the top of the sensitivity ranking; no plant statement.

**Operating point of plant A (Befund 103; upper bound under placeholders and proxy constraints, no
operating recommendation).** Another process-air flow and supply temperature would save at most 758 kg
CO2/d (14.6 %) on product 1 and 90 kg CO2/d (2.8 %) on product 2 at the reference placeholders. The
result is decided by the minimum air flow, the fan law and the waste-heat side; the proxies do not
replace a product-moisture measurement.

**Dynamic headroom of the laboratory line (Befund 105; placeholders, full knowledge of the state).** Tracking the operating point is worth about 9 % energy in the median (fixed point against a quasi-stationary optimiser, 8.9 %); the dynamics add under 1 % (MPC without disturbance preview 0.55 %, perfect preview 1.51 %, almost entirely through a smaller margin). Reading L0: dynamic control does not pay in continuous operation, the RL line for the granulator in continuous operation is closed. Partial observability was not tested.

**Surrogate-based steady RTO with few true evaluations (Befund 107; models of the lab line and of a heat-recovery line, placeholders).** Only a Gaussian process with adaptive design passes the pre-registered gate: from 64 true evaluations on the line model and from 256 on the grey-box model (reading S2); space-filling designs do not. A grey-box fit with a margin is economical in data where the model error shifts the constraints uniformly (feasible from 8 evaluations on one problem, a post-hoc declared variant), but not where compensating parameters hide a shift of the optimum; fitted parameters then are physically wrong and not measured values.

**Global sensitivity of the steady-state models (Befund 108; placeholder ranges, independent inputs).** For the laboratory line the drying kinetics (critical moisture, `nu_max`) decide the energy at the optimum, the operating point and the saving of the quasi-stationary optimiser, the sorption isotherm hardly matters; for the heat-recovery line the exchanger approach and the heater efficiency decide. The pre-registered convergence criterion is met for 19 of 20 outputs at 2^14 samples (not for the optimal gas velocity).

**One-step bandit (Befund 93, follow-up of Befund 68).** A residual action and a separated
moisture cost lift SAC from a negative to a clearly positive closed-regret fraction, but no cell
meets the feasibility gate; RL stays a diagnostic finding. **Citation rule:** the median 0.91 of
the cell `ressep` lies above the reachable admissible ceiling 0.82 of the residual box, so it is
carried by infeasible points and is not to be quoted as a performance figure.

**Margin needed by the RL policy (Befund 96, addendum to Befund 93).** RL needs a margin of 5 m
(m = the margin of the earlier cells) to be feasible against the true quality limit in all five
seeds: 6.9 % extra energy over the optimiser with margin m (median, range 6.3-8.1 %), of which
6.5 points are the margin itself (the optimiser pays it too) and 0.4 points the policy. The gate is
met only by the strict pre-registered reading and only at the 25 grid contexts; quality limit,
lower gas-velocity bound and penalty weight are placeholders.

**Inference cost (Befund 97).** On an idle machine, one thread: a prediction of one window by a network of
the eight architectures takes 4.8 ms (GRU) to 31.6 ms (Transformer); the hybrid of study A is dominated (98.8 %) by the
simulator channel it needs ahead of the prediction (about 253 ms of 256 ms per prediction).

## Installation

One virtual environment with every extra of `pyproject.toml`:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[torch,sklearn,plant-data,train,rl,dev]"
python -c "import twinlink_ml; print(twinlink_ml.__file__)"   # must point to this repo's src/
```

The grey-box simulator, the heat network and the partner-model rebuilds need only NumPy and
SciPy; PyTorch is needed for the sequence models, `gymnasium` / `stable-baselines3` for RL.

```bash
pytest                      # full suite takes a long time (tens of minutes)
ruff check . && ruff format --check .
```

In a git worktree, start every run with `tools/run_here.sh <python args>` so that the worktree
loads its own code; never run `pip install -e .` from a worktree.

## A first look

```python
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, load_process_config

config = load_process_config("conf/process/greybox_gab_placeholder.yaml")
simulator = GreyBoxFluidBedGranulator(config)
trajectory = simulator.rollout(actions, disturbances, seed=0)   # (batch, time, channel)
```

New studies generate their data with `DataSpec.new_study(...)` (preset version 2: planned
future set points, own disturbance process for the solid fraction, per-run cap of the gas
velocity). The plain `DataSpec` defaults reproduce old data only.

The scripts in `scripts/` are the entry points of the studies (`report_*`, `analyze_*`,
`probe_*`, `run_*`). Each artifact records the git commit, the dirty flag, the config
fingerprint and where the code was loaded from. Long runs (hours) belong on a local machine and
are resumable with `--resume`.

## Working rules (short version, details in `CLAUDE.md`)

- **Run-level splits, never window-level.** Learning curves are the primary result, 3-5 seeds,
  median and range. Baselines come before any architecture comparison.
- **Always name the test set** (excited vs. constant-run holdout). Rank claims only after
  checking against the resolution limit (paired resolution over seeds).
- Commit scripts before running them; clean working tree before any run that writes artifacts;
  never overwrite an artifact.
- Lab-book entries are append-only; corrections are dated addenda.
- If a finding changes existing results, report first, fix after confirmation.

## Confidentiality

Partner companies, their customers' products and third-party persons are never named in the
repository; established pseudonyms are used (application partner, consortium coordinator,
software integration partner, associated partner). `tests/test_denylist.py` checks tracked files
against a local denylist before a commit is trusted. Workbooks and raw plant data stay in the
git-ignored `local/` directory (see `local.example`).

## Caveats

- Most process parameters are placeholders (`provenance.status: placeholder`; the loader warns).
  Treat absolute energy and CO2 numbers as relative.
- The simulator is a laboratory-scale model; statements about the partner's plant scale are
  extrapolations and are labelled as such.
- Results marked *suspended*, *withdrawn* or *provisional* in the lab book are not to be cited
  until the entry that supersedes them says so.

## Further reading

[`docs/01_projektsteckbrief.md`](docs/01_projektsteckbrief.md) (project outline),
[`docs/02_prozess_setup.md`](docs/02_prozess_setup.md) (process set-up and open questions to the
partner), [`docs/04_stand_ap1.md`](docs/04_stand_ap1.md) (lab book),
[`docs/08_synthese_m2.md`](docs/08_synthese_m2.md) (synthesis),
[`docs/11_partner_model_v4.md`](docs/11_partner_model_v4.md) (partner model and heat recovery),
[`CONTRIBUTING.md`](CONTRIBUTING.md).
