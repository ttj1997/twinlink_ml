# A1 -- simulator pretraining under a structurally different plant: design of the main run (DRAFT)

Status: draft for review, written and committed BEFORE any computation of the main run; updated after the second pilot (see the section below the title). The file name
carries the placeholder `NN` (the number is assigned when the finding is entered into the lab
book, `docs/04_stand_ap1.md`, entry "Befund 88"). Nothing below has been run except the pilot
(artifact `artifacts/runs/a1_pilot_pp.json`, script commit `46998d6`, clean tree).

## Decisions after the second pilot, the confirmation on the new data, and how the main run is operated

Decisions of the project lead (2026-10-13):

1. **The excitation of all studies is changed** (Befund 86; `DataSpec.new_study` preset version 2): per-run cap
   of the gas velocity at `u_t` of the initial diameter, the projection stays on as a counted guard. `master` was
   MERGED into this branch (no rebase: the provenance of the pilot artifacts stays). The pilots and the action-limit
   analysis ran on data of preset version 1; their numbers below are version-1 numbers, they are NOT the numbers of
   the main run.
2. **Training budget per method, not shared:** "from scratch" 4000 gradient steps, hybrid 8000 at the time of this
   decision, changed to 16000 after the confirmation (see "Result of the confirmation"; the shared 8000 would
   have cost "from scratch" about 9 % of C against its own optimum and shifted the comparison towards the hypotheses
   under test). Fine-tuning: all parameters, learning rate 3e-4, 320 epochs.
3. Pretraining on 256 + 32 P0 runs stays (same excitation as the plants, own seeds 101-105).
4. Five seeds, all N (2, 4, 8, 16, 32, 64); constant-run holdout kept; one LSTM over three targets for all methods.
5. **Small confirmation on the new data instead of a full pilot** (`scripts/run_a1_confirm.py`, rule fixed in its
   docstring and committed before the run): on PP and PP2 (pilot seeds, N = 2 and 16, the 12 cells of pilot 2), the
   chosen fine-tuning setting and its direct neighbours (learning rate 1e-4 / 3e-4 / 1e-3 x epochs 160 / 320 / 640),
   the two step budgets and one neighbour each (the neighbour with the lower pilot-2 score: from scratch 8000, hybrid
   16000). A chosen setting stays if its mean criterion C is at most 3 % above the best tested setting of its part;
   otherwise it is reported and the main run is not started on the old setting. The projection guard is recorded per
   pool there too.
6. The main run starts only after the confirmation, with at most 8 workers, in a worktree on `C:\`, detached with a log
   file and `--resume` (`scripts/run_a1_main.py`).

**Result of the confirmation (`a1_confirm_preset2.json`, script `f382272`, clean tree) and the decision on the hybrid.**
By the rule fixed in advance (a chosen setting stays if its mean C over the 12 cells is at most 3 % above the best
tested setting of its part): fine-tuning (3e-4, 320 epochs; C 0.0510 against 0.0499 for 1e-4 / 640 epochs, +2.1 %)
stays; "from scratch" (4000 steps; C 0.2471 against 0.2841 for 8000) stays; the hybrid (8000 steps; C 0.2527 against
0.2235 for 16000, +13.0 %) does NOT stay. Decision of the project lead: the hybrid gets **16000 steps** (the rule:
each method at its best tested budget). To be stated in the finding: **16000 is the EDGE of the tested range**
(8000 and 16000 were tested; whether more steps would help is not known), and **the selection is driven by the
strongly scattering N = 2 cells**: at N = 2 the C of one setting ranges from 0.13 to 1.05 across the three pools; at
N = 16 the two hybrid budgets differ by less than the scatter (PP 0.0264 against 0.0270, PP2 0.0261 against 0.0253). As a
SENSITIVITY without a hypothesis of its own the main run also trains the hybrid with 8000 steps at N = 2 and 4
(method `hybrid_steps8000`); the evaluation reports whether any conclusion depends on the hybrid budget.

Pretraining on data of preset version 2: on the 224 P0 test runs the pretrained model reaches C = 0.017 (0.0169; 132
epochs, best epoch 111), on data of version 1 it reached C = 0.054 (0.0535; 31 epochs, best epoch 10; pilot 2); the
data and the test runs differ, so the comparison is of the pretraining on the two excitations, not of one model on one
test set. The same guard check found 0 projected steps in all seven pools of the confirmation and the preset version 2 in
all; the simulator's own projection in the hybrid channel acted at 223 of 36 864 samples in pool 11 and never in
pools 12 and 13.

**Timing and the machine.** The main run starts in parallel to Study B (both with 8 workers on 8 physical cores).
Every time measured in the run (training time, inference time) is marked `timing_context` = "parallel load, not a
result" and is not used as one. The inference time per method is measured ONCE after the run, separately: a machine
without other runs, one thread, warm-up excluded. Cost with the budgets above, extrapolated from the times measured
in pilot 2 for a machine of one job (EXTRAPOLATION): per (plant, seed) fine-tuning about 22 min, from scratch about
24 min, hybrid 16000 steps about 95 min plus the sensitivity about 13 min, data and evaluation about 10 min: about
164 min; 25 jobs = about 68 h of CPU = about 8.5 h of wall time on 8 workers alone; under the parallel load of Study B
1.5-2 times that.

**Guard in the main run.** Before any training of a unit, every pool it generates is checked: `data/new_study_version`
must be 2 and the number of projected steps must be 0. Otherwise the unit writes a `STOP` file and the run halts
(running units end at their next record) and is reported. Every result record carries `data/new_study_version` and the
projection guard of both pools (steps, runs with projection, per pool); the simulator's own projection inside the
hybrid channel is recorded per pool as well (not a stop criterion).

Cost with the per-method budgets -- EXTRAPOLATION from the times measured in pilot 2 (see below), not measured for
the main run: per (plant, seed) fine-tuning about 22 min, from scratch (4000 steps) about 24 min, hybrid (8000 steps)
about 48 min, data, simulator channels and evaluation about 10 min: about 104 min; 25 jobs = about 43 h of CPU = about
5.4 h of wall time on 8 workers, plus five pretrainings (one thread each, in parallel with the first units); with
margin 6-7 h. The confirmation costs about 1.3 h (extrapolated the same way).

## Update after the second pilot (decisions of the project lead and measured settings; data of preset version 1)

Status before the decisions above: the main run was on hold until the cross-cutting decision on the excitation. That decision is made (see above), so the pilots are CONFIRMED on new data instead of repeated.

Decisions: (1) a second pilot on TWO pilot plants, PP and PP2 (PP2: summary deviation 0.301 process /
0.305 measured against PP 0.130 / 0.134; both pilot-only), grid extended downward and, where needed, upward
until the selection is no longer at the edge, selection = lowest mean criterion C over PP and PP2, and the
step budget of "from scratch" and "hybrid" by the same rule; (2) pretraining data 256 + 32 P0 runs, same
excitation as the plants, own seeds; (4) one LSTM over three targets for all methods; (5) constant-run
holdout kept; (6) the simulator channel is not changed.

Measured in `a1_pilot2_pp_pp2.json` (script `765c233`, clean tree; pilot seeds 11-13 for the plant pools, 201
for the pretraining; 3 pools x N = 2, 16 x PP, PP2 = 12 cells per setting):

* **Fine-tuning setting: all parameters trained, learning rate 3e-4, 320 epochs** (mean C 0.0771; PP 0.0738,
  PP2 0.0805). After three upward extensions of the epochs the selection is INSIDE the grid (640 epochs: 0.0800;
  160 epochs: 0.0774; lr 1e-4 at 320 / 640 epochs: 0.0775 / 0.0775): a flat optimum, not an edge. Best
  head-only setting: 0.1031 (lr 1e-5, 640 epochs). The first pilot's setting (1e-4, 5 epochs) was the edge of
  a grid too short in epochs; 320 epochs are 640 gradient steps at N = 2 and about 2900 at N = 16.
* **Training budget of "from scratch" and "hybrid": 8000 gradient steps for both** (combined score 0.2657),
  inside the grid (4000: 0.2724; 16000: 0.2761). Per method the best budget differs: from scratch 4000 (0.2503,
  against 0.2735 at 8000), hybrid 8000 (0.2579). One shared budget costs "from scratch" about 9 % of C
  against its own optimum -- a decision for the project lead (see below).
* Pretraining on 256 runs stopped after 31 epochs, best epoch 10 (early stopping, patience 20); in-distribution
  test (224 P0 runs) C 0.0535. The first pilot, 64 runs, reached 0.0517 on its 32 P0 test runs (different
  runs, not strictly comparable): four times the data did not visibly improve the pretrained model.
* References (mean over the 3 pools, val / test): persistence PP 0.490 / 0.455, PP2 0.474 / 0.441; simulator
  alone PP 0.181 / 0.249, PP2 0.276 / 0.332; pretrained without adaptation PP 0.156 / 0.197, PP2 0.213 / 0.249.
* Selected settings on the test runs (mean over pools): fine-tuning N = 2: PP 0.123, PP2 0.137; N = 16: PP 0.045,
  PP2 0.047. From scratch N = 2: PP 0.593, PP2 0.765; N = 16: 0.047, 0.041. Hybrid N = 2: 0.585, 0.473; N = 16:
  0.035, 0.033. (Pilot cells, selected on the validation runs; no rank statement is made from them.) The
  re-run of every selected cell reproduces its validation criterion bit for bit (determinism).

Measured times (8 workers x 1 thread, except pretraining: 8 threads): fine-tuning 0.078 s per epoch at
N = 2 and 0.512 s at N = 16; from-scratch / hybrid training 0.045 s per step at N = 2 and 0.067 s at N = 16;
pretraining on 256 runs 134 s (4.3 s per epoch, 31 epochs); P0 pool of 512 runs 14 s; first-use cost of torch
about 23-26 s per process, measured apart. The pilot took 16 324 s of wall time (the upward extensions to 640
epochs and 16 000 steps dominate).

Cost of the main run with the shared budget of the pilot -- superseded by the per-method budgets above; EXTRAPOLATION from the measured times: per (plant, seed) fine-tuning 320 epochs over N = 2...64 about 22 min, from scratch and hybrid 8000 steps
each about 48 min each, data, simulator channels and evaluation about 10 min: about 130 min; 25 (plant, seed)
jobs = about 54 h of CPU = about 6.8 h of wall time on 8 workers, plus five pretrainings; with margin 8-9 h.
This replaces the earlier 2.5 h estimate, which assumed 3000 steps and 5 epochs. Options to reduce: 3 seeds
instead of 5 (about 4.1 h), or method-specific budgets (no saving), or dropping N = 64 for the neural
methods (about 3 h saved).

Open for the project lead: (a) one shared budget (8000) or method-specific budgets (from scratch 4000, hybrid
8000); (b) whether to keep the pretraining at 256 runs given that it did not visibly improve on 64; (c) the
cost options above; (d) the excitation decision, which comes first.

## Question

How much does a simulator help a data-poor model when the plant is structurally different from
it -- and from how many plant runs on does it no longer help?

## Plants (`twinlink_ml.simulation.plant_variants`, commit `39a1390`)

| Plant | Deviation from the simulator (P0) |
|---|---|
| P0 | none (control) |
| P1 | parameters only: heat loss UA x1.30, gas-solid transfer x0.75 (PROXY: `max_saturation_degree`, the model has no separate heat-transfer coefficient), critical moisture x1.20 (kinetics only; the quality limit of the product moisture is pinned to the base config) |
| P2a | kinetics only: drying curve `f(eta) = eta ** 2` (exact option `MaterialConfig.drying_curve_exponent`) |
| P2b | measurement only: sensor lag and dead time (bed temperature 30 s / 50 s, moisture 60 s / 75 s), offset between bed and exhaust (1.0 K + 0.08 (T - 20 degC)) |
| P3 | P2a + P2b + fouling within a run (heat loss +40 % and temperature-sensor offset +1.5 K over 8 h; restarts with every run) |
| PP | pilot only (not in the main run) |

Deviation of each plant from P0 on the fixed reference excitation (8 runs of 8 h, reserved seed
20261012; RMSE per channel in physical units; artifact `a1_pilot_pp.json`, `deviation_of_plants`).
Summary = mean over energy target, bed temperature, product moisture of RMSE divided by the
standard deviation of the P0 reference (d50 is not predicted and is unaffected by every plant).

| Plant | bed temp. [K] (process / measured) | moisture [kg/kg] (process / measured) | summary (process / measured) |
|---|---|---|---|
| P1 | 1.21 / 1.21 | 0.0321 / 0.0320 | 0.285 / 0.285 |
| P2a | 1.09 / 1.09 | 0.0351 / 0.0351 | 0.305 / 0.305 |
| P2b | 0 / 4.09 | 0 / 0.0039 | 0 / 0.128 |
| P3 | 1.10 / 4.94 | 0.0358 / 0.0361 | 0.312 / 0.405 |
| PP | 0.86 / 0.86 | 0.0136 / 0.0140 | 0.130 / 0.134 |

PP deviates about half as much as P1/P2a/P3 (process summary 0.13 vs 0.29-0.31). A fine-tuning
setting chosen on PP may therefore favour little adaptation (see "Open decisions", 1).

## Pilot A1.3 (what is measured, artifact `a1_pilot_pp.json`)

One LSTM (hidden 64, 1 layer, three targets, not the per-target composite of Befund 26), pretrained
on a P0 pool (64 train / 32 validation / 32 test runs), adapted to PP pools (master seeds 1-3) with
N = 2 and 16. Selection criterion C (mean over channels of median-over-runs RMSE at h = 36 divided by
the channel standard deviation, on the 8 PP validation runs, which are in no fit).

* Selected: all parameters trained, learning rate 1e-4, 5 epochs (mean C 0.096 over the 6 cells).
  Both values are the EDGE of the grid (grid: lr 1e-4...3e-3, epochs 5...80, freeze none / core
  frozen). The grid is not extended without a decision.
* Head-only fine-tuning did not beat full fine-tuning (best head-only: lr 1e-4, 20 epochs, C 0.103).
* References on the PP validation runs, mean of C over the 3 pools: persistence 0.518,
  simulator alone 0.212, pretrained model without adaptation 0.181, selected fine-tuning 0.110 (N = 2)
  and 0.082 (N = 16). On the PP test runs (excited): persistence 0.279, simulator alone 0.087,
  zero shot 0.087, selected 0.070 (N = 2) and 0.052 (N = 16).
* Pretraining reached the P0 test runs with C 0.052 (bed temperature 0.74 degC at h = 36).
* The selected cell was re-run for the test evaluation: its validation criterion is bit-identical to
  the grid run in all 6 cells (determinism across processes).
* Hybrid channels (32 runs): steady state found at the set points for 32 of 32 runs (0 nominal
  fallbacks; 2 solutions flagged infeasible by the solver). In a separate 8-run sample of excited
  starts 2 of 8 had no steady state (low gas velocity at high spray rate), so fallbacks DO occur.

## Methods

1. **from scratch** -- a new LSTM on the N plant runs (`TransferSequenceModel.fit`, 4000 gradient steps).
2. **simulator only** -- the noise-free P0 measurement as prediction (`SimulatorOnlyModel`); no
   training, no plant data; one value per plant and seed pool.
3. **pretraining + fine-tuning** -- LSTM pretrained on P0 data, then `fine_tune` on the N plant runs
   with the pretraining scalers kept; setting fixed by the pilot (see Open decisions, 1).
4. **hybrid** (16000 gradient steps; plus a sensitivity with 8000 steps at N = 2 and 4) -- LSTM with the open P0 simulation (set points and disturbances of the run only, steady
   state at the set points as initial state, noise-free P0 measurement) as three more exogenous
   channels, trained on the N plant runs from scratch.
5. Baselines before any comparison: **persistence** and **ARX** (`LinearARXModel`, N plant runs).

All neural methods: one LSTM, hidden 64, one layer, three targets (energy target, bed temperature,
product moisture), horizon 36 (900 s), history 32, stride 8, 25 s sample time, process config
`greybox_gab_placeholder.yaml` with the gas-velocity projection ON (as Befunde 51/58, process
fingerprint `bddca37f716f5cfc`).

## Design

* Plants P0, P1, P2a, P2b, P3. N = 2, 4, 8, 16, 32, 64 plant runs, nested prefixes of the same pool.
  Five seeds (master seeds 1-5; one master seed = one plant pool, one noise realisation, one model
  seed; the same seeds for every method and plant, so the design is paired).
* Data only with `DataSpec.new_study(...)` (planned future set points, own disturbance process for the
  solid fraction). Run-level splits; per plant and seed a pool of 128 runs: 64 train (the nested
  prefixes), 32 reserved validation runs (in no fit of the main run; they keep the pool at a power of
  two), 32 test runs.
* **Test sets (named):** primary: the 32 EXCITED test runs of the plant's own pool. Secondary: a
  constant-run holdout of the same plant (16 constant runs, fixed holdout seed distinct from every
  study seed and from the reference seed), reported separately and never pooled with the primary.
* **Pretraining data:** P0 only, so one pretrained model per seed (5 models, shared by all plants).
  Same excitation design as the plant data (`new_study`: excited design with 10 % constant runs,
  independent solid-fraction process, 8 h at 25 s, projection on); own master seeds (100 + seed), never
  the plant seeds. Size: see Open decisions, 2. The pretraining uses 32 P0 validation runs for early
  stopping (defaults: 200 epochs, patience 20).
* **Metrics:** multistep error (h = 36) and one-step error (h = 1) as the median over test runs of the
  RMSE in physical units, per channel (the primary quantities; no normalised metric is used for a rank
  statement); inference time per window. Learning curves are the primary result: median and range over
  the 5 seeds per N.
* **Rank statements** only with the paired resolution (`evaluation/paired_resolution.paired_comparison`,
  paired t on the per-seed differences of the physical metric, two-sided 95 %, n - 1 = 4 degrees of
  freedom), never with the older `resolution.py`; the number of tests is reported next to every
  result (no multiplicity correction, review item C-L1 is open). The tested quantity is the MEAN of
  the paired differences; medians are reported alongside.
* **Baseline rule:** a method is only called useful on a plant where it beats persistence AND ARX with
  a resolved paired difference at that N; where it does not, that is stated and no architecture claim is
  made from it.
* Provenance: scripts committed before they run; clean tree; new artifact names; one `.venv` of the main
  checkout; runs in a worktree on `C:\` (`tools/run_here.sh`), not on the network drive; records carry
  `code/package_dir`, `code/interpreter`, `code/root_checked`.

## Hypotheses and the criteria at which they count as supported

Throughout: "better" / "worse" = a resolved paired difference (as above) of the h = 36 RMSE (physical
units) on the primary test set; "in k of 3 channels" refers to energy target, bed temperature, product
moisture; "low deviation" = P0 and P1, "strong deviation" = P3 (largest summary deviation,
measured 0.405) and P2a (0.305). The classification is by the table above, fixed here.

* **HA1 -- pretraining beats from scratch at small N and low deviation; the advantage shrinks with N.**
  Supported if (a) on P0 and on P1, at N = 2 AND N = 4, pretraining + fine-tuning is better than from
  scratch in at least 2 of 3 channels, AND (b) on both plants the ratio RMSE(from scratch) /
  RMSE(pretraining + fine-tuning) (median over seeds, mean over channels) at N = 64 is smaller than
  at N = 2. Not supported if (a) fails on either plant. The crossover N* (smallest N from which
  pretraining is no longer resolved better) is reported as a bracket between grid points -- or
  as "not reached" if the advantage persists at N = 64.
* **HA2 -- under strong deviation pretraining harms from some N on.** Supported if on P3 or on P2a there
  is an N at which pretraining + fine-tuning is WORSE than from scratch in at least 2 of 3 channels;
  the smallest such N is reported. Not supported if no such N exists (cells without resolution are
  listed as unresolved, not as "no harm").
* **HA3 -- the hybrid is the most data-efficient method under structural deviation.** Supported if on at
  least 2 of the 3 structural plants (P2a, P2b, P3) the hybrid is better than each of the other three
  methods (from scratch, simulator only, pretraining + fine-tuning) in at least 2 of 3 channels at
  at least 3 of the 4 small N (2, 4, 8, 16). The separation that motivated P2a / P2b -- which method wins
  under wrong physics (P2a) and which under wrong measurement (P2b) -- is reported plant by plant
  whether or not HA3 is supported.

## Cost (extrapolated from the pilot, NOT measured for the main run)

Measured in the pilot, one torch thread per process: fine-tuning 0.075 s per epoch at N = 2 and 0.52 s per
epoch at N = 16 (derived: about 0.24 ms per window, so at most 61 ms per gradient step of 256
windows); pretraining on 64 runs 78 s for 84 epochs (4 threads); generation of the 128-run P0 pool
8 s; hybrid channels 17.5 s for 32 runs.

Extrapolation for 8 workers (one thread each), if the from-scratch and hybrid fits get a budget of about
3000 gradient steps each (to be fixed, Open decisions, 3): about 3 min per fit independent of N (steps,
not epochs, set the cost), i.e. 6 N x 2 methods x 3 min = 36 min per (plant, seed); plus data, channels,
evaluation and the cheap methods about 7 min; 43 min x 25 (plant, seed) = 18 h of CPU = about 2.2 h of
wall time on 8 workers; five pretrainings of a 256-run P0 pool of about 6 min each, run beside; total
about 2.5 h, say 3-3.5 h with margin. Everything resumable per (plant, seed) (`--resume`).
All numbers in this section are extrapolations (placeholders without a measured source for the main
run) and are replaced by measured times in the report of the run.

## Open decisions (before the main run) -- as they stood BEFORE the second pilot; see the update above

1. **Fine-tuning setting at the edge of the pilot grid.** Selected lr 1e-4, 5 epochs = the smallest
   learning rate and the fewest epochs of the grid, so the true optimum may lie below. At N = 2 five epochs
   are 10 gradient steps. Proposal: a second pilot on PP only (about 7 min of wall time on 8 workers, a
   new artifact), same selection rule, extended downward (lr 1e-5, 3e-5; epochs 1, 2, 3), and one
   setting for all plants as before. Caveat to decide on: PP deviates about half as much as P1/P2a/P3, so
   a setting that favours little adaptation may disadvantage pretraining in exactly the plants of HA2.
   Alternatives: select on a plant of the main run (violates "pilot only on PP"), or accept the setting
   and state the caveat.
2. **Size of the pretraining data.** The question is about the value of the simulator, so the pretrained
   model should not itself be limited by data relative to the largest N. Proposal: 256 train + 32
   validation P0 runs (four times the largest N), same excitation as the plants. The pilot used 64 + 32 (its
   pretraining stopped early at epoch 84, best epoch 63). Cost at 256 runs is about 6 min per seed.
   Alternative: 64 + 32 as in the pilot (the pretrained model of the pilot is then the one used).
3. **Budget of from scratch and hybrid** (no validation set at small N). A fixed number of epochs gives
   N = 2 only 2 steps per epoch and N = 64 about 68, so equal epochs mean very different amounts of
   training. Proposal: a budget in gradient steps, fixed by the same pilot rule on PP (a second pilot:
   steps 500, 1000, 2000, 4000 at N = 2 and 16, learning rate 1e-3, last epoch kept), one setting for both
   methods and all plants; the fine-tuning budget stays in epochs as specified. The first pilot ran both
   methods once with 100 epochs for TIMING only; their test values in the artifact are not a result.
4. **Model class.** One LSTM over three targets (as in the pilot) instead of the per-target composite of
   Befund 26: `CompositeSequenceModel` has no `fine_tune`. Differences to Befunde 51/58 (composite) are to
   be kept in mind when the two sets of numbers are compared; no comparison across the two is planned.
5. **Constant-run holdout** as a secondary test set (16 runs per plant): keep, or drop to save evaluation
   time (small).

## Files of the main run

New scripts under `scripts/`: `run_a1_confirm.py` (confirmation on the new data), `run_a1_main.py` (the main run:
units, guard, resume, collection of the records into `artifacts/runs/a1_main_records.json`). The analysis with the
paired resolution (HA1-HA3, learning curves, crossover N*) is a separate script written after the run and committed
before it is applied. New modules: `models/transfer.py`, `models/hybrid.py`, `evaluation/plant_data.py`,
`evaluation/plant_deviation.py`, `simulation/plant_variants.py`. Shared files changed: `thermo.py` (two lines) and the
config schema (field `drying_curve_exponent`, only in the fingerprint when set); no change to `greybox.py`,
`torch_sequence.py`, `experiments.py`.
