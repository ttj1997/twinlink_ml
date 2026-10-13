# Plan: re-run of the architecture comparison (Befunde 51 and 58) -- NOT STARTED

Status: **plan only, nothing computed, nothing committed to run.** Drafted 2026-10-12 on instruction of the
project lead; waits for approval and for the decisions in section 9. All numbers below are measured values of
committed records (`arch_learning_curve_projected(_extended).jsonl`, field `training/elapsed_s`, median over 5
seeds, excited test set, one thread per process) or arithmetic on them; every estimate is labelled as such.

## 1. Question

1. How much do the errors of the architecture comparison change when it is run again on **one machine and
   one environment** (the `.venv`), with identical data specification? (Platform and library effect.)
2. How much do they change with the new data specification `DataSpec.new_study()` (planned future set
   points, review D1; own solid-fraction process, review E3)?
3. **Which share of that change is due to planned instead of limited future actions (D1)?** Platform and D1
   must be separated; for that an arm with the old data specification on the new machine is needed.

The records of Befund 51 and 58 come from two platforms (GRU, LSTM, baselines: Linux, torch 2.13.0+cu130,
numpy 2.4.6; Transformer, xLSTM, Mamba: Windows, torch 2.13.0+cpu, numpy 2.5.1; docs/12 F1.4). The `.venv`
has torch 2.14.0+cpu, numpy 2.5.3, scipy 1.18.1 (Befund 83: all records since 2026-10-01).

## 2. What is compared (scope fixed by the instruction)

* **Methods:** persistence, static regression, ARX (baselines) and GRU, LSTM, Transformer, xLSTM, Mamba, as in
  `run_arch_learning_curve_worker.py` (`ARCHITECTURE_LABELS`, `BASELINE_FACTORIES`), hyperparameters and
  `max_epochs = 200` unchanged (the epoch cap is part of the protocol; the 600-epoch probe is not repeated).
  Not included: the capacity-matched variants of Befund 67, NARX (Befund 44), the constant-trained curve
  (Befund 55). They can be added later with the same arms.
* **Budgets and seeds:** `n_train` 8, 16, 32, 64; `master_seed` 1-5 (same as before, so that the pairing
  with the old records is by seed).
* **Targets, horizons, test sets, metric:** `energy_target`, `product_moisture_kg_per_kg`,
  `bed_temperature_celsius`; h = 1 and 36; excited test pool and constant-run holdout (seed 999); primary
  metric `rmse_run_median_physical`. Test sets are named in every result (CLAUDE.md).
* **Process config:** `greybox_gab_placeholder.yaml` with `enforce_gas_velocity_projection=True`,
  `process/fingerprint` `bddca37f716f5cfc` (the worker's existing guard).

## 3. Arms

| arm | data spec | window future inputs | solid fraction | purpose |
|---|---|---|---|---|
| **O** (old records) | legacy | realised | diurnal | Befunde 51/58 as stored, two platforms; not re-run |
| **A0** | `DataSpec(...)` legacy | realised | diurnal | same specification on the new machine: **platform and library effect** (A0 against O) |
| **A1** | legacy data, `future_actions="planned"` | planned | diurnal | **D1 alone**: the same trajectories as A0, only the future inputs of the windows differ (A1 against A0) |
| **N** | `DataSpec.new_study(...)` | planned | independent | the preset for new studies: D1 + E3 together (N against A0) |
| *(A2, optional)* | `future_actions="realised"`, `solid_fraction_process="independent"` | realised | independent | E3 alone; completes the 2 x 2 factorial and shows an interaction D1 x E3 |

Why A1 is cheap in data but not in compute: D1 changes only `future_exogenous`, not the trajectory, so A0 and
A1 share the pool of every seed; training is repeated. E3 changes the pool (new disturbance, new
trajectories, new `excitation/fingerprint`); the constant-run holdout inherits both options through
`dataclasses.replace` and therefore also differs in N.

**Cross-evaluation (cheap, inference only).** Because D1 acts at training *and* at test time, every trained
model is evaluated on both window variants of its own test pool: on `realised` and on `planned` future inputs
(the checkpoints of the worker, `utils/checkpoints.py`, are saved; evaluation of a stored model is cheap
except for the Transformer, about 2.4 ms per window). This separates "the model was trained on cleaner inputs"
from "the test no longer hands it state information through the clipped future velocity". For N the models
are in addition evaluated on the legacy (diurnal) pools of the same seed, so that a change of the test pool
is not mistaken for a change of the model. Each model thus has four evaluations (2 window variants x 2 pools);
the primary one of an arm is its own specification.

## 4. Gates before any arm counts

1. **Data identity of A0 with O.** `process/fingerprint`, `excitation/fingerprint`, `split/digest`,
   `measurement/digest` equal the old records for every seed; the persistence, static and ARX records
   (no training randomness) equal the old ones to within a stated tolerance (proposal: relative 1e-6 on
   `rmse_run_median_physical`). If scipy 1.18 (Sobol) or numpy changed the pools, A0 is not a platform-only
   arm and the plan stops here.
2. **Provenance:** one `git/commit` per arm, `git/dirty=False`, `code/package_dir` `src/twinlink_ml`
   (Befund 82 f), run from a detached worktree of the committed commit with `tools/run_here.sh` so that
   commits in the main working tree cannot touch the run (lesson of Befunde 58 and 67).
3. **Pilot:** one seed, `n_train` 8, all eight methods, arm A0 and N; checks the gates, the timing below,
   and that `--resume` works. Metrics of the pilot are not used.

## 5. Evaluation (pre-registered here; the evaluation script is committed before it runs)

* **Pairing:** all arms use the same `master_seed` set; `evaluation/paired_resolution.py` (paired t, n-1
  degrees of freedom) on `rmse_run_median_physical`; sensitivity on the log ratio; per cell the medians, the
  relative difference of the medians and the number of seeds in which the arm is worse (as in Befund 83).
* **How much do the errors change** (per method, channel, test set, horizon, budget): log ratio
  `log(arm / reference arm)` per seed with its paired t verdict, for A0 against O (separately for the Linux
  group GRU/LSTM/baselines and the Windows group Transformer/xLSTM/Mamba, because the platform part
  differs: platform + library for the first group, library only for the second), A1 against A0, N against A0.
* **Share due to D1:** per method and cell, `share = log(A1/A0) / log(N/A0)`, reported only where the
  denominator is resolved; with A2: `log(N/A0) - log(A1/A0) - log(A2/A0)` as the interaction. Cross-evaluation
  splits `log(A1/A0)` into a training part (A1-trained model on the realised windows against the A0 model) and
  a test part (the same A1 model on planned against realised windows). No share is quoted for a cell
  where the total change is unresolved.
* **Rank statements:** the paired verdicts of Befund 83 repeated on the new records, so that one can see which
  rank statements are robust against platform and data specification. Multiplicity: the families are named
  beforehand (all tests; per Befund; per question), Holm and BH as sensitivity, as in the addendum to
  Befund 83. The seeds-needed table is recomputed for the new data.
* **Not claimed:** a change of the ranking is reported as a change of the ranking in this model and these
  data, not as a property of the architectures.

## 6. Compute (estimate from committed timing records; to be calibrated by the pilot)

Training time of one seed job per method, sum over `n_train` 8/16/32/64, one thread per process:

| method | sum over the four budgets [s] | [core-hours] |
|---|---|---|
| Transformer | 3267 + 5610 + 9304 + 14698 = 32 879 | 9.1 |
| xLSTM | 418 + 1095 + 1579 + 3477 = 6569 | 1.8 |
| LSTM (Linux) | 553 + 891 + 2949 + 1284 = 5677 | 1.6 |
| GRU (Linux) | 527 + 865 + 1502 + 1534 = 4428 | 1.2 |
| Mamba | 292 + 761 + 1078 + 1992 = 4123 | 1.1 |
| baselines | seconds | 0 |
| **per arm and seed** | | **about 14.9** |

Per arm (5 seeds) about 75 core-hours; **A0, A1, N: about 225 core-hours; with A2 about 300.** Data generation
per (arm, seed), the evaluation on the four pool/window variants and the saving of checkpoints are small against
this except possibly the Transformer inference (to be measured). The GRU/LSTM figures come from Linux runs;
on Windows they may be higher (Befund 66: per epoch 12.9 s LSTM, 3.6 s GRU, three heads, idle machine, the
two sources are not comparable).

Machine: 16 logical CPUs, 31 GB. **8 workers** (one thread each: `MKL_NUM_THREADS=OMP_NUM_THREADS=1`,
`torch.set_num_threads(1)`), because studies A and B run in parallel on the other cores. Jobs are
(arm, seed, method) triples so that the nine-hour Transformer jobs do not serialise everything; scheduled
longest first.

| arms | core-hours | wall, 8 workers, ideal | wall, if per-job cost is 1.5-2x (Befund 67 saw 2x with ten processes) |
|---|---|---|---|
| A0, A1, N | about 225 | about 28 h | about 42-56 h |
| A0, A1, N, A2 | about 300 | about 37 h | about 56-75 h |

Memory: peak per process of the order of 1-2 GB for the larger cells (Befund 67 measured 1.3 GB for the mLSTM
step); eight processes fit in 31 GB with room for studies A and B (to be confirmed by the pilot). Disk: one
checkpoint per trained model (about 5 methods x 4 budgets x 5 seeds x arms), not committed; the records
(JSONL) are.

## 7. Code changes needed (committed and tested before the run; no change of defaults)

1. A worker/sweep variant of `run_arch_learning_curve_worker.py` that takes the arm (`DataSpec` options) and a
   method subset per job, writes `data/future_actions` and `data/solid_fraction_process` and the arm name
   into every record, saves checkpoints, and evaluates every trained model on the four pool/window variants
   (records tagged by `eval/window_variant`, `eval/pool`). The old worker stays unchanged for reproduction.
2. A sweep driver with resumable (arm, seed, method) jobs and the provenance guard of
   `run_arch_learning_curve_sweep.py` (commit drift, dirty tree) extended by the code-location check.
3. Tests: fingerprint and persistence gate of section 4 as a test on a small spec; cross-evaluation on a tiny
   pool; job-key resume.
4. The evaluation script of section 5.

## 8. Risks and limits

* scipy/numpy changes could alter the pools (gate 1 catches it).
* Compute is dominated by the Transformer (61 % of the core-hours); the Transformer is epoch-capped in 73 %
  of its fits (Befund 58 OA5b), so its learning curve is partly a budget statement, as before.
* A0 against O mixes platform and library version for the Linux group and library version only for the Windows
  group; it cannot separate the two within the Linux group (no Linux run is planned).
* GRU/LSTM nondeterminism of the platform kernels is not controlled; five seeds give the platform effect only
  with the resolution of Befund 83 (energy and bed temperature a few seeds, moisture many more).
* E3 changes the test pools; the cross-evaluation addresses this only for the pools of the same seed.
* The new-study defaults are placeholders (solid-fraction standard deviation 0.09, Befund 82 d).

## 9. Decisions requested

1. Arms: A0, A1, N (about 225 core-hours) or additionally A2 (E3 alone, about 300)?
2. Cross-evaluation on four pool/window variants (section 3): yes or only the own specification?
3. Scope of methods: the eight methods of Befunde 51/58 only, or add the capacity-matched variants / NARX?
4. Pilot first (about 1.4 core-hours at `n_train` 8 for one seed, scaled from the table above), then the full run.
5. Wall time of about 1.5-2.5 days on 8 workers acceptable, or reduce (for instance Transformer only at
   `n_train` 8 and 64, which cuts about 4.1 core-hours per arm and seed (the 5610 s and 9304 s of the table))?

## 10. Decisions of the project lead (2026-10-12) and the resulting scope

1. Arms **A0, A1 and N**; no A2.
2. Cross-evaluation on all pool/window variants (arm N: 4; A0/A1: the pool is the legacy pool, so the two
   window variants only).
3. The eight methods of Befunde 51/58, no others.
4. **Budgets:** A0 and A1 at `n_train` 8 and 64 only (two points suffice to separate platform and D1); N at 8, 16,
   32, 64 (basis of the synthesis, version 5).
5. **Pilot now:** one seed (1), `n_train` 8, all three arms; at most **4 workers** (studies A and B run in
   parallel); detached worktree on `C:\`, started with `tools/run_here.sh`, code-location check active. Gate as
   planned (A0 reproduces fingerprints and baseline errors of the old records, `scripts/check_rerun_gate.py`);
   if it fails: stop and report.
6. **The full run is not started**; its time is decided by the project lead when cores are free.

Compute for this scope (same timing records as section 6, one thread per process):

| arm | budgets | core-hours per seed | five seeds |
|---|---|---|---|
| A0 | 8, 64 | 7.8 (Transformer 5.0, xLSTM 1.1, LSTM 0.5, GRU 0.6, Mamba 0.6) | 38.9 |
| A1 | 8, 64 | 7.8 | 38.9 |
| N | 8, 16, 32, 64 | 14.9 | 74.5 |
| **total** | | **30.5** | **about 152** |

Wall time (estimate, jobs of (arm, seed, method group), longest first): **4 workers about 38 h ideal, about 57-76 h
if per-job cost is 1.5-2 times the committed timings; 8 workers about 19 h ideal, about 29-38 h.** The
pilot (3 arms x one seed x `n_train` 8) is about 4.2 core-hours, about 1-1.5 h on 4 workers; it also gives the
first measured per-job costs on this machine under the 4-worker load.

Code for the run (committed with tests, `tests/test_rerun.py`): `scripts/run_arch_rerun_worker.py` (arm,
cross-evaluation, checkpoints, records tagged `rerun/arm`, `eval/pool`, `eval/window`, `eval/primary`),
`scripts/run_arch_rerun_sweep.py` (jobs, at most 8 workers, resume, provenance guard, merge),
`scripts/check_rerun_gate.py`. The evaluation script of section 5 is written after the pilot and committed
before the full run.

## 11. Decisions after the pilot (2026-10-12) and the resulting scope

* **Arm A0 is not re-run.** The old records of Befunde 51/58 serve as A0 (the pilot reproduced them; confirmed for
  GRU and LSTM at seed 1, `n_train` 64, maximum deviation 8.6e-5 < 1e-3, Befund 85 b).
* The full run therefore consists of **A1** (`n_train` 8, 64) and **N** (`n_train` 8, 16, 32, 64), seeds 1-5,
  `scripts/run_arch_rerun_sweep.py --arms A1 N`: 22.7 core-hours per seed, about 113 for five seeds (committed
  timings; about 105 with the cost ratios of the pilot), about 28 h on four workers (ideal), 43-57 h with the
  margin of section 6; on eight workers about 14 h, 21-28 h.
* The evaluation script (`scripts/analyze_rerun.py`) is committed; the pilot analysis of limited actions
  (Befund 85) is done; **the full run is not started**, its time and the question of a limited excitation
  window are decisions of the project lead.

## 12. Redefinition of arm N (decision of 2026-10-13, Befund 86) -- before any computation

* **Arm N = `DataSpec.new_study()` preset version 2:** planned future inputs, own solid-fraction process **and the
  per-run cap of the gas velocity at `u_t` of the run's initial diameter**; the projection stays on as a guard and
  its count per pool is logged in every record (`pool/projected_steps`, `holdout_pool/projected_steps`; expected 0;
  a count above 0 is reported, the run is not silently accepted). N is the basis of the synthesis, version 5.
  The pilot arm N of Befund 84 (preset version 1, without the cap) was a pilot only; it is not part of the full run.
* **Arm A1 unchanged:** legacy trajectories (no cap, projection acting), planned future inputs; `n_train` 8 and 64.
  It measures how much the old findings profited from the information leak.
* **A0 = the old records** (Befund 85 b). Comparisons as in section 5; N against A0 now contains three changes
  together (D1, E3, the cap); A1 against A0 isolates D1; the share of D1 in the N effect is reported as before
  (only where resolved), the remainder is E3 and the cap together and is not split further (no arm for the cap
  alone, no A2).
* **Gate additions:** every record of N carries `data/new_study_version` 2 and `data/gas_velocity_cap`
  `initial_diameter`; `pool/projected_steps` and `holdout_pool/projected_steps` are 0 (else report).
* **Schedule:** the full run starts when the main runs of studies A and B have finished, then with **all 16
  workers** (jobs of (arm, seed, method group), longest first). Estimate: about 113 core-hours of committed
  timings (about 105 with the pilot ratios): about 7 h ideal on 16 workers, about 11-14 h with the margin of
  section 6. The start is announced to the project lead; nothing is started before.
