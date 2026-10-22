# 27 -- Robustness of the architecture ranking against outliers in the sensor signal

Status: design fixed and committed before the run (sections 1-5); run and results in sections 6-8
(filled in after the run; numbers only from the artifacts named there). Branch `outlier-robustness`
(from `origin/master` 4b0ab66), not merged, not pushed; the finding number is assigned by the
managing session (lab book entry "Befund NN").

## 1. Question

The ranking of the architectures is robust against the **level** of Gaussian sensor noise (step 5),
but it was never tested against outliers. The lab data of the predecessor project lost 17-34 % of
the raw points per batch as outliers (docs/04, "Das Sensormodell ist zu gutmuetig"; docs/04 open
question 7). That is the realistic disturbance. Question: does the ranking on product moisture
(networks ahead of the baselines; order of the networks) survive a realistic outlier share
(`p = 0.25`), with and without a causal outlier filter?

**Step 0 (check for earlier work).** `docs/04` and `docs/08` searched for outlier, heavy-tail,
Hampel, Ausreisser (2026-10-22): `docs/04` names the outlier process only as an open item (section
"Das Sensormodell ist zu gutmuetig", open question 7, and Befund on the window-level split, item 4:
"not adopted"); the other hits concern reward outliers of the RL studies and single-seed outliers
of learning curves; `docs/08` has no hit. No earlier study; the study was started.

## 2. Sensor model

On top of the unchanged sensor model (PT1 with 10 s plus Gaussian noise, `relative_noise_std`
0.005 relative to the PT1 output, `MeasurementConfig`), applied to the measurements returned by
`remeasure` (no new simulation), only on the moisture channel, operating mode INLINE.
Settings: `conf/sensor/outliers_placeholder.yaml`, **all placeholders** (fixed beforehand by the
project lead; the only plant anchor is the 17-34 % share):

| item | setting | source |
|---|---|---|
| independent outliers | each sample with probability `p` becomes value + error | placeholder |
| error | Student-t, 3 degrees of freedom, scale = 10 x std of the existing Gaussian noise at that sample (`10 x 0.005 x |PT1 output|`) | placeholder |
| bursts | same total rate `p`; contiguous sections of geometric length, mean 10 steps (250 s); one constant offset per burst | placeholder |
| causal Hampel filter | window of the last 8 values incl. the current one; threshold 3 x scaled MAD (1.4826 x MAD); replacement by the window median; fixed, not tuned on test data | placeholder |
| side check | GRU only, `p = 0.25`, windows 4 and 16 | placeholder |

Outliers act on the input history AND the training target (as at the plant). Code:
`src/twinlink_ml/outliers/` (process, filter, contamination, evaluation), tests
`tests/test_outliers.py`: `p = 0` reproduces the existing sensor model bit-identically; the filter
is causal (output at `t` independent of `t+1...`); the outlier rate is met on average
(independent and burst); the mean burst length is met on average; the burst offset is constant;
the noise scale equals `relative_noise_std x |PT1 output|`.

## 3. Design

* `p in {0, 0.05, 0.25}`; type in {independent, burst} for `p > 0`.
* Preprocessing in {none, causal Hampel (window 8)}, identical for ALL models, persistence and ARX
  included (persistence uses the filtered last value).
* Models persistence, linear ARX, GRU, LSTM, Transformer (not xLSTM, Mamba, static).
* `n_train` 16 and 64, seeds 1-5, preset and protocol of Befund 94 arm N (`DataSpec.new_study()`,
  preset version 2; run-level splits; validation pool; constant-run holdout seed 999;
  `max_epochs` 200, patience 20; primary evaluation variant: own pool, planned windows).
* Target `product_moisture_kg_per_kg`, h = 1 and 36; test sets excited and constant-run holdout
  (always named).
* Main metric `rmse_run_median_physical_truth`: per-run RMSE in kg/kg against the noise-free
  simulator output, median over the runs. Reported in addition: against the disturbed measurement
  (before any filter) and against the processed signal the models see.
* Condition names `p<rate>_<type>_<filter>`, e.g. `p0.25_burst_hampel8`.

Driver `scripts/run_outlier_robustness.py` (at most 8 workers), worker
`scripts/run_outlier_robustness_worker.py`.

## 4. Pre-registered reading

Fixed in the docstring of `scripts/analyze_outlier_robustness.py` (commit `8d4407a`, before the
first run; tests `tests/test_outlier_robustness_scripts.py`). Short form:

* 9 pairs per cell (net order: GRU-LSTM, Transformer-LSTM, Transformer-GRU; nets against baselines:
  ARX and persistence against each network), 8 cells (`n_train` x test set x h), 72 rank cells per
  condition. Paired t over five seeds (`evaluation/paired_resolution.py`), Holm 5 % as main form,
  family = the 72 cells of one condition; effect size (ratio of the medians), sign consistency,
  median and range next to it.
* Reference C0 = `p0_none_none` (the pipeline of Befund 94). RANKING UNCHANGED in C: (a) no
  reversal of a Holm verdict; (b) no new cell with a baseline ahead; (c) net order: no new resolved
  cell and at most half of the C0-resolved cells lost; (d) LEAD (number of Holm-resolved cells with
  the network ahead of a baseline, of 48) at least half of LEAD(C0). LEAD VANISHED: LEAD = 0.
* Per outlier type: **R3** lead vanished at `p = 0.25` with the filter (the comparison does not
  carry over to real sensor quality; limit for docs/08); **R1** not R3, ranking unchanged without
  and with the filter (robust); **R2** not R3, changed without, unchanged with the filter (filter
  becomes mandatory; recommendation for AP 2); otherwise no reading (deviating cells listed,
  decision to the project lead).
* Control: the same rule for `p0_none_hampel8` (does the filter alone change the ranking on clean
  data?), reported as a caveat.
* Gate before any reading: clean tree and one commit for all records; completeness; at `p = 0`
  without filter the metric against the measurement equals the moisture metric of Befund 94
  (`rerun_full_N.jsonl`) for all five models, budgets, seeds, test sets and horizons (relative
  1e-6), with equal fingerprints and equal moisture-head epochs.

## 5. Decisions taken without asking (unattended run, most conservative reading)

1. **Base of the branch.** The task says "from the current `origin/master`". Local `master` was 8
   commits ahead (Befund 107, not pushed). Taken literally: `origin/master` (4b0ab66), the published
   state. Nothing in this study depends on the 8 commits; the lab book entry will need a trivial
   merge at the end of `docs/04`.
2. **Scale of the outlier error.** The existing Gaussian noise is relative (`0.005 x |PT1 output|`),
   so its standard deviation differs per sample. "10 x the standard deviation" is taken per sample:
   `10 x 0.005 x |PT1 output|`. The PT1 output is taken from the sensor model itself (`remeasure`
   with both noise levels 0), not re-implemented.
3. **Burst offset.** One Student-t draw per burst, scaled with the noise std at the first sample of
   the burst, added unchanged to every sample of the burst (absolute offset).
4. **Burst process.** Two-state Markov chain with stationary share `p`: a burst ends with
   probability 1/10 per step, one starts with probability `p/(1-p)/10`; the first state of a run is
   drawn from the stationary distribution; at least one clean sample between two bursts.
5. **Random numbers.** The outlier seed is derived from the noise seed of the data set and the type,
   not from the rate: independent outliers of `p = 0.05` are a subset of those of `p = 0.25` (common
   random numbers), and the comparison between conditions is paired over the same Gaussian noise.
   The fixed constant-run holdout (seed 999) gets one fixed realisation per type, as it has one
   fixed noise realisation.
6. **Where outliers act.** On every run of both pools (training, validation, test, holdout): it is
   the same sensor. Early stopping therefore sees disturbed (or filtered) validation targets.
7. **No clipping.** A heavy-tailed error may give a non-physical (e.g. negative) moisture; kept, as a
   faulty sensor delivers it.
8. **Filter details.** Only the moisture channel is filtered (the only one with outliers); the
   window holds raw values (standard non-recursive Hampel); at the start of a run the window holds
   the available values only; the filter runs over the whole run before windowing (it is causal, so
   no window sees a future value).
9. **Networks train only the moisture head.** The composite networks have one independent network
   per target with its own history and a seed derived from (model seed, target name), so the
   moisture-only view trains exactly the moisture head of Befund 94 (checked by the gate). Saves two
   thirds of the training time. Persistence and ARX are fitted on the three-target data as in
   Befund 94 (ARX regresses on all three histories). No checkpoints are written.
10. **"Against the disturbed measurement"** is the measurement with outliers before the filter
    (the same reference with and without the filter).
11. **Operationalisation of R1-R3.** The task names the readings, not the rule. Reference for
    "unchanged" is `p = 0` without filter (the established ranking); the "at most half" rules are
    taken from reading K2 of Befund 104; the family (72 cells of one condition) is the analogue of
    "all rank cells of one arm" of Befund 94/98; both test sets and both horizons enter the rules
    together. R3 has precedence (it is the strongest negative statement).
12. **Side check** for both outlier types; it is outside every family and every reading.
13. **Only the primary evaluation variant** of arm N (own pool, planned windows); the cross-evaluation
    on the legacy pool of Befund 94 is not repeated.
14. **Budget rule beyond its two cuts.** If the estimate stays above 12 h after both cuts, no further
    cut is made (none is authorised); the jobs run in tiers: tier 1 the conditions of the readings,
    tier 2 the `p = 0` filter control and `p = 0.05`, tier 3 the side check.

## 6. Budget and run

**Probe (2026-10-22, 16:55-17:36, worktree state before commit `8d4407a`, outputs outside the
repository, not artifacts).** Seed 1, while another computation occupied the machine (16
processes, until about 19:00): persistence and ARX (`n_train` 16, 64), GRU `n_train` 16, LSTM
`n_train` 64, Transformer `n_train` 16 at `p0_none_none`, and GRU `n_train` 16 at
`p0.25_independent_hampel8`. Wall times: GRU 206 s, LSTM 337 s, Transformer 2457 s (200 epochs,
12.3 s per epoch), setup (data of both pools) 25 s plus about 1 min start-up. At `p = 0` every
probed value of the metric against the measurement equals the Befund 94 record (persistence, ARX,
GRU, LSTM, Transformer; both test sets, h 1 and 36), with the same moisture-head epochs -- the gate
G3 in small. At `p0.25_independent_hampel8` the realised outlier share of the pool was 0.253 and
the filter replaced 9.1 % of the samples.

**Estimate** (cost per job in `run_outlier_robustness.py`, `COST_H`: GRU/LSTM/baselines job 0.4 h,
Transformer 0.6 h at `n_train` 16 and 2.2 h at 64, side-check job 0.15 h per budget; list
scheduling over 8 workers):

| plan | jobs | conditions | core-hours | wall, 8 workers |
|---|---|---|---|---|
| full design | 170 | 14 (10 + 4 side) | 166.0 | 20.9 h |
| cut 1: bursts only at `p = 0.25` | 140 | 12 | 134.0 | 16.9 h |
| cut 1 + cut 2: no window side check | 120 | 8 | 128.0 | 16.2 h |

**Cuts (budget rule).** The estimate is above 12 h, so (1) the bursts are restricted to
`p = 0.25` (`p0.05_burst_*` dropped) and, still above 12 h, (2) the window side check (GRU, windows
4 and 16) is dropped. After both cuts the estimate is still 16.2 h; no further cut is authorised
(decision 14), so the remaining plan runs completely, in tiers: tier 1 (`p0_none_none`,
`p0.25_independent_none`, `p0.25_independent_hampel8`, `p0.25_burst_none`,
`p0.25_burst_hampel8`; about 80 core-hours) first, then tier 2 (`p0_none_hampel8`,
`p0.05_independent_none`, `p0.05_independent_hampel8`).

Command: `tools/run_here.sh scripts/run_outlier_robustness.py --run-name main --workers 8
--bursts-only-at-025 --skip-window-check` (resumable with `--resume`). The working tree stays
untouched during the run (every job records `git/dirty`).

## 7. Results

(after the run.)

## 8. Caveats

(after the run.)
