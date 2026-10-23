# 27 -- Robustness of the architecture ranking against outliers in the sensor signal

Status: run and evaluated. Design fixed and committed before the run (sections 1-5); run and
results in sections 6-8 (numbers only from the artifacts named there). Pre-registered readings:
**independent outliers R1, bursts R2** -- the R2 hinges on one cell that sharpens, not reverses,
the reference (section 7.4). Branch `outlier-robustness`
(from `origin/master` 4b0ab66), merged into `master` on 2026-10-23; the finding number is 109
(lab book).

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

Records `artifacts/runs/outlier_robustness_main.jsonl.gz` (800, run at `09404ef`, clean tree,
`git/dirty=False`; commit `256fcfc`) and `outlier_robustness_main_timing.json`; analysis
`artifacts/runs/outlier_robustness_analysis_main.json` (script of `8d4407a`, run at `256fcfc`,
clean tree, `code/package_dir` `src/twinlink_ml`; commit `c5341d0`). Every number below is from
these artifacts. Errors in kg/kg, `rmse_run_median_physical_truth` unless stated.

### 7.1 Gate: passed (0 failures)

One commit for all 800 records, `git/dirty` False, preset version 2, process fingerprint
`bddca37f716f5cfc`, zero projected steps, one outlier config digest; every planned record present
once. Reproduction: at `p0_none_none` the metric against the measurement equals the moisture metric
of Befund 94 in all 400 checked values (5 models x 2 budgets x 5 seeds x 2 test sets x 2 horizons x
2 references), largest relative deviation 2.1e-15, with equal fingerprints and equal moisture-head
epochs. The moisture-only training of the networks is therefore exactly the moisture head of
Befund 94.

### 7.2 Outliers and filter

| condition | realised outlier share, pool (min-max over seeds) | holdout | replaced by the filter, pool | holdout |
|---|---|---|---|---|
| `p0_none_hampel8` | 0 | 0 | 0.059-0.065 | 0.041 |
| `p0.05_independent_*` | 0.049-0.051 | 0.051 | 0.067-0.072 (hampel8) | 0.069 |
| `p0.25_independent_*` | 0.249-0.253 | 0.250 | 0.091-0.096 (hampel8) | 0.151 |
| `p0.25_burst_*` | 0.242-0.253 | 0.244 | 0.078-0.082 (hampel8) | 0.100 |

The filter replaces far fewer samples than there are outliers (about 9 % against 25 %), and on clean
data (`p = 0`) it replaces about 6 % of the samples.

### 7.3 Holm verdicts per condition (family = 72 rank cells of the condition)

B / W / U = candidate better / candidate worse / unresolved. Net order: GRU-LSTM, Transformer-LSTM,
Transformer-GRU (24 cells); nets against baselines: baseline as candidate (48 cells; W = network
ahead = LEAD).

| condition | net order, Holm | nets vs baselines, Holm | net order, uncorrected | nets vs baselines, uncorrected |
|---|---|---|---|---|
| `p0_none_none` (C0) | 0/0/24 | 0/23/25 | 0/13/11 | 1/35/12 |
| `p0_none_hampel8` | 0/0/24 | 0/25/23 | 0/7/17 | 1/39/8 |
| `p0.05_independent_none` | 0/0/24 | 0/22/26 | 0/7/17 | 0/35/13 |
| `p0.05_independent_hampel8` | 0/0/24 | 0/23/25 | 0/10/14 | 1/34/13 |
| `p0.25_independent_none` | 0/0/24 | 0/16/32 | 0/1/23 | 2/31/15 |
| `p0.25_independent_hampel8` | 0/0/24 | 0/26/22 | 0/5/19 | 4/33/11 |
| `p0.25_burst_none` | 0/1/23 | 0/18/30 | 0/3/21 | 1/32/15 |
| `p0.25_burst_hampel8` | 0/0/24 | 0/17/31 | 0/0/24 | 2/30/16 |

No baseline is ahead of a network after Holm in any condition. Already at `p = 0` no net-order cell
is resolved after Holm (13 of 24 uncorrected): on moisture the order of the networks is not resolved
after Holm in this family, consistent with Befund 98.

### 7.4 Readings (pre-registered, Holm)

| type | without filter (`p0.25_T_none`) | with filter (`p0.25_T_hampel8`) | reading |
|---|---|---|---|
| independent | unchanged; LEAD 16 (C0 23); no reversal, no new cell | unchanged; LEAD 26 | **R1** |
| burst | **changed**: criterion (c) fails on one cell; LEAD 18 | unchanged; LEAD 17 | **R2** |

Control `p0_none_hampel8`: ranking unchanged (LEAD 25). `p = 0.05` (descriptive): unchanged for
both filter settings (LEAD 22 / 23).

**The R2 of the bursts hinges on one cell, and that cell is not a reordering.** In
`p0.25_burst_none` the cell Transformer against GRU, `n_train` 16, excited test set, h = 1 becomes
resolved after Holm, "Transformer worse" (p 4.5e-4, Holm-adjusted 0.028, ratio of the medians
1.093, worse in 5 of 5 seeds). In C0 the same cell points in the same direction (uncorrected
resolved, p 0.015, ratio 2.03, 5 of 5 seeds) but is not resolved after Holm (adjusted 0.561). With
the filter it is unresolved again (p 0.234, ratio 1.094, 3 of 5). Criterion (c) counts every new
resolution in the net-order block as a change; it was meant for a tie that becomes an order, and
here it catches a sharpening in the direction the reference already had. There is no reversal and
no new cell with a baseline ahead in any condition. **The reading stays R2 as fixed beforehand;
read as content, the burst result is closer to R1 (no reordering). The consequence (filter as
mandatory part of the pipeline, recommendation for AP 2) is left to the project lead.**

Readings with other forms (descriptive, not used):

* **Uncorrected verdicts: no reading for either type** (criteria a, b, c fail with and without the
  filter). Uncorrected reversals: ARX ahead of LSTM at `n_train` 64, excited, h = 1 (independent,
  with and without filter) and ARX ahead of GRU at `n_train` 16, excited, h = 1 (independent with
  filter); new uncorrected baseline-ahead cells: ARX against Transformer (`n_train` 64, excited,
  h = 1, independent without filter), ARX against LSTM (`n_train` 16, excited, h = 1, independent
  with filter), persistence against GRU and Transformer (`n_train` 16, constant-run holdout, h = 36,
  bursts). None of them survives Holm.
* **Against the disturbed measurement: R1 for both types** (LEAD independent 12 / 18, burst 13 / 17
  without / with filter).

### 7.5 Order by median (mean rank over the 8 cells, 1 = best)

| condition | order (mean rank) |
|---|---|
| `p0_none_none` | GRU 1.38, LSTM 1.62, Transformer 3.75, persistence 4.12, ARX 4.12 |
| `p0_none_hampel8` | GRU 1.50, LSTM 1.50, Transformer 3.50, persistence 4.12, ARX 4.38 |
| `p0.05_independent_none` | GRU 1.50, LSTM 1.75, Transformer 3.25, persistence 4.00, ARX 4.50 |
| `p0.05_independent_hampel8` | LSTM 1.38, GRU 1.62, Transformer 3.50, persistence 4.12, ARX 4.38 |
| `p0.25_independent_none` | GRU 1.62, Transformer 2.62, LSTM 3.12, ARX 3.62, persistence 4.00 |
| `p0.25_independent_hampel8` | LSTM 2.00, GRU 2.75, Transformer 3.12, persistence 3.38, ARX 3.75 |
| `p0.25_burst_none` | LSTM 1.50, GRU 1.75, Transformer 3.00, persistence 4.25, ARX 4.50 |
| `p0.25_burst_hampel8` | LSTM 1.62, GRU 2.38, Transformer 2.50, persistence 4.12, ARX 4.38 |

Networks before baselines by median in every condition; within the networks the medians move (at
`p0.25_independent_none` LSTM falls to third by mean rank; in the cell `n_train` 16, constant-run
holdout, h = 36 its median is 5.83e-3 against 1.04e-3 in C0, the worst of the five models); none
of these moves is resolved after Holm.

### 7.6 Error levels (median over seeds, kg/kg)

`n_train` 64, excited test set, h = 36:

| condition | GRU | LSTM | Transformer | ARX | persistence |
|---|---|---|---|---|---|
| `p0_none_none` | 6.32e-4 | 5.44e-4 | 9.63e-4 | 6.75e-3 | 1.18e-2 |
| `p0_none_hampel8` | 1.06e-3 | 1.08e-3 | 1.22e-3 | 7.30e-3 | 1.20e-2 |
| `p0.25_independent_none` | 1.32e-3 | 1.61e-3 | 1.37e-3 | 7.46e-3 | 1.20e-2 |
| `p0.25_independent_hampel8` | 1.35e-3 | 1.21e-3 | 1.37e-3 | 7.38e-3 | 1.22e-2 |
| `p0.25_burst_none` | 1.87e-3 | 1.65e-3 | 1.80e-3 | 7.22e-3 | 1.18e-2 |
| `p0.25_burst_hampel8` | 1.97e-3 | 1.78e-3 | 2.07e-3 | 7.65e-3 | 1.20e-2 |

`n_train` 16, excited test set, h = 1:

| condition | GRU | LSTM | Transformer | ARX | persistence |
|---|---|---|---|---|---|
| `p0_none_none` | 3.22e-4 | 3.40e-4 | 6.54e-4 | 4.85e-4 | 1.27e-3 |
| `p0_none_hampel8` | 1.08e-3 | 1.08e-3 | 1.34e-3 | 1.24e-3 | 1.85e-3 |
| `p0.25_independent_none` | 1.86e-3 | 1.87e-3 | 1.92e-3 | 1.73e-3 | 2.37e-3 |
| `p0.25_independent_hampel8` | 1.71e-3 | 1.72e-3 | 1.76e-3 | 1.55e-3 | 2.38e-3 |
| `p0.25_burst_none` | 1.68e-3 | 1.69e-3 | 1.83e-3 | 1.84e-3 | 2.30e-3 |
| `p0.25_burst_hampel8` | 2.16e-3 | 2.12e-3 | 2.36e-3 | 2.24e-3 | 2.75e-3 |

Observations, descriptive (medians, not resolved statements): (1) on the excited test set at
h = 36 the networks stay far ahead of ARX and persistence under every outlier condition; on the
constant-run holdout at `n_train` 16, h = 36, persistence has the lowest median under independent
outliers (with and without filter); (2) at h = 1 the lead shrinks to the level of the
outlier-induced error, and under independent outliers ARX has the lowest median at h = 1 on the
excited test set (both budgets, with and without filter; not resolved after Holm); (3) the causal
Hampel filter with the fixed settings does not restore the clean error level: on clean data it
raises the h = 1 error of every model on the excited test set (GRU 3.22e-4 -> 1.08e-3 in the cell
above), and with bursts it makes the h = 1 error on the excited test set larger for every model
(both budgets), not smaller.

### 7.7 Run time

Wall time 501.1 min (2026-10-22 18:27 to 2026-10-23 02:48), 8 workers, 120 jobs, 66.2 core-hours
(sum of the job wall times). Median job time: Transformer `n_train` 64 1.09 h (largest 1.66 h,
`p0.25_independent_hampel8` seed 3), Transformer `n_train` 16 0.33 h, GRU/LSTM/baselines job
0.16 h. The estimate (128 core-hours, 16.2 h) was about twice too high: the probe ran while another
computation occupied the machine, and the Transformer at `n_train` 64 was assumed at 2.2 h per job.
The two cuts were made by the rule on the estimate as it stood and are not reversed; at the observed
job times the cut parts (bursts at `p = 0.05`, window side check) would cost roughly 19 core-hours
(extrapolation from the job medians, not measured).

## 8. Caveats

* All outlier and filter parameters are placeholders (section 2); only the 17-34 % share has a
  source. One process configuration with placeholder parameters (GAB placeholder).
* Five seeds, four degrees of freedom; Holm over 72 correlated cells is conservative; the cells of a
  condition are not independent.
* The net order on moisture is not resolved after Holm already at `p = 0`, so criterion (c) can only
  register new resolutions; that is how the burst R2 arose (section 7.4).
* The main metric needs the noise-free truth, which exists only in simulation; against the disturbed
  measurement (what a plant check sees) the reading is R1 for both types.
* The Hampel filter was not tuned (by design); a filter with other settings may behave differently.
  It replaces about 9 % of the samples at an outlier share of 25 %.
* Not run (budget cuts): bursts at `p = 0.05`, window side check (GRU, windows 4 and 16).
* Not covered: xLSTM, Mamba, static regression; energy and bed temperature; `n_train` 8 and 32;
  SOFT_SENSOR mode; outliers on other channels.
* The branch starts at `origin/master` (4b0ab66), not at the local `master` with Befund 107.
