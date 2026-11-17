# NN -- Hybrid against GRU against calibration with a structural error to the UNSAFE side

Status: design, strength of the error, cost measure and reading fixed and committed before any
run (sections 1-8); dry run, projection and cut decision in section 9, added before the main run;
main run and results from section 10 on, added after the run (numbers only from the artifacts
named there). Branch `hybrid-structural-2` (worktree `C:/twinlink_work/hybs2`, from
`origin/master` 0e6e3fe, which contains the merge of `hybrid-structural`). Finding number: open
("Befund NN (number open)" in the lab book); document number assigned at merge. Unattended run
from 2026-11-16; no merge, no push.

## 1. Question

Befund 124 (docs/41): with a structural error that makes the truth DRIER than the prior model
predicts, the prior model M0 and the hybrid M3 (grey-box + GRU on the residual) need no margin and
pay for it in energy (M0 29.5 %, M3 16.2 % above the oracle at `n_train` 16, against 8.1-9.1 % for
M1, M2 and M2b); the pre-registered reading on the margin came out "mixed", and docs/41 section 12
notes that a structural error of the other sign would turn the conservatism of M0 into violations.
Here the error has that other sign: the bed is WETTER than the prior model predicts. The reading
is on the energy with the limit held: which model reaches the limit at the lowest energy --
calibration with a bias (M2b), the hybrid (M3) or the pure GRU (M1)?

## 2. Truth with a structural error to the unsafe side

* **Parameters**: the 20 draws of Befund 111 (as Befund 122 and 124).
* **Structure**: exhaust recirculation, an EXISTING switch of the grey-box model
  (`OperationConfig.exhaust_recirculation_fraction`, default `None` = off; off is bit-identical to
  every earlier result, `tests/test_exhaust_recirculation.py`). A fraction `r` of the dry air
  through the bed is taken from the bed exhaust (`T_bed`, `Y_out`) and mixed with fresh ambient air
  ahead of the heater, which brings the mixture to the commanded inlet temperature. The bed sees
  the same inlet temperature and gas velocity but a higher inlet humidity, dries less and is
  wetter. The switch is part of the right-hand side, of the steady solvers and of the energy
  target (the heater intake is the mixture), so the truth needs no emulation by the inputs: it is
  the grey-box model of the config with the switch set, for every data pool and every steady
  evaluation (`src/twinlink_ml/hybrid_structural_2/study.py`). No new switch ("gas bypass") was
  added (section 8, decision 1).
* **Prior knowledge** (M0, basis of M2, M2b and M3): the model WITHOUT the term, nominal
  parameters.

**Strength, fixed before any run.** The request sets the size by what M0 does: it has to
underestimate the moisture at its own operating point by 0.03-0.07 kg/kg in the median (mirror
image of the +0.070 of Befund 124). Rule (config, section `structure`): M0 is optimised with the
optimiser of the study at margin 0 at each of the 100 interior cases (5 per draw, 20 draws); the
bias of a case is M0's predicted moisture minus the moisture of the truth at that point; among the
candidates 0.20, 0.25, ..., 0.60 the fraction whose median bias over the cases with a steady truth
is closest to -0.05 kg/kg, the middle of the band. Result
(`artifacts/runs/hybrid_structural_2_strength_v1.json`, script a1830c0, clean tree):
**fraction 0.35, median bias -0.0523 kg/kg** (neighbours: 0.30 -> -0.0436, 0.40 -> -0.0638; range
over the cases at 0.35: -0.1834 to +0.0128, quartiles -0.0865 and -0.0265). Without the term the
parameter draws alone give a median of -0.0126 (range -0.1028 to +0.0374).

*Amendment of the rule before the selection.* The rule as first committed (cb10bfd) admitted only
candidates at which the truth has a steady state at all 100 points. No candidate meets that, and
neither does the truth without the term: at 4 of the 100 points (lower end of the gas velocity)
the water balance of the draw has no steady state; with the term at 0.35 there are 8 such points.
The first probe run therefore stopped without a selection and without an artifact, after printing
the medians of the candidates. The condition was dropped (a1830c0) and the statistic stated as the
median over the cases with a steady truth, which is how the +0.070 of Befund 124 was computed. A
point without a steady truth lies on the wet side (the bed cannot evaporate the sprayed water);
counting these points as the wettest gives a median of -0.0542 at 0.35 (-0.0463 at 0.30, -0.0761
at 0.40), so the selection is the same under both conventions. The amendment was made after the
medians were visible; it is reported here for that reason (section 8, decision 3).

**Bed-to-exhaust gap: 0 K.** The purge leaves the ideally mixed bed at the bed temperature; the
recirculation changes the inlet state only. Unlike the truth of Befund 124 (gap 7.42 K) this truth
shows no gap between bed and exhaust.

What the term does at the nominal point (same artifact; nominal commands, disturbances and
material parameters): product moisture 0.0356 instead of 0.0334 kg/kg (+0.0022), bed temperature
+0.05 K, inlet humidity 0.0126 instead of 0.0073 kg/kg. The effect is small at the nominal point
and large at M0's operating points, which lie near the limit moisture at low gas velocity. Only
the moisture of M0 against the truth was looked at; no energy of the truth, no oracle and no
margin was computed before the pre-registration. **Known before the run:** the recirculated air
is warmer than the ambient air, so the heater of the truth needs less power than the prior model
computes for the same operating point; the energy target of the truth is therefore lower than the
prior model predicts, while its moisture is higher. The cost measure compares every model with
the oracle of the SAME truth.

## 3. Models

As Befund 124 (docs/41 section 3), unchanged: M0 prior knowledge; M1 GRU of the Befund 94
pipeline; M2 refit of critical moisture and nu_max of the prior model; M2b = M2 plus a constant
bias of the product moisture from the training runs; M3 prior knowledge + GRU on the residual. No
M4. `n_train` in {4, 8, 16}; master seeds 1-5 (the same seeds and the same parameter draws as
Befund 124); every model trained anew for the new truth (600 networks, 300 refits). Checkpoints:
`C:/twinlink_runs/hybrid-structural-2/checkpoints/` (outside the repository; path and sha256 in
the training records).

## 4. Reused unchanged from Befund 124

Code (`src/twinlink_ml/hybrid_model/`, `src/twinlink_ml/hybrid_structural/`: model M2b, packed
records; the task functions of `scripts/run_hybrid_evaluation.py`), the one-step operating-point
task of Befund 113 (limit moisture 0.15 kg/kg, placeholder), the fixed-point map of the networks,
the optimiser (grid 21 x 21 + zoom 9 x 9, three levels, recentring, 4 starts), the 140 test cases,
the inside / outside definition (oracle = the truth of THIS study), the margin determination
(95 % quantile over the inside cases, bisection on [0, 0.12], tolerance 1e-4, raise rule of
Befund 111) and determinability (>= 20 pooled inside cases). Data: pool of the Befund 94 pipeline,
arm N, `DataSpec.new_study()` (preset version 2), run-level splits, fixed validation pool. The
config `conf/hybrid_structural_2/hybrid_structural_2.yaml` differs from that of Befund 124 in the
section `structure` only (test).

Changes to shared code, each without a change of behaviour for Befund 124 (tests of
`tests/test_hybrid_structural.py` green): the drivers `scripts/train_hybrid_structural.py` and
`scripts/run_hybrid_structural_evaluation.py` take `--config` (default: the config of Befund 124)
and pass it to their workers; `hybrid_structural.study.build_structural_setup` returns the set-up
named by `structure.mode` (`evaporation`: Befund 124; `recirculation`: this study); every set-up
writes its structural term and its study label into the records (for Befund 124 the same values
as before).

## 5. Cost measure, statistics and pre-registered reading

Full text: docstring of `scripts/analyze_hybrid_structural_2.py` (committed in cb10bfd, before any
data, training or evaluation run and before the strength was computed).

**Cost measure** (per model, master seed and `n_train`): energy relative to the oracle at the
margin that holds in >= 95 % of the inside cases -- the median over the inside cases of
`E_truth(model's operating point at that margin) / E_truth(oracle point)`. A cell is **admissible**
iff it is determinable, its margin is reached within the upper end of the search (0.12 kg/kg) and
the final evaluation holds in >= 95 % of the inside cases; otherwise it is **not admissible**,
counted as such, and takes part in no comparison (no substitute value).

**Statistics**: paired over the master seeds on the seeds at which both cells are admissible
(t-test of the per-seed costs); four families of three tests (`n_train` 4, 8, 16) -- {M3 vs M2b},
{M3 vs M2}, {M1 vs M2b}, {M1 vs M3} --, Holm within each family (5 %); **resolved** iff Holm
rejects AND at least 4 of the 5 seeds have the sign of the mean difference. The resolution
thresholds (95 %, uncorrected) are reported for every test.

* **E0**: at at least one `n_train` M2b has a resolved lower energy than M3, and at none M3 a
  resolved lower energy than M2b -> calibration with a bias is the first choice in both directions
  of the error.
* **E1**: at at least one `n_train` M3 has a resolved lower energy than M2b AND than M2 -> the
  residual pays off when the prior knowledge errs to the unsafe side.
* **E2**: at at least one `n_train` M1 has a resolved lower energy than M2b AND than M3 -> pure
  learning suffices.
* **E3**: nothing resolved -> reported as such, with the thresholds.

If more than one holds or none: "mixed", reported and not reinterpreted.

Without a reading: share of inside cases violated at margin 0 (M0 and M3 named in the request;
expectation of the project lead: above 5 %); margins; rollout errors; violation shares outside the
training range; share of the M2 refits at their parameter bounds.

**Post hoc, separate** (key `post_hoc_befund_124` of the analysis artifact): the same cost
measure, statistics and criteria applied to the existing artifact of Befund 124
(`hybrid_structural_v1_margins_per_draw`, nothing run anew).

## 6. Budget rule and long-run rules

The main run has to be finished by Wednesday 2026-11-18 07:00. If the projection after the dry run
lies beyond that: first `n_train` 4 is dropped, then M0; the cut is decided before the main run
and documented in section 9. Rules of `CLAUDE.md`, "Long runs": dry run over ALL units at minimal
cost before the main run; every unit saved at once; `--resume`; a failing unit is recorded and
does not discard the others; at most 8 workers (physical cores); workers x memory peak below 80 %
of the RAM; the main run started detached from the Claude Code session; stage outputs written
directly as `.jsonl.gz`; watcher with a status block every 6 h
(`C:/twinlink_runs/hybrid-structural-2/status.log`).

## 7. Scripts and artifacts

| stage | script | artifact |
|---|---|---|
| strength | `scripts/probe_hybrid_structural_2_strength.py` | `hybrid_structural_2_strength_v1.json` |
| training | `scripts/train_hybrid_structural.py --config ...` | `hybrid_structural_2_training_v1.jsonl.gz` |
| oracle | `scripts/run_hybrid_structural_evaluation.py --config ... --part oracle` | `hybrid_structural_2_v1_oracle.jsonl.gz` |
| errors | `... --part errors` | `hybrid_structural_2_v1_errors.jsonl.gz` |
| margins | `... --part margins` | `hybrid_structural_2_v1_margins_per_draw.jsonl.gz` |
| analysis | `scripts/analyze_hybrid_structural_2.py` | `hybrid_structural_2_v1_analysis.json` |

Settings `conf/hybrid_structural_2/hybrid_structural_2.yaml`; tests
`tests/test_hybrid_structural_2.py`.

## 8. Decisions without consultation

The request ruled out questions; where it left room, the most conservative reading was taken.

1. **Existing switch instead of a new one.** The request prefers an existing switch of the model
   and names a new "gas bypass" switch as the fallback. The exhaust recirculation exists, is
   tested, is part of the dynamics, the steady solvers and the energy target, and makes the bed
   wetter; no model code was changed. Consequences accepted with it: the bed-to-exhaust gap is
   0 K, and the truth needs less heater power than the prior model computes (section 2).
2. **Quantity that fixes the strength.** "M0 underestimates the moisture at its own operating point
   in the median by 0.03-0.07": taken as the median over the 100 interior cases of the study at
   margin 0, target the middle of the band, candidates on a 0.05 grid, written into the config
   before the selection. The +0.070 of Befund 124 is a median over the inside cases of trained
   cells; the inside classification needs the training pools and the oracle and was not available
   (and not wanted) before the pre-registration, so the interior cases stand in for it.
3. **Amendment of the strength rule** (section 2): the first rule admitted no candidate. The
   replacement keeps target, candidates and band and uses the convention of Befund 124 for points
   without a steady truth; the selection is the same if these points are counted as the wettest.
   A scratch check on four draws at fractions 0.3, 0.6, 0.8 and 0.9 (moisture only, not an
   artifact) preceded the choice of the candidate grid; it showed only that the band is within the
   reach of the switch.
4. **Cost measure made operational** (section 5): "the margin that holds inside in >= 95 % of the
   cases" is the margin the study determines; "not reached within the upper end" also covers a
   cell whose final evaluation holds in fewer than 95 % of the inside cases after the raise rule.
   A cell that is not determinable (fewer than 20 inside cases) has no margin and is listed with
   the not admissible ones, with its reason.
5. **Reading made operational.** One family per pair of models that the criteria compare, three
   tests each; E1 and E2 need both differences at the SAME `n_train`; E2 ("lowest energy against
   M2b and M3") was read like E1 as "at at least one `n_train`"; E3 refers to the twelve tests of
   the four families. A test with fewer than 4 pairs cannot be resolved (sign rule), which is
   reported next to the verdict.
6. **Task, targets, models, seeds, cases unchanged**; the exhaust state is not added as an input.
7. **Stage outputs, chain, checkpoints** as Befund 124 (docs/41 section 8, decisions 7-11 and 13):
   one gzip member per record; the run chain commits each stage output on this branch (denylist
   test first) so that the next stage starts on a clean tree; errors before margins; margins of
   `n_train` 8 and 16 first, `n_train` 4 added only if that part starts before a gate fixed before
   the main run (section 9); the chain also runs the analysis.
8. **Full suite** on the code commit 079508f (after the suite of the managing session had ended;
   no other suite or run was active). Later commits before the main run add documentation only.
9. **Post hoc part** in the same analysis artifact under its own key, labelled, never merged with
   the verdict of this study.

## 9. Dry run, projection and cut decision

**Dry run** (`C:/twinlink_work/hybs2_dry_chain.sh dry1`, commit 82c4863, clean tree, 8 workers,
2026-11-16 11:20-14:43, outputs gitignored `_hybs2_dry1_*`), all units of every stage:

| stage | units | errors | wall time | peak commit per worker |
|---|---|---|---|---|
| training (`--dry-run`) | 900 of 900 | 0 | 4111 s | 755 MB |
| oracle and discretisation check (full) | 30 of 30 | 0 | 1737 s | 445 MB |
| margins, minimal mode, `n_train` 8 and 16 | 2000 of 2000, 50 of 50 cells, none blocked | 0 | 1345 s | 499 MB |
| margins, minimal mode, `n_train` 4 added with `--resume` | 1000 of 1000, 75 of 75 cells, none blocked | 0 | 704 s | 499 MB |
| memory probe of the margins (coarse grid, seed 1, `n_train` 16, draws 0-7, five models) | 80 of 80 | 0 | 932 s | 1247 MB |
| errors (full) | 100 of 100 | 0 | 3333 s | 904 MB |
| analysis (incl. the post hoc part on the artifact of Befund 124) | written | 0 | 15 s | -- |

Memory: 8 workers x 1247 MB = 10.0 GB, 31 % of the 31.9 GB RAM (limit 80 %). The tree was clean
after the chain. The margins and energies of the dry run come from the minimal mode and are not
results; they were not read. Two structural observations from the dry-run analysis: the five cells
of seed 5 at `n_train` 4 have 17 inside cases (fewer than 20, not determinable; all other 70 cells
have at least 20), so every test at `n_train` 4 has at most 4 pairs; the discretisation check of
the oracle (grid + zoom against 8-start Nelder-Mead, 10 cases) shows up to 0.39 % energy
difference, above the 0.2 % threshold, so this error stands next to every energy figure.

**Projection** = stage times of the main run of Befund 124 (docs/41, section 10) x the ratio of
the dry-run times here to the dry-run times there (the truth with recirculation is slower):

| stage | Befund 124 main run | ratio dry here / dry there | projection |
|---|---|---|---|
| training | 24325 s | 4111 / 3826 = 1.07 | 26137 s |
| oracle | 446 s | (dry run is the full stage) | 1737 s |
| errors | 2280 s | (dry run is the full stage) | 3333 s |
| margins `n_train` 8, 16 | 40437 s | 2049 / 1219 = 1.68 (minimal mode); 932 / 881 = 1.06 (memory probe) | 67975 s; 42782 s |
| margins `n_train` 4 | 16454 s | as above | 27659 s; 17408 s |
| sum | | | 35.2 h with the larger ratio; 25.4 h with the smaller |

With a start at about 15:15 on Monday 2026-11-16 the main run ends between Tuesday about 16:40
and Wednesday about 02:30, before Wednesday 07:00 in both cases.

**Cut decision (before the main run): no cut from the start**; all five models and `n_train` 4, 8
and 16 run. Guard as in Befund 124: the margins run `n_train` 8 and 16 first; `n_train` 4 is added
only if that part starts before the gate **Tuesday 2026-11-17 20:00 (+01:00)** (07:00 minus the
7.7 h of the larger projection for that part minus 3 h for analysis, full suite and documents). If
the gate is missed, `n_train` 4 is dropped from the margins (budget rule, first cut); M0 is not
cut after the start.

## 10. Main run

Started detached (`Win32_Process.Create`) on 2026-11-16 14:48:00 on 587970c (code as 079508f, on
which the full suite ran; the later commits add documentation only), clean tree at the start of
every stage except the second part of the margins, which resumes into the not yet committed
output of the first part (the chain commits the margins once, after both parts; as Befund 124).
Ended 2026-11-17 19:33:16: 28.8 h, inside the projection of section 9 (25.4-35.2 h) and before
Wednesday 07:00. No stage needed a second run, no unit failed, nothing was cut: the gate was
passed on 2026-11-17 14:07 and `n_train` 4 was added to the margins.

A first start at 14:45:32 on the same commit ended within about a minute, before any unit had
finished and before anything was written: the project lead closed the two console windows of the
chain and of the watcher, not knowing what they were. The chain log keeps the two lines of that
start. The start at 14:48 is the main run.

| stage | artifact (commit) | units, wall time, memory peak per worker |
|---|---|---|
| training | `hybrid_structural_2_training_v1.jsonl.gz` (748a479) | 900 of 900 units, 0 errors, 26171 s, 787 MB |
| oracle | `hybrid_structural_2_v1_oracle.jsonl.gz` (f24fbb8) | 30 of 30 tasks, 0 errors, 1632 s, 447 MB |
| errors | `hybrid_structural_2_v1_errors.jsonl.gz` (90f08f1) | 100 of 100 tasks, 0 errors, 3279 s, 919 MB |
| margins | `hybrid_structural_2_v1_margins_per_draw.jsonl.gz` (7ffbd8e) | 2000 tasks (`n_train` 8, 16; 52660 s) + 1000 tasks (`n_train` 4; 19417 s), 0 errors, 75 of 75 cells, none blocked, 1318 MB |
| analysis | `hybrid_structural_2_v1_analysis.json` (754f0d5) | reads the four artifacts above and, for the post hoc part, `hybrid_structural_v1_margins_per_draw.jsonl.gz` of Befund 124 |

Watcher: status blocks every 6 h in `C:/twinlink_runs/hybrid-structural-2/status.log`; checkpoints
in `C:/twinlink_runs/hybrid-structural-2/checkpoints/`.

## 11. Results

All figures from `hybrid_structural_2_v1_analysis.json`. Test cases: the interior and
extrapolation cases of section 4; "inside" = inside the training range of the cell. Energy = energy
of the truth at the operating point of the model relative to the oracle, median over the inside
cases. The discretisation error of the oracle (grid + zoom against 8-start Nelder-Mead, 10 cases)
is up to 0.39 % in energy (0.00023 kg/kg in slack); it stands next to every energy figure below.

### 11.1 Pre-registered reading: E3 (nothing resolved)

**Verdict E3.** None of the twelve tests is resolved; E0, E1 and E2 are not met. The reason is not
small differences but missing pairs: **eleven of the twelve tests have fewer than 4 pairs of
admissible cells and cannot be resolved** by the sign rule (4 of 5); the one resolvable test, M1
against M2b at `n_train` 16 (4 pairs), is not resolved (M1 higher by 0.207 on average, threshold
0.348, Holm 0.309, signs 3 of 4).

| family | `n_train` | pairs (seeds) | mean difference reference - candidate | resolution threshold | p (Holm) | signs + / - | resolved |
|---|---|---|---|---|---|---|---|
| M3 vs M2b | 4 | 2 (1, 4) | -0.1803 | 0.1082 | 0.0901 | 0 / 2 | no (not resolvable) |
| M3 vs M2b | 8 | 2 (3, 5) | -0.0594 | 0.8200 | 0.7945 | 1 / 1 | no (not resolvable) |
| M3 vs M2b | 16 | 2 (2, 5) | -0.2240 | 2.0495 | 0.7945 | 0 / 2 | no (not resolvable) |
| M3 vs M2 | 4 | 2 (1, 4) | -0.1778 | 0.0951 | 0.0803 | 0 / 2 | no (not resolvable) |
| M3 vs M2 | 8 | 2 (3, 5) | -0.0611 | 0.8011 | 0.8018 | 1 / 1 | no (not resolvable) |
| M3 vs M2 | 16 | 2 (2, 5) | -0.2231 | 2.0655 | 0.8018 | 0 / 2 | no (not resolvable) |
| M1 vs M2b | 4 | 0 | -- | -- | -- | -- | no (not resolvable) |
| M1 vs M2b | 8 | 2 (3, 5) | -0.0022 | 0.7277 | 0.9755 | 1 / 1 | no (not resolvable) |
| M1 vs M2b | 16 | 4 (2, 3, 4, 5) | -0.2071 | 0.3479 | 0.3090 | 1 / 3 | no |
| M1 vs M3 | 4 | 0 | -- | -- | -- | -- | no (not resolvable) |
| M1 vs M3 | 8 | 2 (3, 5) | +0.0572 | 0.0923 | 0.1609 | 2 / 0 | no (not resolvable) |
| M1 vs M3 | 16 | 2 (2, 5) | +0.0662 | 0.1122 | 0.1609 | 2 / 0 | no (not resolvable) |

A negative difference means the candidate (first model) uses more energy than the reference.

### 11.2 Admissibility (counted as pre-registered, no substitute value)

Of 5 cells (seeds) per model and `n_train`: admissible / margin not reached within the upper end
0.12 / not determinable (fewer than 20 inside cases; seed 5 at `n_train` 4).

| model | `n_train` 4 | `n_train` 8 | `n_train` 16 | admissible of 15 |
|---|---|---|---|---|
| M0 | 3 / 1 / 1 | 3 / 2 / 0 | 5 / 0 / 0 | 11 |
| M1 | 0 / 4 / 1 | 2 / 3 / 0 | 4 / 1 / 0 | 6 |
| M2 | 3 / 1 / 1 | 5 / 0 / 0 | 5 / 0 / 0 | 13 |
| M2b | 3 / 1 / 1 | 5 / 0 / 0 | 5 / 0 / 0 | 13 |
| M3 | 2 / 2 / 1 | 2 / 3 / 0 | 2 / 3 / 0 | 6 |

The two calibrations are admissible in every determinable cell but one; the hybrid and the GRU
fail to hold the limit in >= 95 % of the inside cases within the largest allowed margin in 8 of
their 14 determinable cells each, M0 in 3 of 14.

### 11.3 Energy at the held limit and margins (admissible cells only)

Energy relative to the oracle: median over the admissible seeds (minimum-maximum; number of
seeds).

| model | `n_train` 4 | `n_train` 8 | `n_train` 16 |
|---|---|---|---|
| M0 | 1.421 (1.354-1.430; 3) | 1.385 (1.115-1.431; 3) | 1.385 (1.111-1.427; 5) |
| M1 | -- (0) | 1.197 (1.071-1.323; 2) | 1.368 (1.075-1.463; 4) |
| M2 | 1.017 (1.015-1.368; 3) | 1.016 (1.011-1.403; 5) | 1.014 (1.012-1.407; 5) |
| M2b | 1.015 (1.012-1.375; 3) | 1.016 (1.012-1.405; 5) | 1.014 (1.012-1.405; 5) |
| M3 | 1.194 (1.187-1.201; 2) | 1.254 (1.136-1.373; 2) | 1.434 (1.400-1.468; 2) |

Margin in kg/kg: median over the admissible seeds (minimum-maximum).

| model | `n_train` 4 | `n_train` 8 | `n_train` 16 |
|---|---|---|---|
| M0 | 0.1186 (0.1127-0.1186) | 0.1127 (0.0700-0.1186) | 0.1148 (0.0700-0.1186) |
| M1 | -- | 0.0903 (0.0736-0.1070) | 0.1100 (0.0521-0.1187) |
| M2 | 0.0135 (0.0043-0.1077) | 0.0145 (0.0074-0.1106) | 0.0112 (0.0097-0.1100) |
| M2b | 0.0128 (0.0064-0.1090) | 0.0145 (0.0083-0.1107) | 0.0124 (0.0106-0.1104) |
| M3 | 0.0713 (0.0608-0.0818) | 0.0881 (0.0663-0.1099) | 0.1127 (0.1097-0.1157) |

The calibrations are bimodal over the seeds: most cells hold the limit with a margin of about
0.01 kg/kg at 1.1-1.7 % energy above the oracle, single cells (M2 and M2b: seed 3 at `n_train` 4,
seeds 2 and 3 at `n_train` 8, seed 5 at `n_train` 16) need about 0.11 kg/kg and 37-41 % more
energy. These comparisons are descriptive; none is resolved (section 11.1).

### 11.4 Descriptive, without a reading

**Violation share at margin 0** (inside cases of the determinable cells, pooled over seeds;
`n_train` 4 / 8 / 16): M0 95.3 / 96.2 / 96.6 %, M1 79.1 / 91.1 / 94.3 %, M2 67.4 / 82.3 / 89.1 %,
M2b 72.9 / 86.5 / 86.6 %, M3 59.7 / 73.8 / 66.0 %. The expectation of the project lead (M0 and M3
above 5 %) holds; without a margin every model violates the limit in the majority of the cases.

**Violation rates outside the training range** at the determined margin ("holds outside" = at
most 5 % violations; (a) extrapolation contexts, 200 cases; (b) interior cases outside the
coverage of the cell). (a), `n_train` 4 / 8 / 16: M0 7.5 / 7.5 / 7.5 %, M1 20.0 / 9.0 / 7.5 %,
M2 9.5 / 9.0 / 8.5 %, M2b 10.0 / 10.0 / 9.0 %, M3 8.0 / 8.5 / 12.0 %. (b): M0 5.4 / 4.2 / 6.0 %,
M1 13.6 / 6.5 / 6.0 %, M2 7.1 / 7.6 / 9.3 %, M2b 7.9 / 8.7 / 9.3 %, M3 5.4 / 6.1 / 10.7 %. The
margin holds outside only for M0 in (b) at `n_train` 8.

**Rollout error of the product moisture** (36 steps, RMSE in kg/kg, median over the 100 units;
excited test set / constant-run holdout): M0 0.0362 / 0.0528 (bias -0.0157 / -0.0260); M1 0.0377 /
0.0490, 0.0279 / 0.0383, 0.0107 / 0.0220 at `n_train` 4, 8, 16; M2 0.0380 / 0.0676, 0.0400 /
0.0679, 0.0031 / 0.0045; M2b 0.0381 / 0.0678, 0.0401 / 0.0681, 0.0027 / 0.0045; M3 0.0054 /
0.0064, 0.0046 / 0.0049, 0.0038 / 0.0044. The hybrid has the smallest rollout error at
`n_train` 4 and 8 and is admissible least often: the rollout error on the test sets does not order
the models as the held limit at their own optimised operating point does.

**M2 refits at parameter bounds:** 56 / 58 / 29 of 100 refits at `n_train` 4 / 8 / 16 end at a
bound of at least one parameter (critical moisture lower 4 / 7 / 10, upper 17 / 17 / 6; `nu_max`
upper 35 / 34 / 13); none at a bound of both.

## 12. Post hoc (not pre-registered for that data): the same measure on Befund 124

Key `post_hoc_befund_124` of the analysis artifact: the cost measure, statistics and criteria of
section 5 applied to the existing margins artifact of Befund 124 (structural error to the SAFE
side); nothing was run anew. This part is not a verdict of this study and does not change the
reading "mixed" of Befund 124, which was pre-registered on the margin.

Read by the criteria of this study, that artifact gives **E0**: M2b has a resolved lower energy
than M3 at `n_train` 16 (difference 0.0945, threshold 0.0411, Holm 0.0093, 5 of 5 seeds) and M3 is
nowhere resolved lower. Also resolved at `n_train` 16: M2 lower than M3 (0.0850, threshold 0.0485,
Holm 0.0247, 5 of 5) and M1 lower than M3 (0.0929, threshold 0.0637, Holm 0.0465, 5 of 5); M1
against M2b is not resolved (0.0016, threshold 0.0283). The four tests at `n_train` 4 have fewer
than 4 pairs. Admissibility there: every determinable cell except M3 at `n_train` 4, seed 1.
Energy relative to the oracle (median, `n_train` 4 / 8 / 16): M0 1.311 / 1.307 / 1.295, M1 1.164 /
1.090 / 1.082, M2 1.118 / 1.091 / 1.091, M2b 1.076 / 1.076 / 1.081, M3 1.178 / 1.153 / 1.162.

Both directions side by side (the second post hoc): with the error to the unsafe side nothing is
resolved because the learned models are mostly not admissible; with the error to the safe side
the calibration with bias is resolved cheaper than the hybrid at `n_train` 16. "First choice in
both directions" (E0) is therefore met post hoc for the safe side and NOT established for the
unsafe side.

## 13. Caveats

- The verdict E3 rests on missing pairs, not on small differences. The pre-registered rule
  excludes not admissible cells without a substitute value, so the statistics cannot express
  that a model fails to hold the limit; the admissibility counts of section 11.2 carry that
  information and have no pre-registered test.
- The upper end of the margin (0.12 kg/kg at a limit of 0.15) is a setting of Befund 122; "not
  reached" refers to it.
- Energies of cells that need about 0.11 kg/kg lie 37-47 % above the oracle; at such margins the
  comparison measures the cost of retreating from the limit, not model accuracy.
- Discretisation error of the oracle up to 0.39 % energy, above the 0.2 % reporting threshold;
  the differences among the calibrations at small margins (1.1-1.7 %) are of that order.
- `n_train` 4: seed 5 is not determinable (17 inside cases); at most 4 pairs.
- One structural term (exhaust recirculation, fraction 0.35), one strength; the rule for the
  strength was amended once after medians were visible (section 2). Bed-exhaust gap 0 K by
  construction; the heater power of the truth is lower than the prior computes.
- Why M3 and M1 fail to hold the limit although their rollout errors are small was not
  investigated; no cause is claimed.
- Limit moisture and kinetics are placeholders; all data synthetic; no statement about a plant.

## 14. Open

- Cause of the missing admissibility of M3 and M1 (operating points the optimiser picks against
  the cases of the test sets; seeds that need about 0.11 kg/kg also for the calibrations).
- A pre-registered test on admissibility itself (share of cells that hold the limit), which the
  reading of this study lacks.
- Number of the finding and of this document: assigned by the managing session at the merge.
