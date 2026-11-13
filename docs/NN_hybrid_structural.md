# NN -- Hybrid against GRU against parameter refit with a STRUCTURALLY deviating truth

Status: design, dry run and cut decision fixed and committed before the main run (sections 1-9);
main run and results in sections 10-13, added after the run (numbers only from the artifacts named there). Branch
`hybrid-structural` (worktree `C:/twinlink_work/hybs`, from `origin/master` 68a805b). Finding
number: open ("Befund NN (number open)" in the lab book); document number assigned at merge.
Unattended run over the weekend of 2026-11-13; no merge, no push.

## 1. Question

docs/12 section I, item 30. Befund 122 (docs/39): with a truth of the same structure as the prior
model, a refit of two kinetic parameters (M2) needs margins of 0.0019-0.0039 kg/kg and the hybrid
(M3, grey-box + GRU on the residual) is not resolvably better than a pure GRU (M1); that truth
favours M2 by construction. Befund 117: the grey-box model does not match plant A structurally
(bed-to-exhaust gap of 5-10 K). Here the truth of the simulation study deviates structurally from
the prior model: does the learned residual now pay off against a calibration of the prior model,
with or without a constant bias?

## 2. Truth with a structural deviation

* **Parameters**: the 20 draws of Befund 111 (as Befund 122).
* **Structure**: a fixed share `f` of the spray water `W = s (1 - x_s)` evaporates in the freeboard
  above the bed instead of in the bed -- candidate C2 of Befund 117 (`plant_scale.model.Freeboard`,
  mode `evaporation`), with the same emulation by the inputs: the bed receives the spray rate
  `s - f W` with the solid fraction `x_s s / (s - f W)` (solid feed unchanged); the gas leaving the
  bed evaporates `f W`, `m_L c_h (T_bed - T_exh) = f W (r_0 + c_v T_exh - c_w T_feed)`. Code
  `src/twinlink_ml/hybrid_structural/truth.py`; the grey-box code is not changed. The simulator of
  every data pool sees the bed feed in its right-hand side (commands, disturbances, clipping,
  projection and sensor model stay those of the commanded inputs); the steady truth is the config
  whose nominal spray rate and solid fraction are the bed feed. Tests: switched off the truth is
  the grey-box model bit for bit; simulator and steady config have the same steady state; the
  exhaust balance equals `plant_scale.model.exhaust_after_freeboard`.
* **Prior knowledge** (M0, basis of M2, M2b and M3): the model WITHOUT this term, nominal
  parameters, as Befund 122.

**Share, fixed before any run.** Rule committed in the config before the share was computed
(979cd1b): among the candidates 0.05, 0.10, ..., 0.60 the share whose bed-to-exhaust gap at the
nominal point (nominal commands and disturbances of the process config: inlet 90 degC, gas velocity
1.6 m/s, spray 1.5 kg/h, ambient 20 degC / 0.50, solid fraction 0.30; nominal material parameters)
is closest to 7.5 K, the middle of the 5-10 K of Befund 117. Result
(`artifacts/runs/hybrid_structural_share_v1.json`, script 979cd1b, clean tree): **share 0.30, gap
7.42 K** (neighbours: 0.25 -> 6.16 K, 0.35 -> 8.69 K; over the 20 draws at 0.30: 7.42-7.43 K).

What the term does to the bed at the nominal point (same artifact; "without" = the prior
structure with the same parameters): product moisture 0.0213 instead of 0.0334 kg/kg (-0.0121;
over the 20 draws -0.0208 to -0.0092, median -0.0130), bed temperature +6.23 K (draws 6.17-6.26 K),
energy target -1.76 %. **Known before the run:** the truth is drier and warmer than the prior
structure predicts, i.e. the prior model errs to the safe side in the moisture; a model that keeps
the prior structure can therefore need a small margin and pay in energy instead. Margin and energy
relative to the oracle are both reported. Side effect of the emulation (as in Befund 117): pump
and atomiser power are computed with the bed's spray rate; they are 2.5 % of the total power at the
nominal point.

The exhaust temperature is not a target of the study (targets as Befund 122: energy target,
product moisture, bed temperature), so no model is given the gap itself; the models see the
deviation only through its effect on the bed.

## 3. Models

| | model | data |
|---|---|---|
| M0 | prior knowledge only (nominal grey-box, no freeboard term) | none |
| M1 | GRU of the Befund 94 pipeline (`gru`, 200 epochs, training seed of the seed bundle) | truth pool |
| M2 | refit of critical moisture and nu_max of the prior model (as Befund 122: least squares on all three targets of the training runs, open-loop replay, bounds [0.15, 0.45] and [0.2, 1.0]) | truth pool |
| M2b | M2 plus a constant bias of the product moisture: mean of `measured - replay of the refitted model` over every sample of the training runs | truth pool |
| M3 | prior knowledge + GRU on the residual `truth - nominal replay` (same architecture, budget and training seed as M1) | residual pool |

No M4. `n_train` in {4, 8, 16}; master seeds 1-5; every model trained anew for the new truth
(600 networks, 300 refits). Checkpoints: `C:/twinlink_runs/hybrid-structural/checkpoints/`
(outside the repository; path and sha256 in the training records).

## 4. Reused unchanged from Befund 122

Code (`src/twinlink_ml/hybrid_model/`, the task functions of `scripts/run_hybrid_evaluation.py`),
the one-step operating-point task of Befund 113 (limit moisture 0.15 kg/kg, placeholder; bed
80 degC), the fixed-point map of the networks, the optimiser (grid 21 x 21 + zoom 9 x 9, three
levels, recentring, 4 starts), the 140 test cases (5 interior and 2 extrapolation contexts per
draw, rotation of the config), the inside / outside definition (Chebyshev distance <= 0.2 of the
oracle operating point to a training sample; oracle = the truth of THIS study), the margin
determination (95 % quantile over the inside cases of the per-case needed margin, bisection on
[0, 0.12], tolerance 1e-4, raise rule of Befund 111, "not reached" counted as 0.12) and the
addendum on determinability (a cell needs >= 20 pooled inside cases). Data: pool of the Befund 94
pipeline, arm N, `DataSpec.new_study()` (preset version 2), run-level splits, fixed validation
pool. See docs/39 sections 2-6.

Changes to shared code, each without a change of behaviour for Befund 122 (tests of
`tests/test_hybrid_model.py` green): `generate_data` takes an optional simulator factory;
`StudySetup.generate` (the pool generator of the study, overridden here);
`scripts/run_hybrid_evaluation.py` generates its pools through it, reads its inputs plain or
packed, knows M2b, and keeps the model order of its scheduler as a class attribute. The new
evaluation driver `scripts/run_hybrid_structural_evaluation.py` installs the set-up of this study
in every worker and then calls the task functions of Befund 122.

## 5. Statistics and pre-registered reading

Full text: docstring of `scripts/analyze_hybrid_structural.py` (committed in d341793, before any
run). In short: comparisons paired over the master seeds (t-test of the per-seed margins), two
families of three tests -- {M3 vs M2 at `n_train` 4, 8, 16} and {M3 vs M2b at 4, 8, 16} --, Holm
within each family (5 %), and a difference counts as **resolved** only if Holm rejects it AND at
least 4 of the 5 seeds have the sign of the mean difference. The resolution thresholds (95 %,
uncorrected) are reported for every test.

* **Z0**: at at least one `n_train` <= 16 M3 needs a resolved smaller margin than M2 AND than M2b
  -> with a structural error the residual pays off.
* **Z1**: M2 or M2b needs a resolved smaller margin than M3, or one of equal size, at every
  `n_train` -> calibration (possibly with a bias) stays the first choice with a structural error.
* **Z2**: nothing resolved -> reported as such, with the thresholds.

Without a reading: rollout errors; violation shares outside the training range at the margin
determined inside; the M2 margin against Befund 122 (cost of the structural error).

## 6. Budget rule and long-run rules

The main run has to be finished by Monday 2026-11-16 07:00. If the projection after the dry run
lies beyond that: first `n_train` 4 is dropped, then M0; the cut is decided before the main run
and documented in section 9. Rules of `CLAUDE.md`, "Long runs":
dry run over ALL units at minimal cost before the main run; every unit saved at once; `--resume`;
a failing unit is recorded and does not discard the others; at most 8 workers (physical cores);
workers x memory peak below 80 % of the RAM; the main run started detached from the Claude Code
session; stage outputs written directly as `.jsonl.gz`; no new exception in the clean-tree check
(the only path left out is the output the stage itself writes).

## 7. Scripts and artifacts

| stage | script | artifact |
|---|---|---|
| share | `scripts/probe_hybrid_structural_share.py` | `hybrid_structural_share_v1.json` |
| training | `scripts/train_hybrid_structural.py` | `hybrid_structural_training_v1.jsonl.gz` |
| oracle | `scripts/run_hybrid_structural_evaluation.py --part oracle` | `hybrid_structural_v1_oracle.jsonl.gz` |
| margins | `... --part margins` | `hybrid_structural_v1_margins_per_draw.jsonl.gz` |
| errors | `... --part errors` | `hybrid_structural_v1_errors.jsonl.gz` |
| analysis | `scripts/analyze_hybrid_structural.py` | `hybrid_structural_v1_analysis.json` |

Settings `conf/hybrid_structural/hybrid_structural.yaml`; tests `tests/test_hybrid_structural.py`.

## 8. Decisions without consultation

The request ruled out questions; where it left room, the most conservative reading was taken.

1. **Share by a rule on a grid.** The request asks for a gap "of the order of Befund 117 (5-10 K)"
   at the nominal point. Taken: the middle of the band as the target and a 0.05 grid, written into
   the config before the gap was computed, so that the share is not tuned to any result. "Nominal
   point" = nominal commands and disturbances of the process config with the nominal material
   parameters.
2. **The deviation as in Befund 117, nothing added.** C2 is emulated by the inputs exactly as the
   switch of Befund 117; dynamically the share applies to the current spray water at every
   instant (no freeboard hold-up, no dependence of the share on the operating point). The side
   effect on pump and atomiser power is kept and reported rather than corrected, since a
   correction would need a change of the grey-box code.
3. **Task and targets unchanged.** The exhaust temperature is not added as a target or input:
   "task unchanged" was read strictly. The models therefore see the deviation only through the bed.
4. **M2b.** "Constant bias of the moisture (from the training runs)": the mean of measured minus
   open-loop replay of the refitted model over all samples of the training runs, moisture only,
   added to the steady map and to the replay. No joint re-estimation of parameters and bias (that
   would be another model than "M2 + bias").
5. **Reading made operational.** Families: one per reference model (M2, M2b), three tests each
   (`n_train`); Holm within each family. Sign consistency: at least 4 seeds with the sign of the
   mean difference, a zero difference counting for neither. Z0 needs both references at the SAME
   `n_train`. Z1: the SAME model (M2 or M2b) at every `n_train`; "of equal size" = not resolved in
   either direction and the median margin of that model not larger than that of M3; and Z1 needs at
   least one resolved difference, otherwise the verdict is Z2 (nothing resolved is reported as not
   resolved, not as a preference). Order Z0, Z2, Z1, else "mixed". All of this is in the docstring
   committed before any run.
6. **Master seeds 1-5** (seeds 1-3 are those of Befund 122; 4 and 5 are new).
7. **Stage outputs** as `.jsonl.gz` with one gzip member per record (written and synced per unit;
   an incomplete last member after a kill is cut off before a resume). Between the stages the
   run chain commits the stage output on this branch (after the denylist test), so that the next
   stage starts on a clean tree; nothing is merged or pushed.
8. **Hooks instead of a copy** of the Befund 122 driver (section 4), so that the task functions are
   literally the same code.
9. **Checkpoints** are written directly to `C:/twinlink_runs/hybrid-structural/checkpoints/`.
10. **Dry run of the training**: one task per (draw, seed) runs its nine units through the same
    unit function (one epoch, three refit evaluations), so that every unit is covered without
    generating each pool nine times.
11. **The run chain also runs the analysis** when the errors stage is done, so that the
    pre-registered verdict exists even if the session has ended by then.
12. **Full suite** on the code commit of the main run (d341793; the commit that adds this document
    changes documentation only); intermediate commits were made after the targeted tests
    (denylist, fingerprints, experiments, hybrid model, hybrid structural).
13. **Budget rule applied as a gate inside the run** (section 9): the projection lies below the
    budget but with a reserve of about 10 h only, and the scheduler of Befund 122 orders the margin
    tasks by model first, so that an overrun would hit M2, M2b and M0 instead of `n_train` 4. The
    run chain therefore computes the margins of `n_train` 8 and 16 first and adds `n_train` 4 to
    the same output only if that part starts before 2026-11-15 15:00; otherwise `n_train` 4 is
    dropped from the margins, as the request orders for a projection beyond the budget. The stage
    of the errors runs before the margins (it needs only the training records and the oracle).
    Decided and committed before the main run.
14. **Determinability is not repaired.** The dry run shows that 3 of the 15 (seed, `n_train`)
    combinations have fewer than 20 inside cases (section 9). The rule of Befund 122 (such a cell
    is not determinable) and the pre-registered reading (at least 4 seeds with the sign of the
    mean) are kept as committed; no seed is replaced and no threshold is changed.

## 9. Dry run, projection and cut decision (before the main run)

**Dry run over all units** (d341793, clean tree, gitignored outputs `_hybs_dry1_*`, 2026-11-13
14:52-17:30, 8 workers): training 900 of 900 units, 0 errors (one epoch per network, three
evaluations per refit; 3826 s; peak commit 740 MB per worker); oracle and discretisation check 30
of 30 tasks, 0 errors (569 s; 422 MB); margins in the minimal mode 3000 of 3000 tasks, 0 errors, 75
of 75 cells, none blocked (1219 s; 499 MB); memory probe of the margins with the coarse grid on
8 draws (seed 1, `n_train` 16, all five models) 80 of 80 tasks, 0 errors (881 s; **peak commit
1245 MB per worker**); errors 100 of 100 tasks, 0 errors (2909 s; 882 MB); analysis written. A
second dry run (`_hybs_dry2_*`, 17:33-17:56) covers the two-phase margins stage of decision 13:
2000 tasks for `n_train` 8 and 16, then 1000 tasks added with `--resume`, 0 errors, 3000 distinct
keys, 75 of 75 cells; the analysis runs with and without `n_train` 4 (without: `n_train` 4 listed
as not assessed). Memory: 8 workers x 1.25 GB = 10 GB, below 80 % of the 31.2 GB RAM.

**Inside cases (known before the main run).** The number of inside cases depends on the training
pool and the oracle only, not on the trained model, so the dry run gives the final counts
(`_hybs_dry1_analysis.json`, not committed; the main run reproduces them in
`hybrid_structural_v1_analysis.json`):

| `n_train` | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
|---|---|---|---|---|---|
| 4 | 37 | 16 | 51 | 33 | 0 |
| 8 | 46 | 16 | 67 | 46 | 48 |
| 16 | 56 | 44 | 79 | 66 | 48 |

With the rule of Befund 122 (>= 20 inside cases) the cells of seed 2 at `n_train` 4 and 8 and of
seed 5 at `n_train` 4 are not determinable. Consequence for the pre-registered reading: at
`n_train` 4 only three seeds can be paired, so that the condition "at least 4 seeds with the sign
of the mean" cannot be met and no difference can count as resolved there; at `n_train` 8 all four
remaining seeds have to agree. The reading is left as committed (decision 14).

**Projection** (per-task times of Befund 122, `hybrid_model_training_v1` and
`hybrid_model_v1_margins_per_draw`, three seeds, scaled by 5/3; 8 workers): training 43.6 CPU-h x
5/3 = 72.6 CPU-h, about 9.1 h wall (pool generation with the new truth 21 s per pool in the dry
run, not slower than in Befund 122); margins (M0 7.0 + M1 84.3 + M2 6.6 + M3 87.1) CPU-h x 5/3 =
308 CPU-h plus M2b like M2 (11 CPU-h) = 319 CPU-h, about 39.9 h wall, of which `n_train` 4 about
11.2 h; errors about 1 h; oracle 0.2 h. Total about 50 h; time to the deadline at the start
(2026-11-13 about 18:15): about 60 h. Not covered by the projection: refits that need more
evaluations with the structural error, and cases that need more bisection steps.

**Cut decision.** The projection lies below the budget: nothing is dropped from the start. The
reserve is about 10 h, so the gate of decision 13 holds the cut ready inside the run: `n_train` 4
is dropped from the margins if its part would start after 2026-11-15 15:00. The outcome of the
gate is reported in section 10.
