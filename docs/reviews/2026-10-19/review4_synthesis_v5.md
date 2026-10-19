# Review 4: independent review of `docs/08_synthese_m2.md` (version 5)

- **Object:** `docs/08_synthese_m2.md`, version 5 ("state 2026-10-16"), as on `origin/master`
  at the start of this review: **commit `c3f13a8`** (last change of `docs/08`: `5233b6f`).
  The local `master` of the main checkout was 7 commits ahead of `origin/master` (re-check of
  Befund 99); those commits were not reviewed and do not touch `docs/08`.
- **Reviewer role:** external; not involved in writing the synthesis. Read only; no file other
  than this report was changed. Python only through `tools/run_here.sh` in the worktree
  `C:/twinlink_work/rev4` (branch `review-4`).
- **Date:** 2026-10-19.
- **Severity scale:** high = must be changed before the document goes to the consortium;
  medium = should be changed, a reader can be misled; low = editorial or robustness.
- **Reviewer computations** (marked "RC") are numbers computed for this review from committed
  artifacts. They are NOT findings of the lab book and must not be quoted in the synthesis
  before they are entered as a Befund (CLAUDE.md, documentation rule).

## Overall assessment

**Not ready to send (no).** Every number I recomputed independently from the artifacts (more
than 60, section A) reproduces, the figures regenerate byte for byte from the Befund-94
records, and no confidential name or value was found in the synthesis or its figures. Before it
goes out, the architecture comparison needs the capacity/variant caveat of SX3 (finding 1), the
milestone gap on tools, templates and prerequisites must be closed (finding 2), the moisture
seed numbers must be taken from (or at least labelled with) the right data arm (finding 3), and
the "uncorrected tendency" wording must be calibrated with effect sizes (finding 4); the medium
findings 5-12 are short text fixes.

## Table of findings (the three most important first)

| # | Severity | Location in `docs/08` | Finding | Proposal |
|---|---|---|---|---|
| 1 | high | section 3 "Transformer, xLSTM, Mamba"; changelog "Rank statements"; `docs/12` G (S9 "not taken over") | **Capacity and variant confound not stated.** The "xLSTM" of the comparison is the mLSTM-only variant with 8,649 parameters and no hidden-to-hidden weights (Befund 67 SX0); Mamba has 8,139, the LSTM 57,027, i.e. a factor of about 7 (the parameter counts appear only in the cost table of section 4). All runs use the fixed protocol without tuning (`max_epochs` 200; the Transformer's epoch cap binds at `n_train` 64, Befund 58). SX3 (Befund 67; paired re-evaluation Befund 83 d) answers exactly the question a reader will ask: capacity-matched sLSTM and mixed variants come within 0-16 % of the LSTM median on energy and bed temperature, stay 1.6-3.3 times behind on moisture; a 53k Mamba stays behind (6 of 6 cells; 3 of 6 survive Holm in the family of 96, Befund 83 addendum a); the size control is not met. The reason for leaving S9 out ("after Holm hardly any cell outside Mamba survives; scope") does not justify omitting the caveat itself: without it "xLSTM is behind LSTM" reads as a statement about the published xLSTM architecture. | Add one paragraph to section 3: variant (mLSTM only), parameter counts of all networks, fixed protocol without tuning, and the SX3 readings of Befund 83 d with their Holm status, explicitly marked as old data (A0, platform confound). Name the method "xLSTM (mLSTM-only variant, 8.6k parameters)" wherever it is ranked; keep "statements about this protocol, not about the architectures in general" (Befund 67). |
| 2 | high | whole document; title quote of M2 | **Milestone M2 only partly met.** M2 in `docs/01` (row M2, month 21): "Methodik der Simulations- und ML-Modellierung dokumentiert (Vorgehensweisen, Werkzeuge, Vorlagen; Reduzierung des Trainingsdatenbedarfs)". Procedures (*Vorgehensweisen*) are documented well (sections 2, 5). **Tools (*Werkzeuge*)** and **templates (*Vorlagen*)** are almost absent: no description of the software (package, `DataSpec.new_study`, `evaluation/paired_resolution.py`, provenance and fingerprints, `tools/run_here.sh`, record schema), no script or command per artifact (the appendix names artifacts and commits only), no study template (pre-registration, gate, clean-tree rule). **Prerequisites** (the wording of the review request: *Voraussetzungen*) are missing: compute cost of the studies (e.g. the re-run took 1246.5 min with 8 workers, Befund 94; study A 37,874 s, Befund 88; study B 6.0 h, Befund 87 addendum 4), software environment, data volumes (run length, windows, stride), model hyperparameters, simulator inputs/outputs and its parameter status beyond "four states". **Reduction of the training-data need** is answered only indirectly (learning curves, study A, study B); there is no summary statement "how many runs for which quality, and how much the simulator saves". A consortium reader without the repository also meets undefined terms (dE, arm, preset version 2, excited set, closed fraction, margin m, E-Q1) and no executive summary (the changelog is the de-facto abstract). | Add (a) a short executive summary of 6-8 sentences, (b) a section "Tools and templates" (software, how a study is run and recorded, pre-registration/gate template, provenance rules, a table artifact -> script -> command), (c) "Prerequisites" (compute times from the Befunde, environment, data volumes, hyperparameters, simulator I/O and placeholder status), (d) a paragraph "Data need" that collects the answer from Figures 1-2, Befund 88 and Befund 87 with their caveats, (e) a glossary. |
| 3 | high | section 2 "Number of seeds"; changelog "Seeds"; section 5 "Seed budget"; section 7 | **Moisture seed numbers come from the old data (A0) although the synthesis rests on arm N.** "37 and 45 seeds ... for GRU, 33 and 193 for the Transformer" are Befund 83 addendum b (A0, `n_train` 64, h = 36); the text says "old data A0" in brackets but the changelog turns it into the general statement "the M2 target of 12.5 % is not reachable with five seeds". The artifact of Befund 94 (`seeds_needed`, which Befund 94 c points to) contains the same planning figure for arm N; RC: GRU against LSTM, moisture, `n_train` 64, h = 36, 12.5 %: **5 (excited) and 16 (constant-run holdout)** seeds, Transformer 210 and 69. On the basis of the synthesis the GRU-LSTM moisture comparison is therefore not in the "dozens of seeds" regime on the excited set; the statement holds for the Transformer and for GRU on the holdout. (Five-seed estimates of the standard deviation are poor either way; Befund 83 gives the 80 % interval.) | Enter the arm-N planning figures of Befund 94 as a lab-book addendum (they are in the artifact but in no Befund text), then report A0 and N side by side in section 2 and restrict the changelog sentence to the cells where it holds. Until then label every seed number "old data A0" in the changelog and in section 5 as well. |
| 4 | high | changelog "Rank statements"; section 3 "Only Mamba, and the baselines, are clearly behind LSTM; the deficits of Transformer and xLSTM are an uncorrected tendency"; README | **"Uncorrected tendency" understates large effects.** Holm over 336 cells with a t test on four degrees of freedom has little power, and the family includes the baselines. The mean test is weak when one seed diverges (Befund 83 f). RC (arm N, `rerun_evaluation_rerun_full.json`): on `energy_target`, h = 36, the xLSTM median is 10 to 25 times the LSTM's in all 8 cells (`n_train` 8-64, both test sets) and every one of the 5 seeds is worse in every cell; at `n_train` 64 the xLSTM is worse than the static regression (median error 3.5 and 2.3 times the static one, excited / holdout). Calling this a "tendency" tells the reader the effect may be noise; the correct reading is "large and consistent, but not resolved after Holm in this family with five seeds". (RC sensitivity: Holm on the log-ratio p values that Befund 94 also stores resolves 130 instead of 70 cells, xLSTM 6, Transformer 3; this is not a Befund.) | Keep Holm as the main form, but report next to each verdict the effect size (ratio of medians) and the number of seeds worse; replace "tendency" by "not resolved after Holm (large effect, all seeds worse in n of m cells)". Decide (project lead) whether a pre-registered smaller family (architectures only, or per candidate) and the log-ratio test are added as marked sensitivities in a new Befund. |
| 5 | medium | section 3, xLSTM sentence: "On the product moisture xLSTM stays unresolved in 7 of 16 cells (uncorrected); its optimisation is unstable there (Befund 58, 63)" | **Instability attributed to the wrong channel.** Befund 58 (OA5) and Befund 67 SX3 document the divergent xLSTM optimisation on `energy_target`; on moisture Befund 67 says the spread is large "but not of that kind". | "its optimisation diverges on the energy target (Befund 58 OA5, 67 SX3); on moisture the seed spread is large". |
| 6 | medium | section 3 "A1 against A0" | **The 24 Holm-resolved N-against-A0 tests are all persistence** (RC: 16 worse, 8 better; recomputed from the per-seed values). Persistence changes with the pool, not with a model (Befund 94 a). In the same paragraph "the recurrent models and ARX worse" is an A1 statement; in N against A0 ARX is better in 16 cells uncorrected. N against A0 mixes three data changes (Befund 94 d). | Split the paragraph: A1 against A0 (3 Holm-resolved tests: 2 GRU, 1 LSTM, all worse); N against A0: 24 Holm-resolved tests, all persistence, i.e. no method effect survives Holm. |
| 7 | medium | section 6, study A | **Uncorrected verdicts not marked, and one statement stronger than the lab book.** Study A rests on 2640 paired tests without multiplicity correction (Befund 88 d, i); the synthesis declares Holm the main form but does not mark these verdicts as uncorrected. "The hybrid is never better than pre-training plus fine-tuning": Befund 88 d reports 1 of 90 cells resolved better (bed temperature, P3, N = 32) and "never better in 2 of 3 channels". "So it does not pay" goes beyond Befund 88 g ("does not say that the hybrid is useless"). | Add "uncorrected, 2640 paired tests; statements rest on consistent patterns (Befund 88 i)"; write "never better in 2 of 3 channels (1 of 90 cells resolved better)"; replace "so it does not pay" by the cost statement of section 4 and the reading of Befund 88 g. |
| 8 | medium | section 6, study B, first sentence | **Descriptive result stated as a finding before the failed hypothesis.** "A better designed excitation is worth several additional runs of the plain random-step design at small N" is a descriptive, uncorrected run-equivalent reading (Befund 87 addendum 5); the primary test HB1 is not established. "At equal cost per run" omits that the plans are not equal in how often the rate limiter and the box bounds shape the signal (B2: 18 % of the inlet-temperature and 35-36 % of the spray-rate steps; Befund 87 addendum 2). | Lead with the hypothesis verdicts; give the run-equivalent reading as "descriptively (addendum 5, uncorrected intervals)"; add the limiter reservation next to the gas-velocity one. |
| 9 | medium | section 8, model-based optimisation | **Pre-registered stability verdict omitted.** The 9.0-16.5 % saving is S1 "adapting the operating point saves energy", which is **NOT stable** by the unchanged rule of Befund 69 (Befund 80 c: changes 74-87 %); only S2 (margin cost) and S3 (remaining benefit) are stable. | Add the verdicts: "S1 not stable, S2 and S3 stable (Befund 80 c)". |
| 10 | medium | section 6, plant-data paragraph (Befund 55) | **Test set, data arm and caveat missing; lab-book error still open.** The factor 2 to 3 is on the constant-run holdout, old data configuration, NRMSE, GRU and LSTM; the test set is not named (CLAUDE.md: always name the test set). Befund 55 itself says that at LSTM / `n_train` 64 the gap is not separable from the seed spread; "no catching up" is stated without that caveat and without a paired test. RC: the range is 2.22-2.52 for GRU and 2.40-3.06 for LSTM, so "2 to 3" is correct, but the lab-book text "2,2-3,1 (GRU)" (Befund 55, line 4622) is wrong and register item R2-17 is closed in `docs/12` G only through the rounding. | Name test set, metric, models and data arm; add the LSTM-64 caveat; correct the GRU range in a dated addendum to Befund 55. |
| 11 | medium | section 5 "Seed budget" | **Overstated.** "Five seeds resolve ... 10 to 12.5 % on energy and bed temperature": by Befund 83 addendum b this holds for GRU against LSTM (3-5 seeds) but not for the Transformer (6-11 seeds at 12.5 % and 10 %). | "for GRU against LSTM; the Transformer needs 6-11 seeds (old data A0)". |
| 12 | medium | lab book, outside `docs/08` (cited by section 6 as Befund 56) | **Confidentiality, lab book only.** The heading of Befund 56 (`docs/04_stand_ap1.md`, line 4692) contains a plant identifier and an apparatus number; line 4764 contains two product abbreviations. None of them is in `local/denylist.txt`, `local/mapping.md` or `local/confidential_values.md`, so the denylist test does not catch them. A consortium reader who follows the citation of Befund 56 reaches them. (Not quoted here on purpose.) | Project lead to decide whether these are confidential; if so, add them to the denylist and replace them in the lab book (pseudonymisation like the history cut of 2026-09-25), not only in new text. |
| 13 | low | section 3 GRU sentence; `docs/12` G row F1.3 | The register says F1.3 ("h = 1 cells named in the GRU counts") is taken over in section 3; it is not. Of the 9 bed-temperature cells with GRU better, 7 are at h = 1 (Befund 98 a). | Add "(mostly at h = 1: 7 of 9 on bed temperature)" or correct the register. |
| 14 | low | section 3 "Only Mamba, and the baselines, are clearly behind" | "Clearly" for 12 of 48 Holm cells (ARX) and 19 of 48 (persistence); persistence is better than LSTM in 1 cell uncorrected (energy; Befund 98 b). | "resolved behind after Holm in 12-21 of 48 cells (baselines) and 14 of 48 (Mamba)". |
| 15 | low | section 3 "Static explainability of dE" | The factors 5.2 to 7.8 are old data A0 (Befund 83 e; Transformer on the other platform) inside a section titled "(arm N)"; not said. RC: arm N gives 5.3-7.3 for GRU, LSTM, Transformer (all seeds better), i.e. the statement holds, but that figure is in no Befund. "dE" is not defined anywhere in the document. | Write "old data A0"; define dE / energy target with its unit. |
| 16 | low | section 8 storage bullet; README | Befund 99 (storage time-slice test, merged 2026-10-16, gate reading L2) is in the README but not in the outlook, which reads only "L1 (L2 post hoc)" for storage. A reader comparing README and synthesis sees two gate readings for "storage of plant A". (An addendum to Befund 99 exists on the local `master`, not on `origin/master`.) | Add one sentence on Befund 99 (L2, scenario band, no RL) or state that the outlook stops at Befund 92. |
| 17 | low | section 7 "One timing run ... the networks were fitted for two epochs only" | Ambiguous: reads as if the networks of the comparison had two epochs. | "the networks used for the timing were fitted for two epochs (cost of `predict` only)". |
| 18 | low | section 8 RL bullet | "Needs a margin of five times the base margin": k was tested at 1, 2, 3, 5; k = 4 untested (Befund 96 d). | "k = 5 is the smallest tested factor that meets the gate (k = 4 not tested)". |
| 19 | low | section 2 run-level splits | Metric and targets of Befund 39 not named: `nrmse_run_median`, h = 1, targets energy, moisture and `d50_um` (not the bed temperature of the rest of the document). | Name metric and targets. |
| 20 | low | header and changelog | "Version 5", changes "from the previous version (version 3)": version 4 is not explained. Section 5 is titled "Holm and Benjamini-Hochberg" but gives no BH figure (251 of 336 in N, Befund 94 b). | One sentence on version 4; give the BH count or drop it from the title. |
| 21 | low | figures | Byte-identical reproduction (finding B4 below). Editorial: both learning-curve files are named `fig1_*` while the text calls them Figures 1 and 2; the inference-cost figure has no number and no caption; the energy panel has no unit; the bed-temperature RMSE is labelled "degC" (a difference, K); no provenance (commit) is stamped into the PNG metadata. | Number and caption all three figures; units; optional provenance in metadata. |
| 22 | low | `scripts/check_m2_synthesis_consistency.py` | Not a tautology (see B2), but: three exemptions are verified by trivial greps; one claim (0.80 m/s) is checked against a free-text string inside an artifact; "Holm (5 %)" is checked against the alpha of the Befund-83 addendum, not of the Befund-94 analysis; numbers written as words ("ten seeds", "two epochs", "four of the five plants" except where a truth check exists) escape the coverage step; two tolerances are loose; the committed artifact `check_m2_synthesis_consistency_v5` predates the last text commit `5233b6f`. | See B3; re-run with `--write` on the final text. |

## A. Numbers recomputed against the artifacts (review point 1)

Computed with my own scripts (scratchpad, not committed) from the committed artifacts, without
using `scripts/check_m2_synthesis_consistency.py`. Where possible from raw per-seed values
(paired t test and Holm re-implemented from scratch, scipy only for the t distribution), not
from the derived summary artifacts.

| # | Statement (section) | Written | Recomputed | Source of the recomputation | Result |
|---|---|---|---|---|---|
| 1-2 | Run-level against point-wise split (2, 5) | 29 to 68 % | 28.7 %, 51.7 %, 68.1 % (energy, d50, moisture) | `extra_trees_split_comparison.jsonl`, medians of `nrmse_run_median` h1 | ok |
| 3-5 | Rank-cell families (2) | 336, 336, 168 | 336, 336, 168 | `rerun_evaluation_rerun_full.json`, `rank_cells` | ok |
| 6 | Paired tests on old data (2) | 632 | 632 (field `n_tests_total`; not recounted) | `paired_reevaluation_58_63_67.json` | ok |
| 7-8 | Seeds, energy/bed temperature (2) | 3 to 11 | 3-11 (GRU 3-5, Transformer 4-11), A0, n 64, h 36 | `rerun_evaluation...` `seeds_needed` A0 (same values as Befund 83 addendum b) | ok, see finding 3 |
| 9-12 | Seeds, moisture (2) | 37, 45; 33, 193 | 37, 45; 33, 193 (A0) | as above | ok for A0; arm N: 16, 5; 69, 210 (finding 3) |
| 13 | Paired sd on moisture (2) | 25 to 60 % | 24.8-61.5 % | as above | ok (rounded) |
| 14-16 | GRU vs LSTM, moisture (3) | 15/16 unresolved, 1 worse | 15, 1 (worse) | per-seed t tests, arm N | ok |
| 17-18 | GRU better, energy / bed temperature (3) | 5/16, 9/16, never worse | 5, 9, 0 worse | per-seed t tests | ok |
| 19 | GRU after Holm (3) | 0 | 0 (main family and GRU-only family; smallest GRU p 0.0040) | own Holm | ok |
| 20-22 | Transformer / xLSTM / Mamba worse, uncorrected (3) | 39 / 38 / 44 of 48, never better | 39 / 38 / 44, 0 better | per-seed t tests | ok |
| 23-25 | Same after Holm (3) | 2 / 2 / 14 | 2 / 2 / 14 | own Holm, family of 336 | ok |
| 26 | xLSTM moisture unresolved (3) | 7 of 16 | 7 | per-seed t tests | ok |
| 27-29 | Baselines after Holm (3) | 21 / 19 / 12 | 21 / 19 / 12 | own Holm | ok |
| 30-31 | Holm survivors (3) | 70 of 336 (N), 82 (A0) | 70, 82 (A1: 20) | own Holm | ok |
| 32 | Uncorrected resolved, arm N (5) | 260 of 336 | 260 | per-seed t tests | ok |
| 33-35 | Rank verdicts differing from A0 (3) | 85, 45, 40 | 85, 45, 40 | recomputed by joining the verdicts of the three arms | ok |
| 36-39 | A1 / N against A0 after Holm (3) | 3 of 192; 24 of 384 | 3, 192; 24, 384 | own t tests and Holm on `a1_vs_a0`, `n_vs_a0` | ok, see finding 6 |
| 40-41 | Gain over static regression (3) | 5.2 to 7.8, all seeds better | 5.20-7.82 (A0); all seeds better | `paired_reevaluation_58_63_67.json`, `part_58_cells` | ok, see finding 15 |
| 42-43 | Parameters GRU / LSTM, "about 25 % fewer" (3) | 42,819; 57,027 | 42,819; 57,027; 24.9 % | `joint_timing_idle.json`; also Befund 28 text | ok |
| 44-67 | Inference table, 8 methods x (parameters, latency, batch) (4) | table | identical after rounding | `joint_timing_idle.json` | ok |
| 68 | Test windows (4) | 4352 | 4352 | same | ok |
| 69-73 | Fit times per epoch (4) | 4.3 / 4.7 / 14.8 / 27.8 / 67.6 s | same | same | ok |
| 74-77 | Hybrid simulator share (4) | 253,136; 256,123; 98.8 %; 2,987 | 253,136.0; 256,123.4; 98.83 %; 2,987.5 | `a1_inference_time_v2_idle.json` | ok |
| 78-81 | Study A ratios (6) | 12.69, 10.27, 1.32, 1.27 | 12.694, 10.269, 1.323, 1.273 | `a1_main_analysis.json` | ok |
| 82 | Study A crossover (6) | four of five plants | not reached on P0, P1, P2a, P2b; P3 n* = 8 | same | ok |
| 83-85 | Study B verdicts (6) | HB1 n.e.; HB2 n.e. (T3) / refuted (T1); HB3 n.e. | same | `b1_main_analysis.json` | ok, see finding 8 |
| 86 | Constant-trained vs excited (6) | factor 2 to 3 | GRU 2.22-2.52, LSTM 2.40-3.06 | both JSONL files, `nrmse_run_median` moisture h36, constant holdout, medians | ok, see finding 10 |
| 87-96 | Model-based optimisation (8) | 9.0 / 16.5 %; 1.30 / 1.46 %; 94 / 85 %; 22 / 17; 13 / 25 | 9.03 / 16.52; 1.295 / 1.458; 93.9 / 84.9; 22 / 17; 13 / 25 | `m1_k1_comparison.json`, `rerun_k1` | ok, see finding 9 |
| 97-98 | Coupling test (8) | 0 of 6 (20 K), 1 of 6 (10 K) | 0, 1 (G min over seeds and best seed) | `m4r_test_dt*_umin126.json`, W scenarios | ok |
| 99-101 | RL margin (8) | k = 5; 6.9 %; 6.5 points | 5; 6.91; 6.53 | `bres_margin.json` | ok |
| 102 | Partner recovery (8) | 60 to 88 % | 59.55-88.1 % | `heat_recovery_comparison_r4_consistent.json`, variant (i), a / c | ok |
| 103 | Storage readings (8) | L1, L2 post hoc | L1 (all products), L2 with R1b | `gate_storage_shift*.json` | ok |
| - | Plant-data variation (6) | 0.12-2.01 % | not checkable (raw data not in the repository); agrees with Befund 56a text, lines 4764-4765 | lab book | text only |

**Result:** no deviation. The paired t tests recomputed from the per-seed values agree with the
stored p values to the last digit (maximum difference 0) in all 840 rank cells, and the verdicts
agree in all of them.

## B. The consistency script (review point 1, second part)

**B1. Result on `c3f13a8`.** Run without `--write` (nothing written): 112 claims, 112
confirmed, 12 exemptions verified, 0 uncovered numeric tokens, exit code 0.

**B2. Does it check against artifacts or text against text?** Against artifacts, with three
qualifications. (a) Most claims of section 3 read the derived summary `holm_rank_cells_rerun_full.json`
(produced by the project's own script, Befund 98) rather than recomputing from the per-seed
values; this is not circular, and section A shows the summary is right, but the check is not
independent of the analysis code. (b) Two checks are text against text: "0.80 m/s" is checked
by searching the free-text field `placeholders` of `bres_margin.json`; the M2 note "10 to 15 %"
and the exemptions below are greps in documents. (c) The table rows of section 4 use
`report.truth(row, True, ...)`; this looks tautological but is not, because the row is
assembled from the artifact and must occur in the text verbatim.

**B3. Weaknesses.** (1) "Holm (5 %)" is compared with `alpha` of `paired_reevaluation_83_addendum.json`,
not with the alpha of `scripts/analyze_rerun.py` (Befund 94). (2) Coverage handles digits only:
numbers in words ("ten seeds", "two epochs", "one LSTM", "three targets", "five times") are not
checked; for study B the "ten seeds" is correct (artifact `seeds`), but nothing would catch a
wrong one. (3) Generic reference patterns (`[A-Za-z]{1,3}-?\d+`, `horizons? \d+`, `$N$ = \d+`)
remove tokens before the coverage test; acceptable, but a wrong horizon or N would pass.
(4) Loose tolerances: "25 to 60 %" accepts 23-62 %, "factor 2 to 3" accepts 1.95-3.2.
(5) Qualitative claims ("never better", "unstable there", "does not pay", "clearly behind")
are outside its scope by design; findings 4, 5, 7, 14 are of that kind. (6) The committed
artifact `check_m2_synthesis_consistency_v5.json` (commit `b9bf7b0`, 09:09) predates the text
change `5233b6f` (09:39, citations only); a re-run with `--write` on the final text is needed.

**B4. Are the 12 declared exemptions justified?**

| Exemption | Justified? | Verification in the script | Comment |
|---|---|---|---|
| `n_train ∈ {8, 16, 32, 64}` (two spellings), `n_train = 64`, "curve still falls at 64" | yes | n_train set of the N records | sound |
| "arm N, the preset version 2" | yes | `data/new_study_version` of the N records | sound |
| "time constant 600 s" | yes (design constant) | **weak:** any "600" in `docs/02` passes | check `mass_control_time_constant_s: 600.0` in `conf/process/greybox_gab_placeholder.yaml` |
| "planned for month 21" | yes (plan fact) | **weak:** any "21" in `docs/01` passes | anchor on the M2/population-balance row |
| "horizon 36 (900 s)" | yes | 36 x sample time from the records | sound |
| "The M2 definition of 10 to 15 %" | yes | regex in `docs/01` | adequate |
| "two 24-hour exports", "0.12 to 2.01 %" | yes (raw data never in the repository) | none (declared not checkable) | at least compare with the Befund 56a text, which agrees |
| "nozzle-pressure kink at 1 bar" | yes (register question) | **weak:** checks that "E-Q1" exists, not the value | check the value in the register row or the config |

## C. Statements against the lab book (review point 2)

- **Holm as main form:** applied consistently in section 3 (rank cells) and in study B (HB1
  has its Holm step). **Not** applied or marked in study A (finding 7), in the Befund-55
  paragraph (finding 10) and in the outlook (descriptive studies, acceptable there).
- **Uncorrected tendencies marked:** the GRU counts and the Transformer/xLSTM deficits are
  marked; the marking itself understates large effects (finding 4). The A1/N paragraph mixes a
  Holm count of persistence with a model statement (finding 6).
- **Summary stronger than the results?** There is no abstract; the changelog acts as one. Its
  "Rank statements" bullet matches section 3. Its "Seeds" bullet generalises an A0 figure to the
  synthesis (finding 3). Within the results, study A ("does not pay"), study B ("worth several
  runs") and section 5 ("five seeds resolve 10-12.5 %") are stronger than their Befunde
  (findings 7, 8, 11); section 8 omits the "not stable" verdict (finding 9).
- Statements checked and consistent with the lab book: GRU recommendation on practical grounds
  (Befunde 28, 97, 98), batch/latency orders (97), A1 reading (94 a), ends of the order (94 c),
  static explainability (83 e, 59), negative transfer (26), test-set rule (51 K5), coupling
  test (81), RL margin (96), partner model (90 addendum, 95), storage (91, 92).

## D. Omissions (review point 3)

| Befund | Relevance | Omitted? | Justified? |
|---|---|---|---|
| **SX3 / Befund 67, re-evaluated in 83 d** (capacity-matched mLSTM, sLSTM, mixed, Mamba against LSTM) | high: answers whether the xLSTM/Mamba deficit is architecture or size | yes (`docs/12` G: S9 not taken over) | **No** for the caveat (finding 1). The numbers may stay out, but the variant, the parameter counts and the qualitative SX3 reading must be in. |
| **Befund 70** (M2, one line with fixed source and sink: set-point effect larger than routing effect) | medium, outlook only | yes | **Yes.** Computed with the warm-end cascade check that review A1 found faulty; not re-computed (`docs/12` B5). The sentence "Set points are optimised line by line before any interconnection" in section 8 should be marked as a procedure, not a finding, so that it is not read as Befund 70. |
| **Befund 62 HA1** (agent margin against 3 x RMSE of the soft sensor) | medium: the only link between soft-sensor accuracy and control | yes (margin factors removed, `docs/12` M7) | **Partly.** The factors rest on a yardstick later called wrong (third addendum to Befund 65), so leaving them out is right. But the margin m used in section 8 is never defined: it is the soft-sensor margin of Befund 64/65 (FA2), based on the 3-sigma convention introduced in HA1, a convention and not a plant or norm requirement. Add one sentence. |
| **Befund 55** (2.2-3.1 against 2.2-2.5) | medium | no, quoted as "factor 2 to 3" | The rounding is correct (GRU 2.22-2.52, LSTM 2.40-3.06, RC), so there is no contradiction in `docs/08`; the error sits in the lab book (line 4622, "2,2-3,1 (GRU)") and needs a dated addendum; test set and caveat are missing in `docs/08` (finding 10). |
| Befund 99 (storage time slice, L2) | low, outlook | yes | Not justified by date (merged the same day); see finding 16. |
| Befund 83 c (Transformer moisture deficit resolvable again, does not survive Holm) | low | yes | Acceptable under Holm as main form. |
| Befund 58 / 83 e (Transformer epoch cap binding; catch-up on bed temperature) | medium as part of finding 1 | yes | Belongs in the protocol caveat of finding 1. |
| Befund 88 g (hybrid better than from scratch from N = 16/32) | low | yes | Should accompany "never better than fine-tuning" (finding 7). |

## E. Figures (review point 4)

- **Reproducible:** `tools/run_here.sh scripts/plot_synthesis_v5_figures.py --output-dir <scratch>`
  regenerates all three PNGs **byte-identical** to `docs/figures/` (SHA-256 equal, pixel
  difference 0).
- **Data:** the script reads `rerun_full_N.jsonl` (records with `eval/primary`) and
  `joint_timing_idle.json`. RC: each of the 192 plotted points (8 methods x 4 `n_train` x 2 test
  sets x 3 channels) has exactly 5 seeds, and the per-seed values equal those of the rank cells
  of Befund 94 in all 192 points (0 mismatches). Lines are the median, bands min-max over the
  five seeds, as captioned.
- **Labels:** test set in the figure title and in the caption (excited / constant-run holdout);
  horizon 36 on the y axis; x axis = training runs. Editorial points in finding 21.

## F. Confidentiality (review point 5)

- `docs/08`, both scripts and the three figures (text inspected visually): **no hit** against
  `local/denylist.txt` (7 entries) and no numeric match with `local/confidential_values.md`
  (checked as counts only). Pseudonyms are used throughout ("application partner", "plant A",
  "partner model"); no person, product name, apparatus number, measuring-point tag or recipe
  value appears. Plant data appear only as aggregated statistics (coefficient of variation),
  which I consider uncritical.
- Lab book outside the review object: finding 12.

## G. Milestone M2 (review point 6)

See finding 2. In short: *procedures* yes; *tools* and *templates* no; *prerequisites* no;
*reduction of the training-data need* only implicitly. A consortium reader who does not know the
repository needs, in addition: an executive summary; a self-contained description of the
simulator (inputs, outputs, states, which parameters are placeholders and which are sourced) and
of the models (architecture, size, training protocol); the study workflow as a reusable
template (pre-registration, gate, clean-tree rule, provenance, fingerprints, paired resolution
with Holm); compute and data requirements; a glossary; and a definition of terms that today are
only explained in the lab book (much of it in German).

## H. Internal consistency (review point 7)

- `docs/08` against `docs/12` G: the closure table is accurate except F1.3 (finding 13) and
  S9 (finding 1); R2-17 is closed only by rounding (finding 10).
- `docs/08` against README: consistent on the architecture comparison, study A, Befund 80,
  coupling and partner model. Differences: README contains Befund 99 (finding 16); README's
  architecture paragraph still opens with "per cell and not corrected for multiplicity" before
  the Holm precision, whereas `docs/08` puts Holm first (harmless, but the README should lead
  with the same main form); README says the margin "costs about 1.5 %" (1.26 m/s only) where
  `docs/08` gives 1.30-1.46 % (consistent).
- Within `docs/08`: findings 3/11 (seed statements of sections 2, 5 and changelog differ in
  scope), 15 (A0 figure in an arm-N section), 17, 20.

## Method of this review

Worktree `C:/twinlink_work/rev4`, branch `review-4` from `origin/master` = `c3f13a8`. Read
`docs/08` completely, then the Befunde it cites (16, 26, 28, 39, 51, 55, 56, 58, 62 HA1, 67 SX0/SX3,
70, 80-99) and `docs/12` sections C-G, README. Recomputations with throw-away scripts outside the
repository, run with `tools/run_here.sh`; the consistency script was read only after section A was
complete, and run once without `--write`. No file except this report was changed; not merged, not
pushed.
