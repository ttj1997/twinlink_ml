# Review 4b: follow-up check of `docs/08_synthese_m2.md` version 5b

- **Object:** `docs/08_synthese_m2.md` version 5b at `origin/master` **`484d1a8`** (text commit
  `161c8d7`), with the lab-book addenda to Befunde 55, 94 and 98 (`f3d8e33`), their artifacts
  (`198afc1`, `9fd51df`, `6ef5501`), the figure script and figures (`2323915`, `821cd70`), the
  extended consistency check and its artifact (`161c8d7`, `b0faf2b`) and `docs/12` section H.
- **Scope:** only the four questions of the follow-up request; reference is the review report
  `review4_synthesis_v5.md` (findings numbered as there). Read only; worktree `C:/twinlink_work/rev4b`,
  branch `review-4b`, Python through `tools/run_here.sh`. Not merged, not pushed.
- "RC" = reviewer computation from committed artifacts, not a Befund; not to be quoted.

## Result

**Ready to send: yes, after one sentence of the summary is changed (N1 below).** All four high
findings are implemented and correct; all eight medium findings are implemented (finding 12 by a
documented decision of the project lead); the ten sampled new numbers reproduce exactly from the raw
records; the consistency check (382 claims, 15 exemptions, 0 uncovered tokens) and the figures
(pixel-identical) reproduce. The remaining points N2-N6 are low and can follow; N2 concerns the
README, not the document to be sent.

## 1. The four high findings

| # | Finding (review 4) | Implemented? | Correct? |
|---|---|---|---|
| 1 | Capacity / variant confound, SX3 | yes: section 3 "Capacity and protocol", section 7, appendix B, summary | yes. Parameter counts, mLSTM-only variant, fixed untuned protocol and the SX3 readings agree with Befunde 67 and 83 d / addendum a; the SX3 parameter counts (sLSTM / mixed 47,955-50,739, mLSTM 50,301, Mamba 53,211, small LSTM 8,517) agree with `sx3_capacity_matched_a/b.jsonl`. See N4 for one incomplete clause. |
| 2 | M2 completeness | yes: summary, appendices A-E, appendix F with scripts | yes, with one declared omission (wall-clock times of whole studies; `docs/12` H item 2). I accept the reason: those times are in no committed artifact, and the times in the records are marked as taken under load. Appendix B agrees with the record parameters (learning rate 0.001, weight decay 0.0001, batch 256, 200 epochs, patience 20, gradient clip 1.0, dropout 0, teacher forcing 0, equal horizon weights; AdamW in `models/torch_sequence.py`); appendix C agrees with `conf/process/greybox_gab_placeholder.yaml` and `simulation/excitation.py` (bounds, rate limits, disturbance ranges, RK4 with 5 substeps, signal weights 0.40/0.30/0.30, amplitudes 15/35/70/100 %), except N5. |
| 3 | Moisture seed numbers from A0 | yes: addendum to Befund 94 a; sections 2, 5, 7, summary use arm N | yes (sample below). The reading "the moisture problem is smaller for the GRU-LSTM pair, not gone" is supported by the addendum (5 / 16 seeds; 3-753 over all budgets). |
| 4 | Effect size next to Holm | yes: addendum to Befund 98; wording rule in section 3, section 5, glossary | yes (sample below). The declined part (smaller family, log-ratio test) is a documented decision of the project lead; the wording no longer depends on it. |

**Sample of ten new numbers, recomputed independently** from `rerun_full_N.jsonl` (records with
`eval/primary`, `rmse_run_median_physical`) and the stored rank cells, with my own paired t test and
Holm, not from the new analysis artifacts:

| # | Number in 5b / addendum | Recomputed | |
|---|---|---|---|
| 1 | Arm N seeds, `n_train` 64, h 36, 12.5 %: GRU 3-6 (energy, bed), 5 / 16 (moisture), Transformer 4-7, 210 / 69 | 3, 6, 3, 3; 5, 16; 7, 6, 4, 4; 210, 69 | ok |
| 2 | Same at 10 %: GRU 3-8, Transformer 4-10 (section 5) | 4, 8, 3, 4; 10, 8, 4, 5 | ok |
| 3 | GRU moisture, excited, all budgets and horizons: 3 to 753 seeds | 3-753 (8 cells) | ok |
| 4 | Appendix D ratio table (12 cells x 3 values) | identical to two decimals in all 36 values | ok |
| 5 | Absolute moisture at 64 runs: 7.08e-4 / 6.17e-4, 6.71e-4 / 5.43e-4; persistence 1.18e-2 / 1.62e-3 kg/kg | identical | ok |
| 6 | 64 against 32 runs: 12 of 12 cells lower (GRU, LSTM), 2 after Holm in the family of 48 (LSTM energy and bed temperature, excited); five networks 22 / 6; xLSTM none | identical (all seeds lower in 11 of 12) | ok |
| 7 | Baselines beaten in every seed: static from 8; persistence from 8 (excited), on the holdout from 32 (moisture), 16 (energy) | identical | ok |
| 8 | Effect sizes: GRU median ratio 0.97 (energy 0.94, bed 0.98); Transformer 1.30 (energy 1.26, moisture 1.67, bed 1.16), 39 cells all seeds worse; xLSTM 3.26 (energy 8.42), 48 of 48 cells all seeds worse; Mamba 3.19 | identical | ok |
| 9 | xLSTM on energy, h 36: 10.4 to 24.5 times the LSTM, all seeds worse in all 8 cells | identical | ok |
| 10 | xLSTM against static, `n_train` 64, h 36, energy: 3.4 / 2.3 times, 4 / 3 of 5 seeds worse, p 0.11 / 0.18 | 3.43 / 2.32; 4 / 3; 0.11 / 0.18 | ok |

Also confirmed: Befund 55 addendum ranges (GRU 2.22-2.52, LSTM 2.40-3.06; identical to review 4);
the margin m as "median half-width of the calibrated ensemble band, Befund 65 FA2" (lab book, Befund 68
settings); every artifact commit and script of appendix F that is new in 5b exists and matches
`git log --diff-filter=A`.

## 2. The medium findings

| # | Finding | Implemented? |
|---|---|---|
| 5 | xLSTM instability on the wrong channel | yes |
| 6 | N-against-A0 Holm tests all persistence | yes (the 16/8 split is rightly not quoted) |
| 7 | Study A uncorrected; hybrid statement | yes |
| 8 | Study B order and limiter reservation | yes |
| 9 | Stability verdict of Befund 80 | yes |
| 10 | Befund 55 test set, arm, caveat; lab-book range | yes (addendum to Befund 55) |
| 11 | Seed budget overstated | yes |
| 12 | Identifiers in the lab book | no change in the tracked text, by decision of the project lead (kept until the planned history cut, recorded in `local/`). Acceptable for sending `docs/08`, which contains none of them; the repository itself should not go to the consortium before the cut. |

The low findings 13-22 are all implemented as listed in `docs/12` H; I checked 13, 14, 15, 16, 18,
19, 20 and 21 in the text and the figures.

## 3. The objection to "xLSTM worse than the static regression at `n_train` 64"

**I agree.** Review 4 compared medians only ("median error 3.5 and 2.3 times the static one") and
called that "worse than the static regression". With 4 and 3 of 5 seeds on that side and p 0.11 and
0.18 (RC confirms), the difference is neither consistent over the seeds nor resolved, i.e. by the
synthesis's own rule it is not even "large and consistent". The wording of 5b ("3.4 and 2.3 times ...
but with 4 and 3 of 5 seeds worse this is not resolved") and of the addendum to Befund 98 b is the
correct one. (My 3.5 was also a rounding slip; the ratio is 3.43.)

## 4. New problems introduced by the corrections

| # | Severity | Location | Problem | Proposal |
|---|---|---|---|---|
| N1 | medium | Summary, paragraph 2, first sentence | **The summary is stronger than section 3 for the Transformer.** "GRU and LSTM are the best of the eight methods tested" has no qualifier, and the Transformer is not mentioned in the summary at all. Section 3: Transformer worse after Holm in 2 of 48 cells, large and consistent only on the moisture (11 of 16), on energy and bed temperature an uncorrected tendency (ratios 1.26 / 1.16); "the middle of the order is not supported by five seeds". A reader of the summary alone takes "best" as an established rank over the Transformer. | e.g. "GRU and LSTM have the lowest errors of the eight methods; against the Transformer this is large and consistent on the moisture and an uncorrected tendency on energy and bed temperature." |
| N2 | medium (README, not `docs/08`) | README, architecture paragraph | **Contradicts section 3:** "capacity-matched variants (Befund 67, old data) do not remove the xLSTM and Mamba deficit". Section 3 (and Befund 67 SX3): the sLSTM and mixed variants come close to the LSTM on energy and bed temperature; only on the moisture they stay 1.6-3.3 times behind; for Mamba the statement holds. The README also still says "clearly behind only Mamba and the baselines", which 5b removed (review 4 finding 14). | "capacity-matched variants with memory mixing close most of the xLSTM gap on energy and bed temperature but not on the moisture; a larger Mamba stays behind"; drop "clearly". |
| N3 | low | Summary paragraph 3, section 6, appendix D | "The curves still fall at 64 runs" is an uncorrected statement (12 of 12 cells, 2 after Holm, none on the moisture); appendix D says so, the summary does not. The conclusion drawn ("need beyond 64 not determined") is cautious and unaffected. | Add "(in the medians, all seeds lower in 11 of 12 cells; 2 after Holm)" or "do not saturate". |
| N4 | low | Section 3 "Capacity and protocol" | "The epoch cap of the Transformer binds at `n_train` = 64 (Befund 58)" names one network. RC (arm N, `n_train` 64, heads with `training/epochs_run` = 200): Transformer 20 of 30, Mamba 18 of 30, LSTM 10 of 30, GRU 8 of 30, xLSTM 4 of 30. The cap therefore shapes the comparison more widely than the sentence says, in particular for Mamba. Not a Befund. | Either enter the cap counts of arm N as a lab-book note and quote them, or write "the epoch cap binds for part of the heads (Befund 58 for the Transformer on the old data)". |
| N5 | low | Appendix C "Time" and "Status" | The sample time of 25 s is set by `DataSpec` (`sample_time_s = 25.0`, `evaluation/experiments.py`); the YAML named as the source of the process configuration still says `sample_time_s: 5.0`. A reader who reproduces from the YAML alone gets 5 s. | One clause: "set by the data specification (`DataSpec`), overriding the 5 s of the YAML". |
| N6 | low | Summary paragraph 3 | "Models trained on constant operation alone stay far behind" rests on Befund 55 (legacy data configuration of A0, unpaired, LSTM-64 gap not separable) and the descriptive B0 result of study B; "an excitation experiment at the plant is needed" is a recommendation drawn from simulator results. Section 6 carries the caveats, the summary not. | Add "on the simulator" / "(old data, factor 2 to 3 on the moisture)" and mark the last clause as a recommendation. |

No contradiction found between the appendices A-F and the main text: the figures of appendix B equal
the section-4 table and the records, appendix D equals section 6 and the summary, appendix F names the
right artifacts, commits and scripts, and the glossary matches the wording rule of section 3.

**Reproduction.** `scripts/check_m2_synthesis_consistency.py` at `484d1a8` without `--write`:
382 claims confirmed, 0 deviating, 15 exemptions verified, 0 uncovered numeric tokens.
`scripts/plot_synthesis_v5b_figures.py` into a scratch directory: all three PNGs pixel-identical to
`docs/figures/fig*_v5b_*.png`; the bytes differ only in the embedded `Source` metadata (commit of the
run, here `484d1a8`, committed `2323915`), which is intended. Units (kW, kg/kg, K), numbering and test
sets in the titles are correct.
