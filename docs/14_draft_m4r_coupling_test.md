# Draft: re-fixed coupling test M4-R after the corrections A1 and K1 (NOT released, NOT computed)

Status: draft of 2026-10-09 for the decision of the project lead. Nothing of this has been computed. On
release, the hypothesis, the scenarios, the criterion and the script are committed BEFORE any scenario is
computed (as M4-C, Befund 77), in `docs/09` and in the script docstring; this file is then superseded.

## 1. Why a new test

Befund 77 (M4-C) confirmed H1 by K_lab 6 of 6, but Befund 78 showed that the reported solutions rest on
the warm-end check of `dT_min` (A1) and that the line optima sat on the solver bound `T_amb + 10 K` (K1);
both are corrected (Befund 79). The candidate list of M4-C (two interconnections taken from the optima of
Befunde 76 and the S2 interconnection) is invalid: it was selected from optima of the faulty model. The
new test therefore searches the joint optimum over ALL 441 interconnections.

## 2. Hypothesis (unchanged in wording)

**H1:** at warm and humid ambient air the joint optimum of the two lines (M3') saves more than 1.0 kg
CO2e/h of the site against "each line on its own, then the best interconnection" (S2), in the great
majority of the scenarios of group W.

## 3. Test set and settings

- Scenarios as M4-C, none among the 25 Befund 52 points: **W** (decision rule) T_amb 30.0 / 32.5 degC x
  phi 0.60 / 0.72 / 0.82 (six); **N** (negative control, no rule) (32.5; 0.30), (30.0; 0.35); **H**
  (descriptive, no rule) (20.0; 0.82), (25.0; 0.82). For every scenario the absolute humidity of the
  ambient air is reported.
- Lines A (1.5 kg/h) and B (2.0 kg/h), scaled site (k = 271.797, a scaling computation), external source
  Q1 and sink S1 of the partner data case, no exchanger limit, `T_kalt` 29.1 degC, `eta_H` 0.80,
  `H_u` 45 MJ/kg, quality limit 0.15, margin m = 0.005961819615175404 kg/kg for both lines (D8), nozzle
  pressure 1.0 bar (D6), lower gas-velocity bound 1.26 m/s (0.80 m/s NOT computed).
- **Inlet-temperature box 60-100 degC** (laboratory range; the box 60-120 is not computed).
- **Corrected model:** `approach_check = "along_exchanger"`, steady-state solver with the fallback
  bracket (Befund 79). **`dT_min` = 20 K main case, 10 K sensitivity** (the same procedure, separately
  reported).
- Provenance of every artifact: solver search bound, `approach_check`, `SiteSpec.fingerprint`
  (Befund 79, step 3).

## 4. Method

**S2 (new).** For each line and each DE seed 0-4 the optimum of the line on its own (line objective of
M1/M2, three variables, box 60-100 degC, u >= 1.26 m/s, Befund 52 budget), with the corrected solver.
S2 of a pair of seeds = the best of the 441 interconnections at the set points of the pair (1.0 bar, true
solves, corrected cascade at the `dT_min` of the case); **best S2 = minimum over the 25 pairs**. The S2
set points do not depend on the network, so they serve both `dT_min` cases.

**M3' (full nested search, scheme C of `joint_search`, every rule unchanged except the one below).**
1. Response tables of both lines per scenario with the corrected solver: inlet temperature 60-100 degC
   in steps of 2.5 K (17 nodes) x gas velocity 1.26-2.5 m/s (9 nodes), 1.0 bar.
2. For EVERY one of the 441 interconnections: DE + Nelder-Mead of the four set points on the tables
   (Befund 52 budget: popsize 10, maxiter 50, tolerance 1e-4).
3. Polishing with true solves of every interconnection within the window W = 5 kg CO2e/h of the best
   table value; error rule (widen to 4 times the largest table error while it exceeds W / 2); the
   duplicate rule (same table value AND same table optimum within 1e-6) unchanged; the S2
   interconnection is always polished.
4. M3'(seed) = the polished interconnection with the lowest true value.
5. **Change against Befund 73, found while drafting:** the shortcut of `nested_search` that reuses the
   table result of an interconnection for its order variant (Q1 -> SA before or after Q1 -> SB) "because
   without a limit the order cannot matter" is NOT valid: at the stored M4-C solutions the two orders
   differ in 1344 of 5880 pairs (up to 14.9 kg CO2e/h) already with the warm-end check, and in 1512 of
   5880 with the check along the exchanger (Q1 is cooled by `Q1 -> S1` first, so the second stage of Q1
   is limited by `dT_min`). For M4-R both orders are searched as separate problems (441 instead of 343
   table searches). [The consequence for Befund 73 is reported separately and not corrected here.]
6. **Five DE seeds (0-4) of the whole nested search** per scenario and `dT_min` case.
   `G_s = value(best S2) - value(M3'(seed s))`, **`G = min_s G_s`** (the weakest seed, as K_lab).

**Checks before the decision is read (no rule, reported):** (a) table accuracy: the direct DE on true
solves (four variables, seed 0) for the interconnection chosen by M3' at two W scenarios, as
`m3_check`; (b) the tables reproduce the true solves at their nodes (as in Befund 73); (c) for the best
M3' and S2 solutions: smallest approach and its position per stage, condensate per stage, solver
fallback used, distances to the box bounds, to the gas-velocity bound and to the tightened limit.

## 5. Criterion (fixed before any computation on release)

**K_R (main, `dT_min` 20 K, box 60-100 degC): G >= 1.0 kg CO2e/h in at least 5 of the 6 scenarios of W,
with all solutions feasible.** If K_R is missed: H1 is NOT confirmed in the corrected model; the gains of
Befunde 73, 76, 77 do not survive the corrections; no recomputation and no change of a threshold
afterwards. The 10 K case is evaluated by the same rule and reported as a sensitivity, without a
decision. N: the expectation "G about zero" is stated, not tested. H: descriptive.

## 6. Run-time estimate

From the measurement of Befund 79, step 4 (artifact `probe_cascade_runtime_2`, single-threaded, medians):
table objective 3.3 ms with the check along the exchanger (0.23 ms with the warm-end check, factor 12.7);
polishing objective about 0.35-0.41 s (dominated by the two steady-state solves, unchanged); one site
evaluation of 441 interconnections 0.55 s (S2 pairs).

| part | per scenario, seed and `dT_min` case (CPU) | basis |
|---|---|---|
| table search | 441 x about 2300 calls x 3.3 ms = about 56 min | DE popsize 10, 4 variables, maxiter 50, plus Nelder-Mead |
| polishing, W-like scenarios | about 300-600 s | 790-1580 true evaluations in the warm, humid scenarios 18, 19, 22-24 of Befund 73 |
| polishing, N-like scenarios | about 4000-4700 s | 10500-12300 true evaluations in the warm, dry scenarios 20, 21 of Befund 73 |
| polishing, H-like scenarios | about 550 s | scenario 14 of Befund 73 |

Per seed and `dT_min` case: about 46000 s CPU (12.8 h) for the 10 scenarios. Five seeds on 16 workers
(50 tasks per case): **about 4 h wall per `dT_min` case, about 8 h for 20 K and 10 K**; plus S2 (100 line
searches, about 1 h wall), tables (about 10 min) and checks (about 30 min): **about 10 h wall in total,
range 7-14 h** (the windows of the polishing are not known before the run; a widened window costs most).
Variant with one seed and a seed check at two scenarios (as Befund 73): about 1.5 h wall for the 20 K case
plus S2. Run on this local machine, resumable with `--resume` (CLAUDE.md, long runs).

## 7. Open points for the project lead

1. Name and numbering of the test (proposal: M4-R).
2. Five seeds of the full nested search (proposal) or one seed with a seed check at two scenarios
   (as Befund 73; cheaper, but K_R uses the weakest seed).
3. Whether the 10 K case is run in the same go or only after K_R is read.
4. Whether the consequence for Befund 73 (order variants) is documented before the test.
