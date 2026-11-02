# Work plan ML start (TUHH-B-AP 1)

> Version 5. Order chosen deliberately: first the continuous thread from data to
> metric, then the interesting architectures. Not the other way round.
>
> Status: 13.08.2026 · steps 1–5 done, step 6 half done. Implementation status and
> measured values: `04_stand_ap1.md`. Process assumptions: `02_prozess_setup.md` (version 3, the
> earlier addendum is incorporated).

## Guiding idea

The value of AP 1 does not lie in getting xLSTM to run, but in a
**robust comparison framework**. Every figure that decides whether a comparison is
meaningful is computed and logged, not assumed. So far the
steps have refuted what looked plausible beforehand:

| Step | refuted assumption |
|---|---|
| 1 | "there is one dominant time constant" |
| 1 | "`dE` is a dynamic target" (R² static 0.97) |
| 2 | "a PRBS excites all channels broadband" (rate limiting dominates) |
| 3 | "pooled NRMSE is a neutral metric" (factor 31 for the moisture) |
| 4 | "more training runs help" (linear ARX saturates at 16) |
| 4 | "fine Δt is better" (rollout error at constant horizon almost invariant) |
| 5 | "three seeds are enough for an architecture comparison" (resolution 10–15 %) |
| 5 | "`d50` is a target" (84 % noise floor) |
| 5 | "the moisture fails because of the noise" (even noise-free, persistence beats the ARX) |
| 6 | "the moisture needs nonlinearity" (the NARX is worse than the linear ARX) |
| 6 | "the run-level NRMSE is *the* primary metric" (holds only with inline measurement) |
| 6 | "more history helps" (jump between 1 and 2 steps, flat afterwards) |
| 6 | "the ARX saturates, so the task saturates" (the NARX curve keeps going at n=64) |

The last four shift the focus again: **the open problem is error
accumulation in the rollout, not the function class and not the window length.**

---

## Steps 1 to 5 — **done**

**1 Grey-box process model.** Four states, ideally mixed bed, gas phase
quasi-stationary.

**2 Excitation design.** Step sequences, PRBS as a true m-sequence, ramps, constant runs
(evenly distributed). Sobol instead of Latin hypercube, support points as powers of two,
rate limiting as a mandatory component, scattered initial states. Rule: build the maximum design
exactly once, everything else via `subset()`.

**3 Vertical slice.** Data generation → windowing → splits → model → evaluation →
result log. Three-block windows, `RunSplitter` ABC, rollout-centred `SequenceModel`,
`ExperimentLogger` with fingerprints. Δt = 25 s as standard, chosen by computational cost.

**4 Baselines.** Persistence, static quadratic regression, linear ARX(4,2) with
`j = 0` term. Result: `energy_target` has to be beaten at 0.172, `product_moisture` at
0.794 — and the linear ARX does not manage the latter.

**5 Evaluation protocol.** Variance decomposition by source (`SeedPlan`), resolving power
(13.2 % / 9.7 % / 14.5 % with three seeds), model-free noise floor (`NoiseFloor`), both
validation operating modes, learning-curve notebook. Side benefit: the sensor model is decoupled from
the integrator (`remeasure`), noise variants cost 0.005 s instead of 0.78 s.

---

## Step 6 – architecture slots (part 1 done, part 2 open)

### Part 1 — the framework, measured

- **Observability as an experimental factor.** `ObservationMode.INLINE` and `SOFT_SENSOR`;
  masking in the windowing, not in the model. Parameter-free reference is persistence
  or mean prediction, primary metric run-level median or pooled, respectively.
- **Architecture registration** (`models/registry.py`) with availability check; slots not
  implemented are declared and report what is missing.
- **Verdicts** (`architecture_verdicts`): a difference appears only together with the
  resolution threshold and as a ready-formulated sentence.
- **`RandomFeatureNARX`** — nonlinear, without PyTorch, with a real model seed. Changes
  **exactly one** thing compared with the linear ARX: the basis function. This makes it possible, if the
  GRU wins, to separate whether the gain comes from nonlinearity, learned state or
  rollout training.
- **`torch_sequence.py`** with an exchangeable core (GRU / LSTM) behind *one* trainer.

### Part 2 — what is still missing, and in which order

**Before the first neural run, without exception:**

1. **Get six skipped torch tests green.** Among them the assertions
   "zero-initialized head = persistence, exact" and "GRU and LSTM share the trainer".
2. **Learning curves up to 128 and 256** for ARX and NARX. The NARX curve is still going at n = 64;
   without these points a saturation is reported that only concerns the linear baseline.
3. **Repeat the Δt pre-study** with corrected moisture level (τ_fast ≈ 35 s instead of
   13.7 s). Δt = 60 s is thus no longer ruled out, and the sequence length of all
   architectures depends on it.
4. **Fix the seed budget and the relevance limit.** Three seeds resolve 10–15 %; for 5 %,
   12–26 are needed. The relevance limit has to be set separately — in soft-sensor operation,
   5 % above the mean prediction is statistically detectable and practically meaningless.

**Then the architectures**, in this order and with this purpose:

| Slot | What it is meant to answer |
|---|---|
| GRU | Does rollout training bring anything over one-step fitting? This is the main question. |
| LSTM | Does a second, additively updated memory help with few runs, or is the additional capacity a risk? |
| xLSTM (mLSTM-heavy) | Does the matrix-valued memory cell help with the slow mode? mLSTM first, sLSTM only if needed (in the paper xLSTM[7:1]). |
| Mamba | Is a selective state space better than a gating mechanism given three decades of stiffness? Write as a pure PyTorch reference, not via `mamba-ssm` (CUDA kernel). **Implemented (Befund 38):** O(1) core property confirmed (step time flat over 68 steps), but 5.2-9.7x more expensive than GRU/LSTM depending on the comparison method (composite inference 365 µs/window) and only runnable after three stability-fix cycles (RMSNorm fixes a NaN collapse caused by an unboundedly growing rollout input; weight-decay exclusion for `A_log`/`D` without measurable effect at `n_train=16`). In the reduced single data point (1 seed, `n_train=16`) worse than pure persistence for `product_moisture_kg_per_kg` (h=36). |
| Transformer | Control group without recurrent state. **Implemented (Befund 37):** at H=36 (Befund 35, no longer the 24 steps of Befund 23) the quadratic attention is **not** uncritical -- no KV caching in the decoder loop, 14-18x more expensive than GRU/LSTM in training and inference alike, and in a reduced single data point (1 seed, `n_train=16`) at the same time worse in NRMSE than GRU/LSTM. |

**Target: `product_moisture_kg_per_kg` in the multi-step rollout.** `energy_target` runs
along as a control quantity, `d50_um` is dropped (84 % noise floor). Rationale sharpened after
step 6, part 1: for the moisture, undercutting persistence **and**
nonlinearity both fail; the ARX rollout accumulates errors instead of damping them. That is the
gap for which rollout training exists.

**Requirements for all architectures:**

- **Residual parametrization at an anchor.** The head predicts Δy; zero weights
  correspond exactly to the parameter-free baseline. **Refinement from Befund 21:** where there is
  no anchor (soft-sensor operation), the head is to be parametrized **directly** —
  otherwise the prediction is an integrator without feedback and drifts over the horizon.
- **No teacher forcing as standard.** Primary metric is the rollout; parameters for a
  curriculum are present, default 0.0.
- **Sequence length is a model decision, not a hyperparameter question.** At Δt = 25 s
  and 600 s horizon it is 24 steps; the slow mode (16,000 s) lies structurally
  outside and is covered by the scatter of the initial conditions. Befund 23 shows
  that the baselines do not profit from more than two lags — whether a model that
  *can read* 32 steps profits from them is one of the open questions of part 2.
- Physics constraints: non-negativity and value ranges as output projection, monotonicity
  (∂dE/∂T_in > 0, ∂dE/∂u_gas > 0) via monotonic networks / ICNN.

---

## What is deliberately NOT done yet

- **No RL.** Stable multi-step rollouts must be in place first. For the moisture, the
  best rollout is worse than persistence even noise-free and even nonlinear — an
  RL agent on this surrogate would exploit model artifacts. Month 5–6 at the earliest.
- **No PSD statements from the grey-box model.**
- **Do not release `p_nozzle` in RL** as long as the coupling droplet size → wetting →
  growth is missing. The lab range is moreover narrower than the placeholder (1.0–1.8 bar
  vs. 1.0–4.0), which shrinks the apparent leeway further.
- **No architecture code written blind.** xLSTM and Mamba were only written
  after it had been checked that they can be executed on this machine (both
  implemented by now, Befund 38 for Mamba).
- **No further literature.** The open questions are empirical.

---

## Which gaps block which step

| Gap | Blocks | Workaround until then |
|---|---|---|
| Max. rates of change of the actions | Sequence length | Placeholder; Δt pre-study shows low sensitivity |
| **Inline measurability of the moisture at application partner** | **which result holds** | both operating modes implemented and reported |
| **Sorption isotherm** | Time constants, hence Δt and sequence length | Sensitivity calculation available (`02_prozess_setup.md`) |
| Time constant of the air heater | Dynamic share of `dE` | Moisture as the main target |
| Definition of `dE` | Validity of the monotonicity constraints | both variants implemented |
| Computing time per Dyssol run | End point of the learning curves | Reference value from the previous project: ~69 h for 128 runs |
| Noise level of the plant sensors | ~~Saturation height~~ | Ranking stable over two orders of magnitude; **the outlier process is open** |
| **PyTorch in the development environment** | **Step 6, part 2 in full** | NARX carries the comparison until then |

**Conclusion:** One step is now genuinely blocked — part 2 of step 6 needs a machine
with PyTorch. Everything else runs. Placeholders are permissible as long as they are explicit and
versioned, but two of them (moisture level, sorption isotherm) co-determine the
experiment planning and are now recognized as wrong, not just as unknown.
