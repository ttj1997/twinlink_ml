# 48 -- Conclusion of AP 1: method selection for the dynamic control of fluidised-bed spray granulation

> **Status:** final, version 3, 2026-11-23 (version 1: 2026-11-20; version 2: 2026-11-23). The
> selection of section 7 and the deviation of section 6 are accepted by the project lead (decisions
> of 2026-11-23, section 11). Document number 48 (no lab-book entry); merged into `master` on 2026-11-23. Branch `ap1-conclusion`, rebased onto `master` 82dcc84 (lab book up to
> Befund 129, with the addenda of 2026-11-23 to Befunde 105, 126, 127 and 128). No code, no run, no new number: every number
> below is quoted from a finding of the lab book (`docs/04_stand_ap1.md`) and traced to its artifact
> in section 10. `docs/08_synthese_m2.md` stays frozen at version 5f (2026-10-20) and is not changed;
> it documents the method and the comparison of the forecast models. This document adds what
> version 5f does not contain: the control studies after it (Befunde 105 to 129), the explicit
> selection of methods, and the reasoned deviation from the proposal on reinforcement learning.
> **All results are statements about a grey-box simulator with placeholder parameters or about
> synthetic data, not about a plant.** Findings are cited as "Befund N".

## 1. Summary

AP 1 of sub-project TUHH-B asks which methods of machine learning suit the dynamic control of a
single process, with fluidised-bed spray granulation as the example, and in particular the
prediction of optimised transient manipulated variables. The studies answer this on the simulator
as follows.

1. **Forecast model.** Among eight methods GRU and LSTM have the lowest errors; after Holm no
   GRU-against-LSTM cell is resolved. GRU is the default on practical grounds (fewer parameters,
   lower single-window latency). Transformer, xLSTM (mLSTM-only variant) and Mamba are not selected
   in the tested, untuned configuration. Every method is far below the sample time of 25 s
   (`docs/08_synthese_m2.md`, version 5f).
2. **Control.** The energetic lever is tracking the operating point, not the dynamics: the
   quasi-stationary set-point optimisation (RTO) saves 8.93 % energy in the median against a fixed
   operating point, an MPC adds 0.55 %, and the upper bound with perfect preview is 1.51 %
   (Befund 105). With the throughput as a command under an hourly electricity price, a two-level
   price rule saves 2.07 % of the cost per kg of dry product, the economic MPC 2.88 % and the
   perfect preview 3.89 % (Befund 126); this gain exists only with an electric heater. The numbers
   are quoted as recorded; with the cold multi-start of the steady optimiser the dated addenda give
   9.10 % instead of 8.93 % and 2.08 % instead of 2.07 %, and the readings are untouched (Befund 129;
   section 4).
3. **Learned models in the control (prediction of optimised transient set points).** No learned
   arm comes within 2 % of the same control law with the known model: the GRU in the loop costs
   2.98 % more per kg, the GRU map with an inline moisture feedback 3.87 %, without feedback
   7.07 %. That is more than the 2.07 % load following gains. Networks that output the commands
   directly do not beat a fixed set point (Befund 127).
4. **Reinforcement learning.** RL was trained on the one-step operating-point problem, also with
   the learned GRU as its environment, and on a storage problem. It does not reach the direct
   optimisation in any of them. A sequential RL agent for the granulator was not trained, because
   the pre-registered headroom tests bound its possible gain below the thresholds fixed beforehand
   (section 6).
5. **What decides.** A model is chosen for the control by admissibility and by the cost at the held
   limit, that is by the safety margin it needs at the moisture limit, not by its mean prediction
   error (Befunde 113, 122, 125, 127). The feedback of the product moisture is the most effective
   lever in continuous operation (Befunde 111, 112; section 7).

Not complete in AP 1: training on data of dynamic Dyssol models (A6, partial) and the integration
of the pipeline into the framework of the consortium coordinator (A7: the interface is built,
`docs/47_module_api.md`; the integration is open, the specification is requested; section 9). The selection of section 7 is accepted by the project lead (decision of 2026-11-23,
section 11); each of its lines rests on a reading the project lead has accepted.

## 2. The tasks of AP 1 and where each stands

The tasks are paraphrased from the approved proposal (section 4.2.5; the proposal is not in the
repository; short form in `docs/01_projektsteckbrief.md`).

| No. | Task of AP 1 (paraphrase) | Status | Evidence | Section |
|---|---|---|---|---|
| A1 | Compare ML methods for the dynamic control of single processes, in particular for the prediction of optimised transient manipulated variables; example process fluidised-bed spray granulation | done on the simulator | Befunde 105, 113, 126, 127 | 4, 5 |
| A2 | Emphasis: reinforcement learning in combination with time-based models (recurrent networks, Transformer) and their suitability for process optimisation | done with a reasoned deviation: no sequential RL agent for the granulator | Befunde 93, 96, 113, 115; gates of Befunde 105, 126 | 6 |
| A3 | Select the most suitable methods by accuracy, generalisability and real-time ability | done: selection of section 7, accepted by the project lead on 2026-11-23 | `docs/08_synthese_m2.md` 5f; Befunde 94, 97, 98, 104, 109 | 3, 7 |
| A4 | Reliable prediction with few training data through a suitable sampling of the parameter range; low data need | done; the data need beyond 64 runs is not determined | Befunde 55, 87, 88, 94 (addendum), 107, 122, 127 | 3, 8 |
| A5 | Contribute to a lower energy demand at equal product quality | done on the simulator; quality is the product moisture only | Befunde 105, 126 | 4 |
| A6 | Training and comparison first with synthetic data from existing Dyssol models and from TUHH-A-AP 2 | partial: stationary Dyssol flowsheets done (Befund 128); training on data of dynamic Dyssol models depends on the granulation model of TUHH-A | Befund 128 | 8, 9 |
| A7 | Data and model pipeline as a separate module in the framework of the consortium coordinator, usable for other optimisation tasks such as the distribution of waste heat | interface built (`docs/47_module_api.md`); integration into the framework of the consortium coordinator open, specification requested | `docs/47_module_api.md` (infrastructure, no finding) | 9 |

## 3. Forecast model: accuracy, generalisability, real-time ability

This section restates `docs/08_synthese_m2.md`, version 5f, without a new number; the method
(learning curves, run-level splits, two named test sets, baselines first, paired comparison over
five seeds with Holm correction) is described there.

**Accuracy.** Eight methods (persistence, static regression, ARX, GRU, LSTM, Transformer, xLSTM,
Mamba), three targets, two test sets, 8 to 64 training runs, five seeds, horizons 1 and 36
(Befund 94). After Holm no GRU-against-LSTM cell is resolved in any channel (Befund 98). Against
the LSTM the Transformer is worse in 39 of 48 cells without correction, the xLSTM in 38 and Mamba
in 44, never better; after Holm 2, 2 and 14 of these cells remain (Befund 98). The convergence
check with longer training leaves the order GRU >= LSTM > Transformer on energy and bed
temperature unchanged (Befund 104). All statements hold for the tested, untuned configuration with
networks of unequal size, not for the architectures in general.

**Generalisability.** Every number is quoted with its test set (excited test set or constant-run
holdout), because one hides what the other shows (Befund 51). The ranking does not reverse under outliers
in the moisture signal, and no baseline moves ahead after Holm (Befund 109). Models trained on
constant operation only stay worse on the product moisture by a factor of 2 to 3 than the same
model trained on excited data (Befund 55 with its addendum). Against a plant that deviates from
the simulator, pre-training on the simulator plus fine-tuning beats training from scratch at small
numbers of runs on every plant variant tested: the error ratio falls from 12.69 (identical plant)
and 10.27 (parameter deviation) at 2 runs to 1.32 and 1.27 at 64 runs (Befund 88).

**Real-time ability.** On an idle machine the latency of one window is 4,753 microseconds for the
GRU, 6,502 for the LSTM and 31,567 for the Transformer, the most expensive method (Befund 97):
every method is far below the sample time of 25 s. A hybrid that needs the simulator ahead of every
prediction spends 98.8 % of its inference cost in the simulator (Befund 97).

**Default.** GRU: never resolved worse than LSTM, 42,819 against 57,027 parameters, lower
single-window latency. This is a practical default, not a performance rank.

## 4. Control with the known model: where the energy is

Truth in all studies of this section: the grey-box model of the laboratory line; the quality limit
is the product moisture (0.15 kg/kg, placeholder) as a path condition. Each controller gets the
smallest family-wide safety margin with which the limit holds; costs are compared at that held
limit. The controllers of this section carry the true model as their internal model.

| Befund | Operation | Commands | Compared | Result (median) | Pre-registered reading |
|---|---|---|---|---|---|
| 105 | continuous operation, disturbances ridden out | inlet temperature and gas velocity every 5 min | R0 fixed operating point, R1 quasi-stationary RTO, R3 MPC without disturbance preview (2 h), R2 perfect preview (upper bound) | R0 -> R1 8.93 % energy (0.69 .. 25.23 %; addendum: 9.10 % with the cold multi-start); R1 -> R3 0.55 % (0.05 .. 1.50 %); R1 -> R2 1.51 % (1.02 .. 2.44 %) | L0: dynamic control does not pay in continuous operation |
| 105 | product change (moisture target 0.15 -> 0.12) | as above | R1, R3, R2 | time to the new specification: R1 37-55 min, R3 0.9-1.8 min | none (per changeover; frequency unknown) |
| 126 | load following under an hourly price with a daily production target, main band 1.2-1.8 kg/h | in addition the spray rate (throughput), hourly | C0 constant throughput with the steady optimum, C1 two-level price rule, C2 schedule on the steady-state map, C3 economic MPC, C4 perfect preview (upper bound) | gain in cost per kg against C0: 2.07 % / 2.06 % / 2.88 % / 3.89 % (C1 / C2 / C3 / C4; addendum: C1 2.08 %, C2 1.98 % with the cold multi-start and a newly searched margin); `gap(C1, C3)` 0.83 % | T1: a price rule suffices |

**Reading of the table.** (1) The step from a fixed operating point to the steady optimum at the
current disturbance carries the saving (8.93 %); the dynamics add 0.55 %, and even the perfect
preview only 1.51 % (Befund 105). (2) The MPC differs from the simple methods by its smaller
margin, not by its trajectory or schedule: in Befund 105 the margin share of R1 -> R3 is 0.92 % and
the trajectory share -0.35 %; in Befund 126 the schedule share is about 2.5 % for every controller.
(3) The value of an MPC lies in the product change, as time outside the specification, not in
energy (decision of the project lead on Befund 105). (4) Load following shifts consumption in time
and saves no energy: C1 and C2 use slightly more energy than C0 (1135.3 against 1132.9 kWh). With a
gas heater there is no gain (C1 and C2 are 0.04 % and 0.08 % more expensive than C0; Befund 126).

**Addenda of 2026-11-23 (cold multi-start, Befund 129).** The table and the text quote Befunde 105
and 126 as recorded. The steady optimiser behind the reference controllers depends on the order of
its queries. Recomputed with a cold multi-start (8 starts; R0, R1, C0, C1 and C2 only), R1 holds
the limit with the margin 0.005859 instead of 0.006641, and its median saving against R0 is 9.10 %
instead of 8.93 % (addendum to Befund 105). C2 does not hold the limit at its recorded margin;
with the newly searched margin 0.007812 instead of 0.007422 its gain against C0 is 1.98 % instead
of 2.06 %, and that of C1 is 2.08 % instead of 2.07 % at an unchanged margin (addendum to
Befund 126). No statement of Befund 105 is withdrawn, so L0 stands, and the reading T1 is
untouched. The controllers with a dynamic optimiser (R3, R2, C3, C4) were not recomputed
(`docs/12`, item 38): the gaps to them (0.55 %, 1.51 %, 0.83 %, 1.05 %) and the shares are as
recorded.

**Manipulated variables.** Inlet temperature, gas velocity and spray rate were commands in the
dynamic studies; the studies of Befunde 126 and 127 run with these three commands.

**Other optimisation tasks (waste heat).** For a heat pump with storage under the same hourly
prices the schedule-aware price rule is 0.65 % above the MPC with perfect day-ahead prices
(Befund 115, reading H0: a simple price rule suffices). For the campaign heat network a rolling
planning with an exact standstill plan stays within 0.98 % of the bound with foresight in every
cell (Befund 119).

## 5. Learned models in the control: the prediction of optimised transient set points

**One-step problem, learned model as the environment (Befund 113).** RL (SAC, TD3) and the RTO
were trained or optimised on the composite GRU of the architecture pipeline and evaluated on the
grey-box model. With the margin 5 m they stay feasible (reading G0 for SAC, TD3 and RTO). The
margin carries this: at the margin m the optimiser on the GRU trained on 64 runs violates the limit
in 39 of 75 cases, because the optimisation seeks out the errors of the model in its favour. RL
does not exploit the model more than the RTO.

**Load following with learned models (Befund 127).** Frame of Befund 126 unchanged; the twin of
every learned arm is the schedule C2 with the true model. Three families: M, the steady map of the
unchanged law replaced by the map of a learned model (static quadratic regression as the baseline,
fixed point of the composite GRU), without feedback and with an inline moisture bias correction;
D, the GRU rollout optimised every 5 min over 900 s; P, policies that output the three commands
(imitation of the twin on 5 to 40 solved days). Gap to the twin in cost per kg, median over the
admissible seeds:

| Arm | Data | Gap to the twin | Median margin (kg/kg) | Admissible seeds |
|---|---|---|---|---|
| D-gru (GRU in the loop) | `n_train` 16 / 64 | +5.44 / +2.98 % | 0.0155 / 0.0088 | 5 of 5 |
| M-gru with inline feedback | `n_train` 8 / 16 / 32 / 64 | +6.06 / +3.98 / +3.23 / +3.87 % | 0.0285 / 0.0224 / 0.0195 / 0.0221 | 5 of 5 |
| M-gru without feedback | `n_train` 8 / 16 / 32 / 64 | +11.28 / +7.44 / +7.04 / +7.07 % | 0.0711 / 0.0463 / 0.0416 / 0.0357 | 5 of 5 |
| M-quad with inline feedback | `n_train` 64 | +8.51 % | 0.0203 | 5 of 5 |
| M-quad without feedback | `n_train` 64 | +22.36 % | 0.0934 | 4 of 5 |
| P-gru | `K` 10 / 20 / 40 | +21.18 / +12.05 / +10.47 % | 0.0752 / 0.0330 / 0.0338 | 5 of 5 |
| P-lin | `K` 40 | +14.15 % | 0.0344 | 5 of 5 |
| P-mlp | `K` 40 | no number | - | 1 of 5 |

27 of 170 learned cells reach no margin within 0.12 kg/kg and are not admissible. The margins of
the rows M-quad, P-gru and P-lin and the admissible seeds of M-gru with feedback are taken from the
table of `docs/44_learned_setpoints.md`, section 16.1; all other entries are in the lab-book entry
as well.

**Pre-registered reading: U3 in the families M (without and with feedback), D and P**, accepted by
the project lead: no arm is within 2 % of its twin, so no arm is named and no data need is
determined. Family P is U3 by the letter only: the median cost of P-gru is 0.015 % below the fixed
set points, and in substance the policies do not beat a fixed set point at 40 solved days.

**Where the gap comes from.** For M-gru with feedback the margin share is +3.82 % and the set-point
share +0.05 %: the learned map chooses the set points of the twin and pays for its distance to the
limit. On the twin curve a margin of 0.015 kg/kg costs 1.92 % and 0.02 kg/kg 3.26 %, so the 2 % of
the reading is a margin of about 0.015. No learned arm is cheaper in the median than constant
throughput with the true steady optimum (C0) at any budget; the closest is D-gru at `n_train` 64,
0.85 % above C0. After Holm 1 of 27 rank statements between learned arms is resolved.

**Computing time per decision (Befund 127).** Policies below 0.1 ms, table arms about 6 ms, D-gru
84 ms, the MPC on the true model used as its control 119 ms: all far below the command interval of
5 min.

**Feedback.** In continuous operation under a model mismatch the value of a moisture feedback
against no feedback is 13.40 % energy in the median (Befund 111, reading V2); a laboratory sample
every 4 h captures 93 % of the value of an inline sensor (Befund 112). A GRU soft sensor fed with
commands and disturbances only is blind to the mismatch (it captures at most an eighth of the
value; Befund 111), and one with measured bed temperature and exhaust humidity is worse than no
feedback (Befund 112). Under load following the inline feedback about halves margin and gap of the
GRU map, while the laboratory sample every 4 h does not help (+7.62 % at `n_train` 64 against
+7.07 % without feedback; Befund 127, descriptive).

**Calibrate before learning.** When the structure of the physical model fits, a refit of two
kinetic parameters needs a margin of 0.0019-0.0039 kg/kg, the pure GRU 0.1173 / 0.12 / 0.0456 /
0.0318 at 4 / 8 / 16 / 32 runs (Befund 122). With a structural error to the unsafe side the
calibrations are admissible in 13 of 15 cells, the hybrid and the GRU in 6 of 15 each
(Befund 125). The hybrid has a rollout error of the moisture 8 to 12 times smaller than the GRU at
4 and 8 runs without a resolvably smaller margin (decision of the project lead on Befund 122): the
margin is set by the least favourable cases, not by the mean error.

## 6. Reinforcement learning: what was done and why no sequential agent was trained

The proposal puts an emphasis on reinforcement learning in combination with time-based models.
What was done:

- **One-step operating-point problem on the grey-box model.** With a residual action and a
  separated moisture cost SAC does not meet the pre-registered feasibility gate in any cell
  (feasibility 0.88, range 0.60-1.00, in the cell with both measures; Befund 93). With the margin 5 m the policy
  is feasible in all five seeds at 6.9 % extra energy over the optimiser with the margin m; of
  that, 6.5 points are the margin itself and 0.4 points the policy (Befund 96).
- **RL in combination with the sequence model.** SAC and TD3 trained on the composite GRU as the
  environment (Befund 113, section 5): feasible on the grey-box model only with the margin 5 m,
  and no better than the RTO on the same learned model.
- **Sequential RL on a storage problem.** SAC and TD3 (three seeds, 400 000 steps) for the heat
  pump with storage end behind the constant rule, by +3.13 % and +2.98 % (Befund 115); untuned.

Why no sequential RL agent for the granulator was trained:

1. **The headroom was measured first, with thresholds fixed beforehand.** In continuous operation
   the upper bound with the true model and the true course of the disturbances is 1.51 % above the
   quasi-stationary RTO; the reading is L0 (pooled median below 2 %, no scenario type in another
   class), and by the rule fixed beforehand the RL line for the granulator in continuous operation
   was closed (Befund 105, decision of the project lead of 2026-10-22).
2. **Load following did not open it again.** With the throughput as a command, `gap(C1, C3)` is
   0.83 % (threshold 2 %) and `gap(C3, C4)` 1.05 %; the RL threshold of 5 % is far away (reading
   T1, Befund 126, decision of the project lead of 2026-11-19).
3. **Where RL was trained it did not reach the direct optimisation** (Befunde 93, 96, 113, 115),
   and its cost is the margin it needs, which the optimiser pays too.

**What this deviation does not cover.** (a) Both gates use controllers that see the true state;
partial observability (state estimator, measurement noise, soft-sensor operation) was not tested
and is expressly open (decision of the project lead on Befund 105). (b) In the whole throughput box
0.4-3.0 kg/h the perfect preview gains 8.96 % and the schedule on the map 1.98 %, without a
reading; a follow-up waits for the answers of the application partner (`docs/12`, item 36).
(c) Frequent product changes were not assessed (frequency unknown). In each of these cases the
gate would have to be measured again before an RL study.

## 7. The selection (accepted by the project lead on 2026-11-23)

Accepted with four additions, marked (a) to (d) here and listed in section 11.

**Criterion for the choice of a model for the control (a).** A model for the control is chosen
by admissibility first (does it reach a safety margin with which the moisture limit holds) and
by the cost at the held limit second, not by its mean prediction error. Evidence: the hybrid has
a rollout error of the moisture 8 to 12 times smaller than the GRU without a resolvably smaller
margin (Befund 122); with a structural error to the unsafe side the calibrations are admissible
in 13 of 15 cells, the hybrid and the GRU in 6 of 15 each (Befund 125); under load following the
gap of the GRU map with feedback is its margin (margin share +3.82 % of a gap of +3.87 %;
Befund 127).

| Role | Selected | Not selected | Reason |
|---|---|---|---|
| Forecast model of moisture, energy target and bed temperature | GRU, composite of per-target networks; LSTM equivalent | Transformer, xLSTM (mLSTM-only variant), Mamba in the tested, untuned configuration | Befunde 94, 98, 104, 97 |
| Energy-optimal operation in continuous operation | quasi-stationary set-point optimisation on a calibrated physical model with a safety margin at the moisture limit | MPC and RL as energy measures | Befund 105 (L0) |
| Product change | MPC, for the time outside the specification | - | Befund 105 (no reading; value depends on the frequency of changes) |
| Load following under an hourly price | two-level price rule, with an electric heater only (c) | schedule, economic MPC, RL; load following with a gas heater (no gain) | Befund 126 (T1; with a gas heater C1 and C2 are 0.04 % and 0.08 % more expensive than C0) |
| Internal model of the controller | the order "calibrate, bias, only then learn": a calibrated physical model first, a bias correction second, a learned model last | learned model in place of a fitting physical model | Befunde 122, 125 (decision, item 5: resolved on the safe side, only descriptive on the unsafe side), 127 |
| Learned model in the loop, where one is needed | GRU in the loop (D-gru) or GRU map with inline moisture feedback, each with its own margin | direct prediction of the commands (family P); static quadratic map; GRU map without feedback | Befund 127 (U3; no arm within 2 %) |
| Moisture information | feedback of a moisture measurement, the most effective lever in continuous operation (b); there a laboratory sample every 4 h | GRU soft sensor as the feedback | Befunde 111, 112, 127 |
| Reinforcement learning | not selected for the granulator | - | section 6 |
| Training data | excitation experiment at the plant; pre-training on the simulator plus fine-tuning | constant operating records alone; hybrid with the simulator as an input channel as the default | Befunde 55, 87, 88 |
| Steady optimiser of the set-point optimisation | cold multi-start; new studies set `steady_cold_starts = 8` (d) | warm start from the previous solution (the default 0 stays only to reproduce old results) | Befund 129 (O0; largest change 3.78 %) |

The line "learned model in the loop" names the two cheapest learned arms of Befund 127; the reading
U3 names no arm, and D-gru against M-gru is not resolved.

**Moisture feedback (b).** Under a model mismatch in continuous operation the value of a moisture
feedback is 13.40 % energy in the median (Befund 111, reading V2), and a laboratory sample every
4 h captures 93 % of the value of an inline sensor (Befund 112). Under load following the inline
feedback about halves margin and gap of the GRU map, while the laboratory sample every 4 h does
not help (Befund 127, descriptive). The 13.40 % are measured under a model mismatch against the
bias-corrected set-point optimisation without feedback, the 8.93 % of Befund 105 with the true
model against a fixed operating point: the two numbers come from different frames and are not
steps of one ladder.

**Load following (c).** The gain of Befund 126 exists with an electric heater only; with a gas
heater there is none (section 4).

**Steady optimiser (d).** The steady optimiser starts every solve from the previous solution, so
its result depends on the order of the queries: the warm start returns a worse local optimum than
a cold multi-start in 1.82 % of 94211 queries, by up to 2.58 % of the power. Pre-registered reading
O0: no key number of Befunde 105, 126 and 127 that rests on it changes by more than 10 % when the
dependence is removed (largest change 3.78 %). New studies set `steady_cold_starts = 8`; the
defaults stay at 0 to reproduce old results (Befund 129, decision of the project lead of
2026-11-23).

## 8. Data need

- **Forecast model.** The moisture error of GRU and LSTM at 8 training runs is 3.7 to 5.1 times
  the error at 64 runs, and the curves still fall at 64 runs: the need beyond 64 runs is not
  determined (addendum to Befund 94). Every point also uses 32 validation runs.
- **Control with a learned model.** No arm is within 2 % of its twin, so no data need is
  determined (Befund 127). In the medians the gap of the GRU map does not shrink from 16 runs on
  (table of section 5); the gap of D-gru is +5.44 % at 16 and +2.98 % at 64 runs.
- **Sampling of the parameter range.** A planned excitation was not established as better than a
  plain random-step design in the primary test (Befund 87); excited data against constant
  operation is the difference that matters (Befund 55). For a steady set-point optimisation with
  few true evaluations only the Gaussian process with an adaptive design passes the gate, from 64
  true evaluations on the line model and only at 256 on the grey-box model (Befund 107).
- **Dyssol.** For stationary Dyssol flowsheets a surrogate needs 32 Dyssol runs with one line and
  128 with two, reached by a quadratic regression before the MLP (64 and 256); reading D2
  (Befund 128). The units are closed formulas: the chain (design of experiments, run, control, fit,
  reading) carries over to dynamic Dyssol models, the verdict does not.

## 9. Limits and open items

**Limits of every statement above.**

- Grey-box simulator with placeholder parameters (drying kinetics, moisture limit, lower bound of
  the gas velocity among them); one laboratory line. Whether the moisture limit binds at the plant
  is an open question to the application partner (Befund 105).
- Product quality is the product moisture only; the median particle diameter does not depend on
  the operating point in the model (Befund 126, caveat 3).
- The controllers of Befunde 105 and 126 see the true state; the product moisture is not measured
  at the plant (Befund 56), and the inline feedback of Befund 127 assumes a measurement the plant
  does not have.
- The twin of Befund 127 is a perfectly calibrated physical model, the most favourable case for
  the physics. The plant-near model of plant A misses the exhaust temperature of the other day by
  up to 5.51 K (Befund 117); the studies with a structural error are Befunde 124 and 125.
- The margins of Befund 127 are found on the test days; a margin searched on other days holds on
  the test days in every seed for no learned arm, so the reported margins are optimistic.
- Five seeds (three in Befunde 113 and 122), untuned networks and RL agents throughout.
- **Order dependence of the steady optimiser** (Befund 129): found in Befund 127, where the re-run
  of C2 differs from the records of Befund 126 by 0.0016 % in the sum, and measured in Befund 129
  with the reading O0 (largest change of a key number 3.78 %; section 7, (d)). The numbers of
  Befunde 105 and 126 are quoted as recorded, with the dated addenda named in section 4. The twin
  of Befund 127 was recomputed as well: at newly searched margins its cost changes by 0.075 % and
  the smallest gap of a learned arm by 2.603 % relative (0.029000 against 0.029775, with the costs
  of the arm held), so that gap stays above the 2 % of the reading U3. Not recomputed: the
  controllers with a dynamic optimiser (R2, R3, C3, C4) and the learned arms (`docs/12`, item 38).

**Not complete in AP 1.**

- **A6, Dyssol data: partial.** The sequence models were trained on data of the project's own
  grey-box simulator. Stationary Dyssol flowsheets are done (Befund 128). Training and comparison
  on data of dynamic Dyssol models depend on the granulation model of TUHH-A and are open.
- **A7, module: interface built, integration open.** The package `twinlink_ml.api` makes the
  pipeline integrable: forecast, set-point, calibration and heat-distribution service, a
  transferable model bundle and versioned JSON schemas (`docs/47_module_api.md`; infrastructure,
  no finding). It contains no framework adapter. The framework, its transport and its information
  model are the consortium coordinator's to define; the specification is requested. The task is
  not done as long as no adapter exists (decision of the project lead of 2026-11-23).

**Left to the managing session at merge.** The number of this document, its row in `docs/12` and a
README paragraph.

## 10. Traceability table

Columns: statement (section) · number(s) · Befund · artifact in `artifacts/runs/` · study document.
Rows that restate `docs/08_synthese_m2.md` 5f carry the artifact named in its appendix F.

| Statement (section) | Number(s) | Befund | Artifact | Document |
|---|---|---|---|---|
| GRU against LSTM after Holm (3) | no cell resolved | 98 | `holm_rank_cells_rerun_full.json` | `docs/08`, section 3 |
| Transformer, xLSTM, Mamba against LSTM (3) | 39 / 38 / 44 of 48; Holm 2 / 2 / 14 | 98 | `holm_rank_cells_rerun_full.json` | `docs/08`, section 3 |
| Convergence check (3) | order unchanged | 104 | `convergence_check_94_conv600_main_attempt2.json` | `docs/08`, sections 3, 7 |
| Outliers in the moisture signal (3) | no reversal, no baseline ahead after Holm | 109 | `outlier_robustness_analysis_main.json` | `docs/27_outlier_robustness.md` |
| Constant-trained against excited (3, 8) | factor 2 to 3 | 55 with addendum | `befund55_addendum_ratio_range.json` | `docs/08`, section 6 |
| Pre-training plus fine-tuning (3) | 12.69, 10.27, 1.32, 1.27 | 88 | `a1_main_analysis.json` | `docs/08`, section 6 |
| Latency and parameters (3) | 4,753 / 6,502 / 31,567 microseconds; 42,819 / 57,027; 25 s; 98.8 % | 97 | `joint_timing_idle.json`, `a1_inference_time_v2_idle.json` | `docs/08`, section 4 |
| Headroom in continuous operation (4, 6) | 8.93 % (0.69 .. 25.23); 0.55 % (0.05 .. 1.50); 1.51 % (1.02 .. 2.44); shares 0.92 % / -0.35 %; L0 | 105 | `dynamic_headroom_v2.json.gz`, `dynamic_headroom_v2_analysis.json` | `docs/24_dynamic_headroom.md` |
| Product change (4) | 37-55 min; 0.9-1.8 min | 105 | `dynamic_headroom_v2_analysis.json` | `docs/24_dynamic_headroom.md` |
| Load following (4, 6) | 2.07 / 2.06 / 2.88 / 3.89 %; gaps 0.83 %, 1.05 %; schedule share about 2.5 %; 1135.3 / 1132.9 kWh; gas 0.04 / 0.08 %; whole box 8.96 % / 1.98 %; T1 | 126 | `load_following_v1.json.gz`, `load_following_v1_analysis.json` | `docs/43_load_following.md` |
| Heat pump and storage; RL there (4, 6) | 0.65 %; +3.13 % / +2.98 %; 400 000 steps; H0 | 115 | `storage_price_v2.json.gz`, `storage_price_v2_analysis.json` | `docs/32_storage_price.md` |
| Campaign network (4) | 0.98 % | 119 | `campaign_network_2_v1.json.gz`, `campaign_network_2_v1_analysis.json` | `docs/36_campaign_network_2.md` |
| Learned model as the environment (5, 6) | 39 of 75; margin 5 m; G0 | 113 | `surrogate_env_v1_analysis.json` | `docs/30_surrogate_env.md` |
| Learned set points: gaps, margins, admissibility (5, 8) | table of section 5; 27 of 170; 0.12 kg/kg; U3 | 127 | `learned_setpoints_v1.json.gz`, `learned_setpoints_v1_analysis.json` | `docs/44_learned_setpoints.md` |
| Learned set points: decomposition, twin curve, ladder, ranks, time (5) | +3.82 % / +0.05 %; 1.92 %, 3.26 %; 0.85 %; 0.015 %; 1 of 27; 0.1 / 6 / 84 / 119 ms; +7.62 % | 127 | `learned_setpoints_v1_analysis.json`; the twin curve from the records of `learned_setpoints_v1.json.gz` | `docs/44_learned_setpoints.md` |
| Value of the moisture feedback (5) | 13.40 %; an eighth; V2 | 111 | `sensor_value_v1.json.gz`, `sensor_value_v1_analysis.json` | `docs/28_sensor_value.md` |
| Laboratory sample, soft sensor with measured responses (5) | 93 % (4 h); worse than no feedback | 112 | `sensor_value_2_v1.json.gz`, `sensor_value_2_v1_analysis.json` | `docs/29_sensor_value_2.md` |
| Refit against GRU, rollout error against margin (5) | 0.0019-0.0039; 0.1173 / 0.12 / 0.0456 / 0.0318; 8 to 12 times | 122 | `hybrid_model_v1_analysis.json` | `docs/39_hybrid_model.md` |
| Structural error to the unsafe side (5) | 13 of 15; 6 of 15 | 125 | `hybrid_structural_2_v1_analysis.json` | `docs/42_hybrid_structural_2.md` |
| One-step RL, gate (6) | feasibility 0.88 (0.60-1.00) | 93 | `bres_gate.json` | lab book |
| One-step RL, margin (6) | 5 m; 6.9 %; 6.5 and 0.4 points | 96 | `bres_margin.json` | lab book |
| Data need of the forecast model (8) | 3.7 to 5.1; 32 validation runs | 94 addendum | `befund94_addendum_seeds_data_need_rerun_full.json` | `docs/08`, section 6, appendix D |
| Excitation (8) | not established | 87 | `b1_main_analysis.json` | `docs/08`, section 6 |
| Steady optimisation with few true evaluations (8) | 64; 256 | 107 | `surrogate_rto_v1_analysis.json`, `surrogate_rto_v2_analysis.json` | `docs/25_surrogate_rto.md` |
| Stationary Dyssol flowsheets (8) | 32 / 128; 64 / 256; D2 | 128 | `dyssol_steady_v1_one_line_analysis.json`, `dyssol_steady_v1_two_lines_analysis.json` | `docs/45_dyssol_steady_state.md` |
| Plant-near model (9) | 5.51 K | 117 | `plant_scale_v1_analysis.json` | `docs/34_plant_scale.md` |
| Order dependence of the steady optimiser: where found (9) | 0.0016 % | 127 | `learned_setpoints_v1.json.gz` | `docs/44_learned_setpoints.md` |
| Order dependence: reading, mechanism, rule (7, 9) | O0; 3.78 %; 1.82 % of 94211; 2.58 %; 8 starts; 0.075 %; 2.603 % (0.029000 against 0.029775) | 129 | `order_dependence_v1.json.gz`, `order_dependence_v1_analysis.json` | `docs/46_order_dependence.md` |
| Cold multi-start: addenda to Befunde 105 and 126 (1, 4) | 9.10 % instead of 8.93 %, margin 0.005859 instead of 0.006641; 1.98 % instead of 2.06 %, margin 0.007812 instead of 0.007422; 2.08 % instead of 2.07 % | 129; addenda of 2026-11-23 to 105 and 126 | `order_dependence_v1_analysis.json` | `docs/46_order_dependence.md` |
| Module interface (2, 9) | no number | none (infrastructure) | none | `docs/47_module_api.md` |

## 11. Decisions of the project lead (2026-11-23)

Entered with version 2; the sections above are changed accordingly.

1. **Section 7 is accepted, with four additions.**
   (a) The criterion for the choice of a model for the control is admissibility and the cost at the
   held limit, not the mean error (Befunde 122, 125, 127).
   (b) The moisture feedback is the most effective lever in continuous operation (Befunde 111, 112,
   127).
   (c) Load following only with an electric heater; with gas there is no gain (Befund 126).
   (d) The steady optimiser runs with a cold multi-start (finding on the order dependence, reading
   O0). The caveat on the order dependence in section 9 is replaced by the reference as soon as that
   finding is merged.
2. **Status of the tasks.** A6 "partial": stationary, Befund 128; dynamic, depending on the
   granulation model of TUHH-A. A7 "in progress", with the reference to the branch `module-api`.
3. **Finalisation.** Rebase and finalise only when `order-dependence` and `module-api` are merged;
   until then this documentation commit, and wait.

**Supplement (2026-11-23, second decision of the project lead on this document).**

4. **Section 6 is accepted** as well: the wording of the deviation on reinforcement learning,
   including the three cases it does not cover (partial observability, the whole throughput box,
   frequent product changes). The first decision had not addressed section 6, which version 1 had
   put to the project lead together with section 7.
5. **The sentence on the frames in addition (b) stays** (section 7: the 13.40 % of Befund 111 and
   the 8.93 % of Befund 105 come from different frames and are not steps of one ladder).

**Supplement (2026-11-23, third decision of the project lead: finalisation).** Both conditions of
item 3 are met: `order-dependence` is merged as Befund 129 (`docs/46_order_dependence.md`),
`module-api` as `docs/47_module_api.md` (infrastructure, no finding). Entered with version 3:

6. **Addition (d) is entered** in sections 7 and 9: the caveat is replaced by the reference to
   Befund 129 (O0, largest change 3.78 %; new studies set `steady_cold_starts = 8`).
7. **Numbers of Befunde 105 and 126:** quoted as recorded, with the dated addenda named (R1 9.10 %
   instead of 8.93 %; C2 1.98 % instead of 2.06 % at a newly searched margin; C1 2.08 % instead of
   2.07 %); the readings are untouched.
8. **A7:** "interface built (`docs/47_module_api.md`), integration into the framework of the
   consortium coordinator open, specification requested"; not "done" as long as no adapter exists.
   This replaces the status "in progress" of item 2.
9. **The document is final** with version 3.

## 12. Addendum (2026-11-23): two precisions after the independent check of the M2 synthesis

Decision of the project lead of 2026-11-23, after the independent check of `docs/08_synthese_m2.md`
version 6 (commit d49bdaf of the branch `synthesis-v6`). The sections above are unchanged and the
document stays final; the selection of section 7 is not touched. No code, no run, no new number.

1. **Status of the order "calibrate, bias, only then learn"** (section 7, row "Internal model of the
   controller"; decision on Befund 125, item 5: "resolved on the safe side, only descriptive on the
   unsafe side"). The resolution on the safe side is a post-hoc result: the cost measure of Befund 125
   applied to the records of Befund 124 resolves the refit with bias as cheaper than the hybrid at
   `n_train` 16. Against the plain GRU the order is not resolved (there the GRU against the refit with
   bias is not resolved; in Befund 122 the paired comparison of the refit with the hybrid is not
   resolved either). The precise status is therefore: resolved post hoc against the hybrid on the safe
   side, not resolved against the plain GRU, only descriptive on the unsafe side.
2. **Reinforcement learning** (section 1, item 4: "It does not reach the direct optimisation in any of
   them"; section 6, item 3: "Where RL was trained it did not reach the direct optimisation"). For
   Befund 113 this is too strong: with the learned GRU as the environment the comparison of RL with
   the optimiser shows no resolved difference in the moisture gap or the violation share (12 tests).
   The statement that holds for all the findings named there is: where reinforcement learning was
   trained it was **not better than** the direct optimisation. The reasoned deviation of section 6
   rests on the two headroom gates and is not affected.
