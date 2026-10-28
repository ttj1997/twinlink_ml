# 32 -- Heat pump and storage under an hourly electricity price: rule, MPC or RL?

Status: design fixed and committed before the main run (sections 1-6); runs and results in sections
7-9, added after the runs (numbers only from the artifacts named there). Follow-up of `docs/12`
section I item 1 (Befund 101: schedule-aware rule against a day-ahead MPC). Branch `storage-price`
(from `origin/master` a84b764); the document number is 32 and the finding number 115
(lab book). **Scenario on placeholders, not a plant statement.**

## 1. Question

A single storage without power limits is run optimally by the greedy rule (Befund 99); for the heat
pump, a tuned constant rule closes almost the whole gap to a programme with foresight (Befund 101,
decision of the project lead 2026-10-23). Open is the one case left in which decisions carry over in
time: a storage with charge and discharge power limits, a heat pump with a size limit, an hourly
electricity price and a gas boiler as the alternative. Question: does a simple schedule-aware price
rule suffice, or does it take a day-ahead MPC -- and if the MPC pays, is RL (SAC, TD3) an
equivalent alternative to it?

## 2. Model (structure of plant A, placeholders)

Package `src/twinlink_ml/storage_price/`, settings `conf/storage_price/storage_price.yaml` (every
number there carries its reason and band), tests `tests/test_storage_price.py`.

* **Streams, one hour per step, calendar year 2024 (8784 h).** Hot: low-temperature waste heat
  50 -> 35 degC, 1000 kW at nominal production. Cold: process air in the preheater from the intake
  temperature to 80 degC, 10 kW/K. Production plan: weekdays 1, Saturday and Sunday 0.6 (air flow and
  waste heat alike). Only the preheater is inside the system boundary (the main heater above it does
  not depend on the dispatch), so relative costs are costs of the controllable part.
* **Heat pump: model of Befund 101, only read** (`heatpump.cop.design_pairs`, `HeatPumpSpec`
  defaults): design pair 30 -> 90 degC, COP = 0.5 x Carnot = 3.03, heating capacity 800 kW;
  evaporator and condenser as isothermal pseudo-streams in the problem table (approach `dT_min/2`).
* **Storage: model of Befund 99, only read** (`storage_timeslice.table`): one level 75 degC,
  approach `dT_min` = 20 K on both sides (charging stream at shifted 85 degC: only the condenser can
  charge it; discharging stream at shifted 65 degC: serves the air up to 55 degC),
  `s+ = (1 - lambda dt) s + (q_c - q_d) dt`, capacity 2 MWh, loss 0.2 %/h. **New: power limits**
  charge 300 kW, discharge 300 kW.
* **Gas heat** = hot utility of the problem table, `U = max(0, max_g (M_g + alpha_g Q + beta_g q_c -
  gamma_g q_d))` (`heatpump.storage.coupled_pieces`, only read); fuel `U / 0.85`.
* What this gives in the main case: the waste heat covers the air below 30 degC in every hour, the
  demand above it is 500 kW on weekdays and 300 kW at the weekend; of it the storage can serve the
  lower half (30-55 degC), the upper half (55-80 degC) only the pump or gas. Pump heat costs
  `c_el / 3.03`, gas heat 0.0601 EUR/kWh (gas 0.040 + 55 EUR/t CO2, efficiency 0.85; Befund-101
  `Economics`), so the pump beats gas below an electricity price of 0.182 EUR/kWh.
* CO2 with the repository factors (electricity 0.366, gas 0.201 kg/kWh), constant; the controllers
  minimise cost, CO2 is reported.

## 3. Prices and split

* **Day-ahead prices DE-LU 2024, hourly**, Energy-Charts API (Fraunhofer ISE; data Bundesnetzagentur |
  SMARD.de, licence CC BY 4.0), retrieved 2026-10-27; URL and SHA-256 in the config; the raw file
  stays in `local/prices/` (gitignored). 8784 values, no gap, mean 0.0785 EUR/kWh, range -0.135 to
  0.936 EUR/kWh, 457 negative hours. 2024 is the last complete year with a native hourly day-ahead
  product (the coupled market trades 15-minute products since 2025-10-01).
* **Tariff** = day-ahead price + 0.08 EUR/kWh (grid fees, levies, tax; placeholder, band 0.05-0.12);
  mean 0.159 EUR/kWh, i.e. the 0.16 EUR/kWh of Befund 101. Gas price constant.
* **Split:** training and tuning on the 8 months Jan, Mar, Apr, Jun, Jul, Sep, Oct, Dec; **test on
  Feb, May, Aug, Nov** (one month per quarter, disjoint). Nothing of a test month is used for
  tuning, training or selection; the end value (below) is the mean price of the training months.
* **Price information** (R1, M2, RL): at hour `t` of day `D` the prices of `D` are known to its end;
  every later hour is forecast by the price of the same hour of `D` ("previous day as forecast" for
  `D + 1`). M1 sees the true prices of its horizon.
* **Protocol:** one run per month from an empty storage; the content left at the end is credited at
  `end_value` = mean tariff of the training months / COP = 0.0519 EUR/kWh (value of average pump
  heat). The same projection on the power and charge limits for every controller; cuts are counted.

## 4. Controllers

| label | controller |
|---|---|
| N0 | neither pump nor storage (reference) |
| P0 | pump without storage, whenever cheaper than gas (reference) |
| R0 | tuned constant rule of Befund 101: per hour the exact step programme `min c_U U + p_t Q - v (q_c - q_d)` with `v = kappa c_U`; `kappa` on the training months (grid 0..1, step 0.02) |
| R1 | schedule-aware rule: the hour is cheap when its price is at most the `q`-quantile of the price information of the next 24 h; cheap: `v = (1 - eps) c_U` (pump covers the demand and charges whenever cheaper than gas), else `v = eps c_U` (storage first, pump only for the rest, no charging); `q` on the training months (grid 0.05..0.95, step 0.05) |
| M1 | MPC with perfect prices over the horizon, 36 h (24 h reported), re-solved every hour, terminal value `end_value` |
| M2 | MPC as M1 with the price information of R1 |
| P* | hindsight: one programme per test month with the true prices and the end credit (the exact optimum of the protocol; reference) |
| SAC, TD3 | stable-baselines3, 3 seeds, 400 000 steps on episodes of one week inside one training month (random start content); observation: content, demand, waste heat, the 36 h price information of M2 (entry 0 = current price), hours to midnight, hour of the day and day of the week; action: pump heat and storage power (power limits hold by construction, the content limits by the projection, counted); reward = -cost/30 EUR; the final policy is evaluated |

Variants (rules and MPC only, one at a time): capacity 1 / 4 MWh, power limits 150 kW, pump 600 kW,
loss 0.5 %/h, adder 0.12 EUR/kWh.

Metrics on the test months: cost and CO2 relative to R0; gap to M1; violations of power and charge
limits; time per decision (in the run under parallel load; re-measured in one process at the end).

## 5. Pre-registered reading

In the docstring of `scripts/analyze_storage_price.py` (committed with this section, before the main
run). `gap(X, Y) = (J_X - J_Y) / J_Y` of the summed test-month costs; MPC horizon 36 h; RL = median
over the seeds of the final policies, study-level value = the algorithm with the lower median.

* **H0** if `gap(R1, M1) <= 2 %`: a simple price rule suffices; neither MPC nor RL justified.
* **H1** if `gap(R1, M2) > 2 %` and `gap(RL, M2) <= 1 %` and RL not behind R1: RL is an equivalent
  alternative to MPC (advantage only computing time / no model in operation).
* **H2** if `gap(RL, M2) > 1 %` or RL behind R1: RL not justified for this case.
* **none** otherwise (reported as not covered).

First that applies. Every condition is also reported on its own, per algorithm, with the seed
minimum and maximum, and per test month.

## 6. Decisions taken without asking (unattended run)

1. **Year 2024** instead of 2025 (native hourly product; 2025 would need an averaging rule for the
   15-minute products from October).
2. **Clock UTC+1 all year**: every day has 24 h, midnight is 00:00 CET; daylight saving is ignored.
3. **Test months Feb, May, Aug, Nov** (one per quarter) instead of a contiguous block, so that every
   season is in training and test; a contiguous out-of-time test was not run.
4. **System boundary = preheater**, placeholders of section 2; demand and waste heat follow the
   production plan and are known to every controller (no load-forecast error); the intake
   temperature is a deterministic climate normal and in the main case does not reach the
   controllable range (the waste heat covers the air below 30 degC in every hour).
5. **Tariff = day-ahead + constant adder** (0.08 EUR/kWh).
6. **Start empty, end credit and MPC terminal value = mean training tariff / COP.**
7. **R1 interpretation** of "pump and charge when the price is below the q-quantile of the next
   24 h": in the other hours the pump still runs for the direct demand when cheaper than gas, but
   the storage is used first and not charged; R1 uses the price information of M2 (a rule with the
   true next 24 h is not realistic).
8. **M2 forecast** after midnight = today's price of the same hour; M1's horizon may reach into the
   following month (true future prices, no tuning on it).
9. **Primary MPC horizon 36 h**, 24 h reported.
10. **RL observation** includes the clock (hour, day of the week, hours to midnight): M2 knows the
    production plan, so the agents get it through the clock; the price part is exactly M2's 36 h
    information.
11. **RL: final policy, no selection**; the learning curve is recorded on the training months
    (in-sample) and on the test months every 20 000 steps, neither is used for a choice.
    Hyperparameters: library defaults except those in the config (no tuning).
12. **Study-level RL value = better of SAC and TD3 by the median** (optimistic for RL, declared);
    both reported.
13. **Reading "none"** added for the combination not covered by H0-H2.
14. **Variants** one at a time around the main case, rules and MPC only.
15. **Rules use the exact step solver of Befund 101** (`heatpump.storage._case_table`/`_case_step`,
    private there, imported, not copied); the power limits enter as the bounds of `q_c`, `q_d`.
16. **Disclosure (development order).** The config (section 2-4) was written first. During the
    development, before this section and the pre-registration were committed, test-month numbers of
    rules and MPC were seen twice: a development check on May (untuned `kappa` and `q`), and the
    smoke test of the driver and of the analysis on all four test months (not an artifact). The
    reading in `scripts/analyze_storage_price.py` was written before the smoke test; its thresholds
    are those of the task; neither the config nor the reading was changed after these numbers were
    seen. The artifacts of section 7 are new runs of the committed scripts.
17. **Aborted first run `storage_price_v1` (commit 3931f9c, not an artifact).** It stopped after
    about 1.5 h with an IndexError in the training environment: the last episode start of December
    ended at the last hour of the year, and the observation after the last step (the bootstrap
    target) was read one hour beyond it. The same off-by-one meant that for episodes ending at the
    end of January, April, July or October the bootstrap target was the first hour of the following
    test month, i.e. it carried that day's true day-ahead prices -- information from the test
    period. Fix (commit after 3931f9c, test `test_env_episodes_stay_in_training_months` extended):
    the last start of a month is `episode_h` hours before its last hour, and the environment refuses
    a step whose next observation leaves the training months. Every job was re-run under the new
    name `storage_price_v2` from scratch (rules included); the v1 checkpoints and log are kept outside
    the repository (`C:/twinlink_runs/storage-price/storage_price_v1_aborted/`) and are not used.

## 7. Runs

* **`storage_price_v1`** (commit 3931f9c): aborted, not an artifact (section 6 item 17).
* **`storage_price_v2`**: driver `scripts/run_storage_price.py` on commit 1cc411e (`git/dirty=False`),
  6 workers, all 13 jobs (7 rule/MPC cases, SAC and TD3 x seeds 0, 1, 2), runtime 5384 s.
  Artifact `artifacts/runs/storage_price_v2.json.gz` (0.07 MB, commit 73f1586). Full suite on 3931f9c
  and on 1cc411e: 1330 passed, 30 skipped.
* **Budget:** probe of 15 000 steps per algorithm on 3931f9c (`local/probe/`, copied into the
  record), **measured under parallel load** (6 workers): SAC 0.0243 s/step, TD3 0.0130 s/step,
  estimated total 2.81 h <= 30 h, so **no cut** (both algorithms, 3 seeds). Actual training time per
  job 3500-3513 s (TD3) and 5042-5153 s (SAC).
* **Timing re-measured** with `scripts/measure_storage_price_timing.py` in one process, one thread,
  controllers one after the other, from the committed checkpoints (artifact
  `storage_price_v2_timing.json`, commit 9e66de2); the machine CPU load is sampled before and after
  every controller (0-29 %, other processes of the machine). Every controller reproduces its test cost
  of the run exactly (largest relative difference 0).
* **Analysis** `scripts/analyze_storage_price.py` (reading as committed in 3931f9c), artifact
  `storage_price_v2_analysis.json` (commit d421a05).
* Checkpoints (126 files, 458 622 747 bytes) in `artifacts/checkpoints/storage_price/storage_price_v2/`
  (gitignored), copied to `C:/twinlink_runs/storage-price/storage_price_v2/` (count and size checked).

## 8. Results (test months Feb, May, Aug, Nov; sums of the four month runs)

Tuning on the training months: `kappa` = 0.76, `q` = 0.40, neither at a grid edge.

| controller | cost EUR | vs R0 | CO2 vs R0 | gap to M1 | power / content violations | ms per decision, one process (median) |
|---|---|---|---|---|---|---|
| N0 | 77411 | +22.28 % | +54.05 % | +25.04 % | 0 / 0 | 0.0003 |
| P0 | 65327 | +3.19 % | +1.57 % | +5.52 % | 0 / 0 | 0.088 |
| R0 | 63308 | 0 | 0 | +2.26 % | 0 / 0 | 0.089 |
| **R1** | **62313** | **-1.57 %** | **-4.79 %** | **+0.65 %** | 0 / 0 | 0.088 |
| M1 36 h | 61908 | -2.21 % | -5.34 % | 0 | 0 / 0 | 1.64 |
| M2 36 h | 62020 | -2.03 % | -5.31 % | +0.18 % | 0 / 0 | 1.66 |
| M1 24 h | 61909 | -2.21 % | -5.34 % | +0.00 % | 0 / 0 | 1.43 |
| M2 24 h | 62035 | -2.01 % | -5.30 % | +0.20 % | 0 / 0 | 1.43 |
| P* | 61902 | -2.22 % | -5.34 % | -0.01 % | 0 / 0 | (one programme per month) |
| SAC, median [range] | 65287 [64760, 65694] | +3.13 % | +6.31 % | +5.46 % [+4.61, +6.11] | 0 / 1953, 1721, 2209 | 0.108-0.109 |
| TD3, median [range] | 65191 [65128, 65554] | +2.98 % | +4.81 % | +5.30 % [+5.20, +5.89] | 0 / 1639, 1984, 1598 | 0.078-0.079 |

Content violations of RL = hours (of 2904) in which the action would have over- or under-run the
storage content and was cut by the projection (power limits hold by construction). In the run under
parallel load the medians were R0 0.186, R1 0.201, M1 36 h 2.94, M2 36 h 2.85, SAC 0.110, TD3 0.117 ms
(`decide_ms_median_under_load` in the analysis; marked there as measured under parallel load).

**Pre-registered reading: H0.** `gap(R1, M1)` = +0.65 % <= 2 %: the schedule-aware price rule
suffices; neither MPC nor RL is justified for this case. Every condition on its own: C0 holds;
C1a fails (`gap(R1, M2)` = +0.47 %, not > 2 %); C1b fails (`gap(RL, M2)` = +5.11 %, RL = TD3); C2
holds (RL is +5.11 % behind M2 and +4.62 % behind R1). The reading is the same for SAC and TD3, for
the seed minimum and maximum of each, and **in every test month**:

| month | gap(R1, M1) | gap(R1, M2) | gap(RL, M2) | gap(RL, R1) | reading |
|---|---|---|---|---|---|
| Feb | +0.62 % | +0.49 % | +5.38 % | +4.87 % | H0 |
| May | +0.88 % | +0.74 % | +7.37 % | +6.59 % | H0 |
| Aug | +0.87 % | +0.61 % | +6.29 % | +5.65 % | H0 |
| Nov | +0.31 % | +0.12 % | +2.29 % | +2.17 % | H0 |

* **MPC.** M1 is +0.010 % above the hindsight optimum P*; the 24 h horizon gives the same cost as
  36 h to within 15 EUR (M1 61909 / 61908, M2 62035 / 62020 EUR). The realistic forecast costs M2 +0.18 % (36 h) against M1. R0 (constant rule, Befund 101)
  is +2.26 % above M1: the power limits and the hourly price leave a gap for the constant rule that
  the schedule-aware rule closes to +0.65 %.
* **RL.** Both algorithms end above R0 (+2.98 % / +3.13 % in the median) and close to the pump
  without storage P0 (65327 EUR against medians 65191 and 65287 EUR); every seed of both algorithms
  is behind R0. RL is also behind on its own training months (in-sample cost of the final policies
  128615-130952 EUR against R0 125994 and R1 124142 EUR), so this is an optimisation gap, not a
  generalisation gap. The learning curves still fall slowly at the end (e.g. SAC seed 0 test
  65896 EUR at 300 000 and 65694 EUR at 400 000 steps); the budget of 400 000 steps was fixed in the
  config before the run. The raw RL actions hit the content limits in 1598-2209 of the 2904 test hours.
* **CO2** follows cost (CO2 price in `c_U`): R1 -4.79 %, M1 -5.34 % against R0; RL +4.81 % / +6.31 %.
* **Variants** (rules and MPC): C0 holds in every variant; `gap(R1, M1)` ranges from +0.22 %
  (adder 0.12 EUR/kWh) to +1.33 % (capacity 1 MWh), `gap(R0, M1)` from +0.53 % to +2.26 %.

## 9. Caveats

* **Scenario on placeholders** (section 2): one plant structure, one year of prices (2024), a
  constant tariff adder, a deterministic production plan without load-forecast error. A larger gap of
  R1 needs a case where shifting pays more per kWh than here (capacity 1 MWh: +1.33 %).
* **H0 is decided by rules and MPC alone**, and these are deterministic: the smoke test before the
  run (section 6 item 16) produced the same numbers, i.e. C0 was visible before the main run. The
  reading and its thresholds were committed before that smoke test and were not changed.
* **RL is not tuned** (library defaults, one reward scale, one network size, 400 000 steps, final
  policy). The result says that RL with this budget does not reach the rules, not that it cannot;
  under H0 there is no gap above R1 larger than +0.65 % for any controller to win.
* **Information:** R1, M2 and RL get the same price information (known until midnight, then today's
  prices); the real day-ahead auction publishes tomorrow's prices around midday, which this study does
  not use (it would favour M2 and RL over persistence, not R1 over M1).
* **Timing:** one-process values on a shared machine (CPU load sampled 0-29 %); per-decision time
  excludes the hindsight programme of P*, which is not an online controller. All controllers decide
  within a few milliseconds, far below the hourly step.
* **Aborted run v1** (section 6 item 17) not used; its checkpoints are kept only for traceability.
