# Campaign network 2: does a better simple rule or a rolling planning close the gap to K3?

Status: design (sections 1-4) fixed before the run; results after the run. Lab book: Befund 119
in `docs/04_stand_ap1.md` (document number 36). Basis:
Befund 118, `docs/35_campaign_network.md`, artifact `campaign_network_v2` (model, configuration and
schedule draws unchanged, paired). Code `src/twinlink_ml/campaign_network/dynamic.py` (K0b, K1d) and
`planning.py` (P1, P2), tests `tests/test_campaign_network_2.py`, driver
`scripts/run_campaign_network_2.py`, reading in the docstring of
`scripts/analyze_campaign_network_2.py`. **Scenario band on placeholders, not a plant statement.**
Rules and LP only, no training, no RL.

## 1. Question

Befund 118 left the simple rules up to 6.30 % (K2b) below the bound with foresight K3 (reading RB1),
about half from the cascade order and half from the storage. The project lead named a rolling
dispatch planning on the production plan as the appropriate alternative (docs/12, item 26). Does a
better simple rule, or such a planning, close the gap -- also when the plan deviates?

## 2. Configurations

Main variant (approach 10 K, supply 80 degC), process water 0 / 200 kW / seasonal (levels of the
post-hoc extension of Befund 118), source power 400 / 600 / 800 / 1200 kW, availability
1.0 / 0.8 / 0.57 / 0.4, independent or joint standstills, the same 5 draws, storage 0 / 2 / 8 / 24 h.

- **K0b** fair baseline without a network: the source water in fixed shares to the preheaters of A,
  B, C (shares by expected mean demand = air-flow shares 20 : 15 : 10), surplus of each preheater to
  the cooling tower; hall and process water get nothing. While all plants run, K0b equals the
  process-air stage of K1 (tested).
- **K1d** dynamic cascade: the stages of K1 ordered in every hour by the gas each could replace now
  (its duty with the full source water at the top / efficiency), storage with the logic of K2b.
- **P1** rolling dispatch planning: LP over 7 days with the exactly known standstill plan, executed
  until the next plan; a new plan every 24 h and at once when the observed state differs from the
  plan (decision of the project lead, 2026-11-04). No value of the stored heat at the window end.
- **P2** as P1, with a plan whose every change (start and end of a campaign) is shifted by
  U[-24, +24] h (fixed per draw, order kept; joint mode: one plan for all plants). Correction: a
  change that happened is taken at its true time; a planned change that is overdue is expected in
  the next hour. This is a **naive adjustment of the plan**; P2 is therefore a lower bound for
  planning with deviations.
- **K3** the hindsight LP of Befund 118 (bound); **K1** and **K2b** for comparison.

Planning model: every consumer has the same net CO2 weight per kWh (same efficiency 0.90, same pump
share), so the hourly allocation reduces to the min-cut of the problem-table cascade
(`planning.py`); over the whole year this LP gives the optimum of K3 (tested: the rolling plan with
a horizon to the end of the year reproduces K3, with hourly and with daily planning). The heat used
in an hour is always computed with the TRUE demand at the executed storage flow.

## 3. Pre-registered reading

Gap of the net CO2 saving to K3 with the same storage, median over the draws per cell. **Q0**: K1d
<= 1 % below K3 in every cell (S = 0 / 2 / 8 / 24 h) -> a simple, well-chosen rule suffices. **Q1**:
not Q0, but P2 <= 1 % below K3 in every cell with storage -> planning with the production plan is
justified also with deviations; RL is not needed. **Q2**: otherwise gap and cells. Without reading:
the network benefit against the fair baseline, `V_N_fair = (use K1 - use K0b) / source energy`.

## 4. Checks and budget

Tests: K3 >= every configuration in every draw (K3 >= P1 >= P2 not assumed); P1 with a horizon to the
end of the year reproduces K3; P1 with an exact plan plans once a day; plan shifts and the naive
correction; K0b = K1 air stage; reduced model = K3 without storage; storage balance of K1d. Run:
the same order check on every draw, LP statuses, control against `campaign_network_v2` (K0, K1,
K2b, K3 identical), and the check hourly against daily planning on one draw (seen before the run:
daily planning 0.21 % below hourly in heat used on that draw). Budget 30 h wall-clock; cut order
storage 2 h, then 3 draws, applied also if the run would end after the free machine time (decision of 2026-11-05); at most 8 workers; times measured under parallel load.

## 5. Run

Script commit `680f160` (model, tests, driver and reading committed together before the run; full
suite 1440 passed, 30 skipped, 1 xfailed), clean tree, `git/dirty=False`. Artifact
`artifacts/runs/campaign_network_2_v1.json.gz` (0.78 MB, committed as `e608f06`), reading
`campaign_network_2_v1_analysis`. Probe (8 workers, measured under parallel load): 0.9 h expected
for the full grid, below 30 h and before the end of the free machine time, so nothing was cut
(storage 0 / 2 / 8 / 24 h, 5 draws). The run took 31.7 min with 8 workers (measured under parallel
load). Checks: no configuration above K3 in any draw (heat used and net CO2), every LP optimal, no
failed planning LP; K0, K1, K2b and K3 identical to `campaign_network_v2` (6696 values, largest
difference 0.0). Price of planning once a day instead of every hour (exact plan, one draw: water
200 kW, a = 0.57, independent, 800 kW, 24 h): 0.21 % less heat used.

## 6. Results

**Reading: Q2.** Neither K1d nor P2 stays within 1 % of K3 in every cell. Median gap of the net CO2
saving to K3 (cell medians over the draws):

| configuration | cells | cells above 1 % | largest | median of the cells |
| --- | --- | --- | --- | --- |
| K1 (S = 0) | 96 | 26 | 3.15 % | 0.06 % |
| K1d (S = 0 / 2 / 8 / 24 h) | 384 | 151 | 6.30 % | 0.41 % |
| K2b (S > 0) | 288 | 123 | 5.74 % | 0.61 % |
| **P1** exact plan (S > 0) | 288 | **0** | **0.98 %** | 0.00 % |
| **P2** plan +-24 h, naive correction (S > 0) | 288 | **67** | **3.29 %** | 0.37 % |

- **The dynamic cascade does not help.** K1d is never better than K2b with the same storage (worse
  in 41 of 288 cells, by at most 0.58 points) and, without storage, never better than K1 (worse in
  14 of 96 cells, by at most 0.33 points). With equal efficiencies "most replaceable gas first"
  means "largest current demand first", which is no better than the temperature order.
- **With an exact plan the planning closes the dispatch gap:** P1 stays within 0.98 % of K3 in
  every cell (largest: process water 200 kW, independent, 600 kW, a = 0.4, 24 h).
- **With plan deviations of up to 24 h and the naive correction it does not everywhere:** P2 is
  above 1 % in 67 of 288 cells, largest 3.29 % (no process water, joint standstills, 600 kW,
  a = 0.4, 24 h). The cells above 1 % are mostly joint standstills (54 of 67) and the larger tanks
  (24 h: 42, 8 h: 25, 2 h: 0); without process water 25, with 200 kW 18, seasonal 24. P2 - P1 in
  the gap: median 0.35 points, at most 3.29 points. P2 is a lower bound for planning with
  deviations (naive correction: an overdue change is expected in the next hour).
- **What the gaps mean (not like for like).** P1 and P2 use in every hour the heat allocation of
  the ideal network of K3 (min-cut of the cascade); their gap to K3 measures only the storage
  dispatch. The rules K1, K1d and K2b work with the physical series cascade and carry its cascade
  part (K1 against K3 without storage: up to 3.15 % at 1200 kW, median 0.06 %). A planning of the
  storage on top of the physical cascade rule was not computed; so the result says that the
  storage part of the gap is closed by planning with an exact plan, not that a real network
  reaches K3.

**Network benefit against the fair baseline K0b** (no reading; `V_N_fair = (use K1 - use K0b) /
source energy`, median [range] over the draws, 400 kW): at a = 1 it is 0.00 % at every level (A, B
and C together take the whole 400 kW, also in summer; V_N against K0 was 1.60 %). At a = 0.57:
process water 200 kW 39.57 % [37.07, 41.72] independent, 26.73 % [22.77, 28.16] joint; seasonal
35.48 / 17.70 %; no process water 30.82 / 4.00 %. Against the fair baseline the network benefit at
400 kW therefore comes entirely from the standstills; with 200 kW process water it is close to V_N
against K0 of Befund 118 (42.08 / 27.60 %). With 200 kW process water and a = 0.57 it falls with
the source power (independent 39.57 / 33.99 / 25.59 / 17.08 % at 400 / 600 / 800 / 1200 kW),
because the fixed shares of K0b also take more of a larger source.

## 7. Caveats

(1) Placeholders throughout; a scenario band, not a plant statement. (2) P1 and P2 use the ideal
hourly heat allocation (see above); a plan for the storage combined with the physical cascade was
not computed. (3) P2: plan deviations uniform in +-24 h, every change independent, naive
correction; a better correction (e.g. a statistical expectation of the delay) may do better, so P2
is a lower bound. (4) Rolling horizon 7 d without a value of the stored heat at the window end;
planning once a day costs 0.21 % against every hour on the one draw checked. (5) Equal efficiencies
of every replaced supply are required by the reduced planning model (checked in the code); with
different efficiencies the planning model would need the full interval LP. (6) No RL and no
training were part of the study; nothing here argues for RL: the gap that remains with an exact
plan is below 1 %, and the gap with deviations depends on how the plan is corrected.
