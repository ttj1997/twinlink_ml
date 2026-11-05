# Campaign network 2: does a better simple rule or a rolling planning close the gap to K3?

Status: design (sections 1-4) fixed before the run; results after the run. Lab book: "Befund NN
(number open)" in `docs/04_stand_ap1.md` (the number is assigned by the managing session). Basis:
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
