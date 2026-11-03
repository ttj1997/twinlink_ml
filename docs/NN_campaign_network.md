# Campaign network: does a heat network pay when the consumers do not run all the time?

Status: design (sections 1-5) fixed before the run; results in sections 6-9 after the run. Lab
book: "Befund NN (number open)" in `docs/04_stand_ap1.md` (the number is assigned by the managing
session). Code `src/twinlink_ml/campaign_network/`, settings
`conf/campaign_network/campaign_network.yaml`, tests `tests/test_campaign_network.py`, driver
`scripts/run_campaign_network.py`, reading `scripts/analyze_campaign_network.py` (pre-registered
in its docstring). **Scenario band on placeholders, not a plant statement; every number carries its
scenario.** No RL.

## 1. Question and step 0

Does a heat network pay when the process consumers have campaigns and standstills? Earlier
network, storage and site studies were checked first (step 0 of the task):

- Befunde 60-81: steady state (pinch target, two-line coupling, M1-M4); no time axis, no
  standstills.
- Befund 99 (storage, time-slice pinch) and Befund 101 (heat pump, same streams): deterministic
  modes `shift` and `weekly` (`storage_timeslice/profiles.py`): lines C1/C2 and process water off
  at night or at weekends, BUT the compressors drop to 30 % at the same time (source and
  consumers coupled), plant A runs 24/7 in every mode, and the question is storage against ideal
  integration, not network against a fixed assignment.
- Befund 100 (site pinch): 12 representative days with continuous operation; "today" (fixed
  groups) against the target resembles V_N, but without standstills.
- Befund 110: steady six-case network of the partner model.

None modelled campaign availability or independent against joint standstills of the consumers
with this question; the study was not stopped.

## 2. Model

One central waste-heat source: compressor station, constant 24/7, water from the supply
temperature (80 degC, variant 70 degC) down to a return floor of 50 degC; source power P = 400 /
600 / 800 / 1200 kW (expansion path). Consumers (cold streams, hourly):

| consumer | temperatures | size | operation |
| --- | --- | --- | --- |
| process air A, B, C | ambient -> supply - approach | 20 000 / 15 000 / 10 000 Nm3/h | campaigns, availability a |
| process water | 15 -> 60 degC | 200 kW | constant |
| hall heating | 35 -> 50 degC | 200 kW peak at -10 degC, limit 15 degC, day factor 0.4 | seasonal |

Every value is a placeholder with its source in the YAML (mostly the site data set of Befund 100,
which is only read). Exchangers counter-current, minimum approach 10 K (variant 5 K, reference
Befund 114). Pump electricity 1.5 % of the delivered heat (site data set), charged on every
configuration. Gas replaced = heat / 0.90; CO2 0.201 kg/kWh gas, 0.366 kg/kWh electricity.
Climate: monthly means of the site data set plus a daily cosine (4 K), deterministic.

Campaigns: alternating on/off renewal process, 1 h step, one year after one year of burn-in;
campaign length uniform 1-14 d, standstill uniform in 1-14 d x (1 - a) / a (expected availability
exactly a); a in {1.0, 0.8, 0.57, 0.4}; plants independent or jointly standing still (all share
plant A's schedule); 5 draws per combination, the same draws for every P, configuration and
variant (paired).

## 3. Configurations

- **K0** no network: the source water passes plant A's preheater only (fixed assignment of today,
  Befund 100); the rest goes to the cooling tower, also while A stands still.
- **K1** network with the cascade rule of practice (fixed before the run, docstring of
  `campaign_network/cascade.py`): supply water first to the stage with the highest required
  temperature (process-air preheaters, parallel branches A/B/C with the flow split by air flow),
  the return cascades to the next lower stage (process water 60 degC, then hall 50 degC); ties by
  the gas replaced per kWh; a consumer that stands still takes nothing, the next stage moves up.
  The task's first wording "greedy, highest gas replacement first" and the addendum "highest
  required temperature first" coincide here: every replaced supply has the same efficiency 0.90,
  so the gas order adds nothing to the temperature order.
- **K2** K1 + a stratified hot-water tank on the loop (0 / 2 / 8 / 24 h of P, loss 0.1 %/h,
  discharge at most P) with a rule without foresight: discharge the smallest amount that reaches
  every stage duty of the largest possible discharge; otherwise charge the largest part of the
  source water whose removal lowers no stage duty.
- **K3** LP with full foresight over the year (heat cascade of the problem table, same storage):
  an ideal-network upper bound (any split, any match), so K3 >= K2 >= K1 >= K0.

## 4. Key figures and pre-registered reading

Use of the waste heat (share of P x 8760 h), gas replaced, CO2 (gas, pump, net);
`V_N = (use K1 - use K0) / source energy`; gap of K1 (S = 0) and K2 (S > 0) to K3 in net CO2.
Readings (docstring of `scripts/analyze_campaign_network.py`, committed before the run), per P and
mode on the median over the draws: **N0** V_N < 2 % at every a >= 0.57; **N1** V_N >= 5 % at
a = 0.57; otherwise between. Rule: **R0** (simple rule suffices) if every median gap <= 1 %,
otherwise **R1** with the gaps reported. Saturation: first P with marginal use < 0.25 (grid), plus
a descriptive sweep 100-3000 kW at a = 1 (reference Befund 100).

## 5. Checks

Tests: K3 >= K2 >= K1 >= K0 on every draw (three variants, three availability/mode pairs, two
powers, every storage size); a = 1 and P within plant A's own demand -> V_N = 0; exchanger
approach, floor and balance on random cases; schedule availability and durations; joint mode
shares plant A's schedule; K2 without storage = K1; storage and LP by hand (two hours, with and
without loss); pump and CO2 bookkeeping. The run repeats the order check on every draw and records
every LP status.
