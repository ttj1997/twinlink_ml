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

## 6. Run

Script commit `d2ebbf9` (model, settings, tests, driver and reading committed together before the
run; full suite 1418 passed, 30 skipped, 1 xfailed), clean tree, `git/dirty=False`. Artifact
`artifacts/runs/campaign_network_v1.json.gz` (0.49 MB, committed as `2f9124d`), reading
`artifacts/runs/campaign_network_v1_analysis.json`. 96 jobs, 6 workers, 403 s wall-clock; the
probe estimated 0.14 h, far below the 30 h budget, so nothing was cut (storage 0/2/8/24 h, 5
draws). Checks: 0 order violations (K3 >= K2 >= K1 >= K0 in heat used and net CO2 in every draw,
variant and cell), every LP optimal. Realised availability per plant at a = 0.57: plant A median
0.545 [0.52, 0.609] (about 25 campaigns per year, so a draw deviates by several points).

## 7. Results (main variant: approach 10 K, supply 80 degC)

**V_N, median [min, max] over 5 draws, %** (a = 1 is one schedule):

| P (kW) | mode | a = 1.0 | 0.8 | 0.57 | 0.4 |
| --- | --- | --- | --- | --- | --- |
| 400 | independent | 1.60 | 19.80 [19.56, 20.35] | 42.08 [36.95, 45.81] | 55.00 [46.75, 57.37] |
| 400 | joint | 1.60 | 12.34 [12.16, 12.52] | 27.60 [23.68, 28.97] | 38.24 [33.52, 40.28] |
| 600 | independent | 27.53 | 39.43 | 49.61 | 55.21 |
| 600 | joint | 27.53 | 29.52 | 32.74 | 35.53 |
| 800 | independent | 45.65 | 50.66 | 52.75 | 49.57 |
| 800 | joint | 45.65 | 42.40 | 38.19 | 36.07 |
| 1200 | independent | 56.44 | 50.50 | 43.78 | 37.08 |
| 1200 | joint | 56.44 | 49.45 | 39.66 | 33.32 |

**Reading: N1 in every (P, mode)**, in every draw (min and max give the same letter), and in both
sensitivity variants. Overall N1: distribution pays with consumers that stand still. Two effects
are mixed in V_N (declared split, `V_N(a) - V_N(1)`, median at a = 0.57):

- At today's 400 kW and continuous operation the network adds only 1.60 % (below the N0 limit):
  plant A alone takes almost the whole source (K0 uses 98.4 %). Everything above that at 400 kW is
  due to standstills: +40.48 points (independent) and +26.01 points (joint) at a = 0.57.
- From 600 kW on, plant A alone cannot take the source even at a = 1 (V_N 27.53 / 45.65 / 56.44 %
  at 600 / 800 / 1200 kW): the network pays by serving further consumers, as in Befund 100. The
  standstill part shrinks with P and turns negative at 800 kW (joint) and 1200 kW (both modes),
  because then all consumers together have less demand than the source.
- Joint standstills (common plant holiday) lower the benefit at 400 kW by about a third compared
  with independent ones (27.60 against 42.08 %): while A, B and C stand still together only process water and hall
  heating take heat.

Utilisation of the waste heat (median, %), K0 / K1 / K3 0 h / K3 24 h, at a = 0.57: 400 kW
independent 53.7 / 96.9 / 97.2 / 99.2, joint 53.7 / 81.3 / 81.3 / 88.9; 800 kW independent
29.7 / 82.8 / 84.3 / 89.9, joint 29.7 / 67.9 / 67.9 / 75.5; 1200 kW independent
19.8 / 63.6 / 65.5 / 66.1. Net CO2 saving (t/a, gas minus pump electricity, median): 400 kW,
a = 0.57, independent: K0 410, K1 740, K3 24 h 757; joint: K1 620, K3 24 h 679; 1200 kW, a = 1:
K0 830, K1 2122, K3 24 h 2203.

## 8. Rule against foresight; saturation

**Reading R1** (the gap is reported; not "a simple rule suffices" by the pre-registered 1 %
limit): 67 of the 128 cells have a median gap of the net CO2 saving above 1 %; the largest is
7.30 % (independent, 800 kW, a = 0.57, storage 24 h). Two sources:

- **Cascade order without storage (K1 against K3 0 h):** with joint standstills 0.00 % at every
  P <= 800 kW; with independent standstills up to 2.25 % (800 kW, a = 0.4); at 1200 kW 2.43-3.15 %
  in both modes. The rule heats the process air first and gives the process water less than the
  bound does (draw 0, 800 kW, a = 0.57, independent: 1093 against 1733 MWh/a to process water).
- **Storage rule (K2 against K3, same tank):** the gap grows with the tank size. Mechanism (draw
  0, 800 kW, a = 0.57, independent): K2 charges 317 MWh/a, but the heat used rises by only 36
  MWh/a over K1, because the rule discharges whenever any stage duty rises and the extra supply
  water returns largely unused (warm) to the cooling tower; K3 with the same tank charges 426
  MWh/a and uses 389 MWh/a more than K3 without storage. With joint standstills at 400 kW the rule
  works (charged 264, used +231 MWh/a; K3 245 / +235) and the gaps stay at 0.00-1.36 %.

So a learning or predictive controller is not justified by this result as such: most of the gap
is a weakness of the specific storage rule fixed before the run (discharge although the return
stays above the floor), which a better rule without foresight may close; this was not tested
(it would be a declared post-hoc variant).

**Saturation.** On the grid (marginal use < 0.25 between neighbouring levels) the consumers do
not saturate up to 1200 kW at a >= 0.8 (K1 marginal 1.0 / 1.0 / 0.78 at a = 1); with standstills
they saturate from 1200 kW (independent, a = 0.4, every configuration; independent, a = 0.57,
K3 24 h). Fine sweep at a = 1 (descriptive): K1 uses 100 % up to 800 kW, 99.3 % at 1000 kW,
92.7 % at 1200 kW and saturates from 1400 kW (K3 0 h the same); the consumers can take at most
10 626 MWh/a (K1) of a demand of 10 639 MWh/a. Befund 100 found saturation between 800 and
1200 kW without further sinks; here the placeholder consumers B and C (25 000 Nm3/h together) and
the smaller approach (10 instead of 20 K) move it to about 1400 kW. The saturation point is a
property of the assumed consumers, not of the source.

## 9. Sensitivity, caveats, questions

**Sensitivity.** approach_5: N1 everywhere, R1 (largest gap 5.33 %, 48 cells above 1 %), sweep
saturation from 1500 kW. supply_70: N1 everywhere, R1 (largest gap 9.82 %, 83 cells above 1 %),
sweep saturation from 1200 kW (K1) / 1300 kW (K3).

**Caveats.** (1) Placeholders throughout; the letter N1 is carried by consumers that run while
plant A stands still -- above all the constant 200 kW of process water (24/7) and, with
independent standstills, plants B and C. Without such consumers V_N would fall back to the
sharing among A, B, C; this was not varied (no pre-registered variant). (2) V_N measures heat
used against a fixed assignment to plant A; it says nothing about the cost of the network
(pipes, distances, pressure drops, part load), which decide whether it pays in money. (3) K3 is
an ideal-network bound (problem-table cascade), not a design. (4) The storage stores supply water
only; a tank for warm return water was not modelled. (5) Campaign lengths 1-14 d, independent
renewal processes, no correlation with season or with the hall heating; the climate is
deterministic. (6) Process-air preheating is limited by the water (supply - approach); the main
heater and the dryer are outside the system boundary.

**Questions for the partner.** Real schedules (campaign lengths, standstills, common shutdowns)
of the plants that could join the network; process-water and hall demand over the year;
compressor-water temperatures; which consumers could be connected at all.
