# Heat-network module: specification for the future Dyssol balance module

> Status: 18.09.2026 · version 5 (pinch target/problem table algorithm added as the yardstick for
> the six interconnection cases; pinch added as an operating-point-dependent quantity,
> `lp-benchmark` branch, Befund 60 and 61).
>
> This file describes permanent matters (module purpose, the three corrections to the
> template, the scale caveat), not a measurement result with a point in time -- hence a separate
> document analogous to `02_prozess_setup.md`, not a further entry in
> `04_stand_ap1.md`. The caveats that concern Befund 52/57 (there: gross consumption without or with
> recovery credit) remain in the finding log -- see the reference below and
> `04_stand_ap1.md`, Befund 52d/57.

## Purpose

The grey-box simulator (`simulation/greybox.py`) delivers, via `simulation/exhaust.py`,
the exhaust air state of the process, but so far no model of heat recovery. The
optimization so far (Befund 52, `04_stand_ap1.md`) therefore minimizes only the
gross consumption (`power_total`) -- not the net consumption after recovery, which is the
actual project goal (Befund 52d already names this as one of four
caveats: "Ein Prozess, kein Verbund").

`heatnetwork/` closes this gap: from exhaust air states (of one or several sources)
and the process air sinks that have to be heated, it computes the
recoverable power and the resulting fuel demand. This is needed for

- the future net objective function (successor of Befund 52, see below),
- a later LP benchmark,
- an RL environment (`routing.MatrixRouting` is already intended as its action
  space),
- the interface to the consortium partner.

**This module is the specification for the future Dyssol balance module
(AP3 of the software integration partner).** It ports an existing Excel balance of the consortium partner (two
dryer lines, two heat exchangers, two air heaters) 1:1 in its calculation rules,
but in doing so corrects three defects of the template (see below) -- a future
Dyssol module that reads the same template must not inherit these defects just because it
has the same source.

## The three corrections to the Excel template

In detail (with the exact formulas) in the module docstring of
`src/twinlink_ml/heatnetwork/balance.py`. In summary:

1. **Dew-point reference per source.** The template has the second heat exchanger compute against the
   dew point of the FIRST line -- a copy error between two
   worksheets, not a modelling decision. Every source in the module has its own
   dew-point limit, computed from its own exhaust air humidity.

2. **Cascade the residual enthalpy instead of discarding it.** The template caps the
   sink temperature at its set point and discards the unused
   source enthalpy -- documented in a case of 891.8 kW at 87.4 °C, far
   above any dew-point limit and thus still usable. In the module every source is a
   stream with a temperature decreasing along the cascade: unused enthalpy is available to the
   next stage (possibly a different sink) until the dew-point limit
   is reached. `tests/test_heatnetwork_balance.py::
   test_cascading_beats_the_uncorrected_excel_template` recomputes the uncorrected
   template and shows that the cascading needs strictly less fuel --
   the number that goes to the consortium partner.

3. **Minimum approach temperature per stage.** The template nowhere checks
   `T_Luft,aus <= T_Quelle - dT_min`; a 120 °C cap there prevents a
   violation only by chance. In the module the approach temperature is a hard constraint of every
   stage, with the cascade order low-temperature sources first (otherwise the
   sink air is already too warm to still take up a cold source).

## Condensation mode (MA1, Befund 57)

Occasion: the consortium partner extends its heat model with a CONDENSING
heat exchanger (Model v3) instead of the purely sensible operation assumed so far. This
reverses the meaning of the exhaust air loading `Y_out` -- without condensation, moisture only
raises the dew point (a nuisance, narrows the sensible cooling limit); with condensation it supplies
additional latent heat (a resource). Verified at the nominal point of the simulator (Befund 57,
MA2): a `+10 %` disturbance of the product moisture lowers `Q_rec` in the sensible mode by
1.60 % (-4.95 kW, site-network mass flow), raises it in the condensing mode by 2.73 %
(+14.20 kW) -- the same sign change that a rough estimate at the operating point of the
consortium partner (8 kg/s, Y=0.01998, T_out=81.4 °C) had already suggested before the
implementation. **Restriction (addendum NB1, Befund 57h): holds only as long as the line
actually condenses.** With the 15 °C cold side it does; with the more realistic
cold side of ~29 °C (see below) it no longer does -- then, for exactly this operating point,
the sensible view applies again (moisture LOWERS `Q_rec`).

`HeatNetworkConfig.recovery_mode` (`"sensible"` default/bit-identical to the state before MA1,
`"condensing"` new) selects the mode -- a config field, not a function parameter, so that the
selected mode is documented in the `RunRecord`. In the condensing mode the source cools
down to `cold_temperature_celsius` (`T_kalt`, not to `Taupunkt + dT_approach`) and supplies
in addition the latent heat of the condensate produced in the process
(`Kondensat = max(0, Y_out - Y_s(T_kalt)) · ṁ_L`, `Q_latent = Kondensat · Δh_v(T_kalt)`).
Formulas, cascade integration (latent reserve per source, tapped only after the sensible reserve)
and the three MA1 tests (sensible bit-identical, Q_rec monotonicity, continuous fallback
above the dew point, mass balance of the condensate) in the module docstring of
`heatnetwork/balance.py`.

**`T_kalt` -- placeholder, OPTIMISTIC, with an open question like `dT_min`.** Default:
**15 °C** (`heatnetwork.config.DEFAULT_COLD_TEMPERATURE_CELSIUS`, unchanged -- this
addendum does not change the code default, only its classification in the documentation).
Rationale of the original value: plausible value for a cold-water circuit
(recooling plant or slightly chilled cooling water), consistent with our own rough estimate at the
operating point of the consortium partner, which uses this value as an anchor point -- neither
a measurement nor a statement by the consortium partner.

**Addendum NB1 (Befund 57h): 15 °C is too low to be reachable without a chiller.**
15 °C corresponds to the model ambient temperature -- the cold side would thus have a
driving temperature difference of zero. The control system of the application partner states
**29.1 °C cooling-water temperature** for the compressor station, a real reference point for the
actual cold side. `Q_rec` falls between 15 °C and 29.1 °C from 5893.2 kW to
4502.2 kW (sum of both dryer lines, spray tower + fluidized bed, MA5 values); from
`T_kalt≈25 °C` (above the dew point of the fluidized-bed line, 24.92 °C) onwards, the
fluidized-bed line no longer condenses at all -- for it, the purely sensible
view applies again. Details and the full sweep table in Befund 57h. The code default remains
15 °C (no unjustified change without an explicit task), but every statement that reads the
default as a realistic operating point is to be corrected -- 15 °C is the
FAVOURABLE case, not the expected one.

**Open question (unchanged):** which temperature level the actual cold-water circuit
runs at in the partner model, in particular whether it coincides with the
hot-water storage intermediate circuit confirmed in Befund 56 (`B-3236`, for `dT_min`) or is a separate circuit
-- the 29.1 °C cooling-water temperature of the compressor station is a reference point, not a
confirmed answer for THIS circuit. From the reconstruction of the six-case model (MA5,
Befund 57, `Model_v3.xlsx`) it is also known that the two dryer lines there run
DIFFERENT `T_kalt` (spray-tower line 10 °C, fluidized-bed line 15 °C) --
`HeatNetworkConfig.cold_temperature_celsius` is network-wide, not per source; a
multi-source calculation with a different `T_kalt` per source (as needed in MA5)
therefore has to be built OUTSIDE `evaluate_cascade` (see
`scripts/probe_six_case_reconciliation.py`, deliberately not generalized into `heatnetwork/`
there, so as not to complicate the network-wide configuration as long as only
one study needs it). Enters the fingerprint, triggers the same
`PlaceholderConfigWarning` as `dT_min`.

## `dT_min` -- placeholder, with an open question to the consortium partner

Default: **20 K** (`heatnetwork.config.DEFAULT_MINIMUM_PINCH_K`). Rationale: the
consortium partner models the intermediate circuit between exhaust air and process air as a black
box -- presumably it contains two exchanger stages in series (exhaust air → water,
water → process air). The effective approach temperature over the chain is approximately the
sum of the single-stage approach temperatures; 10 K is a customary guide value for gas/gas exchangers
for ONE stage, 20 K assumes two stages. In addition: an approach temperature set too small
overestimates the recoverable power -- the conservative value
is the more honest error in this direction.

**Open question:** Does the intermediate circuit really consist of two stages, or is
exhaust air transferred directly against process air? With direct transfer, 10-15 K would be
more appropriate. As with `critical_moisture_kg_per_kg` (Befund 54, `04_stand_ap1.md`),
the next step is a sensitivity calculation over 5/10/15/20/25 K -- **not part
of this task**, but noted: if the result is robust over the plausible range,
the query to the consortium partner becomes unnecessary, as already with
`critical_moisture_kg_per_kg`.

`dT_min` is a mandatory parameter with a default (`HeatNetworkConfig.minimum_pinch_k`), enters
the fingerprint, and like every other placeholder in the project triggers a
`PlaceholderConfigWarning` (`HeatNetworkConfig.warn_if_placeholder`).

**Partial answer to the open question (Befund 56, `04_stand_ap1.md`).** Two
control-system screenshots of the application partner (heat recovery, plant F-5) show the
intermediate circuit concretely: compressors (waste-heat source) → heat exchanger `W 3237` →
hot-water storage `B-3236` (three temperature sensors, 57-69 °C there depending on the time) →
heat exchanger `W 3223` (process air preheating). This directly confirms two things from the above
assumption: the circuit really is **two-stage** (two exchangers in series, not
direct transfer), and a **buffer storage really exists** (`B-3236`) -- supports the
choice `dT_min = 20 K` as the sum of two single-stage approach temperatures. Not a complete
answer: the screenshots show the compressor waste heat as the source, not explicitly the
granulator exhaust air itself -- whether and how both sources feed the same buffer circuit is thus
not clarified and remains the actual open query to the consortium partner.

## Scale caveat

The site-network model computes with about 8 kg/s of process air
(`evaluation.recoverable_power.DISTRICT_MASS_FLOW_KG_PER_S`), the lab simulator with about
0.03 kg/s (`dry_air_mass_flow_kg_per_s` in `data.trajectory.diagnostics`) -- two
orders of magnitude. **Every kW number from coupling lab sources with site-network sinks
(or vice versa) is a scaling calculation, not a plant statement** -- the same
classification as in `evaluation/recoverable_power.py` (Befund 50) and already maintained there for the
NRMSE/RMSE numbers of the X3 evaluation.

## Coupling to the simulator

`simulation/exhaust.reconstruct_exhaust_state` delivers `outlet_temperature_celsius`
(the OUTLET duct) -- not `bed_temperature_celsius`. At
`ExhaustConfig.outlet_duct_heat_loss_k = 0.0` (default) both are numerically equal, but the
contract at the site-network interface is the outlet duct (the same distinction
that `exhaust.py` already carries as a contract correction). The dry-air mass flow comes
from `dry_air_mass_flow_kg_per_s` in the diagnostics/`SteadyStateSolution`, not recomputed from the
gas velocity. Details (literal signature, what is delivered and what
is missing) in the module docstring of `balance.py`.

## Not part of this task

No LP, no optimizer, no RL environment, no reading of the Excel file itself (MA5,
Befund 57, reads `Model_v3.xlsx` values only as constants in a separate script,
not as a file parser). `routing.py` provides the interconnection abstraction
(`HeatNetworkRouting`), with a fixed and a matrix-based implementation
(the latter the later RL action space) -- it decides nothing by itself.

## Pinch target (problem table algorithm)

`heatnetwork/pinch.py` (branch `lp-benchmark`) answers a different question than
`balance.evaluate_cascade`: not "how much fuel does THIS interconnection need", but
"how much fuel does ANY CONCEIVABLE interconnection need at least" -- the yardstick against
which the six interconnection cases of the partner model (Befund 57) and every later controller
are measured.

**Method.** For the pure targeting problem (minimum heating/cooling demand, no question of
number/arrangement of the heat exchangers) the classical problem table algorithm
(pinch analysis) is exact and needs no solver -- an LP would only be needed for the
network synthesis (which exchanger where), not part of this module. Shifted
temperatures (hot `-dT_min/2`, cold `+dT_min/2`), cascade of the
interval heat balances from the highest temperature downwards, largest negative
cascade load = minimum heating demand, zero crossing of the cascade (shifted by the heating demand)
= pinch temperature.

**Condensation without a second formula.** Below their dew point the exhaust air sources
no longer have a constant heat capacity. Instead of a separate condensation formula,
`hot_segments_for_source` samples the enthalpy-temperature curve with the existing
`balance.evaluate_cascade` (`recovery_mode="condensing"`) -- the same, already tested
code, only called repeatedly at different `T_kalt` values. Above the dew point
the curve is exactly linear (one segment, no discretization error possible); below it
is sampled at `n_segments_below_dew_point` support points, with the dew point itself
as an exact interval boundary.

**Parameters (`scripts/run_pinch_benchmark.py`, PA2 data case).** `T_kalt` over
`{15; 25; 29,1; 35} °C`, `dT_min` over `{10; 15; 20} K` -- 12 combinations, each compared
against the best of the six cases (`heatnetwork.balance.evaluate_cascade`, the same module,
no second cascade implementation -- `T_kalt` is ONE network-wide parameter here, which fits
directly into `HeatNetworkConfig.cold_temperature_celsius`, unlike in the older
`probe_six_case_reconciliation.py` with source-specific `T_kalt`).

**Lower bound, not a construction rule (caveat, MUST be stated with every use).**
The pinch target presupposes arbitrarily many heat exchangers, ideally matched against
each other. A small gap to the best concrete case (in the tested
data case: 2.3-4.2 %) can be fully explained by its restriction to two exchangers
-- it does not prove that no better realizable interconnection exists,
only that it can save at most that much.
`tests/test_heatnetwork_pinch.py::test_pinch_target_never_exceeds_the_best_of_six_cases`
makes the inequality (pinch target <= best case) explicitly checkable -- if this test
fails, the implementation is wrong, not the cases.

Details, numbers (incl. correction of an earlier dew-point mix-up, 47.9 °C
instead of 44.94 °C for Q1) and the side finding that for this data case the pinch sits at the
hot end and is therefore independent of `T_kalt`, in Befund 60
(`04_stand_ap1.md`).

**The pinch is itself an operating-point-dependent quantity, not a fixed property of the
site:** because here it sits at the inlet of the fluidized-bed line (source 2) and its
exhaust air temperature is adjustable via the process set points, a change in the
process operation shifts the site pinch -- up to a jump to a structurally different
bottleneck (dew point of source 1) as soon as the fluidized-bed exhaust air temperature falls below that
dew point (Befund 61, `scripts/run_process_pinch_sensitivity.py`).

**Addendum RA2 -- the T_kalt independence of the minimum heating demand holds ALSO in the lower
regime, confirmed at two feasible QA3 points.** So far only shown for the upper regime
(PA2 data case, pinch throughout at the inlet of source 2). `T_kalt` over
{15; 25; 29.1; 35} °C (`dT_min=20 K`) at both feasible lower-regime points from QA3
(`T_out`(Q2)=40.94 °C and 43.74 °C, pinch = Q1 condensation curve at 44.94 °C): the
minimum heating demand stays exactly bit-identical at BOTH points over the entire sweep
(4548.378 kW), only the minimum cooling demand falls with rising `T_kalt` (e.g. point 1:
3532.0 kW at 15 °C to 1372.8 kW at 35 °C). Mechanistically expected, not a surprise
in hindsight: in the problem table algorithm the minimum heating demand is determined by the cascade
ABOVE the pinch; `T_kalt` changes only the LOWER limit of the
source segments (how far they are cooled) -- as long as the pinch location itself
does not shift (here 44.94 °C throughout, unchanged over the whole sweep), the
cascade above the pinch remains untouched. The statement "condensation/`T_kalt` does not lower the
minimum heating demand" is thus NOT to be restricted to the upper regime --
Befund 57h and the pinch finding draft remain valid unchanged, no correction
needed. Caveat: checked at only two points (the coarse QA2 grid does not provide more feasible
lower-regime points, see RA3), numbers and reproducibility in
`04_stand_ap1.md`, Befund 61.

## Relation to Befund 52 and 57

Befund 52 (`04_stand_ap1.md`) evaluates a fixed operating point against the
disturbance range, exclusively via the gross consumption. Caveat d there already
named "Ein Prozess, kein Verbund" as a restriction. **Befund 57 quantifies this
caveat** (MA3/MA4): net objective function (`evaluation/net_energy.py`,
fuel-equivalent credit `Q_rec/eta_H`, two weighting variants kW sum/
CO2-weighted) via the same optimization mechanics as Befund 52
(`scripts/study_operating_point_regret_net.py`). Result: the gas velocity
leaves its lower bound ONLY under kW-summed weighting, not under
CO2 weighting -- the expectation formulated beforehand (condensation drives the shift)
does not hold; the MA3 weighting dominates the effect. **Addendum NB2 (Befund 57i):**
under the two substantively justified valuations (cost, CO2 -- industrial electricity roughly
3-4x more expensive per kWh than gas, 1.8x more CO2e) the trade of more electricity for more
recovery does NOT pay off; only the kW sum, which equates both forms of energy without justification,
shows the shift. The choice of target is thus an open
**consortium decision with direct consequences for the later RL action space**,
not a question of implementation detail. Details, numbers and caveats in Befund 57.
