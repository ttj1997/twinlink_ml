# 47 -- Module interface for the digital-twin framework

Status: infrastructure, no study and no finding. Branch `module-api`. Stage 1 (forecast service,
set-point service, model bundle, schemas) and stage 2 (calibration service, heat-distribution
service, recovery-case rule) are built and tested. Document number 47; merged into `master` on
2026-11-23.

## 1. Purpose and limits

The grant proposal ends the description of TUHH-B-AP 1 with: the data and model pipeline for
dynamic control is integrated as a separate module into the framework defined in the first work
package of the consortium coordinator, so that it can be used for other optimisation tasks as
well, for example the distribution of waste heat. That framework, its transport and its
information model are the coordinator's to define. This package therefore stops one step before
the integration:

* it makes the pipeline **integrable**: a narrow, stable Python package `twinlink_ml.api` above
  the research code, with versioned request and response schemas (JSON Schema files in
  `docs/api/`) and a transferable format for trained models;
* it contains **no** framework adapter, no REST layer and no asset administration shell. Section 7
  describes how an adapter would map the services; section 10 lists what has to be asked first.

The research code is imported. It was changed in one place, by decision of the project lead:
two additions to the campaign-network code that leave its results unchanged (section 3.6). No
model was trained for this package beyond the small networks of the tests, and no bundle of a
study model was built (section 8).

## 2. Rules of the interface

| Rule | Meaning | Where it is enforced |
|---|---|---|
| Stateless | a response depends on the request and the injected model only | tests `test_service_is_stateless` (every service with a model behind it) |
| Advisory | every response carries `advisory_only = true`; the twin recommends, it does not act on a plant | field of `ResponseMeta`, type `Literal[True]` |
| Labelled | every response and every description names schema version, model fingerprint, the plant and model it stands for (`model_scope`) and the status of the process parameters (`placeholder` for everything in the repository today) | `ResponseMeta`, `ServiceDescription`, tests `test_response_says_what_it_is` |
| Units in the names | `_celsius`, `_k`, `_m_per_s`, `_kg_per_h`, `_nm3_per_h`, `_bar`, `_kg_per_kg`, `_kw`, `_kwh`, `_kw_per_k`, `_kg`, `_s`, `_per_h`; dimensionless quantities are named as fractions, ratios or sums | test `test_every_physical_field_carries_its_unit` |
| Roles, not channels | no plant, partner or channel name in a schema; the mapping of plant channels to roles comes from outside (a role file of the kind of `conf/plant/plant_local.yaml.example`) | schemas; `tests/test_denylist.py` covers `docs/api/` like every tracked file |
| No silent repair | a wrong time grid, a short history, a wrong observation mode, a wider bound or a parameter that cannot be calibrated is rejected with a code, never resampled, clipped or dropped | tests `test_wrong_grid_is_rejected_not_resampled`, `test_what_cannot_be_calibrated_is_rejected` and neighbours |
| Extras on demand | importing the package loads neither PyTorch nor CasADi; a service names the extra it needs when it is called | test `test_import_loads_neither_torch_nor_casadi` |
| Strict input | unknown fields, NaN and infinity are errors; values are immutable | `ApiModel` (`extra="forbid"`, `allow_inf_nan=False`, `frozen=True`) |

Errors a service raises on purpose carry a machine-readable `code`: `request_rejected`,
`no_solution`, `model_failure`, `bundle_integrity`, `missing_extra`. A payload that does not follow
the schema raises a `pydantic.ValidationError`.

## 3. Services

Every service is a class behind the ABC `twinlink_ml.api.base.Service` with four methods:
`describe()` (version, model fingerprint, parameter status, valid input ranges), `handle(request)`,
`handle_json(payload)` (the whole surface an adapter needs) and `json_schemas()`. What a service
computes with is injected through its constructor.

| Service | Model behind it | `model_scope` | Extras |
|---|---|---|---|
| `ForecastService` | model bundle (network or linear ARX) trained on runs of the grey-box simulator | `lab line, grey-box` (from the bundle) | `api`; `torch` for a network |
| `SetpointService` | steady-state optimiser on the grey-box model | `lab line, grey-box` | `api`, `optimal-control` |
| `CalibrationService` | plant-near steady model | `plant A, plant-near model` | `api` |
| `HeatDistributionService` | campaign-network model (one source, cold streams, a tank) | `site heat network, campaign-network model` | `api` |
| `RecoveryCaseService` | rule of Befund 121 | `partner workbook rebuild` | `api` |

**One scope per answer.** The services do not share one model. Every description and every
response therefore says in `model_scope` which plant or line and which model it stands for, and a
result of one scope must not be applied to another. The scope of a forecast comes from its bundle
(a required argument when a bundle is built), the others are constructor arguments with the
defaults above.

### 3.1 `ForecastService`

| | |
|---|---|
| Question | history and planned inputs -> energy target, bed temperature, product moisture over the horizon |
| Model | a model bundle (section 5): one network per target (the per-target GRU composite is the recommended default of the architecture comparison, Befund 94) or the linear ARX; a second bundle can be injected as fallback |
| Request | `sample_time_s`, `observation_mode` (`inline` or `soft_sensor`), `history` (four manipulated variables and three disturbances), `measured_history` (the measured predicted quantities), `planned` (the same seven series from the next sample on) |
| Response | one series per predicted quantity, `time_offset_s`, `fallback_used`, `meta` |
| Rejected | sample time not the model's; history shorter than the model's window; horizon longer than trained; observation mode not the model's; measured history of a quantity the model reads missing or of another length |
| Remarked | input outside the range of the training data (`input_outside_training_range`); measured history the model does not read (`measurement_ignored`); fallback used (`fallback_model_used`) |
| Fallback | used when the extra of the primary model is not installed or its prediction is not finite; never to hide a rejected request |

A history longer than the window is cut to the last samples; a planned series shorter than the
trained horizon gives the corresponding prefix. The window is built with the arithmetic of
`TrajectoryWindowDataset` (standardise with the scalers of the training runs, then set the history
of an unmeasured quantity to zero), so the response equals
`target_scaler.inverse(model.predict(batch))` bit for bit.

### 3.2 `SetpointService`

| | |
|---|---|
| Question | disturbances, limits and margin -> inlet temperature and gas velocity |
| Model | the steady-state optimiser of the dynamic headroom study (`headroom.controllers.SteadyStateOptimizer`, controller R1 of Befund 105): minimum total power with the steady product moisture at or below limit minus margin and the bed temperature below its limit |
| Request | ambient temperature and humidity, solid fraction, spray rate (the throughput, given), optional inlet-temperature offset, moisture limit, margin, optional narrower bounds |
| Response | `setpoints` as a list of time-stamped entries (one entry at `time_offset_s = 0`; `spray_rate_kg_per_h` is `null` because the throughput is not decided), the applied limits, the steady state the model expects (bed temperature, product moisture, total power), `meta` |
| Rejected | a bound wider than the box of the process configuration or than its bed-temperature limit |
| Remarked | a disturbance outside the ranges of the process configuration (`input_outside_model_range`) |
| Error | `no_solution` when the optimiser finds no admissible point |

**One optimiser per request.** `SteadyStateOptimizer` starts every solve from its previous
solution. For a controller that follows a scenario this is right; for a stateless service it makes
the answer depend on the requests before it. The service builds a fresh optimiser for every
request and thereby equals a direct call on a new optimiser bit for bit (test). Controller R1 of
Befund 105 used one warm-started optimiser along a scenario; the service is its stateless
counterpart, not a reproduction of those trajectories.

**Why a list of set points.** The studies on load following and on predicted transient set points
run in parallel to this package. A trajectory of set points, and a decided spray rate, fit into
the same response without a new schema version.

### 3.3 `CalibrationService`

| | |
|---|---|
| Question | stationary sections of plant operation -> constant bias per measured quantity; optionally calibrated parameters |
| Model | the plant-near steady model of plant A (`plant_calibration`, Befund 123, on the parametrisation of Befund 117): evaluated at the measured operating point of every section, compared in bed and exhaust temperature |
| Request | `sections` (product role, intake temperature, process-air and atomising-air flow, supply-air temperature, spray rate, measured bed and exhaust temperature), optional `calibrate_parameters` |
| Response | the temperatures of the uncalibrated model, the bias (mean of measured minus model), the errors without and with the bias, the fit when asked for, a fingerprint of the result, `meta` |
| Rejected | an unknown product role; a parameter that is not calibratable on stationary sections (kinetics, heat capacity); more parameters than the configuration allows (three) or than measured values; a product-specific parameter without a section of its product |
| Remarked | bias from a single section (`bias_from_one_section`); every fit (`calibration_not_validated`); a fitted parameter at a bound (`parameter_at_bound`) |
| Error | `no_solution` when the model has no steady state at a section |

**Bias first.** The default answer is the bias and nothing is fitted unasked. With few points a
constant bias is a hard baseline (Befund 120), and in Befund 123 a stationary calibration was no
better on the other day than the bias, because per day only two directions of the parameter space
are determinable. A fit is therefore an explicit request, limited to the parameters the study
configuration marks as calibratable, and its answer always carries a notice.

**A calibration belongs to its model.** The calibrated model is the plant-near model of plant A;
the forecast and the set-point service compute with the laboratory line. A bias of the one must
not be applied to the other, so those two services do not take a calibration result as input
(confirmed by the project lead, 2026-11-18). The result carries a fingerprint (model, bias,
parameters) so that a caller can store and name it.

### 3.4 `HeatDistributionService`

| | |
|---|---|
| Question | one source of waste heat, consumers and optionally a tank -> heat per consumer and hour, storage operation, key figures |
| Model | the campaign-network model (`campaign_network`, Befunde 118 and 119): counter-current exchangers with a minimum approach, water as the carrier, a stratified tank on the loop |
| Request | `mode`, `source_power_kw`, `storage_capacity_kwh`, `storage_content_kwh` (content of the tank at the first hour), `storage_rule`, `sinks` (cold streams hour by hour: inlet and outlet temperature, heat-capacity flow while running, required temperature, efficiency of the replaced supply, optional on/off plan `running`, optional `group` for parallel branches of one stage) |
| Mode `cascade_rule` | the rule without foresight (`run_k1`; with a tank the storage rule K2 or K2b, `run_k2`): heat per consumer and hour, storage flows, key figures of `key_figures` |
| Mode `planning` | one linear programme over the whole request with the on/off plan known (`rolling_plan`, the plan as the forecast, one window): heat used per hour, storage operation, key figures; no split to the consumers |
| Rejected | a grid other than hourly; parallel branches with different temperatures; in the mode `planning` consumers with different efficiencies; a content above the capacity (schema) |

Supply temperature, return floor, minimum approach, storage loss, discharge limit, pump share and
CO2 factors come from the injected configuration (placeholders, `conf/campaign_network`), the
consumers from the request.

**The tank is state of the caller.** A request carries the content at its first hour and the
response returns the content after every hour. The cascade rule has no foresight, so calling the
service hour by hour with the content handed over gives the same operation as one call over all
hours (tests `test_calls_with_the_content_handed_over_equal_one_call` and, in the research code,
`test_k2_continues_from_a_handed_over_content`).

**One computation of its own.** The key figures of the mode `planning` repeat the formulas of
`evaluate.key_figures` for a total instead of a sum over consumers (as `plan_figures` of the
driver of Befund 119 does).

### 3.5 `RecoveryCaseService`

| | |
|---|---|
| Question | exhaust temperatures of the two dryers and the ambient temperature -> which heat-recovery case of the two-dryer arrangement of the partner workbook |
| Rule | case 1 (each exhaust feeds its own sink) where exhaust 1 minus exhaust 2 is above -0.541 K **and** exhaust 2 is warmer than ambient plus 20 K; case 2 (crossed) where exhaust 2 is the warmer one; no recommendation where exhaust 1 is warm enough but exhaust 2 has no usable heat |
| Response | `recommended_case` (1, 2 or `null`), `case_1_expected_best`, the two temperature differences, and with every answer `reference`, `valid_for`, `hit_rate_physical_unpressurised`, `hit_rate_physical_pressurised`, `hit_rate_basis` |
| Remarked | no case named (`no_case_named`); exhaust 2 above 118.5 degC, where the tree of the unpressurised mode has one more branch (`unpressurised_branch_not_represented`); ambient temperature outside the investigated box (`outside_investigated_box`) |

The research code holds the study that found the rule (decision trees of depth 3 fitted on its
samples), not the rule as a function, so this is the one service whose logic is written in the
package (decision of the project lead, 2026-11-18). What it implements and what it claims:

* **The rule** is the effective leaves of the trees of the two physical modes (Befund 121,
  `docs/38_case_ranking.md` section 6.4; artifact `case_ranking_v1_analysis`): exactly those of
  the pressurised mode; the unpressurised tree has one more branch above 118.5 degC of exhaust 2,
  which is remarked, not represented.
* **Valid for** the physical modes of the rebuild of the partner workbook with its fixed circuit
  flows, inside the investigated box; not the partner mode (its thresholds differ), not the
  optimised circuit flows, not a plant. The rule describes where the rebuild ranks case 1 first;
  it is not physics.
* **Hit rate**: 0.966 (unpressurised) and 0.971 (pressurised), the test accuracy of those trees on
  the study's own test stream for the statement "case 1 is strictly best" (Befund 121). The
  response says that the rates belong to the trees: they are the rates of this rule in the
  pressurised mode, and in the unpressurised mode the rate of a tree with one branch more.
* Thresholds and rates are constants of `api/heat.py`; the test
  `test_rule_constants_are_those_of_the_study_artifact` reads them off the artifact.

It is an operation of its own next to the heat distribution, because a service maps one request
to one response.

### 3.6 Changes of the campaign-network code

Decided by the project lead (2026-11-18) so that the study and the service call the same code:

1. `planning.cut_data_for_streams(streams, schedulable, supply_celsius, floor_celsius,
   approach_k)`: the arithmetic of `cut_data` as a pure function of cold streams; `cut_data(config)`
   builds the consumers of the configuration and calls it.
2. `cascade.run_k2(..., initial_content_kwh=0.0)` and `planning.rolling_plan(...,
   initial_content_kwh=0.0)`: content of the tank at the beginning of the first hour, before the
   loss of that hour; the default is the empty tank of the studies.

Evidence that no result moves (`tests/test_campaign_network_artifacts.py`): at the cell
availability 0.57, independent schedules, draw 0, 400 kW, the key figures equal the stored ones
bit for bit in `campaign_network_v1` (Befund 118: K1, and K2 for the three tank sizes),
`campaign_network_v2` (addendum: K2b) and `campaign_network_2_v1` (Befund 119: K2b, and P1 and P2
at 8 h, every stored figure but the solve time). The same comparison was made on the unchanged
code before the change, with the same result. In addition: a run over consecutive hours equals two
runs with the content handed over (K2, K2b and the rolling planning), and a content outside the
tank is refused. The lab book has no entry for this change; a dated addendum to Befunde 118 and
119 is left to the session that merges.

## 4. Schemas

* Source: the pydantic models in `src/twinlink_ml/api/schemas.py` (services), `base.py` (shared
  parts) and `bundle.py` (manifest).
* Files: `docs/api/<service>.request.schema.json`, `.response.schema.json`,
  `.description.schema.json` for `forecast`, `setpoint`, `calibration`, `heat_distribution` and
  `recovery_case`, and `docs/api/model_bundle_manifest.schema.json`; JSON Schema draft 2020-12.
* Export: `tools/run_here.sh scripts/export_api_schemas.py` (`--check` compares without writing).
  `tests/test_api_schemas.py::test_exported_schema_files_are_current` fails when a file is stale,
  so a change of a model and of its file always travel in one commit.
* Version: `schema_version` is `major.minor`, now `1.0` for all five services (nothing has been
  released, so stage 2 did not raise it). A request is accepted when its major version is the
  service's; a minor version only adds optional fields or services. Renaming or removing a field,
  or changing a unit, is a new major version.
* Scope of version 1: for the forecast and the set point the interface of the laboratory line of
  the grey-box model (four manipulated variables, three disturbances, three predicted quantities);
  a model with other channels, for example a soft sensor with measured process responses as inputs
  (`docs/29`), needs a new schema version and is refused by the forecast service when it is
  constructed. The calibration works on the roles of the plant-near steady model, the heat
  distribution on cold streams on an hourly grid.

## 5. Model bundle

A directory, or a `.zip` archive of it, with two files:

| File | Content |
|---|---|
| `manifest.json` | format version; kind, label, scope (`model_scope`) and settings of the model; input and target names; measured targets; window (history length, horizon, planned or realised future inputs); sample time; both scalers; ranges of the training inputs; data specification including the preset version; data fingerprints (process, excitation, split, measurement); status of the process parameters; number of training runs and model seed; code commit, dirty flag and versions; hash of the weights; fingerprint |
| `weights.npz` | the weights as plain NumPy arrays, read with `allow_pickle=False` |

* **No pickle.** The checkpoints of `utils/checkpoints.py` are pickled dictionaries read with
  `weights_only=False`; that is acceptable for a local file and not for a package that is handed
  to another system. A bundle holds arrays and JSON only.
* **Fingerprints.** `weights_sha256` covers names, dtypes, shapes and bytes of the arrays;
  `fingerprint` is the SHA-256 of the canonical manifest without the fingerprint field and so
  covers weights, scalers, settings and provenance. `load_bundle` recomputes both and refuses a
  bundle in which either differs; the caller can pin the fingerprint it expects
  (`expected_fingerprint`).
* **No local paths.** A bundle records commit, dirty flag, versions and whether the code location
  was checked, but not `code/package_dir` and `code/interpreter`: both name a local user
  directory, and a bundle leaves the machine. Exception to the provenance rule of `CLAUDE.md`,
  decision of the project lead of 2026-11-18. The check itself stays: building a bundle calls
  `environment_fingerprint()`, which aborts when the loaded package is not the checkout's.
* **From runs to a bundle.** `fit_bundle(data, n_train=..., model_seed=..., kind=...)` builds the
  data sets of one budget from a generated pool (run-level split, scalers from the training runs
  of that budget), fits and packs. `build_bundle(model, train, ...)` packs a model that is already
  fitted. Supported: the per-target composite network of any core and the linear ARX.
* **Rebuilding.** `instantiate_model(bundle)` rebuilds the fitted model, a network always on the
  CPU. The composite network goes through its public `state_dict` / `load_checkpoint`; the linear
  ARX has no such pair, so `BundledARXModel` sub-classes it and adds an export and a restore
  without touching `fit` or `predict`.
* Bundles are not committed (`artifacts/` rules apply to whoever builds one); the tests build
  theirs in a temporary directory.

## 6. Checks

`tests/test_api_schemas.py`, `test_api_bundle.py`, `test_api_forecast.py`, `test_api_setpoint.py`,
`test_api_calibration.py`, `test_api_heat.py` (`tests/_api_common.py` holds the shared small pool
of the forecast tests, generated with `DataSpec.new_study`):

* **End to end per service** on synthetic data against the research code called directly, **bit
  for bit**, also through `handle_json` (a Python float survives the JSON round trip exactly).
  Forecast: linear ARX in the inline and in the soft-sensor mode, and a small GRU composite, on
  requests built from the trajectory in engineering units. Set point: three disturbance cases and
  a narrowed box. Calibration: bias and errors on two synthetic sections, and a fit of two
  parameters on synthetic measurements of the model itself. Heat distribution: the cascade rule,
  both storage rules and the planning on two weeks of the study configuration with standstills,
  each with an empty tank and with a content at the first hour, and three calls in a row against
  one call. Recovery case: the rule against the trees of the study artifact.
* **Schema round trip**, strictness (unknown field, NaN, wrong major version, unequal series) and
  the unit rule.
* **Bundle**: round trip as directory and archive; a wrong fingerprint, an altered manifest, a
  one-bit change of a weight, an unexpected bundle, an incomplete directory, a foreign archive and
  an unknown format version are refused; no local path in the manifest; nothing is overwritten.
* **Statelessness**, fallback behaviour and the rejection of every case of section 3.
* **The campaign-network code** against the artifacts of Befund 118 and 119 (section 3.6).

## 7. How an adapter would map the services

The adapter is the coordinator's side of the interface; the sketches below only show that nothing
in the package stands in its way.

**REST.** One resource per service: `POST /v1/<service>` with the request JSON, answered by
`service.handle_json(body)`; `GET /v1/<service>/description` answered by
`service.describe().model_dump_json()`; `GET /v1/<service>/schema` from `json_schemas()`. Error
mapping: `pydantic.ValidationError` -> 422 with the pydantic error list; `request_rejected` -> 422
with code and message; `no_solution` and `model_failure` -> 409 or 422 (to be agreed);
`missing_extra` and `bundle_integrity` are deployment errors and belong in the start-up of the
adapter, not in a request. Services are constructed once at start-up from a bundle path and an
expected fingerprint, or from their configuration.

**Asset administration shell.** One submodel per service: the fields of `describe()` as
properties (schema version, model fingerprint, parameter status, valid ranges, `advisory_only`),
`handle` as an operation whose input and output variables are the request and the response, either
as one JSON blob or unfolded into submodel elements along the JSON Schema (the unit suffixes map to
unit references). A model bundle is a file element with its fingerprint as a property; replacing a
model is replacing that file and the pinned fingerprint. Whether results are pulled (operation) or
pushed (property updates on every sample) depends on the framework.

**Message bus.** Request and response as messages with a correlation identifier; statelessness
makes replays and several workers harmless.

In every variant the adapter owns: the mapping of plant channels to roles, resampling to the model
grid, the detection of stationary sections, authentication, logging of requests and responses, and
what happens with a recommendation.

**Effort (estimate of the author, no source):** a REST adapter of the kind above with start-up
checks and error mapping is a matter of two to four working days for the five services; an
administration-shell adapter depends mainly on the SDK and server the coordinator has chosen and
on how far the schemas are unfolded into submodel elements, of the order of one to two weeks
including the alignment of the information model. Neither figure includes the channel mapping and
resampling at a plant.

## 8. Limits

* **Placeholder parameters.** Every model behind the services rests on placeholders and every
  response says so. No response is a plant statement.
* **Two different models.** Forecast and set point compute with the grey-box model of the
  laboratory line, which does not represent plant A (Befund 117); the calibration computes with
  the plant-near model of plant A. They are not interchangeable and a calibration result is not an
  input of the other two services (section 3.3).
* **Product moisture.** It is not measured at the plant (Befund 56); there the forecast service is
  a soft sensor. The schema carries the observation mode for this reason.
* **Time grids.** The models of the architecture comparison work on 25 s; the plant exports are on
  one-minute sampling (Befund 56); the network model is hourly. A service refuses another grid;
  resampling, or a model trained on the plant grid, is open (section 10).
* **No uncertainty.** A forecast is a point prediction. The safety margin of the control studies
  (Befund 65, 96, 113) is a convention applied by the caller through `moisture_margin_kg_per_kg`;
  the services do not compute it.
* **Full knowledge.** The set-point optimiser is given the disturbances and returns a steady
  optimum; it contains no state estimator (as in Befund 105, where the controllers saw the true
  state). The planning mode of the heat distribution takes the on/off plan as true.
* **Calibration says nothing about identifiability per request.** The service limits what can be
  fitted and warns; it does not compute the sensitivity and collinearity measures of the study for
  the sections it is given.
* **Network model.** One source of constant power and an hourly grid. The planning mode does not
  split the heat to the consumers, needs one efficiency for all of them, and plans one window:
  nothing after the request has a value in the model, so heat left in the tank at the end counts
  for nothing.
* **Recovery-case rule.** A description of a study on a rebuilt workbook at laboratory scale, with
  the hit rates of the trees it was read from (section 3.5).
* **No bundle of a study model.** The package was built without a compute run. Turning the GRU
  composite of the architecture comparison into a bundle needs its checkpoint or a retraining and
  is a step of its own.
* **Side effect of the provenance.** `environment_fingerprint()` imports PyTorch when it is
  installed in order to record its version, so building a bundle loads PyTorch even for the linear
  ARX. Loading a bundle and answering requests with the ARX does not.
* **Cost of the set-point service.** Building the optimiser dominates a request, about one second
  on the development machine (a single measurement while writing, no artifact).
* **Base of the branch.** `origin/master` (`ba02e0e`) as instructed; the local `master` of the
  main checkout was 12 commits ahead at that time (not pushed). The package adds files and touches
  none of those commits.

## 9. Open points

**For the fourth work package (model operation and updates): one model per plant with its
calibration.** Today the services stand on four different scopes (laboratory line, plant A,
site network, partner workbook), and a calibration result has no consumer. A plant in the
framework needs one model with a stated scope, a calibration that belongs to that model and is
versioned with it, and services that take the two together; the fingerprints of the bundle and of
the calibration result are the handles for this. Until then `model_scope` keeps the scopes apart.

| Further step | What it needs |
|---|---|
| A service on the plant-near model that takes a calibration result (for example the operating point of plant A, Befund 103) | a decision which question it answers; the roles of the calibration request can be reused |
| Identifiability per calibration request | the measures of `plant_calibration.identifiability` on the sections of the request |
| A value of the tank content at the end of a planning window | a terminal term in the linear programme of `rolling_plan` (a change of the research model) |
| A bundle of the GRU composite of the architecture comparison | its checkpoint or a retraining (a compute run) |
| A set-point trajectory and a decided spray rate | the results of the running studies; the response already has the fields |

## 10. Questions to the consortium coordinator

No requirement specification or data concept of the coordinator was at hand when this package was
written (2026-11-18); if one exists, the schemas have to be aligned with it first.

1. **Transport and hosting.** REST, administration-shell operations, a message bus? Does the
   module run as a process of its own next to the framework, or is it imported into it?
2. **Information model.** Is there a data concept (names, units, identifiers of the quantities)
   the schemas should follow? Are the JSON Schemas of `docs/api/` a usable basis, or is a
   submodel template expected?
3. **Runtime.** Python version and operating system of the target system; are PyTorch (CPU) and
   CasADi/IPOPT acceptable there, and how large may the environment be? Would a network have to be
   delivered in another format (for example ONNX)?
4. **Pull or push.** Does the framework ask for a forecast or a recommendation when it needs one,
   or does it expect a value on every sample?
5. **Time grid and history.** On which grid does the framework hold the data, and who resamples to
   the grid of a model? Can it deliver the last window of samples (at 25 s, 32 samples) with every
   request?
6. **Planned inputs.** Who supplies the planned set points and the expected disturbances over the
   horizon (ambient air: a forecast, or the last value held)?
7. **Channel mapping.** Where does the mapping of plant channels to the roles of the schemas live,
   and who maintains it?
8. **Models as files.** How are model bundles distributed, versioned and exchanged in the
   framework, and where is the expected fingerprint pinned? This is also the entry point of the
   later work on model monitoring and updates.
9. **Recommendations.** How is a recommendation shown and logged, given that the twin must not
   change operating parameters in this project phase? Is `advisory_only` enough as a marker?
10. **Errors and validity.** How should a rejected request, a missing solution and a notice
    (input outside the valid range, placeholder parameters) appear to a user?
11. **Latency and load.** Which response times are expected? (The forecast is far below the sample
    time, Befund 97; a set-point request rebuilds its optimiser.)
12. **Confidentiality.** May requests and responses with plant values be logged by the module, or
    only by the framework?
13. **Stationary sections.** Does the framework detect stationary sections of plant operation and
    deliver their means, or does it deliver time series and expect the module to find them?
14. **Production plan.** In which form do planned standstills of the consumers arrive (a table of
    start and end times, a planning system), on which grid and how far ahead?
15. **Consumers and sources.** Does the framework hold the heat sources and sinks of the site with
    their temperatures and flows, so that a heat-distribution request can be assembled from it?

## 11. Decisions

Decisions of the project lead (2026-11-18):

1. **Bundles carry no local paths**: no absolute path and no user name; commit, fingerprints and
   preset version are recorded. Exception to the provenance rule, noted in `CLAUDE.md` (section 5).
2. **A calibration result is not passed to the forecast or the set point.** Every service names
   its plant and model in `describe()` and in every response (`model_scope`, section 3). Open for
   the fourth work package: one model per plant with its calibration (section 9).
3. **`cut_data` is not duplicated**: its arithmetic is a pure function of the research code that
   the study and the service both call; the artifacts of Befund 118 and 119 are reproduced bit
   for bit (section 3.6).
4. **The recovery-case rule lives in the package**, with the reference to Befund 121, its range of
   validity (physical modes, investigated box) and its hit rate in every answer (section 3.5).
5. **Initial content of the tank**: optional argument of `run_k2` and `rolling_plan`, default
   empty, artifacts reproduced bit for bit with the default; the service takes the current content
   as an input (sections 3.4, 3.6).

Also decided by the project lead on that day: stage 2 with the plant-near model as the model of
the calibration, the bias as its only default with a fit on explicit request, both modes of the
heat distribution, and the recovery-case rule as an operation of its own.

Taken without consultation:

* Explicit fields per quantity instead of a generic list of named channels: a schema then states
  names and units itself; the price is a new schema version for a model with other channels.
* The existing channel names of the process configuration are kept as field names; only the energy
  target gets its unit (`energy_target_kw`).
* A history longer than the window is cut, not rejected; a shorter one is rejected.
* Bounds of a set-point request may only narrow the box of the configuration.
* One optimiser per set-point request (section 3.2), at the price of about a second.
* A network is rebuilt on the CPU whatever device it was trained on.
* The bundle archive is a `.zip` (the suffix `.bundle` is taken by git bundles in `.gitignore`).
* `pydantic>=2.10,<3` as the extra `api` (2.13.5 is installed in the project environment).
* The schema version stays 1.0 for stage 2 (new services and fields, nothing released before).
* The wording of the scopes (`lab line, grey-box`, `plant A, plant-near model`, `site heat
  network, campaign-network model`, `partner workbook rebuild`); `model_scope` is a required
  argument when a bundle is built.
* The planning mode solves one window over the whole request; re-planning under plan deviations
  (P2 of Befund 119) is an evaluation of a plan against a truth and not a service question.
* `k2` is the default storage rule (the rule fixed before the run of Befund 118); `k2b` on request.
* The recovery-case rule recommends nothing where the trees of the study name no case, instead of
  falling back to case 1 or 2; it takes the ambient temperature as a third input, because the
  second condition of the trees needs it.
* An initial content may exceed the capacity by the duty tolerance of the configuration (1e-7 kW),
  so that a content handed over from a run that ended at the full tank is accepted unchanged.
* Sink and group names are identifiers (letters, digits, underscore).
* Commits: stage 1 as package and export script first, then schema files, tests and document after
  the full suite; the campaign-network change as a commit of its own after a full suite; stage 2
  in one commit after a full suite, with the schema files checked against the committed code.
