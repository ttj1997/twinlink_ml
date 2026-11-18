# NN -- Module interface for the digital-twin framework (stage 1)

Status: infrastructure, no study and no finding. Branch `module-api`, stage 1 built and tested
(forecast service, set-point service, model bundle, schemas); stage 2 (calibration service,
heat-distribution service) is planned in section 9 and not started. The number of this document is
assigned at the merge.

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

The research code is imported, never changed. No model was trained for this package beyond the
small networks of the tests, and no bundle of a study model was built (section 8).

## 2. Rules of the interface

| Rule | Meaning | Where it is enforced |
|---|---|---|
| Stateless | a response depends on the request and the injected model only | tests `test_service_is_stateless` (both services) |
| Advisory | every response carries `advisory_only = true`; the twin recommends, it does not act on a plant | field of `ResponseMeta`, type `Literal[True]` |
| Labelled | every response names schema version, model fingerprint and the status of the process parameters (`placeholder` for everything in the repository today) | `ResponseMeta`, tests `test_response_says_what_it_is` |
| Units in the names | `_celsius`, `_k`, `_m_per_s`, `_kg_per_h`, `_bar`, `_kg_per_kg`, `_kw`, `_s`; two named fractions (`ambient_relative_humidity`, `suspension_solid_fraction`) | test `test_every_physical_field_carries_its_unit` |
| Roles, not channels | no plant, partner or channel name in a schema; the mapping of plant channels to roles comes from outside (a role file of the kind of `conf/plant/plant_local.yaml.example`) | schemas; `tests/test_denylist.py` covers `docs/api/` like every tracked file |
| No silent repair | a wrong time grid, a short history, a wrong observation mode or a wider bound is rejected with a code, never resampled or clipped | tests `test_wrong_grid_is_rejected_not_resampled` and neighbours |
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

### 3.1 `ForecastService` (built)

| | |
|---|---|
| Question | history and planned inputs -> energy target, bed temperature, product moisture over the horizon |
| Model | a model bundle (section 5): one network per target (the per-target GRU composite is the recommended default of the architecture comparison, Befund 94) or the linear ARX; a second bundle can be injected as fallback |
| Request | `sample_time_s`, `observation_mode` (`inline` or `soft_sensor`), `history` (four manipulated variables and three disturbances), `measured_history` (the measured predicted quantities), `planned` (the same seven series from the next sample on) |
| Response | one series per predicted quantity, `time_offset_s`, `fallback_used`, `meta` |
| Rejected | sample time not the model's; history shorter than the model's window; horizon longer than trained; observation mode not the model's; measured history of a quantity the model reads missing or of another length |
| Remarked | input outside the range of the training data (`input_outside_training_range`); measured history the model does not read (`measurement_ignored`); fallback used (`fallback_model_used`) |
| Fallback | used when the extra of the primary model is not installed or its prediction is not finite; never to hide a rejected request |
| Extras | `api`; `torch` for a network |

A history longer than the window is cut to the last samples; a planned series shorter than the
trained horizon gives the corresponding prefix. The window is built with the arithmetic of
`TrajectoryWindowDataset` (standardise with the scalers of the training runs, then set the history
of an unmeasured quantity to zero), so the response equals
`target_scaler.inverse(model.predict(batch))` bit for bit.

### 3.2 `SetpointService` (built)

| | |
|---|---|
| Question | disturbances, limits and margin -> inlet temperature and gas velocity |
| Model | the steady-state optimiser of the dynamic headroom study (`headroom.controllers.SteadyStateOptimizer`, controller R1 of Befund 105): minimum total power with the steady product moisture at or below limit minus margin and the bed temperature below its limit |
| Request | ambient temperature and humidity, solid fraction, spray rate (the throughput, given), optional inlet-temperature offset, moisture limit, margin, optional narrower bounds |
| Response | `setpoints` as a list of time-stamped entries (one entry at `time_offset_s = 0`; `spray_rate_kg_per_h` is `null` because the throughput is not decided), the applied limits, the steady state the model expects (bed temperature, product moisture, total power), `meta` |
| Rejected | a bound wider than the box of the process configuration or than its bed-temperature limit |
| Remarked | a disturbance outside the ranges of the process configuration (`input_outside_model_range`) |
| Error | `no_solution` when the optimiser finds no admissible point |
| Extras | `api`, `optimal-control` (CasADi, IPOPT) |

**One optimiser per request.** `SteadyStateOptimizer` starts every solve from its previous
solution. For a controller that follows a scenario this is right; for a stateless service it makes
the answer depend on the requests before it. The service builds a fresh optimiser for every
request and thereby equals a direct call on a new optimiser bit for bit (test). Controller R1 of
Befund 105 used one warm-started optimiser along a scenario; the service is its stateless
counterpart, not a reproduction of those trajectories.

**Why a list of set points.** The studies on load following and on predicted transient set points
run in parallel to this package. A trajectory of set points, and a decided spray rate, fit into
the same response without a new schema version.

## 4. Schemas

* Source: the pydantic models in `src/twinlink_ml/api/schemas.py` (services), `base.py` (shared
  parts) and `bundle.py` (manifest).
* Files: `docs/api/<service>.request.schema.json`, `.response.schema.json`,
  `.description.schema.json` for `forecast` and `setpoint`, and
  `docs/api/model_bundle_manifest.schema.json`; JSON Schema draft 2020-12.
* Export: `tools/run_here.sh scripts/export_api_schemas.py` (`--check` compares without writing).
  `tests/test_api_schemas.py::test_exported_schema_files_are_current` fails when a file is stale,
  so a change of a model and of its file always travel in one commit.
* Version: `schema_version` is `major.minor`, now `1.0`. A request is accepted when its major
  version is the service's; a minor version only adds optional fields. Renaming or removing a
  field, or changing a unit, is a new major version.
* Scope of version 1: the interface of the laboratory line of the grey-box model (four manipulated
  variables, three disturbances, three predicted quantities). A model with other channels, for
  example a soft sensor with measured process responses as inputs (`docs/29`), needs a new schema
  version; the forecast service refuses such a bundle when it is constructed.

## 5. Model bundle

A directory, or a `.zip` archive of it, with two files:

| File | Content |
|---|---|
| `manifest.json` | format version; kind, label and settings of the model; input and target names; measured targets; window (history length, horizon, planned or realised future inputs); sample time; both scalers; ranges of the training inputs; data specification including the preset version; data fingerprints (process, excitation, split, measurement); status of the process parameters; number of training runs and model seed; code commit, dirty flag and versions; hash of the weights; fingerprint |
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

`tests/test_api_schemas.py`, `test_api_bundle.py`, `test_api_forecast.py`, `test_api_setpoint.py`
(`tests/_api_common.py` holds the shared small pool, generated with `DataSpec.new_study`):

* **End to end per service** on synthetic runs: a request built from the trajectory in engineering
  units against the research code called directly, **bit for bit**, also through
  `handle_json` (a Python float survives the JSON round trip exactly). Forecast: linear ARX in the
  inline and in the soft-sensor mode, and a small GRU composite; set point: three disturbance
  cases and a narrowed box.
* **Schema round trip**, strictness (unknown field, NaN, wrong major version, unequal series) and
  the unit rule.
* **Bundle**: round trip as directory and archive; a wrong fingerprint, an altered manifest, a
  one-bit change of a weight, an unexpected bundle, an incomplete directory, a foreign archive and
  an unknown format version are refused; no local path in the manifest; nothing is overwritten.
* **Statelessness** of both services, fallback behaviour, rejection of every case of section 3.

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
expected fingerprint.

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
grid, authentication, logging of requests and responses, and what happens with a recommendation.

**Effort (estimate of the author, no source):** a REST adapter of the kind above with start-up
checks and error mapping is a matter of two to four working days; an administration-shell adapter
depends mainly on the SDK and server the coordinator has chosen and on how far the schemas are
unfolded into submodel elements, of the order of one to two weeks including the alignment of the
information model. Neither figure includes the channel mapping and resampling at a plant.

## 8. Limits

* **Laboratory line, placeholder parameters.** Both services compute with the grey-box model of
  the laboratory line; every process parameter is a placeholder and the responses say so. The
  model does not represent plant A (Befund 117); a set-point recommendation is not a plant
  statement.
* **Product moisture.** It is not measured at the plant (Befund 56); there the forecast service is
  a soft sensor. The schema carries the observation mode for this reason.
* **Time grid.** The models of the architecture comparison work on 25 s; the plant exports are on
  one-minute sampling (Befund 56). The service refuses another grid; resampling, or a model trained
  on the plant grid, is open (section 10).
* **No uncertainty.** A forecast is a point prediction. The safety margin of the control studies
  (Befund 65, 96, 113) is a convention applied by the caller through `moisture_margin_kg_per_kg`;
  the services do not compute it.
* **Full knowledge.** The optimiser is given the disturbances and returns a steady optimum; it
  contains no state estimator (as in Befund 105, where the controllers saw the true state).
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

## 9. Stage 2 (planned, not started)

| Service | Question | Procedure in the research code |
|---|---|---|
| `CalibrationService` | steady measured sections -> bias per measured quantity, optionally calibrated parameters | constant bias as the default (it is the hard baseline with few points, Befund 120, and no worse than a stationary calibration across days, Befund 123); calibration of a few physical parameters only where the structure fits |
| `HeatDistributionService` | sources, sinks and a production plan -> assignment and key figures | cascade rule and linear programme of the campaign network (Befund 118, 119), rule for the heat-recovery case (Befund 121) |

Both need new schemas (minor version: new services, no change of the existing ones). Open before
they are built: which measured quantities a calibration request may carry for a plant whose
exhaust the laboratory model does not represent (Befund 117), and whether the production plan
arrives as a table of standstills or from a planning system.

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

## 11. Decisions taken without consultation

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
* Two commits: package and export script first, then the schema files generated from a clean
  tree, the tests and this document after the full suite.
