# Code conventions

## Language and tools

- **Python** ≥ 3.11
- **PyTorch**, PyTorch Lightning welcome for training loops
- **Hydra** or OmegaConf for configuration
- **MLflow** or Weights & Biases for experiment tracking (from day 1, not retrofitted)
- **Ruff** (linting + formatting), **mypy** (strict), **pytest**
- Package management: `uv` or Poetry

## Structure

Installable package with `src` layout:

    twinlink_ml/
    ├── pyproject.toml
    ├── README.md
    ├── conf/                     # Hydra configurations, versioned
    ├── src/twinlink_ml/
    │   ├── simulation/           # grey-box process model, excitation design
    │   ├── data/                 # datasets, windowing, splits
    │   ├── models/               # architectures behind a common interface
    │   ├── evaluation/           # metrics, rollouts, learning curves
    │   ├── constraints/          # physical constraints (projection, monotonicity)
    │   └── utils/                # seeding, logging, serialization
    ├── notebooks/                # import the modules, duplicate no code
    └── tests/

Strict separation: **config** (`conf/`) – **code** (`src/`) – **artifacts** (outside version
control, referenced via the tracking tool).

## Object orientation

- **Type hints throughout**, including internal functions. mypy in strict mode must pass.
- **Dataclasses** (`frozen=True` where possible) or **Pydantic** for all configuration objects.
  No loose dicts as parameter containers.
- **Registration instead of selection in the script.** Models are listed exactly once in
  `models/registry.py`, with label, factory and prerequisites. A dict in the driver script
  only lasts until the second driver; after that, two studies compare different
  sets of models under the same name. Factories take no arguments, so that each seed and
  support point gets a fresh instance and no state lives across learning-curve points.
- **Abstract base classes (ABC)** for the four core abstractions:
  - `ProcessSimulator` – delivers trajectories (grey-box model, later Dyssol adapter,
    later possibly a learned surrogate)
  - `SequenceDataset` – windowing, normalisation, splits
  - `SequenceModel` – common interface for GRU/LSTM/xLSTM/Mamba/Transformer,
    incl. multi-step rollout
  - `Evaluator` – metrics and result formats
- **Dependency injection** instead of hard coupling: models are handed the simulator or the dataset,
  they do not construct it themselves. This later allows the swap grey-box → Dyssol → real data
  without changing the training code.
- Observe **SOLID**, in particular open-closed: adding a new architecture must not require any
  change to training or evaluation code.
- Composition over inheritance. Avoid inheritance depth > 2.

**Deliberate deviation: `steady_state.py` is a module function, not an ABC.** The exact
steady-state solver (`solve_steady_state`) is deliberately *not* a fifth item in the
ABC list above. An abstraction (e.g. `SteadyStateSolver`) over exactly one implementation
is speculative building: the only foreseeable second case -- a steady-state flowsheet mode of
Dyssol -- provides no right-hand side whose root could be bisected, so the
interface of the second case is not even foreseeable in its form yet. The ABC
will be added once this second case is implemented, not before.

**Confirmation of the convention: `sorption.SorptionIsotherm` is an ABC.** Unlike the
`SteadyStateSolver` case above, two concrete implementations existed here from the start
(`ConstantIsotherm`, `GabIsotherm`) -- the same threshold that `excitation.DisturbanceDesign`
already draws over two implementations. An ABC here is therefore not a speculative abstraction,
but follows the existing convention; see `04_stand_ap1.md`, Befund 34.

## Reproducibility

- **Central seed handling** in one place (Python, NumPy, PyTorch, CUDA)
- **Config serialization**: the complete resolved configuration is saved with every run
- **Deterministic splits**: split assignment computed from run IDs and seed, not drawn at random
  at run time
- Git commit hash in the run log

## Tests

- Unit tests for all physical balances of the grey-box model (mass, energy close
  within tolerance)
- Test that splits contain **no overlapping windows** between train and test
  (leakage regression)
- Property-based tests (Hypothesis) for constraint projections: output never leaves the
  physical value range
- Smoke test that runs the entire pipeline with a tiny data set
- **Leakage test for observability:** if a target is configured as not measured inline,
  it must not appear anywhere in the window history. A leak there looks like an excellent
  soft sensor in all metrics and is not reproducible at the plant — it makes a comparison
  useless without making it suspicious.
- **Test for residual equivalence:** a model with a zeroed head must reproduce *exactly* the
  parameter-free baseline (persistence with inline measurement,
  mean prediction without). Without PyTorch this can be checked on the `RandomFeatureNARX`, so
  the requirement itself is checked and not just one implementation.
- **No architecture code written blind.** A module in which no tensor operation has been
  executed is marked as not executed in the status document and delivers
  no reportable number. The associated tests skip themselves with
  `pytest.importorskip` and are the first task on a machine with the missing
  dependency.

## Notebooks

- Serve **visualization and exploration**, not implementation
- Import from `src/`, contain no logic that does not also exist as a module
- Commented and narrative: what is shown, what can be seen in it
- Standard visualizations: trajectories of the process model, excitation signals,
  learning curves with spread band, rollout comparisons against ground truth
- **Every learning curve carries its resolution threshold and, where known, its noise floor.** The
  band is part of the plot function in `evaluation/plots.py` and not of the caller —
  if the logic sits in the notebook cell, sooner or later a variant without the
  threshold band appears, and that one ends up in a talk.
- **Do not put curves of different observability operating modes into one diagram.** The
  primary metric differs (run-level vs. pooled); the y-axes mean
  different things.
- Clear outputs before the commit (`nbstripout`)

## Documentation

- Docstrings in NumPy or Google style, with units for all physical quantities
- Units always explicit: `T_in_celsius`, `m_spray_kg_per_h` – never just `T` and `m`
- **Report time spans in seconds, not in sampling steps.** A history of "two
  steps" is something completely different at Δt = 2 s than at Δt = 25 s. Step counts are
  implementation quantities, seconds are comparable with the process time constants — and only
  the comparison with them is a statement.
