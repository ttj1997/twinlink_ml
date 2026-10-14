# Contributing

This is a research code base. The goal is results that can be trusted and traced back, so most
rules below are about provenance and honesty of reporting, not style. `CLAUDE.md` holds the
project rules in full; `docs/03_code_konventionen.md` the code conventions (German).

## Setup

One virtual environment in the main checkout with every extra of `pyproject.toml`:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[torch,sklearn,plant-data,train,rl,dev]"
python -c "import twinlink_ml; print(twinlink_ml.__file__)"   # must point to this repo's src/
```

**Worktrees:** an editable install points at one directory only. Start every run in a worktree
with `tools/run_here.sh <python args>` (it loads the worktree's own code) and never run
`pip install -e .` from a worktree. A run that loads the code of another checkout aborts
(`twinlink_ml.utils.code_location`).

## Before every commit

```bash
ruff check --fix .        # fixes the mechanical findings
ruff format <files you changed>
ruff check .              # must be empty
mypy src
pytest -q                 # the full suite takes tens of minutes
nbstripout notebooks/*.ipynb
```

- The order matters: `ruff check --fix` before `ruff format`, otherwise the fixer undoes some
  line breaks of the formatter.
- Apply the formatter only to files you changed. Formatting the whole repository is a separate
  commit, with evidence that nothing changed in behaviour (`tools/check_translation.py` against
  the previous commit reports no deviation, the fingerprint regression tests and the full suite
  are green).
- Ruff and mypy are pinned from above in `pyproject.toml` on purpose; raising a pin is a
  deliberate change with a run behind it.
- Disable a rule in `pyproject.toml` with a reason at that place instead of a bare `# noqa`.
  A `noqa` line is for a local exception whose comment names the reason.
- The config-fingerprint regression tests must stay green. If one fails, a default or a config
  changed; find out why before touching the test.

## Branches, merging, pushing

- Work on a branch of your own (a worktree is fine). **Only one session merges into `master` and
  pushes**; everyone else delivers branches.
- Push only the commit on which the full suite has run. Commits that change documentation only
  (`docs/`, `CLAUDE.md`; no files under `src/`, `tests/` or
  `scripts/`) may be pushed after ruff and the denylist test. If `master` has moved in the
  meantime, check again and do not push.
- When in doubt, ask. If a finding changes existing results, report it first and fix it after
  confirmation. Do not change defaults, penalty weights, tolerances or configs that underpin
  findings on your own.

## Confidentiality

- Never commit names of partner companies, their customers' products or third-party persons.
  Use the established pseudonyms (application partner, consortium coordinator, software
  integration partner, associated partner). The mapping to real names lives in `local/`, which is
  git-ignored. Do not guess real names.
- Raw plant data and partner workbooks never go into the repository (`local/`, see
  `local.example`).
- `tests/test_denylist.py` checks `local/denylist.txt` (if present) against all tracked files.
  Run it before pushing; set `TWINLINK_LOCAL_DIR` when `local/` is not in your checkout.

## Provenance of results

- Commit a script **before** you run it. Use a clean working tree for any run that writes
  artifacts (`git/dirty` is logged).
- Never overwrite an artifact; use a new name. A corrected re-run gets a new name and the old
  one stays.
- Every artifact records the commit, the dirty flag, the config fingerprint and where the code
  was loaded from (`code/package_dir`, `code/interpreter`, `code/root_checked`). Artifact scripts
  that write their own provenance call `require_code_location(ROOT)`.
- Long runs (hours) belong on a local machine, not in cloud sessions, and are resumable with
  `--resume`.

## Methodology rules

- New studies generate their data with `DataSpec.new_study(...)` (preset version 2). The plain
  `DataSpec` defaults reproduce old data only.
- **Run-level splits, never window-level.** Learning curves are the primary result, 3-5 seeds,
  median and range. **Baselines before any architecture comparison.**
- **Always name the test set** (excited vs. constant-run holdout). Factors against persistence
  are meaningful only with the test set named.
- Rank claims only after checking against the resolution limit, with paired resolution over
  seeds (`evaluation/paired_resolution.py`) for every rank claim of a new study.
- Report time spans in seconds, not in sampling steps. A module in which no tensor operation has
  been executed produces no reportable number and is marked as not executed.
- Known pitfalls: R2 compresses the decisive range, compare in physical units; a return value
  made for an optimiser (a 1e6 sentinel) can poison a learned value function; do not confuse
  relative and absolute humidity; normal cubic metres are a mass unit.

## Code

Type hints throughout, frozen dataclasses for configs, ABCs for the core abstractions,
dependency injection, `src/` layout. A new architecture is added once in `models/registry.py`
and must not require a change in training or evaluation code. Scripts in `scripts/` import
modules and contain no domain logic; notebooks import modules and do not duplicate code.

## Documentation

- Language: English. The lab book `docs/04_stand_ap1.md` keeps its existing German entries; new
  entries are written in English.
- Lab-book entries are **append-only**; corrections are dated addenda. A new entry that is not
  yet merged carries the placeholder number "NN", replaced at the merge.
- In syntheses only numbers that appear in a finding **and** in an artifact; never numbers from
  conversations. Mark placeholders without a source as such.
- Translation passes must pass `tools/check_translation.py`.
