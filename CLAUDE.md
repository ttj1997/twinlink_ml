# twinlink_ml

Part of TwinLink AP1 (BMWE-funded digital-twin research project): comparing RL and
RNN/xLSTM/Transformer architectures for controlling a fluidized-bed spray granulation
process. The core is a grey-box simulator of that process, used to generate training data
and to evaluate control strategies before any real-plant deployment. See `docs/` for the
lab book (`04_stand_ap1.md`), process setup, and synthesis documents.

## Confidentiality (first, most important)

- Never commit names of partner companies, their customers' products, or third-party
  persons. Use the pseudonyms already established in the repo (application partner,
  consortium coordinator, software integration partner, associated partner); the mapping
  to real names is in `local/`, which is gitignored and absent in cloud sessions. Do not
  guess real names.
- Raw plant data is never in the repository.
- `tests/test_denylist.py` checks `local/denylist.txt` (if present) against all
  git-tracked files before a commit is trusted.

## Working rules

- When in doubt, ask instead of deciding.
- If a finding changes existing results: report first, fix after confirmation.
- No unilateral changes to defaults, penalty weights, tolerances or configs that underpin
  findings.
- Only one session merges into master and pushes; other sessions deliver branches.
- Push only the commit on which the full suite has run. Commits that change documentation
  only (docs/, CLAUDE.md; no files under src/, tests/ or scripts/) may be pushed after ruff
  and the denylist test. If master has moved in the meantime: check again, do not push.
- Edits by script: chain every step with `&&` (or set -e) so that a failing edit stops the
  sequence. After any scripted edit to docs, verify each expected change with grep (the new
  heading, the finding number, the decision paragraph, the docs/12 row, the README paragraph)
  and report the grep result. A change counts as done only when its grep check passes. Never
  report a merge as complete from the exit status of later commands.
- Before removing a worktree: list ignored and untracked content (`git status --ignored
  --porcelain`) and copy everything that is not reproducible in seconds (checkpoints, job logs,
  caches of long runs, unversioned reports) to `C:/twinlink_runs/<branch>/` (reports to
  `local/reports/`). Report the list and the copy target. Remove the worktree only after the
  copy is verified (file count and total size match).
- Budget rules in study prompts refer to wall-clock time.
- Parallel studies name their document `docs/NN_<topic>.md`; the managing session assigns the
  number at merge.

## Provenance

- Commit scripts BEFORE running them; clean working tree before any run that writes
  artifacts (`git/dirty` is logged).
- Never overwrite existing artifacts; use a new name.
- The config-fingerprint regression tests must stay green.
- Environment: one `.venv` in the main checkout with every extra of `pyproject.toml`
  (`pip install -e ".[torch,sklearn,plant-data,train,rl,optimal-control,surrogate,sensitivity,dev]"`, includes gymnasium and,
  with `optimal-control`, CasADi 3.8.1 for `src/twinlink_ml/headroom/`; `sensitivity` carries SALib for
  `src/twinlink_ml/sensitivity/`; `plant-data` carries openpyxl and h5py,
  the latter for the Dyssol file of the partner model). Use it, not
  the system interpreter (its editable install points at an old, unrelated checkout).
  Check: `python -c "import twinlink_ml; print(twinlink_ml.__file__)"` shows this repo's `src/`.
- Code location: every new artifact records `code/package_dir`, `code/interpreter` and
  `code/root_checked` (`twinlink_ml.utils.code_location`, via `environment_fingerprint()` and
  `scripts/verify/_common.provenance()`). If the loaded `twinlink_ml` is not
  `<root>/src/twinlink_ml` of the repository or worktree the run belongs to, the run aborts.
  Artifact scripts that write their own provenance call `require_code_location(ROOT)`.
- Worktrees: an editable install points at one directory only, so a worktree must load its own
  code. Start every run there with `tools/run_here.sh <python args>` (changes into the worktree
  root, sets `PYTHONPATH=<worktree>/src`, uses the main checkout's `.venv`); detached drivers do
  the same (`cd <worktree>; PYTHONPATH=<worktree>/src <venv python> ...`). Never re-run
  `pip install -e .` from a worktree (it would redirect every other run on the machine).

## Methodology

- New studies generate their data with `DataSpec.new_study(...)` (preset version 2: planned future
  set points, own disturbance process for the solid fraction, per-run cap of the gas velocity at
  `u_t` of the initial diameter with the projection kept as a counted guard; review D1/E3,
  Befund 86) and no other data configuration. The preset version is part of the fingerprints,
  records and provenance. The defaults of `DataSpec` stay at the behaviour up to Befund 81 and exist only
  to reproduce old data. Paired resolution over seeds
  (`evaluation/paired_resolution.py`) for every rank claim of a new study; the older
  `resolution.py` only to reproduce old findings.

- Run-level splits, never window-level.
- Learning curves as primary result; 3-5 seeds, median and range.
- Baselines before any architecture comparison.
- Always name the test set (excited vs. constant-run holdout).
- Rank claims only after checking against the resolution limit.

## Documentation

- Language: English. The lab book `docs/04_stand_ap1.md` keeps its existing entries in
  German; new entries are written in English.
- Lab book entries are append-only; corrections as dated addenda.
- In syntheses, only numbers that appear in a finding and an artifact -- never numbers
  from conversations.
- Label placeholders without a source as such.
- Translation passes must pass `tools/check_translation.py`.

## Code

Type hints, frozen dataclasses for configs, ABCs, dependency injection, `src/` layout. Run
ruff and pytest before every commit. Apply the formatter only to files you changed; format the
whole repo only as a separate commit with the evidence of the FM2 kind (`tools/check_translation.py`
against the previous commit reports no deviation, fingerprint regression tests and full suite green).

## Known pitfalls (one line each)

- R2 compresses the decisive range; compare in physical units.
- A return value fit for an optimizer (e.g. a 1e6 sentinel) can poison a learned value
  function.
- Do not confuse relative and absolute humidity; normal cubic metres are a mass unit.
- Factors against persistence are only meaningful with the test set named.

## Long runs

Long runs (hours) belong on a local machine, not in cloud sessions; sweeps are resumable
with `--resume`.
