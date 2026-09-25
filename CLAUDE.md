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

## Provenance

- Commit scripts BEFORE running them; clean working tree before any run that writes
  artifacts (`git/dirty` is logged).
- Never overwrite existing artifacts; use a new name.
- The config-fingerprint regression tests must stay green.

## Methodology

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
ruff and pytest before every commit.

## Known pitfalls (one line each)

- R2 compresses the decisive range; compare in physical units.
- A return value fit for an optimizer (e.g. a 1e6 sentinel) can poison a learned value
  function.
- Do not confuse relative and absolute humidity; normal cubic metres are a mass unit.
- Factors against persistence are only meaningful with the test set named.

## Long runs

Long runs (hours) belong on a local machine, not in cloud sessions; sweeps are resumable
with `--resume`.
