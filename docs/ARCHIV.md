# Archive

This repository's history was cut on **2026-09-25**. The predecessor repository (198
commits) is kept permanently as a local archive:

```
T:\ResearchProjects\BMWE TwinLink (ML)\Code\twinlink_ml (4)\twinlink_ml
```

**This archive must never be deleted.** Every `git/commit` value recorded in
`artifacts/runs/*.jsonl`/`.json` (provenance field, see
`twinlink_ml.utils.tracking.JsonlExperimentLogger`) refers to a commit in the archive's
history, not in this repository's fresh history. Without the archive, those commit hashes
are unresolvable and the affected records lose their code-provenance trail.

## Why the cut

A confidentiality audit (2026-09-25) found that the application partner's company name,
the consortium partners' company names, one spelled-out third-party product name, and one
third-party contact name were committed throughout the repository's history (config
comments, docstrings, docs, commit messages) — not only in raw plant data, which was
already kept outside the repository. Rewriting the existing history in place was rejected:
records reference it by commit hash (see above), so the old repository stays intact as an
archive, and this repository starts a fresh history from a pseudonymized working copy.

## What changed

Company/person names were replaced by role-based pseudonyms throughout code, comments,
docstrings, config comments and docs (not in existing artifacts — none contained a literal
partner name as a value). Plant designations, apparatus numbers, instrument tags and
operating values were intentionally left unchanged; only the identity of companies and
persons was pseudonymized. See `local.example/README.md` for the pseudonym mapping
mechanism and `local.example/mapping.md` for the pseudonym scheme itself (the real names
live in `local/`, gitignored, not in this repository).

## Config-fingerprint shift

One of the pseudonymized fields, `provenance.open_todos` in
`conf/process/greybox_gab_placeholder.yaml`, is not a comment but a real YAML value that
was part of the hashed payload behind `ProcessConfig.fingerprint`. Replacing the partner
name in it therefore moved the fingerprint, even though no number changed. Since this cut,
`twinlink_ml.simulation.config.without_documentary_provenance` excludes
`provenance.description`/`provenance.open_todos` from the hash — a documentation edit
cannot move the fingerprint again. See the `ProcessConfig.fingerprint`/
`without_documentary_provenance` docstrings in `src/twinlink_ml/simulation/config.py`.

**Equivalence proof (2026-09-25):** for each pair below, `config_fingerprint(
without_documentary_provenance(...))` was computed once over the unmodified archive config
(with the real partner name) and once over the pseudonymized config in this repository.
Both sides produced the identical value for all three variants (projection off, projection
on, as-loaded with no override) — proving the pairs denote the same numeric configuration,
differing only in documentation text.

| Value before the cut | Value after the cut | `greybox_gab_placeholder.yaml` variant |
|---|---|---|
| `abbe8325f1326d80` | `2acd06f38963c3af` | as-loaded, no override (VA2 anchor point, `twinlink_ml.rl.env`) |
| `078572b3a7a0fb45` | `6ecedc6bd993fb3d` | `sample_time_s=25.0`, `enforce_gas_velocity_projection=False` |
| `2d2679eeed716577` | `bddca37f716f5cfc` | `sample_time_s=25.0`, `enforce_gas_velocity_projection=True` |

This table is **closed** — it contains exactly these three pairs, enforced by
`tests/test_fingerprint_history.py`. A future fingerprint shift gets no new entry here: it
means a real configuration change, and every guard using
`twinlink_ml.utils.fingerprint_history.fingerprints_equivalent` must report it like any
other unexpected drift. The table only recognizes old, frozen artifact values as
equivalent to the current one; new runs always produce the new fingerprint, never the old
one.

Roughly 18 scripts (and `src/twinlink_ml/rl/env.py`) hardcode expected fingerprints as
drift guards before running or evaluating; all of them were updated to compare via
`fingerprints_equivalent` instead of `==`/`!=`, so they keep recognizing pre-cut artifacts
(anchor files, checkpoints) without weakening the guard against genuine drift. Continuation
workers that append to an existing shared JSONL series (e.g.
`run_arch_learning_curve_worker.py` → `artifacts/runs/arch_learning_curve_projected.jsonl`)
now produce the new fingerprint on any future run; the file's existing pre-cut records keep
their old value, recognized as equivalent by the same helper.

## Documentation

`docs/04_stand_ap1.md`'s existing entries (German, append-only lab book) still name the
pre-cut fingerprint values as originally recorded — see its "Fingerprint-Historie" section
(Befund 46) for the full chain including this cut's addendum. Values are never edited in
place there; only this file and the closed equivalence table are authoritative for what a
pre-cut value corresponds to today.
