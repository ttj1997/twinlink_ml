# Synchronization between working machines

Workflow, not subject matter -- concerns who has which state on which machine (Windows
network drive, Linux laptop, cloud sessions), not the model or the findings themselves.

GitHub is the shared remote for this repository, replacing the earlier Zip-copy and
`git bundle` workflow (see `docs/ARCHIV.md` for why this repository's history starts fresh
here rather than continuing the old one).

## Order rule

**`git pull` before working, `git push` after working.** Starting work on a machine
without first pulling the current remote state risks exactly what caused divergent copies
under the old Zip workflow: two histories that continue independently until someone has to
merge them.

## `local/` does NOT sync over GitHub

`local/` (the real names behind the pseudonyms, and `denylist.txt`) is gitignored --
`git push`/`git pull` never touch it. A machine or cloud session that needs it (to run
`tests/test_denylist.py` meaningfully, or to resolve a pseudonym back to a real name) needs
it transferred separately, outside GitHub. Its absence is the normal case in cloud
sessions: everything else in this repository works without it, just with that one test
skipped. Never commit it, and never paste its contents into a cloud session's context.

## `safe.directory` on the Windows network drive

Git refuses all operations on the network drive (`\\...\ou-t2a-team$\...`) with
`fatal: detected dubious ownership` unless an exception is set:

    git config --global --add safe.directory '*'

Without this, `twinlink_ml.utils.tracking.environment_fingerprint()` returns no
`git/commit`, only `"unknown"` -- every record produced on that drive is then
provenance-incomplete, regardless of anything else in this file. Set this once per
machine, before running anything else here.

## Commit scripts before running them

**Before every run: working tree clean, the script about to run already committed.**
Reason for this rule (see `docs/04_stand_ap1.md`): several Befund-52/53/54 artifacts
carried `git/dirty=True` because the script had not been committed yet at run time -- the
logged `git/commit` was then the parent state BEFORE the script's first version, not the
state that actually ran. `git diff HEAD` does not see a brand-new, untracked script at all
(it only shows changes to already-tracked files) -- `git/diff_digest` is then the digest of
an EMPTY diff and does not make the run identifiable after the fact despite
`dirty=True`. Commit first, then run -- otherwise the provenance gap is real, not just a
caveat.

## Pull

    git pull

Fails loudly on a genuine conflict instead of silently overwriting -- read what changed
(`git log`, `git diff`) before resolving. If unclear which side is the complete one: ask,
do not guess and do not merge automatically.

## Push

    git push

Push only after `ruff`/`pytest` are green and the working tree is clean (see
`docs/04_stand_ap1.md`'s provenance rules and `CLAUDE.md`). A dirty push spreads an
unreproducible state to every other machine that pulls next.
