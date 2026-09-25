# local/ -- confidential, gitignored configuration

This directory is a template. In a real working copy (never in a cloud session
without it being deliberately transferred), copy it to `local/`:

```
cp -r local.example local
```

Then fill in:

- `mapping.md` -- the real names behind the pseudonyms used throughout the repo
  (application partner, consortium coordinator, software integration partner,
  associated partner, and any third-party contact names).
- `denylist.txt` -- the same real names, one per line. `tests/test_denylist.py`
  greps every git-tracked file against this list before a commit is trusted; if
  `local/` is absent (the normal case in a cloud session), that test is skipped.

`local/` is gitignored and must never be committed. It does not sync over
GitHub -- see `docs/07_synchronisation.md` for how to transfer it separately.
