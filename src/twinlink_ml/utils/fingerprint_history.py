r"""Equivalence of the ``greybox_gab_placeholder.yaml`` fingerprints across the name cut.

Reason: separating the repo from its history (see ``docs/ARCHIV.md``, 2026-09-25) replaced
partner names in ``provenance.open_todos`` with pseudonyms. This free-text list was part
of the hashed config and therefore shifted ``ProcessConfig.fingerprint`` -- although not a
single number was changed. Since this cut,
:func:`twinlink_ml.simulation.config.without_documentary_provenance` excludes ``open_todos`` (and
``description``) from the hash; the three pairs below are therefore the LAST ones that
this cut produces -- future changes to documentary fields no longer move the
fingerprint.

Evidence of equivalence (one-off, see audit conversation 2026-09-25): for every pair,
``config_fingerprint(without_documentary_provenance(...))`` was computed both over the unchanged
archive config (with partner names, from the predecessor repo) and over the cleaned
config -- both sides returned the same value for each of the three variants (projection off/on,
and without override "as loaded"). This shows: the config pairs
denote the same numerical configuration, only the documentation differs.

This table is CLOSED. It contains exclusively the three pairs covered by the above
evidence -- a future change that moves the fingerprint gets
NO new entry here. Such a change is real (a number has changed)
and is to be reported by every guard that uses :func:`fingerprints_equivalent`, like
any other unexpected drift. See ``tests/test_fingerprint_history.py``, which enforces exactly
these three pairs -- no more, no fewer.
"""

from __future__ import annotations

PRE_CUT_FINGERPRINT_EQUIVALENTS: dict[str, str] = {
    # old (before the cut) -> new (from without_documentary_provenance onwards)
    "abbe8325f1326d80": "2acd06f38963c3af",  # greybox_gab_placeholder.yaml, as loaded
    "078572b3a7a0fb45": "6ecedc6bd993fb3d",  # ... sample_time_s=25.0, projection OFF
    "2d2679eeed716577": "bddca37f716f5cfc",  # ... sample_time_s=25.0, projection ON
}
"""Nur lesen, nie erweitern -- siehe Moduldocstring."""


def fingerprints_equivalent(a: str, b: str) -> bool:
    """``True`` if ``a``/``b`` are identical or equivalent according to the table.

    "Equivalent" means: the same numerical configuration, one side before, the other
    after the pseudonymisation cut (see module docstring). New runs always produce
    the new fingerprint -- the table is only used to ACKNOWLEDGE old, frozen
    artifact values, never for writing. Symmetric: the order of ``a``/``b``
    does not matter.
    """
    if a == b:
        return True
    return (
        PRE_CUT_FINGERPRINT_EQUIVALENTS.get(a) == b or PRE_CUT_FINGERPRINT_EQUIVALENTS.get(b) == a
    )
