"""Haelt die Fingerprint-Aequivalenztabelle exakt auf den drei nachgewiesenen Paaren fest.

Siehe :mod:`twinlink_ml.utils.fingerprint_history` (Audit 2026-09-25) fuer den Nachweis.
Die Tabelle ist geschlossen; ein Test, der das nicht erzwingt, waere nur so lange richtig,
bis jemand in gutem Glauben einen vierten Eintrag ergaenzt, der nicht durch denselben
Beleg gedeckt ist.
"""

from __future__ import annotations

from twinlink_ml.utils.fingerprint_history import (
    PRE_CUT_FINGERPRINT_EQUIVALENTS,
    fingerprints_equivalent,
)

PROVEN_PAIRS = {
    "abbe8325f1326d80": "2acd06f38963c3af",
    "078572b3a7a0fb45": "6ecedc6bd993fb3d",
    "2d2679eeed716577": "bddca37f716f5cfc",
}


def test_table_contains_exactly_the_proven_pairs() -> None:
    assert PRE_CUT_FINGERPRINT_EQUIVALENTS == PROVEN_PAIRS


def test_identical_fingerprints_are_equivalent() -> None:
    assert fingerprints_equivalent("abc123", "abc123")


def test_known_pairs_are_equivalent_in_either_order() -> None:
    for old, new in PROVEN_PAIRS.items():
        assert fingerprints_equivalent(old, new)
        assert fingerprints_equivalent(new, old)


def test_unrelated_fingerprints_are_not_equivalent() -> None:
    assert not fingerprints_equivalent("abbe8325f1326d80", "078572b3a7a0fb45")
    assert not fingerprints_equivalent("deadbeefdeadbeef", "2acd06f38963c3af")
