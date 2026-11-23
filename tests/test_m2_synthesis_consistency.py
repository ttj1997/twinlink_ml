"""Consistency check of the M2 synthesis (docs/08): version 5f at its tag, version 6 in the tree.

Version 5f stays retrievable under the tag ``docs08-v5f`` and has to keep its 455 confirmed claims;
version 6 has to confirm completely (no deviating claim, no numeric token without a check). The test
of version 5f FAILS when the tag is missing: the tag belongs to the repository like the denylist
does (it is set locally on the branch ``synthesis-v6`` and pushed with its merge).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_m2_synthesis_v6 as version6  # noqa: E402

TAG = "docs08-v5f"
CLAIMS = re.compile(r"^claims: (\d+) checked, (\d+) confirmed, (\d+) deviating$", re.MULTILINE)
COVERAGE = re.compile(r"^coverage: (\d+) numeric tokens without a check", re.MULTILINE)
EXEMPTIONS = re.compile(
    r"^exemptions: (\d+) declared, (\d+) not in text or not verified$", re.MULTILINE
)


def run_check(version: str) -> tuple[int, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONIOENCODING"] = "utf-8"
    done = subprocess.run(
        [
            sys.executable,
            "-u",
            str(SCRIPTS / "check_m2_synthesis_consistency.py"),
            "--version",
            version,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def counts(output: str) -> dict[str, int]:
    claims = CLAIMS.search(output)
    coverage = COVERAGE.search(output)
    exemptions = EXEMPTIONS.search(output)
    assert claims and coverage and exemptions, output[-2000:]
    return {
        "checked": int(claims.group(1)),
        "confirmed": int(claims.group(2)),
        "deviating": int(claims.group(3)),
        "uncovered": int(coverage.group(1)),
        "exemptions": int(exemptions.group(1)),
        "exemption_problems": int(exemptions.group(2)),
    }


def test_version_5f_at_its_tag_still_confirms_455_claims() -> None:
    tag = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{TAG}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if tag.returncode != 0:
        pytest.fail(
            f"git tag {TAG!r} is missing: version 5f of docs/08_synthese_m2.md is read from it. "
            f"Fetch the tags, or set it on the commit of version 5f (git tag {TAG} 803ee5f)."
        )
    code, output = run_check("5f")
    found = counts(output)
    assert found == {
        "checked": 455,
        "confirmed": 455,
        "deviating": 0,
        "uncovered": 0,
        "exemptions": 15,
        "exemption_problems": 0,
    }, output[-2000:]
    assert code == 0


def test_version_6_confirms_completely() -> None:
    code, output = run_check("6")
    found = counts(output)
    assert found["deviating"] == 0, output[-3000:]
    assert found["uncovered"] == 0, output[-3000:]
    assert found["exemption_problems"] == 0, output[-3000:]
    assert found["confirmed"] == found["checked"] > 455
    assert "verified references outside their allowed values: []" in output
    assert code == 0


class FakeReport:
    """The two lists of the report that a text claim writes to."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.fragments: list[str] = []


def text_claims(sentence: str) -> list[dict[str, Any]]:
    report = FakeReport()
    version6.Sources(report, version6.normalise(sentence)).passage(
        "Dynamic control:", "(Befund 105).", befunde=(105,)
    )
    return report.results


def test_a_text_claim_finds_a_number_that_is_not_in_the_cited_entry() -> None:
    right = text_claims(
        "Dynamic control: an MPC adds 0.55 % in the median, at most 1.50 % (Befund 105)."
    )
    assert [row["written"] for row in right] == ["0.55", "1.50"]
    assert {row["status"] for row in right} == {"confirmed"}
    wrong = text_claims(
        "Dynamic control: an MPC adds 0.56 % in the median, at most 1.50 % (Befund 105)."
    )
    verdicts = {row["written"]: row["status"] for row in wrong}
    assert verdicts == {"0.56": "DEVIATING", "1.50": "confirmed"}


def test_a_passage_that_is_not_in_the_text_is_reported_not_skipped() -> None:
    report = FakeReport()
    sources = version6.Sources(report, "some other text")
    sources.passage("Dynamic control:", "(Befund 105).", befunde=(105,))
    assert [row["status"] for row in report.results] == ["DEVIATING"]
    assert report.results[0]["fragment_in_text"] is False


def test_tokens_and_their_lookup() -> None:
    passage = (
        "Befund 105 (section 7.3, rule 6, 2026-11-23): 8.93 % in 5 of 5 seeds, "
        "three classes, C2 and 1.7e-13 K"
    )
    assert version6.tokens_of(passage) == ["1.7", "13", "5", "8.93", "three"]
    assert version6.written_in("8.93", "a saving of 8.93 % in the median")
    assert not version6.written_in("8.9", "a saving of 8.93 % in the median")
    assert not version6.written_in("93", "a saving of 8.93 % in the median")
    assert version6.written_in("five", "in 5 of 5 seeds")
    assert version6.written_in("1.7", "at most 1.7e-13 K")
