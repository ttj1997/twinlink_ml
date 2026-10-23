"""Check of the sensor-value study: value of the measurement at F0's first criterion-(ii) candidate.

In ``sensor_value_v1`` the order statistic of the F0 per-pair margins (0.0478515625 kg/kg) held in
379 of 400 pairs at its verification (one pair is non-monotone), one short of the 380 of criterion
(ii); by the rule of the driver the candidate was raised to the next per-pair margin (0.059375),
where 382 hold. That rule raises F0's energy and with it the value of the measurement V. This
script quantifies the effect from the verification runs at the first candidate, which the run kept
in its cache (``artifacts/runs/_sensor_value_v1_cache.jsonl``, gitignored): V per pair at both F0
margins against F1 and F2 at their criterion-(ii) margins of the artifact. Descriptive; the
pre-registered reading stays the one at the raised margin.

    tools/run_here.sh scripts/check_sensor_value_f0_candidate.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree  # noqa: E402

NAME = "sensor_value_v1"
OUTPUT = RUNS_DIR / f"{NAME}_f0_candidate_check.json"


def main() -> int:
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists -- artifacts are never overwritten")
    require_clean_tree()
    artifact = RUNS_DIR / f"{NAME}.json.gz"
    cache = RUNS_DIR / f"_{NAME}_cache.jsonl"
    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload["results"]
    tried = results["F0"]["criteria"]["ii"]["tried"]
    first, raised = float(tried[0][0]), float(results["F0"]["criteria"]["ii"]["margin"])
    with cache.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline())["header"]
        if header["commit"] != payload["provenance"]["git/commit"]:
            raise SystemExit("cache and artifact come from different commits")
        entries = [json.loads(line) for line in handle]
    at_first = {
        (e["value"]["draw"], e["value"]["scenario"]): e["value"]
        for e in entries
        if e["key"].startswith("verify|F0|") and e["value"]["margin"] == first
    }
    tol = payload["violation_tolerance_kg_per_kg"]

    def by_pair(v: str) -> dict:
        return {(r["draw"], r["scenario"]): r for r in results[v]["criteria"]["ii"]["records"]}

    f0, f1, f2 = by_pair("F0"), by_pair("F1"), by_pair("F2")
    pairs = sorted(f1)
    if sorted(at_first) != pairs:
        raise SystemExit(
            f"cache holds {len(at_first)} of {len(pairs)} pairs at the first candidate"
        )

    def stats(values: list[float]) -> dict:
        a = np.asarray(values)
        return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}

    out = {
        "provenance": provenance("not used"),
        "source": {
            "artifact": str(artifact.name),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "cache_header": header,
        },
        "f0_first_candidate": first,
        "f0_raised_margin": raised,
        "holding_at_first_candidate": int(
            sum(r["max_violation"] <= tol for r in at_first.values())
        ),
        "value_vs_F1_at_first_candidate": stats(
            [(at_first[p]["energy_kwh"] - f2[p]["energy_kwh"]) / f1[p]["energy_kwh"] for p in pairs]
        ),
        "value_vs_F1_at_raised_margin": stats(
            [(f0[p]["energy_kwh"] - f2[p]["energy_kwh"]) / f1[p]["energy_kwh"] for p in pairs]
        ),
        "f0_energy_raised_over_first": stats(
            [f0[p]["energy_kwh"] / at_first[p]["energy_kwh"] - 1.0 for p in pairs]
        ),
    }
    OUTPUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("provenance", "source")}, indent=1))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
