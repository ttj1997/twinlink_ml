"""Re-run of the architecture comparison (docs/15): arms, cross-evaluation variants, jobs, gate.

The smoke tests run the worker on a tiny pool with the baselines only (no torch training).
"""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import check_rerun_gate as gate  # noqa: E402
import run_arch_rerun_sweep as sweep  # noqa: E402
import run_arch_rerun_worker as worker  # noqa: E402


def test_arm_specs() -> None:
    legacy = worker.legacy_spec()
    a0, a1, n = (worker.arm_spec(arm) for arm in worker.ARMS)
    assert a0 == legacy
    assert (legacy.future_actions, legacy.solid_fraction_process) == ("realised", "diurnal")
    assert a1.future_actions == "planned" and a1.solid_fraction_process == "diurnal"
    assert n.is_new_study_preset
    for spec in (a1, n):
        for field in ("n_train_max", "horizon", "target_names", "constant_run_fraction"):
            assert getattr(spec, field) == getattr(legacy, field)
    with pytest.raises(ValueError):
        worker.arm_spec("A2")


def test_evaluation_variants() -> None:
    a0 = worker.evaluation_variants("A0")
    a1 = worker.evaluation_variants("A1")
    n = worker.evaluation_variants("N")
    assert sorted((p, w) for p, w, _ in a0) == [("own", "planned"), ("own", "realised")]
    assert [w for p, w, primary in a0 if primary] == ["realised"]
    assert [w for p, w, primary in a1 if primary] == ["planned"]
    assert len(n) == 4 and sum(primary for *_, primary in n) == 1
    assert [(p, w) for p, w, primary in n if primary] == [("own", "planned")]
    assert worker.expected_record_count("N", [8, 16, 32, 64], 8) == 8 * 4 * 4 * 2
    assert worker.expected_record_count("A0", [8, 64], 8) == 8 * 2 * 2 * 2


def test_jobs_follow_the_decided_budgets_and_cost_order() -> None:
    jobs = sweep.build_jobs(["A0", "A1", "N"], [1, 2, 3, 4, 5], [8, 64], [8, 16, 32, 64])
    assert len(jobs) == 3 * 5 * 2
    assert {job.sizes for job in jobs if job.arm != "N"} == {(8, 64)}
    assert {job.sizes for job in jobs if job.arm == "N"} == {(8, 16, 32, 64)}
    costs = [job.cost_s for job in jobs]
    assert costs == sorted(costs, reverse=True)
    assert jobs[0].group == "transformer" and jobs[0].arm == "N"
    transformer = next(j for j in jobs if j.group == "transformer")
    others = next(j for j in jobs if j.group == "others")
    assert transformer.n_methods == 1 and transformer.skip_baselines
    assert others.n_methods == 4 + 3
    # core-hours of the decided cut: per seed 7.8 (A0) + 7.8 (A1) + 14.9 (N), five seeds, about 152
    total = sum(job.cost_s - 60.0 for job in jobs) / 3600.0
    assert 150.0 < total < 155.0


def _smoke(arm: str, tmp_path: Path, seed: int = 1) -> list[dict]:
    out = tmp_path / f"{arm}.jsonl"
    arguments = worker.parse_arguments(
        [
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--output",
            str(out),
            "--smoke",
            "--sizes",
            "4",
            "--architectures",
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        written = worker.run(arguments)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written == len(records) == worker.expected_record_count(arm, [4], 3)
    return records


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict]]:
    root = tmp_path_factory.mktemp("rerun")
    return {arm: _smoke(arm, root) for arm in worker.ARMS}


def _primary(smoke: dict[str, list[dict]], arm: str, label: str, test_set: str) -> dict:
    return next(
        r
        for r in smoke[arm]
        if r["params"]["model/label"] == label
        and r["params"]["test_set"] == test_set
        and r["params"]["eval/primary"]
    )


def test_smoke_records_are_tagged(smoke: dict[str, list[dict]]) -> None:
    n = smoke["N"]
    assert {(r["params"]["eval/pool"], r["params"]["eval/window"]) for r in n} == {
        ("own", "realised"),
        ("own", "planned"),
        ("legacy", "realised"),
        ("legacy", "planned"),
    }
    assert all(r["params"]["rerun/arm"] == "N" for r in n)
    primary = [r for r in n if r["params"]["eval/primary"]]
    assert len(primary) == 3 * 2 and all(r["params"]["eval/pool"] == "own" for r in primary)
    assert all(r["params"]["code/package_dir"] == "src/twinlink_ml" for r in n)
    assert smoke["A1"][0]["params"]["window/future_actions"] == "planned"
    assert "window/future_actions" not in smoke["A0"][0]["params"]


def test_arms_share_the_trajectories_and_n_changes_the_pool(
    smoke: dict[str, list[dict]],
) -> None:
    a0 = _primary(smoke, "A0", "persistence", "excited")
    a1 = _primary(smoke, "A1", "persistence", "excited")
    for name in gate.FINGERPRINTS:
        assert a0["params"][name] == a1["params"][name]
    # persistence reads the history only: identical in A0 and A1
    worst, _, count = gate.max_relative_deviation(a0["metrics"], a1["metrics"])
    assert count > 10 and worst == 0.0
    n = _primary(smoke, "N", "persistence", "excited")
    assert n["params"]["excitation/fingerprint"] != a0["params"]["excitation/fingerprint"]
    # the legacy pool inside arm N is the pool of A0: same persistence error
    legacy_eval = next(
        r
        for r in smoke["N"]
        if r["params"]["model/label"] == "persistence"
        and r["params"]["test_set"] == "excited"
        and r["params"]["eval/pool"] == "legacy"
        and r["params"]["eval/window"] == "realised"
    )
    assert (
        legacy_eval["params"]["eval/pool_excitation_fingerprint"]
        == a0["params"]["excitation/fingerprint"]
    )
    worst, _, count = gate.max_relative_deviation(legacy_eval["metrics"], a0["metrics"])
    assert count > 10 and worst < 1e-9


def test_gate_logic(smoke: dict[str, list[dict]]) -> None:
    arms = {arm: smoke[arm] for arm in worker.ARMS}
    old = copy.deepcopy(smoke["A0"])  # pretend the old records are the A0 records
    report = gate.run_checks(old, arms, 1e-6)
    assert report["passed"], report["failures"]
    broken = copy.deepcopy(old)
    for r in broken:
        if r["params"]["model/label"] == "static":
            key = next(k for k in r["metrics"] if k.startswith("rmse_run_median_physical"))
            r["metrics"][key] *= 1.01
    assert not gate.run_checks(broken, arms, 1e-6)["passed"]
    wrong = copy.deepcopy(old)
    for r in wrong:
        r["params"]["split/digest"] = "x"
    assert not gate.run_checks(wrong, arms, 1e-6)["passed"]


def test_incomplete_job_is_detected(tmp_path: Path) -> None:
    job = sweep.build_jobs(["A0"], [1], [8], [8])[0]
    path = tmp_path / "j.jsonl"
    assert not sweep.job_complete(path, job)
    path.write_text("\n".join(["{}"] * (job.expected_records - 1)) + "\n", encoding="utf-8")
    assert not sweep.job_complete(path, job)
    path.write_text("\n".join(["{}"] * job.expected_records) + "\n", encoding="utf-8")
    assert sweep.job_complete(path, job)


def test_smoke_with_a_trained_network_saves_a_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "gru.jsonl"
    arguments = worker.parse_arguments(
        [
            "--arm", "N", "--seed", "1", "--output", str(out), "--smoke", "--sizes", "4",
            "--architectures", "gru", "--skip-baselines", "--net-max-epochs", "2",
            "--checkpoint-dir", str(tmp_path / "ckpt"),
        ]
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        written = worker.run(arguments)
    assert written == worker.expected_record_count("N", [4], 1)
    assert (tmp_path / "ckpt" / "gru_N_seed1_n4.pt").exists()
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(r["params"]["checkpoint/path"].endswith("gru_N_seed1_n4.pt") for r in records)


def test_gate_accepts_the_documented_fingerprint_history(smoke: dict[str, list[dict]]) -> None:
    arms = {arm: smoke[arm] for arm in worker.ARMS}
    old = copy.deepcopy(smoke["A0"])
    for r in old:
        r["params"]["process/fingerprint"] = "2d2679eeed716577"  # same configuration before the cut
        r["params"].pop("measurement/digest", None)  # not recorded in the old records
    for r in arms["A0"]:
        r["params"]["process/fingerprint"] = "bddca37f716f5cfc"
    report = gate.run_checks(old, arms, 1e-6)
    process_failures = [f for f in report["failures"] if "process/fingerprint" in f]
    assert not process_failures and not [f for f in report["failures"] if "measurement" in f]


def test_gate_checks_the_preset_version_and_the_projection_guard(
    smoke: dict[str, list[dict]],
) -> None:
    old = copy.deepcopy(smoke["A0"])
    arms = {arm: copy.deepcopy(smoke[arm]) for arm in worker.ARMS}
    prim = [r for r in arms["N"] if r["params"]["eval/primary"]]
    assert all(r["params"]["data/new_study_version"] == 2 for r in prim)
    assert all(r["params"]["pool/projected_steps"] == 0 for r in prim)
    assert all(r["params"]["holdout_pool/projected_steps"] == 0 for r in prim)
    assert gate.run_checks(old, arms, 1e-6)["passed"]

    def failures(mutate) -> list[str]:
        changed = {arm: copy.deepcopy(smoke[arm]) for arm in worker.ARMS}
        for r in changed["N"]:
            mutate(r["params"])
        return gate.run_checks(old, changed, 1e-6)["failures"]

    assert any(
        "preset version" in f for f in failures(lambda p: p.update({"data/new_study_version": 1}))
    )
    assert any("preset version" in f for f in failures(lambda p: p.pop("data/new_study_version")))
    assert any(
        "gas-velocity cap" in f
        for f in failures(lambda p: p.update({"data/gas_velocity_cap": "none"}))
    )
    assert any(
        "pool/projected_steps is 3" in f
        for f in failures(lambda p: p.update({"pool/projected_steps": 3}))
    )
    assert any(
        "holdout_pool/projected_steps is 1" in f
        for f in failures(lambda p: p.update({"holdout_pool/projected_steps": 1}))
    )
    assert any(
        "pool/projected_steps is None" in f
        for f in failures(lambda p: p.pop("pool/projected_steps"))
    )
