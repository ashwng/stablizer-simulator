import json

import pytest

pytest.importorskip("stim")

from benchmarks.full_benchmark import build_plan, main


def test_full_plan_covers_scenarios_without_duplicates():
    plan = build_plan()
    assert len({row["id"] for row in plan}) == len(plan)
    assert {row["family"] for row in plan} == {"gates", "measurement", "protocol", "lifecycle", "shots", "stress"}
    assert any(row["parameters"].get("kind") == "parity" for row in plan)
    assert any(row["parameters"].get("kind") == "interleaved" for row in plan)
    assert any(row["parameters"].get("n") == 4096 and row["parameters"].get("count") == 4096
               for row in plan)


def test_checkpoint_and_resume(tmp_path):
    args = ["--output", str(tmp_path / "run"), "--repeats", "1", "--limit", "2"]
    first = main(args)
    assert len(first) == 2
    assert all(row["status"] == "success" for row in first)
    resumed = main([*args, "--resume"])
    assert first == resumed
    root = tmp_path / "run"
    assert len((root / "results.jsonl").read_text().splitlines()) == 2
    assert json.loads((root / "progress.json").read_text())["completed"] == 2
    assert (root / "summary.md").exists()
