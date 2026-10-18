"""Exercise isolated workers and check equivalence of the stress workloads."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("stim")

from benchmarks.benchmark_stim import build_circuits, verify_tableau
from benchmarks.stress_qubits import compare_results, gate_stream


@pytest.mark.parametrize("n", [1, 2, 63, 64, 65, 129])
def test_stress_gate_stream_matches_full_stim_tableau(n):
    import stabilizer
    import stim

    gates = list(gate_stream(n, 1000, 123))
    a, b = build_circuits(n, gates, stabilizer, stim)
    rust, reference = stabilizer.Tableau(n), stim.TableauSimulator()
    reference.set_num_qubits(n)
    rust.apply_circuit(a)
    reference.do(b)
    verify_tableau(rust, reference)


@pytest.mark.parametrize("mode", ["gates", "measurement"])
def test_isolated_comparison_workers(mode, tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "stress.json"
    run = subprocess.run(
        [sys.executable, str(root / "benchmarks/stress_qubits.py"), "--qubits", "1", "65",
         "--backends", "stabilizer", "stim", "--mode", mode, "--gates", "100",
         "--repeats", "3", "--json", str(output)],
        cwd=root, capture_output=True, text=True, timeout=30, env=os.environ.copy(),
    )
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(output.read_text())
    assert len(report["results"]) == 4
    assert len(report["comparisons"]) == 2
    for result in report["results"]:
        assert result["status"] == "success", result
        assert len(result["gate_samples_seconds"]) == 3
        assert result["peak_rss_bytes"] > 0
        assert result["worker_swap_bytes"] == 0
        assert (result["measurement_seconds"] is not None) == (mode == "measurement")


def test_comparison_rejects_different_gate_streams():
    row = {"status": "success", "qubits": 65}
    with pytest.raises(RuntimeError, match="gate streams differ"):
        compare_results([{**row, "backend": "stabilizer", "gate_sha256": "a"},
                         {**row, "backend": "stim", "gate_sha256": "b"}])
