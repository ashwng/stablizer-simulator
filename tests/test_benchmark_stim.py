"""Cross-backend checks for circuit translation and benchmark reporting."""

import gc
import json

import pytest

stim = pytest.importorskip("stim")
import stabilizer

from benchmarks.benchmark_stim import (
    benchmark_case,
    build_circuits,
    generate_gates,
    main,
    verify_tableau,
)


@pytest.mark.parametrize("n", [1, 2, 63, 64, 65, 129])
@pytest.mark.parametrize("workload", ["h", "s", "cx", "mixed"])
def test_signed_tableaus_match(n, workload):
    if n == 1 and workload == "cx":
        return
    gates = generate_gates(n, 7 * n + 1, "mixed", 8)
    gates += generate_gates(n, 257, workload, 9)
    a, b = build_circuits(n, gates, stabilizer, stim)
    rust = stabilizer.Tableau(n)
    reference = stim.TableauSimulator()
    reference.set_num_qubits(n)
    rust.apply_circuit(a)
    reference.do(b)
    verify_tableau(rust, reference)


def test_deterministic_generation_and_distinct_cnot_targets():
    gates = generate_gates(65, 1000, "mixed", 123)
    assert gates == generate_gates(65, 1000, "mixed", 123)
    assert gates != generate_gates(65, 1000, "mixed", 124)
    assert len(gates) == 1000
    assert {g[0] for g in gates} == {"h", "s", "cx"}
    assert all(g[1] != g[2] for g in gates if g[0] == "cx")


def test_verification_catches_sign_errors_and_idle_qubits():
    rust = stabilizer.Tableau(65)
    reference = stim.TableauSimulator()
    reference.set_num_qubits(65)
    reference.z(64)
    with pytest.raises(RuntimeError, match="tableau mismatch at X64"):
        verify_tableau(rust, reference)


@pytest.mark.parametrize("n", [65, 257, 1025])
def test_persistent_columns_across_short_calls_and_inspection(n):
    rust = stabilizer.Tableau(n)
    reference = stim.TableauSimulator()
    reference.set_num_qubits(n)
    for seed, count in enumerate([1, 3, 17, 2, 1]):
        gates = generate_gates(n, count, "mixed", seed)
        a, b = build_circuits(n, gates, stabilizer, stim)
        rust.apply_circuit(a)
        reference.do(b)
        verify_tableau(rust, reference)
        # Clone a populated read cache; changing the copy must not affect rust.
        copy = rust.copy()
        copy.apply_h(n - 1)
        copy.apply_h(n - 1)
        verify_tableau(copy, reference)


def test_from_stabilizers_row_mutations_and_gates_match_stim():
    stabs = ["+XXX", "+ZZI", "-IZZ"]
    rust = stabilizer.Tableau.from_stabilizers(stabs)
    reference = stim.TableauSimulator()
    reference.set_inverse_tableau(stim.Tableau.from_stabilizers(
        [stim.PauliString(s) for s in stabs]
    ).inverse())
    # The destabilizer completion need not be the same; compare states here.
    for q in range(3):
        rust.apply_h(q)
        reference.h(q)
        for p in rust.stabilizers():
            assert reference.peek_observable_expectation(stim.PauliString(p)) == 1


def test_json_report(tmp_path, capsys):
    output = tmp_path / "nested" / "results.json"
    gc_before = gc.isenabled()
    main(["--qubits", "2", "--gates", "30", "--repeats", "3", "--warmup", "1",
          "--json", str(output)])
    assert gc.isenabled() == gc_before
    report = json.loads(output.read_text())
    assert len(report["results"]) == 4
    for result in report["results"]:
        assert result["verified"]
        assert result["gates"] == 30
        assert result["prep_gates"] == 8
        a, b = result["stabilizer"], result["stim"]
        assert len(a["samples_seconds"]) == len(b["samples_seconds"]) == 3
        assert result["stabilizer_over_stim"] == a["median_seconds"] / b["median_seconds"]
        assert a["ns_per_gate"] == a["median_seconds"] * 1e9 / 30
    assert "Rust / Stim" in capsys.readouterr().out


def test_gc_restored_on_failure(monkeypatch):
    class BrokenTableau:
        n = 1

        def apply_circuit(self, circuit):
            raise RuntimeError("failed during timing")

    import benchmarks.benchmark_stim as harness

    original = harness.prepare_states
    calls = 0

    def prepare(*args):
        nonlocal calls
        calls += 1
        if calls == 1:  # Allow the correctness check before timing.
            return original(*args)
        return BrokenTableau(), stim.TableauSimulator()

    monkeypatch.setattr(harness, "prepare_states", prepare)
    gc_before = gc.isenabled()
    with pytest.raises(RuntimeError, match="failed during timing"):
        benchmark_case(1, [("h", 0)], [], 1, 0, stabilizer, stim)
    assert gc.isenabled() == gc_before


@pytest.mark.parametrize("args", [
    ["--gates", "0"], ["--qubits", "0"], ["--repeats", "0"],
    ["--warmup", "-1"], ["--qubits", "1", "--workloads", "cx"],
])
def test_invalid_cli_arguments(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
