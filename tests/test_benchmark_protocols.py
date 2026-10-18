"""Validate logical Pauli decompositions and circuit benchmark cases."""

import pytest
import stabilizer

stim = pytest.importorskip("stim")

from benchmarks.benchmark_protocols import benchmark_protocol, benchmark_lifecycle, benchmark_shots


@pytest.mark.parametrize("n", [8, 65])
@pytest.mark.parametrize("stim_mode", ["decomposed", "native_pauli"])
@pytest.mark.parametrize("kind,basis,weight", [("basis", "X", 1), ("basis", "Y", 1),
                                               ("parity", "Z", 2), ("parity", "Z", 8),
                                               ("interleaved", "Z", 1), ("repeated", "Z", 1)])
def test_protocols_match_full_logical_replay(n, stim_mode, kind, basis, weight):
    row = benchmark_protocol(n, "scrambled", kind, 8, 1, 731, stabilizer, stim,
                             basis=basis, weight=weight, stim_mode=stim_mode)
    assert row["random_measurements"] + row["deterministic_measurements"] == 8
    assert all(len(values) == 1 for values in row["samples_seconds"].values())
    assert row["verification"] == "full logical replay state"


@pytest.mark.parametrize("kind", ["initialize", "copy", "export_cold", "export_warm", "peek"])
def test_lifecycle_benchmark(kind):
    row = benchmark_lifecycle(8, kind, 1, 731, stabilizer, stim)
    assert all(seconds > 0 for seconds in row["median_seconds"].values())


def test_fresh_tableau_shots():
    row = benchmark_shots(8, 3, "cluster", 1, 731, stabilizer, stim)
    assert row["shots"] == 3
    assert all(seconds > 0 for seconds in row["median_seconds"].values())
