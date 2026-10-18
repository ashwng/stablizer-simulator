"""Verify benchmark preparation, measurement branches, and collapsed states."""

import json

import pytest

import stabilizer

stim = pytest.importorskip("stim")

from benchmarks.benchmark_measurement import benchmark_measurements, main


@pytest.mark.parametrize("n", [8, 65])
@pytest.mark.parametrize("state", ["ghz", "cluster", "scrambled", "deterministic"])
@pytest.mark.parametrize("mode", ["native", "sequential"])
@pytest.mark.parametrize("selection", ["one", "half", "all"])
def test_measurement_cases_match_full_stim_state(n, state, mode, selection):
    count = {"one": 1, "half": n // 2, "all": n}[selection]
    row = benchmark_measurements(n, state, count, mode, 2, 731, stabilizer, stim)
    assert row["verification"] == "full collapsed state"
    assert row["random_measurements"] + row["deterministic_measurements"] == count
    assert len(set(row["targets"])) == count
    assert all(len(samples) == 2 for samples in row["samples_seconds"].values())
    assert all(seconds > 0 for seconds in row["median_seconds"].values())
    if state in ("ghz", "cluster"):
        assert row["sampled_maximally_mixed_qubits"] == min(n, 64)
    if state == "ghz":
        assert row["random_measurements"] == 1
    elif state == "cluster":
        assert row["random_measurements"] == count
    elif state == "deterministic":
        assert row["random_measurements"] == 0


def test_measurement_json_and_capped_duplicate_counts(tmp_path):
    output = tmp_path / "measurement.json"
    report = main(["--qubits", "2", "--states", "ghz", "cluster", "--counts", "1", "64", "all",
                   "--modes", "native", "sequential", "--repeats", "1", "--json", str(output)])
    assert json.loads(output.read_text()) == report
    assert len(report["results"]) == 8
    assert report["stim_core"] == stim.TableauSimulator.__module__
    assert len(report["stabilizer_core_sha256"]) == 64


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_invalid_measurement_counts(value):
    with pytest.raises(SystemExit):
        main(["--counts", value])
