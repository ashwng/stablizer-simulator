"""Compare native batches and sequential Z measurements against Stim.

Run from the repository root after a release build:
    python -m benchmarks.benchmark_measurement --qubits 256 1024 4096 --json /tmp/measurement.json

Preparation and verification are outside the timer. Timings include Python
calls and result conversion, with one warmup and fresh preparation per sample.
The default compares normal measurement, with independent seeded outcomes.
--workloads retains the older forced-outcome benchmark for historical reports.
Use stress_qubits.py for isolated peak-RAM and large-qubit product-state probes.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import time

from .benchmark_stim import build_circuits, generate_gates, git_output


def benchmark(n, workload, repeats, batch, seed, stabilizer, stim):
    count = 1 if workload != "entangled_batch" else min(n, batch)
    gates = generate_gates(n, 40 * n, "cx" if workload == "deterministic" else "mixed", seed)
    if workload == "deterministic":
        # Create nonzero outcome signs before scrambling the generator basis.
        gates = [gate for q in range(0, n, 7)
                 for gate in [("h", q), ("s", q), ("s", q), ("h", q)]] + gates
    prep, reference_prep = build_circuits(n, gates, stabilizer, stim)
    planner = stim.TableauSimulator(seed=seed)
    planner.set_num_qubits(n)
    planner.do(reference_prep)
    if workload != "deterministic" and planner.peek_z(0) != 0:
        # Guarantee that the first projection is random, even for tiny states.
        prep.h(0)
        reference_prep.append("H", [0])
        planner.h(0)
    plan = [(q, planner.measure(q)) for q in range(count)]
    del planner
    samples = {"stabilizer": [], "stim": []}
    for sample in range(repeats + 1):
        order = ("stabilizer", "stim") if sample % 2 == 0 else ("stim", "stabilizer")
        for backend in order:
            if backend == "stabilizer":
                state = stabilizer.Tableau(n)
                state.apply_circuit(prep)
                start = time.perf_counter()
                for q, outcome in plan:
                    result = state.measure(q, force=int(outcome))
                    if result != outcome:
                        raise RuntimeError("local measurement disagrees with planned outcome")
                elapsed = time.perf_counter() - start
            else:
                state = stim.TableauSimulator()
                state.set_num_qubits(n)
                state.do(reference_prep)
                start = time.perf_counter()
                for q, outcome in plan:
                    if workload == "deterministic":
                        # Normal Stim measurement avoids transposition when the
                        # result is deterministic; postselect_z always transposes.
                        result = state.measure(q)
                        if result != outcome:
                            raise RuntimeError("Stim measurement disagrees with planned outcome")
                    else:
                        state.postselect_z(q, desired_value=outcome)
                elapsed = time.perf_counter() - start
            for q, outcome in plan:
                expected_z = 1 - 2 * int(outcome)
                actual = state.peek(q) if backend == "stabilizer" else state.peek_z(q)
                if backend == "stim":
                    assert actual == expected_z
                    continue
                assert actual == outcome
            del state
            if sample:
                samples[backend].append(elapsed)
    return {"qubits": n, "workload": workload, "measurement_count": count,
            "seed": seed, "preparation_gate_count": len(prep),
            "samples_seconds": samples,
            "median_seconds": {backend: statistics.median(values) for backend, values in samples.items()}}


def preparation_gates(n, state, seed):
    """GHZ/cluster have guaranteed entanglement; scrambled is a deeper Clifford."""
    if state == "zero":
        return []
    if state == "plus":
        return [("h", q) for q in range(n)]
    if state == "bell":
        return [gate for q in range(0, n - 1, 2) for gate in [("h", q), ("cx", q, q + 1)]]
    if state == "ghz":
        return [("h", 0)] + [("cx", 0, q) for q in range(1, n)]
    if state == "deterministic":
        return [gate for q in range(0, n, 7)
                for gate in [("h", q), ("s", q), ("s", q), ("h", q)]] + (generate_gates(n, 40 * n, "cx", seed) if n > 1 else [])
    gates = [("h", q) for q in range(n)]
    for q in range(n - 1):
        # CZ(q, q+1) from the shared H/CNOT gate set.
        gates.extend([("h", q + 1), ("cx", q, q + 1), ("h", q + 1)])
    if state == "scrambled":
        gates.extend(generate_gates(n, 40 * n, "mixed", seed))
    return gates


def prepared_state(n, prep, backend, seed, stabilizer, stim):
    if backend == "stabilizer":
        state = stabilizer.Tableau(n, seed=seed)
        state.apply_circuit(prep[0])
    else:
        state = stim.TableauSimulator(seed=seed)
        state.set_num_qubits(n)
        state.do(prep[1])
    return state


def execute(state, backend, mode, targets, measurement_circuit):
    """Time the calls and binding result conversion; normalize after timing."""
    start = time.perf_counter_ns()
    if mode == "native":
        raw = state.apply_circuit(measurement_circuit) if backend == "stabilizer" else state.measure_many(*targets)
    else:
        raw = [state.measure(q) for q in targets]
    elapsed = (time.perf_counter_ns() - start) * 1e-9
    if mode == "native" and backend == "stabilizer":
        return elapsed, [bool(outcome) for _, outcome, _ in raw], [det for _, _, det in raw]
    return elapsed, [bool(outcome) for outcome in raw], None


def verify_projection(state, backend, outcomes, targets, n, prep, seed, stabilizer, stim, full=False):
    """Replay the observed branch outside timing and check the collapsed state."""
    if len(outcomes) != len(targets):
        raise RuntimeError("wrong number of measurement outcomes")
    reference = prepared_state(n, prep, "stim", seed, stabilizer, stim)
    # Z projections commute, so grouping zero/one outcomes needs at most two
    # transposes and checks that the entire observed bit string is possible.
    for desired in (False, True):
        group = [q for q, outcome in zip(targets, outcomes) if outcome == desired]
        if group:
            reference.postselect_z(group, desired_value=desired)
    checks = sorted({0, len(targets) // 2, len(targets) - 1})
    for index in checks:
        q, outcome = targets[index], outcomes[index]
        actual = state.peek(q) if backend == "stabilizer" else state.peek_z(q)
        expected = int(outcome) if backend == "stabilizer" else 1 - 2 * int(outcome)
        if actual != expected:
            raise RuntimeError("collapsed qubit disagrees with its measurement result")
    if full:
        def canonical(sim):
            forward = sim.current_inverse_tableau().inverse()
            return stabilizer.Tableau.from_stabilizers(
                [str(forward.z_output(q)).replace("_", "I") for q in range(n)]
            ).canonical_stabilizers()
        actual = state.canonical_stabilizers() if backend == "stabilizer" else canonical(state)
        if actual != canonical(reference):
            raise RuntimeError("full collapsed state disagrees with Stim replay")


def benchmark_measurements(n, state_name, count, mode, repeats, seed, stabilizer, stim):
    gates = preparation_gates(n, state_name, seed)
    prep = build_circuits(n, gates, stabilizer, stim)
    targets = list(range(n))
    random.Random(seed + 1).shuffle(targets)
    targets = targets[:count]
    measurements = stabilizer.Circuit(n)
    for q in targets:
        measurements.measure(q)
    planner = prepared_state(n, prep, "stim", seed, stabilizer, stim)
    sampled_qubits = list(range(min(n, 64)))
    mixed_qubits = sum(all(peek(q) == 0 for peek in (planner.peek_x, planner.peek_y, planner.peek_z))
                       for q in sampled_qubits)
    deterministic = []
    for q in targets:
        deterministic.append(planner.peek_z(q) != 0)
        planner.measure(q)
    del planner
    samples = {"stabilizer": [], "stim": []}
    outcome_hashes = {}
    for sample in range(repeats + 1):
        order = ("stabilizer", "stim") if sample % 2 == 0 else ("stim", "stabilizer")
        for backend in order:
            state = prepared_state(n, prep, backend, seed + sample, stabilizer, stim)
            elapsed, outcomes, flags = execute(state, backend, mode, targets, measurements)
            if flags is not None and flags != deterministic:
                raise RuntimeError("random/deterministic branches disagree with Stim")
            if sample == 0:
                verify_projection(state, backend, outcomes, targets, n, prep, seed, stabilizer, stim,
                                  full=n <= 129)
                outcome_hashes[backend] = hashlib.sha256(bytes(outcomes)).hexdigest()
            else:
                samples[backend].append(elapsed)
            del state
    medians = {backend: statistics.median(values) for backend, values in samples.items()}
    return {"qubits": n, "state": state_name, "mode": mode, "measurement_count": count,
            "basis": "Z", "seed": seed, "targets": targets,
            "random_measurements": count - sum(deterministic), "deterministic_measurements": sum(deterministic),
            "sampled_initial_qubits": len(sampled_qubits), "sampled_maximally_mixed_qubits": mixed_qubits,
            "preparation_gate_count": len(gates),
            "preparation_sha256": hashlib.sha256(json.dumps(gates).encode()).hexdigest(),
            "samples_seconds": samples, "median_seconds": medians,
            "ns_per_measured_qubit": {b: seconds * 1e9 / count for b, seconds in medians.items()},
            "stabilizer_over_stim": medians["stabilizer"] / medians["stim"],
            "verification": "full collapsed state" if n <= 129 else "whole outcome vector replay and sampled collapsed qubits",
            "warmup_outcome_sha256": outcome_hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--states", nargs="+", choices=["zero", "plus", "bell", "ghz", "cluster", "scrambled", "deterministic"],
                        default=["ghz", "cluster", "scrambled"])
    parser.add_argument("--counts", nargs="+", default=["1", "64"], help="number of distinct Z targets, or all")
    parser.add_argument("--modes", nargs="+", choices=["native", "sequential"], default=["native", "sequential"])
    parser.add_argument("--workloads", nargs="+", choices=["entangled_first", "entangled_batch", "deterministic"],
                        help="run the legacy forced-outcome harness instead")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if min(args.qubits) < 2 or args.repeats < 1 or args.batch < 1:
        parser.error("qubits must be >= 2; repeats and batch must be positive")
    try:
        if any(value != "all" and int(value) < 1 for value in args.counts):
            raise ValueError
    except ValueError:
        parser.error("counts must be positive integers or all")
    import stabilizer
    import stim
    results = []
    report = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": platform.platform(), "stabilizer_core": stabilizer._core.__file__,
              "stabilizer_core_sha256": hashlib.sha256(Path(stabilizer._core.__file__).read_bytes()).hexdigest(),
              "stim_version": importlib.metadata.version("stim"),
              "stim_core": stim.TableauSimulator.__module__,
              "git_revision": git_output("rev-parse", "HEAD"),
              "warmup": 1, "repeats": args.repeats,
              "description": "Fresh shared preparation per sample. Z measurements of distinct qubits; no compiled sampler. Native: one apply_circuit/measure_many call; sequential: one unforced measure call per qubit. Preparation and verification excluded. Independent seeded random streams; outcomes may differ between backends. Random/deterministic branch classification is outcome-independent for Clifford circuits.",
              "legacy_postselection": bool(args.workloads), "results": results}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        report["cpu_model"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                                    if line.startswith("model name")), platform.processor())
    for n in args.qubits:
        cases = [(state, min(n, n if count == "all" else int(count)), mode)
                 for state in args.states for count in args.counts for mode in args.modes]
        if args.workloads:
            cases = [(workload, None, None) for workload in args.workloads]
        for state, count, mode in dict.fromkeys(cases):
            row = (benchmark(n, state, args.repeats, args.batch, args.seed, stabilizer, stim)
                   if args.workloads else
                   benchmark_measurements(n, state, count, mode, args.repeats, args.seed, stabilizer, stim))
            results.append(row)
            print(f"{n:,} {state} {mode or 'legacy'} ({row['measurement_count']} measurements): "
                  + ", ".join(f"{backend} {seconds * 1000:.3f} ms" for backend, seconds in row["median_seconds"].items()),
                  flush=True)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
