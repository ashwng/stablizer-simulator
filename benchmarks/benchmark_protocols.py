"""Shared Clifford decompositions for basis, parity, and dynamic circuits.

An event is either a unitary gate or a logical Pauli measurement. The local
simulator implements Pauli measurement by a reversible Clifford parity network.
Stim can run the identical network or its native MPP operation. These are
separate benchmark modes; gate/network overhead is included in both timers.
"""

import hashlib
import json
import random
import statistics
import time

from .benchmark_measurement import prepared_state, preparation_gates
from .benchmark_stim import build_circuits, generate_gates


def inverse_gates(gates):
    return [gate for gate in reversed(gates) for _ in range(3 if gate[0] == "s" else 1)]


def pauli_network(observable):
    support = [q for q, letter in enumerate(observable) if letter != "I"]
    if not support:
        raise ValueError("measurement needs nonempty Pauli support")
    before = []
    for q in support:
        if observable[q] == "X":
            before.append(("h", q))
        elif observable[q] == "Y":
            before.extend([("s", q)] * 3 + [("h", q)])
    pivot = support[-1]
    before.extend(("cx", q, pivot) for q in support[:-1])
    return before + [("m", pivot)] + inverse_gates(before)


def make_events(n, kind, count, seed, basis="Z", weight=2, layer=8, locality="random"):
    rng = random.Random(seed)
    events = []
    order = list(range(n))
    rng.shuffle(order)
    for step in range(count):
        if kind == "interleaved":
            for gate in generate_gates(n, layer, "mixed", seed + step):
                if locality == "local" and gate[0] == "cx":
                    gate = ("cx", gate[1], (gate[1] + 1) % n)
                events.append(("gate", gate))
        letters = ["I"] * n
        if kind == "parity":
            for q in rng.sample(range(n), min(weight, n)):
                letters[q] = rng.choice("XYZ")
        else:
            q = order[0] if kind == "repeated" else order[step % n]
            letters[q] = basis
        events.append(("measure", "".join(letters)))
    return events


def compile_events(n, events, stabilizer, stim, stim_mode):
    local = stabilizer.Circuit(n)
    reference = stim.Circuit()
    methods = {"h": local.h, "s": local.p, "cx": local.cnot, "m": local.measure}
    for kind, value in events:
        gates = [value] if kind == "gate" else pauli_network(value)
        for gate, *targets in gates:
            methods[gate](*targets)
        if kind == "measure" and stim_mode == "native_pauli":
            targets = []
            factories = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
            for q, letter in enumerate(value):
                if letter != "I":
                    if targets:
                        targets.append(stim.target_combiner())
                    targets.append(factories[letter](q))
            reference.append("MPP", targets)
        else:
            for gate, *targets in gates:
                reference.append("M" if gate == "m" else gate.upper(), targets)
    return local, reference


def replay_events(sim, events, stim, outcomes=None):
    flags = []
    index = 0
    for kind, value in events:
        if kind == "gate":
            gate, *targets = value
            {"h": sim.h, "s": sim.s, "cx": sim.cnot}[gate](*targets)
        else:
            observable = stim.PauliString(value)
            flags.append(sim.peek_observable_expectation(observable) != 0)
            if outcomes is None:
                sim.measure_observable(observable)
            else:
                sim.postselect_observable(observable, desired_value=outcomes[index])
            index += 1
    if outcomes is not None and index != len(outcomes):
        raise RuntimeError("wrong number of logical measurement results")
    return flags


def canonical_stim(sim, n, stabilizer):
    forward = sim.current_inverse_tableau().inverse()
    return stabilizer.Tableau.from_stabilizers(
        [str(forward.z_output(q)).replace("_", "I") for q in range(n)]
    ).canonical_stabilizers()


def verify_state(actual, reference, backend, n, stabilizer, stim):
    if n <= 129:
        a = actual.canonical_stabilizers() if backend == "stabilizer" else canonical_stim(actual, n, stabilizer)
        if a != canonical_stim(reference, n, stabilizer):
            raise RuntimeError("logical protocol collapsed state mismatch")
    elif backend == "stabilizer":
        # Inspect physical stabilizers, including unmeasured qubits. All must
        # have expectation +1 in the replay state; no equality of bases needed.
        for q in sorted({0, n // 7, n // 3, n // 2, n - 1}):
            pauli = actual.row_to_pauli(n + q).replace("I", "_")
            if reference.peek_observable_expectation(stim.PauliString(pauli)) != 1:
                raise RuntimeError("sampled stabilizer disagrees with logical replay")
    else:
        for q in sorted({0, n // 7, n // 3, n // 2, n - 1}):
            for axis in "xyz":
                method = "peek_" + axis
                if getattr(actual, method)(q) != getattr(reference, method)(q):
                    raise RuntimeError("sampled observable disagrees with logical replay")


def benchmark_protocol(n, state_name, kind, count, repeats, seed, stabilizer, stim,
                       basis="Z", weight=2, layer=8, locality="random", stim_mode="decomposed"):
    prep = build_circuits(n, preparation_gates(n, state_name, seed), stabilizer, stim)
    events = make_events(n, kind, count, seed, basis, weight, layer, locality)
    circuits = compile_events(n, events, stabilizer, stim, stim_mode)
    planner = prepared_state(n, prep, "stim", seed, stabilizer, stim)
    flags = replay_events(planner, events, stim)
    del planner
    samples = {"stabilizer": [], "stim": []}
    for iteration in range(repeats + 1):
        order = ("stabilizer", "stim") if iteration % 2 == 0 else ("stim", "stabilizer")
        for backend in order:
            state = prepared_state(n, prep, backend, seed + iteration, stabilizer, stim)
            start = time.perf_counter_ns()
            if backend == "stabilizer":
                raw = state.apply_circuit(circuits[0])
            else:
                state.do(circuits[1])
                raw = state.current_measurement_record()
            elapsed = (time.perf_counter_ns() - start) * 1e-9
            outcomes = [bool(x[1]) for x in raw] if backend == "stabilizer" else [bool(x) for x in raw]
            if len(outcomes) != count:
                raise RuntimeError("protocol outcome count mismatch")
            if backend == "stabilizer" and [x[2] for x in raw] != flags:
                raise RuntimeError("protocol random/deterministic classification mismatch")
            if iteration == 0:
                reference = prepared_state(n, prep, "stim", seed, stabilizer, stim)
                replay_events(reference, events, stim, outcomes)
                verify_state(state, reference, backend, n, stabilizer, stim)
                del reference
            else:
                samples[backend].append(elapsed)
            del state
    medians = {b: statistics.median(s) for b, s in samples.items()}
    return {"qubits": n, "state": state_name, "kind": kind, "measurement_count": count,
            "basis": basis, "weight": min(weight, n) if kind == "parity" else 1,
            "unitary_layer": layer if kind == "interleaved" else 0, "locality": locality,
            "stim_mode": stim_mode, "local_compiled_gates": len(circuits[0]),
            "seed": seed, "event_sha256": hashlib.sha256(json.dumps(events).encode()).hexdigest(),
            "random_measurements": count - sum(flags), "deterministic_measurements": sum(flags),
            "samples_seconds": samples, "median_seconds": medians,
            "stabilizer_over_stim": medians["stabilizer"] / medians["stim"],
            "verification": "full logical replay state" if n <= 129 else "logical outcome replay plus sampled final stabilizers/observables",
            "timing_scope": "one circuit call and result conversion; includes Clifford decomposition/basis restoration; native_pauli uses Stim MPP"}


def benchmark_lifecycle(n, kind, repeats, seed, stabilizer, stim):
    samples = {"stabilizer": [], "stim": []}
    prep = build_circuits(n, preparation_gates(n, "cluster", seed), stabilizer, stim)
    for iteration in range(repeats + 1):
        order = ("stabilizer", "stim") if iteration % 2 == 0 else ("stim", "stabilizer")
        for backend in order:
            source = prepared_state(n, prep, backend, seed, stabilizer, stim) if kind != "initialize" else None
            if kind == "export_warm":
                if backend == "stabilizer":
                    source.to_numpy()
                else:
                    source.current_inverse_tableau().inverse().to_numpy()
            start = time.perf_counter_ns()
            if kind == "initialize":
                if backend == "stabilizer":
                    result = stabilizer.Tableau(n, seed=seed)
                else:
                    result = stim.TableauSimulator(seed=seed)
                    result.set_num_qubits(n)
            elif kind == "copy":
                result = source.copy() if backend == "stabilizer" else source.copy(copy_rng=True)
            elif kind.startswith("export"):
                result = source.to_numpy() if backend == "stabilizer" else source.current_inverse_tableau().inverse().to_numpy()
            else:
                result = [source.peek(q) for q in range(min(64, n))] if backend == "stabilizer" else [source.peek_z(q) for q in range(min(64, n))]
            elapsed = (time.perf_counter_ns() - start) * 1e-9
            del result, source
            if iteration:
                samples[backend].append(elapsed)
    medians = {b: statistics.median(s) for b, s in samples.items()}
    return {"qubits": n, "kind": kind, "samples_seconds": samples, "median_seconds": medians,
            "stabilizer_over_stim": medians["stabilizer"] / medians["stim"],
            "timing_scope": "initial allocation without full touch" if kind == "initialize" else
                            "full forward tableau arrays, including Stim inversion" if kind.startswith("export") else kind,
            "state": "zero" if kind == "initialize" else "cluster"}


def benchmark_shots(n, shots, state_name, repeats, seed, stabilizer, stim):
    prep = build_circuits(n, preparation_gates(n, state_name, seed), stabilizer, stim)
    measurements = stabilizer.Circuit(n)
    reference_measurements = stim.Circuit()
    for q in range(n):
        measurements.measure(q)
    reference_measurements.append("M", list(range(n)))
    samples = {"stabilizer": [], "stim": []}
    for iteration in range(repeats + 1):
        order = ("stabilizer", "stim") if iteration % 2 == 0 else ("stim", "stabilizer")
        for backend in order:
            start = time.perf_counter_ns()
            for shot in range(shots):
                state = prepared_state(n, prep, backend, seed + iteration + shot, stabilizer, stim)
                if backend == "stabilizer":
                    outcomes = state.apply_circuit(measurements)
                else:
                    state.do(reference_measurements)
                    outcomes = state.current_measurement_record()
                if len(outcomes) != n:
                    raise RuntimeError("shot measurement count mismatch")
                del state, outcomes
            elapsed = (time.perf_counter_ns() - start) * 1e-9
            if iteration:
                samples[backend].append(elapsed)
    medians = {b: statistics.median(s) for b, s in samples.items()}
    return {"qubits": n, "shots": shots, "state": state_name, "samples_seconds": samples,
            "median_seconds": medians, "stabilizer_over_stim": medians["stabilizer"] / medians["stim"],
            "timing_scope": "fresh tableau allocation, preparation, all-qubit Z measurement, result conversion and destruction for every shot; no compiled sampler"}
