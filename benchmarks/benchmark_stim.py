"""Compare native Clifford circuit execution in stabilizer and Stim.

Build the Python extension in release mode first; see benchmarks/README.md.
"""

import argparse
import gc
import importlib.metadata
import json
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns


WORKLOADS = ("h", "s", "cx", "mixed")
ROOT = Path(__file__).resolve().parents[1]


def generate_gates(n, count, workload, seed):
    """One shared, deterministic gate stream; count means elementary gates."""
    if n < 1 or count < 0 or workload not in WORKLOADS:
        raise ValueError("invalid qubit count, gate count, or workload")
    if n == 1 and workload == "cx":
        raise ValueError("the cx workload needs at least two qubits")
    rng = random.Random(seed)
    choices = ("h", "s", "cx") if n > 1 else ("h", "s")
    gates = []
    for _ in range(count):
        kind = rng.choice(choices) if workload == "mixed" else workload
        a = rng.randrange(n)
        if kind == "cx":
            b = rng.randrange(n - 1)
            b += b >= a
            gates.append((kind, a, b))
        else:
            gates.append((kind, a))
    return gates


def build_circuits(n, gates, stabilizer, stim):
    rust = stabilizer.Circuit(n)
    reference = stim.Circuit()
    methods = {"h": rust.h, "s": rust.p, "cx": rust.cnot}
    for kind, *targets in gates:
        methods[kind](*targets)
        reference.append(kind.upper(), targets)
    return rust, reference


def prepare_states(n, prep, stabilizer, stim):
    rust = stabilizer.Tableau(n, seed=0)
    reference = stim.TableauSimulator()
    # Allocate even idle qubits before starting the clock.
    reference.set_num_qubits(n)
    rust.apply_circuit(prep[0])
    reference.do(prep[1])
    return rust, reference


def verify_tableau(rust, reference):
    """Compare every signed X/Z generator, not just the state stabilizers."""
    forward = reference.current_inverse_tableau().inverse()
    for q in range(rust.n):
        for label, actual, expected in (
            ("X", rust.row_to_pauli(q), forward.x_output(q)),
            ("Z", rust.row_to_pauli(rust.n + q), forward.z_output(q)),
        ):
            if actual != str(expected).replace("_", "I"):
                raise RuntimeError(
                    f"tableau mismatch at {label}{q}: Rust={actual}, Stim={expected}"
                )


def summarize(samples, gate_count):
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "mad_seconds": statistics.median(abs(s - median) for s in samples),
        "ns_per_gate": median * 1e9 / gate_count,
        "gates_per_second": gate_count / median,
    }


def benchmark_case(n, gates, prep_gates, repeats, warmup, stabilizer, stim):
    circuits = build_circuits(n, gates, stabilizer, stim)
    prep = build_circuits(n, prep_gates, stabilizer, stim)
    rust, reference = prepare_states(n, prep, stabilizer, stim)
    rust.apply_circuit(circuits[0])
    reference.do(circuits[1])
    verify_tableau(rust, reference)
    del rust, reference

    samples = {"stabilizer": [], "stim": []}
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(warmup + repeats):
            rust, reference = prepare_states(n, prep, stabilizer, stim)
            # Bind methods and allocate argument tuples outside the timed region.
            calls = (
                ("stabilizer", rust.apply_circuit, circuits[0]),
                ("stim", reference.do, circuits[1]),
            )
            # Alternate execution order to reduce systematic ordering bias.
            ordered_calls = calls if iteration % 2 == 0 else calls[::-1]
            for name, apply, circuit in ordered_calls:
                start = perf_counter_ns()
                apply(circuit)
                elapsed = (perf_counter_ns() - start) * 1e-9
                if iteration >= warmup:
                    samples[name].append(elapsed)
            # Release the completed states before preparing the next pair.
            del ordered_calls, calls, apply, rust, reference
    finally:
        if gc_was_enabled:
            gc.enable()

    result = {name: summarize(values, len(gates)) for name, values in samples.items()}
    result["stabilizer_over_stim"] = (
        result["stabilizer"]["median_seconds"] / result["stim"]["median_seconds"]
    )
    result["verified"] = True
    return result


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=positive_int, nargs="+", default=[10000])
    parser.add_argument("--gates", type=positive_int, default=10_000, help="gates per circuit")
    parser.add_argument("--workloads", choices=WORKLOADS, nargs="+", default=list(WORKLOADS))
    parser.add_argument("--repeats", type=positive_int, default=9)
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prep-gates", type=nonnegative_int, default=None,
        help="untimed random H/S/CX preparation gates (default: 4 * qubits)",
    )
    parser.add_argument("--json", type=Path, metavar="FILE", help="save samples and metadata")
    args = parser.parse_args(argv)
    if 1 in args.qubits and "cx" in args.workloads:
        parser.error("the cx workload needs --qubits >= 2; select h/s/mixed for one qubit")
    try:
        import stabilizer
        import stim
    except ImportError as exc:
        parser.exit(2, f"{exc}\nInstall benchmark dependencies; see benchmarks/README.md.\n")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "stabilizer_version": importlib.metadata.version("stabilizer"),
        "stim_version": stim.__version__,
        "stim_simulator_module": stim.TableauSimulator.__module__,
        "stabilizer_module": stabilizer.__file__,
        "git_revision": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "parameters": {**vars(args), "json": str(args.json) if args.json else None},
        "timing_scope": "one native circuit call, including internal packing and allocation; excludes circuit construction, initial tableau allocation, preparation, verification",
    }
    print(f"stabilizer {metadata['stabilizer_version']} vs Stim {stim.__version__}", flush=True)
    print("Gate execution only; fresh prepared states; full signed tableau verified.", flush=True)
    print("Ratio = Rust / Stim time (>1 means Stim is faster).", flush=True)
    print(f"{'qubits':>7} {'workload':>9} {'gates':>8} {'Rust ms':>11} {'Stim ms':>11} "
          f"{'Rust ns/g':>11} {'Stim ns/g':>11} {'ratio':>9}", flush=True)
    results = []
    for n in args.qubits:
        prep_count = 4 * n if args.prep_gates is None else args.prep_gates
        prep_gates = generate_gates(n, prep_count, "mixed", args.seed + n)
        for workload in args.workloads:
            gates = generate_gates(n, args.gates, workload, args.seed)
            result = benchmark_case(
                n, gates, prep_gates, args.repeats, args.warmup, stabilizer, stim
            )
            result.update(qubits=n, workload=workload, gates=len(gates), prep_gates=prep_count)
            results.append(result)
            a, b = result["stabilizer"], result["stim"]
            print(f"{n:7d} {workload:>9} {len(gates):8d} "
                  f"{a['median_seconds'] * 1e3:11.4f} {b['median_seconds'] * 1e3:11.4f} "
                  f"{a['ns_per_gate']:11.2f} {b['ns_per_gate']:11.2f} "
                  f"{result['stabilizer_over_stim']:8.2f}x", flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
