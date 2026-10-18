"""Probe fully resident tableau capacity in isolated Linux worker processes.

Run from the repository root after building the release Python extension:
    python benchmarks/stress_qubits.py --auto --json /tmp/qubit-limits.json
    python benchmarks/stress_qubits.py --auto --mode measurement --json /tmp/measurement-limits.json
    python benchmarks/stress_qubits.py --qubits 65536 98304 --backends stabilizer stim

The default leaves 512 MiB of currently available RAM for the host. It does not
count swap as RAM. See benchmarks/README.md for the workload and limitations.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time


MIB = 1024**2
GIB = 1024**3


def memory_info():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value, *_ = line.split()
        fields[name.rstrip(":")] = int(value) * 1024
    available = fields["MemAvailable"]
    # Account for a container's memory limit as well as host memory.
    relative = next((s.split(":", 2)[2] for s in Path("/proc/self/cgroup").read_text().splitlines()
                     if s.startswith("0::")), "/")
    root = Path("/sys/fs/cgroup")
    group = root / relative.lstrip("/")
    while group.is_relative_to(root):
        limit_file, current_file = group / "memory.max", group / "memory.current"
        if limit_file.exists() and current_file.exists():
            limit = limit_file.read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit) - int(current_file.read_text())))
        if group == root:
            break
        group = group.parent
    return {"total_bytes": fields["MemTotal"], "available_bytes": available,
            "swap_used_bytes": fields["SwapTotal"] - fields["SwapFree"]}


def storage_bytes(n, mode, backend="stabilizer"):
    if backend == "stim":
        # Conservative SIMD padding estimate. Actual peak RSS is measured;
        # Stim's measurement/transposition overhead is not assumed to be zero.
        # set_num_qubits uses Stim's 1.1x growth reserve on each matrix axis.
        grown = (11 * n + 9) // 10
        padded = ((grown + 511) // 512) * 512
        return padded * padded // 2 + padded // 4
    column_words = ((2 * n + 1 + 255) // 256) * 4
    total = 2 * n * column_words * 8 + column_words * 8
    if mode == "measurement":
        # Column-native projection uses only linear masks/phase workspace.
        # Include spare Vec capacity; it no longer allocates a row tableau.
        total += 4 * n + 4096
    return total


def capacity_for_budget(budget, mode, backend="stabilizer"):
    lo, hi = 1, math.isqrt(2 * budget) + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if storage_bytes(mid, mode, backend) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def gate_stream(n, count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        q = rng.randrange(n)
        kind = rng.randrange(3 if n > 1 else 2)
        if kind < 2:
            yield ("h" if kind == 0 else "s", q)
        else:
            target = rng.randrange(n - 1)
            target += target >= q
            yield ("cx", q, target)


def compare_results(results):
    pairs = {}
    for result in results:
        if result["status"] == "success":
            pairs.setdefault(result["qubits"], {})[result["backend"]] = result
    comparisons = []
    for n, pair in sorted(pairs.items()):
        if set(pair) != {"stabilizer", "stim"}:
            continue
        a, b = pair["stabilizer"], pair["stim"]
        if a["gate_sha256"] != b["gate_sha256"]:
            raise RuntimeError(f"gate streams differ for {n} qubits")
        comparisons.append({"qubits": n, "gate_sha256": a["gate_sha256"],
                            "stabilizer_over_stim_gate_time": a["gate_seconds"] / b["gate_seconds"],
                            "stabilizer_over_stim_peak_rss": a["peak_rss_bytes"] / b["peak_rss_bytes"],
                            "stabilizer_over_stim_measurement_time":
                                a["measurement_seconds"] / b["measurement_seconds"]
                                if a["measurement_seconds"] is not None else None})
    return comparisons


def worker(args):
    worker_started = time.perf_counter()
    resource.setrlimit(resource.RLIMIT_AS, (args.limit_bytes, args.limit_bytes))

    if args.backend == "stabilizer":
        import stabilizer
        make_circuit = lambda: stabilizer.Circuit(args.worker)
        def append(circuit, gate):
            kind, *targets = gate
            {"h": circuit.h, "s": circuit.p, "cx": circuit.cnot}[kind](*targets)
        core_module = stabilizer._core.__file__
        backend_version = "local release extension"
    else:
        import stim
        make_circuit = stim.Circuit
        def append(circuit, gate):
            kind, *targets = gate
            circuit.append(kind.upper(), targets)
        core_module = stim.TableauSimulator.__module__
        backend_version = stim.__version__

    def event(stage, **values):
        print(json.dumps({"stage": stage, **values}), flush=True)

    n = args.worker
    started = time.perf_counter()
    if args.backend == "stabilizer":
        tableau = stabilizer.Tableau(n)
        apply = tableau.apply_circuit
        h = tableau.apply_h
    else:
        tableau = stim.TableauSimulator()
        tableau.set_num_qubits(n)
        apply = tableau.do
        h = tableau.h
    initialization = time.perf_counter() - started
    event("allocated", initialization_seconds=initialization)

    sweep = make_circuit()
    for q in range(n):
        append(sweep, ("h", q))
    started = time.perf_counter()
    # Both implementations write generator data throughout the tableau.
    apply(sweep)
    touch_seconds = time.perf_counter() - started
    event("fully_touched", touch_seconds=touch_seconds,
          rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

    circuit = make_circuit()
    digest = hashlib.sha256()
    for gate in gate_stream(n, args.gates, args.seed):
        append(circuit, gate)
        digest.update((repr(gate) + "\n").encode())
    inverse = circuit.inverse()
    apply(circuit)  # Warm up; restore the same state before every timed call.
    apply(inverse)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        apply(circuit)
        samples.append(time.perf_counter() - started)
        apply(inverse)
    gate_seconds = statistics.median(samples)
    event("gates_completed", gates=args.gates, gate_seconds=gate_seconds)
    apply(sweep)  # Restore |0...0>, including Clifford action.

    # Direct bit/observable access never constructs a second full tableau.
    # These are sampled checks, not an exhaustive comparison at large n.
    rng = random.Random(args.seed + 1)
    checks = sorted({0, n - 1, n // 2, *[rng.randrange(n) for _ in range(61)]})
    for q in checks:
        if args.backend == "stim":
            # Query observables directly; current_inverse_tableau() would copy
            # the entire state and distort the capacity measurement.
            assert tableau.peek_z(q) == 1
            assert tableau.peek_x(q) == 0
            continue
        other = (q + 1) % n
        assert tableau.get_x(q, q) == 1
        assert tableau.get_z(q, q) == 0
        assert tableau.get_x(n + q, q) == 0
        assert tableau.get_z(n + q, q) == 1
        assert tableau.get_r(q) == tableau.get_r(n + q) == 0
        if n > 1:
            assert tableau.get_x(q, other) == tableau.get_z(n + q, other) == 0
    event("sampled_identity_verified", checked_qubits=len(checks))

    measurement_seconds = None
    if args.mode == "measurement":
        h(n - 1)
        started = time.perf_counter()
        if args.backend == "stabilizer":
            outcome = tableau.measure(n - 1, force=0)
            assert outcome == 0
        else:
            tableau.postselect_z(n - 1, desired_value=False)
        measurement_seconds = time.perf_counter() - started
        if args.backend == "stim":
            assert tableau.peek_z(n - 1) == 1
        event("measurement_completed", measurement_seconds=measurement_seconds)

    swap_bytes = next(int(line.split()[1]) * 1024
                      for line in Path("/proc/self/status").read_text().splitlines()
                      if line.startswith("VmSwap:"))

    event("success", backend=args.backend, qubits=n, mode=args.mode,
          storage_bytes=storage_bytes(n, args.mode, args.backend),
          initialization_seconds=initialization, full_touch_seconds=touch_seconds,
          gate_seconds=gate_seconds, gates=args.gates,
          gate_samples_seconds=samples, gate_sha256=digest.hexdigest(),
          measurement_seconds=measurement_seconds, checked_qubits=len(checks),
          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
          worker_swap_bytes=swap_bytes,
          total_seconds=time.perf_counter() - worker_started,
          core_module=core_module, backend_version=backend_version,
          verification="sampled Clifford identity" if args.backend == "stabilizer" else "sampled state identity")


def probe(n, args, limit_bytes, backend="stabilizer"):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(n),
               "--mode", args.mode, "--gates", str(args.gates), "--seed", str(args.seed),
               "--limit-bytes", str(limit_bytes), "--backend", backend,
               "--repeats", str(args.repeats)]
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    before = memory_info()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env)
    started = time.monotonic()
    reason = None
    while True:
        try:
            output, _ = process.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            info = memory_info()
            if time.monotonic() - started > args.timeout:
                reason = "timeout"
            elif info["available_bytes"] < 128 * MIB:
                reason = "host_memory_pressure"
            elif info["swap_used_bytes"] - before["swap_used_bytes"] > 64 * MIB:
                reason = "swap_pressure"
            if reason:
                process.kill()
                output, _ = process.communicate()
                break
    events = []
    for line in output.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    result = next((e for e in events if e.get("stage") == "success"), None)
    if result is None:
        if reason is None and "memory allocation of" in output:
            reason = "allocation_failed_under_worker_cap"
        result = {"backend": backend, "qubits": n, "mode": args.mode, "status": reason or "worker_failed",
                  "returncode": process.returncode, "events": events, "output": output[-3000:]}
    else:
        result["status"] = "success"
    result["address_space_limit_bytes"] = limit_bytes
    result["available_before_bytes"] = before["available_bytes"]
    result["memory_after"] = memory_info()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qubits", nargs="+", type=int)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--mode", choices=["gates", "measurement"], default="gates")
    parser.add_argument("--gates", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", choices=["stabilizer", "stim"], nargs="+", default=["stabilizer"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--limit-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["stabilizer", "stim"], default="stabilizer", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gates < 1 or args.repeats < 1 or args.reserve_mib < 128 or args.timeout <= 0:
        parser.error("gates/repeats/timeout must be positive and reserve-mib must be >= 128")
    if args.worker is not None:
        worker(args)
        return
    if args.auto == bool(args.qubits):
        parser.error("select either --auto or --qubits N [...]")
    if args.qubits and min(args.qubits) < 1:
        parser.error("qubit counts must be positive")
    info = memory_info()
    limit_bytes = info["available_bytes"] - args.reserve_mib * MIB
    # Leave room inside the worker's virtual-address cap for Python, native
    # modules, stack and circuits. Limits are conservative, not physical OOMs.
    tableau_budget = limit_bytes - 128 * MIB
    if tableau_budget <= 0:
        parser.error("insufficient available memory for the requested reserve")
    ceilings = {backend: capacity_for_budget(tableau_budget, args.mode, backend) for backend in args.backends}
    if args.auto:
        tops = [n // 1024 * 1024 for n in ceilings.values()]
        candidates = sorted({n for n in [8192, 16384, 32768, 65536, 98304, 131072,
                            *tops, *[t * 9 // 10 // 1024 * 1024 for t in tops]]
                             if 0 < n <= max(ceilings.values())})
    else:
        candidates = args.qubits
    report = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "initial_memory": info, "mode": args.mode, "seed": args.seed,
              "gates": args.gates, "repeats": args.repeats, "backends": args.backends,
              "reserve_mib": args.reserve_mib,
              "budget_ceiling_qubits": ceilings, "results": []}
    print(f"Available {info['available_bytes'] / GIB:.2f} GiB; worker cap {limit_bytes / GIB:.2f} GiB; "
          f"estimated {args.mode} ceilings {ceilings}.", flush=True)
    disabled = set()
    for i, candidate in enumerate(candidates):
        # Alternate order at each size; never hold both native states in RAM.
        for backend in (args.backends if i % 2 == 0 else args.backends[::-1]):
            if backend in disabled:
                continue
            n = candidate
            # Recheck free memory before each child; earlier children have exited.
            limit = min(limit_bytes, memory_info()["available_bytes"] - args.reserve_mib * MIB)
            if args.auto and storage_bytes(n, args.mode, backend) + 128 * MIB > limit and limit > 128 * MIB:
                n = capacity_for_budget(limit - 128 * MIB, args.mode, backend) // 1024 * 1024
                if n < 1 or any(r["qubits"] == n and r["backend"] == backend for r in report["results"]):
                    continue
            if storage_bytes(n, args.mode, backend) + 128 * MIB > limit:
                result = {"backend": backend, "qubits": n, "mode": args.mode, "status": "skipped_memory_budget",
                          "storage_bytes": storage_bytes(n, args.mode, backend)}
            else:
                print(f"Testing {backend} {n:,} qubits ({storage_bytes(n, args.mode, backend) / GIB:.2f} GiB estimated storage)...", flush=True)
                result = probe(n, args, limit, backend)
            report["results"].append(result)
            report["comparisons"] = compare_results(report["results"])
            if result["status"] == "success":
                print(f"  PASS: RSS {result['peak_rss_bytes'] / GIB:.2f} GiB, "
                      f"full touch {result['full_touch_seconds']:.3f}s, "
                      f"{args.gates:,} gates {result['gate_seconds']:.3f}s, "
                      f"measurement {result['measurement_seconds']}", flush=True)
            else:
                print(f"  {result['status']}", flush=True)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(report, indent=2) + "\n")
            if result["status"] not in ("success", "skipped_memory_budget"):
                disabled.add(backend)


if __name__ == "__main__":
    main()
