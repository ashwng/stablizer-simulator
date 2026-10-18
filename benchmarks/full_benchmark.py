"""Checkpointed pure-state Clifford benchmark suite against Stim.

Run after a release build, from the repository root:
    python -m benchmarks.full_benchmark --output benchmarks/full_results/RUN

The plan is finite and explicit. No timing cases run concurrently. Use --resume
with the same extension and parameters to continue a partially completed run.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

from .benchmark_stim import benchmark_case, generate_gates, git_output
from .benchmark_measurement import benchmark_measurements
from .benchmark_protocols import benchmark_protocol, benchmark_lifecycle, benchmark_shots
from .stress_qubits import memory_info


def build_plan():
    plan = []
    def add(family, **parameters):
        key = hashlib.sha256(json.dumps([family, parameters], sort_keys=True).encode()).hexdigest()[:16]
        if not any(case["id"] == key for case in plan):
            plan.append({"id": key, "family": family, "parameters": parameters})

    for n in [1, 2, 63, 64, 65, 127, 128, 129, 256, 1024, 4096]:
        for workload in ["h", "s", "cx", "mixed"]:
            if n == 1 and workload == "cx":
                continue
            for count in [1, 1000, 50000]:
                for prep in [0, 40 * n]:
                    add("gates", n=n, workload=workload, count=count, prep=prep, seed=731, locality="random")
    for n in [65, 1024, 4096]:
        for workload in ["cx", "mixed"]:
            for locality in ["local", "hotspot"]:
                add("gates", n=n, workload=workload, count=50000, prep=40 * n, seed=731, locality=locality)

    states = ["zero", "plus", "bell", "ghz", "cluster", "scrambled", "deterministic"]
    for n in [2, 63, 64, 65, 129, 256, 1024, 4096]:
        for state in states:
            for count in sorted({min(n, c) for c in [1, 8, 64, 256, n]}):
                for mode in ["native", "sequential"]:
                    add("measurement", n=n, state=state, count=count, mode=mode, seed=731)
    for n in [65, 1024]:
        for seed in [0, 2026]:
            for count in [1, 64]:
                for mode in ["native", "sequential"]:
                    add("measurement", n=n, state="scrambled", count=count, mode=mode, seed=seed)

    for n in [65, 1024, 4096]:
        for state in states:
            for basis in ["X", "Y"]:
                for count in [1, 64]:
                    for mode in ["decomposed", "native_pauli"]:
                        add("protocol", n=n, state=state, kind="basis", count=count, seed=731,
                            basis=basis, stim_mode=mode)
    for n in [65, 1024]:
        for state in ["ghz", "cluster", "scrambled"]:
            for weight in sorted({2, 16, 64, n}):
                for count in [1, 64]:
                    for mode in ["decomposed", "native_pauli"]:
                        add("protocol", n=n, state=state, kind="parity", count=count, seed=731,
                            weight=weight, stim_mode=mode)
    for n in [65, 256, 1024, 4096]:
        for state in ["zero", "cluster", "scrambled"]:
            for layer in [1, 8, 64]:
                for locality in ["random", "local"]:
                    add("protocol", n=n, state=state, kind="interleaved", count=128,
                        seed=731, layer=layer, locality=locality)
    for n in [65, 1024, 4096]:
        for state in ["plus", "scrambled"]:
            for count in [64, 1024]:
                add("protocol", n=n, state=state, kind="repeated", count=count, seed=731)
    for n in [1, 65, 1024, 4096, 16384]:
        for kind in ["initialize", "copy", "peek"]:
            add("lifecycle", n=n, kind=kind, seed=731)
        if n <= 1024:
            for kind in ["export_cold", "export_warm"]:
                add("lifecycle", n=n, kind=kind, seed=731)
    for n in [65, 256, 1024]:
        for state in ["cluster", "scrambled"]:
            for shots in [1, 32]:
                add("shots", n=n, state=state, shots=shots, seed=731)
    for mode in ["gates", "measurement"]:
        add("stress", mode=mode, auto=True, gates=10000, repeats=3, seed=731)
    add("stress", mode="gates", qubits=[100000], gates=1000000, repeats=3, seed=731)
    return plan


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def execute_case(case, repeats, output, stabilizer, stim):
    p = case["parameters"]
    family = case["family"]
    if family == "gates":
        gates = generate_gates(p["n"], p["count"], p["workload"], p["seed"])
        if p["locality"] != "random":
            local = []
            for gate in gates:
                if p["locality"] == "hotspot":
                    gate = (gate[0], *(q % 2 for q in gate[1:]))
                    if gate[0] == "cx" and gate[1] == gate[2]:
                        gate = ("cx", gate[1], 1 - gate[1])
                elif gate[0] == "cx":
                    gate = ("cx", gate[1], (gate[1] + 1) % p["n"])
                local.append(gate)
            gates = local
        prep = generate_gates(p["n"], p["prep"], "mixed", p["seed"] + 1)
        result = benchmark_case(p["n"], gates, prep, repeats, 2, stabilizer, stim)
        result["gate_sha256"] = hashlib.sha256(json.dumps(gates).encode()).hexdigest()
        return result
    if family == "measurement":
        return benchmark_measurements(p["n"], p["state"], p["count"], p["mode"], repeats,
                                      p["seed"], stabilizer, stim)
    if family == "protocol":
        extra = {k: v for k, v in p.items() if k not in ["n", "state", "kind", "count", "seed"]}
        return benchmark_protocol(p["n"], p["state"], p["kind"], p["count"], repeats,
                                  p["seed"], stabilizer, stim, **extra)
    if family == "lifecycle":
        return benchmark_lifecycle(p["n"], p["kind"], repeats, p["seed"], stabilizer, stim)
    if family == "shots":
        return benchmark_shots(p["n"], p["shots"], p["state"], repeats, p["seed"], stabilizer, stim)
    path = output / ("stress-" + case["id"] + ".json")
    command = [sys.executable, "-m", "benchmarks.stress_qubits", "--mode", p["mode"],
               "--gates", str(p["gates"]), "--repeats", str(p["repeats"]), "--seed", str(p["seed"]),
               "--backends", "stabilizer", "stim", "--reserve-mib", "768", "--timeout", "240",
               "--json", str(path)]
    command += ["--auto"] if p.get("auto") else ["--qubits", *map(str, p["qubits"])]
    log_path = output / ("stress-" + case["id"] + ".log")
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
    return json.loads(path.read_text())


def timing_medians(result):
    if "median_seconds" in result:
        return result["median_seconds"]
    if "stabilizer" in result:
        return {b: result[b]["median_seconds"] for b in ["stabilizer", "stim"]}
    return None


def write_summary(output, metadata, plan, results):
    completed = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    lines = ["# Full Clifford benchmark against Stim", "", f"Started: {metadata['started_at_utc']}", "",
             f"Machine: {metadata['cpu_model']}. Stim {metadata['stim_version']} ({metadata['stim_core']}); local release extension.", "",
             f"Completed {len(completed)} of {len(plan)} planned cases; {len(failed)} case errors. Native timing backends ran sequentially.", "",
             f"{metadata['repeats']} samples after warmup per backend (isolated stress: three gate samples); all raw samples are saved. Times include binding/result conversion. Ratios above 1 mean Stim was faster.", "",
             "## Scope and interpretation", "",
             "This is a finite matrix for the pure-state Clifford/Tableau APIs: gate kinds, circuit lengths, target locality, identity/prepared inputs, word/vector boundaries, product/Bell/GHZ/cluster/scrambled states, random/deterministic and single/partial/full Z measurements, native/sequential calls, X/Y measurements, joint Pauli products, interleaved circuits, repeated measurements, repeated fresh-tableau shots, allocation/copy/export/query, and isolated large-memory probes.", "",
             "Noise, non-Clifford algorithms, mixed-state backends, adaptive classical control, and compiled multi-shot samplers are outside this shared API comparison. Circuit and preparation seeds are fixed and recorded; selected scrambled measurement cases are repeated with three seeds. These circuits are not uniform random Cliffords, and completed memory sizes are demonstrated bounds under the measured workload.", "",
             "Normal measurement uses independent seeded RNG streams. Random/deterministic branches have the same classification despite differing signs. Small collapsed states are checked exhaustively; large cases use logical outcome replay and sampled state checks. All signed generators are compared for unitary gate cases. The local Pauli implementation uses Clifford decomposition; native_pauli compares it with Stim's native MPP. Shared decomposed cases use the same gate network in both backends. Basis/parity networks include reversal to preserve the physical post-measurement state.", "",
             "Initialization does not fully touch zero-filled pages; isolated capacity workers apply a full H layer before reporting peak RSS. Copy includes RNG state. Forward-tableau export includes Stim's inverse-to-forward conversion and produces different array organization in the two APIs. Repeated-shot cases include fresh allocation, preparation, measurement, and destruction for every shot.", "",
             "## Coverage", "", "| Family | Planned | Successful | Case errors |", "|---|---:|---:|---:|"]
    for family in dict.fromkeys(c["family"] for c in plan):
        subset = [r for r in results if r["family"] == family]
        lines.append(f"| {family} | {sum(c['family'] == family for c in plan)} | {sum(r['status'] == 'success' for r in subset)} | {sum(r['status'] != 'success' for r in subset)} |")
    lines.extend(["", "## Results", "", "Per-case times are total elapsed milliseconds for that case. Measurement counts include every logical measurement, including repeats of the same target. Fresh-shot counts are listed separately. Aggregate win counts can be dominated by the chosen matrix; inspect matching workloads.", "",
                  "| Case | Family | Parameters | Ours ms | Stim ms | Ours / Stim |", "|---|---|---|---:|---:|---:|"])
    for record in results:
        if record["status"] != "success":
            lines.append(f"| {record['id']} | {record['family']} | ERROR: {record.get('error', '')} | | | |")
            continue
        result = record["result"]
        medians = timing_medians(result)
        if medians:
            params = ", ".join(f"{k}={v}" for k, v in record["parameters"].items())
            ratio = medians["stabilizer"] / medians["stim"]
            lines.append(f"| {record['id']} | {record['family']} | {params} | {medians['stabilizer']*1000:.6f} | {medians['stim']*1000:.6f} | {ratio:.3f} |")
    lines.extend(["", "## Isolated memory/depth probes", "", "| Mode | Backend | Qubits | Status | Peak RSS GiB | Gate time s | Projection s |", "|---|---|---:|---|---:|---:|---:|"])
    for record in completed:
        if record["family"] == "stress":
            for row in record["result"]["results"]:
                lines.append(f"| {row.get('mode', record['parameters']['mode'])} | {row['backend']} | {row['qubits']} | {row['status']} | {row.get('peak_rss_bytes', 0)/1024**3:.3f} | {row.get('gate_seconds', 0):.6f} | {row.get('measurement_seconds') or 0:.6f} |")
    lines.extend(["", "Raw records: [results.json](results.json), [results.jsonl](results.jsonl). Exact matrix and machine/build metadata: [manifest.json](manifest.json). Per-case JSON and tracebacks are in `cases/`; stress worker reports and logs are adjacent to this summary."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--families", nargs="+", choices=["gates", "measurement", "protocol", "lifecycle", "shots", "stress"])
    parser.add_argument("--limit", type=int, help="run the first N planned cases for a smoke check")
    args = parser.parse_args(argv)
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeats and limit must be positive")
    import stabilizer
    import stim
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    manifest = output / "manifest.json"
    core_hash = hashlib.sha256(Path(stabilizer._core.__file__).read_bytes()).hexdigest()
    plan = [c for c in build_plan() if not args.families or c["family"] in args.families]
    if args.limit:
        plan = plan[:args.limit]
    if manifest.exists():
        if not args.resume:
            parser.error("output already contains a run; select a new directory or --resume")
        saved = json.loads(manifest.read_text())
        metadata = saved["metadata"]
        if metadata["core_sha256"] != core_hash or metadata["repeats"] != args.repeats or saved["plan"] != plan:
            parser.error("resume requires the same binary, repeats, and plan")
    else:
        cpu = Path("/proc/cpuinfo").read_text()
        metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "cpu_model": next(line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")),
                    "platform": platform.platform(), "python": sys.version,
                    "stabilizer_version": importlib.metadata.version("stabilizer"),
                    "stim_version": stim.__version__, "stim_core": stim.TableauSimulator.__module__,
                    "core_path": stabilizer._core.__file__, "core_sha256": core_hash,
                    "git_revision": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--short"),
                    "memory_at_start": memory_info(), "repeats": args.repeats,
                    "harness_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ["full_benchmark.py", "benchmark_protocols.py", "benchmark_measurement.py", "benchmark_stim.py", "stress_qubits.py"]}}
        atomic_json(manifest, {"metadata": metadata, "plan": plan})
    results = []
    for case in plan:
        path = output / "cases" / (case["id"] + ".json")
        if path.exists():
            results.append(json.loads(path.read_text()))
    completed_ids = {r["id"] for r in results}
    print(f"Output {output}; {len(plan)} cases, {len(results)} already completed; repeats={args.repeats}", flush=True)
    write_summary(output, metadata, plan, results)
    for case in plan:
        if case["id"] in completed_ids:
            continue
        print(f"[{len(results)+1}/{len(plan)}] {case['family']} {case['parameters']}", flush=True)
        started = time.monotonic()
        record = {**case, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            record.update(status="success", result=execute_case(case, args.repeats, output, stabilizer, stim))
        except Exception as error:
            record.update(status="error", error=str(error), traceback=traceback.format_exc())
        record["wall_seconds"] = time.monotonic() - started
        results.append(record)
        atomic_json(output / "cases" / (case["id"] + ".json"), record)
        with (output / "results.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        atomic_json(output / "progress.json", {"completed": len(results), "planned": len(plan),
                    "errors": sum(r["status"] != "success" for r in results), "last_case": case,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        if len(results) % 20 == 0 or record["status"] != "success":
            write_summary(output, metadata, plan, results)
        medians = timing_medians(record.get("result", {}))
        print("  " + (str(medians) if medians else record["status"]), flush=True)
    # Use plan order for a stable final artifact, including resumed runs.
    by_id = {r["id"]: r for r in results}
    results = [by_id[c["id"]] for c in plan]
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "results.json", {"metadata": metadata, "results": results})
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    write_summary(output, metadata, plan, results)
    print(f"Saved {output / 'summary.md'}", flush=True)
    return results


if __name__ == "__main__":
    main()
