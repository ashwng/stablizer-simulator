"""Export a checkpointed full run as CSV and a compact overview."""

import argparse
import csv
import json
from pathlib import Path

from .full_benchmark import timing_medians


def analyze(output):
    manifest = json.loads((output / "manifest.json").read_text())
    records = ([json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
               if not (output / "results.json").exists() else json.loads((output / "results.json").read_text())["results"])
    fields = ["id", "family", "status", "parameters", "ours_seconds", "stim_seconds", "ours_over_stim",
              "ours_samples_seconds", "stim_samples_seconds", "verification", "wall_seconds"]
    with (output / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            result = record.get("result", {})
            medians = timing_medians(result)
            if not medians:
                continue
            samples = result.get("samples_seconds") or {b: result[b]["samples_seconds"] for b in ["stabilizer", "stim"]}
            writer.writerow({"id": record["id"], "family": record["family"], "status": record["status"],
                             "parameters": json.dumps(record["parameters"], sort_keys=True),
                             "ours_seconds": medians["stabilizer"], "stim_seconds": medians["stim"],
                             "ours_over_stim": medians["stabilizer"] / medians["stim"],
                             "ours_samples_seconds": json.dumps(samples["stabilizer"]),
                             "stim_samples_seconds": json.dumps(samples["stim"]),
                             "verification": result.get("verification", result.get("verified", "see suite methodology")),
                             "wall_seconds": record["wall_seconds"]})
    successful = [r for r in records if r["status"] == "success"]
    metadata = manifest["metadata"]
    lines = ["# Benchmark overview", "", f"{len(successful)} successful cases of {len(manifest['plan'])} planned; {len(records)-len(successful)} case errors.", "",
             f"{metadata['cpu_model']}; Stim {metadata['stim_version']} ({metadata['stim_core']}); local release extension.", "",
             f"Medians of {metadata['repeats']} timed samples after warmup. Preparation and validation are excluded except where an end-to-end lifecycle/shot scope is specified. Backends run sequentially. Full case parameters and all raw samples are retained.", ""]
    selectors = [
        ("50,000 gates at 4,096 qubits", lambda r: r["family"] == "gates" and r["parameters"]["n"] == 4096 and r["parameters"]["count"] == 50000),
        ("Z measurement at 4,096 qubits", lambda r: r["family"] == "measurement" and r["parameters"]["n"] == 4096 and r["parameters"]["count"] in [1, 64, 256, 4096]),
        ("X/Y measurements and parity at 1,024 qubits", lambda r: r["family"] == "protocol" and r["parameters"]["n"] == 1024 and r["parameters"]["kind"] in ["basis", "parity"] and r["parameters"]["count"] == 64),
        ("Interleaved circuits at 4,096 qubits", lambda r: r["family"] == "protocol" and r["parameters"]["n"] == 4096 and r["parameters"]["kind"] == "interleaved"),
        ("32 fresh-tableau shots at 1,024 qubits", lambda r: r["family"] == "shots" and r["parameters"]["n"] == 1024 and r["parameters"]["shots"] == 32),
    ]
    for heading, predicate in selectors:
        lines.extend(["## " + heading, "", "Times are total milliseconds; ratio >1 means Stim is faster.", "",
                      "| Parameters | Ours ms | Stim ms | Ours / Stim |", "|---|---:|---:|---:|"])
        for record in successful:
            if predicate(record):
                p = record["parameters"]
                medians = timing_medians(record["result"])
                label = ", ".join(f"{k}={v}" for k, v in p.items() if k not in ["n", "seed"])
                lines.append(f"| {label} | {medians['stabilizer']*1000:.3f} | {medians['stim']*1000:.3f} | {medians['stabilizer']/medians['stim']:.2f} |")
        lines.append("")
    memory = []
    for record in successful:
        if record["family"] == "stress":
            for row in record["result"]["results"]:
                memory.append({**row, "mode": row.get("mode", record["parameters"]["mode"]), "case_id": record["id"]})
    memory_fields = ["case_id", "mode", "backend", "qubits", "status", "peak_rss_bytes", "gate_seconds", "measurement_seconds", "worker_swap_bytes", "gates", "address_space_limit_bytes"]
    with (output / "memory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=memory_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(memory)
    lines.extend(["## Largest completed isolated probes", "", "These sizes are demonstrated under the measured workload and available RAM; they are not universal maximum qubit counts. Initialization and full-touch times, host snapshots, worker swap, failures and budget skips are in the raw reports.", "",
                  "| Mode | Backend | Qubits | Peak RSS GiB |", "|---|---|---:|---:|"])
    for mode in ["gates", "measurement"]:
        for backend in ["stabilizer", "stim"]:
            candidates = [r for r in memory if r["mode"] == mode and r["backend"] == backend and r["status"] == "success"]
            if candidates:
                row = max(candidates, key=lambda r: r["qubits"])
                lines.append(f"| {mode} | {backend} | {row['qubits']:,} | {row['peak_rss_bytes']/1024**3:.2f} |")
    lines.extend(["", "Native batches allow Stim to share conversion work. Sequential calls and first measurements have different costs. Deterministic measurement benefits from Stim's inverse tableau. Joint Pauli rows compare local Clifford decomposition either with the same decomposition or with Stim MPP, as labeled; those two modes should be interpreted separately.", "",
                  "See [summary.md](summary.md) for every case and complete methodology, [timings.csv](timings.csv) for numeric analysis, [memory.csv](memory.csv) for every isolated probe, and [results.json](results.json) for the full machine-readable run. Scope is the shared pure-state Clifford/Tableau backend; other algorithms require their own benchmarks."])
    (output / "overview.md").write_text("\n".join(lines) + "\n")
    return {"completed": len(records), "successful": len(successful), "planned": len(manifest["plan"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(analyze(args.output))
