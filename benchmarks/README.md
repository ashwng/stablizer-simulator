# Clifford gate benchmarks against Stim

From the repository root, build the current Rust extension with optimizations:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'maturin>=1.7,<2' '.[benchmark]'
maturin develop --release
python benchmarks/benchmark_stim.py --json /tmp/clifford-benchmark.json
```

`maturin develop --release` is essential after changing Rust code. A debug build
or a previously installed extension gives misleading results. The harness prints
package versions and records the imported module path in JSON.

A short run, or a larger sweep:

```bash
python benchmarks/benchmark_stim.py --qubits 64 65 --gates 1000 --repeats 3
python benchmarks/benchmark_stim.py --qubits 64 256 1024 4096 \
    --gates 50000 --workloads mixed cx --repeats 15 --warmup 3 \
    --seed 123 --json /tmp/clifford-benchmark.json
```

The workloads apply H only, S only (Rust's P), CNOT only, or a uniform random mix
of H/S/CNOT. Targets are uniform random qubits; CNOT endpoints always differ.
Each backend receives exactly the same elementary gates, generated once using
Python's seeded RNG. Stim can merge adjacent instructions of the same kind;
reported gate counts count each single-qubit operation or CNOT pair separately.

Before each execution, both simulators allocate fresh n-qubit tableaus and apply
the same random preparation circuit of `4 * n` gates. This preparation is outside
the timer and avoids benchmarking only sparse identity tableaus. Override with
`--prep-gates COUNT`, including `0` for the identity. This short preparation is
not a uniform random Clifford. Workloads containing CNOT require n >= 2; for
n = 1 select `--workloads h s mixed` (mixed then uses H/S).

Each sample times one `stabilizer.Tableau.apply_circuit` or
`stim.TableauSimulator.do` call using `perf_counter_ns`. Both gate loops execute
in native code. Circuit generation, initial tableau allocation, preparation, and
correctness checks are excluded. Internal temporary allocations and layout
conversions during execution are included. Each timer includes one Python/native call;
increase `--gates` if this overhead matters. Warmup samples are discarded,
backend order alternates, and Python garbage collection is disabled during the
timing loop and restored afterward.

Every case first checks **all signed destabilizer and stabilizer rows** against
Stim's forward tableau (the inverse of `current_inverse_tableau()`). A mismatch
aborts the run. This checks Clifford action up to global phase, including idle
qubits. The harness compares the current Aaronson–Gottesman `Tableau` backend.

The terminal reports medians, nanoseconds per elementary gate, and Rust/Stim
time ratios. A ratio of 10 means Rust took ten times as long. JSON also contains
all raw timings, min/max, median absolute deviations, throughput, seed, workload
parameters, Python/package/platform versions, and Git revision/status. Run on an
otherwise idle machine; compare repeated runs before interpreting small gaps.
The JSON also records Stim's loaded simulator module, which identifies whether
the installed wheel uses SSE2 or AVX2. SIMD support varies between wheels.

## Vectorized Rust execution

`Tableau` initializes the identity directly in column-packed storage. Gates,
including individual calls, short circuits, and repeated `apply_circuit` calls,
update those columns in place. There is no gate-count threshold or conversion
at the end of a unitary circuit. H, S, and CNOT update 64 rows per word using
AND/XOR operations, including all row signs. On x86-64, runtime CPU detection
selects AVX2 kernels operating on 256 rows per vector. Other CPUs use portable
word operations; no `target-cpu=native` flag is required.

Measurement works directly on column storage. Random projection multiplies the
pivot into selected rows in groups of 64, tracking each product's phase with two
packed bit planes. Only words containing affected rows are visited, and identity
terms of the pivot are skipped. Deterministic measurement computes a selected
stabilizer product's sign using packed parity and prefix-XOR operations. Both
paths use O(n) temporary storage and leave columns ready for the next gate.
Forced measurement checks the branch once rather than recomputing its outcome.

Other row algorithms lazily obtain a synchronized row view using a 64x64 bit
transpose. Read-only inspection caches that view without
invalidating the columns. Row mutation makes the columns stale; the next gate
repopulates the existing column buffers once. Consecutive row operations need
no further conversion. Discarded row views are retained as reusable conversion
buffers. Clones copy authoritative state and discard derived read caches;
concurrent readers can safely share the lazily constructed row view. If explicit
row edits have made rows authoritative, measurement uses the existing row
algorithm until a gate restores column storage.

The unitary benchmark therefore measures persistent gate updates after the
initial tableau construction and preparation. With `--prep-gates 0`, identity
construction still initializes columns directly outside the timer. If a circuit
contains explicit row operations, any required conversions happen inside
execution and count toward its time. State storage and initialization remain
quadratic in the number of qubits. Alternating gates and measurements no longer
requires layout conversions. Dense measurement still has quadratic worst-case
work; this change does not add parallel threads or switch to an inverse tableau.

The original bottleneck was updating one bit per row while stepping over
row-major storage. Packing across rows eliminates most of that work and enables
contiguous SIMD loads. Stim also uses a cache-friendly SIMD layout and an
inverse tableau; its implementation is described in
[Stim's paper](https://arxiv.org/abs/2103.02202). Its inverse tableau and compiled
sampling optimizations have benefits beyond this unitary-only benchmark.

This measures unitary Clifford gate updates on a single tableau. Measurement,
noise, and repeated-shot sampling are separate workloads. In particular, Stim's
compiled samplers use additional optimizations and are not used here. See the
[Stim API reference](https://github.com/quantumlib/Stim/wiki/Stim-v1.15-Python-API-Reference)
for the simulator APIs and
[Stim's paper](https://arxiv.org/abs/2103.02202) for its algorithms.

To run the harness regression checks:

```bash
python -m pip install pytest
python -m pytest tests/test_benchmark_stim.py
```

## Maximum-qubit stress tests (Linux)

```bash
python benchmarks/stress_qubits.py --auto --json /tmp/qubit-limits.json
python benchmarks/stress_qubits.py --auto --mode measurement \
    --json /tmp/measurement-limits.json
python benchmarks/stress_qubits.py --qubits 65536 98304 131072 \
    --gates 10000 --reserve-mib 512 --timeout 180 --json /tmp/explicit-limits.json
```

Each candidate runs in a separate process with an address-space cap based on
current available RAM (including any cgroup limit). By default, 512 MiB remains
for the host and 128 MiB inside the cap is reserved for Python/native libraries
and circuits. `--reserve-mib` may be reduced to 128, but available memory varies
while other applications run. Swap is not counted as RAM; workers stop on low
host memory, increased swap pressure, or the per-candidate timeout. A skipped
candidate exceeds the configured budget; it is not a demonstrated physical OOM.

The worker initializes the column tableau and applies H to **every qubit**.
This writes the full X/Z storage, so the reported peak resident memory cannot be
explained by Linux's lazy allocation of untouched zero pages. It then times
10,000 seeded random H/S/CNOT gates (override with `--gates`), applies their
inverse and the inverse H layer, and checks identity bits/signs at up to 64
sampled qubits. These checks are sampled, not exhaustive. No duplicate Stim
tableau, clone, statevector, or full row export is created in gate-only mode.

Measurement mode additionally applies H to the last qubit and forces a random
Z measurement to zero, exercising projection directly in column storage.
This tests memory capacity for one simple measurement, not throughput on dense
measurement-heavy circuits. Column storage uses approximately `n² / 2` bytes;
measurement adds only O(n) workspace. Explicit row exports or row mutations can
still create a second full tableau. Padding, Python, circuits, and allocator
overhead add to these estimates.

Reports contain initialization/full-touch/gate/measurement timings, peak RSS,
worker swap usage, sampled-check counts, memory snapshots, and each worker's
address-space cap. The largest passing candidate is a demonstrated lower bound
under that workload, not a universal maximum for all circuits or environments.

Before column-native measurement, the recorded run on this 15.32 GiB machine reached 155,648 gate-only qubits
(11.31 GiB peak RSS; 10,000 mixed gates in 0.090 s) and 111,616 qubits including
one forced measurement (11.63 GiB peak RSS; measurement/row conversion in
16.64 s). The workers reported zero swap usage at successful completion. Probes
at 157,696 gate-only qubits and 112,640 measurement qubits were stopped because
host swap usage increased. Available RAM varied between probes; 155,648 also
triggered swap pressure in an earlier attempt before completing on retry.
These historical results are workload-specific demonstrated capacities, not absolute hardware or
algorithm limits. Full measurements and memory snapshots are in
[qubit_limits.json](qubit_limits.json).

## Isolated stress comparison with Stim

```bash
python benchmarks/stress_qubits.py --qubits 65536 98304 131072 139264 \
    --backends stabilizer stim --repeats 3 --json /tmp/stress-gates.json
python benchmarks/stress_qubits.py --qubits 65536 98304 106496 110592 131072 139264 \
    --backends stabilizer stim --mode measurement --json /tmp/stress-measurements.json
python benchmarks/stress_qubits.py --qubits 100000 --gates 1000000 \
    --backends stabilizer stim --repeats 3 --json /tmp/stress-depth.json
```

Use this harness for capacity comparisons: the ordinary `benchmark_stim.py`
holds both tableaus and creates additional correctness views, which may exceed
RAM at large qubit counts. Stress workers hold only their own backend, alternate
backend order between sizes, and record the native implementation/version. Each
worker builds the same seeded H/S/CNOT stream and records its SHA-256 hash;
the parent checks matching hashes before calculating comparison ratios.

Gate timing is the median of `--repeats` native circuit calls after one warmup.
Each call starts from the same prepared state; the inverse is applied outside
the timer between calls. Allocation, the full H sweep, circuit construction,
and sampled correctness checks remain outside the gate timer. Measurement is
one forced-random Z projection on an otherwise restored zero state:
`Tableau.measure(force=0)` versus `TableauSimulator.postselect_z(desired_value=False)`.
Projection timing includes any layout conversion and is not a dense-measurement
throughput claim. No compiled Stim sampling is used.

Our worker checks sampled signed Clifford identity bits. Stim's worker uses
`peek_z`/`peek_x` to check sampled zero-state observables without copying its
tableau. Regression tests compare the full signed tableaus at small sizes.
These large probes therefore verify sampled restoration, not exhaustive Clifford
equivalence. Both workers write generator data throughout their tableaus and
report measured peak RSS and swap usage.

The Stim memory-budget estimate includes its default 1.1x growth reserve along
both tableau axes, yielding roughly 21% more allocation than exact dimensions
before SIMD padding. See its
[allocation implementation](https://github.com/quantumlib/Stim/blob/main/src/stim/simulators/tableau_simulator.inl).
Measured RSS is authoritative; estimates only determine which candidates fit
the configured budget. A skipped candidate is not an observed OOM. After a
worker fails or encounters pressure, that backend stops increasing, while the
other backend may continue. Previous single-backend commands remain supported.

The comparison below predates column-native measurement and used the local
AVX2 extension and Stim 1.16.0's SSE2 wheel. At 100,000 qubits, the median of
three one-million-gate executions was
3.688 s for the local simulator and 4.406 s for Stim, with peak worker RSS of
4.75 GiB and 5.79 GiB respectively. Both completed warmup, inverse, full-memory
touch and sampled restoration checks with identical gate-stream hashes.

For the forced-random measurement at 98,304 qubits, the local simulator took
12.030 s with 9.03 GiB peak RSS; Stim took 6.381 s with 5.48 GiB peak RSS.
The largest completed gate-only probes in this comparison were 147,456 local
qubits and 141,312 Stim qubits. Including projection, the largest completed
probes were 108,544 local qubits and 139,264 Stim qubits. Available memory varied;
higher local probes encountered swap pressure, and oversized Stim candidates
were skipped by the memory budget. These are completed test sizes, not proven
maximum capacities. All samples, memory snapshots, stream hashes and failed or
skipped cases are in [stim_stress_results.json](stim_stress_results.json).

## Column-native measurement benchmarks

Use the isolated stress harness above for large product-state projections and
peak RAM. For entangled single measurements and batches, run:

```bash
python -m benchmarks.benchmark_measurement --qubits 256 1024 4096 \
    --states ghz cluster scrambled deterministic --counts 1 64 256 \
    --modes native --repeats 7 --json /tmp/measurement-native.json
python -m benchmarks.benchmark_measurement --qubits 1024 4096 \
    --states ghz cluster scrambled --counts all --modes native \
    --repeats 7 --json /tmp/measurement-all.json
python -m benchmarks.benchmark_measurement --qubits 256 1024 4096 \
    --states ghz cluster scrambled --counts 1 64 --modes sequential \
    --repeats 7 --json /tmp/measurement-sequential.json
```

This harness applies the same seeded preparation to both backends, with fresh
states for each sample. Preparation and verification are outside timing; one
warmup is discarded. `native` times one local `apply_circuit` call containing Z
measurements versus one Stim `measure_many` call. `sequential` times one normal
`measure` call per qubit in each backend. Timings include binding overhead and
result conversion. Targets are distinct, chosen by one seeded permutation;
counts larger than the state are capped and duplicate cases are removed.

GHZ is globally entangled but has only one random Z result; subsequent Z
outcomes are deterministic. A connected linear cluster state has random Z
outcomes for every target. Scrambled states start from that cluster and apply
`40 * n` seeded H/S/CNOT gates. Deterministic cases use signed computational
basis states in a scrambled CNOT generator basis. The report counts random and
deterministic projections and samples the initial single-qubit X/Y/Z
expectations: all three being zero establishes entanglement with the rest of
the pure state. These circuits are not uniform random Cliffords.

Both backends use normal measurement with independent seeded RNGs, so their
outcome bit strings can differ. For Clifford/Pauli circuits the sequence of
random versus deterministic branches is independent of outcome signs; the
local native flags are checked against Stim's plan. Each warmup outcome vector
is replayed through Stim postselection outside timing to verify the branch is
possible, and sampled collapsed qubits are checked. At n <= 129, the full
collapsed state is compared by canonical stabilizers. Regression tests exercise
all state families and both call modes. Full generators are also compared in
the separate unitary benchmark. No compiled shot sampler is used.

The former forced-outcome harness is retained for reproducing the historical
optimization comparison:

```bash
python -m benchmarks.benchmark_measurement --qubits 256 1024 4096 \
    --workloads entangled_first entangled_batch deterministic --repeats 5 \
    --json /tmp/measurement-legacy.json
```

That legacy mode uses one forced call per qubit for random measurements, which
prevents Stim from sharing a transpose over a batch. Use the native mode above
to assess batched measurement throughput.

The before/after experiment preserves the previous release extension and runs
the same harness against each build. At 98,304 qubits the isolated simple
projection changed from 11.802 s / 9.03 GiB peak RSS to 0.00672 s / 4.53 GiB;
Stim measured 6.293 s / 5.48 GiB in the new comparison. A projection at 131,072
local qubits also completed using 8.03 GiB. These single large projection probes
show the benefit of avoiding a full transpose and allocation; dense measurement
does more work. Median entangled and deterministic timings and raw isolated
probe reports are saved in [measurement_optimization.json](measurement_optimization.json).
Stim's inverse tableau remains advantageous for deterministic measurements;
this optimization retains the forward Aaronson–Gottesman tableau.

The current gate, entangled-state, and multi-qubit comparison is summarized in
[stim_comparison.md](stim_comparison.md). It covers 256, 1,024, and 4,096 qubits,
including batches of 1/64/256 Z measurements and full-register measurements at
1,024 and 4,096 qubits. Raw samples and metadata are saved in
[stim_gate_comparison.json](stim_gate_comparison.json),
[stim_measurement_native.json](stim_measurement_native.json),
[stim_measurement_all.json](stim_measurement_all.json), and
[stim_measurement_sequential.json](stim_measurement_sequential.json).

## Comprehensive checkpointed suite

```bash
python -m benchmarks.full_benchmark --output benchmarks/full_results/NEW_RUN --repeats 5
# Continue an interrupted run using the same extension, parameters and matrix:
python -m benchmarks.full_benchmark --output benchmarks/full_results/NEW_RUN --repeats 5 --resume
# Export numeric CSVs and a compact overview from a complete or partial run:
python -m benchmarks.analyze_full_benchmark benchmarks/full_results/NEW_RUN
```

The default plan contains 1,090 cases. It covers H/S/CNOT/mixed gates; short and
long circuits; random, local and concentrated targets; identity and prepared
inputs; 64-bit and SIMD boundary sizes; zero/plus/Bell/GHZ/cluster/scrambled and
dense deterministic generator states; single/partial/full-register Z
measurements; native batches and sequential calls; X/Y basis measurements;
joint Pauli products; interleaved gates and measurements; repeated measurement
of the same target; repeated fresh-tableau shots; initialization, copying,
inspection and forward-tableau export; and isolated RAM/depth probes.

Joint Pauli and X/Y operations are implemented locally as reversible Clifford
networks. `decomposed` runs that same network in Stim; `native_pauli` lets Stim
use MPP. Both modes include measurement and restoration of the physical basis.
Repeated-shot cases allocate and simulate a fresh tableau per shot in each
backend. This suite describes the shared pure-state Clifford backend; other
algorithms such as noise simulation, non-Clifford states, adaptive classical
control, mixed states and compiled multi-shot sampling require separate suites.

Timing runs are sequential. The manifest records the exact finite matrix,
seeds, machine details, binary hash, source hashes, and dirty Git status. Every
case saves all raw samples to its own JSON file and to an append-only JSONL
checkpoint. `progress.json` records completed/error counts, and `summary.md`
is refreshed every 20 cases. On completion, `results.json` contains the entire
run. Small protocol states are compared exhaustively against logical Stim
replay; large cases use sampled checks. Resource-limited stress workers report
their status, peak RSS, swap use and memory snapshots explicitly.

The full run for this workspace is saved under
[full_results/2026-10-18/](full_results/2026-10-18/), including its
[manifest](full_results/2026-10-18/manifest.json),
[progress](full_results/2026-10-18/progress.json), and
[summary](full_results/2026-10-18/summary.md).
The same directory also contains `overview.md`, `timings.csv`, `memory.csv`,
`environment.json`, and a `source_snapshot/` with the Rust and harness sources
used for this run. Existing benchmark reports are retained separately.
