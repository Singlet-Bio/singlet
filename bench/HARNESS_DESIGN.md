# singlet-gpu Benchmark Harness — Design

Status: design only (no drivers committed yet — see `state/self-repair-log.md` 2026-05-13 OBS-loop-operator).
Owner: singlet-gpu agent; gpu-bench sub-agent implements.
Last revised: Cycle 90, 2026-05-13.

## Why this exists

Cycles 85, 86, and earlier reference bench drivers (`bench_de_wilcoxon_perf_c85.cpp`, `bench_de_ttest_perf.cpp`, `bench/refs/ttest_ref.py`) that produced real numbers in `state/gpu/benchmark-registry.md` but **were never committed** — they lived in scratch dirs and were lost. Without a committed harness, every new feature has to re-derive timing infrastructure and `Cycle 85 OOM follow-up` cannot be fixed (there's nothing to patch).

This document defines the layout, signatures, and patterns so any cycle can recreate or add a driver without rediscovering the conventions.

## Directory layout

```
bench/
├── HARNESS_DESIGN.md         # this file
├── CMakeLists.txt            # one add_bench() macro; per-driver target
├── common/
│   ├── bench_harness.h       # OOM-guard, timing helpers, scale presets
│   ├── csc_loader.h          # load .1pz → DeviceCSC, with size precheck
│   └── ref_dispatch.h        # invoke Python/R refs via subprocess + capture
├── refs/                     # baseline implementations (CPU / SOTA)
│   ├── ttest_ref.py          # scanpy/scipy CPU baseline
│   ├── wilcoxon_ref.py       # scanpy wilcoxon
│   ├── nmf_ref.py            # sklearn.decomposition.NMF
│   ├── qc_ref.py             # sc.pp.calculate_qc_metrics
│   ├── scran_ref.R           # R scran (when installed)
│   └── env/                  # pinned conda env + R lockfile
├── bench_de_wilcoxon.cpp     # one driver per feature × phase
├── bench_de_ttest.cpp
├── bench_qc.cpp
├── bench_nmf.cpp
├── bench_pca.cpp             # adopt-winner consolidation lives here
└── runner/
    ├── run_all.sh            # SLURM submission script
    └── parse_results.py      # aggregate driver stdout → benchmark-registry rows
```

## Driver signature contract

Every bench driver follows this signature so `runner/parse_results.py` can extract uniformly:

```cpp
// bench_<feature>.cpp
#include "common/bench_harness.h"
#include "singlet/gpu/<feature_header>.h"

int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);   // --scale {small,medium,large}
                                                // --rep {N}
                                                // --warmup {N}
                                                // --seed {N}
                                                // --matrix {path.1pz | synth:rows:cols:density}
    log_machine_info();                          // GPU model, mem, CUDA ver, driver

    // PRE-OOM GUARD: estimate working set + check device + host
    auto plan = estimate_workspace(cfg);
    if (!fits_in_device(plan)) {
        emit_skip("scale exceeds device memory", plan);
        return 0;                                // exit 0; skipped is not a failure
    }
    if (!fits_in_host_pinned(plan)) {            // <-- CYCLE 85 BUG WAS HERE
        emit_skip("scale exceeds host pinned budget", plan);
        return 0;
    }

    auto mat = load_matrix(cfg);                 // pinned host → device

    // Warmup
    for (int i = 0; i < cfg.warmup; ++i) run_once(mat, cfg);

    // Timed
    std::vector<double> times;
    for (int i = 0; i < cfg.rep; ++i) times.push_back(time_once_ms(mat, cfg));

    emit_result(cfg.feature, cfg.scale, median(times), peak_mem_mb(), throughput(cfg, times));
    return 0;
}
```

`emit_result` writes a single stdout line in benchmark-registry's pipe-delimited format:

```
RESULT | <date> | <feature> | <scale> | <impl> | <median_ms> | <peak_mb> | <throughput> | ...
```

`runner/parse_results.py` `grep`s `^RESULT` and appends matching rows to `state/gpu/benchmark-registry.md`.

## Cycle 85 OOM-guard pattern (the bug being recreated to fix)

The bug: `std::vector::reserve` for the synthetic matrix's host CSR arrays threw `bad_alloc` *before* the GPU OOM check fired, so the driver crashed without an actionable skip message.

The fix that goes in `common/bench_harness.h`:

```cpp
struct WorkspacePlan {
    size_t device_bytes;        // PCA workspace + matrices on device
    size_t host_pinned_bytes;   // staging for transfer
    size_t host_pageable_bytes; // synthesizer or .1pz decompress
};

bool fits_in_host_pageable(const WorkspacePlan& p) {
    struct sysinfo si{}; sysinfo(&si);
    size_t available = si.freeram * si.mem_unit;
    return p.host_pageable_bytes < available * 0.8;   // 80% headroom
}
```

`estimate_workspace()` must be **called before any `vector::reserve` or `cudaMalloc`** and must cover synthesizer overhead.

## Scale presets

Standard sizes used by all drivers (matches existing benchmark-registry data):

| Scale  | rows (genes) | cols (cells) | density | nnz       | Use when                   |
|--------|--------------|--------------|---------|-----------|----------------------------|
| small  | 200          | 500          | 0.10    | ~10K      | smoke test, correctness    |
| medium | 30,000       | 20,000       | 0.05    | ~30M      | typical real-world         |
| large  | 30,000       | 100,000      | 0.05    | ~150M     | SOTA-pressure scale        |
| huge   | 30,000       | 1,000,000    | 0.01    | ~300M     | streaming/billion-cell     |

Drivers MUST support `--scale huge` even if they `emit_skip` for it — the harness needs to record "tried, can't fit" not "didn't try."

## Reference baselines

Each feature dispatches a CPU/SOTA reference via `refs/<feature>_ref.py` (or `.R`):

- Same `--matrix` and `--seed` arguments so the reference runs on identical input.
- Reference outputs are compared post-hoc by `analysis-validator` (separate concern from this harness).
- Reference timing is recorded as a distinct row in benchmark-registry with `impl=scanpy-cpu` / `impl=sklearn-nmf-cpu` / etc.

`refs/env/` pins versions (one conda env per language). Reproducibility matters more than freshness.

## CMake integration

`bench/CMakeLists.txt` (to be written) follows the `tests/cpp/CMakeLists.txt` macro pattern:

```cmake
# only built when SINGLET_BUILD_BENCH=ON
if(NOT SINGLET_BUILD_BENCH)
    return()
endif()

macro(add_gpu_bench name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE singlet-gpu CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)
    target_compile_options(${name} PRIVATE -O3)
endmacro()

add_gpu_bench(bench_de_wilcoxon)
add_gpu_bench(bench_de_ttest)
add_gpu_bench(bench_qc)
add_gpu_bench(bench_nmf)
add_gpu_bench(bench_pca)
```

Top-level `CMakeLists.txt` adds `option(SINGLET_BUILD_BENCH "Build benchmarks" OFF)` and `if(SINGLET_BUILD_BENCH) add_subdirectory(bench) endif()`.

## What this design intentionally does NOT cover

- The actual kernel correctness tests — `tests/cpp/` owns those.
- Multi-GPU benchmarks — punt to a later cycle when NCCL is wired.
- Profile-guided variants (Nsight Systems) — separate `profile/` dir if/when needed.
- Cluster autoscaling — keep `runner/run_all.sh` SLURM-only for now.

## Next-cycle pickup list

1. Implement `common/bench_harness.h` with `estimate_workspace`, `fits_in_*`, `emit_result`, `emit_skip`, `BenchConfig`, `parse_args`.
2. Implement `common/csc_loader.h` (uses existing `io/pz_device_loader.h`).
3. Port the `refs/` Python scripts (signatures are in `state/gpu/benchmark-registry.md` Cycle 85/86 comments).
4. Write `bench_qc.cpp` first (smallest, most-isolated feature). Validate end-to-end with `--scale small`.
5. Add `bench/CMakeLists.txt`. Verify `cmake -B build -DSINGLET_BUILD_BENCH=ON && cmake --build build/bench` succeeds.
6. After bench_qc works, port bench_nmf, bench_pca, bench_de_wilcoxon, bench_de_ttest from the Cycle 85/86 numbers as references for expected runtimes.

Each step above is a separate cycle. Don't bundle.
