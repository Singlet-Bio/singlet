# singlet-gpu Benchmark Harness

This directory contains the benchmark framework for singlet-gpu. Every kernel in
`include/singlet-gpu/` ships with a `bench_{feature}_perf.cpp` driver here that
measures wall time, peak device memory, and throughput against SOTA baselines.

---

## Quick Start

```bash
# 1. Build the library and bench drivers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. Run all benchmarks (requires GPU)
bash bench/run_all.sh

# 3. View results
cat state/benchmark-registry.md
```

Results are appended to `state/benchmark-registry.md`. Logs land in `bench/logs/`.

---

## Directory Layout

```
bench/
  include/singlet_gpu/bench/harness.h   -- header-only bench utilities (BenchTimer, etc.)
  refs/
    common.py                            -- shared SOTA subprocess launcher
    requirements.txt                     -- pip deps for all baselines
  logs/                                  -- per-run stderr logs ({timestamp}_{feature}_{scale}.log)
  bench_{feature}_perf.cpp               -- one driver per feature (built by CMake)
  run_all.sh                             -- orchestrator: runs every driver, seeds registry
  README.md                              -- this file
```

---

## The BenchRow Schema

Every driver produces rows matching this schema in `state/benchmark-registry.md`:

| Column | Type | Description |
|---|---|---|
| `feature` | string | Roadmap identifier, e.g. `io/pz_loader`, `preprocess/lognorm` |
| `scale` | enum | `tiny` \| `10k` \| `100k` \| `1m` |
| `wall_ms_min` | float | Min wall time over 5 timed iterations (ms) |
| `wall_ms_med` | float | Median wall time over 5 timed iterations (ms) |
| `wall_ms_max` | float | Max wall time over 5 timed iterations (ms) |
| `mem_mb_peak` | float | Peak device memory delta (MB) via `cudaMemGetInfo`; RSS for CPU |
| `cells_per_s` | float | Throughput: cells ÷ `wall_ms_med` × 1000 |
| `sota_wall_sec` | float | SOTA reference wall time (s); `TBD` if not yet measured |
| `sota_mem_mb` | float | SOTA peak memory (MB); `TBD` if not yet measured |
| `ratio_wall` | float | `sota_wall_sec × 1000 / wall_ms_med`; >1 means we are faster |
| `commit` | string | Short git SHA of singlet-gpu commit benchmarked, or `no-git` |
| `timestamp` | string | ISO-8601 UTC run timestamp |

---

## How to Write a New Bench Driver

When a new feature lands in `include/singlet-gpu/{module}/{name}.h`, the author MUST
write a corresponding `bench/bench_{module}_{name}_perf.cpp`. Follow this template:

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// bench/bench_{module}_{name}_perf.cpp
//
// Benchmark driver for singlet-gpu::{module}::{name}.
// Runs 3 warmup + 5 timed GPU iterations at each requested scale.
// Logs BenchRow to state/benchmark-registry.md via harness::log_row().

#include "singlet_gpu/bench/harness.h"
#include "singlet-gpu/{module}/{name}.h"
#include <cstdlib>
#include <string>

// Parse --scale, --sample-dir, --commit, --timestamp, --registry from argv.
// Use harness::parse_args() (defined in harness.h).

int main(int argc, char** argv) {
    auto cfg = singlet_gpu::bench::parse_args(argc, argv);

    // 1. Skip if no GPU
    if (!singlet_gpu::bench::gpu_available()) {
        std::puts("NO GPU — skipping " BENCH_FEATURE_NAME);
        return 0;
    }

    // 2. Load canonical sample for this scale (or generate tiny synthetic)
    auto mat = singlet_gpu::bench::load_sample(cfg);  // returns DeviceCSC or nullptr
    if (!mat) {
        std::puts("SKIP — sample not found for this scale");
        return 0;
    }

    // 3. SOTA reference via subprocess
    double sota_wall_s = singlet_gpu::bench::run_sota_ref(
        cfg.scale, cfg.sample_dir,
        "python bench/refs/common.py --feature {name} --scale " + cfg.scale_str
    );  // returns -1.0 if SOTA not available; log as TBD

    // 4. Warm-up (3 discarded iterations)
    for (int i = 0; i < 3; ++i) {
        singlet_gpu::{module}::{the_kernel}(*mat /*, ...params... */);
    }

    // 5. Timed iterations (5 real iters via cuEvents)
    singlet_gpu::bench::BenchTimer timer;
    for (int i = 0; i < 5; ++i) {
        timer.start();
        singlet_gpu::{module}::{the_kernel}(*mat /*, ...params... */);
        timer.stop();
    }

    // 6. Build and log the row
    singlet_gpu::bench::BenchRow row{};
    row.feature      = "{module}/{name}";
    row.scale        = cfg.scale_str;
    row.wall_ms      = timer.stats();   // {min, med, max}
    row.mem_mb_peak  = singlet_gpu::bench::peak_mem_delta_mb(mat->bytes_device());
    row.cells_per_s  = mat->n_cols() / (row.wall_ms.med / 1000.0);
    row.sota_wall_s  = sota_wall_s;
    row.commit       = cfg.commit;
    row.timestamp    = cfg.timestamp;
    singlet_gpu::bench::log_row(row, cfg.registry_path);

    return 0;
}
```

### Checklist for a new driver

- [ ] File is named `bench_{module}_{name}_perf.cpp` (underscores, not slashes)
- [ ] Added to `bench/CMakeLists.txt` via `add_bench_driver({module}_{name})`
- [ ] 3 warmup + 5 timed iterations using `BenchTimer` (cuEvents, not `chrono`)
- [ ] SOTA reference invoked via `bench/refs/common.py` or an R subprocess; failure is logged as `TBD`, not a hard error
- [ ] `--scale tiny` works with no sample on disk (synthetic 500×200 fixed-seed)
- [ ] `--scale 10k` requires `SAMPLE_DIR` pointing to GSM4037629
- [ ] `--scale 100k` skips cleanly if no concat found
- [ ] `--scale 1m` skips with message ("use bench_streaming_perf")
- [ ] Driver exits 0 on skip/no-GPU, non-zero only on internal crash
- [ ] Row appended to `state/benchmark-registry.md` via `harness::log_row()`

---

## Fallback Behavior

| Condition | Behavior |
|---|---|
| No GPU (`nvidia-smi` absent) | `run_all.sh` prints "NO GPU — skipping bench" and exits 0 |
| No bench binaries in `build/bench/` | `run_all.sh` prints build hint and exits 0 |
| Sample directory missing for a scale | Driver skips that scale with a SKIP message; other scales still run |
| SOTA subprocess fails (package not installed) | Driver logs `sota_wall_sec=TBD`, `sota_mem_mb=TBD` and continues |
| Driver binary crashes | `run_all.sh` logs the non-zero exit, records the log path, continues with next driver |
| `state/benchmark-registry.md` missing | `run_all.sh` creates it with the correct header before first append |

---

## Installing SOTA Baselines

```bash
# Activate or create the bench venv
python -m venv bench_env && source bench_env/bin/activate

# Install all baselines (GPU node with CUDA 12.x required for rapids)
pip install -r bench/refs/requirements.txt

# R baselines (fgsea, DESeq2, mgatk) — run once on the compute node
Rscript -e 'install.packages(c("BiocManager"), repos="https://cran.r-project.org")'
Rscript -e 'BiocManager::install(c("fgsea","DESeq2","mgatk","scran"))'
```

SOTA baselines are optional — missing packages cause `TBD` entries in the registry,
not failures. The harness never blocks a benchmark run on SOTA availability.

---

## Benchmark Registry Seed (cycle 53a)

Cycle 53a pre-populates `state/benchmark-registry.md` with placeholder rows for all
9 core drivers being written in parallel. These rows have `scale=pending` and
`wall_ms_min=TBD` until the first GPU dispatch populates them.

Features seeded: `io/pz_loader`, `preprocess/lognorm`, `preprocess/hvg`,
`reduce/svd`, `reduce/nmf`, `graph/knn`, `graph/leiden`, `embed/umap`, `de/wilcoxon`.

---

## Contributing

1. Write the kernel in `include/singlet-gpu/{module}/{name}.h` (via `gpu-kernel-dev`).
2. Write the correctness test in `tests/{name}_correctness.cpp` (via `analysis-validator`).
3. Write the bench driver in `bench/bench_{module}_{name}_perf.cpp` (same cycle).
4. Add `add_bench_driver({module}_{name})` to `bench/CMakeLists.txt`.
5. Add a placeholder row to `state/benchmark-registry.md` via the seed pattern in cycle 53a.
6. Mark the feature `in-progress` → `done` in `state/feature-roadmap.md` only after
   the bench row shows a real number (not TBD).

No feature is on the Pareto frontier without a populated bench row.
