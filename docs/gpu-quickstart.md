# Quickstart

Run the standard single-cell EDA pipeline on a singlify `.1pz` output, end-to-end on device. This is the canonical "what does the library do" walkthrough.

## What you'll build

```
.1pz file
   │
   ▼  (zero-copy)
DeviceCSC
   │
   ▼  qc::compute_qc_metrics + filter_cells/genes
filtered DeviceCSC
   │
   ▼  preprocess::log_normalize
normalized DeviceCSC
   │
   ▼  preprocess::select_hvg
HVG-subset DeviceCSC
   │
   ▼  preprocess::scale
scaled DeviceCSC
   │
   ▼  reduce::svd
PCA embedding (n_cells × k)
   │
   ▼  graph::knn + graph::leiden
cluster labels (n_cells)
   │
   ▼  embed::umap
2D embedding (n_cells × 2)
   │
   ▼  de::wilcoxon
per-cluster marker genes
```

Every step runs on the GPU. No host-device round-trips between steps. The matrix lives in cuSPARSE CSC the whole time.

## C++

> **Status**: pre-1.0. The umbrella header is on the roadmap (CYCLE-92 in `state/dag.md`). Until it lands, include the per-module headers directly — at your own risk for API churn.

```cpp
#include <singlet-gpu/singlet_gpu.hpp>

namespace sg = singlet_gpu;

int main() {
    auto ctx = sg::core::GpuContext{};
    auto mat = sg::io::load_pz("/path/to/exon_counts.1pz", ctx);

    // To be filled in as features reach `released` state.
    // See state/public-api.md for the current frozen surface.
}
```

## Python

> **Status**: pre-1.0. Wrappers exist module-by-module under `singlet.gpu.preprocess`, `singlet.gpu.reduce`, etc. The flat `singlet.gpu.run_pipeline(...)` convenience entry point is on the roadmap.

```python
import singlet.gpu as sg

mat = sg.io.load_pz("/path/to/exon_counts.1pz")

# Pipeline calls land here as features reach `released`.
```

## R

> **Status**: pre-1.0.

```r
library(singletGpu)

mat <- singletGpu::load_pz("/path/to/exon_counts.1pz")

# Pipeline calls land here as features reach `released`.
```

## Streaming (1M+ cells)

For datasets larger than device memory, drop in `streaming::PzShardIterator`:

```cpp
#include <singlet-gpu/singlet_gpu.hpp>

auto ctx = singlet_gpu::core::GpuContext{};
auto iter = singlet_gpu::streaming::PzShardIterator{
    "/path/to/shards/",
    /*vram_budget_gb=*/40,
};

while (auto shard = iter.next(ctx)) {
    // each shard is a DeviceCSC fitting under the budget;
    // accumulate sufficient statistics, run two-pass algorithms.
}
```

Streaming kernel contract documentation is forthcoming.

## Reproducibility

Every kernel that uses randomness takes an explicit `uint64_t seed`. The default config seeds with a documented constant — set your own seed for reproducibility across runs.

## Where to next

- Per-feature deep dives: [`api/`](api/)
- Notebooks (real-data correctness, speedup at 3 scales): [`notebooks/`](notebooks/)
- Live benchmark frontier: https://singlet.bio/benchmarks
