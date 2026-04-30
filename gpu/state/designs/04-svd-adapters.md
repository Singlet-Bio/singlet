---
feature: svd_adapters
roadmap_id: 4
module: include/singlet-gpu/reduce/svd/{lanczos,irlba_factornet,randomized,krylov_constrained,deflation,auto_select}.h
status: design
tolerance: bit-identical to factornet `*_svd_gpu` (same algorithm, same config — adapter must not corrupt the call). Validation = round-trip CPU vs GPU on identical input.
target_perf: ≤2% overhead vs direct factornet call (the adapter only adds parameter marshaling + optional pinned-host extraction from PzDeviceMatrix)
ooc_plan: factornet SVD does not stream; for chunks ≥1M cells, the orchestrator must concat in host memory or run incremental SVD via repeated calls (deferred to cycle 7+)
---

## Algorithm

Six **adapter** headers, each ~50–80 LOC, wiring our `PzDeviceMatrix` (or any equivalent host-pinned CSC) to factornet's `*_svd_gpu` functions. Algorithm logic is entirely owned by factornet — we only marshal arguments.

The **major architectural correction from cycle 4 code-reader**: factornet's GPU SVD functions take HOST pointers, not a `SparseMatrixGPU<float>`. Our adapters bridge host→factornet, with factornet doing the internal H2D. To avoid wasted decompression work, the loader needs to retain its pinned host buffers when SVD is in the pipeline.

## Loader extension (prerequisite, cycle 5 ships this too)

Extend `singlet_gpu::io::PzDeviceMatrix` and `load_pz`:

```cpp
struct PzDeviceMatrix {
    factornet::gpu::SparseMatrixGPU<float> mat;
    Metadata meta;
    cudaStream_t producer_stream;

    // NEW (cycle 5):
    // Optional pinned host buffers, retained when load_pz(..., keep_host_pinned=true).
    // SVD adapters use these to call factornet without re-staging.
    std::shared_ptr<int>   host_indptr;     // size n+1
    std::shared_ptr<int>   host_indices;    // size nnz
    std::shared_ptr<float> host_values;     // size nnz
    bool host_retained = false;
};

PzDeviceMatrix load_pz(const std::string& path,
                       cudaStream_t stream = nullptr,
                       bool keep_host_pinned = false);
```

The default is `keep_host_pinned = false` so existing kernels (lognorm, hvg) get no overhead. SVD adapters set it to `true` via the orchestrator-level pipeline. The shared_ptrs use a custom deleter that calls `cudaFreeHost`.

## Adapter headers

Each header is ≤80 LOC including doc.

### 1. `reduce/svd/lanczos.h`

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/svd/lanczos_gpu.cuh — factornet::svd::lanczos_svd_gpu<float>

#pragma once
#include <factornet/svd/lanczos_gpu.cuh>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/svd/types.h>  // SvdConfig + SvdResult re-exports

namespace singlet_gpu::reduce::svd {

inline SvdResult lanczos(const PzDeviceMatrix& m, const SvdConfig& cfg) {
    require_host_retained(m, "lanczos");
    return factornet::svd::lanczos_svd_gpu<float>(
        m.host_indptr.get(), m.host_indices.get(), m.host_values.get(),
        m.mat.rows, m.mat.cols, m.mat.nnz,
        cfg);
}

}  // namespace
```

That is the entire header. Same shape for all five backends.

### 2. `reduce/svd/irlba_factornet.h`

Wraps `factornet::svd::irlba_svd_gpu<float>`. Same shape.

### 3. `reduce/svd/randomized.h`

Wraps `factornet::svd::randomized_svd_gpu<float>`. Same shape.

### 4. `reduce/svd/krylov_constrained.h`

Wraps `factornet::svd::krylov_svd_gpu<float>`. Same shape, but documents that constraint fields (L1/L2/non-neg/L21/angular/graph) on `SvdConfig` are honored by this backend.

### 5. `reduce/svd/deflation.h`

Wraps `factornet::svd::deflation_svd_gpu<float>`. Same shape, plus documents that `robust_delta > 0` triggers IRLS, `graph_u`/`graph_v` triggers Laplacian regularization.

### 6. `reduce/svd/auto_select.h`

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/svd/auto_select.hpp + all five backend headers

#pragma once
#include <factornet/svd/auto_select.hpp>
#include <singlet-gpu/reduce/svd/lanczos.h>
#include <singlet-gpu/reduce/svd/irlba_factornet.h>
#include <singlet-gpu/reduce/svd/randomized.h>
#include <singlet-gpu/reduce/svd/krylov_constrained.h>
#include <singlet-gpu/reduce/svd/deflation.h>

namespace singlet_gpu::reduce::svd {

inline bool has_constraints(const SvdConfig& cfg) {
    return cfg.L1_u != 0 || cfg.L2_u != 0 || cfg.nonneg_u
        || cfg.L1_v != 0 || cfg.L2_v != 0 || cfg.nonneg_v
        || cfg.upper_bound_u > 0 || cfg.upper_bound_v > 0
        || cfg.L21_u != 0 || cfg.L21_v != 0
        || cfg.angular_u != 0 || cfg.angular_v != 0
        || cfg.graph_u != nullptr || cfg.graph_v != nullptr;
}

inline SvdResult auto_select(const PzDeviceMatrix& m, int k, const SvdConfig& cfg) {
    auto method = factornet::svd::auto_select_svd_method(
        k, m.mat.rows, m.mat.cols,
        /*is_gpu*/ true,
        /*has_constraints*/ has_constraints(cfg),
        /*prefer_memory_efficiency*/ false);
    SvdConfig cfg_with_k = cfg;
    cfg_with_k.k = k;
    switch (method) {
        case factornet::svd::SVDMethod::LANCZOS:    return lanczos(m, cfg_with_k);
        case factornet::svd::SVDMethod::IRLBA:      return irlba_factornet(m, cfg_with_k);
        case factornet::svd::SVDMethod::RANDOMIZED: return randomized(m, cfg_with_k);
        case factornet::svd::SVDMethod::KRYLOV:     return krylov_constrained(m, cfg_with_k);
        case factornet::svd::SVDMethod::DEFLATION:  return deflation(m, cfg_with_k);
    }
    return SvdResult{};  // unreachable
}

}  // namespace
```

### `reduce/svd/types.h` (new, prerequisite for the six adapters)

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/svd/{auto_select.hpp, gateway.hpp} — type re-exports

#pragma once
#include <factornet/svd/auto_select.hpp>  // pulls SVDConfig + SVDResult

namespace singlet_gpu::reduce::svd {
    using SvdConfig = factornet::svd::SVDConfig<float>;
    using SvdResult = factornet::svd::SVDResult<float>;
    using SvdMethod = factornet::svd::SVDMethod;
}
```

### `require_host_retained` helper

A small inline helper that throws (or returns an error code, depending on style) when `m.host_retained` is false. Drives the orchestrator-level invariant that SVD requires the loader to be called with `keep_host_pinned=true`.

## Numerical stability

Owned by factornet. We pass `cfg.tol`, `cfg.max_iter` through. fp32 default. Document in each header that fp64 (`double`) is available by re-exporting templates if a future cycle needs it.

## Memory layout

- Input: `PzDeviceMatrix` with `host_retained = true`. Memory cost: device CSC (used by lognorm/hvg upstream) + pinned host CSC (used by SVD).
- factornet does its own internal H2D copy and SVD workspace. Workspace size depends on backend and `k` — see factornet docs.
- Output: `SvdResult` containing host-side `U`, `d`, `V` matrices + metadata.

## Streams

Stream is owned by factornet's `GPUContext` (which it creates internally on each call). Our adapter does NOT pass a `cudaStream_t`. Document this — SVD calls do not overlap with caller-provided streams unless factornet exposes a stream-aware overload (it does not currently).

**Open follow-up**: file `CYCLE-5-FOLLOWUP-FACTORNET-STREAM-OVERLOAD` to request stream-aware factornet GPU APIs.

## Out-of-core

factornet's GPU SVD does not stream — it allocates the full matrix on device internally. For chunks ≥1M cells, the orchestrator must:

- Use `randomized_svd_gpu` (lowest memory footprint due to fixed work).
- OR run `factornet::svd::streaming_matvec` (factornet/svd/streaming.hpp) — read in cycle 7 to confirm signature.

Defer the streaming SVD path to cycle 7.

## Determinism

Owned by factornet. Lanczos with full reorthogonalization is deterministic given a fixed seed; randomized SVD is deterministic given a fixed seed (cfg.seed). Document this in each adapter header.

## Correctness test spec

Test file: `tests/reduce_svd_correctness.cpp`. The diff-test confirms the adapter does not corrupt the factornet call:

1. **Round-trip via direct factornet call**: load GSM4037629 with `keep_host_pinned=true`, run `singlet_gpu::reduce::svd::lanczos(m, cfg)` and ALSO call `factornet::svd::lanczos_svd_gpu<float>` directly with the same arguments. Compare `U`, `d`, `V` element-wise — must be **bit-identical** (rel_err == 0).
2. **Per-backend smoke test**: tiny synthetic 500 × 200 fixed-seed CSC. Run all 5 backends + auto_select. Confirm:
   - Singular values agree pairwise across backends to relative L2 ≤ 1e-4 (different algorithms, different convergence criteria).
   - Subspace angles on top-k vectors ≤ 1e-3 radians.
3. **`auto_select` routing test**: synthetic matrices of various m, n, k. Confirm:
   - k=10 (no constraints) → LANCZOS selected.
   - k=40 (no constraints) → RANDOMIZED selected (GPU path).
   - k=80 (no constraints) → IRLBA selected.
   - k=10, L1_u=0.1 → KRYLOV selected.
   - k=4, L1_u=0.1 → DEFLATION selected.
4. **Constraint test** (KRYLOV): non-negativity on `u` → all elements of `U` ≥ 0.

## Target performance

| Scale | Cells | k | Backend | factornet wall (target) | adapter overhead | our wall |
|---|---|---|---|---|---|---|
| 10k | 11,560 | 50 | RANDOMIZED | ~30ms | <1ms | <31ms |
| 100k | ~120k | 50 | RANDOMIZED | ~250ms | <1ms | <251ms |
| 1M | ~1M | 50 | RANDOMIZED | ~3s | <2ms | <3.005s |
| 10k | 11,560 | 200 | IRLBA | ~150ms | <1ms | <151ms |

The adapter contributes essentially zero runtime. The story is "we provide the bridge — factornet provides the speed."

SOTA to beat (factornet handles this already):
- rapids-singlecell PCA: ~50ms / 10k, ~500ms / 100k. factornet's randomized GPU is ≥1.5× faster per its own benchmarks.
- cuml randomized SVD: ~40ms / 10k. factornet matches.
- IRLBA Python: 100× slower (CPU).

## Implementation notes

- Six headers under `include/singlet-gpu/reduce/svd/` plus one `types.h`.
- Each adapter ≤80 LOC including license + doc.
- Total module: ~500 LOC across 7 files.
- Cycle-2 loader extension also required: add `keep_host_pinned` flag and the three `shared_ptr` fields. Estimated +30 LOC.
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: cycle 2 (loader), cycle 1 (core types).
- `// SPDX-License-Identifier: GPL-2.0-or-later` first line on every file.
- `// integrates: factornet/svd/{file}_gpu.cuh` second comment.

## Risks

1. **SvdConfig field set may differ from what we expect**. The cycle 5 code-reader summary listed the fields, but we did not verify the full struct. Validator's first run on the GPU node will catch any signature mismatch.
2. **factornet may evolve its API** — pinning to factornet at a specific commit / tag is recommended once GPU dispatch starts. Currently we read whatever is on disk at `/mnt/home/debruinz/factornet/`.
3. **The host-pinned retention burns memory** — for very large inputs the 2× cost may bite. Cycle 7 streaming work can address this by chunking before SVD.
