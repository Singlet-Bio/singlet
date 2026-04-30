# singlet-gpu — Integration Notes (HISTORICAL)

> **⚠ HISTORICAL — superseded by CYCLE-105 / CYCLE-107 (2026-04-29).**
>
> Through Cycle 104, factornet (GPL-2.0, Zach DeBruine, 2021–2026) was the GPU
> linear-algebra backend for singlet-gpu's PCA + NMF + cross-validation. After
> hitting persistent nvcc-with-pybind11 build failures from latent template
> instantiation bugs in factornet's `gpu/loss.cuh` and `gpu/batch_nnls.cuh`,
> CYCLE-105 (per user directive) forked the GPU portions of factornet's
> algorithms internally. CYCLE-106 + CYCLE-107 finished removing every
> `#include <factornet/...>` from the foundational compile path.
>
> **Current state (2026-04-29)**: native kernels in `include/singlet-gpu/core/`
> (DeviceCSC, DeviceDense, DeviceMemory, GPUContext) and `include/singlet-gpu/
> reduce/{svd,nmf}/` (deflation/randomized SVD, MU+CD NMF, speckled CV).
> No external linear-algebra backend. License remains GPL-2.0-or-later;
> algorithm credit to factornet preserved in SPDX + `// derived from factornet/...`
> comments where directly ported. Multi-modal `FactorGraph` + CSI-GEP +
> OmniDoublet remain deferred-indefinitely (gated behind
> `SINGLET_GPU_BUILD_DEFERRED`); their factornet refs stay in those headers
> until either ported or the scope reopens.
>
> The CMakeLists.txt migration valve (`-DFACTORNET_INCLUDE_DIR=...`) still
> exists for old build commands but is no longer required. New builds skip
> factornet entirely.
>
> The contents below are preserved as **historical reference** only — they
> describe an integration model that no longer exists. Do NOT use them to
> guide new code.

---

## License

singlet-gpu is **GPL-2.0-or-later**, inherited from factornet. Algorithm credit
to Zach DeBruine (factornet, 2021–2026) preserved in SPDX + per-file `derived
from factornet/...` comments where directly ported. Every singlet-gpu source
file carries:

```cpp
// SPDX-License-Identifier: GPL-2.0-or-later
```

---

## --- Below this line: pre-CYCLE-105 historical content ---

---

## factornet on disk

- Root: `/mnt/home/debruinz/factornet/include/factornet/`
- Top-level umbrella: `factornet.hpp`
- API guide: `GUIDE.md` (15 sections — read once at orchestrator boot)
- License: GPL-2.0
- Author: Zach DeBruine (2021–2026)

### Subtree map

```
factornet/include/factornet/
├── core/              types, traits, constants, memory, logging, resource_tags
├── math/              blas, loss
├── rng/               rng
├── primitives/        backend abstraction (CPU vs GPU dispatch)
├── svd/               5 SVD backends — both .hpp (CPU) and *_gpu.cuh (GPU)
│                       lanczos, irlba, randomized, krylov, deflation
│                       gateway.hpp + auto_select.hpp + spmv.hpp + streaming.hpp
├── nmf/               full NMF stack
│                       fit_cpu.hpp, fit_gpu.cuh, fit_chunked_gpu.cuh
│                       fit_cv.hpp, fit_cv_gpu.cuh, speckled_cv.hpp
│                       nmf_init.hpp, masked_nnls.hpp, nnls_streaming.hpp
│                       loss_tracker.hpp, explicit_loss.hpp
│                       fit_streaming_spz.hpp  ← streaming from .spz
│                       cpu_alignment_guide.hpp + gpu_alignment_guide.cuh  ← batch-effect steering
│                       variant_helpers.hpp
├── gpu/               GPU bridge & primitives
│                       types.cuh        ← SparseMatrixGPU<T>, DenseMatrixGPU<T>, DeviceMemory<T>, GPUContext
│                       loader.hpp       ← runtime dlopen for CPU-only builds
│                       bridge_nmf.hpp   ← function-pointer bridge
│                       gram.cuh         ← cuBLAS SYRK
│                       rhs.cuh          ← cuSPARSE SpMM
│                       nnls.cuh         ← per-column CD on device
│                       loss.cuh         ← reduction kernel
│                       batch_nnls.cuh, bipartition.cuh, context.cuh,
│                       cv_delta.cuh, cv_kernels.cuh, dclust.cuh,
│                       dispersion.cuh, fused_cv.cuh, graph_reg_gpu.cuh, k2.cuh
├── graph/             FactorGraph DAG: hierarchical, multi-modal joint, shared-H NMF
│                       graph_all.hpp (umbrella)
│                       node.hpp, graph.hpp, fit.hpp, result.hpp
├── clustering/        clustering primitives
├── features/          feature extraction
├── io/                CPU IO
│                       loader.hpp, file_reader.hpp, in_memory.hpp
│                       spz_loader.hpp           ← streampress .spz v2 reader
│                       dense_spz_loader.hpp
│                       caching_loader.hpp
│                       chunk_size.hpp
│                       ping_pong_prefetch.hpp
├── profiling/         timing harness
└── util/              misc utilities
```

---

## API touchpoints (what singlet-gpu actually calls)

| singlet-gpu adapter | factornet entry point | Header |
|---|---|---|
| `core/types.h` | `factornet::gpu::SparseMatrixGPU<float>` | `gpu/types.cuh` |
| `core/types.h` | `factornet::gpu::DenseMatrixGPU<float>` | `gpu/types.cuh` |
| `core/types.h` | `factornet::gpu::DeviceMemory<T>` | `gpu/types.cuh` |
| `core/handles.h` | `factornet::gpu::GPUContext` | `gpu/types.cuh` |
| `reduce/svd/lanczos.h` | `factornet::svd::lanczos_gpu(...)` | `svd/lanczos_gpu.cuh` |
| `reduce/svd/irlba_factornet.h` | `factornet::svd::irlba_gpu(...)` | `svd/irlba_gpu.cuh` |
| `reduce/svd/randomized.h` | `factornet::svd::randomized_gpu(...)` | `svd/randomized_gpu.cuh` |
| `reduce/svd/krylov_constrained.h` | `factornet::svd::krylov_gpu(...)` | `svd/krylov_gpu.cuh` |
| `reduce/svd/deflation.h` | `factornet::svd::deflation_gpu(...)` | `svd/deflation_gpu.cuh` |
| `reduce/svd/auto_select.h` | `factornet::svd::auto_select(...)` | `svd/auto_select.hpp` |
| `reduce/svd/streaming.h` | `factornet::svd::streaming_matvec(...)` | `svd/streaming_matvec.hpp` |
| `reduce/nmf/fit.h` | `factornet::nmf::fit_gpu(A, k, config)` | `nmf/fit_gpu.cuh` |
| `reduce/nmf/cv.h` | `factornet::nmf::speckled_cv(...)` | `nmf/speckled_cv.hpp` |
| `reduce/nmf/streaming.h` | `factornet::nmf::fit_streaming_spz(...)` (with our chunk iterator) | `nmf/fit_streaming_spz.hpp` |
| `reduce/nmf/graph.h` | `factornet::graph::FactorGraph<float>` + `factornet::graph::fit(net, A)` | `graph/graph_all.hpp` |
| `reduce/nmf/alignment.h` | `factornet::nmf::gpu_alignment_guide(...)` | `nmf/gpu_alignment_guide.cuh` |
| `streaming/chunk_iter.h` | `factornet::io::loader<T>` interface | `io/loader.hpp` (we implement the interface for `.1pz` chunks) |

---

## Build flag

Every translation unit that includes a factornet GPU header must compile with `-DFACTORNET_HAS_GPU=1`. The CMake INTERFACE target sets this for everyone who links `singlet-gpu::singlet-gpu`.

---

## Compile-time / runtime checklist

1. `find_path(CUDAToolkit)` — required.
2. `find_package(Eigen3)` — required (factornet uses Eigen::Sparse for the CPU side; `gpu/types.cuh` does not need it directly but other factornet headers do, transitively).
3. `FACTORNET_INCLUDE_DIR` set in `CMakeLists.txt` (default `/mnt/home/debruinz/factornet/include`).
4. `FACTORNET_HAS_GPU=1` exported by the INTERFACE target.

---

## What we implement, what we do NOT

| Thing | Source |
|---|---|
| `.1pz` device loader | **singlet-gpu** (factornet has no `.1pz` reader) |
| Device CSC type | factornet `gpu::SparseMatrixGPU` (we re-export under `singlet_gpu::core::DeviceCSC`) |
| Device dense type | factornet `gpu::DenseMatrixGPU` |
| RAII device memory | factornet `gpu::DeviceMemory` |
| cuBLAS / cuSPARSE / cuSOLVER handles + streams | factornet `gpu::GPUContext` |
| PCA / SVD (5 backends + auto-select) | factornet `svd/*_gpu.cuh` |
| NMF (every loss + every constraint + auto-switching) | factornet `nmf/fit_gpu.cuh` |
| Hierarchical NMF / multi-modal joint NMF | factornet `graph::FactorGraph` |
| Speckled-mask CV for auto-rank | factornet `nmf::speckled_cv` |
| Batch-effect alignment guide | factornet `nmf::gpu_alignment_guide` |
| Log-normalize / size factors | **singlet-gpu** |
| Highly variable genes | **singlet-gpu** |
| kNN graph (brute-force, IVF-PQ, HNSW-GPU) | **singlet-gpu** |
| Leiden / Louvain | **singlet-gpu** |
| UMAP | **singlet-gpu** |
| Wilcoxon / t-test / NB GLM DE | **singlet-gpu** |
| Marker scoring / reference annotation | **singlet-gpu** |
| GSEA / AUCell | **singlet-gpu** |
| Harmony / scVI-lite / BBKNN | **singlet-gpu** |
| Intron-aware velocity prep | **singlet-gpu** (unique to us) |
| MT heteroplasmy lineage | **singlet-gpu** (unique to us) |
| Donor-aware pseudobulk DE | **singlet-gpu** (unique to us) |
| Out-of-core streaming driver | **singlet-gpu** (factornet has streaming for `.spz`; we adapt the interface for `.1pz`) |
| Python wrappers (pybind11) | **singlet-gpu** |
| R wrappers (Rcpp) | **singlet-gpu** |

---

## Risks / open questions

0. **🔴 CRITICAL — int32 nnz cap in `SparseMatrixGPU<float>`**: factornet's device CSC type uses `int` (not `int32_t` or `int64_t`) for `rows`, `cols`, `nnz`. Hard cap ~2.1B nnz per matrix. At typical scRNA density (1k–3k nonzeros per cell), this means **a single device matrix holds 0.7M–2M cells max**. Implications:
   - The streaming driver (feature 16) is **required** for the 1M+ benchmark scale, not just billion-cell. Roadmap must promote it earlier.
   - Cycle-2 loader's `PzChunkIterator` is the right primitive — every kernel feature 6+ should consume chunks, not whole matrices, when targeting 1M+ cells.
   - Possible future workaround: fork factornet to use `int64_t` indices, or upstream a PR. Defer until we know the kernels actually hit the cap in practice.
   - Not blocking for cycles 3–5 (lognorm, HVG, SVD adapters): all of those run on a single chunk at a time at the 100k scale we benchmark first.

0a. **🔴 CRITICAL — factornet GPU SVD/NMF take HOST pointers, not device pointers**: The top-level GPU SVD functions in factornet have signatures like:
   ```cpp
   factornet::svd::lanczos_svd_gpu<float>(
       const int* h_col_ptr, const int* h_row_idx, const float* h_values,
       int m, int n, int nnz,
       const SVDConfig<float>& config);
   ```
   They take CSC arrays in **host memory** and presumably copy to device internally. They do NOT take a `factornet::gpu::SparseMatrixGPU<float>` despite that type existing. Same expected for `nmf::fit_gpu`. Implications:
   - The "thin adapter" model still holds, but adapters bridge **host→host** at the call site, with factornet doing its own H2D copy.
   - Our `pz_device_loader` already produces a pinned host CSC during decompression. To avoid wasted work, extend `PzDeviceMatrix` with an optional `keep_host_pinned` flag — when set, the pinned host buffers are retained alongside the device CSC so SVD adapters can pass them directly to factornet without re-staging. Memory cost: 2× the matrix while both live.
   - For kernels that need both (e.g., HVG on device, then SVD via factornet), the dual-residency is acceptable (small fraction of total RAM).
   - **Open follow-up**: submit a PR to factornet adding `*_svd_gpu_device(int* d_col_ptr, ...)` overloads that skip the H2D copy. File as `CYCLE-5-FOLLOWUP-FACTORNET-DEVICE-OVERLOAD` after cycle 5 ships.
   - **Open follow-up**: confirm `factornet::nmf::fit_gpu` signature when reading it for cycle 6. Same correction may apply.

0b. **`SVDConfig<Scalar>` and `SVDResult<Scalar>` are unified across backends**: All five SVD GPU functions take the same `SVDConfig<Scalar>` and return the same `SVDResult<Scalar>`. This means our adapter is effectively a re-export of these two types plus six pass-through functions. Even thinner than originally planned. We re-export under `singlet_gpu::reduce::{SvdConfig, SvdResult}`.

0c. **`auto_select_svd_method` requires explicit `has_constraints` flag**: It is NOT auto-detected from the config. Our adapter for `auto_select.h` must inspect the SvdConfig and set the flag based on whether any constraint field is non-zero. Wrap this in a helper.

0d. **NMF / SVD config field names diverge** (cycles 5 + 6 corrections):
   - SVD: `SVDConfig<float>::k_max` (the rank field is named `k_max`, not `k`)
   - NMF: `NMFConfig<float>::rank` (named `rank`, not `k` or `k_max`)
   - Confirmed at `factornet/core/config.hpp:61`. Two different config types use two different field names. Adapters and tests must match each.

0e. **NMF SolverMode and InitMode are int fields, not enums** (cycle 6). `NMFConfig::solver_mode` and `NMFConfig::init_mode` are plain `int`. We cannot create `using` aliases for non-existent enum types. Document the integer values in the adapter header docs.

0f. **NMF type namespace placement** (cycle 6 corrections):
   - `LossType` lives in `factornet::` namespace (header `factornet/math/loss.hpp`), NOT `factornet::nmf::`.
   - `FactorConfig<float>` lives in `factornet::` namespace (header `factornet/core/factor_config.hpp`), NOT `factornet::nmf::`.
   - `LazySpeckledMask<float>` lives in `factornet::nmf::` (`factornet/nmf/speckled_cv.hpp`).

0g. **`factornet::io::DataLoader<Scalar>` interface uses out-params** (cycle 6 correction): `next_forward(Chunk<Scalar>& out)` and `next_transpose(Chunk<Scalar>& out)` take an output reference, NOT return an `Eigen::SparseMatrix<Scalar>`. The `Chunk<Scalar>` struct wraps the Eigen sparse plus chunk metadata. Our `PzDataLoader` implements this interface as out-param.

0h. **`SharedNode` exists** for true shared-H multi-modal NMF (cycle 6 correction; reverses cycle 5's incorrect claim). Use `SharedNode<float>({inputs}, k, name)`, not `ConcatNode`. The cycle 5 code-reader's GUIDE.md-only read missed this — the actual `node.hpp` header has it. Lesson: read the actual `node.hpp`, not just the API guide.

1. **Eigen on CPU side**: factornet's CPU API takes `Eigen::SparseMatrix<float>`. If singlet-gpu ever needs to round-trip through CPU (e.g., to dump for debugging), we go via Eigen. The GPU path does NOT touch Eigen.
2. **`io::loader<T>` interface**: factornet's streaming NMF takes a `loader<T>` interface. We need to implement this interface for `.1pz` chunks. Unknown until we read `io/loader.hpp` — first cycle todo for `code-reader`.
3. **`.spz` vs `.1pz`**: different formats. factornet's `spz_loader.hpp` is irrelevant for `.1pz`. But the loader *interface* (`loader.hpp`) is reusable.
4. **Threadblock count for `nnls.cuh`**: factornet may set its own launch config based on GPU compute capability. We trust it.
5. **Determinism**: factornet's `fit_gpu` may use atomicAdd. Our determinism opt-in (rule §⛔18) cannot promise reproducibility for PCA/NMF unless factornet itself supports it. Document this caveat in `style-rules.md` §I.
6. **`FactorConfig` vs our `FactorConfig`**: we re-export factornet's `FactorConfig<float>` directly. Do not invent a parallel one.
7. **Scalar type**: factornet templates on `float` and `double`. We standardize on `float` (fp32) per absolute rule §⛔5; expose a `double` overload only when a kernel design doc justifies it.
8. **GPU on login node**: there is no GPU here. All compilation, testing, and benchmarking must dispatch to a compute node. Cycle work that does NOT need GPU (design docs, header source, test specs, benchmark drivers) completes locally.
