---
feature: lognorm
roadmap_id: 2
module: include/singlet-gpu/preprocess/lognorm.h
status: design
tolerance: relative L_inf ≤ 1e-6 vs scanpy normalize_total + log1p (fp64 reference); element-wise rel ≤ 1e-5
target_perf: 1M cells in ≤6ms on A100 (beat rapids-singlecell 7.39ms by 18%); 100k in ≤0.8ms; 10k in ≤0.1ms
ooc_plan: streamed via PzChunkIterator — per chunk: compute partial column sums on device, optional global median pass on host (single fp64 reduce), apply normalization in place
---

## Algorithm

Single fused CUDA kernel that takes a `singlet_gpu::core::DeviceCSC` (= `factornet::gpu::SparseMatrixGPU<float>`) and computes per-cell size factors then `log1p(x_i / s_j * scale)` for every nonzero, in place.

Three modes via `LogNormConfig::method`:

1. **`TotalCount`** (default — consensus per lit-scout): scale per cell to the median total count then log1p.
   - Per-cell total: `t_j = sum_{i in col j} x_ij`
   - Target: `T = median(t_j)` over cells with `t_j > 0`, or user-supplied scalar.
   - Per-cell size factor: `s_j = t_j / T` (cells with `t_j == 0` get `s_j = 1` and a flag in the QC mask).
   - In-place update: `x_ij ← log1p(x_ij / s_j)` for every nonzero.

2. **`ScranDeconvolution`**: deferred. Pool-based deconvolution requires the QR / pool layout that scran uses, and lit-scout flagged it as not GPU-friendly (no current GPU implementation). Implement the API, return `STATUS_NOT_IMPLEMENTED` until cycle ≥ 8.

3. **`Downsample`**: deferred. Hypergeometric per-cell resampling is also not GPU-friendly per lit-scout. Same status.

`Mode::TotalCount` is the only mode shipped in cycle 3. The others are API-shaped and return early.

### Kernel passes

**Pass 1 — `compute_col_sums`** (one warp per column, parallel reduce over the column's nonzeros):
- Each warp reads `indptr[j]` and `indptr[j+1]`, then sweeps `values[indptr[j] .. indptr[j+1])` summing in fp32 with **Kahan compensation** (lit-scout pitfall: fp32 sum overflows on >50M-read cells).
- Warp-level reduction via `__shfl_down_sync`, then one thread writes `t[j]`.
- Output: `DeviceMemory<float> t(n)` — column totals.

**Pass 2 — `compute_target`** (single block, n entries):
- Cast `t` to fp64 in shared memory if `n ≤ 4096`, else use a deterministic radix select on `t`.
- Compute median of `t_j > 0` (skip cells with zero counts — matches scanpy semantics).
- Or: if `LogNormConfig::target_count` is user-supplied (>0), use it directly.
- Output: scalar `T` on device.

**Pass 3 — `compute_size_factors`** (one thread per column):
- `s[j] = (t[j] > 0) ? (t[j] / T) : 1.0f`.
- Mark a per-cell flag in `qc_mask[j]` (uint8) when `t[j] == 0`.
- Output: `DeviceMemory<float> s(n)`, `DeviceMemory<uint8_t> qc_mask(n)`.

**Pass 4 — `apply_lognorm`** (one warp per column, fused):
- For each nonzero `x_ij`: load `s[j]` once into shared memory at warp start, then `x_ij ← log1pf(x_ij / s[j])`.
- Single global memory pass over `values[]`. Indices and indptr untouched.
- This is the ScaleSC "fused VectorAdd+Div+Log1p" pattern lit-scout flagged as the SOTA to beat.

The four passes use one stream; passes 3 and 4 can be merged into one kernel launch since they touch disjoint memory and are O(nnz) bound. Pass 2 is a sync point.

### What we beat

- **rapids-singlecell normalize_total**: 7.39 ms on 1M cells (A100). Target: ≤6 ms via fused 4-in-1 vs their separate normalize + log1p calls.
- **Scanpy CPU**: ~45 min on 1M cells. Trivially.
- **ScaleSC**: comparable fused kernel; aim to match within 10% on A100 and beat them via stream concurrency on the ooc path.

## Numerical stability

- **Kahan summation in pass 1** to handle ≥50M-read cells without overflow (lit-scout pitfall).
- **fp64 accumulator for the median target** (pass 2) — small `n` so cost is negligible. Cast back to fp32 on store.
- **`log1pf` not `logf(1+x)`** — log1pf is accurate to fp32 ULP for `x` near 0; `logf(1+x)` loses the entire mantissa for tiny x.
- **Division by zero guarded**: `s[j] = 1.0f` when `t[j] == 0`. The QC mask records the event so downstream kernels can filter.
- **No fp64 in pass 4** (the hot loop) — fp32 throughout. Welford / Kahan only in passes 1 and 2.

Reference comparison: scanpy uses fp64 internally for `normalize_total` and `log1p`. Our fp32 must match within `rel ≤ 1e-5` element-wise. Lit-scout cites rapids-singlecell hitting fp64 equivalence with the same fp32 strategy.

## Memory layout

- Input: `DeviceCSC` (m × n, fp32 values, int32 indptr/indices). Untouched except `values[]`.
- Workspace: `t[n]` (fp32) + `s[n]` (fp32) + `qc_mask[n]` (uint8) — total `9n` bytes for outputs, all owned via `factornet::gpu::DeviceMemory<T>`.
- Peak device memory increase: `9n` bytes above the input CSC. For 1M cells: 9 MB. Negligible.
- No host copies during the kernel.

## Streams

- One stream, passed in by the caller (`core::default_context().stream()` if none provided).
- All four passes launched async on the same stream — no cross-stream sync needed since they're sequential within the column dimension.
- Producer-consumer overlap with downstream kernels (HVG, PCA) is the caller's responsibility — they pass their stream and we run on it.

## Out-of-core chunking

`PzChunkIterator` yields fixed-column-width slices. For lognorm:

1. **First pass over all chunks**: compute per-chunk `t[j_chunk]` arrays, accumulate into a global `t[n_total]` on host (single fp64 reduce).
2. **Single host-side median** over `t[n_total]` → `T`.
3. **Second pass over all chunks**: load chunk, compute `s_chunk = t_chunk / T`, apply `log1p(x / s)`, write back (or stream to next downstream kernel).

Memory bound by chunk size, not total cell count. Streaming is two-pass (median requires it); a single-pass approximation (running median estimator) is offered as `LogNormConfig::approximate_median = true` for the billion-cell case where two passes are infeasible.

For in-memory mode the four passes run sequentially on one stream as described above.

## Determinism

The `compute_col_sums` pass uses warp-level reductions which are deterministic given fixed warp ordering on a fixed architecture. The median pass is deterministic by construction. The `apply_lognorm` pass is a pure elementwise op — bit-identical across runs.

The `deterministic` flag in `LogNormConfig` is therefore a no-op — this kernel is always deterministic. Document this in the header.

## Correctness test spec

Test file: `tests/preprocess_lognorm_correctness.cpp`.

Inputs:
1. **Tiny synthetic**: 500 × 200 fixed-seed CSC, dense fraction 0.1, values uniform [1, 100]. Run our kernel; in a Python subprocess run `scanpy.pp.normalize_total(adata) → log1p(adata)` on the same matrix (scipy CSC); copy our device result back; compare element-wise.
2. **GSM4037629** (11,560 cells) exon_counts.1pz: load via `pz_device_loader` (cycle 2 dependency), normalize, compare to scanpy.
3. **100k concat**: 10 scRNA samples concatenated, same comparison.
4. **Edge cases**:
   - Cell with `t_j == 0` → size factor 1, qc_mask bit set, values untouched.
   - Cell with single huge count (`>2^24`) → kahan kicks in, no overflow, matches scanpy.
   - User-supplied `target_count` → bypass median, use directly.

Tolerance: per-element `rel_err ≤ 1e-5` AND `abs_err ≤ 1e-7` AND `L_inf ≤ 1e-6` (scanpy is fp64 reference). Spearman ρ on per-cell totals after normalization ≥ 0.9999.

Reference: scanpy `sc.pp.normalize_total` + `sc.pp.log1p` in a subprocess. Subprocess command:
```
python -c "
import scanpy as sc, scipy.sparse as sp, numpy as np
adata = sc.read_h5ad('input.h5ad')
sc.pp.normalize_total(adata, target_sum=None)  # median
sc.pp.log1p(adata)
np.save('expected.npy', adata.X.toarray())
"
```

Validator dumps our device result via a small `tests/refs/dump_csc.cpp` driver, then `numpy.allclose` compares.

## Target performance

| Scale | Cells | nnz | Target wall | SOTA (rapids-singlecell) | Notes |
|---|---|---|---|---|---|
| tiny | 200 | 2k | <0.05ms | n/a | smoke test |
| 10k | 11,560 | ~30M | <0.1ms | ~0.5ms | beat by ≥5× |
| 100k | ~120k | ~300M | <0.8ms | ~1.2ms | beat by ≥1.5× |
| 1M | ~1M | ~3B | <6ms | 7.39ms | beat by ≥18% (lit-scout's reported number) |

Memory: ≤ input_csc_size + 9n bytes (negligible overhead).

## Implementation notes (for gpu-kernel-dev when cycle 3 dispatches)

- Header path: `include/singlet-gpu/preprocess/lognorm.h`
- API:
  ```cpp
  namespace singlet_gpu::preprocess {
      enum class LogNormMethod { TotalCount, ScranDeconvolution, Downsample };
      struct LogNormConfig {
          LogNormMethod method = LogNormMethod::TotalCount;
          float target_count = 0.0f;          // 0 = use median
          bool approximate_median = false;     // streaming mode
          uint64_t seed = 0;                   // unused (deterministic)
      };
      struct LogNormResult {
          singlet_gpu::core::DeviceMemory<float> size_factors;  // s[n]
          singlet_gpu::core::DeviceMemory<uint8_t> qc_mask;     // 1 = zero-count cell
          float target_used;                                     // median or user value
      };
      LogNormResult log_normalize(
          singlet_gpu::core::DeviceCSC& mat,            // mutated in place (values[])
          const LogNormConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- The kernel is in-place on `mat.values`. Indptr/indices untouched.
- Test data: GSM4037629 (cycle 2 loader output).
- Build flag: needs `FACTORNET_HAS_GPU=1` (set by INTERFACE target).
- Dependencies: `core/types.h` (DeviceCSC alias), `core/handles.h` (default stream), `core/memory.h` (DeviceMemory through factornet).
- `// SPDX-License-Identifier: GPL-2.0-or-later` first line.
