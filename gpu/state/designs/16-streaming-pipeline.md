---
feature: streaming_pipeline
roadmap_id: 16
module: include/singlet-gpu/streaming/streamed_pipeline.h
status: design
tolerance: end-to-end equivalence vs in-memory at 100k cells (HVG top-2000 Jaccard ≥ 0.95; lognorm rel_err ≤ 1e-5; PCA singular values rel_err ≤ 1e-4)
target_perf: 1M cells full pipeline (lognorm → HVG → PCA k=50) in ≤30s on A100; 10M cells in ≤6 min; 100M cells in ≤2 hours; bounded by chunk size, not total cell count
ooc_plan: this IS the OOC driver; chunk size is the only memory tunable
---

## Why this exists (the int32 nnz cap motivation)

Cycle 2 surfaced the critical finding: `factornet::gpu::SparseMatrixGPU<float>` uses `int` for `rows`, `cols`, `nnz`. Hard cap ~2.1B nnz per matrix → ~0.7M–2M cells max in a single in-memory matrix at typical scRNA density. The full-resident in-memory pipeline (cycles 3–6) cannot reach the 1M+ cell benchmark scale. **Streaming is mandatory, not optional.**

This driver composes our existing in-memory kernels (lognorm, HVG, optional PCA via SVD, optional NMF via factornet's `nmf_chunked_gpu`) over `PzDataLoader` chunks. The pipeline runs in a strict two-pass-or-one-pass schedule per kernel:

| Kernel | Passes over chunks | Reason |
|---|---|---|
| lognorm (TotalCount) | 2 | Need median of cell totals across all cells before applying log1p |
| lognorm (user target_count) | 1 | Skip the median pass |
| HVG (SeuratV3) | 2 | Need global per-gene mean + var, then global LOWESS, then per-chunk variance norm |
| HVG (PearsonResiduals) | 2 | Need global per-gene mean + sum, then per-chunk residual variance |
| PCA via factornet `*_svd_gpu` | NOT STREAMED | factornet SVD GPU loads full matrix on device — incompatible with chunks. PCA at billion-scale defers to randomized SVD over a sketched submatrix (cycle ≥ 8). |
| NMF via factornet `nmf_chunked_gpu` | factornet handles | We pass the `PzDataLoader` directly; factornet's chunked NMF iterates internally |
| kNN graph | NOT STREAMED in cycle 7 | Requires the PCA embedding; covered in feature 6 (cycle 8) |
| Leiden / UMAP / DE | NOT STREAMED in cycle 7 | Each has its own chunking story; covered in their respective features |

## Algorithm

The driver exposes one entry point:

```cpp
namespace singlet_gpu::streaming {

struct PipelineConfig {
    std::vector<std::string> input_paths;     // .1pz files to concatenate logically
    int chunk_cols = 100'000;                 // PzDataLoader chunk width

    // Lognorm
    bool run_lognorm = true;
    singlet_gpu::preprocess::LogNormConfig lognorm_cfg = {};

    // HVG
    bool run_hvg = true;
    singlet_gpu::preprocess::HvgConfig hvg_cfg = {};

    // PCA (deferred for chunked path; only runs in-memory after HVG selection if total cells ≤ 1M)
    bool run_pca = false;
    int pca_k = 50;
    singlet_gpu::reduce::svd::SvdConfig pca_cfg = {};

    // NMF (uses factornet's chunked path)
    bool run_nmf = false;
    singlet_gpu::reduce::nmf::NmfConfig nmf_cfg = {};
};

struct PipelineResult {
    int64_t n_cells;
    int64_t n_genes_total;
    int64_t n_nnz_total;

    // Lognorm outputs
    std::vector<float> size_factors;          // host-side, length n_cells
    std::vector<uint8_t> qc_mask;
    float lognorm_target_used = 0.0f;

    // HVG outputs
    std::vector<int> hvg_indices;             // host-side, length top_n
    std::vector<float> hvg_scores;
    std::vector<float> gene_means;
    std::vector<float> gene_vars;

    // PCA outputs (in-memory path only)
    singlet_gpu::reduce::svd::SvdResult pca = {};
    bool pca_ran = false;

    // NMF outputs (chunked path)
    singlet_gpu::reduce::nmf::NmfResult nmf = {};
    bool nmf_ran = false;

    // Diagnostics
    double wall_lognorm_s = 0;
    double wall_hvg_s = 0;
    double wall_pca_s = 0;
    double wall_nmf_s = 0;
    int n_chunks_processed = 0;
};

PipelineResult run_pipeline(const PipelineConfig& cfg, cudaStream_t stream = nullptr);

}  // namespace
```

### Execution

1. **Open all input paths**: each `.1pz` is opened via `singlet_gpu::io::PzDataLoader`. The driver iterates the loaders in order (logical concatenation) — column indices are offset by the running total of `n_cells`.
2. **Lognorm pass 1** (if enabled and not user-target):
   - For each chunk: load via the loader, compute per-chunk col sums (Kahan), accumulate into a host-side `cell_totals[n_cells_global]` vector.
   - Compute global median `T`.
3. **Lognorm pass 2**:
   - For each chunk: compute `s = t / T`, apply `log1p(x / s)` in place, `qc_mask[j] = (t == 0)`.
   - Optionally write the normalized chunk to a temporary cache (memory-mapped file) so HVG and PCA can re-read it without re-decompressing. Cache is opt-in — `cfg.cache_normalized = true`.
4. **HVG pass 1**:
   - For each chunk: compute per-gene `(sum_x, sum_xx, n_nz)` partial accumulators.
   - Welford-merge across chunks into global `(mean[m], var[m])`.
5. **HVG pass 2**:
   - For each chunk: compute the per-gene normalized variance contribution (SeuratV3) or Pearson residual variance contribution (PearsonResiduals).
   - Final pass: select top-N globally via `cub::DeviceRadixSort`.
6. **PCA** (in-memory only, if `n_cells ≤ in_memory_threshold`):
   - Concatenate all chunks back into a single `PzDeviceMatrix` (with `keep_host_pinned=true`).
   - Run `singlet_gpu::reduce::svd::auto_select(m, k, cfg)` → `pca`.
   - For `n_cells > in_memory_threshold`, leave `pca_ran = false` and document.
7. **NMF** (chunked path, if `cfg.run_nmf`):
   - Pass the `PzDataLoader` directly to `singlet_gpu::reduce::nmf::chunked_fit(loader, cfg)`.
   - factornet's `nmf_chunked_gpu` iterates the loader internally; we just hand it over.

### Multi-input concatenation

Multiple input paths are treated as logical concatenation along the cell axis. The driver tracks `n_cells_so_far` and offsets column indices appropriately. Genes (rows) MUST match across inputs — the driver checks `m` and `rownames` from each manifest and errors out on mismatch.

For multi-modal (different genes per input), the user constructs a `FactorGraph` with `SharedNode` and dispatches NMF directly — `run_pipeline` is single-modality only.

## Numerical stability

- Lognorm pass 1 cell totals: fp32 with Kahan compensation per chunk; chunks summed into fp64 global totals to handle the `>50M reads` pitfall. Median uses fp64.
- HVG pass 1: per-chunk fp32 with Kahan; merge into fp64 globals across chunks.
- HVG pass 2: per-chunk fp32; global selection on fp32 because rank-sensitive comparisons don't need fp64.
- PCA: factornet's choice (fp32 default).

## Memory layout

- Input: list of `.1pz` paths.
- Per-chunk transient: one `PzDataLoader` chunk worth of pinned host CSC + Eigen sparse view (for factornet) — bounded by `cfg.chunk_cols × max_nnz_per_col × 4 bytes` ≈ 200 MB for 100k cols × 5k nnz/col.
- Global accumulators:
  - `cell_totals[n_cells]` fp64 — `8n` bytes globally on host.
  - `gene_sum[m]`, `gene_sum_xx[m]` fp64 — `16m` bytes.
  - `qc_mask[n_cells]` uint8 — `n` bytes.
  - HVG outputs.
- Total host overhead: ~`8n + 16m` bytes for accumulators. For 100M cells × 30k genes: 800 MB + 480 KB. Manageable.
- Device peak: bounded by chunk size, NOT total cell count. Independent of n_cells.
- Optional normalized cache: a memory-mapped temporary file the size of the full nnz × 4 bytes. Opt-in.

## Streams

One stream, caller-provided. Each chunk's load + decompress + kernel sequence runs on the same stream; ping-pong prefetch (factornet `io::ping_pong_prefetch`) overlaps the next chunk's decode with the current chunk's compute.

## Out-of-core

This IS the OOC driver. There is no smaller scale.

## Determinism

Deterministic given:
- Fixed input file order
- Fixed `chunk_cols`
- Fixed reduction order across chunks (the driver iterates in deterministic order)

The lognorm + HVG kernels are themselves deterministic (cycles 3 + 4 confirmed). factornet NMF determinism depends on factornet's choice; document.

## Correctness test spec

Test file: `tests/streaming_pipeline_correctness.cpp`.

1. **`Pipeline_LognormOnly_EquivalentToInMemory`** — load GSM4037629 in two ways: (a) `load_pz` + `log_normalize` directly, (b) `run_pipeline({input_paths={path}, chunk_cols=2000, run_hvg=false, run_pca=false, run_nmf=false})`. Compare `size_factors` element-wise: rel_err ≤ 1e-5 (Welford-merge precision).
2. **`Pipeline_HvgOnly_EquivalentToInMemory`** — same shape, but with `run_lognorm=false, run_hvg=true`. Compare top-2000 HVG indices: Jaccard ≥ 0.95.
3. **`Pipeline_LognormHvg_EquivalentToInMemory`** — both passes. Same comparison.
4. **`Pipeline_LognormHvgPca_InMemoryFallback`** — small input. Confirm PCA runs and singular values match in-memory result within rel_err ≤ 1e-4.
5. **`Pipeline_NmfChunked_RunsToCompletion`** — small input. Confirm `nmf` field is populated and loss decreased.
6. **`Pipeline_MultipleInputs_ConcatenatedCorrectly`** — pass 3 copies of GSM4037629 as 3 input paths. Confirm `n_cells = 3 × 11560` and the lognorm result matches running on a manually concatenated single matrix.
7. **`Pipeline_GeneMismatch_Errors`** — pass two `.1pz` files with different gene rows. Confirm `run_pipeline` returns an error or throws.
8. **`Pipeline_ChunkSize_DoesNotAffectResult`** — run the same input with `chunk_cols=1000, 5000, 11560`. Confirm the lognorm + HVG outputs are bitwise identical (up to float reduction order tolerance).

Tolerance:
- Lognorm rel_err ≤ 1e-5
- HVG Jaccard ≥ 0.95
- PCA singular values rel_err ≤ 1e-4
- Multi-input concat: bitwise identical to single-matrix run within Welford merge precision

## Target performance

| Scale | n_cells | n_chunks | Wall (target) | Notes |
|---|---|---|---|---|
| 100k | 120k | 12 | <2s | smoke; equivalent to in-memory |
| 1M | ~1M | 100 | <30s | first benchmark scale unreachable in-memory due to int32 nnz cap |
| 10M | ~10M | 1000 | <6min | medium-scale demo |
| 100M | ~100M | 10000 | <2 hours | billion-cell capability proof |

Memory peak: bounded by `chunk_cols`, not `n_cells`. Constant ~1 GB device + ~1 GB host pinned for `chunk_cols=100k`.

## Implementation notes (for cycle 7 kernel-dev dispatch)

- Header path: `include/singlet-gpu/streaming/streamed_pipeline.h`.
- Total LOC budget: ~600 LOC.
- Reuses cycle 2–6 headers extensively.
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: cycles 2 (loader), 3 (lognorm), 4 (hvg), 5 (svd adapters), 6 (nmf adapters + PzDataLoader).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (composes singlet-gpu kernels over PzDataLoader chunks)` first comment.
- Determinism: deterministic by construction. `cfg.seed` reserved for future stochastic features.
- Errors: throw `std::runtime_error` on gene mismatch, file open failure, factornet errors.

## Risks

1. **Lognorm streaming pass 1 + pass 2 means the input is read twice from disk** — if the `.1pz` file is on slow NFS, this doubles I/O time. Mitigation: optional `cfg.cache_normalized = true` writes the normalized data to a temp memory-mapped file after pass 2, and HVG / PCA / NMF read from the cache instead of re-decompressing.
2. **HVG SeuratV3 LOWESS fit needs ALL gene means + vars at once** — cannot be streamed. We accumulate per-gene moments across all chunks, then run LOWESS once globally on the host. For very large gene panels (>100k features), the LOWESS host-side cost may dominate — defer to a later cycle.
3. **PCA fall-back to in-memory when n_cells ≤ threshold** — the threshold defaults to 2M cells (just below the int32 nnz cap). Above that, the user gets HVG but no PCA from this pipeline. Workaround: subsample to 1M cells for PCA, project the rest. Document.
4. **Multi-input file ordering matters** — the driver concatenates in file-list order, so `cell_totals[j]` is in that order. Document.
5. **Factornet `nmf_chunked_gpu` may not converge well at billion-cell scale** — needs validation. The first 100M-cell run will tell us.
