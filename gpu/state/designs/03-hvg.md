---
feature: hvg
roadmap_id: 3
module: include/singlet-gpu/preprocess/hvg.h
status: design
tolerance: per-gene rank Spearman ρ ≥ 0.99 vs scanpy reference (top-N gene selection); residual variance rel_err ≤ 1e-4
target_perf: 1M cells × 20k genes in ≤80ms on A100 (beat rapids-singlecell on the seurat_v3 path); 100k cells in ≤8ms; 10k cells in ≤1ms
ooc_plan: streamed via PzChunkIterator — accumulate per-gene first/second moments across chunks (Welford-merge); polynomial fit and selection happen once at the end on the full per-gene reduction
---

## Algorithm

`hvg.h` implements two flavors in one fused header — both target raw count matrices (NOT lognormed) per lit-scout's pitfall #3 (sctransform/seurat_v3 require raw counts):

1. **`SeuratV3`** (consensus default): variance-stabilizing local polynomial fit on `log10(mean) → log10(var)`, then standardized residuals scored per gene, top-N selected.
2. **`PearsonResiduals`** (Lause et al. 2021): regularized NB regression with shared overdispersion `θ` across all genes (the Hafemeister-Satija trick), per-gene Pearson residual variance, top-N selected.

Both flavors share the same Pass 1 (gene moments). They differ only in Pass 2/3.

### Shared Pass 1 — Per-gene moments (`compute_gene_moments`)

One block per gene; threads in the block iterate over the cells in that gene's row. Since the input is CSC (genes × cells = rows × cols, but we want **per-gene** = per-row stats), we either:

- **Option A**: Use a CSR view if available (factornet `SparseMatrixGPU` may keep both). Each block walks one row → one gene's nonzeros.
- **Option B**: Two-pass scatter — first pass computes `gene_id_of_nz[]` via `cusparse::cusparseCsr2csc` or our own scatter, then each block walks contiguous gene-grouped nonzeros.
- **Option C** (fallback): reduce-by-key on `(gene_id, value)` pairs using `cub::DeviceSegmentedReduce`.

Pick option A first — factornet's `SparseMatrixGPU` likely caches the transpose. If not, option C using cub is the simplest fallback. The kernel must work on either CSC-only or CSR-cached input; document the cost difference.

Per gene g, in fp32 with Welford online compensation:
- `count_nz[g]` = number of nonzero cells (from indptr).
- `mean[g]` = `sum(x_ig) / n_cells_total` (NOT just nonzero cells — sparse zeros count).
- `var[g]` = `sum((x_ig − mean)²) / (n_cells_total − 1)`, computed two-pass (single-pass Welford on a sparse stream is awkward because most entries are zero; two-pass is cleaner).

For sparse matrices, the trick: `var = (sum_x² − n_total · mean²) / (n_total − 1)`, where `sum_x²` is over nonzero entries only. The all-zero contribution is `0` to `sum_x²`, but enters `mean²`. This avoids iterating over all `n_cells × n_genes` entries.

```
for each gene g (one block):
    sum_x = 0, sum_xx = 0
    for each nonzero (cell j, value v) in row g:    // O(nnz_g)
        sum_x  += v
        sum_xx += v * v
    mean[g] = sum_x / n_cells
    var[g]  = (sum_xx - n_cells * mean[g] * mean[g]) / (n_cells - 1)
```

Numerical stability: fp32 accumulation is fine when nnz_g is small (most genes), but for housekeeping genes with high counts and 1M cells, fp32 sum_xx can overflow. Use **Kahan compensation on `sum_xx`** (lit-scout pitfall #1 + #2). For genes where Kahan-fp32 still risks overflow (`max_val * sqrt(nnz_g) > 2^16`), promote `sum_xx` to fp64 for that gene only — flagged via a per-gene status bit.

Output: `mean[m]`, `var[m]` (both fp32 device arrays).

### `SeuratV3` Pass 2 — Local polynomial fit

Per scanpy `_highly_variable_genes_seurat_v3`:

1. Compute `log10_mean = log10(mean)`, `log10_var = log10(var)` per gene. Filter genes with `mean > 0` and `var > 0`.
2. Fit a **local polynomial** of degree 2 on `(log10_mean, log10_var)` over a sliding window of ~50% of genes per query point, OR a **single global LOWESS** with span 0.3.
   - scanpy uses `numpy.polyfit` style local fit via `scikit-misc.smoothers_lowess.lowess`. We replicate the LOWESS with span 0.3, tricube weights, two robustness iterations.
   - GPU LOWESS is non-trivial — implement on device via per-query-point local weighted least squares. Each thread handles one query gene, gathers its nearest neighbors in `log10_mean`, solves a 3×3 normal equation. ~50ms target on 20k genes.
3. Predicted `log10_var_expected[g] = lowess(log10_mean[g])`.
4. Standardized count value `clip = sqrt(n_cells)`, normalized variance `v_norm[g] = ((x_ig − mean[g])² / 10^log10_var_expected[g])` averaged over cells, clipped to `clip`.
5. Sort genes by `v_norm[g]` descending → top N.

The expensive step is step 4 (full pass over the matrix to compute the clipped per-gene normalized variance). Fuse with Pass 1 by holding `log10_var_expected[g]` and computing `v_norm` in the same kernel pass over the values.

Actually — the cleanest implementation is a **three-pass kernel**:
- Pass 1 (`compute_gene_moments`): mean + var per gene.
- Pass 2 (`fit_lowess` on host or single-block GPU): fit `log10_var_expected[g]` from `log10_mean[g]`. Small (m=20k genes), can be done on a single block in shared memory or even host-side via cuSOLVER for the local LSQ.
- Pass 3 (`compute_v_norm`): one block per gene, second sparse pass to compute clipped normalized variance.
- Pass 4 (`top_n_select`): cub::DeviceRadixSort on `v_norm[]`, then cub partition for top-N indices.

### `PearsonResiduals` Pass 2 — NB regression + residual variance

Per Lause et al. 2021:

1. Compute `mu_ig = (sum_g · sum_j) / total_count` — the multiplicative-null expected count for gene g in cell j (genes × cells = rank-1 product of gene totals × cell totals / grand total).
2. Pearson residual: `r_ig = (x_ig − mu_ig) / sqrt(mu_ig + mu_ig² / θ)` with shared overdispersion `θ` (default `100`, sometimes fit per dataset by binary search to make residual variance = 1 on housekeeping genes).
3. Per-gene residual variance: `var_r[g] = sum((r_ig − mean_r[g])²) / (n_cells − 1)`. With shared θ and rank-1 `mu`, this can be computed without materializing `r` densely.
4. Top-N select on `var_r[]`.

Step 3 requires care because `r_ig` is dense (every cell has a value, even zero counts have residuals `−mu_ij / sqrt(...)`). But the all-zero-cell contribution is computable in closed form per gene from `(mean_g, sum_j)`. Specifically:

```
var_r[g] = (sum over nonzero cells of (r_ig − mean_r[g])²)
         + (n_zero_cells_g) * (r_zero[g] − mean_r[g])²
```

where `r_zero[g]` is the residual for a zero entry against the dense `mu_ig` for that gene-cell pair. This still requires per-gene-per-cell `mu_ig`, but `mu_ig = (gene_sum[g] * cell_sum[j]) / grand_sum`, which factorizes. So the closed form is:

```
var_r[g] = sum_nonzero ((x − μ)² / (μ + μ²/θ)) + closed_form_zeros(gene_sum[g], cell_sums, θ)
```

The closed-form zeros term can be computed in O(n_cells) per gene by summing `cell_sums²` weighted by `(gene_sum[g])²` factors. Per-gene cost: O(nnz_g + n_cells), so total `O(nnz + m·n)`. For 1M cells × 20k genes that's 20B ops — borderline. Optimization: precompute `sum(cell_sums²)`, `sum(cell_sums)` once on host; per gene only needs O(nnz_g) work.

After that optimization the cost drops to `O(nnz + m)`.

### Selection (both flavors)

`top_n_select`: `cub::DeviceRadixSort::SortPairs` on the per-gene score, then take the first `top_n` indices. Output: `DeviceMemory<int> hvg_indices(top_n)` plus `DeviceMemory<float> hvg_scores(top_n)`.

### Filtering by mean / variance bounds

scanpy supports `min_mean`, `max_mean`, `min_disp` filters. Replicate exactly. Default values: `min_mean=0.0125`, `max_mean=3`, `min_disp=0.5` (these are the scanpy defaults for the seurat-v1 flavor; v3 has different defaults — match per-flavor).

## Numerical stability

- **fp32 with Kahan compensation** in pass 1 for `sum_xx`. Per-gene fp64 promotion when `max_val * sqrt(nnz_g) > 2^16` (lit-scout pitfall #1).
- **Welford NOT used** because the sparse two-pass formula is cleaner.
- **`log10` not `log`**: scanpy uses log10 explicitly. Match.
- **Polynomial fit**: cuSOLVER for the 3×3 normal equations is fp32; the matrices are small enough that fp32 LU is stable. Use cuSOLVER's `cusolverDnSgesv` if needed.
- **`mean = 0` and `var = 0` handling**: skip genes where mean == 0 (no expression). For genes where var == 0 but mean > 0, treat as "constant" and exclude from HVG.
- **NB θ**: default 100; future cycle can fit it via binary search to set median residual variance to 1 (Lause et al.).

## Memory layout

- Input: `DeviceCSC` (m × n, fp32). Untouched.
- Workspace:
  - `mean[m]`, `var[m]` (fp32) — `8m` bytes.
  - SeuratV3: `log10_var_expected[m]`, `v_norm[m]` — `8m` bytes.
  - PearsonResiduals: `gene_sum[m]`, `var_r[m]` — `8m` bytes (cell sums computed once, shared with lognorm).
  - Output: `hvg_indices[top_n]`, `hvg_scores[top_n]` — `8 * top_n` bytes.
- Peak overhead: ≤24m bytes for SeuratV3, ≤16m bytes for PearsonResiduals. For m=20k genes: ~480 KB / 320 KB. Negligible.

## Streams

One stream, caller-provided. All passes sequential on that stream. The LOWESS fit (SeuratV3) is small enough to run synchronously on the host without losing GPU concurrency — the next async kernel queues immediately after.

## Out-of-core chunking

`PzChunkIterator` yields fixed-column-width slices. For HVG:

1. **Single pass over chunks** for pass 1: each chunk contributes to per-gene `(sum_x, sum_xx)` accumulators. Welford merge across chunks (the parallel-merge formula by Chan et al. 1979) — exact for fp32 if accumulators promoted to fp64 between chunks.
2. **Single pass for pass 3** (SeuratV3) or **single pass for the residuals** (PearsonResiduals). Same chunk iterator, same per-gene reductions.
3. **Selection** runs once on the full per-gene array.

Memory bound by chunk size. Two-pass over the data, but each pass is a single sparse traversal.

## Determinism

The two-pass formula `var = (sum_xx − n·mean²) / (n−1)` is deterministic for a fixed reduction order. cub `DeviceReduce` is deterministic per-launch on a fixed architecture.

LOWESS local fits are deterministic given a fixed neighbor set. cuSOLVER `gesv` is deterministic.

cub `DeviceRadixSort::SortPairs` is deterministic.

The kernel as a whole is deterministic. The `deterministic` flag in `HvgConfig` is a no-op.

## Correctness test spec

Test file: `tests/preprocess_hvg_correctness.cpp`.

Three flavors, three scales:

1. **tiny** (500 × 200 fixed-seed CSC): both flavors. Compare top-50 gene indices to scanpy. Spearman ρ on `v_norm` / `var_r` per gene ≥ 0.99.
2. **GSM4037629 exon_counts** (11,560 cells): both flavors, top-2000 genes. Same comparison.
3. **100k concat**: both flavors. Same comparison.

Reference: scanpy `sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)` and `sc.experimental.pp.highly_variable_genes(adata, flavor='pearson_residuals', n_top_genes=2000)` in a Python subprocess. Dump the top-N indices + per-gene scores for diff.

Tolerances:
- **Top-N gene set Jaccard ≥ 0.95** (different fp accumulation orders give 1–5% gene shuffle on the boundary; 95% Jaccard is the realistic gate).
- **Per-gene score Spearman ρ ≥ 0.99**.
- **Per-gene rank rel-error ≤ 5%** for genes in the top-2N.

Looser than lognorm because HVG is rank-sensitive; small fp32 differences shuffle gene boundaries even when math is correct.

Edge cases:
- Genes with `var == 0` → excluded from HVG, no error.
- Genes with `nnz == 0` → excluded.
- `top_n > m` → return all `m` genes.
- `top_n == 0` → return empty array.

## Target performance

| Scale | Cells | Genes | nnz | Target wall | SOTA (rapids-singlecell) | Notes |
|---|---|---|---|---|---|---|
| tiny | 200 | 500 | 2k | <0.5ms | n/a | smoke |
| 10k | 11,560 | ~30k | ~30M | <1ms | ~5ms | beat ≥5× |
| 100k | ~120k | ~30k | ~300M | <8ms | ~30ms | beat ≥3× |
| 1M | ~1M | ~30k | ~3B | <80ms | ~300ms | beat ≥3× |

Memory: ≤ 24m bytes overhead.

ScaleSC reports <2 min for 1.3M cells + 18k→4k gene selection on **8 GPUs**. We target ≤80ms on a single A100 by exploiting our smaller gene panel and the closed-form-zeros optimization for PearsonResiduals.

## Implementation notes

- Header path: `include/singlet-gpu/preprocess/hvg.h`.
- API:
  ```cpp
  namespace singlet_gpu::preprocess {
      enum class HvgFlavor { SeuratV3, PearsonResiduals };
      struct HvgConfig {
          HvgFlavor flavor = HvgFlavor::SeuratV3;
          int top_n = 2000;
          float min_mean = 0.0125f;
          float max_mean = 3.0f;
          float pearson_theta = 100.0f;
          uint64_t seed = 0;  // unused
      };
      struct HvgResult {
          singlet_gpu::core::DeviceMemory<int>   indices;   // top_n gene indices, sorted by score desc
          singlet_gpu::core::DeviceMemory<float> scores;    // top_n scores
          singlet_gpu::core::DeviceMemory<float> mean;      // per-gene mean (m)
          singlet_gpu::core::DeviceMemory<float> var;       // per-gene var (m)
      };
      HvgResult select_hvg(
          const singlet_gpu::core::DeviceCSC& mat,    // raw counts, NOT lognormed
          const HvgConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- The kernel does NOT mutate `mat`. It is read-only on the input.
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: `core/types.h`, `core/handles.h`, `core/memory.h`. Optional dependency on cuSOLVER for the polynomial fit.
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (no factornet equivalent for HVG selection)` first comment.
