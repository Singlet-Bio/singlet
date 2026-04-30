# Cycle 88 — feature #3 sub-variant: scry deviance feature selection (Townes 2019)

**Date**: 2026-04-18  
**Phase**: C (Design decision)  
**Target feature**: #3 (HVG), sub-variant 3.2  
**Status**: Ready for Phase D (kernel-dev + validator dispatch)

---

## Problem Statement

Bioconductor `scry::devianceFeatureSelection` (Townes 2019) is the only GPU-absent, GPU-native port of a widely-cited HVG selection method in the Seurat/scran/Bioconductor ecosystem. Scanpy does not offer it; rapids-singlecell does not port it; factornet does not ship HVG kernels at all. The method provides an alternative deviance-based scoring as a binomial-approximation-to-multinomial null model, complementing the Pearson-residuals path (frontier since Cycle 3).

Recent benchmarks (Lause 2021, Bredikhin 2025) show Pearson residuals outperform deviance for downstream integration and querying tasks — yet deviance remains popular in the R ecosystem and is the explicit default in Bioconductor workflows. **Shipping both variants gives Bioconductor users a familiar, correct option; Scanpy/rapids users will rely on the demonstrated-superior Pearson path.**

---

## Phase B Findings

### Algorithm summary (from lit-scout return)
- **Deviance formula** (binomial approximation to multinomial null):
  ```
  D_g = Σ_c d(y_gc, n_c, π_g)
  d(y,n,π) = 2·[y·log(y/(nπ)) + (n-y)·log((n-y)/(n(1-π)))]
  0·log(0) = 0 (by convention)
  π_g = gene_total / grand_total = (Σ_c y_gc) / (Σ_c n_c)
  ```
  Cell library size `n_c = Σ_g y_gc`; gene total `s_g = Σ_c y_gc`.
- **Poisson alternative** (faster, better for low counts):
  ```
  D_g^{Poisson} = Σ_c d_poisson(y_gc, λ)
  d_poisson(y,λ) = 2·[y·log(y/λ) - (y - λ)]
  λ = π_g · n_c (same π_g)
  ```
- **Correctness benchmarks** (Lause 2021 HVG selection on downstream tasks):
  - Deviance vs Pearson for scRNA-seq: Pearson wins on integration Silhouette, query assignment F1, and cell-type stability.
  - Deviance still widely used; R scry is the only mature implementation.
- **Reference implementation**: R `scry::devianceFeatureSelection` (Bioconductor); no GPU port exists.
- **SOTA comparison**: Scanpy has Seurat VST + Pearson residuals (both on frontier). scran has Poisson dispersion. cuml/rapids-singlecell do not ship HVG kernels at all.

### Key insights
1. **Proven inferior to Pearson on downstream tasks** (Lause 2021, Bredikhin 2025) — yet we ship it for ecosystem compatibility.
2. **Two-pass map-reduce structure** — excellent GPU fit:
   - Pass 1: compute π_g (gene total / grand total) — tree sum.
   - Pass 2: per-cell deviance contribution — only nnz entries computed.
3. **Sparse-friendly decomposition** (critical for GPU):
   - `D_g = D_g^{nnz} + D_g^{zero}` (closed form for zero contributions).
   - `D_g^{zero} = -2·log(1-π_g)·(T - L_g)` where T=grand total, L_g=gene total, O(n_cells) for the constant, O(1) to apply.
   - `D_g^{nnz} = Σ_{c: y_gc>0} d(y_gc, n_c, π_g)` (only nnz terms computed).
   - Typical scRNA: 50-100× reduction vs naive O(n_genes × n_cells).

---

## Design Specification

### Scope (Phase C decision)
**Ship: `devianceFeatureSelection` (per-gene deviance vector + top-N selection).**
This is the most-used entry point and the one benchmarked in the literature.

**Defer: `nullResiduals`** — we already have Pearson residuals on the frontier (Cycle 3) which Lause 2021 benchmarks show outperform deviance residuals for downstream tasks. Add null-residuals scaling only if a future cycle opts into generalized-residuals workflows.

### Configuration

```cpp
namespace singlet_gpu::preprocess::hvg {

struct DevianceHvgConfig {
  int top_n = 2000;             // number of HVGs to select (ascending deviance)
  float min_gene_total = 1.0f;  // minimum total UMI per gene to consider
  bool use_poisson = false;     // binomial (default) vs Poisson null
  uint64_t seed = 0;            // reproducibility (unused but required by style)
};

struct DevianceHvgResult {
  DeviceMemory<float> deviance;           // per-gene deviance scores [n_genes]
  std::vector<int> top_gene_idx;          // indices of top-N HVGs (host copy)
  std::vector<bool> is_variable;          // boolean mask [n_genes]
  int n_genes_considered;                 // genes passing min_gene_total filter
};

// Primary entry point
DevianceHvgResult deviance_feature_selection(
  const DeviceCSC& counts,
  const DevianceHvgConfig& cfg,
  cudaStream_t stream = nullptr
);

// Variant accepting precomputed library sizes (for workflow fusion)
DevianceHvgResult deviance_feature_selection_with_lib_sizes(
  const DeviceCSC& counts,
  const DeviceMemory<float>& lib_sizes,  // n_cells precomputed column sums
  const DevianceHvgConfig& cfg,
  cudaStream_t stream = nullptr
);

} // namespace singlet_gpu::preprocess::hvg
```

### Algorithm structure

**Input**: sparse CSC matrix `counts [n_genes × n_cells]`, config, stream.

**Pass 1 — Statistics accumulation**:
1. Compute grand total `T = sum(counts)` → 1 scalar (tree reduce, 1 KB D2H).
2. Compute column sums (lib sizes) `L_c = sum_g counts[g,c]` → n_cells vector.
3. Compute row sums (gene totals) `s_g = sum_c counts[g,c]` → n_genes vector.
4. Compute `π_g = s_g / T` → n_genes vector.
5. Apply filter: `mask_g = (s_g >= min_gene_total)` → n_genes boolean.

**Pass 2 — Deviance computation (sparse-friendly)**:

For each gene `g`:
- Closed-form zero contribution: `D_g^{zero} = -2·log(1-π_g)·(T - s_g)` (branch on π_g near 1.0 to avoid log(0)).
- Per-cell nonzero loop: for `c` where `y_gc > 0`, accumulate:
  - Binomial: `d = 2·[y_gc·log(y_gc/(L_c·π_g)) + (L_c - y_gc)·log((L_c - y_gc)/(L_c·(1-π_g)))]`
  - Poisson: `d = 2·[y_gc·log(y_gc/(L_c·π_g)) - (y_gc - L_c·π_g)]`
  - Handle 0·log(0) = 0 explicitly.
- Sum: `D_g = D_g^{zero} + sum_nnz(d)`.

**Pass 3 — Top-N selection** (device-side):
- `argsort(D_g)` ascending → top N indices.
- Return indices + sorted `D_g` vector.

### Memory footprint

- **Per-gene state**: `deviance[n_genes]` (float32, n_genes KB for 20k genes).
- **Per-cell state**: `lib_sizes[n_cells]` (float32), precomputed or reused from lognorm.
- **Intermediate**: scalar tree-reduce scratch (~64 bytes), sorted index/value (~400 KB for 20k genes).
- **Peak**: max(nnz region in one sparse column for the per-cell loop) + output.
- Typical: **~50 MB for 20k genes × 100k cells.**

### Streaming (billion-cell support, per Rule 14)

**Contract**: `devianceFeatureSelection` is **associative across cell shards** — every reduction in Pass 1-2 can be split and merged.

- **Shard mode**: Load shard-i cells. Accumulate partial sums (L_c for shard-i only, s_g^{(i)} partial gene totals).
- **Reduction tree**: Host-side merge of partial s_g and T across shards; broadcast final π_g to all shards.
- **Pass 2 (per-shard)**: Recompute deviance on shard-i cells using global π_g.
- **Number of passes**: 2 over data (+ 1 small D2H/H2D).
- **Result**: Concatenate per-shard deviance vectors, take global top-N.

**Key**: Gene totals `s_g` are sufficient statistics — no per-cell tracking required between shards.

---

## Numerical Stability & Correctness

### Stability measures
1. **Kahan summation** for gene total accumulation `s_g` (log-likelihood is sensitive to rounding).
2. **Explicit 0·log(0) = 0 handling** — branch `if (y > 0)` before any log.
3. **Guard near-unit π_g**: if `π_g > 0.9999`, use `log(1e-4)` instead of `log(small)` to avoid -∞.
4. **Two-pass variance check**: No fp32 precision loss detectable on gene totals (compare Pass 1 sum vs Pass 2 partial).

### Correctness tolerances (vs R scry on real data)

- **Per-gene deviance**: Spearman rank correlation ≥ 0.999 (scry uses same algorithm; GPU rounding will differ, but ranking is stable).
- **Per-cell max deviation**: max |D_g^{GPU} - D_g^{R}| / (mean |D_g^{R}|) < 5% (floating-point rounding + tree-reduce variance).
- **Top-N Jaccard index** (intersection / union of top-N genes): ≥ 0.95 vs R scry on N={100,500,2000}.
- **Binomial vs Poisson agreement** (low-count regime, π_g < 0.05): Spearman ≥ 0.95 (approximation error is acceptable).

### Reference implementations

- R `scry::devianceFeatureSelection` (Bioconductor) — ground truth.
- **Test will fetch**: 2-3 real datasets from GEO (via `code-reader` fetch) to compare GPU vs R.

---

## Test Specification (for analysis-validator)

### Test 1: Deterministic, tiny synthetic data
- **Input**: 1000 cells, 200 genes, planted spike in genes 0-9 (high UMI) + Poisson background.
- **Expected**: Top 10 should include genes 0-9 with correct deviance ordering.
- **Correctness gate**: Top-10 Jaccard = 1.0, all D_g exact to float precision.

### Test 2: Seurat tutorial data (processed scRNA)
- **Input**: GSM4037629 (11.5k cells, ~18k genes after QC).
- **Expected**: Agree with R scry on deviance ranking; top-2000 Jaccard ≥ 0.95.
- **Correctness gate**: Spearman(D_g^{GPU}, D_g^{R}) ≥ 0.999.

### Test 3: Real large-scale data (if R scry available on g008)
- **Input**: 100k cells, 30k genes (PBMC or similar).
- **Expected**: Spearman ≥ 0.999, top-2000 Jaccard ≥ 0.95.
- **Correctness gate**: Wall time < 500 ms (H100).

### Test 4: Poisson vs Binomial agreement
- **Input**: Real data with wide π_g range.
- **Expected**: Low π_g genes (sparse) should agree Spearman ≥ 0.95 between Poisson and binomial.
- **Rationale**: Poisson is an approximation; validates that the approximation holds.

### Test 5: Streaming correctness (billion-cell)
- **Input**: Virtual 1M cells (load in 4× 250k shards).
- **Expected**: Global top-2000 genes match single-pass result exactly.
- **Correctness gate**: Jaccard = 1.0 (associativity check).

### Test 6: Null model check (synthetic)
- **Input**: Uniform random counts (no signal).
- **Expected**: Deviance should follow the null distribution (χ² with 1 dof); flat ranking.
- **Rationale**: Confirms null model is correct.

---

## SOTA Benchmark Targets

### Comparison baselines
- **R scry** (CPU): reference truth. Expect 10–60 seconds on 11.5k × 18k.
- **Scanpy VST** (CPU): shipped on frontier. Expect ~5 seconds on 11.5k × 18k.
- **Pearson residuals (GPU)** (Cycle 3 frontier): expect <100 ms on H100. Deviance will be similar.

### Target wall times at three scales

| Scale | R scry | Our target (H100) | Expected GPU speedup |
|---|---|---|---|
| 10k cells, 15k genes | ~1 sec | <10 ms | 100–150× |
| 100k cells, 20k genes | ~10 sec | <100 ms | 100–150× |
| 1M cells, 25k genes | 100+ sec | <1 s (streaming) | 100–200× |

(scry is single-threaded by default; parallelized scry ~2–3× faster, still an order of magnitude slower than GPU.)

### Win criteria for frontier promotion
- **Correctness**: All 6 tests pass (Test 3 SKIP gracefully if R scry unavailable).
- **Wall time**: < 100 ms at 11.5k × 18k (H100).
- **Memory**: < 200 MB peak.

---

## API Integration (Rule 33: Compose existing module)

**Location**: Extend `include/singlet-gpu/preprocess/hvg.h` (Cycle 3, currently 701 LOC + test infra).

- Add `struct DevianceHvgConfig` + `struct DevianceHvgResult`.
- Add `deviance_feature_selection(counts, cfg, stream)` + `_with_lib_sizes` variant.
- Extend existing `test/preprocess_hvg_correctness.cpp` with 6 new test cases.
- Update `state/designs/3-hvg.md` with sub-variant 3.2 section.

**Not a new file.** Compose with Cycle 3's Pearson path in the same header via overload resolution or `variant<>` if the API needs to expose both HVG methods at once. Current plan: keep as separate functions; users call explicitly.

---

## Novel Algorithmic Contribution

**Sparse-friendly decomposition** (above). Standard deviance impl iterates over all (gene, cell) pairs — O(n_genes × n_cells), per-pass. Ours: `D_g^{zero}` (closed form) + `D_g^{nnz}` (only nnz terms). On scRNA with 5% density: **50–100× reduction in kernel invocations**. Not novel in HPC (trace-form identities are standard), but **novel for GPU single-cell HVG** (rapids-singlecell and cuml do not ship this; R scry does not use it).

---

## Cycle 88 Decision

- **Design complete**: Yes.
- **Ready for Phase D**: Yes. Dispatch `gpu-kernel-dev` (Sonnet) + `analysis-validator` (Sonnet) in parallel.
- **Assigned streams**: H100 (g001), full access.

---
