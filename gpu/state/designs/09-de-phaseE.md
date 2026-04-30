---
feature: 9
module: de/wilcoxon.h, de/ttest.h, de/glm_nb.h, de/donor_pseudobulk.h
cycle: 66 or later
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy
extends: 09-de.md
status: draft
parent_features: 9 (three-backend DE), 15 (donor-aware DE — unique to us)
---

# Feature 9 + 15 — DE (Wilcoxon + t-test + pseudobulk NB-GLM + donor-aware) Phase E

Feature 9 is the main DE entry point. Feature 15 (donor-aware DE) is a unique-to-us extension that exploits singlify's `donor_assignments.tsv` artifact — no other single-cell library can read it. This doc merges both.

## 1. Success metrics

On GSM4037629 with synthetic cell-type labels (a 2-group comparison across ~10k cells):

- **Wilcoxon wall p50**: ≤ `scanpy rank_genes_groups(method='wilcoxon')` × 0.02 (50× speedup headline). The scanpy path is the de-facto CPU standard and is embarrassingly parallel across genes but currently unoptimized.
- **Wilcoxon GPU wall**: ≤ `rapids-singlecell.tl.rank_genes_groups` × 0.5.
- **t-test wall**: ≤ scanpy × 0.02 (trivially faster on GPU).
- **Pseudobulk NB-GLM wall** (novel: closed-form Wald): ≤ `DESeq2` wall × 0.02 (50× faster than the R reference).
- **Correctness**:
  - Wilcoxon: gene set jaccard ≥ 0.99 for top-100 DE genes vs scanpy; `-log10(p-value)` Pearson ≥ 0.995.
  - t-test: bit-identical to scipy within fp32 ULP.
  - NB-GLM: β coefficient correlation ≥ 0.99 vs DESeq2 closed-form path; `-log10(p-value)` Pearson ≥ 0.99.
  - Donor-aware: ARI of significant-gene set ≥ 0.95 vs a reference mixed-effects R implementation (glmmTMB or pseudobulk+DESeq2).

Frontier promotion gate: dominance on wall for all four backends, correctness on all four metrics, Rule 31 autonomy delta ≤ 10%.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **scanpy `rank_genes_groups` (wilcoxon)** | Python / CPU | Primary CPU reference |
| **rapids-singlecell `tl.rank_genes_groups`** | Python / cupy | Primary GPU competitor |
| **Seurat `FindMarkers` (wilcox)** | R / C++ | R gold standard |
| **DESeq2 (pseudobulk)** | R / C++ | Gold standard for pseudobulk NB-GLM |
| **edgeR (pseudobulk)** | R / C++ | Secondary NB-GLM baseline |
| **MAST** | R | Two-part hurdle model — alternative DE framework; correctness sanity check |
| **glmmTMB (mixed effects)** | R | Donor-aware gold standard |

## 3. Bench configurations

Scales:
- **tiny**: 500 cells, 2 groups, 200 genes (smoke)
- **small**: GSM4037629 2 cell-type groups, top-2000 HVG
- **medium**: 5-sample concat, 5 cell-type groups, top-2000 HVG
- **large (pseudobulk only)**: 1M synthetic with 10 donors × 5 cell types (stress test for pseudobulk matrix construction)

Configurations × every scale:

1. `ours_wilcoxon_fused` — Rule 30 fused warp-sort-and-rank (§4a)
2. `ours_ttest` — trivially fast reference
3. `ours_glm_nb_wald` — Rule 30 closed-form Wald for pseudobulk (§4b)
4. `ours_glm_nb_irls` — iterative IRLS fallback for ill-conditioned dispersion
5. **`ours_donor_aware`** — Rule 30 unique donor-mixed-effect DE via singlify donor assignments (§4c)
6. `ours_auto` — Rule 31 backend autoselect
7. `scanpy_wilcoxon`
8. `rapids_singlecell_wilcoxon`
9. `seurat_findmarkers`
10. `deseq2_pseudobulk` — R subprocess
11. `edger_pseudobulk` — R subprocess
12. `glmmtmb_donor` — R subprocess (small scale only — it's extremely slow)

Metrics: wall p50, peak dev mem, top-100 jaccard vs reference, `-log10(p)` Pearson, FDR correlation.

## 4. Novel pursuit (Rule 30)

### 4a. Fused Wilcoxon via `cub::WarpSort` + rank, eliminating per-gene sort

**The scanpy / rapids-singlecell bottleneck**: Wilcoxon computes per-gene `U` statistics which requires ranking the combined-group values per gene. Standard approach: sort n_genes separate arrays of n_cells each, accumulate ranks, compute U. Total work: `O(n_genes × n_cells × log n_cells)`.

**Our fused approach**:
- Blocks of 32 threads (one warp) process one gene each.
- For genes where `n_cells ≤ 32 × warp_size = 1024`, use `cub::WarpMergeSort` entirely in registers — no global memory sort. ~100× faster than global sort for small cell counts.
- For genes where `n_cells ≤ 32768`, use `cub::BlockRadixSort` in shared memory — ~10× faster than global sort.
- For larger (rare in scRNA), fall back to `cub::DeviceRadixSort` with block-per-gene parallelism.
- Ranks accumulated via warp-scan of equal-value runs for tie-adjusted Wilcoxon.
- U, z-statistic, and p-value all computed from closed form in the same kernel — zero inter-kernel memory traffic.

**Expected wall**: one kernel launch for ALL genes vs scanpy's Python-per-gene loop. At 2000 genes × 10000 cells, scanpy takes ~30 sec and our fused kernel targets ≤1 sec. **30–50× speedup** achievable even before the constant-factor tuning.

**Gate**: top-100 jaccard ≥ 0.99 vs scanpy, `-log10(p)` Pearson ≥ 0.995, wall ≤ scanpy × 0.02.

**Risk**: `cub::WarpMergeSort` has register pressure caps — may need a fallback block-sort path at `n_cells ≥ 1024`.

### 4b. Pseudobulk NB-GLM via closed-form Wald (no IRLS)

**The DESeq2 bottleneck**: DESeq2 fits a negative binomial GLM per gene via iterative reweighted least squares (IRLS), typically 10–20 iterations. For pseudobulk with 10s–100s of pseudosamples per gene, each iteration fits an R-level `glm.fit` — painfully slow.

**Closed-form Wald alternative**: when the dispersion `θ` is well-conditioned (which is the common case — most genes have reasonable dispersion estimated from the shrinkage prior), the NB MLE for the mean `μ` is available in closed form via:

```
β̂ = (X^T W X)^(-1) X^T W y        where W = diag(μ / (1 + μ/θ))
```

This is a single iteration of IRLS starting from a good initial guess. For pseudobulk with few covariates (intercept + group + maybe batch), the `(X^T W X)` matrix is 2–5 × 2–5 — tiny, invertible on device via cuSOLVER's batched Cholesky.

**Algorithm**:
```
Input: pseudobulk counts matrix Y (n_genes × n_samples), design X (n_samples × n_cov), dispersion θ (n_genes)
Output: β̂ (n_genes × n_cov), SE (n_genes × n_cov), Wald p-values (n_genes)

1. Initial guess: β̂_0 = OLS(log(Y+1), X) per gene — batched GEMM.
2. W_g = diag(μ_g / (1 + μ_g/θ_g))   // per gene, per sample
3. β̂_g = (X^T W_g X)^(-1) X^T W_g z_g  where z_g is the working response
4. SE_g = diag((X^T W_g X)^(-1))^(1/2)
5. Wald_g = β̂_g / SE_g; p = 2 × (1 − Φ(|Wald_g|))
6. Dispersion check: if |β̂_g − β̂_{g,OLS}| / |β̂_{g,OLS}| > 0.5, fall back to `ours_glm_nb_irls` for that gene (rare).
```

All batched across genes via cuBLAS strided GEMM + cuSOLVER `gesvBatched`. Total wall dominated by the batched Cholesky on ~5×5 matrices — trivially fast on H100 (~1 ms for 2000 genes).

**Expected wall**: ~10 ms for 2000-gene 10-sample pseudobulk vs ~5 min for DESeq2 (R loop). **30000× headline** on small pseudobulks, falling to ~1000× on larger ones where DESeq2 parallelizes.

**Gate**: β correlation ≥ 0.99 vs DESeq2, `-log10(p)` Pearson ≥ 0.99. Fallback rate < 5% on real data.

### 4c. Donor-aware DE via `donor_assignments.tsv` (UNIQUE TO US, feature 15)

**The problem no other library solves cleanly**: single-cell DE across individuals suffers from pseudo-replication — cells from the same donor are not independent. The statistically correct approach is a **mixed-effects model** with donor as a random effect. Standard tools:

- R `glmmTMB`: correct but ~100× slower than DESeq2.
- `dream` (variancePartition): R, CPU, complex setup.
- Pseudobulk + DESeq2: fast but requires the user to manually pseudobulk by donor × cell type.

**Our unique contribution**: singlify's pipeline produces `donor_assignments.tsv` automatically for every sample run with `--snps` (probabilistic donor labels from SNP-pileup demultiplexing). This file is sitting unused on every singlify output.

The donor-aware DE path:
1. Auto-detect `donor_assignments.tsv` in the sample directory alongside the `.1pz`.
2. Convert soft donor probabilities into hard donor labels (threshold 0.9).
3. **Pseudobulk by donor × cell-type × condition** on device in one kernel.
4. Run the closed-form Wald NB-GLM (§4b) on the pseudobulk matrix.
5. Because pseudobulking already accounts for the donor random effect (each donor contributes one sample), the fixed-effects Wald test is the correct estimator. No mixed-effects machinery needed.

This is a published statistically-valid shortcut (Squair et al. 2021, Crowell et al. 2020 "muscat") but requires automatic donor labels — which singlify uniquely provides.

**Expected wall**: ~50 ms for a donor×celltype×condition pseudobulk on 20k cells with 5 donors vs glmmTMB's ~10 min. **10000× headline** on the gold-standard mixed-effects comparison.

**Gate**: significant-gene set jaccard ≥ 0.95 vs glmmTMB on a multi-donor reference dataset. FDR rank correlation ≥ 0.95.

**Dependency**: this path ONLY works when singlify's `--snps` was enabled at alignment time. Detect absence and fall back to unpooled DE with a loud warning.

## 5. Autonomy pass (Rule 31)

No-args `de::compare(counts, group_labels)` returns DE results with:

| Config | Auto |
|---|---|
| `method` | Auto: `wilcoxon` for n_cells ≤ 5000 (exact rank fits in smem); `t_test` for simple gaussian case (auto-detected from library-size distribution); `pseudobulk_nb` for n_donors ≥ 2. |
| `correction` | Auto: `benjamini_hochberg`. |
| `min_cells_per_group` | Auto: 10. |
| `logfc_threshold` | Auto: 0 (return all, user filters downstream). |
| `dispersion_shrinkage` | Auto: `trended` (shared-across-genes, gene-wise). |
| `donor_aware` | Auto: true if `donor_assignments.tsv` is detected alongside the input `.1pz` path. |
| `max_cells_per_group` | Auto: unlimited unless ooc. |
| `tie_correction` | Auto: true (affects Wilcoxon at low n). |

## 6. OOC streaming contract (Rule 14)

DE is embarrassingly gene-parallel but cell-streaming requires care:
- Wilcoxon: per-gene ranks need the global cell distribution. Two-pass: (1) per-gene quantile histogram across chunks, (2) per-gene rank assignment. Approximate via TDigest per gene; exact via global sort per chunk merge.
- NB-GLM pseudobulk: streaming-trivial — accumulate per-group sum + sum_of_squares per gene across chunks. Final GLM fit on the accumulated pseudobulk matrix.
- Donor-aware: same as pseudobulk, add the donor dimension to the accumulator.

Document; implement with feature 16.

## 7. Determinism contract

- Wilcoxon with stable ties is deterministic up to fp32 ULP.
- NB-GLM is deterministic (closed-form path is deterministic by construction; IRLS path is deterministic with fixed initial guess and tolerance).
- Pseudobulk accumulation is deterministic if segmented-scan reduction is used (opt-in). Default atomic accumulation is not.

## 8. Phase E dispatch spec

Dispatch two workers in parallel:

**Worker A (gpu-kernel-dev)**: implement 4a (fused Wilcoxon), 4b (closed-form Wald NB-GLM), 4c (donor-aware pseudobulk) as prototype kernels.

**Worker B (gpu-bench)**: run the 12-config bench table at all scales. Verify correlations against scanpy / DESeq2 / glmmTMB references. Write to benchmark-registry.md, pareto-frontier.md, novel-attempts.md.

## 9. Open questions

- **`cub::WarpMergeSort` register pressure on sm_70**: V100S has 64 registers per thread. For n_cells=1024 per warp that's 32 per thread — likely too much. Fall back to block-sort at n_cells ≥ 512 on sm_70.
- **DESeq2 shrinkage prior**: DESeq2 uses a shared dispersion prior across genes that our closed-form path should replicate. Port their trended-dispersion estimation as a one-shot device kernel.
- **glmmTMB reference for the donor-aware gate**: glmmTMB on 10k-cell datasets takes hours. Use a subsampled 2000-cell reference dataset for the gate.
- **Donor confidence thresholding**: `donor_assignments.tsv` has a `prob_max` column. Threshold at 0.9 by default; expose as a config field.

## 10. Links

- Original: `state/designs/09-de.md`
- scanpy rank_genes_groups: https://github.com/scverse/scanpy
- DESeq2: Love, Huber, Anders 2014
- Squair et al. 2021 "Confronting false discoveries in single-cell differential expression"
- Crowell et al. 2020 "muscat: detecting state changes in replicated single-cell RNA-seq"
- Mandate v2 §A (EDA parity): DE row
- singlify `donor_assignments.tsv` format: `singlify/include/singlet-pileup/pileup_writer.h`
