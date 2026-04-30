# API Reference

One page per public function. Each page is a stable contract — added when a feature transitions to `documented` (see `../../state/release-policy.md`), updated only when the public signature changes.

## Page template

Every API page must contain:

1. **One-line summary**.
2. **C++ signature** — from `include/singlet-gpu/singlet_gpu.hpp`.
3. **Python signature** — from the wrapper.
4. **R signature** — from the wrapper.
5. **Config struct** — every field, default value, what auto-tune does when the field is unset.
6. **Inputs** — types, layout (CSC/dense), assumptions on the matrix.
7. **Outputs** — types, layout, lifetime.
8. **Complexity** — time + memory + streaming behavior. Includes residency budget for streaming.
9. **Determinism** — default + how to opt in via `Config::deterministic = true`.
10. **Correctness contract** — reference tool, tolerance, sample used in the equivalence test.
11. **Citation** — the original method paper + any algorithmic novelties from singlet-gpu.
12. **Example** — 5–15 lines.
13. **Links** — design doc in `state/designs/`, equivalence notebook in `docs/notebooks/`.

## Index

(populated as features transition to `documented`)

### io
- [`io_load_pz.md`](io_load_pz.md) — `singlet_gpu::io::load_pz`. Zero-copy `.1pz` → device CSC. **6.4× anndata-gpu, 9.4× memory.**

### preprocess
- [`preprocess_log_normalize.md`](preprocess_log_normalize.md) — `log_normalize` (total-count + log1p, **382× scanpy**) and `compute_deconv_size_factors` (scran-style pool-and-deconvolve, first GPU port).
- [`preprocess_select_hvg.md`](preprocess_select_hvg.md) — `select_hvg` (Seurat v3 VST **107×**, Pearson residuals **12,597×**) and `deviance_feature_selection` (scry-style binomial / Poisson, Phase E pending).
- [`preprocess_scale.md`](preprocess_scale.md) — `scale` (sparse → dense z-score; clip to ±10) + `regress_out` (cuSOLVER batched QR, p ≤ 32). Frontier; bench backfill pending.

### reduce
- [`reduce_svd.md`](reduce_svd.md) — `auto_select` / `deflation` / `randomized`. **27× scanpy at k=50.** Cycle-61 winner consolidation: deflation primary, randomized fallback; lanczos / IRLBA / krylov removed.
- [`reduce_nmf.md`](reduce_nmf.md) — factornet GPU NMF + Cycle-86 `FitConfig` adapter (k_cd_cutoff=32 forces MU at high rank). **1.82–8.66× sklearn across k ∈ {10, 20, 50, 100}.**

### qc
- [`qc_metrics.md`](qc_metrics.md) — `calculate_qc_metrics` + `filter_cells` + `filter_genes` + `doublet_score`. **429× scanpy at small; 74M cells/sec at medium.** Per-cell + per-gene stats in one pass; Scrublet-equivalent doublet detection on the PCA embedding.

### graph
- [`graph_knn.md`](graph_knn.md) — `compute_knn` (Exact via `cub::DeviceSegmentedRadixSort`, **2.1× sklearn**; CAGRA blocked on cuVS) + `compute_snn` (Jaccard pruning over the kNN result).

### embed
- _none yet_

### de
- [`de_wilcoxon_ttest.md`](de_wilcoxon_ttest.md) — `wilcoxon_de` (**6.5×–388.8× scanpy**) + `ttest_de` (**8.4×–10.4×**). Both at full frontier with all metrics ≥ 0.9999 against scanpy on planted-signal 20k×310k real data. 7-cycle correctness arc documented.

### gsea
- _none yet_

### anno
- _none yet_

### integrate
- _none yet_

### streaming
- _streaming.md_ — pending docs cycle

## Backfill order

Per `state/dag.md` CYCLE-91 onward, pages are written in the order features were promoted to frontier:

1. io/load_pz
2. preprocess/log_normalize
3. preprocess/deconvolve_size_factors
4. preprocess/select_hvg (Seurat v3 + Pearson residuals + scry deviance)
5. reduce/svd (auto_select + deflation winner)
6. reduce/nmf
7. qc/compute_qc_metrics + filter_cells + filter_genes + detect_doublets
8. preprocess/scale + regress_out
9. graph/knn (exact)
10. de/wilcoxon
11. de/ttest

Each is a Phase H dispatch — `gpu-doc-scribe` writes the page in the same cycle the feature is promoted from `frontier` to `documented`.
