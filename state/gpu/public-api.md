# singlet-gpu — Public API Surface

The contract. Every symbol listed here is exported from `include/singlet-gpu/singlet_gpu.hpp` and is API-frozen across MINOR versions per `state/release-policy.md`.

> **Status**: pre-1.0. The umbrella header [`singlet_gpu.hpp`](../include/singlet-gpu/singlet_gpu.hpp) exists as of CYCLE-92 (2026-04-28) and exports the foundational types + `io::load_pz`. The remaining frontier features get added one Phase H cycle at a time; until 1.0, anything not listed here can churn in any patch.

## Conventions

- Namespace: `singlet::gpu::`.
- Every entry: `singlet::gpu::<module>::<function>` with full signature.
- Every entry has a Python binding (`singlet_gpu.<module>.<function>`) and an R binding (`singletGpu::<module>_<function>`).
- Config structs: `<Function>Config` next to the function. Sane defaults trigger on-device auto-tune (Rule 31 in orchestrator).
- Result types: `<Function>Result` aggregating outputs.

## Versioning

- `singlet::gpu::version_major()` → 0 (constexpr int)
- `singlet::gpu::version_minor()` → 1 (constexpr int)
- `singlet::gpu::version_patch()` → 0 (constexpr int)
- `singlet::gpu::commit_sha()` → `const char*` (literal `"pre-1.0"` until git is initialized + CMake passes `-DSINGLET_GPU_COMMIT_SHA=...`)

Defined in [`include/singlet-gpu/version.h`](../include/singlet-gpu/version.h). All four are `constexpr` / link-time constants; no runtime cost.

## Module: `core` (types)

- `singlet::gpu::core::DeviceCSC` — `using DeviceCSC = factornet::gpu::SparseMatrixGPU<float>;` (genes × cells, fp32).
- `singlet::gpu::core::DeviceDense` — `using DeviceDense = factornet::gpu::DenseMatrixGPU<float>;`.
- `singlet::gpu::core::DeviceMemory<T>` — `using DeviceMemory = factornet::gpu::DeviceMemory<T>;` RAII device buffer.
- `singlet::gpu::core::GPUContext` — `using GPUContext = factornet::gpu::GPUContext;` owns cuBLAS / cuSPARSE / cuSOLVER handles.
- `singlet::gpu::core::PinnedBuffer` — RAII pinned host buffer (move-only).
- `singlet::gpu::core::Metadata` — typed GEO sidecar from the `.1pz` TLV block:
  ```cpp
  struct Metadata {
      std::string              gsm_id, gse_id;
      std::string              organism;
      int32_t                  taxon_id = 0;
      std::string              protocol, modality;
      std::vector<std::string> srr_ids;
      int64_t                  read_count = 0;
      std::string              geo_title, geo_source_name;
      std::string              singlet_version, pipeline_date;
      std::vector<std::string> rownames, colnames;
  };
  ```

Top-level aliases (re-exported into `singlet::gpu::`): `DeviceCSC`, `DeviceDense`, `DeviceMemory`, `GPUContext`, `PinnedBuffer`, `Metadata`.

## Module: `io`

- `singlet::gpu::io::load_pz(const std::string& path, cudaStream_t stream = nullptr, bool keep_host_pinned = false) -> PzDeviceMatrix` — zero-copy `.1pz` → DeviceCSC. See [`docs/api/io_load_pz.md`](../docs/api/io_load_pz.md).
- `singlet::gpu::io::PzDeviceMatrix` — return type: `{ DeviceCSC mat, Metadata meta, cudaStream_t producer_stream, PinnedBuffer indptr/indices/values, optional retained host buffers, n_genes, n_cells }`.
- `singlet::gpu::io::PzChunkIterator` — streaming iterator for matrices that exceed device memory.

Top-level aliases: `load_pz`, `PzDeviceMatrix`.

## Module: `preprocess`

- `singlet::gpu::preprocess::log_normalize(core::DeviceCSC& mat, const LogNormConfig&, cudaStream_t) -> LogNormResult` — total-count + log1p (382× scanpy). See [`docs/api/preprocess_log_normalize.md`](../docs/api/preprocess_log_normalize.md).
- `singlet::gpu::preprocess::compute_deconv_size_factors(const core::DeviceCSC&, const int32_t* clusters, const DeconvSizeFactorsConfig&, cudaStream_t) -> DeconvSizeFactorsResult` — scran-style pool-and-deconvolve via cuSOLVER batched QR.
- `singlet::gpu::preprocess::select_hvg(const core::DeviceCSC&, const HvgConfig&, cudaStream_t) -> HvgResult` — Seurat v3 VST (107×) or Pearson residuals (12,597×). See [`docs/api/preprocess_select_hvg.md`](../docs/api/preprocess_select_hvg.md).
- `singlet::gpu::preprocess::deviance_feature_selection(const core::DeviceCSC&, const DevianceHvgConfig&, cudaStream_t) -> DevianceHvgResult` — scry-style binomial / Poisson deviance.
- `singlet::gpu::preprocess::scale(const core::DeviceCSC&, const float* d_mean, const float* d_std, const ScaleConfig&, cudaStream_t) -> core::DeviceDense` — sparse → dense z-score with ±10 clip; 3 overloads. See [`docs/api/preprocess_scale.md`](../docs/api/preprocess_scale.md).
- `singlet::gpu::preprocess::regress_out(float* X, int n_genes, int n_cells, const float* C, int p, cudaStream_t)` — in-place QR residualization, p ≤ 32.

Top-level aliases: `log_normalize`, `compute_deconv_size_factors`, `select_hvg`, `deviance_feature_selection`, `scale`, `regress_out`.

## Module: `reduce`

- `singlet::gpu::reduce::svd::auto_select(const io::PzDeviceMatrix&, int k, const SvdConfig&) -> SvdResult` — primary SVD entry point. Routes to deflation (Cycle-61 winner) with randomized fallback. **27× scanpy at k=50.** See [`docs/api/reduce_svd.md`](../docs/api/reduce_svd.md).
- `singlet::gpu::reduce::svd::deflation` / `randomized` — direct backends, qualified-only access.
- `singlet::gpu::reduce::nmf::fit(const io::PzDeviceMatrix&, const NmfConfig&, const FitConfig&, const DenseMatrix* W_init, const DenseMatrix* H_init) -> NmfResult` — factornet adapter + Cycle-86 `FitConfig` shim (`k_cd_cutoff=32` → MU at high rank). **1.82–8.66× sklearn.** See [`docs/api/reduce_nmf.md`](../docs/api/reduce_nmf.md).

Both SVD and NMF require `keep_host_pinned=true` on the input `PzDeviceMatrix` (factornet GPU adapters take host pointers).

Top-level aliases: `auto_select`. (`reduce::nmf::fit` qualified-only — too generic a name to bring to top level.)

## Module: `qc`

- `singlet::gpu::qc::calculate_qc_metrics(const core::DeviceCSC&, const core::DeviceMemory<uint8_t>& is_mt, const core::DeviceMemory<uint8_t>& is_ribo, cudaStream_t, QcConfig) -> QcResult` — per-cell + per-gene QC in one pass. **429× scanpy.** See [`docs/api/qc_metrics.md`](../docs/api/qc_metrics.md).
- `singlet::gpu::qc::filter_cells(const core::DeviceCSC&, const QcResult&, const FilterConfig&, cudaStream_t) -> core::DeviceCSC` — column-compact CSC.
- `singlet::gpu::qc::filter_genes(const core::DeviceCSC&, const QcResult&, const FilterConfig&, cudaStream_t) -> core::DeviceCSC` — row-compact + relabel.
- `singlet::gpu::qc::doublet_score(const core::DeviceDense& embedding, const DoubletScoreConfig&, cudaStream_t) -> DoubletScoreResult` — Scrublet-equivalent. Operates on a PCA embedding.

Top-level aliases: `calculate_qc_metrics`, `filter_cells`, `filter_genes`, `doublet_score`.

## Module: `graph`

- `singlet::gpu::graph::compute_knn(const core::DeviceDense& embedding, const KnnConfig&, cudaStream_t) -> KnnResult` — Exact (cub::DeviceSegmentedRadixSort, **2.1× sklearn**) or CAGRA (blocked on cuVS install). See [`docs/api/graph_knn.md`](../docs/api/graph_knn.md).
- `singlet::gpu::graph::compute_snn(const KnnResult&, const SnnConfig&, cudaStream_t) -> SnnResult` — shared-nearest-neighbour Jaccard pruning.

Top-level aliases: `compute_knn`, `compute_snn`.

## Module: `embed`

- _none yet_

## Module: `de`

- `singlet::gpu::de::wilcoxon_de(const core::DeviceCSC& mat, const core::DeviceMemory<int>& labels, int n_clusters, const WilcoxonConfig&, cudaStream_t) -> WilcoxonResult` — rank-sum DE. **6.5×–388.8× scanpy.** See [`docs/api/de_wilcoxon_ttest.md`](../docs/api/de_wilcoxon_ttest.md).
- `singlet::gpu::de::ttest_de(const core::DeviceCSC&, const core::DeviceMemory<int>& labels, int n_clusters, const TtestConfig&, cudaStream_t) -> TtestResult` — Welch's t-test. **8.4×–10.4×.**

Both produce `std::vector<ClusterMarkers>` per_cluster, with `gene_indices`, `z_scores` / `t_values`, `log2_fc`, `p_values`, `p_adj` (BH FDR).

Top-level aliases: `wilcoxon_de`, `ttest_de`.

## Module: `gsea`

- _none yet_

## Module: `anno`

- _none yet_

## Module: `integrate`

- _none yet_

## Module: `models`

- _none yet — scVI/scANVI/totalVI deferred_

## Module: `fate`

- _none yet_

## Module: `streaming`

- `singlet::gpu::streaming::PzShardIterator` — iterates `.1pz` shards under a fixed VRAM budget.
- `singlet::gpu::streaming::run_pipeline(...)` — reference end-to-end streaming driver.

## Backfill plan

✅ **Complete as of CYCLE-100 (2026-04-29)**: 16 public functions exposed via the umbrella `singlet_gpu.hpp`. Every documented frontier feature is now declarable as `singlet::gpu::<function>` (top-level alias) or via its module namespace.

The next pass (CYCLE-101) audits Python/R wrappers against this surface to identify gaps blocking the `documented → released` transition.

## Future additions (not yet on frontier — see `state/roadmap.md`)

- `embed::umap` (feature 10) — blocked on cuVS install.
- `graph::leiden` (feature 9) — blocked on cuGraph install.
- `integrate::harmony`, `integrate::bbknn` (feature 14).
- `gsea::fgsea`, `gsea::aucell` (feature 12).
- `anno::*` (feature 13).
- `de::donor_pseudobulk` (feature 11 sub-variant — frontier-pending).
- `models::scvi` / `scanvi` / `totalvi` (feature 15 — P2 todo).
- `fate::velocity` / `pseudotime` (feature 16 — P2 todo).
- `streaming::PzShardIterator`, `streaming::run_pipeline` (feature 17 — partial frontier).

## CYCLE-118 through CYCLE-137 — frontier additions (2026-04-29 session)

Thirteen new GPU kernels added in one autonomous loop session, all 5+ correctness tests PASSing. Need backfilling into the `singlet_gpu.hpp` umbrella header + Python wrapper exposure. Tracked via `CYCLE-138-FOLLOWUP-UMBRELLA-EXPOSURE`:

| Cycle | Kernel | Header | Status |
|-------|--------|--------|--------|
| 118 | Pearson residuals (Lause-Berens-Kobak 2021) | `preprocess/pearson_residuals.h` | frontier |
| 124 | MAGIC graph-diffusion imputation (van Dijk 2018) — **FIRST GPU** | `preprocess/magic.h` | frontier |
| 127 | scran::modelGeneVarByPoisson (Lun 2016) | `preprocess/model_gene_var.h` | frontier |
| 128 | decoupleR WSUM + WMEAN (Badia-i-Mompel 2022) | `enrich/decoupler_wsum.h` | frontier |
| 129 | scanpy.tl.score_genes (Satija 2015) | `enrich/score_genes.h` | frontier |
| 130 | decoupleR ULM | `enrich/decoupler_ulm.h` | frontier |
| 131 | scanpy.pp.combat (Johnson 2007) | `integrate/combat.h` | frontier |
| 132 | decoupleR ORA | `enrich/decoupler_ora.h` | frontier |
| 133 | LISI batch-integration metric (Korsunsky 2019) | `integrate/lisi.h` | frontier |
| 134 | DropletUtils::emptyDrops (Lun 2019) — **FIRST GPU** | `qc/empty_drops.h` | frontier |
| 135 | CellTypist.predict (Domínguez Conde 2022) | `anno/celltypist.h` | frontier |
| 136 | decoupleR MLM | `enrich/decoupler_mlm.h` | frontier |
| 137 | decoupleR VIPER (Alvarez 2016) | `enrich/decoupler_viper.h` | frontier |
| 138 | Symphony reference mapping (Kang 2021) | `anno/symphony.h` | frontier |
| 139 | Average Silhouette Width (Rousseeuw 1987) | `integrate/asw.h` | frontier |
| 140 | kBET batch-effect test (Buttner 2019) | `integrate/kbet.h` | frontier |
| 141 | SoupX ambient RNA correction (Young 2020) — **FIRST GPU** | `qc/soupx.h` | frontier |
| 142 | Diffusion Pseudotime (Haghverdi 2016) | `embed/dpt.h` | frontier |

decoupleR coverage on GPU = **6 of 6 main methods complete** as of CYCLE-137.
scIB integration evaluation triplet = **3 of 3 complete** (LISI + ASW + kBET) as of CYCLE-140.
Raw-10X preprocessing duo = **complete** (emptyDrops + SoupX) as of CYCLE-141.
Cell-type annotation = **2 paradigms** (CellTypist logreg + Symphony centroid) as of CYCLE-138.
Trajectory inference foundation = **DPT** as of CYCLE-142.

**Three literature firsts** as of CYCLE-141: MAGIC GPU, emptyDrops GPU, SoupX GPU.

Documentation pages (`docs/api/*.md`) and Python wrapper exposure (`python/singlet_gpu/*/`) are CYCLE-143-FOLLOWUP queued.
