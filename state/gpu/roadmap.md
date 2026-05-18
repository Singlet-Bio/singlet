# singlet-gpu — Roadmap

Single source of truth for the feature queue. The orchestrator reads this in Phase 0 of every cycle.

Focused on the foundational single-cell EDA workflow. Advanced niche methods (Cell2fate, CSI-GEP, STAGATE, …) are deferred indefinitely — their headers remain in-tree but receive no development. See `state/followups.md` § advanced/deferred-indefinitely if scope reopens.

> **As of 2026-04-28**: 9 features on frontier (0, 2, 3, 4, 5, 6, 7, 8 partial, 11). Features 9, 10 blocked on cuVS/cuGraph install (`state/blockers.md` → INFRA-CUVS-CUGRAPH-INSTALL). Active P0 work: feature 17 streaming completion + website-backfill cycle for the 9 frontier rows. Three-state ladder (`frontier → documented → released`) introduced in Cycle 89 makeover; existing frontier features will be brought through Phase H + wrappers in subsequent cycles.

## Active Roadmap

| # | Status | Feature | Module | SOTA to beat | Tier |
|---|---|---|---|---|---|
| 0 | **frontier** | Zero-copy `.1pz` -> `SparseMatrixGPU<float>` | `io/pz_device_loader.h` | scanpy, anndata-gpu | -- |
| 1 | done | Core types (factornet re-export) | `core/` | -- | -- |
| 2 | **partial frontier** | Normalization (total-count+log1p FRONTIER; scran, SCTransform, CLR, TF-IDF todo) | `preprocess/lognorm.h` | scanpy, rapids-sc, scran | -- |
| 3 | **frontier** | HVG (Seurat v3 VST + Pearson residuals FRONTIER; Cell Ranger, deviance todo) | `preprocess/hvg.h` | scanpy, rapids-sc | -- |
| 4 | **frontier** | PCA/SVD (5 factornet backends; ADOPT WINNER pending) | `reduce/svd/` | scanpy, cuml, factornet CPU | -- |
| 5 | **frontier** | NMF (factornet adapter + speckled CV auto-rank) — 1.82-8.66× sklearn across k=10-100 (Cycle 86 fix: k_cd_cutoff=32 MU forcing) | `reduce/nmf/` | RcppML, sklearn NMF, factornet CPU | P0 |
| 6 | **frontier** | QC metrics + cell/gene filtering + doublet detection (all GPU tests pass; bench vs scanpy 429× small, 74M cells/sec medium) | `qc/` | scanpy, rapids-sc scrublet | P0 |
| 7 | **frontier** (all tests pass after TF32 tolerance fix) | Scaling + regress_out | `preprocess/scale.h` | scanpy, rapids-sc | P0 |
| 8 | **partial frontier** | kNN + SNN graph (Exact 2.1x sklearn; CAGRA pending cuVS install) | `graph/` | cuml, RAFT CAGRA, FAISS-GPU | P0 |
| 9 | todo | Leiden + Louvain clustering (multi-resolution, subclustering) | `graph/leiden.h` | cuGraph | P0 |
| 10 | todo | UMAP + t-SNE embedding | `embed/` | cuml UMAP, cuml t-SNE | P0 |
| 11 | **frontier** (wilcoxon: TinyPlanted + RealDataPlanted at 20k × 310k all metrics = 1.0; t-test: TinyPlanted Jaccard/LFC/PvalRank = 1.0 + RealDataPlanted all metrics ≥0.9999) | DE: Wilcoxon, t-test, logreg, pseudobulk NB GLM, donor-aware DE | `de/` | scanpy, rapids-sc, DESeq2 | P1 |
| 12 | todo | Gene set scoring + fgsea + AUCell + collapse-pathways | `gsea/` | fgsea, AUCell, scanpy | P1 |
| 13 | todo | Marker scoring + reference-based annotation (SingleR-style + CellTypist) | `anno/` | SingleR, CellTypist | P1 |
| 14 | todo | Batch integration: Harmony (GPU) + BBKNN (GPU) | `integrate/` | rapids-sc Harmony, BBKNN | P1 |
| 15 | todo | scVI / scANVI / totalVI (bare-metal CUDA, no PyTorch) | `models/` | scvi-tools | P2 |
| 16 | todo | Trajectory: velocity prep (from intron_counts.1pz), pseudotime, PAGA | `fate/` | scVelo, scanpy DPT | P2 |
| 17 | **partial frontier** (9/11 pass; lognorm+NMF streaming correct; HVG diverges from in-memory) | Out-of-core streaming driver (billion-cell, multi-GPU) | `streaming/` | Dask-AnnData, rapids-sc | P0 |

## Priority tiers

- **P0**: Core pipeline — must ship before anything else. Features 5-10, 17.
- **P1**: Complete the workflow — DE, gene sets, annotation, integration. Features 11-14.
- **P2**: Advanced foundational — probabilistic models, trajectory. Features 15-16.
- After all tiers: continuous optimization (Rule 25).

## Sub-features within modules

### Feature 2 — Normalization variants
- [x] total-count + log1p (FRONTIER, 370x scanpy)
- [x] scran deconvolution size factors (FRONTIER, 4/5 tests PASS, wall-SOTA gate pending R scran install)
- [ ] SCTransform (regularized NB residuals)
- [ ] CLR normalization (for CITE-seq ADT)
- [ ] TF-IDF (for ATAC peaks)
- [ ] Downsample-to-median

### Feature 3 — HVG variants
- [x] Seurat v3 VST (FRONTIER, 107x scanpy)
- [x] Pearson residuals (FRONTIER, 12609x scanpy)
- [ ] Cell Ranger method
- [ ] scry deviance

### Feature 4 — PCA/SVD (ADOPT WINNER)
Current backends (all from factornet):
- randomized_gpu — general purpose
- lanczos_gpu — Lanczos eigendecomposition
- irlba_gpu — implicitly restarted Lanczos (most popular in R)
- krylov_gpu — constrained subspace
- deflation_gpu — successive rank-1 (currently fastest at k=50: 28ms vs 148ms randomized)
- auto_select — routes by (n, p, k)

**Action**: Next cycle benchmarks all 5 at k={10,30,50,100} at 3 scales. Consolidate to <=2 winners.

### Feature 6 — QC
- Per-cell: n_genes, n_umis, pct_mt, pct_ribo, pct_hb, pct_intronic
- Per-gene: mean, variance, dropout, n_cells
- Threshold filtering (cell + gene)
- Doublet detection (Scrublet-GPU equivalent)

### Feature 8 — kNN + SNN (ADOPT WINNER)
Candidates:
- brute-force fp16 (exact, cells <=50k)
- IVF-Flat (approximate, 50k-500k)
- IVF-PQ (approximate, 500k-10M)
- CAGRA/HNSW-GPU (approximate, 10M+)
SNN: Seurat FindNeighbors Jaccard pruning on kNN output -> CSR adjacency.

**Action**: benchmark all kNN backends, adopt <=2.

### Feature 11 — DE variants
- Wilcoxon rank-sum (scanpy default)
- Welch's t-test
- Logistic regression one-vs-rest
- Pseudobulk NB GLM (DESeq2-equivalent, novel GPU contribution)
- ROC/AUC markers
- Donor-aware pseudobulk DE (from singlet donor_assignments.tsv)

### Feature 15 — Probabilistic models (bare-metal CUDA)
- scVI: VAE for batch-corrected latent space
- scANVI: semi-supervised scVI
- totalVI: joint RNA + protein (CITE-seq)
Design: cuBLAS GEMMs, fused activations, cuRAND reparameterization, in-house Adam. DeviceCSC input via zero-copy from .1pz.

### Feature 16 — Trajectory
- RNA velocity prep from intron_counts.1pz (singlet-unique)
- Diffusion pseudotime
- PAGA graph abstraction

### Feature 17 — Out-of-core streaming
- Shard iterator with fixed VRAM budget
- Streamed HVG, PCA, kNN, clustering, UMAP
- Multi-GPU via NCCL allreduce
- Landmark UMAP for billion-cell

## Deferred indefinitely

Existing headers remain in-tree but receive NO further development:
Cell2fate, CSI-GEP, STAGATE, discrete_diffusion, PerturbGraph, FlashDeconv, chromVAR, Numbat, GRaNIE, CellChat, NEBULA, OmniDoublet, DAESC, Cospar, Monopogen, CellRank2, ssGSEA, PROGENy, Palantir, hdWGCNA, Milo, scDRS.

## Status legend

- `todo` — not started
- `in-progress` — design doc exists, kernel being written or benchmarked
- `blocked` — see DAG; root cause documented
- `frontier` — on the Pareto frontier
- `partial frontier` — some variants on frontier, others todo
- `done` — frontier + wrappers shipped
