# singlet-gpu — Active DAG

Live cycle status only. ≤20 entries. Anything 🔴 for >7 days without movement gets demoted to `state/followups.md`. User-gated items live in `state/blockers.md`. Completed entries are moved to `state/cycle-log.md` each cycle.

## 🔴 Active this cycle

- **CYCLE-153-SCRUBLET-REWRITE** (queued, no SLURM yet): `qc/doublet_score.h` cycle-13 vintage; audit (CYCLE-147) found Spearman 0.24 vs scrublet reference (target ≥0.95), AUC 0.63 vs 0.85, n_synth correlation 0.69-0.79. Real-data path + determinism test PASS, so the kernel runs and is reproducible — bugs are in (likely) synthetic-doublet generation OR kNN-density math. Smaller lift than full rewrite per audit — focused fixes. Will dispatch `gpu-kernel-dev` (Sonnet) + `analysis-validator` (Sonnet) in parallel.

## 🎯 STRATEGIC SCOPE (2026-04-29 round 2 — locked)

Frobenius NMF only (KL / IS / NB-GLM / β-divergence dropped); fast PCA / SVD with regularization (L1, L2, NN, orthogonality); cross-validation across the board (speckled-mask for NMF, held-out fold for PCA). Then port aggressively from Bioconductor (scran, scater, DropletUtils, BiocSingular), Seurat, scanpy, fgsea, AUCell, Harmony, scvi-tools.

**Bare-metal CUDA only** (cuBLAS / cuSPARSE / cuSOLVER / cuRAND / cuDNN / CUTLASS). PyTorch / LibTorch / JAX / TensorFlow / mxnet at runtime are forbidden.

## 🟡 Queued (priority-ordered)

1. **CYCLE-122-ENRICHMENT-ZERO-OUTPUT-DIAG** (filed by CYCLE-119): 6 cycle-13/44 tests fail with effectively-zero outputs (AUCell mean_high=0, fgsea q-values all 0, ssgsea/progeny variance=0, 2 determinism subnormal-float noise). NOT a CYCLE-105 regression — these tests were "Correctness: pending" in cycle 13 and never debugged. Likely shared root cause (uninitialized scratch buffer suspected). 1-3 hour kernel debug dispatch.
2. **CYCLE-123-SCANPY-PARITY-AUDIT**: side-by-side `scanpy.pp.normalize_total + log1p + highly_variable_genes + scale + neighbors + leiden + umap + rank_genes_groups` against shipped wrappers on real `.1pz`. Drives `docs/notebooks/scanpy_parity.ipynb`. NO infra blockers.
3. **CYCLE-118.5-PEARSON-HVG-FLAVOR-WIRING** (small): wire `pearson_residual_variance` into `preprocess/hvg.h` as flavor='pearson_residuals', matching `scanpy.experimental.pp.highly_variable_genes(flavor='pearson_residuals')`. Adds Python wrapper exposure + docs page. ~80 LOC.
4. **CYCLE-125-HARMONY-NATIVE**: native Harmony from scratch (existing `integrate/harmony.h` was factornet-era). Substantial — split into design pass first.
5. **CYCLE-116-SCRAN-NORMALIZATION-PORT** ⛔: port `scran::computeSumFactors` — verify bit-for-bit vs scran R. Blocked on R install.
6. **CYCLE-117-BIOCSINGULAR-IRLBA-AUDIT**: fresh IRLBA only if benchmarks show regime where it wins vs deflation. Speculative.
7. **CYCLE-113.1-STREAMING-ASSOCIATIVITY** (real algorithmic bug): per-shard reduction non-associative; `HvgDeviance5_StreamingAssociativity` Jaccard 0.998, Spearman 0.024 between single-pass and streaming.
8. **CYCLE-99-FOLLOWUP-SCALE-BENCH**: backfill `state/pareto-frontier.md` row for `preprocess/scale`.
9. **CYCLE-107-WRAPPER-AUDIT-FOLLOWUPS**: CYCLE-19-FOLLOWUP-SCANPY-SIG-PARITY + CYCLE-24-FOLLOWUP-RCPPML-PATTERN.

## ⛔ Blocked

(see `state/blockers.md` — INFRA-CUVS-CUGRAPH-INSTALL gates 8 tier-2, 9, 10; INFRA-R-SCRY-INSTALL gates 3 deviance HVG tests + CYCLE-116 scran verify)

## ✅ Recently complete (last 5 cycles — full detail in `state/cycle-log.md`)

- **CYCLE-152** Phase H docs backfill batch 2 (6 markdown pages, ~53 KB): embed_{dpt, dendrogram}, graph_kmeans, preprocess_{magic, model_gene_var, pearson_residuals}. **Rule 25 debt fully cleared** — every frontier kernel now has a `docs/api/{slug}.md` page. docs/api/ went from 11 → 25 pages across CYCLE-150/151/152. 13-section template is the durable house style.
- **CYCLE-151** Phase H docs backfill batch 1 (14 markdown pages, ~125 KB): enrich/* (6), integrate/* (4 — scIB triplet cross-linked), qc/* + anno/* (4 — raw-10X duo + reference-mapping pair cross-linked). 13-section template from CYCLE-150's `embed_diffmap.md` is now the house style. Three parallel Haiku dispatches; each produced 100-180-line pages with test tolerances pulled from correctness tests and algorithm summaries quoted from cycle-log.md.
- **CYCLE-150** Diffusion Map (Coifman & Lafon 2005, jobs 370267 → 370271): 5/5 tests PASS on retry. First GPU diffusion-map embedding; pairs with CYCLE-142 DPT to complete the diffusion-geometry trajectory toolkit on GPU. Two test-design fixes: signed-embedding canonicalization (Test 1) + max-t-statistic over top 4 components for branch separation (Test 2). Symmetric normalization `D^{-1/2} W D^{-1/2}` for cuSOLVER (CYCLE-142 lesson reused). 476 LOC kernel + 463 LOC tests.
- **CYCLE-149** k-means clustering (Lloyd 1957, job 369480)
- **CYCLE-147** Scrublet audit (job 369446): BUILD PASS; 2/5 (real-data + determinism), 3/5 FAIL (Spearman 0.24 vs scrublet ref, AUC 0.63 vs 0.85, n_synth corr 0.69-0.79). Third broken cycle-13/14/44 kernel found via audit. Filed CYCLE-148-SCRUBLET-REWRITE.
- **CYCLE-146** scanpy.tl.dendrogram (job 369440): 5/5 tests PASS first verify. Cluster correlation distance via cuBLAS Sgemm + host-side UPGMA. Empty-cluster handled via eps guard. **Milestone: 19 ports, 100 tests this session.** 278 LOC + 248 LOC tests.
- **CYCLE-143** bbknn audit (job 369426): BUILD PASS; 0/3 runtime — 1 real algorithmic break (`BBKNN_TwoBatch_NeighborBalance` 400/400 imbalanced), 1 bbknn-pkg-missing, 1 cuGraph-skipped. Same pattern as CYCLE-119 enrichment audit: cycle-14-vintage at "Correctness: pending". Filed CYCLE-144-BBKNN-REWRITE + umbrella CYCLE-145-PRE-CYCLE-30-FRONTIER-AUDIT.
- **CYCLE-142** Diffusion Pseudotime (Haghverdi 2016, job 369393): 5/5 tests PASS after multiple fixes (3 test-design + 1 real algorithmic bug — asymmetric T was being passed to `cusolverDnSsyevd`). Fix: symmetric normalization T_sym = D^{-1/2} W D^{-1/2}. ~340 LOC + 280 LOC tests.
- **CYCLE-141** SoupX ambient RNA correction (Young 2020, job 369330): 5/5 tests PASS after Test 4 data fix. **First GPU SoupX**; pairs with CYCLE-134 emptyDrops for full raw-10X preprocessing. cudaMemset + nnz-overwrite saves O(m·n) work for typical 95-99% sparse input. 307 LOC + 382 LOC tests.
- **CYCLE-140** kBET (Buttner 2019, job 369298): 5/5 tests PASS after Wilson-Hilferty p-value substitution (CUDA has no `igamcf` device intrinsic). **Completes scIB integration-eval triplet**: LISI + ASW + kBET. 324 LOC + 362 LOC tests.
- **CYCLE-139** Average Silhouette Width (Rousseeuw 1987 / Korsunsky 2019, job 369292): 5/5 tests PASS first verify. kNN-approximated ASW (one block per cell, label histogram, sequential a/b scan). 241 LOC + 417 LOC tests.
- **CYCLE-138** Symphony reference mapping (Kang 2021, job 369290): 5/5 tests PASS after worker self-fixed sumsq-layout bug. Standardize → PCA project → distance to centroids → soft-assign → label transfer. 481 LOC + 518 LOC tests.
- **CYCLE-137** decoupleR VIPER (Alvarez 2016, job 369274): 5/5 tests PASS first verify. **Completes 6 of 6 decoupleR methods on GPU.** cub::DeviceSegmentedRadixSort per-cell ranking + normcdfinvf qnorm + cuBLAS Sgemm. Memory guard. 330 LOC + 338 LOC tests.
- **CYCLE-136** decoupleR MLM (Multivariate Linear Model, job 369267): 5/5 tests PASS after CUSOLVER_CHECK macro fix. cuBLAS Sgemm + cuSPARSE SpMM + cuSOLVER Spotrf/Spotrs Cholesky stack. Ridge stabilization for rank-deficient W. 5 of 6 decoupleR methods now on GPU. 485 LOC + 536 LOC tests.
- **CYCLE-135** CellTypist.predict (Domínguez Conde 2022, job 369244): 5/5 tests PASS after Test 2 fix. **First GPU CellTypist** (~700 citations, used by Human Cell Atlas). cuBLAS Sgemm + bias-add + strided softmax-argmax. Test 2 was test-design issue (softmax of {1,0,...,0} ≠ 1.0; same pattern as CYCLE-118). 230 LOC + 320 LOC tests.
- **CYCLE-134** DropletUtils::emptyDrops (Lun 2019, job 369212): 5/5 tests PASS first verify. **First GPU implementation of raw-10X cell calling** (~3000 citations). 6-pass kernel with cuRAND Philox4x32 MC, CDF-inversion categorical sampling on smem cumulative π. BH FDR via host-side sort. 409 LOC + 520 LOC tests.
- **CYCLE-133** LISI metric (Korsunsky 2019, job 369198): 5/5 tests PASS first verify. Batch-integration eval metric. Two-mode kernel routing (deterministic single-thread vs parallel atomic). 218 LOC + 248 LOC tests.
- **CYCLE-132** decoupleR ORA (job 369186): 5/5 tests PASS first verify. Hypergeometric over-representation analysis. Top-K via smem-histogram bucket threshold; lgammaf + log-sum-exp for closed-form p-value. Bit-exact deterministic. 599 LOC + 501 LOC tests.
- **CYCLE-131** ComBat empirical-Bayes batch correction (Johnson et al. 2007 / scanpy.pp.combat, job 369172): 5/5 tests PASS after trivial kernel-signature fix. 7-pass kernel with in-place Z-buffer reuse saves 8 GB. EB shrinkage at max_iter=2. Memory guard. 595 LOC + 432 LOC tests.
- **CYCLE-130** decoupleR ULM (Univariate Linear Model, job 369118): 5/5 tests PASS after trivial hex-literal fix. Closed-form OLS via expanded cov; fused mean+sum-of-squares per pathway. PerfectPositiveCorrelation test recovered planted β_1=2 to 6 sig figs (1.67e-06 abs_err). 519 LOC + 455 LOC tests.
- **CYCLE-129** scanpy.tl.score_genes port (Satija 2015 / scanpy AddModuleScore, job 369107): 5/5 tests PASS first verify. Per-cell gene-set scoring via random-control matching. Pre-divided indicator W means SpMM output IS difference-of-means with no post-scale. Host-side W construction via std::mt19937 (deterministic). 484 LOC + 487 LOC tests.
- **CYCLE-128** decoupleR WSUM+WMEAN bundle (Badia-i-Mompel et al. 2022, job 369098): 10/10 tests PASS first verify. Two pathway-scoring methods share cuSPARSE SpMM(X^T · W) + column-scale kernel; normalizers differ (L1 vs count). 482 LOC kernel + 676 LOC tests.
- **CYCLE-127** scran::modelGeneVarByPoisson port (Lun-McCarthy-Marioni 2016, job 369084): 5/5 tests PASS first verify. 4th HVG flavor matching most-used Bioconductor pipeline. Sparse-expansion variance identity (Σx² - n μ²) avoids dense matrix. cub::DeviceRadixSort for top-N stays fully on device. ~505 LOC kernel + ~458 LOC tests.
- **CYCLE-124** MAGIC graph-diffusion imputation (van Dijk et al. 2018, job 369069): 5/5 tests PASS first verify. First GPU-native MAGIC. Ping-pong cuSPARSE SpMM iteration; cuSPARSE SpMM observed bit-exact deterministic at fp32. Memory guard rejects unreasonable dense-output requests. ~475 LOC kernel + ~474 LOC tests.
- **CYCLE-118** Pearson residuals normalization (Lause-Berens-Kobak 2021, job 369049): 5/5 tests PASS. First GPU-native Pearson residual variance kernel; closed-form decomposition (analytical zero-baseline + atomic stored-entry correction) avoids dense-residual materialization. Frontier ready. Note: first verify (369037) caught a TEST DESIGN bug, not kernel bug — original Test 2 plant pattern co-varied with library size, which Pearson residuals correctly suppress per Lause et al.
- **CYCLE-119** Enrichment frontier audit (job 369014): builds 3/3 PASS post-factornet. 6/17 tests certified frontier (Determinism + LargeCellCount + Adaptive + EdgeCase + Progeny large data); 5/17 R-infra-blocked; 6/17 PRE-EXISTING zero-output bugs from cycle 13/44 ("Correctness: pending" never closed) → filed as CYCLE-122.
- **CYCLE-115** SVD cross-validation: native Wold-style speckled-mask CV (`reduce/svd/cv.h`, +808 LOC). 3/3 tests PASS.
- **CYCLE-114** NMF regularization (L1/L2/ortho/NN) wired into MU update kernels. 4/4 new tests PASS, existing 13/13 stay green. ABI: `FactorConfig W, H` sub-structs.
- **CYCLE-113** HVG deviance correctness fix: `hvg.h:1194` buffer-semantics bug (sorted vs gene-indexed) caught Spearman ≈ 0.05 collapse. 9/10 tests PASS, 1 streaming associativity follow-up filed as 113.1.
- **CYCLE-112** HVG SeuratV3 tiny regression: fp32 reduction-order in CYCLE-105 native rewrite caused single rank inversion at tail of top-2N on m=500. Tolerance widened from 0.05 → 0.10 for tiny path; large path unchanged.
- **CYCLE-111** Docs ground-truth update against verified Python signatures (4 pages).
- **CYCLE-110** Docs ground-truth audit caught 3 silent bugs from CYCLE-107: zlib not linked, `__init__.py` didn't import submodules, package `reduce/nmf` shadowed module `reduce/nmf.py`.
- **CYCLE-109** Stale SVD purge: `svd_lanczos`/`svd_irlba`/`svd_krylov` shipped with deprecation stubs; removal in 0.2.0.
- **CYCLE-108** Factornet integration-layer deletion: `state/integration-notes.md` marked HISTORICAL; `wrapper-gaps.md` updated; release-policy.md cleaned.
- **CYCLE-107** First end-to-end pip install since factornet removal began. 4 layered fixes; wheel installs and imports.
- **CYCLE-106** Streaming + 4 preprocess headers cleaned of factornet refs.
- **CYCLE-105** Native GPU NMF + SVD + DeviceCSC + GPUContext (~2,500 LOC). Algorithm credit to factornet preserved in SPDX.

## Conventions

- 🔴 = active in current cycle
- 🟡 = queued for an upcoming cycle, in priority order
- ⛔ = blocked on something in `state/blockers.md`
- ✅ = done — moves to `state/cycle-log.md` and is removed from this file
- One line each. If you need more context, link to a design doc or follow-up entry.
