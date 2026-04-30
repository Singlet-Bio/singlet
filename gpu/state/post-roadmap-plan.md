# singlet-gpu — Post-Roadmap Plan (Cycle 18+)

The original 17-feature roadmap completes with cycle 17 (`de/donor_pseudobulk.h`). This document plans what comes after, based on the cycle 17 lit-scout's wrapper-best-practices and future-feature surveys.

## Phase A: Wrappers (cycles 18–22)

Per absolute rule §⛔10, wrappers come AFTER the kernel is on the Pareto frontier. Cycle 18 begins the Python + R wrapper sprint.

### Cycle 18 — Python wrapper foundation

- **Module**: `python/src/_singlet_gpu_core.cpp` (pybind11)
- Exposes `core::DeviceCSC` as a `cupy.sparse.csr_matrix` view via the cuda-array-interface protocol.
- Wraps `io::load_pz` returning a `dict` with `cupy_csr` + `metadata` + `host_pinned` shared_ptrs (per cycle 5 loader extension).
- Provides `from_anndata(adata, layer='X')` → `core::DeviceCSC` and `to_anndata(...)` → AnnData.
- Distribution: PyPI wheel via `manylinux2014` runner with CUDA 12.x static link.

### Cycle 19 — Python wrappers for cycles 2–6 kernels (loader, lognorm, hvg, svd, nmf)

One pybind11 binding per cycle, each ~50 LOC. Returns are cupy arrays / `cunnData`-compatible.

### Cycle 20 — Python wrappers for cycles 7–12 (streaming, knn, leiden, umap, de, anno)

### Cycle 21 — Python wrappers for cycles 13–17 (gsea, integration, velocity, mt, donor)

### Cycle 22 — R wrapper foundation

- **Module**: `r/src/singlet_gpu_core.cpp` (Rcpp + RcppArrayFire OR raw Rcpp + manual CUDA copy)
- `read_pz_to_sce(path)` → `SingleCellExperiment` (matrix copied back to `Matrix::dgCMatrix`).
- The R wrapper accepts the host-copy cost — R does not have native GPU sparse types.
- Distribution: NOT submitted to CRAN initially due to native CUDA. Provide via `remotes::install_github`.

## Phase B: New features (cycles 23+)

The lit-scout's top 3 priorities, in order:

### Cycle 23 — Cell2fate velocity modules (extends feature 13)

- **Module**: `embed/cell2fate_modules.h`
- Bayesian ODE + linearization for fate prediction (Nature Methods 2025).
- Consumes `intron_counts.1pz` (already exploited by cycle 15 velocity_prep). Adds a downstream module-assignment step.
- Why GPU: Bayesian inference scales poorly on CPU; vectorized likelihood over cells.
- Builds on cycle 15 velocity_prep output as input.

### Cycle 24 — MultiVI multimodal integration adapter (extends feature 12)

- **Module**: `integrate/multivi.h`
- VAE for CITE-seq + GEX + ATAC joint integration (scvi-tools).
- Wraps the scvi-tools PyTorch model — this is mostly a Python wrapper, so it lives in `python/src/multivi_wrapper.py` rather than the C++ tree.
- Reuses singlify's `snp_ad/adt/fragments` outputs.
- Why GPU: 10–100× speedup on 100k+ cells via PyTorch CUDA.

### Cycle 25 — Spatial graph convolution for Xenium / Visium HD

- **Module**: `spatial/spatial_gcn.h`
- GCN on spot/cell neighborhood + gene expression.
- Consumes `spatial_coords.parquet` from singlify's Visium output.
- New "spatial" module path. First cycle in this subtree.
- Why GPU: O(k² edges) on dense neighborhoods.
- No established GPU SOTA — opportunity for "first GPU" #4.

## Phase C: Long-tail features (cycles 26+)

### scGPT / Geneformer transformer foundation models

- New module path `foundation/`.
- Wraps the published transformers (PyTorch) for cell representation + perturbation simulation.
- Heavyweight; defer until users explicitly request.

### snmCT-seq joint analysis (chromatin + methylation + transcriptome)

- Requires singlify to add `mC_counts.1pz` and `ATAC_counts.1pz` as new modalities — this is a CROSS-AGENT request to the singlify-dev orchestrator. File a `CROSS-AGENT-METHYLATION-OUTPUT` task in `dag.md` when ready.
- Cannot be implemented until the singlify side ships.

### CITE-seq antibody-only dim reduction

- Module `anno/cite_classifier.h`.
- cuML logistic regression on ADT embedding vs reference panels.
- Reuses cycle 12 reference_map pattern.

### BD Rhapsody multiplexing demux

- Module `qc/rhapsody_demux.h`.
- GMM on barcode freq + UMI cutoff.
- New "qc" module path.

## Wrapper distribution channels

| Channel | Status | Notes |
|---|---|---|
| PyPI wheels (manylinux2014) | Cycle 18 first push | Static CUDA 12.x link |
| conda-forge | Cycle 18+1 (auto from PyPI) | rapids label |
| GitHub Releases | Always | source dist |
| CRAN | Probably never | native CUDA disqualifies most submissions |
| `remotes::install_github` | Cycle 22 | R install path |

## Top 3 cycle 18+ priorities (from lit-scout)

1. **Cell2fate velocity modules** — extends our intron_counts.1pz unique advantage; Nature 2025 validation.
2. **MultiVI multi-GPU adapter** — unblocks CITE+GEX+ATAC joint; aligns with scvi-tools ecosystem.
3. **Spatial graph convolution (Xenium/Visium HD)** — fills the spatial gap; no established GPU SOTA = potential first-GPU #4.

But **wrappers come first** (cycles 18–22) because the kernels are useless without them.

## Summary of singlet-gpu after cycle 17

- 17 features, ~17000 LOC of GPU C++ kernels, ~16000 LOC of test infrastructure
- 3 "first GPU" implementations (cycle 13 GSEA, cycle 16 MT lineage, cycle 17 donor pseudobulk)
- 3 "unique to us" features exploiting singlify-only outputs
- Integrates factornet (PCA/NMF), cuGraph (Leiden), cuml (UMAP, kNN), cuVS (HNSW), cuSOLVER (LSQ), cuRAND (Philox4x32)
- Header-only C++20, CUDA 12+, GPL-2.0-or-later
- Open followups (mostly cycle 14 Harmony PCIe fix, cycle 11 deterministic mode, cycle 8 knn radix select) all blocked on first GPU dispatch session
