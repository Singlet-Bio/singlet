# Bringing the legacy `singlet` NMF package into Singlet-Bio/singlet

Goal: one package for exploratory single-cell analysis that does everything `zdebruine/singlet` (v0.99.x, R,
Seurat-first) does and more (SVD, more flexible NMF), and that works with Seurat, SingleCellExperiment, AnnData,
scran, scater and the Python and R ecosystems, with the legacy function names still working.

This is a plan, not an implementation. What is already true is marked **(done)**.

## 1. Where the two packages stand

| | `zdebruine/singlet` (legacy R) | `Singlet-Bio/singlet` (this repo) |
|---|---|---|
| Purpose | NMF with automatic rank search on a Seurat/SCE object | Find, download and load the atlas; read `.singlet` files |
| Languages | R + C++ (`src/singlet.cpp`, RcppEigen) | Python, R, C++ |
| Analysis | `RunNMF`, `RunLNMF`, `RunGCNMF`, `ProjectData`, `AnnotateNMF`, `RunGSEA`, plots | `singlet.nmf()` = a 94-line scikit-learn wrapper; GPU NMF headers (`include/singlet/gpu/reduce/nmf`, experimental, CUDA 12) |
| Data in | Seurat v3/v4/**v5 (done)**, SingleCellExperiment | `.singlet` -> AnnData / SingleCellExperiment / Seurat (`as_sce`, `as_seurat`) |
| R package name | `singlet` 0.99.9 | `singlet` 1.0.0 (**the same name**) |
| Licence | GPL (>= 2), built on RcppML (GPL) | MIT |

## 2. Decisions that block the work

1. **R package name.** Two packages named `singlet` cannot coexist; installing one replaces the other. Recommended:
   keep `singlet` for Singlet-Bio (1.0.0), fold the NMF functions into it, archive `zdebruine/singlet` after a
   final 0.99.9 release, and tell its users to `install_github("Singlet-Bio/singlet", subdir = "r")`. The
   alternative is renaming the data client (for example `singletbio`), which breaks the documented install line
   and the `singlet::load()` examples already on the website.
2. **Licence of the NMF code.** The legacy C++ is derived from RcppML (GPL). Making it MIT needs every copyright
   holder's consent: you for the code you wrote, anyone else who contributed (Tim Triche is listed as an author;
   PR #53 came from another contributor), and a decision on whether RcppML is a dependency (GPL propagates to the
   combined work in the usual reading) or is reimplemented. Until then, new NMF code in this repo should be
   written from the algorithm descriptions, not copied from `src/singlet.cpp`.
3. **Masking.** `singlet::load` and `singlet::find` mask `base::load` and `utils::find`. Keep them (the website
   documents them) but make `singlet_load()` / `singlet_find()` the documented spelling.

## 3. Target API

One set of verbs, the same in R and Python; results stored where each ecosystem expects them.

| Verb | Python (`singlet.*`) | R (`singlet::*`) | Legacy name it replaces |
|---|---|---|---|
| NMF at a fixed rank | `nmf(adata, k=...)` | `nmf(x, k = ...)` | `RunNMF(k = )`, `run_nmf` |
| Rank search (cross-validation) | `nmf(adata, k="auto")`, `nmf_cv(adata, ranks)` | `nmf(x, k = "auto")`, `nmf_cv()` | `RunNMF()`, `ard_nmf`, `cross_validate_nmf` |
| Linked / integrated NMF | `nmf(adata, k, batch="col")` | `nmf(x, k, batch = "col")` | `RunLNMF`, `split.by` |
| Graph-regularised NMF | `nmf(adata, k, graph=)` | `nmf(x, k, graph = )` | `RunGCNMF` |
| Project onto a model | `project(adata, model)` | `project(x, model)` | `ProjectData`, `project_model` |
| SVD / PCA (new) | `svd(adata, k)`, `pca()` | `svd(x, k)` | (RcppML `svd`) |
| Annotate factors | `annotate(model, obs=)` | `annotate(model, meta)` | `AnnotateNMF`, `getModelFit` |
| Gene-set enrichment | `enrich(model, sets)` | `enrich(model, sets)` | `RunGSEA` |

Where results live:

- AnnData: `obsm["X_nmf"]` (cells x k), `varm["nmf_loadings"]` (genes x k), `uns["nmf"]` (`d`, `rank_search`, parameters).
- SingleCellExperiment / scran / scater: `reducedDim(sce, "NMF")`, loadings in `rowData` or `metadata(sce)$nmf`.
- Seurat: a `DimReduc` named `nmf` with `feature.loadings`, as the legacy package does.
- Conversion between the three goes through the existing `as_sce()`, `as_seurat()` and AnnData readers; add
  `from_seurat()`/`from_sce()` so a model fitted in one world can be projected in another.

## 4. Phases

**Phase 0 (done).** Stabilise the legacy package: rank-search crashes fixed, Seurat v5 layers supported, tests and a
GitHub Actions workflow added, dependency and Makevars cleanup. Branch `stabilize/seurat-v5-and-ci`.

**Phase 1: contract tests.** Before porting anything, turn the legacy behaviour into fixtures: for pbmc3k and
the synthetic matrices in `tests/testthat/helper-data.R`, record the rank chosen, `d`, and the loadings up to
column order and scale. These are the numbers the new implementation must reproduce (within tolerance, since
random initialisation differs).

**Phase 2: one CPU engine.** A header-only C++ NMF/SVD core in `include/singlet/reduce/` (CPU; the CUDA headers
become an optional backend behind the same interface): MU and coordinate descent, L1/L2, masked test-set
cross-validation, randomised and deflation SVD. Bindings exist for both Python (`src/bindings/python`) and R
(`r/src`); reuse them.

**Phase 3: Python verbs** over that engine, replacing the scikit-learn wrapper, with the AnnData layout above.

**Phase 4: R verbs** for SCE and Seurat (v4 and v5, using the layer helpers in the legacy `R/seurat_compat.R`),
plus **compat shims**: `RunNMF()`, `RunLNMF()`, `RunGCNMF()`, `ProjectData()`, `AnnotateNMF()`, `RunGSEA()`,
`ard_nmf()`, `cross_validate_nmf()` as thin wrappers that call the new verbs and emit a once-per-session message
naming the replacement. The legacy test suite runs against the shims unchanged.

**Phase 5: interop tests.** Round-trip a model Seurat -> SCE -> AnnData and back; check scran and scater pipelines
that read `reducedDim(sce, "NMF")`; add a vignette that goes from `singlet_load("GSE138867")` to an NMF embedding.

**Phase 6: release.** Submit `singlet-bio` to PyPI and `singlet` to CRAN or Bioconductor once the licence and
name decisions above are settled. CRAN needs the zstd system requirement declared and the `load`/`find` masking
explained in the package documentation.

## 5. Risks

- Numerical drift: NMF is non-convex. Compare reconstruction error and rank, not factor values.
- Rank search is the legacy package's selling point and its most fragile part (see the 2025 issues): port the test
  cases in `tests/testthat/test-rank-search.R` first.
- The GPU headers need CUDA 12 and a source build; keep them optional and out of the wheels.
- Two R packages, one name: until decision 1 is made, do not publish either to a package index.
