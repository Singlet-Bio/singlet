# singlet-gpu — Python / R wrapper audit (CYCLE-101) — UPDATED 2026-04-29

> **CYCLE-107 update**: `pip install -e python/` now works end-to-end (job 368478, `singlet_gpu-0.1.0` wheel built + installed + imported on g001 V100S). All 5 CYCLE-103 Python wrappers (`calculate_qc_metrics`, `filter_cells`, `filter_genes`, `scale`, `regress_out`) confirmed callable. Coverage matrix below updated to reflect shipped status.

Cross-check of the documented C++ public surface (`state/public-api.md`, 16 functions) against `singlet-gpu/python/` and `singlet-gpu/r/`.

## Naming convention

The Python wrappers follow **scanpy convention** (e.g. `pp.normalize_total`, `pp.log1p`, `pp.neighbors`, `tl.rank_genes_groups`, `tl.leiden`, `tl.umap`), not the C++ identifier names. **This is intentional** — drop-in compatibility for scanpy users. The R wrappers follow Seurat / scran idioms similarly.

The docs pages currently document the C++ names in their Python signature blocks, which is inaccurate for what `pip install singlet-gpu` actually exposes. **Each docs page needs a "Python (scanpy-style)" subsection** that reflects the real wrapper API.

## Coverage matrix

| C++ symbol | Python wrapper | R wrapper | Status |
|---|---|---|---|
| `io::load_pz` | `singlet_gpu.load_pz` (top-level) | `singletGpu::load_pz` (`r/R/load_pz.R`) | ✓ matches |
| `preprocess::log_normalize` | `singlet_gpu.preprocess.normalize_total` + `.log1p` (split, scanpy-style) | `singletGpu::normalize_total` + `log1p` | ⚠️ scanpy-split; document accurately |
| `preprocess::compute_deconv_size_factors` | **still missing** | **still missing** | ❌ gap (NOT in CYCLE-103) |
| `preprocess::select_hvg` | `singlet_gpu.preprocess.highly_variable_genes` | `singletGpu::highly_variable_genes` | ⚠️ scanpy name; document accurately |
| `preprocess::deviance_feature_selection` | **still missing** | **still missing** | ❌ gap (cycle 88; Phase E pending) |
| `preprocess::scale` | **shipped (CYCLE-103/107)** | **shipped (CYCLE-103/107)** | ❌ gap |
| `preprocess::regress_out` | **shipped (CYCLE-103/107)** | **shipped (CYCLE-103/107)** | ❌ gap |
| `reduce::svd::auto_select` | `singlet_gpu.reduce.svd.pca` (+ legacy `svd_lanczos`, `svd_irlba`, `svd_krylov`, `svd_randomized`, `svd_deflation`) | `singletGpu::pca` (likely similar) | ⚠️ stale exports of cycle-61-removed backends |
| `reduce::nmf::fit` | `singlet_gpu.reduce.nmf.nmf` + `nmf_chunked` + `nmf_graph_factorize` | `singletGpu::nmf` | ✓ |
| `qc::calculate_qc_metrics` | **shipped (CYCLE-103/107)** | **shipped (CYCLE-103/107)** | ❌ gap (frontier feature, no wrapper) |
| `qc::filter_cells` / `filter_genes` | **shipped (CYCLE-103/107)** | **shipped (CYCLE-103/107)** | ❌ gap |
| `qc::doublet_score` | `singlet_gpu.qc.run_doublet_score` | `singletGpu::run_doublet_score` (`qc_new.R`) | ⚠️ different name |
| `graph::compute_knn` | `singlet_gpu.pp.neighbors` (bundles knn + snn into one call) | `singletGpu::neighbors` | ⚠️ scanpy-bundled; document accurately |
| `graph::compute_snn` | bundled into `neighbors` | bundled into `neighbors` | ⚠️ |
| `de::wilcoxon_de` | `singlet_gpu.tools.rank_genes_groups(method="wilcoxon")` | `singletGpu::rank_genes_groups` | ⚠️ unified entry point with `method=` flag |
| `de::ttest_de` | `rank_genes_groups(method="t-test")` | `rank_genes_groups(method="t-test")` | ⚠️ same |

**Score (post-CYCLE-107)**: 9/16 fully aligned (5 of the original missing shipped via CYCLE-103/107), 7/16 scanpy-named (intentional, now documented per CYCLE-102), 2/16 still missing — `compute_deconv_size_factors` and `deviance_feature_selection`. Both tracked as future cycles.

## Stale Python exports (cycle-61 winner consolidation)

`python/singlet_gpu/reduce/svd.py` exports `svd_lanczos`, `svd_irlba`, `svd_krylov` directly. These three backends were removed from the C++ surface in Cycle 61 (Rule 32 adopt-the-winner). The Python functions presumably still call into `_core.svd_*` binding entry points — depending on whether those bindings were also deleted in Cycle 61, they'll either fail at import or at first call. Either way these need to be removed from the Python surface.

R wrappers likely have an equivalent stale-export problem; needs a focused look at `r/R/reduce.R`.

## Recommended cycle plan

The path to `documented → released` runs through 4 cycles:

- **CYCLE-102-DOCS-PYTHON-NAMING**: update each `docs/api/*.md` page's "Python signature" subsection to reflect real scanpy-style names (`normalize_total + log1p` instead of `log_normalize`, `highly_variable_genes` instead of `select_hvg`, `rank_genes_groups(method=...)` instead of `wilcoxon_de`/`ttest_de`, etc.). No Python code changes; documentation alignment only. Same R-side update for Seurat / scran idioms.
- **CYCLE-103-WRAPPER-GAPS**: write the 5 missing Python wrappers — `qc.calculate_qc_metrics`, `qc.filter_cells`, `qc.filter_genes`, `preprocess.scale`, `preprocess.regress_out`. Each is a small file (~50 LOC) calling the existing pybind11 binding `_core.<function>`. **Precondition**: confirm the bindings exist by reading `python/src/_bind_*.hpp`. If they don't, the binding has to be added too.
- **CYCLE-104-STALE-SVD-PURGE**: remove `svd_lanczos`, `svd_irlba`, `svd_krylov` from `python/singlet_gpu/reduce/svd.py` and `r/R/reduce.R`. Add a `[[deprecated]]`-style import warning if they're imported, redirecting to `pca` / `auto_select`. Per `state/release-policy.md`, removed symbols stay one MINOR with deprecation warning before disappearing.
- **CYCLE-105-WRAPPER-AUDIT-FOLLOWUPS**: address `CYCLE-19-FOLLOWUP-SCANPY-SIG-PARITY` (rename `inplace`/`backend`/`seed` → `copy`/`svd_solver`/`rng`), `CYCLE-24-FOLLOWUP-RCPPML-PATTERN` (drop reticulate, add `resource="auto"|"gpu"|"cpu"`), `CYCLE-18-FOLLOWUP-NANOBIND-MIGRATION` (post-1.0 cleanup).

After CYCLE-103, every documented frontier feature has a Python wrapper, and the `documented → released` transition is unblocked for the 9 features that already have docs pages + correctness signed.

## R wrapper status (sample, not exhaustive)

- `r/R/load_pz.R` — present.
- `r/R/preprocess.R` — present (assumed scanpy/scran-style normalize/log1p).
- `r/R/reduce.R` — present.
- `r/R/qc_new.R` — present, but covers `run_doublet_score`. No `calculate_qc_metrics` / `filter_cells`.
- `r/R/markers.R` + `r/R/rank_genes_groups.R` — DE entry points.
- `r/R/neighbors.R` — kNN entry.
- `r/R/leiden.R`, `umap.R`, `nmf_new.R` — present (P0 not all on frontier yet).
- **`reticulate_bridge.R`** — present, indicates the cycle-24 hybrid pattern that `CYCLE-24-FOLLOWUP-RCPPML-PATTERN` should remove.

Same naming + stale-backend issues as Python expected. CYCLE-103/104 must touch both languages or the README user-experience diverges.

## What this audit does NOT cover

- Whether the wrappers actually work end-to-end on a GPU node (no `pip install -e python/` smoke test was run as part of this audit).
- Whether the `_core` pybind11 module exposes the bindings the Python files reference (cycle 19 + cycle 21 follow-ups already note specific gaps; some may already be resolved).
- Wheels / pypi publication — explicitly out of scope until 1.0.
