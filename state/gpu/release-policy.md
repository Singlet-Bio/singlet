# singlet-gpu — Release Policy

## Three feature states

| State | Gate |
|---|---|
| `todo` | not started |
| `in-progress` | design doc exists, kernel being written |
| `frontier` | compiles + correctness on real data + benchmark recorded |
| `documented` | frontier + `docs/api/{feature}.md` + `docs/notebooks/{slug}.ipynb` |
| `released` | documented + Python wrapper + R wrapper + entry in `state/public-api.md` + umbrella export in `singlet_gpu.hpp` + install smoke-test |
| `blocked` | see `state/dag.md` for root cause |

A feature can never be `released` while any predecessor in its dependency tree is below `released`.

## Versioning — semantic, with library-specific rules

`MAJOR.MINOR.PATCH`.

- **PATCH**: bug fixes, perf improvements, no API change. Internal headers can churn freely.
- **MINOR**: new features, new public symbols, new optional `Config` fields. Existing public symbols frozen. Deprecations marked, not removed.
- **MAJOR**: removals from `state/public-api.md`, signature breaks, license change.

The umbrella header `include/singlet-gpu/singlet_gpu.hpp` is API-frozen across MINOR. Internal headers under module subdirs (`preprocess/`, `de/`, `graph/`, etc.) are NOT frozen — users must include them at their own risk.

## Public API surface

`state/public-api.md` is the contract. Every released feature contributes exactly one stable function (or struct + free function) to `singlet_gpu.hpp`. Anything not in `state/public-api.md` is internal and may break in any PATCH.

Adding a public symbol: append to `state/public-api.md` + add umbrella export + bump MINOR at next release tag.

Removing a public symbol:
1. Mark `[[deprecated("Use X instead. Removal in vN+1.0.")]]` and bump MINOR.
2. After at least one MINOR release with the deprecation warning, remove and bump MAJOR.

Renames go through the same deprecation cycle (old name `[[deprecated]]`, new name added, both for one MINOR).

## Supported platforms

| Axis | Supported |
|---|---|
| CUDA | 12.x |
| GPU arch | sm_70 (V100), sm_80 (A100), sm_90 (H100) |
| C++ | C++20 (gcc-toolset-13 on Clipper; clang ≥17) |
| Python | 3.10, 3.11, 3.12 (pybind11) |
| R | 4.3+ (Rcpp) |
| OS | Linux x86_64 (RHEL 9 / Ubuntu 22.04 tested) |

Anything outside this matrix is best-effort.

## Install paths (smoke-tested per release)

| Path | Smoke test |
|---|---|
| C++ via CMake FetchContent | `cmake --build` of `examples/cpp/cpp_minimal/` succeeds |
| C++ via system install | `make install` + downstream `find_package(singlet-gpu)` works |
| Python via pip | `pip install singlet-gpu` then `import singlet_gpu; singlet_gpu.__version__` |
| R via remotes::install_github | `library(singletGpu); singletGpu::lognorm(...)` |

The smoke tests live at `tests/install/` and run on at least one V100 + one H100 node before any release tag.

## Release cadence

Continuous frontier development. **Cut a MINOR release when**:
- 3+ features have transitioned to `released` since the last MINOR, OR
- A user-visible improvement hits a stable benchmark (e.g., new SOTA-beating kernel).

Cut a PATCH release for any urgent correctness fix.

## Wrapper naming convention

The C++ public surface uses singlet-gpu's own identifiers (`log_normalize`, `compute_knn`, `wilcoxon_de`, etc.). The **Python and R wrappers follow scanpy / Seurat conventions** intentionally — drop-in compatibility is more valuable to users than name parity with the C++ side.

Mapping table (canonical; CYCLE-101 audit):

| C++ | Python (scanpy-style) | R (Seurat / scran idiom) |
|---|---|---|
| `io::load_pz` | `singlet_gpu.load_pz` (top-level) | `singletGpu::load_pz` |
| `preprocess::log_normalize` | `pp.normalize_total` + `pp.log1p` (split) | `normalize_total` + `log1p` |
| `preprocess::compute_deconv_size_factors` | _(planned, CYCLE-103)_ | _(planned)_ |
| `preprocess::select_hvg` | `pp.highly_variable_genes` | `highly_variable_genes` |
| `preprocess::deviance_feature_selection` | _(planned, CYCLE-103)_ | _(planned)_ |
| `preprocess::scale` | _(planned)_ `pp.scale` | _(planned)_ |
| `preprocess::regress_out` | _(planned)_ `pp.regress_out` | _(planned)_ |
| `reduce::svd::auto_select` | `reduce.svd.pca` (scanpy `tl.pca`) | `pca` |
| `reduce::nmf::fit` | `reduce.nmf.nmf` | `nmf` |
| `qc::calculate_qc_metrics` | _(planned)_ `pp.calculate_qc_metrics` | _(planned)_ |
| `qc::filter_cells` / `filter_genes` | _(planned)_ `pp.filter_cells` / `pp.filter_genes` | _(planned)_ |
| `qc::doublet_score` | `qc.run_doublet_score` | `run_doublet_score` |
| `graph::compute_knn` + `compute_snn` | `pp.neighbors` (single call, bundled) | `neighbors` |
| `de::wilcoxon_de` / `ttest_de` | `tools.rank_genes_groups(method=...)` (one entry point with `method=` flag) | `rank_genes_groups(method=...)` |

Per-feature docs pages under `docs/api/` carry an explicit "Python signature (scanpy convention)" subsection that uses the column-2 names. The C++ signature blocks remain authoritative for direct C++ consumers.

## What we promise users

1. Public API surface is stable across MINOR (non-breaking adds only).
2. Every released feature has a docs page, a notebook, a benchmark row, and a correctness row.
3. Install paths are smoke-tested on real hardware.
4. Tag = git tag. (Currently aspirational — repo not yet initialized at workspace root. See Rule 35.)

## What we do NOT promise

- Internal header stability (everything except `singlet_gpu.hpp`).
- Determinism by default (Rule 19 — opt-in).
- Backward compatibility across MAJOR.
- Support for unsupported GPU arch (sm_60 and below, sm_100+).
