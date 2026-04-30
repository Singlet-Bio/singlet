---
feature: python_kernel_wrappers_part_1
roadmap_id: 19
module: python/singlet_gpu/preprocess/{lognorm,hvg}.py + python/singlet_gpu/reduce/{svd,nmf}.py + python/singlet_gpu/io/loader.py
status: design
tolerance: per-function output element-wise rel_err ≤ 1e-5 vs the C++ direct call (round-trip through AnnData should not change values)
target_perf: pybind11 marshal overhead ≤2% of kernel wall time
ooc_plan: each function exposes a `chunk_cols` parameter that, when set, drives the underlying streaming variant
---

## Architectural correction from cycle 18 code-reader

The cycle 18 code-reader found that rapids-singlecell **does NOT use a `cunnData` wrapper class** in their main codebase anymore. They operate directly on AnnData with cupy.sparse-backed `.X`. The original cycle 18 design doc mentioned `cunnData` — that was outdated guidance. Going forward (cycles 19+), all Python wrappers follow the **scanpy-compatible AnnData-native pattern**:

```python
def normalize_total(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | None:
    ...
```

Inputs: `AnnData` with `.X` (or `.layers[layer]`) as a `cupy.sparse.csr_matrix`. The wrapper extracts the device CSR, marshals into `core::DeviceCSC` (transpose if needed), calls the C++ kernel, marshals the result back.

Returns: `None` if `inplace=True` (default), or a modified `AnnData` if `inplace=False`.

This matches scanpy + rapids-singlecell exactly, so users can substitute `singlet_gpu.preprocess.normalize_total(adata)` for `scanpy.pp.normalize_total(adata)` without other changes.

## Future migration to nanobind (deferred)

rapids has moved from pybind11 to **nanobind** for smaller binaries and better CUDA support. Cycle 18 already shipped with pybind11; we accept that. **Cycle X (post-cycle-22)** will be a "migrate to nanobind" cycle. Add to `state/dag.md` as a low-priority followup.

## Module layout

```
python/singlet_gpu/
├── __init__.py                 (cycle 18)
├── version.py                  (cycle 18)
├── _core.<so>                  (cycle 18, compiled pybind11 module)
├── io/
│   ├── __init__.py
│   ├── loader.py               # NEW cycle 19: read_pz_to_anndata + write_pz
│   └── _utils.py
├── preprocess/
│   ├── __init__.py
│   ├── lognorm.py              # NEW cycle 19: normalize_total + log1p
│   ├── hvg.py                  # NEW cycle 19: highly_variable_genes
│   └── _common.py
└── reduce/
    ├── __init__.py
    ├── svd.py                  # NEW cycle 19: pca + factornet SVD adapters
    └── nmf.py                  # NEW cycle 19: nmf + factornet NMF + graph adapters
```

The compiled C++ extension `_core.so` from cycle 18 exposes the raw bindings. Cycles 19+ Python files wrap them in scanpy-compatible APIs.

## Per-module API (cycle 19 scope)

### `preprocess/lognorm.py` (~150 LOC)

```python
def normalize_total(
    adata: AnnData,
    *,
    target_sum: float | None = None,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """GPU-native total-count normalization. Mirrors scanpy.pp.normalize_total."""

def log1p(
    adata: AnnData,
    *,
    base: float | None = None,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """GPU-native log1p. Mirrors scanpy.pp.log1p."""
```

Both wrap the cycle-3 `singlet_gpu::preprocess::log_normalize` kernel. The combined `normalize_total → log1p` is exposed as a fused call when `target_sum` is set.

### `preprocess/hvg.py` (~150 LOC)

```python
def highly_variable_genes(
    adata: AnnData,
    *,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",  # or "pearson_residuals"
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """GPU-native HVG selection. Mirrors scanpy.pp.highly_variable_genes."""
```

Wraps cycle-4 `select_hvg`. Writes results into `adata.var['highly_variable']`, `adata.var['means']`, `adata.var['variances']`, `adata.var['variances_norm']`.

### `reduce/svd.py` (~200 LOC)

```python
def pca(
    adata: AnnData,
    *,
    n_comps: int = 50,
    layer: str | None = None,
    backend: str = "auto",  # auto / lanczos / irlba / randomized / krylov / deflation
    center: bool = True,
    scale: bool = False,
    seed: int = 0,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """GPU-native PCA via factornet 5-method SVD with auto-select. Mirrors scanpy.pp.pca."""
```

Wraps cycle-5 SVD adapters. Writes `adata.obsm['X_pca']`, `adata.uns['pca']['variance']`, `adata.uns['pca']['variance_ratio']`, `adata.varm['PCs']`.

### `reduce/nmf.py` (~250 LOC)

```python
def nmf(
    adata: AnnData,
    *,
    n_factors: int = 20,
    loss: str = "MSE",  # MSE / KL / NB / GP / Gamma / Tweedie
    solver_mode: int = 3,  # auto
    init_mode: int = 2,  # IRLBA SVD init
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 0,
    layer: str | None = None,
    inplace: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """GPU-native NMF via factornet. Writes adata.obsm['X_nmf'] and adata.varm['NMF_loadings']."""

def nmf_chunked(
    paths: list[str],
    *,
    n_factors: int = 20,
    chunk_cols: int = 100_000,
    **kwargs,
) -> NmfResult:
    """OOC streaming NMF over multiple .1pz files via PzDataLoader."""
```

Wraps cycle-6 NMF adapters. The `nmf_chunked` is the streaming entry point — bypasses AnnData since the input is too large to materialize.

### `io/loader.py` (~120 LOC)

```python
def read_pz_to_anndata(
    pz_dir: str,
    *,
    modality: str = "exon",
    keep_host_pinned: bool = False,
) -> AnnData:
    """Load a .1pz directory into a GPU-resident AnnData."""

def write_anndata_to_pz(
    adata: AnnData,
    pz_dir: str,
    *,
    modality: str = "exon",
) -> None:
    """Write an AnnData back to a .1pz directory (round-trip support)."""
```

Wraps the cycle 18 `_core.load_pz` and adds the AnnData wrapper logic that cycle 18's `singlet_gpu/io.py` started.

## Implementation notes (for cycle 19 kernel-dev)

- The wrappers are PURE Python (no C++ changes). They call into `_core` (the cycle 18 pybind11 module) and into `cupy.sparse` for the conversions.
- AnnData → DeviceCSC marshalling: extract `adata.X` (which is a `cupy.sparse.csr_matrix` if loaded via `read_pz_to_anndata`), convert CSR → CSC if needed (`.tocsc()`), call `_core.from_cupy_csr(...)` to wrap in a `DeviceCsc`. **Cycle 18 must expose `from_cupy_csr` for this to work.** Note this dependency.
- Result back to AnnData: same in reverse — `_core.to_cupy_csr(device_csc)` → `cupy.sparse.csr_matrix` → assign to `adata.X`.
- Layer support: when `layer != None`, operate on `adata.layers[layer]` instead of `adata.X`.
- inplace/copy semantics match scanpy exactly.

## Constraints

- **NO host data copies.** All adata.X manipulations stay on device via cupy.sparse.
- **scanpy API parity**: function signatures + parameter names match scanpy.pp.* exactly so users can `import singlet_gpu as sgpu; sgpu.preprocess.normalize_total(adata)` as a drop-in.
- **inplace=True is the default** per rapids-singlecell convention.

## Correctness test spec

Tests in `python/tests/test_preprocess.py`, `test_reduce.py`, `test_io.py`. Pytest fixtures from cycle 18.

Test cases (per function):
1. **`test_{func}_vs_scanpy`**: load GSM4037629 → run our `singlet_gpu.preprocess.normalize_total(adata)` AND `scanpy.pp.normalize_total(adata_cpu)`. Compare `adata.X` element-wise (after copying our cupy view to scipy): rel_err ≤ 1e-5.
2. **`test_{func}_inplace_vs_copy`**: confirm `inplace=True` returns None and modifies adata; `inplace=False, copy=True` returns a new adata with the original unchanged.
3. **`test_{func}_layer_param`**: write the result to `adata.layers['normalized']` instead of `adata.X`.
4. **`test_{func}_drop_in_replacement_for_scanpy`**: import `scanpy as sc; sc.pp.normalize_total = singlet_gpu.preprocess.normalize_total`, run a scanpy tutorial, confirm it passes.

## Build / test

- Cycle 19 produces only Python source. The first GPU dispatch session must `pip install -e python/` (which builds the C++ pybind11 from cycle 18) and `pytest python/tests/`.
- No nvcc on this login node.
