---
feature: python_kernel_wrappers_part_2
roadmap_id: 20
module: python/singlet_gpu/{tools/leiden,tools/umap,tools/rank_genes_groups,pp/neighbors,tools/markers}.py + matching test files
status: design
tolerance: per-function output element-wise rel_err ≤ 1e-5 vs the C++ direct call; full scanpy API parity for drop-in replacement
target_perf: pybind11 marshal overhead ≤2% of kernel wall time
ooc_plan: each function exposes a `chunk_cols` parameter when applicable
---

## Cycle 20 scope

Python wrappers for cycles 7–12 kernels:
- Cycle 7: `streamed_pipeline.h` → `singlet_gpu.streaming.run_pipeline()`
- Cycle 8: `graph/knn.h` → `singlet_gpu.pp.neighbors()` (scanpy parity)
- Cycle 9: `graph/leiden.h` → `singlet_gpu.tools.leiden()`
- Cycle 10: `embed/umap.h` → `singlet_gpu.tools.umap()`
- Cycle 11: `de/{wilcoxon,ttest}.h` → `singlet_gpu.tools.rank_genes_groups()`
- Cycle 12: `anno/{marker_score,reference_map}.h` → `singlet_gpu.tools.score_genes()` + `singlet_gpu.tools.celltypist_predict()`

## Verified scanpy signatures (from cycle 19 code-reader)

These are the EXACT signatures we must match for drop-in compatibility:

### `sc.pp.pca` (referenced by cycle 19 too — note for cycle 19 followup)
```python
def pca(
    data: AnnData | np.ndarray | CSBase,
    n_comps: int | None = None,
    *,
    layer: str | None = None,
    obsm: str | None = None,
    zero_center: bool = True,
    svd_solver: SvdSolver | None = None,
    chunked: bool = False,
    chunk_size: int | None = None,
    rng: SeedLike | RNGLike | None = None,
    return_info: bool = False,
    mask_var: NDArray[np.bool] | str | None | Default = ...,
    use_highly_variable: bool | None = None,
    dtype: DTypeLike = "float32",
    key_added: str | None | Default = ...,
    copy: bool = False,
) -> AnnData | np.ndarray | CSBase | None
```
**Note**: NO `inplace` parameter. Use `copy=False` (default) for in-place. `n_comps` is positional. `zero_center` not `center`. `svd_solver` not `backend`. `rng` not `seed`. PCA always writes to `obsm['X_pca']`, `varm['PCs']`, `uns['pca']`.

### `sc.tl.leiden`
```python
def leiden(
    adata: AnnData,
    resolution: float = 1,                      # positional, not keyword!
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    key_added: str = "leiden",
    adjacency: CSBase | None = None,
    directed: bool | None = None,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: type | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    flavor: LeidenFlavor | Default = ...,
) -> AnnData | None
```
**Note**: `resolution` is positional. `rng` not `seed`. Reads neighbors from `adata.obsp['connectivities']` (or `obsp` argument). `key_added` defaults to "leiden". Result goes in `adata.obs['leiden']`.

### `sc.tl.umap`
```python
def umap(
    adata: AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: _InitPos | np.ndarray | None = "spectral",
    rng: SeedLike | RNGLike | None = None,
    a: float | None = None,
    b: float | None = None,
    method: Literal["umap"] = "umap",
    key_added: str | None = None,
    neighbors_key: str = "neighbors",
    copy: bool = False,
) -> AnnData | None
```
**Note**: `init_pos="spectral"` is scanpy's default. **We will accept the user's input** but document that `init_pos="random"` is recommended for reproducibility (cycle 10 lit-scout pitfall #1 — cuML spectral init non-deterministic). Result goes in `adata.obsm['X_umap']`.

### `sc.pp.neighbors` (kNN graph)
```python
def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    knn: bool = True,
    method: Literal["umap", "gauss"] = "umap",
    transformer: ... = None,
    metric: str = "euclidean",
    metric_kwds: dict = ...,
    rng: SeedLike | RNGLike | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None
```
Reads from `adata.obsm['X_pca']` (or `use_rep`). Writes to `adata.obsp['distances']`, `adata.obsp['connectivities']`, `adata.uns['neighbors']`.

### `sc.tl.rank_genes_groups`
```python
def rank_genes_groups(
    adata: AnnData,
    groupby: str,                               # positional!
    *,
    mask_var: ... = ...,
    use_raw: bool | None = None,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = None,
    copy: bool = False,
    method: DETest | Default = ...,            # default in {wilcoxon, t-test, t-test_overestim_var, logreg}
    corr_method: _CorrMethod = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: str | None = None,
) -> AnnData | None
```

## API mapping (singlet-gpu cycle 20)

```python
# python/singlet_gpu/streaming/__init__.py
def run_pipeline(input_paths, *, chunk_cols=100_000, run_lognorm=True, run_hvg=True, run_pca=False, pca_k=50, run_nmf=False, nmf_factors=20, ...):
    """Streaming end-to-end pipeline. Returns a PipelineResult."""

# python/singlet_gpu/pp/neighbors.py
def neighbors(adata, n_neighbors=15, n_pcs=None, *, use_rep=None, ...):
    """GPU-native neighbors. Drop-in for sc.pp.neighbors."""

# python/singlet_gpu/tools/leiden.py
def leiden(adata, resolution=1, *, rng=None, key_added="leiden", ...):
    """GPU-native Leiden via cuGraph. Drop-in for sc.tl.leiden."""

# python/singlet_gpu/tools/umap.py
def umap(adata, *, min_dist=0.5, spread=1.0, n_components=2, ..., init_pos="random", rng=None, ...):
    """GPU-native UMAP via cuML. Default init_pos='random' for reproducibility."""

# python/singlet_gpu/tools/rank_genes_groups.py
def rank_genes_groups(adata, groupby, *, method="wilcoxon", ...):
    """GPU-native DE. Drop-in for sc.tl.rank_genes_groups."""

# python/singlet_gpu/tools/markers.py
def score_genes(adata, gene_list, *, score_name="score", method="mlm", ...):
    """GPU marker scoring via DecoupleR-style mlm/ulm/wsum/UCell."""

def celltypist_predict(adata, model_path, *, key_added="celltypist", ...):
    """GPU CellTypist projection."""
```

## Constraints

- **scanpy parameter name parity for the affecting params**. Cosmetic params (verbose, key_added, obsm, etc.) can be subsetted in cycle 20; full parity in a final cleanup cycle.
- **NO host copies** in any wrapper.
- **rng vs seed**: scanpy uses `rng` (numpy Generator). Our C++ kernels use `uint64_t seed`. Convert: `rng = np.random.default_rng(seed); seed_for_cpp = rng.integers(0, 2**63)`.
- **Result locations match scanpy exactly**: leiden → `adata.obs['leiden']`; umap → `adata.obsm['X_umap']`; neighbors → `adata.obsp['distances']` + `adata.obsp['connectivities']`; rank_genes_groups → `adata.uns['rank_genes_groups']`.

## Test spec (cycle 20)

`python/tests/test_pp_neighbors.py`, `test_tl_leiden.py`, `test_tl_umap.py`, `test_tl_rank_genes_groups.py`, `test_tl_markers.py`, `test_streaming.py`.

Each ~150 LOC. ~5 test cases each. Drop-in replacement test for each (monkey-patch scanpy with our function and run a tiny pipeline).
