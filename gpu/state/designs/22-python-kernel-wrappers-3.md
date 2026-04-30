---
feature: python_kernel_wrappers_part_3
roadmap_id: 22
module: python/singlet_gpu/{enrichment,integrate,velocity,lineage,de}/*.py
status: design
tolerance: per-function output element-wise rel_err ≤ 1e-5 vs the C++ direct call
target_perf: pybind11 marshal overhead ≤2% of kernel wall time
ooc_plan: streaming variants where applicable
---

## Cycle 22 scope

Python wrappers for cycles 13–17 kernels — the FINAL cycle of the wrapper sprint:

- Cycle 13 (GSEA + AUCell) → `singlet_gpu.enrichment`
- Cycle 14 (Harmony + BBKNN) → `singlet_gpu.integrate`
- Cycle 15 (Velocity prep) → `singlet_gpu.velocity`
- Cycle 16 (MT lineage) → `singlet_gpu.lineage`
- Cycle 17 (Donor pseudobulk DE) → `singlet_gpu.de` (extending the cycle 19 reduce/de namespace)

After cycle 22, all 17 original kernels have Python wrappers. Cycle 23 starts the R wrapper sprint.

## API mapping (verified by cycle 21 lit-scout)

### `singlet_gpu.enrichment`

```python
# python/singlet_gpu/enrichment/__init__.py
from .gsea import run_gsea
from .aucell import run_aucell

# python/singlet_gpu/enrichment/gsea.py
def run_gsea(
    mat: AnnData | cupy.sparse.csr_matrix,    # or per-gene preranked stats vector
    net: pd.DataFrame,                          # gene-set DataFrame, source/target columns
    *,
    source: str = "source",
    target: str = "target",
    times: int = 1000,                          # max permutations
    min_n: int = 5,                             # min genes per set
    seed: int = 42,
    use_raw: bool = False,
) -> AnnData | pd.DataFrame:
    """GPU-native preranked GSEA via cycle-13 fgsea kernel. decoupleR-compatible."""

# python/singlet_gpu/enrichment/aucell.py
def run_aucell(
    mat: AnnData | cupy.sparse.csr_matrix,
    net: pd.DataFrame,
    *,
    source: str = "source",
    target: str = "target",
    min_n: int = 5,
    seed: int = 42,
    use_raw: bool = False,
) -> AnnData | pd.DataFrame:
    """GPU-native AUCell via cycle-13 aucell kernel. decoupleR-compatible."""
```

Result location: `adata.obsm['X_aucell']` (n_cells × n_pathways) for AUCell; `adata.obs['gsea_norm_es_<set>']` columns for GSEA per pathway.

### `singlet_gpu.integrate`

```python
# python/singlet_gpu/integrate/harmony.py
def harmony_integrate(
    adata: AnnData,
    key: str | list[str],                       # batch column(s)
    *,
    basis: str = "X_pca",                       # input embedding from sc.pp.pca
    adjusted_basis: str = "X_pca_harmony",      # output
    n_clusters: int = 20,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 0,
) -> None:
    """GPU-native Harmony. Drop-in for sc.external.pp.harmony_integrate."""

# python/singlet_gpu/integrate/bbknn.py
def bbknn(
    adata: AnnData,
    *,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    neighbors_within_batch: int = 3,
    n_pcs: int | None = None,
    metric: str = "euclidean",
    set_op_mix_ratio: float = 1.0,
    local_connectivity: int = 1,
    copy: bool = False,
) -> AnnData | None:
    """GPU-native BBKNN. Drop-in for sc.external.pp.bbknn."""
```

### `singlet_gpu.velocity`

```python
# python/singlet_gpu/velocity/__init__.py
from .moments import moments
from .velocity import velocity

# python/singlet_gpu/velocity/moments.py
def moments(
    adata: AnnData,
    *,
    n_neighbors: int = 30,
    n_pcs: int | None = None,
    mode: str = "connectivities",
    use_rep: str | None = None,
    layer_spliced: str = "spliced",
    layer_unspliced: str = "unspliced",
    copy: bool = False,
) -> AnnData | None:
    """GPU-native first/second-order moments smoothing. Drop-in for scvelo.pp.moments."""

# python/singlet_gpu/velocity/velocity.py
def velocity(
    adata: AnnData,
    *,
    mode: str = "steady_state",     # only steady_state supported in cycle 22
    fit_offset: bool = False,
    perc: list[int] = [5, 95],      # outlier percentile
    layer_spliced: str = "spliced",
    layer_unspliced: str = "unspliced",
    copy: bool = False,
) -> AnnData | None:
    """GPU-native steady-state RNA velocity. Drop-in for scvelo.tl.velocity (mode='steady_state' only)."""
```

Note: cycle 15 velocity_prep is steady-state only. Dynamical and stochastic modes are deferred to a future cycle (cycle 24+ Cell2fate replacement).

### `singlet_gpu.lineage`

```python
# python/singlet_gpu/lineage/__init__.py
from .mt import detect_clones

# python/singlet_gpu/lineage/mt.py
def detect_clones(
    adata: AnnData,
    *,
    alt_layer: str = "mt_alt",
    depth_layer: str = "mt_depth",
    min_depth: int = 10,
    min_cells_alt: int = 5,
    min_vaf: float = 0.01,
    min_K: int = 2,
    max_K: int = 10,
    seed: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """GPU MT heteroplasmy clone calling. NEW (no scanpy/MQuad equivalent that takes AnnData)."""
```

Result: `adata.obs['mt_clone_id']` (int per cell), `adata.obsm['mt_heteroplasmy']` (n_cells × n_informative_sites).

This wraps cycle 16 `anno/mt_lineage.h`. The function is a NEW addition to the scanpy ecosystem.

### `singlet_gpu.de` (extending cycle 19 reduce/de — actually new namespace)

```python
# python/singlet_gpu/de/__init__.py
from .pseudobulk import pseudobulk_de

# python/singlet_gpu/de/pseudobulk.py
def pseudobulk_de(
    adata: AnnData,
    *,
    sample_col: str = "donor_id",       # cell-level donor labels
    groupby: str = "cell_type",          # cell-level cluster labels
    mode: str = "sum",                   # sum aggregation
    min_cells_per_pseudobulk: int = 10,
    apeglm_shrinkage: bool = True,
    seed: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """GPU-native donor-aware pseudobulk DE via cycle-17 NB GLM kernel. decoupler-compatible."""
```

Result: `adata.uns['donor_pseudobulk']` containing per-gene per-cluster LFC, p-values, q-values, dispersions.

## Module layout (cycle 22)

```
python/singlet_gpu/
├── enrichment/
│   ├── __init__.py
│   ├── gsea.py
│   └── aucell.py
├── integrate/
│   ├── __init__.py
│   ├── harmony.py
│   └── bbknn.py
├── velocity/
│   ├── __init__.py
│   ├── moments.py
│   └── velocity.py
├── lineage/
│   ├── __init__.py
│   └── mt.py
├── de/
│   ├── __init__.py
│   └── pseudobulk.py
```

## Constraints

- scanpy / decoupleR / scvelo parameter name parity for the affecting params.
- NO host copies in the wrapper paths.
- `apeglm_shrinkage` is a singlet-gpu addition (DESeq2 has it built-in via the `apeglm` shrinkage estimator).
- AUCell + GSEA take pandas DataFrames as gene-set input (decoupleR convention).
- `pseudobulk_de` accepts donor labels via `sample_col` from `adata.obs` — the user is responsible for loading singlify's `donor_assignments.tsv` into `adata.obs['donor_id']` (we provide a helper in `singlet_gpu.io.load_donor_assignments(path) -> pd.Series`).

## Test spec (cycle 22)

5 test files (one per module). ~150 LOC each. ~4 cases each. Reuse cycle 18-21 fixtures.
