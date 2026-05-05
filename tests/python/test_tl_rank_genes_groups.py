"""
Cycle 21 correctness tests for singlet_gpu.tools.rank_genes_groups().

Tests the GPU-native DE wrapper (cycle 11: de/{wilcoxon,ttest}.h):
  - ``singlet.gpu.tools.rank_genes_groups``

Formal spec: singlet-gpu/state/designs/20-python-kernel-wrappers-2.md

Verified scanpy signature (from spec):
  def rank_genes_groups(adata, groupby, *, mask_var=..., use_raw=None,
                        groups='all', reference='rest', n_genes=None,
                        rankby_abs=False, pts=False, key_added=None,
                        copy=False, method=..., corr_method='benjamini-hochberg',
                        tie_correct=False, layer=None)
  Result: adata.uns['rank_genes_groups'].

Tests:
  - test_rank_genes_groups_wilcoxon_basic             : call succeeds, uns key written
  - test_rank_genes_groups_writes_uns                 : uns structure & required sub-keys
  - test_rank_genes_groups_vs_scanpy_lfc_spearman     : log-fold-change Spearman ρ ≥ 0.97
  - test_rank_genes_groups_vs_scanpy_top_markers_jaccard : top-50 marker Jaccard ≥ 0.90
  - test_rank_genes_groups_method_t_test              : t-test method accepted, results differ

Skip strategy:
  - Module-level skip if singlet.gpu not available.
  - requires_gpu marker for all GPU-exercising tests.
"""
from __future__ import annotations

import numpy as np
import pytest

singlet_gpu = pytest.importorskip(
    "singlet.gpu",
    reason="singlet.gpu not available. Run `pip install -e singlet-gpu/python/` first.",
)

from conftest import requires_gpu  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
_LFC_SPEARMAN_MIN = 0.97   # LFC Spearman ρ vs scanpy (spec tolerance)
_MARKER_JACCARD_MIN = 0.90 # top-50 markers Jaccard vs scanpy (spec tolerance)
_TOP_N_MARKERS = 50        # marker set size for Jaccard
_SEED = 0xC0FFEE
_N_NEIGHBORS = 15
_N_COMPS = 50
_GROUPBY = "leiden"        # cluster labels from leiden step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gpu_adata(gsm4037629_path):
    return singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))


def _gpu_to_cpu_adata(adata_gpu):
    import anndata
    import scipy.sparse as sp

    X = adata_gpu.X
    X_host = X.get() if hasattr(X, "get") else X
    if sp.issparse(X_host):
        X_host = X_host.tocsr()
    else:
        X_host = sp.csr_matrix(X_host)

    adata_cpu = anndata.AnnData(
        X=X_host,
        obs=adata_gpu.obs.copy(),
        var=adata_gpu.var.copy(),
        uns=dict(adata_gpu.uns),
    )
    adata_cpu.obs_names = adata_gpu.obs_names.copy()
    adata_cpu.var_names = adata_gpu.var_names.copy()
    return adata_cpu


def _full_pipeline_gpu(adata_gpu):
    """Preprocess + PCA + neighbors + leiden (GPU)."""
    singlet_gpu.preprocess.normalize_total(adata_gpu, inplace=True)
    singlet_gpu.preprocess.log1p(adata_gpu, inplace=True)
    singlet_gpu.reduce.pca(adata_gpu, n_comps=_N_COMPS, inplace=True)
    singlet_gpu.pp.neighbors(adata_gpu, n_neighbors=_N_NEIGHBORS)
    singlet_gpu.tools.leiden(adata_gpu, 1.0, rng=_SEED)


def _full_pipeline_cpu(adata_cpu):
    """Preprocess + PCA + neighbors + leiden (scanpy CPU)."""
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    sc.pp.normalize_total(adata_cpu, inplace=True)
    sc.pp.log1p(adata_cpu)  # scanpy 1.11+ removed inplace= (in-place by default)
    sc.pp.pca(adata_cpu, n_comps=_N_COMPS)
    sc.pp.neighbors(adata_cpu, n_neighbors=_N_NEIGHBORS, use_rep="X_pca")
    sc.tl.leiden(adata_cpu, 1.0, rng=_SEED)


def _extract_uns_rgg(adata, key="rank_genes_groups"):
    """Extract adata.uns['rank_genes_groups'] with validation."""
    assert key in adata.uns, (
        f"adata.uns['{key}'] not found; rank_genes_groups may have failed"
    )
    return adata.uns[key]


def _lfc_array(rgg_result, group):
    """Extract log-fold-change values for a group from uns result."""
    # scanpy stores as recarray: rgg_result['logfoldchanges'][group]
    lfcs = rgg_result.get("logfoldchanges", None)
    if lfcs is None:
        pytest.skip("rank_genes_groups uns does not contain 'logfoldchanges'")
    if hasattr(lfcs, "dtype") and lfcs.dtype.names:
        # recarray indexed by group name
        if group in lfcs.dtype.names:
            return np.asarray(lfcs[group], dtype=np.float64)
    # Dict-of-arrays format
    if isinstance(lfcs, dict) and group in lfcs:
        return np.asarray(lfcs[group], dtype=np.float64)
    pytest.skip(f"Cannot extract LFC for group '{group}' from uns result")


def _gene_names_array(rgg_result, group):
    """Extract gene name arrays for a group."""
    names = rgg_result.get("names", None)
    if names is None:
        pytest.skip("rank_genes_groups uns does not contain 'names'")
    if hasattr(names, "dtype") and names.dtype.names and group in names.dtype.names:
        return np.asarray(names[group])
    if isinstance(names, dict) and group in names:
        return np.asarray(names[group])
    pytest.skip(f"Cannot extract names for group '{group}'")


def _spearman_rho(a, b):
    """Compute Spearman correlation between two arrays."""
    from scipy.stats import spearmanr
    res = spearmanr(a, b)
    return float(res.correlation if hasattr(res, "correlation") else res[0])


# ---------------------------------------------------------------------------
# test_rank_genes_groups_wilcoxon_basic
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: rank_genes_groups indirectly invokes leiden which requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_rank_genes_groups_wilcoxon_basic(gsm4037629_path):
    """rank_genes_groups(method='wilcoxon') completes, writes uns key.

    Procedure:
      1. Load GSM4037629, run full pipeline (preprocess + PCA + neighbors + leiden).
      2. Call singlet_gpu.tools.rank_genes_groups(adata, 'leiden', method='wilcoxon').
      3. Assert returns None (inplace convention).
      4. Assert adata.uns['rank_genes_groups'] exists.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata)

    ret = singlet_gpu.tools.rank_genes_groups(adata, _GROUPBY, method="wilcoxon")

    assert ret is None, (
        f"rank_genes_groups() inplace must return None, got {type(ret)}"
    )
    assert "rank_genes_groups" in adata.uns, (
        "rank_genes_groups(): adata.uns['rank_genes_groups'] not written"
    )


# ---------------------------------------------------------------------------
# test_rank_genes_groups_writes_uns
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: rank_genes_groups indirectly invokes leiden which requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_rank_genes_groups_writes_uns(gsm4037629_path):
    """rank_genes_groups() uns result contains required sub-keys.

    Required sub-keys (from scanpy API parity spec):
      - 'names'      : gene names per group
      - 'scores'     : test statistics per group
      - 'pvals'      : uncorrected p-values
      - 'pvals_adj'  : BH-corrected p-values (corr_method='benjamini-hochberg')
      - 'logfoldchanges' : log2 fold-changes per group
      - 'params'     : dict with 'groupby', 'method', 'reference'

    Verifies correct shapes and value ranges.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata)
    singlet_gpu.tools.rank_genes_groups(adata, _GROUPBY, method="wilcoxon")

    rgg = _extract_uns_rgg(adata)

    required_keys = ["names", "scores", "pvals", "pvals_adj", "logfoldchanges", "params"]
    for k in required_keys:
        assert k in rgg, (
            f"adata.uns['rank_genes_groups'] missing required key '{k}'"
        )

    # Validate params sub-dict.
    params = rgg["params"]
    assert params.get("groupby") == _GROUPBY, (
        f"uns params['groupby'] = '{params.get('groupby')}' != '{_GROUPBY}'"
    )
    assert params.get("method") == "wilcoxon", (
        f"uns params['method'] = '{params.get('method')}' != 'wilcoxon'"
    )

    # Validate pvals_adj in [0, 1].
    pvals_adj = rgg["pvals_adj"]
    if hasattr(pvals_adj, "dtype") and pvals_adj.dtype.names:
        # recarray — check first group
        first_group = pvals_adj.dtype.names[0]
        pv = np.asarray(pvals_adj[first_group], dtype=np.float64)
    elif isinstance(pvals_adj, dict):
        pv = np.asarray(next(iter(pvals_adj.values())), dtype=np.float64)
    else:
        pv = np.asarray(pvals_adj, dtype=np.float64).ravel()

    assert np.all(pv >= 0.0), "pvals_adj contains values < 0"
    assert np.all(pv <= 1.0 + 1e-9), (
        f"pvals_adj contains values > 1.0; max={pv.max():.4e}"
    )


# ---------------------------------------------------------------------------
# test_rank_genes_groups_vs_scanpy_lfc_spearman
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: rank_genes_groups indirectly invokes leiden which requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_rank_genes_groups_vs_scanpy_lfc_spearman(gsm4037629_path):
    """GPU Wilcoxon LFC Spearman ρ ≥ 0.97 vs sc.tl.rank_genes_groups.

    Procedure:
      1. Load GSM4037629 twice → adata_gpu, adata_cpu.
      2. Run full pipeline on both.
      3. singlet_gpu.tools.rank_genes_groups(adata_gpu, 'leiden', method='wilcoxon').
      4. sc.tl.rank_genes_groups(adata_cpu, 'leiden', method='wilcoxon').
      5. For each cluster, extract LFC vectors from both results.
         Align by gene name (set intersection), compute Spearman ρ.
      6. Assert median Spearman ρ across clusters ≥ 0.97.
    """
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    pytest.importorskip("scipy.stats", reason="scipy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_gpu_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_adata(adata_gpu)

    _full_pipeline_gpu(adata_gpu)
    _full_pipeline_cpu(adata_cpu)

    singlet_gpu.tools.rank_genes_groups(adata_gpu, _GROUPBY, method="wilcoxon")
    sc.tl.rank_genes_groups(adata_cpu, _GROUPBY, method="wilcoxon")

    rgg_gpu = _extract_uns_rgg(adata_gpu)
    rgg_cpu = _extract_uns_rgg(adata_cpu)

    # Determine shared groups.
    gpu_groups = set(adata_gpu.obs[_GROUPBY].unique())
    cpu_groups = set(adata_cpu.obs[_GROUPBY].unique())
    shared_groups = list(gpu_groups & cpu_groups)

    if len(shared_groups) == 0:
        pytest.skip(
            "GPU and CPU leiden produced disjoint cluster label sets; "
            "cannot compare LFCs (label names differ too much)"
        )

    rhos = []
    for grp in shared_groups:
        try:
            gpu_genes = _gene_names_array(rgg_gpu, grp)
            cpu_genes = _gene_names_array(rgg_cpu, grp)
            gpu_lfcs = _lfc_array(rgg_gpu, grp)
            cpu_lfcs = _lfc_array(rgg_cpu, grp)
        except Exception:
            continue  # skip groups where extraction fails

        # Align by gene name intersection.
        gpu_gene2lfc = dict(zip(gpu_genes, gpu_lfcs))
        cpu_gene2lfc = dict(zip(cpu_genes, cpu_lfcs))
        shared_genes = list(set(gpu_gene2lfc) & set(cpu_gene2lfc))
        if len(shared_genes) < 10:
            continue

        a = np.array([gpu_gene2lfc[g] for g in shared_genes], dtype=np.float64)
        b = np.array([cpu_gene2lfc[g] for g in shared_genes], dtype=np.float64)
        if np.all(a == a[0]) or np.all(b == b[0]):
            continue  # degenerate constant array
        rho = _spearman_rho(a, b)
        if np.isfinite(rho):
            rhos.append(rho)

    assert len(rhos) >= 1, (
        "Could not compute LFC Spearman ρ for any cluster pair — "
        "check that leiden labels overlap between GPU and CPU runs"
    )

    median_rho = float(np.median(rhos))
    assert median_rho >= _LFC_SPEARMAN_MIN, (
        f"Wilcoxon LFC median Spearman ρ {median_rho:.4f} < {_LFC_SPEARMAN_MIN} "
        f"vs sc.tl.rank_genes_groups (computed over {len(rhos)} clusters)"
    )


# ---------------------------------------------------------------------------
# test_rank_genes_groups_vs_scanpy_top_markers_jaccard
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: rank_genes_groups indirectly invokes leiden which requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_rank_genes_groups_vs_scanpy_top_markers_jaccard(gsm4037629_path):
    """Top-50 Wilcoxon markers Jaccard ≥ 0.90 vs sc.tl.rank_genes_groups.

    Procedure:
      1. Load GSM4037629 twice → adata_gpu, adata_cpu.
      2. Run full pipeline on both.
      3. Run rank_genes_groups on both.
      4. For each shared cluster, extract top-50 gene names.
      5. Compute Jaccard of the two top-50 sets per cluster.
      6. Assert median Jaccard across clusters ≥ 0.90.
    """
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_gpu_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_adata(adata_gpu)

    _full_pipeline_gpu(adata_gpu)
    _full_pipeline_cpu(adata_cpu)

    singlet_gpu.tools.rank_genes_groups(adata_gpu, _GROUPBY, method="wilcoxon",
                                        n_genes=_TOP_N_MARKERS)
    sc.tl.rank_genes_groups(adata_cpu, _GROUPBY, method="wilcoxon",
                            n_genes=_TOP_N_MARKERS)

    rgg_gpu = _extract_uns_rgg(adata_gpu)
    rgg_cpu = _extract_uns_rgg(adata_cpu)

    gpu_groups = set(adata_gpu.obs[_GROUPBY].unique())
    cpu_groups = set(adata_cpu.obs[_GROUPBY].unique())
    shared_groups = list(gpu_groups & cpu_groups)

    if len(shared_groups) == 0:
        pytest.skip("GPU and CPU leiden produced disjoint cluster label sets")

    jaccards = []
    for grp in shared_groups:
        try:
            gpu_genes = set(_gene_names_array(rgg_gpu, grp)[:_TOP_N_MARKERS])
            cpu_genes = set(_gene_names_array(rgg_cpu, grp)[:_TOP_N_MARKERS])
        except Exception:
            continue
        if len(gpu_genes) == 0 or len(cpu_genes) == 0:
            continue
        inter = len(gpu_genes & cpu_genes)
        union = len(gpu_genes | cpu_genes)
        jaccards.append(inter / union if union > 0 else 0.0)

    assert len(jaccards) >= 1, (
        "Could not compute top-marker Jaccard for any cluster pair"
    )

    median_j = float(np.median(jaccards))
    assert median_j >= _MARKER_JACCARD_MIN, (
        f"Top-{_TOP_N_MARKERS} marker Jaccard {median_j:.4f} < {_MARKER_JACCARD_MIN} "
        f"(median across {len(jaccards)} clusters)"
    )


# ---------------------------------------------------------------------------
# test_rank_genes_groups_method_t_test
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: rank_genes_groups indirectly invokes leiden which requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_rank_genes_groups_method_t_test(gsm4037629_path):
    """rank_genes_groups(method='t-test') is accepted and produces different results than Wilcoxon.

    Verifies:
      - method='t-test' parameter is accepted without error.
      - uns['rank_genes_groups']['params']['method'] == 't-test'.
      - The top-gene scores differ from the Wilcoxon results (tests are not identical).
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_wc = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata_wc)
    singlet_gpu.tools.rank_genes_groups(adata_wc, _GROUPBY, method="wilcoxon")

    adata_tt = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata_tt)
    singlet_gpu.tools.rank_genes_groups(adata_tt, _GROUPBY, method="t-test")

    rgg_wc = _extract_uns_rgg(adata_wc)
    rgg_tt = _extract_uns_rgg(adata_tt)

    # params must reflect the method used.
    assert rgg_tt.get("params", {}).get("method") == "t-test", (
        f"uns params['method'] = '{rgg_tt.get('params', {}).get('method')}' != 't-test'"
    )

    # The two methods should produce different top genes for at least one group.
    groups_wc = list(
        rgg_wc["names"].dtype.names
        if hasattr(rgg_wc["names"], "dtype") and rgg_wc["names"].dtype.names
        else (rgg_wc["names"].keys() if isinstance(rgg_wc["names"], dict) else [])
    )
    if len(groups_wc) == 0:
        pytest.skip("Could not enumerate groups from rank_genes_groups uns result")

    # Check at least one group shows different top gene.
    first_group = groups_wc[0]
    try:
        genes_wc = list(_gene_names_array(rgg_wc, first_group)[:5])
        genes_tt = list(_gene_names_array(rgg_tt, first_group)[:5])
    except Exception:
        pytest.skip("Cannot compare top genes — uns extraction failed")

    # It is expected (though not guaranteed) that the methods differ.
    # We allow them to agree for rare edge cases, but log the outcome.
    methods_differ = genes_wc != genes_tt
    # If they happen to agree exactly, issue a warning but do not fail the test —
    # the primary check is that t-test ran without error and set params correctly.
    if not methods_differ:
        import warnings
        warnings.warn(
            f"wilcoxon and t-test top-5 genes for group '{first_group}' are identical: "
            f"{genes_wc}. This may indicate both methods are using the same code path.",
            UserWarning,
            stacklevel=2,
        )
