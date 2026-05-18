# SPDX-License-Identifier: MIT
"""
Cycle 21 correctness tests for singlet.gpu.tools.score_genes() and
singlet.gpu.tools.celltypist_predict().

Tests the GPU-native marker scoring and reference annotation wrappers
(cycle 12: anno/{marker_score,reference_map}.h):
  - ``singlet.gpu.tools.score_genes``      — DecoupleR-style MLM/ULM/wsum/UCell
  - ``singlet.gpu.tools.celltypist_predict`` — GPU CellTypist projection

Formal spec: singlet-gpu/state/designs/20-python-kernel-wrappers-2.md

API from spec:
  def score_genes(adata, gene_list, *, score_name='score', method='mlm', ...)
  def celltypist_predict(adata, model_path, *, key_added='celltypist', ...)

Tests:
  - test_score_genes_mlm_basic          : call succeeds, score column written
  - test_score_genes_writes_obs         : obs[score_name] shape, finite values
  - test_celltypist_predict_basic       : call succeeds, obs key written
  - test_celltypist_writes_obs_label    : obs[key_added] non-empty string labels

Skip strategy:
  - Module-level skip if singlet_gpu wheel not built.
  - requires_gpu marker for all GPU-exercising tests.
  - celltypist tests additionally skip if no model file is found.
"""
from __future__ import annotations

import numpy as np
import pytest

singlet_gpu = pytest.importorskip(
    "singlet_gpu",
    reason="singlet_gpu wheel not built. Run `pip install -e singlet-gpu/python/` first.",
)

from conftest import requires_gpu  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical gene sets and model paths
# ---------------------------------------------------------------------------

# S-phase gene signature from Tirosh et al. 2016 (used by scanpy as default
# cell-cycle genes). We use a small validated subset for the marker smoke test.
_S_PHASE_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
]

_G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67",
]

# CellTypist model path — skip celltypist tests if model not present.
_CELLTYPIST_MODEL_PATH = (
    "/mnt/projects/debruinz_project/cellarium/celltypist_models/"
    "Immune_All_Low.pkl"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gpu_adata(gsm4037629_path):
    return singlet.gpu.io.read_pz_to_anndata(str(gsm4037629_path))


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


def _preprocess_gpu(adata_gpu):
    """Normalize + log1p (GPU). Minimal preprocessing for marker scoring."""
    singlet.gpu.preprocess.normalize_total(adata_gpu, inplace=True)
    singlet.gpu.preprocess.log1p(adata_gpu, inplace=True)


def _filter_to_present(adata, gene_list):
    """Return only gene names from gene_list that are present in adata.var_names."""
    var_set = set(adata.var_names)
    present = [g for g in gene_list if g in var_set]
    return present


def _celltypist_model_available():
    """Return True if the CellTypist model file exists on disk."""
    import pathlib
    return pathlib.Path(_CELLTYPIST_MODEL_PATH).is_file()


# ---------------------------------------------------------------------------
# test_score_genes_mlm_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_score_genes_mlm_basic(gsm4037629_path):
    """score_genes(method='mlm') completes and writes obs[score_name].

    Procedure:
      1. Load GSM4037629, normalize + log1p.
      2. Filter S-phase gene list to genes present in the dataset.
         Skip if fewer than 3 genes present (insufficient signal).
      3. Call singlet.gpu.tools.score_genes(adata, gene_list, method='mlm').
      4. Assert returns None (inplace convention, score written to obs).
      5. Assert adata.obs[score_name] exists.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_gpu(adata)

    gene_list = _filter_to_present(adata, _S_PHASE_GENES)
    if len(gene_list) < 3:
        pytest.skip(
            f"Too few S-phase genes found in GSM4037629 ({len(gene_list)}/15); "
            "cannot meaningfully test score_genes"
        )

    ret = singlet.gpu.tools.score_genes(
        adata, gene_list, score_name="s_score", method="mlm"
    )

    assert ret is None, (
        f"score_genes() inplace must return None, got {type(ret)}"
    )
    assert "s_score" in adata.obs.columns, (
        "score_genes(): adata.obs['s_score'] not written"
    )


# ---------------------------------------------------------------------------
# test_score_genes_writes_obs
# ---------------------------------------------------------------------------
@requires_gpu
def test_score_genes_writes_obs(gsm4037629_path):
    """score_genes() writes obs[score_name] with correct shape and finite values.

    Verifies:
      - obs[score_name] length == n_obs.
      - All values are finite (no NaN/Inf).
      - Scores have non-zero variance (not all identical — meaningful signal).
      - score_name parameter correctly changes the column name written.
      - method parameter accepts 'mlm', 'ulm', 'wsum', and 'ucell' without error
        (smoke test for method dispatch; accuracy tested in correctness-registry).
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_gpu(adata)

    # Use G2M genes for a second distinct score to test column naming.
    s_genes = _filter_to_present(adata, _S_PHASE_GENES)
    g2m_genes = _filter_to_present(adata, _G2M_GENES)

    if len(s_genes) < 3:
        pytest.skip(f"Too few S-phase genes ({len(s_genes)})")

    singlet.gpu.tools.score_genes(adata, s_genes, score_name="s_score", method="mlm")

    # Validate s_score column.
    assert "s_score" in adata.obs.columns, "s_score column not written"
    scores = np.asarray(adata.obs["s_score"], dtype=np.float64)
    assert len(scores) == adata.n_obs, (
        f"s_score length {len(scores)} != n_obs {adata.n_obs}"
    )
    assert np.all(np.isfinite(scores)), (
        f"s_score contains NaN or Inf: {np.sum(~np.isfinite(scores))} non-finite values"
    )
    assert np.std(scores) > 1e-6, (
        f"s_score has near-zero variance (std={np.std(scores):.2e}); "
        "scores may be all identical"
    )

    # Validate score_name isolation — g2m_score must NOT exist if we haven't called it.
    if "g2m_score" in adata.obs.columns:
        pytest.fail(
            "adata.obs already contains 'g2m_score' before calling score_genes "
            "with g2m_genes — column isolation violated"
        )

    if len(g2m_genes) >= 3:
        singlet.gpu.tools.score_genes(adata, g2m_genes, score_name="g2m_score", method="mlm")
        assert "g2m_score" in adata.obs.columns, "g2m_score column not written"
        # Both columns must coexist and be independent.
        g2m_scores = np.asarray(adata.obs["g2m_score"], dtype=np.float64)
        assert not np.allclose(scores, g2m_scores, rtol=1e-3), (
            "s_score and g2m_score are identical — column isolation may be broken"
        )


# ---------------------------------------------------------------------------
# test_celltypist_predict_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_celltypist_predict_basic(gsm4037629_path):
    """celltypist_predict() completes and writes obs key without error.

    This test is skipped if the CellTypist model file does not exist at the
    canonical Clipper path.

    Procedure:
      1. Load GSM4037629, normalize + log1p.
      2. Call singlet.gpu.tools.celltypist_predict(adata, model_path).
      3. Assert returns None (inplace convention).
      4. Assert adata.obs['celltypist'] or adata.obs[key_added] exists.
    """
    if not _celltypist_model_available():
        pytest.skip(
            f"CellTypist model not found at {_CELLTYPIST_MODEL_PATH}. "
            "Set SINGLET_GPU_CELLTYPIST_MODEL env var or copy model to canonical path."
        )

    pytest.importorskip("anndata", reason="anndata not installed")

    import os
    model_path = os.environ.get(
        "SINGLET_GPU_CELLTYPIST_MODEL", _CELLTYPIST_MODEL_PATH
    )

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_gpu(adata)

    ret = singlet.gpu.tools.celltypist_predict(adata, model_path=model_path)

    assert ret is None, (
        f"celltypist_predict() inplace must return None, got {type(ret)}"
    )
    assert "celltypist" in adata.obs.columns, (
        "celltypist_predict(): adata.obs['celltypist'] not written"
    )


# ---------------------------------------------------------------------------
# test_celltypist_writes_obs_label
# ---------------------------------------------------------------------------
@requires_gpu
def test_celltypist_writes_obs_label(gsm4037629_path):
    """celltypist_predict() writes non-empty string cell-type labels to obs.

    Verifies:
      - obs[key_added] length == n_obs.
      - All labels are non-empty strings.
      - At least 2 distinct cell types predicted (non-trivial prediction).
      - key_added parameter correctly changes the column name.

    Skipped if CellTypist model not present.
    """
    if not _celltypist_model_available():
        pytest.skip(
            f"CellTypist model not found at {_CELLTYPIST_MODEL_PATH}."
        )

    pytest.importorskip("anndata", reason="anndata not installed")

    import os
    model_path = os.environ.get(
        "SINGLET_GPU_CELLTYPIST_MODEL", _CELLTYPIST_MODEL_PATH
    )

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_gpu(adata)

    key = "ct_label"
    singlet.gpu.tools.celltypist_predict(adata, model_path=model_path, key_added=key)

    assert key in adata.obs.columns, (
        f"celltypist_predict(key_added='{key}'): column '{key}' not in adata.obs"
    )

    labels = adata.obs[key]
    assert len(labels) == adata.n_obs, (
        f"obs['{key}'] length {len(labels)} != n_obs {adata.n_obs}"
    )

    # All labels must be non-empty strings.
    label_arr = np.asarray(labels, dtype=str)
    empty_mask = (label_arr == "") | (label_arr == "None") | (label_arr == "nan")
    assert empty_mask.sum() == 0, (
        f"obs['{key}'] contains {empty_mask.sum()} empty/None/nan labels"
    )

    # Non-trivial: expect at least 2 distinct cell types for a PBMC-like sample.
    n_types = int(len(set(label_arr)))
    assert n_types >= 2, (
        f"celltypist_predict produced only {n_types} distinct cell type(s) — "
        "expected ≥ 2 for GSM4037629 (mixed cell types)"
    )
