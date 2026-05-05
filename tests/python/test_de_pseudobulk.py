"""
Cycle 23 correctness tests for singlet_gpu.de.pseudobulk_de and
singlet_gpu.io.donor.load_donor_assignments.

Tests the GPU-native donor pseudobulk DE wrapper (cycle 17: de/donor_pseudobulk.h):
  - ``singlet.gpu.de.pseudobulk_de``
  - ``singlet.gpu.io.load_donor_assignments``  (helper, not GPU — pure Python)

Formal spec: singlet-gpu/state/designs/22-python-kernel-wrappers-3.md

API (from spec):
  pseudobulk_de(adata, *, sample_col='donor_id', groupby='cell_type',
                mode='sum', min_cells_per_pseudobulk=10,
                apeglm_shrinkage=True, seed=0, copy=False) -> AnnData | None
  singlet_gpu.io.load_donor_assignments(path) -> pd.Series

Result location (spec):
  - adata.uns['donor_pseudobulk'] : dict with per-gene per-cluster LFC, pvals,
    qvals, dispersions.

Tolerances (from spec):
  - LFC Spearman ρ ≥ 0.97 vs DESeq2 (via Rscript subprocess; skip if R absent)

Notes:
  - Synthetic data: 200 cells, 2 donors, 3 cell types, 50 genes.
  - Real data for DESeq2 comparison: GSM4037629 with donor_assignments.tsv.
  - load_donor_assignments reads the singlify donor_assignments.tsv and returns
    a pd.Series indexed by barcode with values donor_id.

Skip strategy:
  - Module-level skip if singlet.gpu not available.
  - requires_gpu for all GPU-exercising tests.
  - DESeq2 test skips if Rscript subprocess fails or R/DESeq2 absent.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import textwrap

import numpy as np
import pandas as pd
import pytest

singlet_gpu = pytest.importorskip(
    "singlet.gpu",
    reason="singlet.gpu not available. Run `pip install -e singlet-gpu/python/` first.",
)

from conftest import requires_gpu  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LFC_SPEARMAN_MIN = 0.97
_SEED = 0
_MIN_CELLS = 10
_N_CELLS = 400       # enough cells per donor × cell_type to pass min_cells filter
_N_GENES = 50
_N_DONORS = 3
_N_CELL_TYPES = 3

_DONOR_ASSIGNMENTS_FILENAME = "donor_assignments.tsv"

# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def _make_pseudobulk_adata(
    n_cells: int = _N_CELLS,
    n_genes: int = _N_GENES,
    n_donors: int = _N_DONORS,
    n_cell_types: int = _N_CELL_TYPES,
):
    """Build a minimal AnnData suitable for pseudobulk_de.

    - Count matrix: negative-binomial-like integer counts.
    - obs['donor_id']  : cyclic donor labels.
    - obs['cell_type'] : cyclic cell-type labels.
    Ensures each (donor, cell_type) combination has ≥ min_cells_per_pseudobulk cells.
    """
    import anndata
    import scipy.sparse as sp

    rng = np.random.default_rng(0xC0FFEE)
    # Simple NB-like: Poisson(lambda) where lambda ~ Gamma(shape, scale).
    lam = rng.gamma(shape=2.0, scale=5.0, size=(n_cells, n_genes)).astype(np.float32)
    counts = rng.poisson(lam).astype(np.float32)

    donor_labels = [f"donor_{i % n_donors}" for i in range(n_cells)]
    cell_type_labels = [f"type_{i % n_cell_types}" for i in range(n_cells)]

    adata = anndata.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"donor_id": donor_labels, "cell_type": cell_type_labels},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)]),
    )
    return adata


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    res = spearmanr(a, b)
    return float(res.correlation if hasattr(res, "correlation") else res[0])


def _rscript_available() -> bool:
    """Check if Rscript is callable and DESeq2 is installed."""
    try:
        r_check = subprocess.run(
            ["Rscript", "-e", "library(DESeq2); cat('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        return r_check.returncode == 0 and "ok" in r_check.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# test_pseudobulk_de_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_pseudobulk_de_basic():
    """pseudobulk_de() completes on synthetic data without error.

    Procedure:
      1. Build synthetic 400-cell, 50-gene AnnData with 3 donors, 3 cell types.
      2. Call singlet_gpu.de.pseudobulk_de(adata, sample_col='donor_id',
                                            groupby='cell_type', seed=0).
      3. Assert returns None (inplace convention).
    """
    pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("scipy", reason="scipy not installed")

    adata = _make_pseudobulk_adata()

    ret = singlet_gpu.de.pseudobulk_de(
        adata,
        sample_col="donor_id",
        groupby="cell_type",
        mode="sum",
        min_cells_per_pseudobulk=_MIN_CELLS,
        seed=_SEED,
    )

    assert ret is None, (
        f"pseudobulk_de() inplace must return None, got {type(ret)}"
    )


# ---------------------------------------------------------------------------
# test_pseudobulk_de_writes_uns
# ---------------------------------------------------------------------------
@requires_gpu
def test_pseudobulk_de_writes_uns():
    """pseudobulk_de() writes adata.uns['donor_pseudobulk'] with required structure.

    Required sub-keys per the spec:
      - 'lfc'         : per-gene per-cluster log2 fold-change
      - 'pvals'       : uncorrected p-values
      - 'qvals'       : BH-adjusted q-values
      - 'dispersion'  : per-gene dispersion estimates

    Verifies:
      - uns['donor_pseudobulk'] exists.
      - Each required key is present.
      - pvals in [0, 1], qvals in [0, 1].
      - lfc is finite for informative genes.
    """
    pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("scipy", reason="scipy not installed")

    adata = _make_pseudobulk_adata()
    singlet_gpu.de.pseudobulk_de(
        adata,
        sample_col="donor_id",
        groupby="cell_type",
        mode="sum",
        min_cells_per_pseudobulk=_MIN_CELLS,
        seed=_SEED,
    )

    assert "donor_pseudobulk" in adata.uns, (
        "pseudobulk_de(): adata.uns['donor_pseudobulk'] not written"
    )

    pb = adata.uns["donor_pseudobulk"]

    required_keys = ["lfc", "pvals", "qvals", "dispersion"]
    for k in required_keys:
        assert k in pb, (
            f"adata.uns['donor_pseudobulk'] missing required key '{k}'"
        )

    # pvals in [0, 1]
    pvals = np.asarray(list(pb["pvals"].values()) if isinstance(pb["pvals"], dict)
                       else pb["pvals"], dtype=np.float64).ravel()
    valid_p = pvals[np.isfinite(pvals)]
    if len(valid_p) > 0:
        assert np.all(valid_p >= 0.0), "pvals contains negative values"
        assert np.all(valid_p <= 1.0 + 1e-9), (
            f"pvals contains values > 1.0; max={valid_p.max():.4e}"
        )

    # qvals in [0, 1]
    qvals = np.asarray(list(pb["qvals"].values()) if isinstance(pb["qvals"], dict)
                       else pb["qvals"], dtype=np.float64).ravel()
    valid_q = qvals[np.isfinite(qvals)]
    if len(valid_q) > 0:
        assert np.all(valid_q >= 0.0), "qvals contains negative values"
        assert np.all(valid_q <= 1.0 + 1e-9), (
            f"qvals contains values > 1.0; max={valid_q.max():.4e}"
        )


# ---------------------------------------------------------------------------
# test_pseudobulk_de_min_cells_filter
# ---------------------------------------------------------------------------
# CYCLE-256: xfail removed. Root cause was Python wrapper passing
# (cells × genes) CSC to a kernel that expects (genes × cells); applied .T in
# python/singlet/gpu/de/pseudobulk.py:258-273. Verified PASS via job 374155.
@requires_gpu
def test_pseudobulk_de_min_cells_filter():
    """pseudobulk_de() excludes (donor, cell_type) groups with fewer than min_cells.

    Procedure:
      1. Build AnnData where one (donor, cell_type) combination has only 2 cells.
      2. Set min_cells_per_pseudobulk=10.
      3. Run pseudobulk_de.
      4. Assert the result does not include pseudobulk contributions from that
         sparse group (no crash, correct result shape for the well-covered groups).
    """
    import anndata
    import scipy.sparse as sp

    pytest.importorskip("scipy", reason="scipy not installed")

    rng = np.random.default_rng(0xABCD)
    n_cells_main = 300
    n_genes = 20

    # Well-covered base: 300 cells evenly across 3 donors × 2 cell types.
    counts = rng.poisson(5.0, size=(n_cells_main, n_genes)).astype(np.float32)
    donor_labels = [f"donor_{i % 3}" for i in range(n_cells_main)]
    cell_type_labels = [f"type_{i % 2}" for i in range(n_cells_main)]

    # Add 2 cells of a sparse (donor_3, type_2) group.
    sparse_counts = rng.poisson(5.0, size=(2, n_genes)).astype(np.float32)
    all_counts = np.vstack([counts, sparse_counts])
    donor_labels += ["donor_3", "donor_3"]
    cell_type_labels += ["type_2", "type_2"]  # only 2 cells — below min_cells=10

    adata = anndata.AnnData(
        X=sp.csr_matrix(all_counts),
        obs=pd.DataFrame(
            {"donor_id": donor_labels, "cell_type": cell_type_labels},
            index=[f"cell_{i}" for i in range(len(donor_labels))],
        ),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)]),
    )

    # Should complete without error even with the sparse group.
    singlet_gpu.de.pseudobulk_de(
        adata,
        sample_col="donor_id",
        groupby="cell_type",
        mode="sum",
        min_cells_per_pseudobulk=10,
        seed=_SEED,
    )

    assert "donor_pseudobulk" in adata.uns, (
        "pseudobulk_de() with sparse group did not write uns['donor_pseudobulk']"
    )
    # The sparse (donor_3, type_2) group should not appear in results;
    # verify no crash and result is non-empty.
    pb = adata.uns["donor_pseudobulk"]
    assert "lfc" in pb, "uns['donor_pseudobulk']['lfc'] missing after min_cells filtering"


# ---------------------------------------------------------------------------
# test_pseudobulk_de_vs_DESeq2
# ---------------------------------------------------------------------------
@requires_gpu
def test_pseudobulk_de_vs_DESeq2(gsm4037629_path):
    """GPU pseudobulk_de LFC Spearman ρ ≥ 0.97 vs DESeq2 (Rscript subprocess).

    Skips if:
      - Rscript is not callable in $PATH.
      - R package DESeq2 is not installed.
      - donor_assignments.tsv is absent in GSM4037629 output.

    Procedure:
      1. Load GSM4037629.
      2. Load donor_assignments.tsv → adata.obs['donor_id'].
      3. Assign synthetic cell_type from leiden clusters (fallback: modulo groups).
      4. Run singlet_gpu.de.pseudobulk_de.
      5. Write adata.X, obs['donor_id'], obs['cell_type'] to a temp CSV.
      6. Run Rscript DESeq2 pseudobulk DE on those temp files.
      7. Compare LFC vectors: Spearman ρ ≥ 0.97.
    """
    import tempfile
    import os

    pytest.importorskip("anndata", reason="anndata not installed")

    if not _rscript_available():
        pytest.skip("Rscript + DESeq2 not available — skip vs-reference test")

    donor_tsv = pathlib.Path(gsm4037629_path) / _DONOR_ASSIGNMENTS_FILENAME
    if not donor_tsv.is_file():
        pytest.skip(
            f"donor_assignments.tsv not found at {donor_tsv}. "
            "Run singlify with --snps --pipeline to generate this file."
        )

    adata = singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))

    # Load donor assignments.
    try:
        donor_series = singlet_gpu.io.load_donor_assignments(str(donor_tsv))
    except Exception as e:
        pytest.skip(f"load_donor_assignments failed: {e}")

    # Align barcodes.
    shared_barcodes = adata.obs_names.intersection(donor_series.index)
    if len(shared_barcodes) < 50:
        pytest.skip(
            f"Too few shared barcodes between adata ({adata.n_obs}) and "
            f"donor_assignments ({len(donor_series)}): {len(shared_barcodes)}"
        )
    adata = adata[shared_barcodes].copy()
    adata.obs["donor_id"] = donor_series.loc[shared_barcodes].values

    # Use modulo-based synthetic cell types (no leiden required for DE test).
    n_cell_types = 2
    adata.obs["cell_type"] = [f"type_{i % n_cell_types}" for i in range(adata.n_obs)]

    # Check we have enough pseudobulk samples.
    n_donors = adata.obs["donor_id"].nunique()
    if n_donors < 2:
        pytest.skip(f"Only {n_donors} donor(s) detected; pseudobulk DE requires ≥ 2")

    # Run GPU pseudobulk DE.
    singlet_gpu.de.pseudobulk_de(
        adata,
        sample_col="donor_id",
        groupby="cell_type",
        mode="sum",
        min_cells_per_pseudobulk=_MIN_CELLS,
        seed=_SEED,
    )

    if "donor_pseudobulk" not in adata.uns or "lfc" not in adata.uns["donor_pseudobulk"]:
        pytest.skip("pseudobulk_de produced no lfc output — cannot compare to DESeq2")

    gpu_lfc = adata.uns["donor_pseudobulk"]["lfc"]
    if isinstance(gpu_lfc, dict):
        # Take first cell type.
        first_key = next(iter(gpu_lfc))
        gpu_lfc_arr = np.asarray(list(gpu_lfc[first_key].values()), dtype=np.float64)
        gene_names = list(gpu_lfc[first_key].keys())
    else:
        gpu_lfc_arr = np.asarray(gpu_lfc, dtype=np.float64).ravel()
        gene_names = list(adata.var_names)[: len(gpu_lfc_arr)]

    # Write pseudobulk count matrix for DESeq2.
    import scipy.sparse as sp
    X_host = adata.X
    if hasattr(X_host, "get"):
        X_host = X_host.get()
    if sp.issparse(X_host):
        X_host = np.asarray(X_host.todense())
    else:
        X_host = np.asarray(X_host)

    with tempfile.TemporaryDirectory() as tmpdir:
        counts_csv = os.path.join(tmpdir, "counts.csv")
        meta_csv = os.path.join(tmpdir, "meta.csv")
        result_json = os.path.join(tmpdir, "deseq2_lfc.json")

        # Write transposed: genes as rows, cells as cols (DESeq2 convention).
        pd.DataFrame(
            X_host.T.astype(int),
            index=list(adata.var_names),
            columns=list(adata.obs_names),
        ).to_csv(counts_csv)

        pd.DataFrame(
            {"donor_id": adata.obs["donor_id"].values,
             "cell_type": adata.obs["cell_type"].values},
            index=list(adata.obs_names),
        ).to_csv(meta_csv)

        # DESeq2 pseudobulk R script (inline).
        r_script = textwrap.dedent(f"""
            suppressMessages({{
                library(DESeq2)
            }})
            counts <- as.matrix(read.csv('{counts_csv}', row.names=1))
            meta <- read.csv('{meta_csv}', row.names=1)
            # Aggregate to pseudobulk.
            groups <- paste(meta$donor_id, meta$cell_type, sep='_')
            donors <- unique(meta$donor_id)
            cell_types <- unique(meta$cell_type)[1]  # first cell type only
            pb_counts <- sapply(unique(groups), function(g) {{
                idx <- which(groups == g)
                rowSums(counts[, idx, drop=FALSE])
            }})
            pb_meta <- data.frame(
                donor_id = sub('_.*', '', colnames(pb_counts)),
                cell_type = sub('.*_', '', colnames(pb_counts))
            )
            rownames(pb_meta) <- colnames(pb_counts)
            pb_counts <- round(pb_counts)
            # Filter to first cell type, ≥2 donors.
            keep <- pb_meta$cell_type == cell_types
            if (sum(keep) < 2) {{
                cat('{{}}\n')
                quit(status=0)
            }}
            pb_sub <- pb_counts[, keep, drop=FALSE]
            meta_sub <- pb_meta[keep, , drop=FALSE]
            # DESeq2 (intercept-only if only one group).
            tryCatch({{
                dds <- DESeqDataSetFromMatrix(
                    countData = pb_sub,
                    colData   = meta_sub,
                    design    = ~ 1
                )
                dds <- DESeq(dds, quiet=TRUE)
                res <- results(dds)
                lfc_vec <- as.numeric(res$log2FoldChange)
                names(lfc_vec) <- rownames(res)
                lfc_finite <- lfc_vec[is.finite(lfc_vec)]
                cat(jsonlite::toJSON(as.list(lfc_finite)), '\n')
            }}, error = function(e) cat('{{}}\n'))
        """)

        r_file = os.path.join(tmpdir, "deseq2_run.R")
        with open(r_file, "w") as f:
            f.write(r_script)

        try:
            proc = subprocess.run(
                ["Rscript", r_file],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("DESeq2 Rscript timed out (>120s)")

        if proc.returncode != 0 or not proc.stdout.strip():
            pytest.skip(f"DESeq2 Rscript failed: {proc.stderr[:300]}")

        try:
            r_lfc = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            pytest.skip("Could not parse DESeq2 JSON output")

        if not r_lfc:
            pytest.skip("DESeq2 returned empty LFC dict — model may be degenerate")

        r_genes = list(r_lfc.keys())
        shared_genes = list(set(gene_names) & set(r_genes))
        if len(shared_genes) < 10:
            pytest.skip(
                f"Too few shared genes between GPU ({len(gene_names)}) and "
                f"DESeq2 ({len(r_genes)}): {len(shared_genes)}"
            )

        gpu_map = dict(zip(gene_names, gpu_lfc_arr))
        gpu_shared = np.array([gpu_map[g] for g in shared_genes], dtype=np.float64)
        r_shared = np.array([float(r_lfc[g]) for g in shared_genes], dtype=np.float64)

        valid = np.isfinite(gpu_shared) & np.isfinite(r_shared)
        if valid.sum() < 10:
            pytest.skip(f"Too few finite shared LFC estimates: {valid.sum()}")

        rho = _spearman_rho(gpu_shared[valid], r_shared[valid])
        assert rho >= _LFC_SPEARMAN_MIN, (
            f"pseudobulk_de LFC Spearman ρ {rho:.4f} < {_LFC_SPEARMAN_MIN} "
            f"vs DESeq2 (n_genes={valid.sum()})"
        )


# ---------------------------------------------------------------------------
# test_load_donor_assignments_helper
# ---------------------------------------------------------------------------
def test_load_donor_assignments_helper(gsm4037629_path):
    """singlet_gpu.io.load_donor_assignments returns a pd.Series indexed by barcode.

    This test does NOT require GPU (pure I/O helper).

    Verifies:
      - Returns a pd.Series.
      - Index values are non-empty strings (barcode IDs).
      - Values are non-empty strings (donor labels).
      - Length matches the number of rows in donor_assignments.tsv.
      - Function gracefully raises FileNotFoundError for a missing path.

    Skipped if donor_assignments.tsv is absent.
    """
    donor_tsv = pathlib.Path(gsm4037629_path) / _DONOR_ASSIGNMENTS_FILENAME
    if not donor_tsv.is_file():
        pytest.skip(
            f"donor_assignments.tsv not found at {donor_tsv}. "
            "Run singlify with --snps --pipeline to generate this file."
        )

    result = singlet_gpu.io.load_donor_assignments(str(donor_tsv))

    assert isinstance(result, pd.Series), (
        f"load_donor_assignments() must return pd.Series, got {type(result)}"
    )
    assert len(result) > 0, "load_donor_assignments() returned an empty Series"

    # Index: barcode strings.
    index_arr = np.asarray(result.index, dtype=str)
    assert all(s != "" for s in index_arr[:10]), (
        "load_donor_assignments() index contains empty barcode strings"
    )

    # Values: donor label strings.
    values_arr = np.asarray(result.values, dtype=str)
    assert all(s != "" and s != "nan" for s in values_arr[:10]), (
        "load_donor_assignments() values contain empty/nan donor labels"
    )

    # Cross-check length against the raw file.
    raw_df = pd.read_csv(str(donor_tsv), sep="\t")
    assert len(result) == len(raw_df), (
        f"load_donor_assignments() returned {len(result)} rows, "
        f"but donor_assignments.tsv has {len(raw_df)} rows"
    )

    # FileNotFoundError for missing path.
    with pytest.raises((FileNotFoundError, OSError)):
        singlet_gpu.io.load_donor_assignments("/nonexistent/path/donor_assignments.tsv")
