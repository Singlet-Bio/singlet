# SPDX-License-Identifier: MIT
"""Export quantification output to .1pz format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def export_to_1pz(
    quant_dir: str | Path,
    output_path: str | Path,
    *,
    sample_id: Optional[str] = None,
) -> bool:
    """Export simpleaf/alevin quantification to .1pz format.

    Reads the count matrix from ``af_quant/alevin/`` and writes a compressed
    .1pz file (VOCSC + zstd-3).

    Parameters
    ----------
    quant_dir : path
        simpleaf output directory (containing ``af_quant/``).
    output_path : path
        Output .1pz file path.
    sample_id : str, optional
        Sample identifier for logging.

    Returns
    -------
    bool
        True if export succeeded.
    """
    import scipy.io
    import scipy.sparse as sp

    qdir = Path(quant_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Find count matrix
    mtx_candidates = [
        qdir / "af_quant" / "alevin" / "quants_mat.mtx",
        qdir / "af_quant" / "alevin" / "quants_mat.mtx.gz",
    ]
    mtx_path = None
    for p in mtx_candidates:
        if p.exists():
            mtx_path = p
            break

    if mtx_path is None:
        logger.error("No count matrix found in %s", qdir)
        return False

    # Load cells × genes matrix
    mat = scipy.io.mmread(mtx_path)
    if sp.issparse(mat):
        mat = mat.tocsc()
    else:
        mat = sp.csc_matrix(mat)

    # Transpose to genes × cells for .1pz container
    mat_gc = mat.T.tocsc()

    import singlepress

    singlepress.write_1pz(str(out), mat_gc)

    sid = sample_id or out.stem
    logger.info(
        "Exported %s: %d genes × %d cells → %s",
        sid,
        mat_gc.shape[0],
        mat_gc.shape[1],
        out,
    )
    return True


# Keep legacy name for backward compatibility
def export_to_spz(
    quant_dir: str | Path,
    output_path: str | Path,
    *,
    sample_id: Optional[str] = None,
    row_sort: bool = True,
) -> bool:
    """Export to .spz format (legacy). Prefer export_to_1pz()."""
    import numpy as np
    import scipy.io
    import scipy.sparse as sp

    qdir = Path(quant_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Find count matrix
    mtx_candidates = [
        qdir / "af_quant" / "alevin" / "quants_mat.mtx",
        qdir / "af_quant" / "alevin" / "quants_mat.mtx.gz",
    ]
    mtx_path = None
    for p in mtx_candidates:
        if p.exists():
            mtx_path = p
            break

    if mtx_path is None:
        logger.error("No count matrix found in %s", qdir)
        return False

    # Load cells × genes matrix
    mat = scipy.io.mmread(mtx_path)
    if sp.issparse(mat):
        mat = mat.tocsc()
    else:
        mat = sp.csc_matrix(mat)

    # Read gene names
    gene_names = []
    for name in ["quants_mat_cols.txt", "features.tsv", "genes.tsv"]:
        gfile = mtx_path.parent / name
        if gfile.exists():
            with open(gfile) as f:
                gene_names = [line.strip().split("\t")[0] for line in f]
            break

    # Read cell barcodes
    barcodes = []
    for name in ["quants_mat_rows.txt", "barcodes.tsv"]:
        bfile = mtx_path.parent / name
        if bfile.exists():
            with open(bfile) as f:
                barcodes = [line.strip() for line in f]
            break

    # Transpose to genes × cells for .spz container
    # mat is cells × genes, so we need mat.T
    mat_gc = mat.T.tocsc()

    from singlet._singlepress import sp_write, sp_write_int

    if np.issubdtype(mat_gc.dtype, np.integer):
        sp_write_int(
            mat_gc.indptr.astype(np.int32),
            mat_gc.indices.astype(np.int32),
            mat_gc.data.astype(np.int32),
            mat_gc.shape[0],
            str(out),
            row_sort=row_sort,
            rownames=gene_names if gene_names else None,
            colnames=barcodes if barcodes else None,
        )
    else:
        sp_write(
            mat_gc.indptr.astype(np.int32),
            mat_gc.indices.astype(np.int32),
            mat_gc.data.astype(np.float64),
            mat_gc.shape[0],
            str(out),
            row_sort=row_sort,
            rownames=gene_names if gene_names else None,
            colnames=barcodes if barcodes else None,
        )

    sid = sample_id or out.stem
    logger.info(
        "Exported %s: %d genes × %d cells → %s",
        sid,
        mat_gc.shape[0],
        mat_gc.shape[1],
        out,
    )
    return True
