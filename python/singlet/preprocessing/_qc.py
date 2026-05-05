"""Quality control metrics for single-cell RNA-seq outputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_MIN_CELLS = 50
DEFAULT_MIN_MAPPING_RATE = 0.10
DEFAULT_MIN_GENES_PER_CELL = 200


@dataclass
class QCMetrics:
    """QC metrics for a processed sample."""

    n_cells: int = 0
    n_genes: int = 0
    median_genes_per_cell: int = 0
    median_counts_per_cell: int = 0
    total_counts: int = 0
    mapping_rate: float = 0.0
    pass_qc: bool = False
    qc_status: str = "unknown"  # "pass", "warn", "fail"
    fail_reasons: list = field(default_factory=list)


def _read_mtx_shape(mtx_path: Path) -> tuple:
    """Read MTX header to get shape without loading full matrix."""
    import gzip

    opener = gzip.open if str(mtx_path).endswith(".gz") else open
    with opener(mtx_path, "rt") as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.strip().split()
            return int(parts[0]), int(parts[1]), int(parts[2])
    return (0, 0, 0)


def run_qc(
    quant_dir: str | Path,
    *,
    min_cells: int = DEFAULT_MIN_CELLS,
    min_mapping_rate: float = DEFAULT_MIN_MAPPING_RATE,
    min_genes_per_cell: int = DEFAULT_MIN_GENES_PER_CELL,
) -> QCMetrics:
    """Compute QC metrics from simpleaf output.

    Parameters
    ----------
    quant_dir : path
        simpleaf output directory (containing ``af_quant/``).
    min_cells : int
        Minimum number of cells to pass QC.
    min_mapping_rate : float
        Minimum mapping rate (0-1) to pass QC.
    min_genes_per_cell : int
        Minimum median genes per cell.

    Returns
    -------
    QCMetrics
    """
    import numpy as np
    import scipy.io
    import scipy.sparse as sp

    qdir = Path(quant_dir)

    # Find the count matrix
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
        return QCMetrics(qc_status="fail", fail_reasons=["No count matrix found"])

    # Load matrix (cells × genes)
    mat = scipy.io.mmread(mtx_path)
    if sp.issparse(mat):
        mat = mat.tocsr()
    else:
        mat = sp.csr_matrix(mat)

    n_cells, n_genes = mat.shape
    genes_per_cell = np.diff(mat.indptr)  # nnz per row ≈ genes per cell
    counts_per_cell = np.array(mat.sum(axis=1)).flatten()

    median_genes = int(np.median(genes_per_cell)) if n_cells > 0 else 0
    median_counts = int(np.median(counts_per_cell)) if n_cells > 0 else 0
    total_counts = int(mat.sum())

    # Mapping rate
    mapping_rate = 0.0
    map_info = qdir / "af_quant" / "alevin" / "map_info.json"
    if map_info.exists():
        with open(map_info) as f:
            mi = json.load(f)
        n_proc = mi.get("num_processed", 0)
        n_mapped = mi.get("num_mapped", 0)
        mapping_rate = n_mapped / n_proc if n_proc > 0 else 0.0

    # Apply thresholds
    fail_reasons = []
    if n_cells < min_cells:
        fail_reasons.append(f"Too few cells: {n_cells} < {min_cells}")
    if mapping_rate < min_mapping_rate:
        fail_reasons.append(f"Low mapping rate: {mapping_rate:.1%} < {min_mapping_rate:.0%}")
    if median_genes < min_genes_per_cell:
        fail_reasons.append(f"Low genes/cell: {median_genes} < {min_genes_per_cell}")

    if not fail_reasons:
        qc_status = "pass"
        pass_qc = True
    elif fail_reasons and mapping_rate >= min_mapping_rate:
        qc_status = "warn"
        pass_qc = False
    else:
        qc_status = "fail"
        pass_qc = False

    return QCMetrics(
        n_cells=n_cells,
        n_genes=n_genes,
        median_genes_per_cell=median_genes,
        median_counts_per_cell=median_counts,
        total_counts=total_counts,
        mapping_rate=mapping_rate,
        pass_qc=pass_qc,
        qc_status=qc_status,
        fail_reasons=fail_reasons,
    )
