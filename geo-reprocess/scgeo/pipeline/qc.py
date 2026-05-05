"""
Quality control module for single-cell RNA-seq outputs.

Computes QC metrics from quantification outputs:
- Cell counts
- Gene counts per cell
- UMI counts per cell
- Mapping rates
- Mitochondrial percentages
- Quality thresholds
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from scipy.io import mmread
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


@dataclass
class QCMetrics:
    """Quality control metrics for a sample.
    
    Attributes:
        n_cells: Number of cells detected
        n_genes: Number of genes detected
        median_genes_per_cell: Median genes per cell
        median_counts_per_cell: Median UMI counts per cell
        total_counts: Total UMI counts
        mapping_rate: Mapping rate (0-1)
        frac_mt: Mitochondrial fraction (0-1)
        frac_unspliced: Unspliced fraction (RNA velocity specific)
        pass_qc: Whether sample passes QC thresholds
        qc_status: QC status ("pass", "warn", "fail")
        fail_reasons: List of reasons for QC failure
    """
    n_cells: int = 0
    n_genes: int = 0
    median_genes_per_cell: int = 0
    median_counts_per_cell: int = 0
    total_counts: int = 0
    mapping_rate: float = 0.0
    frac_mt: float = 0.0
    frac_unspliced: float = 0.0
    pass_qc: bool = False
    qc_status: str = "unknown"
    fail_reasons: list = None
    
    def __post_init__(self):
        if self.fail_reasons is None:
            self.fail_reasons = []


def read_mtx_matrix(mtx_path: Path) -> Optional[np.ndarray]:
    """Read sparse MTX matrix file.
    
    Args:
        mtx_path: Path to MTX file (.mtx or .mtx.gz)
        
    Returns:
        Scipy sparse matrix or None if failed
    """
    try:
        matrix = mmread(mtx_path)
        return matrix
    except Exception as e:
        logger.error(f"Failed to read MTX matrix {mtx_path}: {e}")
        return None


def compute_basic_metrics(mtx_path: Path) -> Dict[str, float]:
    """Compute basic metrics from count matrix.
    
    Args:
        mtx_path: Path to MTX file (genes × cells)
        
    Returns:
        Dictionary with n_cells, n_genes, median_genes, median_counts, total_counts
    """
    matrix = read_mtx_matrix(mtx_path)
    if matrix is None:
        return {}
    
    # alevin-fry quants_mat.mtx is cells × genes (rows=cells, cols=genes)
    # Convert to CSR for efficient row operations (per-cell)
    if hasattr(matrix, 'tocsr'):
        matrix = matrix.tocsr()
    
    n_cells, n_genes = matrix.shape
    
    # Compute per-cell statistics (sum across genes for each cell = axis=1)
    genes_per_cell = np.array((matrix > 0).sum(axis=1)).flatten()
    counts_per_cell = np.array(matrix.sum(axis=1)).flatten()
    
    median_genes = int(np.median(genes_per_cell)) if len(genes_per_cell) > 0 else 0
    median_counts = int(np.median(counts_per_cell)) if len(counts_per_cell) > 0 else 0
    total_counts = int(matrix.sum())
    
    return {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "median_genes_per_cell": median_genes,
        "median_counts_per_cell": median_counts,
        "total_counts": total_counts,
    }


def compute_qc_simpleaf(output_dir: Path, config, protocol: str = "") -> QCMetrics:
    """Compute QC metrics from simpleaf output.
    
    Args:
        output_dir: simpleaf output directory
        config: Configuration object with QC thresholds
        protocol: Protocol name (e.g., "seqwell") for protocol-aware thresholds
        
    Returns:
        QCMetrics object
    """
    metrics = QCMetrics()
    fail_reasons = []
    
    # Find quantification output
    af_quant_dir = output_dir / "af_quant"
    if not af_quant_dir.exists():
        metrics.qc_status = "fail"
        metrics.fail_reasons = ["simpleaf output directory not found"]
        return metrics
    
    # Read count matrix
    mtx_path = af_quant_dir / "alevin" / "quants_mat.mtx"
    if not mtx_path.exists():
        # Try alternate location
        mtx_path = af_quant_dir / "alevin" / "quants_mat.mtx.gz"
    
    if not mtx_path.exists():
        metrics.qc_status = "fail"
        metrics.fail_reasons = ["Count matrix not found"]
        return metrics
    
    # Compute basic metrics
    basic = compute_basic_metrics(mtx_path)
    if not basic:
        metrics.qc_status = "fail"
        metrics.fail_reasons = ["Failed to read count matrix"]
        return metrics
    
    metrics.n_cells = basic["n_cells"]
    metrics.n_genes = basic["n_genes"]
    metrics.median_genes_per_cell = basic["median_genes_per_cell"]
    metrics.median_counts_per_cell = basic["median_counts_per_cell"]
    metrics.total_counts = basic["total_counts"]
    
    # Read mapping rate from piscem map_info.json or quant.json
    import json
    
    # piscem writes map_info.json in af_map/ (parent of af_quant)
    af_map_dir = af_quant_dir.parent / "af_map"
    map_info_path = af_map_dir / "map_info.json"
    if map_info_path.exists():
        try:
            with open(map_info_path) as f:
                map_info = json.load(f)
                n_reads = map_info.get("num_reads", 0)
                num_mapped = map_info.get("num_mapped", 0)
                if n_reads > 0:
                    metrics.mapping_rate = num_mapped / n_reads
        except Exception as e:
            logger.warning(f"Failed to parse map_info.json: {e}")
    
    # Fallback: legacy meta_info.json (salmon-based pipelines)
    if metrics.mapping_rate == 0.0:
        meta_path = af_quant_dir / "meta_info.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                    metrics.mapping_rate = meta.get("mapping_rate", 0.0)
            except Exception as e:
                logger.warning(f"Failed to parse meta_info.json: {e}")
    
    # Apply QC thresholds
    # All sample-level metrics are now WARNINGS only — no hard gates.
    # Downstream consumers filter individual cells at query/merge time.
    # We record QC metrics for catalog reconciliation and downstream filtering.
    
    if metrics.mapping_rate < config.qc.min_mapping_rate:
        fail_reasons.append(f"Low mapping rate ({metrics.mapping_rate:.2%} < {config.qc.min_mapping_rate:.2%})")
    
    # Log cell count and genes/cell as metadata — not used as sample-level gates.
    from scgeo.config.protocols import LOW_SENSITIVITY_PROTOCOLS
    if protocol.lower() in LOW_SENSITIVITY_PROTOCOLS:
        genes_threshold = config.qc.low_sensitivity_min_genes_per_cell
    else:
        genes_threshold = config.qc.min_genes_per_cell
    
    warn_reasons = []
    if metrics.n_cells < config.qc.min_cells:
        warn_reasons.append(f"Low cell count ({metrics.n_cells} < {config.qc.min_cells})")
    if metrics.median_genes_per_cell < genes_threshold:
        warn_reasons.append(f"Low median genes per cell ({metrics.median_genes_per_cell} < {genes_threshold})")
    
    # Determine QC status
    if not fail_reasons and not warn_reasons:
        metrics.qc_status = "pass"
        metrics.pass_qc = True
    elif not fail_reasons and warn_reasons:
        # Low genes/cell is a warning, not a failure — data is still exported
        metrics.qc_status = "warn"
        metrics.pass_qc = True
    elif len(fail_reasons) == 1 and "mapping rate" in fail_reasons[0].lower():
        metrics.qc_status = "warn"  # Mapping rate alone is a warning
        metrics.pass_qc = True
    else:
        metrics.qc_status = "fail"
    
    metrics.fail_reasons = fail_reasons + warn_reasons
    
    return metrics


def check_qc_thresholds(metrics: QCMetrics, config, protocol: str = "") -> Tuple[bool, str]:
    """Check if metrics pass QC thresholds.
    
    Args:
        metrics: QCMetrics object
        config: Configuration object with thresholds
        protocol: Protocol name for protocol-aware thresholds
        
    Returns:
        (pass_qc, status_string)
        status_string is "pass", "warn", or "fail"
    """
    if metrics.qc_status == "pass":
        return True, "pass"
    elif metrics.qc_status == "warn":
        return True, "warn"
    else:
        return False, "fail"
