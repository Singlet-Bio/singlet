"""Free cell type annotation via NMF gene program projection.

Unlike the token-priced nmf.transform/annotate, these functions run entirely
locally. No API key, no quota, no internet required after initial model download.

The annotation pipeline:
  1. Download species-appropriate W matrix (gene programs) from R2 (cached)
  2. Project: H = NNLS(W, X) — non-negative least squares, runs on any laptop
  3. Map H loadings → cell type labels using reference label mappings
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _models_dir() -> Path:
    d = Path.home() / ".singlet" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_model(organism: str, k: int) -> Path:
    """Download W matrix for an organism from R2 if not cached."""
    import requests

    from singlet._catalog import _R2_BASE

    safe_name = organism.lower().replace(" ", "_")
    filename = f"{safe_name}_k{k}.1pz"
    local_path = _models_dir() / filename

    if local_path.exists():
        return local_path

    url = f"{_R2_BASE}/gene_programs/{filename}"
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot reach model server ({_R2_BASE}). "
            "Check your internet connection, or cache models locally with "
            "'singlet.gene_programs(organism)' while online."
        ) from None
    except requests.HTTPError as e:
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"No gene program model for organism={organism!r}, k={k}. "
                "Available: Homo sapiens, Mus musculus."
            ) from None
        raise RuntimeError(f"Failed to download model: {e}") from e
    local_path.write_bytes(resp.content)
    return local_path


def _detect_organism(adata) -> str:
    """Infer organism from adata metadata."""
    # Check obs columns
    for col in ("organism", "species"):
        if col in adata.obs.columns:
            val = adata.obs[col].mode()
            if len(val) > 0:
                return str(val.iloc[0])

    # Check uns
    if "organism" in adata.uns:
        return str(adata.uns["organism"])

    # Heuristic: infer from gene name capitalization
    # Human genes: ALL CAPS (TP53, BRCA1, EGFR)
    # Mouse genes: Title Case (Tp53, Brca1, Egfr)
    if hasattr(adata, "var_names") and len(adata.var_names) >= 100:

        def _alpha_part(g: str) -> str:
            return "".join(c for c in g if c.isalpha())

        sample = [
            g
            for g in adata.var_names[:500]
            if len(g) >= 3 and g[0].isalpha() and len(_alpha_part(g)) >= 2
        ]
        if len(sample) >= 50:
            upper_frac = sum(1 for g in sample if _alpha_part(g).isupper()) / len(sample)
            if upper_frac > 0.7:
                return "Homo sapiens"
            title_frac = sum(
                1 for g in sample if _alpha_part(g)[0].isupper() and not _alpha_part(g).isupper()
            ) / len(sample)
            if title_frac > 0.7:
                return "Mus musculus"

    raise ValueError(
        "Could not detect organism from adata. Pass organism= explicitly "
        "or ensure adata.obs['organism'] or adata.uns['organism'] is set."
    )


def gene_programs(organism: str, k: int = 100) -> pd.DataFrame:
    """Download and return the NMF gene program dictionary (W matrix).

    This is a free, citable scientific resource. The W matrix contains
    ``k`` gene programs (columns), each defined by a set of gene loadings.

    Parameters
    ----------
    organism : str
        Species name, e.g. ``"Homo sapiens"``, ``"Mus musculus"``.
    k : int
        Number of gene programs. Default 100.

    Returns
    -------
    pandas.DataFrame
        Genes × k matrix of gene program loadings.
        Index: gene names. Columns: program IDs (``"P001"``–``"P{k}"``).

    Examples
    --------
    >>> W = singlet.gene_programs("Homo sapiens")
    >>> W.shape
    (38606, 100)
    >>> W.iloc[:5, :3]  # top 5 genes, first 3 programs
    """
    import pandas as pd

    from singlet._io import read_1pz

    model_path = _download_model(organism, k)
    adata = read_1pz(model_path)

    # W is stored as genes × programs in the 1pz file
    # After read_1pz transpose, it's programs × genes → we need to transpose back
    W = adata.X.T  # type: ignore[union-attr]
    if hasattr(W, "toarray"):
        W = W.toarray()

    gene_names = list(adata.obs_names) if len(adata.obs_names) > 0 else None
    program_names = list(adata.var_names) if len(adata.var_names) > 0 else None

    if program_names is None:
        program_names = [f"P{i + 1:03d}" for i in range(W.shape[1])]
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(W.shape[0])]

    return pd.DataFrame(W, index=gene_names, columns=program_names)  # type: ignore[arg-type]


def project(adata, *, organism: Optional[str] = None, k: int = 100) -> np.ndarray:
    """Project cells onto NMF gene programs. **Free, runs locally.**

    Computes H = NNLS(W, X) where W is the gene program dictionary.
    This maps each cell to a k-dimensional space of biological programs.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells to project. Gene names must overlap with the W matrix.
    organism : str, optional
        Species name. Auto-detected from adata if not provided.
    k : int
        Number of gene programs. Default 100.

    Returns
    -------
    numpy.ndarray
        Cells × k matrix of gene program loadings.

    Examples
    --------
    >>> H = singlet.project(my_adata, organism="Homo sapiens")
    >>> H.shape  # (n_cells, 100)
    """
    import scipy.sparse as sp

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"project() requires an AnnData object, got {type(adata).__name__}")

    if organism is None:
        organism = _detect_organism(adata)

    W_df = gene_programs(organism, k)

    # Align genes: find intersection
    shared_genes = adata.var_names.intersection(W_df.index)
    if len(shared_genes) < 100:
        raise ValueError(
            f"Only {len(shared_genes)} genes overlap between your data and "
            f"the {organism} gene program model. Need at least 100."
        )

    W = W_df.loc[shared_genes].values  # (n_shared_genes, k)
    X = adata[:, shared_genes].X  # (n_cells, n_shared_genes)
    if sp.issparse(X):
        X = X.toarray()

    # Non-negative least squares: solve min ||x - W @ h|| s.t. h >= 0
    from scipy.optimize import nnls

    n_cells = X.shape[0]
    H = np.zeros((n_cells, k), dtype=np.float64)

    for i in range(n_cells):
        H[i], _ = nnls(W, X[i])

    return H


def annotate(adata, *, organism: Optional[str] = None, k: int = 100) -> pd.DataFrame:
    """Annotate cells with types via NMF projection. **Free, runs locally.**

    Projects cells onto gene programs, then maps dominant programs to
    cell type labels using a reference label mapping.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells to annotate.
    organism : str, optional
        Species name. Auto-detected from adata if not provided.
    k : int
        Number of gene programs. Default 100.

    Returns
    -------
    pandas.DataFrame
        One row per cell with columns:
        - ``cell_type``: Predicted cell type label.
        - ``confidence``: Confidence score (0–1).
        - ``top_program``: Dominant gene program ID.
        - ``top_program_loading``: Loading value for the top program.

    Examples
    --------
    >>> annotations = singlet.annotate(my_adata)
    >>> annotations["cell_type"].value_counts()
    """
    import pandas as pd

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"annotate() requires an AnnData object, got {type(adata).__name__}")

    if organism is None:
        organism = _detect_organism(adata)

    H = project(adata, organism=organism, k=k)

    # Load label mapping (program_id → cell_type)
    labels = _load_program_labels(organism, k)

    # For each cell, find the dominant program
    top_program_idx = np.argmax(H, axis=1)
    top_loading = H[np.arange(len(H)), top_program_idx]

    # Normalize confidence: loading / sum of all loadings
    row_sums = H.sum(axis=1)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    confidence = top_loading / row_sums

    # Map to cell type labels
    program_ids = [f"P{i + 1:03d}" for i in top_program_idx]
    cell_types = [labels.get(pid, "unknown") for pid in program_ids]

    return pd.DataFrame(
        {
            "cell_type": cell_types,
            "confidence": confidence,
            "top_program": program_ids,
            "top_program_loading": top_loading,
        },
        index=adata.obs_names,
    )


def _load_program_labels(organism: str, k: int) -> dict:
    """Load the program → cell type label mapping from R2."""
    import json

    import requests

    from singlet._catalog import _R2_BASE

    safe_name = organism.lower().replace(" ", "_")
    filename = f"{safe_name}_k{k}_labels.json"
    local_path = _models_dir() / filename

    if local_path.exists():
        return json.loads(local_path.read_text())

    url = f"{_R2_BASE}/gene_programs/{filename}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        labels = resp.json()
        local_path.write_text(json.dumps(labels))
        return labels
    except Exception:
        # If label mapping not available, return empty → all "unknown"
        return {}
