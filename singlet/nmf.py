"""NMF model serving — project data onto interpretable biological programs."""

from __future__ import annotations

from typing import Optional


def transform(adata, *, model: str = "human_global_k100"):
    """Project cells onto NMF factors. **Token-priced feature.**

    Adds ``.obsm["X_nmf"]`` to the AnnData in-place.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells to project.
    model : str
        Model name, e.g. ``"human_global_k100"``.
    """
    from singlet._auth import _headers, _API_BASE

    import requests
    import numpy as np

    genes = list(adata.var_names)
    resp = requests.post(
        f"{_API_BASE}/nmf/transform",
        json={"model": model, "genes": genes, "n_cells": adata.n_obs},
        headers=_headers(),
        timeout=120,
    )
    resp.raise_for_status()

    adata.obsm["X_nmf"] = np.array(resp.json()["loadings"])


def annotate(adata, *, model: str = "human_global_k100") -> dict:
    """Annotate NMF factors with biological programs. **Token-priced feature.**

    Parameters
    ----------
    adata : anndata.AnnData
        Must have ``.obsm["X_nmf"]`` (call ``transform`` first).
    model : str
        Model used for factor annotations.

    Returns
    -------
    dict
        Mapping from factor index to annotation string, e.g.
        ``{0: "ventricular cardiomyocyte — repolarization", ...}``.
    """
    from singlet._auth import _headers, _API_BASE

    import requests

    resp = requests.get(
        f"{_API_BASE}/nmf/annotations",
        params={"model": model},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()

    return resp.json()["annotations"]
