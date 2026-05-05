"""Token-priced: cross-atlas query and semantic search."""

from __future__ import annotations

from typing import Optional


def query(
    *,
    species: Optional[str | list[str]] = None,
    tissue: Optional[str | list[str]] = None,
    disease: Optional[str | list[str]] = None,
    cell_type: Optional[str | list[str]] = None,
    perturbation: Optional[str] = None,
    developmental_stage: Optional[str] = None,
    sex: Optional[str] = None,
    modality: Optional[str] = None,
    min_cells: int = 0,
    max_results: int = 100_000,
):
    """Query the atlas by structured metadata. **Token-priced feature.**

    Streams matching cells from AWS as a single AnnData object with
    NMF embeddings and standardized annotations.

    Parameters
    ----------
    species : str or list, optional
        Species name(s), e.g. ``"human"`` or ``["human", "mouse"]``.
    tissue : str or list, optional
        Tissue name(s), e.g. ``"lung"`` or ``["lung", "liver"]``.
    disease : str or list, optional
        Disease name(s), e.g. ``"Crohn's disease"``.
    cell_type : str or list, optional
        Cell type(s), e.g. ``"macrophage"``.
    perturbation : str, optional
        Perturbation condition, e.g. ``"anti-TNF"``.
    developmental_stage : str, optional
        Stage filter, e.g. ``"fetal"``, ``"adult"``.
    sex : str, optional
        ``"male"`` or ``"female"``.
    modality : str, optional
        Data modality, e.g. ``"RNA-seq"``, ``"ATAC-seq"``.
    min_cells : int
        Minimum cells per dataset to include.
    max_results : int
        Cap on total cells returned.

    Returns
    -------
    anndata.AnnData
        Merged cells across matching datasets with:
        - ``.X``: raw counts (sparse)
        - ``.obs``: cell_type, disease, tissue, species, accession, ...
        - ``.obsm["X_nmf"]``: NMF embeddings
    """
    import requests

    from singlet._auth import _API_BASE, _headers

    params = {}
    for field, val in [
        ("species", species),
        ("tissue", tissue),
        ("disease", disease),
        ("cell_type", cell_type),
        ("perturbation", perturbation),
        ("developmental_stage", developmental_stage),
        ("sex", sex),
        ("modality", modality),
    ]:
        if val is not None:
            params[field] = val if isinstance(val, str) else ",".join(val)

    params["min_cells"] = str(min_cells)
    params["max_results"] = str(max_results)

    resp = requests.post(
        f"{_API_BASE}/query",
        json=params,
        headers=_headers(),
        timeout=120,
    )
    resp.raise_for_status()

    import tempfile

    from singlet._io import read_spz

    with tempfile.NamedTemporaryFile(suffix=".spz", delete=False) as f:
        f.write(resp.content)
        tmp_path = f.name

    return read_spz(tmp_path)


def search(text: str, max_results: int = 100_000):
    """Natural-language search across the atlas. **Token-priced feature.**

    Parameters
    ----------
    text : str
        Free-text query, e.g. ``"exhausted T cells in pediatric leukemia"``.
    max_results : int
        Cap on total cells returned.

    Returns
    -------
    anndata.AnnData
        Matching cells with metadata and NMF embeddings.
    """
    import requests

    from singlet._auth import _API_BASE, _headers

    resp = requests.post(
        f"{_API_BASE}/search",
        json={"query": text, "max_results": max_results},
        headers=_headers(),
        timeout=120,
    )
    resp.raise_for_status()

    import tempfile

    from singlet._io import read_spz

    with tempfile.NamedTemporaryFile(suffix=".spz", delete=False) as f:
        f.write(resp.content)
        tmp_path = f.name

    return read_spz(tmp_path)
