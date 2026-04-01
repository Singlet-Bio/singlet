"""Load datasets from Zenodo (free) or AWS (token-priced streaming)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

_ZENODO_BASE = "https://zenodo.org/records/XXXXXX/files"
_AWS_BASE = "https://data.singletdb.com/v1"


def _cache_dir() -> Path:
    d = Path.home() / ".singlet" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(
    accession: str,
    output_dir: Optional[str | Path] = None,
    force: bool = False,
    source: str = "zenodo",
) -> Path:
    """Download a dataset file.

    Parameters
    ----------
    accession : str
        GEO series accession (e.g. "GSE136831").
    output_dir : path, optional
        Where to save. Defaults to ``~/.singlet/data/``.
    force : bool
        Re-download even if cached.
    source : str
        ``"zenodo"`` (free, deep archive) or ``"aws"`` (fast streaming,
        costs tokens proportional to data size).

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    import requests
    from tqdm import tqdm

    if source not in ("zenodo", "aws"):
        raise ValueError(f"source must be 'zenodo' or 'aws', got {source!r}")

    dest_dir = Path(output_dir) if output_dir else _cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{accession}.spz"

    if dest.exists() and not force:
        return dest

    if source == "aws":
        from singlet._auth import _get_headers
        url = f"{_AWS_BASE}/{accession}"
        headers = _get_headers()
    else:
        url = f"{_ZENODO_BASE}/{accession}.spz"
        headers = {}

    resp = requests.get(url, stream=True, timeout=60, headers=headers)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=accession
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def load(
    source: str | Path,
    *,
    layers: Optional[Sequence[str]] = None,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    backend: str = "zenodo",
):
    """Load a dataset as AnnData.

    This is the primary entry point. Pass a GEO accession to download from
    Zenodo (free) or AWS (token-priced), or a local path to read directly.

    Parameters
    ----------
    source : str or Path
        GEO accession (e.g. "GSE136831") or path to a local file.
        Supports ``.spz``, ``.h5ad``, and ``.zarr`` formats.
    layers : list of str, optional
        Which layers to include. Default: all available.
    genes : list of str, optional
        Subset to these gene names (column slice).
    obs_filter : dict, optional
        Filter cells by obs columns, e.g. ``{"tissue": "lung"}``.
    backend : str
        ``"zenodo"`` (default, free) or ``"aws"`` (fast streaming,
        costs tokens). The ``source`` keyword in the public API maps
        to this parameter.

    Returns
    -------
    anndata.AnnData
        Count matrix with layers, obs metadata, and var gene names.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE136831")
    >>> adata = singlet.load("GSE136831", backend="aws")
    >>> adata = singlet.load("GSE136831", genes=["TP53", "BRCA1"])
    >>> adata = singlet.load("local/file.h5ad")
    """
    path = Path(source)

    # If it's a local file, read it directly
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".h5ad":
            import anndata as ad
            adata = ad.read_h5ad(path)
        elif suffix == ".zarr":
            import anndata as ad
            adata = ad.read_zarr(path)
        else:
            from singlet._io import read_spz
            adata = read_spz(path)
    else:
        # Treat as accession — download
        from singlet._io import read_spz
        local = download(str(source), source=backend)
        adata = read_spz(local)

    # Gene subset
    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        adata = adata[:, gene_mask].copy()

    # Obs filter
    if obs_filter is not None:
        mask = True
        for col, val in obs_filter.items():
            if isinstance(val, list):
                mask = mask & adata.obs[col].isin(val)
            else:
                mask = mask & (adata.obs[col] == val)
        adata = adata[mask].copy()

    # Layer selection
    if layers is not None:
        drop = [k for k in list(adata.layers) if k not in layers]
        for k in drop:
            del adata.layers[k]

    return adata
