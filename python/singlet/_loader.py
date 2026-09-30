# SPDX-License-Identifier: MIT
"""Load Singlet datasets as AnnData.

Users work with two things only: ``.singlet`` files and GEO accession strings
(``GSE…`` / ``GSM…``). Public per-GSE data is hosted at::

    https://data.singlet.bio/data/<GSE>/<GSE>.singlet

(free, no authentication). A ``.singlet`` file opens via
:class:`singlet.SingletBundle`. The host is configurable with
``$SINGLET_DATA_BASE``: the base URL such that files live at
``<base>/data/<GSE>/<GSE>.singlet`` (default ``https://data.singlet.bio``).
A value that already ends in ``/data`` (the form the R client documents) is
accepted too.

Resolution priority for :func:`load`:
  1. A local ``.singlet`` file path (also accepts ``.h5ad`` / ``.zarr`` you exported)
  2. A local catalog directory (cluster use: SINGLET_CATALOG_DIR or singlet.set_catalog_dir())
  3. A free, cached download of the public ``.singlet`` bundle for an accession

GSM accessions are resolved to their parent GSE (via the public REST API at
``https://singlet.bio/api``); the parent bundle is downloaded and the returned
AnnData is filtered to that GSM's cells.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata

# Public REST API base (used only to resolve a GSM → its parent GSE).
_API_BASE_DEFAULT = "https://singlet.bio/api"
_DATA_BASE_DEFAULT = "https://data.singlet.bio"
_USER_AGENT = "singlet-loader/1"


def _api_base() -> str:
    """Base URL for the public REST API. Override with ``$SINGLET_API_BASE``."""
    return os.environ.get("SINGLET_API_BASE", _API_BASE_DEFAULT).rstrip("/")


def _cache_dir() -> Path:
    """Local download cache. Override with ``$SINGLET_CACHE_DIR``.

    Defaults to ``~/.singlet/cache``.
    """
    env = os.environ.get("SINGLET_CACHE_DIR")
    d = Path(env) if env else (Path.home() / ".singlet" / "cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _data_base() -> str:
    """Bundle host: files live at ``<base>/data/<GSE>/<GSE>.singlet``.

    Override with ``$SINGLET_DATA_BASE``. Both ``https://host`` and
    ``https://host/data`` are accepted, so the same value works for the
    Python and R clients.
    """
    base = os.environ.get("SINGLET_DATA_BASE", "").strip() or _DATA_BASE_DEFAULT
    base = base.rstrip("/")
    if base.endswith("/data"):
        base = base[: -len("/data")]
    return base


def _bundle_url(accession: str) -> str:
    """Public R2 URL for a GSE's ``.singlet`` bundle."""
    return f"{_data_base()}/data/{accession}/{accession}.singlet"


def _read_pz_record(path: str) -> dict:
    """Decode a ``.1pz`` with the compiled codec (imported only when needed)."""
    from singlet._pz import read_1pz as _native_read

    return _native_read(path)


def _resolve_gsm_to_gse(gsm_id: str) -> Optional[str]:
    """Look up the parent GSE accession for a GSM via the public REST API.

    Returns the GSE id, or ``None`` if it cannot be resolved.
    """
    base = _api_base()
    # Primary: /api/gsm?q=<GSM> (the /api/gsm/<id> detail endpoint is currently
    # unreliable). Fallback: /api/search?q=<GSM>.
    candidates = (
        f"{base}/gsm?q={gsm_id}&limit=5",
        f"{base}/search?q={gsm_id}",
    )
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
        except Exception:  # noqa: BLE001 — try the next endpoint
            continue
        rows = payload.get("data") or payload.get("gsm") or []
        for row in rows:
            if row.get("gsm_id") == gsm_id and row.get("gse_id"):
                return row["gse_id"]
        # If exactly one row came back, trust it even without an exact id match.
        if len(rows) == 1 and rows[0].get("gse_id"):
            return rows[0]["gse_id"]
    return None


def _resolve_gse_path(accession: str) -> Optional[Path]:
    """Resolve a GSE accession to a local .1pz path via the catalog."""
    from singlet._catalog import _get_catalog_dir, _load_catalog

    cat_dir = _get_catalog_dir()
    if cat_dir is None:
        return None

    cat = _load_catalog()
    rows = cat[cat["gse_id"] == accession]
    if rows.empty:
        return None

    row = rows.iloc[0]
    rel_path = row.get("path", "")
    if not rel_path:
        return None

    # path is relative like "pipeline/quant/GSE264667"
    # cat_dir is the catalog root (e.g. /mnt/projects/.../cellarium)
    base = cat_dir.parent  # go up from catalog/ to cellarium/
    gse_dir = base / rel_path

    counts_path = gse_dir / "counts.1pz"
    if counts_path.exists():
        return counts_path

    return None


def download(
    accession: str,
    output_dir: Optional[str | Path] = None,
    force: bool = False,
) -> Path:
    """Download a GSE's public ``.singlet`` bundle from Cloudflare R2.

    Fetches ``https://data.singlet.bio/data/<GSE>/<GSE>.singlet`` (free, no
    auth) to the local cache. The download is path-cached: an existing,
    non-empty file is reused unless ``force=True``.

    Parameters
    ----------
    accession : str
        GEO series accession (e.g. "GSE138867").
    output_dir : path, optional
        Where to save. Defaults to ``default_cache_dir()``
        (``$SINGLET_CACHE_DIR`` or ``~/.singlet/cache``).
    force : bool
        Re-download even if cached.

    Returns
    -------
    Path
        Path to the downloaded ``.singlet`` bundle.
    """
    dest_dir = Path(output_dir) if output_dir else _cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{accession}.singlet"

    # On-disk cache: reuse an existing, non-empty bundle.
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    url = _bundle_url(accession)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    # Write to a temp file first, then atomic rename (interrupted downloads
    # don't corrupt the cache).
    tmp_dest = dest.with_suffix(".singlet.part")
    try:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment]

        import shutil

        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("content-length", 0))
            if tqdm is not None and total:
                with (
                    open(tmp_dest, "wb") as f,
                    tqdm(total=total, unit="B", unit_scale=True, desc=accession) as pbar,
                ):
                    for chunk in iter(lambda: resp.read(1 << 16), b""):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                with open(tmp_dest, "wb") as f:
                    shutil.copyfileobj(resp, f, length=1 << 16)
        tmp_dest.replace(dest)
    except urllib.error.HTTPError as e:
        tmp_dest.unlink(missing_ok=True)
        if e.code == 404:
            raise FileNotFoundError(
                f"Dataset '{accession}' not found at {url}. "
                f"Check the accession or use singlet.catalog() to browse available datasets."
            ) from None
        raise RuntimeError(f"Failed to download '{accession}' from {url}: HTTP {e.code}") from e
    except BaseException:
        tmp_dest.unlink(missing_ok=True)
        raise

    return dest


def open_bundle(source: str | Path, *, force: bool = False):
    """Open a study's ``.singlet`` bundle for modality-level access.

    :func:`load` gives you the assembled gene-count AnnData. ``open_bundle``
    gives you the file itself, so you can reach every other modality the
    pipeline produced for each sample — mitochondrial heteroplasmy, donor
    demultiplexing, non-host species, splice junctions, V(D)J usage and the
    per-feature (pre-aggregation) count matrices.

    Parameters
    ----------
    source : str or Path
        A GEO series accession (``"GSE138867"``) or a local ``.singlet`` path.
        Accessions are downloaded to the cache on first use.
    force : bool
        Re-download even if the bundle is already cached.

    Returns
    -------
    singlet.SingletBundle

    Examples
    --------
    >>> import singlet
    >>> b = singlet.open_bundle("GSE138867")        # doctest: +SKIP
    >>> b.modalities()                              # doctest: +SKIP
    >>> b.raw_counts(b.gsm_ids[0])                  # doctest: +SKIP
    >>> b.mt_variants(b.gsm_ids[0])                 # doctest: +SKIP
    """
    from singlet.bundle import SingletBundle

    path = Path(str(source)).expanduser()
    if path.exists():
        return SingletBundle.open(path)

    accession = str(source).strip().upper()
    if accession.startswith("GSM"):
        parent = _resolve_gsm_to_gse(accession)
        if parent is None:
            raise ValueError(f"Could not resolve {accession} to a parent GSE")
        accession = parent
    if not accession.startswith("GSE"):
        raise ValueError(
            f"open_bundle() expects a GSE/GSM accession or a .singlet path, got {source!r}"
        )
    return SingletBundle.open(download(accession, force=force))


def load(
    source: str | Path | Sequence[str | Path],
    *,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    force: bool = False,
) -> anndata.AnnData:
    """Load one or more datasets as a single AnnData.

    This is the primary entry point. Pass a GEO accession to load from the
    public Singlet data host (free download, cached) or a local ``.singlet``
    file path to read directly. Pass a list/tuple of accessions and/or
    ``.singlet`` paths to load them all and concatenate into one AnnData.

    For a ``GSE`` accession, the public ``.singlet`` bundle is downloaded and
    assembled into one AnnData. For a ``GSM`` accession, the parent GSE bundle
    is downloaded and the result is filtered to that sample's cells.

    Parameters
    ----------
    source : str, Path, or list/tuple of those
        A GEO accession (e.g. ``"GSE138867"``, ``"GSM4120733"``), a local
        ``.singlet`` file path, or a list/tuple mixing any of those. When a
        list/tuple is given, every item is loaded and concatenated into one
        AnnData (with a ``source`` column in ``obs`` recording each item's
        accession/path).
    genes : list of str, optional
        Subset to these genes (column slice). Each entry may be a
        ``var_names`` id (Ensembl gene id for bundles) or a gene symbol from
        ``var["gene_name"]``; symbols are matched exactly first, then
        case-insensitively. Unmatched entries are reported in a warning.
        ``var_names`` stay Ensembl ids.
    obs_filter : dict, optional
        Filter cells by obs columns, e.g. ``{"organism": "Homo sapiens"}``.
    force : bool
        Re-download even if the bundle is already cached.

    Returns
    -------
    anndata.AnnData
        Count matrix with obs metadata, var gene annotations, and uns study
        info. Samples whose count matrix is empty are skipped with a warning
        and listed in ``adata.uns["skipped_samples"]``.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE138867")                       # doctest: +SKIP
    >>> adata = singlet.load("/path/to/GSE138867.singlet")      # doctest: +SKIP
    >>> adata = singlet.load("GSE138867", genes=["CD3E", "MS4A1"])  # doctest: +SKIP
    >>> adata = singlet.load("GSM4120733")  # parent GSE, filtered to this GSM  # doctest: +SKIP
    >>> # Load and concatenate several studies at once:
    >>> adata = singlet.load(["GSE138867", "GSE146974"])        # doctest: +SKIP
    """
    # ---- Multiple sources → load each and concatenate -------------------
    if isinstance(source, (list, tuple)):
        items = list(source)
        if not items:
            raise ValueError("load() received an empty list of sources")
        import anndata as ad

        parts = []
        for item in items:
            sub = load(item, genes=genes, obs_filter=obs_filter, force=force)
            sub.obs["source"] = str(item)
            parts.append(sub)
        if len(parts) == 1:
            return parts[0]
        # anndata.concat's `keys` (used for the index suffix and `dataset`
        # label) must be unique; disambiguate any repeated sources while the
        # per-cell `source` column above keeps the true accession/path.
        keys = []
        seen: dict[str, int] = {}
        for item in items:
            k = str(item)
            if k in seen:
                seen[k] += 1
                keys.append(f"{k}#{seen[k]}")
            else:
                seen[k] = 0
                keys.append(k)
        combined = ad.concat(
            parts,
            join="outer",
            label="dataset",
            keys=keys,
            index_unique="-",
            merge="first",
        )
        # concat drops .uns; keep the record of samples skipped as empty.
        import pandas as pd

        skipped = [
            p.uns["skipped_samples"]
            for p in parts
            if isinstance(p.uns.get("skipped_samples"), pd.DataFrame)
        ]
        if skipped:
            combined.uns["skipped_samples"] = pd.concat(skipped, ignore_index=True)
        return combined

    return _load_one(source, genes=genes, obs_filter=obs_filter, force=force)


def _load_one(
    source: str | Path,
    *,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    force: bool = False,
) -> anndata.AnnData:
    """Load a single dataset (accession or local ``.singlet``/file path)."""
    from singlet._io import read_1pz, read_matrix

    if source is None:
        raise TypeError("load() requires a file path or GEO accession, got None")
    if not str(source).strip():
        raise ValueError("load() requires a non-empty file path or GEO accession")

    path = Path(source).expanduser()

    # Detect if source looks like a file path (has directory separators or known extension)
    _is_path = (
        "/" in str(source)
        or "\\" in str(source)
        or path.suffix.lower()
        in (
            ".singlet",
            ".1pz",
            ".spz",
            ".h5ad",
            ".zarr",
        )
    )

    # Track a GSM accession so we can filter to its cells after loading the
    # parent GSE bundle.
    _gsm_filter: Optional[str] = None

    if path.exists():
        # A ``.zarr`` store is a directory; every other directory is a mistake.
        if path.is_dir() and path.suffix.lower() != ".zarr":
            raise IsADirectoryError(
                f"'{path}' is a directory. Use singlet.load_dir() for pipeline output directories, "
                f"or provide a .singlet file path (or an accession like 'GSE138867')."
            )
        suffix = path.suffix.lower()
        if suffix == ".singlet":
            from singlet.bundle import SingletBundle

            adata = SingletBundle.open(path).to_anndata(verbose=False)
        elif suffix == ".h5ad":
            import anndata as ad

            adata = ad.read_h5ad(path)
        elif suffix == ".zarr":
            import anndata as ad

            adata = ad.read_zarr(path)
        elif suffix in (".1pz", ".spz"):
            adata = read_matrix(path)
        else:
            adata = read_matrix(path)
    elif _is_path:
        raise FileNotFoundError(f"File not found: {path}")
    else:
        # Treat as a GEO accession.
        accession = str(source).strip()
        gse_accession = accession

        if accession.upper().startswith("GSM"):
            # Resolve the GSM to its parent GSE, then filter after loading.
            _gsm_filter = accession
            parent = _resolve_gsm_to_gse(accession)
            if parent is None:
                raise FileNotFoundError(
                    f"Could not resolve parent GSE for sample {accession!r} via the "
                    f"public API ({_api_base()}). Load the parent series directly with "
                    f"singlet.load('<GSE>')."
                )
            gse_accession = parent

        # Prefer a local catalog hit for the GSE (instant); else fetch the
        # public R2 bundle.
        local_path = _resolve_gse_path(gse_accession)
        if local_path is not None and _gsm_filter is None:
            adata = read_1pz(local_path)
        else:
            local = download(gse_accession, force=force)
            from singlet.bundle import SingletBundle

            adata = SingletBundle.open(local).to_anndata(verbose=False)

    # Filter to a single GSM's cells (obs has a 'gsm_id' column; index is
    # '<GSM>_<barcode>').
    if _gsm_filter is not None:
        if "gsm_id" in adata.obs.columns:
            mask = adata.obs["gsm_id"] == _gsm_filter
        else:
            mask = adata.obs_names.str.startswith(f"{_gsm_filter}_")
        n = int(mask.sum())
        if n == 0:
            reason = _skip_reason(adata, _gsm_filter)
            if reason is not None:
                raise KeyError(
                    f"Sample {_gsm_filter!r} has no usable cells in its parent GSE "
                    f"bundle: {reason}. See https://singlet.bio/sample/{_gsm_filter}."
                )
            raise KeyError(f"Sample {_gsm_filter!r} has no cells in its parent GSE bundle.")
        adata = adata[mask].copy()

    # Gene subset
    if genes is not None:
        adata = _subset_genes(adata, genes)

    # Obs filter
    if obs_filter is not None:
        mask = True
        for col, val in obs_filter.items():
            if col not in adata.obs.columns:
                raise KeyError(
                    f"obs_filter column {col!r} not found. Available: {list(adata.obs.columns)}"
                )
            if isinstance(val, list):
                mask = mask & adata.obs[col].isin(val)
            else:
                mask = mask & (adata.obs[col] == val)
        adata = adata[mask].copy()

    return adata


def _skip_reason(adata, gsm_id: str) -> Optional[str]:
    """Why *gsm_id* was skipped while assembling *adata*, if it was."""
    import pandas as pd

    skipped = adata.uns.get("skipped_samples")
    if not isinstance(skipped, pd.DataFrame) or "gsm_id" not in skipped.columns:
        return None
    rows = skipped[skipped["gsm_id"] == gsm_id]
    if rows.empty:
        return None
    return str(rows["reason"].iloc[0]) if "reason" in rows.columns else "skipped"


def _subset_genes(adata, genes: Sequence[str]):
    """Column-subset *adata* to *genes*, given as ids or symbols.

    Each requested gene is matched, in order, against ``var_names`` (exact),
    ``var["gene_name"]`` (exact), then both case-insensitively. A symbol that
    maps to several ids keeps all of them. The original column order is kept
    and one warning lists anything left unmatched.
    """
    import warnings

    import numpy as np

    requested = [genes] if isinstance(genes, str) else [str(g) for g in genes]
    var_names = [str(v) for v in adata.var_names]
    symbols = [str(s) for s in adata.var["gene_name"]] if "gene_name" in adata.var.columns else []

    def _index(names):
        exact: dict = {}
        folded: dict = {}
        for i, name in enumerate(names):
            exact.setdefault(name, []).append(i)
            folded.setdefault(name.lower(), []).append(i)
        return exact, folded

    id_exact, id_folded = _index(var_names)
    sym_exact, sym_folded = _index(symbols)

    keep = np.zeros(len(var_names), dtype=bool)
    missing = []
    for gene in requested:
        hits = (
            id_exact.get(gene)
            or sym_exact.get(gene)
            or id_folded.get(gene.lower())
            or sym_folded.get(gene.lower())
        )
        if hits:
            keep[hits] = True
        else:
            missing.append(gene)

    if not keep.any():
        example = symbols[:5] if symbols else var_names[:5]
        warnings.warn(
            f"None of the {len(requested)} requested genes were found in the dataset "
            f"(matched against var_names and var['gene_name']). "
            f"First few available: {example}",
            UserWarning,
            stacklevel=3,
        )
    elif missing:
        warnings.warn(
            f"{len(missing)} of {len(requested)} requested genes not found: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            UserWarning,
            stacklevel=3,
        )
    return adata[:, keep].copy()


def load_sample(
    gsm_id: str,
    *,
    genes: Optional[Sequence[str]] = None,
) -> anndata.AnnData:
    """Load a single GSM sample from a local processing tree (cluster use).

    Reads the sample's column range out of its study's ``counts.1pz`` in a
    local catalog directory (``SINGLET_CATALOG_DIR`` or
    :func:`singlet.set_catalog_dir`). This is for machines that hold the
    pipeline output itself; everyone else should use
    ``singlet.load("GSM…")``, which downloads the public bundle.

    Parameters
    ----------
    gsm_id : str
        GEO sample accession (e.g. "GSM4120733").
    genes : list of str, optional
        Subset to these gene names.

    Returns
    -------
    anndata.AnnData
        Count matrix for just this sample.

    Raises
    ------
    RuntimeError
        If no local catalog directory is configured.
    KeyError
        If the sample is not in the local sample index.
    """
    import anndata as ad
    import scipy.sparse as sp

    from singlet._catalog import _get_catalog_dir, _load_sample_index

    # Check the catalog before touching the compiled codec, so a machine
    # without one gets this message rather than an ImportError.
    cat_dir = _get_catalog_dir()
    if cat_dir is None:
        raise RuntimeError(
            "load_sample requires a local catalog (a directory holding the "
            "pipeline's processing tree). Set SINGLET_CATALOG_DIR or call "
            "singlet.set_catalog_dir(). To load a public sample, use "
            f"singlet.load({gsm_id!r}) instead."
        )

    idx = _load_sample_index()
    rows = idx[idx["gsm_id"] == gsm_id]
    if rows.empty:
        raise KeyError(f"Sample {gsm_id!r} not found in sample index")

    row = rows.iloc[0]
    base = cat_dir.parent
    gse_path = row["gse_id"]
    subdir = row.get("species_subdir", "")
    if subdir:
        counts_path = base / "pipeline" / "quant" / gse_path / subdir / "counts.1pz"
    else:
        counts_path = base / "pipeline" / "quant" / gse_path / "counts.1pz"

    col_start = int(row["col_offset"])
    col_end = col_start + int(row["col_count"])

    # TODO: native read_1pz_columns slice — currently we read all then slice
    r = _read_pz_record(str(counts_path))
    full_mat = sp.csc_matrix(
        (r["data"], r["indices"], r["indptr"]),
        shape=(r["m"], r["n"]),
    )
    mat = full_mat[:, col_start:col_end]

    adata = ad.AnnData(X=mat.T)

    if r["rownames"]:
        import pandas as pd

        adata.var_names = pd.Index(r["rownames"])

    adata.obs["gsm_id"] = gsm_id
    adata.obs["gse_id"] = row["gse_id"]
    adata.obs["organism"] = row["organism"]

    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        adata = adata[:, gene_mask].copy()

    return adata


def load_dir(
    path: str | Path,
    *,
    layer: str = "gene_counts",
    with_qc: bool = True,
    with_doublets: bool = True,
) -> anndata.AnnData:
    """Load a singlet pipeline output directory as AnnData.

    Reads the count matrix (.1pz), attaches gene names from
    gene_expression.tsv, cell barcodes from auto_barcodes.tsv,
    and optionally merges QC metrics and doublet scores.

    Parameters
    ----------
    path : str or Path
        Path to a singlet sample output directory (containing
        gene_counts.1pz, auto_barcodes.tsv, gene_expression.tsv).
    layer : str
        Which .1pz to load: "gene_counts", "exon_counts", "intron_counts",
        or "gene_counts_em". Default "gene_counts".
    with_qc : bool
        Merge cell_qc_metrics.tsv into obs if available.
    with_doublets : bool
        Merge doublet_scores.tsv into obs if available.

    Returns
    -------
    anndata.AnnData
        Sparse count matrix (cells × genes) with metadata.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load_dir("/path/to/quant/GSM3573650")
    >>> adata
    AnnData object with n_obs × n_vars = 75420 × 38606
    """
    import pandas as pd

    from singlet._io import read_1pz

    path = Path(path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")

    # Read count matrix
    pz_file = path / f"{layer}.1pz"
    if not pz_file.exists():
        raise FileNotFoundError(f"Missing {pz_file}")
    adata = read_1pz(pz_file)

    # Attach gene names
    gene_file = path / "gene_expression.tsv"
    if gene_file.exists():
        genes = pd.read_csv(gene_file, sep="\t", usecols=["gene_id", "gene_name"])  # type: ignore[arg-type]
        if len(genes) == adata.n_vars:
            adata.var_names = pd.Index(genes["gene_name"].values)
            adata.var["gene_id"] = genes["gene_id"].values

    # Attach barcodes
    bc_file = path / "auto_barcodes.tsv"
    if bc_file.exists():
        barcodes = pd.read_csv(bc_file, header=None)[0].values
        if len(barcodes) == adata.n_obs:
            adata.obs_names = pd.Index(barcodes)

    # Merge QC metrics
    if with_qc:
        qc_file = path / "cell_qc_metrics.tsv"
        if qc_file.exists():
            qc = pd.read_csv(qc_file, sep="\t", index_col="barcode")
            overlap = adata.obs_names.intersection(qc.index)
            if len(overlap) > 0:
                for col in qc.columns:
                    adata.obs[col] = qc[col].reindex(adata.obs_names).values

    # Merge doublet scores
    if with_doublets:
        dub_file = path / "doublet_scores.tsv"
        if dub_file.exists():
            dub = pd.read_csv(dub_file, sep="\t", index_col="barcode")
            overlap = adata.obs_names.intersection(dub.index)
            if len(overlap) > 0:
                for col in dub.columns:
                    adata.obs[col] = dub[col].reindex(adata.obs_names).values

    # Merge cell cycle scores
    cc_file = path / "cell_cycle_scores.tsv"
    if cc_file.exists():
        cc = pd.read_csv(cc_file, sep="\t", index_col="barcode")
        overlap = adata.obs_names.intersection(cc.index)
        if len(overlap) > 0:
            for col in cc.columns:
                adata.obs[col] = cc[col].reindex(adata.obs_names).values

    # Store ancestry call if available
    ancestry_file = path / "ancestry_call.json"
    if ancestry_file.exists():
        import json as _json

        with open(ancestry_file) as f:
            adata.uns["ancestry"] = _json.load(f)

    # Store sex call if available
    sex_file = path / "sex_call.json"
    if sex_file.exists():
        import json as _json

        with open(sex_file) as f:
            adata.uns["sex_call"] = _json.load(f)

    # Store summary metrics if available
    summary_file = path / "summary.json"
    if summary_file.exists():
        import json as _json

        with open(summary_file) as f:
            adata.uns["summary"] = _json.load(f)

    # Store saturation curve if available
    sat_file = path / "saturation_curve.tsv"
    if sat_file.exists():
        adata.uns["saturation_curve"] = pd.read_csv(sat_file, sep="\t")

    # Store source path
    adata.uns["singlet_dir"] = str(path)

    return adata
