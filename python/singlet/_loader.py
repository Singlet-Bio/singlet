# SPDX-License-Identifier: MIT
"""Load Singlet datasets as AnnData.

Users work with GEO accession strings (``GSE…`` / ``GSM…``) and local files.
Public data is hosted in the per-sample format (``docs/format-s1.md``)::

    https://data.singlet.bio/s1/m/<GSE>.json         study manifest
    https://data.singlet.bio/s1/s/<GSM>/<d>.s1       one immutable object per sample

(free, no authentication). :func:`load` fetches only the members it needs with
HTTP Range requests (:mod:`singlet.s1.client`). The legacy per-study archive
``https://data.singlet.bio/data/<GSE>/<GSE>.singlet`` is being retired: it is
read only with an explicit ``legacy=True`` (or, as a transition aid, when
``$SINGLET_LEGACY_FALLBACK=1`` and a study has no .s1 manifest yet). A local
``.singlet`` file always opens via :class:`singlet.SingletBundle`. The host is
configurable with ``$SINGLET_DATA_BASE`` (default ``https://data.singlet.bio``;
a value that already ends in ``/data`` — the form the R client documents — is
accepted too; a directory laid out like the server works as well).

Resolution priority for :func:`load`:
  1. A local ``.singlet`` file path (also accepts ``.h5ad`` / ``.zarr`` you exported), a ``.s1`` sample object or a .s1 manifest
  2. A local catalog directory (cluster use: SINGLET_CATALOG_DIR or singlet.set_catalog_dir())
  3. The public per-sample study for an accession (ranged reads; nothing is stored unless you call :func:`download`)

GSM accessions are resolved to their parent GSE (via the public REST API at
``https://singlet.bio/api``); only that sample's members are fetched.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from singlet._integrity import (
    BundleIntegrityError,
    normalize_accession,
    normalize_gse,
    verify_bundle,
)

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
    gsm_id = normalize_accession(gsm_id)
    base = _api_base()
    q = urllib.parse.quote(gsm_id, safe="")
    # Primary: /api/gsm?q=<GSM> (the /api/gsm/<id> detail endpoint is currently
    # unreliable). Fallback: /api/search?q=<GSM>.
    candidates = (
        f"{base}/gsm?q={q}&limit=5",
        f"{base}/search?q={q}",
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
                parent = str(row["gse_id"]).strip().upper()
                if re.match(r"^GSE[0-9]{1,9}$", parent):
                    return parent
        # Only an exact id match counts: a search can return a different sample,
        # and loading the wrong study silently is worse than failing.
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


# Transition switch (docs/s1-migration.md): legacy bundles are retired (data.singlet.bio answers 410 for them), so a study
# without a .s1 manifest is not read from one. ``$SINGLET_LEGACY_FALLBACK`` still overrides either way.
_LEGACY_FALLBACK_DEFAULT = False


def _legacy_fallback_enabled() -> bool:
    """Transition aid: allow the legacy bundle for studies without a .s1 manifest (off since the retirement)."""
    v = os.environ.get("SINGLET_LEGACY_FALLBACK", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return _LEGACY_FALLBACK_DEFAULT


def _warn_fallback(err: Exception, stacklevel: int = 3) -> None:
    """Say so only when the user asked for the fallback explicitly; the default transition path stays quiet."""
    if os.environ.get("SINGLET_LEGACY_FALLBACK", "").strip():
        warnings.warn(
            f"{err}; using the legacy bundle (SINGLET_LEGACY_FALLBACK)",
            FutureWarning,
            stacklevel=stacklevel,
        )


def _no_s1_message(err: Exception) -> str:
    return (
        f"{err}. The study is not in the atlas: check the accession, or see https://singlet.bio/known-issues for studies "
        f"Research Alpha withholds (legacy .singlet bundles are retired)."
    )


def download(
    accession: str,
    output_dir: Optional[str | Path] = None,
    force: bool = False,
    verify: bool = True,
    *,
    legacy: Optional[bool] = None,
) -> Path:
    """Download a study for offline use.

    By default this mirrors the study's .s1 objects (manifest, one ``.s1`` per sample, the packs they need) under
    ``<output_dir or cache>/s1/`` with the server's layout, verifies every sample object against the manifest's
    sha256, and returns the path of the manifest (``.../s1/m/<GSE>.json``). :func:`load` and :func:`open_bundle`
    accept that path and read the local copies without network access; :func:`load` also finds them in the default
    cache on its own. With ``legacy=True`` the legacy ``.singlet`` bundle is fetched instead.
    """
    if legacy is not True:
        from singlet.s1 import client as _store

        try:
            return _store.download_study(
                normalize_gse(accession), output_dir, force=force, verify=verify
            )
        except _store.StudyNotFound as e:
            if legacy is None and _legacy_fallback_enabled():
                _warn_fallback(e)
            else:
                raise _store.StudyNotFound(_no_s1_message(e)) from None
    return _download_legacy(accession, output_dir, force, verify)


def _download_legacy(
    accession: str,
    output_dir: Optional[str | Path] = None,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """Download a GSE's legacy ``.singlet`` bundle from Cloudflare R2 (deprecated: the per-sample format replaces it).

    Fetches ``https://data.singlet.bio/data/<GSE>/<GSE>.singlet`` (free, no
    auth) to the local cache. The download is path-cached: an existing file
    that is still a complete archive is reused unless ``force=True``.

    A download is only moved into the cache after it passes
    :func:`singlet._integrity.verify_bundle`: the byte count matches what the
    server announced, the archive is complete, the schema is one this version
    reads, and (with ``verify=True``, the default) every member matches the
    SHA-256 in ``manifest.json``. A cached file that no longer passes the
    structural check is discarded and fetched again.

    Parameters
    ----------
    accession : str
        GEO series accession (e.g. "GSE138867").
    output_dir : path, optional
        Where to save. Defaults to ``default_cache_dir()``
        (``$SINGLET_CACHE_DIR`` or ``~/.singlet/cache``).
    force : bool
        Re-download even if cached.
    verify : bool
        Re-hash every member against the manifest before caching (reads the
        file once more; leave on unless you are verifying elsewhere).

    Returns
    -------
    Path
        Path to the downloaded ``.singlet`` bundle.

    Raises
    ------
    ValueError
        ``accession`` is not a GEO accession.
    FileNotFoundError
        The study has no published file.
    BundleIntegrityError
        The download was truncated, altered, or uses a newer schema.
    """
    accession = normalize_gse(accession)
    dest_dir = Path(output_dir) if output_dir else _cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (
        dest_dir / f"{accession}.singlet"
    )  # the accession matched ^GSE[0-9]+$, so this stays in dest_dir

    # On-disk cache: reuse a file that is still a complete, readable archive.
    if dest.exists() and not force:
        try:
            verify_bundle(dest, hashes=False)
            return dest
        except BundleIntegrityError:
            try:
                dest.unlink(missing_ok=True)
            except OSError as e:  # Windows: still open elsewhere
                raise OSError(
                    f"{dest} is damaged but cannot be replaced while it is open ({e}); close it and retry"
                ) from e

    url = _bundle_url(accession)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    # Write to a temp file first, then atomic rename (interrupted or invalid
    # downloads never reach the cache). The name is unique per process so two
    # simultaneous downloads of one study cannot interleave.
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=f".{dest.name}.", suffix=".part")
    os.close(fd)
    tmp_dest = Path(tmp_name)
    try:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment]

        written = 0
        with urllib.request.urlopen(req, timeout=120) as resp:
            announced = resp.headers.get("content-length")
            total = int(announced) if announced and announced.isdigit() else None
            bar = (
                tqdm(total=total, unit="B", unit_scale=True, desc=accession)
                if tqdm is not None and total
                else None
            )
            try:
                with open(tmp_dest, "wb") as f:
                    for chunk in iter(lambda: resp.read(1 << 16), b""):
                        f.write(chunk)
                        written += len(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()
        if total is not None and written != total:
            raise BundleIntegrityError(
                f"download of {accession} ended after {written:,} of {total:,} bytes; nothing was cached. Try again."
            )
        verify_bundle(tmp_dest, expected_size=total, hashes=verify)
        try:
            tmp_dest.replace(dest)
        except OSError as e:  # Windows will not replace a file that is open elsewhere
            raise OSError(
                f"downloaded {accession} but cannot replace {dest} while it is open ({e}); close it and retry"
            ) from e
    except urllib.error.HTTPError as e:
        tmp_dest.unlink(missing_ok=True)
        if e.code == 404:
            raise FileNotFoundError(
                f"Dataset '{accession}' not found at {url}. "
                f"Check the accession or use singlet.catalog() to browse available datasets."
            ) from None
        if e.code == 410:  # retired legacy bundle: the server names the study's .s1 manifest
            link = e.headers.get("Link", "") if e.headers else ""
            raise FileNotFoundError(
                f"The legacy bundle of '{accession}' is retired{(' (' + link + ')') if link else ''}. "
                f"Load the study with singlet.load('{accession}') or singlet.download('{accession}') without legacy=True."
            ) from None
        raise RuntimeError(f"Failed to download '{accession}' from {url}: HTTP {e.code}") from e
    except BaseException:
        tmp_dest.unlink(missing_ok=True)
        raise

    return dest


def open_bundle(source: str | Path, *, force: bool = False, legacy: Optional[bool] = None):
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
    legacy : bool, optional
        ``True`` opens the legacy ``.singlet`` bundle of an accession (deprecated). The default opens the per-sample study.

    Returns
    -------
    singlet.SingletBundle
        For a per-sample study (the default), a :class:`singlet.s1.bundle.S1Bundle` with the same interface.

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
        if path.suffix.lower() in (".json", ".s1", ".s2"):  # a .s1 manifest or sample object
            from singlet.s1.bundle import S1Bundle

            return S1Bundle.open(path)
        return SingletBundle.open(path)

    try:
        accession = normalize_accession(source)
    except ValueError:
        raise ValueError(
            f"open_bundle() expects a GSE/GSM accession or a .singlet path, got {source!r}"
        ) from None
    if accession.startswith("GSM"):
        parent = _resolve_gsm_to_gse(accession)
        if parent is None:
            raise ValueError(f"Could not resolve {accession} to a parent GSE")
        accession = parent
    if legacy is not True:
        from singlet.s1 import client as _store
        from singlet.s1.bundle import S1Bundle

        try:
            return S1Bundle.open(accession, force=force)
        except _store.StudyNotFound as e:
            if not (legacy is None and _legacy_fallback_enabled()):
                raise _store.StudyNotFound(_no_s1_message(e)) from None
            _warn_fallback(e)
    return SingletBundle.open(_download_legacy(accession, force=force))


def load(
    source: str | Path | Sequence[str | Path],
    *,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    force: bool = False,
    legacy: Optional[bool] = None,
    layers: bool = True,
    modalities: Optional[str | Sequence[str]] = None,
    **s1_options,
) -> anndata.AnnData:
    """Load one or more datasets as a single AnnData.

    This is the primary entry point. Pass a GEO accession to load from the
    public Singlet data host (free download, cached) or a local ``.singlet``
    file path to read directly. Pass a list/tuple of accessions and/or
    ``.singlet`` paths to load them all and concatenate into one AnnData.

    For a ``GSE`` accession, the study's .s1 members are fetched with HTTP Range
    requests (only the matrices and cell tables the gene-level result needs) and
    assembled into one AnnData. For a ``GSM`` accession, only that sample's
    members are fetched. A ``.s1`` sample object or a .s1 manifest path is read
    from disk.

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
        Re-validate the study manifest even if it was checked a moment ago
        (.s1), or re-download the bundle (legacy).
    legacy : bool, optional
        ``True`` reads the legacy ``.singlet`` bundle of an accession (deprecated,
        being retired). The default reads the per-sample format.
    layers : bool
        Attach ``spliced`` / ``unspliced`` layers next to ``X = spliced + unspliced``
        (.s1 reader; the legacy reader always does).
    modalities : str or list of str, optional
        Fetch only what these modalities need (.s1 reader: only those members'
        byte ranges are downloaded). ``"counts"`` (``X`` = spliced + unspliced;
        needs the exon and intron matrices), ``"spliced"`` (``layers["spliced"]``,
        exon matrix only), ``"unspliced"`` (``layers["unspliced"]``, intron matrix
        only), ``"splice_junctions"`` and ``"psi"`` (``obsm``, junction matrix
        only; ``psi`` is the true float PSI), ``"mt"`` and ``"vdj"`` (``obsm``).
        Feature names of an ``obsm`` modality are in
        ``uns["obsm_features"][name]``; :func:`singlet.s1.client.modality_anndata`
        turns one into its own AnnData. Without ``"counts"``, ``X`` is ``None``.
        The default keeps ``counts`` + ``spliced`` + ``unspliced`` (``counts``
        alone with ``layers=False``). A modality no requested sample carries
        raises :class:`singlet.s1.client.ModalityUnavailable` listing what the
        study has. A legacy bundle is always downloaded whole; the result is
        then subset (with a warning).
    **s1_options
        ``samples`` (restrict a study to some GSMs), ``cell_order`` (``"legacy"``,
        ``"barcode"`` or ``"column"``), ``concurrency`` (parallel samples and
        range requests, default 4), ``threads`` (decoder threads), ``verify``
        (recompute canonical matrix digests), ``whole_max`` (bytes below which a
        sample's members are fetched whole), ``base`` (data host),
        ``cell_type_level`` (deepest level of ``obs["cell_type"]``, default 2),
        ``cell_type_min_score`` (extra minimum path score for it, default 0),
        ``include_unvalidated`` (also keep ``qc_doublet_score``).

    Returns
    -------
    anndata.AnnData
        Count matrix with obs metadata, var gene annotations, and uns study
        info. Samples whose count matrix is empty are skipped with a warning
        and listed in ``adata.uns["skipped_samples"]``. For a per-sample (.s1)
        study, ``obs`` carries the per-cell labels (``cell_type``,
        ``cell_type_ontology_term_id``, ``sex_chrom``, ``ancestry_hancestro``,
        ...), the cell calls (``called``, ``called_ed``) and the QC columns of
        each sample's cells table, then the GEO sample metadata; see
        :mod:`singlet.s1.labels`.

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
            sub = load(
                item,
                genes=genes,
                obs_filter=obs_filter,
                force=force,
                legacy=legacy,
                layers=layers,
                modalities=modalities,
                **s1_options,
            )
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

    return _load_one(
        source,
        genes=genes,
        obs_filter=obs_filter,
        force=force,
        legacy=legacy,
        layers=layers,
        modalities=modalities,
        s1_options=s1_options,
    )


_STORE_OPTIONS = frozenset(
    {
        "samples",
        "cell_order",
        "concurrency",
        "threads",
        "verify",
        "whole_max",
        "base",
        "verbose",
        "cell_type_level",
        "cell_type_min_score",
        "include_unvalidated",
    }
)


def _load_store(
    accession: str,
    gsm: Optional[str],
    *,
    genes,
    obs_filter,
    force: bool,
    layers: bool,
    options: dict,
    modalities=None,
) -> anndata.AnnData:
    """Gene-level AnnData of a per-sample study (or of one sample of it) through :mod:`singlet.s1.client`."""
    from singlet.s1 import client as _store

    unknown = set(options) - _STORE_OPTIONS
    if unknown:
        raise TypeError(f"load() got unexpected keyword argument(s): {sorted(unknown)}")
    opts = dict(options)
    base = opts.pop("base", None)
    samples = opts.pop("samples", None)
    if gsm is not None:
        samples = [gsm]
    study = _store.open_study(accession, base=base, force=force)
    try:
        return study.to_anndata(
            samples,
            genes=genes,
            obs_filter=obs_filter,
            layers=layers,
            modalities=modalities,
            **opts,
        )
    except _store.NoUsableSamples as e:
        if gsm is not None:
            raise KeyError(
                f"Sample {gsm!r} has no usable cells in its parent GSE: {e.reasons.get(gsm, 'skipped')}. "
                f"See https://singlet.bio/sample/{gsm}."
            ) from None
        raise


def _load_one(
    source: str | Path,
    *,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    force: bool = False,
    legacy: Optional[bool] = None,
    layers: bool = True,
    modalities: Optional[str | Sequence[str]] = None,
    s1_options: Optional[dict] = None,
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

            bundle = SingletBundle.open(path)
            adata = _legacy_modalities(bundle.to_anndata(verbose=False), bundle, modalities)
        elif suffix in (".s1", ".s2", ".json"):  # a .s1 sample object or study manifest on disk
            from singlet.s1 import client as _store
            from singlet.s1.write import is_user_object, read_s1

            if suffix != ".json" and is_user_object(
                path
            ):  # written by singlet.write_s1, not an atlas sample
                adata = read_s1(path)
                if genes is not None:
                    adata = adata[:, [g for g in genes if g in adata.var_names]].copy()
                return adata

            return _store.load(
                path,
                genes=genes,
                obs_filter=obs_filter,
                layers=layers,
                modalities=modalities,
                force=force,
                **(s1_options or {}),
            )
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
        try:
            accession = normalize_accession(source)
        except ValueError:
            raise FileNotFoundError(
                f"{str(source).strip()!r} is neither a file that exists nor a GEO accession (GSE… or GSM…)."
            ) from None
        gse_accession = accession

        if accession.startswith("GSM"):
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
            if legacy is not True:
                from singlet.s1 import client as _store

                try:
                    return _load_store(
                        gse_accession,
                        _gsm_filter,
                        genes=genes,
                        obs_filter=obs_filter,
                        force=force,
                        layers=layers,
                        options=s1_options or {},
                        modalities=modalities,
                    )
                except _store.StudyNotFound as e:
                    if not (legacy is None and _legacy_fallback_enabled()):
                        raise _store.StudyNotFound(_no_s1_message(e)) from None
                    _warn_fallback(e, stacklevel=4)
            local = _download_legacy(gse_accession, force=force)
            from singlet.bundle import SingletBundle

            bundle = SingletBundle.open(local)
            adata = _legacy_modalities(bundle.to_anndata(verbose=False), bundle, modalities)

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


_LEGACY_MODALITY = {
    "splice_junctions": "junctions",
    "psi": "junctions",
    "mt": "mt_heteroplasmy",
    "vdj": "vdj_gene_usage",
}


def _legacy_modalities(adata, bundle, modalities):
    """``modalities=`` on a legacy bundle: the whole bundle was downloaded already, so this only shapes the result like the
    .s1 reader does (X / layers / obsm). A modality the bundle cannot supply is warned about and left out, never fatal."""
    if modalities is None:
        return adata
    from singlet.s1 import client as _store

    mods = _store.normalize_modalities(modalities)
    warnings.warn(
        "this study is read from its legacy .singlet bundle, which is downloaded whole; modalities= only subsets the "
        "result (the .s1 reader fetches only the members asked for)",
        UserWarning,
        stacklevel=4,
    )
    for lname in ("spliced", "unspliced"):
        if lname not in mods and lname in adata.layers:
            del adata.layers[lname]
    if "counts" not in mods:
        adata.X = None
    if not any(_store.MODALITIES[k].slot != "obsm" for k in mods):
        adata = adata[:, []].copy()
    names_out = {}
    for k in mods:
        if _store.MODALITIES[k].slot != "obsm":
            continue
        try:
            mat, names = _legacy_feature_block(adata, bundle, _LEGACY_MODALITY[k], psi=(k == "psi"))
        except Exception as e:  # noqa: BLE001 - a legacy bundle may lack or mangle a member
            warnings.warn(
                f"{k} is not available from the legacy bundle: {e}", UserWarning, stacklevel=4
            )
            continue
        adata.obsm[k] = mat
        names_out[k] = names
    if names_out:
        adata.uns["obsm_features"] = names_out
    return adata


def _legacy_feature_block(adata, bundle, legacy_name: str, *, psi: bool):
    """cells x features CSR aligned to ``adata.obs_names`` (``<GSM>_<barcode>``) from ``bundle.read(legacy_name, gsm)``."""
    import numpy as np
    import scipy.sparse as sp

    gsms = (
        list(dict.fromkeys(adata.obs["gsm_id"].astype(str)))
        if "gsm_id" in adata.obs.columns
        else list(bundle.gsm_ids)
    )
    row_of = {n: i for i, n in enumerate(adata.obs_names)}
    union: dict = {}
    rows, cols, vals = [], [], []
    for g in gsms:
        if legacy_name not in bundle.modalities(g):
            continue
        sub = bundle.read(legacy_name, g)
        m = sp.csr_matrix(sub.X)
        m.sort_indices()
        names = [str(x) for x in sub.var_names]
        if psi:
            from singlet.s1.client import _Part, _psi_part

            part, _ = _psi_part(
                _Part(m.indptr, m.indices.astype(np.int32), m.data, len(names)), names
            )
            m = part.csr()
        remap = np.fromiter(
            (union.setdefault(n, len(union)) for n in names), dtype=np.int64, count=len(names)
        )
        target = np.array([row_of.get(f"{g}_{b}", -1) for b in sub.obs_names], dtype=np.int64)
        coo = m.tocoo()
        keep = target[coo.row] >= 0
        rows.append(target[coo.row[keep]])
        cols.append(remap[coo.col[keep]])
        vals.append(coo.data[keep])
    if not rows:
        raise KeyError(f"no sample carries {legacy_name!r}")
    mat = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(adata.n_obs, len(union)),
    )
    return mat, np.array(list(union), dtype=object)


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


def _gene_keep_mask(
    var_names: Sequence[str], symbols: Sequence[str], genes: Sequence[str], *, stacklevel: int = 3
):
    """Boolean mask over the gene axis for the requested ``genes`` (ids or symbols).

    Each requested gene is matched, in order, against ``var_names`` (exact),
    ``symbols`` (exact), then both case-insensitively. A symbol that maps to
    several ids keeps all of them. One warning lists anything left unmatched.
    """
    import numpy as np

    requested = [genes] if isinstance(genes, str) else [str(g) for g in genes]
    var_names = [str(v) for v in var_names]
    symbols = [str(s) for s in symbols]

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
            stacklevel=stacklevel,
        )
    elif missing:
        warnings.warn(
            f"{len(missing)} of {len(requested)} requested genes not found: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            UserWarning,
            stacklevel=stacklevel,
        )
    return keep


def _subset_genes(adata, genes: Sequence[str]):
    """Column-subset *adata* to *genes*, given as ids or symbols.

    Each requested gene is matched, in order, against ``var_names`` (exact),
    ``var["gene_name"]`` (exact), then both case-insensitively. A symbol that
    maps to several ids keeps all of them. The original column order is kept
    and one warning lists anything left unmatched.
    """
    symbols = [str(s) for s in adata.var["gene_name"]] if "gene_name" in adata.var.columns else []
    keep = _gene_keep_mask([str(v) for v in adata.var_names], symbols, genes, stacklevel=4)
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
