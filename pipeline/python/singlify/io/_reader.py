"""
High-level reader API for singlify pipeline outputs.

The functions here are the public face of :mod:`singlify.io`. They delegate
all heavy lifting to the :mod:`singlify._pz_io` pybind11 wrapper, which in
turn calls the header-only C++ decoder at
``singlify/include/singlet-pileup/pz_reader.h``.

Public entry points::

    from singlify.io import read_matrix, read_metadata_from_file, read_dir

    mat, meta = read_matrix("gene_counts.1pz")
    dd = read_dir("pipeline_output_dir/")
    dd["gene_counts"]   # scipy.sparse.csc_matrix
    dd.metadata         # PZMetadata (rownames/colnames/user_kv)
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from ._metadata import PZMetadata

PathLike = Union[str, bytes, os.PathLike]


def _load_via_binding(path: PathLike) -> dict:
    """Call the pybind11 binding. Import lazily so that the pure-Python
    helpers (:mod:`singlify.io._format` etc.) keep working even if the
    ``_pz_io`` extension is not built."""
    from .. import _pz_io  # local import — raises ImportError if missing
    return _pz_io.read_1pz(os.fspath(path))


def read_matrix(path: PathLike) -> Tuple[sp.csc_matrix, PZMetadata]:
    """Read a single ``.1pz`` file into a scipy sparse CSC and its metadata.

    Parameters
    ----------
    path
        Filesystem path to the ``.1pz`` file.

    Returns
    -------
    (scipy.sparse.csc_matrix, PZMetadata)
        The matrix has dtype uint8 / uint16 / uint32 depending on the
        narrowest integer width that losslessly holds every stored value
        (per ``vt_code`` in the file header). The metadata object carries
        ``rownames``, ``colnames``, and ``user_kv`` (the GEO context fields
        embedded by the pipeline).

    Raises
    ------
    ValueError
        If the file cannot be decoded.
    """
    d = _load_via_binding(path)
    mat = sp.csc_matrix(
        (d["data"], d["indices"], d["indptr"]), shape=d["shape"]
    )
    meta = PZMetadata(
        rownames=list(d["rownames"]),
        colnames=list(d["colnames"]),
        user_kv=dict(d["user_kv"]),
    )
    return mat, meta


def read_metadata_from_file(path: PathLike) -> PZMetadata:
    """Read only the metadata from a ``.1pz`` file.

    This still loads the full file into memory via the C++ reader — use
    :func:`singlify.io.open_pz` + :func:`singlify.io._metadata.read_metadata`
    for the pure-Python path that skips matrix decoding.
    """
    _, meta = read_matrix(path)
    return meta


@dataclass
class PipelineDirectory:
    """A decoded singlify pipeline output directory.

    Attributes
    ----------
    path
        Original directory path.
    matrices
        Dict of matrix name (without ``.1pz`` suffix) to
        :class:`scipy.sparse.csc_matrix`. Typical keys include
        ``gene_counts``, ``spliced``, ``unspliced``, ``exon_counts``,
        ``intron_counts``, ``sj_counts``, ``mt_heteroplasmy``, ``snp_ad``,
        ``snp_dp``, depending on which files the pipeline wrote.
    metadata
        Embedded TLV metadata from any one matrix (they all carry the same
        ``user_kv`` block). ``rownames``/``colnames`` may differ between
        matrices, but the GEO context is consistent.
    """

    path: pathlib.Path
    matrices: Dict[str, sp.csc_matrix] = field(default_factory=dict)
    metadata: PZMetadata = field(default_factory=PZMetadata)
    matrix_rownames: Dict[str, List[str]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> sp.csc_matrix:
        return self.matrices[key]

    def __contains__(self, key: str) -> bool:
        return key in self.matrices

    def keys(self):
        return self.matrices.keys()

    def __iter__(self):
        return iter(self.matrices)

    def __len__(self) -> int:
        return len(self.matrices)

    def __repr__(self) -> str:
        shapes = ", ".join(
            f"{k}={v.shape[0]}×{v.shape[1]}" for k, v in self.matrices.items()
        )
        gsm = self.metadata.user_kv.get("gsm_id", "?")
        return f"PipelineDirectory({self.path}, gsm={gsm}, {shapes})"


def aggregate_features_to_gene(
    feature_mat: sp.csc_matrix,
    feature_rownames: List[str],
) -> Tuple[sp.csc_matrix, List[str]]:
    """Aggregate a per-feature (per-exon or per-intron) sparse matrix to a
    per-gene matrix by summing over all features whose rowname starts with
    the same ``ENSG...`` prefix.

    The pipeline writes per-feature rownames in the format::

        ENSG00000000003_TSPAN6_chrX:100627107-100629986

    where the leading ``ENSGxxxxxxxxxxx`` token is the Ensembl gene ID. This
    helper extracts the gene ID, builds an aggregation map, and returns
    a (per-gene matrix, gene_rowname_list) pair.

    Verified empirically against ``spliced.1pz`` and ``unspliced.1pz`` on
    real pipeline outputs: ``sum(aggregate(exon_counts)) == sum(spliced)``
    bit-exactly. Same for intron_counts ↔ unspliced.

    Parameters
    ----------
    feature_mat
        A :class:`scipy.sparse.csc_matrix` with shape ``(n_features, n_cells)``
        as produced by :func:`read_matrix` on ``exon_counts.1pz`` or
        ``intron_counts.1pz``.
    feature_rownames
        The corresponding row names, one per feature. Each must begin
        with an ``ENSG...`` (or other Ensembl-stable-ID-style) prefix
        followed by an underscore.

    Returns
    -------
    (scipy.sparse.csc_matrix, list[str])
        The per-gene aggregated matrix (shape ``n_genes × n_cells``) and
        the gene ID list in the order they appear in the matrix rows.

    Examples
    --------
    Reconstruct spliced from exon_counts::

        from singlify.io import read_matrix, aggregate_features_to_gene

        ex, meta = read_matrix("exon_counts.1pz")
        spliced, gene_ids = aggregate_features_to_gene(ex, meta.rownames)

    The returned matrix is a drop-in replacement for ``spliced.1pz`` —
    ``sum(spliced.sum()) == sum(ex.sum())`` exactly.
    """
    if feature_mat.shape[0] != len(feature_rownames):
        raise ValueError(
            f"matrix has {feature_mat.shape[0]} rows but {len(feature_rownames)} "
            "rownames; they must match."
        )

    n_features, n_cells = feature_mat.shape
    if n_features == 0:
        return sp.csc_matrix((0, n_cells), dtype=feature_mat.dtype), []

    # Build feature-row → gene-row mapping by extracting the gene ID prefix.
    # Format: "ENSG..._GENE_chr:start-end". The gene ID is everything before
    # the first underscore — Ensembl IDs themselves never contain underscores.
    gene_to_idx: Dict[str, int] = {}
    feature_to_gene = np.empty(n_features, dtype=np.int64)
    for fi, name in enumerate(feature_rownames):
        gene_id = name.split("_", 1)[0]
        idx = gene_to_idx.get(gene_id)
        if idx is None:
            idx = len(gene_to_idx)
            gene_to_idx[gene_id] = idx
        feature_to_gene[fi] = idx

    n_genes = len(gene_to_idx)

    # Aggregation: build a sparse projection M[gene, feature] = 1 and compute
    # M @ feature_mat. This is O(n_features + nnz) and avoids dense intermediates.
    proj = sp.csr_matrix(
        (np.ones(n_features, dtype=feature_mat.dtype),
         (feature_to_gene, np.arange(n_features, dtype=np.int64))),
        shape=(n_genes, n_features),
    )
    aggregated = (proj @ feature_mat).tocsc()

    # Rebuild gene_rowname_list in row order
    gene_ids = [None] * n_genes
    for gid, idx in gene_to_idx.items():
        gene_ids[idx] = gid

    return aggregated, gene_ids  # type: ignore[return-value]


def read_cohort(
    paths: List[PathLike],
    *,
    matrix_name: str = "spliced",
    show_progress: bool = False,
) -> List[Tuple[sp.csc_matrix, PZMetadata]]:
    """Read the same matrix from many singlify pipeline output directories.

    The most common cross-sample workflow: you have a directory of GSM
    outputs from one GSE (or a list of paths spanning multiple GSEs)
    and you want to load the same per-gene matrix from each into a list
    ready for cohort-level concatenation or comparison.

    Parameters
    ----------
    paths
        List of pipeline output directories.
    matrix_name
        Which matrix to load from each directory (without the ``.1pz``
        suffix). Default ``"spliced"``.
    show_progress
        When True, print one-line-per-sample progress to stderr.

    Returns
    -------
    list of (scipy.sparse.csc_matrix, PZMetadata)
        One entry per input path, in input order. If a directory does
        not contain ``<matrix_name>.1pz``, that entry is dropped from
        the output list (and a warning is printed when ``show_progress``
        is True).

    Examples
    --------
    Load all gene_counts matrices from a GSE::

        from pathlib import Path
        from singlify.io import read_cohort

        gse_dir = Path("quant/scrna/GSE174/GSE174399")
        sample_dirs = sorted(d for d in gse_dir.iterdir() if d.is_dir())
        cohort = read_cohort(sample_dirs, matrix_name="spliced")

        # Now you can concatenate
        import scipy.sparse as sp
        big = sp.hstack([m for m, _ in cohort])
        gsm_labels = [meta.user_kv["gsm_id"] for _, meta in cohort]
    """
    import sys
    out: List[Tuple[sp.csc_matrix, PZMetadata]] = []
    for i, p in enumerate(paths):
        pdir = pathlib.Path(os.fspath(p))
        if show_progress:
            print(f"[{i+1}/{len(paths)}] {pdir.name}", file=sys.stderr, flush=True)
        target = pdir / f"{matrix_name}.1pz"
        if not target.is_file():
            if show_progress:
                print(f"  skip — {target.name} not present", file=sys.stderr)
            continue
        mat, meta = read_matrix(target)
        out.append((mat, meta))
    return out


def read_dir(
    path: PathLike,
    *,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> PipelineDirectory:
    """Read every ``.1pz`` file in a pipeline output directory.

    Parameters
    ----------
    path
        Directory containing ``.1pz`` files (the per-sample output tree
        produced by :file:`pilot_job.sh`).
    include
        Optional allow-list of matrix names to load (without the ``.1pz``
        suffix). ``None`` means load everything.
    exclude
        Optional deny-list of matrix names to skip. Applied after
        ``include``.

    Returns
    -------
    PipelineDirectory
        Container whose ``matrices`` attribute is a dict keyed by
        matrix name.

    Examples
    --------
    >>> dd = read_dir("quant/scrna/GSE174/GSE174399/GSM5293863")
    >>> dd["gene_counts"].shape
    (38606, 16079)
    >>> dd.metadata.user_kv["gsm_id"]
    'GSM5293863'
    """
    p = pathlib.Path(os.fspath(path))
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")

    files = sorted(q for q in p.iterdir() if q.suffix == ".1pz")
    if not files:
        raise FileNotFoundError(f"no .1pz files in {p}")

    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()

    matrices: Dict[str, sp.csc_matrix] = {}
    matrix_rownames: Dict[str, List[str]] = {}
    meta: Optional[PZMetadata] = None

    for f in files:
        name = f.stem  # "gene_counts" etc.
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        mat, m = read_matrix(f)
        matrices[name] = mat
        matrix_rownames[name] = m.rownames
        if meta is None:
            meta = m
        elif not meta.user_kv and m.user_kv:
            meta = m  # prefer a metadata block that actually has user_kv

    if meta is None:
        meta = PZMetadata()

    return PipelineDirectory(
        path=p,
        matrices=matrices,
        metadata=meta,
        matrix_rownames=matrix_rownames,
    )
