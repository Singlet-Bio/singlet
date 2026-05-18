# SPDX-License-Identifier: MIT
"""singlet.io.sample — Canonical v2 per-sample reader API.

Loads a singlet v2 sample directory (see ``docs/CANONICAL_OUTPUT_FORMAT.md``)
and exposes lazy CSC views over each row block. No data is decompressed
until a view is materialized.

Per-sample layout (always present)::

    {sample_dir}/
    ├── counts.1pz                  # exon_body | intron_body | junctions
    ├── snp.1pz                     # snp_alt_ad + snp_dp (two-layer CSC)
    ├── mt.1pz                      # mt_alt_ad  + mt_dp  (two-layer CSC)
    ├── cell_meta.parquet
    ├── summary.json
    ├── saturation_curve.tsv
    └── star_Log.final.out

Optional siblings: ``nonhost.json``, ``nonhost_species.1pz``,
``donor_consensus.fa``, ``donor_variants.vcf.gz``, ``ambient_profile.npy``,
``splice_events.tsv``, ``guides.1pz``, ``antibodies.1pz``,
``vdj_gene_usage.1pz``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from singlet.pz_v2 import PzReader, read_pz_v2

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix


__all__ = [
    "SingletSample",
    "SingletCounts",
    "SingletSnp",
    "SingletMt",
    "SingletNonhost",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _safe_vaf(ad: "csc_matrix", dp: "csc_matrix") -> "csc_matrix":
    """Element-wise ``ad / dp`` over the shared sparsity pattern.

    Returns a CSC matrix; entries where ``dp == 0`` are dropped (i.e.
    become implicit zero, which downstream code must interpret as
    "undefined" rather than "0%").
    """
    from scipy.sparse import coo_matrix

    if ad.shape != dp.shape:
        raise ValueError("ad/dp shape mismatch")
    if ad.nnz != dp.nnz:
        raise ValueError(
            "ad/dp nnz mismatch — two-layer CSC requires shared sparsity"
        )
    defined = dp.data > 0
    ratio = np.zeros_like(ad.data, dtype=np.float32)
    ratio[defined] = ad.data[defined].astype(np.float32) / dp.data[defined].astype(
        np.float32
    )
    coo = ad.tocoo(copy=False)
    return coo_matrix(
        (ratio[defined], (coo.row[defined], coo.col[defined])),
        shape=ad.shape,
    ).tocsc()


# --------------------------------------------------------------------------
# Row-block readers
# --------------------------------------------------------------------------


class SingletCounts:
    """Reader for ``counts.1pz`` — the three-block gene-expression tensor.

    Three row blocks satisfy the partition invariant: for every cell
    ``c``, ``sum(exon_body[:, c]) + sum(intron_body[:, c]) +
    sum(junctions[:, c]) == n_umi(c)``. Each block shares the
    cell-barcode axis but stores its own CSC arrays.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader: Optional[PzReader] = None

    def _open(self) -> PzReader:
        if self._reader is None:
            self._reader = read_pz_v2(self.path)
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> "SingletCounts":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def barcodes(self) -> List[str]:
        return list(self._open().cell_barcodes)

    @property
    def n_cells(self) -> int:
        return self._open().n_cells

    @property
    def block_names(self) -> List[str]:
        return self._open().block_names

    def _block(self, name: str) -> "csc_matrix":
        from scipy.sparse import csc_matrix as _csc

        rd = self._open()
        if name not in rd.block_names:
            return _csc((0, rd.n_cells), dtype=np.int32)
        return rd.block(name).data()

    def exon_body(self) -> "csc_matrix":
        """Per-exon-interval UMI counts (rows = ``features.fbin`` exon order)."""
        return self._block("exon_body")

    def intron_body(self) -> "csc_matrix":
        """Per-intron-interval UMI counts."""
        return self._block("intron_body")

    def junctions(self) -> "csc_matrix":
        """Per-junction UMI counts."""
        return self._block("junctions")


class _TwoLayerReader:
    """Shared implementation for SNP and mt — two-layer CSC (ad, dp)."""

    _block_name: str = ""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._reader: Optional[PzReader] = None

    def _open(self) -> PzReader:
        if self._reader is None:
            self._reader = read_pz_v2(self.path)
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def barcodes(self) -> List[str]:
        return list(self._open().cell_barcodes)

    def ad(self) -> "csc_matrix":
        """Alternate-allele read depth per (site, cell)."""
        return self._open().block(self._block_name).data("ad")

    def dp(self) -> "csc_matrix":
        """Total read depth per (site, cell)."""
        return self._open().block(self._block_name).data("dp")

    def vaf(self) -> "csc_matrix":
        """Variant-allele frequency = ad / dp. Only defined where dp > 0."""
        return _safe_vaf(self.ad(), self.dp())


class SingletSnp(_TwoLayerReader):
    """Reader for ``snp.1pz``. Rows = ``snp_sites.fbin`` order."""

    _block_name = "snp"


class SingletMt(_TwoLayerReader):
    """Reader for ``mt.1pz``. Rows = mitochondrial genome positions."""

    _block_name = "mt"


class SingletNonhost:
    """Reader for ``nonhost.json`` + ``nonhost_species.1pz``.

    Holds raw Kraken2 + Bracken outputs (no filtering, no z-scores).
    Rows of the per-cell matrix are NCBI taxids detected in this sample
    (sample-specific); the canonical taxid → row mapping lives in the
    JSON's ``species`` array.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._summary_cache: Optional[dict] = None

    def summary(self) -> dict:
        """Parsed ``nonhost.json`` — bulk Kraken2 + Bracken metrics."""
        if self._summary_cache is None:
            with open(self.path / "nonhost.json", "r") as f:
                self._summary_cache = json.load(f)
        return self._summary_cache

    def species_table(self):
        """Per-species summary as a pandas DataFrame.

        Columns: ``row``, ``taxid``, ``name``, ``rank``, ``kraken_reads``,
        ``kraken_kmer_hits``, ``bracken_reads``, ``bracken_abundance``,
        ``lineage``. Missing columns are filled with ``None``.
        """
        import pandas as pd

        rows = self.summary().get("species", [])
        df = pd.DataFrame(rows)
        expected = [
            "row",
            "taxid",
            "name",
            "rank",
            "kraken_reads",
            "kraken_kmer_hits",
            "bracken_reads",
            "bracken_abundance",
            "lineage",
        ]
        for c in expected:
            if c not in df.columns:
                df[c] = None
        return df[expected]

    def per_cell(self) -> "csc_matrix":
        """Per-cell × taxid read counts. Row order matches the
        ``species`` array in :meth:`summary`."""
        from scipy.sparse import csc_matrix as _csc

        path = self.path / "nonhost_species.1pz"
        if not path.exists():
            n_species = len(self.summary().get("species", []))
            return _csc((n_species, 0), dtype=np.int32)
        with read_pz_v2(path) as rd:
            return rd.block(rd.block_names[0]).data()


# --------------------------------------------------------------------------
# Top-level sample reader
# --------------------------------------------------------------------------


class SingletSample:
    """Top-level reader for one canonical v2 sample directory.

    Lazy: nothing is read until a sub-reader or :attr:`summary` is touched.

    Examples
    --------
    >>> from singlet.io import SingletSample
    >>> s = SingletSample("/path/to/sample")  # doctest: +SKIP
    >>> s.counts.exon_body().shape            # doctest: +SKIP
    (n_exons, n_cells)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not (self.path / "summary.json").exists():
            raise FileNotFoundError(
                f"Not a singlet v2 sample directory (no summary.json): {self.path}"
            )

    @property
    def summary(self) -> dict:
        """Parsed ``summary.json``."""
        with open(self.path / "summary.json", "r") as f:
            return json.load(f)

    @property
    def counts(self) -> SingletCounts:
        return SingletCounts(self.path / "counts.1pz")

    @property
    def snp(self) -> SingletSnp:
        return SingletSnp(self.path / "snp.1pz")

    @property
    def mt(self) -> SingletMt:
        return SingletMt(self.path / "mt.1pz")

    @property
    def cell_meta(self):
        """``cell_meta.parquet`` as a pyarrow Table."""
        import pyarrow.parquet as pq

        return pq.read_table(self.path / "cell_meta.parquet")

    @property
    def nonhost(self) -> Optional[SingletNonhost]:
        if not (self.path / "nonhost.json").exists():
            return None
        return SingletNonhost(self.path)

    @property
    def guides_path(self) -> Optional[Path]:
        p = self.path / "guides.1pz"
        return p if p.exists() else None

    @property
    def antibodies_path(self) -> Optional[Path]:
        p = self.path / "antibodies.1pz"
        return p if p.exists() else None

    @property
    def vdj_path(self) -> Optional[Path]:
        p = self.path / "vdj_gene_usage.1pz"
        return p if p.exists() else None

    def __repr__(self) -> str:
        return f"SingletSample({str(self.path)!r})"
