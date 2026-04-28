"""
singlify — Streaming BAM pileup for single-cell multi-omics.

A Python wrapper around the C++ singlet-pileup engine, returning results
as scipy sparse matrices ready for downstream analysis with scanpy/anndata.

Quick start::

    import singlify

    result = singlify.pileup(
        bam_path="sample.bam",
        barcode_path="barcodes.tsv",
        exon_gtf_path="genes.gtf.gz",
    )

    # Access as scipy CSC matrices
    exons = result.exons        # features × barcodes
    introns = result.introns
    print(result.stats)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import scipy.sparse as sp
import numpy as np

if TYPE_CHECKING:
    from ._core import PileupStats  # noqa: F401

__version__ = "0.2.0"
__all__ = ["pileup", "PileupResult", "PileupStats", "io"]


def __getattr__(name: str):
    """Lazy import of the pybind11 _core module and submodules.

    The native ``_core`` extension is only required for the live-BAM
    :func:`pileup` entry point. Output readers in :mod:`singlify.io` are
    pure Python and work even if ``_core`` is not built, so we defer the
    import until the symbol is actually needed.
    """
    if name == "PileupStats":
        from ._core import PileupStats
        return PileupStats
    if name == "io":
        from . import io
        return io
    raise AttributeError(f"module 'singlify' has no attribute {name!r}")


class PileupResult:
    """Container for pileup results with scipy sparse matrices.

    Attributes
    ----------
    stats : PileupStats
        Read-level statistics (total_reads, mapped_reads, etc.).
    barcodes : list[str]
        Barcode strings (column labels).
    exons : scipy.sparse.csc_matrix or None
        Exon count matrix (features × barcodes).
    introns : scipy.sparse.csc_matrix or None
        Intron count matrix.
    splice_junctions : scipy.sparse.csc_matrix or None
        Splice junction count matrix.
    snp_ad : scipy.sparse.csc_matrix or None
        SNP alternate allele depth.
    snp_dp : scipy.sparse.csc_matrix or None
        SNP total depth.
    mt_alleles : scipy.sparse.csc_matrix or None
        chrM allele count matrix.
    exon_names : list[str] or None
        Feature names for exon matrix.
    intron_names : list[str] or None
        Feature names for intron matrix.
    sj_names : list[str] or None
        Feature names for splice junction matrix.
    snp_names : list[str] or None
        Feature names for SNP matrices.
    mt_names : list[str] or None
        Feature names for mt allele matrix.
    """

    def __init__(self, raw: dict):
        self.stats: PileupStats = raw["stats"]
        self.barcodes: list[str] = raw["barcodes"]

        self.exons = self._to_csc(raw.get("exons"))
        self.introns = self._to_csc(raw.get("introns"))
        self.splice_junctions = self._to_csc(raw.get("splice_junctions"))
        self.snp_ad = self._to_csc(raw.get("snp_ad"))
        self.snp_dp = self._to_csc(raw.get("snp_dp"))
        self.mt_alleles = self._to_csc(raw.get("mt_alleles"))

        self.exon_names = self._names(raw.get("exons"))
        self.intron_names = self._names(raw.get("introns"))
        self.sj_names = self._names(raw.get("splice_junctions"))
        self.snp_names = self._names(raw.get("snp_ad"))
        self.mt_names = self._names(raw.get("mt_alleles"))

    @staticmethod
    def _to_csc(d: Optional[dict]) -> Optional[sp.csc_matrix]:
        if d is None:
            return None
        return sp.csc_matrix(
            (np.asarray(d["data"]), np.asarray(d["indices"]), np.asarray(d["indptr"])),
            shape=d["shape"],
        )

    @staticmethod
    def _names(d: Optional[dict]) -> Optional[list[str]]:
        if d is None:
            return None
        return d.get("features")

    def __repr__(self) -> str:
        parts = [f"PileupResult(reads={self.stats.total_reads:,}"]
        if self.exons is not None:
            parts.append(f"exons={self.exons.shape[0]:,}×{self.exons.shape[1]:,}")
        if self.introns is not None:
            parts.append(f"introns={self.introns.shape[0]:,}×{self.introns.shape[1]:,}")
        if self.snp_ad is not None:
            parts.append(f"snps={self.snp_ad.shape[0]:,}")
        if self.mt_alleles is not None:
            parts.append(f"mt={self.mt_alleles.shape[0]:,}")
        return ", ".join(parts) + ")"

    def to_anndata(self):
        """Convert exon counts to an AnnData object.

        Requires the ``anndata`` package.

        Returns
        -------
        anndata.AnnData
            AnnData with exon counts in ``.X``, barcodes as ``.obs_names``,
            and feature names as ``.var_names``. Intron and SNP matrices
            are stored in ``.layers`` if available.
        """
        import anndata as ad

        if self.exons is None:
            raise ValueError("No exon data — provide exon_gtf_path to pileup()")

        n_var = self.exons.shape[0]
        adata = ad.AnnData(
            X=self.exons.T.tocsr(),
            obs={"barcode": self.barcodes},
            var={"feature_name": self.exon_names},
        )
        adata.obs_names = self.barcodes
        adata.var_names = self.exon_names

        # Layers require same var dimension as X. Matrices with different
        # feature counts go into .obsm instead.
        if self.introns is not None:
            mat = self.introns.T.tocsr()
            if mat.shape[1] == n_var:
                adata.layers["introns"] = mat
            else:
                adata.obsm["introns"] = mat
        if self.splice_junctions is not None:
            mat = self.splice_junctions.T.tocsr()
            if mat.shape[1] == n_var:
                adata.layers["splice_junctions"] = mat
            else:
                adata.obsm["splice_junctions"] = mat

        return adata


def pileup(
    bam_path: str,
    barcode_path: str,
    exon_gtf_path: str = "",
    snp_path: str = "",
    *,
    threads: int = 4,
    min_mapq: int = 20,
    min_baseq: int = 10,
    count_introns: bool = True,
    count_sj: bool = True,
    umi_dedup: bool = True,
    count_mt: bool = False,
) -> "PileupResult":
    """Run streaming BAM pileup for single-cell multi-omics.

    Processes a BAM file in a single pass, extracting per-barcode counts for
    exon/intron expression, splice junctions, SNP genotypes, and chrM alleles.

    Parameters
    ----------
    bam_path : str
        Path to input BAM file.
    barcode_path : str
        Path to filtered barcode list (one per line).
    exon_gtf_path : str, optional
        GTF for gene model (enables exon, intron, splice junction counting).
    snp_path : str, optional
        VCF of SNP positions (enables genotyping).
    threads : int
        BAM decompression threads.
    min_mapq : int
        Minimum mapping quality filter.
    min_baseq : int
        Minimum base quality for SNP/editing calls.
    count_introns : bool
        Count intronic reads (requires exon_gtf_path).
    count_sj : bool
        Extract splice junctions (requires exon_gtf_path).
    umi_dedup : bool
        Deduplicate by UMI.
    count_mt : bool
        Enable chrM allele pileup for heteroplasmy analysis.

    Returns
    -------
    PileupResult
        Container with scipy CSC matrices and statistics.

    Examples
    --------
    >>> result = singlify.pileup("sample.bam", "barcodes.tsv", "genes.gtf.gz")
    >>> result.exons.shape
    (33538, 5000)
    >>> result.stats.total_reads
    72000000
    """
    if not exon_gtf_path and not snp_path:
        raise ValueError("At least one of exon_gtf_path or snp_path is required")

    from ._core import pileup as _pileup

    raw = _pileup(
        bam_path=bam_path,
        barcode_path=barcode_path,
        exon_gtf_path=exon_gtf_path,
        snp_path=snp_path,
        threads=threads,
        min_mapq=min_mapq,
        min_baseq=min_baseq,
        count_introns=count_introns,
        count_sj=count_sj,
        umi_dedup=umi_dedup,
        count_mt=count_mt,
    )
    return PileupResult(raw)
