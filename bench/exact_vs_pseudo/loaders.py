"""Load per-cell gene UMI matrices from Singlet, STARsolo, alevin-fry (USA) and a published Cell Ranger h5.

Every loader returns a `Mat` (cells x genes, CSR, UMI counts) keyed by Ensembl gene id and bare barcode (no `-1`).
Standard library + numpy/scipy/h5py only (the `pbench` conda env on Clipper has them).
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass

import numpy as np
import scipy.io
import scipy.sparse as sp


@dataclass
class Mat:
    barcodes: np.ndarray      # (n_cells,) str
    genes: np.ndarray         # (n_genes,) Ensembl ids (version stripped)
    X: sp.csr_matrix          # cells x genes
    names: np.ndarray | None = None

    def subset_cells(self, bcs) -> "Mat":
        idx = {b: i for i, b in enumerate(self.barcodes)}
        rows = [idx[b] for b in bcs if b in idx]
        return Mat(self.barcodes[rows], self.genes, self.X[rows], self.names)


def _bare(bc: str) -> str:
    return bc.split("-")[0]


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _read_tsv(path, col=0):
    with _open(path) as f:
        return np.array([ln.rstrip("\n").split("\t")[col] for ln in f])


def _ens(x):
    import re
    return np.array([re.sub(r"^(hg19|mm10)_", "", g).split(".")[0] for g in x])


def load_10x_dir(d, genes_by_rows=True):
    """A Cell Ranger / STARsolo style directory (matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz])."""
    def f(stem):
        for ext in ("", ".gz"):
            if os.path.exists(os.path.join(d, stem + ext)):
                return os.path.join(d, stem + ext)
        raise FileNotFoundError(os.path.join(d, stem))
    M = scipy.io.mmread(f("matrix.mtx")).tocsr()
    bcs = np.array([_bare(b) for b in _read_tsv(f("barcodes.tsv"))])
    feat = f("features.tsv") if any(os.path.exists(os.path.join(d, "features.tsv" + e)) for e in ("", ".gz")) else f("genes.tsv")
    genes = _ens(_read_tsv(feat, 0))
    names = _read_tsv(feat, 1)
    X = M.T.tocsr() if genes_by_rows else M.tocsr()
    return Mat(bcs, genes, X.astype(np.float64).tocsr(), names)


def load_singlet(d):
    """Singlet `raw_feature_bc_matrix`: gene x barcode, exon + intron molecules counted once (docs/pipeline-validation.md)."""
    return load_10x_dir(os.path.join(d, "raw_feature_bc_matrix"))


def load_starsolo(d, feature="GeneFull"):
    return load_10x_dir(os.path.join(d, "Solo.out", feature, "raw"))


def load_starsolo_velocyto(d, layer):
    return load_10x_dir(os.path.join(d, "Solo.out", "Velocyto", "raw"), genes_by_rows=True) if layer is None else _velo(d, layer)


def _velo(d, layer):
    r = os.path.join(d, "Solo.out", "Velocyto", "raw")
    M = scipy.io.mmread(os.path.join(r, layer + ".mtx")).tocsr()
    bcs = np.array([_bare(b) for b in _read_tsv(os.path.join(r, "barcodes.tsv"))])
    genes = _ens(_read_tsv(os.path.join(r, "features.tsv"), 0))
    return Mat(bcs, genes, M.T.tocsr().astype(np.float64))


def load_alevinfry(d, quant="af_quant", part="usa"):
    """alevin-fry USA: columns are [all spliced | all unspliced | all ambiguous] in gene order.

    part: 'usa' = S+U+A (the GeneFull / exon+intron analogue), 'sa' = S+A (the exonic analogue),
    'S', 'U', 'A' = single layers.
    """
    q = os.path.join(d, "af", quant, "alevin")
    M = scipy.io.mmread(os.path.join(q, "quants_mat.mtx")).tocsr()
    bcs = np.array([_bare(b) for b in _read_tsv(os.path.join(q, "quants_mat_rows.txt"))])
    cols = _read_tsv(os.path.join(q, "quants_mat_cols.txt"))
    g = len(cols) // 3
    genes = _ens(cols[:g])
    S, U, A = M[:, :g], M[:, g:2 * g], M[:, 2 * g:]
    X = {"usa": S + U + A, "sa": S + A, "S": S, "U": U, "A": A}[part]
    return Mat(bcs, genes, sp.csr_matrix(X, dtype=np.float64))


def load_cr_h5(path, feature_type="Gene Expression"):
    import h5py
    with h5py.File(path, "r") as h:
        m = h["matrix"]
        data, ind, ptr = m["data"][:], m["indices"][:], m["indptr"][:]
        shape = tuple(m["shape"][:])
        bcs = np.array([b.decode() for b in m["barcodes"][:]])
        ids = np.array([b.decode() for b in m["features"]["id"][:]])
        names = np.array([b.decode() for b in m["features"]["name"][:]])
        ft = np.array([b.decode() for b in m["features"]["feature_type"][:]])
    X = sp.csc_matrix((data, ind, ptr), shape=shape).T.tocsr().astype(np.float64)   # cells x features
    keep = np.where(ft == feature_type)[0]
    return Mat(np.array([_bare(b) for b in bcs]), _ens(ids[keep]), X[:, keep].tocsr(), names[keep])


def gene_chrom(gtf):
    """gene_id -> (chrom, gene_name, gene_type) from a GTF."""
    out = {}
    with _open(gtf) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.split("\t")
            if len(p) < 9 or p[2] != "gene":
                continue
            a = p[8]
            def tag(k):
                i = a.find(k + ' "')
                return a[i + len(k) + 2:a.find('"', i + len(k) + 2)] if i >= 0 else ""
            out[tag("gene_id").split(".")[0]] = (p[0], tag("gene_name"), tag("gene_type") or tag("gene_biotype"))
    return out
