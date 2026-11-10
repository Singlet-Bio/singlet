# SPDX-License-Identifier: MIT
"""Readers for the tool outputs of the public benchmark runs (Clipper side).

Everything here returns gene- or junction-level sparse matrices aligned to a requested list of barcodes. Only counts are read:
Singlet's ``private/`` directory (donor genotypes) is never opened. numpy, scipy and pandas only.

Barcodes are the bare 14/16-mer without the ``-1`` suffix; gene ids are Ensembl ids without version and without a
species prefix (``canon_gene``).
"""
from __future__ import annotations

import gzip
import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

_GENE_RE = re.compile(r"(ENS(?:MUS)?G\d{11}(?:_PAR_Y)?)")


def bare(bc: str) -> str:
    return bc.split("-")[0]


def canon_gene(g: str) -> str:
    """Ensembl gene id without version or species prefix ('GRCh38_ENSG..', 'mm10___ENSMUSG..', 'ENSG...13')."""
    m = _GENE_RE.search(g)
    return m.group(1) if m else g.split(".")[0]


def opener(p: Path):
    p = Path(p)
    with open(p, "rb") as f:
        magic = f.read(2)
    return gzip.open(p, "rt") if magic == b"\x1f\x8b" else open(p, "rt")


def read_lines(p: Path) -> list[str]:
    with opener(p) as f:
        return [ln.rstrip("\n") for ln in f]


def read_mtx(p: Path) -> sp.csc_matrix:
    with opener(p) as f:
        return sp.csc_matrix(scipy.io.mmread(f))


def aggregator(groups: list[str]) -> tuple[list[str], sp.csr_matrix]:
    """Sparse (n_groups x n_items) 0/1 matrix summing items into groups; returns (sorted group ids, matrix)."""
    uniq = sorted(set(groups))
    idx = {g: i for i, g in enumerate(uniq)}
    rows = np.fromiter((idx[g] for g in groups), dtype=np.int64, count=len(groups))
    return uniq, sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, np.arange(len(rows)))),
                               shape=(len(uniq), len(rows)))


class GeneMat:
    """cells x genes counts for an ordered list of cells; ``present[i]`` is False when cell i is not in the tool's output."""

    def __init__(self, genes: list[str], X: sp.csr_matrix, cells: list[str], present: np.ndarray):
        self.genes = np.asarray(genes)
        self.X = sp.csr_matrix(X)
        self.cells = cells
        self.present = np.asarray(present, bool)

    def totals(self) -> np.ndarray:
        t = np.asarray(self.X.sum(axis=1)).ravel().astype(float)
        t[~self.present] = np.nan
        return t

    def pad(self, genes: list[str]) -> "GeneMat":
        """The same counts on the given gene axis: genes the tool did not report (no features of that kind) become zeros."""
        pos = {g: i for i, g in enumerate(genes)}
        keep = [i for i, g in enumerate(self.genes) if g in pos]
        c = self.X[:, keep].tocoo()
        cols = np.array([pos[self.genes[i]] for i in keep], dtype=np.int64)[c.col]
        return GeneMat(genes, sp.csr_matrix((c.data, (c.row, cols)), shape=(self.X.shape[0], len(genes))), self.cells, self.present)

    def add(self, other: "GeneMat") -> "GeneMat":
        genes = sorted(set(self.genes) | set(other.genes))
        return GeneMat(genes, _reindex(self, genes) + _reindex(other, genes), self.cells, self.present | other.present)


def _reindex(m: GeneMat, genes: list[str]) -> sp.csr_matrix:
    pos = {g: i for i, g in enumerate(genes)}
    cols = np.fromiter((pos[g] for g in m.genes), dtype=np.int64, count=len(m.genes))
    c = m.X.tocoo()
    return sp.csr_matrix((c.data, (c.row, cols[c.col])), shape=(m.X.shape[0], len(genes)))


def _cells_x_genes(M_gc: sp.csc_matrix, bcs: list[str], genes: list[str], cells: list[str]) -> GeneMat:
    """Pick the requested cells out of a genes x barcodes matrix, collapsing duplicate canonical gene ids."""
    pos = {}
    for i, b in enumerate(bcs):
        pos.setdefault(bare(b), i)
    take = [pos.get(c, -1) for c in cells]
    present = np.array([t >= 0 for t in take])
    sel = [t for t in take if t >= 0]
    sub = M_gc[:, sel].tocsr()
    uniq, agg = aggregator([canon_gene(g) for g in genes])
    sub = sp.csr_matrix(agg @ sub)  # genes x selected cells
    out = sub.T.tocsr()
    rows = np.nonzero(present)[0]
    # place the selected rows back in the order of `cells`
    P = sp.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(cells), len(rows)))
    X = sp.csr_matrix(P @ out)
    return GeneMat(uniq, X, cells, present)


# --------------------------------------------------------------------------------------------------------- Singlet
def singlet_known_genes(run: Path) -> set[str]:
    f = pd.read_csv(run / "filtered_feature_bc_matrix" / "features.tsv.gz", sep="\t", header=None)
    return set(f[0].astype(str))


def _singlet_gene_of(known: set[str]):
    def gene_of(name: str) -> str:
        parts = name.split("_")
        if len(parts) > 2 and parts[1] == "PAR" and (parts[0] + "_PAR_" + parts[2]) in known:
            return parts[0] + "_PAR_" + parts[2]
        return parts[0] if parts[0] in known else canon_gene(name)

    return gene_of


def load_singlet(run: Path, cells: list[str]) -> dict[str, GeneMat]:
    """Gene-level Singlet layers for ``cells``: E (exon_counts = spliced + ambiguous), U (intron_counts = unspliced),
    A (ambiguous_counts). Spliced S = E - A and total = E + U are derived by the callers."""
    known = singlet_known_genes(run)
    gene_of = _singlet_gene_of(known)
    out = {}
    for key, prefix in (("E", "exon_counts"), ("U", "intron_counts"), ("A", "ambiguous_counts")):
        feats = read_lines(run / f"{prefix}_features.tsv.gz")
        bcs = read_lines(run / f"{prefix}_barcodes.tsv.gz")
        M = read_mtx(run / f"{prefix}.mtx.gz")
        genes = [gene_of(f) for f in feats]
        out[key] = _cells_x_genes(M, bcs, genes, cells)
    return out


def load_singlet_raw_total(run: Path) -> tuple[list[str], list[str], sp.csc_matrix]:
    """Singlet raw gene x barcode matrix (exon + intron, every barcode with >= 100 reads): (genes, barcodes, matrix)."""
    d = run / "raw_feature_bc_matrix"
    feats = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None)[0].astype(str).tolist()
    bcs = [bare(b) for b in read_lines(d / "barcodes.tsv.gz")]
    return feats, bcs, read_mtx(d / "matrix.mtx.gz")


# -------------------------------------------------------------------------------------------------------- STARsolo
def starsolo_cells(run: Path, feature: str = "Gene") -> list[str]:
    return [bare(b) for b in read_lines(run / "Solo.out" / feature / "filtered" / "barcodes.tsv")]


def load_starsolo(run: Path, feature: str, cells: list[str], layer: str = "matrix") -> GeneMat:
    """Solo.out/<feature>/raw. feature in Gene / GeneFull; Velocyto layers via feature='Velocyto', layer in
    spliced / unspliced / ambiguous."""
    d = run / "Solo.out" / feature / "raw"
    feats = pd.read_csv(d / "features.tsv", sep="\t", header=None)[0].astype(str).tolist()
    bcs = read_lines(d / "barcodes.tsv")
    M = read_mtx(d / f"{layer}.mtx")
    return _cells_x_genes(M, bcs, feats, cells)


# ------------------------------------------------------------------------------------------------------ alevin-fry
def load_alevinfry(run: Path, cells: list[str]) -> dict[str, GeneMat]:
    """simpleaf USA output: quants_mat.mtx is barcodes x (S genes | U genes | A genes)."""
    q = run / "af" / "af_quant" / "alevin"
    m = sp.csr_matrix(scipy.io.mmread(str(q / "quants_mat.mtx")))
    cols = read_lines(q / "quants_mat_cols.txt")
    rows = read_lines(q / "quants_mat_rows.txt")
    if m.shape[1] == len(cols) and len(cols) % 3 == 0 and cols[len(cols) // 3].endswith("-U"):
        cols = cols[: len(cols) // 3]  # the file lists S genes, then '<gene>-U', then '<gene>-A'
    n = len(cols)
    if m.shape[1] != 3 * n:
        raise ValueError(f"alevin-fry matrix {m.shape} does not hold 3 x {n} genes")
    layers = {}
    for key, sl in (("S", slice(0, n)), ("U", slice(n, 2 * n)), ("A", slice(2 * n, 3 * n))):
        layers[key] = _cells_x_genes(sp.csc_matrix(m[:, sl].T), rows, cols, cells)
    return layers


# ----------------------------------------------------------------------------------------------- Cell Ranger (10x)
def load_cellranger_published(data_dir: Path, cells: list[str]) -> tuple[GeneMat | None, list[str]]:
    """The 10x-published filtered matrix shipped with the dataset (h5, or the Cell Ranger 2 tar.gz). Returns the gene
    expression matrix for ``cells`` and all of the file's barcodes (the published cell calls)."""
    h5 = next(iter(sorted(data_dir.glob("cellranger_*.h5"))), None)
    if h5 is not None:
        import h5py

        with h5py.File(h5, "r") as f:
            g = f["matrix"]
            M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
            bcs = [b.decode() for b in g["barcodes"][:]]
            ids = [x.decode() for x in g["features"]["id"][:]]
            ftype = np.array([x.decode() for x in g["features"]["feature_type"][:]])
        keep = np.nonzero(ftype == "Gene Expression")[0]
        return _cells_x_genes(sp.csc_matrix(M[keep]), bcs, [ids[i] for i in keep], cells), [bare(b) for b in bcs]
    tgz = next(iter(sorted(data_dir.glob("cellranger_*.tar.gz"))), None)
    if tgz is None:
        return None, []
    with tarfile.open(tgz) as t:
        members = {Path(m.name).name: m for m in t.getmembers() if m.isfile()}
        def rd(name):
            return t.extractfile(members[name]).read()
        bcs = rd("barcodes.tsv").decode().split()
        genes = [ln.split("\t")[0] for ln in rd("genes.tsv").decode().splitlines()]
        M = sp.csc_matrix(scipy.io.mmread(io.BytesIO(rd("matrix.mtx"))))
    return _cells_x_genes(M, bcs, genes, cells), [bare(b) for b in bcs]


def cellranger_published_barcodes(data_dir: Path) -> list[str]:
    """Barcodes of the 10x-published filtered matrix (the published cell calls), without reading the counts."""
    h5 = next(iter(sorted(data_dir.glob("cellranger_*.h5"))), None)
    if h5 is not None:
        import h5py

        with h5py.File(h5, "r") as f:
            return [bare(b.decode()) for b in f["matrix"]["barcodes"][:]]
    tgz = next(iter(sorted(data_dir.glob("cellranger_*.tar.gz"))), None)
    if tgz is None:
        return []
    with tarfile.open(tgz) as t:
        m = next(m for m in t.getmembers() if m.isfile() and Path(m.name).name == "barcodes.tsv")
        return [bare(b) for b in t.extractfile(m).read().decode().split()]


# ------------------------------------------------------------------------------------------------ GNU time (-v)
def load_time(p: Path) -> dict:
    d: dict = {}
    if not Path(p).exists():
        return d
    for ln in Path(p).read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock) time"):
            t = ln.split(": ", 1)[1].split(":")
            d["wall_s"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
        elif ln.startswith("Maximum resident set size"):
            d["max_rss_gb"] = int(ln.split(": ")[1]) / 1024 ** 2
        elif ln.startswith("User time"):
            d["user_s"] = float(ln.split(": ")[1])
        elif ln.startswith("System time"):
            d["sys_s"] = float(ln.split(": ")[1])
    if "user_s" in d:
        d["cpu_h"] = (d["user_s"] + d.get("sys_s", 0.0)) / 3600.0
    return d
