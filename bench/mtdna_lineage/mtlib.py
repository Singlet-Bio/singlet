"""Shared helpers for the mtDNA lineage benchmark (docs/mtdna-lineage.md).

Count tensors come from mtpile (bench/mtdna_lineage/mtpile.cpp): records {cell, pos, count[8]}, count[b + 4*rev] with
b = A,C,G,T and pos 0-based on rCRS (= GRCh38 chrM). A `Pile` keeps them as flat arrays; site matrices (AD/DP, cells x
sites) are built on demand.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

L = 16569
BASES = "ACGT"
REC = np.dtype([("cell", "<u4"), ("pos", "<u4"), ("c", "<u4", (8,))])


def read_fasta(path: str) -> str:
    op = gzip.open if path.endswith(".gz") else open
    seq = []
    with op(path, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip().upper())
    return "".join(seq)


@dataclass
class Pile:
    n_cells: int
    cell: np.ndarray   # u4 per record
    pos: np.ndarray    # u4 per record
    c: np.ndarray      # (n, 8) u4

    @staticmethod
    def load(path: str) -> "Pile":
        with open(path, "rb") as f:
            assert f.read(4) == b"MTP1"
            nc = int(np.frombuffer(f.read(4), "<u4")[0])
            n = int(np.frombuffer(f.read(8), "<u8")[0])
            r = np.frombuffer(f.read(n * REC.itemsize), REC)
        return Pile(nc, r["cell"].copy(), r["pos"].copy(), r["c"].astype(np.int64))

    def subset_cells(self, keep: np.ndarray) -> "Pile":
        """keep: old cell indices in the new order."""
        m = np.full(self.n_cells, -1, np.int64)
        m[keep] = np.arange(len(keep))
        new = m[self.cell]
        ok = new >= 0
        return Pile(len(keep), new[ok].astype(np.uint32), self.pos[ok], self.c[ok])

    def thin(self, frac: float, seed: int = 0) -> "Pile":
        """Binomial thinning of every count (molecule-level downsampling)."""
        rng = np.random.default_rng(seed)
        c = rng.binomial(self.c, frac)
        ok = c.sum(1) > 0
        return Pile(self.n_cells, self.cell[ok], self.pos[ok], c[ok])

    def by_pos(self):
        """(order, ptr): records sorted by position; records at pos p are order[ptr[p]:ptr[p+1]]."""
        if getattr(self, "_bp", None) is None:
            order = np.argsort(self.pos, kind="stable")
            ptr = np.concatenate([[0], np.cumsum(np.bincount(self.pos, minlength=L))])
            self._bp = (order, ptr)
        return self._bp

    def depth(self) -> np.ndarray:
        return self.c.sum(1)

    def cov_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.depth(), (self.cell, self.pos)), shape=(self.n_cells, L))

    def base_matrix(self, b: int, strand: str = "both") -> sp.csr_matrix:
        if strand == "fwd":
            v = self.c[:, b]
        elif strand == "rev":
            v = self.c[:, b + 4]
        else:
            v = self.c[:, b] + self.c[:, b + 4]
        ok = v > 0
        return sp.csr_matrix((v[ok], (self.cell[ok], self.pos[ok])), shape=(self.n_cells, L))

    def pseudobulk(self, cells: np.ndarray | None = None) -> np.ndarray:
        """(L, 8) summed counts over the given cells (all if None)."""
        sel = np.ones(len(self.cell), bool) if cells is None else np.isin(self.cell, cells)
        out = np.zeros((L, 8), np.int64)
        np.add.at(out, self.pos[sel], self.c[sel])
        return out


def site_matrices(p: Pile, sites: list[tuple[int, int]]):
    """AD, DP (cells x sites, CSC) for sites [(pos0, alt_base_idx)]; plus fwd/rev alt counts."""
    order, ptr = p.by_pos()
    cell_s, c_s = p.cell[order], p.c[order]
    rows_ad, cols_ad, ad, adf, adr = [], [], [], [], []
    rows_dp, cols_dp, dp = [], [], []
    for j, (ps, b) in enumerate(sites):
        sl = slice(ptr[ps], ptr[ps + 1])
        cc, cm = cell_s[sl], c_s[sl]
        if len(cc) == 0:
            continue
        rows_dp.append(cc); cols_dp.append(np.full(len(cc), j)); dp.append(cm.sum(1))
        a = cm[:, b] + cm[:, b + 4]
        k = a > 0
        rows_ad.append(cc[k]); cols_ad.append(np.full(k.sum(), j)); ad.append(a[k])
        adf.append(cm[k, b]); adr.append(cm[k, b + 4])
    shape = (p.n_cells, len(sites))
    cat = lambda x: np.concatenate(x) if x else np.zeros(0, np.int64)
    AD = sp.csc_matrix((cat(ad), (cat(rows_ad), cat(cols_ad))), shape=shape)
    DP = sp.csc_matrix((cat(dp), (cat(rows_dp), cat(cols_dp))), shape=shape)
    ADF = sp.csc_matrix((cat(adf), (cat(rows_ad), cat(cols_ad))), shape=shape)
    ADR = sp.csc_matrix((cat(adr), (cat(rows_ad), cat(cols_ad))), shape=shape)
    return AD, DP, ADF, ADR


def candidate_sites(p: Pile, ref: str, min_cells: int = 2, min_alt: int = 1):
    """All (pos, alt) with alt counts >= min_alt in >= min_cells cells; alt != reference base."""
    refidx = np.array([BASES.find(x) for x in ref[:L]])
    out = []
    for b in range(4):
        a = p.c[:, b] + p.c[:, b + 4]
        m = (a >= min_alt) & (refidx[p.pos] != b) & (refidx[p.pos] >= 0)
        cnt = np.bincount(p.pos[m], minlength=L)
        for ps in np.nonzero(cnt >= min_cells)[0]:
            out.append((int(ps), b))
    out.sort()
    return out


def mgatk_stats(p: Pile, ref: str, sites=None, low_cov: int = 2):
    """Per-variant statistics as computed by mgatk / Signac IdentifyVariants:
    mean_af, VMR (variance / mean of per-cell AF over all cells, AF = 0 where uncovered), strand_correlation (Pearson
    of fwd vs rev alt counts over cells with any alt read), n_cells_conf_detected (alt >= low_cov on both strands),
    n_cells_over_{0.05,0.1,0.2,0.5,0.9}.
    """
    if sites is None:
        sites = candidate_sites(p, ref)
    AD, DP, ADF, ADR = site_matrices(p, sites)
    n = p.n_cells
    rows = []
    DPc = DP.tocsc(); ADc = AD.tocsc(); Fc = ADF.tocsc(); Rc = ADR.tocsc()
    for j, (ps, b) in enumerate(sites):
        dsl = slice(DPc.indptr[j], DPc.indptr[j + 1])
        asl = slice(ADc.indptr[j], ADc.indptr[j + 1])
        dcell, dval = DPc.indices[dsl], DPc.data[dsl]
        acell, aval = ADc.indices[asl], ADc.data[asl]
        af = np.zeros(n)
        dd = np.zeros(n); dd[dcell] = dval
        af[acell] = aval / np.maximum(dd[acell], 1)
        mean = af.mean(); var = af.var(ddof=1) if n > 1 else 0.0
        f = Fc.data[asl]; r = Rc.data[asl]
        sc = np.corrcoef(f, r)[0, 1] if len(f) > 2 and f.std() > 0 and r.std() > 0 else np.nan
        conf = int(((f >= low_cov) & (r >= low_cov)).sum())
        rows.append(dict(pos=ps + 1, ref=ref[ps], alt=BASES[b], variant=f"{ps + 1}{ref[ps]}>{BASES[b]}",
                         mean_af=mean, vmr=var / mean if mean > 0 else np.nan, strand_correlation=sc,
                         n_cells_conf_detected=conf, n_cells_alt=len(acell), n_cells_cov=len(dcell),
                         mean_cov=dval.sum() / n, alt_fwd=int(f.sum()), alt_rev=int(r.sum()),
                         ref_fwd=None, **{f"n_cells_over_{t}": int((af >= t).sum()) for t in (0.05, 0.1, 0.2, 0.5, 0.9)}))
    import pandas as pd
    df = pd.DataFrame(rows)
    df["mgatk_pass"] = (df.n_cells_conf_detected >= 5) & (df.strand_correlation >= 0.65) & (np.log10(df.vmr) > -2)
    return df, (AD, DP)


def write_mtx_pair(AD: sp.spmatrix, DP: sp.spmatrix, prefix: str):
    """MQuad / vireo input: variants x cells Matrix Market files."""
    import scipy.io
    scipy.io.mmwrite(prefix + ".AD.mtx", sp.coo_matrix(AD.T).astype(np.int64), field="integer")
    scipy.io.mmwrite(prefix + ".DP.mtx", sp.coo_matrix(DP.T).astype(np.int64), field="integer")


def clone_fit(AD, DP, ks=range(2, 9), n_init=50, seed=0):
    """vireoSNP binomial mixture (cardelino-style, no tree) over cells; AD/DP cells x variants.
    Returns dict k -> (labels, ELBO, model); the caller picks k."""
    from vireoSNP import BinomMixtureVB
    ADv = sp.csr_matrix(AD.T); DPv = sp.csr_matrix(DP.T)
    out = {}
    for k in ks:
        m = BinomMixtureVB(n_var=ADv.shape[0], n_cell=ADv.shape[1], n_donor=k)
        m.fit(ADv, DPv, min_iter=30, n_init=n_init, max_iter=500, random_seed=seed)
        out[k] = (m.ID_prob.argmax(1), float(m.ELBO_iters[-1]), m)
    return out


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def matched_f1(truth: np.ndarray, pred: np.ndarray):
    """Hungarian-match predicted clusters to truth labels; per-truth-label precision / recall / F1."""
    from scipy.optimize import linear_sum_assignment
    tl = np.unique(truth); pl = np.unique(pred)
    M = np.array([[np.sum((truth == t) & (pred == q)) for q in pl] for t in tl])
    r, c = linear_sum_assignment(-M)
    res = {}
    for i, j in zip(r, c):
        tp = M[i, j]; prec = tp / max(M[:, j].sum(), 1); rec = tp / max(M[i].sum(), 1)
        res[str(tl[i])] = dict(precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / max(prec + rec, 1e-12)),
                               n_truth=int(M[i].sum()))
    for i in set(range(len(tl))) - set(r):
        res[str(tl[i])] = dict(precision=0.0, recall=0.0, f1=0.0, n_truth=int(M[i].sum()))
    return res


def dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
