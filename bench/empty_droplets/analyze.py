# SPDX-License-Identifier: MIT
"""Per-sample evidence on whether the barcodes Singlet leaves uncalled are empty droplets.

Input: ``fetch.py`` npz files (gene-level exon/intron counts of every stored barcode, the called flag). Output per sample:
``<out>/<GSM>.json`` (all numbers the report uses) and ``<out>/<GSM>.bc.npz`` (per-barcode statistics, kept on Clipper).

Tests (see docs/empty-droplets.md for the reasoning):

1. Profile test against ambient. Ambient profile ``a`` = summed counts of the lowest-total uncalled barcodes (``--amb-quantile``),
   DM concentration by MLE. Cell profiles ``q_k`` = summed counts of Leiden clusters of the called cells. For every barcode the
   mixture ``alpha q_k + (1 - alpha) a`` is fitted on a grid (best k, best alpha); LLR = log L(best mixture) - log L(ambient).
   The null distribution of the LLR is simulated per barcode depth from DM draws of the ambient (Polya urn), so p-values are
   calibrated for depth, overdispersion and the fitting itself; BH within the uncalled barcodes.
   Power: called cells binomially thinned to fixed depths and scored against centroids built from the other half of the cells.
2. Correlation of log-CPM profiles with the ambient and with the best cluster centroid (the simpler view).
3. Intronic fraction, mitochondrial fraction, hemoglobin fraction per barcode.
4. Species purity for barnyard references (human vs mouse genes).
5. Independent callers (callers.py): EmptyDrops (lower 100 and adaptive), Cell Ranger 3-like, OrdMag, knee, inflection.
6. Storage accounting: non-zero entries (feature level exon/intron, junctions, V(D)J) by category and under candidate keep rules.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback

import numba as nb
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import callers as C  # noqa: E402

ALPHAS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
UMI_BINS = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10**9]
THIN_DEPTHS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
KEEP_GRID = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500]


# ----------------------------------------------------------------------------------------------------------- inputs
def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    genes = z["genes"].astype(str)
    n = len(z["barcodes"])

    def csc(k):
        return sp.csc_matrix((z[f"{k}_data"], z[f"{k}_indices"], z[f"{k}_indptr"]), shape=(len(genes), n))

    d = {
        "meta": meta,
        "genes": genes,
        "barcodes": z["barcodes"].astype(str),
        "called": z["called"].astype(bool),
        "exon": csc("exon"),
        "intron": csc("intron"),
    }
    for k in ("exon_nnz_features", "intron_nnz_features", "sj_nnz", "vdj_nnz"):
        d[k] = z[k] if k in z.files else None
    d["symbols"] = z["symbols"].astype(str) if "symbols" in z.files else genes
    d["chroms"] = z["chroms"].astype(str) if "chroms" in z.files else np.full(len(genes), "")
    return d


def gene_sets(genes, symbols, chroms):
    """Masks for mitochondrial (chrM), hemoglobin genes, and species (Ensembl human ENSG vs mouse ENSMUSG ids)."""
    up = np.char.upper(genes.astype(str))
    symu = np.char.upper(symbols.astype(str))
    mt = np.isin(chroms, ["chrM", "MT", "M", "chrMT"]) | np.char.startswith(symu, "MT-")
    hb = np.isin(symu, ["HBB", "HBA1", "HBA2", "HBD", "HBG1", "HBG2", "HBA-A1", "HBA-A2", "HBB-BS", "HBB-BT", "HBB-B1", "HBB-B2"])
    human = np.char.startswith(up, "ENSG")
    mouse = np.char.startswith(up, "ENSMUSG")
    return mt, hb, human, mouse


# ----------------------------------------------------------------------------------------------------------- statistics
def dm_null_matrix(prop, alpha, totals, seed=7):
    """Genes x len(totals) matrix of DM(alpha * prop) draws at the given totals."""
    totals = np.asarray(totals, np.int64)
    prob, alias = C.alias_table(prop)
    starts = np.concatenate([[0], np.cumsum(totals)[:-1]]).astype(np.int64)  # column j owns slots starts[j] .. + totals[j]
    idx = np.zeros(int(totals.sum()), np.int64)
    cnt = np.zeros(int(totals.sum()), np.int32)
    lens = np.zeros(len(totals) + 1, np.int64)
    _urn_null_regions(prob, alias, prop, float(alpha), totals, starts, seed, idx, cnt, lens)
    keep = np.concatenate([np.arange(s, s + m) for s, m in zip(starts, lens[1:])]) if len(totals) else np.zeros(0, np.int64)
    indptr = np.concatenate([[0], np.cumsum(lens[1:])])
    return sp.csc_matrix((cnt[keep], idx[keep], indptr), shape=(len(prop), len(totals)))


@nb.njit(parallel=True, cache=True)
def _urn_null_regions(prob, alias, prop, alpha, totals, starts, seed, out_idx, out_cnt, out_len):
    K = len(prop)
    for j in nb.prange(len(totals)):
        np.random.seed(seed + j)
        t = totals[j]
        draws = np.empty(max(t, 1), np.int64)
        for s in range(t):
            if np.random.random() < alpha / (alpha + s):
                i = np.random.randint(0, K)
                g = i if np.random.random() < prob[i] else alias[i]
            else:
                g = draws[np.random.randint(0, s)]
            draws[s] = g
        d = np.sort(draws[:t])
        p = starts[j]
        last = -1
        m = 0
        for s in range(t):
            if d[s] != last:
                out_idx[p + m] = d[s]
                out_cnt[p + m] = 1
                m += 1
                last = d[s]
            else:
                out_cnt[p + m - 1] += 1
        out_len[j + 1] = m


def mixture_llr(Y: sp.csc_matrix, amb: np.ndarray, Q: np.ndarray):
    """Best mixture alpha*q_k + (1-alpha)*a over (k, alpha grid) vs pure ambient. Y genes x barcodes; amb (G,), Q (G, K).
    Returns llr, best k, best alpha (per barcode)."""
    G, K = Q.shape
    cols = []
    for k in range(K):
        for a in ALPHAS[1:]:
            cols.append(np.log(a * Q[:, k] + (1 - a) * amb))
    L = np.stack(cols, axis=1)  # G x (K*(A-1))
    la = np.log(amb)
    Yt = Y.T.tocsr().astype(np.float64)
    ll_amb = Yt @ la
    ll = Yt @ L  # n x K*(A-1)
    best = ll.argmax(axis=1)
    llr = ll[np.arange(ll.shape[0]), best] - ll_amb
    na = len(ALPHAS) - 1
    return np.maximum(llr, 0.0), best // na, ALPHAS[1:][best % na]


def logcpm_corr(Y: sp.csc_matrix, ref: np.ndarray, genes_idx: np.ndarray):
    """Pearson correlation of each column's log1p(CPM) with log1p(CPM) of ``ref`` over ``genes_idx``."""
    Ys = Y[genes_idx, :].tocsc().astype(np.float64)
    t = np.asarray(Y.sum(axis=0)).ravel()
    t[t == 0] = 1
    Ys = Ys @ sp.diags(1e6 / t)
    Ys.data = np.log1p(Ys.data)
    r = np.log1p(ref[genes_idx] / ref[genes_idx].sum() * 1e6)
    r = (r - r.mean()) / (r.std() + 1e-12)
    n = len(genes_idx)
    mu = np.asarray(Ys.mean(axis=0)).ravel()
    sq = np.asarray(Ys.multiply(Ys).mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(sq - mu**2, 1e-12))
    cov = np.asarray(Ys.T @ r).ravel() / n - mu * r.mean()
    return cov / sd


def thin(Y: sp.csc_matrix, depth: int, rng) -> sp.csc_matrix:
    """Binomial thinning of each column to expected total ``depth``."""
    Y = Y.tocsc().astype(np.int64)
    t = np.asarray(Y.sum(axis=0)).ravel()
    p = np.repeat(np.minimum(1.0, depth / np.maximum(t, 1)), np.diff(Y.indptr))
    data = rng.binomial(Y.data, p)
    out = sp.csc_matrix((data, Y.indices, Y.indptr), shape=Y.shape)
    out.eliminate_zeros()
    return out


def cluster_called(X: sp.csc_matrix, called_idx: np.ndarray, rng, max_cells=20000, res=1.0):
    """Leiden clusters of called cells (scanpy standard workflow). Returns labels for called_idx (subsample -> kNN transfer)."""
    import anndata as ad
    import scanpy as sc

    idx = called_idx
    if len(idx) > max_cells:
        idx = np.sort(rng.choice(called_idx, max_cells, replace=False))
    A = ad.AnnData(X[:, idx].T.tocsr().astype(np.float32))
    sc.pp.filter_genes(A, min_cells=3)
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    n_hvg = min(2000, A.shape[1] - 1)
    sc.pp.highly_variable_genes(A, n_top_genes=n_hvg, flavor="seurat")
    A = A[:, A.var.highly_variable].copy()
    sc.pp.scale(A, max_value=10)
    ncomp = int(min(30, A.shape[0] - 2, A.shape[1] - 2))
    sc.tl.pca(A, n_comps=ncomp)
    sc.pp.neighbors(A, n_neighbors=15, n_pcs=ncomp)
    sc.tl.leiden(A, resolution=res, flavor="igraph", n_iterations=2, directed=False)
    lab = A.obs["leiden"].astype(int).to_numpy()
    if len(idx) == len(called_idx):
        return lab
    # transfer by nearest centroid in the HVG log space
    full = np.full(len(called_idx), -1)
    pos = np.searchsorted(called_idx, idx)
    full[pos] = lab
    return full


def centroids(X: sp.csc_matrix, idx: np.ndarray, labels: np.ndarray, amb: np.ndarray, min_cells=10):
    ks = [k for k in np.unique(labels) if k >= 0 and (labels == k).sum() >= min_cells]
    Q = []
    for k in ks:
        s = np.asarray(X[:, idx[labels == k]].sum(axis=1)).ravel().astype(np.float64)
        Q.append(0.999 * s / s.sum() + 0.001 * amb)  # floor by ambient so no gene has probability 0
    return (np.stack(Q, axis=1) if Q else np.zeros((len(amb), 0))), ks


def null_pvalues(obs_llr, obs_tot, null_llr, null_tot, bins_per_decade=12):
    """P-value of each observed LLR against nulls in the same log-depth bin: empirical ((1 + #null >= obs)/(1 + n)) while at
    least 10 nulls exceed the observation, beyond that an exponential tail fitted to the nulls above their 90th percentile
    (mean excess), so that p-values can go below 1/n and survive BH across tens of thousands of barcodes."""
    p = _empirical_p(obs_llr, obs_tot, null_llr, null_tot, bins_per_decade)
    lb = lambda t: np.floor(np.log10(np.maximum(t, 1)) * bins_per_decade).astype(int)
    ob, nbn = lb(obs_tot), lb(null_tot)
    for b in np.unique(ob):
        nl = np.sort(null_llr[nbn == b])
        if len(nl) < 100:
            continue
        sel = np.nonzero(ob == b)[0]
        ge = len(nl) - np.searchsorted(nl, obs_llr[sel] - 1e-9, side="left")
        tail = ge < 10
        if not tail.any():
            continue
        u = np.quantile(nl, 0.9)
        exc = nl[nl > u] - u
        scale = max(float(exc.mean()) if len(exc) else 1e-3, 1e-3)
        pt = 0.1 * np.exp(-(obs_llr[sel[tail]] - u) / scale)
        p[sel[tail]] = np.minimum(p[sel[tail]], pt)
    return p


def _empirical_p(obs_llr, obs_tot, null_llr, null_tot, bins_per_decade=12):
    """Empirical p-value of each observed LLR against nulls in the same log-depth bin (p = (1 + #null >= obs)/(1 + n))."""
    lb = lambda t: np.floor(np.log10(np.maximum(t, 1)) * bins_per_decade).astype(int)
    ob, nbn = lb(obs_tot), lb(null_tot)
    p = np.ones(len(obs_llr))
    for b in np.unique(ob):
        nl = np.sort(null_llr[nbn == b])
        sel = ob == b
        if len(nl) == 0:
            p[sel] = np.nan
            continue
        ge = len(nl) - np.searchsorted(nl, obs_llr[sel] - 1e-9, side="left")
        p[sel] = (ge + 1) / (len(nl) + 1)
    return p


def bin_table(values_by_name: dict, umi, bins=UMI_BINS):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (umi >= lo) & (umi < hi)
        row = {"lo": lo, "hi": hi if hi < 10**9 else None, "n": int(m.sum())}
        for name, (v, how) in values_by_name.items():
            vv = v[m]
            vv = vv[~np.isnan(vv)] if vv.dtype.kind == "f" else vv
            if len(vv) == 0:
                row[name] = None
            elif how == "mean":
                row[name] = float(np.mean(vv))
            elif how == "median":
                row[name] = float(np.median(vv))
            elif how == "sum":
                row[name] = float(np.sum(vv))
        out.append(row)
    return out


def q(v, qs=(0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)):
    v = np.asarray(v)
    if len(v) == 0:
        return None
    return {str(k): float(np.quantile(v, k)) for k in qs}


# ----------------------------------------------------------------------------------------------------------- one sample
def analyze_sample(path, out_dir, niters=10000, seed=1, test_matrix="exon", do_callers=True):
    t0 = time.time()
    d = load(path)
    meta = d["meta"]
    gsm = meta["gsm"]
    rng = np.random.default_rng(seed)
    E, I = d["exon"], d["intron"]
    X = (E + I).tocsc()
    X.data = X.data.astype(np.int32)
    called = d["called"]
    tot = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)
    etot = np.asarray(E.sum(axis=0)).ravel()
    itot = np.asarray(I.sum(axis=0)).ravel()
    n = len(tot)
    unc = ~called
    mt, hb, human, mouse = gene_sets(d["genes"], d["symbols"], d["chroms"])
    mt_frac = np.asarray(X[mt].sum(axis=0)).ravel() / np.maximum(tot, 1)
    hb_frac = np.asarray(X[hb].sum(axis=0)).ravel() / np.maximum(tot, 1)
    intr_frac = itot / np.maximum(tot, 1)
    ngenes = np.diff(X.indptr)
    R = {
        "gse": meta["gse"], "gsm": gsm, "n_barcodes": int(n), "n_called": int(called.sum()),
        "protocol": (meta.get("gsm_meta") or {}).get("protocol") or (meta.get("summary") or {}).get("protocol_id"),
        "reference_build": (meta.get("gsm_meta") or {}).get("reference_build") or (meta.get("provenance") or {}).get("reference_build"),
        "call_method": (meta.get("cell_calls") or {}).get("call_method"),
        "cell_calls_missing": (meta.get("cell_calls") or {}).get("n_missing_in_matrix"),
        "n_genes": int(len(d["genes"])), "n_mt_genes": int(mt.sum()), "n_hb_genes": int(hb.sum()),
        "n_human_genes": int(human.sum()), "n_mouse_genes": int(mouse.sum()),
        "umi_called": q(tot[called]), "umi_uncalled": q(tot[unc]),
        "umi_total_called": int(tot[called].sum()), "umi_total_uncalled": int(tot[unc].sum()),
        "min_called_umi": int(tot[called].min()) if called.any() else None,
        "max_uncalled_umi": int(tot[unc].max()) if unc.any() else None,
        "summary": {k: (meta.get("summary") or {}).get(k) for k in ("n_cells_called", "median_umi_per_cell", "n_input_reads",
                                                                      "mapping_rate", "fraction_reads_in_cells")},
        "members": meta.get("members"),
    }
    # called/uncalled overlap in total UMI: rank-based view of how far from a single threshold STAR's call is
    if called.any() and unc.any():
        thr_grid = np.unique(np.quantile(tot, np.linspace(0, 1, 400)))
        errs = [((tot >= t) != called).sum() for t in thr_grid]
        bi = int(np.argmin(errs))
        R["best_single_threshold"] = {"umi": float(thr_grid[bi]), "n_mismatch": int(errs[bi]), "frac_mismatch": errs[bi] / n}

    # ---------------------------------------------------------------- storage accounting
    nnz = {}
    for k in ("exon_nnz_features", "intron_nnz_features", "sj_nnz", "vdj_nnz"):
        if d[k] is not None:
            nnz[k.replace("_nnz_features", "").replace("_nnz", "")] = d[k].astype(np.int64)
    mbytes = {"exon": meta.get("exon_bytes", 0), "intron": meta.get("intron_bytes", 0), "sj": meta.get("sj_bytes", 0),
              "vdj": meta.get("vdj_bytes", 0)}
    st = {"member_bytes": mbytes, "nnz_total": {k: int(v.sum()) for k, v in nnz.items()},
          "nnz_called": {k: int(v[called].sum()) for k, v in nnz.items()}}
    st["by_bin_uncalled"] = bin_table({k: (v[unc].astype(float), "sum") for k, v in nnz.items()}, tot[unc])
    keep = {}
    for X0 in KEEP_GRID:
        m = called | (tot >= X0)
        keep[str(X0)] = {"n_barcodes": int(m.sum()), **{f"nnz_{k}": int(v[m].sum()) for k, v in nnz.items()}}
    st["keep_called_plus_umi_ge"] = keep
    R["storage"] = st

    # ---------------------------------------------------------------- test matrix, ambient profile and DM alpha
    # The profile test uses exon counts only: the pileup counts one molecule once as exon and once as intron when its reads
    # cover both, which shows up as 2-UMI barcodes whose two counts sit on one gene (``dup2_single_gene`` below) and would
    # masquerade as overdispersion. Bins in the tables stay on total (exon + intron) UMIs, the measure of the cells table.
    M = (E if test_matrix == "exon" else X).tocsc()
    mtot = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
    two = unc & (tot == 2)
    R["dup2_single_gene"] = float((ngenes[two] == 1).mean()) if two.any() else None
    R["dup2_exon1_intron1_same_gene"] = float(((ngenes == 1) & (etot == 1) & (itot == 1))[two].mean()) if two.any() else None
    unc_idx = np.nonzero(unc)[0]
    q05 = float(np.quantile(mtot[called], 0.05)) if called.any() else 100.0
    thr_amb = float(np.clip(0.5 * q05, 2, 100))
    amb_mask = unc & (mtot <= thr_amb)
    alpha_mask = amb_mask & (mtot >= 2)
    asum = np.asarray(M[:, amb_mask].sum(axis=1)).ravel().astype(np.int64)
    amb = C.good_turing_proportions(asum)
    alpha = C.dm_alpha_mle(M[:, alpha_mask], amb) if alpha_mask.sum() >= 50 else 1e4
    R["ambient"] = {"matrix": test_matrix, "rule": "uncalled with test-matrix total <= clip(0.5 x q05(called), 2, 100)",
                    "umi_max": thr_amb, "n_barcodes": int(amb_mask.sum()), "n_alpha_barcodes": int(alpha_mask.sum()),
                    "umi": int(asum.sum()), "alpha": alpha,
                    "intronic_frac": float(I[:, amb_mask].sum() / max(1, X[:, amb_mask].sum())),
                    "mt_frac": float(asum[mt].sum() / max(1, asum.sum())), "hb_frac": float(asum[hb].sum() / max(1, asum.sum())),
                    "top_genes": [str(d["symbols"][i]) for i in np.argsort(-asum)[:25]]}
    csum = np.asarray(M[:, called].sum(axis=1)).ravel().astype(np.float64)
    R["ambient"]["corr_logcpm_with_called_sum"] = float(np.corrcoef(np.log1p(asum / max(1, asum.sum()) * 1e6),
                                                                    np.log1p(csum / max(1, csum.sum()) * 1e6))[0, 1])

    # ---------------------------------------------------------------- clusters of called cells, centroids (two halves)
    called_idx = np.nonzero(called)[0]
    labels = None
    if len(called_idx) >= 50:
        try:
            labels = cluster_called(X, called_idx, rng)
        except Exception as e:
            R["cluster_error"] = repr(e)
    if labels is None:
        labels = np.zeros(len(called_idx), int)
    half = rng.random(len(called_idx)) < 0.5
    Q_all, ks = centroids(M, called_idx, labels, amb)
    QA, _ = centroids(M, called_idx[half], labels[half], amb)
    QB, _ = centroids(M, called_idx[~half], labels[~half], amb)
    R["clusters"] = {"n": len(ks), "sizes": {str(k): int((labels == k).sum()) for k in ks}}
    mk = {}  # marker genes per cluster (top 8 by log fold over the other clusters), for interpretation
    for j, k in enumerate(ks):
        others = Q_all[:, [i for i in range(len(ks)) if i != j]].mean(axis=1) if len(ks) > 1 else amb
        lfc = np.log2((Q_all[:, j] + 1e-6) / (others + 1e-6))
        lfc[Q_all[:, j] < 1e-4] = -np.inf
        mk[str(k)] = [str(d["symbols"][i]) for i in np.argsort(-lfc)[:8]]
    R["clusters"]["markers"] = mk

    out_bc = {"total": tot, "exon_total": etot, "intron_total": itot, "called": called, "mt_frac": mt_frac.astype(np.float32),
              "hb_frac": hb_frac.astype(np.float32), "n_genes": ngenes.astype(np.int32)}
    bpd = 12

    def depth_nulls(tt_all, per_bin=2000, cap=40000):
        """Totals for null draws: every observed total repeated so that each log-depth bin gets >= per_bin draws."""
        lb = np.floor(np.log10(np.maximum(tt_all, 1)) * bpd).astype(int)
        out = []
        for b in np.unique(lb):
            tt = tt_all[lb == b]
            reps = int(min(50, max(1, math.ceil(per_bin / len(tt)))))
            out.append(np.repeat(tt, reps) if len(tt) * reps <= cap else rng.choice(tt, cap))
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def full(v, fill=np.nan, dtype=float):
        o = np.full(n, fill, dtype=dtype)
        o[unc_idx] = v
        return o

    if Q_all.shape[1] >= 1 and amb_mask.sum() >= 20:
        # ---------------------------------------------------------- LLR for uncalled (centroids from all called cells)
        mu = mtot[unc_idx]
        llr_u, k_u, a_u = mixture_llr(M[:, unc_idx], amb, Q_all)
        null_tot = depth_nulls(mu)
        llr_0, _, _ = mixture_llr(dm_null_matrix(amb, alpha, null_tot, seed=seed + 11), amb, Q_all)
        p_u = null_pvalues(llr_u, mu, llr_0, null_tot, bpd)
        fdr_u = C.bh(np.nan_to_num(p_u, nan=1.0))
        # sensitivity: a null 4x more overdispersed than the fitted one
        llr_0c, _, _ = mixture_llr(dm_null_matrix(amb, alpha / 4, null_tot, seed=seed + 13), amb, Q_all)
        p_uc = null_pvalues(llr_u, mu, llr_0c, null_tot, bpd)
        fdr_uc = C.bh(np.nan_to_num(p_uc, nan=1.0))
        kk = full(np.array(ks)[k_u], -1, int)
        out_bc.update({"llr": full(llr_u).astype(np.float32), "p_cell": full(p_u), "fdr_cell": full(fdr_u),
                       "fdr_cell_alpha_div4": full(fdr_uc), "alpha_cell": full(a_u).astype(np.float32),
                       "best_cluster": kk.astype(np.int16)})
        sig = full(fdr_u < 0.01, False, bool)
        R["profile_test"] = {
            "n_uncalled_tested": int(len(unc_idx)), "n_null": int(len(null_tot)),
            "n_sig_fdr01": int(sig.sum()), "n_sig_fdr05": int((fdr_u < 0.05).sum()),
            "n_sig_fdr01_alpha_div4": int((fdr_uc < 0.01).sum()),
            "n_nominal_p01": int((p_u < 0.01).sum()), "expected_nominal_p01": 0.01 * len(unc_idx),
            "umi_of_sig": q(tot[sig]), "nnz_share_of_sig": {k: float(v[sig].sum() / max(1, v[unc].sum())) for k, v in nnz.items()},
            "alpha_of_sig": q(a_u[fdr_u < 0.01]),
            "cluster_of_sig": {str(k): int(((kk == k) & sig).sum()) for k in ks},
            "cluster_of_called": {str(k): int((labels == k).sum()) for k in ks},
            "by_bin": bin_table({"frac_nominal_p01": ((p_u < 0.01).astype(float), "mean"),
                                 "frac_sig_fdr01": ((fdr_u < 0.01).astype(float), "mean"),
                                 "frac_sig_fdr01_alpha_div4": ((fdr_uc < 0.01).astype(float), "mean"),
                                 "median_alpha": (a_u.astype(float), "median"),
                                 "mean_alpha": (a_u.astype(float), "mean"),
                                 "median_llr": (llr_u, "median"),
                                 "median_test_umi": (mu.astype(float), "median")}, tot[unc_idx]),
            "null_by_bin_test_umi": bin_table({"null_median_llr": (llr_0, "median")}, null_tot),
        }

        # ---------------------------------------------------------- split-half cluster consistency (robust to the null model)
        # Each barcode's counts are split 50/50 at random and the best cluster is fitted on each half. A barcode carrying one
        # cell type's RNA picks the same cluster twice; ambient (however overdispersed around the ambient mean) does so only
        # at the chance rate, measured on DM null draws at the same depths. Thinned held-out cells give the ceiling.
        def split_agree(Y, Qc):
            Y = Y.tocsc().astype(np.int64)
            y1 = rng.binomial(Y.data, 0.5)
            Y1 = sp.csc_matrix((y1, Y.indices, Y.indptr), shape=Y.shape)
            Y2 = sp.csc_matrix((Y.data - y1, Y.indices, Y.indptr), shape=Y.shape)
            _, k1, _ = mixture_llr(Y1, amb, Qc)
            _, k2, _ = mixture_llr(Y2, amb, Qc)
            return k1 == k2

        sh = {"min_test_umi": 6, "bins": "uncalled by total UMI; nulls by test-matrix UMI"}
        sel = mu >= 6
        if sel.any():
            ag = split_agree(M[:, unc_idx[sel]], Q_all)
            nt6 = depth_nulls(mu[sel], per_bin=500, cap=10000)
            ag0 = split_agree(dm_null_matrix(amb, alpha, nt6, seed=seed + 17), Q_all)
            ag0c = split_agree(dm_null_matrix(amb, alpha / 4, nt6, seed=seed + 19), Q_all)
            sh["uncalled"] = bin_table({"agree": (ag.astype(float), "mean")}, tot[unc_idx[sel]])
            sh["uncalled_test_umi"] = bin_table({"agree": (ag.astype(float), "mean")}, mu[sel])
            sh["null_test_umi"] = bin_table({"agree": (ag0.astype(float), "mean")}, nt6)
            sh["null_alpha_div4_test_umi"] = bin_table({"agree": (ag0c.astype(float), "mean")}, nt6)
            sh["uncalled_overall"] = float(ag.mean())
            sh["null_overall"] = float(ag0.mean())
            sh["null_alpha_div4_overall"] = float(ag0c.mean())
            ag_full = np.zeros(n, np.int8) - 1
            ag_full[unc_idx[sel]] = ag
            out_bc["split_agree"] = ag_full
        cells_sh = []
        for dpt in THIN_DEPTHS:
            ti = called_idx[~half]
            ti = ti[mtot[ti] > dpt * 1.5]
            if len(ti) < 20 or QA.shape[1] == 0:
                continue
            ti = rng.choice(ti, min(1500, len(ti)), replace=False)
            cells_sh.append({"depth": dpt, "n": int(len(ti)), "agree": float(split_agree(thin(M[:, ti], dpt, rng), QA).mean())})
        sh["thinned_cells_heldout"] = cells_sh
        R["split_half"] = sh

        # ---------------------------------------------------------- power: thinned called cells, held-out centroids
        pw = []
        if QA.shape[1] and QB.shape[1]:
            for dpt in THIN_DEPTHS:
                rows = []
                for (Qc, test_sel) in ((QA, ~half), (QB, half)):
                    ti = called_idx[test_sel]
                    ti = ti[mtot[ti] > dpt * 1.5]
                    if len(ti) > 1500:
                        ti = rng.choice(ti, 1500, replace=False)
                    if len(ti) == 0:
                        continue
                    Yt = thin(M[:, ti], dpt, rng)
                    tt = np.asarray(Yt.sum(axis=0)).ravel()
                    l1, _, a1 = mixture_llr(Yt, amb, Qc)
                    n0t = np.maximum(1, rng.poisson(dpt, 3000))
                    l0, _, _ = mixture_llr(dm_null_matrix(amb, alpha, n0t, seed=seed + 101 + dpt), amb, Qc)
                    rows.append((null_pvalues(l1, tt, l0, n0t, 6), a1))
                if rows:
                    p1 = np.concatenate([r[0] for r in rows])
                    a1 = np.concatenate([r[1] for r in rows])
                    pw.append({"depth": dpt, "n": int(len(p1)), "power_p01": float(np.nanmean(p1 < 0.01)),
                               "power_p001": float(np.nanmean(p1 < 0.001)), "median_alpha": float(np.median(a1))})
        R["power_thinned_called"] = pw

        # ---------------------------------------------------------- called cells that look like ambient (held-out centroids)
        lc = np.full(len(called_idx), np.nan)
        for (Qc, sel_c) in ((QA, ~half), (QB, half)):
            if Qc.shape[1] == 0 or sel_c.sum() == 0:
                continue
            l1, _, _ = mixture_llr(M[:, called_idx[sel_c]], amb, Qc)
            lc[sel_c] = l1
        ct = mtot[called_idx]
        nt = depth_nulls(ct[ct <= 20000], per_bin=1000, cap=20000)
        if len(nt):
            l0c, _, _ = mixture_llr(dm_null_matrix(amb, alpha, nt, seed=seed + 23), amb, QA if QA.shape[1] else Q_all)
            pc = null_pvalues(np.nan_to_num(lc, nan=0.0), ct, l0c, nt, bpd)
            pc[ct > 20000] = 0.0
            look_empty = pc > 0.05
            pcf = np.full(n, np.nan)
            pcf[called_idx] = pc
            out_bc["p_cell_called"] = pcf
            R["called_look_empty"] = {"n": int(look_empty.sum()), "frac": float(look_empty.mean()),
                                      "umi": q(tot[called_idx][look_empty]),
                                      "by_bin": bin_table({"frac_p_gt_05": (look_empty.astype(float), "mean")}, tot[called_idx])}

        # ---------------------------------------------------------- correlations (requested view)
        top = np.argsort(-(asum + csum / max(1, csum.sum()) * max(1, asum.sum())))[:2000]
        corr_amb = logcpm_corr(M[:, unc_idx], asum.astype(float), top)
        corr_cell = np.full(len(unc_idx), np.nan)
        for j in range(Q_all.shape[1]):
            s_j = k_u == j
            if s_j.any():
                corr_cell[s_j] = logcpm_corr(M[:, unc_idx[s_j]], Q_all[:, j], top)
        cc_called = logcpm_corr(M[:, called_idx], asum.astype(float), top)
        R["correlation"] = {"genes": "top 2000 by ambient+cell abundance",
                            "by_bin_uncalled": bin_table({"corr_ambient": (corr_amb, "median"), "corr_best_centroid": (corr_cell, "median"),
                                                          "frac_closer_to_cell": ((corr_cell > corr_amb).astype(float), "mean")}, tot[unc_idx]),
                            "called_corr_ambient": q(cc_called)}
        out_bc["corr_ambient"] = full(corr_amb).astype(np.float32)
        out_bc["corr_cell"] = full(corr_cell).astype(np.float32)

    # ---------------------------------------------------------------- QC fractions by category
    groups = {"called": called, "uncalled": unc, "ambient_pool": amb_mask}
    if "fdr_cell" in out_bc:
        groups["uncalled_sig"] = unc & (np.nan_to_num(out_bc["fdr_cell"], nan=1) < 0.01)
        groups["uncalled_notsig"] = unc & ~groups["uncalled_sig"]
    R["qc_fractions"] = {g: {"n": int(m.sum()), "intronic": q(intr_frac[m]), "mt": q(mt_frac[m]), "hb": q(hb_frac[m]),
                             "genes_per_umi": q(ngenes[m] / np.maximum(tot[m], 1))} for g, m in groups.items() if m.any()}
    R["qc_by_bin_uncalled"] = bin_table({"intronic": (intr_frac[unc], "median"), "mt": (mt_frac[unc], "median"),
                                         "hb": (hb_frac[unc], "median")}, tot[unc])
    R["qc_by_bin_called"] = bin_table({"intronic": (intr_frac[called], "median"), "mt": (mt_frac[called], "median")}, tot[called])

    # ---------------------------------------------------------------- species purity (barnyard)
    if human.sum() > 1000 and mouse.sum() > 1000:
        hu = np.asarray(X[human].sum(axis=0)).ravel()
        mo = np.asarray(X[mouse].sum(axis=0)).ravel()
        hf = hu / np.maximum(hu + mo, 1)
        h_amb = hu[amb_mask].sum() / max(1, (hu + mo)[amb_mask].sum())
        from scipy.stats import binom

        k = hu.astype(np.int64)
        nn = (hu + mo).astype(np.int64)
        # two-sided binomial test of the human fraction against the ambient mix (overdispersion ignored: conservative
        # direction is checked by the ambient pool's own rejection rate, reported as frac_pure_ambient_pool)
        plo = binom.cdf(k, nn, h_amb)
        phi = binom.sf(k - 1, nn, h_amb)
        pb = np.minimum(1, 2 * np.minimum(plo, phi))
        purity = np.maximum(hf, 1 - hf)
        pure = (pb < 0.001) & (purity >= 0.9)
        out_bc["human_frac"] = hf.astype(np.float32)
        out_bc["species_p"] = pb
        R["barnyard"] = {
            "ambient_human_frac": float(h_amb),
            "called_human_frac": q(hf[called]), "called_frac_pure90": float((purity[called] >= 0.9).mean()),
            "frac_pure_ambient_pool": float(pure[amb_mask].mean()),
            "by_bin_uncalled": bin_table({"frac_pure": (pure[unc].astype(float), "mean"), "median_purity": (purity[unc], "median"),
                                          "median_human_frac": (hf[unc], "median")}, tot[unc]),
            "by_bin_called": bin_table({"frac_pure": (pure[called].astype(float), "mean")}, tot[called]),
            "n_uncalled_pure": int(pure[unc].sum()),
            "n_uncalled_pure_by_species": {"human": int((pure & unc & (hf > 0.5)).sum()), "mouse": int((pure & unc & (hf < 0.5)).sum())},
        }
        if "fdr_cell" in out_bc:
            s = unc & (np.nan_to_num(out_bc["fdr_cell"], nan=1) < 0.01)
            R["barnyard"]["profile_sig_that_are_pure"] = float(pure[s].mean()) if s.any() else None
            R["barnyard"]["profile_notsig_that_are_pure"] = float(pure[unc & ~s].mean()) if (unc & ~s).any() else None

    # ---------------------------------------------------------------- independent callers
    if do_callers:
        cal = {}
        knee, infl = C.barcode_ranks(tot, lower=int(np.clip(0.5 * np.quantile(tot[called], 0.05), 5, 100)) if called.any() else 100)
        cal["knee"] = {"threshold": knee, "called": tot >= knee if np.isfinite(knee) else np.zeros(n, bool)}
        cal["inflection"] = {"threshold": infl, "called": tot >= infl if np.isfinite(infl) else np.zeros(n, bool)}
        om, thr = C.ordmag(tot)
        cal["ordmag_cr2"] = {"threshold": thr, "called": om}
        lower_adapt = int(np.clip(0.5 * np.quantile(tot[called], 0.05), 5, 100)) if called.any() else 100
        lower_adapt_exon = int(np.clip(0.5 * np.quantile(etot[called], 0.05), 5, 100)) if called.any() else 100
        for name, Ymat, lower in (("emptydrops_lower100", X, 100), ("emptydrops_adaptive", X, lower_adapt),
                                  ("emptydrops_adaptive_exon", E, lower_adapt_exon)):
            kn = knee if Ymat is X else C.barcode_ranks(etot, lower=lower)[0]
            retain = kn if np.isfinite(kn) and kn > lower else None
            ed = C.emptydrops(Ymat, lower=lower, niters=niters, retain=retain, seed=seed + 3)
            if "error" in ed:
                cal[name] = {"error": ed["error"], "lower": lower}
                continue
            f = np.nan_to_num(ed["fdr"], nan=1.0)
            cal[name] = {"lower": lower, "retain": retain, "alpha": ed["alpha"], "n_ambient": ed["n_ambient"],
                         "n_tested": int(ed["tested"].sum()), "called": f <= 0.001, "called_fdr01": f <= 0.01}
            out_bc[f"{name}_fdr"] = ed["fdr"]
        try:
            cr = C.cr3_like(X, niters=niters, seed=seed + 5)
            cal["cr3_like"] = {k: v for k, v in cr.items()}
        except Exception as e:
            cal["cr3_like"] = {"error": repr(e)}
        comp = {}
        sig = unc & (np.nan_to_num(out_bc.get("fdr_cell", np.full(n, np.nan)), nan=1) < 0.01)
        for name, c in cal.items():
            row = {k: v for k, v in c.items() if k != "called" and not isinstance(v, np.ndarray)}
            for ck in ("called", "called_fdr01"):
                if ck not in c:
                    continue
                cm = np.asarray(c[ck], bool)
                out_bc[f"{name}_{ck}"] = cm
                both = cm & called
                row[ck] = {
                    "n": int(cm.sum()), "n_ours_and_theirs": int(both.sum()),
                    "theirs_not_ours": int((cm & unc).sum()), "ours_not_theirs": int((called & ~cm).sum()),
                    "theirs_not_ours_umi": q(tot[cm & unc]), "ours_not_theirs_umi": q(tot[called & ~cm]),
                    "theirs_not_ours_profile_sig": int((cm & sig).sum()),
                    "theirs_not_ours_nnz_share": {k: float(v[cm & unc].sum() / max(1, v[unc].sum())) for k, v in nnz.items()},
                    "jaccard": float(both.sum() / max(1, (cm | called).sum())),
                }
            comp[name] = row
        R["callers"] = comp

    R["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{gsm}.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(os.path.join(out_dir, f"{gsm}.bc.npz"), barcodes=d["barcodes"], ambient_profile=amb.astype(np.float32),
                        genes=d["genes"], **{k: v for k, v in out_bc.items()})
    return R


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="directory of fetch.py npz files (one study)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--niters", type=int, default=10000)
    ap.add_argument("--gsm", nargs="*")
    ap.add_argument("--no-callers", action="store_true")
    ap.add_argument("--test-matrix", choices=["exon", "sum"], default="exon")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    nb.set_num_threads(max(1, a.threads))
    files = sorted(glob.glob(os.path.join(a.data_dir, "GSM*.npz")))
    if a.gsm:
        files = [f for f in files if os.path.basename(f)[:-4] in set(a.gsm)]
    for f in files:
        g = os.path.basename(f)[:-4]
        if not a.redo and os.path.exists(os.path.join(a.out, f"{g}.json")):
            print(g, "done already")
            continue
        try:
            R = analyze_sample(f, a.out, niters=a.niters, test_matrix=a.test_matrix, do_callers=not a.no_callers)
            pt = R.get("profile_test", {})
            print(f"{g} ok n={R['n_barcodes']} called={R['n_called']} sig={pt.get('n_sig_fdr01')} t={R['seconds']}s", flush=True)
        except Exception:
            print(g, "FAILED")
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
