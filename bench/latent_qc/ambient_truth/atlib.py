"""Shared helpers for the ambient-truth benchmark: dataset I/O, pool, species truth, harness dump, scoring.

Dataset dir layout (written by prep_lib.py):  X.npz (genes x barcodes csc, barcodes with >=1 UMI), barcodes.npy,
gene_ids.npy, gene_mouse.npy (bool), cell_idx.npy (columns of X that are the called cells).
A "view" restricts to a gene subset (gene-split evaluation): truth and every method are then computed on that subset only.
"""
import json
import os
import time
import resource
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
DATA = f"{LQC}/ambient_truth/data"
OUT = f"{LQC}/ambient_truth/out"
POOL_LO, POOL_HI, POOL_TOP = 1, 100, 50000


class DS:
    pass


def load(lib, view="full"):
    """view: full | A | B (random 50/50 gene split, fixed seed)."""
    d = f"{DATA}/{lib}"
    ds = DS()
    ds.name, ds.view = lib, view
    X = sp.load_npz(f"{d}/X.npz").tocsc()
    ds.gene_ids = np.load(f"{d}/gene_ids.npy", allow_pickle=True)
    ds.gene_mouse = np.load(f"{d}/gene_mouse.npy")
    ds.cell_idx = np.load(f"{d}/cell_idx.npy")
    ds.barcodes = np.load(f"{d}/barcodes.npy", allow_pickle=True)
    if view != "full":
        rng = np.random.default_rng(12345)
        half = rng.random(X.shape[0]) < 0.5
        keep = half if view == "A" else ~half
        X = X.tocsr()[np.where(keep)[0]].tocsc()
        ds.gene_ids, ds.gene_mouse = ds.gene_ids[keep], ds.gene_mouse[keep]
    X.sort_indices()
    ds.X = X
    ds.tot = np.asarray(X.sum(0)).ravel()
    pool = np.where((ds.tot >= POOL_LO) & (ds.tot <= POOL_HI))[0]
    ds.pool = pool[np.argsort(-ds.tot[pool], kind="stable")[:POOL_TOP]]
    ds.amb_counts = np.asarray(X[:, ds.pool].sum(1)).ravel().astype(float)
    ds.a = (ds.amb_counts + 0.5) / (ds.amb_counts + 0.5).sum()
    ds.alpha_m = float(ds.amb_counts[ds.gene_mouse].sum() / ds.amb_counts.sum())
    ds.C = X[:, ds.cell_idx].tocsc()  # genes x cells
    Xr = ds.C.tocsr()
    ds.M = np.asarray(Xr[np.where(ds.gene_mouse)[0]].sum(0)).ravel()
    ds.N = np.asarray(ds.C.sum(0)).ravel()
    ds.H = ds.N - ds.M
    mf = ds.M / np.maximum(ds.N, 1)
    ds.species = np.where(mf <= 0.05, "human", np.where(mf >= 0.95, "mouse", "mixed"))
    ds.hum, ds.mou = ds.species == "human", ds.species == "mouse"
    ds.pure = ds.hum | ds.mou
    ds.alpha_other = np.where(ds.hum, ds.alpha_m, np.where(ds.mou, 1 - ds.alpha_m, np.nan))
    minor = np.where(ds.hum, ds.M, np.where(ds.mou, ds.H, np.nan))
    ds.rho_truth = minor / np.maximum(ds.N, 1) / ds.alpha_other
    ds.depth_bin = pd.qcut(pd.Series(ds.N).rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str).values
    return ds


def dump_harness(ds, out):
    """Binary CSC of the full trimmed raw matrix + called-cell columns for the C++ production harness."""
    os.makedirs(out, exist_ok=True)
    X = ds.X
    X.indptr.astype(np.int64).tofile(f"{out}/raw.indptr")
    X.indices.astype(np.int32).tofile(f"{out}/raw.indices")
    X.data.astype(np.uint32).tofile(f"{out}/raw.data")
    ds.cell_idx.astype(np.uint32).tofile(f"{out}/cells.idx")
    open(f"{out}/dims.txt", "w").write(f"{X.shape[0]} {X.shape[1]} {X.nnz} {len(ds.cell_idx)}\n")


def load_harness_out(ds, d):
    rho = np.fromfile(f"{d}/rho.f64", dtype=np.float64)
    ip = np.fromfile(f"{d}/corr.indptr", dtype=np.int64)
    ix = np.fromfile(f"{d}/corr.indices", dtype=np.int32)
    dv = np.fromfile(f"{d}/corr.data", dtype=np.uint32)
    Cc = sp.csc_matrix((dv.astype(np.float64), ix, ip), shape=ds.C.shape)
    return rho, Cc


def removed_by_species(ds, Cc):
    """Expected/actual removed UMIs per cell split by gene species, from a corrected matrix."""
    D = (ds.C.astype(np.float64) - Cc).tocsr()
    mi = np.where(ds.gene_mouse)[0]
    hi = np.where(~ds.gene_mouse)[0]
    return np.asarray(D[mi].sum(0)).ravel(), np.asarray(D[hi].sum(0)).ravel()


def est_row(ds, name, v):
    m = ds.pure & np.isfinite(v)
    tr = ds.rho_truth
    d = dict(lib=ds.name, view=ds.view, method=name, mean_est=float(np.mean(v[m])), mean_truth=float(np.mean(tr[m])))
    d["ratio"] = d["mean_est"] / d["mean_truth"]
    ok = np.std(v[m]) > 0
    d["spearman"] = float(spearmanr(v[m], tr[m])[0]) if ok else np.nan
    d["pearson"] = float(pearsonr(v[m], tr[m])[0]) if ok else np.nan
    for b in ("low", "mid", "high"):
        mb = m & (ds.depth_bin == b)
        d[f"est_{b}"], d[f"truth_{b}"] = float(np.mean(v[mb])), float(np.mean(tr[mb]))
    return d


def correction_row(ds, name, rho, rem_m, rem_h):
    """Correction table. rem_m / rem_h: removed UMIs per cell from mouse / human genes."""
    hum, mou, pure = ds.hum, ds.mou, ds.pure
    oth_b = np.where(hum, ds.M, np.where(mou, ds.H, np.nan))
    oth_rm = np.where(hum, rem_m, np.where(mou, rem_h, np.nan))
    own_b = np.where(hum, ds.H, np.where(mou, ds.M, np.nan))
    own_rm = np.where(hum, rem_h, np.where(mou, rem_m, np.nan))
    own_share = 1 - ds.alpha_other
    o = dict(lib=ds.name, view=ds.view, method=name)
    o["cross_species_residual_pct"] = 100 * (1 - oth_rm[pure].sum() / oth_b[pure].sum())
    o["own_species_removed_pct"] = 100 * own_rm[pure].sum() / own_b[pure].sum()
    exp_own = (oth_b / ds.alpha_other * own_share)[pure].sum()
    o["own_removed_over_expected"] = own_rm[pure].sum() / exp_own
    o["total_removed_pct"] = 100 * (rem_m + rem_h).sum() / ds.N.sum()
    return o


def reliability_ceiling(ds):
    """Assumption-light noise ceiling. obs_c = minor_c / (N_c alpha_c) = rho_c + e_c, Var(e_c | rho_c) ~ rho_c /(N_c alpha_c).
    reliability = (Var(obs) - mean noise var) / Var(obs); Pearson ceiling of any estimator = sqrt(reliability)."""
    m = ds.pure
    obs = ds.rho_truth[m]
    nv = np.mean(obs / (ds.N[m] * ds.alpha_other[m]))
    vo = np.var(obs, ddof=1)
    rel = max(vo - nv, 0) / vo
    return dict(var_obs=float(vo), noise_var=float(nv), reliability=float(rel), pearson_ceiling=float(np.sqrt(rel)))


def split_half_truth(ds, seed=0, nrep=10):
    """Real-data check of truth noise (no simulated contamination): randomly split each pure cell's observed UMIs into two halves
    (exact hypergeometric split of the real reads), compute rho_true from each half, and correlate the halves across cells.
    Spearman-Brown gives the reliability of the full-depth truth: 2r/(1+r)."""
    rng = np.random.default_rng(seed)
    m = ds.pure
    N = ds.N[m].astype(np.int64)
    minor = np.where(ds.hum, ds.M, ds.H)[m].astype(np.int64)
    al = ds.alpha_other[m]
    sp_, pe_ = [], []
    for _ in range(nrep):
        n1 = N // 2
        m1 = rng.hypergeometric(minor, N - minor, n1)
        m2 = minor - m1
        r1 = m1 / np.maximum(n1, 1) / al
        r2 = m2 / np.maximum(N - n1, 1) / al
        sp_.append(spearmanr(r1, r2)[0]); pe_.append(pearsonr(r1, r2)[0])
    rs, rp = float(np.mean(sp_)), float(np.mean(pe_))
    return dict(split_half_spearman=rs, split_half_pearson=rp, sb_spearman=2 * rs / (1 + rs), sb_pearson=2 * rp / (1 + rp))


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.sec = time.time() - self.t0
        self.maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
