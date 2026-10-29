"""Per-sample CPU doublet scores.  Usage: methods_cpu.py NAME METHOD [METHOD ...]

scrublet   Scrublet 0.2.3 defaults (30 PCs, sim ratio 2); needs the envs/ambient interpreter
prod       Singlet's production scorer (doublet_detect.h of prod/modality-fixes through the dbl_run harness, gene-level, nomap);
           needs DBL_RUN=<binary>
nnls       "two cell types in one droplet": best two-centroid non-negative fit of the cell's HVG profile against the sample's
           own Leiden cluster centroids vs the best single centroid (relative residual drop), runtime venv (leidenalg)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

from dbl_common import SCORES, Timer, has_score, hvg, lognorm, load, put_score


def scrublet(d):
    import scrublet as scr

    with Timer() as t:
        s = scr.Scrublet(d["X"].tocsc(), expected_doublet_rate=0.06, random_state=0)
        sc, _ = s.scrub_doublets(min_counts=2, min_cells=3, min_gene_variability_pctl=85, n_prin_comps=30, verbose=False)
    put_score(d["name"], "scrublet", sc, t.s)


def prod(d):
    exe = os.environ["DBL_RUN"]
    X = d["X"].T.tocsc()  # genes x cells
    with tempfile.TemporaryDirectory() as tmp:
        pre = f"{tmp}/m"
        np.array([X.shape[0], X.shape[1], X.nnz], np.uint32).tofile(pre + ".hdr")
        X.indptr.astype(np.int32).tofile(pre + ".indptr")
        X.indices.astype(np.int32).tofile(pre + ".indices")
        np.minimum(X.data, 65535).astype(np.uint16).tofile(pre + ".data")
        with Timer() as t:
            r = subprocess.run([exe, pre, f"{tmp}/out.tsv", f"threads={os.environ.get('SLURM_CPUS_PER_TASK', '8')}", "nomap=1"],
                               capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        o = pd.read_csv(f"{tmp}/out.tsv", sep="\t")
    put_score(d["name"], "prod", o.score.values, t.s, calls=o.is_doublet.values.astype(int).tolist(), log=r.stderr.strip()[-500:])


def nnls_scores(P: np.ndarray, M: np.ndarray):
    """P cells x genes, M clusters x genes (both pre-whitened). Returns (relative residual drop of the best non-negative
    two-centroid fit over the best single centroid, minor weight share of that pair)."""
    pp = (P * P).sum(1)
    G = P @ M.T  # cells x K
    mm = (M * M).sum(1)
    C = M @ M.T
    sse1 = pp - np.max(np.where(G > 0, G * G / mm, 0), 1)
    best2 = sse1.copy()
    minor = np.zeros(len(P))
    K = len(M)
    for k in range(K):
        for l in range(k + 1, K):
            det = C[k, k] * C[l, l] - C[k, l] ** 2
            if det <= 1e-12:
                continue
            wk = (C[l, l] * G[:, k] - C[k, l] * G[:, l]) / det
            wl = (C[k, k] * G[:, l] - C[k, l] * G[:, k]) / det
            ok = (wk > 0) & (wl > 0)
            sse = pp - (wk * G[:, k] + wl * G[:, l])
            upd = ok & (sse < best2)
            best2[upd] = sse[upd]
            minor[upd] = (np.minimum(wk, wl) / (wk + wl))[upd]
    return 1 - best2 / np.maximum(sse1, 1e-12), minor


def nnls(d):
    import scanpy as sc

    with Timer() as t:
        X = d["X"]
        h = hvg(X, 2000)
        ad = sc.AnnData(lognorm(X)[:, h])
        sc.pp.scale(ad, max_value=10)
        sc.tl.pca(ad, n_comps=30)
        sc.pp.neighbors(ad, n_neighbors=15)
        sc.tl.leiden(ad, resolution=1.0, flavor="igraph", n_iterations=2, random_state=0)
        cl = ad.obs.leiden.astype(int).values
        Xh = X[:, h].astype(np.float64)
        n = np.asarray(Xh.sum(1)).ravel()
        P = np.asarray((sp.diags(1 / np.maximum(n, 1)) @ Xh).todense())
        K = cl.max() + 1
        M = np.stack([np.asarray(Xh[cl == k].sum(0)).ravel() for k in range(K)])
        M = M / M.sum(1, keepdims=True)
        wgt = 1 / np.sqrt(P.mean(0) + 1e-6)
        rel, minor = nnls_scores(P * wgt, M * wgt)
    put_score(d["name"], "nnls", rel, t.s, n_clusters=int(K))
    put_score(d["name"], "nnls_minor", minor, 0.0)
    np.save(f"{SCORES}/{d['name']}/clusters.npy", cl)


def export(d):
    """genes x cells Matrix Market for the R methods (scDblFinder)."""
    import scipy.io

    os.makedirs(f"{SCORES}/{d['name']}", exist_ok=True)
    scipy.io.mmwrite(f"{SCORES}/{d['name']}/counts.mtx", d["X"].T.tocsc())


if __name__ == "__main__":
    name = sys.argv[1]
    d = load(name)
    for m in sys.argv[2:]:
        if m != "export" and has_score(name, m) and not os.environ.get("FORCE"):
            continue
        globals()[m](d)
