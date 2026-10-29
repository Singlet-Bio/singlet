# SPDX-License-Identifier: MIT
"""Four doublet features from the atlas cell-type network (``singlet.celltype_nn``), used as a fixed encoder.

Per sample, 2n doublets are simulated (summed raw counts of random cell pairs, on the network's gene vocabulary), real and
simulated cells are embedded by the network's 2,048-d hidden layer, and:

``atlas_knn``    fraction of simulated doublets among each real cell's k = max(10, 0.5 sqrt(n)) nearest neighbours (PCA-50)
``atlas_clf``    5-fold cross-fitted classifier, real vs simulated, on the same PCA-50 (out-of-fold score per real cell)
``ct_entropy``   entropy of the network's calibrated cell-type distribution
``atlas_nnls``   two cell types in one droplet: relative residual drop of the best non-negative fit by two atlas cell-type
                 centroids over the best single centroid (candidates: the 8 best single centroids)

Port of ``bench/latent_qc/doublets/latent_gpu.py`` (study/latent-qc, PR #43), whose cached scores trained the combiner.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

NNLS_CANDIDATES = 8


def simulate_doublets(X: sp.csr_matrix, n: int, rng) -> sp.csr_matrix:
    N = X.shape[0]
    a = rng.integers(0, N, n)
    b = rng.integers(0, N, n)
    b = np.where(a == b, (b + 1) % N, b)
    return (X[a] + X[b]).tocsr()


def _dense(A, rows, dev):
    import torch

    return torch.from_numpy(np.asarray(A[rows].todense(), np.float32)).to(dev)


def _knn_fraction(E, n_real: int, k: int):
    import torch

    out = torch.zeros(n_real, device=E.device)
    for s in range(0, n_real, 4096):
        dd = torch.cdist(E[s : min(s + 4096, n_real)], E)
        dd[torch.arange(dd.shape[0]), torch.arange(s, s + dd.shape[0])] = float("inf")
        nb = dd.topk(k, largest=False).indices
        out[s : s + dd.shape[0]] = (nb >= n_real).float().mean(1)
    return out.cpu().numpy()


def _fit_classifier(Z, y, seed: int = 0, epochs: int = 100, hidden: int = 128, wd: float = 1e-4):
    """MLP real (0) vs simulated (1), 20% validation split, early stopping."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = Z.device
    torch.manual_seed(seed)
    n = len(y)
    perm = torch.randperm(n, device=dev)
    va, tr = perm[: n // 5], perm[n // 5 :]
    clf = nn.Sequential(
        nn.Linear(Z.shape[1], hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=wd)
    best, state, bad = 1e9, None, 0
    for _ in range(epochs):
        clf.train()
        p = tr[torch.randperm(len(tr), device=dev)]
        for s in range(0, len(p), 512):
            b = p[s : s + 512]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(clf(Z[b]).squeeze(1), y[b]).backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            v = F.binary_cross_entropy_with_logits(clf(Z[va]).squeeze(1), y[va]).item()
        if v < best - 1e-4:
            best, state, bad = v, {k: t.clone() for k, t in clf.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    clf.load_state_dict(state)
    clf.eval()
    return clf


def _nnls(V, M, dev, chunk: int = 2048):
    """Relative residual drop of the best two-centroid non-negative fit over the best single centroid (rows of ``V``: vocabulary
    counts; ``M``: centroids x vocabulary)."""
    import torch

    n = V.shape[0]
    Vd = V.astype(np.float64)
    tot = np.asarray(Vd.sum(1)).ravel()
    M = torch.from_numpy(np.asarray(M)).to(dev).double()
    mean_p = torch.from_numpy(
        np.asarray((sp.diags(1 / np.maximum(tot, 1)) @ Vd).mean(0)).ravel()
    ).to(dev)
    w = 1 / torch.sqrt(mean_p + 1e-6)
    Mw = M * w
    mm = (Mw * Mw).sum(1)
    rel = np.zeros(n)
    kc = NNLS_CANDIDATES
    for s in range(0, n, chunk):
        r = slice(s, min(n, s + chunk))
        p = (
            torch.from_numpy(
                np.asarray((sp.diags(1 / np.maximum(tot[r], 1)) @ Vd[r]).todense())
            ).to(dev)
            * w
        )
        pp = (p * p).sum(1)
        G = p @ Mw.T
        fit1 = torch.where(G > 0, G * G / mm, torch.zeros_like(G))
        top = fit1.topk(kc, 1).indices
        sse1 = pp - fit1.gather(1, top[:, :1]).squeeze(1)
        best2 = sse1.clone()
        for a in range(kc):
            for b in range(a + 1, kc):
                ka, kb = top[:, a], top[:, b]
                caa, cbb = mm[ka], mm[kb]
                cab = (Mw[ka] * Mw[kb]).sum(1)
                ga, gb = G.gather(1, ka[:, None]).squeeze(1), G.gather(1, kb[:, None]).squeeze(1)
                det = (caa * cbb - cab**2).clamp_min(1e-12)
                wa = (cbb * ga - cab * gb) / det
                wb = (caa * gb - cab * ga) / det
                sse = pp - (wa * ga + wb * gb)
                upd = (wa > 0) & (wb > 0) & (sse < best2)
                best2 = torch.where(upd, sse, best2)
        rel[r] = (1 - best2 / sse1.clamp_min(1e-12)).cpu().numpy()
    return rel


def atlas_features(
    X,
    gene_ids,
    species: str,
    predictor,
    centroids,
    *,
    seed: int = 0,
    device: Optional[str] = None,
    timings: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """``X``: called cells x genes raw counts; ``gene_ids``: Ensembl ids of its columns; ``predictor``: a loaded
    :class:`singlet.celltype_nn.CellTypePredictor`; ``centroids``: atlas cell-type centroids (centroids x vocabulary)."""
    import torch
    import torch.nn.functional as F

    from singlet.celltype_nn.features import transform

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)
    net = predictor.net.to(dev).eval()
    t0 = time.perf_counter()
    V = predictor.vocab.project(X, gene_ids, species).tocsr()
    n = V.shape[0]
    A = sp.vstack([V, simulate_doublets(V, 2 * n, rng)]).tocsr()
    emb, probs = [], []
    with torch.no_grad():
        for s in range(0, A.shape[0], 8192):
            rows = np.arange(s, min(s + 8192, A.shape[0]))
            e = net.embed(transform(_dense(A, rows, dev), net.transform))
            emb.append(e)
            if s < n:  # calibrated cell-type distribution of the real cells only
                probs.append(F.softmax(net.head(e) / net.temperature, 1))
        E = torch.cat(emb)
        Pr = torch.cat(probs)[:n]
        E = E - E.mean(0)
        _, _, Vt = torch.pca_lowrank(E, q=50, center=False)
        Z = E @ Vt
    out: Dict[str, np.ndarray] = {}
    k = max(10, int(round(0.5 * np.sqrt(n))))
    out["atlas_knn"] = _knn_fraction(Z, n, k)
    y = torch.cat([torch.zeros(n), torch.ones(2 * n)]).to(dev)
    Zs = (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-6)
    fold = torch.from_numpy(rng.integers(0, 5, n)).to(dev)
    fold_sim = torch.from_numpy(rng.integers(0, 5, 2 * n)).to(dev)
    oof = np.zeros(n)
    for f in range(5):
        tr = torch.cat(
            [torch.nonzero(fold != f).squeeze(1), n + torch.nonzero(fold_sim != f).squeeze(1)]
        )
        clf = _fit_classifier(Zs[tr], y[tr], seed=f)
        te = torch.nonzero(fold == f).squeeze(1)
        with torch.no_grad():
            oof[te.cpu().numpy()] = torch.sigmoid(clf(Zs[te])).squeeze(1).cpu().numpy()
    out["atlas_clf"] = oof
    P = Pr.cpu().numpy().astype(np.float64)
    out["ct_entropy"] = -(P * np.log(P + 1e-12)).sum(1)
    out["atlas_nnls"] = _nnls(V, centroids, dev)
    if timings is not None:
        timings["atlas_s"] = time.perf_counter() - t0
        timings["device"] = str(dev)
    return out
