# SPDX-License-Identifier: MIT
"""Atlas-encoder doublet features (port of bench/latent_qc/doublets/latent_gpu.py ``atlas``, branch study/latent-qc;
docs/latent-doublet-ambient.md there). The cell type network ``singlet.celltype_nn`` (model m1_notissue, trained on 2.8M CELLxGENE
cells) is a fixed encoder: its 2,048-d hidden layer embeds the real cells and 2n simulated doublets (summed raw counts of random
pairs). Per real cell:

  atlas_knn    fraction of simulated doublets among its k = max(10, round(0.5 sqrt(n))) nearest neighbours (PCA-50 of the embedding)
  atlas_clf    5-fold cross-fitted MLP, simulated vs real, on the same PCA-50 (out-of-fold probability)
  ct_entropy   entropy of the network's calibrated cell type distribution
  atlas_nnls   relative residual drop of the best non-negative fit of the cell's vocabulary profile by two atlas cell-type
               centroids over one (candidates: the 8 best single centroids)

CPU and GPU: ``device`` is any torch device string ("cpu", "cuda"); values differ between devices only through float arithmetic and
the classifier's training noise. The atlas centroids (``M``, cell types x vocabulary, 20 MB) are not in the repository: they are a
data file located by ``SINGLET_DOUBLET_CENTROIDS`` (checked against the sha256 in the model's metadata), like the network file
(``SINGLET_CELLTYPE_NN_MODEL``)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from .doublet_util import simulate_doublets

ENV_MODEL = "SINGLET_CELLTYPE_NN_MODEL"
ENV_CENTROIDS = "SINGLET_DOUBLET_CENTROIDS"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def vocab_projection(pred, genes, species=None):
    """genes x vocabulary 0/1 matrix: Ensembl ids (optionally prefixed hg19_/mm10_/GRCh38_, version suffix dropped) through the
    network's vocabulary (mouse by one-to-one orthologs); gene symbols through the vocabulary's human symbols (upper case)."""
    v = pred.vocab
    g = np.array([str(x) for x in genes])
    strip = np.array(
        [
            x.split("_", 1)[1] if x[:5] in ("hg19_", "mm10_", "GRCh3", "GRCm3", "hg38_") and "_" in x else x
            for x in g
        ]
    )
    strip = np.array([x.split(".")[0] for x in strip])
    cols = np.full(len(g), -1)
    is_ens = np.char.startswith(strip, "ENS")
    if is_ens.mean() > 0.5:
        hum = np.char.startswith(strip, "ENSG")
        cols[hum] = v.columns(strip[hum], "human")
        mou = np.char.startswith(strip, "ENSMUSG")
        cols[mou] = v.columns(strip[mou], "mouse")
    else:
        sym = {s.upper(): i for i, s in enumerate(v.symbols)}
        cols = np.array([sym.get(s.upper(), -1) for s in strip])
    keep = np.flatnonzero(cols >= 0)
    P = sp.csr_matrix(
        (np.ones(len(keep), np.float32), (keep, cols[keep])), shape=(len(g), len(v))
    )
    return P, len(keep)


def _fit_classifier(Z, y, dev, seed=0, epochs=100, hidden=128, wd=1e-4):
    """MLP real (0) vs simulated (1), 20% validation split with early stopping."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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


def _knn_fraction(E, n_real, k, dev):
    """For each of the first ``n_real`` rows of E: fraction of its k nearest neighbours (self excluded) that are simulated."""
    import torch

    out = torch.zeros(n_real, device=dev)
    blk = 4096 if str(dev).startswith("cuda") else 1024  # block size only bounds memory; the values do not depend on it
    for s in range(0, n_real, blk):
        dd = torch.cdist(E[s : min(s + blk, n_real)], E)
        dd[torch.arange(dd.shape[0]), torch.arange(s, s + dd.shape[0])] = float("inf")
        nb = dd.topk(k, largest=False).indices
        out[s : s + dd.shape[0]] = (nb >= n_real).float().mean(1)
    return out.cpu().numpy()


def _dense(A, rows, dev):
    import torch

    B = A[rows]
    return torch.from_numpy(np.asarray(B.todense(), np.float32)).to(dev)


def load_predictor(model_path: Optional[str] = None):
    from singlet.celltype_nn import CellTypePredictor

    path = model_path or os.environ.get(ENV_MODEL)
    if not path:
        raise RuntimeError(f"give the cell type network file (model_path or ${ENV_MODEL})")
    return CellTypePredictor.load(path)


def load_centroids(path: Optional[str] = None, sha256: Optional[str] = None) -> np.ndarray:
    p = path or os.environ.get(ENV_CENTROIDS)
    if not p:
        raise RuntimeError(f"give the atlas centroid file (centroids or ${ENV_CENTROIDS})")
    if sha256 and sha256_file(p) != sha256:
        raise RuntimeError(f"{p}: sha256 differs from the one recorded with the doublet model")
    return np.load(p, allow_pickle=True)["M"]


def features(
    X: sp.csr_matrix,
    genes,
    species: Optional[str],
    pred,
    centroids: np.ndarray,
    *,
    device: str = "cpu",
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """``atlas_knn, atlas_clf, ct_entropy, atlas_nnls`` for the rows of ``X`` (raw counts, cells x genes named by ``genes``)."""
    import torch
    import torch.nn.functional as F

    from singlet.celltype_nn.features import transform

    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    net = pred.net.to(dev).eval()
    Pm, _ = vocab_projection(pred, genes, species)
    V = (sp.csr_matrix(X).astype(np.float32) @ Pm).tocsr()
    n = V.shape[0]
    S, _ = simulate_doublets(V, 2 * n, rng)
    A = sp.vstack([V, S]).tocsr()
    emb, prob = [], []
    with torch.no_grad():
        for s in range(0, A.shape[0], 8192):
            e = net.embed(
                transform(_dense(A, np.arange(s, min(s + 8192, A.shape[0])), dev), net.transform)
            )
            emb.append(e)
            prob.append(F.softmax(net.head(e) / net.temperature, 1))
    E = torch.cat(emb)
    del emb
    Pr = torch.cat(prob)[:n]
    del prob
    E = E - E.mean(0)
    _, _, Vt = torch.pca_lowrank(E, q=50, center=False)
    Z = E @ Vt
    del E
    k = max(10, int(round(0.5 * np.sqrt(n))))
    knn = _knn_fraction(Z, n, k, dev)
    # cross-fitted classifier, simulated vs real
    y = torch.cat([torch.zeros(n), torch.ones(2 * n)]).to(dev)
    Zs = (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-6)
    fold = torch.from_numpy(rng.integers(0, 5, n)).to(dev)
    fold_sim = torch.from_numpy(rng.integers(0, 5, 2 * n)).to(dev)
    oof = np.zeros(n)
    for f in range(5):
        tr = torch.cat(
            [torch.nonzero(fold != f).squeeze(1), n + torch.nonzero(fold_sim != f).squeeze(1)]
        )
        clf = _fit_classifier(Zs[tr], y[tr], dev, seed=f)
        te = torch.nonzero(fold == f).squeeze(1)
        with torch.no_grad():
            oof[te.cpu().numpy()] = torch.sigmoid(clf(Zs[te])).squeeze(1).cpu().numpy()
    P = Pr.cpu().numpy().astype(np.float64)
    ent = -(P * np.log(P + 1e-12)).sum(1)
    # two atlas cell types in one droplet
    M = torch.from_numpy(np.asarray(centroids)).to(dev).double()
    Vd = V.astype(np.float64)
    tot = np.asarray(Vd.sum(1)).ravel()
    mean_p = torch.from_numpy(
        np.asarray((sp.diags(1 / np.maximum(tot, 1)) @ Vd).mean(0)).ravel()
    ).to(dev)
    w = 1 / torch.sqrt(mean_p + 1e-6)
    Mw = M * w
    mm = (Mw * Mw).sum(1)
    rel = np.zeros(n)
    for s in range(0, n, 2048):
        r = slice(s, min(n, s + 2048))
        p = torch.from_numpy(np.asarray((sp.diags(1 / np.maximum(tot[r], 1)) @ Vd[r]).todense())).to(dev) * w
        pp = (p * p).sum(1)
        G = p @ Mw.T
        fit1 = torch.where(G > 0, G * G / mm, torch.zeros_like(G))
        top = fit1.topk(8, 1).indices
        sse1 = pp - fit1.gather(1, top[:, :1]).squeeze(1)
        best2 = sse1.clone()
        for a in range(8):
            for b in range(a + 1, 8):
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
    return {"atlas_knn": knn, "atlas_clf": oof, "ct_entropy": ent, "atlas_nnls": rel}
