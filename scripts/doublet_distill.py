#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Distil the atlas cell type network into a one-hidden-layer numpy model for the doublet stacker (build time only; needs torch).

Atlas reference cells only (OUT/lite/sel_*.npz of scripts/doublet_lite.py select: CELLxGENE training corpus), real cells plus
simulated doublets (summed counts of random pairs). Targets, computed by the network on the cells' vocabulary counts: its hidden
layer projected on the global top-K principal axes (neighbour-search space of atlas_knn) and its cell type entropy (ct_entropy).
Input: a gene panel (the genes with the largest between-cell-type variance), log1p CP10k over the panel, standardised.
Validation: atlas datasets held out from training. The 17 doublet evaluation datasets are never used.

  python scripts/doublet_distill.py --out OUT --genes 1000 --hidden 256 --k 32 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def lognorm_panel(C):
    tot = np.asarray(C.sum(1)).ravel()
    d = C.toarray().astype(np.float32)
    d *= (1e4 / np.maximum(tot, 1.0))[:, None]
    np.log1p(d, out=d)
    return d


def main():
    import torch
    import torch.nn.functional as F

    from singlet.celltype_nn import CellTypePredictor
    from singlet.celltype_nn.features import transform

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--genes", type=int, default=1000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default=os.environ.get("SINGLET_CELLTYPE_NN_MODEL"))
    a = ap.parse_args()
    torch.manual_seed(0)
    out = Path(a.out) / "lite"
    Xs = sp.load_npz(out / "sel_X.npz").tocsr()
    m = np.load(out / "sel_meta.npz", allow_pickle=True)
    term, dsid, terms = m["term"], m["dataset"], m["terms"]
    rng = np.random.default_rng(1)
    uds = np.unique(dsid)
    val_ds = set(rng.choice(uds, max(1, len(uds) // 10), replace=False))
    isval = np.array([d in val_ds for d in dsid])
    tr, va = np.flatnonzero(~isval), np.flatnonzero(isval)
    T = len(terms)

    # panel by between-term variance of the mean log profile (training cells)
    tot = np.asarray(Xs[tr].sum(1)).ravel()
    Xl = sp.csr_matrix(sp.diags(1e4 / np.maximum(tot, 1)) @ Xs[tr])
    Xl.data = np.log1p(Xl.data)
    Gt = sp.csr_matrix((np.ones(len(tr)), (term[tr], np.arange(len(tr)))), shape=(T, len(tr)))
    cnt = np.asarray(Gt.sum(1)).ravel()
    tm = np.asarray((Gt @ Xl).todense()) / np.maximum(cnt, 1)[:, None]
    panel = np.sort(np.argsort(-tm[cnt >= 5].var(0))[: a.genes])
    del Xl

    pred = CellTypePredictor.load(a.model)
    net = pred.net.eval()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    def net_targets(C6000, block=4096):
        emb, ent = [], []
        with torch.no_grad():
            for s in range(0, C6000.shape[0], block):
                x = torch.from_numpy(C6000[s : s + block].toarray().astype(np.float32))
                e = net.embed(transform(x, net.transform))
                p = F.softmax(net.head(e) / net.temperature, 1)
                emb.append(e.numpy())
                ent.append((-(p * torch.log(p + 1e-12)).sum(1)).numpy())
        return np.vstack(emb), np.concatenate(ent)

    def doublets(idx, n, seed):
        r = np.random.default_rng(seed)
        i, j = r.choice(idx, n), r.choice(idx, n)
        return (Xs[i] + Xs[j]).tocsr()

    t0 = time.time()
    # global PCA axes of the network embedding (singles of the training cells)
    sub = rng.choice(tr, min(60000, len(tr)), replace=False)
    Es, _ = net_targets(Xs[sub])
    emu = Es.mean(0)
    _, _, Vt = np.linalg.svd(Es - emu, full_matrices=False)
    V = Vt[: a.k].T.astype(np.float32)
    del Es
    sets = {}
    for name, idx, nd in (("train", tr, len(tr)), ("val", va, len(va))):
        D = doublets(idx, nd, 7 if name == "train" else 8)
        C_all = sp.vstack([Xs[idx], D]).tocsr()
        E, H = net_targets(C_all)
        Y = np.hstack([(E - emu) @ V, H[:, None]]).astype(np.float32)
        x = lognorm_panel(C_all[:, panel])
        sets[name] = (x, Y, len(idx))
        print(name, x.shape, "targets", Y.shape, f"{time.time() - t0:.0f}s", flush=True)
    xtr, ytr, ntr = sets["train"]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-3
    ysc = ytr[:, : a.k].std() + 1e-6
    ymu = ytr.mean(0)
    hs = ytr[:, a.k].std() + 1e-6  # entropy target scale

    def prep(x):
        return torch.from_numpy(((x - mu) / sd).astype(np.float32))

    Xt, Xv = prep(xtr), prep(sets["val"][0])
    Yt = torch.from_numpy(ytr)
    Yv = torch.from_numpy(sets["val"][1])
    w = torch.cat([torch.full((a.k,), 1.0 / ysc**2), torch.tensor([4.0 / hs**2])])  # entropy weighted up
    G = len(panel)
    net_d = torch.nn.Sequential(torch.nn.Linear(G, a.hidden), torch.nn.ReLU(), torch.nn.Linear(a.hidden, a.k + 1))
    with torch.no_grad():
        net_d[2].bias.copy_(torch.from_numpy(ymu))
    opt = torch.optim.Adam(net_d.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bs = 1024
    for ep in range(a.epochs):
        net_d.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(perm), bs):
            b = perm[s : s + bs]
            opt.zero_grad()
            (((net_d(Xt[b]) - Yt[b]) ** 2) * w).mean().backward()
            opt.step()
        sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            net_d.eval()
            with torch.no_grad():
                pv = net_d(Xv)
                res = ((pv - Yv) ** 2).mean(0)
                var = Yv.var(0)
                r2_emb = 1 - float(res[: a.k].sum() / var[: a.k].sum())
                r2_ent = 1 - float(res[a.k] / var[a.k])
                ns = sets["val"][2]
                r2_ent_single = 1 - float(((pv[:ns, a.k] - Yv[:ns, a.k]) ** 2).mean() / Yv[:ns, a.k].var())
            print(f"epoch {ep + 1} val R2 embedding {r2_emb:.3f} entropy {r2_ent:.3f} (singles only {r2_ent_single:.3f})", flush=True)
    sdict = {k: v.numpy() for k, v in net_d.state_dict().items()}
    mouse = dict(zip(m["mouse_human"], m["mouse_ids"]))
    f = out / f"distill_{a.genes}_{a.hidden}_{a.k}.npz"
    np.savez_compressed(
        f,
        panel_ids=m["genes"][panel].astype("U15"),
        panel_symbols=m["symbols"][panel].astype("U16"),
        panel_mouse=np.array([mouse.get(g, "") for g in m["genes"][panel]], dtype="U19"),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        W1=sdict["0.weight"].T.astype(np.float16),
        b1=sdict["0.bias"].astype(np.float32),
        W2=sdict["2.weight"].T.astype(np.float16),
        b2=sdict["2.bias"].astype(np.float32),
    )
    print("saved", f, os.path.getsize(f), "bytes", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
