#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compressed atlas reference for the neighbour feature: k-means codebooks of the student's embedding of atlas singles and of
atlas simulated doublets (augmented as the network was trained), then codebook-distance features for a sample in O(n K).

  build   OUT/lite/codebook_<K>.npz  (atlas training cells only; needs torch for the count augmentation, build time only)
  feats   cb_* features of the 17 evaluation datasets -> OUT/lite/feats_<tag>/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))


def cmd_build(a):
    import torch
    from sklearn.cluster import MiniBatchKMeans

    from singlet.celltype_nn.features import AugmentConfig, Augmenter
    from singlet.features.doublet_lite import Distilled

    out = Path(a.out) / "lite"
    mdl = Distilled.load(out / a.model)
    Xs = sp.load_npz(out / "sel_X.npz").tocsr()
    m = np.load(out / "sel_meta.npz", allow_pickle=True)
    dsid = m["dataset"]
    rng = np.random.default_rng(1)
    uds = np.unique(dsid)
    val_ds = set(rng.choice(uds, max(1, len(uds) // 10), replace=False))
    tr = np.flatnonzero(~np.array([d in val_ds for d in dsid]))
    gi = {g: i for i, g in enumerate(m["genes"])}
    panel = np.array([gi[g] for g in mdl.a["panel_ids"]])
    aug = Augmenter(Xs.shape[1], np.load("/mnt/projects/debruinz_project/debruinz/singlet/data/celltype_nn/intron_ratio.npy"),
                    AugmentConfig(), torch.device("cpu"), seed=5)
    r = np.random.default_rng(11)
    use = r.choice(tr, min(a.cells, len(tr)), replace=False)
    emb_s, emb_d = [], []
    t0 = time.time()
    for s0 in range(0, len(use), 4096):
        i = use[s0 : s0 + 4096]
        j = r.choice(tr, len(i))
        a1 = aug(torch.from_numpy(Xs[i].toarray().astype(np.float32))).numpy()[:, panel]
        a2 = aug(torch.from_numpy(Xs[j].toarray().astype(np.float32))).numpy()[:, panel]
        emb_s.append(mdl.outputs(sp.csr_matrix(a1))[:, : mdl.K])
        emb_d.append(mdl.outputs(sp.csr_matrix(a1 + a2))[:, : mdl.K])
    Es, Ed = np.vstack(emb_s), np.vstack(emb_d)
    print("embedded", Es.shape, Ed.shape, f"{time.time() - t0:.0f}s", flush=True)
    res = {}
    for name, E in (("real", Es), ("dbl", Ed)):
        km = MiniBatchKMeans(a.k, random_state=0, n_init=3, batch_size=8192).fit(E)
        res["cb_" + name] = km.cluster_centers_.astype(np.float16)
        res["n_" + name] = np.bincount(km.labels_, minlength=a.k).astype(np.float32)
    f = out / f"codebook_{a.k}.npz"
    np.savez_compressed(f, **res)
    print("saved", f, f.stat().st_size, "bytes")


def cb_features(mdl, cb, X, gene_ids, species, ks=(5, 15, 40)):
    C = mdl.panel_counts(sp.csr_matrix(X, dtype=np.float32), mdl.columns(gene_ids, species))
    o = mdl.outputs(C)
    E = o[:, : mdl.K]
    R, D = cb["cb_real"].astype(np.float32), cb["cb_dbl"].astype(np.float32)
    nr, nd = cb["n_real"], cb["n_dbl"]
    CB = np.vstack([R, D])
    isd = np.r_[np.zeros(len(R), bool), np.ones(len(D), bool)]
    wt = np.r_[nr / nr.sum(), nd / nd.sum()]
    cs = (CB * CB).sum(1)
    scale = np.sqrt(np.median(np.partition(((R[:, None, :] - R[None, :200, :]) ** 2).sum(2), 1, axis=0)[1]))
    out = {k: np.empty(len(E)) for k in ("cb_f5", "cb_f15", "cb_f40", "cb_wf15", "cb_ratio", "cb_soft1", "cb_soft2")}
    for s in range(0, len(E), 4096):
        e = E[s : s + 4096]
        d2 = np.maximum((e * e).sum(1)[:, None] + cs[None, :] - 2.0 * e @ CB.T, 0.0)
        nb = np.argsort(d2, axis=1)[:, : max(ks)]
        for kk in ks:
            out[f"cb_f{kk}"][s : s + 4096] = isd[nb[:, :kk]].mean(1)
        nb15 = nb[:, :15]
        w = wt[nb15]
        out["cb_wf15"][s : s + 4096] = (w * isd[nb15]).sum(1) / w.sum(1)
        dr = np.sqrt(d2[:, : len(R)].min(1))
        dd = np.sqrt(d2[:, len(R) :].min(1))
        out["cb_ratio"][s : s + 4096] = np.log((dr + 1e-3) / (dd + 1e-3))
        for key, tau in (("cb_soft1", scale), ("cb_soft2", 2 * scale)):
            lr = -np.log(np.exp(-np.sqrt(d2[:, : len(R)]) / tau).sum(1) + 1e-30)
            ld = -np.log(np.exp(-np.sqrt(d2[:, len(R) :]) / tau).sum(1) + 1e-30)
            out[key][s : s + 4096] = lr - ld
    out["lite_entropy"] = o[:, mdl.K].astype(np.float64)
    return out


def cmd_feats(a):
    import doublet_v2 as S
    from singlet.features import doublet as D
    from singlet.features.doublet_lite import Distilled

    out = Path(a.out) / "lite"
    mdl = Distilled.load(out / a.model)
    cb = np.load(out / f"codebook_{a.k}.npz")
    dst = out / f"feats_{a.tag}"
    dst.mkdir(parents=True, exist_ok=True)
    cost = {}
    for n in D.STUDY:
        X, y, genes, species = S.load_counts(n)
        w0, c0 = time.perf_counter(), time.process_time()
        F = cb_features(mdl, cb, X, genes, species)
        cost[n] = dict(n_cells=int(X.shape[0]), wall_s=round(time.perf_counter() - w0, 2), cpu_s=round(time.process_time() - c0, 2))
        np.savez(dst / f"{n}.npz", **F)
        print(n, cost[n], flush=True)
    (dst / "cost.json").write_text(json.dumps(cost, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--model", default="distill_aug_1000_256_32.npz")
    b.add_argument("--k", type=int, default=1500)
    b.add_argument("--cells", type=int, default=150000)
    f = sub.add_parser("feats")
    f.add_argument("--out", required=True)
    f.add_argument("--model", default="distill_aug_1000_256_32.npz")
    f.add_argument("--k", type=int, default=1500)
    f.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "feats": cmd_feats}[a.cmd](a)


if __name__ == "__main__":
    main()
