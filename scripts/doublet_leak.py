#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Leakage screen: do the atlas reference cells of the doublet encoder (and the rest of the celltype_nn corpus) overlap with the 17
doublet evaluation datasets?  Clipper job, real data only.

1. metadata: corpus dataset titles / collections matching the evaluation studies (the corpus obs carries no barcodes: cell_id is the
   Census soma_joinid, so barcode-level matching is impossible);
2. cell level: every evaluation cell is projected on the corpus' 6,000-gene vocabulary and compared with every corpus cell by
   (a) a hash of the exact projected count vector, (b) a hash of the set of its 12 most abundant genes (near-duplicate; tolerant of
   rescaling, not of resampling), reported against the 316k encoder reference cells and against the whole corpus (2.8M cells);
   the sensitivity of (b) to 20% / 50% binomial thinning of the evaluation cells is measured as a control.

  python scripts/doublet_leak.py --out OUT
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

CORPUS = "/mnt/projects/debruinz_project/debruinz/singlet/data/celltype_nn"
TOP = 12
M1, M2, M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F), np.uint64(0xBF58476D1CE4E5B9)
KEYWORDS = r"kang|lupus|demux|multiplex|multi-?seq|hashing|hashtag|cell hash|\bHEK|HMEC|293|mixture|hgmm|human and mouse|PBMC|peripheral blood|kidney|nuclei|xenograft|PDX|cell line|mammary|mcginnis|stoeckius|gehring|xi and li"


def mix(x):
    x = x.astype(np.uint64)
    x ^= x >> np.uint64(29)
    x *= M3
    x ^= x >> np.uint64(32)
    return x


def exact_hash(X):
    """Per-row hash of the sparse count vector (columns sorted, integer counts)."""
    X = sp.csr_matrix(X)
    X.sort_indices()
    v = np.rint(X.data).astype(np.int64)
    h = mix((X.indices.astype(np.uint64) + np.uint64(1)) * M1 ^ (v.astype(np.uint64) + np.uint64(1)) * M2)
    out = np.zeros(X.shape[0], np.uint64)
    nz = np.diff(X.indptr) > 0
    if X.nnz:
        out[nz] = np.add.reduceat(h, X.indptr[:-1][nz])
    return out


def topset_hash(X, table):
    X = sp.csr_matrix(X)
    X.sort_indices()
    out = np.zeros(X.shape[0], np.uint64)
    ip, ix, dt = X.indptr, X.indices, np.rint(X.data).astype(np.int64)
    for i in range(X.shape[0]):
        a, b = ip[i], ip[i + 1]
        if b - a < TOP:
            continue
        key = dt[a:b] * 8192 + (8191 - ix[a:b])
        sel = np.argpartition(-key, TOP - 1)[:TOP]
        out[i] = table[ix[a:b][sel]].sum()
    return out


def project(genes, vocab):
    """genes x vocab 0/1 matrix: Ensembl ids (human, or mouse through the ortholog table) or upper-case symbols."""
    g = np.array([str(x).split(".")[0] for x in genes])
    g = np.array([x.split("_", 1)[1] if x[:5] in ("hg19_", "mm10_", "GRCh3", "GRCm3", "hg38_") and "_" in x else x for x in g])
    index = {x: i for i, x in enumerate(vocab["genes"])}
    if np.mean([x.startswith("ENS") for x in g]) > 0.5:
        cols = np.array([index.get(x, index.get(vocab["mouse"].get(x, ""), -1)) for x in g])
    else:
        sym = {s.upper(): i for i, s in enumerate(vocab["symbols"])}
        cols = np.array([sym.get(x.upper(), -1) for x in g])
    keep = np.flatnonzero(cols >= 0)
    return sp.csr_matrix((np.ones(len(keep), np.float32), (keep, cols[keep])), shape=(len(g), len(vocab["genes"])))


def lookup(sorted_h, order, he):
    pos = np.searchsorted(sorted_h, he)
    pos = np.minimum(pos, len(sorted_h) - 1)
    hit = (sorted_h[pos] == he) & (he != 0)
    return hit, order[pos]


def main():
    import doublet_v2 as S
    from singlet.features.doublet import STUDY

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    obs = pd.read_parquet(f"{CORPUS}/corpus_obs.parquet").reset_index(drop=True)
    vocab = json.load(open(f"{CORPUS}/vocab.json"))
    # the encoder reference cells: the selection of scripts/doublet_encoder_select.py (min 50 cells per term, cap 400)
    col = next(c for c in obs.columns if obs[c].astype(str).str.startswith("CL:").mean() > 0.9)
    r = np.random.default_rng(0)
    obs["_r"] = r.random(len(obs))
    counts = obs[col].value_counts()
    terms = counts.index[counts >= 50].tolist()
    obs["_rank"] = obs.groupby([col, "dataset_id"])["_r"].rank(method="first")
    selidx = np.sort(obs[obs[col].isin(terms)].sort_values([col, "_rank", "_r"]).groupby(col).head(400).index.values)
    insel = np.zeros(len(obs), bool)
    insel[selidx] = True
    rep = {"corpus_cells": len(obs), "encoder_reference_cells": int(insel.sum()), "corpus_datasets": int(obs.dataset_id.nunique()),
           "encoder_reference_datasets": int(obs.dataset_id[insel].nunique())}
    # 1. metadata
    meta = obs.groupby("dataset_id").agg(title=("dataset_title", "first"), collection=("collection_id", "first"), assay=("assay", "first"),
                                         tissue=("tissue_general", "first"), cells=("soma_joinid", "size"))
    meta["in_reference"] = obs.groupby("dataset_id")["soma_joinid"].apply(lambda s: bool(insel[s.index].any()))
    hit = meta[meta.title.astype(str).str.contains(KEYWORDS, flags=re.I, regex=True)]
    rep["metadata_keyword_hits"] = [dict(dataset_id=i, title=str(r_.title)[:110], assay=r_.assay, tissue=r_.tissue, cells=int(r_.cells),
                                         in_reference=bool(r_.in_reference)) for i, r_ in hit.iterrows()]
    print("metadata keyword hits:", len(hit), flush=True)
    for h in rep["metadata_keyword_hits"][:60]:
        print("  ", h, flush=True)
    # 2. cell level
    X = sp.load_npz(f"{CORPUS}/corpus_counts.npz").tocsr()
    print("corpus loaded", X.shape, f"{time.time() - t0:.0f}s", flush=True)
    table = mix(np.arange(1, X.shape[1] + 1, dtype=np.uint64) * M1)
    hx = np.concatenate([exact_hash(X[s : s + 200000]) for s in range(0, X.shape[0], 200000)])
    print("exact hashes", f"{time.time() - t0:.0f}s", flush=True)
    ht = np.concatenate([topset_hash(X[s : s + 200000], table) for s in range(0, X.shape[0], 200000)])
    print("top-set hashes", f"{time.time() - t0:.0f}s", flush=True)
    del X
    ox, ot = np.argsort(hx), np.argsort(ht)
    sx, st = hx[ox], ht[ot]
    ordx, ordt = np.argsort(hx[selidx]), np.argsort(ht[selidx])  # reference cells only
    rx, rt = hx[selidx][ordx], ht[selidx][ordt]
    rep["corpus_internal_duplicates_exact"] = int((np.diff(sx[sx != 0]) == 0).sum())
    res = {}
    for n in STUDY:
        Xe, y, genes, species = S.load_counts(n)
        P = project(genes, vocab)
        V = sp.csr_matrix(sp.csr_matrix(Xe, dtype=np.float32) @ P)
        he, hte = exact_hash(V), topset_hash(V, table)
        hit_x, row_x = lookup(sx, ox, he)
        hit_t, row_t = lookup(st, ot, hte)
        hit_xr, _ = lookup(rx, ordx, he)
        hit_tr, _ = lookup(rt, ordt, hte)
        # control: sensitivity of the top-set hash to thinning
        rng = np.random.default_rng(0)
        sub = rng.choice(V.shape[0], min(2000, V.shape[0]), replace=False)
        sens = {}
        for f in (0.8, 0.5):
            T = V[sub].copy()
            T.data = rng.binomial(np.rint(T.data).astype(np.int64), f).astype(np.float32)
            T.eliminate_zeros()
            ok = (hte[sub] != 0)
            sens[str(f)] = round(float((topset_hash(T, table)[ok] == hte[sub][ok]).mean()), 3)
        titles = {}
        for rows in (row_x[hit_x], row_t[hit_t]):
            for t, c in obs.dataset_title.iloc[rows].value_counts().head(3).items():
                titles[str(t)[:80]] = int(c)
        res[n] = dict(cells=int(V.shape[0]), scored_for_topset=int((hte != 0).sum()),
                      exact_match_corpus=int(hit_x.sum()), exact_match_reference=int(hit_xr.sum()),
                      topset_match_corpus=int(hit_t.sum()), topset_match_reference=int(hit_tr.sum()),
                      topset_sensitivity_thinning=sens, matched_titles=titles)
        print(n, res[n], flush=True)
    rep["datasets"] = res
    rep["totals"] = {k: int(sum(d[k] for d in res.values())) for k in ("cells", "exact_match_corpus", "exact_match_reference", "topset_match_corpus", "topset_match_reference")}
    out = Path(a.out) / "leak"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print("TOTALS", rep["totals"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
