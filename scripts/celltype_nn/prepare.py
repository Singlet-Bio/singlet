#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Turn the fetched Census chunks into the training matrix: choose the gene vocabulary, project every cell onto it, and measure
per-gene intron/exon ratios on Singlet's own counts (for the intron augmentation).

Vocabulary = highly variable genes of the corpus (per tissue, human; plus mouse through one-to-one orthologs) and every guide
marker gene, kept only when Singlet's pipeline annotation has the gene (``--singlet-genes``: bundles whose gene ids are read,
nothing else), and without mitochondrial, ribosomal-protein and sex-chromosome-dosage genes (XIST and the chrY genes the sex
model uses), which track protocol or donor rather than cell identity.

  python scripts/celltype_nn/prepare.py --data $D --singlet-genes human=<bundle>,mouse=<bundle> --intron-bundles <b1>,<b2>,...

Writes ``<data>/vocab.json``, ``<data>/corpus_counts.npz`` (cells x vocabulary, float32 raw counts, plan order),
``<data>/corpus_obs.parquet`` and ``<data>/intron_ratio.npy``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

SEX_GENES = {"XIST", "TSIX", "RPS4Y1", "DDX3Y", "UTY", "KDM5D", "EIF1AY", "ZFY", "USP9Y", "TXLNGY", "NLGN4Y", "PRKY"}
_DROP = re.compile(r"^(MT-|RPL\d|RPS\d|RPLP|RPSA$|MRPL|MRPS)", re.I)


def _bundle_genes(path: str):
    from singlet.bundle import SingletBundle

    with SingletBundle(Path(path)) as b:
        for g in b.gsm_ids:
            try:
                a = b.raw_counts(g)
                return [str(x).split(".")[0] for x in a.var_names]
            except Exception:  # noqa: BLE001
                continue
    raise ValueError(f"{path}: no readable sample")


def _hvg(X: sps.csr_matrix, batch: np.ndarray, n_top: int) -> np.ndarray:
    """Dispersion-based HVGs (mean-binned normalised dispersion of log1p CP10k), ranked by how many batches select them."""
    from singlet.celllabels.celltype import lognorm

    L = lognorm(X)
    score = np.zeros(X.shape[1])
    for b in np.unique(batch):
        m = batch == b
        if m.sum() < 500:
            continue
        S = L[np.flatnonzero(m)]
        mu = np.asarray(S.mean(0)).ravel()
        var = np.asarray(S.multiply(S).mean(0)).ravel() - mu**2
        disp = np.log(np.maximum(var, 1e-12) / np.maximum(mu, 1e-12))
        bins = np.digitize(mu, np.quantile(mu[mu > 0], np.linspace(0, 1, 21)))
        z = np.zeros_like(disp)
        for k in np.unique(bins):
            sel = (bins == k) & (mu > 0.01)
            if sel.sum() > 2:
                z[sel] = (disp[sel] - disp[sel].mean()) / (disp[sel].std() + 1e-9)
        top = np.argsort(-z)[:n_top]
        score[top] += 1
    return score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--singlet-genes", required=True, help="human=<bundle>,mouse=<bundle>")
    ap.add_argument("--intron-bundles", default="", help="comma-separated Singlet bundles (not benchmark studies)")
    ap.add_argument("--n-hvg", type=int, default=3000, help="per tissue batch")
    ap.add_argument("--n-vocab", type=int, default=6000)
    ap.add_argument("--hvg-cells", type=int, default=300000)
    a = ap.parse_args(argv)

    import anndata as ad
    import pandas as pd

    from singlet.celllabels.guide import load_nodes
    from singlet.celltype_nn.features import GeneVocab, read_orthologs

    t0 = time.time()
    plan = pd.read_parquet(a.data / "plan.parquet")
    m2h = read_orthologs(a.data / "orthologs_human_mouse.tsv.gz")
    sg = dict(kv.split("=", 1) for kv in a.singlet_genes.split(","))
    singlet_h = set(_bundle_genes(sg["human"]))
    singlet_m = set(_bundle_genes(sg["mouse"])) if "mouse" in sg else set()
    chunks = sorted((a.data / "chunks").glob("*.h5ad"))
    print(f"{len(chunks)} chunks, {len(plan)} planned cells", flush=True)

    # ---- human gene table and HVGs on a subsample
    rng = np.random.default_rng(0)
    first_h = None
    hv_parts, hv_batch = [], []
    per_chunk = max(1000, a.hvg_cells // max(len(chunks), 1))
    mouse_hv = []
    for p in chunks:
        A = ad.read_h5ad(p)
        sp = plan.loc[plan["chunk"] == int(p.stem), "organism"].iloc[0]
        take = rng.permutation(A.n_obs)[:per_chunk]
        pc = plan[plan["chunk"] == int(p.stem)].reset_index(drop=True)
        assert (pc["soma_joinid"].astype(str).to_numpy() == A.obs_names.to_numpy()).all(), p
        obs = pc.iloc[take]
        if sp == "human":
            if first_h is None:
                first_h = A.var[["feature_id", "feature_name"]].copy()
            hv_parts.append(sps.csr_matrix(A.X[take]))
            hv_batch.append(obs["tissue_general"].to_numpy())
        else:
            mouse_hv.append((sps.csr_matrix(A.X[take]), obs["tissue_general"].to_numpy(), A.var["feature_id"].astype(str).to_numpy()))
    genes_h = first_h["feature_id"].astype(str).str.split(".").str[0].to_numpy()
    sym_h = first_h["feature_name"].astype(str).to_numpy()
    Xh = sps.vstack(hv_parts).tocsr()
    score = _hvg(Xh, np.concatenate(hv_batch), a.n_hvg)
    # mouse HVGs, carried to human ids
    if mouse_hv:
        Xm = sps.vstack([x for x, _, _ in mouse_hv]).tocsr()
        sm = _hvg(Xm, np.concatenate([b for _, b, _ in mouse_hv]), a.n_hvg)
        hidx = {g: i for i, g in enumerate(genes_h)}
        for g, s in zip(mouse_hv[0][2], sm):
            h = m2h.get(g.split(".")[0])
            if h in hidx:
                score[hidx[h]] += s
    markers = set()
    for n in load_nodes():
        for k in ("pos", "neg"):
            markers |= {str(g) for g in ((n.get("markers") or {}).get(k) or [])}
    mk = np.isin(np.char.upper(sym_h.astype(str)), [m.upper() for m in markers])
    ok = np.isin(genes_h, list(singlet_h))
    ok &= ~np.array([bool(_DROP.match(s)) or s.upper() in SEX_GENES for s in sym_h])
    detect = np.asarray((Xh > 0).mean(0)).ravel()
    ok &= detect > 0.001
    rank = score + 1e-3 * detect
    sel = set(np.flatnonzero(ok & mk))
    for i in np.argsort(-rank):
        if len(sel) >= a.n_vocab:
            break
        if ok[i] and score[i] > 0:
            sel.add(int(i))
    sel = np.array(sorted(sel))
    mouse_map = {m: h for m, h in m2h.items() if h in set(genes_h[sel]) and (not singlet_m or m in singlet_m)}
    vocab = GeneVocab([str(g) for g in genes_h[sel]], [str(s) for s in sym_h[sel]], mouse_map)
    (a.data / "vocab.json").write_text(json.dumps(vocab.to_dict()))
    print(f"vocabulary {len(vocab)} genes ({int(mk[sel].sum())} guide markers), {len(mouse_map)} mouse orthologs, "
          f"{time.time() - t0:.0f}s", flush=True)

    # ---- project every chunk
    parts, obs_parts = [], []
    for p in chunks:
        A = ad.read_h5ad(p)
        sp = plan.loc[plan["chunk"] == int(p.stem), "organism"].iloc[0]
        parts.append(vocab.project(A.X, A.var["feature_id"].astype(str).to_numpy(), sp))
        o = plan[plan["chunk"] == int(p.stem)].reset_index(drop=True)
        assert (o["soma_joinid"].astype(str).to_numpy() == A.obs_names.to_numpy()).all(), p
        o["total_counts"] = np.asarray(A.X.sum(1)).ravel()
        obs_parts.append(o)
        print(f"  {p.name}: {A.n_obs} cells", flush=True)
    X = sps.vstack(parts).tocsr()
    obs = pd.concat(obs_parts, ignore_index=True)
    obs["vocab_counts"] = np.asarray(X.sum(1)).ravel()
    # soma_joinid is per organism in Census: the cell id carries both
    obs["cell_id"] = obs["organism"] + ":" + obs["soma_joinid"].astype(str)
    # copies of benchmark cells found by corpus.py dedup never train or validate
    leak = a.data / "leak_exclude.parquet"
    if leak.exists():
        L = pd.read_parquet(leak)
        bad = obs["cell_id"].isin(set(L["organism"] + ":" + L["soma_joinid"].astype(str)))
        obs.loc[bad, "split"] = "excluded"
        print(f"excluded {int(bad.sum())} copies of benchmark cells", flush=True)
    sps.save_npz(a.data / "corpus_counts.npz", X, compressed=False)
    obs.to_parquet(a.data / "corpus_obs.parquet", index=False)
    print(f"corpus {X.shape}, nnz {X.nnz}, {time.time() - t0:.0f}s", flush=True)

    # ---- intron/exon ratio per vocabulary gene on Singlet counts (X = exon + intron, layer 'spliced' = exon)
    if a.intron_bundles:
        from singlet.bundle import SingletBundle

        ex = np.zeros(len(vocab))
        it = np.zeros(len(vocab))
        for bp in a.intron_bundles.split(","):
            with SingletBundle(Path(bp)) as b:
                for g in b.gsm_ids[:4]:
                    try:
                        A = b.raw_counts(g)
                    except Exception:  # noqa: BLE001
                        continue
                    sp = "mouse" if str(A.var_names[0]).startswith("ENSMUSG") else "human"
                    tot = vocab.project(A.X, A.var_names, sp)
                    spl = vocab.project(A.layers["spliced"], A.var_names, sp)
                    ex += np.asarray(spl.sum(0)).ravel()
                    it += np.asarray((tot - spl).sum(0)).ravel()
            print(f"  intron ratios: {bp}", flush=True)
        r = np.where(ex > 50, it / np.maximum(ex, 1), np.nan)
        r = np.where(np.isnan(r), np.nanmedian(r), r)
        np.save(a.data / "intron_ratio.npy", r.astype(np.float32))
        print(f"intron/exon ratio: median {np.median(r):.2f}, p90 {np.quantile(r, 0.9):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
