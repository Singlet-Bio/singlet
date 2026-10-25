# SPDX-License-Identifier: MIT
"""Hashtag demultiplexing benchmark against genotype truth: Singlet vs HTODemux, MULTIseqDemux, hashsolo, demuxEM.

    python -P bench/adt/hto_benchmark.py --data $WORK/data/adt --truth $WORK/state/adt/hto_truth.tsv \
        --out $WORK/state/adt/hto_bench [--max-pools 20] [--methods singlet,htodemux,...]

Truth comes from the GEX library, not from the hashtags: the pipeline's genotype-free donor demultiplexing
(``samples/<GSM>/donor/donor_assignments.tsv``: ``cell``, ``donor_id``, ``cell_status``) gives each cell's donor and flags
genotype doublets. Pools are used when they hold >= 2 donors with >= 200 genotype singlets each.

Each hashtag is mapped to a donor once, independently of every method: for genotype singlets, the hashtag with the largest
count relative to its ambient share (the cell's top hashtag), majority donor per hashtag; hashtags whose cells are < 80% one
donor, or with < 30 cells, are left unmapped (their calls are not scored). Metrics per method and pool:

* ``accuracy``: genotype singlets called as a singlet with a hashtag of their own donor / genotype singlets with a mapped call
  or a negative/doublet call (a negative or doublet call on a genotype singlet counts as not correct);
* ``wrong_donor``: called singlet with a hashtag of the other donor (the error that mixes samples);
* ``negative``, ``false_doublet``: genotype singlets called negative / doublet (doublet calls can be right for
  same-donor doublets, which genotypes cannot see, so this is an upper bound of the false-doublet rate);
* ``doublet_recall``: genotype doublets (two donors) called doublet;
* ``seconds``: the method's own wall time.

Writes ``pools.tsv``, ``metrics.tsv`` (pool x method) and ``summary.json`` (medians over pools).
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_methods as HM  # noqa: E402

MIN_SINGLETS_PER_DONOR = 200
MAP_PURITY = 0.80
MAP_MIN_CELLS = 30


def load_pool(data, gse, gsm):
    from singlet.adt import count as C
    from singlet.adt import gex as G
    from singlet.adt import normalize as N
    from singlet.v2.legacy import LegacyBundle

    d = os.path.join(data, gse, gsm)
    M, feats, amb, _, meta = C.load_npz(os.path.join(d, "counts.npz"))
    pan = pd.read_csv(os.path.join(d, "panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
    gx = G.read_legacy(gse, meta["gex_gsm"])  # exon counts: demuxEM's RNA input
    h = np.flatnonzero(pan.kind.values == "hashtag")
    bcs = np.array([b.split("-")[0] for b in gx.barcodes])
    called = np.asarray(gx.called, bool)
    emp = N.empty_droplets(np.asarray(M.sum(axis=0)).ravel(), called)
    names = list(pan.name.iloc[h])
    X = M[h].tocsc()
    hto = pd.DataFrame(X[:, np.flatnonzero(called)].toarray(), index=names, columns=bcs[called])
    empties = pd.DataFrame(X[:, np.flatnonzero(emp)].toarray(), index=names, columns=bcs[emp])
    b = LegacyBundle.open(f"https://data.singlet.bio/data/{gse}/{gse}.singlet")
    dn = pd.read_csv(io.BytesIO(b.read(f"samples/{meta['gex_gsm']}/donor/donor_assignments.tsv")), sep="\t", dtype=str)
    dn["cell"] = dn["cell"].str.split("-").str[0]
    dn = dn.drop_duplicates("cell").set_index("cell").reindex(hto.columns)
    import anndata

    keep = np.flatnonzero(called | emp)
    rna = anndata.AnnData(gx.exon[:, keep].T.tocsr().astype(np.float32))
    rna.obs_names = list(bcs[keep])
    rna.var_names = [str(g) for g in gx.genes]
    return hto, empties, dn, meta["gex_gsm"], rna


def map_hashtags(hto, empties, dn):
    """hashtag -> donor from genotype singlets and each cell's top ambient-scaled hashtag (method-independent)."""
    amb = empties.sum(axis=1).values.astype(float) + 1
    amb = amb / amb.sum()
    top = (hto.values / amb[:, None]).argmax(axis=0)
    sing = dn["cell_status"].astype(str).str.lower().eq("singlet").values & dn["donor_id"].notna().values
    m, rep = {}, {}
    for k, name in enumerate(hto.index):
        sel = sing & (top == k)
        vc = dn["donor_id"][sel].value_counts()
        if vc.sum() < MAP_MIN_CELLS:
            rep[name] = {"cells": int(vc.sum()), "mapped": None}
            continue
        purity = vc.iloc[0] / vc.sum()
        rep[name] = {"cells": int(vc.sum()), "donor": vc.index[0], "purity": round(float(purity), 3)}
        if purity >= MAP_PURITY:
            m[name] = vc.index[0]
    return m, rep


def score(calls, dn, hmap):
    st = dn["cell_status"].astype(str).str.lower()
    donor = dn["donor_id"]
    sing = st.eq("singlet") & donor.notna()
    dbl = st.eq("doublet")
    c = calls["call"].reindex(dn.index)
    called_donor = c.map(hmap)
    scorable = sing & (c.isin(["Negative", "Doublet"]) | called_donor.notna())
    n = max(int(scorable.sum()), 1)
    return {
        "genotype_singlets": int(scorable.sum()),
        "genotype_doublets": int(dbl.sum()),
        "accuracy": float((scorable & (called_donor == donor)).sum() / n),
        "wrong_donor": float((scorable & called_donor.notna() & (called_donor != donor)).sum() / n),
        "negative": float((scorable & c.eq("Negative")).sum() / n),
        "false_doublet": float((scorable & c.eq("Doublet")).sum() / n),
        "doublet_recall": float((dbl & c.eq("Doublet")).sum() / max(int(dbl.sum()), 1)),
        "singlet_call_share": float(c.isin(list(hmap) + [x for x in c.unique() if x not in ("Negative", "Doublet")]).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pools", type=int, default=25)
    ap.add_argument("--methods", default="singlet,htodemux,multiseqdemux,hashsolo,demuxem")
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(a.truth, sep="\t")
    t = t[t.donor_file.astype(str).eq("True") & (t.n_donors >= 2)].sort_values("n_assigned", ascending=False)
    pools, metrics = [], []
    for r in t.itertuples():
        if len([p for p in pools if p["used"]]) >= a.max_pools:
            break
        rec = {"gse": r.gse, "gsm": r.gsm, "used": False}
        try:
            hto, empties, dn, gex, rna_all = load_pool(a.data, r.gse, r.gsm)
            sing = dn["cell_status"].astype(str).str.lower().eq("singlet")
            per_donor = dn["donor_id"][sing].value_counts()
            rec.update(gex_gsm=gex, cells=hto.shape[1], hashtags=hto.shape[0], donors=int(len(per_donor)),
                       min_donor_singlets=int(per_donor.min()) if len(per_donor) else 0)
            if len(per_donor) < 2 or per_donor.min() < MIN_SINGLETS_PER_DONOR:
                rec["skip"] = "fewer than 2 donors with enough genotype singlets"
                pools.append(rec)
                continue
            hmap, rep = map_hashtags(hto, empties, dn)
            rec["hashtag_map"] = json.dumps(rep)
            if len(set(hmap.values())) < 2:
                rec["skip"] = "hashtags do not separate donors (pool design is not one donor per hashtag)"
                pools.append(rec)
                continue
            rec["used"] = True
            pools.append(rec)
            for m in a.methods.split(","):
                fn = HM.METHODS[m]
                wd = os.path.join(a.workdir or a.out, "work", r.gsm, m)
                os.makedirs(wd, exist_ok=True)
                t0 = time.time()
                try:
                    if m == "demuxem":
                        calls = fn(hto, empties, rna_all, workdir=wd)
                    else:
                        calls = fn(hto, empties, workdir=wd)
                    res = score(calls, dn, hmap)
                    res["seconds"] = float(calls.attrs.get("seconds", time.time() - t0))
                except Exception as e:  # a method failing on a pool is a result too
                    res = {"error": f"{type(e).__name__}: {e}"[:300]}
                metrics.append({"gse": r.gse, "gsm": r.gsm, "method": m, **res})
                print(r.gsm, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
        except Exception as e:
            rec["skip"] = f"error {type(e).__name__}: {e}"[:300]
            pools.append(rec)
    P = pd.DataFrame(pools)
    Mt = pd.DataFrame(metrics)
    P.to_csv(os.path.join(a.out, "pools.tsv"), sep="\t", index=False)
    Mt.to_csv(os.path.join(a.out, "metrics.tsv"), sep="\t", index=False)
    summ = {}
    if len(Mt) and "accuracy" in Mt:
        ok = Mt[Mt.accuracy.notna()]
        summ = {m: {k: float(g[k].median()) for k in ("accuracy", "wrong_donor", "negative", "false_doublet",
                                                      "doublet_recall", "seconds")} | {"pools": int(len(g))}
                for m, g in ok.groupby("method")}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"pools_used": int(P.used.sum()) if len(P) else 0, "methods": summ}, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.exit(main())
