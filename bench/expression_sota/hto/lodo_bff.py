# SPDX-License-Identifier: MIT
"""Task 4 (PREREG section 6): does a BFF-style caller close the gap on Howitt 2023 (leave-one-dataset-out)?

    python -P lodo_bff.py --zenodo DIR --artifacts bpub/artifacts/hto --out lodo_bff.json

Per dataset (BAL1-3, cell lines) the captures are called separately and scored over the whole dataset, as in
bench/adt/hto_lodo.py. Reported for every dataset: the paper's mean per-hashtag F and doublets-called-singlet (comparable to the
release gate) and the PREREG primary metric (macro F1 of singlet / doublet / negative with donor-correct singlets).

Variants, fixed before any score was looked at:
  fixed defaults (no selection): current, hasheddrops, gmm, htodemux, bff_raw, bff_cluster(alpha = beta = 0.05, edge / top),
  family ``bffc``: bff_cluster over alpha, beta in {0.02, 0.05, 0.1, 0.2}^2 and t2_rule in {edge_M, top}; nested LODO: for each
  held-out dataset the setting with the best mean macro F1 over the other three datasets.
The authors' published BFF_raw / BFF_cluster calls are scored with the same code for reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hto_callers as H  # noqa: E402

GRID = [(a, b, r) for r in ("edge_M", "top") for a in (0.02, 0.05, 0.1, 0.2) for b in (0.02, 0.05, 0.1, 0.2)]
SHORT = {"BAL1": "bal1", "BAL2": "bal2", "BAL3": "bal3", "CL": "cl"}


def gname(g):
    return f"bffc_{g[0]}_{g[1]}_{g[2]}"


def metrics(call, truth, t2t, hmap, tags, boot=0):
    keep = pd.notna(truth)
    call, truth = call[keep], truth[keep]
    names = np.array([tags[c] if c >= 0 else ("Negative" if c == H.NEG else "Doublet") for c in call], dtype=object)
    s = H.L.score(names, truth, hmap)
    m, per = H.macro_f1(call, truth, t2t)
    out = {"paper_f": s["f"], "dbl_singlet": s["dbl_singlet"], "macro_f1": m, "F_S": per["S"], "F_D": per["D"], "F_N": per["N"],
           "called_SDN": [int((call >= 0).sum()), int((call == H.DBL).sum()), int((call == H.NEG).sum())]}
    if boot:
        ci_m, ci_s = H.bootstrap(call, truth, t2t, n_boot=boot)
        out.update(macro_f1_ci=ci_m, dbl_singlet_ci=ci_s)
    return out


def published(P, allbc, ntags, col, lab2idx=None):
    Pa = P.drop_duplicates("barcode").set_index("barcode").reindex(allbc) if len(P) != len(allbc) or not (P.barcode.values == np.array(allbc)).all() else P.set_index("barcode", drop=False).reset_index(drop=True)
    v = Pa[col].astype(object).values
    out = np.full(len(v), H.NEG)
    for i, x in enumerate(v):
        if x == "Doublet":
            out[i] = H.DBL
        elif isinstance(x, str) and x != "Negative":
            out[i] = lab2idx[x] if lab2idx and x in lab2idx else (int(x.split()[-1]) - 1) % ntags
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", required=True)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--no-grid", action="store_true")
    a = ap.parse_args()
    data = H.L.load_howitt(a.zenodo)
    if a.datasets:
        data = {k: v for k, v in data.items() if k in a.datasets.split(",")}
    fixed = {"current": H.current, "hasheddrops": H.hasheddrops, "gmm": H.gmm, "htodemux": H.htodemux, "bff_raw": lambda X: H.bff_raw(X)[0]}
    table, prep = {}, {}
    for ds, (caps, hmap) in data.items():
        tags = list(caps[0][1].index)
        t2t = {i: hmap[t] for i, t in enumerate(tags)}
        truth = np.concatenate([tr.to_numpy(dtype=object) for _, _, tr in caps])
        allbc = [b for _, h, _ in caps for b in h.columns]
        table[ds] = {}
        for name, fn in fixed.items():
            t0 = time.time()
            call = np.concatenate([np.asarray(fn(h.to_numpy(dtype=float))) for _, h, _ in caps])
            table[ds][name] = {**metrics(call, truth, t2t, hmap, tags, a.boot), "sec": round(time.time() - t0, 1)}
            print(ds, name, {k: round(v, 3) for k, v in table[ds][name].items() if k in ("paper_f", "dbl_singlet", "macro_f1")}, flush=True)
        prep[ds] = [H.bff_prepare(h.to_numpy(dtype=float)) for _, h, _ in caps]
        grid = [(0.05, 0.05, "top")] if a.no_grid else GRID
        for g in grid:
            call = np.concatenate([H.bff_decide(P, g[0], g[1], "edge", g[2])[0] for P in prep[ds]])
            table[ds][gname(g)] = metrics(call, truth, t2t, hmap, tags, a.boot if g == (0.05, 0.05, "top") else 0)
        table[ds]["bff_cluster"] = table[ds][gname((0.05, 0.05, "top"))]
        print(ds, "bff_cluster", {k: round(v, 3) for k, v in table[ds]["bff_cluster"].items() if k in ("paper_f", "dbl_singlet", "macro_f1")}, flush=True)
        if a.artifacts:
            P = pd.read_parquet(os.path.join(a.artifacts, f"hto_howitt2023_{SHORT[ds]}_published.parquet"))
            mp = json.load(open(os.path.join(a.artifacts, f"hto_howitt2023_{SHORT[ds]}_map.json")))["hashtag_to_donor"]
            don2idx = {hmap[t]: i for i, t in enumerate(tags)}
            lab2idx = {lab: don2idx[d] for lab, d in mp.items() if d in don2idx}
            for col in ("BFF_raw", "BFF_cluster", "HTODemux", "hashedDrops_default", "GMMDemux"):
                if col in P.columns:
                    call = published(P, allbc, len(tags), col, lab2idx)
                    table[ds]["published_" + col] = metrics(call, truth, t2t, hmap, tags, 0)
                    print(ds, "published", col, {k: round(v, 3) for k, v in table[ds]["published_" + col].items() if k in ("paper_f", "dbl_singlet", "macro_f1")}, flush=True)
    res = {"datasets": table, "grid": [list(g) for g in GRID]}
    if not a.no_grid:
        held = {}
        for test in table:
            others = [d for d in table if d != test]
            best = max(GRID, key=lambda g: np.mean([table[d][gname(g)]["macro_f1"] for d in others]))
            held[test] = {"chosen": list(best), **table[test][gname(best)]}
        res["bffc_nested_lodo"] = held
        res["bffc_all_dataset_choice"] = list(max(GRID, key=lambda g: np.mean([table[d][gname(g)]["macro_f1"] for d in table])))
        for t, v in held.items():
            print("nested", t, v["chosen"], round(v["paper_f"], 3), round(v["dbl_singlet"], 3), round(v["macro_f1"], 3), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: None)


if __name__ == "__main__":
    main()
