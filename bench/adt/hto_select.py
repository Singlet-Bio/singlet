# SPDX-License-Identifier: MIT
"""Leave-one-dataset-out choice of Singlet's hashtag caller on the Howitt et al. 2023 benchmark (no tuning on test data).

    python -P bench/adt/hto_select.py --fscores OUT/fscores.tsv --out OUT/selection.json

Criterion, fixed before looking at held-out results: among the Singlet variants (``hd_*``, ``mix_*``, ``hyb_*``), the one with
the highest mean F-score over the training datasets, among those calling <= ``MAX_DOUBLET_SINGLET`` of genetic doublets
singlets on every training dataset. Each dataset is held out in turn; its reported number is the chosen variant's. The
variant chosen on all four datasets together is the one Singlet ships, and that choice is reported separately.
"""

import argparse
import json

import pandas as pd

MAX_DOUBLET_SINGLET = 0.20


def choose(df, datasets):
    d = df[df.dataset.isin(datasets)]
    g = d.groupby("method").agg(f=("f_mean", "mean"), dmax=("doublets_called_singlet", "max"),
                                n=("dataset", "nunique"))
    g = g[(g.n == len(datasets)) & (g.dmax <= MAX_DOUBLET_SINGLET)]
    return (g.f.idxmax(), g) if len(g) else (None, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fscores", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = pd.read_csv(a.fscores, sep="\t")
    runs = f[(f.source == "this run") & f.f_mean.notna()]
    cand = runs[runs.method.str.match(r"^(hd_|mix_|hyb_)")]
    pub = f[f.source == "published calls"]
    ds = sorted(cand.dataset.unique())
    held = []
    for test in ds:
        m, _ = choose(cand, [x for x in ds if x != test])
        row = cand[(cand.dataset == test) & (cand.method == m)].iloc[0] if m else None
        best_pub = pub[pub.dataset == test].sort_values("f_mean", ascending=False)
        held.append({"held_out": test, "chosen_on_others": m,
                     "f_mean": None if row is None else float(row.f_mean),
                     "doublets_called_singlet": None if row is None else float(row.doublets_called_singlet),
                     "best_published": best_pub.iloc[0].method if len(best_pub) else None,
                     "best_published_f": float(best_pub.iloc[0].f_mean) if len(best_pub) else None,
                     "best_published_doublets_called_singlet":
                         float(best_pub.iloc[0].doublets_called_singlet) if len(best_pub) else None})
    final, table = choose(cand, ds)
    out = {"criterion": f"max mean F over training datasets; doublets called singlet <= {MAX_DOUBLET_SINGLET} on each",
           "held_out": held, "shipped_variant": final,
           "shipped_variant_per_dataset": cand[cand.method == final][["dataset", "f_mean", "doublets_called_singlet"]]
           .to_dict("records") if final else []}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
