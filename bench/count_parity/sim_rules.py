"""Replay multi-gene UMI rules on a SINGLET_MOLECULE_DUMP table and report the wrong-species fraction per layer.

The dump holds every molecule after UMI collapse, including those the multi-gene UMI rule dropped (label D), so each
alternative rule can be scored without rerunning the pileup. Pooled fractions over pure cells (diagnose.cell_species);
the parity axes use the same cells with per-cell bootstrap CIs, so compare arms with each other and with the baseline.

  python sim_rules.py --dump molecules.tsv.gz --cr CR.h5 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exact_vs_pseudo"))
from diagnose import cell_species  # noqa: E402

LAYERS = {"X": "SUA", "exon": "SA", "intron": "U"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--cr", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = cell_species(a.cr)
    parts = []
    for ch in pd.read_csv(a.dump, sep="\t", usecols=["barcode", "gene_id", "umi", "label", "reads"],
                          dtype={"barcode": str, "gene_id": str, "umi": str, "label": str, "reads": np.int32}, chunksize=5_000_000):
        ch = ch[ch.barcode.isin(sp.keys())]
        parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    d["human_cell"] = d.barcode.map(lambda b: sp[b] == "human")
    d["wrong"] = d.gene_id.str.startswith("ENSG") != d.human_cell
    g = d.groupby(["barcode", "umi"], sort=False)
    d["n"] = g.gene_id.transform("size")
    d["maxr"] = g.reads.transform("max")
    d["nmax"] = (d.reads == d.maxr).groupby([d.barcode, d.umi], sort=False).transform("sum")
    # does the (cell, UMI) also hold a correct-species molecule with more reads (a chimera donor)?
    d["corr_reads"] = np.where(~d.wrong, d.reads, 0)
    d["max_corr"] = d.corr_reads.groupby([d.barcode, d.umi], sort=False).transform("max")
    kept = d.label != "D"
    multi = d.n > 1
    arms = {
        "dumped (CR)": kept,
        "UmiGeneAll": ~multi,
        "CR + drop U/A of multi-gene UMIs": kept & ~(multi & d.label.isin(["U", "A"])),
        "CR + drop U of multi-gene UMIs (velocyto unspliced)": kept & ~(multi & (d.label == "U")),
        "CR, reads>=2 (diagnostic)": kept & (d.reads >= 2),
        "CR, drop wrong-species with a bigger correct-species twin (oracle)": kept & ~(d.wrong & (d.max_corr > d.reads)),
    }
    out = {"n_cells": int(d.barcode.nunique()), "n_molecules": int(len(d)), "arms": {}}
    for name, keep in arms.items():
        r = {}
        for lay, labs in LAYERS.items():
            m = keep & d.label.isin(list(labs))
            tot = int(m.sum()); wr = int((m & d.wrong).sum())
            r[lay] = {"wrong": wr, "total": tot, "frac": wr / max(tot, 1)}
        out["arms"][name] = r
    w = kept & d.wrong & (d.label != "D"); c = kept & ~d.wrong
    out["profile"] = {
        "wrong": {"n": int(w.sum()), "multi_gene_umi": float(multi[w].mean()), "reads1": float((d.reads[w] == 1).mean()),
                  "has_bigger_correct_twin": float((d.max_corr[w] > d.reads[w]).mean()),
                  "label": d.label[w].value_counts().to_dict()},
        "correct": {"n": int(c.sum()), "multi_gene_umi": float(multi[c].mean()), "reads1": float((d.reads[c] == 1).mean())},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    for name, r in out["arms"].items():
        print(f"{name:70s} " + "  ".join(f"{k} {v['frac']*100:.4f}% ({v['wrong']}/{v['total']})" for k, v in r.items()))
    print(json.dumps(out["profile"], indent=1))


if __name__ == "__main__":
    main()
