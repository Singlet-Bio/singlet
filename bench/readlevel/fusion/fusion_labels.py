# SPDX-License-Identifier: MIT
"""Cell-line labels (K562 or Kasumi-1) for the Penter 2024 mixes, independent of any fusion call.

Two independent signals, both from the engine's run on the mix:
* genotype: the engine's donor demultiplexing (donor_assignments.tsv) splits the cells into genotype clusters (an aneuploid line can
  come out as more than one);
* expression: K562 is an erythroleukemia line with high embryonic/fetal globin (HBG1, HBG2, HBZ, HBE1) and GYPA; Kasumi-1 has none.
Each donor is named by its median globin score; a cell gets a label only when it is a singlet of that donor. The table reports how often
the globin score and the genotype agree and the line ratio, which should follow the mix (1:1.5 or 1:15).

    PZ_SO=.../_pz.so python fusion_labels.py --engine OUT --out labels.tsv --summary labels.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.machinery
import importlib.util
import json
import os

import numpy as np
import scipy.sparse as sp

GLOBIN = {"HBG1", "HBG2", "HBZ", "HBE1", "GYPA",
          "ENSG00000213934", "ENSG00000196565", "ENSG00000130656", "ENSG00000213931", "ENSG00000170180"}


def load_pz():
    so = os.environ["PZ_SO"]
    loader = importlib.machinery.ExtensionFileLoader("singlet._pz", so)
    spec = importlib.util.spec_from_file_location("singlet._pz", so, loader=loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def bare(b):
    return b.split("-")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    pz = load_pz()
    r = pz.read_1pz(os.path.join(a.engine, "exon_counts.1pz"))
    M = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(r["m"], r["n"]))
    names = list(r["rownames"])
    rows = [i for i, n in enumerate(names) if any(t in GLOBIN for t in n.replace("|", "_").split("_")[:2])]  # rows: GENEID_NAME_locus (exons)
    tot = np.asarray(M.sum(axis=0)).ravel()
    glob = np.asarray(M[rows].sum(axis=0)).ravel()
    score = np.log1p(1e4 * glob / np.maximum(tot, 1))
    bcs = [bare(b) for b in r["colnames"]]
    sc = dict(zip(bcs, score))

    called = set()
    with open(os.path.join(a.engine, "cell_calls.tsv")) as f:
        h = f.readline().rstrip("\n").split("\t")
        ic = h.index("is_cell")
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if p[ic].upper() in ("TRUE", "1"):
                called.add(bare(p[0]))
    don = {}
    dpath = os.path.join(a.engine, "donor_assignments.tsv")
    if not os.path.exists(dpath):
        dpath = os.path.join(a.engine, "donor", "donor_assignments.tsv")
    with open(dpath) as f:
        h = f.readline().rstrip("\n").split("\t")
        for ln in f:
            p = dict(zip(h, ln.rstrip("\n").split("\t")))
            don[bare(p["cell"])] = p["donor_id"]
    singlets = {c: d for c, d in don.items() if c in called and d.startswith("donor")}
    by_donor = collections.defaultdict(list)
    for c, d in singlets.items():
        if c in sc:
            by_donor[d].append(sc[c])
    means = {d: float(np.mean(v)) for d, v in by_donor.items()}
    med = {d: float(np.median(v)) for d, v in by_donor.items()}
    if len(med) < 2 or not rows:
        raise SystemExit(f"cannot label: donors {means}, globin rows {len(rows)}")
    # the demultiplexer may split one aneuploid line into several genotype clusters: each donor is named by its median globin
    # score against the midpoint between the highest and lowest donor medians
    thr = (max(med.values()) + min(med.values())) / 2
    line_of = {d: ("K562" if v > thr else "Kasumi-1") for d, v in med.items()}
    lab = {c: line_of[d] for c, d in singlets.items()}
    agree = sum(1 for c, l in lab.items() if c in sc and ((sc[c] > thr) == (l == "K562")))
    with open(a.out, "w") as f:
        f.write("barcode\tline\n")
        for c in sorted(lab):
            f.write(f"{c}\t{lab[c]}\n")
    cnt = collections.Counter(lab.values())
    s = {"cells_called": len(called), "labelled_singlets": len(lab), "lines": dict(cnt),
         "ratio_kasumi_per_k562": round(cnt["Kasumi-1"] / max(cnt["K562"], 1), 3),
         "donor_status": dict(collections.Counter(don[c] for c in called if c in don)),
         "globin_mean_by_donor": means, "globin_median_by_donor": med, "line_of_donor": line_of, "globin_threshold": thr, "genotype_expression_agreement": round(agree / max(len(lab), 1), 4),
         "globin_rows": [names[i] for i in rows]}
    with open(a.summary, "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
