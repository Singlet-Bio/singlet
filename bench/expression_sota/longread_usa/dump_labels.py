# SPDX-License-Identifier: MIT
"""Per-molecule analysis of Singlet's new USA labels (72ac6555) against the long-read truth; rules: RULES_USA_LABELS.md (pre-registered).

  dump_labels.py X_DIR OUT.json     X_DIR = .../expression-sota/longread_usa; reads labels_dump/molecules.tsv.gz and truth/molecules.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "extract"))
sys.path.insert(0, str(HERE))
import runio as R  # noqa: E402
from score_usa import score_tool, truth_mats  # noqa: E402
from truth_usa import enc  # noqa: E402

S_, U_, A_ = 0, 1, 2
MULTI, SPL, FLANK, OTHER, ANTI, OUTSIDE = 1, 2, 4, 8, 16, 32
BITS = {"Multi": MULTI, "Spliced": SPL, "Flank": FLANK, "OtherGene": OTHER, "Antisense": ANTI, "Outside": OUTSIDE}


def rules(d):
    """boolean 'stays U' for each pre-registered rule, given arrays of a U-labelled-or-not molecule set."""
    r, g = d["reads"], d["diag"]
    u = d["lab"] == U_
    fl, ot, mu = (g & FLANK) > 0, (g & (OTHER | ANTI)) > 0, (g & MULTI) > 0
    c1 = ~((r == 1) & fl)
    c2 = ~fl
    c4 = ~ot
    amb = ot | mu
    return {"C0": u, "C1": u & c1, "C2": u & c2, "C3": u & (r >= 2), "C4": u & c4, "C5": u & c1 & c4,
            "C6": u & c2 & c4, "C7": u & c1 & c4 & ~mu, "C8": u & ~(amb & ~((r >= 2) & c1)) & c1}


def prf(call, truth, mask):
    c, t = call[mask], truth[mask]
    tp, fp, fn = int((c & t).sum()), int((c & ~t).sum()), int((~c & t).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def load_dump(path, cellcodes, gene_idx):
    cols = {k: [] for k in ("key", "cell", "gene", "lab", "flags", "diag", "reads")}
    canon_cache: dict[str, int] = {}
    for ch in pd.read_csv(path, sep="\t", chunksize=4_000_000, dtype={"barcode": str, "gene_id": str, "umi": str, "label": str}):
        ch = ch[ch.label != "D"]
        cb = enc(ch.barcode.str.split("-").str[0].tolist(), 16)
        ub = enc(ch.umi.str[:11].tolist(), 11)
        cell = np.searchsorted(cellcodes, cb)
        cell = np.minimum(cell, len(cellcodes) - 1)
        ok = (cb >= 0) & (cellcodes[cell] == cb)
        ch, cb, ub, cell = ch[ok], cb[ok], ub[ok], cell[ok]
        codes, uniq = pd.factorize(ch.gene_id)
        gm = np.array([gene_idx.get(R.canon_gene(u), -1) for u in uniq], np.int32)
        cols["key"].append(np.where(ub >= 0, (cb << 22) | np.maximum(ub, 0), -1))
        cols["cell"].append(cell.astype(np.int32))
        cols["gene"].append(gm[codes])
        cols["lab"].append(ch.label.map({"S": S_, "U": U_, "A": A_}).to_numpy().astype(np.int8))
        cols["flags"].append(ch["flags"].to_numpy().astype(np.uint8))
        cols["diag"].append(ch["diag"].to_numpy().astype(np.uint8))
        cols["reads"].append(ch["reads"].to_numpy().astype(np.int32))
        print(f"[dump] {sum(len(x) for x in cols['key']):,} molecules in truth cells", file=sys.stderr, flush=True)
    return {k: np.concatenate(v) for k, v in cols.items()}


def mats(d, uU, ncell, ngene):
    """S/U/A csr (cells x genes) for a relabelling: molecules in uU are U; the others are S if diag Spliced else A."""
    ok = d["gene"] >= 0
    r, c = d["cell"][ok], d["gene"][ok]
    u = uU[ok]
    s = (~u) & ((d["diag"][ok] & SPL) > 0)
    a = (~u) & ~s
    f = lambda m: sp.csr_matrix((np.ones(int(m.sum())), (r[m], c[m])), shape=(ncell, ngene))  # noqa: E731
    return {"S": f(s), "U": f(u), "A": f(a)}


def main():
    X = Path(sys.argv[1])
    mol = pd.read_parquet(X / "truth" / "molecules.parquet")
    cells = sorted(mol.CB.unique())
    genes = sorted(mol.gene.unique())
    gi = {g: i for i, g in enumerate(genes)}
    cellcodes = enc(cells, 16)
    assert (np.diff(cellcodes) > 0).all()
    mk = (enc(mol.CB.tolist(), 16) << 22) | enc(mol.UB.tolist(), 11)
    o = np.argsort(mk)
    mk, mgene = mk[o], mol.gene.map(gi).to_numpy()[o]
    t2, tl = mol.clsR2.to_numpy()[o], mol.clsLR.to_numpy()[o]
    d = load_dump(X / (sys.argv[3] if len(sys.argv) > 3 else "labels_dump") / "molecules.tsv.gz", cellcodes, gi)
    j = np.minimum(np.searchsorted(mk, d["key"]), len(mk) - 1)
    hit = (d["key"] >= 0) & (mk[j] == d["key"]) & (mgene[j] == d["gene"])
    d["has_truth"] = hit
    d["t2"] = np.where(hit, t2[j], -1)
    d["tl"] = np.where(hit, tl[j], -1)
    n = len(hit)
    out = {"singlet_molecules_in_truth_cells": int(n), "with_truth": int(hit.sum()), "truth_molecules": int(len(mk)),
           "truth_molecules_matched": int(len(np.unique(j[hit])))}
    names = ["S", "U", "A"]
    for tn, tcol in (("R2", "t2"), ("LR", "tl")):
        conf = {f"singlet_{names[a]}->truth_{names[b]}": int(((d["lab"] == a) & (d[tcol] == b)).sum()) for a in range(3) for b in range(3)}
        conf.update({f"singlet_{names[a]}->no_truth": int(((d["lab"] == a) & ~hit).sum()) for a in range(3)})
        out[f"confusion_{tn}"] = conf
    isU = d["lab"] == U_
    out["U_calls"] = {"total": int(isU.sum()), "with_truth": int((isU & hit).sum()), "share_with_truth": float((isU & hit).sum() / isU.sum())}
    for nm, lab in (("S", S_), ("A", A_)):
        m = d["lab"] == lab
        out[f"{nm}_calls"] = {"total": int(m.sum()), "share_with_truth": float((m & hit).sum() / m.sum())}
    # strata of false U (U call, truth present, Truth-R2 not U)
    fu = isU & hit & (d["t2"] != U_)
    tu = isU & hit
    strata = {}

    def row(m):
        a = int((tu & m).sum())
        return {"U_with_truth": a, "false_U": int((fu & m).sum()), "false_U_rate": float((fu & m).sum() / max(a, 1)),
                "share_of_all_false_U": float((fu & m).sum() / max(fu.sum(), 1)), "U_calls": int((isU & m).sum()),
                "truth_share_of_U_calls": float((tu & m).sum() / max((isU & m).sum(), 1)),
                "false_U_truthLR_is_U": float(((fu & m) & (d["tl"] == U_)).sum() / max((fu & m).sum(), 1))}

    for b, v in BITS.items():
        strata[f"diag_{b}"] = row((d["diag"] & v) > 0)
    strata["diag_none(excl Spliced)"] = row((d["diag"] & ~np.uint8(SPL)) == 0)
    for lo, hi, nm in ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 10**9, "10+")):
        strata[f"reads_{nm}"] = row((d["reads"] >= lo) & (d["reads"] <= hi))
    for lo, hi, nm in ((1, 1, "1"), (2, 4, "2-4"), (5, 10**9, "5+")):
        strata[f"reads_{nm}_Flank_only"] = row((d["reads"] >= lo) & (d["reads"] <= hi) & ((d["diag"] & ~np.uint8(SPL)) == FLANK))
    top = pd.Series(d["diag"][isU & hit]).value_counts().head(12)
    for v in top.index:
        strata[f"diagvalue_{int(v)}"] = row(d["diag"] == v)
    strata["flags_values"] = {str(int(v)): int((isU & (d["flags"] == v)).sum()) for v in np.unique(d["flags"][isU])}
    out["strata"] = strata
    out["false_U_total"] = int(fu.sum())
    out["false_U_truthR2_class"] = {names[c]: int((fu & (d["t2"] == c)).sum()) for c in (S_, A_)}
    out["false_U_truthLR_class"] = {names[c]: int((fu & (d["tl"] == c)).sum()) for c in (S_, U_, A_)}

    # rules, halves
    rs = rules(d)
    half = d["cell"] % 2            # 0 = even (selection), 1 = odd (report)
    ncell, ngene = len(cells), len(genes)
    TT = truth_mats(mol, "clsR2", cells, genes)
    res = {}
    for rid, uU in rs.items():
        res[rid] = {}
        T = mats(d, uU, ncell, ngene)
        for hname, hv in (("even", 0), ("odd", 1)):
            m = hit & (half == hv)
            call, truth2, truthL = uU, d["t2"] == U_, d["tl"] == U_
            e = {"mol_R2": prf(call, truth2, m), "mol_LR": prf(call, truthL, m),
                 "U_calls": int((uU & (half == hv)).sum()), "U_calls_with_truth": int((uU & m).sum())}
            sc = score_tool(T, TT, np.flatnonzero(np.arange(ncell) % 2 == hv), ngene)
            q = sc["q90"]
            e["pairs_q90"] = {k: q[k] for k in ("pairs", "pair_pearson_log1p_U", "gene_pooled_pearson_log1p_U", "per_cell_median_pearson")}
            e["pairs_q90"].update({"U_precision": q["has_unspliced"]["precision"], "U_recall": q["has_unspliced"]["recall"], "U_f1": q["has_unspliced"]["f1"],
                                   "tool_U_share": q["share"]["tool_U"], "truth_U_share": q["share"]["truth_U"]})
            res[rid][hname] = e
        print(f"[rules] {rid} done", file=sys.stderr, flush=True)
    out["rules"] = res
    base = res["C0"]["even"]["mol_R2"]["recall"]
    ok = [r for r in res if res[r]["even"]["mol_R2"]["recall"] >= 0.95 * base]
    best = max(ok, key=lambda r: (round(res[r]["even"]["mol_R2"]["f1"], 6), -int(r[1:])))
    out["selection"] = {"baseline_even_recall": base, "eligible": ok, "chosen": best}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("with_truth", "U_calls", "confusion_R2", "selection")}, indent=1))


if __name__ == "__main__":
    main()
