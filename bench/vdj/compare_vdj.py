# SPDX-License-Identifier: MIT
"""Compare V(D)J calls of several methods with a reference (Cell Ranger's filtered contigs, 10x's or the authors').

    python -P compare_vdj.py --reference filtered_contig_annotations.csv --singlet OUT/contigs.csv
        [--trust4 TRUST_x_barcode_report.tsv] [--gex-labels labels.csv] --out metrics.json

Every method becomes a table of productive contigs (barcode, chain, cdr3_nt, cdr3, v, j, c, umis). Metrics, on the
reference's cells (Cell Ranger ``is_cell``):

* ``contig_recovery``: reference productive contigs (barcode, chain, junction nt) the method reports exactly;
* ``cdr3_nt_accuracy`` / ``cdr3_aa_accuracy``: for reference (cell, chain)s where the method has a productive contig,
  the method's top contig has the reference's top junction;
* ``v_gene_agreement`` / ``j_gene_agreement``: on exact junction matches, the same gene (alleles and ``/`` dropped);
* ``extra_contigs``: method contigs in reference cells that the reference lacks, per reference contig;
* ``pairing_rate``: reference cells with a productive pair (TRA+TRB or IGH+IGK/L) by the method; the reference's own;
* ``clonotype_pair_precision`` / ``recall``: over pairs of cells both methods assign, the share of same-clonotype pairs
  of one that are same-clonotype in the other; ``clonotype_ari``;
* ``cells_jaccard``: method cells (>= 1 productive contig) vs reference cells.

With ``--gex-labels`` (barcode, lineage in T/B/other from GEX markers) also ``wrong_lineage``: TCR contigs in B cells
and BCR contigs in T cells, a truth that does not depend on any V(D)J caller.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict

import numpy as np
import pandas as pd

TCR = {"TRA", "TRB", "TRG", "TRD"}
BCR = {"IGH", "IGK", "IGL"}


def norm_gene(g) -> str:
    if not isinstance(g, str) or g in ("None", "nan"):
        return ""
    g = g.split("*")[0].replace("/", "").upper()
    return g


def _truthy(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "t", "yes")


def load_cellranger(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "is_cell" in d:
        d = d[d["is_cell"].map(_truthy)]
    keep = d["productive"].map(_truthy)
    if "high_confidence" in d:
        keep &= d["high_confidence"].map(_truthy)
    d = d[keep & d["chain"].isin(TCR | BCR)]
    return pd.DataFrame({"barcode": d["barcode"].astype(str), "chain": d["chain"], "cdr3_nt": d["cdr3_nt"].astype(str),
                         "cdr3": d["cdr3"].astype(str), "v": d["v_gene"].map(norm_gene), "j": d["j_gene"].map(norm_gene),
                         "c": d["c_gene"].map(norm_gene), "umis": pd.to_numeric(d["umis"], errors="coerce").fillna(0),
                         "clonotype": d.get("raw_clonotype_id", pd.Series(index=d.index, dtype=str)).astype(str)})


def cellranger_cells(path: str) -> set:
    d = pd.read_csv(path)
    if "is_cell" in d:
        d = d[d["is_cell"].map(_truthy)]
    return set(d["barcode"].astype(str))


def load_singlet(path: str, cells_path: str = None) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["kept"].map(_truthy) & d["productive"].map(_truthy)]
    out = pd.DataFrame({"barcode": d["barcode"].astype(str), "chain": d["chain"], "cdr3_nt": d["cdr3_nt"], "cdr3": d["cdr3"],
                        "v": d["v_gene"].map(norm_gene), "j": d["j_gene"].map(norm_gene), "c": d["c_gene"].map(norm_gene),
                        "umis": d["umis"]})
    if cells_path:
        c = pd.read_csv(cells_path)[["barcode", "clonotype_id"]]
        out = out.merge(c.rename(columns={"clonotype_id": "clonotype"}), on="barcode", how="left")
    return out


def load_trust4(path: str, suffix: str = "-1") -> pd.DataFrame:
    """TRUST4 ``*_barcode_report.tsv``: chain fields are ``V,D,J,C,cdr3_nt,cdr3_aa,count,consensus_id,...``."""
    d = pd.read_csv(path, sep="\t")
    d.columns = [c.lstrip("#") for c in d.columns]
    rows = []
    for r in d.itertuples(index=False):
        bc = str(r.barcode)
        bc = bc if bc.endswith(suffix) else bc + suffix
        for col in ("chain1", "chain2", "secondary_chain1", "secondary_chain2"):
            f = getattr(r, col, "*")
            if not isinstance(f, str) or f in ("*", ""):
                continue
            p = f.split(",")
            if len(p) < 7:
                continue
            v, dg, j, c, nt, aa, cnt = p[:7]
            chain = (v[:3] if v not in ("*", "") else j[:3]).upper()
            if chain not in TCR | BCR or "_" in aa or "?" in aa or "*" in aa or aa in ("out_of_frame", "partial"):
                continue
            # TRUST4 cdr3 follows IMGT (Cys ... Phe/Trp) like Cell Ranger
            rows.append({"barcode": bc, "chain": chain, "cdr3_nt": nt, "cdr3": aa, "v": norm_gene(v), "j": norm_gene(j),
                         "c": norm_gene(c), "umis": float(cnt)})
    out = pd.DataFrame(rows, columns=["barcode", "chain", "cdr3_nt", "cdr3", "v", "j", "c", "umis"])
    return out


def _clonotypes_from_contigs(t: pd.DataFrame) -> pd.Series:
    """Exact clonotypes: the sorted set of (chain, junction nt) per barcode (for methods without their own)."""
    k = t.sort_values(["barcode", "chain", "cdr3_nt"]).groupby("barcode").apply(
        lambda g: "|".join(g["chain"] + ":" + g["cdr3_nt"]))
    return k


def _pairs_same(labels_a, labels_b):
    """Pairwise co-membership precision/recall of b against a and the adjusted Rand index."""
    from scipy.special import comb
    from sklearn.metrics import adjusted_rand_score

    ct = pd.crosstab(labels_a, labels_b).to_numpy()
    both = comb(ct, 2).sum()
    a = comb(ct.sum(axis=1), 2).sum()
    b = comb(ct.sum(axis=0), 2).sum()
    return {"clonotype_pair_precision": float(both / b) if b else float("nan"),
            "clonotype_pair_recall": float(both / a) if a else float("nan"),
            "clonotype_ari": float(adjusted_rand_score(labels_a, labels_b))}


def compare(ref: pd.DataFrame, ref_cells: set, m: pd.DataFrame, gex: pd.DataFrame = None) -> Dict:
    out = {}
    m_in = m[m["barcode"].isin(ref_cells)]
    refk = set(zip(ref.barcode, ref.chain, ref.cdr3_nt))
    mk = set(zip(m_in.barcode, m_in.chain, m_in.cdr3_nt))
    out["reference_cells"] = len(ref_cells)
    out["reference_contigs"] = len(refk)
    out["contig_recovery"] = len(refk & mk) / max(1, len(refk))
    out["extra_contigs"] = len(mk - refk) / max(1, len(refk))
    rtop = ref.sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
    mtop = m_in.sort_values("umis", ascending=False).drop_duplicates(["barcode", "chain"])
    j = rtop.merge(mtop, on=["barcode", "chain"], suffixes=("_r", "_m"))
    out["cell_chains_compared"] = len(j)
    out["cell_chain_recovery"] = len(j) / max(1, len(rtop))
    out["cdr3_nt_accuracy"] = float((j.cdr3_nt_r == j.cdr3_nt_m).mean()) if len(j) else float("nan")
    out["cdr3_aa_accuracy"] = float((j.cdr3_r == j.cdr3_m).mean()) if len(j) else float("nan")
    ex = j[j.cdr3_nt_r == j.cdr3_nt_m]
    out["v_gene_agreement"] = float((ex.v_r == ex.v_m).mean()) if len(ex) else float("nan")
    out["j_gene_agreement"] = float((ex.j_r == ex.j_m).mean()) if len(ex) else float("nan")
    cg = ex[(ex.c_r != "") & (ex.c_m != "")]
    out["c_gene_agreement"] = float((cg.c_r == cg.c_m).mean()) if len(cg) else float("nan")
    for name, t in (("method", m_in), ("reference", ref)):
        ch = t.groupby("barcode")["chain"].agg(set)
        paired = ch.map(lambda s: bool(({"TRA", "TRB"} <= s) or ({"TRG", "TRD"} <= s) or ("IGH" in s and s & {"IGK", "IGL"})))
        out[f"pairing_rate_{name}"] = float(paired.sum() / max(1, len(ref_cells)))
    mcells = set(m.barcode)
    out["method_cells"] = len(mcells)
    out["cells_jaccard"] = len(mcells & ref_cells) / max(1, len(mcells | ref_cells))
    # clonotypes: exact chain sets for both (so the comparison does not depend on either tool's grouping rule)
    ca = _clonotypes_from_contigs(ref)
    cb = _clonotypes_from_contigs(m_in)
    common = ca.index.intersection(cb.index)
    out["clonotype_cells_compared"] = int(len(common))
    out["clonotype_exact_same_chains"] = float((ca[common] == cb[common]).mean()) if len(common) else float("nan")
    if len(common) > 1:
        out.update(_pairs_same(ca[common].to_numpy(), cb[common].to_numpy()))
    if gex is not None:
        lab = dict(zip(gex.barcode, gex.lineage))
        for name, t in (("method", m), ("reference", ref)):
            ll = t.barcode.map(lab)
            tc = t.chain.isin(TCR)
            bc = t.chain.isin(BCR)
            known = ll.isin(["T", "B"])
            wrong = (tc & (ll == "B")) | (bc & (ll == "T"))
            out[f"wrong_lineage_{name}"] = float(wrong.sum() / max(1, known.sum()))
            out[f"contigs_lineage_known_{name}"] = int(known.sum())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--singlet")
    ap.add_argument("--singlet-cells")
    ap.add_argument("--trust4")
    ap.add_argument("--gex-labels")
    ap.add_argument("--chains", default=None, help="TCR or BCR: restrict every table to these chains")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = load_cellranger(a.reference)
    cells = cellranger_cells(a.reference)
    gex = pd.read_csv(a.gex_labels) if a.gex_labels else None
    keep = TCR if a.chains == "TCR" else BCR if a.chains == "BCR" else TCR | BCR
    ref = ref[ref.chain.isin(keep)]
    res = {"reference": a.reference, "chains": a.chains or "all"}
    for name, path, loader in (("singlet", a.singlet, lambda p: load_singlet(p, a.singlet_cells)),
                               ("trust4", a.trust4, load_trust4)):
        if path:
            t = loader(path)
            t = t[t.chain.isin(keep)]
            res[name] = compare(ref, cells, t, gex)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
