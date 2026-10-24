#!/usr/bin/env python
"""w4_mtx: summarise results/w4_mtx_t4_*.json into results/w4_mtx_t4_summary.tsv (conditional coding of raw/matrix.mtx given the .1pz exon matrix)."""
import csv, glob, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); RES = os.path.join(BC, "results")
ZIP = {r["gsm"]: int(r["csize"]) for r in csv.DictReader(open(os.path.join(RES, "w4_atlas_members.tsv")), delimiter="\t") if r["path"].endswith("Gene/raw/matrix.mtx")}


def lc2(n, k):   # log2 C(n,k) in bytes
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2) / 8


rows = []
for f in sorted(glob.glob(os.path.join(RES, "w4_mtx_t4_GSE*.json"))):
    d = json.load(open(f)); gsm = d["sample"].split(".")[1]; zipb = ZIP.get(gsm)
    NB = d["NB"]
    for tag in ("exon", "exon+intron"):
        v = d["vs_" + tag]; c = v["cond_cost_on_pz_support"]
        cond_support = c["heldout_bits_per_entry"] * c["support_entries"] / 8
        extra_cols = d["mtx_nonzero_columns"] - d["mtx_cols_in_pz"]
        colids = lc2(NB, extra_cols)
        mo = v["mtx_only_standalone"]
        mo_o0 = mo["total_o0"]; mo_xz = mo.get("total_xz")
        total_o0 = cond_support + mo_o0 + colids
        total_xz = (cond_support + mo_xz + colids) if mo_xz else None
        sa = d["standalone_mtx_csc"]; sa_cols = lc2(NB, d["mtx_nonzero_columns"])
        sa_o0 = sa["total_o0"] + sa_cols; sa_xz = (sa["total_xz"] + sa_cols) if sa.get("total_xz") else None
        rows.append(dict(sample=d["sample"], ctx=tag, zip_deflate=zipb, mtx_nnz=d["mtx_nnz"], pz_cols=d["pz_columns_union"], mtx_cols=d["mtx_nonzero_columns"], extra_cols=extra_cols,
                         frac_mtx_nnz_in_pz=round(v["frac_mtx_nnz_also_in_pz"], 4), frac_equal_both=round(v["frac_equal_among_both"] or 0, 4),
                         both=v["both"], mtx_only=v["mtx_only"], pz_only=v["pz_only"],
                         cond_bits_per_support_heldout=round(c["heldout_bits_per_entry"], 4), cond_bits_per_mtx_nonzero_plugin=round(c["plugin_bits_per_mtx_nonzero"], 4),
                         B_support=round(cond_support), B_mtx_only_o0=round(mo_o0), B_mtx_only_xz=round(mo_xz) if mo_xz else "", B_colids=round(colids),
                         B_cond_total_o0=round(total_o0), B_cond_total_xz=round(total_xz) if total_xz else "",
                         B_standalone_o0=round(sa_o0), B_standalone_xz=round(sa_xz) if sa_xz else "",
                         cond_over_zip_o0=round(total_o0 / zipb, 4) if zipb else "", standalone_over_zip_o0=round(sa_o0 / zipb, 4) if zipb else ""))
with open(os.path.join(RES, "w4_mtx_t4_summary.tsv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
