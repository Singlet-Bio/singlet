"""E4: non-overlapping derivable-bytes accounting per sample of the bundle, from the audit and Track A tables.

Reads results/track_a_account_bundle_1pz.tsv (component sizes of every .1pz in the bundle), results/e4_members.tsv
(bytes of every member type and, for cell_calls.tsv, the independent flag information) and results/e4_recompute.tsv.
Writes results/e4_derivable.tsv: per sample, bytes in each category, and the share of the sample's bytes.

Categories are disjoint (a byte is counted once): the whole splice_psi member; rownames of exon/intron/vdj (reference
vocabulary); the 4 other copies of the barcode list (kept: exon's); permutation minus the per-row counts that regenerate it;
ptr and colsums blocks of the four remaining matrices; cell_calls.tsv minus the called-set information; the mt stub.
"""
import csv
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as L  # noqa: E402
import numpy as np  # noqa: E402
from e4_audit import rowcounts_cost  # noqa: E402


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    acc = rd(os.path.join(L.RESULTS, "track_a_account_bundle_1pz.tsv"))
    mem = {r["member"]: r for r in rd(os.path.join(L.RESULTS, "e4_members.tsv"))}
    z = zipfile.ZipFile(L.BUNDLE)
    sizes = {}
    for zi in z.infolist():
        if zi.filename.startswith("samples/"):
            g, fn = zi.filename.split("/")[1:3]
            sizes.setdefault(g, {})[fn] = zi.compress_size
    flag = {}
    for r in rd(os.path.join(L.RESULTS, "e4_recompute.tsv")):
        if r["test"] == "cell_calls_flag_info" and r["sample"].startswith("GSM"):
            note = r["note"]
            flag[r["sample"]] = int(note.split("flag|depth=")[1].split("B")[0])
    pz = L.load_pz()
    rc = {}  # (sample, member) -> bytes of the per-row nnz counts that regenerate the permutation
    for g in sorted({r["sample"] for r in acc}):
        for m in ("exon_counts", "intron_counts", "sj_counts", "vdj_gene_usage"):
            p = os.path.join(L.SCRATCH, g, m + ".1pz")
            if not os.path.exists(p):  # extracted by e4_audit.py
                with open(p, "wb") as fh:
                    fh.write(z.read(f"samples/{g}/{m}.1pz"))
            M = pz.read_1pz(p)
            rc[(g, m + ".1pz")] = rowcounts_cost(np.bincount(M["indices"], minlength=M["m"]))
    by = {}
    for r in acc:
        by.setdefault(r["sample"], {})[r["member"]] = r
    out = []
    for g in sorted(by):
        S = by[g]
        total = sum(sizes[g].values())
        psi = int(S["splice_psi.1pz"]["bytes"])
        cat = {}
        cat["splice_psi member (whole)"] = psi
        cat["rownames exon+intron+vdj (reference vocabulary)"] = sum(int(S[m]["rown_z3"]) for m in ("exon_counts.1pz", "intron_counts.1pz", "vdj_gene_usage.1pz"))
        keep = ["exon_counts.1pz", "intron_counts.1pz", "sj_counts.1pz", "vdj_gene_usage.1pz"]
        cat["barcode lists, 3 redundant copies of 4"] = sum(int(S[m]["coln_z3"]) for m in keep[1:])
        cat["permutation blocks minus per-row counts (4 matrices)"] = sum(int(S[m]["perm_z"]) - rc[(g, m)] for m in keep)
        cat["ptr blocks (4 matrices)"] = sum(int(S[m]["ptr_z"]) for m in keep)
        cat["colsums blocks (4 matrices)"] = sum(int(S[m]["colsums_z"]) for m in keep)
        cat["cell_calls.tsv minus called-set information"] = sizes[g]["cell_calls.tsv"] - flag.get(g, 0)
        cat["mt_heteroplasmy stub"] = sizes[g]["mt_heteroplasmy.1pz"]
        row = {"sample": g, "sample_bytes": total}
        row.update({k: v for k, v in cat.items()})
        row["derivable_total"] = sum(cat.values())
        out.append(row)
    cols = ["sample", "sample_bytes"] + list(out[0].keys())[2:]
    with open(os.path.join(L.RESULTS, "e4_derivable.tsv"), "w", newline="\n") as f:
        f.write("\t".join(cols) + "\n")
        for r in out:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    n = len(out)
    print("per-sample means over", n, "samples (bytes):")
    tot_mean = sum(r["sample_bytes"] for r in out) / n
    print(f"  sample total {tot_mean:,.0f}")
    for c in cols[2:]:
        mu = sum(r[c] for r in out) / n
        print(f"  {c:55s} {mu:12,.0f}  {100 * mu / tot_mean:5.1f}%")


if __name__ == "__main__":
    main()
