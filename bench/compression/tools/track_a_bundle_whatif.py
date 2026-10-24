#!/usr/bin/env python
"""Track A: what would the model-side treatments save on the whole GSE145370.singlet bundle?

Inputs: results/track_a_account_bundle_1pz.tsv (stored component sizes), results/track_a_bundle_whatif.tsv (cost of the
replacement representations measured by harness/track_a_bundle.cpp), the bundle itself (junction vocabulary).

    python -P tools/track_a_bundle_whatif.py SCRATCH_DIR
"""
import csv, lzma, math, os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
from singlet._pz import read_1pz

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
scratch = sys.argv[1]

acc = {(r["sample"], r["member"]): r for r in csv.DictReader(open("results/track_a_account_bundle_1pz.tsv"), delimiter="\t")}
wf = {}
for r in csv.DictReader(open("results/track_a_bundle_whatif.tsv"), delimiter="\t"):
    s, m = r["file"].split("__")
    wf[(s, m)] = r
I = lambda x: int(x)

total_files = sum(I(a["bytes"]) for a in acc.values())
print("matrix members in bundle: %d files, %s bytes" % (len(acc), "{:,}".format(total_files)))

# ---- junction vocabulary (sj / psi share names inside a sample) ----
z = zipfile.ZipFile("corpus/bundles/GSE145370.singlet")
names = {}
for i in z.infolist():
    if i.filename.endswith("/sj_counts.1pz"):
        g = i.filename.split("/")[1]
        p = os.path.join(scratch, "tmp.1pz")
        open(p, "wb").write(z.read(i))
        names[g] = read_1pz(p)["rownames"]
union = sorted(set().union(*[set(v) for v in names.values()]))
U = len(union)
union_raw = "\0".join(union).encode()
union_xz = len(lzma.compress(union_raw, preset=9 | lzma.PRESET_EXTREME))
# per sample: subset of the union, cost = log2 C(U, n) bits (order is by the sample's own file order, which is
# the annotation order: assume it follows the union's order within the sample; ordering cost reported separately)
subset_bits = sum(math.lgamma(U + 1) / math.log(2) - math.lgamma(len(v) + 1) / math.log(2) - math.lgamma(U - len(v) + 1) / math.log(2) for v in names.values())
print("junction union %d names, raw %d, xz %d; subset coding of the 14 samples: %d bytes" % (U, len(union_raw), union_xz, subset_bits / 8))

kinds = ["exon_counts.1pz", "intron_counts.1pz", "sj_counts.1pz", "splice_psi.1pz", "vdj_gene_usage.1pz"]
tot = {k: 0 for k in ["today", "rown", "coln", "perm", "colsums", "ptr", "chunks", "other"]}
rows = []
for kind in kinds:
    a = [v for (s, m), v in acc.items() if m == kind]
    t = {k: sum(I(x[f]) for x in a) for k, f in [("bytes", "bytes"), ("perm", "perm_z"), ("ptr", "ptr_z"), ("colsums", "colsums_z"), ("chunks", "chunk_payload"),
                                                   ("meta", "meta_z"), ("rown", "rown_z3"), ("coln", "coln_z3")]}
    rc = sum(I(wf[(x["sample"], kind)]["rowcounts_best"]) for x in a)
    rows.append((kind, t, rc))
print()
print("| member | today | rownames | colnames | perm | colsums+ptr | chunks | other | -> row counts instead of perm |")
print("|---|---|---|---|---|---|---|---|---|")
for kind, t, rc in rows:
    other = t["bytes"] - t["rown"] - t["coln"] - t["perm"] - t["colsums"] - t["ptr"] - t["chunks"]
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (kind, "{:,}".format(t["bytes"]), "{:,}".format(t["rown"]), "{:,}".format(t["coln"]), "{:,}".format(t["perm"]),
                                                              "{:,}".format(t["colsums"] + t["ptr"]), "{:,}".format(t["chunks"]), "{:,}".format(other), "{:,}".format(rc)))

# ---- scenario ----
today = total_files
S = {}
exon = [r for r in rows if r[0] == "exon_counts.1pz"][0]
S["a_names_as_model (exon+intron row names -> vocabulary; all barcode lists -> whitelist index, still stored per matrix)"] = None
rown_ei = sum(r[1]["rown"] for r in rows if r[0] in kinds[:2])
wl_each = {k: sum(I(wf[(x["sample"], k)]["whitelist_idx_bytes"]) for x in [v for (s, m), v in acc.items() if m == k]) for k in kinds}
coln_all = sum(r[1]["coln"] for r in rows)
perm_all = sum(r[1]["perm"] for r in rows)
rc_all = sum(r[2] for r in rows)
cs_ptr = sum(r[1]["colsums"] + r[1]["ptr"] for r in rows)
psi_rown = [r for r in rows if r[0] == "splice_psi.1pz"][0][1]["rown"]
sj_rown = [r for r in rows if r[0] == "sj_counts.1pz"][0][1]["rown"]
print()
steps = [
    ("today (sum of the 70 matrix members)", today, 0),
    ("1. exon+intron row names -> vocabulary id (model: 3.57 MB human exon+intron lists, once)", -rown_ei + 8 * 28, 3569124),
    ("2. splice_psi row names -> 'same as sj_counts' (verified identical)", -psi_rown, 0),
    ("3. sj row names -> study junction vocabulary + per-sample subset", -sj_rown + int(subset_bits / 8), union_xz),
    ("4. barcodes: store once per sample (5 members share them), as whitelist index", -coln_all + sum(I(wf[(s, 'exon_counts.1pz')]['whitelist_idx_bytes']) for s in {s for s, m in acc}), 3283),
    ("5. permutation -> per-row nnz counts (decoder re-derives it)", -perm_all + rc_all, 0),
    ("6. drop colsums and ptr blocks (derivable)", -cs_ptr, 0),
]
cum = 0
print("| step | change in bundle bytes | cumulative bundle (matrix members) | model bytes needed |")
print("|---|---|---|---|")
for name, delta, model in steps:
    cum += delta
    print("| %s | %s | %s | %s |" % (name, "{:+,}".format(delta) if delta else "", "{:,}".format(cum), "{:,}".format(model) if model else ""))
print("\nremaining = %.1f%% of today (%.2fx smaller), of which count chunks %s (%.1f%% of remaining)" % (100.0 * cum / today, today / cum, "{:,}".format(sum(r[1]["chunks"] for r in rows)),
                                                                                             100.0 * sum(r[1]["chunks"] for r in rows) / cum))
