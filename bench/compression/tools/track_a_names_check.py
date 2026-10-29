#!/usr/bin/env python
"""Track A, deliverable 2 (names): verify that feature names are a function of the reference build and
look at how barcodes (colnames) are ordered.

    python -P tools/track_a_names_check.py SCRATCH_DIR

SCRATCH_DIR is a writable directory outside the repo; bundle members are extracted there so the compiled
reader (which takes paths) can open them.
"""
import hashlib, json, os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))
import numpy as np
from singlet._pz import read_1pz

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
scratch = sys.argv[1]
os.makedirs(scratch, exist_ok=True)


def h(lst):
    m = hashlib.sha256()
    for s in lst:
        m.update(s.encode()); m.update(b"\0")
    return m.hexdigest()[:12]


corp = ["pbmc_v2_typical", "pbmc_v2_shallow", "pbmc_v3_lowgenes", "brain_sn_v3_deep", "brain_sn_v3_intronic",
        "tcell_5p", "tumor_3p"]
print("== corpus rownames by file")
rows = {}
for s in corp:
    build = json.load(open("corpus/%s/summary.json" % s)).get("reference_build")
    for mem in ("exon_counts.1pz", "intron_counts.1pz"):
        r = read_1pz("corpus/%s/%s" % (s, mem))
        rows[(s, mem)] = r
        print("%-22s %-18s build=%-14s m=%d rownames_hash=%s colnames n=%d hash=%s" % (
            s, mem, build, r["m"], h(r["rownames"]), r["n"], h(r["colnames"])))

print("\n== groups with identical rownames")
groups = {}
for (s, mem), r in rows.items():
    groups.setdefault((mem, h(r["rownames"])), []).append(s)
for k, v in groups.items():
    print(k, v)

# order of barcodes within a file
wl_dir = os.path.join("..", "..", "python", "singlet", "data", "whitelists")
wl = {}
for name in ("737K-august-2016", "3M-february-2018"):
    wl[name] = {l.strip(): i for i, l in enumerate(open(os.path.join(wl_dir, name + ".txt")))}
print("\n== barcodes: whitelist membership and ordering")
for s in corp:
    cn = rows[(s, "exon_counts.1pz")]["colnames"]
    for name, d in wl.items():
        idx = np.array([d.get(c, -1) for c in cn])
        frac = (idx >= 0).mean()
        if frac > 0.5:
            inc = (np.diff(idx[idx >= 0]) > 0).mean()
            print("%-22s whitelist=%-18s members=%.4f  frac(consecutive increasing)=%.3f  wl_size=%d" % (s, name, frac, inc, len(d)))
    # same barcodes in same order in exon vs intron?
    ci = rows[(s, "intron_counts.1pz")]["colnames"]
    print("   exon colnames == intron colnames (order too): %s ; same set: %s" % (cn == ci, set(cn) == set(ci)))

print("\n== bundle: extract and compare member rownames / colnames")
z = zipfile.ZipFile("corpus/bundles/GSE145370.singlet")
by = {}
for i in z.infolist():
    if i.filename.endswith(".1pz"):
        parts = i.filename.split("/")
        p = os.path.join(scratch, parts[1] + "__" + parts[2])
        with open(p, "wb") as f:
            f.write(z.read(i))
        r = read_1pz(p)
        by[(parts[1], parts[2])] = (r["m"], r["n"], h(r["rownames"]), h(r["colnames"]), r["colnames"])
        os.remove(p)
types = sorted({k[1] for k in by})
for t in types:
    rn = {(v[0], v[2]) for k, v in by.items() if k[1] == t}
    print("%-22s distinct rowname lists over 14 samples: %d  %s" % (t, len(rn), sorted(rn)[:4]))
print("per-sample: colnames identical across member types (same order)?")
samples = sorted({k[0] for k in by})
for s in samples:
    hs = {t: by[(s, t)][3] for t in types if by[(s, t)][1] > 0}
    print(" ", s, "n=%d" % by[(s, "exon_counts.1pz")][1], "distinct colname lists across %d members: %d" % (len(hs), len(set(hs.values()))))
# bundle: vs corpus
print("bundle exon rownames == corpus tumor_3p:", by[("GSM4317429", "exon_counts.1pz")][2] == h(rows[("tumor_3p", "exon_counts.1pz")]["rownames"]))
feat = json.load(open("corpus/tumor_3p/feature_vocab.json"))
print("feature_vocab.json: build=%s n_genes=%d (gene list, not the interval list)" % (feat["reference_build"], feat["n_genes"]))
