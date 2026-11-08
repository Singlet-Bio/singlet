#!/usr/bin/env python3
"""S15: why Singlet's cell calls differ from STARsolo on GSM7884577 and GSM4987148, and how the demux tools compare on GSM4987148.
usage: diag_pilot.py PILOT_DIR"""
import csv, re, sys, statistics, collections, os, glob
P = sys.argv[1]; C = "0ea48783"


def tsv(path):
    try: return list(csv.DictReader(open(path, encoding="utf-8", errors="replace"), delimiter="\t"))
    except OSError: return []


def med(x): return statistics.median(x) if x else None


def q(x, f):
    x = sorted(x); return x[int(f * (len(x) - 1))] if x else None


for g in ("GSM7884577", "GSM4987148"):
    d = P + "/" + g; O = d + "/s1cap_" + C + "/out"; OF = d + "/s1cpu_" + C + "/out"
    print("==", g)
    for tag, o in (("matched 100 M prefix (s1cap)", O), ("full run (s1cpu)", OF)):
        cc = tsv(o + "/cell_calls.tsv")
        if not cc: continue
        cells = [x for x in cc if x["is_cell"] == "TRUE"]
        print(" ", tag, "barcodes listed", len(cc), "cells", len(cells), "methods", dict(collections.Counter(x["call_method"] for x in cells)))
    cc = tsv(O + "/cell_calls.tsv"); s = {x["barcode"]: x for x in cc if x["is_cell"] == "TRUE"}
    star = {re.sub(r"-\d+$", "", l.strip().split("\t")[0]) for l in open(d + "/starsolo_bam/Solo.out/GeneFull/filtered/barcodes.tsv") if l.strip()}
    both = set(s) & star; only_s = set(s) - star; only_t = star - set(s)
    print("  both", len(both), "singlet only", len(only_s), "starsolo only", len(only_t))
    for name, grp in (("both", both), ("singlet only", only_s)):
        u = [float(s[b]["umi_total"]) for b in grp if s[b]["umi_total"] not in ("", "NA")]
        print("  %-13s UMI/cell median %s  p10 %s  p90 %s" % (name, med(u), q(u, .1), q(u, .9)),
              " flags knee/v2/lowrna:", [sum(1 for b in grp if s[b].get(k) == "1") for k in ("called_knee", "called_v2", "called_lowrna")], " methods", dict(collections.Counter(s[b]["call_method"] for b in grp)))
    allc = {x["barcode"]: x for x in cc}
    u = [float(allc[b]["umi_total"]) for b in only_t if b in allc and allc[b]["umi_total"] not in ("", "NA")]
    print("  starsolo only: in Singlet table %d of %d, UMI/cell median %s" % (sum(1 for b in only_t if b in allc), len(only_t), med(u)),
          " is_cell values", dict(collections.Counter(allc[b]["is_cell"] for b in only_t if b in allc)))
    try:
        us = [int(l.split()[0]) for l in open(d + "/starsolo_bam/Solo.out/GeneFull/UMIperCellSorted.txt")][:200000]
        print("  STARsolo UMI knee: rank 1000 %s, 5000 %s, 10000 %s, 20000 %s, 50000 %s, last called cell %s" % tuple([us[k] if k < len(us) else None for k in (999, 4999, 9999, 19999, 49999)] + [us[len(star) - 1] if len(star) <= len(us) else None]))
    except OSError: pass
    cbf = d + "/tools/cellbender/cb_cell_barcodes.csv"
    if os.path.exists(cbf):
        cbs = {re.sub(r"-\d+$", "", l.strip().split(",")[0]) for l in open(cbf) if l.strip()}
        print("  CellBender cells %d: in STARsolo %d, in Singlet %d, in both %d" % (len(cbs), len(cbs & star), len(cbs & set(s)), len(cbs & star & set(s))))
    ed = open(d + "/tools/edrops/summary.tsv").read().split() if os.path.exists(d + "/tools/edrops/summary.tsv") else []
    print("  edrops summary", ed[:6])
    cb = glob.glob(d + "/tools/cellbender/*cell_barcodes*") + glob.glob(d + "/tools/cellbender/out/*cell_barcodes*")
    print("  cellbender cell barcode files", cb[:2], " (run.log tail:", (open(d + "/tools/cellbender/run.log", errors="replace").read()[-150:].replace("\n", " ") if os.path.exists(d + "/tools/cellbender/run.log") else "no log"), ")")

# demux on the line mixture
g = "GSM4987148"; d = P + "/" + g; O = d + "/s1cap_" + C + "/out"
da = {x["cell"]: x["donor_id"] for x in tsv(O + "/donor_assignments.tsv") if x.get("cell_status") == "cell"}
sc = {re.sub(r"-\d+$", "", x["barcode"]): (x["status"], x["assignment"]) for x in tsv(d + "/tools/souporcell/out/clusters.tsv")}
vi = {re.sub(r"-\d+$", "", x["cell"]): x["donor_id"] for x in tsv(d + "/tools/vireo/vireo_out/donor_ids.tsv")}
print("== demux", g, "Singlet labelled cells", len(da), "Singlet donors", collections.Counter(da.values()).most_common(8))
print(" souporcell status", dict(collections.Counter(v[0] for v in sc.values())), " clusters", dict(collections.Counter(v[1] for v in sc.values() if v[0] == "singlet")))
print(" vireo donors", dict(collections.Counter(vi.values())))
sh = [b for b in da if b in sc and sc[b][0] == "singlet" and b in vi]
print(" shared cells", len(sh))
for a, bn, A, B in (("souporcell", "vireo", {b: sc[b][1] for b in sh}, {b: vi[b] for b in sh}), ("Singlet", "souporcell", {b: da[b] for b in sh}, {b: sc[b][1] for b in sh}), ("Singlet", "vireo", {b: da[b] for b in sh}, {b: vi[b] for b in sh})):
    tab = collections.Counter((A[b], B[b]) for b in sh)
    print(" contingency %s (rows) x %s (cols):" % (a, bn))
    rows = sorted({k[0] for k in tab}); cols = sorted({k[1] for k in tab})
    print("   cols", cols)
    for r in rows: print("   ", r, [tab.get((r, c), 0) for c in cols])
