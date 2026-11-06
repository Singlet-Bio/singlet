#!/usr/bin/env python3
"""S15: score the 13-sample beta pilot. Singlet (0ea48783, CPU and GPU runs) against the reference tools per sample.
Metrics (all computed from kept outputs, no tool tuned on these samples):
  cells     Singlet cell calls vs STARsolo EmptyDrops_CR (Jaccard, precision, recall of Singlet against STARsolo)
  sex       Singlet chromosomal sex call vs the sex stated in GEO (where there is one)
  ambient   Singlet ambient fraction vs SoupX rho and decontX mean rho (same scale, no truth)
  doublets  Singlet doublet rate vs scDblFinder rate
  demux     Singlet donor/line labels vs souporcell and vireo clusters (adjusted Rand index on shared barcodes)
  perf      wall time and peak RSS of Singlet CPU, Singlet GPU and STARsolo
  equiv     CPU vs GPU output differences (equiv_0ea48783.txt)
usage: score_pilot.py PILOT_DIR OUT_PREFIX"""
import csv, json, os, re, sys, math, collections

P, OUT = sys.argv[1], sys.argv[2]
SAMPLES = ["GSM5682562", "GSM8141311", "GSM9038586", "GSM5130155", "GSM5130163", "GSM8183378", "GSM4987144", "GSM8164511", "GSM4987148", "GSM5171610", "GSM7064234", "GSM7884577", "GSM8085472"]
SEX = {"GSM5130155": "female", "GSM5171610": "female", "GSM7884577": "female", "GSM8183378": "female", "GSM9038586": "female", "GSM5130163": "male", "GSM7064234": "male"}
C = "0ea48783"


def rd(path):
    try: return open(path, encoding="utf-8", errors="replace").read()
    except OSError: return None


def tsv(path):
    t = rd(path)
    if t is None: return []
    return list(csv.DictReader(t.splitlines(), delimiter="\t"))


def wall(path):
    t = rd(path) or ""; m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)", t)
    if not m: return None
    s = 0.0
    for x in m.group(1).split(":"): s = s * 60 + float(x)
    return s


def rss(path):
    t = rd(path) or ""; m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    return int(m.group(1)) / 1e6 if m else None


def barcodes(path):
    t = rd(path)
    if t is None: return None
    return {re.sub(r"-" + chr(92) + "d+$", "", l.split(chr(9))[0]) for l in t.splitlines() if l}


def ari(a, b):
    n = len(a);
    if n < 2: return None
    tab = collections.Counter(zip(a, b)); ra = collections.Counter(a); rb = collections.Counter(b)
    c2 = lambda x: x * (x - 1) / 2
    sij = sum(c2(v) for v in tab.values()); sa = sum(c2(v) for v in ra.values()); sb = sum(c2(v) for v in rb.values())
    exp = sa * sb / c2(n); mx = (sa + sb) / 2
    return (sij - exp) / (mx - exp) if mx != exp else 1.0


rows = []
for g in SAMPLES:
    d = P + "/" + g; r = {"sample": g}
    O = d + "/s1cpu_" + C + "/out"
    OM = d + "/s1cap_" + C + "/out"; r["matched_prefix"] = os.path.isdir(OM); OM = OM if r["matched_prefix"] else O   # samples above 100 M pairs: the Singlet run on the same 100 M prefix STARsolo saw
    cc = tsv(OM + "/cell_calls.tsv")
    s_cells = {x["barcode"] for x in cc if x.get("is_cell") == "TRUE"} if cc else None
    star = barcodes(d + "/starsolo_bam/Solo.out/GeneFull/filtered/barcodes.tsv") or barcodes(d + "/starsolo/Solo.out/GeneFull/filtered/barcodes.tsv")
    r["singlet_cells"] = len(s_cells) if s_cells is not None else None
    r["starsolo_cells"] = len(star) if star else None
    if s_cells and star:
        i = len(s_cells & star); r["cells_jaccard"] = round(i / len(s_cells | star), 3); r["cells_precision"] = round(i / len(s_cells), 3); r["cells_recall"] = round(i / len(star), 3)
    ed = (rd(d + "/tools/edrops/summary.tsv") or ""); m = re.search(r"cells_fdr0.001\s+(\d+)", ed); r["edrops_cells"] = int(m.group(1)) if m else None
    try: sx = json.load(open(O + "/sex_call.json")); r["singlet_sex"] = sx.get("sex"); r["sex_conf"] = sx.get("confidence")
    except Exception: r["singlet_sex"] = None
    r["geo_sex"] = SEX.get(g)
    r["sex_match"] = (r["singlet_sex"] == r["geo_sex"]) if r["geo_sex"] and r["singlet_sex"] else None
    try: am = json.load(open(O + "/ambient_contamination.json")); r["singlet_ambient"] = am.get("estimated_contamination_fraction"); r["singlet_mean_rho"] = am.get("mean_rho_cells")
    except Exception: r["singlet_ambient"] = None
    rt = rd(d + "/tools/rtools/summary.tsv") or ""
    for k, key in (("soupx_rho", "soupx_rho"), ("decontx_mean_rho", "decontx_rho"), ("scdblfinder_doublets", "scdbl_n"), ("n_cells", "rtools_cells")):
        m = re.search(k + r"\s+([0-9.eE+-]+)", rt); r[key] = float(m.group(1)) if m else None
    ds = tsv(O + "/doublet_scores.tsv")
    if ds: r["singlet_doublet_rate"] = round(sum(1 for x in ds if x["is_doublet"] == "True") / len(ds), 4)
    if r.get("scdbl_n") and r.get("rtools_cells"): r["scdbl_rate"] = round(r["scdbl_n"] / r["rtools_cells"], 4)
    # demux against souporcell / vireo
    da = {x["cell"]: x["donor_id"] for x in tsv(OM + "/donor_assignments.tsv") if x.get("cell_status") == "cell" and x["donor_id"] not in ("doublet", "unassigned", "NA")}
    sc = {re.sub(r"-\d+$", "", x["barcode"]): x["assignment"] for x in tsv(d + "/tools/souporcell/out/clusters.tsv") if x.get("status") == "singlet"}
    vi = {re.sub(r"-\d+$", "", x["cell"]): x["donor_id"] for x in tsv(d + "/tools/vireo/vireo_out/donor_ids.tsv") if x["donor_id"] not in ("doublet", "unassigned")}
    for name, ref in (("souporcell", sc), ("vireo", vi)):
        sh = sorted(set(da) & set(ref))
        if len(sh) > 50: r["ari_singlet_vs_" + name] = round(ari([da[b] for b in sh], [ref[b] for b in sh]), 3); r["n_shared_" + name] = len(sh)
    if sc and vi:
        sh = sorted(set(sc) & set(vi))
        if len(sh) > 50: r["ari_souporcell_vs_vireo"] = round(ari([sc[b] for b in sh], [vi[b] for b in sh]), 3)
    r["donors_singlet"] = len(set(da.values())) if da else None
    # performance
    for tag, p in (("singlet_cpu", d + "/s1cpu_" + C + "/time.txt"), ("singlet_gpu", d + "/s1gpu_" + C + "/time.txt"), ("starsolo", d + "/starsolo_bam/time.txt")):
        r[tag + "_wall_s"] = wall(p); r[tag + "_rss_gb"] = rss(p)
    eq = rd(d + "/equiv_" + C + ".txt") or ""
    r["equiv_diff_files"] = len(re.findall(r"^\s*DIFF ", eq, re.M)) if eq else None
    r["equiv_identical"] = ("IDENTICAL" in eq.upper() and "DIFF " not in eq) if eq else None
    rows.append(r)

keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open(OUT + ".tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
json.dump(rows, open(OUT + ".json", "w"), indent=1)
print(len(rows), "samples ->", OUT + ".tsv")
