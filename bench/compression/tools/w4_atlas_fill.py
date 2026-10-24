#!/usr/bin/env python
"""w4_atlas: merge the quick recode rows, rerun the lever table, and substitute both into notes/w4_atlas.md."""
import csv, glob, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); RES = os.path.join(ROOT, "results")
rows = {}
hdr = None
for f in sorted(glob.glob(os.path.join(RES, "w4_atlas_recode_quick.tsv.*"))) + [os.path.join(RES, "w4_atlas_recode_quick.tsv")]:
    if not os.path.exists(f): continue
    for r in csv.DictReader(open(f), delimiter="\t"):
        if r.get("deflate6") and int(r["deflate6"]) > 0: rows[r["label"]] = r; hdr = list(r.keys())
if rows:
    with open(os.path.join(RES, "w4_atlas_recode_quick.tsv"), "w") as fo:
        fo.write("\t".join(hdr) + "\n")
        for r in rows.values(): fo.write("\t".join(r[h] for h in hdr) + "\n")
subprocess.run([sys.executable, "-P", os.path.join(HERE, "w4_atlas_levers.py")], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
order = ["star_Solo_out__Gene__raw__matrix_mtx__1", "star_Solo_out__Gene__filtered__matrix_mtx__1", "star_Solo_out__Gene__raw__barcodes_tsv__1", "ambient_profile_tsv__1"]
nice = {"star_Solo_out__Gene__raw__matrix_mtx__1": "raw matrix.mtx", "star_Solo_out__Gene__filtered__matrix_mtx__1": "filtered matrix.mtx",
        "star_Solo_out__Gene__raw__barcodes_tsv__1": "raw barcodes.tsv (whitelist)", "ambient_profile_tsv__1": "ambient_profile.tsv"}
t = ["| member (gen B) | source | slice (of text) | in-bundle deflate ratio, whole member | deflate-6 on slice | zstd-19 | xz-9e |", "|---|---|---|---|---|---|---|"]
for l in order:
    meta = json.load(open(os.path.join(ROOT, "corpus_w4_atlas", "recode", l + ".json")))
    src = f"{meta['gse']}/{meta['gsm']}" + ("" if meta["complete"] else " (first 10.6 MB of deflate stream)")
    if l not in rows:
        t.append(f"| {nice[l]} | {src} | not finished before the stop | | | | |"); continue
    r = rows[l]; d6 = int(r["deflate6"]); sl = int(r["bytes"]); full = int(r["full_bytes"])
    t.append(f"| {nice[l]} | {src} | {sl/1e6:.1f} of {full/1e6:.1f} MB | {meta['baseline_deflate']/meta['text_bytes']:.3f} | {d6/sl:.3f} ({d6/1e6:.2f} MB) | "
             f"{int(r['zstd19'])/1e6:.2f} MB ({100*(int(r['zstd19'])/d6-1):+.1f}%) | {int(r['xz9e'])/1e6:.2f} MB ({100*(int(r['xz9e'])/d6-1):+.1f}%) |")
note = os.path.join(ROOT, "notes", "w4_atlas.md")
s = open(note, encoding="utf-8").read()
lev = open(os.path.join(RES, "w4_atlas_levers.md"), encoding="utf-8").read().split("\n", 2)[2]
s = s.replace("RECODE_TABLE_PLACEHOLDER", "\n".join(t)).replace("LEVER_TABLE_PLACEHOLDER", lev.strip())
open(note, "w", encoding="utf-8").write(s)
print("\n".join(t))
