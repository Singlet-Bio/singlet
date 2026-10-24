#!/usr/bin/env python
"""w4_corpus: markdown tables for layout, row-name invariance, barcode whitelist membership, E4 claims and empties split.

    python -P tools/w4_corpus_report2.py > results/w4_corpus_tables2.md
"""
import csv, os, collections, statistics, json
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); RES = os.path.join(BC, "results")
rd = lambda p: list(csv.DictReader(open(p), delimiter="\t")) if os.path.exists(p) else []
samples = {r["id"]: r for r in rd(os.path.join(RES, "w4_corpus_samples.tsv"))}

print("### Layout of the bundles (what the members of each sample are)\n")
mem = rd(os.path.join(RES, "w4_corpus_members.tsv"))
per = collections.defaultdict(dict)
for r in mem:
    if r["id"] != "-": per[r["id"]][r["member"]] = (int(r["csize"]), int(r["usize"]))
cls = collections.defaultdict(list)
for sid, m in per.items(): cls[len(m)].append(sid)
print("| members per sample | samples | ids |\n|---|---|---|")
for k, v in sorted(cls.items()): print(f"| {k} | {len(v)} | {', '.join(v)} |")
extra_big = ["ambient_profile.tsv", "star_Solo.out/Gene/raw/barcodes.tsv", "star_Solo.out/Gene/raw/matrix.mtx", "star_Solo.out/Gene/filtered/matrix.mtx", "cell_qc_metrics.tsv", "auto_barcodes.tsv", "gene_expression.tsv", "star_SJ.out.tab", "splice_events.tsv"]
print("\nNew-layout samples (34-46 members) carry extra text members; mean compressed bytes per sample where present:\n")
print("| member | samples with it | mean bytes |\n|---|---|---|")
for e in extra_big:
    v = [m[e][0] for m in per.values() if e in m]
    if v: print(f"| {e} | {len(v)} | {statistics.mean(v):,.0f} |")
tot_all = sum(sum(c for c, _ in m.values()) for m in per.values())
tot_1pz = sum(sum(c for k, (c, _) in m.items() if k in ("exon_counts.1pz", "intron_counts.1pz")) for m in per.values())
print(f"\nAcross the {len(per)} samples: exon+intron .1pz are {tot_1pz/1e6:.0f} MB of {tot_all/1e6:.0f} MB of sample members ({100*tot_1pz/tot_all:.1f}%).\n")

print("\n### (d) Row-name invariance per (reference build, member)\n")
v = rd(os.path.join(RES, "w4_corpus_variants.tsv"))
print("| build | member | distinct lists | rows | samples per list | note |\n|---|---|---|---|---|---|")
g = collections.OrderedDict()
for r in v: g.setdefault((r["build"], r["member"]), []).append(r)
for (b, m), rs in g.items():
    real = [r for r in rs if int(r["rows"]) > 0]
    note = "" if len(rs) == len(real) else "plus an empty-matrix stub (0 rows, x_stub) that has no names"
    print(f"| {b} | {m.split('_')[0]} | {len(real)} | {real[0]['rows']} | {', '.join(r['n_samples'] for r in real)} | {note} |")

print("\n### (f) Barcodes: whitelist membership by protocol (exon member; fractions of columns)\n")
bc = [r for r in rd(os.path.join(RES, "w4_corpus_barcodes.tsv")) if r["member"].startswith("exon") and int(r["n"]) > 0]
gp = collections.OrderedDict()
for r in bc: gp.setdefault(r["protocol"], []).append(r)
print("| protocol | samples | barcode length | columns | in 3M-feb-2018 (v3) | in 737K-aug-2016 (v2/5') | in either | worst sample (in either) |\n|---|---|---|---|---|---|---|---|")
for p, rs in gp.items():
    n = sum(int(r["n"]) for r in rs)
    f3 = sum(int(r.get("in_3M-february-2018", 0)) for r in rs) / n
    f7 = sum(int(r.get("in_737K-august-2016", 0)) for r in rs) / n
    fa = sum(int(r["in_any"]) for r in rs) / n
    worst = min(rs, key=lambda r: int(r["in_any"]) / int(r["n"]))
    lens = sorted({k for r in rs for k in json.loads(r["len_hist"])})
    print(f"| {p} | {len(rs)} | {'/'.join(lens)} | {n} | {f3:.4f} | {f7:.4f} | {fa:.4f} | {worst['id']} ({int(worst['in_any'])/int(worst['n']):.4f}) |")
reuse = rd(os.path.join(RES, "w4_corpus_barcode_reuse.tsv"))
print("\nCross-sample reuse (fraction of a sample's barcodes that also occur in another sample of the same protocol in this corpus), only protocols with >= 2 samples:\n")
print("| protocol | samples | mean fraction | min | max |\n|---|---|---|---|---|")
gr = collections.OrderedDict()
for r in reuse:
    if r["frac_in_other"] != "": gr.setdefault(r["protocol"], []).append(float(r["frac_in_other"]))
for p, vs in gr.items(): print(f"| {p} | {len(vs)} | {statistics.mean(vs):.3f} | {min(vs):.3f} | {max(vs):.3f} |")

print("\n### E4 derivability claims on the new samples (pass / compared)\n")
import glob
e4 = []
for _p in sorted(glob.glob(os.path.join(RES, "w4_corpus_e4*.tsv"))): e4 += rd(_p)
if e4:
    c = collections.OrderedDict(); fails = collections.defaultdict(list)
    for r in e4:
        if r["claim"] in ("rowcounts", "cell_calls_info"): continue
        key = (r["claim"], r["item"] if r["claim"] in ("colsums", "ptr", "perm", "colsums_frame_bytes_zstd155") else "")
        a = c.setdefault(key, [0, 0, set()])
        a[0] += int(r["n_compared"]); a[1] += int(r["n_equal"]); a[2].add(r["id"])
        if int(r["n_compared"]) != int(r["n_equal"]): fails[key].append(r["id"] + (":" + r["item"] if r["claim"] not in ("colsums", "ptr", "perm") else ""))
    print("| claim | item | compared | equal | samples | failing |\n|---|---|---|---|---|---|")
    for (cl, it), (n, e, ss) in c.items():
        print(f"| {cl} | {it} | {n} | {e} | {len(ss)} | {', '.join(sorted(set(fails[(cl, it)]))[:12])} |")
else:
    print("(no E4 rows)")

print("\n### (4) Called cells vs the other columns (exon and intron), nnz share\n")
em = rd(os.path.join(RES, "w4_corpus_empties.tsv"))
if em:
    gp = collections.OrderedDict()
    for r in em: gp.setdefault((r["protocol"], r["member"]), []).append(r)
    print("| protocol | member | samples | columns | called | frac of columns called | frac of nnz in called (mean / min / max) | nnz per called col | nnz per other col |\n|---|---|---|---|---|---|---|---|---|")
    for (p, m), rs in sorted(gp.items()):
        fn = [float(r["frac_nnz_called"]) for r in rs]
        nc = sum(int(r["n_cols"]) for r in rs); ncall = sum(int(r["n_called"]) for r in rs)
        npc = sum(int(r["nnz_called"]) for r in rs) / max(ncall, 1)
        npe = sum(int(r["nnz"]) - int(r["nnz_called"]) for r in rs) / max(nc - ncall, 1)
        print(f"| {p} | {m} | {len(rs)} | {nc} | {ncall} | {ncall/nc:.3f} | {statistics.mean(fn):.3f} / {min(fn):.3f} / {max(fn):.3f} | {npc:.0f} | {npe:.1f} |")
else:
    print("(not run)")
