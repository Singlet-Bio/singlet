#!/usr/bin/env python
"""w4_corpus: aggregate the bench runs into the per-claim tables (markdown on stdout, TSVs in results/w4_corpus_*).

    python -P tools/w4_corpus_report.py > results/w4_corpus_tables.md
"""
import csv, glob, os, sys, collections, json, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); RES = os.path.join(BC, "results")
SPECIES = {"GRCh38-2024-A": "human", "GRCm39-2024-A": "mouse", "GRCh38-GRCm39-barnyard": "barnyard", "GRCz11": "zebrafish",
           "Mmul_10": "macaque", "mRatBN7.2": "rat"}
ORIG_CONTAINER = 53.5 / 114.4      # original corpus headline: a_names_perm_model / today
ORIG_PAYLOAD = -0.151


def rd(p):
    return list(csv.DictReader(open(p), delimiter="\t"))


def fnum(x):
    try: return float(x)
    except Exception: return 0.0


samples = {r["id"]: r for r in rd(os.path.join(RES, "w4_corpus_samples.tsv"))}
rows = []
for p in sorted(glob.glob(os.path.join(RES, "w4_corpus_bench_p*.tsv"))):
    rows += rd(p)
by = collections.defaultdict(dict)
for r in rows:
    by[(r["sample"], r["member"])][r["codec"]] = r
recs = []
for (sid, mem), d in sorted(by.items()):
    if "ref_1pz_v1_L3" not in d or "a_names_perm_model" not in d:
        continue
    s = samples.get(sid, {})
    ref, a, b = d["ref_1pz_v1_L3"], d["a_names_perm_model"], d.get("b_rans16")
    orig = fnum(ref["orig_bytes"]); nnz = fnum(ref["nnz"])
    tp = fnum(ref["payload_bytes"])
    rec = dict(id=sid, member=mem.split("_")[0], protocol=s.get("protocol", "?"), build=s.get("build", "?"),
               species=SPECIES.get(s.get("build", ""), "?"), m=int(fnum(ref["m"])), n=int(fnum(ref["n"])), nnz=int(nnz), orig=orig,
               today_payload=tp, names_z=fnum(ref["names_bytes"]), perm_z=fnum(ref["side_bytes"]), ptr_z=fnum(ref["ptr_bytes"]),
               colsums_z=fnum(ref["colsums_bytes"]), a_total=fnum(a["enc_bytes"]), a_payload=fnum(a["payload_bytes"]),
               a_names=fnum(a["names_bytes"]), a_rowc=fnum(a["side_bytes"]), a_ok=int(fnum(a["matrix_ok"])),
               a_row_mode=a["row_mode"], a_bc_mode=a["bc_mode"], a_col_bytes=fnum(a["col_names_bytes"]), a_row_bytes=fnum(a["row_names_bytes"]))
    rec["derivable_z"] = rec["names_z"] + rec["perm_z"] + rec["ptr_z"] + rec["colsums_z"]
    rec["derivable_share"] = rec["derivable_z"] / orig
    rec["names_share"] = rec["names_z"] / orig
    rec["payload_share"] = tp / orig
    rec["container_ratio"] = rec["a_total"] / orig
    rec["a_payload_vs_today"] = rec["a_payload"] / tp - 1
    if b and b["codec"] == "b_rans16" and fnum(b["enc_bytes"]) > 0:
        rec["b_payload"] = fnum(b["payload_bytes"]); rec["b_side"] = fnum(b["side_bytes"]); rec["b_ok"] = int(fnum(b["matrix_ok"]))
        rec["b_payload_vs_today"] = rec["b_payload"] / tp - 1
        rec["b_payload_side_vs_today"] = (rec["b_payload"] + rec["b_side"]) / tp - 1
        rec["combined"] = rec["a_names"] + rec["a_rowc"] + rec["b_payload"] + 64
        rec["combined_note"] = ""
    else:
        rec["b_payload"] = None; rec["b_side"] = None; rec["b_ok"] = None
        rec["b_payload_vs_today"] = None; rec["b_payload_side_vs_today"] = None
        rec["combined"] = rec["a_total"]; rec["combined_note"] = "b_rans16 unsupported (m > 2^19): A payload used"
    rec["combined_ratio"] = rec["combined"] / orig
    recs.append(rec)

cols = list(recs[0].keys()) if recs else []
with open(os.path.join(RES, "w4_corpus_perfile.tsv"), "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(cols)
    for r in recs: w.writerow([r[c] if r[c] is not None else "" for c in cols])


def agg(group_key, metric, num=None, den=None, label_fmt="{:.3f}", recs_=None):
    """rows per group: n files, mean/min/max of metric, worst-case sample (max for loss metrics given by caller), pooled num/den."""
    out = []
    g = collections.OrderedDict()
    for r in (recs_ or recs):
        if r.get(metric) is None: continue
        g.setdefault(r[group_key], []).append(r)
    for k, rs in sorted(g.items()):
        vals = [r[metric] for r in rs]
        pooled = sum(r[num] for r in rs) / sum(r[den] for r in rs) if num else None
        out.append((k, len(rs), len({r["id"] for r in rs}), statistics.mean(vals), min(vals), max(vals), rs, pooled))
    return out


def table(title, group_key, metric, num, den, worst="max", fmt="{:.3f}", pooled_fmt=None, recs_=None, pooled_delta=False):
    print(f"\n### {title}\n")
    print(f"| {group_key} | files | samples | mean | min | max | worst-case file | pooled (byte-weighted) |")
    print("|---|---|---|---|---|---|---|---|")
    for k, n, ns, mean, mn, mx, rs, pooled in agg(group_key, metric, num, den, recs_=recs_):
        wr = max(rs, key=lambda r: r[metric]) if worst == "max" else min(rs, key=lambda r: r[metric])
        pv = pooled - 1 if pooled_delta and pooled is not None else pooled
        print(f"| {k} | {n} | {ns} | {fmt.format(mean)} | {fmt.format(mn)} | {fmt.format(mx)} | {wr['id']}/{wr['member']} ({fmt.format(wr[metric])}) | {pooled_fmt.format(pv) if pooled_fmt and pv is not None else ''} |")
    allv = [r[metric] for r in (recs_ or recs) if r.get(metric) is not None]
    rr = [r for r in (recs_ or recs) if r.get(metric) is not None]
    if rr:
        pooled = sum(r[num] for r in rr) / sum(r[den] for r in rr)
        pv = pooled - 1 if pooled_delta else pooled
        print(f"| **all** | {len(rr)} | {len({r['id'] for r in rr})} | {fmt.format(statistics.mean(allv))} | {fmt.format(min(allv))} | {fmt.format(max(allv))} | | {pooled_fmt.format(pv) if pooled_fmt else ''} |")


if not recs:
    print("no bench rows yet"); sys.exit(0)
ok_a = sum(r["a_ok"] for r in recs); print(f"files with ref+a rows: {len(recs)}; a_names_perm_model matrix roundtrip OK on {ok_a}")
okb = [r for r in recs if r["b_ok"] is not None]; print(f"files with b_rans16 rows: {len(okb)}; roundtrip OK on {sum(r['b_ok'] for r in okb)}; unsupported on {len(recs) - len(okb)}")
tot_o = sum(r["orig"] for r in recs); tot_a = sum(r["a_total"] for r in recs)
print(f"\nPOOLED all files: today {tot_o/1e6:.1f} MB; a_names_perm_model {tot_a/1e6:.1f} MB = {tot_a/tot_o:.3f} ({tot_o/tot_a:.2f}x) [original corpus 0.468 = 2.14x]")
tot_c = sum(r["combined"] for r in recs); print(f"POOLED combined projection (A names + A row counts + B payload): {tot_c/1e6:.1f} MB = {tot_c/tot_o:.3f} ({tot_o/tot_c:.2f}x)")
tp = sum(r["today_payload"] for r in okb); tb = sum(r["b_payload"] for r in okb); tbs = sum(r["b_payload"] + r["b_side"] for r in okb)
print(f"POOLED b_rans16 payload vs today's payload (chunks+ptr+chunk table), n={len(okb)} files: {tb/tp-1:+.1%}; incl. row-count side stream: {tbs/tp-1:+.1%} [original corpus -15.1%]")
for sel in ("exon", "intron"):
    rr = [r for r in okb if r["member"] == sel]
    if rr: print(f"  {sel}: n={len(rr)} payload {sum(r['b_payload'] for r in rr)/sum(r['today_payload'] for r in rr)-1:+.1%}")

for grp in ("protocol", "species"):
    table(f"(a) container ratio = a_names_perm_model bytes / today's file bytes, by {grp} (original corpus pooled 0.468)", grp, "container_ratio", "a_total", "orig", worst="max", fmt="{:.3f}", pooled_fmt="{:.3f}")
for grp in ("protocol", "species"):
    table(f"(b) b_rans16 payload vs today's payload, by {grp} (negative = smaller; original corpus -15.1% pooled, -11..-24% per file)", grp, "b_payload_vs_today", "b_payload", "today_payload", worst="max", fmt="{:+.3f}", pooled_fmt="{:+.3f}", pooled_delta=True)
for grp in ("protocol", "species"):
    table(f"(c) derivable share of today's file bytes = (names + permutation + ptr + colsums) / file, by {grp} (original corpus: 77.9% per sample)", grp, "derivable_share", "derivable_z", "orig", worst="min", fmt="{:.3f}", pooled_fmt="{:.3f}")
for grp in ("protocol", "species"):
    table(f"(c2) names (row + column) share of today's file bytes, by {grp}", grp, "names_share", "names_z", "orig", worst="max", fmt="{:.3f}", pooled_fmt="{:.3f}")
for grp in ("protocol", "species"):
    table(f"Total file ratio, combined projection (vocab id + whitelist/inline barcodes + row counts + b_rans16 payload) / today, by {grp} (original corpus about 0.40)", grp, "combined_ratio", "combined", "orig", worst="max", fmt="{:.3f}", pooled_fmt="{:.3f}")

print("\n### per-file detail\n")
print("| sample | member | protocol | build | m | n | nnz | file MB | derivable % | names % | container ratio | b payload vs today | combined ratio | A row/bc mode |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in recs:
    bp = "n/a" if r["b_payload_vs_today"] is None else f"{r['b_payload_vs_today']:+.1%}"
    print(f"| {r['id']} | {r['member']} | {r['protocol']} | {r['build']} | {r['m']} | {r['n']} | {r['nnz']} | {r['orig']/1e6:.2f} | {100*r['derivable_share']:.1f} | {100*r['names_share']:.1f} | {r['container_ratio']:.3f} | {bp} | {r['combined_ratio']:.3f} | {r['a_row_mode']}/{r['a_bc_mode']} |")
