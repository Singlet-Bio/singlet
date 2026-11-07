#!/usr/bin/env python3
"""E1 profile: tables for docs/gpu-engine/fable/E1-profile.md from a time-<jobid> directory (e1_time_summary.json).

Usage: e1_tables.py RESULTS_DIR   writes e1_nb_sweep.csv, e1_mono.csv, e1_tails.csv, e1_tail_batches.csv next to it and
prints markdown tables.
"""
import csv, json, os, sys

d = sys.argv[1]
S = json.load(open(os.path.join(d, "e1_time_summary.json")))
V = S["variants"]


def lane_ms(v, L):
    return v.get("json", {}).get({"A": "lane_a_ms", "M": "lane_m_ms", "B": "lane_b_ms"}[L])


def gb(v, L):
    return v["log"]["lanes"].get(L, {}).get("work_GB")


def batch(v, L):
    return v["log"]["lanes"].get(L, {}).get("batch")


# ---- NB sweep (staged lanes)
sweep = [k for k in V if k.startswith("nb_") or k == "base"]
rows = []
for k in sweep:
    v = V[k]
    for L in "AMB":
        rows.append(dict(variant=k, lane=L, batch=batch(v, L), ms=lane_ms(v, L), work_GB=gb(v, L), identical_pct=v["log"].get("identical_pct")))
rows.sort(key=lambda r: (r["lane"], r["batch"] or 0))
with open(os.path.join(d, "e1_nb_sweep.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\n| lane | reads in flight (batch) | lane ms | vs best | Work GB |")
print("|---|---|---|---|---|")
for L in "AMB":
    rl = [r for r in rows if r["lane"] == L and r["ms"]]
    best = min(r["ms"] for r in rl)
    seen = set()
    for r in rl:
        if r["batch"] in seen:
            continue
        seen.add(r["batch"])
        print("| %s | %s | %.1f | %+.1f%% | %.2f |" % (L, r["batch"], r["ms"], 100 * (r["ms"] / best - 1), r["work_GB"]))

# ---- mono: global vs local Work
mono = sorted(k for k in V if k.startswith("mono_"))
mrows = []
for k in mono:
    v = V[k]
    la = v["log"]["lanes"].get("A", {})
    mrows.append(dict(variant=k, mem=la.get("mem"), blocks=la.get("blocks"), threads=la.get("threads"), lane_a_ms=lane_ms(v, "A"),
                      work_GB=la.get("work_GB"), cyc_seeds=la.get("cyc_share", {}).get("seeds"), cyc_windows=la.get("cyc_share", {}).get("windows"),
                      cyc_stitching=la.get("cyc_share", {}).get("stitching"), cyc_select=la.get("cyc_share", {}).get("select"),
                      lane_m_reads=v.get("json", {}).get("lane_m_reads"), identical_pct=v["log"].get("identical_pct")))
with open(os.path.join(d, "e1_mono.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(mrows[0]))
    w.writeheader()
    w.writerows(mrows)
print("\n| variant | Work in | blocks x threads | lane A ms | cycles seeds / windows / stitching / select % | stitching-equivalent ms |")
print("|---|---|---|---|---|---|")
for r in mrows:
    print("| %s | %s | %s x %s | %.1f | %s / %s / %s / %s | %.0f |" % (r["variant"], r["mem"], r["blocks"], r["threads"], r["lane_a_ms"], r["cyc_seeds"],
                                                               r["cyc_windows"], r["cyc_stitching"], r["cyc_select"], r["lane_a_ms"] * (r["cyc_stitching"] or 0) / 100))

# ---- tails
T = S.get("tail", {})
if T:
    trow = []
    for s in T["sum"]:
        trow.append(dict(lane=s["lane"], kernel=s["kernel"], span_ms=float(s["span_ms"]), busy_thr=float(s["busy_thr"]), busy_blk=float(s["busy_blk"]),
                         drain_share=float(s["drain_share"]), idle_slot_share=1 - float(s["busy_thr"])))
    with open(os.path.join(d, "e1_tails.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trow[0]))
        w.writeheader()
        w.writerows(trow)
    with open(os.path.join(d, "e1_tail_batches.csv"), "w", newline="") as f:
        keys = ["lane", "batch", "kernel", "n", "slots", "span_ms", "busy_thr", "busy_warp", "busy_blk", "drain_ms", "drain_share", "fin_p50", "fin_p90",
                "fin_p99", "fin_p999", "longest_share", "clk_ghz", "curve"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(T["batches"])
    print("\n| lane | kernel | span ms | warp-slot busy | block-slot busy | idle slot share | drain share |")
    print("|---|---|---|---|---|---|---|")
    for r in trow:
        print("| %s | %s | %.1f | %.3f | %.3f | %.3f | %.3f |" % (r["lane"], r["kernel"], r["span_ms"], r["busy_thr"], r["busy_blk"], r["idle_slot_share"], r["drain_share"]))
    print("\nper batch k_p3: longest single read / span, finish p50/p90/p99")
    for b in T["batches"]:
        if b["kernel"] == "k_p3":
            print("  %s b%s n=%s span %s ms: longest %s, p50 %s p90 %s p99 %s, curve %s" % (b["lane"], b["batch"], b["n"], b["span_ms"], b["longest_share"],
                                                                                    b["fin_p50"], b["fin_p90"], b["fin_p99"], b["curve"]))
    if T.get("events"):
        print("\nper batch events (ms):")
        for e in T["events"]:
            print("  " + " ".join("%s=%s" % kv for kv in e.items()))
