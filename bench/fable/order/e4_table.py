#!/usr/bin/env python3
"""E4 (docs/gpu-engine/fable/E4-order.md): one markdown table from the e4_<label>.json files of e4_run.sbatch.

usage: e4_table.py DIR [BASELINE_LABEL]   (DIR holds e4_*.json; the baseline defaults to v0)
"""
import json, os, sys

d = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "v0"
rows = {}
for f in sorted(os.listdir(d)):
    if f.startswith("e4_") and f.endswith(".json"):
        try:
            rows[f[3:-5]] = json.load(open(os.path.join(d, f)))
        except json.JSONDecodeError as e:
            print(f"skip {f}: {e}", file=sys.stderr)

def lane(j, k):
    return j["e4"]["lanes"][k]

b = rows.get(base)
print("| variant | options | A ms | M ms | B ms | total ms | M distinct/s | vs %s | identical | A/M/B work GB | A/M/B p3 slot fill | A/M/B lane balance | A/M/B drain |" % base)
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, j in rows.items():
    e = j["e4"]
    opts = []
    if e["presort"]: opts.append("presort")
    if e["p3order"] != "none": opts.append(e["p3order"])
    if e["queue"]: opts.append("queue")
    if e["fused"]: opts.append("fused " + e["fused"] + ("" if e["fused_bps"] == [0, 0, 0] else " bps " + "/".join(map(str, e["fused_bps"]))))
    if e["batches"] != [262144, 262144, 65536]: opts.append("batches " + "/".join(str(x // 1024) + "k" for x in e["batches"]))
    if e["tail"]: opts.append("tail")
    L = [lane(j, x) for x in "AMB"]
    tot = j["gpu_ms_median"]
    rel = (b["gpu_ms_median"] / tot) if b else float("nan")
    ident = "%d/%d" % (j["identical_distinct"], j["distinct_reads"])
    def tl(key, fmt="%.2f"):
        return "/".join((fmt % x["tail"][key]) if x["tail"]["launches"] else "-" for x in L)
    print("| %s | %s | %.1f | %.1f | %.1f | %.1f | %.2f | %.2fx | %s | %s | %s | %s | %s |" % (
        k, ", ".join(opts) or "harness order", L[0]["ms"], L[1]["ms"], L[2]["ms"], tot, j["gpu_distinct_reads_per_s"] / 1e6, rel, ident,
        "/".join("%.1f" % x["work_gb"] for x in L), tl("slot_fill"), tl("lane_balance"), tl("drain_share")))
