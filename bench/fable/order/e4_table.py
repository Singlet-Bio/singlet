#!/usr/bin/env python3
"""E4 (docs/gpu-engine/fable/E4-order.md): one markdown table from the e4_<label>.json files of e4_run.sbatch.

usage: e4_table.py DIR [BASELINE_LABEL] [--tail]   (DIR holds e4_*.json; the baseline defaults to v0)
--tail adds a second table: per lane, the phase-3 span (sum of the batches' kernel durations) against the busy time
(span x slot fill x warp balance: the duration if every resident warp slot were busy until its warp finished).
"""
import json, os, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
d = args[0]
base = args[1] if len(args) > 1 else "v0"
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
print("| variant | options | A ms | M ms | B ms | total ms | M distinct/s | vs %s | identical | A/M/B work GB | p3 regs A/M/B | A/M/B p3 slot fill | A/M/B drain |" % base)
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, j in rows.items():
    e = j.get("e4")
    if e is None:   # a reference build without the E4 fields
        print("| %s | reference build, harness order | %.1f | %.1f | %.1f | %.1f | %.2f | %.2fx | %d/%d | | | | |" % (
            k, j["lane_a_ms"], j["lane_m_ms"], j["lane_b_ms"], j["gpu_ms_median"], j["gpu_distinct_reads_per_s"] / 1e6,
            (b["gpu_ms_median"] / j["gpu_ms_median"]) if b else float("nan"), j["identical_distinct"], j["distinct_reads"]))
        continue
    opts = []
    if e["presort"]: opts.append("presort")
    lo = [lane(j, x).get("order", e["p3order"]) for x in "AMB"]
    if len(set(lo)) > 1: opts.append("order " + "/".join(lo))
    elif e["p3order"] != "none": opts.append(e["p3order"])
    if e["queue"]: opts.append("queue")
    if e.get("p3_block", 128) != 128: opts.append("p3 block %d" % e["p3_block"])
    if e.get("p3_maxreg", 0): opts.append("maxreg %d" % e["p3_maxreg"])
    if e["fused"]: opts.append("fused " + e["fused"] + ("" if e["fused_bps"] == [0, 0, 0] else " bps " + "/".join(map(str, e["fused_bps"])))
                               + ("" if e.get("fused_lb", 1) == 1 else " lb %d" % e["fused_lb"]))
    if e["batches"] != [262144, 262144, 65536]: opts.append("batches " + "/".join(str(x // 1024) + "k" for x in e["batches"]))
    if e["tail"]: opts.append("tail")
    L = [lane(j, x) for x in "AMB"]
    tot = j["gpu_ms_median"]
    rel = (b["gpu_ms_median"] / tot) if b else float("nan")
    ident = "%d/%d" % (j["identical_distinct"], j["distinct_reads"])
    def tl(key, fmt="%.2f"):
        return "/".join((fmt % x["tail"][key]) if x["tail"]["launches"] else "-" for x in L)
    regs = "/".join(str(x.get("p3_regs", "")) for x in L) if "p3_regs" in L[0] else ""
    print("| %s | %s | %.1f | %.1f | %.1f | %.1f | %.2f | %.2fx | %s | %s | %s | %s | %s |" % (
        k, ", ".join(opts) or "harness order", L[0]["ms"], L[1]["ms"], L[2]["ms"], tot, j["gpu_distinct_reads_per_s"] / 1e6, rel, ident,
        "/".join("%.1f" % x["work_gb"] for x in L), regs, tl("slot_fill"), tl("drain_share")))

if "--tail" in sys.argv:
    print()
    print("| variant | lane | phase-3 ms | p3 span ms | busy ms | busy / span | slot fill | warp balance | lane balance | drain | resident warps | launches |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, j in rows.items():
        e = j.get("e4")
        if e is None or not e["tail"]:
            continue
        for x in "AMB":
            l = lane(j, x); t = l["tail"]
            if not t["launches"]:
                continue
            busy = t["span_ms"] * t["slot_fill"] * t["warp_balance"]
            print("| %s | %s | %.1f | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f | %.2f | %d | %d |" % (
                k, x, l["phase_ms"][2], t["span_ms"], busy, busy / t["span_ms"] if t["span_ms"] else 0, t["slot_fill"], t["warp_balance"],
                t["lane_balance"], t["drain_share"], t["resident_warps"], t["launches"]))
