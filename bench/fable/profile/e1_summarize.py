#!/usr/bin/env python3
"""E1 profile: summarise one e1_time job directory (stdlib only).

Usage: e1_summarize.py OUTDIR   (OUTDIR = $W/jobs/fable/profile/results/time-<jobid>)
Reads the m5 JSON/logs of every variant, the nsys GPU trace of the base run and the TAIL lines of the m5t run.
Writes OUTDIR/e1_time_summary.json and prints a text summary.
"""
import csv, glob, json, os, re, statistics, sys

out = sys.argv[1]
S = {"job_dir": out}
NAN = float("nan")


def lane_of_caps(name):
    i = name.find("Caps")
    if i < 0:
        return None
    seg = name[i:i + 200]
    nums = re.findall(r"Li(\d+)E", seg) or re.findall(r"\d+", seg.split(">")[0])
    if len(nums) < 4:
        return None
    return {"16": "A", "64": "M", "128": "B"}.get(nums[3], "?")


def parse_log(path):
    d = {"lanes": {}}
    if not os.path.exists(path):
        return d
    for line in open(path, errors="replace"):
        m = re.match(r"lane (\w) \(staged\): (\d+) reads, batches of (\d+), work (\d+) B/read \(([\d.]+) GB\)", line)
        if m:
            d["lanes"].setdefault(m[1], {}).update(reads=int(m[2]), batch=int(m[3]), work_B=int(m[4]), work_GB=float(m[5]), mode="staged")
        m = re.match(r"lane (\w): (\d+) reads, (\d+) blocks x (\d+) threads, work (\d+) B/thread \((\w+), ([\d.]+) GB\)", line)
        if m:
            d["lanes"].setdefault(m[1], {}).update(reads=int(m[2]), blocks=int(m[3]), threads=int(m[4]), work_B=int(m[5]),
                                                   mem=m[6], work_GB=float(m[7]), mode="mono")
        m = re.match(r"\s+lane (\w) run (\d+): ([\d.]+) ms(?: \(phase 1 ([\d.]+), scan ([\d.]+), phase 3 ([\d.]+)\))?", line)
        if m:
            L = d["lanes"].setdefault(m[1], {})
            L.setdefault("runs", []).append(float(m[3]))
            if m[4]:
                L.setdefault("phases", []).append([float(m[4]), float(m[5]), float(m[6])])
        m = re.match(r"\s+lane (\w) thread cycles: seeds ([\d.]+)%, windows ([\d.]+)%, stitching ([\d.]+)%, select\+encode ([\d.]+)%(?:; ([\d.]+) SA entries per read)?", line)
        if m:
            d["lanes"].setdefault(m[1], {}).update(cyc_share=dict(seeds=float(m[2]), windows=float(m[3]), stitching=float(m[4]), select=float(m[5])),
                                                   sa_entries_per_read=float(m[6]) if m[6] else None)
        m = re.match(r"\s+lane (\w) inside stitching: stitchAlignToTranscript ([\d.]+)%, finalization ([\d.]+)% \(of which recording ([\d.]+)%\), copies ([\d.]+)%", line)
        if m:
            d["lanes"].setdefault(m[1], {}).update(inside_stitch=dict(sat=float(m[2]), finalize=float(m[3]), recording=float(m[4]), copies=float(m[5])))
        m = re.match(r"device (.*), (\d+) SMs; sizeof Work: A (\d+), B (\d+), host (\d+)", line)
        if m:
            d.update(device=m[1], sms=int(m[2]), sizeof_work_A=int(m[3]), sizeof_work_B=int(m[4]))
        m = re.match(r"M5: (\d+) / (\d+) distinct reads identical \(([\d.]+)%\), reads weighted (\d+) / (\d+) \(([\d.]+)%\)", line)
        if m:
            d.update(identical=f"{m[1]}/{m[2]}", identical_pct=float(m[3]), identical_reads=f"{m[4]}/{m[5]}", identical_reads_pct=float(m[6]))
        m = re.match(r"\s+heavy lane: (\d+) reads", line)
        if m:
            d["heavy_reads"] = int(m[1])
        if line.startswith("TAIL"):
            kv = dict(re.findall(r"(\w+)=([^ \n]+)", line))
            d.setdefault(line.split()[0], []).append(kv)
    for L in d["lanes"].values():
        if "runs" in L and len(L["runs"]) > 1:
            L["ms_median"] = statistics.median(L["runs"][1:])
        if "phases" in L and len(L["phases"]) > 1:
            ph = L["phases"][1:]
            L["phase_ms_mean"] = [round(sum(p[k] for p in ph) / len(ph), 3) for k in range(3)]
    return d


def variant(label):
    j = os.path.join(out, label + ".json")
    v = {"label": label}
    if os.path.exists(j):
        try:
            v["json"] = json.load(open(j))
        except Exception as e:  # noqa
            v["json_error"] = str(e)
    v["log"] = parse_log(os.path.join(out, label + ".log"))
    return v


labels = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(out, "*.log"))
                if not os.path.basename(p).startswith(("nsys_stats", "summary")))
S["variants"] = {}
hdr = ["variant", "A ms", "M ms", "B ms", "GPU ms", "identical %", "A GB", "M GB", "B GB", "M/B reads"]
print("%-34s %8s %8s %8s %8s %12s %6s %6s %6s  %s" % tuple(hdr))
for lab in labels:
    v = variant(lab)
    S["variants"][lab] = v
    j = v.get("json", {})
    lg = v["log"]
    ln = lg["lanes"]
    gb = [ln.get(L, {}).get("work_GB", NAN) for L in "AMB"]
    print("%-34s %8.1f %8.1f %8.1f %8.1f %12.6f %6.2f %6.2f %6.2f  %s/%s" % (
        lab, j.get("lane_a_ms", NAN), j.get("lane_m_ms", NAN), j.get("lane_b_ms", NAN), j.get("gpu_ms_median", NAN),
        lg.get("identical_pct", NAN), gb[0], gb[1], gb[2], j.get("lane_m_reads", "?"), j.get("lane_b_reads", "?")))

base = S["variants"].get("base")
if base:
    print("\nbase: per-lane phase ms (phase 1, scan = thrust scan + syncs + k_p2, phase 3) and thread-cycle shares")
    for L, d in base["log"]["lanes"].items():
        print("  lane %s: reads %s batch %s work %s B/read; median %s ms; phases %s; cycles %s; SA/read %s; inside stitching %s" % (
            L, d.get("reads"), d.get("batch"), d.get("work_B"), d.get("ms_median"), d.get("phase_ms_mean"), d.get("cyc_share"),
            d.get("sa_entries_per_read"), d.get("inside_stitch")))

# ---- nsys GPU trace of the base run
tr = os.path.join(out, "nsys_base_cuda_gpu_trace.csv")
if os.path.exists(tr):
    rows = list(csv.DictReader(open(tr, errors="replace")))

    def col(r, pre):
        for k in r:
            if k and k.startswith(pre):
                return r[k]
        return ""
    ops = []
    for r in rows:
        name = col(r, "Name")
        st = float(col(r, "Start") or 0)
        du = float(col(r, "Duration") or 0)
        by = col(r, "Bytes")
        low = name.lower()
        if "k_p1" in name: c = "p1"
        elif "k_p2" in name: c = "p2"
        elif "k_p3" in name: c = "p3"
        elif "k_copy_status" in name: c = "copy_status"
        elif "k_lane" in name: c = "mono"
        elif "scan" in low or "cub::" in name or "thrust::" in name: c = "scan"
        elif "memset" in low: c = "memset"
        elif "device-to-host" in low or "dtoh" in low: c = "d2h"
        elif "host-to-device" in low or "htod" in low: c = "h2d"
        else: c = "other"
        try:
            b = float(by) if by else 0.0
        except ValueError:
            b = 0.0
        ops.append(dict(c=c, st=st, en=st + du, du=du, lane=lane_of_caps(name), bytes=b, regs=col(r, "Reg"), grid=col(r, "GrdX"),
                        blk=col(r, "BlkX"), name=name[:120]))
    ops.sort(key=lambda o: o["st"])
    S["nsys_op_counts"] = {}
    for o in ops:
        S["nsys_op_counts"][o["c"]] = S["nsys_op_counts"].get(o["c"], 0) + 1
    # reps: a k_p1 right after a small host-to-device copy (the g_clk / g_scan reset) starts a rep of the lane of that k_p1
    reps = []
    cur = None
    for i, o in enumerate(ops):
        if o["c"] == "p1" and i > 0 and ops[i - 1]["c"] == "h2d" and ops[i - 1]["bytes"] <= 0.0001:
            cur = dict(lane=o["lane"], ops=[])
            reps.append(cur)
        if cur is None:
            continue
        if o["c"] in ("p1", "scan", "d2h", "memset", "p2", "p3", "copy_status"):
            if o["c"] == "d2h" and o["bytes"] > 0.0001:   # result copies after the last rep
                cur = None
                continue
            if o["c"] in ("p1", "p2", "p3") and o["lane"] != cur["lane"]:
                cur = None
                continue
            cur["ops"].append(o)
        elif o["c"] == "h2d":
            cur = None
    lanes = {}
    for r in reps:
        if r["ops"]:
            lanes.setdefault(r["lane"], []).append(r)
    S["nsys"] = {}
    print("\nnsys (base run; mean of reps 1..3 per lane; ms)")
    cats = ["p1", "scan", "d2h", "memset", "p2", "p3", "copy_status"]
    for L, rl in lanes.items():
        use = rl[1:] if len(rl) > 1 else rl
        agg = {c: 0.0 for c in cats}
        gapb = {c: 0.0 for c in cats}
        span = 0.0
        nb = 0
        regs = {}
        p3b = p1b = p2b = None
        for r in use:
            o0 = r["ops"]
            span += (o0[-1]["en"] - o0[0]["st"]) / 1e6
            for k, o in enumerate(o0):
                agg[o["c"]] += o["du"] / 1e6
                if k > 0:
                    gapb[o["c"]] += (o["st"] - o0[k - 1]["en"]) / 1e6
                if o["c"] in ("p1", "p2", "p3"):
                    regs[o["c"]] = o["regs"]
            d3 = [o["du"] / 1e6 for o in o0 if o["c"] == "p3"]
            d1 = [o["du"] / 1e6 for o in o0 if o["c"] == "p1"]
            d2 = [o["du"] / 1e6 for o in o0 if o["c"] == "p2"]
            p3b = [a + b for a, b in zip(p3b, d3)] if p3b else d3
            p1b = [a + b for a, b in zip(p1b, d1)] if p1b else d1
            p2b = [a + b for a, b in zip(p2b, d2)] if p2b else d2
            nb = len(d3)
        n = len(use)
        rec = dict(reps=len(rl), batches=nb, span_ms=span / n, ms={c: agg[c] / n for c in cats}, gap_before_ms={c: gapb[c] / n for c in cats},
                   gap_total_ms=sum(gapb.values()) / n, regs=regs, p3_batch_ms=[x / n for x in p3b], p1_batch_ms=[x / n for x in p1b],
                   p2_batch_ms=[x / n for x in p2b])
        S["nsys"][L] = rec
        print("  lane %s: %d reps, %d batches, span %.2f; %s; gaps %.3f (%s); regs %s" % (
            L, len(rl), nb, rec["span_ms"], ", ".join("%s %.2f" % (c, rec["ms"][c]) for c in cats), rec["gap_total_ms"],
            ", ".join("before %s %.3f" % (c, rec["gap_before_ms"][c]) for c in cats if rec["gap_before_ms"][c] > 0.0005), regs))
        print("      k_p3 per batch ms: " + ", ".join("%.2f" % x for x in rec["p3_batch_ms"]))
        print("      k_p1 per batch ms: " + ", ".join("%.2f" % x for x in rec["p1_batch_ms"]) + " | k_p2: " + ", ".join("%.2f" % x for x in rec["p2_batch_ms"]))
    if S["nsys"]:
        tot = sum(r["span_ms"] for r in S["nsys"].values())
        print("  all lanes span %.1f ms; shares: %s, gaps %.2f%%" % (
            tot, ", ".join("%s %.1f%%" % (c, 100 * sum(S["nsys"][L]["ms"][c] for L in S["nsys"]) / tot) for c in cats),
            100 * sum(S["nsys"][L]["gap_total_ms"] for L in S["nsys"]) / tot))
    print("  op counts:", S["nsys_op_counts"])

# ---- tails (m5t)
tl = S["variants"].get("tail")
if tl and tl["log"].get("TAIL"):
    print("\ntails (m5t, last rep)")
    S["tail"] = {"sum": tl["log"].get("TAILSUM", []), "batches": tl["log"]["TAIL"], "clk": tl["log"].get("TAILCLK", [])}
    for s in tl["log"].get("TAILEVSUM", []):
        print("  events (rep before the analysed one): " + " ".join("%s=%s" % kv for kv in s.items()))
    S["tail"]["events"] = tl["log"].get("TAILEV", [])
    for s in tl["log"].get("TAILSUM", []):
        print("  " + " ".join("%s=%s" % kv for kv in s.items()))
    for t in tl["log"]["TAIL"]:
        print("  %s b%s %s n=%s slots=%s span=%s busy thr/warp/blk=%s/%s/%s drain=%s fin p50/p90/p99/p999=%s/%s/%s/%s longest=%s clk=%s" % (
            t.get("lane"), t.get("batch"), t.get("kernel"), t.get("n"), t.get("slots"), t.get("span_ms"), t.get("busy_thr"), t.get("busy_warp"),
            t.get("busy_blk"), t.get("drain_share"), t.get("fin_p50"), t.get("fin_p90"), t.get("fin_p99"), t.get("fin_p999"), t.get("longest_share"),
            t.get("clk_ghz")))
json.dump(S, open(os.path.join(out, "e1_time_summary.json"), "w"), indent=1, default=str)
print("\nwrote", os.path.join(out, "e1_time_summary.json"))
