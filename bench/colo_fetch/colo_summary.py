#!/usr/bin/env python3
"""colo_summary.py RES0 [PAIRS]: one table over the points of a colo.sbatch job (bench/colo_fetch/README.md).

Per point: drain span rate and cgroup core-s per M pairs (tp_point.py), engine-only core-s per M pairs (sum of the runs'
/usr/bin/time user + sys), lock wait, peaks, fallbacks, identity; for points with a co-located fetch (meta "colo" > 0) the fetch
tier's decoded pairs per second inside the drain span (pro rata by each run's fetch-to-decode interval), its network MB/s, its
core-s per M decoded pairs and its peak RSS. Each fetch point is compared with the mean of the job's points without a fetch.
Writes RES0/summary.md and RES0/summary.json.
"""
import glob, json, os, sys

RES0 = sys.argv[1]
PAIRS = float(sys.argv[2]) if len(sys.argv) > 2 else 66601887.0   # pbmc1k_v3


def tsv_row(path):
    try:
        h, r = open(path).read().splitlines()[:2]
        return dict(zip(h.split("\t"), r.split("\t")))
    except Exception:
        return {}


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def timev(path):
    u = s = 0.0
    try:
        for l in open(path):
            if "User time (seconds)" in l: u = float(l.split(":")[-1])
            if "System time (seconds)" in l: s = float(l.split(":")[-1])
    except Exception:
        return None
    return u + s


rows = []
for meta_p in sorted(glob.glob(f"{RES0}/*/meta.json"), key=os.path.getmtime):
    P = os.path.dirname(meta_p); meta = json.load(open(meta_p))
    runs = [l.split("\t") for l in open(f"{P}/runs.tsv").read().splitlines() if l.strip()] if os.path.exists(f"{P}/runs.tsv") else []
    if not runs:
        continue
    t0 = min(float(r[3]) for r in runs); t1 = max(float(r[4]) for r in runs); span = t1 - t0
    ok = [r for r in runs if r[5] == "0"]
    eng = [timev(f"{P}/c{r[0]}/time.txt") for r in ok]; eng = [e for e in eng if e is not None]
    tp = tsv_row(f"{P}/results.tsv")
    row = {"point": meta["tag"], "colo": int(meta.get("colo", 0)), "runs": len(runs), "failed": len(runs) - len(ok), "span_s": round(span, 1),
           "drain_Mps": round(len(ok) * PAIRS / 1e6 / span, 3),
           "engine_core_s_per_M": round(sum(eng) / (len(eng) * PAIRS / 1e6), 2) if eng else None,
           "cgroup_core_s_per_M": fnum(tp.get("core_s_per_M")), "cores_span": fnum(tp.get("cores_span")),
           "lock_wait_frac": fnum(tp.get("lock_wait_frac")), "host_peak_gb": fnum(tp.get("host_peak_gb")),
           "fallback_runs": tp.get("fallback_runs"), "ident_alone": tp.get("ident_alone"), "ident_cpu": tp.get("ident_cpu"),
           "other_jobs_max": tp.get("other_jobs_max")}
    cj = f"{P}/colo/runs.jsonl"
    if row["colo"] and os.path.exists(cj):
        recs = [json.loads(l) for l in open(cj) if l.strip()]
        done = [r for r in recs if r.get("status") == "done" and "t_dec" in r]
        pairs_in = cpu_in = 0.0
        for r in done:
            a, b = r["t_fetch"], r["t_dec"]
            f = max(0.0, min(b, t1) - max(a, t0)) / (b - a) if b > a else 0.0
            pairs_in += f * r["reads"]; cpu_in += f * (r.get("user", 0) + r.get("sys", 0))
        net = [tuple(map(float, l.split("\t"))) for l in open(f"{P}/colo/net.tsv") if l.strip()] if os.path.exists(f"{P}/colo/net.tsv") else []

        def at(t):
            prev = None
            for tt, bb in net:
                if tt >= t:
                    return bb if prev is None else prev[1] + (bb - prev[1]) * (t - prev[0]) / max(tt - prev[0], 1e-9)
                prev = (tt, bb)
            return net[-1][1] if net else 0.0
        tot_reads = sum(r["reads"] for r in done); tot_cpu = sum(r.get("user", 0) + r.get("sys", 0) for r in done)
        row.update({"colo_runs_done": len(done), "colo_failed": sum(r.get("status") == "failed" for r in recs),
                    "colo_stopped": sum(r.get("status") == "stopped" for r in recs),
                    "colo_Mps_in_span": round(pairs_in / 1e6 / span, 3),
                    "colo_cores_in_span": round(cpu_in / span, 2),
                    "colo_net_MBps": round((at(t1) - at(t0)) / 1e6 / span, 1) if net else None,
                    "colo_core_s_per_M": round(tot_cpu / (tot_reads / 1e6), 2) if tot_reads else None,
                    "colo_peak_rss_gb": round(max((r.get("maxrss_kb", 0) for r in done), default=0) / 1e6, 1),
                    "colo_fail_why": sorted({r.get("why", "")[:120] for r in recs if r.get("status") == "failed"})[:3]})
    rows.append(row)

base = [r["drain_Mps"] for r in rows if r["colo"] == 0]
bmean = sum(base) / len(base) if base else None
for r in rows:
    if r["colo"] and bmean:
        r["drain_vs_base_pct"] = round(100 * (r["drain_Mps"] / bmean - 1), 1)
        r["fetch_covers_drain"] = round(r.get("colo_Mps_in_span", 0) / r["drain_Mps"], 2) if r["drain_Mps"] else None

cols = ["point", "colo", "runs", "failed", "drain_Mps", "drain_vs_base_pct", "engine_core_s_per_M", "cgroup_core_s_per_M", "cores_span",
        "lock_wait_frac", "colo_Mps_in_span", "fetch_covers_drain", "colo_cores_in_span", "colo_net_MBps", "colo_core_s_per_M",
        "colo_runs_done", "colo_failed", "host_peak_gb", "colo_peak_rss_gb", "fallback_runs", "ident_cpu", "other_jobs_max"]
md = ["# Co-located fetch test: " + os.path.basename(RES0), "",
      f"Baseline drain (points without a fetch, mean of {len(base)}): {bmean:.3f} M pairs/s" if bmean else "No baseline point.", "",
      "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in rows:
    md.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
md += ["", "drain_vs_base_pct: drain rate against the baseline mean. fetch_covers_drain: pairs the co-located fetch decoded per pair the GPU drained,",
       "both inside the drain span (>= 1 means one GPU job could feed itself). engine_core_s_per_M excludes the fetch; cgroup_core_s_per_M includes it.",
       "Noise: an A/A gap of up to 12% was seen between jobs (gpu-next review, 2026-11-05); compare points only within this job."]
for r in rows:
    if r.get("colo_fail_why"):
        md.append(f"- {r['point']} fetch failures: {r['colo_fail_why']}")
open(f"{RES0}/summary.md", "w").write("\n".join(md) + "\n")
json.dump({"baseline_drain_Mps": bmean, "points": rows}, open(f"{RES0}/summary.json", "w"), indent=1)
print("\n".join(md))
