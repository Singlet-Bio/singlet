#!/usr/bin/env python3
"""Summarise the step-5 replay runs (thread / contention scaling of export). usage: parse_replay.py replay_dir out_json"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_w11 import parse_log
d, out = sys.argv[1], sys.argv[2]
KEYS = ["export.core", "export.csc_conversion", "cell_calling", "ambient", "doublet", "saturation_curve", "mt.call_and_write_variants",
        "demux.thread", "autoK.total", "demux.final_fit.K=1", "join.write_threads", "mt_donor_outputs(G6)", "demux.extract_covered_snps"]
res = {}
for n in ["full"] + ["t1", "t4", "t8"] + ["c%d" % i for i in range(1, 9)]:
    p = os.path.join(d, n + ".w11.log")
    if not os.path.exists(p):
        continue
    sec, notes = parse_log(p)
    r = {}
    for k in KEYS:
        if k in sec:
            r[k] = dict(wall_s=round(sum(x["wall"] for x in sec[k]), 3), cpu_s=round(sum(x["cpu"] for x in sec[k]), 3),
                        thr_cpu_s=round(sum(x["thr"] for x in sec[k]), 3))
    for k in ("main.total_including_teardown", "stage.pileup", "stage.pileup_streaming"):
        if k in sec:
            r[k] = dict(wall_s=round(sum(x["wall"] for x in sec[k]), 3), cpu_s=round(sum(x["cpu"] for x in sec[k]), 3))
    r["demux_threads_arg"] = next((int(re.search(r"threads=(\d+)", x).group(1)) for x in notes if x.startswith("demux_input")), None)
    r["sizes"] = [x for x in notes if x.startswith(("demux_input", "autoK_done", "demux_done"))]
    res[n] = r
json.dump(res, open(out, "w"), indent=1)
print("%-5s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s" % ("run", "exp_wall", "exp_cpu", "demux_thr", "autoK_thr", "final_thr", "dbl_wall", "dbl_cpu", "amb_cpu"))
for n, r in res.items():
    g = lambda k, f: r.get(k, {}).get(f, float("nan"))
    print("%-5s %-9.1f %-9.1f %-9.1f %-9.1f %-9.1f %-9.2f %-9.2f %-9.2f" % (n, g("export.core", "wall_s"), g("export.core", "cpu_s"), g("demux.thread", "thr_cpu_s"),
          g("autoK.total", "thr_cpu_s"), g("demux.final_fit.K=1", "thr_cpu_s"), g("doublet", "wall_s"), g("doublet", "cpu_s"), g("ambient", "cpu_s")))
