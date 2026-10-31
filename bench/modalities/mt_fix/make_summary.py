"""Collect results/*.json into results/summary.tsv (metrics x runs). Usage: make_summary.py [RESULTS_DIR]"""
import json, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
runs = ["old_q0", "old_q20", "new_q0", "new_q20", "new_q20umi", "new_q20_hap90", "new_q20_hap85", "new_q20_hap75", "new_q20_hap70"]
rs = {r: json.load(open(os.path.join(d, r + ".json"))) for r in runs if os.path.exists(os.path.join(d, r + ".json"))}
keys = []
for r in rs.values():
    for k, v in r.items():
        if not isinstance(v, (list, dict)) and k != "label" and k not in keys:
            keys.append(k)
with open(os.path.join(d, "summary.tsv"), "w", newline="\n") as f:
    f.write("metric\t" + "\t".join(rs) + "\n")
    for k in keys:
        f.write(k + "\t" + "\t".join("" if rs[r].get(k) is None else ("%.4g" % rs[r][k] if isinstance(rs[r][k], float) else str(rs[r][k])) for r in rs) + "\n")
print("wrote", os.path.join(d, "summary.tsv"), len(keys), "metrics x", len(rs), "runs")
