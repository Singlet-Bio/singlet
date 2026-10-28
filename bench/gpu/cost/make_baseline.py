"""Build cpu_baseline.json from runs/single/<GSM>/ and runs/<pack tag>/i*_<GSM>/ (pulled from Clipper). Run: python make_baseline.py"""
import json, os, glob, statistics as st
from parse_cost import parse
HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["GSM4006843", "GSM5411066", "GSM7901264", "GSM7036403", "GSM6470208", "GSM7656149", "pbmc1k_v3", "nuclei2k_v2"]
SEC129 = {  # section 12.9 / bench/gpu/host README (old 28969b87 dev build on Xeon 6730P / c008): non-host index load, core-s
 "GSM4006843": 33.2, "GSM5411066": 28.6, "GSM7901264": 28.0, "GSM7036403": 53.3, "GSM6470208": 46.2, "GSM7656149": 42.7, "pbmc1k_v3": None, "nuclei2k_v2": None}
out = {"binary": {"name": "singlet-w3", "commit_built": "bb51b0638dd8d06b0eaafead9e512835e7aa4b12",
        "equivalent_to": "prod/campaign c4a1be38 (git diff bb51b063..c4a1be38 is only bench/star_accel/w3_timing.sbatch)",
        "pgo": "yes, 3' (pbmc2m) + 5' (pbmc5p_v1) training, jobs 788956", "sha256": "36e13164cea273912a87b55b1b115665f7451c2d8f21c64626923848770420b3",
        "label": "measured"},
       "command_line": "<sample>.1fq --threads 4 --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off [--run-screen] (non-host screen on)",
       "labels": "every number is measured, derived or projected; see *_label fields", "samples": {}}
tot = {"pairs": 0, "core_s": 0, "credited": 0, "wall": 0}
for s in ORDER:
    d = os.path.join(HERE, "runs", "single", s)
    if not os.path.exists(os.path.join(d, "run.log")): continue
    o = parse(d); pairs = o["pairs"]; M = pairs / 1e6
    load = o.get("nonhost_load_core_s_monitor")
    cred = o["core_s"] - load if load is not None else None
    rec = {"M_pairs": round(M, 2), "M_pairs_label": "measured (.1fq header)",
           "wall_s": o["wall_s"], "wall_label": "measured (/usr/bin/time); depends on node and co-tenancy, see node",
           "core_s": o["core_s"], "core_s_label": "measured (user+sys of the process tree; cgroup counter agrees within 0.5%)",
           "core_s_cgroup": o.get("cgroup_core_s"),
           "core_s_per_M": round(o["core_s"] / M, 2), "core_s_per_M_label": "derived",
           "nonhost_index_load_core_s": load, "nonhost_index_load_core_s_label": "measured (cgroup CPU between the 'Loading screening indices' and 'Loading host Bloom filter' log lines, 2 s resolution)",
           "nonhost_index_load_core_s_sec12_9": SEC129[s], "sec12_9_label": "from docs/gpu-engine/exploration.md 12.9 (other binary, other CPU)",
           "core_s_credited": round(cred, 1) if cred is not None else None,
           "core_s_credited_per_M": round(cred / M, 2) if cred is not None else None,
           "credited_label": "derived: core_s minus measured non-host index load (as if a persistent per-node service loaded the three indices once)",
           "max_rss_gb": round(o["max_rss_gb"], 1), "max_rss_label": "measured", "cgroup_peak_mem_gb": o.get("cgroup_peak_mem_gb"),
           "stage_wall_s_engine": {k: o[k] for k in ("fq_pass1_s", "refs_s", "star_s", "bam_index_s", "cb_refresh_s", "pileup_s", "export_s", "total_s") if o.get(k) is not None},
           "stage_windows_label": "measured; wall and core-s between marker lines (2 s resolution): STAR map+solo (incl. fq pass 1, references), sort, index+CB refresh+pileup, non-host index load, non-host classify+EM+writes (+BAM unlink), export incl. demux",
           "stage_windows": o.get("stage_windows"),
           "node": o["node"], "cpu_model": o["cpu_model"], "cpus_on_node": int(o["cpus_on_node"]), "cotenant_cpus_at_start": int(o["cotenant_cpus"]), "loadavg_at_start": o["loadavg"],
           "bam_records": o["bam_records"], "exit_ok": o["exit_ok"]}
    out["samples"][s] = rec
    if s in ORDER[:6]:
        tot["pairs"] += M; tot["core_s"] += o["core_s"]; tot["wall"] += o["wall_s"]
        if cred is not None: tot["credited"] += cred
out["pooled"] = {"M_pairs": round(tot["pairs"], 1), "core_s": round(tot["core_s"]), "core_s_per_M": round(tot["core_s"] / tot["pairs"], 2),
                 "core_s_credited": round(tot["credited"]), "core_s_credited_per_M": round(tot["credited"] / tot["pairs"], 2), "label": "derived (sums over the six samples)"}
# pooled over the six requested samples only
# packing
packs = {}
for pd in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    tag = os.path.basename(pd)
    if tag in ("single",) or tag.endswith("_stalled") or not os.path.exists(os.path.join(pd, "pack_wall.txt")): continue
    w = dict(l.strip().split("=") for l in open(os.path.join(pd, "pack_wall.txt")) if "=" in l)
    if "t1" not in w: continue
    inst = []
    for idir in sorted(glob.glob(os.path.join(pd, "i*_GSM*"))):
        o = parse(idir)
        inst.append({"sample": os.path.basename(idir).split("_", 1)[1], "M_pairs": round(o["pairs"] / 1e6, 2), "wall_s": o["wall_s"], "core_s": o["core_s"], "core_s_per_M": round(o["core_s"] / o["pairs"] * 1e6, 2),
                     "max_rss_gb": round(o["max_rss_gb"], 1), "exit_ok": o["exit_ok"]})
    mk = float(w["t1"]) - float(w["t0"]); P = sum(i["M_pairs"] for i in inst)
    mon = [l.split("\t") for l in open(os.path.join(pd, "monitor.tsv")).read().splitlines()[1:] if l.count("\t") >= 3]
    pk = max(float(r[1]) for r in mon) / 1024
    packs[tag] = {"instances": inst, "makespan_s": round(mk), "M_pairs_total": round(P, 1), "node_M_pairs_per_s": round(P / mk, 3), "samples_per_hour": round(len(inst) * 3600 / mk, 2),
                  "node_core_s_total": round(float(mon[-1][3]) - float(mon[0][3])), "cgroup_peak_mem_gb": round(pk, 0), "label": "measured (makespan includes the tail where fewer samples run)"}
    node = open(os.path.join(pd, "node.txt")).read().splitlines()
    packs[tag]["node_info"] = [l for l in node if l.startswith(("node=", "Model name", "CPU(s)", "memory.max"))]
out["packing"] = packs
json.dump(out, open(os.path.join(HERE, "cpu_baseline.json"), "w"), indent=1)
print(json.dumps({k: (v["core_s_per_M"], v["core_s_credited_per_M"], v["wall_s"]) for k, v in out["samples"].items()}), out["pooled"])
