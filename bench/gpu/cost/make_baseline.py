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
           "node": o["node"], "cpu_model": o["cpu_model"], "cpus_on_node": int(o["cpus_on_node"]), "cotenant_cpus_at_start": int(o["cotenant_cpus"]) - 8, "cotenant_note": "squeue count of running CPUs on the node at job start minus this job's own 8", "loadavg_at_start": o["loadavg"],
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
    w["t1"] = max(float(l.split("t1=")[1]) for f in glob.glob(os.path.join(pd, "i*_*", "wall.txt")) for l in open(f) if "t1=" in l)
    inst = []
    for idir in sorted(d for d in glob.glob(os.path.join(pd, "i*_*")) if os.path.isdir(d)):
        o = parse(idir)
        inst.append({"sample": os.path.basename(idir).split("_", 1)[1], "M_pairs": round(o["pairs"] / 1e6, 2), "wall_s": o["wall_s"], "core_s": o["core_s"], "core_s_per_M": round(o["core_s"] / o["pairs"] * 1e6, 2),
                     "max_rss_gb": round(o["max_rss_gb"], 1), "exit_ok": o["exit_ok"]})
    mk = float(w["t1"]) - float(w["t0"]); P = sum(i["M_pairs"] for i in inst)
    mon = [l.split("\t") for l in open(os.path.join(pd, "monitor.tsv")).read().splitlines()[1:] if l.count("\t") >= 3]
    pk = max(float(r[1]) for r in mon) / 1024
    ncore = float(mon[-1][3]) - float(mon[0][3])
    t_end = w["t1"]; ts = [float(r[0]) for r in mon]; cs = [float(r[3]) for r in mon]
    packs[tag] = {"instances": inst, "makespan_s": round(mk), "M_pairs_total": round(P, 1), "node_M_pairs_per_s": round(P / mk, 3), "samples_per_hour": round(len(inst) * 3600 / mk, 2),
                  "node_core_s_through_makespan": round(max(c for t, c in zip(ts, cs) if t <= t_end + 5) - cs[0]), "mean_cores_busy": round((max(c for t, c in zip(ts, cs) if t <= t_end + 5) - cs[0]) / mk, 1), "cgroup_peak_mem_gb": round(pk, 0), "label": "measured (makespan includes the tail where fewer samples run)"}
    node = open(os.path.join(pd, "node.txt")).read().splitlines()
    packs[tag]["node_info"] = [l for l in node if l.startswith(("node=", "Model name", "CPU(s)", "memory.max"))]
out["packing"] = packs
json.dump(out, open(os.path.join(HERE, "cpu_baseline.json"), "w"), indent=1)
print(json.dumps({k: (v["core_s_per_M"], v["core_s_credited_per_M"], v["wall_s"]) for k, v in out["samples"].items()}), out["pooled"])

# ---- node block (derived / projected from the one completed packing run) ----
d = json.load(open(os.path.join(HERE, "cpu_baseline.json")))
h = d["packing"]["hom4_t4"]
slot_rate = h["M_pairs_total"] / len(h["instances"]) / h["makespan_s"]            # M pairs/s per concurrent sample
ratio = h["node_core_s_through_makespan"] / h["M_pairs_total"] / d["samples"]["pbmc1k_v3"]["core_s_per_M"]
mix_core_s_per_M = d["pooled"]["core_s_per_M"] * ratio
N_RAM = int(0.9 * 488 // max(i["max_rss_gb"] for i in h["instances"]))                  # concurrent samples that fit in MemAvailable
mixM = d["pooled"]["M_pairs"]
mix_s = mixM / (slot_rate * N_RAM)
d["node"] = {
 "node_type": "c100, 2 x Xeon 6730P, 64 cores (62 allocatable), 512 GB", "cores": 64, "cpu_model": "Intel(R) Xeon(R) 6730P", "label": "measured (lscpu in job 789778)",
 "measured_4_concurrent_pbmc1k_v3_t4": {"node_M_pairs_per_s": h["node_M_pairs_per_s"], "samples_per_hour_of_66M_pair_sample": h["samples_per_hour"], "mean_cores_busy": h["mean_cores_busy"],
     "wall_s_each": [round(i["wall_s"]) for i in h["instances"]], "core_s_per_M_each": [i["core_s_per_M"] for i in h["instances"]], "job": 789778, "label": "measured"},
 "mix_core_s_per_M_on_this_node": round(mix_core_s_per_M, 1), "mix_core_s_per_M_label": "derived: six-sample pooled core-s/M (measured on Gold 6230R/6248 nodes) x (pbmc1k_v3 packed on 6730P / pbmc1k_v3 single on Gold 6248 = %.3f); mixes CPU model with co-tenancy, cannot be separated" % ratio,
 "projected_concurrency": N_RAM, "projected_concurrency_basis": "90%% of MemAvailable (488 GB) / max RSS %.0f GB; untested above 4 (an 8-concurrent attempt on c010 hung in the BAM sort and drained the node)" % max(i["max_rss_gb"] for i in h["instances"]),
 "projected_node_M_pairs_per_s": round(slot_rate * N_RAM, 3), "projected_samples_per_hour_six_sample_mix": round(6 * 3600 / mix_s, 1), "projected_mix_makespan_s": round(mix_s),
 "projected_label": "projected: per-slot rate measured at 4 concurrent, assumed unchanged up to the RAM limit (cores busy would be %.0f of 64)" % (h["mean_cores_busy"] / 4 * N_RAM),
 "six_sample_mix_measured": "not measured (pack job 789779 cancelled while pending; no idle node for hours)",
 "threads_choice": "--threads 4 kept (campaign setting); thread-count sweep not measured. Mean busy cores per 4-thread sample = %.1f (STAR 4 threads, pileup 3-4, then serial export), so ~3 cores per sample-in-flight; RAM (~54 GB RSS each), not cores, bounds concurrency." % (h["mean_cores_busy"] / 4)}
d["incidents"] = ["hom8_t4 (job 789621, 8 x GSM4006843, c010, exclusive): all 8 stuck at 'started sorting BAM' for 20 min at ~0 CPU; scancel failed ('Kill task failed') and Slurm drained c010. Cause unknown; data not used. Needs admin undrain.",
                  "hom4_t4 (job 789778): complete (4 runs exit 0); my stall watchdog fired after the runs finished and cancelled the job (state CANCELLED), so pack_wall.txt has no t1: makespan taken from the per-instance wall stamps.",
                  "run_single array 789532: tasks 0-5 show FAILED (exit 1) after the engine finished with exit 0 and all outputs written (script bug in the last lines; fixed in run_single.sbatch with an EXIT trap). Data complete, not rerun.",
                  "$W/data/bam-stream/ was deleted by another session after GSM4006843, GSM5411066, GSM7901264 had been measured (their runs are kept); pbmc1k_v3 and nuclei2k_v2 singles added from pipeline-benchmark/data."]
d["core_hours"] = {"singles_789532": 23.0, "singles_789777": 5.5, "hom8_t4_789621_stalled": 25.3, "hom4_t4_789778": 16.6, "total": 70.4, "label": "derived: allocated cores x elapsed"}
d["jobs"] = {"singles": [789532, 789777], "hom8_t4_stalled": 789621, "hom4_t4": 789778, "cancelled_unrun": [789622, 789623, 789693, 789758, 789759, 789779]}
json.dump(d, open(os.path.join(HERE, "cpu_baseline.json"), "w"), indent=1)
print(json.dumps(d["node"], indent=1)[:2500])
