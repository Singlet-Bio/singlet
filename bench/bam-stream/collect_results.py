#!/usr/bin/env python3
"""Assemble bench/bam-stream/results.json from the run directories on Clipper (runs in a Slurm job; no data is read, only
stages.json / tally / timing files and out_equiv.py's file-by-file comparison of two runs' output directories)."""
import glob, json, os, re, subprocess, sys
W = "/mnt/projects/debruinz_project/debruinz/singlet"; B = f"{W}/bam-stream"
res = {"binaries": {}, "baseline": {}, "p2_stream": {}, "p3_cb": {}, "p4": {}, "identity": {}, "atlas_bound": {}}
for sub in ("", "_fa", "_memsort", "_stream", "_s0", "_s1", "_b0", "_b1", "_t1"):
    for f in sorted(glob.glob(f"{B}/runs{sub}/*/t*_r*/stages.json")):
        set_, tr = f.split(f"/runs{sub}/")[1].split("/")[:2]
        d = json.load(open(f)); d.pop("dir", None)
        nodef = open(os.path.dirname(f) + "/node.txt").read().split("\n") if os.path.exists(os.path.dirname(f) + "/node.txt") else []
        d["binary_sha256"] = next((l.split()[0] for l in nodef if re.match(r"^[0-9a-f]{64} ", l)), None)
        d["exit_error"] = "exit " in open(os.path.dirname(f) + "/run.log", errors="replace").read()[-300:]
        res["baseline"].setdefault({"": "sorted_disk", "_fa": "sorted_disk_28969b8", "_memsort": "sorted_memory", "_stream": "existing_stream_path", "_s0": "proto_binary_sorted", "_s1": "proto_binary_stream_own_barcodes", "_b0": "proto_binary_sorted_fixed_barcodes", "_b1": "proto_binary_stream_fixed_barcodes", "_t1": "proto_binary_sorted_1thread_fixed_barcodes"}[sub], {}).setdefault(set_, {})[tr] = d
for p in glob.glob(f"{B}/p2/*/"):
    s = os.path.basename(p.rstrip("/")); o = {}
    for m in ("serial", "parallel"):
        if os.path.exists(f"{p}run_{m}.log"):
            t = re.search(r"\[singlet\] Pileup: (\d+) reads, ([\d.]+)s", open(f"{p}run_{m}.log", errors="replace").read())
            if t: o[m] = {"reads": int(t.group(1)), "pileup_s": float(t.group(2))}
    res["p2_stream"][s] = o
for p in glob.glob(f"{B}/p3/*/"):
    s = os.path.basename(p.rstrip("/"))
    if os.path.exists(p + "tally.txt"):
        res["p3_cb"][s] = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", open(p + "tally.txt").read()))
        if os.path.exists(p + "Barcodes.stats"): res["p3_cb"][s]["barcodes_stats"] = open(p + "Barcodes.stats").read().split("\n")
for p in glob.glob(f"{B}/p4/*/"):
    s = os.path.basename(p.rstrip("/"))
    if os.path.exists(p + "times.txt"): res["p4"][s] = open(p + "times.txt").read().strip().split("\n")
def equiv(a, b):
    r = subprocess.run([sys.executable, f"{W}/jobs/deep-perf/out_equiv.py", a, b], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().split("\n")[-40:]
for s in ("GSM4006843", "GSM7036403", "GSM7901264", "GSM5411066", "GSM6470208", "GSM7656149"):
    pairs = {"t4_r1_vs_t4_r2": (f"{B}/runs/{s}/t4_r1/out", f"{B}/runs/{s}/t4_r2/out"),
             "t4_r1_vs_t16_r1": (f"{B}/runs/{s}/t4_r1/out", f"{B}/runs/{s}/t16_r1/out"),
             "t16_r1_vs_t16_r2": (f"{B}/runs/{s}/t16_r1/out", f"{B}/runs/{s}/t16_r2/out"),
             "sorted_disk_vs_memsort": (f"{B}/runs/{s}/t4_r1/out", f"{B}/runs_memsort/{s}/t4_r1/out"),
             "campaign_02aa9b9_vs_28969b8": (f"{B}/runs/{s}/t4_r1/out", f"{B}/runs_fa/{s}/t4_r1/out")}
    for k, (a, b) in pairs.items():
        if os.path.isdir(a) and os.path.isdir(b): res["identity"].setdefault(s, {})[k] = equiv(a, b)
# identity logs of the prototype comparisons (out_equiv.py output kept in the job logs)
res["identity_proto"] = {}
for jid, label in (("789498", "sorted_vs_stream_own_barcodes"), ("789542", "sorted_vs_stream_fixed_barcodes"), ("789731", "sorted_1thread_vs_stream_fixed_barcodes"), ("789732", "big_sets_baseline_runs_vs_stream_fixed_barcodes")):
    lf = f"{W}/logs/bs-eq-{jid}.out"
    if os.path.exists(lf):
        res["identity_proto"][label] = [l[:300] for l in open(lf, errors="replace").read().splitlines() if l.startswith(("########", "files:", "RESULT", "DIFF", "only in"))]
for f in sorted(glob.glob(f"{B}/results/atlas_bound_T*.json")):
    res["atlas_bound"][os.path.basename(f)[:-5]] = json.load(open(f))
json.dump(res, open(f"{B}/results.json", "w"), indent=1)
print("wrote", f"{B}/results.json", {k: len(v) for k, v in res.items()})
