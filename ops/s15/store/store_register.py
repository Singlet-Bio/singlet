#!/usr/bin/env python3
"""S15 benchmark store: register one finished reference-tool run.
    store_register.py DATASET SAMPLE TOOL VERSION SRC_DIR [--cmd 'command' | --cmd-file F] [--time-file time.txt] [--input-dir DIR]
Layout: $STORE/<dataset>/<sample>/<tool>/<version>/{out -> SRC_DIR, run.json}; one row appended to $STORE/index.tsv (locked).
Outputs only: SRC_DIR is linked, never copied; BAMs and reads are not part of the store (the chain deletes them)."""
import argparse, fcntl, hashlib, json, os, re, sys, time
STORE = os.environ.get("STORE", "/mnt/projects/debruinz_project/debruinz/singlet/benchmark-store")
ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("sample"); ap.add_argument("tool"); ap.add_argument("version"); ap.add_argument("src")
ap.add_argument("--cmd", default=""); ap.add_argument("--cmd-file", default=""); ap.add_argument("--time-file", default=""); ap.add_argument("--input-dir", default="")
a = ap.parse_args()
cmd = a.cmd or (open(a.cmd_file).read().strip() if a.cmd_file and os.path.exists(a.cmd_file) else "")
wall = rss = ""
tf = a.time_file or os.path.join(a.src, "time.txt")
if os.path.exists(tf):
    t = open(tf).read()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t); wall = m.group(1) if m else ""
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t); rss = str(int(m.group(1)) // 1024) + " MB" if m else ""
inputs = {}
if a.input_dir and os.path.isdir(a.input_dir):
    rj = os.path.join(a.input_dir, "runs.json")
    if os.path.exists(rj): inputs["runs.json_sha256"] = hashlib.sha256(open(rj, "rb").read()).hexdigest(); inputs["runs"] = json.load(open(rj)).get("runs")
    sd = os.path.join(a.input_dir, "sra")
    if os.path.isdir(sd): inputs["sra_bytes"] = {f: os.path.getsize(os.path.join(sd, f)) for f in sorted(os.listdir(sd))}
d = os.path.join(STORE, a.dataset, a.sample, a.tool, a.version); os.makedirs(d, exist_ok=True)
link = os.path.join(d, "out")
if os.path.islink(link) or os.path.exists(link): os.remove(link)
os.symlink(os.path.abspath(a.src), link)
rec = {"dataset": a.dataset, "sample": a.sample, "tool": a.tool, "version": a.version, "command": cmd, "container_or_env": os.environ.get("S15_ENV", ""),
       "parameters": "see command", "job_id": os.environ.get("SLURM_JOB_ID", ""), "wall": wall, "peak_rss": rss, "inputs": inputs,
       "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "outputs": os.path.realpath(a.src)}
json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)
with open(os.path.join(STORE, "index.tsv"), "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    if f.tell() == 0: f.write("dataset\tsample\ttool\tversion\tjob_id\twall\tpeak_rss\tfinished\tpath\n")
    f.write("\t".join([a.dataset, a.sample, a.tool, a.version, rec["job_id"], wall, rss, rec["finished"], d]) + "\n")
print("registered", d)
