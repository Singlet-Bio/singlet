#!/usr/bin/env python3
"""Collect bench/gpu/m13/runs/<SET>/ (pulled from $W/jobs/w11/m13/results/<SET>/) into ../results/m13.json.
Usage: python make_m13_json.py [SET ...]   (default: the samples found under runs/)"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "..", "results", "m13.json")

# section 12.9 (m10b, campaign build 28969b87, --threads 4, process CPU core-s) for the same samples, labelled as such
M10B = {"GSM4006843": {"pairs_m": 66.7, "load_indices": 33.2, "classify": 20.5, "em": 4.9},
        "GSM6470208": {"pairs_m": 121.8, "load_indices": 46.2, "classify": 77.2, "em": 24.7},
        "GSM5411066": {"pairs_m": 69.6, "load_indices": 28.6, "classify": 20.3, "em": 7.8}}
M10B_POOLED = {"load_indices_per_m": 0.32, "classify_per_m": 0.43, "em_per_m": 0.11, "nonhost_screen_share_of_W": 0.43}


def laps(path):
    d = {}
    if os.path.exists(path):
        for l in open(path, errors="replace"):
            m = re.match(r"\[m13t\] (\w+) wall_s=([\d.]+) cpu_s=([\d.]+)", l)
            if m:
                d[m.group(1)] = {"wall_s": float(m.group(2)), "cpu_s": float(m.group(3))}
    return d


def compare(path):
    txt = open(path, errors="replace").read() if os.path.exists(path) else ""
    ident = [l for l in txt.splitlines() if l.startswith("IDENTICAL")]
    diff = [l for l in txt.splitlines() if l.startswith("DIFFERENT") or "NO" in re.findall(r":(yes|NO)\b", l)]
    files = [l for l in txt.splitlines() if l.startswith("FILE ")]
    return {"dump_comparisons_identical": len(ident), "differences": diff, "files_compared": len(files),
            "all_identical": bool(ident) and not diff and all(":NO" not in l for l in files)}


def one(s):
    d = os.path.join(RUNS, s)
    g = json.load(open(os.path.join(d, "gpu.json")))
    pairs = None
    p = os.path.join(d, "n_pairs.txt")
    if os.path.exists(p):
        m = re.search(r"\.1fq: (\d+) reads", open(p).read())
        pairs = int(m.group(1)) if m else None
    cap, A, B = laps(os.path.join(d, "engine.log")), laps(os.path.join(d, "replayA.log")), laps(os.path.join(d, "replayB.log"))
    C, D = laps(os.path.join(d, "replayC.log")), laps(os.path.join(d, "replayD.log"))
    r = {"sample": s, "pairs": pairs, "gpu": g, "compare": compare(os.path.join(d, "compare.txt")),
         "host_timers_engine_capture_24thr": cap, "host_timers_replay_cpu_4thr": C, "host_timers_replay_gpu_results_4thr": D}
    if C and D and pairs:
        rem = C["load_indices"]["cpu_s"] + C["classify"]["cpu_s"] - D["classify"]["cpu_s"]
        r["host_core_s_removed"] = {
            "cpu_load_indices_s": C["load_indices"]["cpu_s"], "cpu_classify_s": C["classify"]["cpu_s"], "host_left_with_gpu_classify_s": D["classify"]["cpu_s"],
            "removed_total_s": rem, "removed_per_m_pairs": rem / (pairs / 1e6),
            "removed_classify_only_per_m_pairs": (C["classify"]["cpu_s"] - D["classify"]["cpu_s"]) / (pairs / 1e6),
            "removed_load_only_per_m_pairs": C["load_indices"]["cpu_s"] / (pairs / 1e6),
            "label": "this job's [m13t] process-CPU timers of nhscreen replays at 4 threads (campaign setting); replay D still loads the indexes on the host (dev expedient) and keeps Phase D, EM and writers"}
    if s in M10B:
        m = M10B[s]
        r["section_12_9_same_sample"] = {"label": "m10b timers (campaign build, core-s, from exploration.md 12.9)", **m,
                                         "load_per_m": m["load_indices"] / m["pairs_m"], "classify_per_m": m["classify"] / m["pairs_m"]}
    return r


def main():
    sets = sys.argv[1:] or sorted(x for x in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, x, "gpu.json")))
    res = {"milestone": "M13 (host step 2): non-host k-mer screen on the GPU", "cpu_truth": "prod/campaign c4a1be38",
           "section_12_9_pooled": M10B_POOLED, "samples": [one(s) for s in sets]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
