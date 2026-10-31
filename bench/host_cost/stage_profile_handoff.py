#!/usr/bin/env python3
"""Core-seconds per stage of the hand-off path (SINGLET_GPU_PILEUP=on, --engine gpu), from a scripts/engine_gpu/stage_profile.sbatch run.

usage: stage_profile_handoff.py RESULT_DIR M_READS [engine]      (engine defaults to gpu)

RESULT_DIR holds cpu_<e>.txt (wall clock, job cgroup CPU in usec, every second) and log_<e>.txt (the engine's log, every line
prefixed with the wall clock). STAR's own log lines (finished mapping, ENGINE_GPU: ...) carry no prefix there; stage_profile.sbatch
copies star_Log.out without timestamps, so the stages inside STAR come from the singlet log (STAR's stderr lines) where they appear,
and otherwise from starlog_<e>.txt (the same run with `ts`-style prefixes, as W11's job 791104 profile-791104/starlog_gpu.txt has).
Each marker closes the interval since the previous one; the core-s of an interval is the cgroup counter's increase over it.
Output: stage, wall s, core-s, average cores, core-s per M reads. Used to build docs/research-beta/host-cost-model.md (job 791104).
"""
import bisect
import re
import sys
import os

# (regex, name of the stage that ENDS at the first line matching it)
MARKERS = [
    (r"\[1fq-decode\] Done", "decode .1fq to FASTQ text (18 threads)"),
    (r"\[singlet\] STAR pid", "references, barcodes (pileup engine)"),
    (r"Finished loading the genome", "STAR start, genome load"),
    (r"ENGINE_GPU: cache filled", "GPU prefill (read R2, mapDistinct, host lane, hand-off build, inserts)"),
    (r"\.\.\.\.\. finished mapping", "STAR per-read pass (readLoad+clip, barcode, Solo record, summary replay)"),
    (r"Finished collapsing UMIs", "STAR Solo counting (serial)"),
    (r"\[gpu_pileup\] hand-off pass:", "pileup hand-off pass (serial, per read)"),
    (r"\[gpu_pileup\] \d+ reads, \d+ signatures", "prepend_handoff + build GPU input (serial)"),
    (r"\[gpu_pileup\] on:", "GPU pileup + apply + host pass (serial)"),
    (r"\[nonhost\] Loading screening", "pileup finish (molecule labelling, UMI collapse)"),
    (r"\[singlet\] Exporting feature matrices", "non-host screen, EM, writers"),
    (r"\[export\] Total", "export (matrices, QC, demux, writers)"),
]


def load_cpu(path):
    ts, us = [], []
    for line in open(path):
        p = line.split()
        if len(p) == 2:
            ts.append(float(p[0])); us.append(float(p[1]) / 1e6)
    return ts, us


def cpu_at(ts, us, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return us[0]
    if i >= len(ts):
        return us[-1]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return us[i - 1] + f * (us[i] - us[i - 1])


def main():
    d = sys.argv[1]; m = float(sys.argv[2]); eng = sys.argv[3] if len(sys.argv) > 3 else "gpu"
    ts, us = load_cpu(f"{d}/cpu_{eng}.txt")
    lines = []
    for f in (f"{d}/log_{eng}.txt", f"{d}/starlog_{eng}.txt"):
        if os.path.exists(f):
            for line in open(f, errors="replace"):
                p = line.split(" ", 1)
                try:
                    lines.append((float(p[0]), p[1] if len(p) > 1 else ""))
                except ValueError:
                    pass
    lines.sort(key=lambda x: x[0])
    t0 = lines[0][0]; t_end = lines[-1][0]
    seen = {}
    for t, txt in lines:
        for i, (rx, name) in enumerate(MARKERS):
            if i not in seen and re.search(rx, txt):
                seen[i] = t
                break
    bounds = [(t0, "start")] + [(seen[i], MARKERS[i][1]) for i in sorted(seen)] + [(t_end, None)]
    bounds = [bounds[0]] + sorted(bounds[1:-1], key=lambda x: x[0]) + [bounds[-1]]
    tot_c = cpu_at(ts, us, t_end) - cpu_at(ts, us, t0)
    print(f"{'stage':<78}{'wall s':>8}{'core-s':>9}{'cores':>7}{'per M':>7}")
    for (ta, _), (tb, name) in zip(bounds, bounds[1:]):
        if name is None:
            continue
        c = cpu_at(ts, us, tb) - cpu_at(ts, us, ta)
        w = tb - ta
        print(f"{name:<78}{w:8.1f}{c:9.1f}{(c / w if w > 0 else 0):7.1f}{c / m:7.2f}")
    print(f"{'total':<78}{t_end - t0:8.1f}{tot_c:9.1f}{tot_c / (t_end - t0):7.1f}{tot_c / m:7.2f}")


if __name__ == "__main__":
    main()
