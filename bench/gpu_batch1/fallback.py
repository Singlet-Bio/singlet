#!/usr/bin/env python3
"""Fallback lines of a GPU run, with an explicit allow-list.

Usage: fallback.py RUN_DIR      (RUN_DIR holds run.log.ts and star_Log.out)
Prints one line per hit:  DISALLOWED<TAB>source<TAB>line   or   ALLOWED<TAB>reason<TAB>line
and last:  SUMMARY<TAB>n_disallowed<TAB>n_allowed
Also importable: scan(run_dir) -> (disallowed, allowed).
"""
import os
import re
import sys

# A line that matches one of these means the GPU path or the GPU pileup did not run as asked.
# The first nine are the patterns of the old gate (bench/gpu_device/jobs/gate.sbatch, fallbacks.txt).
DISALLOWED = [
    r"\[engine_gpu\] fallback",
    r"\[1fq-ingest\] not used",
    r"SINGLET_GPU_PILEUP ignored",
    r"E8: no shared memory",
    r"prefill wrote none",
    r"FATAL",
    r"FELL BACK",
    r"\[gpu_pileup\] (ERROR|not run|failed)",
    r"screening on the CPU",
    r"built without CUDA",
    r"no usable GPU",
    r"ENGINE_GPU: not used",
    r"per-base quality needed but unavailable",
]

# Lines that match a pattern above but are informational. Each entry: (regex, reason).
ALLOWED = [
    (r"\[1fq-ingest\] R11 exact CB histogram from the \.1fq \(STAR's prefill wrote none\)",
     "info: the pipeline rebuilt STAR's exact barcode histogram from the .1fq. The GPU path still ran. Identity is checked by the gate."),
    (r"Shared memory is not used",
     "info: STAR's Log.out line about genome shared memory, unrelated to the GPU engine."),
]

# star_Log.out: only its engine lines count (STAR prints many unrelated 'not used' lines).
STARLOG_ENGINE = re.compile(r"ENGINE_GPU|engine_gpu")
STARLOG_BAD = re.compile(r"not used|not built|FATAL|fallback", re.I)

_bad = [re.compile(p) for p in DISALLOWED]
_ok = [(re.compile(p), why) for p, why in ALLOWED]


def _strip_ts(l):
    return l[11:] if len(l) > 11 and l[:10].isdigit() and l[10] == " " else l


def _classify(line, src, bad, ok):
    for rx, why in _ok:
        if rx.search(line):
            ok.append((why, line[:300]))
            return
    if any(rx.search(line) for rx in _bad):
        bad.append((src, line[:300]))


def scan(run_dir):
    bad, ok = [], []
    p = os.path.join(run_dir, "run.log.ts")
    if os.path.exists(p):
        for l in open(p, errors="replace"):
            _classify(_strip_ts(l.rstrip("\n")), "run.log", bad, ok)
    p = os.path.join(run_dir, "star_Log.out")
    if os.path.exists(p):
        for l in open(p, errors="replace"):
            l = l.rstrip("\n")
            if "FATAL" in l or (STARLOG_ENGINE.search(l) and STARLOG_BAD.search(l)):
                _classify(l, "star_Log.out", bad, ok)
    return bad, ok


if __name__ == "__main__":
    b, o = scan(sys.argv[1])
    for src, l in b:
        print(f"DISALLOWED\t{src}\t{l}")
    for why, l in o:
        print(f"ALLOWED\t{why}\t{l}")
    print(f"SUMMARY\t{len(b)}\t{len(o)}")
