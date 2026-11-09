#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Type HLA for a batch of finished runs on CPU nodes (the deferred route of `singlet --hla-defer`).

A run started with --hla-defer does the k-mer prefilter in the GPU pool and leaves <run>/private/hla/pending/: the candidate reads and a
manifest. The input reads may be deleted as soon as the pool finishes the sample. This script takes the batch's run directories (one per
line), submits `singlet hla --pending` as a Slurm array on CPU nodes, waits for it, and exits nonzero if any human run lacks a complete
private/hla/hla_summary.json afterwards. `singlet hla --pending` deletes pending/ once the summary has landed and keeps it on any failure,
so running this script again retries only the runs that did not land.

States of a run:
  landed          hla_summary.json is there and pending/ is gone
  pending         pending/manifest.json is there (also when an earlier job died after writing the summary)
  not_applicable  the genome is not human (the pipeline types HLA on human non-barnyard genomes only)
  missing         a human run with neither a summary nor a complete pending/

Exit status: 0 when every run is landed or not_applicable, 1 otherwise (each such run is named on stderr), 2 on a usage error.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

LANDED, PENDING, NOT_APPLICABLE, MISSING = "landed", "pending", "not_applicable", "missing"


def is_human_run(run):
    """The pipeline's rule (hla_t1k_for_run in singlet.cpp) on the genome the run recorded. An unreadable record counts as human, so that a
    damaged run is reported instead of skipped."""
    try:
        with open(os.path.join(run, "pileup_stats.json")) as f:
            genome = json.load(f)["provenance"]["genome_dir"]
    except (OSError, ValueError, KeyError, TypeError):
        return True
    return ("GRCh38" in genome or "hg38" in genome) and "GRCm39" not in genome and "mm39" not in genome


def run_state(run):
    hla = os.path.join(run, "private", "hla")
    if os.path.isfile(os.path.join(hla, "pending", "manifest.json")):
        return PENDING
    if os.path.isfile(os.path.join(hla, "hla_summary.json")) and not os.path.exists(os.path.join(hla, "pending")):
        return LANDED
    return NOT_APPLICABLE if not is_human_run(run) else MISSING


def read_runs(path):
    with open(path) as f:
        runs = [s.strip().rstrip("/") for s in f]
    return [r for r in runs if r and not r.startswith("#")]


def sbatch_command(args, list_file, n):
    """A blocking Slurm array, one task per run, each typing the run on the task's line of list_file."""
    task = ('RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {lst}); exec {bin} hla --pending "$RUN/private/hla/pending" --threads $SLURM_CPUS_PER_TASK'
            ).format(lst=shlex.quote(list_file), bin=shlex.quote(args.singlet))
    cmd = ["sbatch", "--wait", "--parsable", "--job-name=hla-pending", f"--array=0-{n - 1}%{args.array_limit}", f"--partition={args.partition}",
           f"--cpus-per-task={args.threads}", f"--mem={args.mem}", f"--time={args.time}", f"--nice={args.nice}",
           "--output=" + os.path.join(args.log_dir, "hla-pending-%A_%a.out")]
    if args.account:
        cmd.append(f"--account={args.account}")
    return cmd + ["--wrap", task]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", help="file with one finished run directory per line")
    ap.add_argument("--singlet", help="the singlet binary (required unless --check-only)")
    ap.add_argument("--log-dir", help="directory for the task logs and the run list (required unless --check-only)")
    ap.add_argument("--partition", default="short-cpu")
    ap.add_argument("--account", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem", default="32G")
    ap.add_argument("--time", default="03:59:00")
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--array-limit", type=int, default=32, help="tasks running at once")
    ap.add_argument("--retries", type=int, default=1, help="further attempts for the runs that did not land")
    ap.add_argument("--check-only", action="store_true", help="report the states and exit; submits nothing")
    args = ap.parse_args(argv)
    if not args.check_only and not (args.singlet and args.log_dir):
        ap.error("--singlet and --log-dir are required unless --check-only")

    runs = read_runs(args.runs)
    if not runs:
        print("HLA batch: the run list is empty", file=sys.stderr)
        return 1
    if not args.check_only:
        os.makedirs(args.log_dir, exist_ok=True)
        for attempt in range(args.retries + 1):
            todo = [r for r in runs if run_state(r) == PENDING]
            if not todo:
                break
            list_file = os.path.join(args.log_dir, f"hla-pending-{os.getpid()}-{attempt}.list")
            with open(list_file, "w") as f:
                f.write("\n".join(todo) + "\n")
            print(f"HLA batch: attempt {attempt + 1}, submitting {len(todo)} of {len(runs)} runs", flush=True)
            # the exit status of the array says little (a task may fail and a retry succeed): the states below decide
            subprocess.run(sbatch_command(args, list_file, len(todo)), check=False)

    states = {r: run_state(r) for r in runs}
    bad = [r for r in runs if states[r] in (PENDING, MISSING)]
    counts = {s: sum(v == s for v in states.values()) for s in (LANDED, NOT_APPLICABLE, PENDING, MISSING)}
    print("HLA batch: " + ", ".join(f"{v} {k}" for k, v in counts.items()), flush=True)
    for r in bad:
        print(f"HLA NOT LANDED ({states[r]}): {r}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
