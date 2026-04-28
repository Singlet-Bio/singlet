#!/usr/bin/env python3
"""
smoke_test_corpus.py — Run M1 encode+decode round-trip on all corpus samples.
Runs on a compute node via SSH. Designed to be called by the singlify-perf agent.

Usage:
    # Run all untested samples (skips confirmed_working + passed):
    python3 smoke_test_corpus.py --node c006 [--dry-run] [--ids C02,C03,C04]

    # Run a single sample:
    python3 smoke_test_corpus.py --node c006 --ids C02

Output: Updates corpus.json status + writes per-sample logs to /dev/shm/corpus_test/
        Prints pass/fail summary. Exit code 0 if all pass, 1 if any fail.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CORPUS_PATH = SCRIPT_DIR / "corpus.json"
CP_SCRIPT   = SCRIPT_DIR / "agent_checkpoint.py"
SINGLIFY    = "/mnt/home/debruinz/Singlet-AI/singlify/build/singlify"
TMPDIR      = "/dev/shm/corpus_test"

ENV_SETUP = """\
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
export PKG_CONFIG_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib/pkgconfig
"""


def ssh_run(node, cmd, timeout=300):
    """Run a command on the given compute node via SSH."""
    full = f"ssh {node} '{ENV_SETUP}\\n{cmd}'"
    result = subprocess.run(
        ["ssh", node, f"bash -c '{ENV_SETUP}\n{cmd}'"],
        capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout + result.stderr


def encode_decode_test(node, sample, dry_run=False):
    """Run encode + decode --verify on one corpus sample. Returns (pass, log)."""
    sid = sample["id"]
    srr = sample["srr_primary"]
    proto = sample["protocol"]
    out_1fq = f"{TMPDIR}/{sid}_{proto}.1fq"

    encode_cmd = (
        f"mkdir -p {TMPDIR} && "
        f"{SINGLIFY} download {srr} -o {out_1fq} --quality binned --verbose 2>&1"
    )
    decode_cmd = (
        f"{SINGLIFY} decode {out_1fq} --r1 /dev/null --r2 /dev/null --verify 2>&1 && "
        f"echo 'VERIFY_PASS' || echo 'VERIFY_FAIL'"
    )
    cleanup_cmd = f"rm -f {out_1fq}"

    if dry_run:
        print(f"  [DRY RUN] {sid}: would run encode+decode on {srr}")
        return True, "dry_run"

    print(f"  [{sid}] Encoding {srr} → {out_1fq} ...")
    rc, log = ssh_run(node, encode_cmd, timeout=600)
    if rc != 0 or "error" in log.lower()[:200]:
        ssh_run(node, cleanup_cmd, timeout=10)
        return False, f"encode_failed\n{log[:2000]}"

    print(f"  [{sid}] Decoding + verifying ...")
    rc2, log2 = ssh_run(node, decode_cmd, timeout=120)
    ssh_run(node, cleanup_cmd, timeout=10)
    
    passed = "VERIFY_PASS" in log2
    return passed, log + "\n" + log2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="c006")
    parser.add_argument("--ids", help="Comma-separated corpus IDs, e.g. C02,C03")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    samples = corpus["corpus"]
    if args.ids:
        id_set = set(args.ids.split(","))
        samples = [s for s in samples if s["id"] in id_set]
    else:
        # Skip already confirmed/passed
        samples = [s for s in samples
                   if s["status"] not in ("confirmed_working", "encode_pass",
                                          "decode_pass", "pipeline_pass")]

    if not samples:
        print("No samples to test.")
        return 0

    print(f"Testing {len(samples)} samples on {args.node} ({'DRY RUN' if args.dry_run else 'LIVE'})")
    print("-" * 60)

    passed_count = 0
    failed_count = 0

    for sample in samples:
        sid = sample["id"]
        proto = sample["protocol"]
        srr = sample["srr_primary"]
        print(f"\n[{sid}] {proto} — {srr}")
        
        passed, log = encode_decode_test(args.node, sample, dry_run=args.dry_run)
        
        status = "encode_pass" if passed else "encode_fail"
        if not args.dry_run:
            subprocess.run([
                sys.executable, str(CP_SCRIPT), "corpus-update",
                "--id", sid, "--status", status
            ], check=False)
            # Update corpus.json status in-place
            for s in corpus["corpus"]:
                if s["id"] == sid:
                    s["status"] = status
            with open(CORPUS_PATH, "w") as f:
                json.dump(corpus, f, indent=2)

        if passed:
            print(f"  ✅ PASS")
            passed_count += 1
        else:
            print(f"  ❌ FAIL")
            print(f"     {log[:500]}")
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed_count} PASS, {failed_count} FAIL out of {len(samples)} tested")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
