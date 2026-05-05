#!/usr/bin/env python3
"""Unified grab-based batch system for the Singlet Bio reprocessing pipeline.

Each SLURM array task calls this script to atomically claim a batch of N samples
from the processing catalog. Race conditions are prevented via fcntl file locking.

The --phase flag selects which eligibility filter to apply:
  1   Human droplet RNA (10x, Drop-seq, InDrop — not ATAC/multiome)
  2a  Human multiome GEX
  2b  Human CITE-seq GEX
  2c  Human reclassified droplet
  2d  Human ambiguous (protocol unknown, needs auto-detect)
  3   Human screen-flagged recovery (SRA fallback)
  4a  Mouse droplet RNA
  4b  Other model organisms

All phases share one claim directory and completed-GSM sidecar. The reconcile
script precomputes per-phase eligibility columns (eligible_1, eligible_2a, etc.)
for fast filtering. If the column is missing, eligibility is computed inline.

Usage:
    python grab_batch.py --phase 1 --batch-size 15
    python grab_batch.py --phase 1 --batch-size 15 --output out.csv
    python grab_batch.py --phase 1 --mark-done CLAIM_ID
    python grab_batch.py --phase 1 --mark-failed CLAIM_ID
    python grab_batch.py --phase 1 --report-gsms CLAIM_ID GSM1,GSM2
    python grab_batch.py --status
"""

from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_BASE = Path(os.environ.get("SCGEO_BASE", "/mnt/projects/debruinz_project/cellarium"))
PIPELINE_DIR = Path(os.environ.get("SCGEO_PIPELINE_DIR", str(_BASE / "pipeline")))
CATALOG_DIR = Path(os.environ.get("SCGEO_CATALOG_DIR", str(_BASE / "catalog")))
CLAIMS_DIR = PIPELINE_DIR / "claims"
LEDGER_PATH = CLAIMS_DIR / "ledger.tsv"
LOCK_PATH = CLAIMS_DIR / "ledger.lock"
COMPLETED_PATH = CLAIMS_DIR / "completed_gsms.txt"

BATCH_COLUMNS = [
    "gsm_id", "gse_id", "organism", "protocol_inferred", "protocol_confidence",
    "ena_fastq_r1", "ena_fastq_r2", "srr_accessions", "read_count",
]

# ── Permanent failures (never retry) ──────────────────────────────────

PERMANENT_FAILURES = {
    "done", "done_qc_warn",
    "skip_plate_based", "skip_vdj",
    "skip_percell", "skip_lowconf", "skip_no_fastq",
    "fail_no_r2", "fail_qc_few_cells",
    "fail_qc_other", "fail_low_mapping", "fail_no_mapping",
    "fail_simpleaf_permit",
}

# ── Phase-specific eligibility filters ────────────────────────────────


def _curator_cleared(pc: pd.DataFrame) -> pd.Series:
    """Samples explicitly cleared by the protocol-classifier agent."""
    cs = pc["curator_status"].fillna("") if "curator_status" in pc.columns else pd.Series("", index=pc.index)
    return cs == "cleared"


def _curator_triaged(pc: pd.DataFrame) -> pd.Series:
    """Samples marked triaged by the protocol-classifier agent."""
    cs = pc["curator_status"].fillna("") if "curator_status" in pc.columns else pd.Series("", index=pc.index)
    return cs == "triaged"


def _curator_not_blocked(pc: pd.DataFrame) -> pd.Series:
    """Samples NOT blocked by the protocol-classifier agent."""
    cs = pc["curator_status"].fillna("") if "curator_status" in pc.columns else pd.Series("", index=pc.index)
    return ~cs.str.startswith("blocked")


def _is_eligible_phase1(pc: pd.DataFrame) -> pd.Series:
    """Human droplet/single-cell RNA (not ATAC/multiome)."""
    human = pc["organism"].str.contains("Homo sapiens", na=False)
    # Exclude multi-species samples (e.g. "Homo sapiens; Mus musculus") —
    # these map to human-only index and get 0-2% mapping rate
    not_multi = ~pc["organism"].str.contains(";", na=False)
    proto = pc["protocol_inferred"].str.lower().fillna("")
    # All supported single-cell protocols (droplet + combinatorial indexing)
    supported = proto.str.contains(
        "10x|drop|indrop|scirna|parse|splitseq|bd_rhapsody|citeseq|seqwell|ddseq|dnbelab",
        regex=True,
    )
    not_atac = ~proto.str.contains("atac|multiome", regex=True)
    prod = pc["production_ready"].fillna(False)
    not_perm = ~pc["processing_status"].isin(PERMANENT_FAILURES)
    # Exclude low-confidence protocol assignments (10x_suspect etc.) —
    # these have ~90% failure rate and waste compute
    not_low_conf = pc["protocol_confidence"].str.lower().fillna("") != "low"
    # Exclude samples flagged by screening rules (non-GEX titles, VDJ/ADT read lengths)
    not_screened = ~pc["screen_any_flag"].fillna(False)
    # Require curator clearance — samples must be classified by protocol-classifier agent
    cleared = _curator_cleared(pc)
    return human & not_multi & supported & not_atac & prod & not_perm & not_low_conf & not_screened & cleared


def _is_eligible_phase2a(pc: pd.DataFrame) -> pd.Series:
    """Human multiome GEX."""
    human = pc["organism"].str.contains("Homo sapiens", na=False)
    multiome = pc["target_assay"] == "multiome" if "target_assay" in pc.columns else pd.Series(False, index=pc.index)
    not_done = ~pc["processing_status"].isin({"done", "done_qc_warn"})
    not_skip = ~pc["processing_status"].str.startswith("skip_")
    not_screen = ~pc["screen_any_flag"].fillna(False)
    not_perm = ~pc["processing_status"].isin(PERMANENT_FAILURES)
    return human & multiome & not_done & not_skip & not_screen & not_perm


def _is_eligible_phase2b(pc: pd.DataFrame) -> pd.Series:
    """Human CITE-seq GEX."""
    human = pc["organism"].str.contains("Homo sapiens", na=False)
    cite = pc["target_assay"] == "cite" if "target_assay" in pc.columns else pd.Series(False, index=pc.index)
    not_done = ~pc["processing_status"].isin({"done", "done_qc_warn"})
    not_skip = ~pc["processing_status"].str.startswith("skip_")
    not_screen = ~pc["screen_any_flag"].fillna(False)
    return human & cite & not_done & not_skip & not_screen


def _is_eligible_phase2c(pc: pd.DataFrame) -> pd.Series:
    """Human reclassified droplet (skip_reclassified, production_ready, scrna)."""
    human = pc["organism"].str.contains("Homo sapiens", na=False)
    reclass = pc["processing_status"] == "skip_reclassified"
    prod = pc["production_ready"].fillna(False)
    scrna = pc["target_assay"] == "scrna" if "target_assay" in pc.columns else pd.Series(True, index=pc.index)
    proto = pc["protocol_inferred"].str.lower().fillna("")
    not_atac = ~proto.str.contains("atac|multiome", regex=True)
    return human & reclass & prod & scrna & not_atac


def _is_eligible_phase2d(pc: pd.DataFrame) -> pd.Series:
    """Human ambiguous (unknown protocol, needs auto-detect)."""
    human = pc["organism"].str.contains("Homo sapiens", na=False)
    ambig = pc["target_assay"] == "ambiguous" if "target_assay" in pc.columns else pd.Series(False, index=pc.index)
    pending = pc["processing_status"] == "pending"
    not_screen = ~pc["screen_any_flag"].fillna(False)
    return human & ambig & pending & not_screen


def _is_eligible_phase3(pc: pd.DataFrame) -> pd.Series:
    """Human screen-flagged recovery (SRA fallback)."""
    human = pc["organism"].str.contains("Homo sapiens", na=False)
    scrna = pc["target_assay"] == "scrna" if "target_assay" in pc.columns else pd.Series(True, index=pc.index)
    pending = pc["processing_status"] == "pending"
    screened = pc["screen_any_flag"].fillna(False)
    # Only single-flag samples (highest recovery probability)
    flag_cols = [c for c in pc.columns if c.startswith("screen_") and c != "screen_any_flag"]
    if flag_cols:
        n_flags = pc[flag_cols].fillna(False).sum(axis=1)
        single_flag = n_flags == 1
    else:
        single_flag = pd.Series(True, index=pc.index)
    return human & scrna & pending & screened & single_flag


def _is_eligible_phase4a(pc: pd.DataFrame) -> pd.Series:
    """Mouse droplet RNA — hardened to match Phase 1 rigor."""
    mouse = pc["organism"].str.contains("Mus musculus", na=False)
    not_human = ~pc["organism"].str.contains("Homo sapiens", na=False)
    # Exclude multi-species samples (e.g. "Mus musculus; Homo sapiens")
    not_multi = ~pc["organism"].str.contains(";", na=False)
    scrna = pc["target_assay"] == "scrna" if "target_assay" in pc.columns else pd.Series(True, index=pc.index)
    proto = pc["protocol_inferred"].str.lower().fillna("")
    # All supported single-cell protocols (droplet + combinatorial indexing)
    supported = proto.str.contains(
        "10x|drop|indrop|scirna|parse|splitseq|bd_rhapsody|citeseq|seqwell|ddseq|dnbelab",
        regex=True,
    )
    not_atac = ~proto.str.contains("atac|multiome", regex=True)
    not_done = ~pc["processing_status"].isin({"done", "done_qc_warn"})
    not_skip = ~pc["processing_status"].str.startswith("skip_")
    not_screen = ~pc["screen_any_flag"].fillna(False)
    not_perm = ~pc["processing_status"].isin(PERMANENT_FAILURES)
    # Exclude low-confidence protocol assignments (10x_suspect etc.)
    not_low_conf = pc["protocol_confidence"].str.lower().fillna("") != "low"
    # Require curator clearance — samples must be classified by protocol-classifier agent
    cleared = _curator_cleared(pc)
    # Also allow triaged samples (lower priority but processable)
    triaged = _curator_triaged(pc)
    curator_ok = cleared | triaged
    return mouse & not_human & not_multi & scrna & supported & not_atac & not_done & not_skip & not_screen & not_perm & not_low_conf & curator_ok


def _is_eligible_phase4b(pc: pd.DataFrame) -> pd.Series:
    """Other model organisms — droplet RNA."""
    not_human = ~pc["organism"].str.contains("Homo sapiens", na=False)
    not_mouse_only = ~(pc["organism"].str.contains("Mus musculus", na=False) & not_human)
    scrna = pc["target_assay"] == "scrna" if "target_assay" in pc.columns else pd.Series(True, index=pc.index)
    not_done = ~pc["processing_status"].isin({"done", "done_qc_warn"})
    not_skip = ~pc["processing_status"].str.startswith("skip_")
    not_screen = ~pc["screen_any_flag"].fillna(False)
    not_perm = ~pc["processing_status"].isin(PERMANENT_FAILURES)
    not_unsupported = ~pc["screen_unsupported_organism"].fillna(False)
    return not_human & not_mouse_only & scrna & not_done & not_skip & not_screen & not_perm & not_unsupported


PHASE_FILTERS = {
    "1": _is_eligible_phase1,
    "2a": _is_eligible_phase2a,
    "2b": _is_eligible_phase2b,
    "2c": _is_eligible_phase2c,
    "2d": _is_eligible_phase2d,
    "3": _is_eligible_phase3,
    "4a": _is_eligible_phase4a,
    "4b": _is_eligible_phase4b,
}

PHASE_NAMES = {
    "1": "Human droplet RNA",
    "2a": "Human multiome GEX",
    "2b": "Human CITE-seq GEX",
    "2c": "Human reclassified droplet",
    "2d": "Human ambiguous (auto-detect)",
    "3": "Human screen-flagged recovery",
    "4a": "Mouse droplet RNA",
    "4b": "Other model organisms",
}

# ── Core logic ────────────────────────────────────────────────────────

RESULTS_DIR = PIPELINE_DIR / "results"


def _find_bad_gses(min_failures: int = 3) -> set[str]:
    """Identify GSEs with 100% failure rate in result CSVs.

    A GSE is 'bad' if it has ≥min_failures failures and 0 successes.
    These are deprioritized (not excluded) to avoid wasting workers.
    """
    gse_success = collections.Counter()
    gse_fail = collections.Counter()
    for f in glob.glob(str(RESULTS_DIR / "results_*.csv")):
        try:
            for row in csv.DictReader(open(f)):
                gse = row.get("gse_id", "")
                if not gse:
                    continue
                if row["status"] in ("success", "qc_warn"):
                    gse_success[gse] += 1
                elif row["status"] == "failed":
                    gse_fail[gse] += 1
        except Exception:
            continue
    return {gse for gse, n in gse_fail.items()
            if n >= min_failures and gse_success[gse] == 0}


def _load_eligible_gsms(phase: str) -> pd.DataFrame:
    """Load the processing catalog, filter to phase-eligible samples.

    Uses the precomputed eligible_{phase} column if present;
    otherwise computes eligibility inline as a fallback.
    """
    pc = pd.read_parquet(CATALOG_DIR / "processing_catalog.parquet")

    col = f"eligible_{phase}"
    if col in pc.columns:
        eligible = pc[pc[col] == True]
    else:
        if phase not in PHASE_FILTERS:
            print(f"ERROR: Unknown phase '{phase}'. Valid: {', '.join(sorted(PHASE_FILTERS))}", file=sys.stderr)
            sys.exit(1)
        mask = PHASE_FILTERS[phase](pc)
        eligible = pc[mask]

    # Priority tiers: process high-success-rate protocols first
    # Tier 0: indrop (85% success), dropseq (73% success)
    # Tier 1: high-confidence 10xv3 (25% success)
    # Tier 2: everything else (10xv2-high, citeseq, scirna, bd_rhapsody, ...)
    # Tier 3: medium-confidence 10xv2 (18% success), citeseq (untested, needs chemistry fix)
    # Tier 4: 10xv3_5prime ANY confidence (6.8% high, 4.3% medium — mostly VDJ/feature libraries)
    # Tier 5: samples from GSEs with 100% failure rate (≥3 failures, 0 successes)
    proto = eligible["protocol_inferred"].str.lower().fillna("")
    conf = eligible["protocol_confidence"].str.lower().fillna("")
    priority = pd.Series(2, index=eligible.index)  # default tier 2
    priority[proto.isin(["indrop", "dropseq"])] = 0
    priority[(proto == "10xv3") & (conf == "high")] = 1
    priority[(proto == "10xv2") & (conf == "medium")] = 3
    priority[proto == "10xv3_5prime"] = 4

    # Demote samples from 100%-failure GSEs to lowest tier
    bad_gses = _find_bad_gses()
    if bad_gses:
        in_bad_gse = eligible["gse_id"].isin(bad_gses)
        priority[in_bad_gse] = 5
        n_demoted = in_bad_gse.sum()
        if n_demoted:
            print(f"Demoted {n_demoted} samples from {len(bad_gses)} bad GSEs to tier 5", file=sys.stderr)

    # Shuffle within each tier to avoid convoy effects
    eligible = eligible.assign(_priority=priority)
    eligible = eligible.sample(frac=1, random_state=int(time.time()) % 10000)
    eligible = eligible.sort_values("_priority", kind="mergesort")
    eligible = eligible.drop(columns=["_priority"])

    return eligible[BATCH_COLUMNS].copy()


def _load_completed_gsms() -> set[str]:
    """Load the completed_gsms.txt sidecar file."""
    completed = set()
    if COMPLETED_PATH.exists():
        with open(COMPLETED_PATH) as f:
            for line in f:
                gsm = line.strip()
                if gsm:
                    completed.add(gsm)
    return completed


def grab_batch(phase: str, batch_size: int, output: str | None = None) -> list[dict]:
    """Atomically claim a batch of unclaimed samples.

    1. Load eligible GSMs (outside lock — slow I/O)
    2. Acquire exclusive file lock
    3. Read claimed + completed GSMs
    4. Pick first batch_size unclaimed
    5. Append claim to ledger
    6. Release lock
    """
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    claim_id = f"{job_id}_{task_id}"

    eligible = _load_eligible_gsms(phase)

    with open(LOCK_PATH, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            exclude = set()

            if LEDGER_PATH.exists():
                # Two-pass: first collect claims and their GSMs, then apply status updates
                claim_gsms = {}   # claim_id → set of gsm_ids
                claim_status = {} # claim_id → effective status
                with open(LEDGER_PATH) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) < 5:
                            continue
                        cid, jid, gsms, ts, status = parts[:5]
                        if status.startswith("update:"):
                            _, target_cid, new_status = status.split(":", 2)
                            if target_cid in claim_status:
                                claim_status[target_cid] = new_status
                        else:
                            gsm_list = [g for g in gsms.split(",") if g]
                            claim_gsms[cid] = gsm_list
                            claim_status[cid] = status
                # Exclude GSMs from all claims that aren't abandoned/failed
                for cid, gsm_list in claim_gsms.items():
                    if claim_status.get(cid) not in ("abandoned", "failed"):
                        exclude.update(gsm_list)

            if COMPLETED_PATH.exists():
                with open(COMPLETED_PATH) as f:
                    for line in f:
                        gsm = line.strip()
                        if gsm:
                            exclude.add(gsm)

            unclaimed = eligible[~eligible["gsm_id"].isin(exclude)]

            if len(unclaimed) == 0:
                print("No unclaimed eligible samples remaining", file=sys.stderr)
                return []

            batch_df = unclaimed.head(batch_size)
            gsm_ids = batch_df["gsm_id"].tolist()

            ts = datetime.now(timezone.utc).isoformat()
            line = f"{claim_id}\t{job_id}\t{','.join(gsm_ids)}\t{ts}\tclaimed\n"
            with open(LEDGER_PATH, "a") as f:
                f.write(line)

        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    print(f"Claimed {len(batch_df)} samples (claim: {claim_id}, phase: {phase})", file=sys.stderr)

    if output:
        batch_df.to_csv(output, index=False)

    return batch_df.to_dict("records")


def report_gsms(claim_id: str, gsm_ids: list[str]):
    """Record processed GSMs in the completed sidecar and update ledger."""
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOCK_PATH, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            with open(COMPLETED_PATH, "a") as f:
                for gsm in gsm_ids:
                    f.write(f"{gsm}\n")

            ts = datetime.now(timezone.utc).isoformat()
            line = f"{claim_id}\t\t{','.join(gsm_ids)}\t{ts}\tupdate:{claim_id}:done\n"
            with open(LEDGER_PATH, "a") as f:
                f.write(line)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    print(f"Reported {len(gsm_ids)} GSMs as completed (claim: {claim_id})", file=sys.stderr)


def mark_claim(claim_id: str, status: str):
    """Update a claim's status in the ledger."""
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{claim_id}\t\t\t{ts}\tupdate:{claim_id}:{status}\n"

    with open(LOCK_PATH, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            with open(LEDGER_PATH, "a") as f:
                f.write(line)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def show_status():
    """Print claim statistics."""
    if not LEDGER_PATH.exists():
        print("No claims yet")
        return

    claims = {}
    with open(LEDGER_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            cid, jid, gsms, ts, status = parts[:5]
            if status.startswith("update:"):
                _, target_cid, new_status = status.split(":", 2)
                if target_cid in claims:
                    claims[target_cid]["status"] = new_status
            else:
                n_gsms = len([g for g in gsms.split(",") if g])
                claims[cid] = {
                    "job_id": jid,
                    "n_gsms": n_gsms,
                    "timestamp": ts,
                    "status": status,
                }

    total = len(claims)
    total_gsms = sum(c["n_gsms"] for c in claims.values())
    by_status = {}
    for c in claims.values():
        by_status.setdefault(c["status"], 0)
        by_status[c["status"]] += 1

    n_completed = 0
    if COMPLETED_PATH.exists():
        with open(COMPLETED_PATH) as f:
            n_completed = sum(1 for line in f if line.strip())

    print(f"═══ Pipeline Claim Status ═══")
    print(f"Total claims: {total}")
    print(f"Total GSMs claimed: {total_gsms}")
    print(f"Completed GSMs (sidecar): {n_completed}")
    for s, n in sorted(by_status.items()):
        print(f"  {s}: {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase",
                        choices=list(PHASE_FILTERS.keys()),
                        help="Processing phase (determines eligibility filter)")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--output", help="Write batch CSV to this path")
    parser.add_argument("--mark-done", metavar="CLAIM_ID")
    parser.add_argument("--mark-failed", metavar="CLAIM_ID")
    parser.add_argument("--mark-abandoned", metavar="CLAIM_ID",
                        help="Mark a claim as abandoned (frees GSMs for re-claim)")
    parser.add_argument("--report-gsms", nargs=2, metavar=("CLAIM_ID", "GSM_LIST"),
                        help="Report completed GSMs (comma-separated)")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.mark_done:
        mark_claim(args.mark_done, "done")
    elif args.mark_failed:
        mark_claim(args.mark_failed, "failed")
    elif args.mark_abandoned:
        mark_claim(args.mark_abandoned, "abandoned")
    elif args.report_gsms:
        claim_id, gsm_list = args.report_gsms
        gsm_ids = [g.strip() for g in gsm_list.split(",") if g.strip()]
        report_gsms(claim_id, gsm_ids)
    else:
        if not args.phase:
            parser.error("--phase is required for batch claiming")
        samples = grab_batch(args.phase, args.batch_size, args.output)
        if not samples:
            sys.exit(1)
        print(json.dumps(samples, default=str))


if __name__ == "__main__":
    main()
