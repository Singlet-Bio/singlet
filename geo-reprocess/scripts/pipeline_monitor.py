#!/usr/bin/env python3
"""Pipeline monitoring and email reporting for Singlet Bio catalog reprocessing.

Derives stats from:
  - Processing catalog parquet (filter funnel, per-protocol breakdown)
  - On-disk truth (ledger, results CSVs, failures, quant dirs)

Shows:
  - Filter funnel: Total catalog → Droplet scRNA-seq → Human → by protocol
  - Per-protocol PASS / FAIL / QUEUED
  - Two time windows: "Last 6 Hours" (recent throughput) and "Total" (all time)
  - Repository cell count (all prior rounds + current run)
  - ETA estimate based on recent 6h success rate

Usage:
  python pipeline_monitor.py                        # Print report to stdout
  python pipeline_monitor.py --phase 1              # Report for phase 1
  python pipeline_monitor.py --email                # Send email report
  python pipeline_monitor.py --json                 # Save JSON metrics snapshot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time as _time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PIPELINE_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline")
CATALOG_PARQUET = Path("/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet")
CLAIMS_DIR = PIPELINE_DIR / "claims"
RESULTS_DIR = PIPELINE_DIR / "results"
QUANT_DIR = PIPELINE_DIR / "quant"
BATCHES_DIR = PIPELINE_DIR / "batches"
FAILURES_DIR = PIPELINE_DIR / "failures"
METRICS_FILE = PIPELINE_DIR / "monitor_metrics.json"

EMAIL_TO = "debruinz@gvsu.edu"

PHASE_LABELS = {
    "1": "Human droplet RNA",
    "2a": "Human multiome GEX",
    "2b": "Human CITE-seq GEX",
    "2c": "Human reclassified droplet",
    "2d": "Human ambiguous (auto-detect)",
    "3": "Human screen-flagged recovery",
    "4a": "Mouse droplet RNA",
    "4b": "Other model organisms",
}

RECENT_WINDOW_HOURS = 6

# Droplet protocols we process — these are NOT plate-based or spatial
DROPLET_PROTOCOLS = ["10xv2", "10xv3", "10xv3_5prime", "dropseq", "indrop"]


# ── Data collection ──────────────────────────────────────────────────────


def get_disk_usage() -> dict:
    """Get disk usage for the projects filesystem."""
    try:
        r = subprocess.run(
            ["df", "-h", "/mnt/projects/debruinz_project/"],
            capture_output=True, text=True, timeout=10,
        )
        parts = r.stdout.strip().split("\n")[-1].split()
        return {"total": parts[1], "used": parts[2], "avail": parts[3], "pct": parts[4]}
    except Exception:
        return {"total": "?", "used": "?", "avail": "?", "pct": "?"}


def get_catalog_funnel() -> dict:
    """Read catalog parquet and build the filter funnel.

    Returns dict with:
      total_catalog: int
      total_scrna: int (RNA-Seq + similar)
      total_droplet: int (droplet protocols only)
      total_human_droplet: int
      protocols: {proto: {total, done, failed, pending, skipped, repo_cells, cells_per_sample}}
      repo_cells_total: int (sum of migration_n_cells across all done samples)
      repo_samples_total: int
    """
    try:
        import pandas as pd
        cols = ["gsm_id", "organism", "protocol_inferred", "processing_status",
                "migration_n_cells", "library_strategy"]
        cat = pd.read_parquet(CATALOG_PARQUET, columns=cols)
    except Exception as e:
        print(f"Warning: could not read catalog parquet: {e}", file=sys.stderr)
        return None

    total_catalog = len(cat)

    # Step 1: RNA-related (not ATAC/ChIP/Bisulfite etc)
    rna_strategies = {"RNA-Seq", "None", "OTHER"}
    scrna = cat[cat["library_strategy"].isin(rna_strategies) | cat["library_strategy"].isna()]
    total_scrna = len(scrna)

    # Step 2: Droplet protocols
    droplet = scrna[scrna["protocol_inferred"].isin(DROPLET_PROTOCOLS)]
    total_droplet = len(droplet)

    # Step 3: Human only
    human = droplet[droplet["organism"] == "Homo sapiens"]
    total_human_droplet = len(human)

    # Per-protocol breakdown
    protocols = {}
    for proto in DROPLET_PROTOCOLS:
        sub = human[human["protocol_inferred"] == proto]
        if len(sub) == 0:
            continue
        done_mask = sub["processing_status"].isin(["done", "done_qc_warn"])
        fail_mask = sub["processing_status"].str.startswith("fail")
        pending_mask = sub["processing_status"] == "pending"
        skip_mask = sub["processing_status"].str.startswith("skip")

        done_with_cells = sub[done_mask & sub["migration_n_cells"].notna() & (sub["migration_n_cells"] > 0)]
        repo_cells = int(done_with_cells["migration_n_cells"].sum())
        cells_per_sample = int(done_with_cells["migration_n_cells"].median()) if len(done_with_cells) > 0 else 0

        protocols[proto] = {
            "total": int(len(sub)),
            "done": int(done_mask.sum()),
            "failed": int(fail_mask.sum()),
            "pending": int(pending_mask.sum()),
            "skipped": int(skip_mask.sum()),
            "repo_cells": repo_cells,
            "cells_per_sample": cells_per_sample,
        }

    # Total repo cells (all organisms, all protocols — everything ever processed)
    all_done = cat[cat["processing_status"].isin(["done", "done_qc_warn"])]
    all_with_cells = all_done[all_done["migration_n_cells"].notna() & (all_done["migration_n_cells"] > 0)]
    repo_cells_total = int(all_with_cells["migration_n_cells"].sum())
    repo_samples_total = int(len(all_with_cells))

    return {
        "total_catalog": total_catalog,
        "total_scrna": total_scrna,
        "total_droplet": total_droplet,
        "total_human_droplet": total_human_droplet,
        "protocols": protocols,
        "repo_cells_total": repo_cells_total,
        "repo_samples_total": repo_samples_total,
    }


def get_slurm_jobs(phase: str | None = None) -> dict:
    """Get current SLURM job status, filtered to pipeline jobs only."""
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "debruinz"),
             "--format=%j %T %P %N", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        jobs = {"running": Counter(), "pending": Counter(), "total": 0, "nodes": set()}
        for line in r.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name, state, partition = parts[0], parts[1], parts[2]
            node = parts[3] if len(parts) > 3 else ""
            # Only count pipeline array jobs
            if phase and not name.startswith(f"p{phase}-"):
                continue
            if not phase and not (name.startswith("p") and "-" in name):
                continue
            if name == "monitor":
                continue
            jobs["total"] += 1
            if state == "RUNNING":
                jobs["running"][partition] += 1
                if node:
                    jobs["nodes"].add(node)
            elif state == "PENDING":
                jobs["pending"][partition] += 1
        return jobs
    except Exception:
        return {"running": Counter(), "pending": Counter(), "total": 0, "nodes": set()}


def get_pool_size() -> int:
    """Count rows in all_processable.csv — the total eligible pool for grab_batch."""
    pool_file = BATCHES_DIR / "all_processable.csv"
    if not pool_file.exists():
        return 0
    try:
        with open(pool_file) as f:
            return sum(1 for _ in f) - 1  # subtract header
    except Exception:
        return 0


def get_corpus_stats() -> dict:
    """Count completed GSMs, failure records, and GSE series on disk."""
    # Completed GSMs sidecar — the real source of truth, instant to read
    completed_file = CLAIMS_DIR / "completed_gsms.txt"
    n_completed = 0
    if completed_file.exists():
        with open(completed_file) as f:
            n_completed = sum(1 for line in f if line.strip())

    # Count failure records
    n_failures = 0
    if FAILURES_DIR.exists():
        try:
            n_failures = len([f for f in os.listdir(FAILURES_DIR) if f.endswith(".json")])
        except Exception:
            pass

    # Count GSE series dirs (fast — just top-level listing)
    n_series = 0
    try:
        if QUANT_DIR.exists():
            n_series = sum(1 for d in os.listdir(QUANT_DIR) if d.startswith("GSE"))
    except Exception:
        pass

    return {
        "n_completed_gsms": n_completed,
        "n_failures": n_failures,
        "n_series": n_series,
    }



def get_claim_stats() -> dict:
    """Parse the current claim ledger — all entries belong to this submission."""
    ledger = CLAIMS_DIR / "ledger.tsv"
    claims = {}
    first_ts = None

    if ledger.exists():
        with open(ledger) as f:
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
                    claims[cid] = {
                        "n_gsms": len([g for g in gsms.split(",") if g]),
                        "timestamp": ts,
                        "status": status,
                    }
                    if first_ts is None or ts < first_ts:
                        first_ts = ts

    by_status = Counter()
    total_gsms = 0
    for c in claims.values():
        by_status[c["status"]] += 1
        total_gsms += c["n_gsms"]

    elapsed_hours = 0
    if first_ts:
        try:
            start = datetime.fromisoformat(first_ts)
            elapsed_hours = (datetime.now(timezone.utc) - start).total_seconds() / 3600
        except Exception:
            pass

    return {
        "total_claims": len(claims),
        "total_gsms_claimed": total_gsms,
        "by_status": dict(by_status),
        "elapsed_hours": round(elapsed_hours, 1),
    }


def get_results_stats() -> dict:
    """Aggregate results_*.csv files into two buckets: recent (last 6h) and total.

    Uses file mtime to determine which results are recent.
    Returns dict with 'total' and 'recent' sub-dicts, each containing
    successes, failures, skipped, total_cells, failure_types, protocols.
    """
    if not RESULTS_DIR.exists():
        empty = _empty_results()
        return {"total": empty, "recent": empty}

    results_files = sorted(RESULTS_DIR.glob("results_*.csv"))
    cutoff = _time.time() - RECENT_WINDOW_HOURS * 3600

    def new_bucket():
        return {
            "n": 0, "successes": 0, "failures": 0, "skipped": 0,
            "total_cells": 0, "failure_types": Counter(), "protocols": Counter(),
            "n_files": 0,
        }

    total = new_bucket()
    recent = new_bucket()

    for rf in results_files:
        try:
            is_recent = rf.stat().st_mtime >= cutoff
        except OSError:
            is_recent = False

        buckets = [total]
        if is_recent:
            buckets.append(recent)
            recent["n_files"] += 1
        total["n_files"] += 1

        try:
            with open(rf) as f:
                for row in csv.DictReader(f):
                    status = row.get("status", "")
                    for b in buckets:
                        b["n"] += 1
                    if status in ("success", "qc_warn"):
                        cells = 0
                        try:
                            cells = int(row.get("n_cells") or 0)
                        except (ValueError, TypeError):
                            pass
                        proto = row.get("protocol", "")
                        for b in buckets:
                            b["successes"] += 1
                            b["total_cells"] += cells
                            if proto:
                                b["protocols"][proto] += 1
                    elif status == "failed":
                        for b in buckets:
                            b["failures"] += 1
                            _classify_failure(row, b["failure_types"])
                    elif status == "skipped":
                        for b in buckets:
                            b["skipped"] += 1
        except Exception:
            continue

    def finalize(b):
        return {
            "total": b["n"],
            "successes": b["successes"],
            "failures": b["failures"],
            "skipped": b["skipped"],
            "total_cells": b["total_cells"],
            "failure_types": dict(Counter(b["failure_types"]).most_common(10)),
            "protocols": dict(Counter(b["protocols"]).most_common(10)),
            "n_files": b["n_files"],
        }

    return {"total": finalize(total), "recent": finalize(recent)}


def _classify_failure(row: dict, failure_types: Counter):
    """Classify a failed result row into a failure category."""
    err = str(row.get("error", ""))[:120].lower()
    stage = row.get("fail_stage", "")

    if "disk" in err:
        failure_types["Disk full"] += 1
    elif stage == "download" or "download" in err or "fasterq" in err or "curl" in err:
        failure_types["Download failed"] += 1
    elif "permit" in err:
        failure_types["Permit-list crash (simpleaf)"] += 1
    elif "low genes" in err or "median_genes" in err:
        failure_types["QC: low genes/cell"] += 1
    elif "few cells" in err or "n_cells" in err:
        failure_types["QC: few cells"] += 1
    elif "mapping" in err:
        failure_types["Low mapping rate"] += 1
    elif "timeout" in err or "budget" in err:
        failure_types["Download timeout"] += 1
    elif "simpleaf" in err or "piscem" in err or stage == "quant":
        failure_types["Quantification error"] += 1
    elif "protocol" in err or stage == "detect":
        failure_types["Protocol detection"] += 1
    elif stage == "qc":
        failure_types["QC failed (other)"] += 1
    else:
        failure_types["Other"] += 1


def _empty_results() -> dict:
    return {
        "total": 0, "successes": 0, "failures": 0, "skipped": 0,
        "total_cells": 0, "failure_types": {}, "protocols": {}, "n_files": 0,
    }


def estimate_eta(remaining: int, rate_per_hour: float) -> str:
    """Estimate time to completion as a calendar date or duration."""
    if rate_per_hour <= 0 or remaining <= 0:
        return "—"
    hours = remaining / rate_per_hour
    finish = datetime.now(timezone.utc) + timedelta(hours=hours)
    if hours < 48:
        return f"{hours:.1f}h ({finish.strftime('%b %d %H:%M')} UTC)"
    return f"{finish.strftime('%b %d, %Y')}"


def collect_all_data(phase: str | None = None) -> dict:
    """Collect all data from on-disk truth + catalog in one pass."""
    disk = get_disk_usage()
    jobs = get_slurm_jobs(phase)
    corpus = get_corpus_stats()
    claims = get_claim_stats()
    results_both = get_results_stats()
    pool = get_pool_size()
    funnel = get_catalog_funnel()

    results_total = results_both["total"]
    results_recent = results_both["recent"]

    # Compute rates — total uses full elapsed, recent uses the window size
    elapsed = max(claims["elapsed_hours"], 0.1)
    rate_total_h = round(results_total["successes"] / elapsed, 1)
    rate_recent_h = round(results_recent["successes"] / RECENT_WINDOW_HOURS, 1)

    # Success rates
    def _success_rate(r):
        non_skip = r["successes"] + r["failures"]
        return round(r["successes"] / non_skip * 100, 1) if non_skip > 0 else 0

    phase_label = PHASE_LABELS.get(phase, f"Phase {phase}") if phase else "All Phases"

    # Human droplet remaining = pending in catalog (not processed yet)
    hd_pending = 0
    hd_total = 0
    if funnel:
        hd_total = funnel["total_human_droplet"]
        for p in funnel["protocols"].values():
            hd_pending += p["pending"]

    # ETA based on recent rate, against human droplet pending
    eta_rate = rate_recent_h if rate_recent_h > 0 else rate_total_h
    eta = estimate_eta(hd_pending, eta_rate)

    # Repository cells: prior rounds (from catalog migration) + current run
    repo_cells_prior = funnel["repo_cells_total"] if funnel else 0
    repo_cells_current = results_total["total_cells"]
    repo_cells_combined = repo_cells_prior + repo_cells_current
    repo_samples_prior = funnel["repo_samples_total"] if funnel else 0

    # Estimate remaining cells based on median cells/sample per protocol
    est_remaining_cells = 0
    if funnel:
        for proto, pd in funnel["protocols"].items():
            if pd["cells_per_sample"] > 0 and pd["pending"] > 0:
                # Use historical success rate to discount pending
                done_and_fail = pd["done"] + pd["failed"]
                hist_success_pct = pd["done"] / max(done_and_fail, 1)
                est_remaining_cells += int(pd["pending"] * hist_success_pct * pd["cells_per_sample"])

    return {
        "disk": disk,
        "jobs": jobs,
        "corpus": corpus,
        "claims": claims,
        "results_total": results_total,
        "results_recent": results_recent,
        "pool": pool,
        "funnel": funnel,
        "elapsed": elapsed,
        "rate_total_h": rate_total_h,
        "rate_recent_h": rate_recent_h,
        "success_rate_total": _success_rate(results_total),
        "success_rate_recent": _success_rate(results_recent),
        "hd_pending": hd_pending,
        "hd_total": hd_total,
        "eta": eta,
        "eta_rate": eta_rate,
        "phase_label": phase_label,
        "repo_cells_prior": repo_cells_prior,
        "repo_cells_current": repo_cells_current,
        "repo_cells_combined": repo_cells_combined,
        "repo_samples_prior": repo_samples_prior,
        "est_remaining_cells": est_remaining_cells,
    }


# ── Report generators ────────────────────────────────────────────────────


def generate_html_report(d: dict, phase: str | None = None) -> str:
    """Generate an HTML email report with filter funnel and time-windowed stats."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    phase_label = d["phase_label"]
    disk, jobs, corpus, claims = d["disk"], d["jobs"], d["corpus"], d["claims"]
    tot, rec = d["results_total"], d["results_recent"]
    funnel = d["funnel"]

    disk_pct = int(disk["pct"].rstrip("%")) if disk["pct"] != "?" else 0
    disk_color = "#e74c3c" if disk_pct >= 95 else "#f39c12" if disk_pct >= 90 else "#27ae60"

    running_total = sum(jobs["running"].values())
    pending_total = sum(jobs["pending"].values())
    running_str = ", ".join(f"{p}: {n}" for p, n in sorted(jobs["running"].items())) or "none"
    pending_str = ", ".join(f"{p}: {n}" for p, n in sorted(jobs["pending"].items())) or "none"
    n_nodes = len(jobs["nodes"])

    def _td(val, color="#2c3e50", size="22px", lbl="", bg="#f8f9fa"):
        return (f'<td width="33%" align="center" style="padding:10px; background:{bg};">'
                f'<div style="font-size:{size}; font-weight:bold; color:{color};">{val}</div>'
                f'<div style="font-size:11px; color:#7f8c8d; text-transform:uppercase;">{lbl}</div></td>')

    def _row3(t1, t2, t3):
        return f"<tr>{t1}{t2}{t3}</tr><tr><td colspan='3' style='height:6px;'></td></tr>"

    html = f"""<!DOCTYPE html><html><head>
<style>body{{font-family:Arial,Helvetica,sans-serif;color:#2c3e50;margin:0;padding:0;background:#f5f5f5;}}
table{{border-collapse:collapse;}}</style></head><body>
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5;"><tr><td align="center">
<table width="700" cellpadding="0" cellspacing="0" style="background:#fff;margin:20px auto;">
<tr><td style="background:#2c3e50;color:#fff;padding:20px;">
  <div style="font-size:20px;font-weight:bold;">Singlet Bio &mdash; Pipeline Report</div>
  <div style="font-size:13px;color:#a0b4c8;margin-top:4px;">{now_str} &nbsp;|&nbsp; {phase_label}</div>
</td></tr>
"""

    # Disk alert
    if disk_pct >= 90:
        html += (f'<tr><td style="padding:16px;border-bottom:1px solid #e1e8ed;">'
                 f'<table width="100%" cellpadding="8" cellspacing="0" style="border-left:4px solid {disk_color};background:#fff5f5;">'
                 f'<tr><td style="color:{disk_color};font-weight:bold;font-size:13px;">&#9888; Disk at {disk["pct"]} &mdash; {disk["avail"]} remaining of {disk["total"]}</td></tr>'
                 f'</table></td></tr>')

    # ── Repository Overview ──
    html += f"""
<tr><td style="padding:16px;border-bottom:1px solid #e1e8ed;">
  <div style="font-size:15px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #8e44ad;padding-bottom:4px;margin-bottom:10px;">Repository Overview</div>
  <table width="100%" cellpadding="0" cellspacing="0">
    {_row3(
        _td(f"{d['repo_cells_combined']:,}", "#8e44ad", lbl="Total Cells in Repository"),
        _td(f"{d['repo_samples_prior'] + tot['successes']:,}", "#2c3e50", lbl="Samples Processed"),
        _td(f"{d['est_remaining_cells']:,}", "#e67e22", lbl="Est. Cells Remaining*"),
    )}
  </table>
  <div style="font-size:11px;color:#aaa;margin-top:2px;">* Based on median cells/sample &times; pending &times; historical success rate per protocol</div>
</td></tr>
"""

    # ── Catalog Filter Funnel ──
    if funnel:
        protos = funnel["protocols"]
        html += f"""
<tr><td style="padding:16px;border-bottom:1px solid #e1e8ed;">
  <div style="font-size:15px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #9b59b6;padding-bottom:4px;margin-bottom:10px;">Catalog Filter Funnel</div>
  <table width="100%" cellpadding="6" cellspacing="0" style="font-size:13px;">
    <tr><td>Total catalog entries</td><td align="right" style="font-weight:bold;">{funnel['total_catalog']:,}</td></tr>
    <tr style="color:#7f8c8d;"><td>&nbsp;&nbsp;&nbsp;&#8627; RNA-Seq (excl ATAC, ChIP, etc)</td><td align="right">{funnel['total_scrna']:,}</td></tr>
    <tr style="color:#7f8c8d;"><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8627; Droplet protocols only</td><td align="right">{funnel['total_droplet']:,}</td></tr>
    <tr><td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8627; <b>Homo sapiens</b> (current batch)</td><td align="right" style="font-weight:bold;color:#3498db;">{funnel['total_human_droplet']:,}</td></tr>
  </table>
"""
        # Per-protocol breakdown table
        html += """
  <table width="100%" cellpadding="6" cellspacing="0" style="font-size:13px;margin-top:12px;">
    <tr style="background:#f8f9fa;">
      <th align="left" style="padding:6px 8px;font-size:12px;color:#7f8c8d;text-transform:uppercase;border-bottom:2px solid #e1e8ed;">Protocol</th>
      <th align="right" style="padding:6px 8px;font-size:12px;color:#7f8c8d;text-transform:uppercase;border-bottom:2px solid #e1e8ed;">Total</th>
      <th align="right" style="padding:6px 8px;font-size:12px;color:#27ae60;text-transform:uppercase;border-bottom:2px solid #e1e8ed;">Done</th>
      <th align="right" style="padding:6px 8px;font-size:12px;color:#e74c3c;text-transform:uppercase;border-bottom:2px solid #e1e8ed;">Failed</th>
      <th align="right" style="padding:6px 8px;font-size:12px;color:#e67e22;text-transform:uppercase;border-bottom:2px solid #e1e8ed;">Queued</th>
      <th align="right" style="padding:6px 8px;font-size:12px;color:#7f8c8d;text-transform:uppercase;border-bottom:2px solid #e1e8ed;">Skipped</th>
      <th align="right" style="padding:6px 8px;font-size:12px;color:#7f8c8d;text-transform:uppercase;border-bottom:2px solid #e1e8ed;">Cells/Sample</th>
    </tr>
"""
        sum_total = sum_done = sum_fail = sum_pend = sum_skip = 0
        for proto in DROPLET_PROTOCOLS:
            if proto not in protos:
                continue
            p = protos[proto]
            sum_total += p["total"]
            sum_done += p["done"]
            sum_fail += p["failed"]
            sum_pend += p["pending"]
            sum_skip += p["skipped"]
            cps = f'{p["cells_per_sample"]:,}' if p["cells_per_sample"] > 0 else "—"
            html += (f'<tr><td style="padding:6px 8px;border-bottom:1px solid #f0f0f0;font-weight:bold;">{proto}</td>'
                     f'<td align="right" style="padding:6px 8px;border-bottom:1px solid #f0f0f0;">{p["total"]:,}</td>'
                     f'<td align="right" style="padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#27ae60;">{p["done"]:,}</td>'
                     f'<td align="right" style="padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#e74c3c;">{p["failed"]:,}</td>'
                     f'<td align="right" style="padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#e67e22;">{p["pending"]:,}</td>'
                     f'<td align="right" style="padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#7f8c8d;">{p["skipped"]:,}</td>'
                     f'<td align="right" style="padding:6px 8px;border-bottom:1px solid #f0f0f0;">{cps}</td></tr>\n')
        # Totals row
        html += (f'<tr style="background:#f8f9fa;font-weight:bold;">'
                 f'<td style="padding:6px 8px;">Total</td>'
                 f'<td align="right" style="padding:6px 8px;">{sum_total:,}</td>'
                 f'<td align="right" style="padding:6px 8px;color:#27ae60;">{sum_done:,}</td>'
                 f'<td align="right" style="padding:6px 8px;color:#e74c3c;">{sum_fail:,}</td>'
                 f'<td align="right" style="padding:6px 8px;color:#e67e22;">{sum_pend:,}</td>'
                 f'<td align="right" style="padding:6px 8px;color:#7f8c8d;">{sum_skip:,}</td>'
                 f'<td align="right" style="padding:6px 8px;"></td></tr>\n')
        html += "  </table>\n"

        # ETA
        html += f"""
  <table width="100%" cellpadding="8" cellspacing="0" style="margin-top:10px;border-left:4px solid #8e44ad;background:#f8f4fc;">
    <tr><td style="font-size:13px;color:#555;">
      <b>Queued:</b> {d['hd_pending']:,} samples &nbsp;|&nbsp;
      <b>Rate (last 6h):</b> {d['rate_recent_h']} success/hr &nbsp;|&nbsp;
      <b>Est. completion:</b> {d['eta']}
    </td></tr>
  </table>
</td></tr>
"""

    # ── Last 6 Hours ──
    rec_sr = f"{d['success_rate_recent']}%" if rec["total"] > 0 else "&mdash;"
    html += f"""
<tr><td style="padding:16px;border-bottom:1px solid #e1e8ed;">
  <div style="font-size:15px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #e67e22;padding-bottom:4px;margin-bottom:10px;">Last {RECENT_WINDOW_HOURS} Hours</div>
  <table width="100%" cellpadding="0" cellspacing="0">
    {_row3(
        _td(f"{rec['successes']:,}", "#27ae60", lbl="Successes"),
        _td(f"{rec['failures']:,}", "#e74c3c", lbl="Failures"),
        _td(f"{rec['total_cells']:,}", "#2c3e50", lbl="Cells"),
    )}
    {_row3(
        _td(f"{d['rate_recent_h']}", "#27ae60", lbl="Successes/hr"),
        _td(rec_sr, "#2c3e50", lbl="Success Rate"),
        _td(f"{rec['n_files']}", "#7f8c8d", lbl="Result Files"),
    )}
  </table>
</td></tr>
"""

    # ── Total (This Run) ──
    tot_sr = f"{d['success_rate_total']}%" if tot["total"] > 0 else "&mdash;"
    html += f"""
<tr><td style="padding:16px;border-bottom:1px solid #e1e8ed;">
  <div style="font-size:15px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px;margin-bottom:10px;">This Run ({d['elapsed']:.1f}h elapsed)</div>
  <table width="100%" cellpadding="0" cellspacing="0">
    {_row3(
        _td(f"{tot['successes']:,}", "#27ae60", lbl="Successes"),
        _td(f"{tot['failures']:,}", "#e74c3c", lbl="Failures"),
        _td(f"{tot['total_cells']:,}", "#2c3e50", lbl="Cells This Run"),
    )}
    {_row3(
        _td(f"{d['rate_total_h']}", "#27ae60", lbl="Successes/hr"),
        _td(tot_sr, "#2c3e50", lbl="Success Rate"),
        _td(f"{tot['skipped']:,}", "#f39c12", lbl="Skipped"),
    )}
  </table>
</td></tr>
"""

    # ── SLURM Jobs ──
    html += f"""
<tr><td style="padding:16px;border-bottom:1px solid #e1e8ed;">
  <div style="font-size:15px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px;margin-bottom:10px;">SLURM Jobs</div>
  <table width="100%" cellpadding="6" cellspacing="0" style="font-size:13px;">
    <tr><td>Running</td><td align="right">{running_total} ({running_str})</td></tr>
    <tr><td>Pending</td><td align="right">{pending_total} ({pending_str})</td></tr>
    <tr><td>Active nodes</td><td align="right">{n_nodes}</td></tr>
    <tr><td>Disk</td><td align="right" style="color:{disk_color}">{disk['used']} / {disk['total']} ({disk['pct']})</td></tr>
  </table>
</td></tr>
"""

    # ── Failure Modes ──
    if tot["failure_types"]:
        html += """
<tr><td style="padding:16px;border-bottom:1px solid #e1e8ed;">
  <div style="font-size:15px;font-weight:bold;color:#2c3e50;border-bottom:2px solid #e74c3c;padding-bottom:4px;margin-bottom:10px;">Failure Modes (This Run)</div>
  <table width="100%" cellpadding="6" cellspacing="0" style="font-size:13px;">
    <tr style="background:#f8f9fa;">
      <th align="left" style="padding:6px 8px;font-size:12px;color:#7f8c8d;border-bottom:2px solid #e1e8ed;">Category</th>
      <th align="right" style="padding:6px 8px;font-size:12px;color:#7f8c8d;border-bottom:2px solid #e1e8ed;">Count</th>
    </tr>
"""
        for fcat, count in sorted(tot["failure_types"].items(), key=lambda x: -x[1]):
            html += f'<tr><td style="padding:6px 8px;border-bottom:1px solid #f0f0f0;">{fcat}</td><td align="right" style="padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#e74c3c;">{count}</td></tr>\n'
        html += "</table></td></tr>"

    html += "</table></td></tr></table></body></html>"
    return html


def generate_text_report(d: dict, phase: str | None = None) -> str:
    """Generate a plain-text version for stdout."""
    disk, jobs, corpus, claims = d["disk"], d["jobs"], d["corpus"], d["claims"]
    tot, rec = d["results_total"], d["results_recent"]
    funnel = d["funnel"]

    phase_label = d["phase_label"]
    running_parts = [f"{p}:{n}" for p, n in sorted(jobs["running"].items())]
    pending_parts = [f"{p}:{n}" for p, n in sorted(jobs["pending"].items())]
    running_total = sum(jobs["running"].values())
    pending_total = sum(jobs["pending"].values())
    n_nodes = len(jobs["nodes"])

    lines = [
        "=" * 65,
        f"  Singlet Bio -- Pipeline Status ({phase_label})",
        f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 65,
        "",
        "  -- Repository --",
        f"  Total cells:        {d['repo_cells_combined']:>14,}",
        f"  Samples processed:  {d['repo_samples_prior'] + tot['successes']:>14,}",
        f"  Est. cells remaining: {d['est_remaining_cells']:>12,}",
    ]

    if funnel:
        lines += [
            "",
            "  -- Catalog Filter Funnel --",
            f"  Total catalog:       {funnel['total_catalog']:>10,}",
            f"    -> RNA-Seq:         {funnel['total_scrna']:>10,}",
            f"      -> Droplet:       {funnel['total_droplet']:>10,}",
            f"        -> Human:       {funnel['total_human_droplet']:>10,}  (current batch)",
            "",
            f"  {'Protocol':<16s} {'Total':>7s} {'Done':>7s} {'Failed':>7s} {'Queued':>7s} {'Skip':>7s} {'Cells/S':>8s}",
            f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}",
        ]
        for proto in DROPLET_PROTOCOLS:
            if proto not in funnel["protocols"]:
                continue
            p = funnel["protocols"][proto]
            cps = str(p["cells_per_sample"]) if p["cells_per_sample"] > 0 else "—"
            lines.append(f"  {proto:<16s} {p['total']:>7,} {p['done']:>7,} {p['failed']:>7,} {p['pending']:>7,} {p['skipped']:>7,} {cps:>8s}")
        lines.append(f"  Queued: {d['hd_pending']:,}  |  Rate (6h): {d['rate_recent_h']}/h  |  ETA: {d['eta']}")

    lines += [
        "",
        f"  -- Last {RECENT_WINDOW_HOURS} Hours --",
        f"  Successes:     {rec['successes']:>8,}",
        f"  Failures:      {rec['failures']:>8,}",
        f"  Cells:         {rec['total_cells']:>10,}",
        f"  Rate:          {d['rate_recent_h']:>8} success/h",
        f"  Success rate:  {d['success_rate_recent']:>7}%",
        "",
        f"  -- This Run ({d['elapsed']:.1f}h) --",
        f"  Successes:     {tot['successes']:>8,}",
        f"  Failures:      {tot['failures']:>8,}",
        f"  Skipped:       {tot['skipped']:>8,}",
        f"  Cells:         {tot['total_cells']:>10,}",
        f"  Rate:          {d['rate_total_h']:>8} success/h",
        f"  Success rate:  {d['success_rate_total']:>7}%",
        "",
        "  -- SLURM Jobs --",
        f"  Running: {running_total} ({', '.join(running_parts) or 'none'})",
        f"  Pending: {pending_total} ({', '.join(pending_parts) or 'none'})",
        f"  Nodes:   {n_nodes}",
        f"  Disk:    {disk['used']} / {disk['total']} ({disk['pct']})",
    ]

    if tot["failure_types"]:
        lines.extend(["", "  -- Failure Modes --"])
        for fcat, count in sorted(tot["failure_types"].items(), key=lambda x: -x[1])[:8]:
            lines.append(f"  [{count:>3}] {fcat}")

    lines.extend(["", "=" * 65])
    return "\n".join(lines)


# ── Email and metrics ─────────────────────────────────────────────────────


def send_email(d: dict, phase: str | None = None):
    """Send HTML email via sendmail, with plain-text fallback via mail."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    phase_tag = f"Phase {phase}" if phase else "All"
    subject = f"[Singlet Bio] Pipeline {phase_tag} -- {now_str} UTC"
    html = generate_html_report(d, phase)
    text = generate_text_report(d, phase)

    # Try HTML via sendmail
    message = (
        f"To: {EMAIL_TO}\n"
        f"Subject: {subject}\n"
        f"Content-Type: text/html; charset=utf-8\n"
        f"MIME-Version: 1.0\n"
        f"\n"
        f"{html}"
    )
    try:
        proc = subprocess.run(["sendmail", "-t"], input=message, text=True, timeout=30)
        if proc.returncode == 0:
            print(f"HTML email sent to {EMAIL_TO}", file=sys.stderr)
            return
    except FileNotFoundError:
        pass

    # Fallback: plain text via mail
    try:
        subprocess.run(["mail", "-s", subject, EMAIL_TO], input=text, text=True, timeout=30)
        print(f"Plain text email sent to {EMAIL_TO}", file=sys.stderr)
    except Exception as e:
        print(f"Email error: {e}", file=sys.stderr)


def save_metrics(d: dict, phase: str | None = None):
    """Save JSON metrics snapshot for trend tracking."""

    history = []
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                history = json.load(f)
        except Exception:
            pass

    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase or "all",
        "successes_total": d["results_total"]["successes"],
        "successes_recent": d["results_recent"]["successes"],
        "failures_total": d["results_total"]["failures"],
        "total_cells_run": d["results_total"]["total_cells"],
        "repo_cells_combined": d["repo_cells_combined"],
        "rate_total_h": d["rate_total_h"],
        "rate_recent_h": d["rate_recent_h"],
        "success_rate_total": d["success_rate_total"],
        "success_rate_recent": d["success_rate_recent"],
        "hd_pending": d["hd_pending"],
        "hd_total": d["hd_total"],
        "pool": d["pool"],
        "eta": d["eta"],
        "disk_pct": d["disk"]["pct"],
    })

    history = history[-200:]
    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Metrics saved ({len(history)} snapshots)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=str, default=None,
                        choices=list(PHASE_LABELS.keys()),
                        help="Scope report to a specific phase")
    parser.add_argument("--email", action="store_true", help="Send HTML email report")
    parser.add_argument("--json", action="store_true", help="Save JSON metrics snapshot")
    args = parser.parse_args()

    # Collect all data once — avoid repeated slow NFS traversals
    d = collect_all_data(args.phase)

    print(generate_text_report(d, args.phase))

    if args.email:
        send_email(d, args.phase)

    if args.json:
        save_metrics(d, args.phase)


if __name__ == "__main__":
    main()
