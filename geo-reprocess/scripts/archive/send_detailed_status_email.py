#!/usr/bin/env python3
"""Send a detailed pipeline status email with processing rates, ETAs, and per-species breakdowns.

This is a more detailed version of send_status_email.py intended for hourly updates
during active processing periods.

Usage:
    python send_detailed_status_email.py              # Send detailed report
    python send_detailed_status_email.py --dry-run    # Print to stdout
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
PIPELINE_BASE = Path("/mnt/projects/debruinz_project/cellarium/pipeline")
QUANT_BASE = PIPELINE_BASE / "quant"
BATCH_DIR = PIPELINE_BASE / "batches_v6_production"
METRICS_JSON = Path("/mnt/home/debruinz/Singlet-AI/singletai-website/public/data/pipeline-metrics.json")
RECIPIENT = "debruinz@gvsu.edu"
FROM_ADDR = "debruinz@gvsu.edu"
DASHBOARD_URL = "https://singletai.com/invest#corpus"


def fmt(n):
    return f"{n:,}"


def fmt_big(n):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return fmt(n)


def collect_batch_status():
    """Analyze batch completion: completed, in-progress, not-started, by organism.

    Uses fast line parsing (not csv.DictReader) since batch CSVs have simple
    comma-delimited fields without quoting.
    """
    completed = {}
    in_progress = {}
    not_started = {}

    batch_files = sorted(BATCH_DIR.glob("batch_?????.csv"))
    for bf in batch_files:
        idx = bf.stem.replace("batch_", "")
        result_file = bf.parent / f"batch_{idx}_results.csv"
        lock_dir = bf.parent / f"batch_{idx}.csv.lock"

        organisms = Counter()
        protocols = Counter()
        n_samples = 0
        try:
            with open(bf) as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        n_samples += 1
                        organisms[parts[2]] += 1
                        protocols[parts[3]] += 1
        except OSError:
            continue

        info = {"idx": idx, "n_samples": n_samples, "organisms": organisms, "protocols": protocols}

        if result_file.exists():
            completed[idx] = info
        elif lock_dir.exists():
            in_progress[idx] = info
        else:
            not_started[idx] = info

    return completed, in_progress, not_started


def collect_result_details():
    """Parse all completed batch result CSVs for failure breakdown, species cells, and protocol stats."""
    statuses = Counter()
    failure_categories = Counter()
    failure_by_protocol = defaultdict(Counter)  # protocol -> {category: count}
    failure_by_species = defaultdict(Counter)   # organism -> {category: count}

    species_cells = Counter()       # organism -> total cells
    species_samples = Counter()     # organism -> total success samples
    species_protocols = defaultdict(Counter)  # organism -> {protocol: count}

    protocol_success = Counter()    # protocol -> success count
    protocol_total = Counter()      # protocol -> total attempted

    skip_categories = Counter()
    total_time_by_status = defaultdict(list)

    for rf in sorted(BATCH_DIR.glob("batch_*_results.csv")):
        try:
            with open(rf) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("status", "unknown")
                    statuses[status] += 1
                    organism = row.get("organism", "Unknown")
                    protocol = row.get("protocol", "") or row.get("mode", "") or "unknown"
                    n_cells = int(float(row.get("n_cells") or 0))

                    protocol_total[protocol] += 1

                    try:
                        t = float(row.get("total_time_s", 0) or 0)
                        if t > 0:
                            total_time_by_status[status].append(t)
                    except (ValueError, TypeError):
                        pass

                    if status in ("success", "qc_warn"):
                        protocol_success[protocol] += 1
                        if n_cells > 0:
                            species_cells[organism] += n_cells
                            species_samples[organism] += 1
                            species_protocols[organism][protocol] += 1

                    elif status == "failed":
                        cat = _categorize_failure(row.get("error", ""))
                        failure_categories[cat] += 1
                        failure_by_protocol[protocol][cat] += 1
                        failure_by_species[organism][cat] += 1

                    elif status == "skipped":
                        cat = _categorize_skip(row.get("error", ""))
                        skip_categories[cat] += 1
        except OSError:
            continue

    # Compute average time per status
    avg_time = {}
    for st, times in total_time_by_status.items():
        if times:
            avg_time[st] = sum(times) / len(times)

    return {
        "statuses": statuses,
        "failure_categories": failure_categories,
        "failure_by_protocol": failure_by_protocol,
        "failure_by_species": failure_by_species,
        "species_cells": species_cells,
        "species_samples": species_samples,
        "species_protocols": species_protocols,
        "protocol_success": protocol_success,
        "protocol_total": protocol_total,
        "skip_categories": skip_categories,
        "avg_time": avg_time,
    }


def _categorize_failure(error: str) -> str:
    """Categorize a failure error message into a short bucket."""
    if not error:
        return "unknown"
    m = re.search(r"Low genes per cell \((\d+)", error)
    if m:
        val = int(m.group(1))
        if val == 0:
            return "zero_genes"
        if val < 10:
            return "very_low_genes (1-9)"
        if val < 20:
            return "low_genes (10-19)"
        return "low_genes (20-49)"
    if "Low cells" in error:
        return "low_cell_count"
    if "No R2 files" in error or "No R2" in error:
        return "no_R2_files"
    if "simpleaf failed" in error:
        if "permit" in error.lower():
            return "simpleaf_permit_list"
        return "simpleaf_other"
    if "Download" in error or "download" in error:
        return "download_failure"
    if "timeout" in error.lower():
        return "timeout"
    if "Protocol detection" in error:
        return "protocol_detection"
    return "other"


def _categorize_skip(error: str) -> str:
    """Categorize a skip reason."""
    if not error:
        return "unknown"
    if "Smart-seq" in error or "smartseq" in error.lower():
        return "smartseq"
    if "GSE-level failure" in error:
        return "gse_failure_cascade"
    if "already" in error.lower():
        return "already_processed"
    if "CITE-seq" in error or "cite" in error.lower():
        return "cite_seq"
    if "ATAC" in error.upper():
        return "atac"
    return "other"


def aggregate_organisms(batch_dict):
    """Sum organism counts across batches."""
    totals = Counter()
    for info in batch_dict.values():
        totals += info["organisms"]
    return totals


def aggregate_protocols(batch_dict):
    """Sum protocol counts across batches."""
    totals = Counter()
    for info in batch_dict.values():
        totals += info["protocols"]
    return totals


def count_species(organisms, target):
    """Count samples matching a species substring."""
    return sum(n for org, n in organisms.items() if target in org)


def collect_log_completions():
    """Parse v7/v8 log files for completion timestamps and rates."""
    completions = []
    log_dir = PIPELINE_BASE / "logs"

    for err_file in sorted(log_dir.glob("v[78]-*.err")):
        try:
            with open(err_file) as f:
                for line in f:
                    if "Complete: " not in line:
                        continue
                    ts_str = line[:23].replace(",", ".")
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                    if "success" in line:
                        status = "success"
                    elif "qc_warn" in line:
                        status = "qc_warn"
                    elif "failed" in line:
                        status = "failed"
                    elif "skipped" in line:
                        status = "skipped"
                    else:
                        continue
                    completions.append({"ts": ts, "status": status})
        except OSError:
            continue

    return sorted(completions, key=lambda x: x["ts"])


def compute_rates(completions, now):
    """Compute processing rates over multiple time windows for realistic ETA.

    Returns dict with rates for 1h, 6h, 24h windows, plus skip ratio.
    The ETA uses *throughput rate* (all samples/hr) because skipped samples
    still consume batch slots. The effective *success rate* is shown separately.
    """
    windows = {"1h": 1, "6h": 6, "24h": 24}
    rates = {}

    for label, hours in windows.items():
        cutoff = now - timedelta(hours=hours)
        window = [c for c in completions if c["ts"] >= cutoff]
        if not window:
            rates[label] = {"throughput": 0.0, "success": 0.0, "total": 0, "successes": 0}
            continue
        span_hours = (now - window[0]["ts"]).total_seconds() / 3600
        if span_hours <= 0:
            rates[label] = {"throughput": 0.0, "success": 0.0, "total": 0, "successes": 0}
            continue
        n_total = len(window)
        n_success = sum(1 for c in window if c["status"] in ("success", "qc_warn"))
        rates[label] = {
            "throughput": n_total / span_hours,
            "success": n_success / span_hours,
            "total": n_total,
            "successes": n_success,
        }

    # Overall skip ratio from all completions (for adjusting remaining work)
    total_all = len(completions)
    total_success = sum(1 for c in completions if c["status"] in ("success", "qc_warn"))
    total_skipped = sum(1 for c in completions if c["status"] == "skipped")
    total_failed = sum(1 for c in completions if c["status"] == "failed")
    skip_ratio = total_skipped / total_all if total_all > 0 else 0
    success_ratio = total_success / total_all if total_all > 0 else 0

    return rates, {
        "skip_ratio": skip_ratio,
        "success_ratio": success_ratio,
        "total": total_all,
        "success": total_success,
        "skipped": total_skipped,
        "failed": total_failed,
    }


def collect_slurm_jobs():
    """Get current SLURM job state."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "debruinz"),
             "-o", "%.18i %.30j %.8T %.12M %.6D %R"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"jobs": [], "summary": {}}

    jobs = []
    summary = defaultdict(lambda: {"running": 0, "pending": 0})
    nodes = set()

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        job_id, name, state = parts[0], parts[1], parts[2]
        node = parts[-1] if len(parts) >= 6 else ""

        if state == "RUNNING":
            summary[name]["running"] += 1
            if node and not node.startswith("("):
                nodes.add(node)
        elif state == "PENDING":
            summary[name]["pending"] += 1

        jobs.append({"id": job_id, "name": name, "state": state, "node": node})

    total_running = sum(v["running"] for v in summary.values())
    total_pending = sum(v["pending"] for v in summary.values())

    return {
        "jobs": jobs,
        "summary": dict(summary),
        "total_running": total_running,
        "total_pending": total_pending,
        "nodes": sorted(nodes),
    }


def load_metrics_history():
    """Load history from metrics JSON for rate calculations."""
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON) as f:
                data = json.load(f)
            return data.get("history", []), data.get("corpus", {})
        except (json.JSONDecodeError, OSError):
            pass
    return [], {}


def build_html(batch_completed, batch_in_progress, batch_not_started,
               completions, slurm, history, corpus_metrics, result_details):
    """Build the detailed HTML email body."""
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%B %d, %Y %H:%M UTC")

    # ── Batch-level stats ──
    n_completed = len(batch_completed)
    n_in_progress = len(batch_in_progress)
    n_not_started = len(batch_not_started)
    n_total_batches = n_completed + n_in_progress + n_not_started

    samples_completed_batches = sum(i["n_samples"] for i in batch_completed.values())
    samples_in_progress = sum(i["n_samples"] for i in batch_in_progress.values())
    samples_not_started = sum(i["n_samples"] for i in batch_not_started.values())
    samples_total = samples_completed_batches + samples_in_progress + samples_not_started

    # Organism breakdowns from batch CSVs
    orgs_completed = aggregate_organisms(batch_completed)
    orgs_in_progress = aggregate_organisms(batch_in_progress)
    orgs_remaining = aggregate_organisms(batch_not_started)
    orgs_all = aggregate_organisms({**batch_completed, **batch_in_progress, **batch_not_started})

    protos_remaining = aggregate_protocols(batch_not_started)

    human_total = count_species(orgs_all, "Homo sapiens")
    human_completed = count_species(orgs_completed, "Homo sapiens")
    human_in_progress = count_species(orgs_in_progress, "Homo sapiens")
    human_remaining = count_species(orgs_remaining, "Homo sapiens")

    mouse_total = count_species(orgs_all, "Mus musculus")
    mouse_completed = count_species(orgs_completed, "Mus musculus")
    mouse_in_progress = count_species(orgs_in_progress, "Mus musculus")
    mouse_remaining = count_species(orgs_remaining, "Mus musculus")

    # ── Processing rates (multi-window) ──
    rates, log_totals = compute_rates(completions, now)

    # Today's numbers
    today_completions = [c for c in completions if c["ts"].date() == now.date()]
    success_today = sum(1 for c in today_completions if c["status"] in ("success", "qc_warn"))
    failed_today = sum(1 for c in today_completions if c["status"] == "failed")
    skipped_today = sum(1 for c in today_completions if c["status"] == "skipped")
    total_today = len(today_completions)

    # Hourly sparkline
    hourly = Counter()
    hourly_all = Counter()
    for c in today_completions:
        hourly_all[c["ts"].hour] += 1
        if c["status"] in ("success", "qc_warn"):
            hourly[c["ts"].hour] += 1
    current_hour = now.hour

    # ── Metrics JSON ──
    total_cells = corpus_metrics.get("total_cells", 0)
    samples_quantified = corpus_metrics.get("samples_quantified", 0)
    avg_cells = corpus_metrics.get("avg_cells_per_sample", 0)
    species_data = corpus_metrics.get("species", {})

    # Cells/hour from metrics history
    cells_rate_per_hour = 0
    if len(history) >= 2:
        latest = history[-1]
        for h_entry in history:
            t = datetime.fromisoformat(h_entry["timestamp"])
            if (now - t).total_seconds() / 3600 >= 5.5:
                t_latest = datetime.fromisoformat(latest["timestamp"])
                t_earlier = datetime.fromisoformat(h_entry["timestamp"])
                span = (t_latest - t_earlier).total_seconds() / 3600
                if span > 0:
                    c_latest = latest.get("total_cells", 0)
                    c_earlier = h_entry.get("total_cells", h_entry.get("total_cells_estimated", 0))
                    cells_rate_per_hour = (c_latest - c_earlier) / span
                break

    # ── ETA calculation ──
    # Use 6h throughput rate (includes skips/fails — reflects actual batch consumption)
    throughput_rate = rates["6h"]["throughput"] or rates["1h"]["throughput"] or rates["24h"]["throughput"]
    success_rate = rates["6h"]["success"] or rates["1h"]["success"] or rates["24h"]["success"]
    rate_window = "6h" if rates["6h"]["throughput"] > 0 else ("1h" if rates["1h"]["throughput"] > 0 else "24h")

    samples_remaining = samples_in_progress + samples_not_started

    if throughput_rate > 0:
        eta_all_hours = samples_remaining / throughput_rate
        eta_all_days = eta_all_hours / 24
        eta_human_hours = human_remaining / throughput_rate
        eta_human_days = eta_human_hours / 24
        expected_new_cells = success_rate * eta_all_hours * avg_cells if success_rate > 0 and avg_cells > 0 else 0
    else:
        eta_all_hours = eta_all_days = eta_human_hours = eta_human_days = 0
        expected_new_cells = 0

    pct_batches = (n_completed / n_total_batches * 100) if n_total_batches > 0 else 0
    bar_width = min(pct_batches, 100)

    # ── Build HTML sections ──

    # Hourly sparkline
    hourly_chart = ""
    for h in range(current_hour + 1):
        n_s = hourly.get(h, 0)
        n_all = hourly_all.get(h, 0)
        n_other = n_all - n_s
        bar_s = "\u2588" * n_s if n_s > 0 else ""
        bar_f = "<span style='color:#dc2626'>" + "\u2591" * n_other + "</span>" if n_other > 0 else ""
        display = bar_s + bar_f if n_all > 0 else "\u00b7"
        hourly_chart += (
            f"<tr><td style='padding:1px 8px 1px 0;font-family:monospace;color:#888;font-size:11px'>{h:02d}:00</td>"
            f"<td style='font-family:monospace;color:#0d9488;font-size:11px'>{display}</td>"
            f"<td style='padding:1px 0 1px 6px;font-family:monospace;font-size:11px;color:#555'>{n_s}</td>"
            f"<td style='padding:1px 0 1px 4px;font-family:monospace;font-size:10px;color:#999'>"
            f"{f'({n_all})' if n_other > 0 else ''}</td></tr>\n"
        )

    # SLURM summary
    slurm_rows = ""
    for name, counts in sorted(slurm["summary"].items()):
        if "meta_" in name or "email" in name:
            continue
        status_parts = []
        if counts["running"] > 0:
            status_parts.append(f"<span style='color:#16a34a;font-weight:bold'>{counts['running']} running</span>")
        if counts["pending"] > 0:
            status_parts.append(f"<span style='color:#d97706'>{counts['pending']} queued</span>")
        slurm_rows += (
            f"<tr><td style='padding:3px 12px 3px 0;font-family:monospace;font-size:12px'>{name}</td>"
            f"<td style='font-size:12px'>{', '.join(status_parts)}</td></tr>\n"
        )

    # ── Failure breakdown ──
    rd = result_details
    statuses = rd["statuses"]
    n_success = statuses.get("success", 0) + statuses.get("qc_warn", 0)
    n_failed = statuses.get("failed", 0)
    n_skipped = statuses.get("skipped", 0)
    n_processed = n_success + n_failed + n_skipped

    failure_rows = ""
    for cat, count in rd["failure_categories"].most_common():
        pct = count / n_failed * 100 if n_failed > 0 else 0
        color = "#dc2626" if cat in ("zero_genes", "very_low_genes (1-9)") else "#d97706"
        failure_rows += (
            f"<tr><td style='padding:2px 10px 2px 0;font-size:12px'>{cat}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:12px;color:{color}'>{count}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:11px;color:#888;padding-left:6px'>"
            f"{pct:.0f}%</td></tr>\n"
        )

    # Failure by protocol
    fail_proto_rows = ""
    proto_fails_sorted = sorted(rd["failure_by_protocol"].items(), key=lambda x: sum(x[1].values()), reverse=True)
    for proto, cats in proto_fails_sorted[:8]:
        total_f = sum(cats.values())
        top_cats = ", ".join(f"{c}({n})" for c, n in cats.most_common(2))
        fail_proto_rows += (
            f"<tr><td style='padding:2px 10px 2px 0;font-family:monospace;font-size:11px'>{proto}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:11px'>{total_f}</td>"
            f"<td style='font-size:10px;color:#888;padding-left:8px'>{top_cats}</td></tr>\n"
        )

    # Skip breakdown
    skip_rows = ""
    for cat, count in rd["skip_categories"].most_common():
        skip_rows += (
            f"<tr><td style='padding:2px 10px 2px 0;font-size:12px'>{cat}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:12px'>{fmt(count)}</td></tr>\n"
        )

    # ── Per-species cells + protocols ──
    all_species = {}
    for org, data in species_data.items():
        all_species[org] = {"cells": data["cells"], "samples": data["samples"], "protocols": Counter()}
    for org, protos in rd["species_protocols"].items():
        if org in all_species:
            all_species[org]["protocols"] = protos
        else:
            all_species[org] = {
                "cells": rd["species_cells"].get(org, 0),
                "samples": rd["species_samples"].get(org, 0),
                "protocols": protos,
            }

    species_detail_rows = ""
    top_species = sorted(all_species.items(), key=lambda x: x[1]["cells"], reverse=True)
    for name, data in top_species:
        if data["cells"] == 0:
            continue
        proto_str = ", ".join(f"{p}({n})" for p, n in data["protocols"].most_common(3)) if data["protocols"] else ""
        species_detail_rows += (
            f"<tr><td style='padding:3px 8px 3px 0;font-style:italic;font-size:12px;color:#333'>{name}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:12px;padding:3px 8px'>{fmt(data['samples'])}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:12px;padding:3px 8px;color:#0d9488;"
            f"font-weight:bold'>{fmt_big(data['cells'])}</td>"
            f"<td style='font-size:10px;color:#888;padding:3px 0 3px 8px'>{proto_str}</td></tr>\n"
        )

    # ── Remaining work ──
    remaining_org_rows = ""
    for org, n in orgs_remaining.most_common(10):
        remaining_org_rows += (
            f"<tr><td style='padding:2px 10px 2px 0;font-size:12px;font-style:italic'>{org}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:12px'>{fmt(n)}</td></tr>\n"
        )

    proto_rows = ""
    for proto, n in protos_remaining.most_common(12):
        proto_rows += (
            f"<tr><td style='padding:2px 10px 2px 0;font-family:monospace;font-size:11px'>{proto}</td>"
            f"<td style='font-family:monospace;text-align:right;font-size:11px'>{fmt(n)}</td></tr>\n"
        )

    # ── Rate comparison + ETA section ──
    rate_comparison = ""
    for label in ["1h", "6h", "24h"]:
        r = rates[label]
        if r["throughput"] > 0:
            marker = " &larr;" if label == rate_window else ""
            rate_comparison += (
                f"<tr><td style='padding:2px 8px 2px 0;font-size:12px;color:#555'>{label}</td>"
                f"<td style='font-family:monospace;font-size:12px;text-align:right'>{r['throughput']:.1f}</td>"
                f"<td style='font-family:monospace;font-size:12px;text-align:right;color:#0d9488'>{r['success']:.1f}</td>"
                f"<td style='font-family:monospace;font-size:12px;text-align:right'>{r['total']}</td>"
                f"<td style='font-size:10px;color:#888'>{marker}</td></tr>\n"
            )

    eta_section = ""
    if throughput_rate > 0:
        eta_completion_all = now + timedelta(hours=eta_all_hours)
        eta_completion_human = now + timedelta(hours=eta_human_hours)
        skip_pct = log_totals["skip_ratio"] * 100

        expected_cells_html = ""
        if expected_new_cells > 0:
            expected_cells_html = (
                f"<tr><td colspan='2' style='padding:4px 0'></td></tr>"
                f"<tr><td style='padding:4px 0;color:#555'>Expected additional cells</td>"
                f"<td style='font-family:monospace'>~{fmt_big(int(expected_new_cells))} "
                f"(at {fmt(avg_cells)} avg/sample)</td></tr>"
            )

        eta_section = f"""
    <div style="margin:16px 0;padding:14px;background:#fffbeb;border:1px solid #fbbf24;border-radius:8px">
      <strong style="color:#92400e;font-size:14px">Estimated Time to Completion</strong>

      <table style="width:100%;font-size:12px;margin-top:8px;border-collapse:collapse">
        <tr style="color:#888;font-size:11px">
          <th style="text-align:left;padding:2px 8px 2px 0;font-weight:normal">Window</th>
          <th style="text-align:right;padding:2px 8px;font-weight:normal">Throughput/hr</th>
          <th style="text-align:right;padding:2px 8px;font-weight:normal">Success/hr</th>
          <th style="text-align:right;padding:2px 0;font-weight:normal">Processed</th>
          <th></th>
        </tr>
        {rate_comparison}
      </table>

      <div style="margin-top:10px;padding-top:8px;border-top:1px solid #fde68a">
        <table style="width:100%;font-size:13px">
          <tr>
            <td style="padding:4px 0;color:#555">Throughput rate (used for ETA)</td>
            <td style="font-family:monospace;font-weight:bold;color:#0d9488">{throughput_rate:.1f} samples/hr ({rate_window} window)</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#555">Cells/hour (from metrics)</td>
            <td style="font-family:monospace">{fmt_big(int(cells_rate_per_hour))}/hr</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#555">Historical skip rate</td>
            <td style="font-family:monospace">{skip_pct:.0f}% of samples skipped (smartseq, etc.)</td>
          </tr>
          <tr><td colspan="2" style="padding:8px 0 2px 0;border-top:1px solid #e5e7eb"></td></tr>
          <tr>
            <td style="padding:4px 0;color:#555"><strong>Human samples</strong></td>
            <td style="font-family:monospace;font-weight:bold;color:#1e40af">{eta_human_days:.1f} days ({fmt(human_remaining)} remaining)</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#888;font-size:11px">Est. completion</td>
            <td style="font-family:monospace;font-size:11px;color:#888">{eta_completion_human.strftime('%B %d, %Y')}</td>
          </tr>
          <tr><td colspan="2" style="padding:4px 0"></td></tr>
          <tr>
            <td style="padding:4px 0;color:#555"><strong>All catalog samples</strong></td>
            <td style="font-family:monospace;font-weight:bold;color:#7c3aed">{eta_all_days:.1f} days ({fmt(samples_remaining)} remaining)</td>
          </tr>
          <tr>
            <td style="padding:4px 0;color:#888;font-size:11px">Est. completion</td>
            <td style="font-family:monospace;font-size:11px;color:#888">{eta_completion_all.strftime('%B %d, %Y')}</td>
          </tr>
          {expected_cells_html}
        </table>
      </div>

      <p style="font-size:10px;color:#92400e;margin:8px 0 0 0">
        ETAs use {rate_window} throughput rate ({throughput_rate:.1f}/hr) sustained 24/7.
        Actual pace varies with sample size, download speed, and cluster load.
      </p>
    </div>"""

    fail_pct = n_failed / n_processed * 100 if n_processed > 0 else 0

    body = f"""\
<html>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:700px;margin:0 auto;padding:20px;color:#1a1a2e;line-height:1.5">

<div style="border-bottom:3px solid #0d9488;padding-bottom:12px;margin-bottom:20px">
  <h1 style="margin:0;font-size:22px;color:#1a1a2e">Singlet Atlas &mdash; Pipeline Report</h1>
  <p style="margin:4px 0 0 0;font-size:12px;color:#888">
    {now_str} &middot;
    <a href="{DASHBOARD_URL}" style="color:#0d9488">Live Dashboard</a>
  </p>
</div>

<!-- HEADLINE METRICS -->
<table style="width:100%;border-collapse:collapse;margin-bottom:16px">
  <tr>
    <td style="text-align:center;padding:12px;background:#f0fdfa;border:1px solid #ccfbf1;border-radius:8px">
      <div style="font-family:monospace;font-size:28px;font-weight:bold;color:#0d9488">{fmt_big(total_cells)}</div>
      <div style="font-size:11px;color:#888;margin-top:2px">Total Cells</div>
    </td>
    <td style="width:8px"></td>
    <td style="text-align:center;padding:12px;background:#f0fdfa;border:1px solid #ccfbf1;border-radius:8px">
      <div style="font-family:monospace;font-size:28px;font-weight:bold;color:#1a1a2e">{fmt(samples_quantified)}</div>
      <div style="font-size:11px;color:#888;margin-top:2px">Samples (.spz)</div>
    </td>
    <td style="width:8px"></td>
    <td style="text-align:center;padding:12px;background:#f0fdfa;border:1px solid #ccfbf1;border-radius:8px">
      <div style="font-family:monospace;font-size:28px;font-weight:bold;color:#1a1a2e">{fmt(avg_cells)}</div>
      <div style="font-size:11px;color:#888;margin-top:2px">Avg Cells/Sample</div>
    </td>
  </tr>
</table>

<!-- BATCH PROGRESS -->
<div style="margin-bottom:20px">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
    <span style="color:#555"><strong>Catalog Batch Progress</strong></span>
    <span style="font-family:monospace;font-weight:bold">{pct_batches:.1f}% ({n_completed}/{n_total_batches} batches)</span>
  </div>
  <div style="background:#e5e7eb;border-radius:6px;height:18px;overflow:hidden;position:relative">
    <div style="background:linear-gradient(90deg,#0d9488,#14b8a6);height:100%;width:{bar_width}%;border-radius:6px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#888;margin-top:4px">
    <span>{n_completed} done ({fmt(samples_completed_batches)} samples)</span>
    <span>{n_in_progress} active ({fmt(samples_in_progress)} samples)</span>
    <span>{n_not_started} queued ({fmt(samples_not_started)} samples)</span>
  </div>
</div>

{eta_section}

<!-- TODAY'S THROUGHPUT -->
<div style="margin:16px 0;padding:14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
  <strong style="font-size:13px">Today's Processing ({now.strftime('%b %d')})</strong>
  <table style="font-size:12px;margin-top:6px;margin-bottom:10px">
    <tr><td style="padding:2px 12px 2px 0;color:#555">Successful</td><td style="font-family:monospace;font-weight:bold;color:#16a34a">{success_today}</td></tr>
    <tr><td style="padding:2px 12px 2px 0;color:#555">Failed</td><td style="font-family:monospace;color:#dc2626">{failed_today}</td></tr>
    <tr><td style="padding:2px 12px 2px 0;color:#555">Skipped</td><td style="font-family:monospace;color:#d97706">{skipped_today}</td></tr>
    <tr><td style="padding:2px 12px 2px 0;color:#555">Total processed</td><td style="font-family:monospace">{total_today}</td></tr>
  </table>
  <div style="font-size:11px;color:#888;margin-bottom:4px">Successes by hour (red = failed/skipped):</div>
  <table style="border-collapse:collapse">{hourly_chart}</table>
</div>

<!-- FAILURE BREAKDOWN -->
<div style="margin:16px 0;padding:14px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px">
  <strong style="font-size:13px;color:#991b1b">Failure Analysis ({n_failed} failures / {n_processed} processed = {fail_pct:.1f}%)</strong>

  <div style="margin-top:10px">
    <div style="font-size:12px;font-weight:bold;color:#555;margin-bottom:4px">By Error Category</div>
    <table style="width:100%;font-size:12px">{failure_rows}</table>
  </div>

  <div style="margin-top:12px;padding-top:8px;border-top:1px solid #fecaca">
    <div style="font-size:12px;font-weight:bold;color:#555;margin-bottom:4px">By Protocol</div>
    <table style="width:100%;font-size:11px">{fail_proto_rows}</table>
  </div>

  <div style="margin-top:12px;padding-top:8px;border-top:1px solid #fecaca">
    <div style="font-size:12px;font-weight:bold;color:#555;margin-bottom:4px">Skip Reasons ({n_skipped} skipped)</div>
    <table style="width:100%;font-size:12px">{skip_rows}</table>
  </div>
</div>

<!-- ACTIVE COMPUTE -->
<div style="margin:16px 0;padding:14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
  <strong style="font-size:13px">Active Compute</strong>
  <div style="font-size:12px;color:#555;margin:6px 0">
    <strong>{slurm['total_running']}</strong> running, <strong>{slurm['total_pending']}</strong> pending across
    <strong>{len(slurm['nodes'])}</strong> nodes ({', '.join(slurm['nodes']) or 'none'})
  </div>
  <table style="width:100%;font-size:12px;margin-top:8px">{slurm_rows}</table>
</div>

<!-- SPECIES PROGRESS -->
<div style="margin:16px 0;padding:14px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px">
  <strong style="font-size:13px;color:#1e40af">Species Progress</strong>
  <table style="width:100%;font-size:12px;margin-top:8px;border-collapse:collapse">
    <tr style="color:#888;font-size:11px">
      <th style="text-align:left;padding:4px 8px 4px 0;font-weight:normal">Species</th>
      <th style="text-align:right;padding:4px 8px;font-weight:normal">Total</th>
      <th style="text-align:right;padding:4px 8px;font-weight:normal">Done</th>
      <th style="text-align:right;padding:4px 8px;font-weight:normal">Active</th>
      <th style="text-align:right;padding:4px 0;font-weight:normal">Remaining</th>
    </tr>
    <tr style="font-weight:bold">
      <td style="padding:4px 8px 4px 0;font-style:italic">Homo sapiens</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace">{fmt(human_total)}</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace;color:#16a34a">{fmt(human_completed)}</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace;color:#d97706">{fmt(human_in_progress)}</td>
      <td style="text-align:right;padding:4px 0;font-family:monospace;color:#dc2626">{fmt(human_remaining)}</td>
    </tr>
    <tr>
      <td style="padding:4px 8px 4px 0;font-style:italic">Mus musculus</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace">{fmt(mouse_total)}</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace;color:#16a34a">{fmt(mouse_completed)}</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace;color:#d97706">{fmt(mouse_in_progress)}</td>
      <td style="text-align:right;padding:4px 0;font-family:monospace;color:#dc2626">{fmt(mouse_remaining)}</td>
    </tr>
    <tr style="color:#555">
      <td style="padding:4px 8px 4px 0">Other species</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace">{fmt(samples_total - human_total - mouse_total)}</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace;color:#16a34a">{fmt(samples_completed_batches - human_completed - mouse_completed)}</td>
      <td style="text-align:right;padding:4px 8px;font-family:monospace;color:#d97706">{fmt(samples_in_progress - human_in_progress - mouse_in_progress)}</td>
      <td style="text-align:right;padding:4px 0;font-family:monospace">{fmt(samples_not_started - human_remaining - mouse_remaining)}</td>
    </tr>
  </table>
</div>

<!-- PER-SPECIES CELLS + PROTOCOLS -->
<div style="margin:16px 0;padding:14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
  <strong style="font-size:13px">Corpus: Cells by Species &amp; Protocol</strong>
  <table style="width:100%;font-size:12px;margin-top:8px;border-collapse:collapse">
    <tr style="color:#888;font-size:11px">
      <th style="text-align:left;padding:2px 8px 2px 0;font-weight:normal">Species</th>
      <th style="text-align:right;padding:2px 8px;font-weight:normal">Samples</th>
      <th style="text-align:right;padding:2px 8px;font-weight:normal">Cells</th>
      <th style="text-align:left;padding:2px 0 2px 8px;font-weight:normal">Top Protocols</th>
    </tr>
    {species_detail_rows}
  </table>
</div>

<!-- REMAINING WORK -->
<div style="margin:16px 0;padding:14px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
  <strong style="font-size:13px">Remaining Work</strong>
  <div style="display:flex;gap:24px;margin-top:10px">
    <div>
      <div style="font-size:12px;font-weight:bold;color:#555;margin-bottom:4px">By Organism (top 10)</div>
      <table style="font-size:12px">{remaining_org_rows}</table>
    </div>
    <div>
      <div style="font-size:12px;font-weight:bold;color:#555;margin-bottom:4px">By Protocol</div>
      <table style="font-size:12px">{proto_rows}</table>
    </div>
  </div>
</div>

<div style="margin-top:24px;padding-top:12px;border-top:1px solid #e5e7eb;font-size:11px;color:#aaa;text-align:center">
  Singlet AI &middot; Pipeline Report &middot; Auto-generated<br>
  <a href="{DASHBOARD_URL}" style="color:#0d9488">View live dashboard</a>
</div>

</body>
</html>"""

    pct_display = f"{pct_batches:.1f}%"
    if throughput_rate > 0:
        subject = (
            f"Singlet Atlas: {fmt_big(total_cells)} cells | "
            f"{n_completed}/{n_total_batches} batches ({pct_display}) | "
            f"{throughput_rate:.0f}/hr | "
            f"ETA {eta_all_days:.0f}d | "
            f"{n_failed} failures"
        )
    else:
        subject = (
            f"Singlet Atlas: {fmt_big(total_cells)} cells | "
            f"{n_completed}/{n_total_batches} batches ({pct_display}) | "
            f"{slurm['total_running']} tasks running | "
            f"{n_failed} failures"
        )

    return subject, body


def send_email(subject, body, recipient, from_addr):
    message = f"""\
From: Singlet Pipeline <{from_addr}>
To: {recipient}
Subject: {subject}
MIME-Version: 1.0
Content-Type: text/html; charset=utf-8

{body}"""

    proc = subprocess.run(
        ["/usr/sbin/sendmail", "-t"],
        input=message.encode("utf-8"),
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        print(f"sendmail failed (rc={proc.returncode}): {proc.stderr.decode()}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print email to stdout instead of sending")
    parser.add_argument("--to", default=RECIPIENT, help=f"Override recipient (default: {RECIPIENT})")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    print(f"[{now.isoformat()}] Collecting batch status...", file=sys.stderr)
    completed, in_progress, not_started = collect_batch_status()
    print(f"  Batches: {len(completed)} done, {len(in_progress)} active, {len(not_started)} queued", file=sys.stderr)

    print(f"[{now.isoformat()}] Analyzing result CSVs...", file=sys.stderr)
    result_details = collect_result_details()
    print(f"  Processed: {sum(result_details['statuses'].values())} rows, "
          f"{result_details['statuses'].get('failed', 0)} failures", file=sys.stderr)

    print(f"[{now.isoformat()}] Parsing log completions...", file=sys.stderr)
    completions = collect_log_completions()
    print(f"  Found {len(completions)} completion entries", file=sys.stderr)

    print(f"[{now.isoformat()}] Collecting SLURM state...", file=sys.stderr)
    slurm = collect_slurm_jobs()
    print(f"  {slurm['total_running']} running, {slurm['total_pending']} pending", file=sys.stderr)

    print(f"[{now.isoformat()}] Loading metrics history...", file=sys.stderr)
    history, corpus = load_metrics_history()

    subject, body = build_html(completed, in_progress, not_started,
                               completions, slurm, history, corpus, result_details)

    if args.dry_run:
        print(f"To: {args.to}")
        print(f"Subject: {subject}")
        print()
        print(body)
        return

    send_email(subject, body, args.to, FROM_ADDR)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Sent detailed status email to {args.to}", file=sys.stderr)


if __name__ == "__main__":
    main()
