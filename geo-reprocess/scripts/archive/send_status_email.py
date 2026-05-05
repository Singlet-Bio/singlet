#!/usr/bin/env python3
"""Send pipeline status email report using data from pipeline-metrics.json.

Sends an HTML email summary of corpus stats and active compute every 6 hours.
Reads the same metrics JSON that powers the investor dashboard.

Usage:
    python send_status_email.py              # Send report
    python send_status_email.py --dry-run    # Print email to stdout
    python send_status_email.py --test       # Send with [TEST] subject prefix
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────
METRICS_JSON = Path("/mnt/home/debruinz/Singlet-AI/singletai-website/public/data/pipeline-metrics.json")
RECIPIENT = "debruinz@gvsu.edu"
FROM_ADDR = "debruinz@gvsu.edu"
DASHBOARD_URL = "https://singletai.com/invest#corpus"


def fmt(n: int) -> str:
    """Format number with commas."""
    return f"{n:,}"


def fmt_big(n: int) -> str:
    """Format large number in human-readable form."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return fmt(n)


def compute_rate(history: list) -> Optional[dict]:
    """Compute processing rate from history entries."""
    if len(history) < 2:
        return None
    latest = history[-1]
    earliest = history[0]
    t_latest = datetime.fromisoformat(latest["timestamp"])
    t_earliest = datetime.fromisoformat(earliest["timestamp"])
    hours = (t_latest - t_earliest).total_seconds() / 3600
    if hours <= 0:
        return None
    samples_diff = latest["samples_quantified"] - earliest["samples_quantified"]
    cells_diff = latest["total_cells"] - earliest["total_cells"]
    return {
        "samples_per_hour": round(samples_diff / hours, 1),
        "cells_per_hour": round(cells_diff / hours),
        "hours_tracked": round(hours, 1),
        "samples_delta": samples_diff,
        "cells_delta": cells_diff,
    }


def build_email(metrics: dict, is_test: bool = False) -> tuple:
    """Build email subject and HTML body from metrics."""
    corpus = metrics["corpus"]
    compute = metrics["compute"]
    history = metrics.get("history", [])
    updated = metrics["last_updated"]

    pct = (corpus["samples_quantified"] / corpus["total_samples"] * 100) if corpus["total_samples"] > 0 else 0
    rate = compute_rate(history)

    # Top species by cells
    species_sorted = sorted(
        corpus["species"].items(),
        key=lambda x: x[1]["cells"],
        reverse=True,
    )[:6]

    prefix = "[TEST] " if is_test else ""
    subject = (
        f"{prefix}Singlet Atlas: {fmt_big(corpus['total_cells'])} cells | "
        f"{fmt(corpus['samples_quantified'])}/{fmt(corpus['total_samples'])} samples ({pct:.1f}%) | "
        f"{compute['running_tasks']} tasks running"
    )

    # ── HTML body ──
    species_rows = ""
    for name, data in species_sorted:
        species_rows += f"""
        <tr>
          <td style="padding:4px 12px 4px 0;font-style:italic;color:#333">{name}</td>
          <td style="padding:4px 12px;font-family:monospace;text-align:right">{fmt(data['samples'])}</td>
          <td style="padding:4px 0 4px 12px;font-family:monospace;text-align:right">{fmt_big(data['cells'])}</td>
        </tr>"""

    job_rows = ""
    for job in compute.get("job_arrays", []):
        status = f"{job['running']} running"
        if job["pending"] > 0:
            status += f", {job['pending']} queued"
        job_rows += f"""
        <tr>
          <td style="padding:4px 12px 4px 0;font-family:monospace;color:#333">{job['name']}</td>
          <td style="padding:4px 0 4px 12px">{status}</td>
        </tr>"""

    rate_section = ""
    if rate and rate["samples_per_hour"] > 0:
        remaining = corpus["total_samples"] - corpus["samples_quantified"]
        eta_hours = remaining / rate["samples_per_hour"] if rate["samples_per_hour"] > 0 else 0
        eta_days = eta_hours / 24
        rate_section = f"""
    <tr><td colspan="2" style="padding:16px 0 8px 0">
      <strong style="color:#0d9488">Processing Rate</strong>
      <span style="color:#888;font-size:12px">(over {rate['hours_tracked']}h)</span>
    </td></tr>
    <tr>
      <td style="padding:4px 12px 4px 0;color:#555">Samples / hour</td>
      <td style="padding:4px 0;font-family:monospace;font-weight:bold;color:#0d9488">{rate['samples_per_hour']}</td>
    </tr>
    <tr>
      <td style="padding:4px 12px 4px 0;color:#555">Cells / hour</td>
      <td style="padding:4px 0;font-family:monospace;font-weight:bold;color:#0d9488">{fmt_big(rate['cells_per_hour'])}</td>
    </tr>
    <tr>
      <td style="padding:4px 12px 4px 0;color:#555">Since tracking started</td>
      <td style="padding:4px 0;font-family:monospace">+{fmt(rate['samples_delta'])} samples, +{fmt_big(rate['cells_delta'])} cells</td>
    </tr>
    <tr>
      <td style="padding:4px 12px 4px 0;color:#555">ETA to completion</td>
      <td style="padding:4px 0;font-family:monospace">{eta_days:.0f} days ({fmt(remaining)} remaining)</td>
    </tr>"""

    bar_width = min(pct, 100)
    nodes_str = ", ".join(compute.get("nodes_list", [])) or "none"

    body = f"""\
<html>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:640px;margin:0 auto;padding:20px;color:#1a1a2e;line-height:1.5">

<div style="border-bottom:3px solid #0d9488;padding-bottom:12px;margin-bottom:20px">
  <h1 style="margin:0;font-size:22px;color:#1a1a2e">Singlet Atlas — Pipeline Report</h1>
  <p style="margin:4px 0 0 0;font-size:12px;color:#888">
    {datetime.fromisoformat(updated).strftime('%B %d, %Y %H:%M UTC')} ·
    <a href="{DASHBOARD_URL}" style="color:#0d9488">Live Dashboard →</a>
  </p>
</div>

<!-- Progress bar -->
<div style="margin-bottom:20px">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
    <span style="color:#555">Overall Progress</span>
    <span style="font-family:monospace;font-weight:bold">{pct:.1f}%</span>
  </div>
  <div style="background:#e5e7eb;border-radius:6px;height:14px;overflow:hidden">
    <div style="background:linear-gradient(90deg,#0d9488,#14b8a6);height:100%;width:{bar_width}%;border-radius:6px;transition:width 0.5s"></div>
  </div>
  <div style="font-size:11px;color:#888;margin-top:3px">
    {fmt(corpus['samples_quantified'])} of {fmt(corpus['total_samples'])} samples quantified
  </div>
</div>

<!-- Key metrics -->
<table style="width:100%;border-collapse:collapse;margin-bottom:16px">
  <tr>
    <td style="text-align:center;padding:12px;background:#f8fffe;border:1px solid #e5e7eb;border-radius:8px">
      <div style="font-family:monospace;font-size:28px;font-weight:bold;color:#0d9488">{fmt_big(corpus['total_cells'])}</div>
      <div style="font-size:11px;color:#888;margin-top:2px">Total Cells</div>
    </td>
    <td style="width:8px"></td>
    <td style="text-align:center;padding:12px;background:#f8fffe;border:1px solid #e5e7eb;border-radius:8px">
      <div style="font-family:monospace;font-size:28px;font-weight:bold;color:#1a1a2e">{fmt(corpus['total_series'])}</div>
      <div style="font-size:11px;color:#888;margin-top:2px">GEO Series</div>
    </td>
    <td style="width:8px"></td>
    <td style="text-align:center;padding:12px;background:#f8fffe;border:1px solid #e5e7eb;border-radius:8px">
      <div style="font-family:monospace;font-size:28px;font-weight:bold;color:#1a1a2e">{len(corpus['species'])}</div>
      <div style="font-size:11px;color:#888;margin-top:2px">Species</div>
    </td>
  </tr>
</table>

<!-- Details table -->
<table style="width:100%;font-size:13px;border-collapse:collapse">
  <tr><td colspan="2" style="padding:12px 0 8px 0"><strong>Corpus</strong></td></tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">Samples quantified</td>
    <td style="padding:4px 0;font-family:monospace">{fmt(corpus['samples_quantified'])}</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">Samples with metadata</td>
    <td style="padding:4px 0;font-family:monospace">{fmt(corpus['samples_with_metadata'])}</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">Avg cells / sample</td>
    <td style="padding:4px 0;font-family:monospace">{fmt(corpus['avg_cells_per_sample'])}</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">QC pass rate</td>
    <td style="padding:4px 0;font-family:monospace">{corpus['qc_pass_rate'] * 100:.1f}%</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">Pipeline version</td>
    <td style="padding:4px 0;font-family:monospace">{corpus['pipeline_version']}</td>
  </tr>

  <tr><td colspan="2" style="padding:16px 0 8px 0"><strong>Active Compute</strong></td></tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">Running tasks</td>
    <td style="padding:4px 0;font-family:monospace;font-weight:bold;color:{'#16a34a' if compute['running_tasks'] > 0 else '#888'}">{compute['running_tasks']}</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">Pending tasks</td>
    <td style="padding:4px 0;font-family:monospace">{compute['pending_tasks']}</td>
  </tr>
  <tr>
    <td style="padding:4px 12px 4px 0;color:#555">Active nodes</td>
    <td style="padding:4px 0;font-family:monospace">{compute['nodes_active']} ({nodes_str})</td>
  </tr>

  {rate_section}
</table>

<!-- Job arrays -->
{"" if not compute.get("job_arrays") else f'''
<div style="margin-top:16px;padding:12px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
  <strong style="font-size:13px">Job Arrays</strong>
  <table style="width:100%;font-size:13px;margin-top:6px">{job_rows}</table>
</div>
'''}

<!-- Species breakdown -->
<div style="margin-top:16px;padding:12px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px">
  <strong style="font-size:13px">Top Species</strong>
  <table style="width:100%;font-size:12px;margin-top:6px">
    <tr style="color:#888">
      <th style="text-align:left;padding:2px 12px 2px 0;font-weight:normal">Species</th>
      <th style="text-align:right;padding:2px 12px;font-weight:normal">Samples</th>
      <th style="text-align:right;padding:2px 0 2px 12px;font-weight:normal">Cells</th>
    </tr>
    {species_rows}
  </table>
</div>

<div style="margin-top:24px;padding-top:12px;border-top:1px solid #e5e7eb;font-size:11px;color:#aaa;text-align:center">
  Singlet AI · Pipeline Status Report · Auto-generated every 6 hours<br>
  <a href="{DASHBOARD_URL}" style="color:#0d9488">View live dashboard</a>
</div>

</body>
</html>"""

    return subject, body


def send_email(subject: str, body: str, recipient: str, from_addr: str) -> None:
    """Send HTML email via sendmail."""
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
    parser.add_argument("--test", action="store_true", help="Send with [TEST] subject prefix")
    parser.add_argument("--to", default=RECIPIENT, help=f"Override recipient (default: {RECIPIENT})")
    args = parser.parse_args()

    if not METRICS_JSON.exists():
        print(f"Metrics file not found: {METRICS_JSON}", file=sys.stderr)
        sys.exit(1)

    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    subject, body = build_email(metrics, is_test=args.test)

    if args.dry_run:
        print(f"To: {args.to}")
        print(f"Subject: {subject}")
        print()
        print(body)
        return

    send_email(subject, body, args.to, FROM_ADDR)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] Sent status email to {args.to}")


if __name__ == "__main__":
    main()
