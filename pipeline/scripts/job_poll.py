#!/usr/bin/env python3
"""
job_poll.py — Poll pending async jobs and report results.

Called by the orchestrator at Phase 0 (harvest) to check which submitted
batch jobs have completed. Reads from state/jobs/<tag>/ directories.

Usage:
    # Show all jobs with their status (human-readable)
    python3 job_poll.py status

    # Return only newly completed jobs since last poll (machine-readable JSON)
    python3 job_poll.py harvest

    # Show full result for a specific job
    python3 job_poll.py result TAG

    # Archive completed jobs older than N hours (default 48)
    python3 job_poll.py archive [--hours 48]

    # Compact summary for Phase 0 context (≤15 lines)
    python3 job_poll.py summary

    # Check for stale/runaway jobs and auto-cancel (>3x expected duration)
    python3 job_poll.py watchdog

    # Retry a failed job (re-submit from saved SLURM script)
    python3 job_poll.py retry TAG [--suffix r2]

    # Generate a resume block for agent_checkpoint (pending + recent jobs)
    python3 job_poll.py resume-block
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
JOBS_DIR = SCRIPT_DIR.parent / "state" / "jobs"
ARCHIVE_DIR = JOBS_DIR / "archive"

# Stale job multiplier: if wall > expected_duration * STALE_FACTOR, flag as stale
STALE_FACTOR = 3.0
# Default expected duration if not specified (seconds)
DEFAULT_EXPECTED = 300


def load_job(tag_dir):
    """Load job metadata and result from a tag directory."""
    meta_path = tag_dir / "job.meta"
    result_path = tag_dir / "result.json"

    job = {"tag": tag_dir.name, "dir": str(tag_dir)}

    if meta_path.exists():
        try:
            job["meta"] = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            job["meta"] = {"status": "corrupt"}
    else:
        job["meta"] = {"status": "missing_meta"}

    if result_path.exists():
        try:
            job["result"] = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            job["result"] = None
    else:
        job["result"] = None

    # Determine effective status
    meta_status = job["meta"].get("status", "unknown")
    if job["result"] is not None:
        exit_code = job["result"].get("exit_code")
        if exit_code == 0:
            job["effective_status"] = "success"
        else:
            job["effective_status"] = "failed"
    elif meta_status == "pending":
        # Check if SLURM job is still running
        jid = job["meta"].get("slurm_job_id", "")
        if jid and jid != "unknown":
            try:
                out = subprocess.run(
                    ["squeue", "-j", str(jid), "-h", "-o", "%T"],
                    capture_output=True, text=True, timeout=5
                )
                state = out.stdout.strip()
                if state:
                    job["effective_status"] = f"running ({state})"
                else:
                    # Job finished but no result.json → crashed
                    job["effective_status"] = "crashed (no result)"
            except Exception:
                job["effective_status"] = "pending (squeue failed)"
        else:
            job["effective_status"] = "pending"
    else:
        job["effective_status"] = meta_status

    return job


def all_jobs():
    """Load all jobs from state/jobs/."""
    if not JOBS_DIR.exists():
        return []
    jobs = []
    for d in sorted(JOBS_DIR.iterdir()):
        if d.is_dir() and d.name != "archive":
            jobs.append(load_job(d))
    return jobs


def cmd_status(_args):
    """Show all jobs with status."""
    jobs = all_jobs()
    if not jobs:
        print("(no jobs)")
        return

    print(f"{'TAG':<35} {'STATUS':<20} {'EXIT':>5} {'WALL':>8} {'SUBMITTED':<20}")
    print(f"{'---':<35} {'------':<20} {'----':>5} {'----':>8} {'---------':<20}")
    for j in jobs:
        tag = j["tag"]
        status = j["effective_status"]
        exit_code = j["result"]["exit_code"] if j["result"] else "-"
        wall = f"{j['result']['wall_seconds']:.1f}s" if j["result"] and "wall_seconds" in j["result"] else "-"
        submitted = j["meta"].get("submitted", "?")[:19]
        print(f"{tag:<35} {status:<20} {str(exit_code):>5} {wall:>8} {submitted:<20}")


def cmd_harvest(_args):
    """Return newly completed jobs (have result.json, meta still says 'pending' or 'done').
    
    After harvest, marks jobs as 'harvested' so they're not returned again.
    Returns JSON array to stdout for machine consumption.
    """
    jobs = all_jobs()
    completed = []

    for j in jobs:
        if j["result"] is None:
            continue
        meta_status = j["meta"].get("status", "")
        if meta_status in ("harvested", "cancelled"):
            continue

        # This is a newly completed job
        completed.append({
            "tag": j["tag"],
            "exit_code": j["result"].get("exit_code"),
            "wall_seconds": j["result"].get("wall_seconds"),
            "completed": j["result"].get("completed"),
            "node": j["result"].get("node"),
            "stdout_tail": j["result"].get("stdout_tail", ""),
        })

        # Mark as harvested
        meta_path = Path(j["dir"]) / "job.meta"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["status"] = "harvested"
            meta["harvested_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            meta_path.write_text(json.dumps(meta, indent=2))

    print(json.dumps(completed, indent=2))

    if completed:
        print(f"\n[harvest] {len(completed)} job(s) harvested", file=sys.stderr)
    else:
        print("\n[harvest] no new results", file=sys.stderr)


def cmd_result(args):
    """Show full result for a specific job."""
    tag = args.tag
    tag_dir = JOBS_DIR / tag
    if not tag_dir.exists():
        print(f"No job found: {tag}", file=sys.stderr)
        sys.exit(1)

    job = load_job(tag_dir)
    print(f"Tag: {job['tag']}")
    print(f"Status: {job['effective_status']}")
    if job["result"]:
        print(f"Exit: {job['result'].get('exit_code')}")
        print(f"Wall: {job['result'].get('wall_seconds')}s")
        print(f"Completed: {job['result'].get('completed')}")
        tail = job["result"].get("stdout_tail", "")
        if tail:
            print(f"\n--- stdout tail ---\n{tail}")
    else:
        print("(no result yet)")

    # Also show full stdout if available
    stdout_path = tag_dir / "stdout.log"
    if stdout_path.exists():
        lines = stdout_path.read_text().strip().split("\n")
        print(f"\n--- stdout.log ({len(lines)} lines, last 30) ---")
        for line in lines[-30:]:
            print(f"  {line}")


def _job_age_seconds(job):
    """Return seconds since job submission, or None."""
    submitted = job["meta"].get("submitted", "")
    if not submitted:
        return None
    try:
        sub_time = datetime.fromisoformat(submitted.replace("Z", "+00:00")).replace(tzinfo=None)
        return (datetime.utcnow() - sub_time).total_seconds()
    except ValueError:
        return None


def _classify_failure(job):
    """Categorize why a job failed: 'code_failure', 'oom', 'timeout', 'slurm_error', 'crashed'."""
    if job["effective_status"].startswith("crashed"):
        return "crashed"
    if job["result"] is None:
        return "unknown"
    exit_code = job["result"].get("exit_code", -1)
    stdout_tail = job["result"].get("stdout_tail", "")

    # Check for OOM patterns
    if "oom" in stdout_tail.lower() or "out of memory" in stdout_tail.lower() or exit_code == 137:
        return "oom"
    # Check for timeout from SLURM
    if exit_code == 124 or "DUE TO TIME LIMIT" in stdout_tail.upper():
        return "timeout"
    # Generic SLURM infrastructure errors
    if exit_code in (126, 127, 130):
        return "slurm_error"
    # Everything else is a code failure (exit != 0)
    if exit_code != 0:
        return "code_failure"
    return "success"


def cmd_summary(_args):
    """Compact summary for Phase 0 context (≤15 lines)."""
    jobs = all_jobs()
    if not jobs:
        print("ASYNC JOBS: none")
        return

    # Fixed: properly classify pending/running/crashed/ready
    running = []
    crashed = []
    ready = []
    harvested = []
    for j in jobs:
        status = j["effective_status"]
        meta_status = j["meta"].get("status", "")
        if meta_status == "harvested":
            harvested.append(j)
        elif j["result"] is not None and meta_status != "harvested":
            ready.append(j)
        elif status.startswith("crashed"):
            crashed.append(j)
        elif "running" in status or status == "pending" or "squeue" in status:
            running.append(j)

    parts = [f"{len(running)} running", f"{len(ready)} ready"]
    if crashed:
        parts.append(f"{len(crashed)} crashed")
    parts.append(f"{len(harvested)} harvested")
    print(f"ASYNC JOBS: {', '.join(parts)}")

    if ready:
        print("\nREADY TO HARVEST:")
        for j in ready:
            exit_code = j["result"]["exit_code"]
            wall = j["result"].get("wall_seconds", "?")
            icon = "✅" if exit_code == 0 else "❌"
            dag = j["meta"].get("dag_task", "")
            suffix = f" [{dag}]" if dag else ""
            print(f"  {icon} {j['tag']}: exit={exit_code} wall={wall}s{suffix}")

    if running:
        print("\nSTILL RUNNING:")
        for j in running:
            age = _job_age_seconds(j)
            expected = j["meta"].get("expected_duration", 0)
            age_str = f"{int(age)}s ago" if age else "?"
            exp_str = f" (expected ~{expected}s)" if expected else ""
            stale = ""
            if age and expected and age > expected * STALE_FACTOR:
                stale = " ⚠️ STALE"
            print(f"  ⏳ {j['tag']}: submitted {age_str}{exp_str}{stale}")

    if crashed:
        print("\nCRASHED (no result.json):")
        for j in crashed:
            age = _job_age_seconds(j)
            age_str = f"{int(age)}s ago" if age else "?"
            print(f"  💥 {j['tag']}: submitted {age_str}")


def cmd_archive(args):
    """Archive completed jobs older than N hours."""
    hours = args.hours
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    archived = 0
    for j in all_jobs():
        if j["meta"].get("status") not in ("harvested", "done", "failed"):
            continue
        submitted = j["meta"].get("submitted", "")
        if not submitted:
            continue
        try:
            sub_time = datetime.fromisoformat(submitted.replace("Z", "+00:00")).replace(tzinfo=None)
            if sub_time < cutoff:
                src = Path(j["dir"])
                dst = ARCHIVE_DIR / j["tag"]
                src.rename(dst)
                archived += 1
        except (ValueError, OSError):
            continue

    print(f"[archive] Archived {archived} job(s) older than {hours}h")


def cmd_watchdog(_args):
    """Flag stale/runaway jobs. Auto-cancel if >3x expected duration."""
    jobs = all_jobs()
    stale = []
    for j in jobs:
        status = j["effective_status"]
        if not ("running" in status or status == "pending"):
            continue
        age = _job_age_seconds(j)
        expected = j["meta"].get("expected_duration", 0)
        if not age or not expected:
            continue
        if age > expected * STALE_FACTOR:
            stale.append((j, age, expected))

    if not stale:
        print("WATCHDOG: no stale jobs")
        return

    print(f"WATCHDOG: {len(stale)} stale job(s)")
    for j, age, expected in stale:
        ratio = age / expected
        jid = j["meta"].get("slurm_job_id", "")
        print(f"  ⚠️  {j['tag']}: {int(age)}s elapsed / {expected}s expected ({ratio:.1f}x) [slurm={jid}]")

        # Auto-cancel if >5x expected
        if ratio > 5.0 and jid and jid != "unknown":
            try:
                subprocess.run(["scancel", str(jid)], capture_output=True, timeout=5)
                meta_path = Path(j["dir"]) / "job.meta"
                meta = json.loads(meta_path.read_text())
                meta["status"] = "cancelled"
                meta["cancel_reason"] = f"watchdog: {ratio:.1f}x expected duration"
                meta_path.write_text(json.dumps(meta, indent=2))
                print(f"    → AUTO-CANCELLED (>{5.0}x expected)")
            except Exception as e:
                print(f"    → cancel failed: {e}")


def cmd_retry(args):
    """Re-submit a failed job with an optional suffix."""
    tag = args.tag
    suffix = args.suffix or "r2"
    tag_dir = JOBS_DIR / tag

    if not tag_dir.exists():
        print(f"No job found: {tag}", file=sys.stderr)
        sys.exit(1)

    job = load_job(tag_dir)
    if job["effective_status"] not in ("failed", "crashed (no result)"):
        print(f"Job {tag} status is '{job['effective_status']}', not failed/crashed", file=sys.stderr)
        sys.exit(1)

    slurm_script = tag_dir / "job.slurm"
    if not slurm_script.exists():
        print(f"No SLURM script found in {tag_dir}", file=sys.stderr)
        sys.exit(1)

    # Create new tag with suffix
    new_tag = f"{tag}-{suffix}"
    new_dir = JOBS_DIR / new_tag
    if new_dir.exists():
        print(f"Retry tag already exists: {new_tag}", file=sys.stderr)
        sys.exit(1)

    new_dir.mkdir(parents=True)

    # Copy and patch SLURM script
    script_text = slurm_script.read_text()
    script_text = script_text.replace(f"sf_{tag}", f"sf_{new_tag}")
    script_text = script_text.replace(f'"{tag}"', f'"{new_tag}"')
    script_text = script_text.replace(str(tag_dir), str(new_dir))
    new_script = new_dir / "job.slurm"
    new_script.write_text(script_text)

    # Copy metadata with updated tag and retry info
    meta = dict(job["meta"])
    meta["tag"] = new_tag
    meta["status"] = "pending"
    meta["submitted"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["retry_of"] = tag
    meta["failure_class"] = _classify_failure(job)
    (new_dir / "job.meta").write_text(json.dumps(meta, indent=2))

    # Submit
    try:
        result = subprocess.run(
            ["sbatch", "--parsable", str(new_script)],
            capture_output=True, text=True, timeout=10
        )
        jid = result.stdout.strip()
        if jid.isdigit():
            meta["slurm_job_id"] = jid
            (new_dir / "job.meta").write_text(json.dumps(meta, indent=2))
            print(f"RETRIED: {new_tag} job_id={jid} (retry of {tag}, failure={meta['failure_class']})")
        else:
            print(f"SUBMIT FAILED: {result.stderr}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"SUBMIT FAILED: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_resume_block(_args):
    """Generate a compact resume block for agent_checkpoint.py.
    
    Outputs lines suitable for embedding in RESUME.txt so new sessions
    know about in-flight async work immediately.
    """
    jobs = all_jobs()
    if not jobs:
        return

    running = [j for j in jobs if "running" in j["effective_status"]
               or j["effective_status"] == "pending"
               or "squeue" in j["effective_status"]]
    ready = [j for j in jobs if j["result"] is not None and j["meta"].get("status") != "harvested"]
    crashed = [j for j in jobs if j["effective_status"].startswith("crashed")]
    failed = [j for j in jobs if j["effective_status"] == "failed" and j["meta"].get("status") != "harvested"]

    if not (running or ready or crashed or failed):
        return

    print("## Pending Async Jobs")
    print("")

    if ready:
        print(f"**{len(ready)} job(s) ready to harvest** — run `job_poll.py harvest` first:")
        for j in ready:
            ec = j["result"]["exit_code"]
            icon = "✅" if ec == 0 else "❌"
            dag = j["meta"].get("dag_task", "")
            cycle = j["meta"].get("cycle", 0)
            suffix = f" [cycle {cycle}, {dag}]" if (dag or cycle) else ""
            print(f"  {icon} {j['tag']}: exit={ec}{suffix}")
        print("")

    if running:
        print(f"**{len(running)} job(s) still running:**")
        for j in running:
            age = _job_age_seconds(j)
            expected = j["meta"].get("expected_duration", 0)
            age_str = f"{int(age)}s" if age else "?"
            exp_str = f" / expected ~{expected}s" if expected else ""
            print(f"  ⏳ {j['tag']}: running {age_str}{exp_str}")
        print("")

    if crashed:
        print(f"**{len(crashed)} job(s) crashed (no result.json) — investigate:**")
        for j in crashed:
            print(f"  💥 {j['tag']}: check stdout.log")
        print("")

    if failed:
        print(f"**{len(failed)} job(s) failed — consider retry:**")
        for j in failed:
            ec = j["result"]["exit_code"]
            cls = _classify_failure(j)
            print(f"  ❌ {j['tag']}: exit={ec} ({cls})")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Poll async batch jobs")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show all jobs")
    sub.add_parser("harvest", help="Return newly completed jobs (JSON)")
    sub.add_parser("summary", help="Compact summary for Phase 0")
    sub.add_parser("watchdog", help="Check for stale/runaway jobs")
    sub.add_parser("resume-block", help="Generate resume block for checkpoint")

    rp = sub.add_parser("result", help="Show result for a specific job")
    rp.add_argument("tag", help="Job tag")

    retp = sub.add_parser("retry", help="Re-submit a failed job")
    retp.add_argument("tag", help="Job tag to retry")
    retp.add_argument("--suffix", default="r2", help="Suffix for retry tag (default: r2)")

    ap = sub.add_parser("archive", help="Archive old completed jobs")
    ap.add_argument("--hours", type=int, default=48, help="Archive jobs older than N hours")

    args = parser.parse_args()

    cmds = {
        "status": cmd_status,
        "harvest": cmd_harvest,
        "result": cmd_result,
        "summary": cmd_summary,
        "archive": cmd_archive,
        "watchdog": cmd_watchdog,
        "retry": cmd_retry,
        "resume-block": cmd_resume_block,
    }
    fn = cmds.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
