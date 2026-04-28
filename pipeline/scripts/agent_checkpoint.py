#!/usr/bin/env python3
"""
agent_checkpoint.py — Resume prompt generator for singlify orchestrator.

Reads the authoritative state from singlify/state/ markdown files and generates
a RESUME.txt that can be pasted into a new VS Code agent session. Also supports
archiving old checkpoint data to prevent unbounded growth.

Source of truth: state/dag.md, state/episodes.md, state/context-index.md, state/model-routing.md
This script is a VIEW LAYER — it never writes to state/ files. doc-scribe owns state/ writes.

Usage:
    # Generate resume prompt from state/ files
    python3 agent_checkpoint.py resume-prompt

    # Show current sprint status (compact)
    python3 agent_checkpoint.py status

    # Archive legacy checkpoint.json (one-time migration)
    python3 agent_checkpoint.py archive
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
STATE_DIR = SCRIPT_DIR.parent / "state"
ARCHIVE_DIR = SCRIPT_DIR / "archive"
RESUME_PATH = SCRIPT_DIR / "RESUME.txt"
LEGACY_CHECKPOINT = SCRIPT_DIR / "agent_checkpoint.json"


def parse_episodes(path, last_n=5):
    """Extract last N cycle entries from episodes.md."""
    if not path.exists():
        return []
    text = path.read_text()
    entries = []
    for m in re.finditer(
        r"^## Cycle (\d+) \(([^)]+)\)\s*\n(.*?)(?=\n## Cycle |\Z)",
        text, re.MULTILINE | re.DOTALL
    ):
        cycle_num = int(m.group(1))
        timestamp = m.group(2)
        body = m.group(3).strip()
        tasks = ""
        decisions = ""
        strategy = ""
        for line in body.split("\n"):
            if line.startswith("- **Tasks**:"):
                tasks = line.split(":", 1)[1].strip()
            elif line.startswith("- **Decisions**:"):
                decisions = line.split(":", 1)[1].strip()
            elif line.startswith("- **Strategy patch**:"):
                strategy = line.split(":", 1)[1].strip()
        entries.append({
            "cycle": cycle_num,
            "timestamp": timestamp,
            "tasks": tasks,
            "decisions": decisions,
            "strategy": strategy,
        })
    return entries[-last_n:]


def parse_dag_ready_set(path):
    """Extract Ready Set section from dag.md."""
    if not path.exists():
        return "(dag.md not found)"
    text = path.read_text()
    m = re.search(r"^## Ready Set.*?\n(.*?)(?=\n## |\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else "(no ready set found)"


def parse_dag_gate_status(path):
    """Extract GATE-A status from dag.md."""
    if not path.exists():
        return "unknown"
    text = path.read_text()
    m = re.search(r"GATE-A:\s*(CLOSED|OPEN)", text)
    return m.group(1) if m else "unknown"


def parse_dag_baseline(path):
    """Extract baseline panel table from dag.md."""
    if not path.exists():
        return ""
    text = path.read_text()
    m = re.search(r"^## Baseline Panel.*?\n(.*?)(?=\n## |\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_dag_deferred(path):
    """Extract deferred tasks section."""
    if not path.exists():
        return ""
    text = path.read_text()
    m = re.search(r"^## Deferred.*?\n(.*?)(?=\n---|\n## |\Z)", text, re.MULTILINE | re.DOTALL)
    if m:
        content = m.group(1).strip()
        if content and content != "(none currently)":
            return content
    return ""


def parse_context_system_level(path):
    """Extract system-level summary from context-index.md (first section)."""
    if not path.exists():
        return "(context-index.md not found)"
    text = path.read_text()
    m = re.search(
        r"^## System Level.*?\n(.*?)(?=\n## Module Level|\Z)",
        text, re.MULTILINE | re.DOTALL
    )
    if m:
        lines = m.group(1).strip().split("\n")
        return "\n".join(lines[:15])
    return ""


def get_max_cycle(episodes_path):
    """Get the highest cycle number from episodes.md."""
    entries = parse_episodes(episodes_path, last_n=9999)
    if not entries:
        return 0
    return max(e["cycle"] for e in entries)


def count_dag_status(path):
    """Count task statuses in DAG."""
    if not path.exists():
        return {}
    text = path.read_text()
    return {
        "complete": len(re.findall(r"\U0001F7E2", text)),
        "in_progress": len(re.findall(r"\U0001F7E1", text)),
        "not_started": len(re.findall(r"\U0001F534", text)),
        "dead_end": len(re.findall(r"\u26AB", text)),
        "blocked": len(re.findall(r"\U0001F535", text)),
    }


def cmd_resume_prompt(_args):
    """Generate resume prompt from authoritative state/ files."""
    episodes_path = STATE_DIR / "episodes.md"
    dag_path = STATE_DIR / "dag.md"
    context_path = STATE_DIR / "context-index.md"

    max_cycle = get_max_cycle(episodes_path)
    recent = parse_episodes(episodes_path, last_n=3)
    ready_set = parse_dag_ready_set(dag_path)
    gate = parse_dag_gate_status(dag_path)
    deferred = parse_dag_deferred(dag_path)
    baseline = parse_dag_baseline(dag_path)
    system_summary = parse_context_system_level(context_path)
    dag_counts = count_dag_status(dag_path)

    recent_lines = ""
    for e in recent:
        recent_lines += (
            f"  Cycle {e['cycle']} ({e['timestamp']}): {e['tasks'][:80]}\n"
            f"    Decisions: {e['decisions'][:80]}\n"
        )
    if not recent_lines:
        recent_lines = "  (no episodes recorded yet)\n"

    deferred_block = f"\nDEFERRED TASKS:\n{deferred}\n" if deferred else ""

    # Async job awareness: check for pending/ready/failed jobs
    job_block = ""
    try:
        import subprocess
        job_poll = SCRIPT_DIR / "job_poll.py"
        if job_poll.exists():
            result = subprocess.run(
                [sys.executable, str(job_poll), "resume-block"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                job_block = f"\n{result.stdout.strip()}\n"
    except Exception:
        pass  # Non-fatal: if job_poll fails, resume still works

    prompt = (
        f"SINGLIFY ORCHESTRATOR RESUME -- {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{'=' * 74}\n\n"
        f"This is a RESUME from a paused session. Read this block, then follow Phase 0.\n\n"
        f"LAST COMPLETED CYCLE: {max_cycle}\n"
        f"NEXT CYCLE: {max_cycle + 1}\n"
        f"GATE-A: {gate}\n"
        f"DAG STATUS: {dag_counts.get('complete', 0)} complete, "
        f"{dag_counts.get('not_started', 0)} not started, "
        f"{dag_counts.get('dead_end', 0)} dead ends\n\n"
        f"SYSTEM SUMMARY (from context-index):\n{system_summary}\n\n"
        f"READY SET (from dag.md):\n{ready_set}\n"
        f"{deferred_block}\n"
        f"BASELINE PANEL:\n{baseline}\n\n"
        f"LAST 3 CYCLES:\n{recent_lines}\n"
        f"{job_block}"
        f"ACTION: Re-read singlify.agent.md + all state/ files (Phase 0). Then plan and\n"
        f"dispatch cycle {max_cycle + 1}. Do NOT ask whether to proceed.\n"
    )
    print(prompt)

    with open(RESUME_PATH, "w") as f:
        f.write(prompt)
    print(f"\n[checkpoint] Resume prompt written to {RESUME_PATH}", file=sys.stderr)


def cmd_status(_args):
    """Show compact sprint status."""
    episodes_path = STATE_DIR / "episodes.md"
    dag_path = STATE_DIR / "dag.md"

    max_cycle = get_max_cycle(episodes_path)
    gate = parse_dag_gate_status(dag_path)
    dag_counts = count_dag_status(dag_path)
    recent = parse_episodes(episodes_path, last_n=1)

    print(f"Last cycle: {max_cycle}")
    print(f"GATE-A: {gate}")
    print(f"DAG: {dag_counts}")
    if recent:
        last = recent[-1]
        print(f"Last entry: Cycle {last['cycle']} -- {last['decisions'][:80]}")


def cmd_archive(_args):
    """Archive legacy checkpoint.json to archive/ directory."""
    if not LEGACY_CHECKPOINT.exists():
        print("No legacy checkpoint.json found -- nothing to archive.")
        return

    ARCHIVE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_DIR / f"agent_checkpoint_{timestamp}.json"

    import shutil
    shutil.move(str(LEGACY_CHECKPOINT), str(dest))
    print(f"[archive] Moved {LEGACY_CHECKPOINT} -> {dest}")

    history = SCRIPT_DIR / "cycle_history.tsv"
    if history.exists():
        hist_dest = ARCHIVE_DIR / f"cycle_history_{timestamp}.tsv"
        shutil.move(str(history), str(hist_dest))
        print(f"[archive] Moved {history} -> {hist_dest}")


def main():
    parser = argparse.ArgumentParser(
        description="singlify orchestrator resume prompt generator"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("resume-prompt", help="Generate RESUME.txt from state/ files")
    sub.add_parser("status", help="Show compact sprint status")
    sub.add_parser("archive", help="Archive legacy checkpoint.json")

    args = parser.parse_args()

    if args.command == "resume-prompt":
        cmd_resume_prompt(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "archive":
        cmd_archive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
