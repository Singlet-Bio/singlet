#!/usr/bin/env python3
"""
Generate singlify sprint status page (docs/status.html).
Called by push_status.sh at the end of every agent cycle.
"""
import json
import csv
import html
import os
import sys
from datetime import datetime, timezone

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPTS_DIR)

CHECKPOINT  = os.path.join(SCRIPTS_DIR, "agent_checkpoint.json")
HISTORY_TSV = os.path.join(SCRIPTS_DIR, "cycle_history.tsv")
STEERING    = os.path.join(SCRIPTS_DIR, "steering.txt")
OUT_HTML    = os.path.join(REPO_DIR, "docs", "status.html")

STEERING_EDIT_URL = (
    "https://github.com/Singlet-Bio/singlify/edit/main/scripts/steering.txt"
)


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def load_history():
    rows = []
    if os.path.exists(HISTORY_TSV):
        with open(HISTORY_TSV, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
    return rows[-10:]  # last 10 cycles


def load_steering():
    if os.path.exists(STEERING):
        txt = open(STEERING).read().strip()
        return txt if txt else None
    return None


def fmt_null(v, unit="", decimals=None):
    if v is None:
        return '<span class="null">—</span>'
    if decimals is not None:
        return f"{v:.{decimals}f}{unit}"
    return f"{v}{unit}"


def progress_bar(value, total, color="#4ade80"):
    if not total:
        return ""
    pct = min(100, int(value / total * 100))
    return (
        f'<div class="bar-bg"><div class="bar-fill" '
        f'style="width:{pct}%;background:{color}"></div></div>'
        f'<span class="bar-label">{value}/{total}</span>'
    )


def sprint_timer(start_iso, end_iso):
    try:
        start = datetime.fromisoformat(start_iso)
        end   = datetime.fromisoformat(end_iso)
        now   = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        elapsed = now - start.replace(tzinfo=None)
        total   = end.replace(tzinfo=None) - start.replace(tzinfo=None)
        pct = min(100, int(elapsed.total_seconds() / total.total_seconds() * 100))
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m = rem // 60
        return pct, f"{h}h {m}m elapsed"
    except Exception:
        return 0, "—"


def render_history_rows(rows):
    if not rows:
        return '<tr><td colspan="5" class="null" style="text-align:center">No cycles yet</td></tr>'
    out = []
    for r in reversed(rows):
        cycle = html.escape(str(r.get("cycle", "")))
        hyp   = html.escape(str(r.get("hypothesis", "") or "")[:60])
        result = r.get("result", "") or ""
        badge_color = {"win": "#4ade80", "dead_end": "#f87171", "in_progress": "#facc15"}.get(result, "#94a3b8")
        before = html.escape(str(r.get("before", "") or ""))
        after  = html.escape(str(r.get("after", "") or ""))
        gain   = html.escape(str(r.get("gain", "") or ""))
        out.append(
            f'<tr><td>{cycle}</td><td>{hyp}</td>'
            f'<td><span class="badge" style="background:{badge_color}">{html.escape(result)}</span></td>'
            f'<td>{before}</td><td>{after}</td><td>{gain}</td></tr>'
        )
    return "\n".join(out)


def generate(to_stdout=False):
    cp = load_checkpoint()
    rows = load_history()
    steering_txt = load_steering()

    m = cp.get("metrics", {})
    cycle         = cp.get("current_cycle", 0)
    phase         = cp.get("current_phase", "not_started")
    hypothesis    = cp.get("current_hypothesis") or "—"
    resume_note   = cp.get("resume_instructions", "")
    sprint_start  = cp.get("sprint_start", "")
    sprint_end    = cp.get("sprint_end_target", "")
    wins          = cp.get("wins_this_sprint", [])
    dead_ends     = cp.get("dead_ends_this_sprint", [])
    corpus_status = cp.get("corpus_status", {})
    notes         = cp.get("notes", "")
    last_updated  = cp.get("last_updated") or "not yet"

    timer_pct, timer_label = sprint_timer(sprint_start, sprint_end)

    m1_pass  = m.get("m1_protocols_passing", 0)
    m1_total = m.get("m1_protocols_tested", 0)
    m2_1     = m.get("m2_1_wall_40m_16t_s")
    m3_r     = m.get("m3_1_exon_pearson_r")
    m4_pass  = m.get("m4_protocols_complete", 0)
    m4_total = 13

    corpus_rows = ""
    for cid in [f"C{i:02d}" for i in range(1, 14)]:
        status = corpus_status.get(cid, "pending")
        color  = {"encode_pass": "#4ade80", "decode_pass": "#22d3ee",
                  "pipeline_pass": "#a78bfa", "failed": "#f87171",
                  "pending": "#475569"}.get(status, "#475569")
        corpus_rows += (
            f'<span class="corpus-tag" style="background:{color}" '
            f'title="{cid}: {status}">{cid}</span>'
        )

    def _label(item):
        if isinstance(item, dict):
            return item.get("hypothesis", str(item))
        return str(item)
    wins_html = "".join(f"<li>{html.escape(_label(w))}</li>" for w in wins) or "<li>—</li>"
    dead_html = "".join(f"<li>{html.escape(_label(d))}</li>" for d in dead_ends) or "<li>—</li>"

    m2_target = 120.0
    m2_color  = "#4ade80" if (m2_1 is not None and m2_1 <= m2_target) else "#f87171"
    m2_display = fmt_null(m2_1, "s", 1) if m2_1 else '<span class="null">—</span>'

    steering_section = ""
    if steering_txt:
        steering_section = f"""
      <div class="card alert">
        <div class="card-title">&#128233; Pending Steering Prompt</div>
        <p style="font-family:monospace;white-space:pre-wrap;font-size:0.85rem">{html.escape(steering_txt)}</p>
        <p style="font-size:0.75rem;color:#94a3b8">Agent will consume this on the next cycle.</p>
      </div>"""

    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <meta http-equiv="refresh" content="300"/>
  <title>singlify sprint status</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 15px; padding: 16px; max-width: 900px; margin: 0 auto; }}
    h1 {{ font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
    .subtitle {{ color: #64748b; font-size: 0.8rem; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: #1e293b; border-radius: 10px; padding: 14px 16px; }}
    .card.alert {{ border: 1px solid #f59e0b; background: #1c1a0e; }}
    .card.full {{ grid-column: 1 / -1; }}
    .card-title {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 8px; }}
    .metric-val {{ font-size: 1.6rem; font-weight: 700; line-height: 1; }}
    .metric-sub {{ font-size: 0.75rem; color: #64748b; margin-top: 4px; }}
    .null {{ color: #475569; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 0.7rem; font-weight: 600; color: #0f172a; }}
    .phase-badge {{ display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 600; background: #334155; color: #e2e8f0; margin-bottom: 10px; }}
    .bar-bg {{ background: #334155; border-radius: 4px; height: 6px; margin-top: 6px; margin-bottom: 3px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.4s; }}
    .bar-label {{ font-size: 0.72rem; color: #94a3b8; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
    th {{ text-align: left; color: #64748b; font-weight: 600; padding: 4px 8px; border-bottom: 1px solid #334155; }}
    td {{ padding: 5px 8px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    .corpus-tag {{ display: inline-block; padding: 3px 7px; border-radius: 5px; font-size: 0.72rem; font-weight: 600; color: #0f172a; margin: 2px; }}
    .steer-btn {{ display: inline-block; margin-top: 10px; padding: 8px 18px; background: #3b82f6; color: #fff; border-radius: 7px; text-decoration: none; font-size: 0.85rem; font-weight: 600; }}
    .steer-btn:hover {{ background: #2563eb; }}
    .wins li {{ color: #4ade80; font-size: 0.82rem; margin: 2px 0; }}
    .dead li {{ color: #f87171; font-size: 0.82rem; margin: 2px 0; }}
    .hyp {{ font-size: 0.88rem; line-height: 1.5; color: #cbd5e1; }}
    .timer-wrap {{ margin-bottom: 6px; }}
    .timer-bar {{ background: #334155; border-radius: 4px; height: 8px; overflow: hidden; margin: 4px 0; }}
    .timer-fill {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }}
    .refresh-note {{ color: #475569; font-size: 0.7rem; text-align: right; margin-top: 12px; }}
    @media (max-width: 480px) {{ .metric-val {{ font-size: 1.3rem; }} }}
  </style>
</head>
<body>
  <h1>singlify &mdash; sprint status</h1>
  <div class="subtitle">Auto-refreshes every 5 min &nbsp;&bull;&nbsp; Last agent update: {html.escape(last_updated)} &nbsp;&bull;&nbsp; Page generated: {now_str}</div>

  <!-- Sprint timer -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Sprint progress</div>
    <div class="timer-wrap">
      <div class="timer-bar"><div class="timer-fill" style="width:{timer_pct}%"></div></div>
      <span style="font-size:0.78rem;color:#94a3b8">{timer_label} &mdash; {timer_pct}% of 3-day sprint</span>
    </div>
    <div style="margin-top:6px"><span class="phase-badge">Cycle {cycle} &nbsp;|&nbsp; Phase: {html.escape(phase)}</span></div>
    <div class="hyp"><strong>Current hypothesis:</strong> {html.escape(hypothesis)}</div>
  </div>

  <!-- Metrics grid -->
  <div class="grid">
    <div class="card">
      <div class="card-title">M1 &mdash; Format coverage</div>
      <div class="metric-val">{m1_pass}<span style="font-size:1rem;color:#64748b">/{m1_total}</span></div>
      <div class="metric-sub">protocols passing encode+decode</div>
      {progress_bar(m1_pass, max(m1_total, 1))}
    </div>
    <div class="card">
      <div class="card-title">M2.1 &mdash; Pipeline speed (40M/16T)</div>
      <div class="metric-val" style="color:{m2_color}">{m2_display}</div>
      <div class="metric-sub">target &le; 120s</div>
    </div>
    <div class="card">
      <div class="card-title">M3.1 &mdash; Pileup accuracy</div>
      <div class="metric-val">{fmt_null(m3_r, decimals=4)}</div>
      <div class="metric-sub">exon Pearson r vs STARsolo</div>
    </div>
    <div class="card">
      <div class="card-title">M4 &mdash; Corpus complete</div>
      <div class="metric-val">{m4_pass}<span style="font-size:1rem;color:#64748b">/{m4_total}</span></div>
      <div class="metric-sub">full pipeline on all 13 accessions</div>
      {progress_bar(m4_pass, m4_total, "#a78bfa")}
    </div>
  </div>

  <!-- Corpus status -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Corpus sample status</div>
    <div style="margin-top:6px">{corpus_rows}</div>
    <div style="margin-top:8px;font-size:0.72rem;color:#64748b">
      <span class="corpus-tag" style="background:#4ade80;color:#0f172a">encode</span>
      <span class="corpus-tag" style="background:#22d3ee;color:#0f172a">decode</span>
      <span class="corpus-tag" style="background:#a78bfa;color:#0f172a">pipeline</span>
      <span class="corpus-tag" style="background:#f87171;color:#0f172a">failed</span>
      <span class="corpus-tag" style="background:#475569;color:#e2e8f0">pending</span>
    </div>
  </div>

  <!-- Wins + dead ends -->
  <div class="grid">
    <div class="card">
      <div class="card-title">&#9989; Wins this sprint</div>
      <ul class="wins">{wins_html}</ul>
    </div>
    <div class="card">
      <div class="card-title">&#10060; Dead ends this sprint</div>
      <ul class="dead">{dead_html}</ul>
    </div>
  </div>

  <!-- Cycle history -->
  <div class="card full" style="margin-top:16px;overflow-x:auto">
    <div class="card-title">Cycle history (last 10)</div>
    <table>
      <thead><tr><th>#</th><th>Hypothesis</th><th>Result</th><th>Before</th><th>After</th><th>Gain</th></tr></thead>
      <tbody>{render_history_rows(rows)}</tbody>
    </table>
  </div>

  {steering_section}

  <!-- Steering -->
  <div class="card" style="margin-top:16px">
    <div class="card-title">&#127919; Steer the agent</div>
    <p style="font-size:0.82rem;color:#94a3b8;line-height:1.5">
      Reply to any agent email with subject containing <code>[singlify-steer]</code>.
      The agent polls for replies at the start of every cycle (~3h interval),
      reads your message, incorporates it as a steering prompt, and emails you back.
    </p>
    <p style="font-size:0.72rem;color:#475569;margin-top:8px">
      Or edit <code>scripts/steering.txt</code> directly on the cluster via SSH.
      The agent reads and clears it on the next cycle.
    </p>
  </div>

  {"<div class='card' style='margin-top:16px'><div class='card-title'>Notes</div><p style='font-size:0.82rem;color:#cbd5e1;white-space:pre-wrap'>" + html.escape(notes) + "</p></div>" if notes else ""}

  <div class="refresh-note">Page auto-refreshes every 5 minutes. Source: <a href="https://github.com/Singlet-Bio/singlify/blob/main/scripts/agent_checkpoint.json" style="color:#475569">agent_checkpoint.json</a></div>
</body>
</html>
"""
    if to_stdout:
        sys.stdout.write(html_out)
    else:
        os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
        with open(OUT_HTML, "w") as f:
            f.write(html_out)
        print(f"Written: {OUT_HTML}")


if __name__ == "__main__":
    if "--email" in sys.argv:
        # Print HTML to stdout for piping to mailx
        generate(to_stdout=True)
    else:
        generate(to_stdout=False)
