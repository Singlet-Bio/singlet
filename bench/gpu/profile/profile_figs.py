#!/usr/bin/env python3
"""Figures of PROFILE.md from the profile items' outputs: the phase timeline of the traced run per point (nvtx_phases.py TSVs) and the
per-run CPU time by stage (the harness's "stage core-s (sampler)" lines in the pilot item log).

  profile_figs.py --tsv DIR --log SAMPLE=ITEM.log [--log ...] --out DIR
"""
import argparse, collections, csv, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--tsv", required=True)
ap.add_argument("--log", action="append", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# Timeline: one row per traced run (point), bars for the phases that matter; the STAR child's and the pipeline's device holds are dark.
PH = [("STAR", "device lock: wait", "#c9c9c9", "device wait (STAR)"), ("STAR", "device lock: held", "#1f4e79", "device held (STAR prefill)"),
      ("pipeline", "device lock: wait", "#c9c9c9", None), ("pipeline", "device lock: held", "#2e75b6", "device held (pileup, non-host)"),
      ("pipeline", "main: non-host screen", "#9dc3e6", "non-host screen"),
      ("pipeline", "export: Phase 1.7: Cell calling (N5, EmptyDrops-style)", "#f4b183", "cell calling"),
      ("pipeline", "export: Phase 3: Demux + parallel writes", "#c55a11", "demux + writes")]
files = sorted(glob.glob(os.path.join(a.tsv, "*.tsv")))
rows = collections.defaultdict(dict)
for f in files:
    for r in csv.DictReader(open(f), delimiter="\t"):
        rows[r["label"]][(r["process"], r["range"])] = r
labels = list(rows)
fig, ax = plt.subplots(figsize=(10, 0.55 * len(labels) + 2.0))
seen = set()
for i, l in enumerate(labels):
    end = max(float(r["last_end_s"]) for r in rows[l].values())
    ax.barh(i, end, left=0, color="#f2f2f2", edgecolor="none", height=0.7, label="other host work" if "o" not in seen else None); seen.add("o")
    for proc, rng, col, lab in PH:
        r = rows[l].get((proc, rng))
        if not r: continue
        s, e, tot = float(r["first_start_s"]), float(r["last_end_s"]), float(r["summed_s"])
        # ranges that occur once are drawn as they ran; repeated ones (the pipeline's three device holds) as their summed time from the first start
        w = (e - s) if int(r["count"]) == 1 else tot
        ax.barh(i, w, left=s, color=col, height=0.7, label=lab if lab and lab not in seen else None); seen.add(lab)
ax.set_yticks(range(len(labels))); ax.set_yticklabels([l.replace("_trace", "").replace("_", " ") for l in labels])
ax.invert_yaxis(); ax.set_xlabel("seconds from the run's start (traced run of each point)")
ax.set_title("Host phases set the wall time until the per-slot device budget runs out")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, ncol=4, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "timeline.png"), dpi=150)

# CPU time per run by stage, mean over the runs of each point.
ST = ["scan", "refs", "STAR", "stream", "handoff", "pileup", "pileupfin", "nonhost", "export", "demux", "start"]
COL = dict(zip(ST, ["#7f7f7f", "#ffc000", "#1f4e79", "#2e75b6", "#9dc3e6", "#5b9bd5", "#bdd7ee", "#70ad47", "#a9d18e", "#c55a11", "#d9d9d9"]))
bars = []
for spec in a.log:
    sample, path = spec.split("=", 1)
    pt = None; acc = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        m = re.match(r"== point (\S+):", line)
        if m: pt = m.group(1).replace("_trace", ""); continue
        m = re.match(r"\s+stage core-s \(sampler\): (.*)", line)
        if m and pt:
            acc[pt].append({k: float(v) for k, v in re.findall(r"(\w+) (-?[\d.]+)", m.group(1))})
    for p, runs in acc.items():
        bars.append((f"{sample} {p}", {s: sum(r.get(s, 0) for r in runs) / len(runs) for s in ST}, len(runs)))
if bars:
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(bars) + 2.0))
    for i, (lab, d, n) in enumerate(bars):
        left = 0
        for s in ST:
            ax.barh(i, d[s], left=left, color=COL[s], height=0.7, label=s if i == 0 else None); left += d[s]
        ax.text(left + 5, i, f"{left:.0f} core-s ({n} runs)", va="center", fontsize=7)
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars]); ax.invert_yaxis()
    ax.set_xlabel("CPU core-seconds per run (mean over the point's runs)")
    ax.set_title("Pileup streaming, reference loading and demux take most of the CPU")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, ncol=6, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "stage_cores.png"), dpi=150)
print("figures in", a.out)
