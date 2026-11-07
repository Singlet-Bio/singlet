"""Pull the small result files of the final runs from Clipper into results/ (kang_*.json, hgmm_*.csv, timing.jsonl).
Usage: python collect_results.py   (needs the clipper bridge)"""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
clip = os.path.expanduser("~/.claude/skills/clipper/clipper.py")
R = "/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/doublet_fix/results"
files = []
for lab in ["final", "final_bayes", "final_exon_only"]:
    for d in ["k8x", "k8g", "k8"]:
        files.append(f"kang_{d}_{lab}.json")
    for tag in ["", "_r05", "_r08"]:
        files.append(f"hgmm_{lab}{tag}.csv")
files.append("timing.jsonl")
for f in files:
    r = subprocess.run([sys.executable, clip, "get", f"{R}/{f}", os.path.join(out, f)], capture_output=True, text=True)
    print(f, r.stdout.strip()[:60], r.stderr.strip()[:80])
