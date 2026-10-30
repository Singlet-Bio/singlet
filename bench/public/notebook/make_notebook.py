#!/usr/bin/env python3
"""Generate singlet_benchmarks.ipynb (the notebook offered on singlet.bio/benchmarking) from the cells below.

    python bench/public/notebook/make_notebook.py

Edit the cells here, not the .ipynb, so diffs stay readable. The notebook is self-contained: it needs only the packages in
requirements.txt and network access to GitHub and data.singlet.bio (or local copies, see the configuration cell).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = [
("md", """# Singlet 1.0.0 benchmarks: reproduce every number

This notebook regenerates every metric and figure published on [singlet.bio/benchmarking](https://singlet.bio/benchmarking).

1. It loads the published result documents: one JSON per capability, in the
   [results contract](https://github.com/Singlet-Bio/singlet/blob/main/bench/public/CONTRACT.md).
2. It checks each document's pass/fail verdicts against its stated rules.
3. It redraws each figure from the data in the document.
4. **Reproduce.** It downloads the small per-cell artifacts (Singlet's output, every competitor's output, and the truth),
   checks their SHA-256, runs the document's own `compute()` function at the commit that produced it, and confirms that
   every number comes out identical.

**Laptop (summary):** steps 1-4 take a few minutes and well under 1 GB of downloads. **Full re-run (Clipper / any Slurm
cluster):** the last section prints the commands that re-run every tool on the cohort from raw reads (about 19 h at 12
concurrent 16-thread tasks)."""),
("code", """# Configuration. Defaults read everything from GitHub and data.singlet.bio; point them at local copies to work offline.
import os
RESULTS = os.environ.get("SINGLET_BENCH_RESULTS",
    "https://singlet.bio/data/benchmarks")   # or GitHub raw bench/public/results, or a local results/ directory
ARTIFACTS = os.environ.get("SINGLET_BENCH_ARTIFACTS", "")   # local folder with <capability>/<file>; empty = each document's URLs
CODE = os.environ.get("SINGLET_BENCH_CODE", "")             # local clone of Singlet-Bio/singlet; empty = download at each commit
WORKDIR = os.path.abspath(os.environ.get("SINGLET_BENCH_WORKDIR", "singlet_bench_cache"))
REPRODUCE = True        # download artifacts and recompute every metric
ONLY = []               # e.g. ["chromosomal-sex"]; empty = every capability with a document
os.makedirs(WORKDIR, exist_ok=True)"""),
("code", """import hashlib, io, json, math, re, shutil, sys, tarfile, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

UA = {"User-Agent": "singlet-benchmarks-notebook/1.0"}

def fetch_bytes(src):
    if re.match(r"https?://", src):
        with urllib.request.urlopen(urllib.request.Request(src, headers=UA), timeout=300) as r:
            return r.read()
    return Path(src).read_bytes()

def fetch_json(src):
    return json.loads(fetch_bytes(src))

def join(base, name):
    return f"{base.rstrip('/')}/{name}" if re.match(r"https?://", base) else str(Path(base) / name)

index = fetch_json(join(RESULTS, "index.json"))
print(f"schema {index['schema_version']}  generated {index['generated_at']}  overall: {index['overall']}  "
      f"({index['n_pass']}/{index['n_capabilities']} capabilities pass)")
pd.DataFrame([{k: c.get(k) for k in ("capability", "status", "n_pass", "n_metrics", "n_fail", "n_na", "owner", "doc_status")}
              for c in index["capabilities"]])"""),
("md", """## The contract's pass rules

Every metric carries an explicit rule, and `pass` is recomputed here from the values alone. The function below matches
`bench/public/contract.py:evaluate_metric`."""),
("code", """def tol(t, ref):
    return 0.0 if not t else max(float(t.get("abs", 0)), float(t.get("rel", 0)) * abs(ref))

def evaluate(m):
    rule, s = m.get("rule") or {}, (m.get("singlet") or {}).get("value")
    if rule.get("type") == "informational" or s is None:
        return "na"
    hi = m["direction"] == "higher"
    comps = {c["tool"]: c.get("value") for c in m.get("competitors", [])}
    if rule["type"] == "match_or_beat":
        ag = rule.get("against", "best")
        if ag == "best":
            vals = [v for v in comps.values() if v is not None]
            if not vals: return "na"
            refs = [max(vals) if hi else min(vals)]
        else:
            refs = [comps.get(t) for t in ag]
            if any(v is None for v in refs): return "na"
        ok = all((s >= r - tol(rule.get("tolerance"), r)) if hi else (s <= r + tol(rule.get("tolerance"), r)) for r in refs)
        return "pass" if ok else "fail"
    if rule["type"] == "threshold":
        return "pass" if ((s >= rule["value"]) if hi else (s <= rule["value"])) else "fail"
    if rule["type"] == "equivalence":
        t = comps.get(rule["target"]) if isinstance(rule["target"], str) else rule["target"]
        return "na" if t is None else ("pass" if abs(s - t) <= tol(rule.get("tolerance"), t) + 1e-12 else "fail")
    raise ValueError(rule)

docs = {}
for c in index["capabilities"]:
    if c.get("file") and (not ONLY or c["capability"] in ONLY):
        docs[c["capability"]] = fetch_json(join(RESULTS, c["file"]))
bad = [(cap, m["id"], m.get("dataset")) for cap, d in docs.items() for m in d["metrics"] if m.get("pass") != evaluate(m)]
print(f"{len(docs)} documents; verdicts that do not follow from their rules: {bad or 'none'}")"""),
("code", """def metric_table(d):
    rows = []
    for m in d["metrics"]:
        comp = "; ".join(f"{c['tool']} {c['value']}" for c in m.get("competitors", []))
        rule = m["rule"]["type"] + (f" (tol {m['rule']['tolerance']})" if m["rule"].get("tolerance") else "")
        rows.append({"metric": m["id"], "dataset": m["dataset"], "unit": m["unit"], "Singlet": m["singlet"]["value"],
                     "competitors": comp, "rule": rule, "pass": evaluate(m)})
    return pd.DataFrame(rows)

for cap, d in docs.items():
    print(f"\\n=== {d['title']} [{cap}]  status: {d['status']}\\n{d['conclusion']}")
    display(metric_table(d)) if "display" in globals() else print(metric_table(d).to_string())"""),
("md", "## Figures, redrawn from the documents' data"),
("code", """COLORS = {}
def color(tool):
    if tool == "singlet": return "#2563eb"
    pal = ["#9ca3af", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6", "#f472b6"]
    return COLORS.setdefault(tool, pal[len(COLORS) % len(pal)])

def draw(fig, ax):
    k = fig["kind"]
    if k == "table":
        ax.axis("off")
        t = ax.table(cellText=[[("" if v is None else v) for v in r] for r in fig["rows"]], colLabels=fig["columns"], loc="center")
        t.auto_set_font_size(False); t.set_fontsize(7)
    elif k == "heatmap":
        z = np.array(fig["z"], float)
        ax.imshow(z, cmap="Blues"); ax.set_xticks(range(len(fig["cols"])), fig["cols"], rotation=45, ha="right")
        ax.set_yticks(range(len(fig["rows"])), fig["rows"])
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                ax.text(j, i, f"{z[i, j]:g}", ha="center", va="center", fontsize=7)
    else:
        series = fig.get("series", [])
        if k in ("bar", "grouped_bar"):
            cats = fig.get("x", {}).get("categories") or series[0].get("x")
            w = 0.8 / max(len(series), 1)
            for i, s in enumerate(series):
                xs = np.arange(len(s["y"])) + i * w
                err = None
                if s.get("err_lo"):
                    err = [np.subtract(s["y"], s["err_lo"]), np.subtract(s["err_hi"], s["y"])]
                ax.bar(xs, s["y"], w, label=s["name"], color=color(s.get("tool", s["name"])), yerr=err, capsize=2)
            ax.set_xticks(np.arange(len(cats)) + w * (len(series) - 1) / 2, cats, rotation=30, ha="right")
        else:
            for s in series:
                xs = s.get("x", list(range(len(s["y"]))))
                if k == "scatter":
                    ax.scatter(xs, s["y"], s=6, label=s["name"], color=color(s.get("tool", s["name"])))
                else:
                    ax.plot(xs, s["y"], marker="o", label=s["name"], color=color(s.get("tool", s["name"])))
            if fig.get("x", {}).get("scale") == "log": ax.set_xscale("log")
        ax.set_xlabel(fig.get("x", {}).get("label", "")); ax.set_ylabel(fig.get("y", {}).get("label", ""))
        y = fig.get("y", {})
        if "min" in y or "max" in y: ax.set_ylim(y.get("min"), y.get("max"))
        if series: ax.legend(fontsize=7)
    ax.set_title(fig["title"], fontsize=9)

for cap, d in docs.items():
    figs = d.get("figures", [])
    if not figs: continue
    n = len(figs); cols = min(n, 2)
    f, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(6 * cols, 4 * math.ceil(n / cols)), squeeze=False)
    for ax, fig in zip(axes.flat, figs): draw(fig, ax)
    for ax in list(axes.flat)[n:]: ax.axis("off")
    f.suptitle(d["title"]); f.tight_layout(); plt.show()"""),
("md", """## Reproduce every number from the artifacts

For each document: download its artifacts and check their SHA-256; get the repository at the document's commit; run
`reproduce.module:function(artifact_dir)`; compare every Singlet and competitor value with the published one. Genotypes
never appear in an artifact. They hold labels, counts of a few public genes, and calls."""),
("code", """def get_code(commit):
    if CODE:
        return Path(CODE)
    dst = Path(WORKDIR) / "code" / commit
    if not dst.exists():
        data = fetch_bytes(f"https://codeload.github.com/Singlet-Bio/singlet/tar.gz/{commit}")
        tmp = dst.with_suffix(".tmp"); tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            members = [m for m in tf.getmembers() if "/bench/public/" in m.name or "/python/singlet/celllabels/" in m.name]
            tf.extractall(tmp, members=members, filter="data")
        shutil.move(str(next(tmp.iterdir())), dst); shutil.rmtree(tmp)
    return dst

def get_artifacts(cap, d):
    dst = Path(WORKDIR) / "artifacts" / cap; dst.mkdir(parents=True, exist_ok=True)
    for a in d.get("artifacts", []):
        p = dst / a["name"]
        if not p.exists():
            src = join(join(ARTIFACTS, cap), a["name"]) if ARTIFACTS else a["url"]
            p.write_bytes(fetch_bytes(src))
        if a.get("sha256") and hashlib.sha256(p.read_bytes()).hexdigest() != a["sha256"]:
            raise ValueError(f"{cap}/{a['name']}: SHA-256 mismatch")
    return dst

def recompute(cap, d):
    import importlib.util
    root = get_code(d["code"][0]["url"].split("/blob/")[1].split("/")[0])
    mod = root / d["reproduce"]["module"]
    for p in (root / "python", root / "bench/public", mod.parent):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(f"repro_{cap.replace('-', '_')}", mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return getattr(m, d["reproduce"]["function"])(get_artifacts(cap, d))

def diff(pub, rec, rel=1e-6, abs_=1e-9):
    key = lambda m: (m["id"], m.get("dataset"))
    a, b = {key(m): m for m in pub["metrics"]}, {key(m): m for m in rec["metrics"]}
    out = [f"{k}: only in {'published' if k in a else 'recomputed'}" for k in set(a) ^ set(b)]
    for k in set(a) & set(b):
        pairs = [("singlet", a[k]["singlet"]["value"], b[k]["singlet"]["value"])]
        ca = {c["tool"]: c["value"] for c in a[k].get("competitors", [])}
        cb = {c["tool"]: c["value"] for c in b[k].get("competitors", [])}
        pairs += [(t, ca.get(t), cb.get(t)) for t in set(ca) | set(cb)]
        for who, x, y in pairs:
            if (x is None) != (y is None) or (x is not None and not math.isclose(x, y, rel_tol=rel, abs_tol=abs_)):
                out.append(f"{k} {who}: published {x} vs recomputed {y}")
    return out

if REPRODUCE:
    summary = []
    for cap, d in docs.items():
        if not d.get("artifacts"):
            summary.append((cap, "no artifacts")); continue
        try:
            dd = diff(d, recompute(cap, d))
            summary.append((cap, "identical" if not dd else f"{len(dd)} differences: {dd[:3]}"))
        except Exception as e:
            summary.append((cap, f"could not reproduce: {type(e).__name__}: {e}"))
    print(pd.DataFrame(summary, columns=["capability", "reproduction"]).to_string(index=False))"""),
("md", """## Full re-run from raw reads (Clipper or another Slurm cluster)

The harness in `bench/public/harness/` re-runs every tool on the cohort (`bench/public/cohort.tsv`) and writes the same
documents. On Clipper, after `setup.sbatch`, run:

```bash
bash bench/public/harness/submit.sh --tier core          # fetch -> every (dataset, tool) run -> collect
bash bench/public/sexancestry/submit_sex.sh              # chromosomal sex (geo_sex21)
bash bench/public/sexancestry/submit_ancestry.sh         # genetic ancestry (MAGE, 150 individuals)
```

Then point `RESULTS` at the new `results/` directory and `ARTIFACTS` at `artifacts/`, and run this notebook again.
On another cluster, edit the paths in `bench/public/harness/common.sh`. Budget: `python bench/public/cohort_plan.py`."""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i:02d}", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = HERE / "singlet_benchmarks.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
