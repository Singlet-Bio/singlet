# SPDX-License-Identifier: MIT
"""Shared helpers for the public-benchmark capability modules (gex-counts, spliced-unspliced, junctions-psi, cell-calling).

Every capability module exposes ``compute(artifact_dir: Path) -> dict`` returning a capability document that
``contract.finalize`` has already stamped with ``pass``. The module is also a script:

    python bench/public/capabilities/gex_counts.py ARTIFACT_DIR [--out bench/public/results/gex-counts.json]

numpy and pandas only (the notebook installs nothing else). Randomness (bootstrap) uses fixed seeds, so a re-run reproduces
every number.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
sys.path.insert(0, str(PUB))
import contract  # noqa: E402  (bench/public/contract.py)

REPO_URL = "https://github.com/Singlet-Bio/singlet"
ARTIFACT_URL = "https://data.singlet.bio/benchmarking/1.0.0"
OWNER = "pipeline-benchmark"

SINGLET_COMMIT = "02aa9b938b54e0f92bba00c44c07b883a8660aa2"  # prod/campaign, merge of prod/modality-fixes
SINGLET_BINARY_SHA256 = "261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c"

BOOTSTRAPS = 400
SEED = 20261025

DATASET_ORDER = ["pbmc1k_v3", "pbmc1k_v2", "neuron1k_v3", "hgmm1k_v3", "pbmc1k_cite_v3", "nuclei2k_v2", "pbmc5p_v1",
                 "kang2018_ctrl"]

# Fallback descriptions for cohort ids when cohort.tsv has no row yet (the validator then asks for the row).
DATASET_FALLBACK = {
    "pbmc1k_v3": ("10x PBMC 1k, 3' v3", "https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-v-3-chemistry-3-standard-3-0-0"),
    "pbmc1k_v2": ("10x PBMC 1k, 3' v2", "https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-v-2-chemistry-3-standard-2-1-0"),
    "neuron1k_v3": ("10x mouse E18 neurons 1k, 3' v3", "https://www.10xgenomics.com/datasets/1-k-brain-cells-from-an-e18-mouse-v-3-chemistry-3-standard-3-0-0"),
    "hgmm1k_v3": ("10x HEK293T + NIH3T3 1k, 3' v3", "https://www.10xgenomics.com/datasets/1-k-1-1-mixture-of-fresh-frozen-human-hek-293-t-and-mouse-nih-3-t-3-cells-v-3-chemistry-3-standard-3-0-0"),
    "pbmc1k_cite_v3": ("10x PBMC 1k + TotalSeq-B, 3' v3", "https://www.10xgenomics.com/datasets/1-k-pbm-cs-from-a-healthy-donor-gene-expression-and-cell-surface-protein-3-standard-3-0-0"),
    "nuclei2k_v2": ("10x mouse brain nuclei 2k, 3' v2", "https://www.10xgenomics.com/datasets/2-k-brain-nuclei-from-an-e18-mouse-2-standard-2-1-0"),
    "pbmc5p_v1": ("10x PBMC 5' v1 (first 60 M read pairs)", "https://www.10xgenomics.com/datasets/human-pbm-cs-5-gene-expression-with-a-panel-of-total-seq-b-antibodies-v-1-chemistry-3-standard-3-0-0"),
    "kang2018_ctrl": ("Kang 2018 PBMC, batch 2 control lane (GSE96583)", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583"),
}


# --------------------------------------------------------------------------------------------------- statistics
def rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def boot_indices(n: int, offset: int = 0, B: int = BOOTSTRAPS) -> np.ndarray:
    """B x n resampling indices, fixed seed."""
    return rng(offset).integers(0, n, size=(B, n))


def boot_stat(x: np.ndarray, fn, idx: np.ndarray) -> np.ndarray:
    return np.array([fn(x[i]) for i in idx])


def ci95(samples: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def half_width(ci: list[float]) -> float:
    return float((ci[1] - ci[0]) / 2.0)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fnum(x) -> float | None:
    """JSON-safe float: NaN / inf become None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def rnd(x, nd: int = 6):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


# ------------------------------------------------------------------------------------------------- tables / files
def read_table(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kw)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_entries(artifact_dir: Path, capability: str, kinds: dict[str, str]) -> list[dict]:
    """One entry per file in the artifact directory. ``kinds`` maps a filename suffix to the artifact kind."""
    out = []
    for p in sorted(Path(artifact_dir).iterdir()):
        if not p.is_file():
            continue
        kind = next((k for suf, k in kinds.items() if p.name.endswith(suf)), "derived")
        e = {"name": p.name, "url": f"{ARTIFACT_URL}/{capability}/{p.name}", "kind": kind,
             "sha256": sha256_of(p), "bytes": p.stat().st_size,
             "format": "tsv" if ".tsv" in p.name else ("json" if p.name.endswith(".json") else "other")}
        ds = p.name.split(".")[0]
        if ds in DATASET_FALLBACK:
            e["dataset"] = ds
        out.append(e)
    return out


def code_commit() -> str:
    """Commit that holds the code listed under ``code`` (bench/public/capabilities/CODE_COMMIT, set when results are committed)."""
    p = HERE / "CODE_COMMIT"
    return p.read_text().strip() if p.exists() else "0" * 40


def code_entries(paths: dict[str, str]) -> list[dict]:
    sha = code_commit()[:40]
    return [{"path": p, "role": role, "url": f"{REPO_URL}/blob/{sha}/{p}"} for p, role in paths.items()]


# --------------------------------------------------------------------------------------------- dataset metadata
def cohort_rows() -> dict[str, dict]:
    p = PUB / "cohort.tsv"
    return contract.load_cohort(p) if p.exists() else {}


def dataset_entries(ids: list[str], truth: dict[str, str], n_cells: dict[str, int] | None = None,
                    notes: dict[str, str] | None = None) -> list[dict]:
    cohort = cohort_rows()
    out = []
    for d in ids:
        row = cohort.get(d, {})
        title, url = DATASET_FALLBACK.get(d, (d, ""))
        acc: dict = {}
        if row.get("gse"):
            acc["gse"] = row["gse"]
        if row.get("gsm"):
            acc["gsm"] = [x for x in row["gsm"].replace(";", ",").split(",") if x]
        if row.get("srr"):
            acc["srr"] = [x for x in row["srr"].replace(";", ",").split(",") if x]
        if row.get("url") or url:
            acc["url"] = row.get("url") or url
        e = {"id": d, "title": row.get("title") or title, "accessions": acc, "truth": truth[d]}
        if n_cells and d in n_cells:
            e["n_cells"] = int(n_cells[d])
        if notes and d in notes:
            e["notes"] = notes[d]
        out.append(e)
    return out


# ---------------------------------------------------------------------------------------------- document pieces
def singlet_block() -> dict:
    return {"version": "1.0.0", "commit": SINGLET_COMMIT, "branch": "prod/campaign", "binary": "singlet-campaign",
            "binary_sha256": SINGLET_BINARY_SHA256, "build_flags": "PGO, -O3 -DNDEBUG, -march=x86-64-v3"}


def tool(name: str, version: str, role: str, params: str | None = None, url: str | None = None,
         citation: str | None = None) -> dict:
    t = {"name": name, "version": version, "role": role}
    if url:
        t["url"] = url
    if citation:
        t["citation"] = citation
    if params:
        t["params"] = params
    return t


def metric(mid: str, name: str, unit: str, category: str, dataset: str, direction: str, value, ci=None,
           competitors=None, rule=None, notes: str | None = None) -> dict:
    m = {"id": mid, "name": name, "unit": unit, "category": category, "dataset": dataset, "direction": direction,
         "singlet": {"value": rnd(value)}}
    if ci is not None:
        m["singlet"]["ci"] = [rnd(ci[0]), rnd(ci[1])]
    m["competitors"] = [{"tool": c["tool"], "value": rnd(c.get("value")),
                         **({"ci": [rnd(c["ci"][0]), rnd(c["ci"][1])]} if c.get("ci") else {})} for c in (competitors or [])]
    m["rule"] = rule
    if notes:
        m["notes"] = notes
    return m


def equivalence_rule(target, tol_abs: float, rationale: str) -> dict:
    return {"type": "equivalence", "target": target, "tolerance": {"abs": rnd(tol_abs)}, "rationale": rationale}


def match_rule(tol_abs: float | None, rationale: str | None, against="best") -> dict:
    r: dict = {"type": "match_or_beat", "against": against}
    if tol_abs:
        r["tolerance"] = {"abs": rnd(tol_abs)}
        r["rationale"] = rationale
    return r


def resource_rows(res: dict, tool_names: dict[str, str]) -> list[dict]:
    """resources.json -> contract resources list. ``res``: {dataset: {tool_key: {wall_s, max_rss_gb, threads, cpu_h?, node?}}}."""
    rows = []
    for ds in DATASET_ORDER:
        for key, r in (res.get(ds) or {}).items():
            if key not in tool_names or r.get("wall_s") is None:
                continue
            row = {"tool": tool_names[key], "dataset": ds, "wall_s": rnd(r["wall_s"], 1),
                   "max_rss_gb": rnd(r.get("max_rss_gb", 0.0), 2), "threads": r.get("threads", 0)}
            if r.get("cpu_h") is not None:
                row["cpu_h"] = rnd(r["cpu_h"], 3)
            if r.get("node"):
                row["node"] = r["node"]
            if r.get("notes"):
                row["notes"] = r["notes"]
            rows.append(row)
    return rows


def stamp(metrics: list[dict]) -> list[dict]:
    """Fill each metric's 'pass' from its rule (what contract.finalize does), so the text can quote results."""
    for m in metrics:
        m["pass"] = contract.evaluate_metric(m)
    return metrics


def finish(doc: dict) -> dict:
    contract.finalize(doc)
    return doc


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def cli(compute, capability: str) -> None:
    ap = argparse.ArgumentParser(description=f"Compute the {capability} capability document from its artifacts.")
    ap.add_argument("artifact_dir", nargs="?", default=os.environ.get("ARTIFACT_ROOT"))
    ap.add_argument("--out", default=str(PUB / "results" / f"{capability}.json"))
    ap.add_argument("--status", choices=["draft", "final"], default=None)
    a = ap.parse_args()
    if not a.artifact_dir:
        ap.error("artifact_dir (or ARTIFACT_ROOT) required")
    doc = compute(Path(a.artifact_dir))
    if a.status:
        doc["status"] = a.status
        contract.finalize(doc)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    st = contract.capability_status(doc)
    print(f"{a.out}: {doc['status']} {st}")
