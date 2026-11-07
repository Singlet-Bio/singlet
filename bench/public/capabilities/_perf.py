"""Shared helpers of the capability modules (bench/public/capabilities/<capability>.py).

Each capability module defines ``compute(artifact_dir: Path) -> dict`` that returns a complete capability document (the
contract in bench/public/CONTRACT.md) from the artifact tables alone, and a ``main()`` that writes it:

    python bench/public/capabilities/non_host.py --artifacts DIR --out bench/public/results/non-host.json

The numbers in the published file and the notebook's recomputation come from the same function. Needs numpy and pandas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # bench/public, for contract.py
import contract  # noqa: E402

REPO_URL = "https://github.com/Singlet-Bio/singlet"
RELEASE_URL = "https://data.singlet.bio/benchmarking/1.0.0"
# The pinned production binary of the 1.0.0 benchmarks (docs: bench/public/CONTRACT.md section 2, `singlet`).
SINGLET = {
    "version": "1.0.0",
    "commit": "02aa9b938b54e0f92bba00c44c07b883a8660aa2",
    "branch": "prod/campaign",
    "binary": "singlet-campaign",
    "binary_sha256": "261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c",
    "build_flags": "PGO, -march=x86-64-v3",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_entries(art: Path, capability: str, kinds: dict[str, str], datasets: dict[str, str] | None = None) -> list[dict]:
    """One entry per file in `art`: kind by file-name prefix (longest match in `kinds`), sha256 and size from the file."""
    out = []
    for p in sorted(Path(art).iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        kind = next((k for pre, k in sorted(kinds.items(), key=lambda kv: -len(kv[0])) if p.name.startswith(pre)), "derived")
        e = {"name": p.name, "url": f"{RELEASE_URL}/{capability}/{p.name}", "kind": kind, "sha256": sha256_file(p),
             "bytes": p.stat().st_size, "format": "".join(p.suffixes).lstrip(".")}
        for k, ds in (datasets or {}).items():
            if k in p.name:
                e["dataset"] = ds
        out.append(e)
    return out


def code_entries(commit: str, files: list[tuple[str, str]]) -> list[dict]:
    """[(path, role)] -> code list with GitHub blob URLs pinned to `commit`."""
    return [{"path": p, "role": role, "url": f"{REPO_URL}/blob/{commit}/{p}"} for p, role in files]


def ci(values: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(values, a)), float(np.quantile(values, 1 - a))]


def half_width(values: np.ndarray, level: float = 0.95) -> float:
    lo, hi = ci(values, level)
    return float((hi - lo) / 2)


def rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a positive > score of a negative) + half the ties (Mann-Whitney)."""
    r = rank(score)
    npos = int((label == 1).sum())
    nneg = int((label == 0).sum())
    return float((r[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def finite(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def finish(doc: dict, out: Path | None = None) -> dict:
    """finalize (compute every 'pass'), validate and optionally write."""
    contract.finalize(doc)
    if out is not None:
        errs = contract.validate(doc, contract.load_cohort(HERE.parent / "cohort.tsv"))
        if errs:
            raise SystemExit("INVALID:\n  " + "\n  ".join(errs))
        Path(out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def cli(compute) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write the validated capability document here")
    a = ap.parse_args()
    doc = compute(a.artifacts)
    finish(doc, a.out)
    st = contract.capability_status(doc)
    print(json.dumps(st))
