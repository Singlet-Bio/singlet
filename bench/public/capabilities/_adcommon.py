# SPDX-License-Identifier: MIT
"""Helpers shared by the capability writers in this folder (standard library + numpy/pandas/scipy only).

Each capability module exposes `compute(artifact_dir: Path) -> dict` returning a document that satisfies
bench/public/contract.py, and a command line that writes it. The harness and the notebook call the same function.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import contract  # noqa: E402

N_BOOT = 1000
SEED = 20261025

SINGLET = {
    "version": "1.0.0",
    "commit": "02aa9b938b54e0f92bba00c44c07b883a8660aa2",
    "branch": "prod/campaign",
    "binary": "pipeline-benchmark/bin/singlet-campaign",
    "binary_sha256": "261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c",
}
REPO_URL = "https://github.com/Singlet-Bio/singlet"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng(*key) -> np.random.Generator:
    """Deterministic generator per (seed, key) so every bootstrap is reproducible and independent of call order."""
    return np.random.default_rng([SEED] + [int(hashlib.md5(str(k).encode()).hexdigest()[:8], 16) for k in key])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)  # average ranks: ties count one half
    return float((r[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (step interpolation, ties broken pessimistically by processing tied scores together)."""
    n1 = int(y.sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order].astype(float), s[order]
    tp = np.cumsum(ys)
    # last index of each tie block
    last = np.r_[ss[1:] != ss[:-1], True]
    tp_b, n_b = tp[last], (np.arange(len(ys)) + 1)[last]
    prec, rec = tp_b / n_b, tp_b / n1
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def f1(y: np.ndarray, c: np.ndarray) -> float:
    tp = float(((c == 1) & (y == 1)).sum())
    fp = float(((c == 1) & (y == 0)).sum())
    fn = float(((c == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def boot_indices(n: int, key, n_boot: int = N_BOOT) -> np.ndarray:
    return rng(*key).integers(0, n, size=(n_boot, n))


def ci95(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return [round(float(np.percentile(x, 2.5)), 6), round(float(np.percentile(x, 97.5)), 6)]


def half_width(x: np.ndarray) -> float:
    lo, hi = np.percentile(x[np.isfinite(x)], [2.5, 97.5])
    return float((hi - lo) / 2)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def r6(x):
    return None if x is None else round(float(x), 6)


def finish(doc: dict) -> dict:
    """Fill `pass` from the rules and refuse to return an invalid document."""
    contract.finalize(doc)
    errs = contract.validate(doc)
    if errs:
        raise ValueError("document violates the contract:\n  " + "\n  ".join(errs))
    return doc


def write_main(compute, description: str, default_out: str) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("artifact_dir", type=Path, help="directory holding this capability's artifacts")
    ap.add_argument("--out", type=Path, default=Path(default_out), help="where to write the capability document")
    a = ap.parse_args()
    doc = compute(a.artifact_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    st = contract.capability_status(doc)
    print(f"{a.out}: {st['status']} ({st['n_pass']}/{st['n_metrics']} pass, {st['n_fail']} fail, {st['n_na']} n/a)")
    return 0
