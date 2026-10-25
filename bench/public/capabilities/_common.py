"""Shared helpers for the capability compute modules (``bench/public/capabilities/<id>.py``).

Every module exposes ``compute(artifact_dir: Path) -> dict``. The harness calls it to write ``results/<id>.json``, and the
notebook calls it on downloaded artifacts. The provenance (commit, tool versions, resources) comes from a
``provenance.json`` artifact, so the same call gives the same document wherever it runs.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
REPO_URL = "https://github.com/Singlet-Bio/singlet"
ARTIFACT_BASE = "https://data.singlet.bio/benchmarking/1.0.0"


def contract():
    spec = importlib.util.spec_from_file_location("bench_public_contract", PUB / "contract.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_url(path: str, commit: str) -> str:
    return f"{REPO_URL}/blob/{commit}/{path}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def artifact_entries(artifact_dir: Path, capability: str, kinds: dict[str, str]) -> list[dict]:
    """Artifact list for the files named in ``kinds`` ({file name: kind}) that exist in ``artifact_dir``."""
    out = []
    for name, kind in kinds.items():
        p = artifact_dir / name
        if p.exists():
            out.append({"name": name, "url": f"{ARTIFACT_BASE}/{capability}/{name}", "kind": kind,
                        "sha256": sha256(p), "bytes": p.stat().st_size, "format": p.suffix.lstrip(".")})
    return out


def bootstrap_ci(groups: np.ndarray, num: np.ndarray, den: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% CI of sum(num)/sum(den), resampling clusters (``groups``: one label per row) with replacement."""
    keys, inv = np.unique(groups, return_inverse=True)
    gn = np.bincount(inv, weights=num, minlength=len(keys))
    gd = np.bincount(inv, weights=den, minlength=len(keys))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    est = gn[idx].sum(1) / np.maximum(gd[idx].sum(1), 1e-12)
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def bootstrap_diff_halfwidth(groups: np.ndarray, num_a, den_a, num_b, den_b, n_boot: int = 2000, seed: int = 0) -> float:
    """Half-width of the paired 95% bootstrap CI of ratio_a - ratio_b (clusters resampled together for both tools).

    Used as the tolerance of per-cell metrics: two tools whose difference is inside this sampling noise are reported as
    matching. The margin is computed the same way for every metric and every tool, before looking at which tool wins."""
    keys, inv = np.unique(groups, return_inverse=True)
    sums = [np.bincount(inv, weights=np.asarray(x, float), minlength=len(keys)) for x in (num_a, den_a, num_b, den_b)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    ra = sums[0][idx].sum(1) / np.maximum(sums[1][idx].sum(1), 1e-12)
    rb = sums[2][idx].sum(1) / np.maximum(sums[3][idx].sum(1), 1e-12)
    d = ra - rb
    return float((np.quantile(d, 0.975) - np.quantile(d, 0.025)) / 2)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def load_provenance(artifact_dir: Path) -> dict:
    return json.loads((artifact_dir / "provenance.json").read_text(encoding="utf-8"))


def r(x, nd: int = 4):
    """Round for publication (the validator and diff compare these rounded numbers)."""
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)
