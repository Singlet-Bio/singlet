"""Helpers shared by the genotype-pool capability modules (donor-demux, donor-snps, mtdna). Standard library + numpy +
pandas only; no genotype, per-site allele count or donor SNP is read or written here."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REPO = "https://github.com/Singlet-Bio/singlet"
SINGLET_COMMIT = "02aa9b938b54e0f92bba00c44c07b883a8660aa2"   # source of the benchmarked binary (prod/campaign)
SINGLET_BINARY = "singlet-campaign"
SINGLET_BINARY_SHA256 = "261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_url(path: str, commit: str, tree: bool = False) -> str:
    return f"{REPO}/{'tree' if tree else 'blob'}/{commit}/{path}"


def contingency(a, b):
    a_codes, a_idx = np.unique(np.asarray(a, dtype=object).astype(str), return_inverse=True)
    b_codes, b_idx = np.unique(np.asarray(b, dtype=object).astype(str), return_inverse=True)
    m = np.zeros((len(a_codes), len(b_codes)), dtype=np.int64)
    np.add.at(m, (a_idx, b_idx), 1)
    return m


def ari(a, b) -> float:
    """Adjusted Rand index of two labelings (Hubert & Arabie)."""
    m = contingency(a, b)
    n = m.sum()
    if n < 2:
        return float("nan")
    c2 = lambda x: x * (x - 1) / 2.0
    sum_ij = c2(m).sum()
    sa, sb = c2(m.sum(1)).sum(), c2(m.sum(0)).sum()
    exp = sa * sb / c2(n)
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else float((sum_ij - exp) / (mx - exp))


def hungarian(m: np.ndarray):
    """Maximise matched count of a contingency table (rows truth, cols prediction): returns [(row, col)]."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-m)
    return list(zip(r.tolist(), c.tolist()))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def boot(values_fn, n_items: int, B: int = 300, seed: int = 7):
    """Percentile 95% CI of values_fn(idx) over resampled item indices."""
    rng = np.random.default_rng(seed)
    vals = [values_fn(rng.integers(0, n_items, n_items)) for _ in range(B)]
    vals = [v for v in vals if v == v]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def paired_halfwidth(stat_fn, n_items: int, B: int = 1000, seed: int = 11) -> float | None:
    """Half-width of the percentile 95% CI of stat_fn(resampled indices): the noise of a paired difference, used as the
    tolerance of a match-or-beat rule. None when it cannot be estimated."""
    if n_items < 2:
        return None
    rng = np.random.default_rng(seed)
    v = np.array([stat_fn(rng.integers(0, n_items, n_items)) for _ in range(B)], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return None
    return float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2.0)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def r6(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)


def artifact_entry(path: Path, name: str, kind: str, base_url: str, dataset: str | None = None, fmt: str | None = None):
    e = {"name": name, "url": f"{base_url}/{name}", "kind": kind, "sha256": sha256(path), "bytes": path.stat().st_size,
         "format": fmt or ("tsv" if ".tsv" in name else "json" if name.endswith(".json") else "csv")}
    if dataset:
        e["dataset"] = dataset
    return e
