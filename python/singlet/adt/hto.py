# SPDX-License-Identifier: MIT
"""Hashtag demultiplexing calls (per called cell): singlet with its hashtag, doublet, or negative.

Method: ``DropletUtils::hashedDrops`` (Lun et al.; Bioconductor), the ambient-aware rule:

1. Ambient proportions per hashtag from the empty droplets; each cell's counts are divided by them, so a hashtag that
   is only ambient in the cell sits at the cell's ambient level (the median of the hashtags below the top two).
2. Log2 fold change between the largest and second-largest scaled count (``lfc_top``) and between the second-largest
   and the ambient level (``lfc_second``), with pseudocount 5.
3. Doublet if ``lfc_second`` is an upper outlier (> median + ``nmads`` MAD among cells) and above ``doublet_min``;
   confident singlet if ``lfc_top >= confident_min`` and it is not a lower outlier; otherwise negative (unassigned).

Output per cell: ``call`` (hashtag index, -1 negative, -2 doublet), ``lfc_top``, ``lfc_second``.
"""

from __future__ import annotations

import numpy as np

CONFIDENT_MIN = 2.0  # hashedDrops default (log2 fold)
DOUBLET_MIN = 2.0
DOUBLET_NMADS = 3.0
CONFIDENT_NMADS = 3.0
PSEUDO = 5.0


def _mad(x):
    m = np.median(x)
    return m, 1.4826 * np.median(np.abs(x - m))


def hashed_drops(cells: np.ndarray, ambient: np.ndarray):
    """cells: hashtags x cells counts; ambient: per-hashtag counts in empty droplets."""
    X = np.asarray(cells, dtype=float)
    k, n = X.shape
    amb = np.asarray(ambient, dtype=float) + 1e-9
    amb = amb / amb.sum()
    # counts relative to the ambient proportions (an ambient-only hashtag then sits at the cell's ambient level)
    adj = X / (amb[:, None] * k)
    o = np.argsort(-adj, axis=0, kind="stable")
    srt = np.take_along_axis(adj, o, axis=0)
    top = srt[0]
    sec = srt[1] if k > 1 else np.zeros(n)
    # the cell's ambient level: the hashtags below the top two (a doublet's second hashtag must not set it)
    amb_level = np.median(srt[2:], axis=0) if k > 2 else np.zeros(n)
    lfc_top = np.log2((top + PSEUDO) / (sec + PSEUDO))
    lfc_second = np.log2((sec + PSEUDO) / (amb_level + PSEUDO))
    call = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return call, lfc_top, lfc_second
    md, mad = _mad(lfc_second)
    doublet = (lfc_second > md + DOUBLET_NMADS * mad) & (lfc_second >= DOUBLET_MIN)
    mt, madt = _mad(lfc_top[~doublet]) if (~doublet).any() else (0.0, 0.0)
    confident = (~doublet) & (lfc_top >= CONFIDENT_MIN) & (lfc_top >= mt - CONFIDENT_NMADS * madt)
    call[confident] = o[0][confident]
    call[doublet] = -2
    return call, lfc_top, lfc_second


SIDECAR_SCHEMA = "1"
FEATURE_SET = "hto"
METHOD_VERSION = "singlet-adt-hto/1 (hashedDrops rule; confident 2, doublet 2 + 3 MAD)"


def feature_sidecar(gsm: str, cols, call, lfc_top, hashtag_names):
    """Per-cell hashtag calls as a feature sidecar (``docs/CELL-ARCHITECTURE.md`` section 3), called cells only.

    Columns: ``gsm``, ``col`` (int64), ``hto/call`` (uint16 category: 0 negative, 1 doublet, 2 + k hashtag k of
    ``hashtag_names``), ``hto/margin`` (float32 in [0, 1]: ``1 - 2**-lfc_top`` = (top - second) / top on the
    ambient-scaled counts; 0 for negatives). Returns (pyarrow table, category names)."""
    import pyarrow as pa

    call = np.asarray(call)
    cat = np.where(call == -1, 0, np.where(call == -2, 1, call + 2)).astype(np.uint16)
    margin = np.clip(1 - np.power(2.0, -np.asarray(lfc_top, dtype=float)), 0, 1).astype(np.float32)
    margin[call == -1] = 0
    names = ["negative", "doublet"] + list(hashtag_names)
    t = pa.table({"gsm": pa.array([gsm] * len(cat), pa.string()), "col": pa.array(np.asarray(cols, np.int64)),
                  "hto/call": pa.array(cat), "hto/margin": pa.array(margin)})
    md = {b"singlet_features_schema": SIDECAR_SCHEMA.encode(), b"feature_set": FEATURE_SET.encode(),
          b"method_version": METHOD_VERSION.encode(), b"categories": "|".join(names).encode()}
    return t.replace_schema_metadata(md), names
