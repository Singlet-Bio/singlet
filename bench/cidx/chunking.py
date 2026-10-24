# SPDX-License-Identifier: MIT
"""Bytes fetched per gathered cell under the pilot's chunking vs a non-zero-budget chunk cut (docs/format-v2-cell-index.md, 10.2).

    python -P bench/cidx/chunking.py <dir with v2/m/GSE145370.json and v2/s/...>

For every sample and matrix (exon, intron) the chunk bytes are taken from the real chunk index; for the alternative cut
("a chunk ends when it would exceed B non-zeros or 4096 columns") the bytes of a chunk are estimated as its non-zeros times the
member's measured bytes per non-zero. Reported: expected bytes to fetch one random called cell, and 1,000 random called cells spread
over all samples (chunk runs coalesced, prefixes excluded: clients cache them)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main() -> None:
    from singlet import _pz2
    from singlet.v2.cells import CellsTable
    from singlet.v2.packs import PackStore

    root = Path(sys.argv[1])
    m = json.loads((root / "v2/m/GSE145370.json").read_text())
    st = PackStore()
    rng = np.random.default_rng(0)
    budgets = [None, 96_000, 48_000, 24_000, 12_000]
    one = {b: [] for b in budgets}
    picks = {b: 0 for b in budgets}
    called_total = 0
    per_sample = []
    for s in m["samples"]:
        b = (root / s["key"]).read_bytes()
        cm = s["members"]["cells"]
        t = CellsTable.decode(b[cm["off"] : cm["off"] + cm["len"]], st)
        called = np.flatnonzero(np.asarray(t.column("called"), bool))
        called_total += len(called)
        pick = np.sort(rng.choice(called, max(1, round(1000 * len(called) / 40614)), replace=False))
        cost_one = {bb: 0.0 for bb in budgets}
        cost_pick = {bb: 0 for bb in budgets}
        for name in ("exon_counts.1pz", "intron_counts.1pz"):
            me = s["members"][name]
            mem = _pz2.Member(b[me["off"] : me["off"] + me["len"]])
            ci = mem.chunk_index().astype(np.int64)  # off, len, first_col, nnz_before
            first, lens, nnzb = ci[:, 2], ci[:-1, 1], ci[:, 3]
            n = int(mem.n)
            colnnz = mem.column_counts(0, n).astype(np.int64)
            bpn = lens.sum() / max(1, nnzb[-1])
            for bb in budgets:
                if bb is None:
                    edges, cbytes = first, lens.astype(float)
                else:
                    edges = [0]
                    acc = 0
                    for j in range(n):
                        if (acc + colnnz[j] > bb and acc > 0) or j - edges[-1] >= 4096:
                            edges.append(j)
                            acc = 0
                        acc += colnnz[j]
                    edges.append(n)
                    edges = np.array(edges)
                    csum = np.concatenate([[0], np.cumsum(colnnz)])
                    cbytes = (csum[edges[1:]] - csum[edges[:-1]]) * bpn + 16
                k = np.searchsorted(edges, called, side="right") - 1
                cost_one[bb] += float(cbytes[k].mean())
                kk = np.unique(np.searchsorted(edges, pick, side="right") - 1)
                cost_pick[bb] += float(cbytes[kk].sum())
        for bb in budgets:
            one[bb].append((cost_one[bb], len(called)))
            picks[bb] += cost_pick[bb]
        per_sample.append({"gsm": s["gsm"], "n_columns": s["n_columns"], "called": len(called), "chunk_cols": s["members"]["exon_counts.1pz"]["chunk_cols"]})
    out = {}
    for bb in budgets:
        w = np.array([c for _, c in one[bb]], float)
        v = np.array([x for x, _ in one[bb]])
        out["current" if bb is None else f"nnz<= {bb}"] = {
            "bytes_one_random_cell": round(float((v * w).sum() / w.sum())),
            "bytes_1000_random_cells": round(picks[bb]),
            "bytes_per_cell_at_1000": round(picks[bb] / 1000),
        }
    print(json.dumps({"called_cells": called_total, "samples": per_sample, "results": out}, indent=1))


if __name__ == "__main__":
    main()
