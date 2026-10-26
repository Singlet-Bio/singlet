# SPDX-License-Identifier: MIT
"""pz2 chunk budget (EncodeOptions.chunk_nnz_max): member size, chunk count, encode/decode time and bytes moved per gathered cell.

    python -P bench/pz2_chunks/measure.py <corpus dir> [--out results.json]

For every corpus sample (``<dir>/<id>/{exon,intron}_counts.1pz`` + ``cell_calls.tsv``) and the first samples of every
``<dir>/bundles/*.singlet`` (``samples/<GSM>/...``), the exon and intron matrices are re-encoded with the default chunk-width policy and each budget (none =
fixed-width chunks, the behaviour before; then 96k, 48k, 24k, 12k non-zeros). Names are not stored (the migration stores them as
pack/cells references), so sizes are the matrix payload: header + chunk index + tables + rows + chunks. Reported per budget:

* ``bytes``: member size; ``chunks``: number of chunks; ``enc_s`` / ``dec_s``: best of 3 (full encode, full decode, 1 thread);
* ``one_cell``: expected chunk bytes to fetch one random called cell (exon + intron; prefixes excluded, clients cache them);
* ``pick_1pct`` / ``pick_10pct``: chunk bytes for a uniform random 1 % / 10 % of the called cells (each touched chunk once);
* ``all_called``: chunk bytes for every called cell.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

BUDGETS = [-1, 98304, 49152, 24576, 12288]
NONE = {"kind": "none"}
METRICS = ("bytes", "chunks", "enc_s", "dec_s", "one_cell", "pick_1pct", "pick_10pct", "all_called")


def called_mask(colnames, cell_calls: bytes) -> np.ndarray:
    called = set()
    for line in cell_calls.decode().splitlines()[1:]:
        p = line.split("\t")
        if len(p) > 1 and p[1].upper() == "TRUE":
            called.add(p[0])
    return np.array([c in called for c in colnames], bool)


def best(f, k=3):
    t = []
    out = None
    for _ in range(k):
        t0 = time.perf_counter()
        out = f()
        t.append(time.perf_counter() - t0)
    return min(t), out


def measure_pair(name: str, mats: dict, called: np.ndarray, rng) -> dict:
    from singlet import _pz2
    from singlet.store.matrix import Csc

    res = {"sample": name, "n_columns": int(len(called)), "n_called": int(called.sum()), "budgets": {}}
    cc = np.flatnonzero(called)
    picks = {
        "pick_1pct": np.sort(rng.choice(cc, max(1, len(cc) // 100), replace=False)),
        "pick_10pct": np.sort(rng.choice(cc, max(1, len(cc) // 10), replace=False)),
    }
    csc = {k: Csc.from_legacy(v) for k, v in mats.items()}
    for bud in BUDGETS:
        row = {k: 0 for k in METRICS}
        row["chunk_cols"] = {}
        for k, c in csc.items():
            ip = c.indptr.astype(np.int64)
            ix = np.ascontiguousarray(c.indices, np.uint32)
            dv = np.ascontiguousarray(c.data, np.uint32)

            def enc(ip=ip, ix=ix, dv=dv, c=c):
                return _pz2.encode(
                    ip, ix, dv, int(c.m), int(c.n), chunk_nnz_max=bud, row_names=NONE, col_names=NONE, compute_digest=False, threads=1
                )

            te, (blob, st) = best(enc)
            blob = bytes(blob)
            td, _ = best(lambda blob=blob: _pz2.decode(blob, mode="native", threads=1))
            ci = _pz2.Member(blob).chunk_index().astype(np.int64)
            first, lens = ci[:, 2], ci[:-1, 1]
            k_of = np.searchsorted(first, cc, side="right") - 1
            row["bytes"] += len(blob)
            row["chunks"] += int(len(lens))
            row["enc_s"] += te
            row["dec_s"] += td
            row["one_cell"] += float(lens[k_of].mean()) if len(cc) else 0.0
            row["all_called"] += int(lens[np.unique(k_of)].sum())
            for pk, sel in picks.items():
                row[pk] += int(lens[np.unique(np.searchsorted(first, sel, side="right") - 1)].sum())
            row["chunk_cols"][k] = int(st["chunk_cols"])
        row["enc_s"] = round(row["enc_s"], 3)
        row["dec_s"] = round(row["dec_s"], 3)
        row["one_cell"] = round(row["one_cell"])
        res["budgets"]["none" if bud < 0 else str(bud)] = row
        print(name, bud, {k: v for k, v in row.items() if k != "chunk_cols"}, file=sys.stderr, flush=True)
    return res


def main() -> None:
    from singlet.store.verify import read_1pz

    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out")
    ap.add_argument("--bundle-samples", type=int, default=3, help="samples per bundle")
    a = ap.parse_args()
    root = Path(a.corpus)
    rng = np.random.default_rng(0)
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "exon_counts.1pz").exists()):
        mats = {k: read_1pz((d / f"{k}_counts.1pz").read_bytes()) for k in ("exon", "intron") if (d / f"{k}_counts.1pz").exists()}
        called = called_mask(mats["exon"]["colnames"], (d / "cell_calls.tsv").read_bytes())
        out.append(measure_pair(d.name, mats, called, rng))
    bundles = sorted((root / "bundles").glob("*.singlet")) if (root / "bundles").exists() else []
    for b in bundles:
        z = zipfile.ZipFile(b)
        names = set(z.namelist())
        dirs = sorted({n.rsplit("/", 1)[0] for n in names if n.endswith("/exon_counts.1pz")})[: a.bundle_samples]
        for g in dirs:  # samples/<GSM>
            mats = {k: read_1pz(z.read(f"{g}/{k}_counts.1pz")) for k in ("exon", "intron") if f"{g}/{k}_counts.1pz" in names}
            called = called_mask(mats["exon"]["colnames"], z.read(f"{g}/cell_calls.tsv"))
            out.append(measure_pair(f"{b.stem}/{g.rsplit('/', 1)[-1]}", mats, called, rng))
    tot = {}
    for bud in out[0]["budgets"]:
        tot[bud] = {k: round(sum(r["budgets"][bud][k] for r in out), 3) for k in METRICS}
    base = tot["none"]
    for t in tot.values():
        for k in ("bytes", "one_cell", "pick_1pct", "pick_10pct", "all_called", "dec_s"):
            t[k + "_vs_none"] = round(t[k] / base[k], 4) if base[k] else None
    doc = {"samples": out, "totals": tot}
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1))
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
