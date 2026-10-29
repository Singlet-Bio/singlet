# SPDX-License-Identifier: MIT
"""Write one synthetic partition in each candidate shard format and measure bytes and Python scan speed.

    python bench/cidx/synth.py --cells 4000000 --out <dir>
    python bench/cidx/formats.py <dir>            # writes <dir>/fmt/* and prints a table; node bench/cidx/scan.mjs <dir> scans them in JS

Candidates (docs/format-store-cell-index.md, "Format choice"):
  parquet-zstd     Parquet, zstd level 3, dictionary where pyarrow picks it, row groups of 131,072 rows, page index
  parquet-snappy   same, snappy
  parquet-plain    same, uncompressed, DELTA_BINARY_PACKED for the integer columns that are not dictionary-coded
  scx-exact        custom columnar (singlet.store.atlas SCX1): per row group per column frame-of-reference bit packing, no
                   general-purpose compression; sample column run-length coded; QC counts exact
  scx-quant        same, QC counts as 1/256-octave log codes (u16)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

ROW_GROUP = 131072
QC = ("umi_exon", "umi_intron", "feat_exon", "feat_intron")


def qcode(x) -> np.ndarray:
    """1/256-octave log code (the rejected "scx-quant" variant): 0 -> 0, x -> 1 + 256 e + floor((x - 2^e) 256 / 2^e)."""
    x = np.asarray(x, dtype=np.uint64)
    out = np.zeros(x.shape, dtype=np.uint16)
    nz = x > 0
    xv = x[nz]
    e = np.floor(np.log2(xv.astype(np.float64))).astype(np.int64)
    e = np.where((np.uint64(1) << e.astype(np.uint64)) > xv, e - 1, e)
    p = np.uint64(1) << e.astype(np.uint64)
    out[nz] = (
        1 + 256 * e + (((xv - p) << np.uint64(8)) >> e.astype(np.uint64)).astype(np.int64)
    ).astype(np.uint16)
    return out


def parquet(d: dict, path: Path, compression: str, plain: bool = False) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.table({k: pa.array(v) for k, v in d.items()})
    kw: dict = dict(
        compression=compression,
        row_group_size=ROW_GROUP,
        write_page_index=True,
        write_statistics=True,
    )
    if plain:
        dict_cols = [
            "sample",
            "tissue",
            "ct",
            "sex",
            "anc",
            "ct_score",
            "anc_prob",
            "mt",
            "doublet",
        ]
        kw.update(
            use_dictionary=dict_cols,
            column_encoding={k: "DELTA_BINARY_PACKED" for k in d if k not in dict_cols},
        )
    pq.write_table(t, path, **kw)


def main() -> None:
    from singlet.store import atlas

    root = Path(sys.argv[1])
    d = dict(np.load(root / "part.npz"))
    n = len(d["sample"])
    out = root / "fmt"
    out.mkdir(exist_ok=True)
    files = {}
    for name, comp, plain in (
        ("parquet-zstd", "zstd", False),
        ("parquet-snappy", "snappy", False),
        ("parquet-plain", "none", True),
    ):
        p = out / f"{name}.parquet"
        parquet(d, p, comp, plain)
        files[name] = p
    p = out / "scx-exact.scx"
    p.write_bytes(atlas.encode_shard(d, group_rows=ROW_GROUP))
    files["scx-exact"] = p
    q = dict(d)
    for k in QC:
        q[k] = qcode(d[k])
    p = out / "scx-quant.scx"
    p.write_bytes(atlas.encode_shard(q, group_rows=ROW_GROUP))
    files["scx-quant"] = p
    import zstandard

    raw = sum(v.nbytes for v in d.values())
    print(f"{n:,} cells, {len(np.unique(d['sample'])):,} samples, numpy bytes {raw / n:.2f} B/cell")
    print(f"{'format':16s} {'MB':>8s} {'B/cell':>7s} {'+zstd B/cell':>12s} {'py scan ms':>10s}")
    for name, p in files.items():
        b = p.read_bytes()
        z = len(zstandard.ZstdCompressor(level=3).compress(b))
        t0 = time.perf_counter()
        for _ in range(3):
            if name.startswith("parquet"):
                import pyarrow.parquet as pq

                t = pq.read_table(p, columns=["sample", "col", "umi_exon", "ct", "tissue"])
                m = (t.column("umi_exon").to_numpy() >= 1000) & (t.column("ct").to_numpy() == 101)
            else:
                sh = atlas.Shard(b)
                m = None
                for g in range(sh.n_groups):
                    u = sh.column(g, "umi_exon")
                    c = sh.column(g, "ct")
                    sh.column(g, "sample")
                    sh.column(g, "col")
                    mm = (u >= (1000 if name == "scx-exact" else qcode(np.array([1000]))[0])) & (
                        c == 101
                    )
                    m = mm if m is None else np.concatenate([m, mm])
        dt = (time.perf_counter() - t0) / 3 * 1000
        print(
            f"{name:16s} {len(b) / 1e6:8.2f} {len(b) / n:7.2f} {z / n:12.2f} {dt:10.1f}   matches={int(m.sum()):,}"
        )


if __name__ == "__main__":
    main()
