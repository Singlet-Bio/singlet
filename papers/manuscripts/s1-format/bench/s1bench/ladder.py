"""Stage `ladder` (Q6): payload bits per non-zero as each idea is added, from raw arrays to .s1.

Only the count payload is measured here (row ids and values; no names, no container), so the steps are comparable.
L0  CSC as int32 row ids + int32 values                         64 bits/nnz by construction
L1  absolute row ids and values as LEB128 varints               what a naive varint format stores
L2  per-column counts, row gaps and values as three LEB128 streams, zstd level 3
L3  the same, zstd level 19
L4  the .s1 pz2 payload: entropy tables + rows section + chunks (header, chunk index and names excluded)
The model floor (F) is added by `report` from the `floor` stage.
"""

from __future__ import annotations

import numpy as np

from .util import Canon, Config, Unit, load_canon, log, result_path, write_tsv
from .write import format_dir


def leb128_len(v: np.ndarray) -> np.ndarray:
    """Bytes per value of an unsigned LEB128 varint."""
    v = v.astype(np.uint64)
    n = np.ones(v.shape, dtype=np.int64)
    t = v >> np.uint64(7)
    while t.any():
        n += t > 0
        t >>= np.uint64(7)
    return n


def leb128_encode(v: np.ndarray) -> bytes:
    v = v.astype(np.uint64)
    out = bytearray()
    lens = leb128_len(v)
    buf = np.zeros(int(lens.sum()), dtype=np.uint8)
    pos = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rem = v.copy()
    for k in range(int(lens.max()) if len(lens) else 0):
        live = lens > k
        byte = (rem[live] & np.uint64(0x7F)).astype(np.uint8)
        more = lens[live] > k + 1
        byte[more] |= 0x80
        buf[pos[live] + k] = byte
        rem[live] >>= np.uint64(7)
    out += buf.tobytes()
    return bytes(out)


def gaps(c: Canon) -> np.ndarray:
    """row - previous row - 1 within each column, previous row = -1 at a column start (the codec 2 transform)."""
    idx = c.indices.astype(np.int64)
    g = np.empty_like(idx)
    if len(idx):
        g[0] = idx[0]
        g[1:] = idx[1:] - idx[:-1] - 1
        starts = c.indptr[:-1][np.diff(c.indptr) > 0]
        g[starts] = idx[starts]
    return g


def s1_payload_bytes(path) -> dict:
    from singlet.s1.codec import parse_header
    from singlet.s1.container import S2Reader

    body = S2Reader(str(path)).read_member("counts")
    h = parse_header(body)
    sec = h["sections"]
    payload = sum(sec[k][1] for k in ("tables", "rows", "chunks"))
    return {"s1_member": len(body), "s1_payload": payload, "s1_chunks": sec["chunks"][1], "s1_tables": sec["tables"][1],
            "s1_codec": h["codec"]}


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    import zstandard as zstd

    for u in units:
        out = result_path(cfg, "ladder", u)
        if out.exists() and not force:
            continue
        c = load_canon(cfg, u)
        nnz = max(1, c.nnz)
        counts = np.diff(c.indptr)
        streams = [leb128_encode(counts), leb128_encode(gaps(c)), leb128_encode(c.data)]
        row = {"dataset": u.id, "matrix": u.name, "nnz": c.nnz,
               "L0_csc_i32": 64.0,
               "L1_leb128": 8.0 * (leb128_len(c.indices).sum() + leb128_len(c.data).sum()) / nnz}
        for lvl, key in ((3, "L2_gaps_leb_zstd3"), (19, "L3_gaps_leb_zstd19")):
            z = zstd.ZstdCompressor(level=lvl, threads=int(cfg["threads"]["headline"]))
            row[key] = 8.0 * sum(len(z.compress(s)) for s in streams) / nnz
        s1 = format_dir(cfg, u) / "s1.s1"
        if s1.exists():
            b = s1_payload_bytes(s1)
            row["L4_s1_payload"] = 8.0 * b["s1_payload"] / nnz
            row["L4_s1_member"] = 8.0 * b["s1_member"] / nnz
            row["s1_codec"] = b["s1_codec"]
        write_tsv([row], out)
        log(f"{u.key}: ladder {row}")
