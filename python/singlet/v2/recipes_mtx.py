# SPDX-License-Identifier: MIT
"""STARsolo MatrixMarket text <-> canonical CSC, numpy only (owner: W3).

STARsolo ``Solo.out/Gene/{raw,filtered}/matrix.mtx``::

    %%MatrixMarket matrix coordinate integer general
    %
    <n_rows> <n_cols> <nnz>
    <row> <col> <value>          1-based, sorted by column then row, "%d %d %d\\n"

``parse_mtx`` turns the text into a canonical ``Matrix`` (rows = STAR feature order, columns = barcode order of the
barcodes.tsv next to it); ``render_mtx`` writes it back byte for byte (``mtx-text-v1``).  W2 stores the parsed
matrix as a pz2 matrix (decision D7) and calls ``regenerate`` with the decoded matrix in the context.

Both directions work in blocks of about 16 MB of text / 2 M entries, so the peak memory for the largest sample seen
(126 MB of text, 20 M entries) stays near 300 MB.
"""
from __future__ import annotations

import numpy as np

from .recipes_ctx import Matrix

DEFAULT_HEADER = b"%%MatrixMarket matrix coordinate integer general\n%\n"
PARSE_BLOCK = 16 << 20
RENDER_BLOCK = 2_000_000


def _parse_ints(body) -> "np.ndarray | None":
    """Whitespace (space/newline) separated non-negative decimal integers -> int64 array; None on any other byte."""
    a = np.frombuffer(body, dtype=np.uint8)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    isdig = (a >= 48) & (a <= 57)
    isws = (a == 32) | (a == 10)
    if not (isdig | isws).all():
        return None
    prev = np.concatenate([[False], isdig[:-1]])
    start = np.flatnonzero(isdig & ~prev)
    nxt = np.concatenate([isdig[1:], [False]])
    end = np.flatnonzero(isdig & ~nxt)  # inclusive
    ln = end - start + 1
    if len(ln) == 0:
        return np.zeros(0, dtype=np.int64)
    if ln.max() > 18:
        return None
    out = np.zeros(len(start), dtype=np.int64)
    for k in range(int(ln.max())):
        has = ln > k
        idx = start[has] + k
        out[has] = out[has] * 10 + (a[idx].astype(np.int64) - 48)
    return out


def parse_mtx(b: bytes):
    """-> (Matrix, header_prefix[bytes]) or None when ``b`` is not in STARsolo's format (see ``render_mtx``)."""
    pos = 0
    n = len(b)
    while pos < n and b[pos : pos + 1] == b"%":
        e = b.find(b"\n", pos)
        if e < 0:
            return None
        pos = e + 1
    prefix = bytes(b[:pos])
    e = b.find(b"\n", pos)
    if e < 0:
        return None
    try:
        dims = [int(x) for x in b[pos:e].split(b" ")]
    except ValueError:
        return None
    if len(dims) != 3 or min(dims) < 0:
        return None
    m, ncol, nnz = dims
    mv = memoryview(b)[e + 1 :]
    if nnz == 0:
        return (Matrix(np.zeros(ncol + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), (m, ncol)), prefix) if len(mv) == 0 else None
    rows = np.empty(nnz, dtype=np.int32 if m < 2**31 else np.int64)
    cols = np.empty(nnz, dtype=np.int32 if ncol < 2**31 else np.int64)
    vals = np.empty(nnz, dtype=np.int64)
    got = 0
    p = 0
    L = len(mv)
    while p < L:
        q = min(L, p + PARSE_BLOCK)
        if q < L:
            nl = bytes(mv[q : q + 64]).find(b"\n")
            if nl < 0:
                return None
            q += nl + 1
        flat = _parse_ints(mv[p:q])
        if flat is None or flat.size % 3 != 0:
            return None
        k = flat.size // 3
        if got + k > nnz:
            return None
        t = flat.reshape(k, 3)
        rows[got : got + k] = t[:, 0] - 1
        cols[got : got + k] = t[:, 1] - 1
        vals[got : got + k] = t[:, 2]
        got += k
        p = q
    if got != nnz:
        return None
    if rows.min() < 0 or rows.max() >= m or cols.min() < 0 or cols.max() >= ncol:
        return None
    # canonical order: strictly increasing (col, row)
    if nnz > 1:
        prev = None
        for a in range(0, nnz, RENDER_BLOCK):
            bnd = min(nnz, a + RENDER_BLOCK + 1)
            key = cols[a:bnd].astype(np.int64) * np.int64(m) + rows[a:bnd].astype(np.int64)
            if not (np.diff(key) > 0).all():
                return None
    indptr = np.zeros(ncol + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(cols, minlength=ncol))
    return Matrix(indptr, rows.astype(np.int64), vals, (m, ncol), None, None, None), prefix


def _digits(v: np.ndarray):
    nd = np.ones(v.shape, dtype=np.int64)
    p = 10
    while True:
        m = v >= p
        if not m.any():
            break
        nd += m
        p *= 10
    return nd


def _put(buf, start, v, nd):
    for k in range(int(nd.max()) if len(nd) else 0):
        has = nd > k
        d = (v[has] // (10 ** (nd[has] - 1 - k))) % 10
        buf[start[has] + k] = 48 + d.astype(np.uint8)


def render_mtx(mat: Matrix, header_prefix: bytes = DEFAULT_HEADER) -> bytes:
    m, ncol = mat.shape
    ip = np.asarray(mat.indptr, dtype=np.int64)
    nnz = int(ip[-1])
    parts = [header_prefix + ("%d %d %d\n" % (m, ncol, nnz)).encode("ascii")]
    for a in range(0, nnz, RENDER_BLOCK):
        b = min(nnz, a + RENDER_BLOCK)
        cols = np.searchsorted(ip, np.arange(a, b, dtype=np.int64), side="right")   # count of indptr entries <= e == 1-based column
        rows = np.asarray(mat.indices[a:b], dtype=np.int64) + 1
        vals = np.asarray(mat.data[a:b], dtype=np.int64)
        dr, dc, dv = _digits(rows), _digits(cols), _digits(vals)
        ln = dr + dc + dv + 3
        off = np.concatenate([[0], np.cumsum(ln)])
        buf = np.empty(int(off[-1]), dtype=np.uint8)
        s = off[:-1]
        _put(buf, s, rows, dr)
        buf[s + dr] = 32
        _put(buf, s + dr + 1, cols, dc)
        buf[s + dr + 1 + dc] = 32
        _put(buf, s + dr + dc + 2, vals, dv)
        buf[s + ln - 1] = 10
        parts.append(buf.tobytes())
    return b"".join(parts)


def select_columns(mat: Matrix, keep: np.ndarray) -> Matrix:
    """Columns ``keep`` (ascending indices) of ``mat`` renumbered 0..k-1."""
    keep = np.asarray(keep, dtype=np.int64)
    ip = np.asarray(mat.indptr, dtype=np.int64)
    cnt = ip[keep + 1] - ip[keep]
    new_ip = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    tot = int(new_ip[-1])
    if tot:
        starts = np.repeat(ip[keep] - new_ip[:-1], cnt)
        idx = np.arange(tot, dtype=np.int64) + starts
    else:
        idx = np.zeros(0, dtype=np.int64)
    return Matrix(new_ip, mat.indices[idx], mat.data[idx], (mat.shape[0], len(keep)), None, None, None)
