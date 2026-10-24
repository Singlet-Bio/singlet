# SPDX-License-Identifier: MIT
"""2-bit sequence codes and Hamming-1 neighbourhoods, vectorised with numpy.

A sequence of length L <= 32 over ACGT is the integer sum(code(b_i) << 2*(L-1-i)) with A=0, C=1, G=2, T=3, so codes sort
like the strings. Positions holding anything else (N) are reported in a mask; such sequences have no code.
"""

from __future__ import annotations

import numpy as np

_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _LUT[_b] = _i
    _LUT[_b + 32] = _i  # lower case
BASES = "ACGT"


def as_bytes_matrix(seqs, width: int) -> np.ndarray:
    """Fixed-width byte matrix (n, width) from an ``S<width>`` array or a list of str/bytes of exactly ``width``."""
    a = np.asarray(seqs)
    if a.dtype.kind == "U":
        a = np.char.encode(a, "ascii")
    if a.dtype.kind != "S":
        raise TypeError("expected strings")
    a = a.astype(f"S{width}")
    return a.view(np.uint8).reshape(len(a), width)


def encode(mat: np.ndarray):
    """(n, L) byte matrix -> (codes uint64, n_invalid uint8 per row). Rows with N get code 0 and n_invalid > 0."""
    if mat.shape[1] > 32:
        raise ValueError("sequences longer than 32 nt do not fit a 64-bit code")
    v = _LUT[mat]
    bad = (v > 3).sum(axis=1).astype(np.uint8)
    v = np.where(v > 3, 0, v).astype(np.uint64)
    L = mat.shape[1]
    shifts = (2 * np.arange(L - 1, -1, -1)).astype(np.uint64)
    codes = (v << shifts).sum(axis=1, dtype=np.uint64) if L else np.zeros(len(mat), np.uint64)
    return codes, bad


def encode_strings(seqs, width: int):
    return encode(as_bytes_matrix(seqs, width))


def decode(codes, L: int) -> list:
    codes = np.asarray(codes, dtype=np.uint64)
    out = []
    for c in codes.tolist():
        out.append("".join(BASES[(c >> (2 * (L - 1 - i))) & 3] for i in range(L)))
    return out


def neighbours(codes: np.ndarray, L: int) -> np.ndarray:
    """All Hamming-1 substitutions: (n, 3L) uint64, column 3*p + k = base at position p replaced by its k-th alternative."""
    codes = np.asarray(codes, dtype=np.uint64)
    out = np.empty((len(codes), 3 * L), dtype=np.uint64)
    for p in range(L):
        sh = np.uint64(2 * (L - 1 - p))
        cur = (codes >> sh) & np.uint64(3)
        base = codes & ~(np.uint64(3) << sh)
        for k in range(1, 4):
            alt = (cur + np.uint64(k)) & np.uint64(3)
            out[:, 3 * p + k - 1] = base | (alt << sh)
    return out


def n_fill(mat: np.ndarray) -> tuple:
    """For rows with exactly one N: the 4 codes with that N replaced by A/C/G/T, as (row index, (k,4) codes)."""
    v = _LUT[mat]
    badpos = v > 3
    one = np.flatnonzero(badpos.sum(axis=1) == 1)
    if len(one) == 0:
        return one, np.zeros((0, 4), np.uint64)
    sub = v[one].copy()
    pos = badpos[one].argmax(axis=1)
    out = np.empty((len(one), 4), np.uint64)
    L = mat.shape[1]
    shifts = (2 * np.arange(L - 1, -1, -1)).astype(np.uint64)
    for b in range(4):
        sub[np.arange(len(one)), pos] = b
        out[:, b] = (sub.astype(np.uint64) << shifts).sum(axis=1, dtype=np.uint64)
    return one, out


def lookup(sorted_codes: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Index of each query in ``sorted_codes`` or -1."""
    if len(sorted_codes) == 0:
        return np.full(q.shape, -1, dtype=np.int64)
    i = np.searchsorted(sorted_codes, q)
    i = np.minimum(i, len(sorted_codes) - 1)
    hit = sorted_codes[i] == q
    return np.where(hit, i, -1)
