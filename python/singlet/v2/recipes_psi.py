# SPDX-License-Identifier: MIT
"""psi-v1: ``splice_psi.1pz`` from ``sj_counts.1pz`` (owner: W3).

Definition (include/singlet/pileup/splice_psi.h, bit-identical numpy port, checked against the production C++ in
bench/compression/notes/w4_bugs.md and e4.md):

  * junction names ``chrom:start-end`` (first ':' splits the chromosome, first '-' after it splits start and end;
    numbers read like ``std::stoul`` and held as uint32); a junction that does not parse belongs to no event
  * donor group ``chrom:start-D`` / acceptor group ``chrom:end-A``; a group with >= 2 junctions is an event
  * events are sorted by their id string, bytewise; a junction takes the FIRST event of that order it belongs to
    (not "donor first": ``chr1:100007033-A`` sorts before ``chr1:99993741-D``)
  * per cell, event sum = sum of the junction counts of the event; entries with a positive sum are emitted in the
    sj column order with PSI = float32(count) / float32(event sum)
  * the member stores the PSI cast to an integer.  The rule depends on the pipeline build:
        psi-v1-rint   nearbyint (round half to even) - GSE145370 (pipeline 0.3.0, git 8e1803b) and 8 corpus samples
        psi-v1-trunc  static_cast<uint32_t> truncation, i.e. only PSI == 1.0 survives - 30 corpus samples
        psi-v1-stub   the constant 151-byte 0x0 file the pipeline writes for a failed output - 4 corpus samples
    (the stored values are what the legacy files contain; ``singlet.psi()`` computes the true float PSI with
    :func:`compute_psi`, which is the same function.)
"""
from __future__ import annotations

import numpy as np

from . import recipes_pzw as pzw
from .recipes_ctx import Matrix, decode_1pz_bytes

MAX_EXACT_SUM = 1 << 24  # float32 holds integers exactly below this; larger event sums would need ordered float adds


# ---------------------------------------------------------------------------------------------------
# junction names and events
# ---------------------------------------------------------------------------------------------------
def _stoul(s: str):
    """std::stoul semantics on a str (leading blanks, optional sign, digits, trailing garbage ignored); None on failure."""
    i, n = 0, len(s)
    while i < n and s[i] in " \t\n\v\f\r":
        i += 1
    neg = False
    if i < n and s[i] in "+-":
        neg = s[i] == "-"
        i += 1
    j = i
    while j < n and "0" <= s[j] <= "9":
        j += 1
    if j == i:
        return None
    v = int(s[i:j])
    if neg:
        v = (-v) % (1 << 64)
    if v >= 1 << 64:
        return None
    return v


def parse_junction_names(names):
    """Return (chrom_code[int64], chroms[list], start[int64 uint32 values], end[int64], ok[bool])."""
    n = len(names)
    cc = np.full(n, -1, dtype=np.int64)
    st = np.zeros(n, dtype=np.int64)
    en = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    chrom_ix = {}
    chroms = []
    for i, nm in enumerate(names):
        c = nm.find(":")
        if c < 0:
            continue
        d = nm.find("-", c + 1)
        if d < 0:
            continue
        a = nm[c + 1 : d]
        b = nm[d + 1 :]
        if a.isascii() and a.isdigit() and b.isascii() and b.isdigit() and len(a) < 19 and len(b) < 19:
            s, e = int(a), int(b)
        else:
            s, e = _stoul(a), _stoul(b)
            if s is None or e is None:
                continue
        ch = nm[:c]
        k = chrom_ix.get(ch)
        if k is None:
            k = chrom_ix[ch] = len(chroms)
            chroms.append(ch)
        cc[i] = k
        st[i] = s & 0xFFFFFFFF
        en[i] = e & 0xFFFFFFFF
        ok[i] = True
    return cc, chroms, st, en, ok


def build_events(names):
    """Splice events of a junction list (splice_psi.h:build_splice_events): [(event_id, type, member junction indices
    ascending)] sorted bytewise by event id.  Groups with >= 2 junctions only; unparsable names belong to no group."""
    cc, chroms, st, en, ok = parse_junction_names(names)
    idx = np.flatnonzero(ok)
    groups = []
    if idx.size == 0:
        return groups
    for kind, pos, typ in (("D", st, "alt_donor"), ("A", en, "alt_acceptor")):
        key = (cc[idx] << 32) | pos[idx]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        brk = np.flatnonzero(np.diff(ks)) + 1
        starts = np.concatenate([[0], brk])
        ends = np.concatenate([brk, [len(ks)]])
        for g in np.flatnonzero(ends - starts >= 2):
            members = np.sort(idx[order[starts[g] : ends[g]]])
            j0 = members[0]
            groups.append(("%s:%d-%s" % (chroms[cc[j0]], pos[j0], kind), typ, members))
    groups.sort(key=lambda g: g[0].encode("utf-8"))
    return groups


def junction_events(names):
    """For every junction the index (in the bytewise-sorted event list) of the first event it belongs to, -1 if none.

    Returns (junction_event[int64], event_ids[list[str]] sorted bytewise)."""
    groups = build_events(names)
    je = np.full(len(names), -1, dtype=np.int64)
    # first event wins: assign in reverse order so the smallest event index is written last
    for e in range(len(groups) - 1, -1, -1):
        je[groups[e][2]] = e
    return je, [g[0] for g in groups]


def compute_psi(sj: Matrix, names=None):
    """float32 PSI per emitted entry.  Returns (indptr[int64], rows[int64], psi[float32], max_event_sum).

    Bit-identical to singlet::compute_psi as long as every event sum is < 2^24 (checked via ``max_event_sum``)."""
    names = names if names is not None else sj.rownames
    m, n = sj.shape
    if not names or sj.nnz == 0 or n == 0:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), 0
    je, ids = junction_events(names)
    if not ids:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), 0
    rows = sj.indices.astype(np.int64)
    ev = je[rows]
    sel = np.flatnonzero(ev >= 0)
    col = sj.col_ids()
    key = col[sel] * len(ids) + ev[sel]
    uq, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=sj.data[sel].astype(np.float64), minlength=len(uq))
    maxsum = int(sums.max()) if len(sums) else 0
    esum = sums[inv]
    keep = esum > 0
    sel = sel[keep]
    psi = (sj.data[sel].astype(np.float32) / esum[keep].astype(np.float32)).astype(np.float32)
    out_col = col[sel]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(out_col, minlength=n))]).astype(np.int64)
    return indptr, rows[sel], psi, maxsum


def quantise(psi: np.ndarray, rule: str) -> np.ndarray:
    if rule == "rint":
        return np.rint(psi).astype(np.uint32)
    if rule == "trunc":
        return psi.astype(np.uint32)
    raise ValueError("unknown psi rule %r" % rule)


def psi_matrix(sj: Matrix, rule: str) -> Matrix:
    indptr, rows, psi, _ = compute_psi(sj)
    return Matrix(indptr, rows, quantise(psi, rule), (sj.shape[0], sj.shape[1]), sj.rownames, sj.colnames, {})


# ---------------------------------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------------------------------
class NotByteExact(LookupError):
    """The plan only guarantees the decoded matrix (no available writer reproduces the container bytes)."""


_LAST_WRITER = []


def _order_writers(ws):
    ids = list(ws)
    for w in _LAST_WRITER:
        if w in ids:
            ids.remove(w)
            ids.insert(0, w)
    return ids


def plan_splice_psi(member, legacy_bytes, ctx):
    """Plan for ``splice_psi.1pz``.  Returns (recipe_id, params, needs) or None."""
    if legacy_bytes == pzw.STUB_1PZ:
        return ("psi-v1-stub", {}, [])
    sj = ctx.matrix("sj_counts.1pz")
    if sj is None or sj.rownames is None:
        return None
    hdr = pzw.parse_header(legacy_bytes) if len(legacy_bytes) >= 96 else None
    if hdr is None or hdr["magic"] != 0x5A315054:
        return None
    leg = ctx.matrix(member)
    if leg is None:
        leg = decode_1pz_bytes(legacy_bytes)
    if tuple(leg.shape) != tuple(sj.shape):
        return None
    indptr, rows, psi, maxsum = compute_psi(sj)
    if maxsum >= MAX_EXACT_SUM:
        return None
    if not (np.array_equal(indptr, leg.indptr.astype(np.int64)) and np.array_equal(rows, leg.indices.astype(np.int64))):
        return None
    if leg.rownames != list(sj.rownames) or leg.colnames != list(sj.colnames or []):
        return None
    rule = None
    for r in ("rint", "trunc"):
        if np.array_equal(quantise(psi, r).astype(np.int64), leg.data.astype(np.int64)):
            rule = r
            break
    if rule is None:
        return None
    params = {"level": int(hdr["codec_level"]), "chunk_cols": int(hdr["chunk_cols"]), "writer": None}
    if leg.user_kv:
        params["user_kv"] = dict(leg.user_kv)
    m = Matrix(indptr, rows, quantise(psi, rule), tuple(sj.shape), sj.rownames, sj.colnames, {})
    ws = pzw.available_writers()
    for wid in _order_writers(ws):
        try:
            b = ws[wid].write(m, params["level"], params["chunk_cols"], 4, params.get("user_kv"))
        except Exception:
            continue
        if b == legacy_bytes:
            params["writer"] = wid
            _LAST_WRITER[:] = [wid]
            break
    return ("psi-v1-" + rule, params, ["matrix:sj_counts.1pz"])


def regenerate_matrix_psi(recipe_id, params, ctx) -> Matrix:
    if recipe_id == "psi-v1-stub":
        return Matrix(np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint32), (0, 0), [], [], {})
    rule = recipe_id.rsplit("-", 1)[1]
    sj = ctx.matrix("sj_counts.1pz")
    m = psi_matrix(sj, rule)
    if params.get("user_kv"):
        m.user_kv = dict(params["user_kv"])
    return m


def regenerate_psi(recipe_id, params, ctx) -> bytes:
    if recipe_id == "psi-v1-stub":
        return pzw.STUB_1PZ
    wid = params.get("writer")
    if wid is None:
        raise NotByteExact("no writer reproduced the container bytes of this member; use regenerate_matrix")
    m = regenerate_matrix_psi(recipe_id, params, ctx)
    return pzw.write_legacy_1pz(m, wid, params.get("level", 3), params.get("chunk_cols", 1024), 4, params.get("user_kv"))
