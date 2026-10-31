# SPDX-License-Identifier: MIT
"""Count antibody / hashtag UMIs per (cell barcode, feature) with Cell Ranger's feature-barcoding logic.

Steps (``count_library``):

1. **Collapse reads.** R1 gives the cell barcode (CB) and UMI, R2 the feature barcode at a fixed offset (measured per
   library). Reads collapse to unique ``(CB, UMI, feature sequence)`` keys with a read count (a sort, not a hash table,
   so memory is about 2x the unique keys).
2. **Feature barcodes.** Exact match to the panel, else a unique Hamming-1 neighbour of one panel sequence (one N is
   tried as each base). Ambiguous and unmatched sequences are counted, and the most frequent unmatched ones are kept
   for the panel-completeness check.
3. **Cell barcodes.** Exact whitelist match, else the Hamming-1 whitelist neighbours weighted by how often each was
   seen exactly (pseudocount 0.5); the best is taken when its posterior is >= 0.975 (Cell Ranger and STARsolo
   ``1MM_multi`` use the same rule, Cell Ranger also weights by base quality, which is not kept here). On 3' v3
   TotalSeq-B libraries the corrected feature-side barcode is translated to its GEX barcode.
4. **UMIs.** Within one (CB, feature), a UMI moves to its Hamming-1 neighbour with the most reads when that has more
   reads, or as many and is lexicographically larger (one step, exactly Cell Ranger's ``correct_umis``); UMIs with N are
   dropped.
5. **Matrix.** Molecules whose CB is a column of the paired GEX sample (the v2 ``cells`` table, in ``col`` order) give
   the features x columns matrix. Molecules on other whitelist barcodes are kept as the per-feature *ambient* profile and
   a per-barcode total histogram (empty droplets for DSB and for the QC background).

Read-level statistics follow Cell Ranger's antibody metrics: valid barcodes, reads with a recognised feature,
fraction in GEX columns and in called cells, sequencing saturation ``1 - molecules / reads``.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import seq as S

POSTERIOR_MIN = 0.975
PSEUDOCOUNT = 0.5
CHUNK = 4_000_000
N_UNMATCHED_KEEP = 200


# ----------------------------------------------------------------------------------------------------------- reading
def _open(path):
    if path.endswith(".gz"):
        try:
            from isal import igzip  # fastest when present

            return igzip.open(path, "rb")
        except ImportError:
            return gzip.open(path, "rb")
    return open(path, "rb")


def iter_pairs(r1_paths: Sequence[str], r2_paths: Sequence[str]) -> Iterable[Tuple[bytes, bytes]]:
    """(R1 sequence, R2 sequence) byte strings of every read pair, file pair by file pair."""
    for p1, p2 in zip(r1_paths, r2_paths):
        try:
            import dnaio

            with dnaio.open(p1, file2=p2, mode="r") as rd:
                for a, b in rd:
                    yield a.sequence.encode(), b.sequence.encode()
        except ImportError:  # pragma: no cover - plain parser, slow
            with _open(p1) as f1, _open(p2) as f2:
                while True:
                    h = f1.readline()
                    if not h:
                        break
                    s1 = f1.readline().rstrip()
                    f1.readline(), f1.readline()
                    f2.readline()
                    s2 = f2.readline().rstrip()
                    f2.readline(), f2.readline()
                    yield s1, s2


BLOCK_BYTES = 64 << 20  # decompressed bytes read per refill in the FASTQ reader


def key_blocks_from_pairs(
    pairs,
    r1_len: int,
    offset: int,
    feat_len: int,
    max_reads: Optional[int] = None,
    block: int = CHUNK,
):
    """``(keys (k, r1_len + feat_len) uint8, records, short)`` per block, from (R1, R2) byte-string pairs."""
    end = offset + feat_len
    buf: List[bytes] = []
    n = n_short = total = 0
    for s1, s2 in pairs:
        n += 1
        total += 1
        if len(s1) < r1_len or len(s2) < end:
            n_short += 1
        else:
            buf.append(s1[:r1_len] + s2[offset:end])
        if len(buf) >= block:
            yield (
                np.array(buf, dtype=f"S{r1_len + feat_len}").view(np.uint8).reshape(len(buf), -1),
                n,
                n_short,
            )
            buf, n, n_short = [], 0, 0
        if max_reads is not None and total >= max_reads:
            break
    a = (
        np.array(buf, dtype=f"S{r1_len + feat_len}")
        if buf
        else np.zeros(0, f"S{r1_len + feat_len}")
    )
    yield a.view(np.uint8).reshape(len(buf), r1_len + feat_len), n, n_short


class _SeqLines:
    """Sequence lines of a FASTQ(.gz) file, read in blocks of records with numpy (no per-read Python).

    Decompression runs in a ``pigz -dc`` subprocess when pigz is on the PATH (else Python's gzip)."""

    def __init__(self, path: str):
        import shutil
        import subprocess

        self.proc = None
        if path.endswith(".gz") and shutil.which("pigz"):
            self.proc = subprocess.Popen(
                ["pigz", "-dc", "-p", "4", path], stdout=subprocess.PIPE, bufsize=1 << 20
            )
            self.fh = self.proc.stdout
        else:
            self.fh = _open(path)
        self.parts: List[
            np.ndarray
        ] = []  # unread bytes, as chunks (joined only when a block is cut)
        self.lines = 0  # newlines in self.parts
        self.eof = False

    def take(self, n: int):
        """Up to ``n`` records: (concatenated bytes as uint8, sequence-line starts, sequence-line lengths).

        Newlines are counted per new chunk and the chunks are joined once per block, so a block costs O(its bytes)."""
        while not self.eof and self.lines < 4 * n:
            chunk = self.fh.read(BLOCK_BYTES)
            if not chunk:
                self.eof = True
                if self.parts and self.parts[-1][-1] != 10:
                    self.parts.append(np.array([10], np.uint8))
                    self.lines += 1
                break
            a = np.frombuffer(chunk, np.uint8)
            self.parts.append(a)
            self.lines += int(np.count_nonzero(a == 10))
        if not self.parts:
            return np.zeros(0, np.uint8), np.zeros(0, np.int64), np.zeros(0, np.int64)
        buf = self.parts[0] if len(self.parts) == 1 else np.concatenate(self.parts)
        nl = np.flatnonzero(buf == 10)
        k = min(n, len(nl) // 4)
        if k == 0:
            self.parts, self.lines = [buf], len(nl)
            return buf[:0], np.zeros(0, np.int64), np.zeros(0, np.int64)
        cut = nl[4 * k - 1] + 1
        data, rest = buf[:cut], buf[cut:]
        self.parts = [rest] if len(rest) else []
        self.lines = len(nl) - 4 * k
        starts = nl[0 : 4 * k : 4] + 1
        ends = nl[1 : 4 * k : 4]
        if len(ends) and data[ends - 1].max(initial=0) == 13:  # CRLF files
            ends = ends - (data[ends - 1] == 13)
        return data, starts.astype(np.int64), (ends - starts).astype(np.int64)

    def close(self):
        try:
            self.fh.close()
        finally:
            if self.proc is not None:
                self.proc.wait()


def _windows(data, starts, lens, s0: int, w: int):
    """(k, w) uint8 window ``[s0, s0 + w)`` of each line; rows of lines shorter than ``s0 + w`` are garbage (masked)."""
    idx = starts[:, None] + (s0 + np.arange(w))[None, :]
    np.minimum(idx, len(data) - 1, out=idx)
    return data[idx], lens >= s0 + w


def key_blocks_from_fastq(
    r1_paths,
    r2_paths,
    r1_len: int,
    offset: int,
    feat_len: int,
    max_reads: Optional[int] = None,
    block: int = CHUNK,
):
    """The same blocks as :func:`key_blocks_from_pairs`, read straight from FASTQ files with numpy (~20x faster)."""
    total = 0
    for p1, p2 in zip(r1_paths, r2_paths):
        a, b = _SeqLines(p1), _SeqLines(p2)
        try:
            while True:
                want = block if max_reads is None else min(block, max_reads - total)
                if want <= 0:
                    return
                d1, s1, l1 = a.take(want)
                d2, s2, l2 = b.take(want)
                if len(s1) != len(s2):
                    raise ValueError(f"{p1} and {p2} have different numbers of reads")
                if len(s1) == 0:
                    break
                w1, ok1 = _windows(d1, s1, l1, 0, r1_len)
                w2, ok2 = _windows(d2, s2, l2, offset, feat_len)
                ok = ok1 & ok2
                total += len(s1)
                yield np.ascontiguousarray(np.hstack([w1[ok], w2[ok]])), len(s1), int((~ok).sum())
        finally:
            a.close()
            b.close()


def _collapse_blocks(blocks, width: int):
    uniq: List[np.ndarray] = []
    cnts: List[np.ndarray] = []
    n = n_short = 0
    for mat, k, short in blocks:
        n += k
        n_short += short
        if len(mat):
            u, c = np.unique(
                np.ascontiguousarray(mat).view(f"S{width}").ravel(), return_counts=True
            )
            uniq.append(u)
            cnts.append(c.astype(np.int64))
            if len(uniq) > 8:
                _merge(uniq, cnts)
    _merge(uniq, cnts)
    keys = uniq[0] if uniq else np.zeros(0, dtype=f"S{width}")
    counts = cnts[0] if cnts else np.zeros(0, np.int64)
    return keys, counts, {"reads": n, "reads_short": n_short}


def collapse_reads(
    pairs: Iterable[Tuple[bytes, bytes]],
    r1_len: int,
    offset: int,
    feat_len: int,
    max_reads: Optional[int] = None,
):
    """Unique ``R1[:r1_len] + R2[offset:offset+feat_len]`` keys and their read counts.

    Returns (keys ``S<r1_len+feat_len>``, counts int64, stats dict). Pairs with a short R1 or R2 are counted and skipped."""
    return _collapse_blocks(
        key_blocks_from_pairs(pairs, r1_len, offset, feat_len, max_reads), r1_len + feat_len
    )


def _merge(uniq, cnts):
    if len(uniq) <= 1:
        return
    a = np.concatenate(uniq)
    c = np.concatenate(cnts)
    u, inv = np.unique(a, return_inverse=True)
    s = np.bincount(inv.ravel(), weights=c, minlength=len(u)).astype(np.int64)
    uniq[:] = [u]
    cnts[:] = [s]


# ----------------------------------------------------------------------------------------------------------- features
@dataclass
class _FixedIndex:
    sequences: List[str]
    exact: np.ndarray = field(init=False)  # sorted codes
    exact_idx: np.ndarray = field(init=False)
    h1: np.ndarray = field(init=False)  # sorted neighbour codes
    h1_idx: np.ndarray = field(init=False)  # feature index, -2 = ambiguous

    def __post_init__(self):
        L = len(self.sequences[0])
        if any(len(s) != L for s in self.sequences):
            raise ValueError("panel sequences must have one length")
        self.L = L
        codes, bad = S.encode_strings(self.sequences, L)
        if bad.any():
            raise ValueError("panel sequences must be ACGT")
        if len(np.unique(codes)) != len(codes):
            raise ValueError("duplicate panel sequences")
        o = np.argsort(codes)
        self.exact, self.exact_idx = codes[o], o.astype(np.int64)
        nb = S.neighbours(codes, L)
        owner = np.repeat(np.arange(len(codes)), nb.shape[1])
        nb = nb.ravel()
        keep = S.lookup(self.exact, nb) < 0  # an exact panel sequence is never a neighbour
        nb, owner = nb[keep], owner[keep]
        u, first, cnt = np.unique(nb, return_index=True, return_counts=True)
        idx = owner[first].astype(np.int64)
        # a neighbour shared by two features is ambiguous; shared by the same feature twice cannot happen (distinct subs)
        idx[cnt > 1] = -2
        self.h1, self.h1_idx = u, idx

    def assign(self, mat: np.ndarray):
        """Feature index per row of a (n, L) byte matrix: >= 0 matched, -1 unmatched, -2 ambiguous; and a 'corrected' flag."""
        codes, bad = S.encode(mat)
        out = np.full(len(codes), -1, dtype=np.int64)
        corr = np.zeros(len(codes), dtype=bool)
        ok = bad == 0
        e = S.lookup(self.exact, codes[ok])
        sub = np.where(e >= 0, self.exact_idx[np.maximum(e, 0)], -1)
        h = S.lookup(self.h1, codes[ok])
        hv = np.where(h >= 0, self.h1_idx[np.maximum(h, 0)], -1)
        res = np.where(sub >= 0, sub, hv)
        out[ok] = res
        corr[np.flatnonzero(ok)[(sub < 0) & (hv >= 0)]] = True
        rows, fills = S.n_fill(mat)
        if len(rows):
            e = S.lookup(self.exact, fills.ravel()).reshape(fills.shape)
            hitn = (e >= 0).sum(axis=1)
            one = hitn == 1
            pick = np.where(
                one,
                self.exact_idx[np.maximum(e, 0)][np.arange(len(rows)), (e >= 0).argmax(axis=1)],
                -1,
            )
            pick[hitn > 1] = -2
            out[rows] = pick
            corr[rows[one]] = True
        return out, corr


class FeatureIndex:
    """Panel index over one or several barcode lengths (e.g. 15-nt TotalSeq barcodes with a 14-nt control).

    The read window is as long as the longest barcode; a barcode of length L is matched on the first L bases. Exact
    matches win (longest length first), then Hamming-1; a window claimed by two features is ambiguous."""

    def __init__(self, sequences: Sequence[str]):
        self.sequences = list(sequences)
        if len(set(self.sequences)) != len(self.sequences):
            raise ValueError("duplicate panel sequences")
        lens = sorted({len(x) for x in self.sequences}, reverse=True)
        self.L = lens[0]
        self.groups = []
        for L in lens:
            idx = [i for i, x in enumerate(self.sequences) if len(x) == L]
            self.groups.append(
                (L, np.array(idx, dtype=np.int64), _FixedIndex([self.sequences[i] for i in idx]))
            )

    def assign(self, mat: np.ndarray):
        n = len(mat)
        if len(self.groups) == 1:
            return self.groups[0][2].assign(mat[:, : self.L])
        exact = np.full(n, -1, np.int64)
        near = np.full(n, -1, np.int64)
        for L, idx, fx in self.groups:
            a, corr = fx.assign(np.ascontiguousarray(mat[:, :L]))
            g = np.where(a >= 0, idx[np.maximum(a, 0)], a)
            ex = (a >= 0) & ~corr
            take = ex & (exact == -1)
            exact[take] = g[take]
            nr = ((a >= 0) & corr) | (a == -2)
            both = nr & (near != -1)
            near[nr & (near == -1)] = g[nr & (near == -1)]
            near[both] = -2
        out = np.where(exact >= 0, exact, near)
        return out, (exact < 0) & (near >= 0)


# ----------------------------------------------------------------------------------------------------------- barcodes
def load_barcode_list(path: str) -> List[str]:
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        return [ln.split()[0] for ln in f if ln.strip()]


def load_barcode_codes(path: str) -> np.ndarray:
    """Sorted 2-bit codes of a 16-nt barcode list, cached next to the list as ``<path>.codes.npy``."""
    cache = path + ".codes.npy"
    try:
        if os.path.getmtime(cache) >= os.path.getmtime(path):
            return np.load(cache)
    except OSError:
        pass
    codes = np.sort(S.encode_strings(load_barcode_list(path), 16)[0])
    _save_atomic(cache, codes)
    return codes


def _save_atomic(path, arr):
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except OSError:  # read-only cache dir: recompute next time
        pass


def load_translation(path: str, reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Cell Ranger translation table (two barcode columns) -> (sorted feature-side codes, matching GEX codes).

    Column 1 is taken as the GEX barcode and column 2 as the feature-side barcode; ``reverse`` swaps them (the
    orientation is checked on the data by :func:`best_translation`)."""
    cache = f"{path}.{'rev' if reverse else 'fwd'}.npy"
    try:
        if os.path.getmtime(cache) >= os.path.getmtime(path):
            t = np.load(cache)
            return t[0], t[1]
    except OSError:
        pass
    gex, fb = [], []
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 2 and not p[0].startswith("#"):
                gex.append(p[0])
                fb.append(p[1])
    if reverse:
        gex, fb = fb, gex
    g, _ = S.encode_strings(gex, 16)
    b, _ = S.encode_strings(fb, 16)
    o = np.argsort(b)
    _save_atomic(cache, np.vstack([b[o], g[o]]))
    return b[o], g[o]


def correct_barcodes(mat: np.ndarray, reads: np.ndarray, whitelist: np.ndarray):
    """Correct unique cell barcodes against a sorted whitelist of codes.

    ``mat``: (n, L) bytes of the unique observed barcodes; ``reads``: reads per observed barcode.
    Returns (corrected code per row, valid bool, n corrected)."""
    L = mat.shape[1]
    codes, bad = S.encode(mat)
    exact = (bad == 0) & (S.lookup(whitelist, codes) >= 0)
    # prior: reads per whitelist barcode among exact matches
    wl_hit = S.lookup(whitelist, codes[exact])
    prior = np.zeros(len(whitelist), dtype=np.float64)
    np.add.at(prior, wl_hit, reads[exact])
    out = np.where(exact, codes, 0).astype(np.uint64)
    valid = exact.copy()
    # one mismatch (no N)
    cand = np.flatnonzero((bad == 0) & ~exact)
    if len(cand):
        nb = S.neighbours(codes[cand], L)
        idx = S.lookup(whitelist, nb.ravel()).reshape(nb.shape)
        w = np.where(idx >= 0, prior[np.maximum(idx, 0)] + PSEUDOCOUNT, 0.0)
        tot = w.sum(axis=1)
        best = w.argmax(axis=1)
        pb = np.where(tot > 0, w[np.arange(len(cand)), best] / np.where(tot > 0, tot, 1), 0)
        ok = pb >= POSTERIOR_MIN
        out[cand[ok]] = nb[np.arange(len(cand)), best][ok]
        valid[cand[ok]] = True
    # one N
    rows, fills = S.n_fill(mat)
    if len(rows):
        idx = S.lookup(whitelist, fills.ravel()).reshape(fills.shape)
        w = np.where(idx >= 0, prior[np.maximum(idx, 0)] + PSEUDOCOUNT, 0.0)
        tot = w.sum(axis=1)
        best = w.argmax(axis=1)
        pb = np.where(tot > 0, w[np.arange(len(rows)), best] / np.where(tot > 0, tot, 1), 0)
        ok = pb >= POSTERIOR_MIN
        out[rows[ok]] = fills[np.arange(len(rows)), best][ok]
        valid[rows[ok]] = True
    return out, valid, int(valid.sum() - exact.sum())


def translate(codes: np.ndarray, table: Tuple[np.ndarray, np.ndarray]):
    fb, gex = table
    i = S.lookup(fb, codes)
    return np.where(i >= 0, gex[np.maximum(i, 0)], codes), i >= 0


# ----------------------------------------------------------------------------------------------------------- UMIs
def correct_umis(group: np.ndarray, umi: np.ndarray, reads: np.ndarray, umi_len: int) -> np.ndarray:
    """Cell Ranger UMI correction within groups. Inputs are per unique (group, umi) (no duplicates).

    Returns the index of the record each record merges into (itself when not corrected)."""
    n = len(group)
    if n == 0:
        return np.zeros(0, np.int64)
    group = np.asarray(group, dtype=np.uint64)
    umi = np.asarray(umi, dtype=np.uint64)
    key = (group << np.uint64(2 * umi_len)) | umi
    o = np.argsort(key, kind="stable")
    ks = key[o]
    best_reads = reads.astype(np.int64).copy()
    best_key = key.copy()
    target = np.arange(n, dtype=np.int64)
    nb = S.neighbours(umi, umi_len)
    gshift = group << np.uint64(2 * umi_len)
    for j in range(nb.shape[1]):
        q = gshift | nb[:, j]
        i = S.lookup(ks, q)
        hit = i >= 0
        t = np.where(hit, o[np.maximum(i, 0)], -1)
        r = np.where(hit, reads[np.maximum(t, 0)], -1)
        # Cell Ranger (tx_annotation mark_dups.rs correct_umis): move to the neighbour with more reads; on equal reads
        # to the lexicographically larger UMI (codes sort like the strings). One step, no chains.
        tk = np.where(hit, key[np.maximum(t, 0)], np.uint64(0))
        better = hit & ((r > best_reads) | ((r == best_reads) & (tk > best_key)))
        target[better] = t[better]
        best_reads[better] = r[better]
        best_key[better] = tk[better]
    return target


# ----------------------------------------------------------------------------------------------------------- library
@dataclass
class LibraryCounts:
    features: List[str]
    matrix: "object"  # scipy.sparse.csc_matrix, features x GEX columns, uint32
    ambient: np.ndarray  # UMIs per feature on whitelist barcodes outside the GEX columns
    outside_totals: (
        np.ndarray
    )  # UMI totals of each whitelist barcode outside the GEX columns (descending)
    stats: dict
    unmatched: List[
        Tuple[str, int]
    ]  # most frequent unmatched feature sequences (in valid barcodes) and reads


def _barcode_map(ucb: np.ndarray, cb_reads: np.ndarray, cb_len: int, whitelist_codes, translation):
    """Corrected code, validity and translation flag per unique raw barcode (``ucb`` sorted ``S<cb_len>``)."""
    ucb_mat = ucb.view(np.uint8).reshape(len(ucb), cb_len)
    cb_code, cb_valid, n_corr = correct_barcodes(ucb_mat, cb_reads, whitelist_codes)
    translated = np.zeros(len(ucb), bool)
    if translation is not None:
        cb_code_t, translated = translate(cb_code, translation)
        cb_code = np.where(cb_valid, cb_code_t, cb_code)
    return cb_code, cb_valid, translated, n_corr


def _add(st, k, v):
    st[k] = st.get(k, 0) + int(v)


def _process(keys, reads, bmap, fx, cb_len, umi_len, n_features, st, unm):
    """Features, barcodes and UMIs of a set of collapsed keys whose barcodes are all in ``bmap``.

    Returns the surviving molecules (corrected barcode code, feature, reads); adds read statistics to ``st`` and
    unmatched feature sequences on valid barcodes to ``unm`` (sequence -> reads)."""
    ucb, cb_code, cb_valid, translated = bmap
    L = fx.L
    r1_len = cb_len + umi_len
    mat = keys.view(np.uint8).reshape(len(keys), r1_len + L)
    fidx, fcorr = fx.assign(mat[:, r1_len:])
    _add(st, "reads_feature_exact", reads[(fidx >= 0) & ~fcorr].sum())
    _add(st, "reads_feature_corrected", reads[(fidx >= 0) & fcorr].sum())
    _add(st, "reads_feature_ambiguous", reads[fidx == -2].sum())
    _add(st, "reads_feature_unmatched", reads[fidx == -1].sum())
    cbk = np.ascontiguousarray(mat[:, :cb_len]).view(f"S{cb_len}").ravel()
    bi = np.searchsorted(ucb, cbk)
    rv = cb_valid[bi]
    _add(st, "reads_valid_barcode", reads[rv].sum())
    _add(st, "reads_valid_barcode_feature", reads[rv & (fidx >= 0)].sum())
    _add(st, "reads_translated", reads[rv & translated[bi]].sum())
    um = rv & (fidx == -1)
    if um.any():
        fs = np.ascontiguousarray(mat[um, r1_len:]).view(f"S{L}").ravel()
        u, inv = np.unique(fs, return_inverse=True)
        c = np.bincount(inv.ravel(), weights=reads[um]).astype(np.int64)
        for i in np.argsort(-c)[: N_UNMATCHED_KEEP * 5]:
            unm[u[i]] = unm.get(u[i], 0) + int(c[i])
    keep = rv & (fidx >= 0)
    umi_codes, umi_bad = S.encode(mat[keep, cb_len:r1_len])
    bc = cb_code[bi[keep]]
    ft = fidx[keep]
    rd = reads[keep]
    ok = umi_bad == 0
    _add(st, "reads_umi_n", rd[~ok].sum())
    bc, ft, rd, umi_codes = bc[ok], ft[ok], rd[ok], umi_codes[ok]
    # several observed (CB, UMI, feature) can land on one corrected key: sum them. Packed key:
    # dense barcode index | feature (>= 10 bits; CRISPR libraries have thousands of guides) | UMI (2 bits per nt)
    ubits = 2 * umi_len
    fbits = max(10, int(n_features).bit_length())
    ubc, bci = np.unique(bc, return_inverse=True)
    if ubits + fbits + max(1, int(len(ubc)).bit_length()) > 64:
        raise ValueError("barcodes x features x UMIs do not fit the packed molecule key")
    fshift = np.uint64(ubits)
    bshift = np.uint64(ubits + fbits)
    key = (bci.ravel().astype(np.uint64) << bshift) | (ft.astype(np.uint64) << fshift) | umi_codes
    gu, ginv = np.unique(key, return_inverse=True)
    greads = np.bincount(ginv.ravel(), weights=rd, minlength=len(gu)).astype(np.int64)
    umi_mask = np.uint64((1 << (2 * umi_len)) - 1)
    tgt = correct_umis(gu >> fshift, gu & umi_mask, greads, umi_len)
    mol = np.unique(tgt)  # surviving molecules
    mol_reads = np.bincount(tgt, weights=greads, minlength=len(gu))[mol].astype(np.int64)
    _add(st, "umis_before_correction", len(gu))
    _add(st, "molecules", len(mol))
    _add(st, "reads_usable", greads.sum())
    return (
        ubc[(gu[mol] >> bshift).astype(np.int64)],
        ((gu[mol] >> fshift) & np.uint64((1 << fbits) - 1)).astype(np.int64),
        mol_reads,
    )


def count_library(
    r1_paths,
    r2_paths,
    *,
    cb_len: int,
    umi_len: int,
    offset: int,
    panel_sequences: Sequence[str],
    whitelist_codes: np.ndarray,
    gex_barcodes: Sequence[str],
    translation=None,
    max_reads: Optional[int] = None,
    pairs=None,
    shards: int = 0,
    tmpdir: Optional[str] = None,
    pairs_factory=None,
) -> LibraryCounts:
    """Count one library. ``gex_barcodes``: the GEX sample's column barcodes in ``col`` order (16 nt, ``-1`` suffix ok).

    ``shards > 0`` bounds memory for large libraries: the reads are read once, each block's collapsed keys are spilled
    under ``tmpdir`` while reads per raw barcode are counted; every barcode is then corrected (the same prior as the
    single pass), the spilled keys are re-sharded into ``shards`` files by corrected barcode, and each shard is processed
    alone. UMI correction never crosses barcodes, so the result is identical. ``pairs_factory`` (a callable returning a
    fresh pair iterator) may replace ``pairs``."""
    from scipy import sparse

    fx = FeatureIndex(list(panel_sequences))
    L = fx.L
    r1_len = cb_len + umi_len
    nf = len(panel_sequences)

    def blocks():
        if pairs_factory is not None:
            return key_blocks_from_pairs(pairs_factory(), r1_len, offset, L, max_reads)
        if pairs is not None:
            return key_blocks_from_pairs(pairs, r1_len, offset, L, max_reads)
        return key_blocks_from_fastq(r1_paths, r2_paths, r1_len, offset, L, max_reads)

    st: dict = {}
    unm: dict = {}
    if not shards:
        keys, reads, st0 = _collapse_blocks(blocks(), r1_len + L)
        st.update(st0)
        st["unique_keys"] = int(len(keys))
        cbk = np.ascontiguousarray(keys.view(np.uint8).reshape(len(keys), r1_len + L)[:, :cb_len])
        ucb, inv = np.unique(cbk.view(f"S{cb_len}").ravel(), return_inverse=True)
        cb_reads = np.bincount(inv.ravel(), weights=reads, minlength=len(ucb))
        del cbk, inv
        cb_code, cb_valid, translated, n_corr = _barcode_map(
            ucb, cb_reads, cb_len, whitelist_codes, translation
        )
        parts = [
            _process(
                keys, reads, (ucb, cb_code, cb_valid, translated), fx, cb_len, umi_len, nf, st, unm
            )
        ]
    else:
        import tempfile

        # one pass over the reads: each block's collapsed keys go to disk (spill) while reads per raw barcode are
        # counted; then barcodes are corrected and the spilled keys are re-sharded by corrected barcode
        d = tempfile.mkdtemp(prefix="adt-shards-", dir=tmpdir)
        spill = []
        bu: List[np.ndarray] = []
        bc_: List[np.ndarray] = []
        n = n_short = 0
        for mat, nk, short in blocks():
            n += nk
            n_short += short
            if not len(mat):
                continue
            u, c = np.unique(
                np.ascontiguousarray(mat).view(f"S{r1_len + L}").ravel(), return_counts=True
            )
            f = os.path.join(d, f"spill_{len(spill)}.npy")
            np.save(f, u)
            np.save(f + ".c.npy", c.astype(np.int64))
            spill.append(f)
            cbk = np.ascontiguousarray(u.view(np.uint8).reshape(len(u), r1_len + L)[:, :cb_len])
            ub, ib = np.unique(cbk.view(f"S{cb_len}").ravel(), return_inverse=True)
            bu.append(ub)
            bc_.append(np.bincount(ib.ravel(), weights=c, minlength=len(ub)).astype(np.int64))
            if len(bu) > 8:
                _merge(bu, bc_)
        _merge(bu, bc_)
        ucb = bu[0] if bu else np.zeros(0, f"S{cb_len}")
        cb_reads = bc_[0] if bc_ else np.zeros(0, np.int64)
        cb_code, cb_valid, translated, n_corr = _barcode_map(
            ucb, cb_reads, cb_len, whitelist_codes, translation
        )
        bmap = (ucb, cb_code, cb_valid, translated)
        # shard by corrected barcode; invalid barcodes get their own shard (they still count in the read statistics)
        shard_of = np.where(cb_valid, (cb_code % np.uint64(shards)).astype(np.int64), shards)
        pieces = {k: [] for k in range(shards + 1)}
        for f in spill:
            u, c = np.load(f), np.load(f + ".c.npy")
            os.remove(f)
            os.remove(f + ".c.npy")
            cbk = np.ascontiguousarray(u.view(np.uint8).reshape(len(u), r1_len + L)[:, :cb_len])
            sh = shard_of[np.searchsorted(ucb, cbk.view(f"S{cb_len}").ravel())]
            for k in np.unique(sh):
                sel = sh == k
                f = os.path.join(d, f"{k}_{len(pieces[k])}.npz")
                np.savez(f, keys=u[sel], reads=c[sel].astype(np.int64))
                pieces[k].append(f)
        st.update({"reads": n, "reads_short": n_short, "shards": shards})
        parts = []
        uk = 0
        for k in range(shards + 1):
            if not pieces[k]:
                continue
            us, cs = [], []
            for f in pieces[k]:
                with np.load(f) as z:
                    us.append(z["keys"])
                    cs.append(z["reads"])
                os.remove(f)
            _merge(us, cs)
            uk += len(us[0])
            parts.append(_process(us[0], cs[0], bmap, fx, cb_len, umi_len, nf, st, unm))
        os.rmdir(d)
        st["unique_keys"] = uk
    st["barcodes_observed"] = int(len(ucb))
    st["barcodes_corrected"] = n_corr
    if translation is None:
        st.pop("reads_translated", None)
    m_bc = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.uint64)
    m_ft = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int64)
    mol_reads = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.int64)
    unmatched = sorted(((q.decode(), c) for q, c in unm.items()), key=lambda x: -x[1])[
        :N_UNMATCHED_KEEP
    ]

    # GEX columns
    gb = [b.split("-")[0] for b in gex_barcodes]
    gcode, gbad = S.encode_strings(gb, cb_len)
    if gbad.any():
        raise ValueError("GEX barcodes with non-ACGT characters")
    go = np.argsort(gcode)
    col = S.lookup(gcode[go], m_bc)
    col = np.where(col >= 0, go[np.maximum(col, 0)], -1)
    inside = col >= 0
    n_cols = len(gb)
    M = sparse.coo_matrix(
        (np.ones(inside.sum(), np.uint32), (m_ft[inside], col[inside])), shape=(nf, n_cols)
    ).tocsc()
    M.sum_duplicates()
    M.data = M.data.astype(np.uint32)
    st["molecules_in_gex_columns"] = int(inside.sum())
    st["reads_in_gex_columns"] = int(mol_reads[inside].sum())
    st["saturation"] = float(1 - st.get("molecules", 0) / max(1, st.get("reads_usable", 0)))
    st["saturation_gex_columns"] = float(1 - inside.sum() / max(1, mol_reads[inside].sum()))
    ambient = np.bincount(m_ft[~inside], minlength=nf).astype(np.int64)
    ob, oc = np.unique(m_bc[~inside], return_counts=True)
    outside_totals = np.sort(oc)[::-1].astype(np.int64)
    st["outside_barcodes"] = int(len(ob))
    return LibraryCounts(list(panel_sequences), M, ambient, outside_totals, st, unmatched)


def choose_translation(sample_cb: np.ndarray, gex_codes_sorted: np.ndarray, table) -> dict:
    """Share of sampled read barcodes that are GEX columns as they are and after translation (``table`` from
    :func:`load_translation`). ``use`` when translation at least doubles the overlap and reaches 0.2."""
    direct = float((S.lookup(gex_codes_sorted, sample_cb) >= 0).mean()) if len(sample_cb) else 0.0
    t, hit = translate(sample_cb, table)
    trans = float((S.lookup(gex_codes_sorted, t) >= 0).mean()) if len(sample_cb) else 0.0
    return {"direct": direct, "translated": trans, "use": trans > max(2 * direct, 0.2)}


def best_translation(sample_cb, gex_codes_sorted, path):
    """Both orientations of a translation file; -> (table or None, report)."""
    rep = {}
    best = None
    for rev in (False, True):
        tab = load_translation(path, reverse=rev)
        ch = choose_translation(sample_cb, gex_codes_sorted, tab)
        rep["reverse" if rev else "forward"] = ch
        if ch["use"] and (best is None or ch["translated"] > best[1]["translated"]):
            best = (tab, ch)
    rep["use"] = best is not None
    rep["direct"] = rep["forward"]["direct"]
    rep["translated"] = max(rep["forward"]["translated"], rep["reverse"]["translated"])
    return (best[0] if best else None), rep


def save_npz(path: str, lc: LibraryCounts, extra: Optional[dict] = None):
    M = lc.matrix
    meta = dict(stats=lc.stats, unmatched=lc.unmatched, **(extra or {}))
    np.savez_compressed(
        path,
        indptr=M.indptr,
        indices=M.indices,
        data=M.data,
        shape=np.array(M.shape),
        features=np.array(lc.features),
        ambient=lc.ambient,
        outside_totals=lc.outside_totals,
        meta=np.array(json.dumps(meta)),
    )


def load_npz(path: str):
    from scipy import sparse

    z = np.load(path, allow_pickle=False)
    M = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    meta = json.loads(str(z["meta"]))
    return M, list(z["features"]), z["ambient"], z["outside_totals"], meta
