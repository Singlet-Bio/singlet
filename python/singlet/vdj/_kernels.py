# SPDX-License-Identifier: MIT
"""numba kernels of the V(D)J assembler (``singlet.vdj.assemble``).

Sequences are uint8 codes A=0 C=1 G=2 T=3, anything else 4, concatenated with int64 offsets. k-mers are 2-bit packed
into uint64 (k <= 31) and sort like the strings.

* :func:`classify`: per read, the best V, J and C segment and its diagonal (read position - reference position), from
  exact k-mer hits on the segment index, so the positions of Cys104 and Phe/Trp118 in the read follow from the
  segment anchors.
* :func:`assemble_group`: the reads of one (cell, chain) -> junction contigs (Cys104 ... Phe/Trp118) by greedy
  extension through the k-mer graph of the reads, weighted by distinct UMIs, from seed k-mers at Cys104 to terminal
  k-mers ending at Phe/Trp118; branches with enough UMI support are explored too (second alleles, dual alpha
  chains, doublets). Then every read goes to the contig sharing most of its k-mers.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    LUT[_b] = _i
    LUT[_b + 32] = _i

DIAG_BIAS = 1 << 19
MAX_HITS_PER_KMER = 16


@njit(cache=True)
def _kmer_entries(codes, off, k):
    """(kmer, segment, position) of every valid k-mer of every segment."""
    n = 0
    for s in range(len(off) - 1):
        n += max(0, off[s + 1] - off[s] - k + 1)
    km = np.empty(n, np.uint64)
    sg = np.empty(n, np.int32)
    ps = np.empty(n, np.int32)
    mask = np.uint64((1 << (2 * k)) - 1)
    m = 0
    for s in range(len(off) - 1):
        cur = np.uint64(0)
        valid = 0
        for i in range(off[s], off[s + 1]):
            c = codes[i]
            if c > 3:
                valid = 0
                cur = np.uint64(0)
                continue
            cur = ((cur << np.uint64(2)) | np.uint64(c)) & mask
            valid += 1
            if valid >= k:
                km[m] = cur
                sg[m] = s
                ps[m] = i - off[s] - k + 1
                m += 1
    return km[:m], sg[:m], ps[:m]


def build_index(seqs, k):
    """Sorted k-mer index of segment sequences (list of str): (kmers, segment, position)."""
    buf = np.frombuffer("".join(seqs).encode(), dtype=np.uint8)
    codes = LUT[buf]
    off = np.zeros(len(seqs) + 1, np.int64)
    off[1:] = np.cumsum([len(s) for s in seqs])
    km, sg, ps = _kmer_entries(codes, off, k)
    o = np.lexsort((ps, sg, km))
    return km[o], sg[o], ps[o]


@njit(cache=True, parallel=True)
def classify(codes, off, k, ikm, isg, ips, seg_region, min_hits, seg_class):
    """Best V / J / C segment per read. Region codes: 0 V, 1 D, 2 J, 3 C.

    Returns int32 arrays (n, 10): vseg, vhits, vdiag, jseg, jhits, jdiag, cseg, chits, cdiag (-1 / 0 when absent), and
    in column 9 tie flags: 1 C, 2 V, 4 J when the best segment of that region ties with another gene (TRBC1 / TRBC2,
    IGHG1-4 share most k-mers), so that the read does not vote for either. Genes of one equivalence class
    (``seg_class``: identical up to Cys104, e.g. TRBV6-2 / TRBV6-3) never tie: they are one call."""
    n = len(off) - 1
    out = np.full((n, 10), -1, np.int32)
    mask = np.uint64((1 << (2 * k)) - 1)
    for r in prange(n):
        a = off[r]
        L = off[r + 1] - a
        for t in (1, 4, 7, 9):
            out[r, t] = 0
        if L < k:
            continue
        keys = np.empty(L * MAX_HITS_PER_KMER, np.int64)
        m = 0
        cur = np.uint64(0)
        valid = 0
        for i in range(L):
            c = codes[a + i]
            if c > 3:
                valid = 0
                cur = np.uint64(0)
                continue
            cur = ((cur << np.uint64(2)) | np.uint64(c)) & mask
            valid += 1
            if valid < k:
                continue
            q = i - k + 1
            lo = np.searchsorted(ikm, cur)
            if lo >= len(ikm) or ikm[lo] != cur:
                continue
            hi = lo
            while hi < len(ikm) and ikm[hi] == cur and hi - lo < MAX_HITS_PER_KMER:
                keys[m] = (np.int64(isg[hi]) << 20) | np.int64(q - ips[hi] + DIAG_BIAS)
                m += 1
                hi += 1
        if m == 0:
            continue
        ks = np.sort(keys[:m])
        best = np.zeros(4, np.int64)
        second = np.zeros(4, np.int64)
        bseg = np.full(4, -1, np.int64)
        bdiag = np.zeros(4, np.int64)
        j = 0
        while j < m:
            e = j
            while e < m and ks[e] == ks[j]:
                e += 1
            cnt = e - j
            seg = ks[j] >> 20
            reg = seg_region[seg]
            if cnt > best[reg]:
                if bseg[reg] >= 0 and seg_class[bseg[reg]] != seg_class[seg]:
                    second[reg] = best[reg]
                best[reg] = cnt
                bseg[reg] = seg
                bdiag[reg] = (ks[j] & ((1 << 20) - 1)) - DIAG_BIAS
            elif seg_class[seg] != seg_class[bseg[reg]] and cnt > second[reg]:
                second[reg] = cnt
            j = e
        for reg, col in ((0, 0), (2, 3), (3, 6)):
            if best[reg] >= min_hits:
                out[r, col] = bseg[reg]
                out[r, col + 1] = best[reg]
                out[r, col + 2] = bdiag[reg]
        flags = 0
        if best[3] >= min_hits and second[3] == best[3]:
            flags |= 1
        if best[0] >= min_hits and second[0] == best[0]:
            flags |= 2
        if best[2] >= min_hits and second[2] == best[2]:
            flags |= 4
        out[r, 9] = flags
    return out


@njit(cache=True)
def _find(sorted_arr, x):
    i = np.searchsorted(sorted_arr, x)
    if i < len(sorted_arr) and sorted_arr[i] == x:
        return i
    return -1


@njit(cache=True)
def _codon_at(codes, a, L, st, which):
    """Codon check around an anchor, among st, st-1, st+1 (an indel or a V allele can move the diagonal by one).

    ``which`` 0: Cys104 (TGT/TGC). ``which`` 1: Phe/Trp118 followed by the Gly of the J motif F/W-G-X-G
    (TTY/TGG + GG): without the Gly a poly-T run (``...CTGTTTTTTGG``) accepts the codon one base early. Returns the
    codon start or -1."""
    for d in (0, -1, 1):
        q = st + d
        if q < 0 or q + 3 > L:
            continue
        c0, c1, c2 = codes[a + q], codes[a + q + 1], codes[a + q + 2]
        if which == 0:
            if c0 == 3 and c1 == 2 and (c2 == 3 or c2 == 1):
                return q
        else:
            if q + 5 > L:
                continue
            if (c0 == 3 and ((c1 == 3 and (c2 == 3 or c2 == 1)) or (c1 == 2 and c2 == 2))
                    and codes[a + q + 3] == 2 and codes[a + q + 4] == 2):
                return q
    return -1


@njit(cache=True)
def _vote(ek_lo, ek_hi, eu, ec, js, stamp, bestc, bestb, tie, touched, step):
    """Per-UMI consensus over candidate k-mers ``js`` (indices into the unique k-mer table, -1 = absent): each UMI
    votes for the candidate its reads hold most often (ties do not vote). Returns votes per candidate."""
    nt = 0
    for b in range(len(js)):
        j = js[b]
        if j < 0:
            continue
        for e in range(ek_lo[j], ek_hi[j]):
            u = eu[e]
            c = ec[e]
            if stamp[u] != step:
                stamp[u] = step
                bestc[u] = c
                bestb[u] = b
                tie[u] = False
                touched[nt] = u
                nt += 1
            elif c > bestc[u]:
                bestc[u] = c
                bestb[u] = b
                tie[u] = False
            elif c == bestc[u] and bestb[u] != b:
                tie[u] = True
    votes = np.zeros(len(js), np.int32)
    for i in range(nt):
        u = touched[i]
        if not tie[u]:
            votes[bestb[u]] += 1
    return votes


@njit(cache=True, nogil=True)
def assemble_group(codes, off, ridx, umi, cys, fw, ka, min_seed_umis, seed_frac, min_branch_umis, branch_frac,
                   max_paths, max_len):
    """Junction contigs of one (cell, chain).

    ``ridx``: read indices into ``off``; ``umi``: local UMI id per read (< 2**24); ``cys`` / ``fw``: read position of the
    Cys104 / Phe-Trp118 codon start or -1.

    Support is a per-UMI consensus: at a seed or a branch every UMI votes for the alternative its reads hold most often.
    Counting UMIs that merely contain a k-mer would not do: at the depth of a V(D)J library (~100 reads per UMI) almost
    every UMI holds some read with any given one-base error, so error branches would look as supported as the truth.

    Returns (paths (p, max_len) uint8, lengths (p,), seed UMI support (p,), cand per read (int32: -1 none, -2 several))."""
    mask = np.uint64((1 << (2 * ka)) - 1)
    nr = len(ridx)
    n_umi = 1
    for t in range(nr):
        if umi[t] + 1 > n_umi:
            n_umi = umi[t] + 1
    tot = 0
    for t in range(nr):
        r = ridx[t]
        tot += max(0, off[r + 1] - off[r] - ka + 1)
    keys = np.empty(max(tot, 1), np.uint64)
    m = 0
    for t in range(nr):
        r = ridx[t]
        a = off[r]
        cur = np.uint64(0)
        valid = 0
        for i in range(a, off[r + 1]):
            c = codes[i]
            if c > 3:
                valid = 0
                cur = np.uint64(0)
                continue
            cur = ((cur << np.uint64(2)) | np.uint64(c)) & mask
            valid += 1
            if valid >= ka:
                keys[m] = (cur << np.uint64(24)) | np.uint64(umi[t])
                m += 1
    ks = np.sort(keys[:m])
    # (k-mer, UMI) entries with read counts, grouped by k-mer
    eu = np.empty(m, np.int32)
    ec = np.empty(m, np.int32)
    uk = np.empty(m, np.uint64)
    ek_lo = np.empty(m, np.int64)
    ne = 0
    nu = 0
    prev = np.uint64(0xFFFFFFFFFFFFFFFF)
    prevk = np.uint64(0xFFFFFFFFFFFFFFFF)
    for i in range(m):
        if ks[i] == prev:
            ec[ne - 1] += 1
            continue
        prev = ks[i]
        km = ks[i] >> np.uint64(24)
        if km != prevk:
            uk[nu] = km
            ek_lo[nu] = ne
            nu += 1
            prevk = km
        eu[ne] = np.int32(ks[i] & np.uint64(0xFFFFFF))
        ec[ne] = 1
        ne += 1
    uk = uk[:nu]
    ek_lo = ek_lo[:nu]
    ek_hi = np.empty(nu, np.int64)
    for j in range(nu):
        ek_hi[j] = ek_lo[j + 1] if j + 1 < nu else ne
    stamp = np.full(n_umi, -1, np.int64)
    bestc = np.zeros(n_umi, np.int32)
    bestb = np.zeros(n_umi, np.int32)
    tie = np.zeros(n_umi, np.bool_)
    touched = np.empty(n_umi, np.int32)
    step = 0

    # seeds (k-mer starting at Cys104) and terminals (k-mer ending at the end of the Phe/Trp118 codon)
    seeds = np.empty(nr, np.uint64)
    ns = 0
    term = np.empty(nr, np.uint64)
    term_u = np.empty(nr, np.int32)
    nt = 0
    for t in range(nr):
        r = ridx[t]
        a = off[r]
        L = off[r + 1] - a
        for which in range(2):
            if (which == 0 and cys[t] < 0) or (which == 1 and fw[t] < 0):
                continue
            q = _codon_at(codes, a, L, cys[t] if which == 0 else fw[t], which)
            if q < 0:
                continue
            st = q if which == 0 else q + 3 - ka
            if st < 0 or st + ka > L:
                continue
            cur = np.uint64(0)
            ok = True
            for i in range(a + st, a + st + ka):
                c = codes[i]
                if c > 3:
                    ok = False
                    break
                cur = (cur << np.uint64(2)) | np.uint64(c)
            if not ok:
                continue
            if which == 0:
                seeds[ns] = cur
                ns += 1
            else:
                term[nt] = cur
                term_u[nt] = umi[t]
                nt += 1
    seeds = np.unique(seeds[:ns])
    # terminals need UMI support: one read whose error puts TTYGG a base early would otherwise end the true junction
    # there (that k-mer is part of the true path)
    # there (that k-mer is part of the true path). Each UMI votes for the terminal most of its reads give.
    tk_ = np.empty(nt, np.uint64)
    for i in range(nt):
        tk_[i] = (np.uint64(term_u[i]) << np.uint64(40)) | term[i]  # (UMI, k-mer): ka <= 20 fits 40 bits
    tk_ = np.sort(tk_)
    tu = np.unique(term[:nt])
    tc = np.zeros(len(tu), np.int32)
    ntu = len(tu)
    i = 0
    while i < nt:
        u = tk_[i] >> np.uint64(40)
        best_k = np.uint64(0)
        best_n = 0
        tie_ = False
        e = i
        while e < nt and (tk_[e] >> np.uint64(40)) == u:
            f = e
            while f < nt and tk_[f] == tk_[e]:
                f += 1
            if f - e > best_n:
                best_n = f - e
                best_k = tk_[e] & np.uint64((1 << 40) - 1)
                tie_ = False
            elif f - e == best_n:
                tie_ = True
            e = f
        if not tie_:
            tc[_find(tu, best_k)] += 1
        i = e
    tmax = 0
    for i in range(ntu):
        tmax = max(tmax, tc[i])
    tthr = max(2 if tmax >= 4 else 1, int(np.ceil(0.05 * tmax)))
    term = np.empty(ntu, np.uint64)
    nt = 0
    for i in range(ntu):
        if tc[i] >= tthr:
            term[nt] = tu[i]
            nt += 1
    term = term[:nt]
    js = np.full(len(seeds), -1, np.int64)
    for i in range(len(seeds)):
        js[i] = _find(uk, seeds[i])
    ssup = _vote(ek_lo, ek_hi, eu, ec, js, stamp, bestc, bestb, tie, touched, step)
    step += 1
    order = np.argsort(-ssup)
    seed_min = min_seed_umis
    if len(ssup):
        seed_min = max(min_seed_umis, int(np.ceil(seed_frac * ssup.max())))

    paths = np.zeros((max_paths, max_len), np.uint8)
    plen = np.zeros(max_paths, np.int32)
    psup = np.zeros(max_paths, np.int32)
    np_ = 0
    # DFS stacks
    MS = 16
    stk = np.zeros((MS, max_len), np.uint8)
    stl = np.zeros(MS, np.int32)
    stk_k = np.zeros(MS, np.uint64)
    cur_path = np.zeros(max_len, np.uint8)
    j4 = np.full(4, -1, np.int64)
    if len(term) > 0:
        for oi in range(len(order)):
            si = order[oi]
            if ssup[si] < seed_min or np_ >= max_paths:
                break
            # seed onto the stack
            sp = 0
            for i in range(ka):
                stk[0, i] = (seeds[si] >> np.uint64(2 * (ka - 1 - i))) & np.uint64(3)
            stl[0] = ka
            stk_k[0] = seeds[si]
            sp = 1
            steps = 0
            while sp > 0 and np_ < max_paths and steps < 4 * max_len * MS:
                sp -= 1
                pl = stl[sp]
                for i in range(pl):
                    cur_path[i] = stk[sp, i]
                cur = stk_k[sp]
                while True:
                    steps += 1
                    if _find(term, cur) >= 0:
                        dup = False
                        for q in range(np_):
                            if plen[q] == pl:
                                same = True
                                for i in range(pl):
                                    if paths[q, i] != cur_path[i]:
                                        same = False
                                        break
                                if same:
                                    dup = True
                                    break
                        if not dup:
                            for i in range(pl):
                                paths[np_, i] = cur_path[i]
                            plen[np_] = pl
                            psup[np_] = ssup[si]
                            np_ += 1
                        break
                    if pl >= max_len:
                        break
                    for b in range(4):
                        j4[b] = _find(uk, ((cur << np.uint64(2)) | np.uint64(b)) & mask)
                    bs = _vote(ek_lo, ek_hi, eu, ec, j4, stamp, bestc, bestb, tie, touched, step)
                    step += 1
                    bb = np.argmax(bs)
                    if bs[bb] == 0:
                        # every UMI ties (or none reaches here): follow the k-mer held by most reads
                        best_r = 0
                        bb = -1
                        for b in range(4):
                            if j4[b] >= 0:
                                rr = 0
                                for e in range(ek_lo[j4[b]], ek_hi[j4[b]]):
                                    rr += ec[e]
                                if rr > best_r:
                                    best_r = rr
                                    bb = b
                        if bb < 0:
                            break
                    else:
                        thr = max(min_branch_umis, branch_frac * bs[bb])
                        for b in range(4):
                            if b != bb and bs[b] >= thr and sp < MS:
                                for i in range(pl):
                                    stk[sp, i] = cur_path[i]
                                stk[sp, pl] = b
                                stl[sp] = pl + 1
                                stk_k[sp] = ((cur << np.uint64(2)) | np.uint64(b)) & mask
                                sp += 1
                    cur_path[pl] = bb
                    pl += 1
                    cur = ((cur << np.uint64(2)) | np.uint64(bb)) & mask

    # read -> the contig sharing most of its k-mers (strictly more than any other; -2 on a tie, -1 when none), so an
    # error variant takes only the reads carrying its error
    cand = np.full(nr, -1, np.int32)
    if np_ == 0:
        return paths[:0], plen[:0], psup[:0], cand
    tk = 0
    for q in range(np_):
        tk += plen[q] - ka + 1
    ck = np.empty(tk, np.uint64)
    cc = np.empty(tk, np.int32)
    m = 0
    for q in range(np_):
        cur = np.uint64(0)
        for i in range(plen[q]):
            cur = ((cur << np.uint64(2)) | np.uint64(paths[q, i])) & mask
            if i >= ka - 1:
                ck[m] = cur
                cc[m] = q
                m += 1
    o = np.argsort(ck)
    ck = ck[o]
    cc = cc[o]
    score = np.zeros(np_, np.int32)
    for t in range(nr):
        r = ridx[t]
        cur = np.uint64(0)
        valid = 0
        score[:] = 0
        for i in range(off[r], off[r + 1]):
            c = codes[i]
            if c > 3:
                valid = 0
                cur = np.uint64(0)
                continue
            cur = ((cur << np.uint64(2)) | np.uint64(c)) & mask
            valid += 1
            if valid >= ka:
                j = np.searchsorted(ck, cur)
                while j < m and ck[j] == cur:
                    score[cc[j]] += 1
                    j += 1
        b1 = -1
        s1 = 0
        s2 = 0
        for q in range(np_):
            if score[q] > s1:
                s2 = s1
                s1 = score[q]
                b1 = q
            elif score[q] > s2:
                s2 = score[q]
        if s1 > 0:
            cand[t] = b1 if s1 > s2 else -2
    return paths[:np_], plen[:np_], psup[:np_], cand


@njit(cache=True, parallel=True)
def fw_from_c(codes, off, cdiag, todo):
    """Phe/Trp118 found in the read itself for reads with a C hit and no J call (a J gene the reference lacks: GENCODE
    v44 has no TRAJ15): the F/W-G-X-G motif 20-60 nt upstream of the C region's start, in any frame; the one closest to
    33 nt upstream (where J regions put it). Returns the F/W codon start in the read or -1, per entry of ``todo``."""
    out = np.full(len(todo), -1, np.int32)
    for t in prange(len(todo)):
        r = todo[t]
        a = off[r]
        L = off[r + 1] - a
        c0 = cdiag[r]
        best = 1000
        for q in range(max(0, c0 - 60), max(0, c0 - 20 + 1)):
            if q + 12 > L:
                break
            # F/W codon, then G G N, any, G G N
            c1, c2, c3 = codes[a + q], codes[a + q + 1], codes[a + q + 2]
            fw_ok = c1 == 3 and ((c2 == 3 and (c3 == 3 or c3 == 1)) or (c2 == 2 and c3 == 2))
            if not fw_ok:
                continue
            if codes[a + q + 3] == 2 and codes[a + q + 4] == 2 and codes[a + q + 9] == 2 and codes[a + q + 10] == 2:
                d = abs((c0 - q) - 33)
                if d < best:
                    best = d
                    out[t] = q
    return out


@njit(cache=True, parallel=True)
def cys_from_read(codes, off, fw, todo):
    """Cys104 found in the read itself for reads with a Phe/Trp118 and no anchored V call (a V gene the reference
    lacks, GRCh38 has no IGHV1-8; or a V too mutated for exact k-mers): the in-frame TGT/TGC 24-93 nt upstream of the
    F/W codon that follows a Tyr/Phe/His codon (the V region's "YYC" end); the most upstream one. -1 if none."""
    out = np.full(len(todo), -1, np.int32)
    for t in prange(len(todo)):
        r = todo[t]
        a = off[r]
        f = fw[r]
        for jl in range(93, 23, -3):  # junction length (Cys ... F/W inclusive) = jl + 3 - 3
            q = f - jl + 3
            if q < 3:
                continue
            c0, c1, c2 = codes[a + q], codes[a + q + 1], codes[a + q + 2]
            if not (c0 == 3 and c1 == 2 and (c2 == 3 or c2 == 1)):
                continue
            p0, p1, p2 = codes[a + q - 3], codes[a + q - 2], codes[a + q - 1]
            # TAY (Y), TTY (F), CAY (H)
            if (p2 == 3 or p2 == 1) and ((p0 == 3 and p1 == 0) or (p0 == 3 and p1 == 3) or (p0 == 1 and p1 == 0)):
                out[t] = q
                break
    return out
