"""Reference fragment and peak reducers for the GPU align-and-count engine (phase 1e preparation).

Shape: emit tuples -> sort -> collapse -> reduce, order-free, identical output for any cell-partition count K
(docs/research-beta/align-and-count-design.md section 4.2; rules in docs/research-beta/atac-reducers-prep.md).

Two things live here:

* ``cpu_*``  sequential ports of singlet's CPU ATAC path (branch claude/atac-spatial-benchmark-2ba7df, PR #48):
             atac_fragment.h, atac_peak_caller.h, atac_metrics.h. They are the oracle the reducers are tested against.
             They take records in the order given, as the CPU does, so they also expose the order dependence.
* ``fragment_*``, ``insertion_*``, ``peak_*`` the reducers: array/sort/segment code with no sequential state, the shape
             the CUDA drafts in src/engine_gpu/gpu_pileup/pileup/atac_reducers.cuh implement.

Records are a dict of equal-length int arrays (the BAM fields the CPU extractor reads):
  cell   resolved barcode ordinal (whitelist index), -1 if the barcode did not resolve
  tid, pos, tlen, mtid, flag, mapq
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

# ---- BAM flags
F_PAIRED, F_PROPER, F_UNMAP, F_MUNMAP, F_REV, F_READ1, F_READ2, F_SEC, F_SUPP = 0x1, 0x2, 0x4, 0x8, 0x10, 0x40, 0x80, 0x100, 0x800

SPLITMIX_ADD = np.uint64(0x9E3779B97F4A7C15)


# =====================================================================================================================
# cell partitions (the same hash as cell_part() in pileup.cu)
# =====================================================================================================================
def cell_part(cell: np.ndarray, K: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = cell.astype(np.uint32).astype(np.uint64) + SPLITMIX_ADD
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(K)).astype(np.int64)


def select(rec: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in rec.items()}


# =====================================================================================================================
# 1. fragments
# =====================================================================================================================
MIN_FRAG, MAX_FRAG, MIN_MAPQ = 10, 2000, 30
SHIFT_L, SHIFT_R = 4, 5   # Tn5: start + 4, end - 5


def cpu_fragments(rec: dict, assume_sorted: bool = True) -> dict:
    """Port of ATACFragmentExtractor::process_record + finalise (PR #48), records in the order given.

    The dedup table is cleared whenever (tid, frag_start) changes, so it only merges duplicates that are adjacent
    in the stream: correct for a coordinate-sorted BAM, wrong for any other order (finding F1).
    """
    out = []   # [tid, start, end, cell, read_pairs]
    cur = None
    dedup: dict = {}
    for i in range(len(rec["pos"])):
        flag = int(rec["flag"][i])
        if not flag & F_PROPER or flag & (F_UNMAP | F_MUNMAP | F_SEC | F_SUPP):
            continue
        if rec["mapq"][i] < MIN_MAPQ:
            continue
        tlen = int(rec["tlen"][i])
        if tlen <= 0 or rec["mtid"][i] != rec["tid"][i]:
            continue
        if tlen < MIN_FRAG or tlen > MAX_FRAG:
            continue
        s, e = int(rec["pos"][i]) + SHIFT_L, int(rec["pos"][i]) + tlen - SHIFT_R
        if s >= e:
            continue
        cell = int(rec["cell"][i])
        if cell < 0:
            continue
        tid = int(rec["tid"][i])
        if assume_sorted and (tid, s) != cur:   # assume_sorted=False: the A1 table (atac-orderfree-a1-a2 branch, the new default)
            dedup.clear()
            cur = (tid, s)
        key = (tid, s, e, cell)
        if key in dedup:
            out[dedup[key]][4] += 1
        else:
            dedup[key] = len(out)
            out.append([tid, s, e, cell, 1])
    out.sort(key=lambda f: (f[0], f[1], f[2], f[3]))
    a = np.array(out, dtype=np.int64).reshape(-1, 5)
    return {"tid": a[:, 0], "start": a[:, 1], "end": a[:, 2], "cell": a[:, 3], "read_pairs": a[:, 4]}


def fragment_emit(rec: dict) -> dict:
    """Emit one (cell, contig, start, end) tuple per record that is the leftmost mate of a usable pair."""
    flag = rec["flag"]
    tlen = rec["tlen"].astype(np.int64)
    m = ((flag & F_PROPER) != 0) & ((flag & (F_UNMAP | F_MUNMAP | F_SEC | F_SUPP)) == 0)
    m &= rec["mapq"] >= MIN_MAPQ
    m &= (tlen > 0) & (rec["mtid"] == rec["tid"]) & (tlen >= MIN_FRAG) & (tlen <= MAX_FRAG)
    start = rec["pos"].astype(np.int64) + SHIFT_L
    end = rec["pos"].astype(np.int64) + tlen - SHIFT_R
    m &= start < end
    m &= rec["cell"] >= 0
    return {"cell": rec["cell"][m].astype(np.int64), "tid": rec["tid"][m].astype(np.int64), "start": start[m], "end": end[m]}


def _runs(*keys):
    """Sort by keys (first = most significant) and return (order, first-of-run flags)."""
    order = np.lexsort(keys[::-1])
    n = len(order)
    first = np.ones(n, dtype=bool)
    if n > 1:
        first[1:] = False
        for k in keys:
            ks = k[order]
            first[1:] |= ks[1:] != ks[:-1]
    return order, first


def fragment_reduce(t: dict) -> dict:
    """Collapse identical (cell, contig, start, end) tuples: read_pairs = run length. Output sorted cell-major."""
    order, first = _runs(t["cell"], t["tid"], t["start"], t["end"])
    idx = np.flatnonzero(first)
    runlen = np.diff(np.append(idx, len(order)))
    return {"cell": t["cell"][order][idx], "tid": t["tid"][order][idx], "start": t["start"][order][idx],
            "end": t["end"][order][idx], "read_pairs": runlen.astype(np.int64)}


def fragment_merge(parts: list[dict]) -> dict:
    """Canonical merge of partition outputs (partitions are disjoint in cell, so this is a pure sort)."""
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((cat["end"], cat["start"], cat["tid"], cat["cell"]))
    return {k: v[order] for k, v in cat.items()}


def coord_order(f: dict) -> dict:
    """fragments.tsv order: (contig, start, end, cell), the CPU's finalise() order."""
    order = np.lexsort((f["cell"], f["end"], f["start"], f["tid"]))
    return {k: v[order] for k, v in f.items()}


# =====================================================================================================================
# 2. insertions and the peak caller (pass 1)
# =====================================================================================================================
def insertion_emit(f: dict, is_mito: np.ndarray) -> dict:
    """Both cut sites (start, end - 1) of every unique non-mito fragment; read_pairs does not weight them."""
    keep = ~is_mito[f["tid"]]
    tid, s, e = f["tid"][keep], f["start"][keep], f["end"][keep]
    return {"tid": np.concatenate([tid, tid]), "pos": np.concatenate([s, e - 1])}


def insertion_reduce(t: dict) -> dict:
    """(contig, pos) -> count, sorted. Partition tables merge by summing counts of equal keys (integer, commutative)."""
    order, first = _runs(t["tid"], t["pos"])
    idx = np.flatnonzero(first)
    cnt = np.diff(np.append(idx, len(order)))
    return {"tid": t["tid"][order][idx], "pos": t["pos"][order][idx], "count": cnt.astype(np.int64)}


def insertion_merge(parts: list[dict]) -> dict:
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("tid", "pos", "count")}
    order, first = _runs(cat["tid"], cat["pos"])
    idx = np.flatnonzero(first)
    return {"tid": cat["tid"][order][idx], "pos": cat["pos"][order][idx],
            "count": np.add.reduceat(cat["count"][order], idx) if len(idx) else np.zeros(0, np.int64)}


class PeakParams:
    ext, step, llocal, q_cutoff, min_length, max_gap, genome_size = 200, 10, 10000, 0.05, 200, 30, 0.0
    kbins = 100000
    floor = 1.0


@nb.njit(cache=True)
def poisson_upper_neg_log10(k, lam):
    """-log10 P(X >= k), X ~ Poisson(lam), k > lam: detail::poisson_upper_neg_log10 of atac_peak_caller.h."""
    if k == 0:
        return 0.0
    if lam <= 0:
        return 300.0
    log_t0 = -lam + k * math.log(lam) - math.lgamma(k + 1.0)
    s = 1.0
    term = 1.0
    i = k + 1
    while i < k + 100000:
        term *= lam / i
        s += term
        if term < 1e-15 * s:
            break
        i += 1
    log_p = log_t0 + math.log(s)
    r = -log_p / math.log(10.0)
    return r if r > 0.0 else 0.0


def _chrom_arrays(table: dict, ncontig: int):
    """Per-contig sorted insertion positions (multiplicities expanded)."""
    out = []
    for c in range(ncontig):
        m = table["tid"] == c
        out.append(np.repeat(table["pos"][m], table["count"][m]).astype(np.int64))
    return out


@nb.njit(cache=True)
def _grid_scores(v, L, step, half, lhalf, ext, llocal, lambda_bg):
    """Per grid position: pileup k, and score s (-1 when the position is a zero bin: k == 0 or k <= local lambda)."""
    ng = (L + step - 1) // step
    k = np.zeros(ng, np.int64)
    s = np.full(ng, -1.0)
    for g in range(ng):
        x = g * step + step // 2
        kk = np.searchsorted(v, x + half) - np.searchsorted(v, x - half)
        if kk == 0:
            continue
        nloc = np.searchsorted(v, x + lhalf) - np.searchsorted(v, x - lhalf)
        lam = max(lambda_bg, nloc * ext / llocal)
        if kk <= lam:
            continue
        k[g] = kk
        s[g] = poisson_upper_neg_log10(kk, lam)
    return k, s


def peak_call(table: dict, chrom_len: np.ndarray, skip: np.ndarray, prm=PeakParams) -> dict:
    """Peaks from the merged insertion table. Positions are independent; the only cross-position quantities are the
    background rate (a sum), the score histogram (integer counts) and the BH threshold taken from it."""
    nc = len(chrom_len)
    ins = _chrom_arrays(table, nc)
    half, lhalf = prm.ext // 2, prm.llocal // 2
    total_ins = sum(int(len(ins[c])) for c in range(nc) if not skip[c])
    glen = float(sum(int(chrom_len[c]) for c in range(nc) if not skip[c]))
    gsize = prm.genome_size if prm.genome_size > 0 else 0.87 * glen
    lambda_bg = total_ins * prm.ext / gsize if gsize > 0 else 0.0

    hist = np.zeros(prm.kbins + 1, np.int64)
    n_positions = 0
    cand_c, cand_g, cand_s, cand_k = [], [], [], []
    for c in range(nc):
        if skip[c]:
            continue
        L = int(chrom_len[c])
        k, s = _grid_scores(ins[c], L, prm.step, half, lhalf, prm.ext, prm.llocal, lambda_bg)
        n_positions += len(k)
        zero = s < 0
        b = np.minimum(prm.kbins, np.where(zero, 0, s * 100.0).astype(np.int64))
        hist += np.bincount(b, minlength=prm.kbins + 1)
        sel = np.flatnonzero(~zero & (s >= prm.floor))
        cand_c.append(np.full(len(sel), c, np.int64)); cand_g.append(sel); cand_s.append(s[sel].astype(np.float32)); cand_k.append(k[sel])
    cand_c, cand_g, cand_s, cand_k = (np.concatenate(x) if x else np.zeros(0) for x in (cand_c, cand_g, cand_s, cand_k))

    # BH threshold from the histogram (all integer until the log10)
    q_need = -math.log10(prm.q_cutoff)
    n_ge = np.cumsum(hist[::-1])[::-1]
    raw = np.full(prm.kbins + 1, -1.0)
    nz = hist > 0
    b_idx = np.arange(prm.kbins + 1)
    lg_n = math.log10(n_positions)
    for b in b_idx[nz]:   # math.log10, not np.log10: the same libm call as the CPU (the sums stay in this order)
        raw[b] = b / 100.0 + math.log10(int(n_ge[b])) - lg_n
    run = np.maximum.accumulate(np.maximum(raw, 0.0))
    qscore = run.astype(np.float32)
    ok = np.flatnonzero(qscore.astype(np.float64) >= q_need)
    s_thresh = (ok[0] / 100.0) if len(ok) else 1e9
    s_thresh = max(s_thresh, prm.floor)

    keep = cand_s.astype(np.float64) >= s_thresh
    cc, cg, cs, ck = cand_c[keep], cand_g[keep], cand_s[keep], cand_k[keep]
    empty = {"tid": np.zeros(0, np.int64), "start": np.zeros(0, np.int64), "end": np.zeros(0, np.int64),
             "summit": np.zeros(0, np.int64), "pileup": np.zeros(0, np.float32), "nlp": np.zeros(0, np.float32), "nlq": np.zeros(0, np.float32)}
    if len(cc) == 0:
        return empty
    gs = cg * prm.step
    ge = np.minimum(gs + prm.step, chrom_len[cc])
    qs = qscore[np.minimum(prm.kbins, (cs * np.float32(100.0)).astype(np.int64))]
    new = np.ones(len(cc), bool)
    new[1:] = (cc[1:] != cc[:-1]) | (gs[1:] - ge[:-1] > prm.max_gap)
    st = np.flatnonzero(new)
    seg = np.cumsum(new) - 1
    maxk = np.maximum.reduceat(ck, st)
    big = np.iinfo(np.int64).max
    first_g = np.minimum.reduceat(np.where(ck == maxk[seg], cg, big), st)
    ends = np.append(st[1:], len(cc)) - 1
    res = {"tid": cc[st], "start": gs[st], "end": ge[ends], "summit": first_g * prm.step + prm.step // 2,
           "pileup": maxk.astype(np.float32), "nlp": np.maximum.reduceat(cs, st), "nlq": np.maximum.reduceat(qs, st)}
    ln = res["end"] - res["start"] >= prm.min_length
    return {k: v[ln] for k, v in res.items()}


def cpu_peak_call(insertions: list[list[int]], chrom_len, skip, prm=PeakParams) -> dict:
    """Port of call_peaks_insertions (atac_peak_caller.h), sequential two-pointer scan, float32 where the C++ has float."""
    nc = len(insertions)
    half, lhalf = prm.ext // 2, prm.llocal // 2
    ins = [sorted(v) for v in insertions]
    total_ins = sum(len(ins[c]) for c in range(nc) if not skip[c])
    glen = float(sum(chrom_len[c] for c in range(nc) if not skip[c]))
    gsize = prm.genome_size if prm.genome_size > 0 else 0.87 * glen
    lambda_bg = total_ins * prm.ext / gsize if gsize > 0 else 0.0
    hist = [0] * (prm.kbins + 1)
    cand = []
    n_positions = 0
    f32 = np.float32
    for c in range(nc):
        if skip[c]:
            continue
        v = ins[c]
        ng = (chrom_len[c] + prm.step - 1) // prm.step
        n_positions += ng
        lo = hi = llo = lhi = 0
        zero = 0
        for g in range(ng):
            x = g * prm.step + prm.step // 2
            while hi < len(v) and v[hi] < x + half: hi += 1
            while lo < hi and v[lo] < x - half: lo += 1
            k = hi - lo
            if k == 0:
                zero += 1
                continue
            while lhi < len(v) and v[lhi] < x + lhalf: lhi += 1
            while llo < lhi and v[llo] < x - lhalf: llo += 1
            lam = max(lambda_bg, (lhi - llo) * prm.ext / prm.llocal)
            if k <= lam:
                zero += 1
                continue
            s = poisson_upper_neg_log10(k, lam)
            b = min(prm.kbins, int(s * 100.0))
            hist[b] += 1
            if s >= prm.floor:
                cand.append((c, g, f32(s), k))
        hist[0] += zero
    q_need = -math.log10(prm.q_cutoff)
    qscore = [f32(0)] * (prm.kbins + 1)
    raw = [-1.0] * (prm.kbins + 1)
    n_ge = 0
    for b in range(prm.kbins, -1, -1):
        n_ge += hist[b]
        if hist[b] == 0:
            continue
        raw[b] = b / 100.0 + math.log10(n_ge) - math.log10(n_positions)
    run = 0.0
    for b in range(prm.kbins + 1):
        if raw[b] > run:
            run = raw[b]
        qscore[b] = f32(run)
    s_thresh = 1e9
    for b in range(prm.kbins + 1):
        if float(qscore[b]) >= q_need:
            s_thresh = b / 100.0
            break
    s_thresh = max(s_thresh, prm.floor)
    peaks = []
    cur = None
    last_ge = 0
    for (c, g, sc, k) in cand:
        if float(sc) < s_thresh:
            continue
        gs = g * prm.step
        ge = min(gs + prm.step, chrom_len[c])
        qs = qscore[min(prm.kbins, int(sc * f32(100.0)))]
        if cur is not None and c == cur["tid"] and gs - last_ge <= prm.max_gap:
            cur["end"] = ge
            if k > cur["pileup"]:
                cur["pileup"] = k
                cur["summit"] = gs + prm.step // 2
            cur["nlp"] = max(cur["nlp"], sc)
            cur["nlq"] = max(cur["nlq"], qs)
        else:
            if cur is not None and cur["end"] - cur["start"] >= prm.min_length:
                peaks.append(cur)
            cur = {"tid": c, "start": gs, "end": ge, "summit": gs + prm.step // 2, "pileup": k, "nlp": sc, "nlq": qs}
        last_ge = ge
    if cur is not None and cur["end"] - cur["start"] >= prm.min_length:
        peaks.append(cur)
    return {"tid": np.array([p["tid"] for p in peaks], np.int64), "start": np.array([p["start"] for p in peaks], np.int64),
            "end": np.array([p["end"] for p in peaks], np.int64), "summit": np.array([p["summit"] for p in peaks], np.int64),
            "pileup": np.array([p["pileup"] for p in peaks], np.float32), "nlp": np.array([p["nlp"] for p in peaks], np.float32),
            "nlq": np.array([p["nlq"] for p in peaks], np.float32)}


# =====================================================================================================================
# 3. cell x peak counts and per-cell metrics (pass 2)
# =====================================================================================================================
def _peak_keys(peaks: dict):
    return (peaks["tid"] << 32) | peaks["start"], (peaks["tid"] << 32) | peaks["end"]


def peak_of(peaks: dict, tid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the (disjoint, sorted) peak containing x, or -1. The CPU's upper_bound on the end, then start <= x."""
    ps, pe = _peak_keys(peaks)
    key = (tid << 32) | x
    i = np.searchsorted(pe, key, side="right")
    ok = i < len(pe)
    ok[ok] &= ps[i[ok]] <= key[ok]
    return np.where(ok, i, -1)


def peak_overlaps(peaks: dict, tid: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    ps, pe = _peak_keys(peaks)
    ks, ke = (tid << 32) | s, (tid << 32) | e
    i = np.searchsorted(pe, ks, side="right")
    ok = i < len(pe)
    ok[ok] &= ps[i[ok]] < ke[ok]
    return ok


def peak_count_emit(f: dict, peaks: dict, is_mito: np.ndarray) -> dict:
    """One (cell, peak) tuple per fragment end inside a peak (Cell Ranger cut-site counting). chrM has no peaks."""
    keep = ~is_mito[f["tid"]]
    cell, tid, s, e = f["cell"][keep], f["tid"][keep], f["start"][keep], f["end"][keep]
    p1, p2 = peak_of(peaks, tid, s), peak_of(peaks, tid, e - 1)
    return {"cell": np.concatenate([cell[p1 >= 0], cell[p2 >= 0]]), "peak": np.concatenate([p1[p1 >= 0], p2[p2 >= 0]])}


def peak_count_reduce(t: dict) -> dict:
    order, first = _runs(t["cell"], t["peak"])
    idx = np.flatnonzero(first)
    cnt = np.diff(np.append(idx, len(order)))
    return {"cell": t["cell"][order][idx], "peak": t["peak"][order][idx], "count": cnt.astype(np.int64)}


def peak_count_merge(parts: list[dict]) -> dict:
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("cell", "peak", "count")}
    order = np.lexsort((cat["peak"], cat["cell"]))
    return {k: v[order] for k, v in cat.items()}


def cell_metrics(f: dict, peaks: dict, is_mito: np.ndarray, ncells: int) -> dict:
    """Per-cell integer counters used by the cell caller (atac_metrics.h): read_pairs, fragments, mito_fragments,
    peak_region_fragments, peak_region_cutsites. Reduce-by-cell of integers."""
    mt = is_mito[f["tid"]]
    nm = ~mt
    p1, p2 = peak_of(peaks, f["tid"], f["start"]), peak_of(peaks, f["tid"], f["end"] - 1)
    cuts = ((p1 >= 0).astype(np.int64) + (p2 >= 0))
    ov = (p1 >= 0) | (p2 >= 0) | peak_overlaps(peaks, f["tid"], f["start"], f["end"])
    bc = lambda w: np.bincount(f["cell"], weights=w, minlength=ncells).astype(np.int64)   # exact: integer weights < 2^53
    return {"read_pairs": bc(f["read_pairs"]), "fragments": bc(nm), "mito_fragments": bc(mt),
            "peak_region_fragments": bc(nm & ov), "peak_region_cutsites": bc(np.where(nm, cuts, 0))}


def cpu_cell_metrics_and_matrix(frags: dict, peaks: dict, is_mito, ncells: int):
    """Port of compute_atac_metrics (peak part) + AtacTriplets::to_csc, fragments in the order given."""
    m = {k: [0] * ncells for k in ("read_pairs", "fragments", "mito_fragments", "peak_region_fragments", "peak_region_cutsites")}
    per_chrom = {}
    for i in range(len(peaks["tid"])):
        per_chrom.setdefault(int(peaks["tid"][i]), []).append((int(peaks["start"][i]), int(peaks["end"][i]), i))
    trip = []
    def pk(c, x):
        for (s, e, i) in per_chrom.get(c, []):
            if s <= x < e:
                return i
        return -1
    for j in range(len(frags["cell"])):
        c, t, s, e, rp = (int(frags[k][j]) for k in ("cell", "tid", "start", "end", "read_pairs"))
        m["read_pairs"][c] += rp
        if is_mito[t]:
            m["mito_fragments"][c] += 1
            continue
        m["fragments"][c] += 1
        p1, p2 = pk(t, s), pk(t, e - 1)
        m["peak_region_cutsites"][c] += (p1 >= 0) + (p2 >= 0)
        if p1 >= 0 or p2 >= 0 or any(ps < e and pe > s for (ps, pe, _) in per_chrom.get(t, [])):
            m["peak_region_fragments"][c] += 1
        if p1 >= 0: trip.append((c, p1))
        if p2 >= 0: trip.append((c, p2))
    trip.sort()
    out, last = [], None
    for tp in trip:
        if tp == last: out[-1][2] += 1
        else: out.append([tp[0], tp[1], 1]); last = tp
    a = np.array(out, np.int64).reshape(-1, 3)
    return {k: np.array(v, np.int64) for k, v in m.items()}, {"cell": a[:, 0], "peak": a[:, 1], "count": a[:, 2]}


# =====================================================================================================================
# whole pipeline in partitions
# =====================================================================================================================
MIN_CELL_PEAK_FRAGMENTS = 8


def default_cell_caller(metrics: dict) -> np.ndarray:
    """Stand-in for the host mixture cell call (call_cells_mixture): a floor on fragments in peaks. The real caller runs on the
    host over the merged per-cell table in whitelist order (A5); only its input (peak_region_fragments) matters here."""
    return metrics["peak_region_fragments"] >= MIN_CELL_PEAK_FRAGMENTS


def run_all(rec: dict, K: int, chrom_len: np.ndarray, is_mito: np.ndarray, ncells: int, cell_caller=default_cell_caller) -> dict:
    """Rule A8 (peaks from called cells). PR #48's cell call needs peaks (peak_region_fragments), so it is two-pass:
      1. per partition: fragments and the insertion table of ALL resolved barcodes -> canonical merge -> first peak set
      2. per-cell metrics against the first peak set -> cell call (host)
      3. insertion tables of the CALLED cells only -> canonical merge -> final peak set
      4. per partition: cell x peak counts against the final peak set; metrics recomputed on the final peaks (the cell set is
         not re-called: no iteration)
    Everything returned is a function of the set of records, for any K."""
    part = cell_part(rec["cell"], K)
    frag_parts, ins_all = [], []
    for p in range(K):
        f = fragment_reduce(fragment_emit(select(rec, part == p)))
        frag_parts.append(f)
        ins_all.append(insertion_reduce(insertion_emit(f, is_mito)))
    table_all = insertion_merge(ins_all)
    peaks_all = peak_call(table_all, chrom_len, is_mito)
    frags = fragment_merge(frag_parts)
    is_cell = np.asarray(cell_caller(cell_metrics(frags, peaks_all, is_mito, ncells)), dtype=bool)
    ins_cells = [insertion_reduce(insertion_emit(select(f, is_cell[f["cell"]]), is_mito)) for f in frag_parts]
    table = insertion_merge(ins_cells)
    peaks = peak_call(table, chrom_len, is_mito)
    cnt_parts = [peak_count_reduce(peak_count_emit(f, peaks, is_mito)) for f in frag_parts]
    return {"fragments": frags, "insertions_all": table_all, "peaks_all": peaks_all, "is_cell": is_cell.astype(np.uint8),
            "insertions": table, "peaks": peaks, "peak_counts": peak_count_merge(cnt_parts),
            "metrics": cell_metrics(frags, peaks, is_mito, ncells)}


def to_bytes(d: dict) -> bytes:
    """Deterministic serialisation: dict keys sorted, nested dicts flattened, arrays as little-endian bytes."""
    out = bytearray()
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict):
            out += k.encode() + b"{" + to_bytes(v) + b"}"
        else:
            out += k.encode() + b":" + np.ascontiguousarray(v).astype(v.dtype.newbyteorder("<")).tobytes()
    return bytes(out)
