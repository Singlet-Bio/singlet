"""Synthetic scATAC / Multiome-ATAC BAM-record fixture (seeded, well under 1 MB): paired-end records with the noise the
CPU extractor filters on (duplicates, low MAPQ, secondary, mate unmapped, cross-contig mates, unresolved barcodes, out-of-range
insert sizes, either mate as read 1) and planted accessible regions so the peak caller has something to find."""
from __future__ import annotations

import numpy as np

from atac_ref import F_MUNMAP, F_PAIRED, F_PROPER, F_READ1, F_READ2, F_REV, F_SEC

CHROM_LEN = np.array([300_000, 200_000, 120_000, 2_000], np.int64)   # chr1, chr2, chr3, chrM
IS_MITO = np.array([False, False, False, True])
N_REAL = 60                   # cells
NCELLS = 100                  # barcode space: 40 more barcodes are empty droplets (a few ambient fragments each)
READLEN = 50


def make_records(seed: int = 7, n_bg: int = 2600, n_planted: int = 12, per_peak: int = 120):
    rng = np.random.default_rng(seed)
    frags = []   # (cell, tid, start, flen)
    for _ in range(n_bg):
        t = rng.choice(4, p=[0.45, 0.30, 0.2, 0.05])
        flen = int(np.clip(rng.gamma(2.2, 90), 20, 700))
        frags.append((rng.integers(N_REAL), t, rng.integers(0, CHROM_LEN[t] - flen), flen))
    planted = []
    for _ in range(n_planted):
        t = int(rng.integers(0, 3))
        c = int(rng.integers(5_000, CHROM_LEN[t] - 5_000))
        planted.append((t, c))
        for _ in range(per_peak):
            flen = int(np.clip(rng.gamma(2.0, 80), 40, 600))
            frags.append((rng.integers(N_REAL), t, c + int(rng.integers(-250, 250)), flen))
    for e in range(N_REAL, NCELLS):   # empty droplets: 8-14 uniformly placed fragments, no accessibility structure
        for _ in range(int(rng.integers(8, 15))):
            tt = int(rng.choice(3)); flen = int(np.clip(rng.gamma(2.2, 90), 20, 700))
            frags.append((e, tt, rng.integers(0, CHROM_LEN[tt] - flen), flen))
    # PCR duplicates of a random subset (1-4 extra copies)
    dup = [frags[i] for i in rng.choice(len(frags), len(frags) // 6, replace=False) for _ in range(int(rng.integers(1, 5)))]
    frags += dup
    rows = []
    qid = 0
    for (cell, t, start, flen) in frags:
        cell = int(cell)
        if rng.random() < 0.03:
            cell = -1   # barcode did not resolve
        mapq = 255 if rng.random() > 0.08 else int(rng.choice([0, 1, 3]))
        left_is_read1 = rng.random() < 0.5
        l_rev, r_rev = False, True
        fl = int(flen)
        # left mate
        rows.append((cell, t, int(start), fl, t, F_PAIRED | F_PROPER | (F_READ1 if left_is_read1 else F_READ2), mapq, qid))
        # right mate (negative TLEN, position of its own start)
        rows.append((cell, t, int(start) + max(fl - READLEN, 0), -fl, t, F_PAIRED | F_PROPER | F_REV | (F_READ2 if left_is_read1 else F_READ1), mapq, qid))
        qid += 1
    # noise records
    for _ in range(150):   # mate unmapped / secondary / cross-contig mate / improper
        t = int(rng.integers(0, 3))
        s = int(rng.integers(0, CHROM_LEN[t] - 600))
        kind = int(rng.integers(0, 4))
        flag = F_PAIRED | F_PROPER | F_READ1
        mtid = t
        if kind == 0: flag |= F_MUNMAP
        elif kind == 1: flag |= F_SEC
        elif kind == 2: mtid = (t + 1) % 3
        else: flag &= ~F_PROPER
        rows.append((int(rng.integers(N_REAL)), t, s, 300, mtid, flag, 255, qid)); qid += 1
    a = np.array(rows, dtype=np.int64)
    rec = {"cell": a[:, 0].astype(np.int32), "tid": a[:, 1].astype(np.int32), "pos": a[:, 2].astype(np.int32),
           "tlen": a[:, 3].astype(np.int32), "mtid": a[:, 4].astype(np.int32), "flag": a[:, 5].astype(np.int32),
           "mapq": a[:, 6].astype(np.int32), "qid": a[:, 7].astype(np.int32)}
    return rec, planted


def shuffled(rec: dict, seed: int) -> dict:
    p = np.random.default_rng(seed).permutation(len(rec["pos"]))
    return {k: v[p] for k, v in rec.items()}


def coord_sorted(rec: dict) -> dict:
    o = np.lexsort((rec["pos"], rec["tid"]))
    return {k: v[o] for k, v in rec.items()}


def nbytes(rec: dict) -> int:
    return sum(v.nbytes for v in rec.values())
