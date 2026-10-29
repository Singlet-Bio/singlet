"""Laptop check for ReadAlign::seqCacheLeanAln (src/star/ReadAlign_seqCache.cpp): the BAM core fields of a single-end alignment
computed without building the record.

Run: python bench/host_cost/model_lean_aln.py   (plain Python; no STAR, no compiler)

alignbam_fields(): a transcription of the single-end mapped branch of ReadAlign::alignBAM (src/star/ReadAlign_alignBAM.cpp: nMates,
flag, trimL, CIGAR, trimR, MAPQ, refID/pos, the NH / nM integer tags through bamAttrArrayWriteInt and the parse that
seqCacheAlignRecords does on them), with the C++ integer widths (uint = 64 bit, packedCIGAR and pBAM words = 32 bit).
lean_fields(): a transcription of seqCacheLeanAln plus the record layout of seqCacheAlignRecords.
Random transcripts: 1-8 exons, junctions (canonSJ >= 0 or annotated), deletions, insertions, both strands, soft clips from the
exon positions and from clipMates.clippedN, primary and not, 1-12 alignments (NH / MAPQ steps), clipped reads, outSAMflagAND/OR.
This checks the transcription (the arithmetic and the types), not the C++ build: SINGLET_HANDOFF_LEAN=verify does that on Clipper.
"""
import random
import struct

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1
S, I, D, N, MM = 4, 1, 2, 3, 0   # BAM_CIGAR_S, _I, _D, _N, _M
SHIFT = 4


def attr_int(x):
    """bamAttrArrayWriteInt: (type char, width) for a non-negative x"""
    if x <= 255:
        return 'C', 1
    if x <= 65535:
        return 'S', 2
    return 'I', 4


def alignbam_fields(t, nTrOut, trChrStart, P, readLength, readLengthOriginal, clip, readFilter='N'):
    nEx = t['nExons']
    # nMates: any canonSJ == -3 -> two mates (the hand-off never has them: single-end)
    for i in range(nEx - 1):
        if t['canonSJ'][i] == -3:
            return None
    iEx1, iEx2 = 0, nEx - 1
    samFLAG = 0
    if readFilter == 'Y':
        samFLAG |= 0x200
    if not t['primaryFlag']:
        samFLAG |= 0x100
    Mate = t['exons'][iEx1][3]
    Str = t['Str']
    if Mate == 0:
        samFLAG |= Str * 0x10
    else:
        samFLAG |= (1 - Str) * 0x10
    if Str == 0 and Mate == 0:
        trimL = clip[Mate][0]
    elif Str == 0 and Mate == 1:
        trimL = clip[Mate][1]
    elif Str == 1 and Mate == 0:
        trimL = clip[Mate][1]
    else:
        trimL = clip[Mate][0]
    leftMate = 0
    cig = []
    E = t['exons']  # (R, G, L, iFrag)
    trimL1 = (trimL + E[iEx1][0] - (0 if E[iEx1][0] < readLength[leftMate] else readLength[leftMate] + 1)) & M64
    if trimL1 > 0:
        cig.append(((trimL1 << SHIFT) | S) & M32)
    for ii in range(iEx1, iEx2 + 1):
        if ii > iEx1:
            gapG = (E[ii][1] - (E[ii - 1][1] + E[ii - 1][2])) & M64
            gapR = (E[ii][0] - E[ii - 1][0] - E[ii - 1][2]) & M64
            if gapR > 0:
                cig.append(((gapR << SHIFT) | I) & M32)
            if t['canonSJ'][ii - 1] >= 0 or t['sjAnnot'][ii - 1] == 1:
                cig.append(((gapG << SHIFT) | N) & M32)
            elif gapG > 0:
                cig.append(((gapG << SHIFT) | D) & M32)
        if E[ii][2] > 0:
            cig.append(((E[ii][2] << SHIFT) | MM) & M32)
    base = readLength[leftMate] if E[iEx1][0] < readLength[leftMate] else None
    first = readLengthOriginal[leftMate] if E[iEx1][0] < readLength[leftMate] else readLength[leftMate] + 1 + readLengthOriginal[Mate]
    trimR1 = (first - E[iEx2][0] - E[iEx2][2] - trimL) & M64
    if trimR1 > 0:
        cig.append(((trimR1 << SHIFT) | S) & M32)
    MAPQ = P['mapqUnique']
    if nTrOut >= 5:
        MAPQ = 0
    elif nTrOut >= 3:
        MAPQ = 1
    elif nTrOut == 2:
        MAPQ = 3
    w1 = t['Chr'] & M32
    w2 = (E[iEx1][1] - trChrStart) & M32
    readNameLen = 2   # strlen("@x")
    w3 = ((12345 & 0xFFFF) << 16 | (MAPQ << 8) | readNameLen) & M32
    w4 = ((((samFLAG & P['flagAND']) | P['flagOR']) << 16) | len(cig)) & M32
    # the tags, as written then parsed back
    nh_t, nh_w = attr_int(nTrOut)
    nm_t, nm_w = attr_int(t['nMM'])
    return dict(refID=struct.unpack('<i', struct.pack('<I', w1))[0], pos=struct.unpack('<i', struct.pack('<I', w2))[0],
                flag=(w4 >> 16) & 0xFFFF, mapq=(w3 >> 8) & 0xFF, cig=cig, nh=nTrOut, nm=t['nMM'])


def lean_fields(t, nTrOut, trChrStart, P, readLength, readLengthOriginal, clip, readFilter='N'):
    nEx = t['nExons']
    for ii in range(nEx - 1):
        if t['canonSJ'][ii] == -3:
            return None
    iEx1, iEx2, leftMate = 0, nEx - 1, 0
    samFLAG = 0
    if readFilter == 'Y':
        samFLAG |= 0x200
    if not t['primaryFlag']:
        samFLAG |= 0x100
    E = t['exons']
    Mate = E[iEx1][3]
    Str = t['Str']
    samFLAG |= (Str * 0x10) if Mate == 0 else ((1 - Str) * 0x10)
    if Str == 0 and Mate == 0:
        trimL = clip[Mate][0]
    elif Str == 0 and Mate == 1:
        trimL = clip[Mate][1]
    elif Str == 1 and Mate == 0:
        trimL = clip[Mate][1]
    else:
        trimL = clip[Mate][0]
    n = []
    trimL1 = (trimL + E[iEx1][0] - (0 if E[iEx1][0] < readLength[leftMate] else readLength[leftMate] + 1)) & M64
    if trimL1 > 0:
        n.append(((trimL1 << SHIFT) | S) & M32)
    for ii in range(iEx1, iEx2 + 1):
        if ii > iEx1:
            gapG = (E[ii][1] - (E[ii - 1][1] + E[ii - 1][2])) & M64
            gapR = (E[ii][0] - E[ii - 1][0] - E[ii - 1][2]) & M64
            if gapR > 0:
                n.append(((gapR << SHIFT) | I) & M32)
            if t['canonSJ'][ii - 1] >= 0 or t['sjAnnot'][ii - 1] == 1:
                n.append(((gapG << SHIFT) | N) & M32)
            elif gapG > 0:
                n.append(((gapG << SHIFT) | D) & M32)
        if E[ii][2] > 0:
            n.append(((E[ii][2] << SHIFT) | MM) & M32)
    trimR1 = ((readLengthOriginal[leftMate] if E[iEx1][0] < readLength[leftMate] else readLength[leftMate] + 1 + readLengthOriginal[Mate])
              - E[iEx2][0] - E[iEx2][2] - trimL) & M64
    if trimR1 > 0:
        n.append(((trimR1 << SHIFT) | S) & M32)
    MAPQ = P['mapqUnique']
    if nTrOut >= 5:
        MAPQ = 0
    elif nTrOut >= 3:
        MAPQ = 1
    elif nTrOut == 2:
        MAPQ = 3
    refID = struct.unpack('<i', struct.pack('<I', t['Chr'] & M32))[0]
    pos = struct.unpack('<i', struct.pack('<I', (E[iEx1][1] - trChrStart) & M32))[0]
    flag = ((samFLAG & P['flagAND']) | P['flagOR']) & 0xFFFF
    return dict(refID=refID, pos=pos, flag=flag, mapq=MAPQ & 0xFF, cig=n, nh=nTrOut, nm=t['nMM'])


def random_transcript(rng, L, chrStart):
    nEx = rng.choice([1, 1, 1, 2, 2, 3, 4, 6, 8])
    E = []; canon = []; annot = []
    r = rng.randint(0, 5); g = chrStart + rng.randint(1000, 10 ** 8)
    for k in range(nEx):
        ln = rng.randint(1, 40)
        if r + ln > L:
            ln = max(1, L - r)
        E.append([r, g, ln, 0])
        if k + 1 < nEx:
            kind = rng.choice(['sj', 'sj', 'del', 'ins', 'annot_del', 'insdel'])
            gapG = rng.randint(1, 50000) if kind in ('sj', 'annot_del') else (rng.randint(1, 5) if kind in ('del', 'insdel') else 0)
            gapR = rng.randint(1, 4) if kind in ('ins', 'insdel') else 0
            canon.append(rng.choice([0, 1, 2, 3, 4, 5, 6]) if kind == 'sj' else (-1 if kind in ('del', 'insdel') else (-2 if kind == 'ins' else -1)))
            annot.append(1 if kind in ('annot_del',) or (kind == 'sj' and rng.random() < 0.5) else 0)
            r = r + ln + gapR; g = g + ln + gapG
    return dict(nExons=nEx, exons=[tuple(e) for e in E], canonSJ=canon + [0], sjAnnot=annot + [0], Str=rng.randint(0, 1),
                primaryFlag=rng.random() < 0.7, Chr=rng.randint(0, 40), nMM=rng.choice([0, 0, 0, 1, 2, 5, 40, 300, 70000]))


def main():
    rng = random.Random(7)
    n = bad = 0
    for _ in range(300000):
        L = rng.randint(20, 300)
        chrStart = rng.randint(0, 10 ** 9)
        P = dict(mapqUnique=rng.choice([255, 255, 255, 60]), flagAND=rng.choice([0xFFFF, 0xFFFF, 0xFFF0]), flagOR=rng.choice([0, 0, 0x400]))
        clip = [[rng.choice([0, 0, 3, 10]), rng.choice([0, 0, 5, 30])], [0, 0]]
        readLength = [L, 0]; readLengthOriginal = [L + rng.choice([0, 0, 0, 3, 10, 40]), 0]
        t = random_transcript(rng, L, chrStart)
        nTr = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 9, 10, 300, 70000])
        rf = rng.choice(['N', 'N', 'Y'])
        a = alignbam_fields(t, nTr, chrStart, P, readLength, readLengthOriginal, clip, rf)
        b = lean_fields(t, nTr, chrStart, P, readLength, readLengthOriginal, clip, rf)
        n += 1
        if a != b:
            bad += 1
            print("DIFF", t, a, b)
            break
    # a mate gap falls back to alignBAM in both
    t = random_transcript(rng, 100, 0); t['canonSJ'][0] = -3; t['nExons'] = max(t['nExons'], 2)
    t['exons'] = t['exons'] + [t['exons'][-1]] * (2 - len(t['exons']))
    assert alignbam_fields(t, 1, 0, dict(mapqUnique=255, flagAND=0xFFFF, flagOR=0), [100, 0], [100, 0], [[0, 0], [0, 0]]) is None
    print(f"OK: {n} random alignments, {bad} differ")
    assert bad == 0


if __name__ == "__main__":
    main()
