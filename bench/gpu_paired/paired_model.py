"""Reference model of STAR's paired-end window join, stitching and scoring (phase 1d preparation).

A transliteration, rule by rule, of the STAR 2.7.11b code vendored in src/star/ (the file and lines are cited at each rule;
docs/research-beta/paired-end-prep.md has the full table), restricted to what paired-end alignment touches:

  oneRead (mates -> one combined read) -> mapOneRead seed loop -> storeAligns -> stitchPieces (createExtendWindowsWithAlign,
  flanks, assignAlignToWindow) -> stitchWindowAligns (stitchAlignToTranscript, extendAlign, blocksOverlap, finalization with
  the mate rules) -> multMapSelect -> mappedFilter -> primary flag.

Not modelled (documented in the handoff): sjdb-inserted seeds (sjA is always -1), the suffix array (seed loci are found by
brute force and ordered by position, so tie order between windows is not STAR's SA order), peOverlap merging
(--peOverlapNbasesMin > 0, off by default and not used by Singlet), chimeric detection, variation, BySJout stage 2, clipping.

numba compiles the numeric kernels (extend_align, blocks_overlap, mate_overlap_ok); the control flow is plain Python.
Integers are Python ints, so STAR's unsigned (64-bit) arithmetic is only reproduced where the code depends on it (-1 as a
sentinel, the (int) cast of the extension length).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit

U64 = (1 << 64) - 1
MARK_FRAG_SPACER_BASE = 11
DEF_readSeqLengthMax = 650
MAX_N_EXONS = 20
uintWinBinMax = 65535
MAX_SJ_REPEAT_SEARCH = 255
SJ_NONE = U64  # STAR passes (uint)-1 for "not from sjdb"

# column indices (IncludeDefine.h)
PC_rStart, PC_Length, PC_Str, PC_Dir, PC_Nrep, PC_SAstart, PC_SAend, PC_iFrag = range(8)
WC_Str, WC_Chr, WC_gStart, WC_gEnd = range(4)
WA_Length, WA_rStart, WA_gStart, WA_Nrep, WA_Anchor, WA_iFrag, WA_sjA = range(7)
EX_R, EX_G, EX_L, EX_iFrag, EX_sjA = range(5)

_NUC = {"A": 0, "C": 1, "G": 2, "T": 3}


def nucl(seq: str) -> np.ndarray:
    return np.array([_NUC.get(c, 4) for c in seq.upper()], dtype=np.int8)


def revcomp_str(s: str) -> str:
    return s.translate(str.maketrans("ACGTacgtN", "TGCAtgcaN"))[::-1]


# ---------------------------------------------------------------------------------------------------------------------
# Parameters (parametersDefault) and the genome-dependent redefinitions of Genome_genomeLoad.cpp
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Params:
    # seeds
    maxNsplit: int = 10  # Parameters.cpp:473
    seedSplitMin: int = 12
    seedSearchStartLmax: int = 50
    seedSearchStartLmaxOverLread: float = 1.0
    seedMapMin: int = 5
    seedSearchLmax: int = 0
    seedMultimapNmax: int = 10000
    seedPerWindowNmax: int = 50
    winAnchorMultimapNmax: int = 50
    # windows
    winBinNbits: int = 16
    winAnchorDistNbins: int = 9
    winFlankNbins: int = 4
    alignWindowsPerReadNmax: int = 10000
    alignTranscriptsPerWindowNmax: int = 100
    alignTranscriptsPerReadNmax: int = 10000
    # alignment
    alignIntronMin: int = 21
    alignIntronMax: int = 0
    alignMatesGapMax: int = 0
    alignSJoverhangMin: int = 5
    alignSJDBoverhangMin: int = 3
    alignSplicedMateMapLmin: int = 0
    alignSplicedMateMapLminOverLmate: float = 0.66
    alignSJstitchMismatchNmax: Tuple[int, int, int, int] = (0, -1, 0, 0)
    alignSoftClipAtReferenceEnds: bool = True
    alignEndsTypeExt: Tuple[Tuple[bool, bool], Tuple[bool, bool]] = ((False, False), (False, False))
    alignEndsProtrudeNbasesMax: int = 0
    alignEndsProtrudeConcordantPair: bool = False  # Parameters.cpp:1085: false unless nBasesMax > 0 and the 2nd word is ConcordantPair
    flushRight: bool = False
    # filters and scores
    outFilterMultimapScoreRange: int = 1
    outFilterMultimapNmax: int = 10
    outFilterMismatchNmax: int = 10
    outFilterMismatchNoverLmax: float = 0.3
    outFilterMismatchNoverReadLmax: float = 1.0
    outFilterScoreMin: int = 0
    outFilterScoreMinOverLread: float = 0.66
    outFilterMatchNmin: int = 0
    outFilterMatchNminOverLread: float = 0.66
    outFilterIntronMotifs: str = "None"
    removeInconsistentStrands: bool = True
    outSAMstrandFieldType: int = 0
    scoreGap: int = 0
    scoreGapNoncan: int = -8
    scoreGapGCAG: int = -4
    scoreGapATAC: int = -8
    scoreGenomicLengthLog2scale: float = -0.25
    scoreDelOpen: int = -2
    scoreDelBase: int = -2
    scoreInsOpen: int = -2
    scoreInsBase: int = -2
    scoreStitchSJshift: int = 1
    # derived from the genome (derive())
    winBinChrNbits: int = 0
    winBinN: int = 0

    @staticmethod
    def spliced() -> "Params":
        return Params()

    @staticmethod
    def atac() -> "Params":
        """ATAC / unspliced: --alignIntronMax 1 --alignMatesGapMax 2000"""
        return Params(alignIntronMax=1, alignMatesGapMax=2000)

    def derive(self, nGenome: int, gChrBinNbits: int) -> "Params":
        """Genome_genomeLoad.cpp:419-449"""
        if self.alignIntronMax == 0 and self.alignMatesGapMax == 0:
            pass  # window geometry stays at the defaults (the max intron is 2^winBinNbits * winAnchorDistNbins)
        else:
            gap = 1000 if self.alignMatesGapMax == 0 else self.alignMatesGapMax  # :426
            self.winBinNbits = int(math.floor(math.log2(max(max(4, self.alignIntronMax), gap) / 4) + 0.5))
            self.winBinNbits = max(self.winBinNbits, int(math.floor(math.log2(nGenome // 40000 + 1) + 0.5)))
        if self.winBinNbits > gChrBinNbits:  # :432
            self.winBinNbits = gChrBinNbits
        if not (self.alignIntronMax == 0 and self.alignMatesGapMax == 0):  # :439
            self.winFlankNbins = max(self.alignIntronMax, self.alignMatesGapMax) // (1 << self.winBinNbits) + 1
            self.winAnchorDistNbins = 2 * self.winFlankNbins
        self.winBinChrNbits = gChrBinNbits - self.winBinNbits
        self.winBinN = nGenome // (1 << self.winBinNbits) + 1
        return self


# ---------------------------------------------------------------------------------------------------------------------
# Genome: chromosomes at multiples of 2^gChrBinNbits, padding 5 (STAR's GENOME_spacingChar)
# ---------------------------------------------------------------------------------------------------------------------
class Genome:
    def __init__(self, chroms: Dict[str, str], gChrBinNbits: int = 18):
        self.gChrBinNbits = gChrBinNbits
        self.names = list(chroms)
        step = 1 << gChrBinNbits
        self.chrStart: List[int] = []
        self.chrLength: List[int] = []
        pos = 0
        for n in self.names:
            self.chrStart.append(pos)
            self.chrLength.append(len(chroms[n]))
            pos += ((len(chroms[n]) + step) // step) * step  # chrStart is the next multiple past the sequence
        self.nGenome = pos
        self.G = np.full(self.nGenome, 5, dtype=np.int8)
        for n, s in chroms.items():
            i = self.names.index(n)
            self.G[self.chrStart[i]:self.chrStart[i] + len(s)] = nucl(s)
        self.chrBin = np.zeros(self.nGenome // step + 1, dtype=np.int64)
        for i, st in enumerate(self.chrStart):
            nb = ((self.chrStart[i + 1] if i + 1 < len(self.chrStart) else self.nGenome) - st) // step
            self.chrBin[st // step:st // step + nb] = i
        self.text = chroms

    def prepare(self, P: Params) -> Params:
        return P.derive(self.nGenome, self.gChrBinNbits)


# ---------------------------------------------------------------------------------------------------------------------
# Numeric kernels (numba)
# ---------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _g(G, i):
    if i < 0 or i >= G.shape[0]:
        return 5
    return G[i]


@njit(cache=True)
def _r(R, i):
    if i < 0 or i >= R.shape[0]:
        return 11
    return R[i]


@njit(cache=True)
def extend_align_nb(R, G, rStart, gStart, dR, dG, L, Lprev, nMMprev, nMMmax, pMMmax, extendToEnd):
    """extendAlign.cpp (stitch.cuh:628). Returns (ok, extendL, maxScore, nMatch, nMM)."""
    Score = 0
    nMatch = 0
    nMM = 0
    maxScore = 0
    extendL = 0
    oMatch = 0
    oMM = 0
    if extendToEnd:
        iExt = 0
        while iExt < L:
            iS = dR * iExt
            iG = dG * iExt
            gi = gStart + iG
            if gi == -1 or _g(G, gi) == 5:
                return True, 0, -999999999, 0, nMMmax + 1
            rv = _r(R, rStart + iS)
            if rv == 11:
                break
            gv = _g(G, gi)
            if rv > 3 or gv > 3:
                iExt += 1
                continue
            if gv == rv:
                nMatch += 1
                Score += 1
            else:
                nMM += 1
                Score -= 1
            iExt += 1
        if iExt > 0:
            return True, iExt, Score, nMatch, nMM
        return False, 0, 0, 0, 0
    for i in range(L):
        iS = dR * i
        iG = dG * i
        gi = gStart + iG
        rv = _r(R, rStart + iS)
        gv = _g(G, gi)
        if gi == -1 or gv == 5 or rv == 11:
            break
        if rv > 3 or gv > 3:
            continue
        if gv == rv:
            nMatch += 1
            Score += 1
            if Score > maxScore:
                if nMM + nMMprev <= min(pMMmax * float(Lprev + i + 1), float(nMMmax)):
                    extendL = i + 1
                    maxScore = Score
                    oMatch = nMatch
                    oMM = nMM
        else:
            if nMM + nMMprev >= min(pMMmax * float(Lprev + L), float(nMMmax)):
                break
            nMM += 1
            Score -= 1
    return extendL > 0, extendL, maxScore, oMatch, oMM


def extend_align(R, G, rStart, gStart, dR, dG, L, Lprev, nMMprev, nMMmax, pMMmax, extendToEnd):
    Lc = ((int(L) + (1 << 31)) % (1 << 32)) - (1 << 31)  # the (int) cast of a uint length
    return extend_align_nb(R, G, int(rStart), int(gStart), dR, dG, Lc, int(Lprev), int(nMMprev), int(nMMmax),
                           float(pMMmax), bool(extendToEnd))


@njit(cache=True)
def blocks_overlap_nb(e1, n1, e2, n2):
    """blocksOverlap.cpp (stitch.cuh:930)"""
    i1 = 0
    i2 = 0
    nOv = 0
    while i1 < n1 and i2 < n2:
        rs1 = e1[i1, 0]
        rs2 = e2[i2, 0]
        re1 = e1[i1, 0] + e1[i1, 2]
        re2 = e2[i2, 0] + e2[i2, 2]
        gs1 = e1[i1, 1]
        gs2 = e2[i2, 1]
        if rs1 >= re2:
            i2 += 1
        elif rs2 >= re1:
            i1 += 1
        elif gs1 - rs1 != gs2 - rs2:
            if re1 >= re2:
                i2 += 1
            if re2 >= re1:
                i1 += 1
        else:
            nOv += min(re1, re2) - max(rs1, rs2)
            if re1 >= re2:
                i2 += 1
            if re2 >= re1:
                i1 += 1
    return nOv


@njit(cache=True)
def mate_overlap_ok(exons, canonSJ, nExons, Lread, protrude):
    """stitchWindowAligns.cpp:179-219: the check for mates, for a transcript whose first and last exons are in different mates.
    Returns True if the transcript is kept."""
    if exons[nExons - 1, 1] + exons[nExons - 1, 2] <= exons[0, 1]:  # :180 negative insert size
        return False
    iexM2 = nExons
    for iex in range(nExons - 1):  # :182 first exon of the second mate
        if canonSJ[iex] == -3:
            iexM2 = iex + 1
            break
    if exons[iexM2 - 1, 1] + exons[iexM2 - 1, 2] > exons[iexM2, 1]:  # :189 mates overlap
        if exons[0, 1] > exons[iexM2, 1] + exons[0, 0] + protrude:  # :191 LeftMateStart > RightMateStart + allowance
            return False
        if exons[iexM2 - 1, 1] + exons[iexM2 - 1, 2] > exons[nExons - 1, 1] + Lread - exons[nExons - 1, 0] + protrude:  # :193
            return False
        iex1 = 1
        iex2 = iexM2 + 1
        while iex1 < iexM2:  # :198 first junction that overlaps the 2nd mate
            if exons[iex1, 1] >= exons[iex2 - 1, 1] + exons[iex2 - 1, 2]:
                break
            iex1 += 1
        while iex1 < iexM2 and iex2 < nExons:  # :201 junction consistency
            if canonSJ[iex1 - 1] < 0:
                iex1 += 1
                continue
            if canonSJ[iex2 - 1] < 0:
                iex2 += 1
                continue
            if exons[iex1, 1] != exons[iex2, 1] or (exons[iex1 - 1, 1] + exons[iex1 - 1, 2]) != (exons[iex2 - 1, 1] + exons[iex2 - 1, 2]):
                return False
            iex1 += 1
            iex2 += 1
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------------------------------------------------
class Tr:
    def __init__(self):
        self.exons = np.zeros((MAX_N_EXONS, 5), dtype=np.int64)
        self.shiftSJ = np.zeros((MAX_N_EXONS, 2), dtype=np.int64)
        self.canonSJ = np.zeros(MAX_N_EXONS, dtype=np.int64)
        self.sjAnnot = np.zeros(MAX_N_EXONS, dtype=np.int64)
        self.sjStr = np.zeros(MAX_N_EXONS, dtype=np.int64)
        self.intronMotifs = [0, 0, 0]
        self.sjMotifStrand = 0
        self.sjYes = False
        self.nExons = 0
        self.iFrag = 0
        self.rStart = self.roStart = self.rLength = self.gStart = self.gLength = self.cStart = 0
        self.Chr = self.Str = self.roStr = 0
        self.primaryFlag = False
        self.nMatch = self.nMM = self.mappedLength = self.extendL = 0
        self.maxScore = 0
        self.nGap = self.lGap = self.nDel = self.nIns = self.lDel = self.lIns = 0
        self.nUnique = self.nAnchor = 0

    def copy(self) -> "Tr":
        t = Tr.__new__(Tr)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(t, k, v.copy())
            elif isinstance(v, list):
                setattr(t, k, list(v))
            else:
                setattr(t, k, v)
        return t

    def add(self, e):  # Transcript::add of an extension result
        _ok, extendL, maxScore, nMatch, nMM = e
        self.maxScore += maxScore
        self.nMatch += nMatch
        self.nMM += nMM

    def summary(self) -> dict:
        n = self.nExons
        return dict(
            exons=[[int(x) for x in self.exons[i, :4]] for i in range(n)],
            canonSJ=[int(x) for x in self.canonSJ[:n - 1]],
            maxScore=int(self.maxScore), nMatch=int(self.nMatch), nMM=int(self.nMM), gLength=int(self.gLength),
            rLength=int(self.rLength), iFrag=int(self.iFrag), Str=int(self.Str), Chr=int(self.Chr),
            primary=bool(self.primaryFlag), nGap=int(self.nGap), nDel=int(self.nDel), nIns=int(self.nIns))


# ---------------------------------------------------------------------------------------------------------------------
# Read: oneRead.cpp:40-76
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class PairRead:
    r: List[np.ndarray]  # Read1[0], Read1[1] (complement), Read1[2] (reverse complement)
    Lread: int
    readLength: Tuple[int, int]
    nmates: int


def _comp(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    m = a < 4
    out[m] = 3 - a[m]
    return out


def build_read(m1: str, m2: Optional[str]) -> PairRead:
    a = nucl(m1)
    if m2 is None:  # single end (:60)
        r0 = a
        L = len(a)
        rl = (L, 0)
        nm = 1
    else:  # :40-58: mate1, spacer, reverse complement of mate2
        b = nucl(m2)
        r0 = np.concatenate([a, np.array([MARK_FRAG_SPACER_BASE], dtype=np.int8), _comp(b)[::-1]])
        L = len(a) + len(b) + 1
        rl = (len(a), len(b))
        nm = 2
    r1 = _comp(r0)  # :74 complementSeqNumbers leaves values > 3 unchanged
    r2 = r1[::-1].copy()  # :75
    return PairRead([r0.astype(np.int8), r1.astype(np.int8), r2.astype(np.int8)], L, rl, nm)


def quality_split(r: np.ndarray, L: int, maxNsplit: int, minLsplit: int):
    """SequenceFuns.cpp qualitySplit (stitch.cuh:268). Returns [(start, length, iFrag)], and LgoodMin."""
    out = []
    iR = 0
    LgoodMin = 0
    iFrag = 0
    while iR < L and len(out) < maxNsplit:
        while iR < L and r[iR] > 3:
            if int(r[iR]) == MARK_FRAG_SPACER_BASE:
                iFrag += 1
            iR += 1
        if iR == L:
            break
        iR1 = iR
        while iR < L and r[iR] <= 3:
            iR += 1
        LgoodMin = max(LgoodMin, iR - iR1)
        if iR - iR1 < minLsplit:
            continue
        out.append((iR1, iR - iR1, iFrag))
    return out, LgoodMin


# ---------------------------------------------------------------------------------------------------------------------
# Seeds: a brute-force stand-in for the suffix-array search. mapOneRead.cpp:46-153 and storeAligns are ported; only
# maxMappableLength2strands is replaced by "longest prefix that occurs anywhere, on either strand, plus every locus".
# ---------------------------------------------------------------------------------------------------------------------
def _mmp(genome: Genome, piece: str, maxL: int):
    """Longest L <= maxL such that piece[:L] or its reverse complement occurs on a chromosome; and all loci at that L.
    A locus is (aStr, a1): aStr 0 = piece on the + strand at a1, 1 = reverse complement of the piece on the + strand at a1."""
    best = 0
    for L in range(1, maxL + 1):
        q = piece[:L]
        qr = revcomp_str(q)
        if any((q in t) or (qr in t) for t in genome.text.values()):
            best = L
        else:
            break
    if best == 0:
        return 0, []
    q = piece[:best]
    qr = revcomp_str(q)
    loci = []
    for i, n in enumerate(genome.names):
        t = genome.text[n]
        for s, qq in ((0, q), (1, qr)):
            p = t.find(qq)
            while p >= 0:
                loci.append((s, genome.chrStart[i] + p))
                p = t.find(qq, p + 1)
    return best, loci


class Seeds:
    def __init__(self):
        self.PC: List[list] = []  # [rStart, Length, Dir, Nrep, loci, iFrag]
        self.nA = 0
        self.multNminL = 0
        self.multNmin = 0


def _store_aligns(P: Params, S: Seeds, iDir: int, Shift: int, Nrep: int, L: int, loci, iFrag: int):
    """ReadAlign_storeAligns.cpp (OPTIM_STOREaligns_SIMPLE), stitch.cuh:372"""
    if Nrep > P.seedMultimapNmax:
        if Nrep < S.multNmin or S.multNmin == 0:
            S.multNmin = Nrep
            S.multNminL = L
        return
    S.nA += Nrep
    rStart = Shift if iDir == 0 else Shift + 1 - L
    iP = len(S.PC) - 1
    while iP >= 0:
        if S.PC[iP][0] <= rStart:
            if S.PC[iP][0] == rStart and S.PC[iP][1] < L:
                iP -= 1
                continue
            if S.PC[iP][0] == rStart and S.PC[iP][1] == L:
                return
            break
        iP -= 1
    S.PC.insert(iP + 1, [rStart, L, iDir, Nrep, loci, iFrag])


def find_seeds(genome: Genome, P: Params, rd: PairRead) -> Seeds:
    """ReadAlign::mapOneRead seed loop (mapOneRead.cpp:46-153)."""
    S = Seeds()
    r0 = rd.r[0]
    splits, LgoodMin = quality_split(r0, rd.Lread, P.maxNsplit, P.seedSplitMin)
    seedSearchStartLmax = min(P.seedSearchStartLmax, int(P.seedSearchStartLmaxOverLread * (rd.Lread - 1)))
    text0 = "".join("ACGTN"[min(int(c), 4)] for c in r0)
    for (s0, sl, ifr) in splits:
        Nstart = sl // seedSearchStartLmax + 1 if (P.seedSearchStartLmax > 0 and seedSearchStartLmax < sl) else 1
        Lstart = sl // Nstart
        flagDirMap = True
        for iDir in range(2):
            for istart in range(Nstart):
                if flagDirMap or istart > 0:
                    Lmapped = 0
                    while istart * Lstart + Lmapped + P.seedMapMin < sl:
                        Shift = s0 + istart * Lstart + Lmapped if iDir == 0 else s0 + sl - istart * Lstart - 1 - Lmapped
                        seedLength = sl - Lmapped - istart * Lstart
                        if iDir == 0:
                            piece = text0[Shift:Shift + seedLength]
                        else:  # right to left: the reverse complement of the read ending at Shift is searched
                            piece = revcomp_str(text0[Shift - seedLength + 1:Shift + 1])
                        L, loci = _mmp(genome, piece, seedLength)
                        L = max(L, 1)
                        if iDir == 1:
                            # loci of the reverse-complemented piece, expressed for the forward piece: strand flips
                            loci = [(1 - s, a) for (s, a) in loci]
                        Nrep = len(loci)
                        if Nrep > 0:
                            _store_aligns(P, S, iDir, Shift, Nrep, L, sorted(loci, key=lambda x: (x[1], x[0])), ifr)
                        if iDir == 0 and istart == 0 and Lmapped == 0 and Shift + L == sl:
                            flagDirMap = False
                        Lmapped += L
    return S


# ---------------------------------------------------------------------------------------------------------------------
# Windows: ReadAlign_createExtendWindowsWithAlign.cpp, flanks (stitchPieces:129-170), assignAlignToWindow
# ---------------------------------------------------------------------------------------------------------------------
class Windows:
    def __init__(self, genome: Genome, P: Params):
        self.g = genome
        self.P = P
        self.winBin = [dict(), dict()]
        self.WC: List[list] = []  # [Str, Chr, gStart, gEnd]
        self.WA: List[list] = []
        self.WALrec: List[int] = []
        self.WlastAnchor: List[int] = []
        self.mapMarker = 0

    def get(self, s, b):
        return self.winBin[s].get(b, uintWinBinMax)

    def create_extend(self, a1: int, aStr: int) -> int:
        P, g = self.P, self.g
        aBin = a1 >> P.winBinNbits
        iBinLeft = iBinRight = aBin
        iBin = iWin = iWinRight = -1
        if self.get(aStr, aBin) == uintWinBinMax:
            flagMergeLeft = False
            if aBin > 0:
                lo = aBin - P.winAnchorDistNbins if aBin > P.winAnchorDistNbins else 0
                iBin = aBin - 1
                while iBin >= lo:
                    if self.get(aStr, iBin) < uintWinBinMax:
                        flagMergeLeft = True
                        break
                    if iBin == 0:
                        break
                    iBin -= 1
                flagMergeLeft = flagMergeLeft and (g.chrBin[iBin >> P.winBinChrNbits] == g.chrBin[aBin >> P.winBinChrNbits])
                if flagMergeLeft:
                    iWin = self.get(aStr, iBin)
                    iBinLeft = self.WC[iWin][WC_gStart]
                    for ii in range(iBin + 1, aBin + 1):
                        self.winBin[aStr][ii] = iWin
            flagMergeRight = False
            if aBin + 1 < P.winBinN:
                iBin = aBin + 1
                while iBin < min(aBin + P.winAnchorDistNbins + 1, P.winBinN):
                    if self.get(aStr, iBin) < uintWinBinMax:
                        flagMergeRight = True
                        break
                    iBin += 1
                flagMergeRight = flagMergeRight and (g.chrBin[iBin >> P.winBinChrNbits] == g.chrBin[aBin >> P.winBinChrNbits])
                if flagMergeRight:
                    while self.get(aStr, iBin) == self.get(aStr, iBin + 1):
                        iBin += 1
                    iBinRight = iBin
                    iWinRight = self.get(aStr, iBin)
                    if not flagMergeLeft:
                        iWin = self.get(aStr, iBin)
                    for ii in range(aBin, iBin + 1):
                        self.winBin[aStr][ii] = iWin
            if not flagMergeLeft and not flagMergeRight:
                iWin = len(self.WC)
                self.winBin[aStr][aBin] = iWin
                self.WC.append([aStr, int(g.chrBin[aBin >> P.winBinChrNbits]), aBin, aBin])
                if len(self.WC) >= P.alignWindowsPerReadNmax:
                    del self.WC[P.alignWindowsPerReadNmax - 1:]
                    return 101
            else:
                self.WC[iWin][WC_gStart] = iBinLeft
                self.WC[iWin][WC_gEnd] = iBinRight
                if flagMergeLeft and flagMergeRight:
                    self.WC[iWinRight][WC_gStart] = 1
                    self.WC[iWinRight][WC_gEnd] = 0
        return 0

    def flanks(self):
        P, g = self.P, self.g
        for iWin in range(len(self.WC)):
            wc = self.WC[iWin]
            if wc[WC_gStart] <= wc[WC_gEnd]:
                wb = wc[WC_gStart]
                ii = 0
                while ii < P.winFlankNbins and wb > 0 and g.chrBin[(wb - 1) >> P.winBinChrNbits] == wc[WC_Chr]:
                    wb -= 1
                    self.winBin[wc[WC_Str]][wb] = iWin
                    ii += 1
                wc[WC_gStart] = wb
                wb = wc[WC_gEnd]
                ii = 0
                while ii < P.winFlankNbins and wb + 1 < P.winBinN and g.chrBin[(wb + 1) >> P.winBinChrNbits] == wc[WC_Chr]:
                    wb += 1
                    self.winBin[wc[WC_Str]][wb] = iWin
                    ii += 1
                wc[WC_gEnd] = wb
            self.WA.append([])
            self.WALrec.append(0)
            self.WlastAnchor.append(U64)

    def assign(self, a1, aLength, aStr, aNrep, aFrag, aRstart, aAnchor, sjA):
        """assignAlignToWindow (stitch.cuh:562)"""
        P = self.P
        iW = self.get(aStr, a1 >> P.winBinNbits)
        if iW == uintWinBinMax or (not aAnchor and aLength < self.WALrec[iW]):
            return
        WA = self.WA[iW]
        iA = 0
        while iA < len(WA):
            A = WA[iA]
            if (aFrag == A[WA_iFrag] and A[WA_sjA] == sjA and a1 + A[WA_rStart] == A[WA_gStart] + aRstart
                    and ((aRstart >= A[WA_rStart] and aRstart < A[WA_rStart] + A[WA_Length])
                         or (aRstart + aLength >= A[WA_rStart] and aRstart + aLength < A[WA_rStart] + A[WA_Length]))):
                break
            iA += 1
        if iA < len(WA):  # this piece overlaps an existing one on the same diagonal
            if aLength > WA[iA][WA_Length]:
                iA0 = 0
                while iA0 < len(WA):
                    if iA0 != iA and aRstart < WA[iA0][WA_rStart]:
                        break
                    iA0 += 1
                if iA0 > iA:
                    iA0 -= 1
                if iA0 < iA:
                    for k in range(iA, iA0, -1):
                        WA[k] = list(WA[k - 1])
                elif iA0 > iA:
                    for k in range(iA, iA0):
                        WA[k] = list(WA[k + 1])
                WA[iA0] = [aLength, aRstart, a1, aNrep, int(aAnchor), aFrag, sjA]
            return
        if len(WA) == P.seedPerWindowNmax:
            self.WALrec[iW] = self.rd_Lread + 1
            for A in WA:
                if A[WA_Anchor] != 1:
                    self.WALrec[iW] = min(self.WALrec[iW], A[WA_Length])
            if self.WALrec[iW] == self.rd_Lread + 1:
                self.mapMarker = 999905
                self.WC = []
                return
            if not aAnchor and aLength < self.WALrec[iW]:
                return
            self.WA[iW] = WA = [list(A) for A in WA if A[WA_Anchor] == 1 or A[WA_Length] > self.WALrec[iW]]
        if aAnchor or aLength > self.WALrec[iW]:
            iA = 0
            while iA < len(WA):
                if aRstart < WA[iA][WA_rStart]:
                    break
                iA += 1
            WA.insert(iA, [aLength, aRstart, a1, aNrep, int(aAnchor), aFrag, sjA])
            if aAnchor and self.WlastAnchor[iW] < iA:  # unsigned compare: -1 is the maximum
                self.WlastAnchor[iW] = iA


def build_windows(genome: Genome, P: Params, rd: PairRead, S: Seeds) -> Windows:
    """stitchPieces:71-123 (windows from anchors), :129-170 (flanks), :184-264 (assignment, FASTASSIGN-equivalent)."""
    W = Windows(genome, P)
    W.rd_Lread = rd.Lread
    for pc in S.PC:
        rStart, L, iDir, Nrep, loci, iFrag = pc
        if Nrep <= P.winAnchorMultimapNmax:
            for (aStr, a1) in loci:
                if W.create_extend(a1, aStr) == 101:
                    break
    W.flanks()
    for pc in S.PC:
        rStart, L, iDir, Nrep, loci, iFrag = pc
        aAnchor = Nrep <= P.winAnchorMultimapNmax
        for (aStr, a1) in loci:
            aRstart = rStart if aStr == 0 else rd.Lread - (L + rStart)
            if W.get(aStr, a1 >> P.winBinNbits) != uintWinBinMax:
                W.assign(a1, L, aStr, Nrep, iFrag, aRstart, aAnchor, SJ_NONE)
    return W


# ---------------------------------------------------------------------------------------------------------------------
# stitchAlignToTranscript (stitch.cuh:680-925, STAR stitchAlignToTranscript.cpp); sjdb branch omitted (sjA is always -1)
# ---------------------------------------------------------------------------------------------------------------------
def stitch_align(rAend, gAend, rBstart, gBstart, L, iFragB, sjAB, P: Params, R, G, trA: Tr, nMMmaxTotal):
    nE = trA.nExons
    if nE >= MAX_N_EXONS:
        return -1000010
    Score = 0
    last = nE - 1
    trA.sjAnnot[last] = 0
    trA.sjStr[last] = 0
    if trA.exons[last][EX_iFrag] == iFragB:  # same mate: contiguous / gap / junction / indel
        gBend = gBstart + L - 1
        rBend = rBstart + L - 1
        if rBend <= rAend:
            return -1000001
        if gBend <= gAend:
            return -1000002
        if rBstart <= rAend:
            gBstart += rAend - rBstart + 1
            rBstart = rAend + 1
            L = rBend - rBstart + 1
        Score += rBend - rBstart + 1
        gGap = gBstart - gAend - 1
        rGap = rBstart - rAend - 1
        nMatch = L
        nMM = 0
        Del = Ins = nIns = nDel = 0
        jR = 0
        jCan = 999
        gBstart1 = gBstart - rGap - 1

        def Rv(i):
            return int(_r(R, i))

        def Gv(i):
            return int(_g(G, i))
        if gGap == 0 and rGap == 0:
            pass
        elif gGap > 0 and rGap > 0 and rGap == gGap:
            for ii in range(1, rGap + 1):
                if Gv(gAend + ii) < 4 and Rv(rAend + ii) < 4:
                    if Rv(rAend + ii) == Gv(gAend + ii):
                        Score += 1
                        nMatch += 1
                    else:
                        Score -= 1
                        nMM += 1
        elif gGap > rGap:  # deletion or junction
            nDel = 1
            Del = gGap - rGap
            if Del > P.alignIntronMax and P.alignIntronMax > 0:
                return -1000003
            Score1 = 0
            jR1 = 1
            while True:
                jR1 -= 1
                if Rv(rAend + jR1) != Gv(gBstart1 + jR1) and Gv(gBstart1 + jR1) < 4 and Rv(rAend + jR1) == Gv(gAend + jR1):
                    Score1 -= 1
                if not (Score1 + P.scoreStitchSJshift >= 0 and int(trA.exons[last][EX_L]) + jR1 > 1):
                    break
            maxScore2 = -999999
            Score1 = 0
            jPen = 0
            while True:
                if Rv(rAend + jR1) == Gv(gAend + jR1) and Rv(rAend + jR1) != Gv(gBstart1 + jR1):
                    Score1 += 1
                if Rv(rAend + jR1) != Gv(gAend + jR1) and Rv(rAend + jR1) == Gv(gBstart1 + jR1):
                    Score1 -= 1
                jCan1 = -1
                jPen1 = 0
                Score2 = Score1
                if Del >= P.alignIntronMin:
                    a1_, a2_ = Gv(gAend + jR1 + 1), Gv(gAend + jR1 + 2)
                    b1_, b2_ = Gv(gBstart1 + jR1 - 1), Gv(gBstart1 + jR1)
                    if a1_ == 2 and a2_ == 3 and b1_ == 0 and b2_ == 2:
                        jCan1 = 1
                    elif a1_ == 1 and a2_ == 3 and b1_ == 0 and b2_ == 1:
                        jCan1 = 2
                    elif a1_ == 2 and a2_ == 1 and b1_ == 0 and b2_ == 2:
                        jCan1 = 3
                        jPen1 = P.scoreGapGCAG
                    elif a1_ == 1 and a2_ == 3 and b1_ == 2 and b2_ == 1:
                        jCan1 = 4
                        jPen1 = P.scoreGapGCAG
                    elif a1_ == 0 and a2_ == 3 and b1_ == 0 and b2_ == 1:
                        jCan1 = 5
                        jPen1 = P.scoreGapATAC
                    elif a1_ == 2 and a2_ == 3 and b1_ == 0 and b2_ == 3:
                        jCan1 = 6
                        jPen1 = P.scoreGapATAC
                    else:
                        jCan1 = 0
                        jPen1 = P.scoreGapNoncan
                    Score2 += jPen1
                if maxScore2 < Score2:
                    maxScore2 = Score2
                    jR = jR1
                    jCan = jCan1
                    jPen = jPen1
                jR1 += 1
                if not (jR1 < rBend - rAend):
                    break
            jjL = 0
            jjR = 0
            while (gAend + jR >= jjL and Gv(gAend - jjL + jR) == Gv(gBstart1 - jjL + jR) and Gv(gAend - jjL + jR) < 4
                   and jjL <= MAX_SJ_REPEAT_SEARCH):
                jjL += 1
            nG = G.shape[0]
            while (gAend + jjR + jR + 1 < nG and Gv(gAend + jjR + jR + 1) == Gv(gBstart1 + jjR + jR + 1)
                   and Gv(gAend + jjR + jR + 1) < 4 and jjR <= MAX_SJ_REPEAT_SEARCH):
                jjR += 1
            if jCan <= 0:
                jR -= jjL
                if int(trA.exons[last][EX_L]) + jR < 1:
                    return -1000005
                jjR += jjL
                jjL = 0
            for ii in range(min(1, jR + 1), max(rGap, jR) + 1):
                g1 = (gAend + ii) if ii <= jR else (gBstart1 + ii)
                if Gv(g1) < 4 and Rv(rAend + ii) < 4:
                    if Rv(rAend + ii) == Gv(g1):
                        if 1 <= ii <= rGap:
                            Score += 1
                            nMatch += 1
                    else:
                        Score -= 1
                        nMM += 1
                        if ii < 1 or ii > rGap:
                            Score -= 1
                            nMatch -= 1
            if Del >= P.alignIntronMin:  # no sjdb
                Score += P.scoreGap + jPen
            else:
                Score += Del * P.scoreDelBase + P.scoreDelOpen
                jCan = -1
                trA.sjAnnot[last] = 0
            trA.shiftSJ[last][0] = jjL
            trA.shiftSJ[last][1] = jjR
            trA.canonSJ[last] = jCan
            if trA.sjAnnot[last] == 0:
                trA.sjStr[last] = (2 - jCan % 2) if jCan > 0 else 0
        elif rGap > gGap:  # insertion
            Ins = rGap - gGap
            nIns = 1
            if gGap == 0:
                jR = 0
            elif gGap < 0:
                jR = 0
                Score -= -gGap
            else:
                Score1 = 0
                maxScore1 = 0
                for jR1 in range(1, gGap + 1):
                    if Gv(gAend + jR1) < 4:
                        Score1 += 1 if Rv(rAend + jR1) == Gv(gAend + jR1) else -1
                        Score1 += -1 if Rv(rAend + Ins + jR1) == Gv(gAend + jR1) else 1
                    if Score1 > maxScore1 or (Score1 == maxScore1 and P.flushRight):
                        maxScore1 = Score1
                        jR = jR1
                for ii in range(1, gGap + 1):
                    r1 = rAend + ii + (0 if ii <= jR else Ins)
                    if Gv(gAend + ii) < 4 and Rv(r1) < 4:
                        if Rv(r1) == Gv(gAend + ii):
                            Score += 1
                            nMatch += 1
                        else:
                            Score -= 1
                            nMM += 1
            if P.flushRight:
                while jR < rBend - rAend - Ins:
                    if Rv(rAend + jR + 1) != Gv(gAend + jR + 1) or Gv(gAend + jR + 1) == 4:
                        break
                    jR += 1
                if jR == rBend - rAend - Ins:
                    return -1000009
            Score += Ins * P.scoreInsBase + P.scoreInsOpen
            jCan = -2
        if (trA.nMM + nMM) <= nMMmaxTotal and (jCan < 0 or (jCan < 7 and nMM <= (P.alignSJstitchMismatchNmax[(jCan + 1) // 2] & U64))):  # (uint) cast: -1 = no limit
            trA.nMM += nMM
            trA.nMatch += nMatch
            if Del >= P.alignIntronMin:
                trA.nGap += nDel
                trA.lGap += Del
            else:
                trA.nDel += nDel
                trA.lDel += Del
            if Del == 0 and Ins == 0:
                trA.exons[last][EX_L] += rBend - rAend
            elif Del > 0:
                trA.exons[last][EX_L] += jR
                trA.exons[nE][EX_L] = rBend - rAend - jR
                trA.exons[nE][EX_R] = rAend + jR + 1
                trA.exons[nE][EX_G] = gBstart1 + jR + 1
                trA.nExons += 1
            elif Ins > 0:
                trA.nIns += nIns
                trA.lIns += Ins
                trA.exons[last][EX_L] += jR
                trA.exons[nE][EX_L] = rBend - rAend - jR - Ins
                trA.exons[nE][EX_R] = rAend + jR + Ins + 1
                trA.exons[nE][EX_G] = gAend + 1 + jR
                trA.canonSJ[last] = -2
                trA.sjAnnot[last] = 0
                trA.nExons += 1
        else:
            return -1000007
    elif (gBstart + int(trA.exons[0][EX_R]) + P.alignEndsProtrudeNbasesMax >= int(trA.exons[0][EX_G])
          or int(trA.exons[0][EX_G]) < int(trA.exons[0][EX_R])):  # stitchAlignToTranscript.cpp:352 mates
        if P.alignMatesGapMax > 0 and gBstart > int(trA.exons[last][EX_G]) + int(trA.exons[last][EX_L]) + P.alignMatesGapMax:  # :354
            return -1000004
        Score += L
        e = extend_align(R, G, rAend + 1, gAend + 1, 1, 1, DEF_readSeqLengthMax, trA.nMatch, trA.nMM, nMMmaxTotal,
                         P.outFilterMismatchNoverLmax, P.alignEndsTypeExt[int(trA.exons[last][EX_iFrag])][1])  # :374
        if e[0]:
            trA.add(e)
            Score += e[2]  # maxScore
            trA.exons[last][EX_L] += e[1]
        trA.exons[nE][EX_R] = rBstart
        trA.exons[nE][EX_G] = gBstart
        trA.exons[nE][EX_L] = L
        trA.nMatch += L
        extlen = DEF_readSeqLengthMax if P.alignEndsTypeExt[iFragB][1] else ((gBstart - int(trA.exons[0][EX_G]) + int(trA.exons[0][EX_R])) & U64)  # :390
        e = extend_align(R, G, rBstart - 1, gBstart - 1, -1, -1, extlen, trA.nMatch, trA.nMM, nMMmaxTotal,
                         P.outFilterMismatchNoverLmax, P.alignEndsTypeExt[iFragB][1])  # :391
        if e[0]:
            trA.add(e)
            Score += e[2]
            trA.exons[nE][EX_R] -= e[1]
            trA.exons[nE][EX_G] -= e[1]
            trA.exons[nE][EX_L] += e[1]
        trA.canonSJ[last] = -3
        trA.sjAnnot[last] = 0
        trA.nExons += 1
    else:
        return -1000008
    trA.exons[trA.nExons - 1][EX_iFrag] = iFragB
    trA.exons[trA.nExons - 1][EX_sjA] = sjAB if sjAB < (1 << 63) else -1
    return Score


# ---------------------------------------------------------------------------------------------------------------------
# stitchWindowAligns (finalization with the mate rules, then the recursion)
# ---------------------------------------------------------------------------------------------------------------------
def genomic_length_score(P: Params, x: int) -> int:
    return int(math.ceil(math.log2(float(x)) * P.scoreGenomicLengthLog2scale - 0.5))


class _Win:
    """State of one window's recursion: its recorded transcripts (STAR's wTr / nWinTr) and the read."""

    def __init__(self, P, genome, rd, R, nMMmaxTotal, maxScoreMate):
        self.P, self.g, self.rd, self.R = P, genome, rd, R
        self.nMMmaxTotal = nMMmaxTotal
        self.maxScoreMate = maxScoreMate
        self.tr: List[Tr] = []
        self.G = genome.G


def finalize(Wn: _Win, Score: int, tR2: int, tG2: int, trIn: Tr):
    P, g, rd, R, G = Wn.P, Wn.g, Wn.rd, Wn.R, Wn.G
    Lread = rd.Lread
    trA = trIn.copy()
    ex = trA.exons
    vOrder = (0, 1) if trA.roStr == 0 else (1, 0)  # EXTEND_ORDER 1
    for o in vOrder:
        if o == 0:  # extend at start (:39)
            if trA.rStart > 0:
                imate = int(ex[0][EX_iFrag])
                e = extend_align(R, G, trA.rStart - 1, trA.gStart - 1, -1, -1, trA.rStart, tR2 - trA.rStart + 1, trA.nMM,
                                 Wn.nMMmaxTotal, P.outFilterMismatchNoverLmax, P.alignEndsTypeExt[imate][int(trA.Str != imate)])
                if e[0]:
                    trA.add(e)
                    Score += e[2]
                    trA.rStart -= e[1]
                    trA.gStart -= e[1]
                    ex[0][EX_R] = trA.rStart
                    ex[0][EX_G] = trA.gStart
                    ex[0][EX_L] += e[1]
        else:  # extend at end (:60)
            if tR2 < Lread:
                imate = int(ex[trA.nExons - 1][EX_iFrag])
                e = extend_align(R, G, tR2 + 1, tG2 + 1, 1, 1, Lread - tR2 - 1, tR2 - trA.rStart + 1, trA.nMM,
                                 Wn.nMMmaxTotal, P.outFilterMismatchNoverLmax, P.alignEndsTypeExt[imate][int(imate == trA.Str)])
                if e[0]:
                    trA.add(e)
                    Score += e[2]
                    tR2 += e[1]
                    tG2 += e[1]
                    ex[trA.nExons - 1][EX_L] += e[1]
    n = trA.nExons
    if (not P.alignSoftClipAtReferenceEnds) and (
            (ex[n - 1][EX_G] + Lread - ex[n - 1][EX_R]) > (g.chrStart[trA.Chr] + g.chrLength[trA.Chr])
            or ex[0][EX_G] < (g.chrStart[trA.Chr] + ex[0][EX_R])):  # :83
        return
    trA.rLength = int(ex[:n, EX_L].sum())
    trA.gLength = tG2 + 1 - trA.gStart
    for isj in range(n - 1):  # :97 exon lengths around junctions (no sjdb)
        if trA.canonSJ[isj] >= 0:
            if trA.sjAnnot[isj] == 1:
                pass  # not reachable: sjA is always -1
            else:
                if ex[isj][EX_L] < P.alignSJoverhangMin + trA.shiftSJ[isj][0] or ex[isj + 1][EX_L] < P.alignSJoverhangMin + trA.shiftSJ[isj][1]:
                    return
    if n > 1 and trA.sjAnnot[n - 2] == 1 and ex[n - 1][EX_L] < P.alignSJDBoverhangMin:
        return
    sjN = 0
    trA.intronMotifs = [0, 0, 0]
    trA.sjYes = False
    for iex in range(n - 1):
        if trA.canonSJ[iex] >= 0:
            sjN += 1
            trA.intronMotifs[int(trA.sjStr[iex])] += 1
            trA.sjYes = True
    im = trA.intronMotifs
    trA.sjMotifStrand = 1 if (im[1] > 0 and im[2] == 0) else 2 if (im[1] == 0 and im[2] > 0) else 0
    if im[1] > 0 and im[2] > 0 and P.removeInconsistentStrands:
        return
    if sjN > 0 and trA.sjMotifStrand == 0 and P.outSAMstrandFieldType == 1:
        return
    if P.outFilterIntronMotifs == "RemoveNoncanonical":
        if any(trA.canonSJ[i] == 0 for i in range(n - 1)):
            return
    elif P.outFilterIntronMotifs == "RemoveNoncanonicalUnannotated":
        if any(trA.canonSJ[i] == 0 and trA.sjAnnot[i] == 0 for i in range(n - 1)):
            return
    nsj = 0  # :154 mapped length of each mate (readLength[] of the mate of the exon: the paired-end change vs single-end)
    exl = 0
    for iex in range(n):
        exl += int(ex[iex][EX_L])
        if iex == n - 1 or trA.canonSJ[iex] == -3:
            if nsj > 0 and (exl < P.alignSplicedMateMapLmin
                            or exl < int(P.alignSplicedMateMapLminOverLmate * rd.readLength[int(ex[iex][EX_iFrag])])):
                return
            exl = 0
            nsj = 0
        elif trA.canonSJ[iex] >= 0:
            nsj += 1
    if ex[0][EX_iFrag] != ex[n - 1][EX_iFrag]:  # :179 mates
        if not mate_overlap_ok(ex, trA.canonSJ, n, Lread, P.alignEndsProtrudeNbasesMax):
            return
    if P.scoreGenomicLengthLog2scale != 0:  # :221
        Score += genomic_length_score(P, int(ex[n - 1][EX_G] + ex[n - 1][EX_L] - ex[0][EX_G]))
        Score = max(0, Score)
    trA.roStart = trA.rStart if trA.roStr == 0 else Lread - trA.rStart - trA.rLength
    trA.maxScore = Score
    if ex[0][EX_iFrag] == ex[n - 1][EX_iFrag]:  # :232
        trA.iFrag = int(ex[0][EX_iFrag])
        Wn.maxScoreMate[trA.iFrag] = max(Wn.maxScoreMate[trA.iFrag], Score)
    else:
        trA.iFrag = -1
    trA.maxScore = Score
    wTr = Wn.tr
    best0 = wTr[0].maxScore if wTr else 0  # wTr[0] is the initial (score 0) transcript while the window is empty
    if (Score + P.outFilterMultimapScoreRange >= best0
            or (trA.iFrag >= 0 and Score + P.outFilterMultimapScoreRange >= Wn.maxScoreMate[trA.iFrag])):  # :245
        trA.mappedLength = int(ex[:n, EX_L].sum())
        iTr = 0
        while iTr < len(wTr):  # :267 duplicates and subsets
            old = wTr[iTr]
            nOv = int(blocks_overlap_nb(trA.exons, n, old.exons, old.nExons))
            uNew = trA.mappedLength - nOv
            uOld = old.mappedLength - nOv
            if uNew == 0 and Score < old.maxScore:
                break
            elif uOld == 0:
                del wTr[iTr]
            elif uOld > 0 and (uNew > 0 or Score >= old.maxScore):
                iTr += 1
        if iTr == len(wTr):  # :287 insert
            pos = 0
            while pos < len(wTr):
                if Score > wTr[pos].maxScore or (Score == wTr[pos].maxScore and trA.gLength < wTr[pos].gLength):
                    break
                pos += 1
            if len(wTr) < P.alignTranscriptsPerWindowNmax:
                wTr.insert(pos, trA)
            elif pos < len(wTr):
                wTr.insert(pos, trA)
                wTr.pop()


def stitch_window(Wn: _Win, WA: list, iA: int, nA: int, Score: int, tR2: int, tG2: int, trA: Tr):
    """stitchWindowAligns.cpp recursion: include this align, then exclude it"""
    if iA >= nA and tR2 == 0:
        return
    if iA >= nA:
        finalize(Wn, Score, tR2, tG2, trA)
        return
    a = WA[iA]
    dScore = 0
    trAi = trA.copy()
    if trA.nExons > 0:
        dScore = stitch_align(tR2, tG2, a[WA_rStart], a[WA_gStart], a[WA_Length], a[WA_iFrag], a[WA_sjA], Wn.P, Wn.R, Wn.G,
                              trAi, Wn.nMMmaxTotal)
    else:
        trAi.exons[0][EX_R] = trAi.rStart = a[WA_rStart]
        trAi.exons[0][EX_G] = trAi.gStart = a[WA_gStart]
        trAi.exons[0][EX_L] = a[WA_Length]
        trAi.exons[0][EX_iFrag] = a[WA_iFrag]
        trAi.exons[0][EX_sjA] = -1
        trAi.nExons = 1
        dScore = a[WA_Length]
        trAi.nMatch = a[WA_Length]
    if dScore > -1000000:
        if a[WA_Nrep] == 1:
            trAi.nUnique += 1
        if a[WA_Anchor] > 0:
            trAi.nAnchor += 1
        stitch_window(Wn, WA, iA + 1, nA, Score + dScore, a[WA_rStart] + a[WA_Length] - 1, a[WA_gStart] + a[WA_Length] - 1, trAi)
    if a[WA_Anchor] != 2 or trA.nAnchor > 0:
        stitch_window(Wn, WA, iA + 1, nA, Score, tR2, tG2, trA)


# ---------------------------------------------------------------------------------------------------------------------
# Whole read: stitchPieces tail, multMapSelect, mappedFilter
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    unmapType: int  # -1 mapped, 0 no windows, 1 too short, 2 too many mismatches, 3 too many loci
    nW: int
    transcripts: List[Tr] = field(default_factory=list)  # trMult
    primary: int = -1
    windows: List[dict] = field(default_factory=list)  # window records (Str, Chr, WA, WlastAnchor) for replay

    def summaries(self):
        return [t.summary() for t in self.transcripts]


def map_windows(genome: Genome, P: Params, rd: PairRead, WC, WAs, WlastAnchor) -> Result:
    """stitchPieces:351-445 + multMapSelect + mappedFilter, from the windows (as the kernel dumps them)."""
    nMMmaxTotal = min(P.outFilterMismatchNmax, int(P.outFilterMismatchNoverReadLmax * (rd.readLength[0] + rd.readLength[1])))
    trAll: List[List[Tr]] = []
    trBest = Tr()  # trInit: maxScore 0, gLength 0
    maxScoreMate = [0, 0]
    for iW in range(len(WC)):
        WA = [list(x) for x in WAs[iW]]
        if len(WA) == 0:
            continue
        la = WlastAnchor[iW]
        if la < len(WA):
            WA[la][WA_Anchor] = 2
        trA = Tr()
        trA.Chr = WC[iW][WC_Chr]
        trA.Str = WC[iW][WC_Str]
        trA.roStr = trA.Str
        Wn = _Win(P, genome, rd, rd.r[0 if trA.roStr == 0 else 2], nMMmaxTotal, maxScoreMate)
        stitch_window(Wn, WA, 0, len(WA), 0, 0, 0, trA)
        if not Wn.tr:
            continue
        for t in Wn.tr:
            t.Chr, t.Str, t.roStr = trA.Chr, trA.Str, trA.roStr
        trAll.append(Wn.tr)
        top = Wn.tr[0]
        if top.maxScore > trBest.maxScore or (top.maxScore == trBest.maxScore and top.gLength < trBest.gLength):
            trBest = top
    nW = len(trAll)
    res = Result(unmapType=0, nW=nW)
    if trBest.maxScore == 0:  # :440
        res.nW = 0
        return res
    maxScore = max(w[0].maxScore for w in trAll)  # multMapSelect
    mult: List[Tr] = []
    for w in trAll:
        for t in w:
            if t.maxScore + P.outFilterMultimapScoreRange >= maxScore:
                mult.append(t)
    res.transcripts = mult
    nTr = len(mult)
    if nTr <= P.outFilterMultimapNmax and nTr > 0:
        for t in mult:
            t.roStart = t.rStart if t.roStr == 0 else rd.Lread - t.rStart - t.rLength
            t.cStart = t.gStart - genome.chrStart[t.Chr]
        if nTr == 1:
            mult[0].primaryFlag = True
            res.primary = 0
        else:  # outSAMprimaryFlag OneBestScore, outMultimapperOrder Old_2.4: trBest
            trBest.primaryFlag = True
            res.primary = next(i for i, t in enumerate(mult) if t is trBest)
    # mappedFilter
    Lread = rd.Lread
    b = trBest
    um = -1
    if (b.maxScore < P.outFilterScoreMin or b.maxScore < int(P.outFilterScoreMinOverLread * (Lread - 1))
            or b.nMatch < P.outFilterMatchNmin or b.nMatch < int(P.outFilterMatchNminOverLread * (Lread - 1))):
        um = 1
    elif b.nMM > nMMmaxTotal or (b.nMM / b.rLength if b.rLength else 0.0) > P.outFilterMismatchNoverLmax:
        um = 2
    elif nTr > P.outFilterMultimapNmax:
        um = 3
    res.unmapType = um
    return res


def map_pair(genome: Genome, P: Params, m1: str, m2: Optional[str], seeds: Optional[Seeds] = None) -> Result:
    """One read pair (or single-end read when m2 is None), from the sequences to STAR's final transcripts."""
    rd = build_read(m1, m2)
    S = seeds if seeds is not None else find_seeds(genome, P, rd)
    if rd.Lread < P.outFilterMatchNmin:
        return Result(unmapType=0, nW=0)
    if not quality_split(rd.r[0], rd.Lread, P.maxNsplit, P.seedSplitMin)[0] or S.nA == 0:
        return Result(unmapType=0, nW=0)
    W = build_windows(genome, P, rd, S)
    if not W.WC:
        return Result(unmapType=0, nW=0)
    res = map_windows(genome, P, rd, W.WC, W.WA, W.WlastAnchor)
    res.windows = [dict(Str=W.WC[i][WC_Str], Chr=W.WC[i][WC_Chr], WA=W.WA[i], WlastAnchor=int(W.WlastAnchor[i]))
                   for i in range(len(W.WC))]
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------------------------------------------------
def mates_of(t: Tr):
    """Split a transcript into its mates at the -3 junctions (outputTranscriptSAM.cpp:61-66). Each mate is
    (iFrag, [(g, len), ...] blocks, [canonSJ between blocks])."""
    out = []
    cur = []
    sj = []
    n = t.nExons
    for i in range(n):
        cur.append((int(t.exons[i][EX_G]), int(t.exons[i][EX_L])))
        if i == n - 1 or t.canonSJ[i] == -3:
            out.append((int(t.exons[i][EX_iFrag]), cur, sj))
            cur, sj = [], []
        else:
            sj.append(int(t.canonSJ[i]))
    return out


def proper_pair(P: Params, t: Tr, Lread: int) -> bool:
    """SAM flag 0x2 (outputTranscriptSAM.cpp:79-81) for a transcript with two mates"""
    iExMate = next((i for i in range(t.nExons - 1) if t.canonSJ[i] == -3), t.nExons - 1)
    if iExMate == t.nExons - 1:
        return False
    e = t.exons
    return bool(P.alignEndsProtrudeConcordantPair or (
        e[0][EX_G] <= e[iExMate + 1][EX_G] + e[0][EX_R]
        and e[iExMate][EX_G] + e[iExMate][EX_L] <= e[t.nExons - 1][EX_G] + Lread - e[t.nExons - 1][EX_R]))
