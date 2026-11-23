"""Ranked protocol detection from the head reads of a run.

Every read slot of the head (R1, R2, index reads and any technical read the archive keeps) is scanned at every offset and on
both strands against every known barcode list, and checked for linker, TSO, poly(T), Tn5 and capture motifs, length structure
and bead-barcode multiplicity.  The result is a ranked list of candidate protocols, each with a score in [0, 1], the evidence
behind it and whether the campaign engine can run it as is (``engine_tag`` for ``singlet download --protocol``).  Nothing here
forces one answer: ``decide`` says whether the top candidate is clear, and when it is not it names the candidates a trial
mapping should referee (``trial_map.py``).

Inputs are plain sequences, so the same code serves the pre-flight (head of the .sra through the identify dump, or ENA FASTQ
ranges), the post-run diagnosis and the backfill.  Command line (inside a Slurm job, never on the login node):

    python protocol_detect.py --wl-dir DIR --seg SEG1.fq [--seg SEG2.fq ...] [--stated 10x_3p_v3] [--json OUT]

Families follow bench/protocol-detect/REGISTRATION.md.  Thresholds sit in ``T`` with the reason for each.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------- barcode lists and motifs
# (name, file, word length, kind).  kind "10x" lists name a chemistry; the others are protocol lists of short words.
LISTS = [
    ("v1", "737K-april-2014.txt", 14, "10x"),
    ("v2", "737K-august-2016.txt", 16, "10x"),
    ("v3", "3M-february-2018.txt", 16, "10x"),
    ("v4_3p", "3M-3pgex-may-2023.txt", 16, "10x"),
    ("v4_5p", "3M-5pgex-jan-2023.txt", 16, "10x"),
    ("arc_gex", "gex_737K-arc-v1.txt", 16, "10x"),
    ("visium", "visium-v1.txt", 16, "10x"),
    ("atac", "737K-cratac-v1.txt.gz", 16, "atac"),
    ("arc_atac", "737K-arc-v1-atac-gex.tsv.gz", 16, "atac"),
    ("bd_cls1", "bd_cls1.txt", 9, "short"),
    ("bd_cls2", "bd_cls2.txt", 9, "short"),
    ("indrop_bc1", "indrop_bc1.txt", 8, "short"),
    ("splitseq_rd1", "splitseq_bc_rd1.txt", 8, "short"),
    ("splitseq_rd2", "splitseq_bc_rd2.txt", 8, "short"),
    ("microwell_bc1", "microwell_bc1_bare.txt", 6, "short"),
    ("quartzseq2", "quartzseq2_bc.txt", 15, "short"),
    ("scirna3_rt", "scirna3_rt_bc.txt", 10, "short"),
]
TENX = [n for n, _, _, k in LISTS if k == "10x"]

MOTIFS = {
    "polyT": "TTTTTTTTTTTT",
    "polyA": "AAAAAAAAAAAA",
    "tso": "TTTCTTATATGGG",                 # 10x 5' template-switch oligo tail, right after barcode+UMI in R1
    "tso_rc": "CCCATATAAGAAA",              # the same read through from the cDNA read of a short 5' insert
    "polyA_rt": "AAAAAAAAAAAAAAA",          # poly(A) read-through in the cDNA read of a 3' kit (sense reads reach the tail)
    "tn5_me": "CTGTCTCTTATACACATCT",        # Nextera/Tn5 mosaic end read-through
    "tn5_me_rc": "AGATGTGTATAAGAGACAG",
    "bd_linker1": "ACTGGCCTGCGA",
    "bd_linker2": "GGTAGCGGTGACA",
    "indrop_w1": "GAGTGATTGCTTGTGACGCCTT",
    "split_linker1": "GTGGCCGATGTTTCGCATCGGCGTACGACT",
    "split_linker2": "ATCCACGTGCTTGAGAGGCCAGAGCATTCG",
    "microwell_l1": "CGACTCACTACAGGG",
    "microwell_l2": "TCGGTGACACGATCG",
    "smart3_tag": "ATTGCGCAATG",
    "totalseq_cs1": "GCTTTAAGGCCGGTCCTAGCAA",
    "totalseq_cs2": "GCTCACCTATTAGCGGCTAAGG",
    "sgrna_scaffold": "GTTTAAGAGCTATGCTGG",
}


class T:
    """Thresholds, each with the measurement or argument behind it."""
    # A 10x list is present when at least this share of reads carries an exact list word at one offset and strand.  Random
    # 16-mers hit a 3.7 M list once per million; the 14-mer v1 list (737K words) 4 in 10,000 per position, so over ~300
    # offset-strand cells the noise maximum stays under 1 percent.  Real libraries hit 50 to 97 percent (exact match).
    wl_present = 0.05
    # Two 10x lists within this ratio of each other do not settle the version on the list alone (v2 data hits v3 at ~7 percent,
    # v3 data hits 737K at ~20 percent, so a real library sits far above it).
    wl_ratio = 0.80
    # A motif decides 3'/5' when it marks this share of barcode reads at its expected position (poly(T) after barcode+UMI on a
    # 3' kit; TSO on a 5' kit), and the other motif marks under a third of that.
    motif_call = 0.25
    # Short-word protocol lists (BD, inDrop, SPLiT, microwell): present at this share at one offset (96-384 words of 6-9 bases
    # give 0.2 to 1.5 percent by chance per cell; the noise maximum over all cells stays under 5 percent).
    short_present = 0.25
    linker_present = 0.25
    # Bead-barcode structure: the first 12 bases of the barcode read repeat at least 5 times in this share of a 50k head.
    prefix_mult = 0.50
    # A slot is a barcode read when it is at most this long, or when poly(T) starts at a fixed position in it.
    bc_slot_max = 40
    # Feature barcode: top-100 15-mers (at offset 0 or 10) of the cDNA slot hold this share, or a capture constant this share.
    feature_top100 = 0.85      # tag libraries measured 97-98%; a Visium cDNA read reached 69% (development half)
    feature_const = 0.30
    # TotalSeq-A tag read: the 15-base tag is followed by poly(A), so poly(A) starts within the first `feature_polya_pos` bases
    # in `feature_polya_share` of reads, with top-100 15-mers at least `feature_top100_polya` (an ADT library measured 81% with
    # poly(A) at base 7 in 48%; the Visium cDNA read at 75% had poly(A) in 6%, development half).
    feature_top100_polya = 0.60
    feature_polya_pos = 20
    feature_polya_share = 0.30
    # A list hit away from the engine layout (offset over 8 or reverse strand) counts only at this rate: cDNA of a read with no
    # barcode reached 6-14% on the large v4 lists at offsets 52-64 (development half), while real off-layout libraries (long
    # reads with an adapter, offset 4) sit at offset <= 8.
    offlayout_min = 0.30
    # MARS-seq-like: a diverse barcode/UMI read of 12-20 bases (first 12 bases repeat >= 5x in under this share) next to a
    # cDNA read whose first 12 bases repeat (pool barcode) in at least prefix_mult.
    short_diverse_max = 0.30
    smart3_tag = 0.05
    # cDNA-read read-through decides 3'/5' when one motif marks at least rt_call of cDNA reads and rt_ratio times the other.
    # Set on the development half (see docs/PROTOCOL_DETECTION_2026-11.md).
    rt_call = 0.01
    rt_ratio = 3.0
    # Decision: the top candidate is clear when its score is at least `clear_score` and leads the next candidate of another
    # engine tag by `clear_margin`.
    clear_score = 0.60
    clear_margin = 0.20


# ----------------------------------------------------------------------------- encoding
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate(b"ACGT"):
    _LUT[_c] = _i
    _LUT[_c + 32] = _i
_cache: dict = {}


def _codes(win: np.ndarray) -> np.ndarray:
    code = np.zeros(win.shape[0], dtype=np.uint32)
    for j in range(win.shape[1]):
        code = (code << np.uint32(2)) | win[:, j].astype(np.uint32)
    return code


def _read_words(path: str, w: int) -> list:
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt") as fh:
        for line in fh:
            t = line.split("\t")[0].strip().upper()
            if len(t) >= w:
                out.append(t[:w])
    return out


def load_lists(wl_dir: str) -> dict:
    """{name: (sorted uint32 codes, word length, kind)} for every list file found in wl_dir (missing files are skipped)."""
    key = os.path.abspath(wl_dir)
    if key in _cache:
        return _cache[key]
    out = {}
    for name, f, w, kind in LISTS:
        p = os.path.join(wl_dir, f)
        if not os.path.isfile(p):
            continue
        words = _read_words(p, w)
        if not words:
            continue
        arr = _LUT[np.frombuffer("".join(words).encode(), dtype=np.uint8)].reshape(len(words), w)
        arr = arr[(arr < 4).all(axis=1)]
        out[name] = (np.unique(_codes(arr)), w, kind)
    _cache[key] = out
    return out


def _rolling(b: np.ndarray, w: int):
    """(codes, valid): uint32 codes of every w-mer of every row (n x L-w+1) and whether the window holds no N."""
    n, L = b.shape
    k = L - w + 1
    bad = (b == 4).astype(np.int32)
    cs = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(bad, axis=1)], axis=1)
    valid = (cs[:, w:w + k] - cs[:, 0:k]) == 0
    v = np.where(b < 4, b, 0).astype(np.uint32)
    code = np.zeros((n, k), dtype=np.uint32)
    for j in range(w):
        code = (code << np.uint32(2)) | v[:, j:j + k]
    return code, valid


def _matrix(seqs: list, L: int) -> np.ndarray:
    buf = b"".join(s.encode("ascii", "ignore")[:L].ljust(L, b"N") for s in seqs)
    return _LUT[np.frombuffer(buf, dtype=np.uint8)].reshape(len(seqs), L)


# ----------------------------------------------------------------------------- per-slot evidence
@dataclass
class Slot:
    index: int                  # 1-based slot number in spot order
    n: int                      # reads scanned
    len_mode: int
    len_mode_share: float
    lists: dict = field(default_factory=dict)      # name -> {rate, offset, strand}
    motifs: dict = field(default_factory=dict)     # name -> {share, pos (mode), pos_share}
    polyT_pos: int | None = None                    # modal start of the first T12 run, when it is fixed
    polyT_pos_share: float = 0.0
    prefix12_mult: float = 0.0
    top100_15: float = 0.0
    n_share: float = 0.0
    t_frac: list = field(default_factory=list)      # share of reads with T at each of the first 60 positions

    def to_dict(self):
        return self.__dict__


def scan_slot(seqs: list, idx: int, lists: dict, max_reads: int = 50000, list_reads: int = 20000) -> Slot:
    seqs = [s for s in seqs[:max_reads]]
    n = len(seqs)
    lens = [len(s) for s in seqs]
    lc = Counter(lens)
    mode, mc = lc.most_common(1)[0] if lc else (0, 0)
    sl = Slot(index=idx, n=n, len_mode=mode, len_mode_share=mc / n if n else 0.0)
    if n == 0 or mode == 0:
        return sl
    L = min(max(lens), 300)
    b = _matrix(seqs, L)
    rc = np.where(b < 4, 3 - b, 4).astype(np.uint8)[:, ::-1]
    sl.n_share = float((b == 4).sum()) / max(1, sum(min(x, L) for x in lens))
    # barcode lists: every offset, both strands, on the first `list_reads` reads (rolling 2-bit codes per word length)
    m = min(n, list_reads)
    rolled = {}
    for w in sorted({w for _, w, _ in lists.values()}):
        if L < w:
            continue
        rolled[w] = (_rolling(b[:m], w), _rolling(rc[:m], w))
    for name, (codes, w, kind) in lists.items():
        best = {"rate": 0.0, "offset": None, "strand": None}
        if w not in rolled:
            sl.lists[name] = best
            continue
        for strand, (cm, ok) in zip("+-", rolled[w]):
            pos = np.searchsorted(codes, cm)
            pos[pos >= len(codes)] = len(codes) - 1
            hit = (codes[pos] == cm) & ok
            rate = hit.sum(axis=0) / m
            j = int(np.argmax(rate))
            if rate[j] > best["rate"]:
                off = j if strand == "+" else L - w - j
                best = {"rate": round(float(rate[j]), 4), "offset": off, "strand": strand}
        sl.lists[name] = best
    # motifs: share of reads containing it, modal position
    for name, m in MOTIFS.items():
        pos = [s.find(m) for s in seqs]
        hit = [p for p in pos if p >= 0]
        share = len(hit) / n
        pm, pc = Counter(hit).most_common(1)[0] if hit else (None, 0)
        sl.motifs[name] = {"share": round(share, 4), "pos": pm, "pos_share": round(pc / n, 4)}
    tb = b[:, :min(L, 60)]
    sl.t_frac = [round(float(x), 3) for x in (tb == 3).mean(axis=0)]
    pt = [s.find("TTTTTTTTTT") for s in seqs]
    pth = [p for p in pt if p >= 0]
    if pth:
        pm, pc = Counter(pth).most_common(1)[0]
        sl.polyT_pos, sl.polyT_pos_share = pm, round(pc / n, 4)
    if mode >= 12:
        pre = Counter(s[:12] for s in seqs if len(s) >= 12)
        sl.prefix12_mult = round(sum(c for c in pre.values() if c >= 5) / n, 4)
    if mode >= 25:
        best = 0.0
        for o in (0, 10):
            k = Counter(s[o:o + 15] for s in seqs if len(s) >= o + 15)
            best = max(best, sum(c for _, c in k.most_common(100)) / n)
        sl.top100_15 = round(best, 4)
    return sl


# ----------------------------------------------------------------------------- candidates
@dataclass
class Candidate:
    family: str
    protocol: str            # protocol tag (engine tag for 10x, protocol name otherwise)
    score: float
    engine_tag: str | None   # what `singlet download --protocol` takes, or None when the campaign cannot run it
    evidence: list = field(default_factory=list)
    layout_ok: bool = True   # barcode where the engine reads it (offset 0, forward, in the first or second slot)

    def to_dict(self):
        return self.__dict__


ENGINE_3P = {"v1": "10x-3p-v1", "v2": "10x-3p-v2", "v3": "10x-3p-v3", "v4_3p": "10x-3p-v4"}
ENGINE_5P = {"v2": "10x-5p-v2", "v3": "10x-5p-v3", "v4_5p": "10x-5p-v4"}
UMI = {"v1": 10, "v2": 10, "v3": 12, "v4_3p": 12, "v4_5p": 12, "arc_gex": 12, "visium": 12}


def _cdna_slot(slots: list, bc: Slot) -> Slot | None:
    """The longest slot other than the barcode slot and the index-length slots."""
    others = [s for s in slots if s is not bc and s.len_mode > 20]
    return max(others, key=lambda s: s.len_mode) if others else None


def rank(slots: list) -> list:
    """Ranked candidates (best first) from scanned slots.  No slots: no candidate."""
    C: list = []
    if not slots:
        return C
    # best 10x list over all slots
    hits = []
    for s in slots:
        for name in TENX:
            h = s.lists.get(name)
            if h and h["rate"] > 0:
                if (h["offset"] > 8 or h["strand"] != "+") and h["rate"] < T.offlayout_min:
                    continue
                hits.append((h["rate"], name, s, h))
    hits.sort(key=lambda x: -x[0])
    best10 = hits[0] if hits else None
    atac_hits = sorted(((s.lists[n]["rate"], n, s) for s in slots for n in ("atac", "arc_atac") if n in s.lists), key=lambda x: -x[0])
    feat = None

    if best10 and best10[0] >= T.wl_present:
        top_rate = best10[0]
        # versions within wl_ratio of the best list stay candidates; scores scale with the hit rate
        per_list = {}
        for r, name, s, h in hits:
            if name not in per_list:
                per_list[name] = (r, s, h)
        for name, (r, s, h) in per_list.items():
            if r < T.wl_present or r < T.wl_ratio * top_rate * 0.5:
                continue
            rel = r / top_rate
            # the engine finds the barcode read among the segments itself (a 4-slot I1/I2/R1/R2 deposit passed the campaign)
            layout_ok = h["offset"] == 0 and h["strand"] == "+"
            base_ev = f"{name} list {100 * r:.1f}% at slot {s.index} offset {h['offset']} strand {h['strand']} (slot length {s.len_mode})"
            umi = UMI.get(name, 12)
            after = (h["offset"] or 0) + (14 if name == "v1" else 16) + umi
            pt = s.polyT_pos_share if (s.polyT_pos is not None and abs(s.polyT_pos - after) <= 3) else 0.0
            tf = s.t_frac[after:after + 6]
            if len(tf) >= 4:                       # short read: T at every one of the first bases after barcode+UMI (random: 0.25;
                pt = max(pt, min(1.0, max(0.0, min(tf) - 0.25) / 0.6))   # a TSO, TTTCTT..., fails at its C)
            tso = s.motifs["tso"]["pos_share"] if (s.motifs["tso"]["pos"] is not None and abs(s.motifs["tso"]["pos"] - after) <= 3) else 0.0
            cd = _cdna_slot(slots, s)
            # feature barcode / CRISPR: the cDNA slot is a tag read
            if cd is not None:
                const = max(cd.motifs["totalseq_cs1"]["share"], cd.motifs["totalseq_cs2"]["share"], cd.motifs["sgrna_scaffold"]["share"])
                pa = cd.motifs["polyA"]
                early_pa = pa["pos"] is not None and pa["pos"] <= T.feature_polya_pos and pa["pos_share"] >= T.feature_polya_share
                if cd.top100_15 >= T.feature_top100 or const >= T.feature_const or (cd.top100_15 >= T.feature_top100_polya and early_pa):
                    feat = Candidate("feature_barcode", "10x-feature-barcode", round(rel * min(1.0, max(cd.top100_15, const) + 0.3), 3),
                                     None, [base_ev, f"cDNA slot {cd.index}: top-100 15-mers {100 * cd.top100_15:.0f}%, capture/scaffold "
                                                     f"constant {100 * const:.0f}% (tag library)"], layout_ok)
            if name == "visium":
                probe = cd is not None and cd.top100_15 >= 0.30 and cd.len_mode <= 60
                fam, tag = ("visium_probe", None) if probe else ("visium", "10x-visium")
                C.append(Candidate(fam, "10x-visium-probe" if probe else "10x-visium", round(rel, 3), tag if layout_ok else None,
                                   [base_ev] + ([f"cDNA starts repeat (probe library), top-100 15-mers {100 * cd.top100_15:.0f}%"] if probe else []),
                                   layout_ok))
                continue
            if name == "arc_gex":
                C.append(Candidate("10x_arc_gex", "10x-arc-gex", round(rel, 3), "10x-arc-gex" if layout_ok else None, [base_ev], layout_ok))
                continue
            # 3' versus 5': motif after barcode+UMI when the barcode read reaches it
            reach = s.len_mode >= after + 4
            ev3 = ev5 = 0.5
            why = "barcode read too short to see poly(T) or TSO after barcode+UMI"
            if reach:
                if pt >= T.motif_call and tso < pt / 3:
                    ev3, ev5, why = 1.0, 0.1, f"poly(T) at {after} in {100 * pt:.0f}% of barcode reads"
                elif tso >= T.motif_call and pt < tso / 3:
                    ev3, ev5, why = 0.1, 1.0, f"TSO at {after} in {100 * tso:.0f}% of barcode reads"
                else:
                    why = f"poly(T) {100 * pt:.0f}% / TSO {100 * tso:.0f}% at {after}: undecided"
            if cd is not None and ev3 == ev5:
                pa = cd.motifs["polyA_rt"]["share"]
                tr = cd.motifs["tso_rc"]["share"] + cd.motifs["tso"]["share"]
                if pa >= T.rt_call and pa >= T.rt_ratio * tr:
                    ev3, ev5 = 0.85, 0.3
                    why += f"; cDNA read-through: poly(A) {100 * pa:.1f}% vs TSO {100 * tr:.1f}% (3')"
                elif tr >= T.rt_call and tr >= T.rt_ratio * pa:
                    ev3, ev5 = 0.3, 0.85
                    why += f"; cDNA read-through: TSO {100 * tr:.1f}% vs poly(A) {100 * pa:.1f}% (5')"
                else:
                    why += f"; cDNA read-through poly(A) {100 * pa:.1f}% / TSO {100 * tr:.1f}%: undecided"
            if name in ENGINE_3P:
                e3 = ENGINE_3P[name]
                C.append(Candidate("10x_3p", e3, round(rel * ev3, 3), e3 if layout_ok else None, [base_ev, why], layout_ok))
            if name in ENGINE_5P:
                e5 = ENGINE_5P[name]
                C.append(Candidate("10x_5p", e5, round(rel * ev5, 3), e5 if layout_ok else None, [base_ev, why], layout_ok))
        if feat:
            # the cDNA slot is a tag read: the gene-expression readings of the same barcode read are not the library
            for c in C:
                c.score = round(c.score * 0.3, 3)
                c.evidence.append("demoted: the cDNA slot is a tag read")
            C.append(feat)
    # 10x ATAC (barcode read of 16 bases on an index-like slot, Tn5 in the genomic reads)
    if atac_hits and atac_hits[0][0] >= T.wl_present:
        r, n, s = atac_hits[0]
        me = max((x.motifs["tn5_me"]["share"] + x.motifs["tn5_me_rc"]["share"] for x in slots if x.motifs), default=0.0)
        C.append(Candidate("atac", "10x-arc-atac" if n == "arc_atac" else "10x-atac", round(min(1.0, r / 0.5), 3), None,
                           [f"{n} list {100 * r:.1f}% at slot {s.index} (length {s.len_mode})", f"Tn5 mosaic end in {100 * me:.0f}% of reads"]))
    # short-word protocol lists and linkers
    def lk(name):
        return max((s.motifs[name]["share"] for s in slots if name in s.motifs), default=0.0)

    def sl(name):
        return max((s.lists[name]["rate"] for s in slots if name in s.lists), default=0.0)
    for fam_tag, ls, lks in (("bd_rhapsody", ("bd_cls1", "bd_cls2"), ("bd_linker1", "bd_linker2")),
                             ("indrop", ("indrop_bc1",), ("indrop_w1",)),
                             ("splitseq", ("splitseq_rd1", "splitseq_rd2"), ("split_linker1", "split_linker2")),
                             ("microwell-seq", ("microwell_bc1",), ("microwell_l1", "microwell_l2"))):
        lr = max(sl(x) for x in ls)
        kr = max(lk(x) for x in lks)
        # short words (6 to 9 bases) also match constant sequence (a Smart-seq2 adapter hit a 6-mer list in 98 percent of
        # reads on the development half), so a list counts only next to its linker
        if kr >= T.linker_present:
            sc = min(1.0, 0.5 * min(1.0, kr / 0.5) + 0.5 * min(1.0, lr / 0.5))
            C.append(Candidate("other_barcoded", fam_tag, round(sc, 3), None,
                               [f"{fam_tag} list best {100 * lr:.0f}%, linker best {100 * kr:.0f}%"]))
    # sci-RNA-seq3 (RT barcode list) and Quartz-Seq2
    for tag, nm in (("sci-rna-seq3", "scirna3_rt"), ("quartzseq2", "quartzseq2")):
        r = sl(nm)
        if r >= T.short_present:
            C.append(Candidate("other_barcoded", tag, round(min(1.0, r / 0.5), 3), None, [f"{nm} list {100 * r:.0f}%"]))
    # Smart-seq3 tag
    st = max((s.motifs["smart3_tag"]["pos_share"] for s in slots if s.motifs.get("smart3_tag", {}).get("pos") == 0), default=0.0)
    if st >= T.smart3_tag:
        C.append(Candidate("no_cell_barcode", "smartseq3", round(min(1.0, st / 0.15), 3), None, [f"Smart-seq3 tag at read start in {100 * st:.0f}%"]))
    # barcode-read structure without a known list: bead (Drop-seq/Seq-Well) or plate (CEL-Seq2 family)
    top10 = best10[0] if best10 else 0.0
    bc_slots = [s for s in slots if s.len_mode >= 12 and (s.len_mode <= T.bc_slot_max or (s.polyT_pos is not None and s.polyT_pos <= 30 and s.polyT_pos_share >= 0.3))]
    if top10 < T.wl_present:
        for s in bc_slots:
            if s.prefix12_mult >= T.prefix_mult and (s.len_mode <= 25 or (s.polyT_pos is not None and 18 <= s.polyT_pos <= 22)):
                C.append(Candidate("other_barcoded", "dropseq", round(min(1.0, s.prefix12_mult / 0.8), 3), None,
                                   [f"slot {s.index} ({s.len_mode} b): first 12 bases repeat >= 5x in {100 * s.prefix12_mult:.0f}% (bead barcodes), "
                                    f"poly(T) at {s.polyT_pos}"]))
            elif s.polyT_pos is not None and s.polyT_pos <= 24 and s.polyT_pos_share >= 0.3:
                C.append(Candidate("other_barcoded", "celseq2", round(min(1.0, s.polyT_pos_share / 0.6), 3), None,
                                   [f"slot {s.index}: poly(T) at {s.polyT_pos} in {100 * s.polyT_pos_share:.0f}% without a 10x list (plate/UMI read)"]))
            elif s.prefix12_mult >= T.prefix_mult:
                C.append(Candidate("other_barcoded", "barcoded-unknown", round(0.6 * min(1.0, s.prefix12_mult / 0.8), 3), None,
                                   [f"slot {s.index} ({s.len_mode} b): repeated 12-base prefixes in {100 * s.prefix12_mult:.0f}%"]))
    # MARS-seq-like plate libraries: a diverse 12-20 base barcode/UMI read beside a cDNA read that starts with a pool barcode
    if top10 < T.wl_present:
        shorts = [s for s in slots if 12 <= s.len_mode <= 20 and s.prefix12_mult < T.short_diverse_max]
        def fixed_bases(s):    # positions in the first 12 where every read has T, or none has (constant pool barcode/linker)
            return sum(1 for f in s.t_frac[:12] if f >= 0.95 or f <= 0.02)
        pooled = [s for s in slots if s.len_mode > T.bc_slot_max and (s.prefix12_mult >= T.prefix_mult or fixed_bases(s) >= 2)]
        if shorts and pooled:
            C.append(Candidate("other_barcoded", "marsseq", round(max(0.7, min(1.0, pooled[0].prefix12_mult / 0.8)), 3), None,
                               [f"slot {shorts[0].index} ({shorts[0].len_mode} b) diverse (barcode/UMI read); slot {pooled[0].index} "
                                f"({pooled[0].len_mode} b) starts with a pool barcode (12-base prefix repeats in {100 * pooled[0].prefix12_mult:.0f}%, "
                                f"{fixed_bases(pooled[0])} fixed bases)"]))
    # no cell barcode at all: every slot is cDNA-like (or index), nothing above
    longs = [s for s in slots if s.len_mode > T.bc_slot_max]
    structured = any(c.score >= 0.3 for c in C)
    if not structured:
        me = max((s.motifs["tn5_me"]["share"] + s.motifs["tn5_me_rc"]["share"] for s in slots if s.motifs), default=0.0)
        ev = [f"no 10x list above {100 * T.wl_present:.0f}% in any slot at any offset or strand (best {100 * top10:.2f}%)",
              f"slots: {', '.join(str(s.len_mode) for s in slots)} b", f"Tn5 mosaic end {100 * me:.0f}%"]
        C.append(Candidate("no_cell_barcode", "rna-no-barcode", 0.9 if longs else 0.5, None, ev))
    C.sort(key=lambda c: -c.score)
    return C


def decide(cands: list, stated: str = "") -> dict:
    """{status: clear | trial | unsupported | none, top, trial_tags, why}.  `trial_tags` lists up to three engine tags for the
    trial mapping, ordered by score and then by the stated chemistry (which only orders, it never adds a candidate)."""
    if not cands:
        return {"status": "none", "top": None, "trial_tags": [], "why": "no candidate"}
    top = cands[0]
    runnable = [c for c in cands if c.engine_tag]
    nxt = next((c for c in cands[1:] if (c.engine_tag or c.protocol) != (top.engine_tag or top.protocol)), None)
    margin = top.score - (nxt.score if nxt else 0.0)
    st = stated_tag(stated)
    if top.score >= T.clear_score and margin >= T.clear_margin:
        status = "clear" if top.engine_tag else "unsupported"
        return {"status": status, "top": top.to_dict(), "trial_tags": [], "why": f"top {top.score:.2f}, margin {margin:.2f}"}
    close = [c for c in runnable if c.score >= top.score - T.clear_margin - 1e-9 or c.engine_tag == st]
    tags = []
    for c in sorted(close, key=lambda c: (-c.score, c.engine_tag != st)):
        if c.engine_tag not in tags:
            tags.append(c.engine_tag)
    if len(tags) >= 2 or (tags and not top.engine_tag):
        return {"status": "trial", "top": top.to_dict(), "trial_tags": tags[:3], "why": f"top {top.score:.2f}, margin {margin:.2f}"}
    if top.engine_tag:
        return {"status": "clear", "top": top.to_dict(), "trial_tags": [], "why": f"only runnable candidate; top {top.score:.2f}, margin {margin:.2f}"}
    return {"status": "unsupported", "top": top.to_dict(), "trial_tags": [], "why": f"top {top.score:.2f}, margin {margin:.2f}"}


def stated_tag(stated: str) -> str:
    s = (stated or "").lower().replace("_", "-").replace(" ", "")
    m = {"10x-3p-v1": "10x-3p-v1", "10x-3p-v2": "10x-3p-v2", "10x-3p-v3": "10x-3p-v3", "10x-3p-v3.1": "10x-3p-v3",
         "10x-3p-v4": "10x-3p-v4", "10x-5p-v1": "10x-5p-v2", "10x-5p-v2": "10x-5p-v2", "10x-5p-v3": "10x-5p-v3"}
    return m.get(s, "")


def read_fastq(path: str, max_reads: int = 50000) -> list:
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                out.append(line.strip())
                if len(out) >= max_reads:
                    break
    return out


def detect(seg_seqs: list, wl_dir: str, stated: str = "") -> dict:
    lists = load_lists(wl_dir)
    slots = [scan_slot(seqs, i + 1, lists) for i, seqs in enumerate(seg_seqs) if seqs]
    cands = rank(slots)
    return {"schema": "protocol-detect/1", "slots": [s.to_dict() for s in slots], "candidates": [c.to_dict() for c in cands],
            "decision": decide(cands, stated)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl-dir", required=True)
    ap.add_argument("--seg", action="append", required=True)
    ap.add_argument("--stated", default="")
    ap.add_argument("--max-reads", type=int, default=50000)
    ap.add_argument("--json", default="")
    ap.add_argument("--id", default="")
    a = ap.parse_args(argv)
    res = detect([read_fastq(p, a.max_reads) for p in a.seg], a.wl_dir, a.stated)
    if a.id:
        res["id"] = a.id
    txt = json.dumps(res, indent=1, default=str)
    if a.json:
        open(a.json, "w").write(txt)
    else:
        print(txt)


if __name__ == "__main__":
    sys.exit(main())
