"""Reference for the GPU barcode layer (align-and-count-design.md section 4.1), phases 1c and 1f.

Maps one input read to the normalized record (cell_key, umi_key, clip spans, feature_read) by reproducing the
STARsolo rules in src/star/ line for line. This is a specification the CUDA drafts in barcode_layer.cuh must match;
it is not a faster STAR. Each function names the STAR lines it mirrors. Rules and quirks:
docs/research-beta/barcode-layer-prep.md.

Conventions: STAR's sentinel cbMatch codes are kept (>=0 matched, -1 no WL match, -2 N in CB, -3 too many WL
matches, -11 CB not extractable, -12 mismatches in several segments, -21 no adapter, -22 no UMI, -23 N in UMI,
-24 UMI homopolymer). A numba batch version of the CB_UMI_Simple path (exact histogram, then 1MM_multi posterior)
sits at the end; the tests assert it equals the plain-Python path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numba import njit

# ------------------------------------------------------------------------------------------------------------
# nucleotide helpers (SequenceFuns.cpp:206 convertNt01234, :260 convertNuclStrToInt64)
# ------------------------------------------------------------------------------------------------------------
_NT = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}


def conv(s: str) -> tuple[int, int]:
    """convertNuclStrToInt64: (2-bit value with N as A, posN). posN -1: no N; >=0: one N at that index; -2: two or more.
    On -2 STAR returns early with the value partially built; callers only read the value when posN != -2."""
    v, pos_n = 0, -1
    for i, c in enumerate(s):
        nt = _NT.get(c, 4)
        if nt > 3:
            if pos_n >= 0:
                return v, -2
            pos_n = i
            nt = 0
        v = (v << 2) + nt
    return v, pos_n


def pack3(s: str) -> int:
    """umi_key: raw UMI, 3 bits per base (A C G T N other), the packing rule R9 uses; first base in the high bits."""
    v = 0
    for c in s:
        v = (v << 3) | _NT.get(c, 4 if c in "Nn" else 5)
    return v


# ------------------------------------------------------------------------------------------------------------
# CB matching to a whitelist
# ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchFlags:
    """init_CBmatchWL, ParametersSolo.cpp:679-725."""
    mm1: bool = False
    mm1_multi: bool = False
    mm1_multi_pc: bool = False
    mm1_multi_nbase: bool = False
    one_exact: bool = False
    edit2: bool = False


def flags_from_type(t: str) -> MatchFlags:
    table = {
        "Exact": MatchFlags(one_exact=True),
        "1MM": MatchFlags(mm1=True, one_exact=True),
        "1MM_multi": MatchFlags(mm1=True, mm1_multi=True, one_exact=True),
        "1MM_multi_pseudocounts": MatchFlags(mm1=True, mm1_multi=True, mm1_multi_pc=True),
        "1MM_multi_Nbase_pseudocounts": MatchFlags(mm1=True, mm1_multi=True, mm1_multi_pc=True, mm1_multi_nbase=True),
        "EditDist_2": MatchFlags(edit2=True),
    }
    return table[t]


class Whitelist:
    """Sorted, de-duplicated 2-bit barcodes (ParametersSolo.cpp:334-338); a barcode's index is its sorted position.
    An entry with N at position 0 is skipped, any other N reads as A (the `if (conv(...))` test, :321 and :373)."""

    def __init__(self, seqs: list[str], cb_len: Optional[int] = None):
        keys = set()
        for s in seqs:
            if cb_len is not None and len(s) != cb_len:
                raise ValueError("whitelist entry length differs from soloCBlen (fatal in STAR, :312)")
            v, pos_n = conv(s)
            if pos_n == 0:
                continue
            keys.add(v)
        self.keys = np.array(sorted(keys), dtype=np.uint64)
        self._idx = {int(k): i for i, k in enumerate(self.keys)}
        self.strings_len = cb_len

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, v: int) -> int:
        return self._idx.get(v, -1)


@dataclass
class CbMatch:
    cb_match: int = -1
    ind: list = field(default_factory=list)      # cbMatchInd1: whitelist indices
    qual: list = field(default_factory=list)     # quality char of the changed base, one per ind when cb_match > 1


def match_cb_to_wl(cb_seq: str, cb_qual: str, wl: Optional[Whitelist], f: MatchFlags) -> CbMatch:
    """SoloReadBarcode::matchCBtoWL, SoloReadBarcode_getCBandUMI.cpp:9-85, branch for branch."""
    val, pos_n = conv(cb_seq)
    if wl is None:                                            # :18-27 no whitelist: no search
        if pos_n != -1:
            return CbMatch(-2)
        return CbMatch(0, [val])
    if pos_n == -2:                                           # :29
        return CbMatch(-2)
    if pos_n == -1:                                           # :32-39 exact by hash
        i = wl.lookup(val)
        if i >= 0:
            return CbMatch(0, [i])
    if not f.mm1:                                             # :42 exact only; cbMatch1 is still -1
        return CbMatch(-1)
    m = CbMatch(0)
    n = len(cb_seq)
    if pos_n >= 0:                                            # :46-63 one N: try the four bases at the N
        shift = 2 * (n - 1 - pos_n)
        matched = False
        for jj in range(4):
            i = wl.lookup(val ^ (jj << shift))
            if i >= 0:
                if not f.mm1_multi_nbase and matched:         # :53 second hit without Nbase: -3
                    m.ind, m.qual, m.cb_match = [], [], -3
                    break
                matched = True
                m.ind.append(i)
                m.cb_match += 1
                m.qual.append(cb_qual[pos_n])                 # quality of the N base
    else:                                                     # :64-75 1MM: last base first, substitutions 1..3
        for ii in range(n):
            for jj in range(1, 4):
                i = wl.lookup(val ^ (jj << (ii * 2)))
                if i >= 0:
                    m.ind.append(i)
                    m.cb_match += 1
                    m.qual.append(cb_qual[n - 1 - ii])
    if m.cb_match == 0:                                       # :76-84
        m.cb_match = -1
    elif m.cb_match == 1:
        pass
    elif not f.mm1_multi:
        m.cb_match, m.ind, m.qual = -3, [], []
    return m


def check_umi(umi_seq: str, homopolymer_len: int) -> tuple[int, int]:
    """convertCheckUMI, :127-138. Returns (umiCheck, umiB). homopolymer_len is pSolo.umiL when SoloReadBarcode was
    constructed (SoloReadBarcode.cpp:16-20): the UMI length for Simple/samTagOut, 0 for Complex, which makes all four
    'homopolymers' equal 0, so only an all-A UMI (value 0) is rejected there."""
    v, pos_n = conv(umi_seq)
    if pos_n != -1:
        return -23, v
    homo = []
    for j in range(4):
        h = 0
        for _ in range(homopolymer_len):
            h = (h << 2) + j
        homo.append(h)
    if v in homo:
        return -24, v
    return 0, v


# ------------------------------------------------------------------------------------------------------------
# soloType layouts and per-read extraction
# ------------------------------------------------------------------------------------------------------------
@dataclass
class ReadBarcode:
    """The fields of SoloReadBarcode that leave getCBandUMI."""
    cb_match: int = -1
    match: CbMatch = field(default_factory=CbMatch)
    cb_seq: str = ""
    umi_seq: str = ""
    umi_b: int = 0
    cb_seq_corrected: str = ""
    cb_index: int = -1               # CB_UMI_Complex: mixed-radix index over the segments (cbMatchInd[0])


@dataclass
class SimpleLayout:
    """CB_UMI_Simple and CB_samTagOut: soloCBstart/CBlen/UMIstart/UMIlen are 1-based as in STAR."""
    cb_start: int
    cb_len: int
    umi_start: int
    umi_len: int
    flags: MatchFlags
    wl: Optional[Whitelist] = None
    sam_tag_out: bool = False
    bl: int = 0                      # --soloBarcodeReadLength; singlet passes 0
    wl_strings: Optional[list] = None  # sorted unique whitelist strings, for cbSeqCorrected

    @property
    def cbumi_len(self) -> int:
        return self.cb_len + self.umi_len


def _pad_barcode_read(seq: str, qual: str, lay) -> tuple[str, str]:
    """:222-235. bL == 0: pad with N / 'H' up to CB+UMI length. bL > 0 and a different length is fatal in STAR."""
    if len(seq) != lay.bl:
        if lay.bl > 0:
            raise ValueError("barcode read length differs from --soloBarcodeReadLength (STAR exits, :222-230)")
        if len(seq) < lay.cbumi_len:
            seq = seq + "N" * (lay.cbumi_len - len(seq))
            qual = qual + "H" * (lay.cbumi_len - len(qual))
    return seq, qual


def _substr(s: str, pos: int, n: int) -> str:
    if pos > len(s):
        raise IndexError("std::string::substr position past the end (std::out_of_range)")
    return s[pos:pos + n]


def get_cb_umi_simple(seq: str, qual: str, lay: SimpleLayout) -> ReadBarcode:
    """getCBandUMI for CB_UMI_Simple (:246-303) and CB_samTagOut (:305-322)."""
    rb = ReadBarcode()
    seq, qual = _pad_barcode_read(seq, qual, lay)
    rb.cb_seq = _substr(seq, lay.cb_start - 1, lay.cb_len)
    rb.umi_seq = _substr(seq, lay.umi_start - 1, lay.umi_len)
    cb_qual = _substr(qual, lay.cb_start - 1, lay.cb_len)
    rb.match = match_cb_to_wl(rb.cb_seq, cb_qual, lay.wl, lay.flags)
    rb.cb_match = rb.match.cb_match
    if lay.sam_tag_out:                                        # no UMI check here; the UMI is carried raw
        if rb.cb_match in (0, 1):
            rb.cb_seq_corrected = lay.wl_strings[rb.match.ind[0]] if lay.wl is not None else rb.cb_seq
        else:
            rb.cb_seq_corrected = "-"
        rb.umi_b = conv(rb.umi_seq)[0]
        return rb
    umi_check, rb.umi_b = check_umi(rb.umi_seq, lay.umi_len)
    if umi_check != 0:                                         # :291-303
        rb.cb_match = umi_check
        rb.match = CbMatch(umi_check)
    return rb


@dataclass
class Segment:
    """SoloBarcode: one CB segment (or the UMI) of CB_UMI_Complex. pos is the --soloCBposition string 'a_d_b_e'."""
    pos: str
    wl: Optional[dict] = None        # length -> sorted unique 2-bit list, filled by build_segment_wl
    wl_add: dict = field(default_factory=dict)
    wl_factor: int = 1
    total_size: int = 0
    min_len: int = 0xFFFFFFFF
    anchor_type: tuple = (0, 0)
    anchor_dist: tuple = (0, 0)

    def __post_init__(self):
        p = self.pos.split("_")                                # extractPositionsFromString, SoloBarcode.cpp:37-45
        self.anchor_type = (int(p[0]), int(p[2]))
        self.anchor_dist = (int(p[1]), int(p[3]))


def build_segment_wl(seg: Segment, seqs: list[str], adapter_len: int = 0) -> None:
    """ParametersSolo.cpp:373-396 and SoloBarcode::sortWhiteList (SoloBarcode.cpp:9-35)."""
    by_len: dict[int, list[int]] = {}
    for s in seqs:
        v, pos_n = conv(s)
        if pos_n == 0:
            continue
        by_len.setdefault(len(s), []).append(v)
    seg.wl = {}
    seg.wl_add = {}
    total = 0
    seg.min_len = 0xFFFFFFFF
    for ln in range(1, max(by_len) + 1):
        seg.wl_add[ln] = total
        if ln in by_len:
            seg.min_len = min(seg.min_len, ln)
            seg.wl[ln] = sorted(set(by_len[ln]))
            total += len(seg.wl[ln])
    seg.total_size = total


def extract_barcode(seg: Segment, seq: str, qual: str, adapter_start: int, adapter_len: int):
    """SoloBarcode::extractBarcode, SoloBarcode_extractBarcode.cpp:4-33. Returns (ok, subseq, subqual).
    pos[1] == len(seq) passes the bounds test (:28) and substr then clips, so the segment is one base short."""
    pos = [0, 0]
    for k in range(2):
        t = seg.anchor_type[k]
        if t == 0:
            pos[k] = 0
        elif t == 1:
            pos[k] = len(seq) - 1
        elif t == 2:
            pos[k] = adapter_start
        elif t == 3:
            pos[k] = adapter_start + adapter_len - 1
        pos[k] += seg.anchor_dist[k]
    if pos[0] < 0 or pos[1] > len(seq) or pos[0] > pos[1]:
        return False, "", ""
    return True, seq[pos[0]:pos[1] + 1], qual[pos[0]:pos[1] + 1]


def local_align_hamming_dist(text: str, query: str) -> tuple[int, int]:
    """localAlignHammingDist, SequenceFuns.cpp:352-371: (best distance, position). N in the query is a wildcard; ties
    keep the first position; position stays 0 unless a distance below len(query) is found."""
    if len(text) < len(query):
        return len(text) + 1, 0
    best, pos = len(query), 0
    for i in range(len(text) - len(query) + 1):
        d = sum(1 for j, qc in enumerate(query) if qc != "N" and text[i + j] != qc)
        if d < best:
            best, pos = d, i
    return best, pos


@dataclass
class ComplexLayout:
    """CB_UMI_Complex. singlet passes --soloCBmatchWLtype Exact (singlet.cpp:7309-7316) and no adapter."""
    segments: list
    umi: Segment
    flags: MatchFlags = field(default_factory=lambda: MatchFlags(one_exact=True))
    adapter: str = "-"
    adapter_mm_max: int = 1                                    # parametersDefault:828
    seg_strings: Optional[list] = None                         # per segment: {len: [strings sorted by 2-bit value]}


def get_cb_umi_complex(seq: str, qual: str, lay: ComplexLayout) -> ReadBarcode:
    """getCBandUMI for CB_UMI_Complex, :325-420. EditDist_2 is not implemented (singlet does not use it)."""
    assert not lay.flags.edit2
    rb = ReadBarcode()
    adapter_start = 0
    adapter_yes = lay.adapter != "-"
    alen = len(lay.adapter) if adapter_yes else 0
    if adapter_yes:                                            # :333-339
        d, adapter_start = local_align_hamming_dist(seq, lay.adapter)
        if d > lay.adapter_mm_max:
            rb.cb_match = -21
            return rb
    ok, rb.umi_seq, _ = extract_barcode(lay.umi, seq, qual, adapter_start, alen)
    if not ok:                                                 # :341-345
        rb.cb_match = -22
        return rb
    good = True
    cb_match = -1                                              # :156
    umi_check, rb.umi_b = check_umi(rb.umi_seq, 0)             # homopolymer length 0, see check_umi
    if umi_check != 0:
        good = False
        cb_match = umi_check
    index = 0
    parts = []
    for seg in lay.segments:                                   # :357-413
        ok, s1, q1 = extract_barcode(seg, seq, qual, adapter_start, alen)
        if (not ok) or len(s1) < seg.min_len or len(s1) >= (max(seg.wl) + 1) or len(s1) not in seg.wl:
            if good:
                cb_match = -11
                good = False
        parts.append(s1)
        if not good:
            continue
        m = match_cb_to_wl(s1, q1, Whitelist_from_sorted(seg.wl[len(s1)]), lay.flags)
        if m.cb_match < 0:
            good = False
            cb_match = m.cb_match
        elif m.cb_match > 0 and cb_match > 0:                  # :405
            good = False
            cb_match = -12
        else:
            index += seg.wl_factor * (m.ind[0] + seg.wl_add[len(s1)])
            cb_match = max(cb_match, m.cb_match)
    rb.cb_seq = "_".join(parts)
    rb.cb_match = cb_match
    if good:
        rb.cb_index = index
    return rb


def Whitelist_from_sorted(values: list) -> Whitelist:
    w = Whitelist([])
    w.keys = np.array(values, dtype=np.uint64)
    w._idx = {int(k): i for i, k in enumerate(values)}
    return w


def link_segments(segments: list[Segment]) -> int:
    """wlFactor is the product of the earlier segments' totalSize (ParametersSolo.cpp:393-395): segment 0 varies fastest.
    Returns cbWLsize, the number of composite barcodes."""
    size = 1
    for s in segments:
        s.wl_factor = size
        size *= s.total_size
    return size


# ------------------------------------------------------------------------------------------------------------
# the posterior for non-exact matches (SoloReadFeature_inputRecords.cpp:35-107)
# ------------------------------------------------------------------------------------------------------------
QS_BASE, QS_MAX = 33, 33
CB_MIN_P = np.float32(0.975)           # float in the non-MATCH_CellRanger build (ParametersSolo.h:137-141)


def exact_counts(read_barcodes: list, wl_size: int, flags: MatchFlags) -> np.ndarray:
    """cbReadCountExact: reads whose final cbMatch is 0 (addStats, :92-96), summed over threads, then +1 on every
    whitelist entry for the pseudocount types (Solo.cpp:66-71, SoloFeature_sumThreads.cpp:52-57)."""
    c = np.zeros(wl_size, dtype=np.uint32)
    for rb in read_barcodes:
        if rb.cb_match == 0:
            c[rb.match.ind[0]] += 1
    if flags.mm1_multi_pc:
        c += 1
    return c


def resolve_cb(rb: ReadBarcode, counts: np.ndarray, flags: MatchFlags) -> tuple[int, str]:
    """Which cell a read with cbMatch >= 0 is assigned to. Returns (cell index or -1, reason).
    Reasons: 'exact', 'mm1', 'mm1_no_exact' (noMMtoWLwithoutExact), 'posterior', 'posterior_reject'."""
    m = rb.match
    if rb.cb_match < 0:                                                     # SoloReadFeature_record.cpp:66 never records it
        return -1, "no_cb"
    if rb.cb_match <= 1:
        cb = m.ind[0]
        if flags.one_exact and rb.cb_match == 1 and counts[cb] == 0:        # :38
            return -1, "mm1_no_exact"
        return cb, "exact" if rb.cb_match == 0 else "mm1"
    ptot = np.float32(0.0)
    pmax = np.float32(0.0)
    cb = -1
    for ind, q in zip(m.ind, m.qual):                                       # :75-89
        if counts[ind] > 0:
            qin = ord(q) - QS_BASE
            qin = min(qin, QS_MAX)
            pin = np.float32(float(counts[ind]) * (10.0 ** (-qin / 10.0)))
            ptot = np.float32(ptot + pin)
            if pin > pmax:
                cb, pmax = ind, pin
    if ptot > 0.0 and pmax >= np.float32(CB_MIN_P * ptot):                  # :90
        return cb, "posterior"
    return -1, "posterior_reject"


# ------------------------------------------------------------------------------------------------------------
# no-whitelist protocols: the barcode list singlet builds before the pileup (singlet.cpp Phase 0.9)
# ------------------------------------------------------------------------------------------------------------
def discover_barcodes(raw_counts: dict, min_reads: int, stream: bool) -> list:
    """Barcodes with >= min_reads raw reads (5 on the unsorted stream, 100 on the sorted path), by count descending;
    stream ties go to the smaller barcode string. The sorted path's ties are std::sort order (unspecified), so only the
    stream order is reproducible. cell_key is the position in this list."""
    items = [(c, b) for b, c in raw_counts.items() if c >= min_reads]
    items.sort(key=lambda t: (-t[0], t[1]))
    return [b for _, b in items]


def cell_key_no_wl(cb_seq: str, discovered_index: dict) -> int:
    """STAR with soloCBwhitelist None accepts any N-free CB (matchCBtoWL :18-27); the pileup then keeps only barcodes in
    the discovered list. -2 N in CB, -1 not in the list."""
    if "N" in cb_seq.upper():
        return -2
    return discovered_index.get(cb_seq, -1)


# ------------------------------------------------------------------------------------------------------------
# Visium spot list (include/singlet/pileup/visium_spatial.h:52-99)
# ------------------------------------------------------------------------------------------------------------
_INT = re.compile(r"^\s*[+-]?\d+")
_FLT = re.compile(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def _stoi(s: str) -> int:
    m = _INT.match(s)
    if not m:
        raise ValueError("std::invalid_argument from std::stoi")
    return int(m.group(0))


def _stof(s: str) -> float:
    m = _FLT.match(s)
    if not m:
        raise ValueError("std::invalid_argument from std::stof")
    return float(np.float32(float(m.group(0))))


def load_positions(text: str) -> dict:
    """tissue_positions.csv -> {barcode: (array_row, array_col, pixel_row, pixel_col, in_tissue)}. A repeated barcode
    keeps the last row; the first non-empty line is a header if it contains 'barcode' or 'array_row'; rows with fewer
    than 6 fields are skipped; in_tissue is the field equal to '1'."""
    coords, first = {}, True
    for line in text.split("\n"):
        line = line[:-1] if line.endswith("\r") else line
        if not line:
            continue
        if first:
            first = False
            lo = line.lower()
            if "barcode" in lo or "array_row" in lo:
                continue
        f = line.split(",")
        if len(f) < 6:
            continue
        coords[f[0]] = (_stoi(f[2]), _stoi(f[3]), _stof(f[4]), _stof(f[5]), f[1] == "1")
    return coords


def spot_table(coords: dict, barcodes: list) -> list:
    """write_coordinates_tsv: one row per called barcode in engine order; a barcode missing from the file gets zeros."""
    out = []
    for bc in barcodes:
        c = coords.get(bc)
        out.append((bc,) + ((c[0], c[1], c[2], c[3], int(c[4])) if c else (0, 0, 0, 0, 0)))
    return out


# ------------------------------------------------------------------------------------------------------------
# feature barcodes: ADT / hashtag tag matcher, CRISPR guide matcher (adt_matcher.h, adt_counter.h, crispr_guide.h)
# ------------------------------------------------------------------------------------------------------------
class AdtMatcher:
    """adt_matcher.h:131-177 build_index and :191-232 match. Tag length is the first tag's; offsets 0..max_offset; at
    each offset exact first, then Hamming-1 (-2 when two tags share the neighbour; it returns at once, even if a later
    offset would match exactly)."""

    def __init__(self, tags: list):
        self.tags = [t.upper() for t in tags]
        self.tag_len = len(self.tags[0]) if self.tags else 0
        self.exact = {t: i for i, t in enumerate(self.tags)}      # a duplicate sequence keeps the last id
        self.h1 = {}
        for i, t in enumerate(self.tags):
            for pos in range(len(t)):
                for b in "ACGT":
                    if b == t[pos]:
                        continue
                    nbr = t[:pos] + b + t[pos + 1:]
                    if nbr in self.exact:
                        continue
                    if nbr not in self.h1:
                        self.h1[nbr] = i
                    elif self.h1[nbr] != i:
                        self.h1[nbr] = -2

    def match(self, seq: str, max_offset: int = 5) -> int:
        n = len(seq)
        if self.tag_len <= 0 or n < self.tag_len:
            return -1
        limit = min(max_offset, n - self.tag_len)
        for off in range(limit + 1):
            w = seq[off:off + self.tag_len].upper()
            if w in self.exact:
                return self.exact[w]
            if w in self.h1:
                return self.h1[w]
        return -1


def adt_record(r1: str, r2: str, umi_len: int, cells: dict, matcher: AdtMatcher):
    """One ADT read from the .1fq (singlet.cpp:8540-8572): the cell barcode must be in the called set EXACTLY (no
    correction), UMI = R1[bc_len:], tag from R2. Returns (cell index, tag id, umi) or None."""
    bc_len = len(r1) - umi_len if len(r1) > umi_len else len(r1)
    cell = cells.get(r1[:bc_len])
    if cell is None:
        return None
    tag = matcher.match(r2)
    if tag < 0:
        return None
    return cell, tag, r1[bc_len:]


def adt_count_exact(records: list) -> dict:
    """AdtCounter as it is today (adt_counter.h:44-53): distinct UMI strings per (cell, tag). Kept only to show the
    difference from R9; the rule for the GPU feature reducer is adt_count."""
    seen: dict = {}
    for cell, tag, umi in records:
        seen.setdefault((cell, tag), set()).add(umi)
    return {k: len(v) for k, v in seen.items()}


def _umi_code(c: str) -> int:
    return {"a": 0, "c": 1, "g": 2, "t": 3, "n": 4}.get(c.lower(), 5)


def _one_mismatch(a: str, b: str) -> bool:
    """umi3_one_mismatch (umi_dedup.h): same length, exactly one 3-bit base group differs (N is a base of its own)."""
    if len(a) != len(b):
        return False
    return sum(1 for x, y in zip(a, b) if _umi_code(x) != _umi_code(y)) == 1


def r9_molecules(umi_counts: dict) -> int:
    """R9 (order-free-rules.md; umi_dedup.h umi3_directional_roots, UmiFeatureCollapse::finalize) for one (cell, feature):
    umi_counts maps raw UMI string -> read count. Two same-length UMIs are joined when exactly one base differs and the
    larger count is >= 2 * the smaller - 1; molecules are the connected components. UMIs longer than 21 bases match
    exactly (each distinct one is a molecule); an empty UMI is one molecule per read."""
    mols = 0
    short = {}
    for u, n in umi_counts.items():
        if not u:
            mols += n
        elif len(u) > 21:
            mols += 1
        else:
            short[u] = n
    keys = sorted(short, key=lambda u: (-short[u], [_umi_code(c) for c in u]))   # count descending, then packed UMI
    root = list(range(len(keys)))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if short[keys[i]] >= 2 * short[keys[j]] - 1 and _one_mismatch(keys[i], keys[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    root[rj] = ri
    return mols + sum(1 for i in range(len(keys)) if find(i) == i)


def adt_count(records: list) -> dict:
    """Feature-barcode counts under R9: per (cell, tag) the number of directional-collapse molecules of the raw UMIs.
    Decision of 2026-10-29: the same UMI rule as genes, junctions and SNPs. The CPU AdtCounter (adt_count_exact) has to
    follow before gate (a) can compare the two."""
    per: dict = {}
    for cell, tag, umi in records:
        d = per.setdefault((cell, tag), {})
        d[umi] = d.get(umi, 0) + 1
    return {k: r9_molecules(v) for k, v in per.items()}


def match_guide(read_seq: str, guides: list) -> tuple[int, list]:
    """Guide assignment, decision of 2026-10-29 (implemented in crispr_guide.h, branch fix/crispr-guide-orientation): exact
    substring of the upper-cased read in SEQUENCING orientation. Exactly one guide in the read: its id. None: -1. Two or
    more distinct ids (a guide nested in another, or a sequence listed twice): -2, ambiguous, not counted. Returns
    (id or -1 or -2, all matching ids)."""
    rs = read_seq.upper()
    hits = [i for i, g in enumerate(guides) if g.upper() in rs]
    if not hits:
        return -1, []
    return (hits[0] if len(hits) == 1 else -2), hits


_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def revcomp(s: str) -> str:
    """GuideRef::reverse_complement: any base other than A/C/G/T (either case) becomes N."""
    return "".join(_COMP.get(c.upper(), "N") for c in reversed(s))


def match_guide_bam(bam_seq: str, reverse_strand: bool, guides: list) -> tuple[int, list]:
    """A BAM record of a reverse-strand alignment holds the reverse complement of the read; restore the sequencing
    orientation before matching."""
    return match_guide(revcomp(bam_seq) if reverse_strand else bam_seq, guides)


# ------------------------------------------------------------------------------------------------------------
# the normalized record
# ------------------------------------------------------------------------------------------------------------
@dataclass
class Record:
    cell_key: int            # -1 dropped; for no-WL the discovered-list position; for Complex the composite index
    umi_key: int             # raw UMI, 3 bits per base; 0 with umi_len 0
    umi_len: int
    status: int              # STAR cbMatch code (or 0)
    clip5: int = 0           # cDNA mate: leading bases removed (--clip5pNbases N 0)
    adapter3p: str = ""      # cDNA mate: --clip3pAdapterSeq; the span comes from the phase-1a k_clip
    feature_read: bool = False
    feature_id: int = -1


def simple_record(seq, qual, lay: SimpleLayout, counts: Optional[np.ndarray], discovered: Optional[dict] = None,
                  clip5: int = 0, adapter3p: str = "") -> Record:
    rb = get_cb_umi_simple(seq, qual, lay)
    umi_key = pack3(rb.umi_seq)
    if lay.wl is None:                                         # no whitelist: STAR gives the raw CB, the list gives the cell
        if rb.cb_match < 0:                                    # N in the CB (-2) or a UMI with N / a homopolymer (-23, -24): the stream's
            return Record(-1, umi_key, lay.umi_len, rb.cb_match, clip5, adapter3p)   # inline_scan_cb_from_cr drops the read
        cell = discovered.get(rb.cb_seq, -1) if discovered is not None else -1
        return Record(cell, umi_key, lay.umi_len, 0, clip5, adapter3p)
    if rb.cb_match < 0:
        return Record(-1, umi_key, lay.umi_len, rb.cb_match, clip5, adapter3p)
    cb, _ = resolve_cb(rb, counts, lay.flags)
    return Record(cb, umi_key, lay.umi_len, rb.cb_match if cb >= 0 else -3, clip5, adapter3p)


def complex_record(seq, qual, lay: ComplexLayout, clip5: int = 0) -> Record:
    rb = get_cb_umi_complex(seq, qual, lay)
    umi_len = len(rb.umi_seq)
    if rb.cb_match < 0 or rb.cb_index < 0:
        return Record(-1, pack3(rb.umi_seq), umi_len, rb.cb_match, clip5)
    return Record(rb.cb_index, pack3(rb.umi_seq), umi_len, rb.cb_match, clip5)


# ------------------------------------------------------------------------------------------------------------
# numba batch version of the CB_UMI_Simple path: the form the CUDA kernels take
# ------------------------------------------------------------------------------------------------------------
@njit(cache=False)
def _bsearch(keys, v):
    lo, hi = 0, keys.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    if lo < keys.shape[0] and keys[lo] == v:
        return lo
    return -1


@njit(cache=False)
def _pack(cb, off, n):
    """2-bit value and N position of cb[off:off+n] (codes 0..3, 4 = N), as convertNuclStrToInt64."""
    v = np.uint64(0)
    pos_n = -1
    for i in range(n):
        c = cb[off + i]
        if c > 3:
            if pos_n >= 0:
                return v, -2
            pos_n = i
            c = 0
        v = (v << np.uint64(2)) + np.uint64(c)
    return v, pos_n


@njit(cache=False)
def exact_pass(cbs, quals, n_reads, cb_len, wl, hist):
    """Pass 1: exact-match histogram over all reads that reached addStats with cbMatch 0 (caller drops reads whose UMI
    check failed). Order-free: integer adds."""
    for r in range(n_reads):
        v, pos_n = _pack(cbs[r], 0, cb_len)
        if pos_n == -1:
            i = _bsearch(wl, v)
            if i >= 0:
                hist[i] += 1


@njit(cache=False)
def resolve_pass(cbs, quals, n_reads, cb_len, wl, counts, mm1_multi_nbase, qtab, out_cell, out_code):
    """Pass 2: matchCBtoWL for 1MM_multi(_Nbase)_pseudocounts and the posterior, one read at a time. counts already
    holds exact + 1. out_code is the cbMatch outcome (0 exact, 1 one 1MM/N hit, 2 multi accepted, -1/-2/-3 as STAR,
    -4 posterior rejected). qtab[q] = 10**(-q/10) as double, q in 0..QSmax."""
    cand = np.empty(3 * cb_len + 4, dtype=np.int64)
    cq = np.empty(3 * cb_len + 4, dtype=np.int64)
    for r in range(n_reads):
        v, pos_n = _pack(cbs[r], 0, cb_len)
        out_cell[r] = -1
        if pos_n == -2:
            out_code[r] = -2
            continue
        if pos_n == -1:
            i = _bsearch(wl, v)
            if i >= 0:
                out_cell[r] = i
                out_code[r] = 0
                continue
        n = 0
        if pos_n >= 0:
            shift = np.uint64(2 * (cb_len - 1 - pos_n))
            matched = False
            toomany = False
            for jj in range(4):
                i = _bsearch(wl, v ^ (np.uint64(jj) << shift))
                if i >= 0:
                    if (not mm1_multi_nbase) and matched:
                        toomany = True
                        break
                    matched = True
                    cand[n] = i
                    cq[n] = quals[r, pos_n]
                    n += 1
            if toomany:
                out_code[r] = -3
                continue
        else:
            for ii in range(cb_len):
                for jj in range(1, 4):
                    i = _bsearch(wl, v ^ (np.uint64(jj) << np.uint64(2 * ii)))
                    if i >= 0:
                        cand[n] = i
                        cq[n] = quals[r, cb_len - 1 - ii]
                        n += 1
        if n == 0:
            out_code[r] = -1
        elif n == 1:
            out_cell[r] = cand[0]       # pseudocount types: oneExact is off, counts[cb] >= 1 always
            out_code[r] = 1
        else:
            ptot = np.float32(0.0)
            pmax = np.float32(0.0)
            best = -1
            for k in range(n):
                c = counts[cand[k]]
                if c > 0:
                    q = cq[k] - 33
                    if q > 33:
                        q = 33
                    pin = np.float32(np.float64(c) * qtab[q])
                    ptot = np.float32(ptot + pin)
                    if pin > pmax:
                        best = cand[k]
                        pmax = pin
            if ptot > np.float32(0.0) and pmax >= np.float32(np.float32(0.975) * ptot):
                out_cell[r] = best
                out_code[r] = 2
            else:
                out_code[r] = -4


def qual_table() -> np.ndarray:
    return np.array([10.0 ** (-q / 10.0) for q in range(QS_MAX + 1)], dtype=np.float64)


def simple_batch(cb_strings: list, cb_quals: list, wl: Whitelist, nbase: bool = True):
    """Convenience driver for the tests: ASCII reads in, (cell, code) arrays out, with the pseudocount histogram."""
    n, L = len(cb_strings), len(cb_strings[0])
    cbs = np.array([[_NT.get(c, 4) for c in s] for s in cb_strings], dtype=np.uint8)
    qs = np.array([[ord(c) for c in s] for s in cb_quals], dtype=np.int64)
    hist = np.zeros(len(wl), dtype=np.uint32)
    exact_pass(cbs, qs, n, L, wl.keys, hist)
    hist += 1
    cell = np.empty(n, dtype=np.int64)
    code = np.empty(n, dtype=np.int64)
    resolve_pass(cbs, qs, n, L, wl.keys, hist, nbase, qual_table(), cell, code)
    return cell, code, hist


# ------------------------------------------------------------------------------------------------------------
# protocol rows -> layout, as singlet runs STAR today (include/singlet/fq/protocol.h:66-190, sra_encoder.h:2524-2551,
# singlet.cpp:6600-6840 and :7309-7316). Positions are 0-based [start, end] on the barcode read after singlet's R1/R2 swap.
# mode: 'simple'  CB_UMI_Simple, whitelist, 1MM_multi_Nbase_pseudocounts   (phase 1a, on today)
#       'nowl'    CB_samTagOut, no whitelist, barcodes discovered, Exact    (phase 1c)
#       'samwl'   CB_samTagOut with a whitelist, Exact                      (phase 1c)
#       'complex' CB_UMI_Complex or its concatenated CB_samTagOut form, per-segment whitelists, Exact (phase 1c)
#       'complex_nowl' complex geometry but no whitelist file in the table (discovery on the concatenated segments)
#       'none'    not a STAR barcode read (outside this layer)
# ------------------------------------------------------------------------------------------------------------
TSO_5P = "AAGCAGTGGTATCAACGCAGAGTACATGGG"
PROTOCOL_ROWS = {
    "10x-3p-v3": dict(mode="simple", cb=(0, 16), umi=(16, 12), wl="3M-february-2018.txt"),
    "10x-3p-v2": dict(mode="simple", cb=(0, 16), umi=(16, 10), wl="737K-august-2016.txt"),
    "10x-3p-v1": dict(mode="simple", cb=(0, 14), umi=(14, 10), wl="737K-april-2014.txt"),
    "10x-5p-v3": dict(mode="simple", cb=(0, 16), umi=(16, 12), wl="3M-february-2018.txt", adapter3p=TSO_5P),
    "10x-5p-v2": dict(mode="simple", cb=(0, 16), umi=(16, 10), wl="737K-august-2016.txt", adapter3p=TSO_5P),
    "10x-3p-v4": dict(mode="simple", cb=(0, 16), umi=(16, 12), wl="3M-february-2018.txt"),
    "10x-5p-v4": dict(mode="simple", cb=(0, 16), umi=(16, 12), wl="3M-february-2018.txt", adapter3p=TSO_5P),
    "10x-arc-gex": dict(mode="simple", cb=(0, 16), umi=(16, 12), wl="gex_737K-arc-v1.txt", adapter3p=TSO_5P),
    "cite-seq-gex": dict(mode="simple", cb=(0, 16), umi=(16, 12), wl="3M-february-2018.txt"),
    "quartzseq2": dict(mode="simple", cb=(0, 15), umi=(15, 8), wl="quartzseq2_bc.txt"),
    "dropseq": dict(mode="nowl", cb=(0, 12), umi=(12, 8)),
    "seqwell": dict(mode="nowl", cb=(0, 12), umi=(12, 8)),
    "celseq2": dict(mode="nowl", cb=(0, 6), umi=(6, 6)),
    "marsseq2": dict(mode="nowl", cb=(0, 7), umi=(7, 8)),
    "10x-visium": dict(mode="nowl", cb=(0, 16), umi=(16, 12), spots="tissue_positions.csv"),
    "sci-rna-seq3": dict(mode="samwl", cb=(24, 10), umi=(16, 8), wl="scirna3_rt_bc.txt"),
    "bd-rhapsody": dict(mode="complex", segs=["0_0_0_8", "0_21_0_29", "0_43_0_51"], umi="0_52_0_59",
                        wl=["bd_cls1.txt", "bd_cls2.txt", "bd_cls3.txt"]),
    "splitseq": dict(mode="complex", segs=["0_10_0_17", "0_33_0_40", "0_56_0_63"], umi="0_0_0_9", barcode_in_r2=True,
                     wl=["splitseq_bc_rd3.txt", "splitseq_bc_rd2.txt", "splitseq_bc_rd1.txt"]),
    # inDrop v1/v2 (protocol.h:105-112: BC1 8-11 bp, W1 linker, BC2 8 bp, UMI 6): decision of 2026-10-29, the W1-adapter route
    # (STAR's documented inDrop recipe). Today's singlet reads a fixed 10-base BC1 window, segs_current/umi_current.
    "indrop": dict(mode="complex", segs=["0_0_2_-1", "3_1_3_8"], umi="3_9_3_14", adapter="GAGTGATTGCTTGTGACGCCTT",
                   adapter_mm_max=1, segs_current=["0_0_0_9", "0_32_0_39"], umi_current="0_40_0_45",
                   wl=["indrop_bc1.txt", "indrop_bc2.txt"]),
    "microwell-seq": dict(mode="complex", segs=["0_0_0_5", "0_21_0_26", "0_42_0_47"], umi="0_48_0_53",
                          wl=["microwell_bc1_bare.txt"] * 3),
    "surecell": dict(mode="complex", segs=["0_0_0_5", "0_21_0_26", "0_42_0_47"], umi="0_51_0_58",
                     wl=["surecell_bc.txt"] * 3),
    "ddseq": dict(mode="complex_nowl", segs=["0_0_0_5", "0_9_0_14", "0_18_0_23"], umi="0_27_0_34"),
    "strtseq": dict(mode="none"), "10x-atac": dict(mode="none"), "dnbelab-c4": dict(mode="none"),
}

FEATURE_LIBRARIES = ("adt", "hto", "crispr")


def protocol_record_spec(tag: str, library: str = "gex") -> dict:
    """The row for a protocol, plus the library kind. library in adt/hto/crispr marks feature_read (no alignment)."""
    row = dict(PROTOCOL_ROWS[tag])
    row["feature_read"] = library in FEATURE_LIBRARIES
    row["library"] = library
    return row


def build_complex_layout(tag: str, seg_seqs: list, flags: str = "Exact") -> "ComplexLayout":
    """The ComplexLayout for a complex row of PROTOCOL_ROWS, with the given per-segment whitelist sequences."""
    row = PROTOCOL_ROWS[tag]
    segs = []
    for pos, seqs in zip(row["segs"], seg_seqs):
        sg = Segment(pos)
        build_segment_wl(sg, seqs)
        segs.append(sg)
    link_segments(segs)
    return ComplexLayout(segs, Segment(row["umi"]), flags_from_type(flags), row.get("adapter", "-"),
                         row.get("adapter_mm_max", 1))
