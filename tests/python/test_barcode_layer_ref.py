"""Tests for src/engine_gpu/barcode_layer_ref.py: STARsolo's barcode rules on handcrafted reads (a few KB), one fixture
set per protocol class. Each assertion is the outcome STAR's code produces; the STAR lines are in the reference.
Run: python -P -m pytest tests/python/test_barcode_layer_ref.py -p no:cacheprovider --noconftest
"""
import pathlib
import random
import re
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "engine_gpu"))
import barcode_layer_ref as B  # noqa: E402

Q = "I" * 64  # Phred 40 everywhere


# ---------------------------------------------------------------- helpers
def simple(wl, mtype="1MM_multi_Nbase_pseudocounts", cb_len=None, umi_len=4, sam=False, nowl=False):
    cb_len = cb_len or len(wl[0])
    return B.SimpleLayout(1, cb_len, cb_len + 1, umi_len, B.flags_from_type(mtype),
                          None if nowl else B.Whitelist(wl, cb_len), sam, 0, sorted(wl))


def run_simple(lay, reads, quals=None):
    """Two passes like the engine: parse every read, build the exact histogram, resolve each."""
    rbs = [B.get_cb_umi_simple(r, (quals[i] if quals else Q[:len(r)]), lay) for i, r in enumerate(reads)]
    counts = B.exact_counts(rbs, len(lay.wl), lay.flags) if lay.wl is not None else None
    return rbs, counts, [B.resolve_cb(rb, counts, lay.flags) for rb in rbs] if counts is not None else None


# ---------------------------------------------------------------- nucleotide helpers
def test_conv_n_positions():
    assert B.conv("ACGT") == (0b00011011, -1)
    assert B.conv("acgt") == (0b00011011, -1)
    assert B.conv("ANGT")[1] == 1 and B.conv("ANGT")[0] == 0b00001011   # N reads as A
    assert B.conv("ANNT")[1] == -2


def test_umi_key_three_bit_n_is_its_own_base():
    assert B.pack3("ACGTN") == (0 << 12) | (1 << 9) | (2 << 6) | (3 << 3) | 4
    assert B.pack3("AAAAN") != B.pack3("AAAAA")


# ---------------------------------------------------------------- CB_UMI_Simple (10x class)
WL = ["AAAAAAAA", "AAAAAAAC", "CCCCCCCC", "GGGGGGGG", "TTTTTTTT"]


def test_simple_exact_and_unique_1mm():
    lay = simple(WL)
    rbs, counts, res = run_simple(lay, ["CCCCCCCC" + "ACGT", "CCCCCCGC" + "ACGT", "GGGGGGGG" + "ACGT"])
    assert [rb.cb_match for rb in rbs] == [0, 1, 0]
    assert [r[0] for r in res] == [2, 2, 3]
    assert res[1][1] == "mm1"                                  # pseudocount type: no exact support needed


def test_simple_no_match_two_n_and_exact_type():
    lay = simple(WL)
    rbs, _, _ = run_simple(lay, ["ACACACAC" + "ACGT", "CNCNCCCC" + "ACGT", "NNNNNNNN" + "ACGT"])
    assert [rb.cb_match for rb in rbs] == [-1, -2, -2]
    ex = simple(WL, "Exact")
    rb = B.get_cb_umi_simple("CCCCCCGC" + "ACGT", Q, ex)
    assert rb.cb_match == -1                                   # exact type: a 1MM read is not matched (:42)


def test_simple_one_n_single_hit_and_nbase_multi():
    lay = simple(WL)
    rb = B.get_cb_umi_simple("CCCNCCCC" + "ACGT", Q, lay)
    assert (rb.cb_match, rb.match.ind) == (1, [2])             # the N resolves one way
    rb = B.get_cb_umi_simple("AAAAAAAN" + "ACGT", Q, lay)      # A or C at the N: both in the list
    assert (rb.cb_match, rb.match.ind) == (2, [0, 1])
    rb = B.get_cb_umi_simple("AAAAAAAN" + "ACGT", Q, simple(WL, "1MM_multi_pseudocounts"))
    assert rb.cb_match == -3                                   # without Nbase a second N-hit is rejected (:53)
    assert B.get_cb_umi_simple("AAAAAAAN" + "ACGT", Q, simple(WL, "1MM")).cb_match == -3


def test_simple_1mm_multi_candidate_order_and_quality():
    # AAAAAAAG is one change from AAAAAAAA (pos 7) and AAAAAAAC (pos 7): ii=0 (last base), jj 1..3
    lay = simple(WL)
    rb = B.get_cb_umi_simple("AAAAAAAG" + "ACGT", Q[:7] + "5" + Q[8:], lay)
    assert rb.match.ind == [0, 1] and rb.match.qual == ["5", "5"]


def test_posterior_counts_decide_and_pseudocounts():
    lay = simple(WL)
    ambiguous = "AAAAAAAG" + "ACGT"
    # AAAAAAAA seen 99 times exactly, AAAAAAAC never: with +1 pseudocounts 100 vs 1 -> 100/101 = 0.990 >= 0.975
    reads = ["AAAAAAAA" + "ACGT"] * 99 + [ambiguous]
    rbs, counts, res = run_simple(lay, reads)
    assert counts[0] == 100 and counts[1] == 1
    assert res[-1] == (0, "posterior")
    # 9 exact vs none: 10/11 = 0.909 < 0.975 -> rejected
    rbs, counts, res = run_simple(lay, ["AAAAAAAA" + "ACGT"] * 9 + [ambiguous])
    assert res[-1] == (-1, "posterior_reject")


def test_posterior_quality_can_override_counts():
    # read differs from candidate A at position 3 (Q10) and from candidate B at position 0 (Q40, capped to 33)
    wl = ["AAAAAAAA", "CAAGAAAA"]
    lay = simple(wl)
    read = "AAAGAAAA"
    qual = "I" + "II" + "+" + "IIII" + "IIII"   # '+' is Q10 at index 3
    exact = ["AAAAAAAA"] * 20 + ["CAAGAAAA"] * 50
    rbs = [B.get_cb_umi_simple(r + "ACGT", Q, lay) for r in exact]
    rbs.append(B.get_cb_umi_simple(read + "ACGT", qual[:8] + "IIII", lay))
    counts = B.exact_counts(rbs, 2, lay.flags)
    assert (counts[0], counts[1]) == (21, 51)
    cb, why = B.resolve_cb(rbs[-1], counts, lay.flags)
    # pin_A = 21*10^-1 = 2.1, pin_B = 51*10^-3.3 = 0.0256 -> A holds 0.988 of the mass despite fewer reads
    assert rbs[-1].cb_match == 2 and (cb, why) == (0, "posterior")


def test_one_exact_rule_for_plain_1mm_type():
    lay = simple(WL, "1MM_multi")
    rbs, counts, res = run_simple(lay, ["CCCCCCGC" + "ACGT"])
    assert res[0] == (-1, "mm1_no_exact")                      # no exact read supports CCCCCCCC (:38)
    rbs, counts, res = run_simple(lay, ["CCCCCCCC" + "ACGT", "CCCCCCGC" + "ACGT"])
    assert res[1] == (2, "mm1")


def test_umi_checks_and_exact_histogram_excludes_bad_umi():
    lay = simple(WL)
    rbs, counts, _ = run_simple(lay, ["CCCCCCCC" + "ACGN", "CCCCCCCC" + "AAAA", "CCCCCCCC" + "TTTT",
                                      "CCCCCCCC" + "ACGT"])
    assert [rb.cb_match for rb in rbs] == [-23, -24, -24, 0]
    assert counts[2] == 2                                      # one good exact read + the pseudocount
    # a UMI that is not a homopolymer but has no N is fine
    assert B.check_umi("ACAC", 4) == (0, 0b00010001)


def test_short_barcode_read_is_padded_with_n():
    lay = simple(WL)
    rb = B.get_cb_umi_simple("CCCCCCCC" + "AC", "I" * 10, lay)  # UMI cut short: padded with N
    assert rb.umi_seq == "ACNN" and rb.cb_match == -23
    lay_fixed = simple(WL); lay_fixed.bl = 12
    with pytest.raises(ValueError):
        B.get_cb_umi_simple("CCCCCCCC" + "AC", "I" * 10, lay_fixed)


def test_simple_record_fields_5prime():
    lay = simple(WL)
    _, counts, _ = run_simple(lay, ["CCCCCCCC" + "ACGT"])
    rec = B.simple_record("CCCCCCGC" + "ACGT", Q, lay, counts, clip5=0, adapter3p=B.TSO_5P)
    assert (rec.cell_key, rec.umi_len, rec.adapter3p) == (2, 4, B.TSO_5P) and not rec.feature_read
    assert rec.umi_key == B.pack3("ACGT")


def test_whitelist_n_at_position_zero_is_skipped_other_n_read_as_a():
    w = B.Whitelist(["NCCC", "CNCC"])
    assert len(w) == 1 and int(w.keys[0]) == B.conv("CACC")[0]


# ---------------------------------------------------------------- no whitelist (Drop-seq class)
def test_nowl_any_n_free_barcode_and_discovery_order():
    lay = simple([], nowl=True, cb_len=12, umi_len=8)
    lay.wl_strings = None
    rb = B.get_cb_umi_simple("ACGTACGTACGT" + "ACGTACGT", Q, lay)
    assert (rb.cb_match, rb.match.ind) == (0, [B.conv("ACGTACGTACGT")[0]])
    assert B.get_cb_umi_simple("ACGTACGTACGN" + "ACGTACGT", Q, lay).cb_match == -2
    raw = {"AAA": 5, "CCC": 9, "GGG": 5, "TTT": 4}
    assert B.discover_barcodes(raw, 5, stream=True) == ["CCC", "AAA", "GGG"]   # ties: smaller barcode first
    idx = {b: i for i, b in enumerate(B.discover_barcodes(raw, 5, True))}
    assert B.cell_key_no_wl("GGG", idx) == 2 and B.cell_key_no_wl("TTT", idx) == -1 and B.cell_key_no_wl("GNG", idx) == -2


def test_samtag_out_corrected_cb_and_no_umi_check():
    lay = simple(WL, "Exact", sam=True)
    rb = B.get_cb_umi_simple("CCCCCCCC" + "AAAA", Q, lay)
    assert (rb.cb_match, rb.cb_seq_corrected) == (0, "CCCCCCCC")     # poly-A UMI is NOT rejected here
    rb = B.get_cb_umi_simple("CCCCCCGC" + "ACGT", Q, lay)
    assert (rb.cb_match, rb.cb_seq_corrected) == (-1, "-")
    lay_nowl = simple([], "Exact", sam=True, nowl=True, cb_len=8)
    assert B.get_cb_umi_simple("ACGTACGT" + "ACGT", Q, lay_nowl).cb_seq_corrected == "ACGTACGT"


# ---------------------------------------------------------------- CB_UMI_Complex (BD, SPLiT, inDrop, ...)
def seg(pos, seqs):
    s = B.Segment(pos)
    B.build_segment_wl(s, seqs)
    return s


def complex_layout(segs, umi, flags="Exact", adapter="-"):
    B.link_segments(segs)
    return B.ComplexLayout(segs, B.Segment(umi), B.flags_from_type(flags), adapter)


BD1 = ["AAAAAAAAA", "CCCCCCCCC", "GGGGGGGGG"]
BD2 = ["ACACACACA", "CACACACAC", "TTTTTTTTT"]
BD3 = ["AGAGAGAGA", "GAGAGAGAG", "TCTCTCTCT"]
L1, L2 = "ACTGGCCTGCGA", "GGTAGCGGTGACA"


def bd_read(b1, b2, b3, umi):
    return b1 + L1 + b2 + L2 + b3 + umi


def bd(flags="Exact"):
    return complex_layout([seg("0_0_0_8", BD1), seg("0_21_0_29", BD2), seg("0_43_0_51", BD3)], "0_52_0_59", flags)


def test_complex_bd_exact_composite_index():
    lay = bd()
    r = bd_read("CCCCCCCCC", "TTTTTTTTT", "GAGAGAGAG", "ACGTACGT")
    assert len(r) == 60
    rb = B.get_cb_umi_complex(r, Q, lay)
    # segment sizes 3,3,3: factors 1,3,9 -> indexes (1,2,1) -> 1 + 3*2 + 9*1 = 16
    assert (rb.cb_match, rb.cb_index, rb.cb_seq, rb.umi_seq) == (0, 16, "CCCCCCCCC_TTTTTTTTT_GAGAGAGAG", "ACGTACGT")
    assert B.complex_record(r, Q, lay).cell_key == 16


def test_complex_exact_type_does_not_correct_but_1mm_does_with_limits():
    r1mm = bd_read("CCCCCCCCA", "TTTTTTTTT", "GAGAGAGAG", "ACGTACGT")
    assert B.get_cb_umi_complex(r1mm, Q, bd("Exact")).cb_match == -1
    rb = B.get_cb_umi_complex(r1mm, Q, bd("1MM"))
    assert (rb.cb_match, rb.cb_index) == (1, 1 + 3 * 2 + 9 * 1)
    two = bd_read("CCCCCCCCA", "TTTTTTTTA", "GAGAGAGAG", "ACGTACGT")  # a mismatch in two segments
    assert B.get_cb_umi_complex(two, Q, bd("1MM")).cb_match == -12
    nread = bd_read("CCCCCCCCN", "TTTTTTTTT", "GAGAGAGAG", "ACGTACGT")
    assert B.get_cb_umi_complex(nread, Q, bd("Exact")).cb_match == -1     # Exact: an N is simply no exact match
    assert B.get_cb_umi_complex(nread, Q, bd("1MM")).cb_match == 1
    # 1MM type with two barcodes' worth of ambiguity (-3) inside one segment
    amb = complex_layout([seg("0_0_0_3", ["AAAA", "AAAC"]), seg("0_4_0_7", ["CCCC"])], "0_8_0_11", "1MM")
    assert B.get_cb_umi_complex("AAAG" + "CCCC" + "ACGT", Q, amb).cb_match == -3


def test_complex_umi_checks_are_weaker_poly_a_only():
    lay = bd()
    ok = lambda u: B.get_cb_umi_complex(bd_read("CCCCCCCCC", "TTTTTTTTT", "GAGAGAGAG", u), Q, lay)
    assert ok("AAAAAAAA").cb_match == -24
    assert ok("CCCCCCCC").cb_match == 0 and ok("TTTTTTTT").cb_match == 0   # STAR's homopolymer table is all 0 here
    rb = ok("ACGTACGN")
    assert rb.cb_match == -23 and rb.cb_seq == "CCCCCCCCC_TTTTTTTTT_GAGAGAGAG"   # segments still extracted


def test_complex_short_read_edge_cases():
    lay = bd()
    full = bd_read("CCCCCCCCC", "TTTTTTTTT", "GAGAGAGAG", "ACGTACGT")
    rb = B.get_cb_umi_complex(full[:59], Q, lay)                 # pos[1] == size: allowed, UMI is one base short
    assert rb.cb_match == 0 and rb.umi_seq == "ACGTACG"
    assert B.get_cb_umi_complex(full[:58], Q, lay).cb_match == -22       # UMI beyond the read
    short = B.get_cb_umi_complex(full[:30], Q, lay)
    assert short.cb_match == -22                                  # UMI is checked first
    lay2 = complex_layout([seg("0_0_0_8", BD1), seg("0_21_0_29", BD2)], "0_9_0_12")
    assert B.get_cb_umi_complex(full[:25], Q, lay2).cb_match == -11      # segment 2 cut short (4 bases < minLen 9)


def test_complex_anchors_and_adapter_inDrop_style():
    w1 = "GAGTGATTGCTTGTGACGCCTT"
    bc1 = ["AACCGGTT", "AACCGGTTA", "AACCGGTTAC"]                 # BC1 of 8, 9, 10 bases
    bc2 = ["ACGTACGT", "TGCATGCA"]
    s1 = seg("0_0_2_-1", bc1)                                     # read start .. base before the adapter
    s2 = seg("3_1_3_8", bc2)                                      # the 8 bases after the adapter
    umi = "3_9_3_14"
    lay = complex_layout([s1, s2], umi, "Exact", w1)
    r9 = "AACCGGTTA" + w1 + "TGCATGCA" + "ACGTAC"
    rb = B.get_cb_umi_complex(r9, Q, lay)
    # BC1 length 9 is index 1 of the length-9 list but wlAdd[9] = 1 (one 8-mer before it); BC2 index 1, factor 3
    assert rb.cb_match == 0 and rb.cb_seq == "AACCGGTTA_TGCATGCA"
    assert rb.cb_index == 1 + 3 * 1                          # wlAdd[9] = 1; BC2 index 1 times factor 3
    r10 = "AACCGGTTAC" + w1 + "ACGTACGT" + "ACGTAC"
    rb = B.get_cb_umi_complex(r10, Q, lay)
    assert rb.cb_index == (2 + 0) + 3 * 0 and rb.umi_seq == "ACGTAC"       # lengths 8,9,10 -> wlAdd[10] = 2
    one_mm = r9[:12] + "A" + r9[13:]                                       # 1 adapter mismatch: Nmax = 1 passes
    assert B.get_cb_umi_complex(one_mm, Q, lay).cb_match == 0
    two_mm = r9[:12] + "AA" + r9[14:]
    assert B.get_cb_umi_complex(two_mm, Q, lay).cb_match == -21
    # an adapter with every base wrong: distance == len(query), position stays 0
    assert B.local_align_hamming_dist("T" * 40, "GAGA") == (4, 0)
    assert B.local_align_hamming_dist("AC", "ACGT") == (3, 0)             # text shorter than the query: len(text)+1
    # read-end anchor
    sr = seg("1_-5_1_0", ["ACGTAC"[0:6]])                                 # last 6 bases
    assert B.extract_barcode(sr, "TTTTACGTAC", Q[:10], 0, 0) == (True, "ACGTAC", Q[:6])


def test_complex_segments_with_n_at_position_zero_in_whitelist_skipped():
    s = seg("0_0_0_3", ["NCCC", "ACCC"])
    assert s.total_size == 1


# ---------------------------------------------------------------- Visium spot list
def test_visium_positions_parse_and_table():
    txt = ("barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\r\n"
           "AAAC,1,3,5,1200.5,800.25\r\nbad,row\r\n\r\nAAAC,0,4,6,1,2\nGGGT,1,0,0,0.5,7e1\n")
    c = B.load_positions(txt)
    assert c["AAAC"] == (4, 6, 1.0, 2.0, False)            # a repeated barcode keeps the last row
    assert c["GGGT"] == (0, 0, 0.5, 70.0, True) and "bad" not in c
    assert B.spot_table(c, ["GGGT", "ZZZZ"]) == [("GGGT", 0, 0, 0.5, 70.0, 1), ("ZZZZ", 0, 0, 0, 0, 0)]
    nohdr = B.load_positions("AAAC,1,3,5,1.5,2.5\n")       # headerless: first line is data
    assert nohdr["AAAC"][0] == 3
    with pytest.raises(ValueError):
        B.load_positions("AAAC,1,x,5,1,1\n")               # std::stoi throws


# ---------------------------------------------------------------- feature barcodes
TAGS = ["AACAAGACCCCTCCT", "TGCATGTCATCGGTG", "AACAAGACCCCTCCA"]       # tags 0 and 2 differ in the last base


def test_adt_exact_hamming1_ambiguous_and_offsets():
    m = B.AdtMatcher(TAGS)
    assert m.match("AACAAGACCCCTCCT") == 0
    assert m.match("AACAAGACCCCTCCA") == 2
    assert m.match("AACAAGACCCCTCCG") == -2                   # 1 from both tag 0 and 2 -> ambiguous (-2, not -1)
    assert m.h1["AACAAGACCCCTCCG"] == -2
    assert m.match("TGCATGTCATCGGTC") == 1                    # unique Hamming-1
    assert m.match("TTTTT" + "TGCATGTCATCGGTG") == 1          # offset 5 is tried
    assert m.match("TTTTTT" + "TGCATGTCATCGGTG") == -1        # offset 6 is not
    assert m.match("AAC") == -1
    # ambiguity at offset 0 returns at once even though offset 1 holds an exact tag
    assert m.match("AACAAGACCCCTCCG" + "T") == -2 and m.match("A" + "TGCATGTCATCGGTG") == 1


def test_adt_cell_must_be_exact_and_umi_dedup_is_by_string():
    m = B.AdtMatcher(TAGS)
    cells = {"ACGTACGTACGTACGT": 7}
    r1 = "ACGTACGTACGTACGT" + "AAAAAAAAAAAA"
    recs = [B.adt_record(r1, "AACAAGACCCCTCCT", 12, cells, m),
            B.adt_record(r1, "AACAAGACCCCTCCT", 12, cells, m),                  # same UMI: one molecule
            B.adt_record(r1[:-1] + "C", "AACAAGACCCCTCCT", 12, cells, m),         # UMI 1 apart: its own molecule here
            B.adt_record("ACGTACGTACGTACGA" + "AAAAAAAAAAAA", "AACAAGACCCCTCCT", 12, cells, m)]   # cell 1MM: dropped
    assert recs[3] is None
    assert B.adt_count([r for r in recs if r]) == {(7, 0): 2}


def test_crispr_guide_substring_and_ambiguity():
    guides = ["GACGGATCGGGAGATCTCCC", "TTTTGGGGCCCCAAAATTTT", "GGGAGATC"]   # guide 2 is inside guide 0
    hit, all_hits = B.match_guide("nnGACGGATCGGGAGATCTCCCnn", guides)
    assert (hit, all_hits) == (0, [0, 2])                     # CPU winner depends on hash order: not exactly reproducible
    assert B.match_guide("TTTTGGGGCCCCAAAATTTT", guides) == (1, [1])
    assert B.match_guide("ACGT", guides) == (-1, [])
    assert B.match_guide("AAGGGAGATCAA", guides + ["AAGGGAGATCAA"]) == (2, [2, 3])
    assert B.match_guide("CCCC", ["CCCC", "CCCC"]) == (1, [1])      # a repeated sequence keeps the last id


# ---------------------------------------------------------------- numba batch == plain Python
def test_numba_batch_equals_reference_on_random_reads():
    rng = random.Random(7)
    wl = sorted({"".join(rng.choice("ACGT") for _ in range(16)) for _ in range(300)})
    hot = wl[0:60:3]
    for i in range(0, 60, 3):                                 # neighbours one base apart, so multi-matches exist
        s = list(wl[i]); s[15] = "ACGT"[("ACGT".index(s[15]) + 1) % 4]; wl.append("".join(s))
    wl = sorted(set(wl))
    lay = simple(wl, cb_len=16, umi_len=4)
    reads, quals = [], []
    pool = wl + hot * 40                                      # hot barcodes: their neighbour is rare, so the posterior accepts
    for _ in range(6000):
        s = list(rng.choice(pool))
        for _ in range(rng.choice([0, 0, 0, 1, 1, 2])):
            s[rng.randrange(16)] = rng.choice("ACGTN")
        reads.append("".join(s))
        quals.append("".join(rng.choice("#+5?I") for _ in range(16)))
    full = [r + "ACGT" for r in reads]
    rbs = [B.get_cb_umi_simple(f, q + "IIII", lay) for f, q in zip(full, quals)]
    counts = B.exact_counts(rbs, len(lay.wl), lay.flags)
    ref = [B.resolve_cb(rb, counts, lay.flags)[0] if rb.cb_match >= 0 else -1 for rb in rbs]
    cell, code, hist = B.simple_batch(reads, quals, lay.wl)
    assert np.array_equal(hist, counts)
    assert list(cell) == ref
    assert [int(c) for c in code if c == 2] and any(c == -4 for c in code)    # both multi outcomes occur
    nplain = [rb.cb_match for rb in rbs]
    assert all((nc == pc) or (pc >= 2 and nc in (2, -4)) for nc, pc in zip(code, nplain))


# ---------------------------------------------------------------- protocol table matches the repo's tables
def _protocol_h_rows():
    txt = (ROOT / "include" / "singlet" / "fq" / "protocol.h").read_text(encoding="utf-8")
    rows = {}
    for m in re.finditer(r'\{"([a-z0-9\-]+)",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"([^"]*)"', txt):
        rows[m.group(1)] = dict(r1=int(m.group(3)), bc_off=int(m.group(4)), bc_len=int(m.group(5)),
                                umi_off=int(m.group(6)), umi_len=int(m.group(7)), wl=m.group(8))
    return rows


def test_protocol_rows_agree_with_protocol_h():
    h = _protocol_h_rows()
    assert len(h) >= 25
    for tag, row in B.PROTOCOL_ROWS.items():
        assert tag in h, tag
        if row["mode"] in ("simple", "nowl", "samwl"):
            assert row["cb"] == (h[tag]["bc_off"], h[tag]["bc_len"]), tag
            assert row["umi"] == (h[tag]["umi_off"], h[tag]["umi_len"]), tag
            assert row.get("wl", "") == h[tag]["wl"], tag
    for tag in ("bd-rhapsody", "splitseq", "indrop", "microwell-seq", "surecell", "ddseq", "sci-rna-seq3"):
        assert B.PROTOCOL_ROWS[tag]["mode"] in ("complex", "complex_nowl", "samwl")


def test_complex_positions_agree_with_encoder():
    txt = (ROOT / "include" / "singlet" / "fq" / "sra_encoder.h").read_text(encoding="utf-8")
    for tag in ("bd-rhapsody", "splitseq", "indrop", "microwell-seq", "surecell", "ddseq"):
        blk = next(txt[m.start():m.start() + 700] for m in re.finditer(re.escape(f'canonical_tag == "{tag}"'), txt)
                   if "soloCBposition" in txt[m.start():m.start() + 400])
        segs = re.search(r'soloCBposition\\": \[(.*?)\]', blk).group(1)
        segs = re.findall(r'\\"(\d_\d+_\d_\d+)\\"', segs)
        umi = re.search(r'soloUMIposition\\": \\"(\d_\d+_\d_\d+)\\"', blk).group(1)
        assert segs == B.PROTOCOL_ROWS[tag]["segs"] and umi == B.PROTOCOL_ROWS[tag]["umi"], tag
    r = B.protocol_record_spec("10x-5p-v3", "adt")
    assert r["feature_read"] and r["adapter3p"] == B.TSO_5P
    assert not B.protocol_record_spec("10x-3p-v3")["feature_read"]
