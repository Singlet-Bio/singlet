"""Tests of paired_model.py on handcrafted cases. Run: python -m pytest bench/gpu_paired -q  (needs numpy, numba, pytest).

Expected values are the hand-derived ones in cases.py's header and in the comments here; a failing test means the model and the
rule it cites disagree, so check the STAR line cited in docs/research-beta/paired-end-prep.md before changing either."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from cases import build_genome, params, rc  # noqa: E402
from paired_model import (Genome, Windows, build_read, map_pair, map_windows, mate_overlap_ok, mates_of,  # noqa: E402
                          proper_pair, quality_split, WC_gStart, WC_gEnd)


@pytest.fixture(scope="module")
def G():
    return build_genome()


def run(G, P, m1, m2):
    G.prepare(P)
    return map_pair(G, P, m1, m2)


def seq(G, chrom, a, b):
    return G.text[chrom][a:b]


def pair_transcripts(res):
    return [t for t in res.transcripts if t.iFrag == -1]


# ---- read combination, parameters, windows ----------------------------------------------------------------------------
def test_build_read_layout():
    rd = build_read("ACGTAC", "AAACCC")
    # mate 1, spacer 11, reverse complement of mate 2 (GGGTTT)
    assert list(rd.r[0]) == [0, 1, 2, 3, 0, 1, 11, 2, 2, 2, 3, 3, 3]
    assert rd.Lread == 13 and rd.readLength == (6, 6)
    assert list(rd.r[2]) == list(rd.r[1][::-1])  # Read1[2] is the reverse of the complement
    assert rd.r[1][6] == 11                      # the spacer is not complemented


def test_quality_split_marks_mates():
    rd = build_read("A" * 20, "C" * 30)
    s, _ = quality_split(rd.r[0], rd.Lread, 10, 12)
    assert s == [(0, 20, 0), (21, 30, 1)]


def test_window_geometry_atac_and_spliced(G):
    P = params("atac")
    G.prepare(P)
    assert (P.winBinNbits, P.winFlankNbins, P.winAnchorDistNbins) == (9, 4, 8)  # Genome_genomeLoad.cpp:426,442
    P = params("spliced")
    G.prepare(P)
    assert (P.winBinNbits, P.winFlankNbins, P.winAnchorDistNbins) == (16, 4, 9)
    # a mate gap limit alone redefines the geometry too (ISSUE note at :428 is about intronMax only)
    P = params("spliced", alignMatesGapMax=1000)
    G.prepare(P)
    assert P.winBinNbits == 8 and P.winFlankNbins == 4 and P.winAnchorDistNbins == 8


def test_windows_merge_within_anchor_distance(G):
    P = params("atac")
    G.prepare(P)
    W = Windows(G, P)
    W.create_extend(1000, 0)        # bin 1
    W.create_extend(3100, 0)        # bin 6: 5 bins <= winAnchorDistNbins (8) -> merged
    assert len(W.WC) == 1 and W.WC[0][WC_gStart] == 1 and W.WC[0][WC_gEnd] == 6
    W.create_extend(3100 + 512 * 20, 0)  # far: its own window
    assert len(W.WC) == 2
    W.create_extend(1000, 1)        # other strand: its own window
    assert len(W.WC) == 3


# ---- concordant pairs, overlap, protrusion, mate gap -----------------------------------------------------------------
def test_concordant_pair(G):
    P = params("spliced")
    res = run(G, P, seq(G, "chr1", 1000, 1100), rc(seq(G, "chr1", 1300, 1400)))
    assert res.unmapType == -1 and len(res.transcripts) == 1
    t = res.transcripts[0]
    assert t.iFrag == -1 and t.primaryFlag
    assert t.summary()["exons"] == [[0, 1000, 100, 0], [101, 1300, 100, 1]]
    assert list(t.canonSJ[:1]) == [-3]
    # 200 matches + ceil(log2(400) * -0.25 - 0.5) = -2
    assert t.maxScore == 198 and t.nMM == 0 and t.gLength == 400
    assert proper_pair(P, t, 201)


def test_overlapping_mates(G):
    P = params("spliced")
    res = run(G, P, seq(G, "chr1", 1000, 1100), rc(seq(G, "chr1", 1050, 1150)))
    t = res.transcripts[0]
    assert res.unmapType == -1 and len(res.transcripts) == 1
    assert t.summary()["exons"] == [[0, 1000, 100, 0], [101, 1050, 100, 1]]
    assert t.maxScore == 198 and t.gLength == 150  # ceil(log2(150) * -0.25 - 0.5) = -2


def test_mate_inside_the_other_is_rejected_without_protrusion_allowance(G):
    # mate 2 [1020,1080) lies inside mate 1 [1000,1100): LeftMateEnd 1100 > RightMateEnd 1080 (stitchWindowAligns.cpp:193)
    P = params("spliced")
    res = run(G, P, seq(G, "chr1", 1000, 1100), rc(seq(G, "chr1", 1020, 1080)))
    assert not pair_transcripts(res)


def test_protruding_mate_needs_alignEndsProtrude(G):
    # mate 2 = [990,1090): starts 10 before mate 1 [1000,1100) and ends 10 before it
    m1, m2 = seq(G, "chr1", 1000, 1100), rc(seq(G, "chr1", 990, 1090))
    P0 = params("spliced")                                    # 0 allowed: rejected at stitching (:352, -1000008)
    res0 = run(G, P0, m1, m2)
    assert not pair_transcripts(res0)
    assert res0.unmapType == 1                                # two single-mate transcripts of score 98 < 0.66 * 200
    P10 = params("spliced", alignEndsProtrudeNbasesMax=10)    # :352 accepts, :191 and :193 are met with equality
    res10 = run(G, P10, m1, m2)
    t = res10.transcripts[0]
    assert res10.unmapType == -1 and t.iFrag == -1
    assert t.summary()["exons"] == [[0, 1000, 100, 0], [101, 990, 100, 1]]
    assert t.gLength == 90 and t.maxScore == 198              # ceil(log2(90) * -0.25 - 0.5) = -2
    P9 = params("spliced", alignEndsProtrudeNbasesMax=9)      # one short: :193 rejects (LeftMateEnd 1100 > 1090 + 9)
    assert not pair_transcripts(run(G, P9, m1, m2))
    assert proper_pair(P10, t, 201)                           # ConcordantPair: always proper
    P10d = params("spliced", alignEndsProtrudeNbasesMax=10, alignEndsProtrudeConcordantPair=False)
    assert not proper_pair(P10d, t, 201)                      # DiscordantPair: flag 0x2 only if the ends do not protrude


@pytest.mark.parametrize("start2, accepted", [(3100, True), (3101, False)])
def test_mate_gap_at_the_limit_atac(G, start2, accepted):
    # alignMatesGapMax 2000: stitching refuses gBstart > end of mate 1 + 2000 (stitchAlignToTranscript.cpp:354)
    P = params("atac")
    res = run(G, P, seq(G, "chr1", 1000, 1100), rc(seq(G, "chr1", start2, start2 + 100)))
    if accepted:
        t = res.transcripts[0]
        assert res.unmapType == -1 and t.iFrag == -1
        assert t.gLength == 2200 and t.maxScore == 197          # ceil(log2(2200) * -0.25 - 0.5) = -3
    else:
        assert not pair_transcripts(res) and res.unmapType == 1


def test_no_mate_gap_limit_in_spliced_mode_but_one_in_atac(G):
    m1, m2 = seq(G, "chr3", 1000, 1100), rc(seq(G, "chr3", 101000, 101100))
    res = run(G, params("spliced"), m1, m2)
    t = res.transcripts[0]
    assert res.unmapType == -1 and t.iFrag == -1 and t.maxScore == 196      # ceil(log2(100100) * -0.25 - 0.5) = -4
    assert not pair_transcripts(run(G, params("atac"), m1, m2))


# ---- junctions in one mate -------------------------------------------------------------------------------------------
def test_junction_in_mate1(G):
    P = params("spliced")
    m1 = seq(G, "chr1", 1000, 1050) + seq(G, "chr1", 1500, 1550)
    m2 = rc(seq(G, "chr1", 1650, 1700))
    res = run(G, P, m1, m2)
    t = res.transcripts[0]
    assert res.unmapType == -1 and len(res.transcripts) == 1
    assert t.summary()["exons"] == [[0, 1000, 50, 0], [50, 1500, 50, 0], [101, 1650, 50, 1]]
    assert list(t.canonSJ[:2]) == [1, -3]                      # GT/AG, then the mate break
    assert t.nGap == 1 and t.maxScore == 148                   # 150 matches, scoreGap 0, ceil(log2(700) * -0.25 - 0.5) = -2
    assert [m[0] for m in mates_of(t)] == [0, 1]
    assert mates_of(t)[0][1] == [(1000, 50), (1500, 50)]


def test_junction_in_mate2(G):
    P = params("spliced")
    m1 = seq(G, "chr1", 960, 1010)
    m2 = rc(seq(G, "chr1", 1020, 1050) + seq(G, "chr1", 1500, 1530))
    res = run(G, P, m1, m2)
    t = res.transcripts[0]
    assert res.unmapType == -1 and len(res.transcripts) == 1 and t.iFrag == -1
    assert t.summary()["exons"] == [[0, 960, 50, 0], [51, 1020, 30, 1], [81, 1500, 30, 1]]
    assert list(t.canonSJ[:2]) == [-3, 1]
    assert t.maxScore == 108 and t.gLength == 570            # 110 matches + ceil(log2(570) * -0.25 - 0.5) = -2
    assert [m[0] for m in mates_of(t)] == [0, 1] and mates_of(t)[1][1] == [(1020, 30), (1500, 30)]


def test_overlapping_mates_with_consistent_junctions(G):
    P = params("spliced")
    m1 = seq(G, "chr1", 3000, 3030) + seq(G, "chr1", 3200, 3270)
    m2 = rc(seq(G, "chr1", 3010, 3030) + seq(G, "chr1", 3200, 3290))
    res = run(G, P, m1, m2)
    t = res.transcripts[0]
    assert res.unmapType == -1 and t.iFrag == -1
    assert [list(map(int, e)) for e in t.summary()["exons"]] == [[0, 3000, 30, 0], [30, 3200, 70, 0], [101, 3010, 20, 1], [121, 3200, 90, 1]]
    assert list(t.canonSJ[:3]) == [1, -3, 1]
    assert t.maxScore == 208                                   # 210 matches + ceil(log2(290) * -0.25 - 0.5) = -2


def test_overlapping_mates_with_inconsistent_junctions_lose_the_conflicting_exon(G):
    # both mates start at 3000 and splice from the same donor, to acceptors 3200 and 3300. The 4-exon pair is refused at
    # stitchWindowAligns.cpp:211 (exons[1].G 3200 != exons[3].G 3300); what STAR reports is the pair without mate 2's first
    # exon (soft clipped): 30 + 70 + 70 matches, ceil(log2(370) * -0.25 - 0.5) = -2
    P = params("spliced")
    m1 = seq(G, "chr1", 3000, 3030) + seq(G, "chr1", 3200, 3270)
    m2 = rc(seq(G, "chr1", 3000, 3030) + seq(G, "chr1", 3300, 3370))
    res = run(G, P, m1, m2)
    assert res.unmapType == -1 and len(res.transcripts) == 1
    t = res.transcripts[0]
    assert t.summary()["exons"] == [[0, 3000, 30, 0], [30, 3200, 70, 0], [131, 3300, 70, 1]]
    assert list(t.canonSJ[:2]) == [1, -3] and t.maxScore == 168


def test_mate_overlap_ok_kernel_directly():
    ex = np.zeros((20, 5), dtype=np.int64)
    sj = np.zeros(20, dtype=np.int64)
    # two single-block mates: [1000,1100) and [1050,1150), read 201 long, mate 2 starts at r = 101
    ex[0] = [0, 1000, 100, 0, -1]
    ex[1] = [101, 1050, 100, 1, -1]
    sj[0] = -3
    assert mate_overlap_ok(ex, sj, 2, 201, 0)
    ex[1] = [101, 990, 100, 1, -1]                             # mate 2 starts before mate 1
    assert not mate_overlap_ok(ex, sj, 2, 201, 0)              # :191 (1000 > 990 + 0)
    ex[1] = [101, 1100, 100, 1, -1]                            # abutting, no overlap
    assert mate_overlap_ok(ex, sj, 2, 201, 0)
    ex[1] = [101, 900, 50, 1, -1]                              # ends before mate 1 starts: negative insert size
    assert not mate_overlap_ok(ex, sj, 2, 201, 50)             # :180 holds whatever the allowance


# ---- discordant pairs ------------------------------------------------------------------------------------------------
def test_discordant_same_strand(G):
    res = run(G, params("spliced"), seq(G, "chr1", 1000, 1100), seq(G, "chr1", 1300, 1400))   # FF
    assert not pair_transcripts(res) and res.unmapType == 1
    assert {t.Str for t in res.transcripts} == {0, 1}          # each mate alone, on its own strand


def test_discordant_outie(G):
    res = run(G, params("spliced"), seq(G, "chr1", 1300, 1400), rc(seq(G, "chr1", 1000, 1100)))  # RF: mate 2 upstream
    assert not pair_transcripts(res) and res.unmapType == 1    # stitching returns -1000008 (:406)


def test_discordant_different_chromosomes(G):
    res = run(G, params("spliced"), seq(G, "chr1", 1000, 1100), rc(seq(G, "chr2", 1300, 1400)))
    assert not pair_transcripts(res) and res.unmapType == 1
    assert {t.Chr for t in res.transcripts} == {0, 1}


def test_multimapping_pair_has_one_primary(G):
    res = run(G, params("spliced"), seq(G, "chr1", 5000, 5100), rc(seq(G, "chr1", 5300, 5400)))
    assert res.unmapType == -1 and len(res.transcripts) == 2 and res.nW == 2
    assert all(t.iFrag == -1 and t.maxScore == 198 for t in res.transcripts)
    assert sum(t.primaryFlag for t in res.transcripts) == 1


def test_mismatch_near_the_mate_end_is_extended_through(G):
    m1 = list(seq(G, "chr1", 1000, 1100))
    m1[94] = "A" if m1[94] != "A" else "C"                     # a mismatch 5 bases from the end
    res = run(G, params("spliced"), "".join(m1), rc(seq(G, "chr1", 1300, 1400)))
    t = res.transcripts[0]
    assert t.iFrag == -1 and t.nMM == 1 and t.nMatch == 199
    assert t.summary()["exons"] == [[0, 1000, 100, 0], [101, 1300, 100, 1]]
    assert t.maxScore == 196  # seed 94 + extension 4 (5 matches - 1 mismatch) + 100 + ceil(log2(400) * -0.25 - 0.5) = -2


# ---- the window dump the kernel makes is enough to replay a read ------------------------------------------------------
def test_replay_from_windows_equals_full_run(G):
    P = params("spliced")
    G.prepare(P)
    m1 = seq(G, "chr1", 1000, 1050) + seq(G, "chr1", 1500, 1550)
    m2 = rc(seq(G, "chr1", 1650, 1700))
    full = map_pair(G, P, m1, m2)
    rd = build_read(m1, m2)
    WC = [[w["Str"], w["Chr"], 0, 0] for w in full.windows]
    again = map_windows(G, P, rd, WC, [w["WA"] for w in full.windows], [w["WlastAnchor"] for w in full.windows])
    assert full.summaries() == again.summaries()
