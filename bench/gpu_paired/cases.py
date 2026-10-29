"""Handcrafted paired-end cases on a synthetic genome (deterministic: numpy seed 7). Everything is derived from `build_genome()`,
so the fixtures stay a few hundred kB at most (fixtures/cases.json holds the same genome as text for the CUDA harness's unit mode).

Layout (chr1 unless noted); every expected score below was worked out from the STAR rules, not recorded from the model:
  1000-1100 / 1300-1400 / 1050-1150 ...  plain pairs (concordant, overlapping, protruding, at the mate-gap limit)
  S1: exon [1000,1050) GT..AG intron [1050,1500) exon [1500,1700)           one junction, GT/AG
  S2: exon [3000,3030), GT at 3030, AG ending at 3200 and at 3300            two junctions from the same donor
  chr2 [5000,5400) is a copy of chr1 [5000,5400)                            a multimapper
  chr3 is 150 kb, for mates 100 kb apart
"""
import numpy as np

from paired_model import Genome, Params, revcomp_str


def build_genome() -> Genome:
    rng = np.random.default_rng(7)
    alpha = np.array(list("ACGT"))

    def rnd(n):
        return "".join(alpha[rng.integers(0, 4, n)])

    c1, c2, c3 = list(rnd(8000)), list(rnd(8000)), rnd(150000)

    def put(c, pos, s):
        c[pos:pos + len(s)] = list(s)
    put(c1, 1050, "GT")      # S1 donor, intron [1050,1500)
    put(c1, 1498, "AG")      # S1 acceptor
    put(c1, 3030, "GT")      # S2 donor
    put(c1, 3198, "AG")      # S2 acceptor 1 (exon starts 3200)
    put(c1, 3298, "AG")      # S2 acceptor 2 (exon starts 3300)
    c2[5000:5400] = c1[5000:5400]
    return Genome({"chr1": "".join(c1), "chr2": "".join(c2), "chr3": c3})


def params(kind="spliced", **kw) -> Params:
    P = {"spliced": Params.spliced, "atac": Params.atac}[kind]()
    for k, v in kw.items():
        setattr(P, k, v)
    return P


def rc(s):
    return revcomp_str(s)


def case_list(G):
    """(name, params kind, params overrides, mate 1, mate 2): the handcrafted cases of test_paired_model.py, for the CUDA harness's unit mode"""
    t = G.text
    s = lambda c, a, b: t[c][a:b]  # noqa: E731
    return [
        ("concordant", "spliced", {}, s("chr1", 1000, 1100), rc(s("chr1", 1300, 1400))),
        ("overlap50", "spliced", {}, s("chr1", 1000, 1100), rc(s("chr1", 1050, 1150))),
        ("inside", "spliced", {}, s("chr1", 1000, 1100), rc(s("chr1", 1020, 1080))),
        ("protrude10_0", "spliced", {}, s("chr1", 1000, 1100), rc(s("chr1", 990, 1090))),
        ("protrude10_10", "spliced", {"alignEndsProtrudeNbasesMax": 10}, s("chr1", 1000, 1100), rc(s("chr1", 990, 1090))),
        ("protrude10_9", "spliced", {"alignEndsProtrudeNbasesMax": 9}, s("chr1", 1000, 1100), rc(s("chr1", 990, 1090))),
        ("gap_at_limit", "atac", {}, s("chr1", 1000, 1100), rc(s("chr1", 3100, 3200))),
        ("gap_over_limit", "atac", {}, s("chr1", 1000, 1100), rc(s("chr1", 3101, 3201))),
        ("far_spliced", "spliced", {}, s("chr3", 1000, 1100), rc(s("chr3", 101000, 101100))),
        ("far_atac", "atac", {}, s("chr3", 1000, 1100), rc(s("chr3", 101000, 101100))),
        ("junction_m1", "spliced", {}, s("chr1", 1000, 1050) + s("chr1", 1500, 1550), rc(s("chr1", 1650, 1700))),
        ("junction_m2", "spliced", {}, s("chr1", 960, 1010), rc(s("chr1", 1020, 1050) + s("chr1", 1500, 1530))),
        ("junction_consistent", "spliced", {}, s("chr1", 3000, 3030) + s("chr1", 3200, 3270),
         rc(s("chr1", 3010, 3030) + s("chr1", 3200, 3290))),
        ("junction_inconsistent", "spliced", {}, s("chr1", 3000, 3030) + s("chr1", 3200, 3270),
         rc(s("chr1", 3000, 3030) + s("chr1", 3300, 3370))),
        ("discordant_ff", "spliced", {}, s("chr1", 1000, 1100), s("chr1", 1300, 1400)),
        ("discordant_outie", "spliced", {}, s("chr1", 1300, 1400), rc(s("chr1", 1000, 1100))),
        ("discordant_chr", "spliced", {}, s("chr1", 1000, 1100), rc(s("chr2", 1300, 1400))),
        ("multimapper", "spliced", {}, s("chr1", 5000, 5100), rc(s("chr1", 5300, 5400))),
    ]
