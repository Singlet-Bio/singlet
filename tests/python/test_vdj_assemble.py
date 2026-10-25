# SPDX-License-Identifier: MIT
"""singlet.vdj: reference anchors and assembly of a simulated 10x 5' V(D)J library with known junctions."""

import numpy as np
import pytest

pytest.importorskip("numba")

from singlet.adt import seq as S  # noqa: E402
from singlet.vdj import assemble as A  # noqa: E402
from singlet.vdj import reference as R  # noqa: E402

RNG = np.random.default_rng(11)
SENSE = [c for c, a in R.CODON.items() if a not in "*CFW"]


def codons(n):
    return "".join(RNG.choice(SENSE, n))


def rand(n):
    return "".join(RNG.choice(list("ACGT"), n))


def make_segments():
    segs = []
    for ch in ("TRA", "TRB"):
        for i in range(3):
            v = "ATG" + codons(95) + "TGT" + "GCCAGCAGC"  # Cys104 at len - 12
            segs.append(R.Segment(f"{ch}V{i + 1}", f"V{ch}{i}", ch, "V", v, len(v) - 12, 0))
        for i in range(3):
            j = codons(4) + "TTTGGC" + codons(1) + "GGG" + codons(8) + "AG"
            segs.append(R.Segment(f"{ch}J{i + 1}", f"J{ch}{i}", ch, "J", j, 12, 0))
        c = codons(100)
        segs.append(R.Segment(f"{ch}C", f"C{ch}", ch, "C", c, -1, -1))
    return segs


def test_anchors_from_sequence():
    segs = make_segments()
    for s in segs:
        if s.region == "V":
            assert R._v_anchor(s.seq, 0) == s.anchor
        if s.region == "J":
            assert R._j_anchor(s.seq) == (s.anchor, 0)


def recombine(segs, ch):
    vs = [s for s in segs if s.chain == ch and s.region == "V"]
    js = [s for s in segs if s.chain == ch and s.region == "J"]
    c = [s for s in segs if s.chain == ch and s.region == "C"][0]
    v, j = vs[RNG.integers(len(vs))], js[RNG.integers(len(js))]
    vt = v.seq[:v.anchor + 3 + RNG.integers(2, 8)]  # 3' trimming after Cys
    jt0 = RNG.integers(0, 6)
    jt = j.seq[jt0:]
    # in-frame N region
    base = len(vt) - v.anchor + (j.anchor - jt0)
    n = int(RNG.integers(2, 9))
    n += (-(base + n)) % 3
    while True:
        nreg = rand(n)
        junction = vt[v.anchor:] + nreg + jt[:j.anchor - jt0 + 3]
        if "*" not in R.translate(junction):
            break
    tx = rand(40) + vt + nreg + jt + c.seq
    return tx, junction, v.name, j.name


def rc(s):
    return R.revcomp(s)


def simulate(segs, n_cells=40, mols=(3, 8), pairs_per_mol=(4, 12), r2_len=150):
    cbs = S.decode(RNG.integers(0, 4 ** 16, n_cells * 2, dtype=np.uint64), 16)[:n_cells]
    r1, r2, truth = [], [], {}
    for cb in cbs:
        for ch in ("TRA", "TRB"):
            tx, junc, v, j = recombine(segs, ch)
            truth[(cb, ch)] = (junc, v, j)
            for _ in range(RNG.integers(*mols)):
                umi = rand(10)
                for _ in range(RNG.integers(*pairs_per_mol)):
                    f = int(RNG.integers(r2_len, len(tx) + 1))
                    s2 = rc(tx[f - r2_len:f])
                    s1 = cb + umi + "TTTCTTATATGGG" + tx[:111]
                    if RNG.random() < 0.3:  # a sequencing error
                        p = int(RNG.integers(0, len(s2)))
                        s2 = s2[:p] + RNG.choice([b for b in "ACGT" if b != s2[p]]) + s2[p + 1:]
                    r1.append(s1.encode())
                    r2.append(s2.encode())
    o = RNG.permutation(len(r1))
    return [r1[i] for i in o], [r2[i] for i in o], truth, cbs


@pytest.mark.parametrize("r2_len", [150, 91])
def test_assembly_recovers_junctions(r2_len):
    segs = make_segments()
    ref = A.Reference(segs)
    # short R2 reaches the junction from fewer fragmentation points: real 26x91 V(D)J runs are sequenced deeper
    r1, r2, truth, cbs = simulate(segs, r2_len=r2_len, pairs_per_mol=(4, 12) if r2_len >= 150 else (12, 30))
    wl, _ = S.encode_strings(cbs, 16)
    res = A.run_library([], [], ref=ref, cb_len=16, umi_len=10, whitelist_codes=np.sort(wl), chunks=[(r1, r2)],
                        params=A.Params(threads=2), log=lambda *a: None)
    df = res.contigs
    kept = df[df.kept]
    assert res.stats["orientation"]["r2_rc"]
    found = {(b[:-2], c): (n, v, j) for b, c, n, v, j in kept[["barcode", "chain", "cdr3_nt", "v_gene", "j_gene"]].itertuples(index=False)}
    hit = sum(found.get(k, (None,))[0] == t[0] for k, t in truth.items())
    assert hit >= 0.95 * len(truth), (hit, len(truth))
    genes = sum(found.get(k, (0, 0, 0))[1:] == t[1:] for k, t in truth.items())
    assert genes >= 0.9 * len(truth)
    assert kept.productive.mean() > 0.95
    # one kept contig per chain per cell (no error branches)
    assert kept.groupby(["barcode", "chain"]).size().max() == 1
