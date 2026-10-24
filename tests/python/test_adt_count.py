# SPDX-License-Identifier: MIT
"""singlet.adt counting: sequence codes, feature / barcode / UMI correction, and a simulated library with known truth."""

import numpy as np
import pytest

from singlet.adt import count as C
from singlet.adt import seq as S

RNG = np.random.default_rng(7)
B = np.array(list("ACGT"))


def rand_seq(n, L):
    return ["".join(B[RNG.integers(0, 4, L)]) for _ in range(n)]


def mutate(s, k=1):
    s = list(s)
    for p in RNG.choice(len(s), k, replace=False):
        s[p] = RNG.choice([b for b in "ACGT" if b != s[p]])
    return "".join(s)


def test_encode_roundtrip_and_order():
    seqs = rand_seq(50, 16)
    c, bad = S.encode_strings(seqs, 16)
    assert not bad.any()
    assert S.decode(c, 16) == seqs
    assert list(np.argsort(c)) == list(np.argsort(np.array(seqs)))
    _, bad = S.encode_strings(["ACGTN"], 5)
    assert bad[0] == 1


def test_neighbours_are_hamming1():
    s = "ACGTACGTAC"
    c, _ = S.encode_strings([s], 10)
    nb = S.decode(S.neighbours(c, 10)[0], 10)
    assert len(set(nb)) == 30
    assert all(sum(a != b for a, b in zip(s, x)) == 1 for x in nb)


def test_feature_index_exact_h1_ambiguous_and_n():
    panel = ["AAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCC", "AAAAAAAAAAAAAAC"]  # 0 and 2 are at distance 1
    fx = C.FeatureIndex(panel)
    q = ["CCCCCCCCCCCCCCC", "CCCCCCCGCCCCCCC", "AAAAAAAAAAAAAAG", "GGGGGGGGGGGGGGG", "CCCCNCCCCCCCCCC",
         "AAAAAAAAAAAAAAA"]
    idx, corr = fx.assign(S.as_bytes_matrix(q, 15))
    # AAAA..G is a neighbour of both panel 0 and panel 2 -> ambiguous
    assert idx.tolist() == [1, 1, -2, -1, 1, 0]
    assert corr.tolist() == [False, True, False, False, True, False]


def test_barcode_correction_posterior():
    wl = sorted(rand_seq(500, 16))
    wl_codes = np.sort(S.encode_strings(wl, 16)[0])
    a = wl[3]
    err = mutate(a)
    obs = [a, err, "N" + a[1:]]
    reads = np.array([1000, 5, 3])
    out, valid, ncorr = C.correct_barcodes(S.as_bytes_matrix(obs, 16), reads, wl_codes)
    assert valid.all() and ncorr == 2
    assert S.decode(out, 16) == [a, a, a]


def test_umi_correction_merges_into_larger():
    umis = ["AAAAAAAAAA", "AAAAAAAAAC", "TTTTTTTTTT", "AAAAAAAAAG", "CCCCCCCCCC", "CCCCCCCCCG"]
    codes, _ = S.encode_strings(umis, 10)
    group = np.array([0, 0, 0, 1, 2, 2])
    reads = np.array([10, 1, 4, 1, 2, 2])
    t = C.correct_umis(group, codes, reads, 10)
    # the 1-read neighbour merges; another group is untouched; a tie moves to the lexicographically larger UMI
    assert t.tolist() == [0, 0, 2, 3, 5, 5]


def test_simulated_library_recovers_truth():
    wl = sorted(set(rand_seq(3000, 16)))
    wl_codes = np.sort(S.encode_strings(wl, 16)[0])
    cells = wl[:200]
    gex = cells[:150] + wl[1000:1100]  # 50 cells missing from the GEX columns (they count as ambient)
    panel = rand_seq(12, 15)
    truth = RNG.poisson(8, size=(len(panel), len(cells)))
    pairs = []
    for j, cb in enumerate(cells):
        for f, fs in enumerate(panel):
            for _ in range(truth[f, j]):
                umi = rand_seq(1, 12)[0]
                for r in range(int(RNG.integers(1, 5))):
                    c1, u1, s2 = cb, umi, fs
                    if r > 0 and RNG.random() < 0.05:
                        c1 = mutate(c1)
                    if r > 0 and RNG.random() < 0.05:
                        u1 = mutate(u1)
                    if RNG.random() < 0.03:
                        s2 = mutate(s2)
                    pairs.append(((c1 + u1).encode(), ("NNNNNNNNNN" + s2 + "TTTT").encode()))
    RNG.shuffle(pairs)
    lc = C.count_library(None, None, cb_len=16, umi_len=12, offset=10, panel_sequences=panel,
                         whitelist_codes=wl_codes, gex_barcodes=[g + "-1" for g in gex], pairs=iter(pairs))
    M = lc.matrix.toarray()
    assert M.shape == (len(panel), len(gex))
    got = M[:, :150]
    want = truth[:, :150]
    # UMI collisions on random 12-mers are negligible; corrections may merge a handful of true molecules
    assert np.abs(got - want).sum() <= 0.01 * want.sum()
    assert M[:, 150:].sum() == 0
    assert abs(lc.ambient.sum() - truth[:, 150:].sum()) <= 0.01 * truth[:, 150:].sum()
    st = lc.stats
    assert st["reads"] == len(pairs)
    assert st["reads_valid_barcode"] > 0.99 * len(pairs)
    assert 0 < st["saturation"] < 1


def test_short_reads_are_skipped():
    keys, counts, st = C.collapse_reads(iter([(b"A" * 28, b"C" * 30), (b"A" * 20, b"C" * 30), (b"A" * 28, b"C" * 12)]),
                                        28, 10, 15)
    assert st == {"reads": 3, "reads_short": 2}
    assert counts.tolist() == [1]


@pytest.mark.parametrize("n", [0, 1])
def test_empty_inputs(n):
    t = C.correct_umis(np.zeros(n, np.uint64), np.zeros(n, np.uint64), np.ones(n, np.int64), 10)
    assert len(t) == n


def test_mixed_length_panel():
    panel = ["ACGTACGTACGTACG", "TTTTTTTTTTTTTT", "GGGGGGGGGGGGGGG"]  # one 14-nt control
    fx = C.FeatureIndex(panel)
    q = ["ACGTACGTACGTACG", "TTTTTTTTTTTTTTA", "TTTTTTTTTTTTTAC", "GGGGGGGGGGGGGGC", "CCCCCCCCCCCCCCC"]
    idx, corr = fx.assign(S.as_bytes_matrix(q, 15))
    assert idx.tolist() == [0, 1, 1, 2, -1]
    assert corr.tolist() == [False, False, True, True, False]
