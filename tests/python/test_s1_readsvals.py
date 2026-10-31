# SPDX-License-Identifier: MIT
"""singlet.s1.readsvals: the reads matrix is rebuilt exactly from the count matrix and the stored reads - UMIs vector."""
import numpy as np
import pytest
import scipy.sparse as sp

from singlet.s1 import readsvals as RV
from singlet.s1.matrix import Csc


def _csc(dense, rows, cols):
    m = sp.csc_matrix(dense)
    m.sort_indices()
    return Csc(m=m.shape[0], n=m.shape[1], indptr=m.indptr.astype(np.int64), indices=m.indices.astype(np.uint32),
               data=m.data.astype(np.uint32), rownames=rows, colnames=cols)


def _pair(seed=0, big=False):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 4, size=(40, 15)) * (rng.random((40, 15)) < 0.3)
    extra = rng.integers(0, 6, size=counts.shape) * (counts > 0)
    if big:
        extra[extra > 0] += 70000
    rows = ["g%d" % i for i in range(40)]
    cols = ["BC%02d" % i for i in range(15)]
    return _csc(counts, rows, cols), _csc(counts + extra, rows, cols), cols


def _reads_expected(reads, sel):
    sub = reads.to_scipy()[:, sel].tocsc()
    sub.sort_indices()
    return sub


@pytest.mark.parametrize("big", [False, True])
def test_round_trip_is_exact(big):
    counts, reads, cols = _pair(1, big)
    called = {"BC01", "BC04", "BC05", "BC13"}
    blob = RV.encode(counts, reads, called)
    assert blob is not None
    got = RV.decode(blob, counts, called)
    sel = [i for i, c in enumerate(cols) if c in called]
    ref = _reads_expected(reads, sel)
    g = got.to_scipy().tocsc()
    assert got.colnames == [cols[i] for i in sel] and got.rownames == counts.rownames
    assert np.array_equal(g.indptr, ref.indptr) and np.array_equal(g.indices, ref.indices) and np.array_equal(g.data, ref.data)


def test_boolean_mask_and_set_agree():
    counts, reads, cols = _pair(2)
    called = {"BC00", "BC07"}
    mask = np.array([c in called for c in cols])
    assert RV.encode(counts, reads, called) == RV.encode(counts, reads, mask)


def test_no_called_columns_and_empty_deltas():
    counts, reads, _ = _pair(3)
    blob = RV.encode(counts, reads, set())
    assert RV.decode(blob, counts, set()).n == 0
    blob = RV.encode(counts, counts, {"BC02"})
    assert RV.decode(blob, counts, {"BC02"}).data.tolist() == _reads_expected(counts, [2]).data.tolist()


def test_member_is_not_written_when_the_matrices_do_not_line_up():
    counts, reads, cols = _pair(4)
    called = {"BC03", "BC09"}
    other = _csc(np.vstack([reads.to_scipy().toarray()[1:], reads.to_scipy().toarray()[:1]]), reads.rownames, cols)
    assert RV.encode(counts, other, called) is None  # other nonzero positions
    below = Csc(**{**reads.__dict__, "data": np.maximum(reads.data, 1) - 1 if reads.nnz else reads.data})
    below.data = counts.data.copy()
    below.data[: len(below.data) // 2] = 0
    assert RV.encode(counts, below, called) is None  # reads below UMIs / pattern differs


def test_a_member_for_other_count_matrix_is_refused():
    counts, reads, _ = _pair(5)
    blob = RV.encode(counts, reads, {"BC01", "BC02"})
    with pytest.raises(RV.ReadsValuesError):
        RV.decode(blob, counts, {"BC01", "BC03"})
    with pytest.raises(RV.ReadsValuesError):
        RV.decode(b"S1RD", counts, {"BC01"})


def test_modalities_registered():
    from singlet import bundle

    names = list(bundle.MODALITIES)
    assert names.index("exon_reads") == names.index("intron_counts") + 1
    assert bundle.MODALITIES["intron_reads"].members == ("intron_reads_delta.bin",)
