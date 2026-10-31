# SPDX-License-Identifier: MIT
"""Reads-per-UMI members are packed for the called cells only (bundle._called_only) and registered as modalities."""
import gzip

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("singlet._pz")

from singlet import bundle  # noqa: E402
from singlet._pz import read_1pz, write_1pz  # noqa: E402


def _write(path, mat, rows, cols):
    mat = sp.csc_matrix(mat)
    write_1pz(str(path), mat.indptr.astype(np.int32), mat.indices.astype(np.int32), mat.data.astype(np.uint32),
              int(mat.shape[0]), int(mat.shape[1]), rownames=rows, colnames=cols, user_meta=None)


def test_called_only_1pz_keeps_columns_values_and_order(tmp_path):
    rng = np.random.default_rng(1)
    dense = rng.integers(0, 9, size=(30, 12)) * (rng.random((30, 12)) < 0.4)
    rows = ["g%d" % i for i in range(30)]
    cols = ["BC%02d" % i for i in range(12)]
    p = tmp_path / "exon_reads.1pz"
    _write(p, dense, rows, cols)
    called = {"BC01", "BC05", "BC06", "BC11"}
    out = bundle._called_only("exon_reads.1pz", p.read_bytes(), called)
    q = tmp_path / "out.1pz"
    q.write_bytes(out)
    r = read_1pz(str(q))
    keep = [i for i, c in enumerate(cols) if c in called]
    assert list(r["colnames"]) == [cols[i] for i in keep] and list(r["rownames"]) == rows
    got = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(r["m"], r["n"])).toarray()
    assert np.array_equal(got, dense[:, keep])


def test_called_only_histogram_filters_rows_and_keeps_header():
    text = "barcode\tn1\tn2\nA\t3\t1\nB\t2\t0\nC\t5\t5\n"
    out = bundle._called_only("reads_per_umi_hist.tsv.gz", gzip.compress(text.encode()), {"A", "C"})
    assert gzip.decompress(out).decode() == "barcode\tn1\tn2\nA\t3\t1\nC\t5\t5\n"


def test_reads_modalities_are_registered_after_the_counts():
    names = list(bundle.MODALITIES)
    assert names.index("exon_reads") == names.index("intron_counts") + 1
    assert bundle.MODALITIES["reads_per_umi_hist"].members == ("reads_per_umi_hist.tsv.gz",)
