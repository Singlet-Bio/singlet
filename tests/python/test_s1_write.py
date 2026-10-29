# SPDX-License-Identifier: MIT
"""singlet.write_s1 / read_s1 (a count matrix as a .s1 sample object of your own) and the streaming loader over such files
(singlet.torch.S1Stream / S1Cells), locally and over HTTP ranges.

``tests/data/s1_write_fixture.s1`` is the object written for :func:`fixture_matrix`; the R suite
(``r/tests/testthat/test-s1-write.R``) builds the same matrix and checks ``singlet::write_s1`` against its copy. Both suites
compare everything that does not depend on the zstd library the codec was built with: the container (sample id, member,
kind, meta, digest), the pz2 header (shape, codec, layout, canonical digest) and the decoded counts and names. The names and
tables sections are zstd level 19, whose bytes can differ between zstd releases, so whole-file bytes are equal only between
builds that link the same zstd (Python and R built in one conda env write identical files). Regenerate the fixture only when
the format changes on purpose: ``python tests/python/test_s1_write.py --regen``.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

try:
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
sp = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata")

import singlet  # noqa: E402
from singlet.s1 import write as W  # noqa: E402
from singlet.s1.codec import EncodeOptions, NativeCodec  # noqa: E402
from singlet.s1.container import S2Reader, S2Writer  # noqa: E402
from singlet.s1.matrix import VT_U32, Csc, canonical_digest  # noqa: E402

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "data" / "s1_write_fixture.s1"
R_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "r"
    / "tests"
    / "testthat"
    / "fixtures"
    / "s1_write_fixture.s1"
)


def fixture_matrix():
    """A deterministic 40 genes x 30 cells count matrix (also built by the R test): sparse, mostly small counts, one large."""
    g = np.arange(40)[:, None]
    c = np.arange(30)[None, :]
    x = np.where((g * 3 + c * 5) % 4 == 0, (g * 7 + c * 13) % 11, 0).astype(np.int64)
    x[5, 7] = 70000
    genes = ["g%02d" % (i + 1) for i in range(40)]
    cells = ["c%02d" % (j + 1) for j in range(30)]
    return x, genes, cells  # genes x cells


def random_counts(n_cells, n_genes, density, seed=0):
    rng = np.random.default_rng(seed)
    x = sp.random(
        n_cells,
        n_genes,
        density=density,
        format="csr",
        random_state=seed,
        data_rvs=lambda n: rng.zipf(1.6, n).clip(max=5000),
    )
    return (
        x.astype(np.float32),
        ["ENSG%011d" % i for i in range(n_genes)],
        ["CELL%07d" % j for j in range(n_cells)],
    )


def internal_object(x_cells_genes, genes, cells, sample_id="sample", member="counts", threads=4):
    """The object the .s1 benchmark built before write_s1 existed (bench/s1bench/formats/s1.py)."""
    t = sp.csc_matrix(sp.csr_matrix(x_cells_genes).T)
    t.sort_indices()
    csc = Csc(
        m=t.shape[0],
        n=t.shape[1],
        indptr=t.indptr.astype(np.int64),
        indices=t.indices.astype(np.uint32),
        data=t.data.astype(np.uint32),
        rownames=list(genes),
        colnames=list(cells),
        value_type=VT_U32,
    )
    dg = canonical_digest(csc)
    body = NativeCodec("auto").encode(
        csc, EncodeOptions(row_names="inline", col_names="inline", threads=threads, digest=dg)
    )
    w = S2Writer(sample_id)
    w.add_matrix(member, body, dg, {"modality": "counts"})
    return w.build()


# ----------------------------------------------------------------------------------------------- write / read
def test_round_trip_anndata(tmp_path):
    x, genes, cells = random_counts(500, 300, 0.05)
    a = ad.AnnData(X=x)
    a.var_names, a.obs_names = genes, cells
    p = singlet.write_s1(a, tmp_path / "a.s1")
    b = singlet.read_s1(p, dtype=np.uint32)
    assert list(b.var_names) == genes and list(b.obs_names) == cells
    assert (b.X != sp.csr_matrix(x).astype(np.uint32)).nnz == 0
    c = singlet.load(str(p))  # load() routes a write_s1 object to read_s1
    assert c.shape == (500, 300) and (c.X != sp.csr_matrix(x)).nnz == 0


def test_load_reads_one_matrix_objects(tmp_path):
    """singlet.load, singlet.s1.client.load and load_s2 open a write_s1 file (and a bench-written object) exactly."""
    from singlet.s1 import client

    x, genes, cells = random_counts(400, 250, 0.08, seed=11)
    want = sp.csr_matrix(x).astype(np.uint32)
    p = singlet.write_s1(x, tmp_path / "w.s1", genes=genes, barcodes=cells)
    (tmp_path / "bench.s1").write_bytes(internal_object(x, genes, cells))
    (tmp_path / "other.s1").write_bytes(internal_object(x, genes, cells, member="rna"))
    for path in (p, tmp_path / "bench.s1", tmp_path / "other.s1"):
        for got in (singlet.load(str(path)), client.load(str(path)), client.load_s2(path)):
            assert list(got.var_names) == genes and list(got.obs_names) == cells
            assert got.shape == (400, 250)
            assert (sp.csr_matrix(got.X).astype(np.uint32) != want).nnz == 0
    sub = singlet.load(str(p), genes=[genes[7], "absent", genes[3]])
    assert list(sub.var_names) == [genes[7], genes[3]]
    assert np.array_equal(sub.X.toarray(), x[:, [7, 3]].toarray())
    with pytest.raises(client.ModalityUnavailable, match="counts"):
        singlet.load(str(p), modalities="spliced")


def test_round_trip_scipy_and_dense(tmp_path):
    x, genes, cells = random_counts(80, 60, 0.2, seed=3)
    singlet.write_s1(x, tmp_path / "s.s1", genes=genes, barcodes=cells)
    singlet.write_s1(x.toarray().astype(np.int32), tmp_path / "d.s1", genes=genes, barcodes=cells)
    assert (tmp_path / "s.s1").read_bytes() == (tmp_path / "d.s1").read_bytes()
    b = singlet.read_s1(tmp_path / "s.s1")
    assert np.array_equal(b.X.toarray(), x.toarray())


def test_layer_and_explicit_zeros(tmp_path):
    x, genes, cells = random_counts(50, 40, 0.3, seed=4)
    z = x.copy()
    z.data[::7] = 0  # explicit zeros are dropped, not stored
    a = ad.AnnData(X=np.zeros((50, 40), np.float32))
    a.var_names, a.obs_names = genes, cells
    a.layers["counts"] = z
    singlet.write_s1(a, tmp_path / "l.s1", layer="counts")
    z.eliminate_zeros()
    assert np.array_equal(singlet.read_s1(tmp_path / "l.s1").X.toarray(), z.toarray())


def test_identical_to_internal_writer(tmp_path):
    x, genes, cells = random_counts(700, 400, 0.04, seed=5)
    singlet.write_s1(x, tmp_path / "w.s1", genes=genes, barcodes=cells, sample_id="bench")
    assert (tmp_path / "w.s1").read_bytes() == internal_object(x, genes, cells, sample_id="bench")


def test_threads_do_not_change_bytes(tmp_path):
    x, genes, cells = random_counts(2000, 500, 0.05, seed=6)  # several chunks
    a = singlet.write_s1(x, tmp_path / "t1.s1", genes=genes, barcodes=cells, threads=1).read_bytes()
    b = singlet.write_s1(x, tmp_path / "t8.s1", genes=genes, barcodes=cells, threads=8).read_bytes()
    assert a == b


_HEADER_KEYS = (
    "version",
    "codec",
    "m",
    "n",
    "nnz",
    "value_type",
    "row_order",
    "chunk_cols",
    "n_chunks",
    "lanes",
    "prob_bits",
    "value_prob_bits",
    "ctx_kind",
    "digest",
    "rows_kind",
    "cols_kind",
)


def _fingerprint(path):
    """What a .s1 object fixes independently of the zstd build: container layout, member digest, pz2 header fields."""
    from singlet.s1.codec import parse_header

    rd = S2Reader(str(path))
    out = {"sample_id": rd.sample_id}
    for name, mi in rd.members.items():
        h = parse_header(rd.read_range(name, 0, 256))
        out[name] = (mi.kind, mi.flags, mi.meta, mi.digest, mi.off) + tuple(
            h[k] for k in _HEADER_KEYS
        )
    return out


def test_fixture(tmp_path):
    if not FIXTURE.exists():
        pytest.skip("fixture not generated")
    assert (
        R_FIXTURE.read_bytes() == FIXTURE.read_bytes()
    )  # the R suite checks singlet::write_s1 against its copy
    x, genes, cells = fixture_matrix()
    p = singlet.write_s1(x.T, tmp_path / "f.s1", genes=genes, barcodes=cells)
    assert _fingerprint(p) == _fingerprint(FIXTURE)
    for f in (p, FIXTURE):
        a = singlet.read_s1(f, dtype=np.uint32)
        assert list(a.var_names) == genes and list(a.obs_names) == cells
        assert np.array_equal(a.X.toarray(), x.T)


def test_gene_pack_reference(tmp_path):
    x, genes, cells = random_counts(40, 30, 0.3, seed=7)
    pid = "0123456789abcdef"
    singlet.write_s1(x, tmp_path / "p.s1", genes=genes, barcodes=cells, gene_pack=pid)
    inline = singlet.write_s1(x, tmp_path / "i.s1", genes=genes, barcodes=cells)
    assert (tmp_path / "p.s1").stat().st_size < inline.stat().st_size  # names not stored
    b = singlet.read_s1(tmp_path / "p.s1", genes=genes)
    assert list(b.var_names) == genes and np.array_equal(b.X.toarray(), x.toarray())
    with pytest.raises(ValueError, match="vocabulary pack"):
        singlet.read_s1(tmp_path / "p.s1")


@pytest.mark.parametrize(
    "bad, msg",
    [(np.array([[0.5, 1.0]]), "non-integer"), (np.array([[-1, 2]]), "negative")],
)
def test_rejects_non_counts(tmp_path, bad, msg):
    with pytest.raises(ValueError, match=msg):
        singlet.write_s1(bad, tmp_path / "x.s1", genes=["a", "b"], barcodes=["c"])


def test_needs_names_and_orientation(tmp_path):
    x = np.ones((3, 2), np.int64)
    with pytest.raises(ValueError, match="required"):
        singlet.write_s1(x, tmp_path / "x.s1")
    with pytest.raises(ValueError, match="cells x genes"):
        singlet.write_s1(x, tmp_path / "x.s1", genes=["a", "b", "c"], barcodes=["u", "v"])


def test_object_layout(tmp_path):
    x, genes, cells = random_counts(30, 20, 0.3, seed=8)
    p = singlet.write_s1(
        x, tmp_path / "o.s1", genes=genes, barcodes=cells, sample_id="GSM123", member="counts"
    )
    rd = S2Reader(str(p))
    assert rd.sample_id == "GSM123" and list(rd.members) == ["counts"]
    assert rd.members["counts"].meta == {"modality": "counts"}
    assert W.is_user_object(p)
    assert not (tmp_path / "o.s1.part").exists()


if __name__ == "__main__" and "--regen" in sys.argv:
    x, genes, cells = fixture_matrix()
    for f in (FIXTURE, R_FIXTURE):
        f.parent.mkdir(parents=True, exist_ok=True)
        singlet.write_s1(x.T, f, genes=genes, barcodes=cells)
        print("wrote", f, f.stat().st_size, "bytes")
