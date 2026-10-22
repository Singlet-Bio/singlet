# SPDX-License-Identifier: MIT
"""Unit tests for the v2 migration recipes (W3): synthetic inputs only, no corpus needed.

The golden tests on the real corpora are in test_v2_recipes_corpus.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from singlet.v2 import recipes as R  # noqa: E402
from singlet.v2 import recipes_cells as RC  # noqa: E402
from singlet.v2 import recipes_ctx as CTX  # noqa: E402
from singlet.v2 import recipes_extra as RE  # noqa: E402
from singlet.v2 import recipes_mtx as RM  # noqa: E402
from singlet.v2 import recipes_psi as RP  # noqa: E402
from singlet.v2 import recipes_pzw as PW  # noqa: E402
from singlet.v2 import recipes_star as RS  # noqa: E402
from singlet.v2 import recipes_tsv as RT  # noqa: E402
from singlet.v2.recipes_ctx import Matrix  # noqa: E402


# ------------------------------------------------------------------------------------------------------------ helpers
def rand_csc(m, n, density, seed, vmax=5, zero_cols=()):
    rs = np.random.RandomState(seed)
    indptr = [0]
    idx, dat = [], []
    for j in range(n):
        if j in zero_cols:
            k = 0
        else:
            k = rs.binomial(m, density)
        rows = np.sort(rs.choice(m, size=k, replace=False)) if k else np.zeros(0, dtype=np.int64)
        idx.append(rows)
        dat.append(rs.randint(1, vmax + 1, size=k))
        indptr.append(indptr[-1] + k)
    idx = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)
    dat = np.concatenate(dat) if dat else np.zeros(0, dtype=np.int64)
    return Matrix(
        np.array(indptr, dtype=np.int64), idx.astype(np.int64), dat.astype(np.uint32), (m, n)
    )


class FakeCtx(CTX.Context):
    """Minimal in-memory context with the migration conventions (auxiliary members via provide_*)."""

    def __init__(
        self,
        members=None,
        matrices=None,
        cells=None,
        genes=None,
        vocab=None,
        build="TEST",
        wl=None,
        order=None,
    ):
        self.sample_id = "GSM1"
        self.study_id = "GSE1"
        self.reference_build = build
        self.protocol_name = "10xv2"
        self._members = dict(members or {})
        self._mats = dict(matrices or {})
        self._cells = cells
        self._genes = genes
        self._vocab = vocab or {}
        self._wl = wl or {}
        self._order = order
        self._vec = {}

    def matrix(self, member):
        return self._mats.get(member)

    def member_bytes(self, member):
        return self._members.get(member)

    def cells(self):
        return self._cells

    def vocab_names(self, kind):
        return self._vocab.get(kind)

    def genes(self):
        return self._genes

    def whitelist(self, name):
        return self._wl.get(name)

    def whitelist_names(self):
        return list(self._wl)

    def features_order(self):
        return self._order

    def vector(self, name):
        return self._vec.get(name)

    def provide_matrix(self, name, m):
        assert name not in self._mats, "auxiliary name taken"
        self._mats[name] = m

    def provide_vector(self, name, arr):
        self._vec[name] = np.asarray(arr)


# ------------------------------------------------------------------------------------------------------------ interface
def test_registry_interface():
    ids = set(R.RECIPES)
    for rid in (
        "psi-v1-rint",
        "psi-v1-trunc",
        "psi-v1-stub",
        "cellcalls-v1",
        "tsvcols-v1",
        "whitelist-sorted-v1",
        "features-v1",
        "mtx-text-v1",
        "mtx-filter-v1",
        "ambient-v1",
        "geneexpr-v1",
        "splice-events-v1",
        "vocab-json-v1",
        "stub-v1",
    ):
        assert rid in ids
        assert R.RECIPES[rid].id == rid
    ctx = FakeCtx()
    assert R.plan("exon_counts.1pz", b"x", ctx) is None
    assert R.plan("no_such_member.bin", b"abc", ctx) is None
    assert R.cells_columns("no_such_member.bin", b"abc", ctx) == {}
    with pytest.raises(LookupError):
        R.regenerate("nope-v9", {}, ctx)


def test_stub_member():
    ctx = FakeCtx()
    for member in ("mt_heteroplasmy.1pz", "vdj_gene_usage.1pz"):
        p = R.plan(member, PW.STUB_1PZ, ctx)
        assert p is not None and p.recipe_id == "stub-v1" and p.params == {}
        assert R.regenerate(p.recipe_id, p.params, ctx) == PW.STUB_1PZ
        m = R.regenerate_matrix(p.recipe_id, p.params, ctx)
        assert tuple(m.shape) == (0, 0)
    p = R.plan("splice_psi.1pz", PW.STUB_1PZ, ctx)
    assert p.recipe_id == "psi-v1-stub"
    assert R.regenerate("psi-v1-stub", {}, ctx) == PW.STUB_1PZ


# ------------------------------------------------------------------------------------------------------------ mtx text
@pytest.mark.parametrize(
    "m,n,dens,seed", [(50, 40, 0.1, 1), (7, 300, 0.4, 2), (1000, 3, 0.01, 3), (5, 5, 0.0, 4)]
)
def test_mtx_roundtrip(m, n, dens, seed):
    mat = rand_csc(m, n, dens, seed, vmax=1234)
    b = RM.render_mtx(mat)
    got = RM.parse_mtx(b)
    assert got is not None
    mat2, prefix = got
    assert prefix == RM.DEFAULT_HEADER
    assert mat2.shape == mat.shape and np.array_equal(mat2.indptr, mat.indptr)
    assert np.array_equal(mat2.indices, mat.indices) and np.array_equal(mat2.data, mat.data)
    assert RM.render_mtx(mat2, prefix) == b


def test_mtx_text_format_and_rejects():
    b = b"%%MatrixMarket matrix coordinate integer general\n%\n3 2 3\n1 1 5\n3 1 1\n2 2 10\n"
    mat, pre = RM.parse_mtx(b)
    assert (
        mat.shape == (3, 2)
        and mat.indptr.tolist() == [0, 2, 3]
        and mat.indices.tolist() == [0, 2, 1]
    )
    assert RM.render_mtx(mat, pre) == b
    assert RM.parse_mtx(b.replace(b"2 2 10", b"2 2 -1")) is None  # negative
    assert RM.parse_mtx(b.replace(b"3 1 1\n", b"1 1 1\n")) is None  # not strictly ordered
    assert RM.parse_mtx(b + b"x") is None
    assert RM.parse_mtx(b.replace(b"3 2 3", b"3 2 4")) is None  # nnz mismatch
    hdr = b"%%MatrixMarket matrix coordinate real general\n%comment\n"
    mat2, pre2 = RM.parse_mtx(hdr + b"1 1 0\n")
    assert pre2 == hdr and mat2.nnz == 0


def test_mtx_text_recipe_with_aux_matrix():
    mat = rand_csc(60, 90, 0.05, 7)
    b = RM.render_mtx(mat)
    ctx = FakeCtx(members={RS.RAW_MTX: b})
    p = R.plan(RS.RAW_MTX, b, ctx)
    assert p.recipe_id == "mtx-text-v1" and p.params == {"m": "raw"}
    assert RS.aux_name(RS.RAW_MTX) in ctx._mats  # registered through provide_matrix
    assert R.regenerate(p.recipe_id, p.params, ctx) == b
    # filtered member through the same recipe when it is not a subset of the called cells
    b2 = RM.render_mtx(rand_csc(60, 9, 0.2, 8))
    ctx2 = FakeCtx(members={RS.FIL_MTX: b2})
    p2 = R.plan(RS.FIL_MTX, b2, ctx2)
    assert p2.recipe_id == "mtx-text-v1" and p2.params["m"] == "filtered"
    assert R.regenerate(p2.recipe_id, p2.params, ctx2) == b2


def test_mtx_filter_recipe():
    rs = np.random.RandomState(11)
    wl = np.sort(np.array(["".join(rs.choice(list("ACGT"), 8)) for _ in range(400)], dtype="S8"))
    wl = np.unique(wl)
    raw = rand_csc(30, len(wl), 0.05, 12)
    called_idx = np.sort(rs.choice(len(wl), size=17, replace=False))
    raw_b = RM.render_mtx(raw)
    bcs = [x.decode() for x in wl]
    cells = {
        "barcode": np.array(bcs, dtype=object),
        "called": np.isin(np.arange(len(wl)), called_idx),
    }
    fil_m = RM.select_columns(raw, called_idx)
    fil_b = RM.render_mtx(fil_m)
    fbar_b = RS.array_to_lines(wl[called_idx])
    ctx = FakeCtx(
        members={
            RS.RAW_BC: RS.array_to_lines(wl),
            RS.RAW_MTX: raw_b,
            RS.FIL_MTX: fil_b,
            RS.FIL_BC: fbar_b,
        },
        cells=cells,
    )
    p = R.plan(RS.FIL_MTX, fil_b, ctx)
    assert p.recipe_id == "mtx-filter-v1" and p.params == {"what": "matrix"}
    assert RS.aux_name(RS.RAW_MTX) in ctx._mats
    assert R.regenerate(p.recipe_id, p.params, ctx) == fil_b
    p = R.plan(RS.FIL_BC, fbar_b, ctx)
    assert p.recipe_id == "mtx-filter-v1" and p.params == {"what": "barcodes"}
    assert R.regenerate(p.recipe_id, p.params, ctx) == fbar_b
    # the raw matrix recipe finds the auxiliary matrix that the filtered plan registered (member order is not ours)
    p = R.plan(RS.RAW_MTX, raw_b, ctx)
    assert p.recipe_id == "mtx-text-v1" and R.regenerate(p.recipe_id, p.params, ctx) == raw_b
    # a called barcode that is not in the raw list -> no plan
    bad = dict(cells)
    bad["barcode"] = np.array(["TTTTTTTT"] + bcs[1:], dtype=object)
    bad["called"] = np.isin(np.arange(len(wl)), np.concatenate([[0], called_idx]))
    ctx_bad = FakeCtx(members=ctx._members, cells=bad)
    assert R.plan(RS.FIL_BC, fbar_b, ctx_bad) is None


def test_whitelist_recipe():
    rs = np.random.RandomState(5)
    wl = np.unique(
        np.array(["".join(rs.choice(list("ACGT"), 16)) for _ in range(2000)], dtype="S16")
    )
    b = RS.array_to_lines(wl)
    for form in (
        wl,
        [x.decode() for x in wl],
    ):  # numpy array or list of str, like the two context flavours
        ctx = FakeCtx(wl={"mylist": form, "other": wl[:100]})
        p = R.plan(RS.RAW_BC, b, ctx)
        assert p.recipe_id == "whitelist-sorted-v1" and p.params == {"wl": "mylist"}
        assert R.regenerate(p.recipe_id, p.params, ctx) == b
    assert R.plan(RS.RAW_BC, b[:-17], FakeCtx(wl={"mylist": wl})) is None
    assert RS.lines_to_array(b"AC\nGT") is None and RS.lines_to_array(b"AC\r\nGT\r\n") is None


def test_features_recipe_orders():
    genes = [("G%02d" % i, "N%d" % i if i % 3 else "G%02d" % i) for i in range(30)]
    # vocab order
    b = RS.render_features(genes, np.arange(30))
    ctx = FakeCtx(genes=genes)
    p = R.plan(RS.RAW_FEAT, b, ctx)
    assert p.params == {"order": "vocab"} and R.regenerate(p.recipe_id, p.params, ctx) == b
    # permutation resource
    rs = np.random.RandomState(0)
    perm = rs.permutation(30)
    b2 = RS.render_features(genes, perm)
    ctx2 = FakeCtx(genes=genes, build="NOSUCHBUILD", order=perm)
    p2 = R.plan(RS.FIL_FEAT, b2, ctx2)
    assert p2.params == {"order": "perm"} and R.regenerate(p2.recipe_id, p2.params, ctx2) == b2
    assert R.plan(RS.FIL_FEAT, b2, FakeCtx(genes=genes, build="NOSUCHBUILD")) is None
    # positional order: contigs in first-appearance order, then start, then end
    exons = []
    layout = [
        ("chr2", 500, 900),
        ("chr2", 100, 200),
        ("chr1", 50, 60),
        ("chrM", 1, 10),
        ("chr1", 10, 20),
        ("chr2", 100, 150),
    ]
    genes2 = [("E%d" % i, "S%d" % i) for i in range(len(layout))]
    for (g, _), (c, s, e) in zip(genes2, layout):
        exons.append("%s_%s_%s:%d-%d" % (g, g.replace("E", "S"), c, s, e))
    contigs = ["chr1", "chr2", "chrM"]
    cix = {c: i for i, c in enumerate(contigs)}
    order = sorted(
        range(len(layout)), key=lambda i: (cix[layout[i][0]], layout[i][1], layout[i][2], i)
    )
    b3 = RS.render_features(genes2, np.array(order))
    ctx3 = FakeCtx(genes=genes2, vocab={"exon": exons})
    p3 = R.plan(RS.RAW_FEAT, b3, ctx3)
    assert p3.params["order"] == "pos" and p3.params["contigs"] == contigs
    assert R.regenerate(p3.recipe_id, p3.params, ctx3) == b3


# ------------------------------------------------------------------------------------------------------------ psi
def ref_compute_psi(names, indptr, indices, data, ncols):
    """Line-by-line Python port of include/singlet/pileup/splice_psi.h:compute_psi (float32 accumulators)."""
    f32 = np.float32
    donor, acceptor = {}, {}
    for i, nm in enumerate(names):
        c = nm.find(":")
        d = nm.find("-", c + 1) if c >= 0 else -1
        if c < 0 or d < 0:
            continue
        try:
            s, e = int(nm[c + 1 : d]), int(nm[d + 1 :])
        except ValueError:
            continue
        donor.setdefault("%s:%d-D" % (nm[:c], s), []).append(i)
        acceptor.setdefault("%s:%d-A" % (nm[:c], e), []).append(i)
    events = []
    for grp in (donor, acceptor):
        for k, v in grp.items():
            if len(v) >= 2:
                events.append((k, v))
    events.sort(key=lambda t: t[0].encode())
    jev = [-1] * len(names)
    for ei, (_, js) in enumerate(events):
        for j in js:
            if jev[j] == -1:
                jev[j] = ei
    out_ptr, out_idx, out_val = [0], [], []
    for c in range(ncols):
        sums = {}
        for k in range(indptr[c], indptr[c + 1]):
            e = jev[indices[k]]
            if e >= 0:
                sums[e] = f32(sums.get(e, f32(0)) + f32(data[k]))
        for k in range(indptr[c], indptr[c + 1]):
            j = indices[k]
            e = jev[j]
            if e >= 0 and sums.get(e, f32(0)) > 0:
                out_idx.append(j)
                out_val.append(f32(data[k]) / sums[e])
        out_ptr.append(len(out_idx))
    return np.array(out_ptr), np.array(out_idx, dtype=np.int64), np.array(out_val, dtype=np.float32)


def junction_names(n, seed):
    rs = np.random.RandomState(seed)
    names = []
    for _ in range(n):
        chrom = rs.choice(["chr1", "chr2", "chrX"])
        s = int(rs.choice([99, 100, 995, 1000, 1001, 9999, 10000, 10001, 5, 6, 7]))
        e = int(rs.choice([100, 101, 1000, 1002, 10000, 10002, 100007033, 99993741, 12, 13]))
        names.append("%s:%d-%d" % (chrom, s, e))
    names += [
        "chr1:99993741-100007033",
        "chr1:100007033-100008000",
        "chr1:99993741-100009000",
    ]  # crosses a power of ten
    names += [
        "unparsable",
        "chr1:10",
        "chr1:ab-5",
        "chr3:7-8",
        "chr3:7-9",
        "chr3:7-9",
    ]  # junk and a duplicate name
    return names


def test_psi_numpy_equals_reference_port():
    names = junction_names(120, 1)
    mat = rand_csc(len(names), 60, 0.25, 2, vmax=9)
    mat.rownames = names
    ip, ix, vals, maxsum = RP.compute_psi(mat)
    rip, rix, rval = ref_compute_psi(names, mat.indptr, mat.indices, mat.data, mat.shape[1])
    assert np.array_equal(ip, rip) and np.array_equal(ix, rix)
    assert np.array_equal(vals.view(np.uint32), rval.view(np.uint32))  # bit-identical float32
    assert maxsum < RP.MAX_EXACT_SUM


def test_psi_event_rules():
    # the acceptor event sorts before the donor event of the same junction: chr1:100007033-A < chr1:99993741-D
    names = [
        "chr1:99993741-100007033",
        "chr1:100007033-100008000",
        "chr1:99993741-100009000",
        "chr1:100000000-100007033",
    ]
    ev = RP.build_events(names)
    ids = [e[0] for e in ev]
    assert ids == sorted(ids, key=lambda s: s.encode())
    je, eids = RP.junction_events(names)
    assert eids[je[0]] == "chr1:100007033-A"  # first event in sorted order wins, not "donor first"
    assert eids[je[2]] == "chr1:99993741-D"
    assert RP.build_events(["chr1:5-9"]) == []


def test_psi_quantise_rules():
    psi = np.array([0.0, 0.25, 0.5, 0.5000001, 0.75, 1.0], dtype=np.float32)
    assert RP.quantise(psi, "rint").tolist() == [0, 0, 0, 1, 1, 1]  # half to even: 0.5 -> 0
    assert RP.quantise(psi, "trunc").tolist() == [0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        RP.quantise(psi, "ceil")


@pytest.mark.parametrize("rule", ["rint", "trunc"])
def test_psi_plan_and_regenerate(rule):
    pytest.importorskip(
        "singlet._pz", reason="singlet._pz (the .1pz reader and writer) is not built"
    )
    names = junction_names(200, 3)
    sj = rand_csc(len(names), 80, 0.3, 4, vmax=6)
    sj.rownames = names
    sj.colnames = ["BC%03d" % i for i in range(80)]
    m = RP.psi_matrix(sj, rule)
    ws = PW.available_writers()
    assert ws, "no .1pz writer available"
    wid = next(iter(ws))
    legacy = PW.write_legacy_1pz(m, wid)
    ctx = FakeCtx(matrices={"sj_counts.1pz": sj})
    ctx._mats["splice_psi.1pz"] = CTX.decode_1pz_bytes(legacy)
    p = R.plan("splice_psi.1pz", legacy, ctx)
    assert p is not None and p.needs == ["matrix:sj_counts.1pz"]
    # when rint and trunc coincide on the data the planner reports rint first
    assert p.recipe_id in ("psi-v1-" + rule, "psi-v1-rint")
    assert p.params["writer"] in PW.available_writers()
    assert R.regenerate(p.recipe_id, p.params, ctx) == legacy  # bytes
    leg_m = CTX.decode_1pz_bytes(legacy)
    got = R.regenerate_matrix(p.recipe_id, p.params, ctx)  # canonical matrix
    assert got.equals(leg_m)
    # a legacy member whose values are not the PSI of this sj matrix is refused
    broken = CTX.decode_1pz_bytes(legacy)
    broken.data = broken.data.copy()
    broken.data[0] += 1
    ctx3 = FakeCtx(matrices={"sj_counts.1pz": sj, "splice_psi.1pz": broken})
    assert R.plan("splice_psi.1pz", legacy, ctx3) is None


def test_psi_without_writer_is_matrix_level(monkeypatch):
    pytest.importorskip(
        "singlet._pz", reason="singlet._pz (the .1pz reader and writer) is not built"
    )
    names = junction_names(60, 5)
    sj = rand_csc(len(names), 20, 0.4, 6, vmax=4)
    sj.rownames = names
    sj.colnames = ["B%d" % i for i in range(20)]
    wid = next(iter(PW.available_writers()))
    legacy = PW.write_legacy_1pz(RP.psi_matrix(sj, "rint"), wid)
    monkeypatch.setattr(PW, "_WRITERS", {})  # no writer at all
    ctx = FakeCtx(matrices={"sj_counts.1pz": sj, "splice_psi.1pz": CTX.decode_1pz_bytes(legacy)})
    p = R.plan("splice_psi.1pz", legacy, ctx)
    assert p is not None and p.params["writer"] is None
    with pytest.raises(RP.NotByteExact):
        R.regenerate(p.recipe_id, p.params, ctx)
    assert isinstance(
        RP.NotByteExact(), LookupError
    )  # the adapter treats it as "cannot regenerate bytes"
    assert R.regenerate_matrix(p.recipe_id, p.params, ctx).equals(CTX.decode_1pz_bytes(legacy))


# ------------------------------------------------------------------------------------------------------------ cell_calls
def make_cells(n, n_called, seed, with_bc="acgt"):
    rs = np.random.RandomState(seed)
    bcs = np.unique(np.array(["".join(rs.choice(list("ACGT"), 12)) for _ in range(n * 2)]))[:n]
    rs.shuffle(bcs)
    tot = rs.geometric(0.01, size=n).astype(np.uint32)
    return bcs, tot


def cc_text(rows, method):
    out = [RC.CC_HEADER]
    for bc, is_cell, tot, pv in rows:
        lg = "%.4f" % (np.log10(float(tot)) if tot > 0 else 0.0)
        out.append("%s\t%s\t%s\t%s\t%s\n" % (bc, "TRUE" if is_cell else "FALSE", lg, pv, method))
    return "".join(out).encode()


def test_cellcalls_star_solo_and_zero_total():
    bcs, tot = make_cells(500, 40, 1)
    tot[3] = 0
    called = np.zeros(500, dtype=bool)
    sel = np.random.RandomState(2).choice(500, 40, replace=False)
    called[sel] = True
    rows = sorted([(bcs[i], True, int(tot[i]), RC.SENTINEL_P) for i in sel])
    legacy = cc_text(rows, "star_solo")
    cells = {"barcode": bcs.astype(object), "called": called, "n_umi_exon": tot}
    ctx = FakeCtx(cells=cells)
    p = R.plan("cell_calls.tsv", legacy, ctx)
    assert p.recipe_id == "cellcalls-v1"
    assert p.params == {
        "method": "star_solo",
        "listed": "called",
        "order": "sorted",
        "pvalue": "zero",
    }
    assert R.regenerate(p.recipe_id, p.params, ctx) == legacy
    cols = R.cells_columns("cell_calls.tsv", legacy, ctx)
    assert set(cols) == {"called"} and cols["called"][0] == "bool"


def test_cellcalls_emptydrops_and_top_n_variants():
    n = 600
    bcs, tot = make_cells(n, 0, 3)
    rs = np.random.RandomState(4)
    tested = np.sort(rs.choice(n, 120, replace=False))
    is_cell = rs.rand(120) < 0.4
    pvals = np.where(is_cell, 1.5744e-04, 1.0)
    pvals[::7] = 0.0123456
    rows = [(bcs[i], bool(c), int(tot[i]), "%.4e" % p) for i, c, p in zip(tested, is_cell, pvals)]
    legacy = cc_text(rows, "emptydrops")
    called = np.zeros(n, bool)
    called[tested[is_cell]] = True
    ctx = FakeCtx(cells={"barcode": bcs.astype(object), "called": called, "n_umi_exon": tot})
    p = R.plan("cell_calls.tsv", legacy, ctx)
    assert p.params == {
        "method": "emptydrops",
        "listed": "col",
        "order": "colorder",
        "pvalue": "col",
    }
    assert {"cells:cc_listed", "cells:cc_pvalue"} <= set(p.needs)
    cols = R.cells_columns("cell_calls.tsv", legacy, ctx)
    assert set(cols) == {"called", "cc_listed", "cc_pvalue"}
    view = dict(ctx.cells())
    view.update({k: v[1] for k, v in cols.items()})
    ctx2 = FakeCtx(cells=view)
    assert R.regenerate(p.recipe_id, p.params, ctx2) == legacy
    # top_n_fallback: descending total, ties in an arbitrary (unstable-sort) order
    tot2 = np.array([50] * 6 + [49] * 6 + list(range(1, n - 11)), dtype=np.uint32)[
        :n
    ]  # the 10th cell is inside a tie group
    order = np.lexsort((rs.rand(n), -tot2.astype(np.int64)))[:10]  # random tie order
    rows = [(bcs[i], True, int(tot2[i]), RC.SENTINEL_P) for i in order]
    legacy2 = cc_text(rows, "top_n_fallback")
    called2 = np.zeros(n, bool)
    called2[order] = True
    ctx3 = FakeCtx(cells={"barcode": bcs.astype(object), "called": called2, "n_umi_exon": tot2})
    p3 = R.plan("cell_calls.tsv", legacy2, ctx3)
    assert p3.params["order"] in ("explicit", "sorted", "colorder")
    cols3 = R.cells_columns("cell_calls.tsv", legacy2, ctx3)
    view3 = dict(ctx3.cells())
    view3.update({k: v[1] for k, v in cols3.items()})
    assert R.regenerate(p3.recipe_id, p3.params, FakeCtx(cells=view3)) == legacy2


def test_cellcalls_refused():
    bcs, tot = make_cells(50, 5, 7)
    called = np.zeros(50, bool)
    called[:3] = True
    ctx = FakeCtx(cells={"barcode": bcs.astype(object), "called": called, "n_umi_exon": tot})
    assert R.plan("cell_calls.tsv", b"not a cell calls file\n", ctx) is None
    rows = [("NOTINMATRIX", True, 10, RC.SENTINEL_P)]
    assert R.plan("cell_calls.tsv", cc_text(rows, "star_solo"), ctx) is None
    rows = [
        (bcs[0], True, int(tot[0]) + 1, RC.SENTINEL_P)
    ]  # log10 column disagrees with the matrix
    assert R.plan("cell_calls.tsv", cc_text(rows, "star_solo"), ctx) is None


# ------------------------------------------------------------------------------------------------------------ tsv columns
def test_fit_column_roundtrip_kinds():
    cases = {
        "ints": [str(x) for x in np.random.RandomState(0).randint(0, 70000, 200)],
        "ints_na": ["NA" if i % 5 == 0 else str(i * 3) for i in range(100)],
        "fixed4": ["%.4f" % x for x in np.random.RandomState(1).rand(300) * 100],
        "fixed6": ["%.6f" % x for x in np.random.RandomState(2).rand(300)],
        "g6": [
            "%.6g" % x
            for x in np.random.RandomState(3).rand(300)
            * 10.0 ** np.random.RandomState(3).randint(-6, 4, 300)
        ],
        "sci": ["%.4e" % x for x in np.random.RandomState(4).rand(300) * 1e-3],
        "float_na": ["NA" if i % 4 == 0 else "%.4f" % (i / 7.0) for i in range(120)],
        "bool_py": ["True" if i % 3 else "False" for i in range(50)],
        "bool_up": ["TRUE" if i % 3 else "FALSE" for i in range(50)],
        "cat": [("G1", "S", "G2M")[i % 3] for i in range(60)],
        "const": ["NA"] * 40,
        "text": ["id_%d" % (i * 7919) for i in range(400)],
        "neg": [str(i - 50) for i in range(100)],
    }
    for name, vals in cases.items():
        fit, arr = RT.fit_column(vals)
        assert RT.render_column(fit, arr) == vals, name
    assert RT.fit_column(cases["ints"])[0]["k"] == "int"
    assert RT.fit_column(cases["fixed4"])[0]["fmt"] == "%.4f"
    assert (
        RT.fit_column(cases["bool_py"])[0]["k"] == "bool"
        and RT.fit_column(cases["cat"])[0]["k"] == "cat"
    )
    assert RT.fit_column(cases["float_na"])[0]["na"] == "NA"


def test_parse_tsv_refuses_irregular():
    assert RT.parse_tsv(b"a\tb\n1\t2\n") == (["a", "b"], [["1"], ["2"]])
    assert RT.parse_tsv(b"a\tb\n1\t2") is None  # no final newline
    assert RT.parse_tsv(b"a\tb\r\n1\t2\r\n") is None  # CRLF
    assert RT.parse_tsv(b"a\tb\n1\n") is None  # ragged
    h, c = RT.parse_tsv(b"x\ny\n", has_header=False)
    assert h is None and c == [["x", "y"]]


def test_tsvcols_roundtrip_on_synthetic_table():

    n = 300
    bcs, tot = make_cells(n, 0, 9)
    rs = np.random.RandomState(10)
    called = np.zeros(n, bool)
    called[rs.choice(n, 80, replace=False)] = True
    cells = {"barcode": bcs.astype(object), "called": called, "n_umi_exon": tot}
    # doublet_scores.tsv: rows = called cells, sorted by barcode
    sel = sorted(np.flatnonzero(called), key=lambda i: bcs[i])
    score = np.round(rs.rand(len(sel)), 6)
    isd = score > 0.6
    lines = ["barcode\ttotal_umis\tdoublet_score\tis_doublet\n"]
    for i, s, d in zip(sel, score, isd):
        lines.append("%s\t%d\t%s\t%s\n" % (bcs[i], tot[i], "%.6g" % s, "True" if d else "False"))
    legacy = "".join(lines).encode()
    ctx = FakeCtx(cells=cells)
    p = R.plan("doublet_scores.tsv", legacy, ctx)
    assert p is not None and p.recipe_id == "tsvcols-v1"
    assert p.params["r"] == {"m": "called", "o": "sorted"}
    assert [c["s"] for c in p.params["c"]] == [
        "d",
        "c",
        "c",
    ]  # total_umis derived, score and flag stored
    cols = R.cells_columns("doublet_scores.tsv", legacy, ctx)
    assert set(cols) == {"dbl_doublet_score", "dbl_is_doublet"}
    view = dict(cells)
    view.update({k: v[1] for k, v in cols.items()})
    assert R.regenerate(p.recipe_id, p.params, FakeCtx(cells=view)) == legacy
    # auto_barcodes.tsv: the barcode list itself, no header
    ab = ("\n".join(bcs) + "\n").encode()
    pa = R.plan("auto_barcodes.tsv", ab, ctx)
    assert (
        pa.params["h"] is None
        and pa.params["c"] == []
        and R.regenerate(pa.recipe_id, pa.params, ctx) == ab
    )
    # a table whose barcodes are not matrix columns is refused
    assert (
        R.plan("doublet_scores.tsv", legacy.replace(bcs[sel[0]].encode(), b"ZZZZZZZZZZZZ"), ctx)
        is None
    )


# ------------------------------------------------------------------------------------------------------------ extras
def test_ambient_inference_and_recipe():
    rs = np.random.RandomState(21)
    n = 5000
    k = np.where(rs.rand(n) < 0.7, 0, rs.geometric(0.05, n)).astype(np.int64)
    k[17] = 12345
    names = ["g%d" % i for i in range(n)]
    legacy = RE.render_ambient(names, k)
    assert legacy.startswith(b"feature\tambient_fraction\n")
    ctx = FakeCtx(vocab={"exon": names})
    p = R.plan("ambient_profile.tsv", legacy, ctx)
    assert p is not None and p.recipe_id == "ambient-v1" and p.params == {}
    assert np.array_equal(ctx.vector(RE.AMBIENT_VECTOR).astype(np.int64), k)
    assert R.regenerate(p.recipe_id, p.params, ctx) == legacy
    # no zero rows (min value is not 0.5/D): the inference still finds a consistent vector
    k2 = k + 3
    legacy2 = RE.render_ambient(names, k2)
    ctx2 = FakeCtx(vocab={"exon": names})
    p2 = R.plan("ambient_profile.tsv", legacy2, ctx2)
    assert p2 is not None and R.regenerate(p2.recipe_id, p2.params, ctx2) == legacy2
    assert R.plan("ambient_profile.tsv", legacy[:-5], FakeCtx(vocab={"exon": names})) is None


def test_geneexpr_recipe():
    genes = [("G1", "AAA"), ("G2", "G2"), ("G3", "CCC"), ("G4", "DDD")]
    ex_names = [
        "G1_AAA_chr1:100-200",
        "G1_AAA_chr1:300-350",
        "G2__chr1:1000-1100",
        "G3_CCC_chr2:5-95",
        "G4_DDD_chr2:10-20",
    ]
    in_names = ["G1_AAA_intron_chr1:200-300"]
    ex = Matrix(
        np.array([0, 3, 5]),
        np.array([0, 2, 3, 1, 4]),
        np.array([4, 7, 1, 2, 9], dtype=np.uint32),
        (5, 2),
        ex_names,
        ["c1", "c2"],
    )
    it = Matrix(
        np.array([0, 1, 1]),
        np.array([0]),
        np.array([5], dtype=np.uint32),
        (1, 2),
        in_names,
        ["c1", "c2"],
    )
    ctx = FakeCtx(
        matrices={"exon_counts.1pz": ex, "intron_counts.1pz": it},
        genes=genes,
        vocab={"exon": ex_names, "intron": in_names},
    )
    g, ln, cnt = RE.gene_tables(ctx)
    assert ln.tolist() == [150, 100, 90, 10]  # merged exon bp per gene
    assert cnt.tolist() == [4 + 2 + 5, 7, 1, 9]  # exon + intron UMIs per gene over all cells
    legacy = RE.render_geneexpr(g, ln, cnt, 1234567)
    p = R.plan("gene_expression.tsv", legacy, ctx)
    assert p is not None and p.params == {"total_mapped": 1234567}
    assert R.regenerate(p.recipe_id, p.params, ctx) == legacy
    pb = json.dumps({"mapped_reads": 1234567}).encode()
    ctx._members["pileup_stats.json"] = pb
    assert R.plan("gene_expression.tsv", legacy, ctx).params == {"total_mapped": 1234567}
    assert R.plan("gene_expression.tsv", legacy.replace(b"\t150\t", b"\t151\t"), ctx) is None


def test_splice_events_recipe():
    names = ["chr1:10-20", "chr1:10-30", "chr1:15-30", "chr2:5-6", "bad"]
    sj = Matrix(
        np.array([0]),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.uint32),
        (5, 0),
        names,
        [],
    )
    legacy = RE.render_splice_events(names)
    assert legacy.splitlines()[0] == b"event_id\tevent_type\tn_junctions\tjunction_names"
    assert (
        b"chr1:10-D\talt_donor\t2\tchr1:10-20,chr1:10-30" in legacy
        and b"chr1:30-A\talt_acceptor\t2\tchr1:10-30,chr1:15-30" in legacy
    )
    ctx = FakeCtx(matrices={"sj_counts.1pz": sj})
    p = R.plan("splice_events.tsv", legacy, ctx)
    assert p.recipe_id == "splice-events-v1" and R.regenerate(p.recipe_id, p.params, ctx) == legacy


def test_vocab_json_recipe():
    genes = [("ENSG1", "A"), ("ENSG2", "ENSG2")]
    legacy = RE.render_vocab_json("GRCh38-2024-A", "1.0", genes)
    assert json.loads(legacy)["n_genes"] == 2 and b"\n  " in legacy and not legacy.endswith(b"\n")
    ctx = FakeCtx(genes=genes)
    ctx.sample_id = ""
    p = R.plan("feature_vocab.json", legacy, ctx)
    assert p.params == {"build": "GRCh38-2024-A", "schema": "1.0"}
    assert R.regenerate(p.recipe_id, p.params, ctx) == legacy
    ctx.sample_id = "GSM1"  # sample-level member of that name: not ours
    assert R.plan("feature_vocab.json", legacy, ctx) is None


# ------------------------------------------------------------------------------------------------------------ qc
def test_qc_helpers():
    from singlet.v2 import recipes_qc as Q

    assert Q.split_row_names(["ENSG1_A_B_chr1:5-9", "ENSG2__chrM:1-2"], False) == (
        ["ENSG1", "ENSG2"],
        ["A_B", ""],
        ["chr1", "chrM"],
    )
    assert Q.split_row_names(["ENSG1_A_intron_chr1:9-12"], True) == (["ENSG1"], ["A"], ["chr1"])
    assert (
        Q.is_mt_gene("MT-ND1", "chr1")
        and Q.is_mt_gene("mt-Nd1", "5")
        and Q.is_mt_gene("x", "chrM")
        and not Q.is_mt_gene("MTX1", "chr1")
    )
    assert (
        Q.is_ribo_gene("RPL3")
        and Q.is_ribo_gene("Rps6")
        and not Q.is_ribo_gene("RPA1")
        and not Q.is_ribo_gene("RP")
    )
    names = ["G1_MT-X_chrM:1-5", "G1_MT-X_chrM:8-9", "G2_RPL1_chr1:1-5", "G3_ZZ_chr1:1-5"]
    ex = Matrix(
        np.array([0, 3, 4]),
        np.array([0, 1, 3, 2]),
        np.array([2, 2, 4, 6], dtype=np.uint32),
        (4, 2),
        names,
        ["a", "b"],
    )
    q = Q.compute_qc(ex, None)
    assert q["n_umi"].tolist() == [8, 6] and q["n_genes"].tolist() == [2, 1]
    assert q["pct_mt"].tolist() == [np.float32(100.0 / 8 * 4), np.float32(0)] and q[
        "pct_ribo"
    ].tolist()[1] == np.float32(100.0)


# ------------------------------------------------------------------------------------------------------------ invariant under mutation
def _mutations(b, rs, n):
    out = []
    for _ in range(n):
        kind = rs.randint(0, 4)
        i = int(rs.randint(0, max(len(b), 1)))
        if kind == 0 and b:
            out.append(b[:i] + bytes([(b[i] + 1) % 256]) + b[i + 1 :])
        elif kind == 1:
            out.append(b[:i])
        elif kind == 2 and b:
            out.append(b[:i] + b[i + 1 :])
        else:
            out.append(b[:i] + b"\t" + b[i:])
    return out


def test_plan_implies_exact_regeneration_under_mutation():
    """Whatever bytes a member has, a returned plan must regenerate exactly those bytes (verify-or-store invariant)."""
    rs = np.random.RandomState(99)
    n = 120
    bcs, tot = make_cells(n, 0, 31)
    called = np.zeros(n, bool)
    sel = rs.choice(n, 25, replace=False)
    called[sel] = True
    cells = {"barcode": bcs.astype(object), "called": called, "n_umi_exon": tot}
    cc = cc_text(sorted([(bcs[i], True, int(tot[i]), RC.SENTINEL_P) for i in sel]), "star_solo")
    scores = np.round(rs.rand(25), 6)
    dbl = (
        "barcode\ttotal_umis\tdoublet_score\tis_doublet\n"
        + "".join(
            "%s\t%d\t%s\t%s\n" % (bcs[i], tot[i], "%.6g" % s, "True" if s > 0.5 else "False")
            for i, s in sorted(zip(sel, scores), key=lambda t: bcs[t[0]])
        )
    ).encode()
    ab = ("\n".join(bcs) + "\n").encode()
    names = ["g%d" % i for i in range(300)]
    k = rs.geometric(0.1, 300) - 1
    amb = RE.render_ambient(names, k)
    jn = ["chr1:%d-%d" % (rs.randint(1, 8), rs.randint(8, 16)) for _ in range(40)]
    se = RE.render_splice_events(jn)
    mt = RM.render_mtx(rand_csc(20, 30, 0.2, 5))
    cases = [
        ("cell_calls.tsv", cc),
        ("doublet_scores.tsv", dbl),
        ("auto_barcodes.tsv", ab),
        ("ambient_profile.tsv", amb),
        ("splice_events.tsv", se),
        (RS.RAW_MTX, mt),
    ]
    sj = Matrix(
        np.array([0]),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.uint32),
        (len(jn), 0),
        jn,
        [],
    )
    checked = plans = 0
    for member, legacy in cases:
        for mutated in [legacy] + _mutations(legacy, rs, 60):
            ctx = FakeCtx(cells=cells, vocab={"exon": names}, matrices={"sj_counts.1pz": sj})
            p = R.plan(member, mutated, ctx)
            checked += 1
            if p is None:
                continue
            plans += 1
            if member == RS.RAW_MTX:
                got = R.regenerate(p.recipe_id, p.params, ctx)
            else:
                view = dict(cells)
                view.update({c: v[1] for c, v in R.cells_columns(member, mutated, ctx).items()})
                ctx2 = FakeCtx(cells=view, vocab={"exon": names}, matrices={"sj_counts.1pz": sj})
                ctx2._vec = ctx._vec
                got = R.regenerate(p.recipe_id, p.params, ctx2)
            assert got == mutated, (member, p.recipe_id)
    assert plans >= len(cases)  # every unmutated member got a plan
