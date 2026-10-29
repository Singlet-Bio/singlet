# SPDX-License-Identifier: MIT
"""``modalities=`` on the .s1 read path: a user who asks for some modalities downloads only those members' byte ranges.

Against the synthetic study of _s1_synth built with ``extras=True`` (exon, intron, junction and V(D)J matrices), served both from a
``file://`` directory with the server layout (bytes read from the mapped objects are recorded) and over HTTP by the range server
(bytes on the wire are recorded). Every combination is checked for values (against the independent truth of the builder) and for
bytes (exactly the cells table plus the selected members, never another member). ``test_live_pilot_bytes`` repeats the byte
measurement read-only against the live GSE145370 pilot when ``SINGLET_LIVE=1``."""

from __future__ import annotations

import itertools
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
pytest.importorskip("zstandard")

import _s1_range_server as rs  # noqa: E402
import _s1_synth as synth  # noqa: E402
from singlet.s1 import client  # noqa: E402

GSE = synth.GSE
G1, G2, G3 = "GSM9000001", "GSM9000002", "GSM9000003"
EXON, INTRON, SJ, VDJ = (
    "exon_counts.1pz",
    "intron_counts.1pz",
    "sj_counts.1pz",
    "vdj_gene_usage.1pz",
)
WANTS = {
    "counts": {EXON, INTRON},
    "spliced": {EXON},
    "unspliced": {INTRON},
    "splice_junctions": {SJ},
    "psi": {SJ},
    "vdj": {VDJ},
}


def _layers(adata):
    """Layer names, without the phantom None key anndata 0.13.4 reports for a fresh AnnData."""
    return [k for k in adata.layers if k is not None]


@pytest.fixture(scope="module")
def study_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("storemod")
    try:
        s = synth.Synth(str(root)).build(extras=True)
    except IndexError:  # pragma: no cover - packs not shipped
        pytest.skip("vocabulary packs are not available")
    return root, s


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SINGLET_DATA_BASE", raising=False)
    monkeypatch.delenv("SINGLET_LEGACY_FALLBACK", raising=False)
    client.clear_memory_cache()
    yield tmp_path
    client.clear_memory_cache()


@pytest.fixture()
def mmap_log(monkeypatch):
    """Every (offset, length) read from a locally mapped sample object."""
    log = []
    orig = client._MmapSource.read

    def read(self, off, length, *, parallel=False):
        log.append((int(off), int(length)))
        return orig(self, off, length, parallel=parallel)

    monkeypatch.setattr(client._MmapSource, "read", read)
    return log


def _same(a, b):
    a, b = a.tocsr(), b.tocsr()
    return a.shape == b.shape and (a != b).nnz == 0


def _order(truth, gsm):
    """Matrix columns of the called cells in sorted-barcode order (the default ``cell_order``)."""
    t = truth[gsm]
    called = np.flatnonzero(t["called"])
    bcs = np.array(t["barcodes"], dtype=object)[called]
    return called[np.argsort(bcs, kind="stable")]


def _owner(sm, a, b):
    """Names of the members whose byte range intersects [a, b)."""
    return {n for n, m in sm.members.items() if m.len and a < m.off + m.len and m.off < b}


def _check_ranges(sm, ranges, want_members):
    """The ranges touch only the cells table and the wanted members, and cover each of them."""
    touched = set()
    for a, b in ranges:
        touched |= _owner(sm, a, b)
    assert touched == {"cells"} | set(want_members), touched


# =================================================================================================================== vocabulary
def test_vocabulary_aliases_and_errors(study_dir, env):
    root, s = study_dir
    assert client.normalize_modalities(None) == ("counts", "spliced", "unspliced")
    assert client.normalize_modalities(None, layers=False) == ("counts",)
    assert client.normalize_modalities(["sj", "SPLICED", "junctions"]) == (
        "spliced",
        "splice_junctions",
    )
    assert client.normalize_modalities("splice_psi") == ("psi",)
    with pytest.raises(client.ModalityUnavailable, match="unknown modality 'bogus'.*vocabulary"):
        client.normalize_modalities(["counts", "bogus"])
    st = client.open_study(GSE, base=str(root))
    assert st.modalities() == ["counts", "spliced", "unspliced", "splice_junctions", "psi", "vdj"]
    assert st.modalities(G3) == ["counts", "spliced", "splice_junctions", "psi"]
    # mt is in no sample; unspliced is not in G3: the error names what there is, before any sample byte is read
    with pytest.raises(client.ModalityUnavailable, match=r"\['mt'\].*splice_junctions"):
        st.to_anndata([G1], modalities=["counts", "mt"])
    with pytest.raises(client.ModalityUnavailable, match=f"{G3} has"):
        st.to_anndata([G3], modalities="unspliced")
    with pytest.raises(client.ModalityUnavailable):
        st.cells("called", samples=[G3], modalities="vdj")


# =================================================================================================================== bytes
COMBOS = [
    c
    for r in (1, 2)
    for c in itertools.combinations(
        ["counts", "spliced", "unspliced", "splice_junctions", "psi", "vdj"], r
    )
]


@pytest.mark.parametrize("mods", COMBOS, ids=lambda c: "+".join(c))
def test_file_layout_reads_only_the_selected_members(study_dir, env, mmap_log, mods):
    root, s = study_dir
    st = client.open_study(GSE, base=pathlib.Path(root).as_uri())
    sm = st.samples[G1]
    ad = st.to_anndata([G1], modalities=list(mods))
    want = set().union(*(WANTS[m] for m in mods))
    ranges = [(a, a + n) for a, n in mmap_log]
    _check_ranges(sm, ranges, want)
    exact = sm.members["cells"].len + sum(sm.members[n].len for n in want)
    # adjacent members are read as one range, which spans the 64-byte alignment gaps between them
    assert exact <= sum(n for _, n in mmap_log) <= exact + 64 * len(want)
    assert ad.uns["singlet_s1"]["modalities"] == [m for m in client.MODALITIES if m in mods]
    assert (ad.X is not None) == ("counts" in mods)


@pytest.mark.parametrize(
    "mods",
    [None, ("counts",), ("spliced",), ("unspliced",), ("splice_junctions",), ("psi", "vdj")],
    ids=lambda c: "default" if c is None else "+".join(c),
)
def test_http_bytes_on_the_wire(study_dir, env, mods):
    root, s = study_dir
    with rs.serve(str(root)) as (base, rec):
        st = client.open_study(GSE, base=base)
        sm = st.samples[G1]
        rec.log.clear()
        st.to_anndata([G1], modalities=None if mods is None else list(mods))
        got = [r for r in rec.log if r[1].startswith("/s1/s/")]
        want = set().union(*(WANTS[m] for m in (mods or ("counts", "spliced", "unspliced"))))
        _check_ranges(sm, [(r[2][0], r[2][1] + 1) for r in got], want)
        exact = sm.members["cells"].len + sum(sm.members[n].len for n in want)
        # one coalesced request may span the 64-byte alignment gaps between adjacent members
        nb = sum(r[4] for r in got)
        assert exact <= nb <= exact + 64 * len(want)
        plan = st.request_plan([G1], modalities=mods, include_manifest=False)
        assert plan["bytes"] == nb and plan["requests"] == len(got)


def test_counts_only_equals_default_bytes_spliced_and_sj_fetch_less(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    b = {
        k: st.request_plan([G1], modalities=k, include_manifest=False)["bytes"]
        for k in (None, "counts", "spliced", "splice_junctions")
    }
    sm = st.samples[G1]
    assert b[None] == b["counts"]  # counts needs both matrices: storage stays exon + intron
    assert (
        b["spliced"] < b["counts"]
        and b["spliced"] - sm.members["cells"].len <= sm.members[EXON].len + 64
    )
    assert b["splice_junctions"] <= sm.members["cells"].len + sm.members[SJ].len + 64


def test_where_slicing_fetches_only_the_chunk_runs_of_the_selected_members(
    study_dir, env, monkeypatch
):
    root, s = study_dir
    monkeypatch.setattr(client, "PREFIX_HINT", 4096)
    monkeypatch.setattr(client, "PARTIAL_GAP", 64)
    t = s.truth[G1]
    thr = int(np.sort(t["umi_exon"])[-12])  # the 12 deepest cells: the first chunk of 64 columns
    idx = np.flatnonzero(t["umi_exon"] >= thr)
    with rs.serve(str(root)) as (base, rec):
        st = client.open_study(GSE, base=base)
        sm = st.samples[G1]
        rec.log.clear()
        sub = st.cells(
            f"n_umi_exon >= {thr}", samples=[G1], modalities="splice_junctions", whole_max=1
        )
        got = [r for r in rec.log if r[1].startswith("/s1/s/")]
        _check_ranges(sm, [(r[2][0], r[2][1] + 1) for r in got], {SJ})
        m = sm.members[SJ]
        in_sj = sum(max(0, min(r[2][1] + 1, m.off + m.len) - max(r[2][0], m.off)) for r in got)
        assert (
            0 < in_sj < 0.6 * m.len
        )  # the 4 KB prefix (header, index, names) and one chunk run of five
        assert list(sub.obs["column"]) == idx.tolist() and sub.X is None
        assert _same(sub.obsm["splice_junctions"], t["sj"][:, idx].T)
        assert list(sub.uns["obsm_features"]["splice_junctions"]) == t["sj_names"]
        rec.log.clear()
        sp = st.cells(f"n_umi_exon >= {thr}", samples=[G1], modalities="spliced", whole_max=1)
        got = [r for r in rec.log if r[1].startswith("/s1/s/")]
        _check_ranges(sm, [(r[2][0], r[2][1] + 1) for r in got], {EXON})
        assert _same(sp.layers["spliced"], t["exon_g"][:, idx].T)


# =================================================================================================================== values
def test_gene_modalities_values(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    t = s.truth[G1]
    cols = _order(s.truth, G1)
    ex, un = t["exon_g"][:, cols].T, t["intron_g"][:, cols].T
    a = st.to_anndata([G1], modalities="spliced")
    assert a.X is None and _layers(a) == ["spliced"] and _same(a.layers["spliced"], ex)
    b = st.to_anndata([G1], modalities="unspliced")
    assert b.X is None and _layers(b) == ["unspliced"] and _same(b.layers["unspliced"], un)
    c = st.to_anndata([G1], modalities="counts")
    assert _same(c.X, ex + un) and not _layers(c)
    d = st.to_anndata([G1], modalities=["counts", "spliced"])
    assert _same(d.X, ex + un) and _layers(d) == ["spliced"]
    assert list(a.obs_names) == list(c.obs_names) and a.var_names.equals(c.var_names)
    # genes= composes with modalities
    g = s.exon_pack.gene_table()[5][0]
    e = st.to_anndata([G1], modalities="spliced", genes=[g])
    assert e.shape[1] == 1 and _same(e.layers["spliced"], ex[:, [5]])


def test_feature_modalities_values_union_and_missing_rows(study_dir, env):
    from singlet.s1.recipes_ctx import Matrix
    from singlet.s1.recipes_psi import compute_psi

    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    with pytest.warns(UserWarning, match=r"vdj: \['GSM9000003'\] have no vdj_gene_usage.1pz"):
        ad = st.to_anndata([G1, G3], modalities=["splice_junctions", "psi", "vdj"])
    assert ad.X is None and ad.n_vars == 0 and ad.n_obs == 40 + 25
    names = list(ad.uns["obsm_features"]["splice_junctions"])
    assert (
        names == s.truth[G1]["sj_names"] + s.truth[G3]["sj_names"][30:]
    )  # union, first-seen order
    n1 = 40
    for g, rows in ((G1, slice(0, n1)), (G3, slice(n1, None))):
        t = s.truth[g]
        cols = _order(s.truth, g)
        pos = [names.index(x) for x in t["sj_names"]]
        sj = ad.obsm["splice_junctions"][rows][:, pos]
        assert _same(sj, t["sj"][:, cols].T)
        sub = t["sj"][:, cols].tocsc()
        sub.sort_indices()
        ip, rr, psi, _ = compute_psi(
            Matrix(sub.indptr, sub.indices, sub.data, sub.shape, t["sj_names"], None, None)
        )
        import scipy.sparse as sp

        want = sp.csc_matrix((psi, rr, ip), shape=sub.shape).T
        got = ad.obsm["psi"][rows][:, pos]
        assert got.dtype == np.float32 and _same(got, want)
        assert 0 < got.max() <= 1
    vdj = ad.obsm["vdj"]
    assert vdj[n1:].nnz == 0 and _same(vdj[:n1], s.truth[G1]["vdj"][:, _order(s.truth, G1)].T)
    one = client.modality_anndata(ad, "junctions")
    assert one.shape == (65, len(names)) and list(one.var_names) == names
    assert one.obs_names.equals(ad.obs_names)


def test_obs_filter_and_skips_compose(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad = st.to_anndata(modalities="unspliced")  # G2: no called cells; G3: no intron matrix
    assert set(ad.obs["gsm_id"]) == {G1} and set(ad.uns["skipped_samples"]["gsm_id"]) == {G2, G3}
    f = st.to_anndata(obs_filter={"gsm_id": G3}, modalities="splice_junctions")
    assert set(f.obs["gsm_id"]) == {G3} and f.obsm["splice_junctions"].shape[0] == 25


def test_load_s2_and_loader_pass_modalities(study_dir, env, monkeypatch):
    import singlet._loader as L

    root, s = study_dir
    path = os.path.join(root, *s.truth[G1]["key"].split("/"))
    a = client.load_s2(path, modalities="splice_junctions")
    assert a.X is None and a.obsm["splice_junctions"].shape == (40, 60)
    with pytest.raises(client.ModalityUnavailable):
        client.load_s2(path, modalities="mt")
    monkeypatch.setenv("SINGLET_DATA_BASE", str(root))
    b = L.load(GSE, samples=[G1], modalities=["spliced"])
    assert b.X is None and _layers(b) == ["spliced"]
    c = L.load([GSE, GSE], samples=[G1], modalities="counts")
    assert c.n_obs == 80 and not _layers(c)


def test_legacy_fallback_subsets_after_download_and_warns(study_dir, env, monkeypatch):
    import anndata as ad
    import scipy.sparse as sp
    import singlet._loader as L

    monkeypatch.setenv("SINGLET_DATA_BASE", str(study_dir[0]))
    monkeypatch.setenv("SINGLET_LEGACY_FALLBACK", "1")  # off by default since the retirement
    monkeypatch.setattr(L, "_download_legacy", lambda acc, *a, **k: pathlib.Path("legacy.singlet"))
    X = sp.csr_matrix(np.arange(1, 7).reshape(3, 2))
    full = ad.AnnData(X=X.copy())
    full.obs_names = ["GSM1_AAA", "GSM1_CCC", "GSM1_GGG"]
    full.obs["gsm_id"] = "GSM1"
    full.layers["spliced"] = X.copy()
    full.layers["unspliced"] = X.copy()
    junc = ad.AnnData(X=sp.csr_matrix(np.array([[0, 5], [3, 0]])))
    junc.obs_names = ["GGG", "AAA"]
    junc.var_names = ["chr1:10-20", "chr1:10-30"]

    class FakeBundle:
        gsm_ids = ["GSM1"]

        @classmethod
        def open(cls, p, restricted=None):
            return cls()

        def to_anndata(self, **k):
            return full.copy()

        def modalities(self, gsm):
            return ["junctions"]

        def read(self, name, gsm):
            assert name == "junctions"
            return junc

    import singlet.bundle as B

    monkeypatch.setattr(B, "SingletBundle", FakeBundle)
    with pytest.warns(UserWarning, match="downloaded whole"):
        out = L.load("GSE0", modalities=["spliced", "splice_junctions", "vdj"])
    assert out.X is None and _layers(out) == ["spliced"]
    assert list(out.uns["obsm_features"]["splice_junctions"]) == list(junc.var_names)
    sj = out.obsm["splice_junctions"].toarray()
    assert sj.tolist() == [[3, 0], [0, 0], [0, 5]] and "vdj" not in out.obsm
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no modalities: the legacy result is untouched, no warning
        same = L.load("GSE0", legacy=True)
    assert set(_layers(same)) == {"spliced", "unspliced"} and same.X is not None


# =================================================================================================================== live pilot
@pytest.mark.skipif(
    os.environ.get("SINGLET_LIVE") != "1", reason="set SINGLET_LIVE=1 to read the live pilot"
)
def test_live_pilot_bytes(env):
    """Read-only: one GSE145370 sample from data.singlet.bio, default vs counts / spliced / splice_junctions."""
    st = client.open_study("GSE145370", base="https://data.singlet.bio")
    g = st.gsm_ids[0]
    sm = st.samples[g]
    out = {}
    for mods in (None, "counts", "spliced", "splice_junctions"):
        st.stats.requests = st.stats.bytes = 0
        ad = st.to_anndata([g], modalities=mods)
        out[mods or "default"] = st.stats.bytes
        assert ad.n_obs == sm.n_called
    print("\nlive", g, "object", sm.bytes, out)
    assert out["counts"] == out["default"]
    assert out["spliced"] < out["counts"] and out["splice_junctions"] < sm.bytes
