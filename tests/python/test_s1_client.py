# SPDX-License-Identifier: MIT
"""singlet.s1.client / transport / bundle against a synthetic per-sample study (real codec, real vocabulary packs, W2's container writer):
gene-level loading checked against an independent computation from the row names, the request plan (how many requests, how many
bytes), the two-phase plan for large samples, cell slicing, integrity failures, retries, offline use, and the loader routing.

Everything is local: no network, no corpus. Skipped when ``singlet._pz2`` or the shipped packs are missing."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
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
from singlet.s1 import transport as T  # noqa: E402

GSE = synth.GSE
G1, G2, G3 = "GSM9000001", "GSM9000002", "GSM9000003"


@pytest.fixture(scope="module")
def study_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("storestudy")
    try:
        s = synth.Synth(str(root)).build()
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


def _expected(truth, gsm, *, order="barcode"):
    """(X, spliced, unspliced, barcodes) of the called cells, cells x genes CSR, in sorted-barcode order."""
    t = truth[gsm]
    called = np.flatnonzero(t["called"])
    bcs = np.array(t["barcodes"], dtype=object)[called]
    perm = np.argsort(bcs, kind="stable") if order == "barcode" else np.arange(len(called))
    cols = called[perm]
    ex = t["exon_g"][:, cols].T.tocsr()
    un = t["intron_g"][:, cols].T.tocsr() if t["intron_g"] is not None else None
    X = ex + un if un is not None else ex
    return X.tocsr(), ex, un, bcs[perm]


def _same(a, b):
    a, b = a.tocsr(), b.tocsr()
    return a.shape == b.shape and (a != b).nnz == 0


# =================================================================================================================== loading
def test_load_one_sample_matches_independent_computation(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    ad = st.to_anndata([G1])
    X, ex, un, bcs = _expected(s.truth, G1)
    assert ad.shape == (40, len(s.truth[G1]["gene_ids"]))
    assert _same(ad.X, X) and _same(ad.layers["spliced"], ex) and _same(ad.layers["unspliced"], un)
    assert ad.X.dtype == np.int32 and ad.layers["spliced"].dtype == np.int32
    assert list(ad.obs_names) == [f"{G1}_{b}" for b in bcs]
    assert list(ad.var_names) == s.truth[G1]["gene_ids"] and ad.var.index.name == "gene_id"
    assert list(ad.var["gene_name"]) == [n for _, n in s.exon_pack.gene_table()]
    assert set(ad.obs.columns) >= {
        "gsm_id",
        "organism",
        "protocol",
        "qc_flag",
        "reference_build",
        "n_cells_sample",
    }
    assert ad.obs["organism"].iloc[0] == "Homo sapiens" and ad.uns["study_meta"]["gse_id"] == GSE
    assert ad.uns["manifest"]["gsm_ids"] == [G1] and ad.uns["singlet_s1"]["gse"] == GSE


def test_cell_order_options(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    col = st.to_anndata([G1], cell_order="column")
    X, *_ = _expected(s.truth, G1, order="column")
    assert _same(col.X, X)
    bar = st.to_anndata([G1], cell_order="barcode")
    assert list(bar.obs_names) == sorted(bar.obs_names)
    with pytest.raises(ValueError):
        st.to_anndata([G1], cell_order="nope")


def test_sample_without_intron_and_study_level_layers(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    one = st.to_anndata([G3])
    X, ex, un, bcs = _expected(s.truth, G3)
    assert (
        un is None and _same(one.X, X) and "spliced" in one.layers and "unspliced" not in one.layers
    )
    both = st.to_anndata([G1, G3])  # a layer is kept only when every sample has it
    assert both.n_obs == 40 + 25 and "unspliced" not in both.layers and "spliced" in both.layers


def test_skipped_sample_is_reported_like_the_legacy_reader(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    with pytest.warns(UserWarning, match=G2):
        ad = st.to_anndata()
    assert ad.n_obs == 40 + 25
    sk = ad.uns["skipped_samples"]
    assert list(sk["gsm_id"]) == [G2] and sk["reason"].iloc[0] == "0 cells called"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(client.NoUsableSamples) as ei:
            st.to_anndata([G2])
    assert ei.value.reasons[G2] == "0 cells called"
    with pytest.raises(KeyError):
        st.to_anndata(["GSM0000000"])


def test_genes_subset_symbols_ids_and_warning(study_dir, env):
    root, s = study_dir
    import singlet._loader as L

    st = client.open_study(GSE, base=str(root))
    tbl = s.exon_pack.gene_table()
    ids = [g for g, _ in tbl]
    sym = tbl[10][1]
    with pytest.warns(UserWarning, match="not found"):
        ad = st.to_anndata([G1], genes=[sym.lower(), ids[5], "NOT_A_GENE"])
    full = st.to_anndata([G1])
    keep = L._gene_keep_mask(ids, [n for _, n in tbl], [sym.lower(), ids[5]])
    assert list(ad.var_names) == list(np.array(ids)[keep])
    assert _same(ad.X, full.X[:, np.flatnonzero(keep)]) and _same(
        ad.layers["spliced"], full.layers["spliced"][:, np.flatnonzero(keep)]
    )
    with pytest.warns(UserWarning, match="None of the"):
        none = st.to_anndata([G1], genes=["NOT_A_GENE"])
    assert none.shape[1] == 0


def test_obs_filter_prefilters_samples_and_filters_cells(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    a = st.to_anndata(obs_filter={"gsm_id": G3})
    assert a.n_obs == 25 and set(a.obs["gsm_id"]) == {G3}
    b = st.to_anndata([G1, G3], obs_filter={"organism": ["Homo sapiens"], "gsm_id": [G1]})
    assert b.n_obs == 40
    with pytest.raises(KeyError):
        st.to_anndata([G1], obs_filter={"nope": 1})


def test_concurrent_study_load_equals_serial(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    serial = st.to_anndata(concurrency=1)
    par = st.to_anndata(concurrency=3, threads=1)
    assert _same(serial.X, par.X) and list(serial.obs_names) == list(par.obs_names)


# =================================================================================================================== network
def test_one_sample_is_two_requests_and_member_bytes_only(study_dir, env):
    root, s = study_dir
    with rs.serve(str(root)) as (base, rec):
        st = client.open_study(GSE, base=base)
        assert rec.requests == 1 and rec.log[0][1] == f"/s1/m/{GSE}.json"
        ad = st.to_anndata([G1])
        assert ad.n_obs == 40 and rec.requests == 2
        sm = st.samples[G1]
        want = sum(sm.members[n].len for n in ("cells", "exon_counts.1pz", "intron_counts.1pz"))
        got = [r for r in rec.log if r[1].startswith("/s1/s/")]
        # one coalesced range: the three members plus at most the 64-byte alignment gaps between them
        assert len(got) == 1 and got[0][3] == 206 and want <= got[0][4] <= want + 2 * 64
        assert want < sm.bytes
        assert "singlet-client" in next(iter(rec.user_agents))


def test_study_load_is_one_request_per_sample(study_dir, env):
    root, s = study_dir
    with rs.serve(str(root)) as (base, rec):
        st = client.open_study(GSE, base=base)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.to_anndata(concurrency=3)
        assert rec.requests == 1 + 3
        plan = st.request_plan()
        assert plan["requests"] == 1 + 3 and plan["bytes"] > 0


def test_manifest_is_revalidated_with_etag_and_survives_offline(study_dir, env, monkeypatch):
    root, s = study_dir
    with rs.serve(str(root)) as (base, rec):
        client.open_study(GSE, base=base)
        client.clear_memory_cache()
        client.open_study(GSE, base=base)
        assert [r[3] for r in rec.log] == [
            200,
            304,
        ]  # second open: conditional request, nothing re-downloaded
        client._transport_for(
            base
        ).close()  # drop the kept-alive connection: a stopped server's handler threads keep serving it
    client.clear_memory_cache()
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.warns(
        RuntimeWarning, match="cached manifest"
    ):  # server gone: the cached copy is used
        st = client.open_study(GSE, base=base)
    assert st.gse == GSE and st.source["origin"] == "cache-offline"


def test_missing_study_is_studynotfound_with_a_hint(study_dir, env):
    root, s = study_dir
    with rs.serve(str(root)) as (base, rec):
        with pytest.raises(client.StudyNotFound, match="no .s1 manifest"):
            client.open_study("GSE1", base=base)
    with pytest.raises(client.StudyNotFound):
        client.open_study("GSE1", base=str(root))


def test_transient_server_errors_are_retried(study_dir, env, monkeypatch):
    root, s = study_dir
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with rs.serve(str(root)) as (base, rec):
        rec.fail_next.extend([503, 502])
        st = client.open_study(GSE, base=base)
        assert (
            sorted(r[3] for r in rec.log[:3]) == [200, 502, 503] and st.stats.retries == 2
        )  # (log order races the client)


def test_clear_memory_cache_forgets_transports(study_dir, env, monkeypatch):
    """A later server on a reused port must not inherit an earlier transport's connections or retry counts."""
    root, s = study_dir
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with rs.serve(str(root)) as (base, rec):
        rec.fail_next.extend([503])
        assert client.open_study(GSE, base=base).stats.retries == 1
        client.clear_memory_cache()
        assert client._transport_for(base).stats.retries == 0


def test_wrong_object_size_is_refused(study_dir, env):
    root, s = study_dir
    with rs.serve(str(root), total_override=12345) as (base, rec):
        st = client.open_study(GSE, base=base)
        with pytest.raises(T.IntegrityError, match="expected"):
            st.to_anndata([G1])


def test_server_that_ignores_range_is_refused(study_dir, env):
    root, s = study_dir
    with rs.serve(str(root), ignore_range=True) as (base, rec):
        st = client.open_study(GSE, base=base)
        with pytest.raises(T.IntegrityError, match="ignored the Range"):
            st.to_anndata([G1])


def test_a_manifest_digest_that_does_not_match_the_object_is_refused(study_dir, env, tmp_path):
    root, s = study_dir
    mirror = tmp_path / "mirror"
    shutil.copytree(root, mirror)
    mp = mirror / "s1" / "m" / f"{GSE}.json"
    doc = json.loads(mp.read_text())
    doc["samples"][0]["members"]["exon_counts.1pz"]["digest"] = "00" * 16
    mp.write_text(json.dumps(doc))
    st = client.open_study(GSE, base=str(mirror))
    with pytest.raises(client.S1Error, match="digest"):
        st.to_anndata([G1])


def test_verify_recomputes_canonical_digests(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    ad = st.to_anndata([G1], verify=True)
    assert ad.n_obs == 40


# =================================================================================================================== two-phase plan, cells
def test_two_phase_plan_gives_the_same_matrix_and_fewer_bytes(study_dir, env, monkeypatch):
    root, s = study_dir
    monkeypatch.setattr(
        client, "PREFIX_HINT", 4096
    )  # the synthetic members are small: do not read them whole as a "prefix"
    monkeypatch.setattr(
        client, "PARTIAL_GAP", 64
    )  # ... and do not bridge the gap between members to save a round trip
    with rs.serve(str(root)) as (base, rec):
        st = client.open_study(GSE, base=base)
        whole = st.to_anndata([G1])
        rec.log.clear()
        part = st.to_anndata([G1], whole_max=1)
        assert _same(whole.X, part.X) and _same(whole.layers["unspliced"], part.layers["unspliced"])
        assert list(whole.obs_names) == list(part.obs_names)
        reqs = [r for r in rec.log if r[1].startswith("/s1/s/")]
        assert (
            3 <= len(reqs) <= 8
        )  # cells + prefixes (2 phases of concurrent reads), then the chunk runs
        sm = st.samples[G1]
        mem_bytes = sum(
            sm.members[n].len for n in ("cells", "exon_counts.1pz", "intron_counts.1pz")
        )
        assert (
            sum(r[4] for r in reqs) < mem_bytes
        )  # chunks that hold no called cell are never fetched


def test_cells_slice_fetches_only_matching_chunks(study_dir, env, monkeypatch):
    root, s = study_dir
    monkeypatch.setattr(client, "PREFIX_HINT", 4096)
    monkeypatch.setattr(client, "PARTIAL_GAP", 64)
    t = s.truth[G1]
    thr = int(np.median(t["umi_exon"]))
    with rs.serve(str(root)) as (base, rec):
        st = client.open_study(GSE, base=base)
        rec.log.clear()
        sub = st.cells(f"n_umi_exon >= {thr}", samples=[G1], whole_max=1)
        idx = np.flatnonzero(t["umi_exon"] >= thr)
        assert list(sub.obs["column"]) == idx.tolist() and sub.n_obs == len(idx)
        want = (t["exon_g"][:, idx] + t["intron_g"][:, idx]).T.tocsr()
        assert _same(sub.X, want)
        assert "n_umi_exon" in sub.obs.columns and sub.obs["gsm_id"].iloc[0] == G1
        got = sum(r[4] for r in rec.log if r[1].startswith("/s1/s/"))
        assert got < st.samples[G1].bytes
    with rs.serve(str(root)) as (base, rec):  # dict / callable / max_cells forms, and layers
        st = client.open_study(GSE, base=base)
        a = st.cells({"n_umi_exon": (thr, None)}, samples=[G1], layers=True)
        b = st.cells(lambda df: df["n_umi_exon"] >= thr, samples=[G1])
        c = st.cells(None, samples=[G1], max_cells=7)
        assert _same(a.X, b.X) and "spliced" in a.layers and c.n_obs == 7
        assert list(c.obs["column"]) == list(range(7))


def test_cells_across_samples(study_dir, env):
    root, s = study_dir
    st = client.open_study(GSE, base=str(root))
    sub = st.cells("called", samples=[G1, G3])
    assert sub.n_obs == 40 + 25 and set(sub.obs["gsm_id"]) == {G1, G3}


# =================================================================================================================== local files
def test_load_s2_and_manifest_paths(study_dir, env):
    root, s = study_dir
    key = s.truth[G1]["key"]
    s2 = root.joinpath(*key.split("/"))
    ad = client.load(str(s2))
    X, ex, un, bcs = _expected(s.truth, G1)
    assert _same(ad.X, X) and _same(ad.layers["spliced"], ex)
    assert list(ad.obs_names) == [f"{G1}_{b}" for b in bcs]
    st_ad = client.load(str(root / "s1" / "m" / f"{GSE}.json"), samples=[G1])
    assert _same(st_ad.X, ad.X)


def test_download_then_offline_load_uses_local_copies(study_dir, env):
    root, s = study_dir
    import singlet._loader as L

    with rs.serve(str(root)) as (base, rec):
        os.environ["SINGLET_DATA_BASE"] = base
        try:
            mp = L.download(GSE)
            assert mp.name == f"{GSE}.json" and mp.exists()
            n_after_download = rec.requests
            before = client.open_study(GSE).to_anndata([G1])
            sample_reqs = [r for r in rec.log[n_after_download:] if r[1].startswith("/s1/s/")]
            assert sample_reqs == []  # the object is read from the downloaded copy
        finally:
            os.environ.pop("SINGLET_DATA_BASE", None)
    X, *_ = _expected(s.truth, G1)
    assert _same(before.X, X)
    assert L.open_bundle(mp).gsm_ids == [G1, G2, G3]
    assert L.load(str(mp), samples=[G3]).n_obs == 25


# =================================================================================================================== bundle API
def test_storebundle_has_the_singletbundle_interface(study_dir, env):
    root, s = study_dir
    from singlet.bundle import SingletBundle
    from singlet.s1.bundle import S1Bundle

    b = SingletBundle.open(str(root / "s1" / "m" / f"{GSE}.json"))
    assert isinstance(b, S1Bundle) and isinstance(b, SingletBundle)
    assert (
        b.gsm_ids == [G1, G2, G3]
        and b.manifest["gse_id"] == GSE
        and b.study_meta["series_title"] == "synthetic"
    )
    assert (
        b.feature_vocab["n_genes"] == len(s.exon_pack.gene_table())
        and b.feature_vocab["reference_build"] == synth.BUILD
    )
    assert "exon_counts.1pz" in b.list_files(G1) and "cell_calls.tsv" in b.list_files(G1)
    assert b.has("exon_counts", G1) and not b.has("intron_counts", G3)
    assert {"exon_counts", "cell_calls", "summary"} <= set(b.modalities(G1))
    assert b.read("summary", G1)["gsm"] == G1
    cc = b.read("cell_calls", G1)
    assert len(cc) == 40 and list(cc.columns) == ["barcode", "is_cell"]
    ex = b.read("exon_counts", G1)
    assert ex.shape == (300, len(s.exon_names)) and list(ex.var_names[:3]) == s.exon_names[:3]
    assert _same(ex.X.T, s.truth[G1]["exon"].astype(np.uint32)) and ex.X.dtype == np.uint32
    assert list(ex.obs_names) == s.truth[G1]["barcodes"] and ex.obs["gsm_id"].iloc[0] == G1
    with pytest.raises(KeyError):
        b.read("intron_counts", G3)
    rc = b.raw_counts(G1)
    X, ex_, un_, bcs = _expected(s.truth, G1)
    assert _same(rc.X, X) and list(rc.obs_names) == list(bcs) and list(rc.obs.columns) == ["gsm_id"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = b.to_anndata(verbose=False)
    assert whole.n_obs == 65
    assert "S1Bundle" in repr(b)


# =================================================================================================================== loader routing
def test_loader_routes_accessions_to_store_and_legacy_is_explicit(study_dir, env, monkeypatch):
    root, s = study_dir
    import singlet._loader as L

    monkeypatch.setenv("SINGLET_DATA_BASE", str(root))
    ad = L.load(GSE, samples=[G1])
    assert ad.n_obs == 40 and "spliced" in ad.layers
    # anndata 0.13.4 reports an empty `layers` as one key, None: check for our layers by name
    no_layers = L.load(GSE, samples=[G1], layers=False).layers
    assert "spliced" not in no_layers and "unspliced" not in no_layers
    monkeypatch.setattr(L, "_resolve_gsm_to_gse", lambda g: GSE)
    one = L.load(G3)
    assert one.n_obs == 25 and set(one.obs["gsm_id"]) == {G3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(KeyError, match="no usable cells"):
            L.load(G2)
    with pytest.raises(TypeError, match="unexpected keyword"):
        L.load(GSE, bogus=1)
    sub = L.load([GSE, GSE], samples=[G1])  # lists concatenate (legacy behaviour)
    assert sub.n_obs == 80
    genes = L.load(GSE, samples=[G1], genes=[s.exon_pack.gene_table()[3][0]])
    assert genes.shape == (40, 1)


def test_unmigrated_study_fallback_and_legacy_true(study_dir, env, monkeypatch):
    root, s = study_dir
    import singlet._loader as L

    monkeypatch.setenv("SINGLET_DATA_BASE", str(root))
    calls = []
    monkeypatch.setattr(
        L,
        "_download_legacy",
        lambda acc, *a, **k: calls.append(acc) or pathlib.Path("legacy.singlet"),
    )

    class FakeBundle:
        def __init__(self, *a):
            pass

        @classmethod
        def open(cls, p, restricted=None):
            return cls()

        def to_anndata(self, **k):
            return "legacy-adata"

    import singlet.bundle as B

    monkeypatch.setattr(B, "SingletBundle", FakeBundle)
    # since the retirement the fallback is off by default: a study without a manifest is not in the atlas, and no
    # legacy bundle is requested
    with pytest.raises(client.StudyNotFound, match="not in the atlas"):
        L.load("GSE0")
    monkeypatch.setenv("SINGLET_LEGACY_FALLBACK", "0")
    with pytest.raises(client.StudyNotFound, match="known-issues"):
        L.load("GSE1")
    assert calls == []
    assert L.load("GSE1", legacy=True) == "legacy-adata" and calls == ["GSE1"]
    monkeypatch.setenv("SINGLET_LEGACY_FALLBACK", "1")  # the transition switch
    with pytest.warns(FutureWarning, match="legacy bundle"):
        assert L.load("GSE2") == "legacy-adata"
    with pytest.raises(client.StudyNotFound):
        L.load("GSE3", legacy=False)
    assert L.download("GSE1", legacy=True).name == "legacy.singlet"
