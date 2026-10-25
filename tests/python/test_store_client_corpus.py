# SPDX-License-Identifier: MIT
"""the store against the legacy reader on a real study: migrate two samples of bench/compression/corpus/bundles/GSE145370.singlet with the
native codec (about 30 s), then require the store AnnData to equal what the legacy reader builds for the same samples (X, spliced and
unspliced layers, var, obs index and columns, uns study_meta), through the local path and through ranged HTTP.

Skipped when the corpus bundle (untracked benchmark data) or the compiled codec is absent. ``SINGLET_TEST_STORE_CORPUS_ROOT`` points at an
already migrated directory (the ``--out`` of ``python -m singlet.store.migrate``) to skip the migration."""

from __future__ import annotations

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

BUNDLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "bench"
    / "compression"
    / "corpus"
    / "bundles"
    / "GSE145370.singlet"
)
GSMS = ["GSM4317426", "GSM4317427"]
pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(), reason="corpus bundle GSE145370.singlet is not available"
)


@pytest.fixture(scope="module")
def migrated(tmp_path_factory):
    env = os.environ.get("SINGLET_TEST_STORE_CORPUS_ROOT")
    if env and (pathlib.Path(env) / "store" / "m" / "GSE145370.json").exists():
        return pathlib.Path(env)
    from singlet.store import migrate as MG

    out = tmp_path_factory.mktemp("corpus_store")
    opts = MG.Options(
        codec="native", recipes="none", created_at="2026-10-22T00:00:00Z", log=lambda *a: None
    )
    res = MG.migrate_bundle(str(BUNDLE), str(out), None, opts, GSMS)
    MG.write_outputs(res, opts)
    return out


@pytest.fixture(scope="module")
def legacy():
    from singlet.bundle import SingletBundle

    b = SingletBundle.open(BUNDLE)
    out = {}
    for g in GSMS:
        mat, bcs = b._load_gsm_gene_counts(g)
        out[g] = (
            mat.tocsr(),
            b._gene_level_part(g, "exon", bcs),
            b._gene_level_part(g, "intron", bcs),
            list(bcs),
        )
    return b, out


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    from singlet.store import client

    client.clear_memory_cache()


def _same(a, b):
    a, b = a.tocsr(), b.tocsr()
    return a.shape == b.shape and (a != b).nnz == 0


def _check(ad, legacy_out, gsms, bundle):
    for g in gsms:
        X, ex, un, bcs = legacy_out[g]
        mask = (ad.obs["gsm_id"] == g).to_numpy()
        got = ad[mask]
        assert list(got.obs_names) == [
            f"{g}_{b}" for b in bcs
        ]  # same cells in the same (cell_calls.tsv) order
        assert (
            _same(got.X, X)
            and _same(got.layers["spliced"], ex)
            and _same(got.layers["unspliced"], un)
        )
        assert got.X.dtype == X.dtype
    genes = bundle.feature_vocab["genes"]
    assert list(ad.var_names) == [x["gene_id"] for x in genes] and list(ad.var["gene_name"]) == [
        x["gene_name"] for x in genes
    ]


def test_store_equals_legacy_through_the_local_path(migrated, legacy, env):
    from singlet.store import client

    b, out = legacy
    ad = client.open_study("GSE145370", base=str(migrated)).to_anndata()
    assert ad.n_obs == sum(len(out[g][3]) for g in GSMS)
    _check(ad, out, GSMS, b)
    st = ad.uns["study_meta"]
    assert st["gse_id"] == "GSE145370"
    assert st["gsm_meta"][GSMS[0]]["organism"] == b.study_meta["gsm_meta"][GSMS[0]]["organism"]
    assert list(ad.obs.columns) == [
        "gsm_id",
        "organism",
        "protocol",
        "protocol_name",
        "sample_source",
        "sample_characteristics",
        "qc_flag",
        "reference_build",
        "n_cells_sample",
    ]


def test_store_equals_legacy_over_ranged_http_and_asks_for_little(migrated, legacy, env):
    import _store_range_server as rs
    from singlet.store import client

    b, out = legacy
    with rs.serve(str(migrated)) as (base, rec):
        st = client.open_study("GSE145370", base=base)
        ad = st.to_anndata([GSMS[0]])
        _check(ad, out, [GSMS[0]], b)
        assert rec.requests == 2
        sm = st.samples[GSMS[0]]
        assert (
            rec.bytes < 0.7 * sm.bytes + 100_000
        )  # the gene-level members only, not the whole object
        rec.log.clear()
        full = st.to_anndata(concurrency=2)
        _check(full, out, GSMS, b)
        assert rec.requests == len(
            GSMS
        )  # one coalesced range per sample (the manifest was cached in memory)


def test_genes_and_cells_slices_against_legacy(migrated, legacy, env):
    from singlet.store import client

    b, out = legacy
    st = client.open_study("GSE145370", base=str(migrated))
    symbols = [x["gene_name"] for x in b.feature_vocab["genes"]][:2000:400]
    sub = st.to_anndata([GSMS[0]], genes=symbols)
    X, ex, un, bcs = out[GSMS[0]]
    keep = np.flatnonzero(np.isin([x["gene_name"] for x in b.feature_vocab["genes"]], symbols))
    assert _same(sub.X, X[:, keep]) and sub.n_vars == len(keep)
    # a cells slice returns exactly the matching columns of the legacy matrix
    cells = st.cells("n_umi_exon >= 400", samples=[GSMS[0]], whole_max=1)
    assert cells.n_obs > 0 and (cells.obs["n_umi_exon"] >= 400).all()
    names = [f"{GSMS[0]}_{x}" for x in bcs]
    pos = {n: i for i, n in enumerate(names)}
    idx = np.array([pos[n] for n in cells.obs_names])
    assert _same(cells.X, X[idx])
    assert _same(
        cells.X,
        (cells.layers["spliced"] + cells.layers["unspliced"])
        if "spliced" in cells.layers
        else cells.X,
    )


def test_storebundle_reads_the_same_modalities(migrated, legacy, env):
    from singlet.store.bundle import StoreBundle

    b, out = legacy
    vb = StoreBundle.open(str(migrated / "store" / "m" / "GSE145370.json"))
    g = GSMS[0]
    assert {"exon_counts.1pz", "intron_counts.1pz", "cell_calls.tsv", "summary.json"} <= set(
        vb.list_files(g)
    )
    assert set(vb.list_files(g)) <= set(b.list_files(g))  # nothing the legacy bundle did not have
    for mod in ("exon_counts", "intron_counts", "junctions", "vdj_gene_usage"):
        if not (b.has(mod, g) and vb.has(mod, g)):
            continue
        lg, v = b.read(mod, g), vb.read(mod, g)
        assert (
            lg.shape == v.shape
            and list(lg.var_names) == list(v.var_names)
            and list(lg.obs_names) == list(v.obs_names)
            and _same(lg.X, v.X)
        ), mod
    for mod in ("summary", "pileup_stats", "provenance"):
        assert b.read(mod, g) == vb.read(mod, g)
    assert b.read("cell_calls", g).equals(vb.read("cell_calls", g))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc_l, rc_v = b.raw_counts(g), vb.raw_counts(g)
    assert list(rc_l.obs_names) == list(rc_v.obs_names) and _same(rc_l.X, rc_v.X)
