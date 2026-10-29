# SPDX-License-Identifier: MIT
"""singlet.torch.AtlasCells: study-held-out splits, cell-level filters, the gene axis, and the streamed batches (every matching
cell once per epoch, counts equal to the stored matrix, a new order per epoch, disjoint worker shards).

The unit tests need numpy and pandas. The fixture tests stream a synthetic labelled study (tests/python/_s1_synth.py) and need
``singlet._pz2`` and torch. ``test_live_*`` stream from data.singlet.bio and run only with ``SINGLET_LIVE_TESTS=1``."""

from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _load_stream():
    """singlet/torch/_stream.py without the singlet.torch package (whose import needs torch), so the unit tests run without it."""
    import importlib.util

    import singlet

    p = pathlib.Path(singlet.__file__).resolve().parent / "torch" / "_stream.py"
    spec = importlib.util.spec_from_file_location("singlet.torch._stream", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("singlet.torch._stream", mod)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return sys.modules["singlet.torch._stream"]


S = _load_stream()

GSE = "GSE900001"
G1, G3 = "GSM9000001", "GSM9000003"


# =================================================================================================================== unit
def test_study_split_is_deterministic_and_follows_the_fractions():
    gses = [f"GSE{100000 + i}" for i in range(20000)]
    a = [S.study_split(g) for g in gses]
    assert a == [S.study_split(g.lower()) for g in gses]  # case-insensitive, stable
    frac = pd.Series(a).value_counts(normalize=True)
    assert (
        abs(frac["train"] - 0.8) < 0.01
        and abs(frac["val"] - 0.1) < 0.01
        and abs(frac["test"] - 0.1) < 0.01
    )
    b = [S.study_split(g, seed=1) for g in gses]
    assert a != b  # another seed, another partition
    assert {S.study_split(g, (0.0, 0.0, 1.0)) for g in gses[:100]} == {"test"}
    with pytest.raises(ValueError):
        S.study_split("GSE1", (0.5, 0.5, 0.5))


def test_path_level():
    paths = ["CL:0000988>CL:0000542>CL:0000084", "CL:0000066", "", None]
    assert S.path_level(paths, 1) == ["CL:0000988", "CL:0000066", None, None]
    assert S.path_level(paths, 3) == ["CL:0000084", None, None, None]


def _obs():
    return pd.DataFrame(
        {
            "called": [True, True, True, False, True],
            "cell_type_path": [
                "CL:0000988>CL:0000542>CL:0000084>CL:0000624",  # CD4 T cell, a T cell
                "CL:0000988>CL:0000542",  # lymphocyte, not a T cell
                "CL:0000066",
                "CL:0000988>CL:0000542>CL:0000084",
                None,
            ],
            "cell_type_deepest_score": [0.3, 0.5, 0.2, 0.9, np.nan],
            "sex_chrom": ["female", "male", "ambiguous", "female", None],
            "ancestry_hancestro": ["HANCESTRO:0005", "HANCESTRO:0005", None, None, None],
            "n_umi_exon": [100, 200, 300, 400, 500],
            "n_umi_intron": [10, 20, 30, 40, 50],
            "n_feat_exon": [50, 60, 70, 80, 90],
        }
    )


def test_cell_mask_semantics():
    o = _obs()
    assert S.cell_mask(o, {}).tolist() == [True, True, True, False, True]  # called cells only
    assert S.cell_mask(o, {"cell_type": "CL:0000084"}).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]  # descendants, called
    assert (
        S.cell_mask(o, {"cell_type": "T cell"}).tolist()
        == S.cell_mask(o, {"cell_type": ["CL:0000084"]}).tolist()
    )
    assert S.cell_mask(o, {"cell_type": "CL:0000542"}).tolist() == [True, True, False, False, False]
    assert S.cell_mask(o, {"sex": "PATO:0000384"}).tolist() == [False, True, False, False, False]
    assert S.cell_mask(o, {"ancestry": "hancestro:0005"}).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert S.cell_mask(o, {"min_umi": 220, "max_genes": 70}).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert S.cell_mask(o, {"min_cell_type_score": 0.25}).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert not S.cell_mask(o.drop(columns=["sex_chrom"]), {"sex": "female"}).any()
    with pytest.raises(ValueError):
        S.cell_mask(o, {"cell_type": "not a cell type"})
    with pytest.raises(ValueError):
        S._check_where({"colour": "blue"})


def test_remap_columns_and_log1p():
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    X = sp.random(
        30, 12, density=0.3, format="csr", random_state=1, data_rvs=lambda k: rng.integers(1, 9, k)
    )
    gmap = np.array([3, -1, 0, 1, -1, 2, 4, -1, 5, 6, -1, 7], np.int32)
    Y = S.remap_columns(X, gmap, 8)
    want = np.zeros((30, 8), np.float32)
    D = X.toarray()
    for j, t in enumerate(gmap):
        if t >= 0:
            want[:, t] = D[:, j]
    assert np.array_equal(Y.toarray(), want)
    Z = S.log1p_cp10k(Y.copy())
    tot = want.sum(1, keepdims=True)
    tot[tot == 0] = 1
    assert np.allclose(Z.toarray(), np.log1p(want * 1e4 / tot), rtol=1e-5)


def test_gene_axis_strips_versions():
    class St:
        gse = "GSE1"

        def gene_table(self):
            return np.array(["ENSG1.2", "ENSG3", "ENSG9"]), np.array(["A", "C", "Z"])

    ax = S.GeneAxis(["ENSG1", "ENSG2", "ENSG3.1"])
    assert ax.map_for(St()).tolist() == [0, 2, -1]


def test_cohort_frame_forms(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text("# truncated note\ngse_id\ttitle\nGSE1\tx\nGSE2\ty\n", encoding="utf-8")
    assert S._cohort_frame(p) == [("GSE1", None), ("GSE2", None)]
    assert S._cohort_frame(["gse5", ("GSE6", "GSM7"), "GSM8"]) == [
        ("GSE5", None),
        ("GSE6", "GSM7"),
        (None, "GSM8"),
    ]
    assert S._cohort_frame(pd.DataFrame({"gsm_id": ["GSM1"], "gse": ["GSE1"]})) == [
        ("GSE1", "GSM1")
    ]


# =================================================================================================================== fixture
@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    pytest.importorskip("zstandard")
    pytest.importorskip("torch")
    import _s1_synth as synth

    root = tmp_path_factory.mktemp("stream")
    try:
        s = synth.Synth(str(root)).build(labels=True)
    except IndexError:  # pragma: no cover - packs not shipped
        pytest.skip("vocabulary packs are not available")
    return root, s


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from singlet.s1 import client

    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SINGLET_DATA_BASE", raising=False)
    client.clear_memory_cache()
    yield tmp_path
    client.clear_memory_cache()


def _plan(s):
    return pd.DataFrame(
        [(GSE, g, int(s.truth[g]["called"].sum())) for g in (G1, G3)],
        columns=["gse", "gsm", "n_cells"],
    )


def _expected(s, gsm):
    t = s.truth[gsm]
    X = t["exon_g"] + (t["intron_g"] if t["intron_g"] is not None else 0)
    return X.T.tocsr()  # cells x genes, in the study's gene order


def _drain(ds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(ds)


def test_every_matching_cell_once_with_its_counts(labelled, env):
    root, s = labelled
    ds = S.AtlasCells(
        plan=_plan(s),
        base=str(root),
        batch_size=16,
        mix_samples=2,
        prefetch=2,
        transform=None,
        output="scipy",
    )
    batches = _drain(ds)
    assert all(len(b["gsm"]) == 16 for b in batches[:-1])
    seen = [(g, int(c)) for b in batches for g, c in zip(b["gsm"], b["col"])]
    want = [(g, int(c)) for g in (G1, G3) for c in np.flatnonzero(s.truth[g]["called"])]
    assert sorted(seen) == sorted(want) and len(seen) == len(set(seen))
    ref = {g: _expected(s, g) for g in (G1, G3)}
    assert len(ds.gene_axis) == ref[G1].shape[1]
    for b in batches:
        for i, (g, c) in enumerate(zip(b["gsm"], b["col"])):
            assert np.array_equal(b["X"][i].toarray().ravel(), ref[g][int(c)].toarray().ravel())
    # labels follow the synthetic patterns (only GSM9000001 has them)
    pats = s.LABEL_PATTERNS
    for b in batches:
        for g, c, l2, sex in zip(b["gsm"], b["col"], b["cell_type_l2"], b["sex"]):
            if g == G1:
                path, _ps, want_sex, _anc = pats[int(c) % len(pats)]
                assert l2 == (S.path_level([path], 2)[0]) and sex == (want_sex or None)
            else:
                assert l2 is None
    st = ds.stats()
    assert st["cells"] == len(want) and st["samples"] == 2 and st["samples_skipped"] == 0


def test_reshuffle_per_epoch(labelled, env):
    """Each sample's cell order is fixed by (seed, epoch); how samples interleave into batches depends on read timing."""
    root, s = labelled
    ds = S.AtlasCells(
        plan=_plan(s), base=str(root), batch_size=8, mix_samples=2, prefetch=2, output="scipy"
    )

    def per_sample():
        seen = [(g, int(c)) for b in _drain(ds) for g, c in zip(b["gsm"], b["col"])]
        return {g: [c for gg, c in seen if gg == g] for g in (G1, G3)}

    e0 = per_sample()
    assert per_sample() == e0  # same epoch, same order inside each sample
    ds.set_epoch(1)
    e1 = per_sample()
    assert all(sorted(e1[g]) == sorted(e0[g]) and e1[g] != e0[g] for g in (G1, G3))


def test_cell_filter_split_and_cap(labelled, env):
    root, s = labelled
    t1 = s.truth[G1]
    ds = S.AtlasCells(
        {"cell_type": "CL:0000084"}, plan=_plan(s), base=str(root), batch_size=64, output="scipy"
    )
    cols = sorted(int(c) for b in _drain(ds) for c in b["col"])
    pats = s.LABEL_PATTERNS
    want = [int(c) for c in np.flatnonzero(t1["called"]) if "CL:0000084" in pats[c % len(pats)][0]]
    assert cols == want  # G3 has no labels: no cell of it matches a label filter
    split = S.study_split(GSE)
    other = next(x for x in S.SPLITS if x != split)
    assert len(S.AtlasCells(plan=_plan(s), base=str(root), split=split).samples) == 2
    assert len(S.AtlasCells(plan=_plan(s), base=str(root), split=other).samples) == 0
    capped = S.AtlasCells(
        plan=_plan(s), base=str(root), max_cells_per_sample=5, batch_size=4, output="scipy"
    )
    got = [g for b in _drain(capped) for g in b["gsm"]]
    assert got.count(G1) == 5 and got.count(G3) == 5


def test_worker_shards_are_disjoint(labelled, env, monkeypatch):
    import torch.utils.data as tud

    root, s = labelled
    seen = []
    for wid in range(2):
        monkeypatch.setattr(
            tud, "get_worker_info", lambda wid=wid: type("W", (), {"id": wid, "num_workers": 2})()
        )
        ds = S.AtlasCells(plan=_plan(s), base=str(root), batch_size=16, output="scipy")
        seen.append({g for b in _drain(ds) for g in b["gsm"]})
    assert seen[0].isdisjoint(seen[1]) and seen[0] | seen[1] == {G1, G3}


def test_tensor_outputs(labelled, env):
    import torch

    root, s = labelled
    d = next(iter(S.AtlasCells(plan=_plan(s), base=str(root), batch_size=8)))
    c = next(iter(S.AtlasCells(plan=_plan(s), base=str(root), batch_size=8, output="csr")))
    assert (
        isinstance(d["X"], torch.Tensor)
        and d["X"].dtype == torch.float32
        and d["X"].layout == torch.strided
    )
    assert c["X"].layout == torch.sparse_csr and torch.allclose(c["X"].to_dense(), d["X"])


# =================================================================================================================== live
live = pytest.mark.skipif(
    os.environ.get("SINGLET_LIVE_TESTS") != "1", reason="set SINGLET_LIVE_TESTS=1"
)


@live
def test_live_counts_match_the_cell_index(env):
    """The cells streamed for a filter are exactly as many as the atlas cell index counts for each sample."""
    plan = S.plan_samples({"studies": ["GSE138867"], "cell_type": "CL:0000084"})
    assert len(plan) and plan["n_cells"].sum() > 0
    ds = S.AtlasCells(
        {"cell_type": "CL:0000084"}, plan=plan.head(2), batch_size=1024, output="scipy"
    )
    got = pd.Series([g for b in _drain(ds) for g in b["gsm"]]).value_counts()
    for _, r in plan.head(2).iterrows():
        assert got.get(r["gsm"], 0) == r["n_cells"]


def test_dataloader_worker_processes(labelled, env, monkeypatch):
    """Worker processes each stream their own samples; together every cell once, and the batches carry each worker's I/O."""
    from torch.utils.data import DataLoader

    root, s = labelled
    monkeypatch.setenv(  # spawned workers must import singlet from this checkout
        "PYTHONPATH",
        os.pathsep.join(
            [str(pathlib.Path(S.__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")]
        ),
    )
    ds = S.AtlasCells(plan=_plan(s), base=str(root), batch_size=16, output="scipy")
    _ = ds.gene_axis  # resolved before pickling: the workers get the ids, not the open study
    seen, io = [], {}
    for b in DataLoader(ds, batch_size=None, num_workers=2):
        seen += [(g, int(c)) for g, c in zip(b["gsm"], b["col"])]
        io[b["stream"]["worker"]] = b["stream"]
    want = [(g, int(c)) for g in (G1, G3) for c in np.flatnonzero(s.truth[g]["called"])]
    assert sorted(seen) == sorted(want)
    assert set(io) == {0, 1} and all(v["requests"] > 0 and v["bytes"] > 0 for v in io.values())


def test_transport_receive_buffer(monkeypatch):
    """Data connections get a 4 MiB receive buffer on Windows (the 64 KiB default caps one connection near 2 MB/s); elsewhere
    the system's autotuning is left alone; SINGLET_RCVBUF overrides."""
    from singlet.s1 import transport as T

    monkeypatch.delenv("SINGLET_RCVBUF", raising=False)
    assert T._rcvbuf() == (4 << 20 if os.name == "nt" else 0)
    monkeypatch.setenv("SINGLET_RCVBUF", "0")
    assert T._rcvbuf() == 0
    monkeypatch.setenv("SINGLET_RCVBUF", str(1 << 20))
    assert T._rcvbuf() == 1 << 20
    assert issubclass(T._HTTPSConnection, T.http.client.HTTPSConnection)


def test_a_batch_draws_from_every_open_sample():
    """_batch takes cells from every open sample, in proportion to what each has left (no reader threads)."""
    import scipy.sparse as sp

    ds = S.AtlasCells.__new__(S.AtlasCells)
    ds.transform, ds.output, ds.levels, ds._t0_req = None, "scipy", (1,), None
    ds._stats = {"samples": 0, "samples_skipped": 0}
    ds.base = None

    def loaded(gsm, n):
        obs = {
            "col": np.arange(n),
            "path": np.array(["CL:0000988"] * n, object),
            "score": np.zeros(n, np.float32),
            "sex": np.full(n, None),
            "ancestry": np.full(n, None),
        }
        return S._Loaded("GSE" + gsm[3:], gsm, sp.csr_matrix(np.ones((n, 3), np.float32)), obs)

    open_ = [loaded("GSM1", 400), loaded("GSM2", 400), loaded("GSM3", 200)]
    b = ds._batch(open_, 100, np.random.default_rng(0))
    counts = pd.Series(b["gsm"]).value_counts()
    assert len(b["gsm"]) == 100 and set(counts.index) == {"GSM1", "GSM2", "GSM3"}
    assert counts["GSM3"] < counts["GSM1"] and b["X"].shape == (100, 3)
    assert sum(x.left for x in open_) == 900 and b["cell_type_l1"] == ["CL:0000988"] * 100
