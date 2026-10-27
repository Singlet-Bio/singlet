# SPDX-License-Identifier: MIT
"""Per-cell labels in AnnData ``obs`` (singlet.s1.labels): the default cell type level and score rule, the deepest call, sex,
ancestry, the cell calls, categorical columns, the hidden unvalidated doublet score, and no collision with GEO sample metadata.

The unit tests need only numpy and pandas. The fixture tests read a synthetic labelled study (tests/python/_s1_synth.py,
``build(labels=True)``) and need ``singlet._pz2``. ``test_live_*`` read a real study from data.singlet.bio and run only with
``SINGLET_LIVE_TESTS=1`` (marker ``live``)."""

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

from singlet.s1 import labels as L  # noqa: E402

T_PATH = "CL:0000988>CL:0000542>CL:0000084>CL:0000624"


def _cols(rows):
    """Stored cells-table columns from rows of (path, per-level confidences)."""
    n = len(rows)
    lev = {f"cell_type_p{k}": np.full(n, np.nan, np.float32) for k in range(1, 7)}
    for i, (_p, ps) in enumerate(rows):
        for k, v in enumerate(ps):
            lev[f"cell_type_p{k + 1}"][i] = v
    paths = [p for p, _ in rows]
    last = [p.split(">")[-1] if p else "" for p in paths]
    return {
        "cell_type_cl": last,
        "cell_type_label": [L.cl_names().get(x, "") for x in last],
        "cell_type_score": np.array([np.prod(ps) if ps else np.nan for _, ps in rows], np.float32),
        "cell_type_path": paths,
        **lev,
    }


# =================================================================================================================== unit
def test_cl_names_ship_with_the_package():
    names = L.cl_names()
    assert names["CL:0000084"] == "T cell" and names["CL:0000988"] == "hematopoietic cell"


def test_default_level_and_score_rule():
    rows = [
        (T_PATH, (0.9, 0.8, 0.7, 0.6)),  # level 2 passes (0.72)
        ("CL:0000988>CL:0000542", (0.9, 0.4)),  # level 2 fails (0.36): level 1
        ("CL:0000066", (0.3,)),  # level 1 fails: no cell_type, the deepest call stays
        ("", ()),  # no call at all
    ]
    c = L.cell_type_columns(_cols(rows), 4)
    names = L.cl_names()
    assert list(c["cell_type_ontology_term_id"]) == ["CL:0000542", "CL:0000988", None, None]
    assert list(c["cell_type"]) == [names["CL:0000542"], "hematopoietic cell", None, None]
    assert list(c["cell_type_level"]) == [2, 1, 0, 0]
    np.testing.assert_allclose(c["cell_type_score"][:2], [0.72, 0.9], rtol=1e-6)
    assert np.isnan(c["cell_type_score"][2:]).all()
    assert list(c["cell_type_deepest_ontology_term_id"]) == [
        "CL:0000624",
        "CL:0000542",
        "CL:0000066",
        None,
    ]
    assert list(c["cell_type_deepest_level"]) == [4, 2, 1, 0]
    assert c["cell_type_deepest"][2] == names["CL:0000066"]


def test_level_and_min_score_options():
    rows = [(T_PATH, (0.9, 0.8, 0.7, 0.6)), ("CL:0000066", (0.3,))]
    deep = L.cell_type_columns(_cols(rows), 2, level=6, min_score=0.0)
    assert list(deep["cell_type_ontology_term_id"]) == ["CL:0000624", "CL:0000066"]
    assert list(deep["cell_type_level"]) == [4, 1]
    one = L.cell_type_columns(_cols(rows), 2, level=1)
    assert list(one["cell_type_ontology_term_id"]) == ["CL:0000988", None]
    strict = L.cell_type_columns(_cols(rows), 2, min_score=0.8)
    assert list(strict["cell_type_ontology_term_id"]) == ["CL:0000988", None]


def test_obs_columns_order_categories_and_hidden_doublet_score():
    rows = [(T_PATH, (0.9, 0.8, 0.7, 0.6)), ("", ())]
    cols = {
        "barcode": ["AAA", "CCC"],
        "called": np.array([True, False]),
        **_cols(rows),
        "sex_chrom": ["female", ""],
        "sex_score": np.array([0.99, np.nan], np.float32),
        "ancestry_hancestro": ["HANCESTRO:0005", ""],
        "ancestry_prob": np.array([0.97, np.nan], np.float32),
        "called_ed": np.array([True, True]),
        "qc_doublet_score": np.array([0.1, 0.9], np.float32),
        "don_donor_id": np.array([0, 1], np.uint8),
        "cc_pos": np.array([0, 1], np.uint32),
    }
    cats = {"don_donor_id": ["donor0", "unassigned"]}
    d = L.obs_columns(cols, 2, cats=cats)
    assert "barcode" not in d and "cc_pos" not in d and "qc_doublet_score" not in d
    assert "cell_type_cl" not in d and "cell_type_label" not in d  # renamed to cell_type_deepest*
    lead = [c for c in L.LEAD if c in d]
    assert list(d)[: len(lead)] == lead
    assert list(d["don_donor_id"]) == ["donor0", "unassigned"]
    assert list(d["sex_chrom"]) == ["female", None]
    assert "qc_doublet_score" in L.obs_columns(cols, 2, include_unvalidated=True)
    only = L.obs_columns(cols, 2, columns=["cell_type", "cell_type_cl", "qc_doublet_score"])
    assert set(only) == {"cell_type", "cell_type_deepest_ontology_term_id", "qc_doublet_score"}


def test_to_frame_fills_columns_a_sample_lacks():
    import pandas as pd

    a = {"gsm_id": np.array(["G1"], object), "cell_type": np.array(["T cell"], object)}
    a.update({"called_ed": np.array([True]), "sex_score": np.array([0.9], np.float32)})
    b = {"gsm_id": np.array(["G2", "G2"], object)}
    df = L.to_frame([a, b], index=pd.Index(["x", "y", "z"]))
    assert isinstance(df["cell_type"].dtype, pd.CategoricalDtype)
    assert df["cell_type"].isna().tolist() == [False, True, True]
    assert str(df["called_ed"].dtype) == "boolean" and df["called_ed"].isna().tolist() == [
        False,
        True,
        True,
    ]
    assert np.isnan(df["sex_score"].to_numpy()[1:]).all()


def test_categories_from_manifest_recipes():
    members = {
        "donor/donor_assignments.tsv": {
            "kind": 4,
            "params": {
                "c": [
                    {
                        "n": "donor_id",
                        "s": "c",
                        "c": "don_donor_id",
                        "f": {"k": "cat", "labels": ["donor0", "unassigned"]},
                    },
                    {"n": "prob_doublet", "s": "k", "v": "0"},
                ]
            },
        },
        "exon_counts.1pz": {"kind": 1},
    }
    assert L.categories(members) == {"don_donor_id": ["donor0", "unassigned"]}


# =================================================================================================================== fixture
@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    pytest.importorskip("zstandard")
    import _s1_synth as synth

    root = tmp_path_factory.mktemp("labelled")
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


def _truth_rows(s, gsm, cells):
    """Pattern index of each cell (cells = matrix columns)."""
    return np.asarray(cells) % len(s.LABEL_PATTERNS)


def test_to_anndata_obs_has_the_labels(labelled, env):
    from singlet.s1 import client

    root, s = labelled
    G1, G3 = "GSM9000001", "GSM9000003"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad = client.open_study(s.manifest["gse"], base=str(root)).to_anndata([G1, G3])
    obs = ad.obs
    for c in (
        "cell_type",
        "cell_type_ontology_term_id",
        "cell_type_level",
        "cell_type_score",
        "cell_type_deepest",
        "cell_type_path",
        "cell_type_p1",
        "sex_chrom",
        "sex_score",
        "ancestry_hancestro",
        "ancestry_prob",
        "called",
        "called_ed",
        "called_ed_fdr",
        "don_donor_id",
        "cyc_phase",
        "mts_pct_heteroplasmic",
        "organism",
        "sample_characteristics",
    ):
        assert c in obs.columns, c
    assert "qc_doublet_score" not in obs.columns and "barcode" not in obs.columns
    assert list(obs.columns[:2]) == ["gsm_id", "cell_type"]
    one = obs[obs["gsm_id"] == G1]
    # the cells are in sorted-barcode order: map each back to its matrix column
    t = s.truth[G1]
    col_of = {b: i for i, b in enumerate(t["barcodes"])}
    cols = np.array([col_of[n.split("_", 1)[1]] for n in one.index])
    pat = _truth_rows(s, G1, cols)
    want_id = np.array(["CL:0000542", "CL:0000988", None, None], dtype=object)[pat]
    got_id = (
        one["cell_type_ontology_term_id"]
        .astype(object)
        .where(one["cell_type_ontology_term_id"].notna(), None)
    )
    assert list(got_id) == list(want_id)
    assert list(one["sex_chrom"].astype(object).fillna("")) == [
        ["female", "male", "ambiguous", ""][p] for p in pat
    ]
    assert one["called"].all() and one["called_ed"].all()
    assert set(one["don_donor_id"].astype(str)) <= {"donor0", "unassigned"}
    assert set(one["cyc_phase"].astype(str)) <= {"G1", "G2M"}
    # the sample without labels: missing, not an error
    three = obs[obs["gsm_id"] == G3]
    assert three["cell_type"].isna().all() and three["sex_chrom"].isna().all()
    assert three["called_ed"].isna().all()
    # GEO metadata is untouched and separate
    assert (obs["organism"] == "Homo sapiens").all()


def test_to_anndata_options_and_writable_h5ad(labelled, env, tmp_path):
    from singlet.s1 import client

    root, s = labelled
    st = client.open_study(s.manifest["gse"], base=str(root))
    ad = st.to_anndata(
        ["GSM9000001"], cell_type_level=6, cell_type_min_score=0.0, include_unvalidated=True
    )
    assert "qc_doublet_score" in ad.obs.columns
    assert ad.obs["cell_type_level"].max() == 4
    p = tmp_path / "x.h5ad"
    ad.write_h5ad(p)
    import anndata

    back = anndata.read_h5ad(p)
    assert list(back.obs.columns) == list(ad.obs.columns)
    assert [k for k in back.layers.keys() if k is not None] == ["spliced", "unspliced"]


def test_cells_where_on_labels(labelled, env):
    from singlet.s1 import client

    root, s = labelled
    st = client.open_study(s.manifest["gse"], base=str(root))
    t = s.truth["GSM9000001"]
    n = len(t["barcodes"])
    sub = st.cells("sex_chrom == 'female'", samples=["GSM9000001"])
    assert list(sub.obs["column"]) == [i for i in range(n) if i % 4 == 0]
    assert (sub.obs["cell_type_ontology_term_id"] == "CL:0000542").all()
    lin = st.cells({"cell_type_ontology_term_id": "CL:0000988"}, samples=["GSM9000001"])
    assert list(lin.obs["column"]) == [i for i in range(n) if i % 4 == 1]
    # the hidden column can still filter, and comes back only when asked for
    hi = st.cells("qc_doublet_score > 0.5", samples=["GSM9000001"])
    assert hi.n_obs > 0 and "qc_doublet_score" not in hi.obs.columns
    shown = st.cells(
        "qc_doublet_score > 0.5", samples=["GSM9000001"], columns=["qc_doublet_score", "cell_type"]
    )
    assert set(shown.obs.columns) == {"gsm_id", "column", "qc_doublet_score", "cell_type"}
    both = st.cells("called", samples=["GSM9000001", "GSM9000003"])
    assert both.n_obs == 40 + 25 and "cell_type" in both.obs.columns


def test_load_s2_has_the_labels(labelled, env):
    from singlet.s1 import client

    root, s = labelled
    key = s.truth["GSM9000001"]["key"]
    ad = client.load(str(root.joinpath(*key.split("/"))))
    assert {"cell_type", "sex_chrom", "called_ed"} <= set(ad.obs.columns)
    assert "qc_doublet_score" not in ad.obs.columns


# =================================================================================================================== live
LIVE = pytest.mark.skipif(
    os.environ.get("SINGLET_LIVE_TESTS") != "1",
    reason="set SINGLET_LIVE_TESTS=1 to read data.singlet.bio",
)


@pytest.mark.live
@LIVE
def test_live_labels_from_data_singlet_bio(tmp_path, monkeypatch):
    """GSE286650 (one human sample, heart left ventricle, with ancestry): the labels reach obs."""
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    from singlet.s1 import client

    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    client.clear_memory_cache()
    ad = client.open_study("GSE286650", force=True).to_anndata(layers=False)
    obs = ad.obs
    assert ad.n_obs > 1000
    for c in (
        "cell_type",
        "cell_type_ontology_term_id",
        "sex_chrom",
        "sex_score",
        "ancestry_hancestro",
        "called",
        "called_ed",
    ):
        assert c in obs.columns, c
    assert "qc_doublet_score" not in obs.columns and None not in ad.layers.keys()
    assert obs["cell_type"].notna().mean() > 0.5
    assert obs["cell_type_ontology_term_id"].dropna().str.match(r"^CL:\d{7}$").all()
    assert set(obs["sex_chrom"].dropna().astype(str)) <= {"female", "male", "ambiguous"}
    assert (obs["sex_chrom"] == "male").mean() > 0.5  # GEO: male donor
    assert obs["ancestry_hancestro"].dropna().str.match(r"^HANCESTRO:\d{4}$").all()
