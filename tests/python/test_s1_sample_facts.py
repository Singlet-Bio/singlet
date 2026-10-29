# SPDX-License-Identifier: MIT
"""singlet.s1.sample_facts: the study API's per-sample facts as obs columns (tissue_type, cell_line, tumor_sample_type, disease,
donor_status, sample_kind). The flattening and the API reply parsing are tested directly; the obs wiring on the synthetic local
study with the facts injected. No network."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from singlet.s1 import sample_facts as SF  # noqa: E402

K562 = {
    "sample_kind": "cell_line",
    "tissue_type": "cell line",
    "cell_line": [{"id": "CVCL_0004", "name": "K-562"}],
}
ADJ = {
    "sample_kind": "primary",
    "tissue_type": "tissue",
    "tumor_sample_type": "adjacent normal",
    "donor_status": "diseased",
    "disease": [{"id": "MONDO:0005061", "label": "lung adenocarcinoma"}],
}


def test_flatten_scalars_lists_and_missing():
    f = SF.flatten(K562)
    assert f["tissue_type"] == "cell line" and f["sample_kind"] == "cell_line"
    assert f["cell_line"] == "K-562" and f["cell_line_cellosaurus_id"] == "CVCL_0004"
    assert f["disease"] is None and f["tumor_sample_type"] is None and f["donor_status"] is None
    g = SF.flatten(ADJ)
    assert (
        g["disease"] == "lung adenocarcinoma" and g["disease_ontology_term_id"] == "MONDO:0005061"
    )
    assert g["tumor_sample_type"] == "adjacent normal"
    assert set(SF.flatten(None).values()) == {None} and list(SF.flatten({})) == list(
        SF.FACT_OBS_COLUMNS
    )


def test_flatten_joins_several_lines_in_order_and_healthy_is_normal():
    f = SF.flatten(
        {"cell_line": [{"id": "CVCL_9771", "name": "WA01"}, {"id": "CVCL_B207", "name": "HUES 8"}]}
    )
    assert (
        f["cell_line"] == "WA01; HUES 8" and f["cell_line_cellosaurus_id"] == "CVCL_9771; CVCL_B207"
    )
    h = SF.flatten(
        {"donor_status": "healthy", "disease": [{"id": "PATO:0000461", "label": "normal"}]}
    )
    assert (
        h["disease"] == "normal"
        and h["disease_ontology_term_id"] == "PATO:0000461"
        and h["donor_status"] == "healthy"
    )
    nolabel = SF.flatten({"disease": [{"id": "MONDO:0100096", "label": None}]})
    assert nolabel["disease"] == "MONDO:0100096"  # an id without a label stands in for the name


def test_parse_study_payload_keeps_samples_with_facts():
    payload = {
        "series": {"id": "GSE1"},
        "samples": [
            {"gsm_id": "gsm1", "facts": K562},
            {"gsm_id": "GSM2", "facts": {}},
            {"gsm_id": "GSM3"},
            "junk",
        ],
    }
    assert SF.parse_study_payload(payload) == {"GSM1": K562}
    assert SF.parse_study_payload(None) == {} and SF.parse_study_payload({"samples": None}) == {}


def test_no_fetch_offline_or_for_local_and_loopback_studies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        SF, "fetch_sample_facts", lambda gse, **kw: calls.append(gse) or {"GSM1": K562}
    )

    class St:
        gse = "GSE1"

        def __init__(self, base):
            self.transport = type("T", (), {"base": base})()

    assert SF.study_facts(St("/data/singlet")) == {}
    assert SF.study_facts(St("http://127.0.0.1:8000")) == {}
    assert SF.study_facts(St("http://localhost:9")) == {}
    assert SF.study_facts(St("https://data.singlet.bio")) == {"GSM1": K562} and calls == ["GSE1"]


def test_fetch_is_empty_when_offline_and_never_raises(monkeypatch):
    SF.clear_memo()
    monkeypatch.setenv("SINGLET_OFFLINE", "1")
    assert SF.fetch_sample_facts("GSE1") == {}
    monkeypatch.delenv("SINGLET_OFFLINE")
    monkeypatch.setenv("SINGLET_API_BASE", "http://127.0.0.1:9")  # nothing listens there
    assert SF.fetch_sample_facts("GSE1", timeout=2) == {}
    SF.clear_memo()


def test_facts_obs_columns():
    d = SF.facts_obs(ADJ, 3)
    assert list(d) == list(SF.FACT_OBS_COLUMNS)
    assert (
        all(len(v) == 3 for v in d.values())
        and d["cell_line"][0] is None
        and d["disease"][2] == "lung adenocarcinoma"
    )


# ---------------------------------------------------------------------------------------------------------- wired into obs
@pytest.fixture(scope="module")
def study_dir(tmp_path_factory):
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    pytest.importorskip("zstandard")
    import _s1_synth as synth

    root = tmp_path_factory.mktemp("factsstudy")
    try:
        s = synth.Synth(str(root)).build()
    except IndexError:  # pragma: no cover - packs not shipped
        pytest.skip("vocabulary packs are not available")
    return root, s


def test_cell_lines_carry_their_line_as_cell_type():
    n = 3
    base = {
        "cell_type": np.array(["transit amplifying cell", None, "T cell"], dtype=object),
        "cell_type_ontology_term_id": np.array(["CL:0009010", None, "CL:0000084"], dtype=object),
        "cell_type_level": np.array([2, 0, 2], dtype=np.int8),
        "cell_type_score": np.array([0.9, np.nan, 0.8], dtype=np.float32),
        "cell_type_deepest": np.array(["transit amplifying cell", None, "T cell"], dtype=object),
        "cell_type_path": np.array(["CL:a>CL:b", None, "CL:c"], dtype=object),
    }
    d = {k: v.copy() for k, v in base.items()}
    SF.relabel_cell_lines(d, K562, n)
    assert list(d["cell_type"]) == ["K-562"] * n and list(d["cell_type_ontology_term_id"]) == ["CVCL_0004"] * n
    assert list(d["cell_type_deepest"]) == ["K-562"] * n and "cell_type_deepest_ontology_term_id" not in d
    assert list(d["cell_type_level"]) == [0] * n and np.isnan(d["cell_type_score"]).all()
    assert list(d["cell_type_path"]) == [None] * n
    unnamed = {k: v.copy() for k, v in base.items()}
    SF.relabel_cell_lines(unnamed, {"tissue_type": "cell line"}, n)
    assert set(unnamed["cell_type"]) == {"cultured cell"} and set(unnamed["cell_type_ontology_term_id"]) == {"CL:0000010"}
    tissue = {k: v.copy() for k, v in base.items()}
    SF.relabel_cell_lines(tissue, ADJ, n)
    assert list(tissue["cell_type"]) == list(base["cell_type"])
    SF.relabel_cell_lines(tissue, None, n)
    assert list(tissue["cell_type"]) == list(base["cell_type"])


def test_load_and_cells_carry_the_facts(study_dir, tmp_path, monkeypatch):
    import _s1_synth as synth
    import pandas as pd
    from singlet.s1 import client

    root, _ = study_dir
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    client.clear_memory_cache()
    g1, g2 = "GSM9000001", "GSM9000003"
    monkeypatch.setattr(SF, "study_facts", lambda study: {g1: K562, g2: ADJ})
    st = client.open_study(synth.GSE, base=str(root))
    ad = st.to_anndata([g1, g2])
    for c in SF.FACT_OBS_COLUMNS:
        assert c in ad.obs.columns and isinstance(ad.obs[c].dtype, pd.CategoricalDtype), c
    a = ad.obs[ad.obs["gsm_id"] == g1]
    b = ad.obs[ad.obs["gsm_id"] == g2]
    assert set(a["cell_line"]) == {"K-562"} and set(a["cell_line_cellosaurus_id"]) == {"CVCL_0004"}
    assert (
        a["disease"].isna().all() and a["tumor_sample_type"].isna().all()
    )  # not stated: missing, not ""
    assert set(b["tumor_sample_type"]) == {"adjacent normal"} and set(
        b["disease_ontology_term_id"]
    ) == {"MONDO:0005061"}
    assert "cell_type" not in SF.FACT_OBS_COLUMNS  # the per-cell label keeps its name
    if "cell_type" in ad.obs.columns:  # a cell line carries its line, a tissue sample the labeller's call
        assert set(a["cell_type"]) == {"K-562"} and set(a["cell_type_ontology_term_id"]) == {"CVCL_0004"}

    sub = st.cells("called", samples=[g1])
    assert set(sub.obs["tissue_type"]) == {"cell line"}

    off = st.to_anndata([g1], sample_facts=False)
    assert not set(SF.FACT_OBS_COLUMNS) & set(off.obs.columns)
    client.clear_memory_cache()


def test_no_facts_means_no_columns(study_dir, tmp_path, monkeypatch):
    import _s1_synth as synth
    from singlet.s1 import client

    root, _ = study_dir
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    client.clear_memory_cache()
    st = client.open_study(synth.GSE, base=str(root))  # a local directory: the API is never called
    ad = st.to_anndata(["GSM9000001"])
    assert not set(SF.FACT_OBS_COLUMNS) & set(ad.obs.columns)
    assert np.all(ad.obs["gsm_id"] == "GSM9000001")
    client.clear_memory_cache()
