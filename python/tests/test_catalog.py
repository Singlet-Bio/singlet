"""Tests for singlet catalog functions using bundled parquet data."""
import pytest
import pandas as pd


def test_catalog_returns_dataframe():
    import singlet
    cat = singlet.catalog()
    assert isinstance(cat, pd.DataFrame)
    assert len(cat) > 1000
    assert "gse_id" in cat.columns
    assert "organism" in cat.columns
    assert "n_cells" in cat.columns


def test_catalog_search_filters():
    import singlet
    human = singlet.catalog("Homo sapiens")
    assert len(human) > 0
    assert all(human["organism"].str.contains("Homo sapiens", case=False, na=False))


def test_sample_index_returns_dataframe():
    import singlet
    idx = singlet.sample_index()
    assert isinstance(idx, pd.DataFrame)
    assert len(idx) > 2000
    assert "gsm_id" in idx.columns
    assert "status" in idx.columns


def test_sample_index_filter_by_gse():
    import singlet
    idx = singlet.sample_index(gse_id="GSE174399")
    assert len(idx) > 0
    assert all(idx["gse_id"] == "GSE174399")


def test_species_returns_list():
    import singlet
    sp = singlet.species()
    assert isinstance(sp, list)
    assert len(sp) >= 5
    assert "Homo sapiens" in sp
    assert "Mus musculus" in sp


def test_summary_returns_string():
    import singlet
    s = singlet.summary()
    assert isinstance(s, str)
    assert "samples" in s
    assert "species" in s
    assert "cells" in s


def test_datasets_filter_organism():
    import singlet
    human = singlet.datasets(organism="Homo sapiens")
    assert len(human) > 50
    assert all(human["organism"].str.contains("Homo sapiens", case=False, na=False))


def test_datasets_filter_min_cells():
    import singlet
    big = singlet.datasets(min_cells=10000)
    assert len(big) > 0
    assert all(big["n_cells"] >= 10000)


def test_samples_filter_status():
    import singlet
    success = singlet.samples(status="SUCCESS")
    assert len(success) > 500
    assert all(success["status"] == "SUCCESS")


def test_samples_filter_organism():
    import singlet
    mouse = singlet.samples(organism="Mus musculus")
    assert len(mouse) > 100
    assert all(mouse["organism"].str.contains("Mus musculus", case=False, na=False))


def test_samples_text_search():
    import singlet
    results = singlet.samples(search="lung")
    assert len(results) > 0


def test_samples_quality_tier_gold():
    import singlet
    gold = singlet.samples(quality_tier="gold")
    assert len(gold) > 0
    assert all(gold["status"] == "SUCCESS")
    assert all(gold["mapping_rate"] >= 0.7)


def test_top_series():
    import singlet
    top = singlet.top_series(n=5)
    assert len(top) == 5
    assert "total_cells" in top.columns
    assert top["total_cells"].iloc[0] >= top["total_cells"].iloc[-1]


def test_top_series_filter_organism():
    import singlet
    human_top = singlet.top_series(organism="Homo sapiens", n=3)
    assert len(human_top) <= 3
    assert all(human_top["organism"].str.contains("Homo sapiens", case=False, na=False))


def test_info_existing_series():
    import singlet
    info = singlet.info("GSE174399")
    assert isinstance(info, dict)
    assert info["gse_id"] == "GSE174399"
    assert info["n_cells"] > 0


def test_info_missing_raises():
    import singlet
    with pytest.raises(KeyError):
        singlet.info("GSE000000")
