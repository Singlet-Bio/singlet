"""Tests for singlet catalog functions using bundled parquet data."""

import pandas as pd
import pytest


def test_catalog_returns_dataframe():
    import singlet

    cat = singlet.catalog()
    assert isinstance(cat, pd.DataFrame)
    assert len(cat) > 100
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
    cells_col = "total_cells" if "total_cells" in big.columns else "n_cells"
    assert all(big[cells_col] >= 10000)


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
    cells_key = "total_cells" if "total_cells" in info else "n_cells"
    assert info[cells_key] > 0


def test_info_gsm_lookup():
    """info() should also accept GSM accessions."""
    import singlet

    df = singlet.samples(status="SUCCESS")
    gsm = df.iloc[0]["gsm_id"]
    info = singlet.info(gsm)
    assert isinstance(info, dict)
    assert info["gsm_id"] == gsm
    assert info["status"] == "SUCCESS"


def test_info_missing_raises():
    import singlet

    with pytest.raises(KeyError):
        singlet.info("GSE000000")


def test_samples_quality_alias():
    """samples(quality='gold') should work as alias for quality_tier='gold'."""
    import singlet

    gold1 = singlet.samples(quality="gold")
    gold2 = singlet.samples(quality_tier="gold")
    assert len(gold1) == len(gold2)
    assert len(gold1) > 0


def test_failure_categories():
    import singlet

    fc = singlet.failure_categories()
    assert len(fc) > 0
    assert "category" in fc.columns
    assert "count" in fc.columns
    assert "pct" in fc.columns
    assert fc["count"].sum() > 0
    assert abs(fc["pct"].sum() - 100.0) < 1.0


def test_tissues():
    import singlet

    t = singlet.tissues()
    assert len(t) > 10
    assert "tissue" in t.columns
    assert "count" in t.columns
    assert t["count"].iloc[0] >= t["count"].iloc[-1]  # sorted desc


def test_cell_types():
    import singlet

    ct = singlet.cell_types()
    assert isinstance(ct, pd.DataFrame)
    assert len(ct) > 10
    assert "cell_type" in ct.columns
    assert "count" in ct.columns
    # PBMC should be top
    assert ct.iloc[0]["cell_type"] == "PBMC"
    # Should cover a meaningful fraction of SUCCESS samples
    assert ct["count"].sum() > 300
    # Should be sorted descending
    assert ct["count"].iloc[0] >= ct["count"].iloc[-1]


def test_samples_filter_cell_type():
    import singlet

    pbmc = singlet.samples(cell_type="PBMC")
    assert len(pbmc) > 50
    assert all(pbmc["cell_type"].str.contains("PBMC", na=False))


def test_quality_tiers():
    import singlet

    qt = singlet.quality_tiers()
    assert isinstance(qt, pd.DataFrame)
    assert len(qt) == 3
    assert set(qt["tier"]) == {"gold", "silver", "bronze"}
    assert "count" in qt.columns
    assert "pct" in qt.columns
    assert qt["count"].sum() > 1000


def test_protocols():
    import singlet

    p = singlet.protocols()
    assert isinstance(p, pd.DataFrame)
    assert len(p) > 10
    assert "protocol" in p.columns
    assert "count" in p.columns
    # 10xv3 should be the top protocol
    assert p.iloc[0]["protocol"] == "10xv3"
    assert p.iloc[0]["count"] > 500


def test_samples_filter_tissue():
    """samples(tissue=...) filters by tissue column."""
    import singlet

    lung = singlet.samples(tissue="lung")
    assert len(lung) > 0
    assert all(lung["tissue"].str.contains("lung", case=False, na=False))


def test_samples_filter_protocol():
    """samples(protocol=...) filters by protocol column."""
    import singlet

    v3 = singlet.samples(protocol="10xv3")
    assert len(v3) > 100
    assert all(v3["protocol"].str.contains("10xv3", case=False, na=False))


def test_samples_filter_min_cells():
    """samples(min_cells=...) filters by cell count."""
    import singlet

    big = singlet.samples(min_cells=5000)
    assert len(big) > 0
    cells_col = "cells_called" if "cells_called" in big.columns else "n_cells"
    assert all(big[cells_col] >= 5000)


def test_samples_quality_silver():
    """samples(quality_tier='silver') returns mid-tier samples."""
    import singlet

    silver = singlet.samples(quality_tier="silver")
    assert len(silver) > 0
    assert all(silver["status"] == "SUCCESS")


def test_samples_quality_bronze():
    """samples(quality_tier='bronze') returns low-tier samples."""
    import singlet

    bronze = singlet.samples(quality_tier="bronze")
    assert len(bronze) > 0
    assert all(bronze["status"] == "SUCCESS")


def test_datasets_filter_protocol():
    """datasets(protocol=...) filters catalog by protocol."""
    import singlet

    v2 = singlet.datasets(protocol="10xv2")
    assert len(v2) > 0


def test_datasets_filter_has_kraken2():
    """datasets(has_kraken2=True) filters catalog."""
    import singlet

    k2 = singlet.datasets(has_kraken2=True)
    if "has_kraken2" in k2.columns:
        assert all(k2["has_kraken2"])


def test_summary_format():
    """summary() returns a formatted string with key stats."""
    import singlet

    s = singlet.summary()
    assert "singlet atlas" in s
    assert "samples" in s
    assert "series" in s
    assert "species" in s
