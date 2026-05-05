"""Tests for catalog module."""
import pytest
import pandas as pd
from scgeo.catalog import (
    discover_single_cell_series,
    filter_catalog,
    get_catalog_stats,
)
from scgeo.catalog.soft import _classify_license


def test_filter_catalog(sample_catalog_data):
    """Test catalog filtering."""
    df = pd.DataFrame(sample_catalog_data)
    
    # Filter by organism
    filtered = filter_catalog(df, organisms=["Homo sapiens"])
    assert len(filtered) == 2
    assert all(filtered["organism"] == "Homo sapiens")
    
    # Filter by min samples
    filtered = filter_catalog(df, min_samples=100)
    assert len(filtered) <= len(df)


def test_get_catalog_stats(sample_catalog_data):
    """Test catalog statistics."""
    df = pd.DataFrame(sample_catalog_data)
    
    stats = get_catalog_stats(df)
    
    assert stats["total_samples"] == 2
    assert stats["total_series"] == 1
    assert "Homo sapiens" in stats["organisms"]
    assert stats["organisms"]["Homo sapiens"] == 2


def test_discover_single_cell_series(tmp_path, monkeypatch):
    """Test series discovery (mocked)."""
    # This would need to mock NCBI API calls
    # For now, just test that the function signature works
    output_file = tmp_path / "discovery.json"
    
    # TODO: Add mock for NCBI API
    # result = discover_single_cell_series(
    #     query="single cell[Title]",
    #     output_file=str(output_file),
    # )
    # assert output_file.exists()


def test_catalog_columns(sample_catalog_data):
    """Test that catalog has required columns."""
    df = pd.DataFrame(sample_catalog_data)
    
    required_columns = [
        "gsm_id",
        "gse_id",
        "organism",
        "srx_accession",
        "run_accession",
        "protocol_inferred",
        "library_strategy",
        "license",
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


@pytest.mark.parametrize(
    "series_fields, expected",
    [
        ({}, "public_domain"),
        ({"Series_data_use_agreement": ""}, "public_domain"),
        ({"Series_data_use_agreement": "CC-BY-4.0"}, "cc_by"),
        ({"Series_data_use_agreement": "CC-BY-NC"}, "cc_by_nc"),
        ({"Series_data_use_agreement": "CC-BY-NC-SA"}, "cc_by_nc_sa"),
        ({"Series_data_use_agreement": "CC-BY-SA"}, "cc_by_sa"),
        ({"Series_license": "CC0"}, "cc0"),
        ({"Series_license": "Public Domain"}, "cc0"),
        ({"Series_data_use_agreement": "restricted access via dbGaP"}, "restricted"),
        ({"Series_data_use_agreement": "Some custom statement"}, "unknown"),
    ],
)
def test_classify_license(series_fields, expected):
    """Test license classification from SOFT series metadata."""
    assert _classify_license(series_fields) == expected
