"""Tests for indices module."""
import pytest
from pathlib import Path
from scgeo.indices import get_index_path, check_index_exists, list_available_indices


def test_get_index_path(test_config):
    """Test getting index path for an organism."""
    # With test config pointing to tmp dir, index won't exist
    path = get_index_path("Homo sapiens", config=test_config)
    # Returns None when index directory doesn't exist
    assert path is None or "homo_sapiens" in str(path).lower()


def test_check_index_missing(test_config):
    """Test index existence check with missing index."""
    exists = check_index_exists("Homo sapiens", config=test_config)
    assert exists is False


def test_list_available_indices(test_config):
    """Test listing available indices."""
    indices = list_available_indices(config=test_config)
    assert isinstance(indices, dict)
