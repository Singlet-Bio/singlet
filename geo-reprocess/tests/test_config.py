"""Tests for configuration module."""
import pytest
from pathlib import Path
from scgeo.config import Config, get_config, set_config, load_config, save_config


def test_get_default_config():
    """Test getting default configuration."""
    config = get_config()
    
    assert config is not None
    assert config.paths.project_base.exists()
    assert len(config.species_ref) >= 39
    # species_ref is keyed by taxon_id
    assert 9606 in config.species_ref  # Homo sapiens


def test_config_save_load(test_config, tmp_path):
    """Test saving and loading configuration."""
    config_file = tmp_path / "config.json"
    
    # Save
    save_config(config_file, test_config)
    assert config_file.exists()
    
    # Load
    loaded_config = load_config(config_file)


def test_set_config(test_config):
    """Test setting global configuration."""
    set_config(test_config)
    
    # Get should return same config
    retrieved = get_config()
    assert retrieved.paths.project_base == test_config.paths.project_base


def test_config_species_refs():
    """Test species references."""
    config = get_config()
    
    # Check some common species (keyed by taxon_id)
    assert 9606 in config.species_ref   # Homo sapiens
    assert 10090 in config.species_ref  # Mus musculus
    assert 7955 in config.species_ref   # Danio rerio
    
    # Check species data structure
    human = config.species_ref[9606]
    assert "assembly" in human
    assert "name" in human
    assert human["name"] == "Homo sapiens"


def test_config_qc_thresholds():
    """Test QC threshold configuration."""
    config = get_config()
    
    assert config.qc.min_cells > 0
    assert config.qc.min_mapping_rate > 0
    assert 0 < config.qc.min_mapping_rate < 1


def test_config_paths_exist():
    """Test that configured paths exist."""
    config = get_config()
    
    assert config.paths.project_base.exists()
    # Note: Other paths may not exist yet (created on demand)
