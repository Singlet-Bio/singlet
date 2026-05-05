"""Configuration API for sc-geo package.

Provides functions to get, load, and save configuration.
"""
import json
from pathlib import Path
from typing import Optional

from .defaults import Config, get_default_config


# Global config instance (lazy-loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get current configuration.
    
    Returns the global config instance, creating it with defaults
    if it doesn't exist yet.
    
    Returns:
        Current Config object
        
    Example:
        >>> from scgeo.config import get_config
        >>> config = get_config()
        >>> config.qc.min_mapping_rate
        0.1
    """
    global _config
    if _config is None:
        _config = get_default_config()
    return _config


def set_config(config: Config):
    """Set global configuration.
    
    Args:
        config: Config object to use globally
    """
    global _config
    _config = config


def reset_config():
    """Reset configuration to defaults."""
    global _config
    _config = get_default_config()


def save_config(path: Path, config: Optional[Config] = None):
    """Save configuration to JSON file.
    
    Args:
        path: Output path for JSON file
        config: Config to save (uses global if None)
        
    Example:
        >>> from scgeo.config import get_config, save_config
        >>> config = get_config()
        >>> config.qc.min_mapping_rate = 0.2
        >>> save_config("my_config.json")
    """
    if config is None:
        config = get_config()
    
    # Convert config to JSON-serializable dict
    # We'll implement a simple version for now
    config_dict = {
        "download": {
            "segments": config.download.segments,
            "retries": config.download.retries,
            "retry_delay": config.download.retry_delay,
            "timeout": config.download.timeout,
            "verify_md5": config.download.verify_md5,
        },
        "qc": {
            "min_mapping_rate": config.qc.min_mapping_rate,
            "min_cells": config.qc.min_cells,
            "min_genes_per_cell": config.qc.min_genes_per_cell,
            "min_unspliced_frac": config.qc.min_unspliced_frac,
        },
        "resources": {
            "simpleaf_threads": config.resources.simpleaf_threads,
        },
        # Add other sections as needed
    }
    
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: Path) -> Config:
    """Load configuration from JSON file.
    
    Args:
        path: Path to JSON config file
        
    Returns:
        Config object with loaded settings
        
    Example:
        >>> from scgeo.config import load_config
        >>> config = load_config("my_config.json")
    """
    with open(path) as f:
        config_dict = json.load(f)
    
    # Start with defaults
    config = get_default_config()
    
    # Override with loaded values
    if "download" in config_dict:
        for key, value in config_dict["download"].items():
            setattr(config.download, key, value)
    
    if "qc" in config_dict:
        for key, value in config_dict["qc"].items():
            setattr(config.qc, key, value)
    
    if "resources" in config_dict:
        for key, value in config_dict["resources"].items():
            setattr(config.resources, key, value)
    
    return config
