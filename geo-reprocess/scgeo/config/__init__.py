"""Configuration management for sc-geo package.

Contains species references, protocol definitions, paths, QC thresholds,
and other configurable parameters.
"""

__all__ = [
    "get_config",
    "get_default_config",
    "set_config",
    "reset_config",
    "save_config",
    "load_config",
    "Config",
]

from .defaults import Config, get_default_config
from .api import get_config, set_config, reset_config, save_config, load_config
