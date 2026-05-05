"""
Indices module for building reference indices.

Provides functions for downloading reference genomes/annotations and
building splici reference indices for transcript quantification.
"""

from scgeo.indices.builder import (
    build_splici_index,
    check_index_exists,
    list_available_indices,
)

from scgeo.indices.downloader import (
    download_references,
    check_references_exist,
)

from scgeo.indices.api import (
    build_index,
    build_indices,
    get_index_path,
)

__all__ = [
    # Builder
    "build_splici_index",
    "check_index_exists",
    "list_available_indices",
    # Downloader
    "download_references",
    "check_references_exist",
    # High-level API
    "build_index",
    "build_indices",
    "get_index_path",
]
