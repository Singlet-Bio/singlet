# SPDX-License-Identifier: MIT
"""singlet.torch — PyTorch integration for Singlet datasets.

Provides DataLoaders and GPU-friendly sparse tensor pipelines over ``.singlet``
files and GEO accessions.

Classes:
    SingletDataset      Map-style dataset over a .singlet file / accession / AnnData
    DataLoader          Convenience wrapper around torch DataLoader; accepts an
                        array of .singlet files (preferred) for concatenated training
    SpzDataset          Legacy dataset (kept for compatibility)
    OnePZDataset        Deprecated alias of SingletDataset

Functions:
    from_anndata        Convert AnnData to a sparse tensor

Example
-------
    from singlet.torch import DataLoader
    loader = DataLoader(["a.singlet", "b.singlet"], batch_size=512, device="cuda")
"""

try:
    import torch as _torch
except ImportError:
    raise ImportError(
        "singlet.torch requires PyTorch. Install with: pip install singlet[torch]"
    ) from None

from singlet.torch._torch import (
    DataLoader,
    OnePZDataset,
    SingletDataset,
    SpzDataset,
    from_anndata,
    to_sparse_coo,
    to_sparse_csr,
)

__all__ = [
    "to_sparse_csr",
    "to_sparse_coo",
    "from_anndata",
    "SpzDataset",
    "SingletDataset",
    "OnePZDataset",
    "DataLoader",
]
