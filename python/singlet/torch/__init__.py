"""singlet.torch — PyTorch integration for .1pz sparse matrices.

Provides zero-copy DataLoaders and GPU-friendly sparse tensor pipelines.

Classes:
    OnePZDataset        Map-style dataset with chunk-level access
    SpzDataset          Legacy .spz dataset
    DataLoader          Convenience wrapper around torch DataLoader

Functions:
    to_sparse_csr       Load .spz as CSR tensor
    to_sparse_coo       Load .spz as COO tensor
    from_anndata        Convert AnnData to sparse tensor
"""
try:
    import torch as _torch
except ImportError:
    raise ImportError(
        "singlet.torch requires PyTorch. Install with: pip install singlet[torch]"
    )

from singlet.torch._torch import (
    to_sparse_csr, to_sparse_coo, from_anndata,
    SpzDataset, OnePZDataset, DataLoader,
)

__all__ = [
    "to_sparse_csr", "to_sparse_coo", "from_anndata",
    "SpzDataset", "OnePZDataset", "DataLoader",
]
