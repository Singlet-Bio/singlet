"""singlet.torch — PyTorch integration for .1pz sparse matrices.

Provides zero-copy DataLoaders and GPU-friendly sparse tensor pipelines.

Classes:
    OnePZDataset        Map-style dataset with chunk-level access
    OnePZCellDataset    Pre-loaded dataset with true cell-level shuffle
    OnePZShuffleDataset Streaming shuffle buffer (bounded memory)
    PZBufferedLoader    Zero-alloc buffer-reuse with GPU double-buffering

Functions:
    collate_sparse      Custom collate for sparse batches
"""
try:
    import torch as _torch
except ImportError:
    raise ImportError(
        "singlet.torch requires PyTorch. Install with: pip install singlet[torch]"
    )

from singlet.io.pz import (
    OnePZDataset, OnePZCellDataset, OnePZShuffleDataset, PZBufferedLoader,
    collate_sparse, read_1pz_torch,
)

__all__ = [
    "OnePZDataset", "OnePZCellDataset", "OnePZShuffleDataset",
    "PZBufferedLoader", "collate_sparse", "read_1pz_torch",
]
