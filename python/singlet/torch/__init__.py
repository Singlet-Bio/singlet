# SPDX-License-Identifier: MIT
"""singlet.torch — PyTorch integration for Singlet datasets.

Provides DataLoaders and GPU-friendly sparse tensor pipelines over ``.singlet``
files and GEO accessions.

Classes:
    AtlasCells          IterableDataset streaming cells of any atlas filter from
                        data.singlet.bio by HTTP range reads, with study-held-out splits
    S1Cells             IterableDataset of block-shuffled minibatches from one .s1 object
                        (a local file or an HTTP URL), read by range requests
    S1Stream            Range-read access to the cells of one .s1 object (batch, cells)
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
        "singlet.torch requires PyTorch. Install with: pip install 'singlet[torch]'"
    ) from None

from singlet.torch._s1stream import S1Cells, S1Stream, block_batches
from singlet.torch._stream import AtlasCells, plan_samples, study_split
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
    "AtlasCells",
    "S1Cells",
    "S1Stream",
    "block_batches",
    "plan_samples",
    "study_split",
    "to_sparse_csr",
    "to_sparse_coo",
    "from_anndata",
    "SpzDataset",
    "SingletDataset",
    "OnePZDataset",
    "DataLoader",
]
