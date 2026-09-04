# SPDX-License-Identifier: MIT
"""PyTorch integration: sparse tensors and DataLoaders for Singlet datasets.

Datasets are built from ``.singlet`` files, GEO accessions, or in-memory
AnnData. The preferred form is an array (list/tuple) of ``.singlet`` file
paths — these are concatenated into one training set.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


def _read_pz_anndata(source: str | Path):
    """Read a ``.1pz`` (or legacy ``.spz``) file with the in-tree codec."""
    from singlet._io import read_matrix

    return read_matrix(source)


def to_sparse_csr(
    source: str | Path,
    *,
    dtype: str = "float32",
    device: str = "cpu",
) -> torch.Tensor:
    """Load a .1pz file as a PyTorch sparse CSR tensor.

    CSR format is optimal for GPU operations through cuSPARSE.
    The returned tensor has shape (cells, genes) — transposed from the
    on-disk genes×cells layout.

    Parameters
    ----------
    source : str or Path
        Path to a .1pz file.
    dtype : str
        ``"float32"`` (default) or ``"float64"``.
    device : str
        ``"cpu"`` or ``"cuda"`` / ``"cuda:0"`` etc.

    Returns
    -------
    torch.Tensor
        Sparse CSR tensor, shape (n_cells, n_genes).
    """
    return from_anndata(_read_pz_anndata(source), dtype=dtype, device=device)


def to_sparse_coo(
    source: str | Path,
    *,
    dtype: str = "float32",
    device: str = "cpu",
) -> torch.Tensor:
    """Load a .1pz file as a PyTorch sparse COO tensor.

    Parameters
    ----------
    source : str or Path
        Path to a .1pz file.
    dtype : str
        ``"float32"`` (default) or ``"float64"``.
    device : str
        ``"cpu"`` or ``"cuda"`` / ``"cuda:0"`` etc.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor, shape (n_cells, n_genes).
    """
    return from_anndata(_read_pz_anndata(source), dtype=dtype, device=device).to_sparse_coo()


def from_anndata(
    adata,
    *,
    layer: Optional[str] = None,
    dtype: str = "float32",
    device: str = "cpu",
) -> torch.Tensor:
    """Convert AnnData to a PyTorch sparse CSR tensor.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data (cells × genes).
    layer : str, optional
        Use this layer instead of X.
    dtype : str
        ``"float32"`` or ``"float64"``.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    torch.Tensor
        Sparse CSR tensor on the specified device.
    """
    import numpy as np
    import scipy.sparse as sp
    import torch

    mat = adata.layers[layer] if layer else adata.X
    if not sp.issparse(mat):
        mat = sp.csr_matrix(mat)
    elif mat.format != "csr":
        mat = mat.tocsr()

    crow = torch.from_numpy(mat.indptr.astype(np.int64))
    col = torch.from_numpy(mat.indices.astype(np.int64))

    if dtype == "float64":
        vals = torch.from_numpy(mat.data.astype(np.float64))
    else:
        vals = torch.from_numpy(mat.data.astype(np.float32))

    t = torch.sparse_csr_tensor(crow, col, vals, size=mat.shape)
    if device != "cpu":
        t = t.to(device)
    return t


class SpzDataset:
    """PyTorch Dataset backed by a .spz file or AnnData (legacy).

    Use :class:`OnePZDataset` for .1pz files (preferred).

    Yields one cell (row) per ``__getitem__`` call as a dense or sparse tensor.

    Parameters
    ----------
    source : str, Path, or AnnData
        GEO accession, path to .spz file, or a loaded AnnData object.
    layer : str, optional
        Which layer to use. Default: X.
    genes : list of str, optional
        Subset to these genes.
    device : str
        ``"cpu"`` or ``"cuda"``.
    sparse : bool
        If True, return sparse row tensors instead of dense.
    """

    def __init__(
        self,
        source,
        *,
        layer: Optional[str] = None,
        genes: Optional[Sequence[str]] = None,
        device: str = "cpu",
        sparse: bool = False,
    ):
        import scipy.sparse as sp
        import torch

        if hasattr(source, "X"):
            self._adata = source
        else:
            from singlet._loader import load

            self._adata = load(source)

        if genes is not None:
            mask = self._adata.var_names.isin(genes)
            self._adata = self._adata[:, mask]

        mat = self._adata.layers[layer] if layer else self._adata.X
        if not sp.issparse(mat):
            mat = sp.csr_matrix(mat)
        elif mat.format != "csr":
            mat = mat.tocsr()

        self._mat = mat
        self._device = torch.device(device)
        self._sparse = sparse
        self.n_genes = mat.shape[1]

    def __len__(self) -> int:
        return self._mat.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        import numpy as np
        import torch

        row = self._mat[idx]

        if self._sparse:
            # Return as sparse COO tensor (single row)
            if hasattr(row, "tocoo"):
                coo = row.tocoo()
                indices = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
                values = torch.tensor(coo.data, dtype=torch.float32)
                return torch.sparse_coo_tensor(indices, values, row.shape).to(self._device)

        if hasattr(row, "toarray"):
            row = row.toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32, device=self._device)


class SingletDataset:
    """PyTorch Dataset backed by a Singlet dataset.

    Loads the matrix into memory at construction, then yields one cell per
    ``__getitem__`` call.

    Parameters
    ----------
    source : str, Path, or AnnData
        Path to a ``.singlet`` file, a GEO accession (e.g. ``"GSE149298"``),
        or a loaded AnnData object.
    genes : list of str, optional
        Subset to these genes.
    normalize : bool
        Apply log1p(x * 10000 / total_counts) normalization.
    device : str
        ``"cpu"`` or ``"cuda"``.
    sparse : bool
        If True, return sparse row tensors instead of dense.
    """

    def __init__(
        self,
        source,
        *,
        genes: Optional[Sequence[str]] = None,
        normalize: bool = False,
        device: str = "cpu",
        sparse: bool = False,
    ):
        import scipy.sparse as sp
        import torch

        if hasattr(source, "X"):
            self._adata = source
        else:
            from singlet._loader import load

            self._adata = load(source)

        if genes is not None:
            mask = self._adata.var_names.isin(genes)
            self._adata = self._adata[:, mask]

        mat = self._adata.X
        if not sp.issparse(mat):
            mat = sp.csr_matrix(mat)
        elif mat.format != "csr":
            mat = mat.tocsr()

        if normalize:
            import numpy as np

            totals = np.array(mat.sum(axis=1)).ravel()
            totals[totals == 0] = 1
            mat = mat.multiply(1.0 / totals[:, None]).multiply(10000)
            mat.data = np.log1p(mat.data)

        self._mat = mat
        self._device = torch.device(device)
        self._sparse = sparse
        self.n_genes = mat.shape[1]
        self.gene_names = list(self._adata.var_names)

    def __len__(self) -> int:
        return self._mat.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        import numpy as np
        import torch

        row = self._mat[idx]

        if self._sparse:
            if hasattr(row, "tocoo"):
                coo = row.tocoo()
                indices = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
                values = torch.tensor(coo.data, dtype=torch.float32)
                return torch.sparse_coo_tensor(indices, values, row.shape).to(self._device)

        if hasattr(row, "toarray"):
            row = row.toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32, device=self._device)


class DataLoader:
    """Convenience wrapper around ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    source : str, Path, AnnData, or list/tuple
        A GEO accession, a ``.singlet`` file path, an AnnData object, or — the
        preferred form for training — an array (list/tuple) of ``.singlet`` file
        paths and/or accessions. Arrays are concatenated into one training set
        (``torch.utils.data.ConcatDataset``).
    batch_size : int
        Batch size.
    shuffle : bool
        Shuffle cells each epoch.
    num_workers : int
        DataLoader workers.
    device : str
        ``"cpu"`` or ``"cuda"``.
    genes : list of str, optional
        Subset to these genes.
    normalize : bool
        Apply log1p normalization.
    sparse : bool
        Return sparse tensors from dataset.
    """

    def __init__(
        self,
        source,
        *,
        batch_size: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
        device: str = "cpu",
        genes: Optional[Sequence[str]] = None,
        normalize: bool = False,
        sparse: bool = False,
    ):
        from torch.utils.data import ConcatDataset
        from torch.utils.data import DataLoader as TorchDL

        if isinstance(source, (list, tuple)):
            datasets = [
                SingletDataset(s, genes=genes, normalize=normalize, device=device, sparse=sparse)
                for s in source
            ]
            dataset = ConcatDataset(datasets)
        else:
            dataset = SingletDataset(
                source,
                genes=genes,
                normalize=normalize,
                device=device,
                sparse=sparse,
            )

        self._loader = TorchDL(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


# Deprecated alias — kept for backward compatibility. Prefer SingletDataset.
OnePZDataset = SingletDataset
