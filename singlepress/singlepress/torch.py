"""
singlepress.torch — Zero-copy PyTorch dataloaders for .1pz files.

Three dataset classes for different shuffle/memory trade-offs:

  OnePZDataset         — chunk-level map-style, optional within-chunk shuffle
  OnePZCellDataset     — pre-loaded, true uniform cell-level shuffle
  OnePZShuffleDataset  — streaming shuffle buffer, bounded memory, ~uniform

Usage (true uniform cell shuffle):
    from singlepress.torch import OnePZCellDataset

    ds = OnePZCellDataset("counts.1pz")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=True,
        collate_fn=ds.collate_fn, num_workers=4, pin_memory=True)

    for batch in loader:
        gpu_batch = batch.to("cuda", non_blocking=True)

Usage (memory-efficient two-level shuffle):
    from singlepress.torch import OnePZDataset, collate_sparse

    ds = OnePZDataset("counts.1pz", shuffle=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=True,
        collate_fn=collate_sparse, num_workers=4)

Usage (streaming shuffle buffer):
    from singlepress.torch import OnePZShuffleDataset

    ds = OnePZShuffleDataset("counts.1pz", buffer_size=10240, batch_size=256)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2)
"""

from __future__ import annotations

import math
import numpy as np
import scipy.sparse as ss
from typing import Optional, Union

import singlepress


def _resolve_dtype(dtype_str: str):
    """Map string dtype to torch dtype (deferred import)."""
    import torch
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
    }.get(dtype_str, torch.float32)


def _to_torch_csr(mat: ss.csc_matrix, dtype=None):
    """Convert scipy CSC matrix to torch sparse CSR tensor (zero-copy where possible).

    Parameters
    ----------
    mat : scipy.sparse.csc_matrix
        Input sparse matrix.
    dtype : torch.dtype, optional
        Value dtype. Default: torch.float32.

    Returns
    -------
    torch.Tensor
        Sparse CSR tensor. Call .to("cuda", non_blocking=True) for GPU transfer.
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    csr = mat.tocsr()
    csr.sort_indices()

    crow_indices = torch.from_numpy(np.asarray(csr.indptr, dtype=np.int32))
    col_indices = torch.from_numpy(np.asarray(csr.indices, dtype=np.int32))

    if dtype == torch.float32:
        values = torch.from_numpy(np.asarray(csr.data, dtype=np.float32))
    elif dtype == torch.float64:
        values = torch.from_numpy(np.asarray(csr.data, dtype=np.float64))
    elif dtype == torch.int32:
        values = torch.from_numpy(np.asarray(csr.data, dtype=np.int32))
    else:
        values = torch.from_numpy(np.asarray(csr.data, dtype=np.float32))

    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values,
        size=csr.shape, dtype=dtype,
    )


def _to_torch_coo(mat: ss.csc_matrix, dtype=None):
    """Convert scipy CSC matrix to torch sparse COO tensor.

    Parameters
    ----------
    mat : scipy.sparse.csc_matrix
        Input sparse matrix.
    dtype : torch.dtype, optional
        Value dtype. Default: torch.float32.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor.
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    coo = mat.tocoo()
    indices = torch.stack([
        torch.from_numpy(np.asarray(coo.row, dtype=np.int64)),
        torch.from_numpy(np.asarray(coo.col, dtype=np.int64)),
    ])

    if dtype == torch.float32:
        values = torch.from_numpy(np.asarray(coo.data, dtype=np.float32))
    elif dtype == torch.float64:
        values = torch.from_numpy(np.asarray(coo.data, dtype=np.float64))
    else:
        values = torch.from_numpy(np.asarray(coo.data, dtype=np.float32))

    return torch.sparse_coo_tensor(indices, values, size=coo.shape, dtype=dtype)


# ============================================================================
# OnePZDataset — chunk-level with optional within-chunk shuffle
# ============================================================================

class OnePZDataset:
    """Map-style dataset that reads .1pz files by column chunks.

    Each item is a sparse tensor covering ``chunk_size`` cells.
    With ``shuffle=True``, cells within each chunk are randomly permuted
    every time they are accessed. Combined with ``DataLoader(shuffle=True)``,
    this gives two-level shuffling (inter-chunk + intra-chunk) that is
    indistinguishable from uniform for SGD training.

    Parameters
    ----------
    path : str
        Path to a .1pz file.
    chunk_size : int
        Cells per item. Default 1024 (matches .1pz chunk alignment).
    shuffle : bool
        Randomly permute cells within each chunk on every access. Default False.
    normalize : bool
        Apply log-normalization using stored column sums. Default False.
    scale : float
        Normalization scale factor. Default 10000.
    dtype : str
        "float32" (default), "float64", or "int32".
    sparse_format : str
        "csr" (default, cuSPARSE-compatible) or "coo".
    seed : int or None
        RNG seed for within-chunk shuffling. None = non-deterministic.
    """

    def __init__(
        self,
        path: str,
        chunk_size: int = 1024,
        shuffle: bool = False,
        normalize: bool = False,
        scale: float = 10000.0,
        dtype: str = "float32",
        sparse_format: str = "csr",
        seed: Optional[int] = None,
    ):
        self.path = str(path)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.scale = scale
        self.sparse_format = sparse_format
        self._dtype_str = dtype

        self._handle = singlepress.open_1pz(self.path)
        self._nrows, self._ncols = self._handle.shape
        self._colsums = self._handle.colsums if normalize else None
        self._n_chunks = math.ceil(self._ncols / self.chunk_size)
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self._n_chunks

    def __getitem__(self, idx: int):
        """Decode chunk ``idx`` and return a sparse tensor.

        Safe for multi-worker DataLoader (each worker gets independent handles).
        """
        torch_dtype = _resolve_dtype(self._dtype_str)

        col_start = idx * self.chunk_size
        col_end = min(col_start + self.chunk_size, self._ncols)

        mat = singlepress.read_1pz_columns(self.path, col_start, col_end)

        if self.normalize and self._colsums is not None:
            chunk_colsums = self._colsums[col_start:col_end]
            mat = singlepress.lognorm(mat, chunk_colsums, scale=self.scale)

        if self.shuffle:
            perm = self._rng.permutation(mat.shape[1])
            mat = mat[:, perm]

        if self.sparse_format == "csr":
            return _to_torch_csr(mat, dtype=torch_dtype)
        else:
            return _to_torch_coo(mat, dtype=torch_dtype)

    def reshuffle(self, seed: Optional[int] = None):
        """Reset the shuffle RNG (call between epochs for fresh permutations)."""
        self._rng = np.random.RandomState(seed)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nrows, self._ncols)

    @property
    def num_features(self) -> int:
        return self._nrows

    @property
    def num_cells(self) -> int:
        return self._ncols


# ============================================================================
# OnePZCellDataset — pre-loaded, true uniform cell-level shuffle
# ============================================================================

class OnePZCellDataset:
    """Cell-level dataset for true uniform shuffling via pre-loaded matrix.

    Loads the full .1pz matrix into memory at construction. Each
    ``__getitem__`` returns one cell's sparse data (row indices + values).
    Use with ``DataLoader(shuffle=True, collate_fn=dataset.collate_fn)``
    for exact uniform cell-level shuffling.

    Memory: O(nnz) — full matrix resident in RAM as scipy CSC.
    CSC column extraction is O(nnz_per_cell).

    Parameters
    ----------
    path : str
        Path to a .1pz file.
    normalize : bool
        Apply log-normalization at load time. Default False.
    scale : float
        Normalization scale factor. Default 10000.
    dtype : str
        "float32" (default), "float64", or "int32".
    sparse_format : str
        "csr" (default) or "coo" for batch output tensors.
    """

    def __init__(
        self,
        path: str,
        normalize: bool = False,
        scale: float = 10000.0,
        dtype: str = "float32",
        sparse_format: str = "csr",
    ):
        self.path = str(path)
        self.sparse_format = sparse_format
        self._dtype_str = dtype

        if dtype == "int32" and not normalize:
            self._mat = singlepress.read_1pz_int(path)
        else:
            self._mat = singlepress.read_1pz(path)

        if normalize:
            cs = singlepress.colsums_1pz(path)
            self._mat = singlepress.lognorm(self._mat, cs, scale=scale)

        self._mat.sort_indices()
        self._nrows, self._ncols = self._mat.shape

    def __len__(self) -> int:
        return self._ncols

    def __getitem__(self, idx: int):
        """Return ``(row_indices, values)`` arrays for cell ``idx``."""
        s = self._mat.indptr[idx]
        e = self._mat.indptr[idx + 1]
        return self._mat.indices[s:e].copy(), self._mat.data[s:e].copy()

    def collate_fn(self, batch: list):
        """Assemble individual cells into a sparse batch tensor.

        Use as ``DataLoader(..., collate_fn=dataset.collate_fn)``.

        Returns a sparse CSR (or COO) tensor of shape ``(num_features, batch_size)``.
        """
        torch_dtype = _resolve_dtype(self._dtype_str)
        ncells = len(batch)
        nnzs = [len(indices) for indices, _ in batch]
        total_nnz = sum(nnzs)

        indptr = np.empty(ncells + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(nnzs, out=indptr[1:])

        if total_nnz > 0:
            all_indices = np.empty(total_nnz, dtype=np.int32)
            all_values = np.empty(total_nnz, dtype=np.float64)
            pos = 0
            for indices, values in batch:
                n = len(indices)
                all_indices[pos:pos + n] = indices
                all_values[pos:pos + n] = values
                pos += n
        else:
            all_indices = np.array([], dtype=np.int32)
            all_values = np.array([], dtype=np.float64)

        csc = ss.csc_matrix(
            (all_values, all_indices, indptr),
            shape=(self._nrows, ncells),
        )

        if self.sparse_format == "csr":
            return _to_torch_csr(csc, dtype=torch_dtype)
        return _to_torch_coo(csc, dtype=torch_dtype)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nrows, self._ncols)

    @property
    def num_features(self) -> int:
        return self._nrows

    @property
    def num_cells(self) -> int:
        return self._ncols


# ============================================================================
# OnePZShuffleDataset — streaming shuffle buffer (bounded memory)
# ============================================================================

class OnePZShuffleDataset:
    """Streaming dataset with shuffle buffer for memory-bounded uniform shuffling.

    Reads .1pz chunks into a cell buffer in random chunk order. Yields
    batches by sampling uniformly from the buffer. Achieves approximately
    uniform cell shuffling without loading the full matrix.

    For exact uniform shuffling, use ``OnePZCellDataset`` instead.
    Shuffle quality improves with larger ``buffer_size``.

    Use with ``DataLoader(batch_size=None)`` since each yielded item is
    already a complete batch.

    Parameters
    ----------
    path : str
        Path to .1pz file.
    buffer_size : int
        Cells held in the shuffle buffer. Default 10240.
    batch_size : int
        Cells per yielded batch. Default 1024.
    normalize : bool
        Apply log-normalization. Default False.
    scale : float
        Normalization scale. Default 10000.
    dtype : str
        "float32" (default), "float64", or "int32".
    sparse_format : str
        "csr" (default) or "coo".
    """

    def __init__(
        self,
        path: str,
        buffer_size: int = 10240,
        batch_size: int = 1024,
        normalize: bool = False,
        scale: float = 10000.0,
        dtype: str = "float32",
        sparse_format: str = "csr",
    ):
        self.path = str(path)
        self.buffer_size = max(buffer_size, batch_size * 2)
        self.batch_size = batch_size
        self.normalize = normalize
        self.scale = scale
        self.sparse_format = sparse_format
        self._dtype_str = dtype

        handle = singlepress.open_1pz(self.path)
        self._nrows, self._ncols = handle.shape
        self._colsums = handle.colsums if normalize else None
        info = singlepress.info_1pz(self.path)
        self._chunk_cols = info.get("chunk_cols", 1024)
        self._n_chunks = math.ceil(self._ncols / self._chunk_cols)

    def _build_batch(self, buf_indices, buf_values, n):
        """Build a sparse tensor from n cells sampled from the buffer."""
        torch_dtype = _resolve_dtype(self._dtype_str)
        pick = np.random.choice(len(buf_indices), n, replace=False)

        indptr = np.empty(n + 1, dtype=np.int32)
        indptr[0] = 0
        batch_idx = []
        batch_val = []
        for k, p in enumerate(pick):
            batch_idx.append(buf_indices[p])
            batch_val.append(buf_values[p])
            indptr[k + 1] = indptr[k] + len(buf_indices[p])

        # Swap-remove picked cells from buffer (reverse order for stability)
        for p in sorted(pick, reverse=True):
            buf_indices[p] = buf_indices[-1]
            buf_values[p] = buf_values[-1]
            buf_indices.pop()
            buf_values.pop()

        all_idx = np.concatenate(batch_idx) if batch_idx else np.array([], dtype=np.int32)
        all_val = np.concatenate(batch_val) if batch_val else np.array([], dtype=np.float64)

        csc = ss.csc_matrix(
            (all_val, all_idx, indptr),
            shape=(self._nrows, n),
        )
        if self.sparse_format == "csr":
            return _to_torch_csr(csc, dtype=torch_dtype)
        return _to_torch_coo(csc, dtype=torch_dtype)

    def __iter__(self):
        import torch.utils.data as data

        # Worker sharding
        worker_info = data.get_worker_info()
        chunk_order = np.random.permutation(self._n_chunks)
        if worker_info is not None:
            per_worker = math.ceil(len(chunk_order) / worker_info.num_workers)
            start = worker_info.id * per_worker
            chunk_order = chunk_order[start:start + per_worker]

        # Buffer: list of (row_indices, values) per cell
        buf_indices = []
        buf_values = []

        def _add_chunk(chunk):
            for j in range(chunk.shape[1]):
                s, e = chunk.indptr[j], chunk.indptr[j + 1]
                buf_indices.append(chunk.indices[s:e].copy())
                buf_values.append(chunk.data[s:e].copy())

        for ci in chunk_order:
            col_start = int(ci) * self._chunk_cols
            col_end = min(col_start + self._chunk_cols, self._ncols)
            chunk = singlepress.read_1pz_columns(self.path, col_start, col_end)

            if self.normalize and self._colsums is not None:
                cs = self._colsums[col_start:col_end]
                chunk = singlepress.lognorm(chunk, cs, scale=self.scale)

            _add_chunk(chunk)

            while len(buf_indices) >= self.buffer_size + self.batch_size:
                yield self._build_batch(buf_indices, buf_values, self.batch_size)

        # Drain remaining buffer
        while len(buf_indices) >= self.batch_size:
            yield self._build_batch(buf_indices, buf_values, self.batch_size)

        # Final partial batch
        if buf_indices:
            yield self._build_batch(buf_indices, buf_values, len(buf_indices))

    def __len__(self) -> int:
        return math.ceil(self._ncols / self.batch_size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nrows, self._ncols)


# ============================================================================
# collate_sparse — for OnePZDataset chunk-level collation
# ============================================================================

def collate_sparse(batch: list):
    """Collate function for DataLoader that concatenates sparse CSR tensors.

    Horizontally stacks sparse tensors along the column dimension (cells).
    Use as: ``DataLoader(dataset, collate_fn=collate_sparse, ...)``.

    Parameters
    ----------
    batch : list of torch.Tensor
        Sparse CSR or COO tensors from OnePZDataset.__getitem__.

    Returns
    -------
    torch.Tensor
        Single sparse tensor with all cells concatenated.
    """
    import torch

    if len(batch) == 1:
        return batch[0]

    scipy_mats = []
    for t in batch:
        if t.is_sparse_csr:
            crow = t.crow_indices().numpy()
            col = t.col_indices().numpy()
            vals = t.values().numpy()
            m = ss.csr_matrix((vals, col, crow), shape=t.shape)
            scipy_mats.append(m.tocsc())
        elif t.is_sparse:
            indices = t.coalesce().indices().numpy()
            vals = t.coalesce().values().numpy()
            m = ss.coo_matrix((vals, (indices[0], indices[1])), shape=t.shape)
            scipy_mats.append(m.tocsc())
        else:
            raise TypeError(f"Expected sparse tensor, got {type(t)}")

    combined = ss.hstack(scipy_mats, format="csc")
    return _to_torch_csr(combined, dtype=batch[0].dtype)


# ============================================================================
# OnePZIterableDataset — sequential streaming (no shuffle)
# ============================================================================

class OnePZIterableDataset:
    """Iterable-style dataset for streaming large .1pz files.

    Yields one chunk at a time in sequential order. Supports worker sharding
    for multi-process DataLoading. For shuffled streaming, use
    ``OnePZShuffleDataset`` instead.

    Usage:
        from singlepress.torch import OnePZIterableDataset

        dataset = OnePZIterableDataset("counts.1pz")
        loader = torch.utils.data.DataLoader(dataset, num_workers=4)
        for batch in loader:
            gpu = batch.to("cuda", non_blocking=True)
    """

    def __init__(
        self,
        path: str,
        chunk_size: int = 1024,
        normalize: bool = False,
        scale: float = 10000.0,
        dtype: str = "float32",
        sparse_format: str = "csr",
    ):
        self.path = str(path)
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.scale = scale
        self.sparse_format = sparse_format
        self._dtype_str = dtype

        handle = singlepress.open_1pz(self.path)
        self._nrows, self._ncols = handle.shape
        self._colsums = handle.colsums if normalize else None
        self._n_chunks = math.ceil(self._ncols / self.chunk_size)

    def __iter__(self):
        import torch
        import torch.utils.data as data

        worker_info = data.get_worker_info()
        if worker_info is None:
            start, end = 0, self._n_chunks
        else:
            per_worker = math.ceil(self._n_chunks / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, self._n_chunks)

        torch_dtype = _resolve_dtype(self._dtype_str)

        for idx in range(start, end):
            col_start = idx * self.chunk_size
            col_end = min(col_start + self.chunk_size, self._ncols)

            mat = singlepress.read_1pz_columns(self.path, col_start, col_end)

            if self.normalize and self._colsums is not None:
                chunk_colsums = self._colsums[col_start:col_end]
                mat = singlepress.lognorm(mat, chunk_colsums, scale=self.scale)

            if self.sparse_format == "csr":
                yield _to_torch_csr(mat, dtype=torch_dtype)
            else:
                yield _to_torch_coo(mat, dtype=torch_dtype)

    def __len__(self) -> int:
        return self._n_chunks
