"""
singlepress.torch — Zero-copy PyTorch dataloaders for .1pz files.

All sparse tensors are returned in CSR(cells × genes) orientation — batch
dimension first, ready for ``nn.Linear`` and cuSPARSE. No format conversion
or index sorting is performed; CSC on-disk layout is reinterpreted as CSR
in O(1) time.

Four dataset/loader classes for different shuffle/memory trade-offs:

  OnePZDataset         — chunk-level map-style, optional within-chunk shuffle
  OnePZCellDataset     — pre-loaded, true uniform cell-level shuffle
  OnePZShuffleDataset  — streaming shuffle buffer, bounded memory, ~uniform
  PZBufferedLoader     — zero-alloc buffer-reuse loader with GPU double-buffering

Usage (zero-alloc GPU training with buffer pool):
    from singlepress.torch import PZBufferedLoader

    loader = PZBufferedLoader("counts.1pz", device="cuda")
    for epoch in range(10):
        loader.reset_timing()
        for batch in loader:
            loss = model(batch)
            loss.backward()
        print(loader.timing_report())

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
import time
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
    """Reinterpret scipy CSC(genes×cells) as torch CSR(cells×genes) — zero-copy.

    CSC(genes×cells) and CSR(cells×genes) share identical memory layout:
    indptr indexes columns (cells), indices hold row (gene) positions.
    No format conversion or index sorting is needed.

    Parameters
    ----------
    mat : scipy.sparse.csc_matrix
        Input sparse matrix in CSC format (genes × cells).
    dtype : torch.dtype, optional
        Value dtype. Default: torch.float32.

    Returns
    -------
    torch.Tensor
        Sparse CSR tensor of shape (cells, genes).
        Call .to("cuda", non_blocking=True) for GPU transfer.
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    # Zero-copy for int32 indptr/indices (C++ decoder returns int32)
    crow_indices = torch.from_numpy(np.asarray(mat.indptr, dtype=np.int32))
    col_indices = torch.from_numpy(np.asarray(mat.indices, dtype=np.int32))

    if dtype == torch.float32:
        values = torch.from_numpy(np.asarray(mat.data, dtype=np.float32))
    elif dtype == torch.float64:
        values = torch.from_numpy(np.asarray(mat.data, dtype=np.float64))
    elif dtype == torch.int32:
        values = torch.from_numpy(np.asarray(mat.data, dtype=np.int32))
    else:
        values = torch.from_numpy(np.asarray(mat.data, dtype=np.float32))

    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values,
        size=(mat.shape[1], mat.shape[0]),  # (cells, genes)
        dtype=dtype,
    )


def _to_torch_coo(mat: ss.csc_matrix, dtype=None):
    """Convert scipy CSC(genes×cells) to torch COO(cells×genes).

    Parameters
    ----------
    mat : scipy.sparse.csc_matrix
        Input sparse matrix in CSC format (genes × cells).
    dtype : torch.dtype, optional
        Value dtype. Default: torch.float32.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor of shape (cells, genes).
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    coo = mat.tocoo()
    # Transpose: CSC rows (genes) → COO cols, CSC cols (cells) → COO rows
    indices = torch.stack([
        torch.from_numpy(np.asarray(coo.col, dtype=np.int64)),
        torch.from_numpy(np.asarray(coo.row, dtype=np.int64)),
    ])

    if dtype == torch.float32:
        values = torch.from_numpy(np.asarray(coo.data, dtype=np.float32))
    elif dtype == torch.float64:
        values = torch.from_numpy(np.asarray(coo.data, dtype=np.float64))
    else:
        values = torch.from_numpy(np.asarray(coo.data, dtype=np.float32))

        indices, values,
        size=(coo.shape[1], coo.shape[0]),  # (cells, genes)
        dtype=dtype,
    
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
        sparse_format: str = "csr",
        seed: Optional[int] = None,
    ):
        self.path = str(path)
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.scale = scale
        self.sparse_format = sparse_format
        self._dtype_str = "float32" if normalize else "int32"

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

        mat = singlepress.read_1pz_columns(self.path, col_start, col_end,
                                            sorted=False)

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
    sparse_format : str
        "csr" (default) or "coo" for batch output tensors.
    """

    def __init__(
        self,
        path: str,
        normalize: bool = False,
        scale: float = 10000.0,
        sparse_format: str = "csr",
    ):
        self.path = str(path)
        self.sparse_format = sparse_format
        self._dtype_str = "float32" if normalize else "int32"

        if not normalize:
            self._mat = singlepress.read_1pz_int(path, sorted=False)
        else:
            self._mat = singlepress.read_1pz(path, sorted=False)

        if normalize:
            cs = singlepress.colsums_1pz(path)
            self._mat = singlepress.lognorm(self._mat, cs, scale=scale)

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
        sparse_format: str = "csr",
    ):
        self.path = str(path)
        self.buffer_size = max(buffer_size, batch_size * 2)
        self.batch_size = batch_size
        self.normalize = normalize
        self.scale = scale
        self.sparse_format = sparse_format
        self._dtype_str = "float32" if normalize else "int32"

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
            chunk = singlepress.read_1pz_columns(self.path, col_start, col_end,
                                                    sorted=False)

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
    """Collate function that row-concatenates sparse CSR tensors (cells x genes).

    Stacks sparse tensors along the cell (row) dimension in pure torch
    -- no scipy round-trip. Use as: ``DataLoader(dataset, collate_fn=collate_sparse, ...)``.

    Parameters
    ----------
    batch : list of torch.Tensor
        Sparse CSR tensors from OnePZDataset.__getitem__, each (chunk_cells, genes).

    Returns
    -------
    torch.Tensor
        Single sparse CSR tensor with all cells concatenated, (total_cells, genes).
    """
    import torch

    if len(batch) == 1:
        return batch[0]

    all_crow = []
    all_col = []
    all_val = []
    offset = 0
    nrows_total = 0
    ncols = batch[0].shape[1]

    for t in batch:
        crow = t.crow_indices()
        all_crow.append(crow[:-1] + offset)
        all_col.append(t.col_indices())
        all_val.append(t.values())
        offset += t._nnz()
        nrows_total += t.shape[0]

    # Append final pointer
    all_crow.append(torch.tensor([offset], dtype=all_crow[0].dtype))

    return torch.sparse_csr_tensor(
        torch.cat(all_crow), torch.cat(all_col), torch.cat(all_val),
        size=(nrows_total, ncols), dtype=all_val[0].dtype,
    )


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

            mat = singlepress.read_1pz_columns(self.path, col_start, col_end,
                                                sorted=False)

            if self.normalize and self._colsums is not None:
                chunk_colsums = self._colsums[col_start:col_end]
                mat = singlepress.lognorm(mat, chunk_colsums, scale=self.scale)

            if self.sparse_format == "csr":
                yield _to_torch_csr(mat, dtype=torch_dtype)
            else:
                yield _to_torch_coo(mat, dtype=torch_dtype)

    def __len__(self) -> int:
        return self._n_chunks


# ============================================================================
# BufferPool + PZBufferedLoader — zero-alloc loader with buffer reuse
# ============================================================================

class BufferPool:
    """Reusable memory pool for .1pz column-range decode output.

    Holds indptr / indices / values arrays that persist across batches.
    Decompression is **destructive**: each ``decode_columns_into`` call
    overwrites the previous batch's data in-place.

    Buffers grow on demand but never shrink, so capacity stabilises after
    the densest batch (typically within the first epoch).

    When ``pinned=True``, arrays are backed by CUDA page-locked memory via
    ``torch.empty(..., pin_memory=True)`` for non-blocking H2D transfers.
    The numpy views share the same underlying memory (zero-copy).
    """

    __slots__ = ("indptr", "indices", "values",
                 "_indptr_t", "_indices_t", "_values_t",
                 "nnz_cap", "ncols_max", "pinned",
                 "grow_ns", "grow_count")

    def __init__(self, nnz_cap: int, ncols_max: int, *, pinned: bool = False):
        self.ncols_max = ncols_max
        self.pinned = pinned
        self.grow_ns = 0
        self.grow_count = 0
        self._alloc_indptr(ncols_max)
        self._alloc_nnz(nnz_cap)

    def _alloc_indptr(self, ncols_max: int):
        if self.pinned:
            import torch
            self._indptr_t = torch.zeros(ncols_max + 1, dtype=torch.int32,
                                         pin_memory=True)
            self.indptr = self._indptr_t.numpy()
        else:
            self.indptr = np.zeros(ncols_max + 1, dtype=np.int32)
            self._indptr_t = None

    def _alloc_nnz(self, nnz_cap: int):
        self.nnz_cap = nnz_cap
        if self.pinned:
            import torch
            self._indices_t = torch.zeros(nnz_cap, dtype=torch.int32,
                                          pin_memory=True)
            self._values_t = torch.zeros(nnz_cap, dtype=torch.int32,
                                         pin_memory=True)
            self.indices = self._indices_t.numpy()
            self.values = self._values_t.numpy()
        else:
            self.indices = np.zeros(nnz_cap, dtype=np.int32)
            self.values = np.zeros(nnz_cap, dtype=np.int32)
            self._indices_t = None
            self._values_t = None

    def ensure_capacity(self, nnz_needed: int) -> int:
        """Grow nnz buffers if ``nnz_needed > nnz_cap``.  Returns ns spent."""
        if nnz_needed <= self.nnz_cap:
            return 0
        t0 = time.monotonic_ns()
        self._alloc_nnz(nnz_needed)
        ns = time.monotonic_ns() - t0
        self.grow_ns += ns
        self.grow_count += 1
        return ns


class PZBufferedLoader:
    """Zero-allocation DataLoader for .1pz files with GPU double-buffering.

    Standard DataLoaders re-open and re-parse the .1pz file for **every
    batch**, allocating fresh file-blob, permutation, indptr, and output
    arrays each time.  ``PZBufferedLoader`` eliminates this overhead:

    1. **File cache** — A C++ ``PZReader`` reads the file once and caches
       the raw blob, permutation table, full indptr, and chunk offset table
       in memory.  Subsequent batches decode directly from the cache.
    2. **Buffer reuse** — Two ``BufferPool`` instances hold indptr / indices /
       values arrays.  Each batch's decompression is **destructive**: it
       overwrites the previous batch's data in-place.  Buffers grow on
       demand but never shrink, stabilising within one epoch.
    3. **GPU double-buffering** (``device="cuda"``) — A background thread
       decompresses the next batch into pool B while pool A's data is on
       the GPU for forward/backward.  Since decompression is always faster
       than a GPU forward pass, the CPU is ready before the GPU finishes.

    Decompression within each chunk is also destructive: the per-thread OMP
    scratch buffers (raw bytes, gap tables, packed/prefix buffers) reuse
    ``std::vector`` storage that only grows, never shrinks — so they
    stabilise after the first chunk and incur zero ``malloc`` thereafter.

    Access ``.timing`` after iteration for a per-batch breakdown of time
    spent on allocation (buffer grows), decompression, and H2D transfer.

    Parameters
    ----------
    path : str
        Path to .1pz file.
    chunk_size : int
        Cells per batch (columns decoded per iteration).  Default 1024.
        Best performance when aligned to the file's ``chunk_cols``.
    device : str
        ``"cpu"`` for single-buffer sequential mode, ``"cuda"`` (or
        ``"cuda:N"``) for double-buffered GPU mode with pinned memory.
    num_threads : int
        OMP threads for zstd decompression + VOCSC decode.  Default 4.
    normalize : bool
        Apply log1p-normalization (``log1p(x * scale / colsum)``) using
        stored per-cell column sums.  Default False.
    scale : float
        Normalization scale factor.  Default 10_000.
    dtype : str
        Output values dtype: ``"float32"`` (default) or ``"int32"``.
        Ignored when ``normalize=True`` (always float32).

    Examples
    --------
    GPU training with timing:

    >>> loader = PZBufferedLoader("counts.1pz", device="cuda")
    >>> for epoch in range(10):
    ...     loader.reset_timing()
    ...     for batch in loader:
    ...         loss = model(batch)
    ...         loss.backward()
    ...     print(loader.timing_report())

    CPU out-of-core (single buffer, destructive overwrite):

    >>> loader = PZBufferedLoader("counts.1pz", device="cpu", dtype="int32")
    >>> for batch in loader:
    ...     # batch wraps pool memory — valid until next iteration
    ...     process(batch)
    """

    def __init__(
        self,
        path: str,
        chunk_size: int = 1024,
        device: str = "cpu",
        num_threads: int = 4,
        normalize: bool = False,
        scale: float = 10_000.0,
        dtype: str = "float32",
    ):
        import singlepress._pz_codec as C

        self._reader = C.PZReader(str(path))
        self._chunk_size = chunk_size
        self._device = device
        self._num_threads = num_threads
        self._normalize = normalize
        self._scale = scale
        self._dtype_str = "float32" if normalize else dtype

        self._nrows = self._reader.nrows
        self._ncols = self._reader.ncols
        self._n_chunks = math.ceil(self._ncols / chunk_size)
        self._colsums = self._reader.colsums() if normalize else None

        # Pre-scan chunk nnz to size pools for the densest batch
        max_nnz = 0
        for ci in range(self._n_chunks):
            cs = ci * chunk_size
            ce = min(cs + chunk_size, self._ncols)
            max_nnz = max(max_nnz, self._reader.range_nnz(cs, ce))

        use_pinned = device != "cpu"
        n_pools = 2 if use_pinned else 1
        self._pools = [
            BufferPool(max_nnz, chunk_size, pinned=use_pinned)
            for _ in range(n_pools)
        ]

        self.timing = {
            "alloc_ns": 0, "decompress_ns": 0, "h2d_ns": 0,
            "total_ns": 0, "batches": 0, "buffer_grows": 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_batch(self, pool: BufferPool, col_start: int, col_end: int):
        """Decompress [col_start, col_end) into *pool* (destructive)."""
        nnz_needed = self._reader.range_nnz(col_start, col_end)

        alloc_ns = pool.ensure_capacity(nnz_needed)
        if alloc_ns > 0:
            self.timing["alloc_ns"] += alloc_ns
            self.timing["buffer_grows"] += 1

        t0 = time.monotonic_ns()
        result = self._reader.decode_columns_into(
            col_start, col_end,
            pool.indptr, pool.indices, pool.values,
            self._num_threads,
        )
        self.timing["decompress_ns"] += time.monotonic_ns() - t0
        return result["nnz"], col_end - col_start

    def _pool_to_csr(self, pool: BufferPool, ncols: int, nnz: int,
                     col_start: int = 0):
        """Build a sparse CSR tensor from *pool*'s buffers.

        For pinned pools the tensor slices reference pinned memory directly
        (zero-copy for indptr/indices; values may be cast to float32).
        For unpinned pools, ``torch.from_numpy`` provides the zero-copy view.
        """
        import torch

        use_float = self._dtype_str == "float32"

        if pool.pinned and pool._indptr_t is not None:
            crow = pool._indptr_t[:ncols + 1]
            col_idx = pool._indices_t[:nnz]
            vals = (pool._values_t[:nnz].float() if use_float
                    else pool._values_t[:nnz])
        else:
            crow = torch.from_numpy(pool.indptr[:ncols + 1])
            col_idx = torch.from_numpy(pool.indices[:nnz])
            vals = (torch.from_numpy(pool.values[:nnz]).float() if use_float
                    else torch.from_numpy(pool.values[:nnz]))

        if self._normalize and self._colsums is not None:
            chunk_cs = self._colsums[col_start:col_start + ncols]
            cell_scale = np.divide(
                self._scale, chunk_cs,
                out=np.empty(ncols, dtype=np.float32),
            )
            nnz_per = np.diff(pool.indptr[:ncols + 1])
            el_scale = torch.from_numpy(np.repeat(cell_scale, nnz_per))
            vals = torch.log1p(vals * el_scale)

        return torch.sparse_csr_tensor(
            crow, col_idx, vals,
            size=(ncols, self._nrows),
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        import torch

        t_total = time.monotonic_ns()

        if (self._device != "cpu"
                and torch.cuda.is_available()
                and len(self._pools) == 2):
            yield from self._iter_gpu_prefetch()
        else:
            yield from self._iter_sequential()

        self.timing["total_ns"] += time.monotonic_ns() - t_total

    def _iter_sequential(self):
        """Single-buffer sequential iteration (CPU or fallback)."""
        pool = self._pools[0]
        for ci in range(self._n_chunks):
            cs = ci * self._chunk_size
            ce = min(cs + self._chunk_size, self._ncols)
            nnz, ncols = self._decode_batch(pool, cs, ce)
            csr = self._pool_to_csr(pool, ncols, nnz, col_start=cs)
            self.timing["batches"] += 1
            yield csr

    def _iter_gpu_prefetch(self):
        """Double-buffered GPU iteration with background decode thread.

        A producer thread decompresses the next batch on the CPU while the
        consumer runs the forward / backward pass on the GPU.  Two buffer
        pools alternate so decompression always writes to memory the GPU
        has already finished reading.
        """
        import torch
        from threading import Thread
        from queue import Queue

        q: Queue = Queue(maxsize=2)

        def _produce():
            cur = 0
            for ci in range(self._n_chunks):
                cs = ci * self._chunk_size
                ce = min(cs + self._chunk_size, self._ncols)
                pool = self._pools[cur]

                nnz, ncols = self._decode_batch(pool, cs, ce)
                csr = self._pool_to_csr(pool, ncols, nnz, col_start=cs)

                t0 = time.monotonic_ns()
                gpu_csr = csr.to(self._device)
                self.timing["h2d_ns"] += time.monotonic_ns() - t0

                self.timing["batches"] += 1
                q.put(gpu_csr)
                cur = 1 - cur
            q.put(None)

        thread = Thread(target=_produce, daemon=True)
        thread.start()

        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

        thread.join()

    # ------------------------------------------------------------------
    # Properties and utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_chunks

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nrows, self._ncols)

    @property
    def num_features(self) -> int:
        return self._nrows

    @property
    def num_cells(self) -> int:
        return self._ncols

    def reset_timing(self):
        """Clear timing accumulators (call between epochs)."""
        for k in self.timing:
            self.timing[k] = 0

    def timing_report(self) -> str:
        """Human-readable timing breakdown."""
        t = self.timing
        n = max(t["batches"], 1)
        total_ms = t["total_ns"] / 1e6
        dec_ms = t["decompress_ns"] / 1e6
        alloc_ms = t["alloc_ns"] / 1e6
        h2d_ms = t["h2d_ns"] / 1e6
        other_ms = max(0, total_ms - dec_ms - alloc_ms - h2d_ms)
        return (
            f"PZBufferedLoader · {n} batches · {total_ms:.1f} ms total\n"
            f"  decompress  {dec_ms:8.1f} ms  ({dec_ms / n:6.2f} ms/batch)\n"
            f"  alloc/grow  {alloc_ms:8.1f} ms  ({t['buffer_grows']} grow events)\n"
            f"  H2D         {h2d_ms:8.1f} ms  ({h2d_ms / n:6.2f} ms/batch)\n"
            f"  overhead    {other_ms:8.1f} ms  ({other_ms / n:6.2f} ms/batch)"
        )
