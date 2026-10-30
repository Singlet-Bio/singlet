# SPDX-License-Identifier: MIT
"""Stream minibatches of cells from one ``.s1`` sample object (a local file or an HTTP URL) into PyTorch.

:class:`S1Stream` opens the object with one prefix read (header, member table, the matrix member's chunk index and entropy
tables). Each minibatch is a list of blocks of contiguous cells; for each run of touching blocks it plans the chunks that hold
those cells, fetches them in one range request, decodes them on ``threads`` threads and writes the columns straight into the
CSR arrays of a ``torch.sparse_csr_tensor`` (cells x genes). Nothing else of the file is read.

Every chunk range is independent once the prefix is read, so a batch's range requests are sent together on ``io_threads``
threads (``"auto"``: 8 for an HTTP URL, where each request waits a network round trip, and 1 for a local file; 1 sends them
one at a time). These threads wait on the network; decoding uses the separate ``threads``.

:class:`S1Cells` is an ``IterableDataset`` over an epoch of such batches. Cells are grouped in blocks of ``block_size``
contiguous cells (the block shuffling of scDataset: random enough for SGD, and every block is one contiguous range of the
file); each epoch shuffles the blocks and deals ``batch_size // block_size`` of them to each batch. With several DataLoader
workers each worker opens the object itself and takes every n-th batch. ``prefetch`` batches are read ahead on a background
thread, so fetching and decoding the next batch overlaps with the work on the current one.

Gene order: ``order="native"`` returns the genes in the stored order (genes sorted by how many cells express them, which the
codec uses), and :attr:`S1Stream.gene_names` / :attr:`S1Stream.gene_order` give that order once, so a model can permute its
input layer instead of every batch. ``order="canonical"`` returns the source gene order at a small decode cost.

Objects written by :func:`singlet.write_s1` (member ``counts``) are read as they are; an atlas sample object is read with
``member="exon_counts.1pz"`` (gene ids then come from its vocabulary pack, which :class:`singlet.torch.AtlasCells` resolves).

Example::

    from torch.utils.data import DataLoader
    from singlet.torch import S1Cells

    ds = S1Cells("https://example.org/pbmc.s1", batch_size=512, block_size=64, threads=4)
    for batch in DataLoader(ds, batch_size=None, num_workers=2):
        x = batch["X"]                    # torch.sparse_csr_tensor, cells x genes
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Optional, Union

import numpy as np

try:  # importable without torch, for the workload helper
    from torch.utils.data import IterableDataset as _Base
except ImportError:  # pragma: no cover
    _Base = object  # type: ignore[assignment,misc]

Fetch = Callable[[int, int], bytes]
Source = Union[str, os.PathLike, Fetch]
Block = tuple[int, int]
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


# --------------------------------------------------------------------------------------------------------- sources
def file_fetch(path: Union[str, os.PathLike]) -> tuple[Fetch, int]:
    """A ``fetch(offset, length)`` over a local file (positional reads, safe from several threads) and its size."""
    f = open(path, "rb")
    size = os.fstat(f.fileno()).st_size
    if hasattr(os, "pread"):

        def fetch(off: int, ln: int) -> bytes:
            return os.pread(f.fileno(), ln, off)

    else:  # pragma: no cover - Windows

        def fetch(off: int, ln: int) -> bytes:
            with open(path, "rb") as g:
                g.seek(off)
                return g.read(ln)

    fetch._file = f  # type: ignore[attr-defined]  # keep the descriptor open as long as the closure lives
    return fetch, size


def http_fetch(url: str, session=None, retries: int = 3) -> tuple[Fetch, int]:
    """A ``fetch(offset, length)`` doing HTTP Range requests on ``url``, and the object's size. Safe from several threads:
    each thread keeps its own connection (unless a ``session`` is given, which is then shared)."""
    import requests

    local = threading.local()

    def sess():
        if session is not None:
            return session
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    def get(headers: dict):
        for attempt in range(retries + 1):
            try:
                r = sess().get(url, headers=headers, timeout=60)
                if r.status_code in (200, 206):
                    return r
                if r.status_code < 500 or attempt == retries:
                    r.raise_for_status()
            except (requests.ConnectionError, requests.Timeout):
                if attempt == retries:
                    raise
            time.sleep(0.5 * 2**attempt)
        raise RuntimeError("unreachable")

    def ranged(r, a: int, b: int) -> Optional[int]:
        """The object size if ``r`` answers bytes a-b with exactly that slice, else None. A Content-Range naming the slice is
        what counts, not the status: some servers (``rclone serve s3``) send the slice with 200 instead of 206, while one
        that ignores Range sends the whole object without Content-Range."""
        m = _CONTENT_RANGE.match(r.headers.get("Content-Range", ""))
        if not m or (int(m.group(1)), int(m.group(2))) != (a, b) or len(r.content) != b - a + 1:
            return None
        return None if m.group(3) == "*" else int(m.group(3))

    size = ranged(get({"Range": "bytes=0-0"}), 0, 0)
    if size is None:
        raise ValueError(f"{url} does not serve byte ranges")

    def fetch(off: int, ln: int) -> bytes:
        r = get({"Range": f"bytes={off}-{off + ln - 1}"})
        if ranged(r, off, off + ln - 1) is None:
            raise IOError(f"short range read of {url} at {off} (+{ln})")
        return r.content

    return fetch, size


def _io_threads(io_threads: Union[int, str], source: Source) -> int:
    """``"auto"``: 8 for an HTTP URL, 1 for a local file or a fetch callable; otherwise the number given (at least 1)."""
    if io_threads == "auto":
        return 8 if not callable(source) and str(source).startswith(("http://", "https://")) else 1
    n = int(io_threads)
    if n < 1:
        raise ValueError("io_threads must be 'auto' or at least 1")
    return n


def _open_source(source: Source) -> tuple[Fetch, Optional[int]]:
    if callable(source):
        return source, None
    s = str(source)
    if s.startswith(("http://", "https://")):
        return http_fetch(s)
    return file_fetch(os.path.expanduser(s))


# --------------------------------------------------------------------------------------------------------- workload
def block_batches(
    n_cells: int,
    batch_size: int = 512,
    block_size: int = 64,
    seed: int = 0,
    n_batches: Optional[int] = None,
    drop_last: bool = False,
) -> list[list[Block]]:
    """Batches of ``batch_size // block_size`` shuffled blocks of ``block_size`` contiguous cells, each as ``(c0, c1)`` ranges
    sorted by position.

    By default one epoch: every cell once, the last block and the last batch possibly short (``drop_last`` drops a short
    last batch). With ``n_batches``, exactly that many batches of whole blocks, cycling through the shuffled blocks if needed
    (the benchmark workload)."""
    if block_size < 1 or batch_size < block_size:
        raise ValueError("need 1 <= block_size <= batch_size")
    per = batch_size // block_size
    rng = np.random.default_rng(seed)
    if n_batches is not None:
        starts = np.arange(0, max(0, n_cells - block_size + 1), block_size)
        if not len(starts):
            return []
        rng.shuffle(starts)
        need = per * int(n_batches)
        starts = np.resize(starts, need) if len(starts) < need else starts[:need]
        return [
            sorted((int(s), int(s) + block_size) for s in starts[i * per : (i + 1) * per])
            for i in range(int(n_batches))
        ]
    starts = np.arange(0, n_cells, block_size)
    rng.shuffle(starts)
    out = []
    for i in range(0, len(starts), per):
        grp = starts[i : i + per]
        if drop_last and len(grp) < per:
            break
        out.append(sorted((int(s), int(min(s + block_size, n_cells))) for s in grp))
    return out


def merge_blocks(blocks: Iterable[Block]) -> Iterator[Block]:
    """Keep the given order but join blocks that touch, so adjacent blocks cost one fetch and one decode call."""
    cur = None
    for a, b in blocks:
        if cur and cur[1] == a:
            cur = (cur[0], b)
        else:
            if cur:
                yield cur
            cur = (a, b)
    if cur:
        yield cur


# --------------------------------------------------------------------------------------------------------- stream
class S1Stream:
    """Range-read access to the cells (columns) of one pz2 member of a ``.s1`` object.

    Parameters
    ----------
    source
        Path, ``http(s)://`` URL, or a ``fetch(offset, length) -> bytes`` callable.
    member
        Matrix member to read (``"counts"`` for :func:`singlet.write_s1` objects).
    threads
        Decoder threads per request.
    order
        ``"native"`` (stored gene order, see :attr:`gene_order`) or ``"canonical"`` (source gene order).
    size, prefix_hint
        Object size if known, and the bytes read by the first request (header, member table, matrix prefix).
    io_threads
        Range requests of one batch sent together: ``"auto"`` (8 for an HTTP URL, 1 for a local file or a callable) or a
        number; 1 sends them one at a time. A callable ``source`` must be safe to call from that many threads.
    """

    def __init__(
        self,
        source: Source,
        *,
        member: str = "counts",
        threads: int = 4,
        order: str = "native",
        size: Optional[int] = None,
        prefix_hint: int = 1 << 16,
        io_threads: Union[int, str] = "auto",
    ):
        from singlet import _pz2  # type: ignore
        from singlet.s1.container import S2Reader

        if order not in ("native", "canonical"):
            raise ValueError("order must be 'native' or 'canonical'")
        fetch, known = _open_source(source)
        self.reader = S2Reader(
            fetch, size=size if size is not None else known, prefetch=prefix_hint
        )
        if member not in self.reader.members:
            raise KeyError(f"no member {member!r} (members: {', '.join(self.reader.members)})")
        self.member, self.threads, self.order = member, int(threads), order
        self.io_threads = _io_threads(io_threads, source)
        self._pool = None  # a ThreadPoolExecutor of io_threads workers, made on the first batch that needs it
        self.mem = _pz2.Member(
            fetch=lambda off, ln: self.reader.read_range(member, off, ln), prefix_hint=prefix_hint
        )
        self.n_genes, self.n_cells = int(self.mem.m), int(self.mem.n)
        perm = self.mem.row_perm
        #: stored row -> source (canonical) row, or None when the two orders agree
        self.gene_order: Optional[np.ndarray] = (
            None if perm is None else np.asarray(perm, dtype=np.int64)
        )
        self.fetch_s = 0.0  # seconds waiting for bytes
        self.decode_s = 0.0  # seconds in the decoder
        self._genes: Optional[list] = None
        self._barcodes: Optional[list] = None

    # ------------------------------------------------------------------ names
    def _names(self, rows: bool) -> Optional[list]:
        d = self.mem.row_names() if rows else self.mem.col_names()
        names = d.get("names") if isinstance(d, dict) else d
        return None if names is None else [str(s) for s in names]

    @property
    def canonical_gene_names(self) -> Optional[list]:
        """Gene ids in the source order (None when the member refers to a vocabulary pack)."""
        if self._genes is None:
            self._genes = self._names(True)
        return self._genes

    @property
    def gene_names(self) -> Optional[list]:
        """Gene ids in the order of the batches' columns (:attr:`order`)."""
        g = self.canonical_gene_names
        if g is None or self.order == "canonical" or self.gene_order is None:
            return g
        return [g[i] for i in self.gene_order]

    @property
    def barcodes(self) -> Optional[list]:
        """Cell barcodes in column order (read on first use)."""
        if self._barcodes is None:
            self._barcodes = self._names(False)
        return self._barcodes

    # ------------------------------------------------------------------ io counters
    @property
    def bytes_fetched(self) -> int:
        return self.reader.bytes_fetched

    @property
    def requests(self) -> int:
        return self.reader.n_requests

    # ------------------------------------------------------------------ reading
    def _fetch_ranges(self, ranges: list[Block]) -> None:
        """Fetch the member bytes of every cell range, all requests at once on ``io_threads`` threads, and register them."""
        want = {}
        for c0, c1 in ranges:
            p = self.mem.plan(c0, c1)
            off, ln = int(p["byte_off"]), int(p["byte_len"])
            if ln and not self.mem.has(off, ln):
                want[(off, ln)] = None
        t0 = time.perf_counter()
        keys = list(want)
        if self.io_threads > 1 and len(keys) > 1:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor

                self._pool = ThreadPoolExecutor(self.io_threads, thread_name_prefix="s1-io")
            bufs = list(
                self._pool.map(lambda k: self.reader.read_range(self.member, k[0], k[1]), keys)
            )
        else:
            bufs = [self.reader.read_range(self.member, off, ln) for off, ln in keys]
        self.fetch_s += time.perf_counter() - t0
        for (off, _), buf in zip(keys, bufs):
            self.mem.add_range(off, buf)

    def _decode(self, c0: int, c1: int) -> dict:
        t0 = time.perf_counter()
        d = self.mem.decode(
            c0, c1, mode=self.order, threads=self.threads, dtype="float32", index_dtype="int32"
        )
        self.decode_s += time.perf_counter() - t0
        return d

    def close(self) -> None:
        """Stop the request threads (the stream can still be used; they are made again when needed)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def batch_arrays(self, blocks: Iterable[Block]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CSR ``(indptr int64, indices int32, data float32)`` of the cells in ``blocks``, in the given order."""
        ptrs, idx, val = [np.zeros(1, dtype=np.int64)], [], []
        base = 0
        ranges = []
        for c0, c1 in merge_blocks(blocks):
            if not 0 <= c0 <= c1 <= self.n_cells:
                raise IndexError(f"cells [{c0}, {c1}) outside [0, {self.n_cells})")
            if c0 < c1:
                ranges.append((int(c0), int(c1)))
        self._fetch_ranges(ranges)
        for c0, c1 in ranges:
            d = self._decode(c0, c1)
            ip = np.asarray(d["indptr"], dtype=np.int64)
            ptrs.append(ip[1:] + base)
            base += int(ip[-1])
            idx.append(np.asarray(d["indices"]))
            val.append(np.asarray(d["data"], dtype=np.float32))
        indptr = np.concatenate(ptrs)
        indices = np.concatenate(idx) if idx else np.zeros(0, dtype=np.int32)
        data = np.concatenate(val) if val else np.zeros(0, dtype=np.float32)
        return indptr, indices, data

    def batch(self, blocks: Iterable[Block]):
        """The cells of ``blocks`` (``(c0, c1)`` ranges, concatenated in order) as a cells x genes ``torch.sparse_csr_tensor``."""
        import torch

        indptr, indices, data = self.batch_arrays(list(blocks))
        it = (
            np.int32 if indptr[-1] < 2**31 else np.int64
        )  # torch wants crow and col indices in one dtype
        crow = torch.from_numpy(indptr.astype(it, copy=False))
        col = torch.from_numpy(indices.astype(it, copy=False))
        return torch.sparse_csr_tensor(
            crow,
            col,
            torch.from_numpy(data),
            size=(len(indptr) - 1, self.n_genes),
            check_invariants=False,
        )

    def cells(self, c0: int, c1: int):
        """Cells ``[c0, c1)`` as a scipy CSR matrix (cells x genes, float32)."""
        import scipy.sparse as sp

        indptr, indices, data = self.batch_arrays([(c0, c1)])
        return sp.csr_matrix((data, indices, indptr), shape=(c1 - c0, self.n_genes))


# --------------------------------------------------------------------------------------------------------- dataset
class S1Cells(_Base):
    """``IterableDataset`` of block-shuffled minibatches from one ``.s1`` object; use with ``DataLoader(ds, batch_size=None)``.

    Each item is ``{"X": torch.sparse_csr_tensor (cells x genes, float32), "index": torch.LongTensor of cell (column) ids}``.
    Call :meth:`set_epoch` before each epoch for a new block order (the order is a function of ``seed`` and the epoch).
    ``io_threads`` is passed to :class:`S1Stream`; ``prefetch`` batches (default 1, 0 for none) are read ahead on a background
    thread of each worker.
    """

    def __init__(
        self,
        source: Union[str, os.PathLike],
        *,
        batch_size: int = 512,
        block_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        member: str = "counts",
        threads: int = 4,
        order: str = "native",
        io_threads: Union[int, str] = "auto",
        prefetch: int = 1,
    ):
        self.source = str(source)
        self.batch_size, self.block_size = int(batch_size), int(block_size)
        self.shuffle, self.seed, self.drop_last = shuffle, int(seed), drop_last
        self.member, self.threads, self.order = member, int(threads), order
        self.io_threads, self.prefetch = io_threads, int(prefetch)
        if self.prefetch < 0:
            raise ValueError("prefetch must be 0 or more")
        self.epoch = 0
        self._stream: Optional[S1Stream] = None
        info = self.stream  # open once here for the shape and names; workers reopen after pickling
        self.n_cells, self.n_genes = info.n_cells, info.n_genes

    @property
    def stream(self) -> S1Stream:
        if self._stream is None:
            self._stream = S1Stream(
                self.source,
                member=self.member,
                threads=self.threads,
                order=self.order,
                io_threads=self.io_threads,
            )
        return self._stream

    @property
    def gene_names(self) -> Optional[list]:
        return self.stream.gene_names

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_stream"] = None  # file handles and HTTP sessions do not cross process boundaries
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def batches(self) -> list[list[Block]]:
        """This epoch's batches (before splitting between workers)."""
        if self.shuffle:
            seed = int(np.random.SeedSequence([self.seed, self.epoch]).generate_state(1)[0])
            return block_batches(
                self.n_cells, self.batch_size, self.block_size, seed=seed, drop_last=self.drop_last
            )
        starts = range(0, self.n_cells, self.block_size)
        blocks = [(s, min(s + self.block_size, self.n_cells)) for s in starts]
        per = self.batch_size // self.block_size
        out = [blocks[i : i + per] for i in range(0, len(blocks), per)]
        return out[:-1] if self.drop_last and out and len(out[-1]) < per else out

    def __len__(self) -> int:
        return len(self.batches())

    def __iter__(self) -> Iterator[dict]:
        import torch

        mine = self.batches()
        wi = torch.utils.data.get_worker_info()
        if wi is not None and wi.num_workers > 1:
            mine = mine[wi.id :: wi.num_workers]
        s = self.stream

        def item(blocks):
            idx = (
                np.concatenate([np.arange(a, b) for a, b in blocks])
                if blocks
                else np.zeros(0, np.int64)
            )
            return {"X": s.batch(blocks), "index": torch.from_numpy(idx.astype(np.int64))}

        if self.prefetch == 0:
            for blocks in mine:
                yield item(blocks)
            return
        yield from _read_ahead(item, mine, self.prefetch)


def _read_ahead(make: Callable, todo: list, depth: int) -> Iterator:
    """``make(x)`` for each x of ``todo``, in order, computed up to ``depth`` items ahead on one background thread. An error
    in ``make`` is raised to the consumer; closing the generator early stops the thread."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def run():
        try:
            for x in todo:
                if stop.is_set():
                    return
                q.put((True, make(x)))
            q.put((True, done))
        except BaseException as e:  # handed to the consumer
            q.put((False, e))

    th = threading.Thread(target=run, name="s1-prefetch", daemon=True)
    th.start()
    try:
        while True:
            ok, v = q.get()
            if not ok:
                raise v
            if v is done:
                return
            yield v
    finally:
        stop.set()
        while th.is_alive():  # unblock a producer waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                th.join(timeout=0.05)
