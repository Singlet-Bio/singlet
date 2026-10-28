"""Streaming .s1 minibatch loader (PLAN.md gap G1). Port into python/singlet/torch/ once it is settled.

One prefix read opens the object and the matrix member (header, chunk index, entropy tables, row keys). Each minibatch
is a list of cell blocks; for each block the loader plans the chunks that cover it, fetches them in one range request
(adjacent blocks are merged), decodes them on `threads` threads in native row order, and copies the wanted columns
straight into the CSR arrays of a `torch.sparse_csr_tensor` (cells x genes).

With `order="canonical"` the gene axis is the source order (the decoder maps rows through the stored permutation and
sorts each column); with `order="native"` the gene axis is the stored frequency order, and `gene_order` gives the permutation once, so a model
can permute its own input layer instead of the data.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

Fetch = Callable[[int, int], bytes]


class S1Stream:
    def __init__(self, fetch: Fetch, size: Optional[int] = None, member: str = "counts", threads: int = 4,
                 order: str = "native", prefix_hint: int = 1 << 16):
        from singlet import _pz2  # type: ignore
        from singlet.s1.container import S2Reader

        self.rd = S2Reader(fetch, size=size, prefetch=prefix_hint)
        self.mi = self.rd.members[member]
        self._member = member
        self.mem = _pz2.Member(fetch=lambda off, ln: self.rd.read_range(member, off, ln), prefix_hint=prefix_hint)
        self.n_genes, self.n_cells = int(self.mem.m), int(self.mem.n)
        self.threads, self.order = threads, order
        perm = self.mem.row_perm                      # stored row -> canonical row, or None
        self.gene_order = None if perm is None else np.asarray(perm)
        self.fetch_s = 0.0                            # time spent waiting for bytes
        self.decode_s = 0.0                           # time spent in the decoder

    def _block(self, c0: int, c1: int):
        kw = dict(mode=self.order, threads=self.threads, dtype="float32", index_dtype="int32")
        p = self.mem.plan(c0, c1)
        t0 = time.perf_counter()
        buf = self.rd.read_range(self._member, int(p["byte_off"]), int(p["byte_len"]))
        t1 = time.perf_counter()
        self.mem.add_range(int(p["byte_off"]), buf)
        d = self.mem.decode(c0, c1, **kw)
        self.fetch_s += t1 - t0
        self.decode_s += time.perf_counter() - t1
        return d

    @property
    def bytes_fetched(self) -> int:
        return self.rd.bytes_fetched

    @property
    def requests(self) -> int:
        return self.rd.n_requests

    def batch(self, blocks: Iterable[tuple[int, int]]):
        """Columns [c0, c1) of every block, concatenated in order, as a cells x genes torch.sparse_csr_tensor."""
        import torch

        ptrs, idx, val = [np.zeros(1, dtype=np.int64)], [], []
        base = 0
        for c0, c1 in _merge(blocks):
            d = self._block(c0, c1)
            ip = np.asarray(d["indptr"], dtype=np.int64)
            ptrs.append(ip[1:] + base)
            base += int(ip[-1])
            idx.append(np.asarray(d["indices"]))
            val.append(np.asarray(d["data"]))
        crow = np.concatenate(ptrs)
        it = np.int32 if base < 2**31 else np.int64           # torch wants crow and col in the same dtype
        crow = torch.from_numpy(crow.astype(it, copy=False))
        col = torch.from_numpy(np.concatenate(idx).astype(it, copy=False)) if idx else torch.zeros(0, dtype=crow.dtype)
        v = torch.from_numpy(np.concatenate(val)) if val else torch.zeros(0)
        return torch.sparse_csr_tensor(crow, col, v, size=(len(crow) - 1, self.n_genes))


def _merge(blocks: Iterable[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    """Keep the caller's order but join blocks that touch, so adjacent blocks cost one decode call."""
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


def file_fetch(path: str) -> tuple[Fetch, int]:
    import os

    f = open(path, "rb")
    size = os.fstat(f.fileno()).st_size

    def fetch(off: int, ln: int) -> bytes:
        return os.pread(f.fileno(), ln, off)

    return fetch, size


def http_fetch(url: str, session=None) -> tuple[Fetch, int]:
    import requests

    s = session or requests.Session()
    size = int(s.head(url).headers["Content-Length"])

    def fetch(off: int, ln: int) -> bytes:
        r = s.get(url, headers={"Range": f"bytes={off}-{off + ln - 1}"})
        r.raise_for_status()
        return r.content

    return fetch, size
