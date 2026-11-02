"""Stage `access` (review round 3): what one random minibatch costs a reader that fetches by range, from each format's own
index, with nothing timed: the bytes it must fetch, the separate byte ranges (requests, adjacent ranges merged), and the
bytes of the index it reads first. The workload is the minibatch stage's (seed 0, `access.n_batches` batches of
`minibatch.batch_size` cells in blocks of `minibatch.block_size`).

  s1        the member's chunk index (the ranges S1Stream plans)
  h5ad_*    the HDF5 chunk index of X/data, X/indices and X/indptr (storage size of every chunk touched)
  tenx_h5   the same for matrix/{data,indices,indptr}
  zarr      AnnData Zarr: the chunk objects of X/data, X/indices and X/indptr
`needed` is the batch's non-zeros times the file's bytes per non-zero: what a reader with a per-cell index would fetch.

The chunk sweep (`access.chunk_sweep`, on `access.sweep_datasets`) re-encodes the matrix member at each chunk budget and
reports its size, the same costs, and the codec's in-memory decode rate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .util import Config, Unit, load_canon, log, read_tsv, result_path, write_tsv
from .write import format_dir


def _runs(blocks):
    cur = None
    for a, b in sorted(blocks):
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            if cur:
                yield cur
            cur = (a, b)
    if cur:
        yield cur


def _merge_spans(spans) -> tuple[int, int]:
    """Total bytes and number of separate ranges once touching ranges are merged."""
    spans = sorted(s for s in spans if s[1] > 0)
    n, total, end = 0, 0, -1
    for off, ln in spans:
        if off > end:
            n += 1
        total += ln
        end = max(end, off + ln)
    return total, n


class _Chunked:
    """1-d chunked arrays: chunk size and stored (offset, size) of chunk k. HDF5 or Zarr v2 directory."""

    def __init__(self, chunks: int, info, itemsize: int, contiguous_off: int = -1):
        self.chunks, self.info, self.itemsize, self.off = chunks, info, itemsize, contiguous_off

    def spans(self, s: int, e: int):
        if e <= s:
            return []
        if not self.chunks:   # contiguous: exactly the elements
            return [(self.off + s * self.itemsize, (e - s) * self.itemsize)]
        return [self.info(k) for k in range(s // self.chunks, (e - 1) // self.chunks + 1)]


def _h5(ds, file_key: str) -> _Chunked:
    if ds.chunks is None:
        return _Chunked(0, None, ds.dtype.itemsize, ds.id.get_offset() or 0)
    c = ds.chunks[0]
    cache = {}

    def info(k):
        if k not in cache:
            si = ds.id.get_chunk_info_by_coord((k * c,))
            cache[k] = (int(si.byte_offset), int(si.size))
        return cache[k]

    return _Chunked(c, info, ds.dtype.itemsize)


def _zarr2(d: Path, base_off: int) -> _Chunked:
    meta = json.loads((d / ".zarray").read_text())
    c = int(meta["chunks"][0])
    # each chunk is its own object; offsets are only used to keep distinct objects apart when merging ranges
    return _Chunked(c, lambda k: (base_off + k * (1 << 40), (d / str(k)).stat().st_size if (d / str(k)).exists() else 0), 0)


def _csr_costs(indptr: np.ndarray, arrays: list[_Chunked], ip: _Chunked, batches) -> list[tuple[int, int, int]]:
    out = []
    for blocks in batches:
        spans, nnz = [], 0
        for a, b in _runs(blocks):
            s, e = int(indptr[a]), int(indptr[b])
            nnz += e - s
            spans += ip.spans(a, b + 1)
            for arr in arrays:
                spans += arr.spans(s, e)
        tot, n = _merge_spans(spans)
        out.append((tot, n, nnz))
    return out


def _s1_costs(mem, batches) -> list[tuple[int, int, int]]:
    out = []
    for blocks in batches:
        spans, nnz = [], 0
        for a, b in _runs(blocks):
            p = mem.plan(a, b)
            spans.append((int(p["byte_off"]), int(p["byte_len"])))
            nnz += int(p.get("nnz", 0)) if hasattr(p, "get") else 0
        tot, n = _merge_spans(spans)
        out.append((tot, n, nnz))
    return out


def _row(u: Unit, fmt: str, costs, file_bytes: int, nnz_total: int, index_bytes, **kw) -> dict:
    c = np.asarray(costs, dtype=float)
    per_nnz = file_bytes / max(1, nnz_total)
    need = c[:, 2] * per_nnz
    return {"dataset": u.id, "matrix": u.name, "format": fmt, "batches": len(costs), **kw,
            "bytes_per_batch": float(np.median(c[:, 0])), "requests_per_batch": float(np.median(c[:, 1])),
            "requests_p90": float(np.quantile(c[:, 1], 0.9)), "needed_per_batch": float(np.median(need)),
            "overread": float(np.median(c[:, 0] / np.maximum(need, 1))), "index_bytes": index_bytes, "file_bytes": file_bytes}


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    from . import formats
    from .minibatch import workload

    ac, mb = cfg["access"], cfg["minibatch"]
    for u in units:
        out = result_path(cfg, "access", u)
        if out.exists() and not force:
            continue
        fdir = format_dir(cfg, u)
        # formats written and present here (a big dataset's timed run stages only some of them)
        written = {r["format"]: r for r in read_tsv(result_path(cfg, "write", u))
                   if r.get("status") == "ok" and r["format"] in formats.registry() and formats.get([r["format"]])[0].path(fdir).exists()}
        if "s1" not in written:
            continue
        nnz = int(written["s1"]["nnz"])
        n_cells = int(written["s1"]["cells"])
        batches = workload(n_cells, int(mb["batch_size"]), int(mb["block_size"]), int(ac["n_batches"]), 0)
        rows = []
        # .s1: the member's own plan, and the batch's non-zeros from the canonical indptr
        from singlet import _pz2  # type: ignore
        from singlet.s1.container import S2Reader

        p = formats.get(["s1"])[0].path(fdir)
        mem = _pz2.Member(S2Reader(str(p)).read_member("counts"))
        cidx = load_canon(cfg, u).indptr
        costs = [(b, r, sum(int(cidx[y]) - int(cidx[x]) for x, y in _runs(bl))) for (b, r, _), bl in zip(_s1_costs(mem, batches), batches)]
        rows.append(_row(u, "s1", costs, int(written["s1"]["bytes"]), nnz, int(mem.prefix_len)))
        import h5py

        for fid, grp in (("h5ad_gzip", "X"), ("h5ad_lzf", "X"), ("h5ad_none", "X"), ("tenx_h5", "matrix")):
            if fid not in written:
                continue
            with h5py.File(formats.get([fid])[0].path(fdir), "r") as f:
                g = f[grp]
                indptr = g["indptr"][:]
                if len(indptr) != n_cells + 1:
                    log(f"{u.key} {fid}: indptr is not over cells; skipped")
                    continue
                costs = _csr_costs(indptr, [_h5(g["data"], fid), _h5(g["indices"], fid)], _h5(g["indptr"], fid), batches)
                rows.append(_row(u, fid, costs, int(written[fid]["bytes"]), nnz, int(g["indptr"].id.get_storage_size()),
                                 chunk_elems=g["data"].chunks[0] if g["data"].chunks else 0))
        if "zarr" in written:
            x = formats.get(["zarr"])[0].path(fdir) / "X"
            import zarr

            indptr = zarr.open(str(x), mode="r")["indptr"][:]
            arrs = [_zarr2(x / "data", 0), _zarr2(x / "indices", 1 << 60)]
            ipz = _zarr2(x / "indptr", 2 << 60)
            costs = _csr_costs(indptr, arrs, ipz, batches)
            ip_bytes = sum(q.stat().st_size for q in (x / "indptr").iterdir() if q.name[0] != ".")
            rows.append(_row(u, "zarr", costs, int(written["zarr"]["bytes"]), nnz, ip_bytes,
                             chunk_elems=json.loads((x / "data" / ".zarray").read_text())["chunks"][0]))
        if u.id in ac.get("sweep_datasets", []):
            rows += _chunk_sweep(cfg, u, batches, cidx)
        for r in rows:
            log(f"{u.key} {r['format']} {r.get('chunk_nnz_max', '')}: {r['bytes_per_batch'] / 1024:.0f} KiB, "
                f"{r['requests_per_batch']:.0f} requests, over-read {r['overread']:.2f}")
        write_tsv(rows, out)


def _chunk_sweep(cfg: Config, u: Unit, batches, cidx) -> list[dict]:
    from singlet import _pz2  # type: ignore

    c = load_canon(cfg, u)
    H = int(cfg["threads"]["headline"])
    rows = []
    for budget in cfg["access"]["chunk_sweep"]:
        t0 = time.perf_counter()
        body, _st = _pz2.encode(c.indptr, c.indices, c.data, c.m, c.n, chunk_nnz_max=int(budget), threads=H,
                                row_names=[str(g) for g in c.genes], col_names=[str(b) for b in c.barcodes])
        enc_s = time.perf_counter() - t0
        mem = _pz2.Member(body)
        costs = [(b, r, sum(int(cidx[y]) - int(cidx[x]) for x, y in _runs(bl))) for (b, r, _), bl in zip(_s1_costs(mem, batches), batches)]
        ts = []
        for k in range(4):   # one warm-up, three timed: the codec on the member in memory, native order
            t0 = time.perf_counter()
            _pz2.decode(body, mode="native", threads=H)
            if k:
                ts.append(time.perf_counter() - t0)
        rows.append(_row(u, "s1_chunk", costs, len(body), c.nnz, int(mem.prefix_len), chunk_nnz_max=int(budget),
                         bits_per_nnz=8 * len(body) / c.nnz, decode_mnnz_s=c.nnz / float(np.median(ts)) / 1e6, encode_s=enc_s))
    return rows
