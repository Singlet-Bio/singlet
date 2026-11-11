#!/usr/bin/env python3
"""Reader of the all-site member's per-cell delta store (include/singlet/pileup/allsite_store.h).

  from delta_reader import read_delta
  barcodes, sites, rows, cols, ad, dp = read_delta("<out>/private/allsite")

allele_delta.bin: a 64-byte header (magic SGALLEL1, version, cells, sites, donors, entries, index offset), one zlib block per cell
(per entry: row gap, ALT count, REF count as varints), then an index of {offset u64, compressed bytes u32, entries u32} per cell.
cells.tsv.gz names the columns and records each cell's assignment; sites.tsv.gz holds the rows (line i + 1 is row i). Only the entries
that are informative against the cell's donor genotype are stored; the rest are concordant with the donor call (or uncovered).
"""
import gzip, struct, zlib
import numpy as np


def _varints(raw):
    a = np.frombuffer(raw, dtype=np.uint8)
    if a.size == 0:
        return np.zeros(0, dtype=np.uint64)
    ends = np.flatnonzero(a < 128)
    starts = np.concatenate(([0], ends[:-1] + 1))
    grp = np.repeat(np.arange(starts.size), ends - starts + 1)
    sh = ((np.arange(a.size) - starts[grp]) * 7).astype(np.uint64)
    return np.add.reduceat((a & 127).astype(np.uint64) << sh, starts)


def read_delta(d):
    with open(f"{d}/allele_delta.bin", "rb") as f:
        magic, ver, nc, ns, k, ne, ixo = struct.unpack("<8sIIIIQQ", f.read(40))
        assert magic == b"SGALLEL1" and ver == 1
        f.seek(ixo)
        ix = np.frombuffer(f.read(16 * nc), dtype=np.dtype([("off", "<u8"), ("len", "<u4"), ("n", "<u4")]))
        rows, cols, ad, dp = [], [], [], []
        for j in range(nc):
            if ix["n"][j] == 0:
                continue
            f.seek(int(ix["off"][j]))
            v = _varints(zlib.decompress(f.read(int(ix["len"][j])))).reshape(-1, 3)
            r = np.cumsum(v[:, 0]).astype(np.int64)
            rows.append(r); cols.append(np.full(r.size, j, dtype=np.int64)); ad.append(v[:, 1].astype(np.int64)); dp.append((v[:, 1] + v[:, 2]).astype(np.int64))
    cat = lambda x: np.concatenate(x) if x else np.zeros(0, dtype=np.int64)
    with gzip.open(f"{d}/cells.tsv.gz", "rt") as f:
        next(f); barcodes = [l.split("\t")[0] for l in f]
    with gzip.open(f"{d}/sites.tsv.gz", "rt") as f:
        next(f); sites = [tuple(l.split("\t")[:4]) for l in f]
    assert len(barcodes) == nc and len(sites) == ns
    return barcodes, sites, cat(rows), cat(cols), cat(ad), cat(dp)
