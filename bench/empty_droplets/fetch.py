# SPDX-License-Identifier: MIT
"""Fetch one sample of a legacy ``.singlet`` bundle into a compact gene-level ``.npz`` for the empty-droplet study.

Only the members needed are read, by HTTP Range requests straight out of the remote ZIP64 archive (``zipfile`` over a
seekable range-reading file object), so a 2 GB bundle costs a few tens of MB per sample.

Output ``<out>/<GSE>/<GSM>.npz``:

* ``exon``, ``intron``: genes x barcodes CSC (``*_data``, ``*_indices``, ``*_indptr``, ``*_shape``), rows aggregated from the
  exon/intron features to genes (gene key = feature name before the first ``_``, see ``gene_key``)
* ``genes``: gene keys (union of both members, same row order in both), ``barcodes``, ``called`` (bool, from ``cell_calls.tsv``)
* ``sj_nnz``, ``vdj_nnz``, ``*_bytes``: per-barcode non-zero counts and the members' byte sizes (storage accounting)
* ``meta``: JSON string (summary.json, pileup_stats.json, the cell_calls header and call_method, protocol, reference build)

Usage: ``python -P fetch.py GSE163793 --gsm GSM4987142 --out DIR`` (``--first N`` takes the first N samples of the bundle).
The ``.1pz`` decoder is ``singlet._pz`` built in place in this checkout (``setup.py build_ext --inplace``); it is loaded from
its file so ``import singlet`` (eager, heavy) is not needed.
"""

from __future__ import annotations

import argparse
import glob
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

BASE = os.environ.get("SINGLET_DATA_BASE", "https://data.singlet.bio")
UA = "singlet-emptydrop-study/1.0 (+https://singlet.bio)"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def load_pz():
    """``singlet._pz`` from the in-place build of this checkout, without importing the ``singlet`` package."""
    cands = glob.glob(os.path.join(REPO, "python", "singlet", "_pz*.so")) + glob.glob(
        os.path.join(REPO, "python", "singlet", "_pz*.pyd")
    )
    cands = [c for c in cands if "_pz2" not in os.path.basename(c)]
    if not cands:
        raise ImportError("singlet._pz is not built here (python setup.py build_ext --inplace)")
    loader = importlib.machinery.ExtensionFileLoader("singlet._pz", cands[0])
    spec = importlib.util.spec_from_file_location("singlet._pz", cands[0], loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with an LRU block cache (enough for ``zipfile``)."""

    BLOCK = 1 << 20

    def __init__(self, url: str, cache_blocks: int = 64):
        self.url = url
        self.pos = 0
        self.cache: "OrderedDict[int, bytes]" = OrderedDict()
        self.cache_blocks = cache_blocks
        self.n_requests = 0
        self.n_bytes = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, start: int, end: int) -> bytes:  # inclusive end
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError(f"HTTP {r.status} for a range request")
                    b = r.read()
                if len(b) != end - start + 1:
                    raise IOError(f"short read {len(b)} != {end - start + 1}")
                self.n_requests += 1
                self.n_bytes += len(b)
                return b
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError

    def _block(self, i: int) -> bytes:
        b = self.cache.get(i)
        if b is not None:
            self.cache.move_to_end(i)
            return b
        start = i * self.BLOCK
        b = self._get(start, min(self.size, start + self.BLOCK) - 1)
        self.cache[i] = b
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return b

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > 4 * self.BLOCK:  # big member reads bypass the cache
            b = self._get(self.pos, self.pos + n - 1)
            self.pos += n
            return b
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.BLOCK)
            blk = self._block(i)
            take = min(n, len(blk) - o)
            out += blk[o : o + take]
            self.pos += take
            n -= take
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def gene_key(name: str) -> str:
    """Gene of an exon/intron feature: the name before the first ``_`` (the convention of ``bundle.py``)."""
    return name.split("_", 1)[0]


def read_member_matrix(zf: zipfile.ZipFile, arc: str, pz):
    """Decode one ``.1pz`` member -> (genes x barcodes CSC, rownames, colnames)."""
    data = zf.read(arc)
    with tempfile.NamedTemporaryFile(suffix=".1pz", delete=False) as fh:
        fh.write(data)
        path = fh.name
    try:
        r = pz.read_1pz(path)
    finally:
        os.unlink(path)
    m = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(r["m"], r["n"]))
    return m, list(r["rownames"]), list(r["colnames"]), len(data)


def aggregate_rows(m: sp.csc_matrix, rownames, gene_index: dict) -> sp.csc_matrix:
    rows = np.fromiter((gene_index[gene_key(n)] for n in rownames), dtype=np.int64, count=len(rownames))
    agg = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, np.arange(len(rows)))), shape=(len(gene_index), len(rows)))
    out = (agg @ m.astype(np.float32)).tocsc()
    out.data = np.rint(out.data).astype(np.int32)
    return out


def parse_cell_calls(text: str):
    lines = [l for l in text.splitlines() if l.strip()]
    header = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    bc_col = 0
    for i, h in enumerate(header):
        if h.lower() in ("barcode", "cell_barcode", "cb"):
            bc_col = i
    called = set()
    methods = set()
    is_cell_col = next((i for i, h in enumerate(header) if h.lower() == "is_cell"), None)
    meth_col = next((i for i, h in enumerate(header) if h.lower() == "call_method"), None)
    for r in rows:
        if is_cell_col is not None and r[is_cell_col].strip().upper() not in ("TRUE", "1", "T", "YES"):
            continue
        called.add(r[bc_col])
        if meth_col is not None:
            methods.add(r[meth_col])
    return header, called, sorted(methods)


def fetch_sample(zf: zipfile.ZipFile, gse: str, gsm: str, out_dir: str, pz, study_meta: dict) -> dict:
    t0 = time.time()
    pre = f"samples/{gsm}/"
    names = {n[len(pre):]: n for n in zf.namelist() if n.startswith(pre)}
    if "exon_counts.1pz" not in names and "intron_counts.1pz" not in names:
        raise RuntimeError(f"{gsm}: no count matrix in the bundle")
    meta = {"gse": gse, "gsm": gsm, "members": {k: zf.getinfo(v).file_size for k, v in names.items()}}
    for j in ("summary.json", "pileup_stats.json", "provenance.json"):
        if j in names:
            try:
                meta[j.replace(".json", "")] = json.loads(zf.read(names[j]))
            except Exception as e:  # pragma: no cover - data dependent
                meta[j.replace(".json", "")] = {"_error": str(e)}
    meta["gsm_meta"] = (study_meta.get("gsm_meta") or {}).get(gsm, {})

    mats = {}
    for kind in ("exon", "intron"):
        arc = names.get(f"{kind}_counts.1pz")
        if arc:
            mats[kind] = read_member_matrix(zf, arc, pz)
    cols = None
    for kind, (m, rn, cn, nb) in mats.items():
        if cols is None:
            cols = cn
        elif cn != cols:
            raise RuntimeError(f"{gsm}: exon and intron column names differ")
    genes = sorted({gene_key(n) for (_, rn, _, _) in mats.values() for n in rn})
    gi = {g: i for i, g in enumerate(genes)}
    n_bc = len(cols)
    # feature names are <gene id>_<symbol>[_intron]_<genome>___<chrom>:<start>-<end>: keep symbol and chromosome per gene
    sym, chrom = {}, {}
    for (_, rn, _, _) in mats.values():
        for nme in rn:
            g = gene_key(nme)
            if g not in sym:
                parts = nme.split("_")
                sym[g] = parts[1] if len(parts) > 1 else g
                chrom[g] = nme.rsplit("___", 1)[1].split(":", 1)[0] if "___" in nme else ""
    arrays = {"genes": np.array(genes), "barcodes": np.array(cols), "symbols": np.array([sym[g] for g in genes]),
              "chroms": np.array([chrom[g] for g in genes])}
    for kind in ("exon", "intron"):
        if kind in mats:
            m, rn, cn, nb = mats[kind]
            g = aggregate_rows(m, rn, gi)
            arrays[f"{kind}_nnz_features"] = np.diff(m.indptr).astype(np.int32)  # per-barcode non-zeros at feature level
            meta[f"{kind}_nnz_features"] = int(m.nnz)
            meta[f"{kind}_n_features"] = int(m.shape[0])
        else:
            g = sp.csc_matrix((len(genes), n_bc), dtype=np.int32)
            arrays[f"{kind}_nnz_features"] = np.zeros(n_bc, np.int32)
        arrays[f"{kind}_data"] = g.data
        arrays[f"{kind}_indices"] = g.indices.astype(np.int32)
        arrays[f"{kind}_indptr"] = g.indptr.astype(np.int64)
        meta[f"{kind}_bytes"] = mats[kind][3] if kind in mats else 0

    for kind, member in (("sj", "sj_counts.1pz"), ("vdj", "vdj_gene_usage.1pz")):
        arc = names.get(member)
        if arc:
            try:
                m, rn, cn, nb = read_member_matrix(zf, arc, pz)
                if cn == cols:
                    arrays[f"{kind}_nnz"] = np.diff(m.indptr).astype(np.int32)
                    meta[f"{kind}_bytes"] = nb
                    meta[f"{kind}_nnz"] = int(m.nnz)
            except Exception as e:  # pragma: no cover
                meta[f"{kind}_error"] = str(e)

    called = np.zeros(n_bc, bool)
    if "cell_calls.tsv" in names:
        header, cset, methods = parse_cell_calls(zf.read(names["cell_calls.tsv"]).decode())
        idx = {b: i for i, b in enumerate(cols)}
        missing = [b for b in cset if b not in idx]
        for b in cset:
            if b in idx:
                called[idx[b]] = True
        meta["cell_calls"] = {"header": header, "n_called": len(cset), "n_missing_in_matrix": len(missing), "call_method": methods}
    arrays["called"] = called
    os.makedirs(os.path.join(out_dir, gse), exist_ok=True)
    meta["fetch_seconds"] = round(time.time() - t0, 1)
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(os.path.join(out_dir, gse, f"{gsm}.npz"), **arrays)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gse")
    ap.add_argument("--gsm", nargs="*", default=None)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--explore", action="store_true", help="print member names and feature-name samples, write nothing")
    a = ap.parse_args(argv)
    pz = load_pz()
    rf = RangeFile(f"{BASE}/data/{a.gse}/{a.gse}.singlet")
    zf = zipfile.ZipFile(rf)
    study_meta = json.loads(zf.read("study_meta.json")) if "study_meta.json" in zf.namelist() else {}
    gsms = sorted({n.split("/")[1] for n in zf.namelist() if n.startswith("samples/")}, key=lambda s: int(s[3:]))
    if a.gsm:
        gsms = [g for g in gsms if g in set(a.gsm)]
    if a.first:
        gsms = gsms[: a.first]
    if a.explore:
        g = gsms[0]
        print(a.gse, "samples:", len(gsms), "size", rf.size)
        print([n for n in zf.namelist() if n.startswith(f"samples/{g}/")])
        for kind in ("exon", "intron"):
            arc = f"samples/{g}/{kind}_counts.1pz"
            if arc in zf.namelist():
                m, rn, cn, nb = read_member_matrix(zf, arc, pz)
                print(kind, m.shape, m.nnz, "rows:", rn[:5], rn[len(rn) // 2 : len(rn) // 2 + 3], rn[-3:], "cols:", cn[:3])
                mt = [n for n in rn if n.upper().startswith(("MT-", "GRCH38_MT-", "MT_")) or "MT-" in n.upper()[:12]]
                print(" mt-like rows:", len(mt), mt[:5])
        cc = f"samples/{g}/cell_calls.tsv"
        if cc in zf.namelist():
            print(zf.read(cc).decode()[:400])
        print("requests", rf.n_requests, "bytes", rf.n_bytes)
        return
    for g in gsms:
        try:
            meta = fetch_sample(zf, a.gse, g, a.out, pz, study_meta)
            print(f"{a.gse} {g} ok cols={meta.get('cell_calls', {}).get('n_called')} t={meta['fetch_seconds']}s", flush=True)
        except Exception as e:
            print(f"{a.gse} {g} FAILED {type(e).__name__}: {e}", flush=True)
    print("requests", rf.n_requests, "MB", round(rf.n_bytes / 1e6, 1))


if __name__ == "__main__":
    sys.exit(main())
