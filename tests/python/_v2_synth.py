# SPDX-License-Identifier: MIT
"""Build a small synthetic v2 study on disk (server layout) with the real codec, real vocabulary packs and W2's container writer,
so the client can be tested end to end without a corpus: three samples (normal / no called cells / no intron matrix).

The matrices have the real 310,797 (exon) and 272,191 (intron) rows of the shipped GRCh38 packs, so the stored-row -> gene map comes
from the pack exactly as in production; the expected gene-level result is computed independently from the row *names*
(``name.split("_")[0]``, the legacy rule)."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
from singlet import _pz2
from singlet.v2 import cells as cells_mod
from singlet.v2 import container as K
from singlet.v2 import manifest as M
from singlet.v2 import util
from singlet.v2.packs import default_store

BUILD = "GRCh38-2024-A"
GSE = "GSE900001"


class Synth:
    def __init__(self, root: str):
        self.root = root
        self.exon_pack = default_store().vocab_packs("exon", BUILD)[0]
        self.intron_pack = default_store().vocab_packs("intron", BUILD)[0]
        self.exon_names = self.exon_pack.names()
        self.intron_names = self.intron_pack.names()
        self.truth: Dict[str, dict] = {}
        self.manifest: dict = {}

    # ---- helpers
    @staticmethod
    def _barcodes(prefix: str, n: int) -> List[str]:
        alpha = "ACGT"
        out = []
        for i in range(n):
            x, s = i * 2654435761 % (4**14), []
            for _ in range(14):
                s.append(alpha[x & 3])
                x >>= 2
            out.append(prefix + "".join(s))
        return out

    def _matrix(self, m: int, n: int, seed: int, row_pack, barcodes: List[str], col_density: np.ndarray, chunk_cols: int = 64):
        rng = np.random.default_rng(seed)
        indptr = np.zeros(n + 1, dtype=np.int64)
        cols_idx, cols_val = [], []
        for j in range(n):
            k = int(col_density[j])
            ids = np.sort(rng.choice(m, size=k, replace=False)).astype(np.uint32) if k else np.zeros(0, np.uint32)
            cols_idx.append(ids)
            cols_val.append(rng.integers(1, 20, k).astype(np.uint32))
            indptr[j + 1] = indptr[j] + k
        indices = np.concatenate(cols_idx) if cols_idx else np.zeros(0, np.uint32)
        data = np.concatenate(cols_val) if cols_val else np.zeros(0, np.uint32)
        names = row_pack.names()
        blob, stats = _pz2.encode(
            indptr, indices, data, m, n, codec="rans", chunk_cols=chunk_cols,
            row_names={"kind": "pack", "pack_id": row_pack.id[:8], "names": names},
            col_names={"kind": "cells", "names": barcodes},
        )
        csc = sp.csc_matrix((data.astype(np.int64), indices.astype(np.int64), indptr), shape=(m, n))
        return blob, csc

    def _gene_level(self, csc: sp.csc_matrix, names: List[str], gene_ids: List[str]) -> sp.csc_matrix:
        gidx = {g: i for i, g in enumerate(gene_ids)}
        f2g = np.array([gidx.get(nm.split("_")[0], -1) for nm in names], dtype=np.int64)
        keep = f2g >= 0
        ind = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.int64), (f2g[keep], np.flatnonzero(keep))), shape=(len(gene_ids), len(names)))
        out = (ind @ csc).tocsc()
        out.eliminate_zeros()
        out.sort_indices()
        return out

    # ---- build
    def build(self, n_cols: int = 300, n_called: int = 40, chunk_cols: int = 64, pad_between: int = 0) -> "Synth":
        gene_tbl = self.exon_pack.gene_table()
        gene_ids = [g for g, _ in gene_tbl]
        specs = [
            ("GSM9000001", n_cols, n_called, True, True, 11),
            ("GSM9000002", 120, 0, True, True, 12),      # no called cell
            ("GSM9000003", 200, 25, True, False, 13),     # no intron matrix
        ]
        samples = []
        for gsm, n, ncalled, has_exon, has_intron, seed in specs:
            rng = np.random.default_rng(seed)
            bcs = self._barcodes("A", n)
            # cells sorted by library size, like the pipeline's column order; the called cells are among the largest, not a prefix
            dens = np.sort(rng.integers(5, 400, n))[::-1]
            called = np.zeros(n, bool)
            if ncalled:
                called[np.sort(rng.choice(max(ncalled * 3, ncalled), size=ncalled, replace=False))] = True
            w = K.S2Writer(gsm)
            ex_blob, ex_csc = self._matrix(len(self.exon_names), n, seed, self.exon_pack, bcs, dens, chunk_cols)
            in_blob = in_csc = None
            if has_intron:
                in_blob, in_csc = self._matrix(len(self.intron_names), n, seed + 100, self.intron_pack, bcs, dens // 2, chunk_cols)
            umi_e = np.asarray(ex_csc.sum(axis=0)).ravel().astype(np.uint32)
            ct = cells_mod.CellsTable(n)
            ct.add("barcode", cells_mod.DT_STR, bcs)
            ct.add("called", cells_mod.DT_BOOL, called)
            ct.add("n_umi_exon", cells_mod.DT_U32, umi_e)
            ct.add("n_umi_intron", cells_mod.DT_U32, np.asarray(in_csc.sum(axis=0)).ravel().astype(np.uint32) if has_intron else np.zeros(n, np.uint32))
            ct.add("n_feat_exon", cells_mod.DT_U32, np.diff(ex_csc.indptr).astype(np.uint32))
            ct.add("n_feat_intron", cells_mod.DT_U32, np.diff(in_csc.indptr).astype(np.uint32) if has_intron else np.zeros(n, np.uint32))
            w.add_cells("cells", ct.encode(), ct.digest(), {"n_rows": n}, flags=K.FLAG_ANNDATA)
            w.add_matrix("exon_counts.1pz", ex_blob, bytes.fromhex(_pz2.read_header(ex_blob)["digest"]),
                         {"m": len(self.exon_names), "n": n}, K.FLAG_LEGACY | K.FLAG_ANNDATA)
            if has_intron:
                w.add_matrix("intron_counts.1pz", in_blob, bytes.fromhex(_pz2.read_header(in_blob)["digest"]),
                             {"m": len(self.intron_names), "n": n}, K.FLAG_LEGACY | K.FLAG_ANNDATA)
            summary = json.dumps({"n_cells_called": int(called.sum()), "gsm": gsm}).encode()
            w.add_stored("summary.json", summary, {}, K.FLAG_LEGACY)
            w.add_stored("cell_calls.tsv", ("barcode\tis_cell\n" + "".join("%s\tTRUE\n" % b for b, c in zip(bcs, called) if c)).encode(), {}, K.FLAG_LEGACY)
            obj = w.build()
            d16 = util.d16(obj).hex()
            key = "v2/s/%s/%s.s2" % (gsm, d16)
            path = os.path.join(self.root, *key.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(obj)
            members = {}
            for info in w.infos:
                d = {"kind": info.kind, "flags": info.flags, "off": info.off, "len": info.len, "digest": info.digest.hex()}
                if info.kind == K.KIND_PZ2:
                    h = _pz2.read_header(obj[info.off:info.off + 256])
                    d.update({"m": h["m"], "n": h["n"], "nnz": h["nnz"], "codec": h["codec"], "chunk_cols": h["chunk_cols"], "n_chunks": h["n_chunks"]})
                members[info.name] = d
            samples.append(M.sample_entry(gsm, key, obj, n, int(called.sum()), members, {}))
            self.truth[gsm] = {
                "barcodes": bcs, "called": called, "exon": ex_csc, "intron": in_csc, "gene_ids": gene_ids,
                "exon_g": self._gene_level(ex_csc, self.exon_names, gene_ids),
                "intron_g": self._gene_level(in_csc, self.intron_names, gene_ids) if has_intron else None,
                "umi_exon": umi_e, "object": obj, "key": key,
            }
        meta = {"schema_version": "1.0", "gse_id": GSE, "series_title": "synthetic", "gsm_meta": {
            s["gsm"]: {"gsm_id": s["gsm"], "organism": "Homo sapiens", "protocol": "10xv3", "protocol_name": "p", "sample_source": "src",
                       "sample_characteristics": "x", "qc_flag": "ok", "reference_build": BUILD, "n_cells": s["n_called"]} for s in samples}}
        sfiles = {"study_meta.json": M.study_file_entry(json.dumps(meta, indent=2).encode())}
        man = M.build_manifest(GSE, samples, sfiles, BUILD, {"singlet_version": "test"},
                               {"exon": self.exon_pack.id.hex(), "intron": self.intron_pack.id.hex(), "genes": None, "whitelist": None},
                               {"key": "data/%s/%s.singlet" % (GSE, GSE)}, "2026-10-22T00:00:00Z")
        mpath = os.path.join(self.root, "v2", "m", GSE + ".json")
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        with open(mpath, "wb") as fh:
            fh.write(M.dumps_manifest(man))
        self.manifest = man
        self._write_ref()
        return self

    def _write_ref(self):
        """v2/ref/<id>.pack + index.json as the server lays them out (clients without the wheel packs fetch them from here)."""
        from singlet.v2 import packs as P

        ref = os.path.join(self.root, "v2", "ref")
        os.makedirs(ref, exist_ok=True)
        entries = []
        for pk in (self.exon_pack, self.intron_pack):
            data = pk.bytes
            with open(os.path.join(ref, pk.id.hex() + ".pack"), "wb") as fh:
                fh.write(data)
            entries.append(P.index_entry(data, "vocab", {"build": pk.build, "member": pk.member}))
        P.write_index(entries, os.path.join(ref, "index.json"))


# ---------------------------------------------------------------------------------------------------------------------------------
# command line: build the synthetic study and write the expected gene-level matrices for other languages' tests (the R tests)
#     python -P tests/python/_v2_synth.py <out_dir>
# writes <out_dir>/v2/... (the study) and, per sample with called cells, truth_<GSM>.mtx (genes x cells, counts), truth_<GSM>_spliced.mtx,
# truth_<GSM>_unspliced.mtx (when the sample has an intron matrix), truth_<GSM>_barcodes.txt (cells in sorted-barcode order) and genes.txt.
if __name__ == "__main__":
    import sys

    import scipy.io

    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    s = Synth(out).build()
    with open(os.path.join(out, "genes.txt"), "w") as fh:
        fh.write("\n".join("%s\t%s" % (g, n) for g, n in s.exon_pack.gene_table()) + "\n")
    for gsm, t in s.truth.items():
        called = np.flatnonzero(t["called"])
        if not len(called):
            continue
        bcs = np.array(t["barcodes"], dtype=object)[called]
        order = np.argsort(bcs, kind="stable")
        cols = called[order]
        ex = t["exon_g"][:, cols]
        un = t["intron_g"][:, cols] if t["intron_g"] is not None else None
        scipy.io.mmwrite(os.path.join(out, "truth_%s_spliced.mtx" % gsm), ex)
        if un is not None:
            scipy.io.mmwrite(os.path.join(out, "truth_%s_unspliced.mtx" % gsm), un)
        scipy.io.mmwrite(os.path.join(out, "truth_%s.mtx" % gsm), ex + un if un is not None else ex)
        with open(os.path.join(out, "truth_%s_barcodes.txt" % gsm), "w") as fh:
            fh.write("\n".join(bcs[order]) + "\n")
    print(out)
