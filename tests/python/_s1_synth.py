# SPDX-License-Identifier: MIT
"""Build a small synthetic per-sample study on disk (server layout) with the real codec, real vocabulary packs and W2's container writer,
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
from singlet.s1 import cells as cells_mod
from singlet.s1 import container as K
from singlet.s1 import manifest as M
from singlet.s1 import util
from singlet.s1.packs import default_store

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

    def _matrix(
        self,
        m: int,
        n: int,
        seed: int,
        row_pack,
        barcodes: List[str],
        col_density: np.ndarray,
        chunk_cols: int = 64,
    ):
        rng = np.random.default_rng(seed)
        indptr = np.zeros(n + 1, dtype=np.int64)
        cols_idx, cols_val = [], []
        for j in range(n):
            k = int(col_density[j])
            ids = (
                np.sort(rng.choice(m, size=k, replace=False)).astype(np.uint32)
                if k
                else np.zeros(0, np.uint32)
            )
            cols_idx.append(ids)
            cols_val.append(rng.integers(1, 20, k).astype(np.uint32))
            indptr[j + 1] = indptr[j] + k
        indices = np.concatenate(cols_idx) if cols_idx else np.zeros(0, np.uint32)
        data = np.concatenate(cols_val) if cols_val else np.zeros(0, np.uint32)
        inline = isinstance(
            row_pack, list
        )  # a list of names: inline row names (junctions, V(D)J segments)
        names = row_pack if inline else row_pack.names()
        blob, stats = _pz2.encode(
            indptr,
            indices,
            data,
            m,
            n,
            codec="rans",
            chunk_cols=chunk_cols,
            row_names=names
            if inline
            else {"kind": "pack", "pack_id": row_pack.id[:8], "names": names},
            col_names={"kind": "cells", "names": barcodes},
        )
        csc = sp.csc_matrix((data.astype(np.int64), indices.astype(np.int64), indptr), shape=(m, n))
        return blob, csc

    def _gene_level(
        self, csc: sp.csc_matrix, names: List[str], gene_ids: List[str]
    ) -> sp.csc_matrix:
        gidx = {g: i for i, g in enumerate(gene_ids)}
        f2g = np.array([gidx.get(nm.split("_")[0], -1) for nm in names], dtype=np.int64)
        keep = f2g >= 0
        ind = sp.csr_matrix(
            (np.ones(int(keep.sum()), dtype=np.int64), (f2g[keep], np.flatnonzero(keep))),
            shape=(len(gene_ids), len(names)),
        )
        out = (ind @ csc).tocsc()
        out.eliminate_zeros()
        out.sort_indices()
        return out

    # ---- build
    @staticmethod
    def junction_names(lo: int, hi: int) -> List[str]:
        """Junctions ``chr1:<start>-<end>`` in groups of three per start (alternative acceptors) and per end (alternative
        donors), so the PSI recipe finds events."""
        out = []
        for i in range(lo, hi):
            out.append("chr1:%d-%d" % (1000 * (1 + i // 3), 50000 + 100 * (i % 3) + 7 * (i // 9)))
        return out

    def build(
        self,
        n_cols: int = 300,
        n_called: int = 40,
        chunk_cols: int = 64,
        pad_between: int = 0,
        extras: bool = False,
        junction_table: bool = False,
    ) -> "Synth":
        """``extras`` adds feature-level members: ``sj_counts.1pz`` (inline junction names; GSM9000001 and GSM9000003 have
        overlapping, different junction sets) and ``vdj_gene_usage.1pz`` (GSM9000001 only). ``junction_table`` stores the junction rows
        as ids into a shared junction table (the 30 junctions both samples have; each sample's others go to ``sj_counts.1pz.local``)."""
        self.jx = None
        if extras and junction_table:
            from singlet.s1 import junctions as J

            shared = self.junction_names(30, 60)
            seg = J.build_segment(J.parse_names(shared), build=BUILD, id_lo=0, tier="core")
            ver = J.build_version([seg], build=BUILD, version=1)
            self.jx = (J.JunctionTable(ver, [seg]), seg, ver)
        gene_tbl = self.exon_pack.gene_table()
        gene_ids = [g for g, _ in gene_tbl]
        specs = [
            ("GSM9000001", n_cols, n_called, True, True, 11),
            ("GSM9000002", 120, 0, True, True, 12),  # no called cell
            ("GSM9000003", 200, 25, True, False, 13),  # no intron matrix
        ]
        samples = []
        for gsm, n, ncalled, has_exon, has_intron, seed in specs:
            rng = np.random.default_rng(seed)
            bcs = self._barcodes("A", n)
            # cells sorted by library size, like the pipeline's column order; the called cells are among the largest, not a prefix
            dens = np.sort(rng.integers(5, 400, n))[::-1]
            called = np.zeros(n, bool)
            if ncalled:
                called[
                    np.sort(rng.choice(max(ncalled * 3, ncalled), size=ncalled, replace=False))
                ] = True
            w = K.S2Writer(gsm)
            ex_blob, ex_csc = self._matrix(
                len(self.exon_names), n, seed, self.exon_pack, bcs, dens, chunk_cols
            )
            in_blob = in_csc = None
            if has_intron:
                in_blob, in_csc = self._matrix(
                    len(self.intron_names),
                    n,
                    seed + 100,
                    self.intron_pack,
                    bcs,
                    dens // 2,
                    chunk_cols,
                )
            umi_e = np.asarray(ex_csc.sum(axis=0)).ravel().astype(np.uint32)
            ct = cells_mod.CellsTable(n)
            ct.add("barcode", cells_mod.DT_STR, bcs)
            ct.add("called", cells_mod.DT_BOOL, called)
            ct.add("n_umi_exon", cells_mod.DT_U32, umi_e)
            ct.add(
                "n_umi_intron",
                cells_mod.DT_U32,
                np.asarray(in_csc.sum(axis=0)).ravel().astype(np.uint32)
                if has_intron
                else np.zeros(n, np.uint32),
            )
            ct.add("n_feat_exon", cells_mod.DT_U32, np.diff(ex_csc.indptr).astype(np.uint32))
            ct.add(
                "n_feat_intron",
                cells_mod.DT_U32,
                np.diff(in_csc.indptr).astype(np.uint32) if has_intron else np.zeros(n, np.uint32),
            )
            w.add_cells("cells", ct.encode(), ct.digest(), {"n_rows": n}, flags=K.FLAG_ANNDATA)
            w.add_matrix(
                "exon_counts.1pz",
                ex_blob,
                bytes.fromhex(_pz2.read_header(ex_blob)["digest"]),
                {"m": len(self.exon_names), "n": n},
                K.FLAG_LEGACY | K.FLAG_ANNDATA,
            )
            if has_intron:
                w.add_matrix(
                    "intron_counts.1pz",
                    in_blob,
                    bytes.fromhex(_pz2.read_header(in_blob)["digest"]),
                    {"m": len(self.intron_names), "n": n},
                    K.FLAG_LEGACY | K.FLAG_ANNDATA,
                )
            extra_truth = {}
            if extras:
                lo = 0 if gsm != "GSM9000003" else 30
                sj_names = self.junction_names(lo, lo + 60)
                sj_blob, sj_csc = self._matrix(
                    len(sj_names),
                    n,
                    seed + 200,
                    sj_names,
                    bcs,
                    np.minimum(dens // 8, 40),
                    chunk_cols,
                )
                if self.jx is not None:
                    from singlet.s1 import junctions as J

                    rows = J.sample_rows(self.jx[0], sj_names)
                    v2 = sj_csc[rows.legacy_perm, :].tocsc()
                    v2.sort_indices()
                    sj_blob, _ = _pz2.encode(
                        v2.indptr.astype(np.int64),
                        v2.indices.astype(np.uint32),
                        v2.data.astype(np.uint32),
                        len(sj_names),
                        n,
                        codec="rans",
                        chunk_cols=chunk_cols,
                        row_names={
                            "kind": "pack_subset",
                            "pack_id": self.jx[0].id[:8],
                            "pack_index": rows.ids,
                            "names": [sj_names[k] for k in rows.legacy_perm.tolist()],
                        },
                        col_names={"kind": "cells", "names": bcs},
                    )
                    w.add_stored("sj_counts.1pz.local", J.encode_local(rows.local), {}, 0)
                    extra_truth["sj_rows"] = rows
                w.add_matrix(
                    "sj_counts.1pz",
                    sj_blob,
                    bytes.fromhex(_pz2.read_header(sj_blob)["digest"]),
                    {"m": len(sj_names), "n": n},
                    K.FLAG_LEGACY,
                )
                extra_truth.update({"sj": sj_csc, "sj_names": sj_names})
                if gsm == "GSM9000001":
                    vdj_names = ["TRBV%d" % i for i in range(1, 31)]
                    vdj_blob, vdj_csc = self._matrix(
                        len(vdj_names),
                        n,
                        seed + 300,
                        vdj_names,
                        bcs,
                        np.minimum(dens // 40, 10),
                        chunk_cols,
                    )
                    w.add_matrix(
                        "vdj_gene_usage.1pz",
                        vdj_blob,
                        bytes.fromhex(_pz2.read_header(vdj_blob)["digest"]),
                        {"m": len(vdj_names), "n": n},
                        K.FLAG_LEGACY,
                    )
                    extra_truth.update({"vdj": vdj_csc, "vdj_names": vdj_names})
            summary = json.dumps({"n_cells_called": int(called.sum()), "gsm": gsm}).encode()
            w.add_stored("summary.json", summary, {}, K.FLAG_LEGACY)
            w.add_stored(
                "cell_calls.tsv",
                (
                    "barcode\tis_cell\n"
                    + "".join("%s\tTRUE\n" % b for b, c in zip(bcs, called) if c)
                ).encode(),
                {},
                K.FLAG_LEGACY,
            )
            obj = w.build()
            d16 = util.d16(obj).hex()
            key = "s1/s/%s/%s.s1" % (gsm, d16)
            path = os.path.join(self.root, *key.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(obj)
            members = {}
            for info in w.infos:
                d = {
                    "kind": info.kind,
                    "flags": info.flags,
                    "off": info.off,
                    "len": info.len,
                    "digest": info.digest.hex(),
                }
                if info.kind == K.KIND_PZ2:
                    h = _pz2.read_header(obj[info.off : info.off + 256])
                    d.update(
                        {
                            "m": h["m"],
                            "n": h["n"],
                            "nnz": h["nnz"],
                            "codec": h["codec"],
                            "chunk_cols": h["chunk_cols"],
                            "n_chunks": h["n_chunks"],
                        }
                    )
                members[info.name] = d
            samples.append(M.sample_entry(gsm, key, obj, n, int(called.sum()), members, {}))
            self.truth[gsm] = {
                "barcodes": bcs,
                "called": called,
                "exon": ex_csc,
                "intron": in_csc,
                "gene_ids": gene_ids,
                "exon_g": self._gene_level(ex_csc, self.exon_names, gene_ids),
                "intron_g": self._gene_level(in_csc, self.intron_names, gene_ids)
                if has_intron
                else None,
                "umi_exon": umi_e,
                "object": obj,
                "key": key,
                **extra_truth,
            }
        meta = {
            "schema_version": "1.0",
            "gse_id": GSE,
            "series_title": "synthetic",
            "gsm_meta": {
                s["gsm"]: {
                    "gsm_id": s["gsm"],
                    "organism": "Homo sapiens",
                    "protocol": "10xv3",
                    "protocol_name": "p",
                    "sample_source": "src",
                    "sample_characteristics": "x",
                    "qc_flag": "ok",
                    "reference_build": BUILD,
                    "n_cells": s["n_called"],
                }
                for s in samples
            },
        }
        sfiles = {"study_meta.json": M.study_file_entry(json.dumps(meta, indent=2).encode())}
        man = M.build_manifest(
            GSE,
            samples,
            sfiles,
            BUILD,
            {"singlet_version": "test"},
            {
                "exon": self.exon_pack.id.hex(),
                "intron": self.intron_pack.id.hex(),
                "genes": None,
                "whitelist": None,
            },
            {"key": "data/%s/%s.singlet" % (GSE, GSE)},
            "2026-10-22T00:00:00Z",
        )
        mpath = os.path.join(self.root, "s1", "m", GSE + ".json")
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        with open(mpath, "wb") as fh:
            fh.write(M.dumps_manifest(man))
        self.manifest = man
        self._write_ref()
        return self

    def _write_ref(self):
        """s1/ref/<id>.pack + index.json as the server lays them out (clients without the wheel packs fetch them from here)."""
        from singlet.s1 import packs as P

        ref = os.path.join(self.root, "s1", "ref")
        os.makedirs(ref, exist_ok=True)
        entries = []
        for pk in (self.exon_pack, self.intron_pack):
            data = pk.bytes
            with open(os.path.join(ref, pk.id.hex() + ".pack"), "wb") as fh:
                fh.write(data)
            entries.append(P.index_entry(data, "vocab", {"build": pk.build, "member": pk.member}))
        for data, kind in (
            ((self.jx[1], "junction-segment"), (self.jx[2], "junction-version")) if self.jx else ()
        ):
            with open(os.path.join(ref, P.pack_id(data).hex() + ".pack"), "wb") as fh:
                fh.write(data)
            entries.append(P.index_entry(data, kind, {"build": BUILD}))
        P.write_index(entries, os.path.join(ref, "index.json"))


# ---------------------------------------------------------------------------------------------------------------------------------
# command line: build the synthetic study and write the expected gene-level matrices for other languages' tests (the R tests)
#     python -P tests/python/_s1_synth.py <out_dir>
# (``--extras`` after <out_dir> adds the sj / vdj members of ``build(extras=True)`` and their truth files)
# writes <out_dir>/store/... (the study) and, per sample with called cells, truth_<GSM>.mtx (genes x cells, counts), truth_<GSM>_spliced.mtx,
# truth_<GSM>_unspliced.mtx (when the sample has an intron matrix), truth_<GSM>_barcodes.txt (cells in sorted-barcode order) and genes.txt.
if __name__ == "__main__":
    import sys

    import scipy.io

    out = sys.argv[1]
    extras = (
        "--extras" in sys.argv[2:]
    )  # also the sj / vdj members, and truth_<GSM>_sj.mtx + truth_<GSM>_sj_names.txt
    os.makedirs(out, exist_ok=True)
    s = Synth(out).build(extras=extras)
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
        if extras and "sj" in t:
            scipy.io.mmwrite(os.path.join(out, "truth_%s_sj.mtx" % gsm), t["sj"][:, cols])
            with open(os.path.join(out, "truth_%s_sj_names.txt" % gsm), "w") as fh:
                fh.write("\n".join(t["sj_names"]) + "\n")
    print(out)
