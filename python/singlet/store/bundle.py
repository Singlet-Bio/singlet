# SPDX-License-Identifier: MIT
"""``StoreBundle``: the :class:`singlet.bundle.SingletBundle` interface over a per-sample study (or one ``.s1`` sample object).

``singlet.open_bundle("GSE...")`` returns one of these when the study has a store manifest. Every public method of
``SingletBundle`` works: ``manifest``, ``study_meta``, ``feature_vocab``, ``gsm_ids``, ``list_files``, ``has``, ``modalities``,
``read``, ``raw_counts``, ``qc``, ``to_anndata``, ``to_h5ad``, ``to_parquet`` and the convenience accessors. Differences from a
legacy bundle, all by construction:

* nothing is a ZIP: ``list_files`` lists the legacy member names the sample object can regenerate (``exon_counts.1pz``,
  ``cell_calls.tsv``, ...), the study-level names are ``manifest.json``, ``study_meta.json``, ``feature_vocab.json``;
* a member is fetched (one HTTP range) when it is first asked for;
* ``read`` of a matrix modality returns the canonical matrix (legacy row order, ``uint32`` values), ``read`` of a table or JSON
  member regenerates the legacy bytes from the sample object (stored blob or verified recipe);
* ``splice_psi`` regenerates the legacy PSI matrix (kept for compatibility, see D8); ``singlet.psi()`` computes the true PSI.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from singlet.bundle import MODALITIES, SingletBundle, _infer_kind

from . import client as C
from . import container as K

_AUX = ("vector:", "#csc")


def _is_aux(name: str) -> bool:
    return name.startswith(_AUX[0]) or name.endswith(_AUX[1])


class StoreBundle(SingletBundle):
    """Reader for a per-sample study with the interface of :class:`SingletBundle`."""

    def __init__(
        self, study: "C.Study", *, source_path: Optional[Path] = None, single: Optional[str] = None
    ) -> None:  # noqa: D107
        self.study = study
        self.path = source_path or Path(f"store/m/{study.gse}.json")
        self.restricted_path = None
        self._zf = None
        self._manifest: Optional[dict] = None
        self._study_meta: Optional[dict] = None
        self._feature_vocab: Optional[dict] = None
        self._empty_reasons: Dict[str, str] = {}
        self._single = single
        self._samples: Dict[str, Any] = {}  # gsm -> StoreSample (members table, for regeneration)
        self._ctx: Dict[str, Any] = {}
        self._recipes = None

    # ---- opening
    @classmethod
    def open(
        cls,
        source: Union[str, os.PathLike],
        restricted=None,
        *,
        force: bool = False,
        base: Optional[str] = None,
    ) -> "StoreBundle":  # type: ignore[override]
        """Open a study by accession (``GSE...``), a ``store/m/<GSE>.json`` manifest path, or a ``.s1`` sample object."""
        if restricted is not None:
            raise ValueError("restricted sidecars are not part of the per-sample format")
        p = Path(str(source)).expanduser()
        if p.exists() and p.suffix.lower() == ".json":
            root, gse = C._manifest_root(p)
            st = C.open_study(gse, transport=C.T.DirTransport(root), force=True)
            return cls(st, source_path=p)
        if p.exists() and p.suffix.lower() in (".s1", ".s2"):
            return cls._open_s2(p)
        from singlet._integrity import normalize_gse

        return cls(C.open_study(normalize_gse(str(source)), base=base, force=force))

    @classmethod
    def _open_s2(cls, p: Path) -> "StoreBundle":
        rd = K.S2Reader(str(p))
        members = {
            n: {
                "kind": mi.kind,
                "flags": mi.flags,
                "off": mi.off,
                "len": mi.len,
                "digest": mi.digest.hex(),
                **{k: v for k, v in (mi.meta or {}).items() if k in ("names",)},
            }
            for n, mi in rd.members.items()
        }
        exon = rd.members.get(C.EXON)
        pack = None
        if exon is not None:
            hdr = C._native().Member(rd.read_member(C.EXON)).header
            pack = hdr["row_names_id"]
        doc = {
            "format": "singlet-store",
            "manifest_version": 1,
            "gse": "GSE0",
            "rev": "",
            "packs": {"exon": pack},
            "samples": [
                {"gsm": rd.sample_id, "key": p.name, "bytes": rd.file_len, "members": members}
            ],
        }
        st = C.Study(doc, C.T.DirTransport(str(p.parent)), {"origin": "s2"}, concurrency=1)
        st.samples[rd.sample_id].raw["_local_path"] = str(p)
        return cls(st, source_path=p, single=str(p))

    # ---- properties
    def _sm(self, gsm: str) -> "C.SampleRef":
        try:
            return self.study.samples[gsm]
        except KeyError:
            raise KeyError(f"{gsm} is not a sample of {self.study.gse}") from None

    @property
    def manifest(self) -> dict:
        """The legacy-style ``manifest.json`` (regenerated when the study stores it, else assembled from the store manifest)."""
        if self._manifest is None:
            st = self.study
            man = None
            ent = st.study_files.get("manifest.json")
            if ent is not None and ("json" in ent or "zstd_b64" in ent):
                try:
                    man = json.loads(st.study_file("manifest.json").decode("utf-8"))
                except Exception:  # noqa: BLE001
                    man = None
            if man is None:
                man = {
                    "schema_version": "1.0",
                    "gse_id": st.gse,
                    "gsm_ids": list(st.gsm_ids),
                    "reference_build": st.reference_build,
                    "n_gsms": len(st.gsm_ids),
                    "created_at": st.doc.get("created_at"),
                    "singlet_version": (st.doc.get("pipeline") or {}).get("singlet_version"),
                    "included_files": {g: self.list_files(g) for g in st.gsm_ids},
                    "excluded_samples": [],
                    "checksums": {},
                    "profile": "public",
                }
            self._manifest = man
        return self._manifest

    @property
    def study_meta(self) -> dict:
        return self.study.study_meta

    @property
    def feature_vocab(self) -> dict:
        if self._feature_vocab is None:
            self._feature_vocab = self.study.feature_vocab()
        return self._feature_vocab

    @property
    def gsm_ids(self) -> List[str]:
        return list(self.study.gsm_ids)

    def close(self) -> None:
        self.study.close()

    # ---- members
    def list_files(self, gsm: Optional[str] = None) -> List[str]:
        def rel(g: str) -> List[str]:
            return [
                n
                for n, m in self._sm(g).members.items()
                if (m.meta.get("flags", 1) & 1 or m.kind == 4) and not _is_aux(n) and n != C.CELLS
            ]

        if gsm is not None:
            return rel(gsm)
        names = list(self.study.study_files)
        for g in self.study.gsm_ids:
            names.extend(f"samples/{g}/{n}" for n in rel(g))
        return names

    def _resolve_member(self, gsm: str, name: str) -> Optional[str]:
        available = set(self.list_files(gsm))
        for cand in MODALITIES[name].members if name in MODALITIES else (name,):
            if cand in available:
                return cand
        return None

    # ---- reading
    def _sample_obj(self, gsm: str):
        """W2's ``StoreSample`` over the sample object, reading through ranged requests (members are fetched when asked for)."""
        if gsm not in self._samples:
            from .sample import StoreSample

            sm = self._sm(gsm)
            if self._single:
                self._samples[gsm] = StoreSample(self._single, packs=self.study._resolver.store)
            else:
                src = self.study.source_for(sm)
                self._samples[gsm] = StoreSample(
                    lambda off, ln: bytes(src.read(off, ln)),
                    packs=self.study._resolver.store,
                    size=sm.bytes,
                )
        return self._samples[gsm]

    def _context(self, gsm: str):
        if gsm not in self._ctx:
            from .context import StoreContext
            from .rcp_adapter import load_recipes

            if self._recipes is None:
                self._recipes = load_recipes()
            files = {}
            for n in self.study.study_files:
                try:
                    files[n] = self.study.study_file(n)
                except Exception:  # noqa: BLE001
                    pass
            self._ctx[gsm] = StoreContext(
                self._sample_obj(gsm),
                self._recipes,
                gse=self.study.gse,
                reference_build=self.study.reference_build,
                study_files=files,
            )
        return self._ctx[gsm]

    def _member_bytes(self, gsm: str, rel: str) -> bytes:
        data = self._context(gsm).member_bytes(rel)
        if data is None:
            raise KeyError(f"{rel} of {gsm} cannot be regenerated")
        return data

    def read(self, name: str, gsm: str):
        member = self._resolve_member(gsm, name)
        if member is None:
            raise KeyError(
                f"{name!r} is not present for {gsm} in {self.path.name}. Available: {sorted(self.modalities(gsm))}"
            )
        kind = MODALITIES[name].kind if name in MODALITIES else _infer_kind(member)
        mkind = self._sm(gsm).members[member].kind
        if kind == "matrix" or member.endswith(".1pz"):
            if mkind == K.KIND_PZ2:
                return self._read_matrix(gsm, member, name)
            return self._read_derived_matrix(gsm, member, name)
        data = self._member_bytes(gsm, member)
        if kind == "json":
            return json.loads(data)
        if kind == "text":
            return data.decode("utf-8", errors="replace")
        if kind == "table":
            import pandas as pd

            return pd.read_csv(io.BytesIO(data), sep="\t")
        raise ValueError(f"Unknown modality kind {kind!r} for {name!r}")

    def _anndata_from_csc(
        self, gsm: str, name: str, indptr, indices, data, m: int, n: int, rownames, colnames, kv
    ):
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        X = sp.csr_matrix((data, indices, indptr), shape=(n, m), copy=False)
        adata = ad.AnnData(X=X)
        if rownames is not None and len(rownames):
            adata.var_names = pd.Index(list(rownames))
        if colnames is not None and len(colnames):
            adata.obs_names = pd.Index(list(colnames))
        adata.obs["gsm_id"] = gsm
        if kv:
            adata.uns.update(kv)
        adata.uns["modality"] = name
        return adata

    def _read_matrix(self, gsm: str, member: str, name: str):
        sm = self._sm(gsm)
        mref = sm.members[member]
        src = self.study.source_for(sm)
        try:
            pz = C._native()
            mem = pz.Member(src.read(mref.off, mref.len))
            hdr = mem.header
            res = mem.decode(mode="canonical", threads=0, dtype="uint32")
            if hdr["row_names_kind"] == "pack_subset" and self._is_junction_table(hdr):
                # junction rows stored as table ids: the bundle API returns the legacy order and names (format-store 12.4)
                from . import junctions as J

                local = self._small_member(gsm, member + J.LOCAL_SUFFIX)
                perm = self._small_member(gsm, member + J.LEGACY_ORDER_SUFFIX)
                ip, ix, dt, rownames = J.to_legacy(
                    res["indptr"],
                    res["indices"],
                    res["data"],
                    mem.row_names()["pack_index"],
                    self.study._resolver.get(hdr["row_names_id"]),
                    local,
                    perm,
                )
                res = dict(res, indptr=ip.astype(np.int32), indices=ix.astype(np.int32), data=dt)
            else:
                rownames = self._row_names(mem, hdr)
            colnames = self._col_names(gsm, mem, hdr)
            kv = mem.kv()
            return self._anndata_from_csc(
                gsm,
                name,
                res["indptr"],
                res["indices"],
                res["data"],
                hdr["m"],
                hdr["n"],
                rownames,
                colnames,
                kv,
            )
        finally:
            src.close()

    def _is_junction_table(self, hdr) -> bool:
        from .junctions import JunctionTable

        return isinstance(self.study._resolver.get(hdr["row_names_id"]), JunctionTable)

    def _small_member(self, gsm: str, name: str) -> Optional[bytes]:
        """Bytes of a small stored member (kind 3 decompressed, kind 5 as is), None when absent."""
        sm = self._sm(gsm)
        mref = sm.members.get(name)
        if mref is None or not mref.len:
            return None
        src = self.study.source_for(sm)
        try:
            raw = bytes(src.read(mref.off, mref.len))
        finally:
            src.close()
        if mref.kind == K.KIND_STORED:
            from .util import zstd_decompress

            raw = zstd_decompress(raw)
        return raw

    def _row_names(self, mem, hdr) -> list:
        kind = hdr["row_names_kind"]
        if kind == "inline":
            return mem.row_names()["names"]
        pk = self.study._resolver.get(hdr["row_names_id"])
        if kind == "pack":
            return pk.names()
        info = mem.row_names()
        return pk.names_at(info["pack_index"])

    def _col_names(self, gsm: str, mem, hdr) -> list:
        kind = hdr["col_names_kind"]
        if kind == "inline":
            return mem.col_names()["names"]
        if kind == "cells":
            sm = self._sm(gsm)
            cm = sm.members[C.CELLS]
            src = self.study.source_for(sm)
            try:
                rd = C._open_cells(src.read(cm.off, cm.len), self.study._resolver)
                return list(rd.column("barcode"))
            finally:
                src.close()
        pk = self.study._resolver.get(hdr["col_names_id"])
        info = mem.col_names()
        return (
            pk.barcodes(info["pack_index"])
            if info.get("pack_index") is not None
            else pk.all_barcodes()
        )

    def _read_derived_matrix(self, gsm: str, member: str, name: str):
        csc = self._context(gsm).csc(member)
        if csc is None:
            raise KeyError(f"{member} of {gsm} cannot be regenerated")
        return self._anndata_from_csc(
            gsm,
            name,
            csc.indptr.astype(np.int32),
            csc.indices.astype(np.int32),
            csc.data.astype(np.uint32),
            csc.m,
            csc.n,
            csc.rownames,
            csc.colnames,
            csc.kv,
        )

    # ---- cells and counts
    def _read_cell_calls(self, gsm: str):
        try:
            data = self._member_bytes(gsm, "cell_calls.tsv")
        except KeyError:
            return None
        if not data.strip():
            return None
        import pandas as pd

        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", dtype=str)
        except ValueError:
            return None

    def _called_barcodes(self, gsm: str) -> Optional[set]:
        sm = self._sm(gsm)
        cm = sm.members.get(C.CELLS)
        if cm is None:
            return None
        src = self.study.source_for(sm)
        try:
            rd = C._open_cells(src.read(cm.off, cm.len), self.study._resolver)
            called = np.asarray(rd.column("called"), dtype=bool)
            return set(np.asarray(rd.column("barcode"), dtype=object)[called].tolist())
        finally:
            src.close()

    def _is_empty_pz(self, gsm: str, fname: str) -> bool:
        m = self._sm(gsm).members.get(fname)
        return m is None or m.len == 0 or (m.meta.get("m") == 0) or (m.meta.get("n") == 0)

    def raw_counts(self, gsm: str, *, gene_level: bool = True, cells: str = "called"):
        if cells not in ("called", "all"):
            raise ValueError("cells must be 'called' or 'all'")
        if not gene_level:
            return super().raw_counts(gsm, gene_level=False, cells=cells)
        try:
            adata = self.study.to_anndata([gsm], layers=True)
        except C.NoUsableSamples as e:
            raise ValueError(
                f"{gsm} has no usable cells in {self.path.name}: {e.reasons.get(gsm, 'no called cells')}"
            ) from None
        prefix = f"{gsm}_"
        adata.obs_names = [n[len(prefix) :] if n.startswith(prefix) else n for n in adata.obs_names]
        adata.obs = adata.obs[["gsm_id"]]
        adata.uns = {"reference_build": self.feature_vocab.get("reference_build", "")}
        adata.var = adata.var[["gene_name"]]
        adata.var_names_make_unique()
        return adata

    # ---- whole study
    def to_anndata(
        self, layer: str = "gene_counts", *, layers: bool = True, verbose: bool = True, **kw
    ):  # type: ignore[override]
        if layer != "gene_counts":
            raise ValueError("only layer='gene_counts' is supported")
        return self.study.to_anndata(None, layers=layers, verbose=verbose, **kw)

    def _zip(self):  # pragma: no cover - there is no archive
        raise NotImplementedError("a per-sample study is not a ZIP archive")

    def __repr__(self) -> str:
        return (
            f"StoreBundle({self.study.gse!r}, {len(self.study.gsm_ids)} GSMs, path={str(self.path)!r})"
        )


V2Bundle = StoreBundle  # DUAL: old name, removed in the cleanup
