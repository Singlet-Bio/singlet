# SPDX-License-Identifier: MIT
"""singlet.v2.client -- the v2 read path.

::

    study = open_study("GSE145370")                 # fetches v2/m/GSE145370.json (ETag revalidated), nothing else
    adata = study.to_anndata(samples=["GSM4317426"])
    adata = load("GSM4317426")                      # same, resolving the parent study first
    sub   = cells("GSE145370", where="n_umi_exon > 5000", samples=[...])   # only those cells' chunks are fetched
    adata = load("/data/GSM4317426.s2")             # a downloaded sample object, read from disk (mmap, zero copy)

What is fetched.  A sample is one immutable object ``v2/s/<GSM>/<d>.s2``.  The manifest gives the byte range of every member, so
``load`` issues HTTP Range requests for the members it needs (``cells``, ``exon_counts.1pz``, ``intron_counts.1pz``), coalesced into
one request per sample when they are close together, and never reads the member table.  Large samples switch to a two-phase plan
(cells table first, then only the chunk runs that cover the wanted cells).  Every ranged response is checked (HTTP 206,
``Content-Range`` start/end/total against the manifest size), a pz2 header must carry the digest the manifest lists, and with
``verify=True`` the canonical matrix digest is recomputed after decoding.

How it decodes.  Each matrix member is decoded once by ``singlet._pz2`` straight into numpy arrays (GIL released, ``threads``
workers), already aggregated to genes (a stored-row -> gene map derived from the vocabulary pack: no string work), restricted to
the wanted columns.  ``X`` is ``spliced + unspliced``; a genes x cells CSC is exactly a cells x genes CSR, so nothing is
transposed or copied on the way to AnnData.  Names come lazily from packs (shipped in the wheel, else fetched once to the cache
and verified by sha256).

Environment: ``SINGLET_DATA_BASE`` (default ``https://data.singlet.bio``; a directory or ``file://`` URL with the server layout also
works), ``SINGLET_CACHE_DIR`` (default ``~/.singlet/cache``), ``SINGLET_V2_VERIFY=1`` (same as ``verify=True``).
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from . import transport as T
from .transport import NotFound, Transport, TransportError

if TYPE_CHECKING:  # pragma: no cover
    import anndata

__all__ = [
    "V2Error",
    "StudyNotFound",
    "NoUsableSamples",
    "Study",
    "SampleRef",
    "MemberRef",
    "open_study",
    "load",
    "cells",
    "load_s2",
    "download_study",
    "clear_memory_cache",
]

MANIFEST_VERSION_MAX = 1
EXON, INTRON, CELLS = "exon_counts.1pz", "intron_counts.1pz", "cells"
WHOLE_MAX_BYTES = 48 << 20  # members up to this size (exon+intron+cells) are fetched whole
COALESCE_GAP = (
    2 << 20
)  # merge ranges closer than this (one round trip costs about 6 MB of transfer)
PART_BYTES = 16 << 20  # a single range larger than 2 parts is split and fetched in parallel
PREFIX_HINT = 512 << 10  # first read of a member in the two-phase plan
PARTIAL_GAP = 256 << 10  # in the two-phase plan, ranges closer than this share a request


class V2Error(RuntimeError):
    """Something is wrong with a v2 object or with how it was asked for."""


class StudyNotFound(V2Error, FileNotFoundError):
    """No v2 manifest exists for the study."""


class NoUsableSamples(V2Error, RuntimeError):
    """Every requested sample is empty (no count matrix, or no called cells). ``reasons`` maps GSM -> why."""

    def __init__(self, message: str, skipped: Sequence[dict]):
        super().__init__(message)
        self.skipped = list(skipped)
        self.reasons = {s["gsm_id"]: s["reason"] for s in self.skipped}


# ================================================================================================================= configuration
def cache_root() -> Path:
    """``<$SINGLET_CACHE_DIR or ~/.singlet/cache>/v2`` (created on demand)."""
    env = os.environ.get("SINGLET_CACHE_DIR")
    return (Path(env) if env else (Path.home() / ".singlet" / "cache")) / "v2"


def _native():
    try:
        from singlet import _pz2
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "singlet._pz2 (the v2 matrix decoder) is not built. Install with `pip install singlet-bio` (wheel) or build it with "
            "`pip install .` / src/bindings/python/build_pz2_mingw.sh."
        ) from e
    return _pz2


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ================================================================================================================= manifest model
@dataclass
class MemberRef:
    name: str
    kind: int
    off: int
    len: int
    digest: Optional[str] = None
    nnz: Optional[int] = None
    chunk_cols: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.off + self.len


@dataclass
class SampleRef:
    gsm: str
    key: str
    bytes: int
    sha256: Optional[str]
    n_columns: Optional[int]
    n_called: Optional[int]
    members: Dict[str, MemberRef]
    stats: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def member(self, name: str) -> Optional[MemberRef]:
        return self.members.get(name)


def _member_ref(name: str, d: dict) -> MemberRef:
    return MemberRef(
        name=name,
        kind=int(d.get("kind", 0)),
        off=int(d.get("off", 0)),
        len=int(d.get("len", 0)),
        digest=d.get("digest"),
        nnz=(int(d["nnz"]) if d.get("nnz") is not None else None),
        chunk_cols=(int(d["chunk_cols"]) if d.get("chunk_cols") is not None else None),
        meta={
            k: v
            for k, v in d.items()
            if k not in ("kind", "off", "len", "digest", "nnz", "chunk_cols")
        },
    )


def _sample_ref(d: dict) -> SampleRef:
    members = {n: _member_ref(n, m) for n, m in (d.get("members") or {}).items()}
    return SampleRef(
        gsm=d["gsm"],
        key=d["key"],
        bytes=int(d["bytes"]),
        sha256=d.get("sha256"),
        n_columns=(int(d["n_columns"]) if d.get("n_columns") is not None else None),
        n_called=(int(d["n_called"]) if d.get("n_called") is not None else None),
        members=members,
        stats=d.get("stats") or {},
        raw=d,
    )


# ================================================================================================================= packs
class _PackResolver:
    """Wheel packs first, then the cache directory, then ``v2/ref/<id>.pack`` from the data host (sha256 verified against
    ``v2/ref/index.json``). One instance per transport."""

    def __init__(self, transport: Transport):
        from .packs import PackStore

        self.transport = transport
        self.lock = threading.RLock()
        self.dir = cache_root() / "ref"
        outer = self

        class _Store(PackStore):
            def _fetch_remote(self, hexid: str) -> str:  # noqa: D401 - PackStore hook
                return outer._download(self, hexid)

        self.store = _Store(download=True)
        if str(self.dir) not in self.store.dirs:
            self.store.dirs.append(str(self.dir))
            self.store.rescan()
        self._index: Optional[dict] = None
        self._gene: Dict[tuple, Any] = {}

    # ---- download hook
    def _index_doc(self) -> dict:
        if self._index is None:
            r = self.transport.get("v2/ref/index.json", None, kind="packs")
            self._index = json.loads(bytes(r.body).decode("utf-8"))
        return self._index

    def _download(self, store, hexid: str) -> str:
        from .packs import PackError, pack_id

        with self.lock:
            idx = self._index_doc()
            ent = [e for e in idx.get("packs", []) if str(e.get("id", "")).startswith(hexid)]
            if len(ent) != 1:
                raise PackError(f"pack {hexid} is not in {self.transport.url('v2/ref/index.json')}")
            e = ent[0]
            r = self.transport.get(f"v2/ref/{e['id']}.pack", None, kind="packs")
            data = bytes(r.body)
            if (
                hashlib.sha256(data).hexdigest() != e.get("sha256")
                or pack_id(data).hex() != e["id"]
            ):
                raise PackError(f"downloaded pack {e['id']} fails its checksum")
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / (e["id"] + ".pack")
            tmp = path.with_suffix(f".part{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            store._files[e["id"]] = str(path)
            return e["id"]

    # ---- lookups
    def get(self, hexid: str):
        with self.lock:
            return self.store.get(hexid)

    def gene_table(self, pack_hex: str) -> Tuple[np.ndarray, np.ndarray]:
        """(gene_ids, gene_names) as numpy object arrays."""
        key = ("gt", pack_hex)
        with self.lock:
            v = self._gene.get(key)
            if v is None:
                tbl = self.get(pack_hex).gene_table()
                v = (
                    np.array([g for g, _ in tbl], dtype=object),
                    np.array([n for _, n in tbl], dtype=object),
                )
                self._gene[key] = v
            return v

    def row_gene(self, pack_hex: str) -> np.ndarray:
        key = ("rg", pack_hex)
        with self.lock:
            v = self._gene.get(key)
            if v is None:
                v = np.ascontiguousarray(self.get(pack_hex).row_gene(), dtype=np.int32)
                self._gene[key] = v
            return v


_RESOLVERS: Dict[str, _PackResolver] = {}
_RES_LOCK = threading.Lock()


def _resolver_for(transport: Transport) -> _PackResolver:
    with _RES_LOCK:
        r = _RESOLVERS.get(transport.base)
        if r is None:
            r = _RESOLVERS[transport.base] = _PackResolver(transport)
        return r


# ================================================================================================================= fetching
def _coalesce(
    ranges: Sequence[Tuple[int, int]], gap: int = COALESCE_GAP
) -> List[Tuple[int, int, List[int]]]:
    """[(start, end, [indices into ranges])] with ranges closer than ``gap`` merged. ``ranges`` are (off, len); zero length skipped."""
    order = sorted((i for i, (_o, ln) in enumerate(ranges) if ln > 0), key=lambda i: ranges[i][0])
    out: List[list] = []
    for i in order:
        o, ln = ranges[i]
        if out and o - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], o + ln)
            out[-1][2].append(i)
        else:
            out.append([o, o + ln, [i]])
    return [(a, b, ix) for a, b, ix in out]


class _Source:
    """Random access to one sample object: ``read(off, len)`` returns a bytes-like object."""

    local = False

    def read(
        self, off: int, length: int, *, parallel: bool = False
    ):  # pragma: no cover - interface
        raise NotImplementedError

    def read_ranges(
        self, ranges: Sequence[Tuple[int, int]], *, parallel: bool = False, gap: int = COALESCE_GAP
    ) -> List[Any]:
        """Bytes-like for every (off, len) range; near ranges share one request, separate requests run concurrently."""
        res: List[Any] = [b""] * len(ranges)
        co = _coalesce(ranges, gap)
        if parallel and len(co) > 1 and self._executor() is not None:
            futs = [self._executor().submit(self.read, a, b - a) for a, b, _ in co]
            bufs = [f.result() for f in futs]
        else:
            bufs = [self.read(a, b - a, parallel=parallel and len(co) == 1) for a, b, _ in co]
        for (a, b, ix), buf in zip(co, bufs):
            mv = memoryview(buf)
            for i in ix:
                o, ln = ranges[i]
                res[i] = mv[o - a : o - a + ln]
        return res

    def _executor(self) -> Optional[ThreadPoolExecutor]:
        return None

    def close(self) -> None:
        pass


class _MmapSource(_Source):
    local = True

    def __init__(self, path: str, expect_size: Optional[int] = None):
        self.path = path
        self._fh = open(path, "rb")
        size = os.fstat(self._fh.fileno()).st_size
        if expect_size is not None and size != expect_size:
            self._fh.close()
            raise V2Error(f"{path} is {size} bytes, expected {expect_size}")
        self.size = size
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ) if size else None
        self._mv = memoryview(self._mm) if self._mm is not None else memoryview(b"")

    def read(self, off, length, *, parallel=False):
        if off < 0 or off + length > self.size:
            raise V2Error(f"{self.path}: range {off}+{length} outside the {self.size}-byte object")
        return self._mv[off : off + length]

    def close(self):
        try:
            self._mv.release()
            if self._mm is not None:
                self._mm.close()
        except (BufferError, ValueError):  # views still alive: the garbage collector closes it
            pass
        try:
            self._fh.close()
        except OSError:
            pass


class _RemoteSource(_Source):
    def __init__(
        self,
        transport: Transport,
        key: str,
        size: int,
        part_pool: Optional[ThreadPoolExecutor],
        part_bytes: int = PART_BYTES,
    ):
        self.transport, self.key, self.size = transport, key, size
        self._pool = part_pool
        self.part_bytes = part_bytes

    def _executor(self) -> Optional[ThreadPoolExecutor]:
        return self._pool

    def read(self, off, length, *, parallel=False):
        if length <= 0:
            return b""
        if off < 0 or off + length > self.size:
            raise V2Error(f"{self.key}: range {off}+{length} outside the {self.size}-byte object")
        if not parallel or self._pool is None or length < 2 * self.part_bytes:
            return self.transport.get(
                self.key, (off, off + length - 1), expect_total=self.size
            ).body
        n_parts = -(-length // self.part_bytes)
        step = -(-length // n_parts)
        parts = [
            (off + k * step, min(off + length, off + (k + 1) * step) - 1) for k in range(n_parts)
        ]
        futs = [
            self._pool.submit(self.transport.get, self.key, p, expect_total=self.size)
            for p in parts
        ]
        out = bytearray(length)
        pos = 0
        for p, f in zip(parts, futs):
            body = f.result().body
            out[pos : pos + len(body)] = body
            pos += len(body)
        return out


# ================================================================================================================= study
_STUDY_CACHE: Dict[Tuple[str, str], Tuple[float, "Study"]] = {}
_STUDY_LOCK = threading.Lock()
_STUDY_TTL = 60.0


def clear_memory_cache() -> None:
    """Forget in-process manifests (not the disk cache)."""
    with _STUDY_LOCK:
        _STUDY_CACHE.clear()


def _manifest_paths(gse: str) -> Tuple[Path, Path]:
    d = cache_root() / "m"
    return d / f"{gse}.json", d / f"{gse}.etag"


def _validate_manifest(doc: dict, gse: str) -> None:
    if not isinstance(doc, dict) or doc.get("format") != "singlet-v2":
        raise V2Error(f"{gse}: not a singlet-v2 manifest")
    if int(doc.get("manifest_version", 1)) > MANIFEST_VERSION_MAX:
        raise V2Error(
            f"{gse}: manifest_version {doc.get('manifest_version')} is newer than this client supports; upgrade singlet-bio"
        )
    if not isinstance(doc.get("samples"), list):
        raise V2Error(f"{gse}: manifest has no sample list")


_TRANSPORTS: Dict[str, Transport] = {}
_TRANSPORT_LOCK = threading.Lock()


def _transport_for(base: Optional[str]) -> Transport:
    """One transport (and so one set of kept-alive connections and one request counter) per base URL."""
    key = (base or T.data_base()).strip()
    with _TRANSPORT_LOCK:
        t = _TRANSPORTS.get(key)
        if t is None:
            t = _TRANSPORTS[key] = T.make_transport(key)
        return t


def open_study(
    gse: str,
    *,
    base: Optional[str] = None,
    force: bool = False,
    transport: Optional[Transport] = None,
    concurrency: int = 4,
    verify: Optional[bool] = None,
) -> "Study":
    """Open the v2 study ``gse``: one GET of ``v2/m/<GSE>.json`` (conditional on the cached ETag), nothing else.

    Raises :class:`StudyNotFound` when the study has no v2 manifest."""
    from singlet._integrity import normalize_gse

    gse = normalize_gse(gse)
    t = transport or _transport_for(base)
    ck = (t.base, gse)
    now = time.time()
    if not force:
        with _STUDY_LOCK:
            hit = _STUDY_CACHE.get(ck)
        if hit and now - hit[0] < _STUDY_TTL:
            return hit[1]
    key = f"v2/m/{gse}.json"
    doc: Optional[dict] = None
    mp, ep = _manifest_paths(gse)
    if t.is_local():
        try:
            doc = json.loads(bytes(t.get(key, None, kind="manifest").body).decode("utf-8"))
        except NotFound as e:
            raise StudyNotFound(f"no v2 manifest for {gse} under {t.base}") from e
        source = {"origin": "local", "path": t.path(key), "etag": None}  # type: ignore[attr-defined]
    else:
        etag = None
        if mp.exists() and ep.exists():
            try:
                etag = ep.read_text().strip() or None
            except OSError:
                etag = None
        try:
            r = t.get(key, None, if_none_match=etag, kind="manifest")
        except NotFound as e:
            raise StudyNotFound(
                f"no v2 manifest for {gse} at {t.url(key)}: the study has not been migrated to the v2 format (or the accession is wrong)"
            ) from e
        except TransportError as e:
            if mp.exists():
                warnings.warn(
                    f"could not revalidate {gse} ({e}); using the cached manifest",
                    RuntimeWarning,
                    stacklevel=2,
                )
                doc = json.loads(mp.read_text(encoding="utf-8"))
                source = {"origin": "cache-offline", "path": str(mp), "etag": etag}
                r = None
            else:
                raise
        if doc is None:
            assert r is not None
            if r.status == 304 and mp.exists():
                doc = json.loads(mp.read_text(encoding="utf-8"))
                source = {"origin": "cache-304", "path": str(mp), "etag": etag}
            else:
                raw = bytes(r.body)
                doc = json.loads(raw.decode("utf-8"))
                _validate_manifest(doc, gse)
                try:
                    mp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = mp.with_suffix(f".part{os.getpid()}")
                    tmp.write_bytes(raw)
                    os.replace(tmp, mp)
                    if r.etag:
                        ep.write_text(r.etag)
                    else:
                        ep.unlink(missing_ok=True)
                except OSError:
                    pass
                source = {"origin": "network", "path": str(mp), "etag": r.etag}
    _validate_manifest(doc, gse)
    st = Study(doc, t, source, concurrency=concurrency, verify=verify)
    with _STUDY_LOCK:
        _STUDY_CACHE[ck] = (now, st)
    return st


class Study:
    """A v2 study: its manifest plus the means to read its samples. Create with :func:`open_study`."""

    def __init__(
        self,
        doc: dict,
        transport: Transport,
        source: dict,
        *,
        concurrency: int = 4,
        verify: Optional[bool] = None,
    ):
        self.doc = doc
        self.transport = transport
        self.source = source
        self.gse: str = doc["gse"]
        self.rev: str = doc.get("rev", "")
        self.reference_build: str = doc.get("reference_build", "")
        self.packs: dict = doc.get("packs") or {}
        self.concurrency = max(1, int(concurrency))
        self.verify = _env_flag("SINGLET_V2_VERIFY") if verify is None else bool(verify)
        self.samples: Dict[str, SampleRef] = {}
        for sd in doc["samples"]:
            sm = _sample_ref(sd)
            self.samples[sm.gsm] = sm
        self.gsm_ids: List[str] = list(self.samples)
        self.study_files: dict = doc.get("study_files") or {}
        self._resolver = _resolver_for(transport)
        self._part_pool: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._maps: Dict[tuple, np.ndarray] = {}
        self._study_meta: Optional[dict] = None

    # ------------------------------------------------------------------ metadata
    @property
    def stats(self) -> T.Stats:
        """Requests and bytes issued through this study's transport so far (all kinds)."""
        return self.transport.stats

    def study_file(self, name: str) -> Optional[bytes]:
        """Bytes of a study-level legacy file (``study_meta.json``, ``manifest.json``, ``feature_vocab.json``, ...), or None."""
        ent = self.study_files.get(name)
        if ent is None:
            return None
        from . import manifest as M

        def regen(recipe_id: str, params: dict) -> bytes:
            if name == "feature_vocab.json":
                return self._resolver.get(self.packs["exon"]).vocab_json()
            raise V2Error(
                f"{name} is derived by recipe {recipe_id}; use the legacy tools to regenerate it"
            )

        return M.study_file_bytes(ent, regen)

    @property
    def study_meta(self) -> dict:
        v = self._study_meta
        if v is None:
            ent = self.study_files.get("study_meta.json")
            if ent is None:
                v = {}
            elif "json" in ent:
                v = ent["json"]
            else:
                raw = self.study_file("study_meta.json")
                v = json.loads(raw.decode("utf-8")) if raw else {}
            self._study_meta = v
        return v

    def feature_vocab(self) -> dict:
        """The study's ``feature_vocab.json`` content, regenerated from the vocabulary pack."""
        pk = self._resolver.get(self.packs["exon"])
        return json.loads(pk.vocab_json().decode("utf-8"))

    def gene_table(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._resolver.gene_table(self.packs["exon"])

    def __repr__(self) -> str:
        return f"Study({self.gse!r}, {len(self.samples)} samples, rev={self.rev[:8]!r}, base={self.transport.base!r})"

    # ------------------------------------------------------------------ objects
    def _pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._part_pool is None:
                self._part_pool = ThreadPoolExecutor(
                    self.concurrency, thread_name_prefix="singlet-part"
                )
            return self._part_pool

    def local_object(self, sm: SampleRef) -> Optional[str]:
        """Path of a locally stored copy of the sample object (a downloaded study or a directory mirror), if present."""
        if self.transport.is_local():
            p = self.transport.path(sm.key)  # type: ignore[attr-defined]
            return p if os.path.exists(p) else None
        p = (
            cache_root() / Path(*sm.key.split("/")[1:])
            if sm.key.startswith("v2/")
            else cache_root() / sm.key
        )
        try:
            if p.exists() and p.stat().st_size == sm.bytes:
                return str(p)
        except OSError:
            pass
        return None

    def source_for(self, sm: SampleRef) -> _Source:
        p = self.local_object(sm)
        if p is not None:
            return _MmapSource(p, sm.bytes)
        return _RemoteSource(self.transport, sm.key, sm.bytes, self._pool())

    # ------------------------------------------------------------------ gene maps
    def _gene_map(
        self,
        pack_hex: str,
        keep_map: Optional[np.ndarray],
        subset: Optional[np.ndarray] = None,
        key_extra=None,
    ) -> np.ndarray:
        """stored-agnostic canonical row -> final gene index (int32, -1 = dropped) for rows described by a pack."""
        k = (pack_hex, id(keep_map) if keep_map is not None else None, key_extra)
        with self._lock:
            v = self._maps.get(k)
        if v is not None:
            return v
        rg = self._resolver.row_gene(pack_hex)
        study_ids, _ = self.gene_table()
        pack_ids, _ = self._resolver.gene_table(pack_hex)
        if len(pack_ids) == len(study_ids) and (
            pack_ids is study_ids or bool(np.all(pack_ids == study_ids))
        ):
            to_study = None
        else:  # a sample on another vocabulary: translate by gene id
            idx = {g: i for i, g in enumerate(study_ids.tolist())}
            to_study = np.array([idx.get(g, -1) for g in pack_ids.tolist()], dtype=np.int32)
        m = (
            rg
            if to_study is None
            else np.where(rg >= 0, to_study[np.maximum(rg, 0)], -1).astype(np.int32)
        )
        if subset is not None:
            m = m[subset]
        if keep_map is not None:
            m = np.where(m >= 0, keep_map[np.maximum(m, 0)], -1).astype(np.int32)
        m = np.ascontiguousarray(m, dtype=np.int32)
        with self._lock:
            self._maps[k] = m
        return m

    # ------------------------------------------------------------------ high level
    def to_anndata(self, samples: Optional[Sequence[str]] = None, **kw) -> "anndata.AnnData":
        return _study_to_anndata(self, samples, **kw)

    def cells(self, where=None, **kw) -> "anndata.AnnData":
        return _study_cells(self, where, **kw)

    def request_plan(
        self, samples: Optional[Sequence[str]] = None, *, include_manifest: bool = True
    ) -> dict:
        """The requests and bytes a whole-sample ``load`` of these samples issues (no network access)."""
        gsms = list(samples) if samples is not None else self.gsm_ids
        reqs = 1 if include_manifest else 0
        total = 0
        per = []
        for g in gsms:
            sm = self.samples[g]
            rngs = [
                (m.off, m.len) for n in (CELLS, EXON, INTRON) if (m := sm.member(n)) is not None
            ]
            co = _coalesce(rngs)
            nb = sum(b - a for a, b, _ in co)
            per.append({"gsm": g, "requests": len(co), "bytes": nb, "object_bytes": sm.bytes})
            reqs += len(co)
            total += nb
        return {"requests": reqs, "bytes": total, "samples": per}

    def close(self) -> None:
        if self._part_pool is not None:
            self._part_pool.shutdown(wait=False)
            self._part_pool = None


# ================================================================================================================= cells table
def _open_cells(buf, resolver: _PackResolver):
    from .cells import CellsReader

    mv = memoryview(buf)
    return CellsReader(lambda off, ln: bytes(mv[off : off + ln]), len(mv), resolver.store)


# ================================================================================================================= per-sample read
@dataclass
class _Part:
    """One matrix read: genes x selected-cells CSC (== cells x genes CSR)."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n_genes: int

    @property
    def n_cells(self) -> int:
        return len(self.indptr) - 1

    def csr(self):
        import scipy.sparse as sp

        m = sp.csr_matrix(
            (self.data, self.indices, self.indptr), shape=(self.n_cells, self.n_genes), copy=False
        )
        m.has_sorted_indices = True
        m.has_canonical_format = True
        return m


@dataclass
class _SampleData:
    gsm: str
    sel: np.ndarray  # selected matrix columns, ascending
    cols: Dict[str, Any]  # cells-table columns restricted to sel (numpy / list)
    spliced: Optional[_Part]
    unspliced: Optional[_Part]
    n_cells_total: int
    skipped: Optional[str] = None


def _runs(
    sel: np.ndarray, chunk_first_col: np.ndarray, gap_chunks: int = 0
) -> List[Tuple[int, int]]:
    """Column ranges [c0, c1) aligned to chunk boundaries that cover ``sel`` (ascending). Chunks that touch (or are at most
    ``gap_chunks`` chunks apart) share a run."""
    if len(sel) == 0:
        return []
    ch = np.searchsorted(chunk_first_col, sel, side="right") - 1
    uniq = np.unique(ch)
    runs: List[List[int]] = []
    for c in uniq.tolist():
        if runs and c <= runs[-1][1] + 1 + gap_chunks:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return [(int(chunk_first_col[a]), int(chunk_first_col[b + 1])) for a, b in runs]


def _open_member(
    src: _Source, mref: MemberRef, *, whole: Optional[Any] = None, first: Optional[Any] = None
):
    """Open a pz2 member from a whole-member buffer, or from its first bytes (the binding fetches what the prefix lacks)."""
    pz = _native()
    if whole is not None:
        return pz.Member(whole)
    if first is None:
        first = src.read(mref.off, min(mref.len, PREFIX_HINT))
    fetch = lambda off, ln: src.read(mref.off + off, min(ln, mref.len - off))  # noqa: E731
    return pz.Member(first, fetch)


def _decode_part(
    member, c_lo: int, c_hi: int, sel: np.ndarray, gene_map: np.ndarray, n_genes: int, threads: int
) -> _Part:
    """Decode columns [c_lo, c_hi), keep ``sel`` (absolute, ascending, inside the range), aggregate to genes."""
    if n_genes == 0:
        return _Part(
            np.zeros(len(sel) + 1, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            0,
        )
    rel = (sel - c_lo).astype(np.uint32) if len(sel) != (c_hi - c_lo) else None
    r = member.decode(
        c_lo, c_hi, mode="gene", threads=threads, cols=rel, gene_map=gene_map, n_genes=n_genes
    )
    return _Part(r["indptr"], r["indices"], r["data"], n_genes)


def _concat_parts(parts: List[_Part]) -> _Part:
    if len(parts) == 1:
        return parts[0]
    n_genes = parts[0].n_genes
    nnz = sum(int(p.indptr[-1]) for p in parts)
    ncell = sum(p.n_cells for p in parts)
    indptr = np.empty(ncell + 1, dtype=np.int64 if nnz >= 2**31 else np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.int32)
    indptr[0] = 0
    c = z = 0
    for p in parts:
        k = p.n_cells
        z_end = z + int(p.indptr[-1])
        indptr[c + 1 : c + k + 1] = p.indptr[1:] + z
        indices[z:z_end] = p.indices
        data[z:z_end] = p.data
        c += k
        z = z_end
    return _Part(indptr, indices, data, n_genes)


def _sample_columns(cells_buf, resolver: _PackResolver, want: Sequence[str]):
    rd = _open_cells(cells_buf, resolver)
    return rd, {n: rd.column(n) for n in want if n in rd.entries}


def _read_sample(
    study: Study,
    sm: SampleRef,
    *,
    sel_fn: Callable[[Any], np.ndarray],
    want_cols: Sequence[str],
    keep_map: Optional[np.ndarray],
    n_genes: int,
    threads: int,
    want_exon: bool = True,
    want_intron: bool = True,
    parallel_parts: bool = False,
    whole_max: int = WHOLE_MAX_BYTES,
    verify: bool = False,
    src: Optional[_Source] = None,
) -> _SampleData:
    """Fetch and decode what one sample contributes.

    ``sel_fn(cells_reader) -> ascending int array`` of the matrix columns to keep (e.g. the called cells)."""
    own_src = src is None
    src = src or study.source_for(sm)
    try:
        cm, em, im = sm.member(CELLS), sm.member(EXON), sm.member(INTRON)
        if cm is None:
            raise V2Error(f"{sm.gsm}: the sample object has no cells table")
        if em is None or em.len == 0:
            return _SampleData(
                sm.gsm,
                np.zeros(0, np.int64),
                {},
                None,
                None,
                0,
                skipped="exon_counts.1pz is missing",
            )
        if not want_intron:
            im = None
        total = cm.len + em.len + (im.len if im else 0)
        whole = src.local or total <= whole_max
        bufs: Dict[str, Any] = {}
        prefixes: Dict[str, Any] = {}
        if whole:
            names = [CELLS, EXON] + ([INTRON] if im else [])
            bl = src.read_ranges(
                [(sm.members[n].off, sm.members[n].len) for n in names], parallel=parallel_parts
            )
            bufs = dict(zip(names, bl))
            cbuf = bufs[CELLS]
        else:
            # phase 1: the cells table and the prefix (header, chunk index, tables, rows) of each matrix, concurrently
            names = [CELLS] + [n for n, r in ((EXON, em), (INTRON, im)) if r is not None]
            bl = src.read_ranges(
                [
                    (
                        sm.members[n].off,
                        min(sm.members[n].len, PREFIX_HINT) if n != CELLS else sm.members[n].len,
                    )
                    for n in names
                ],
                parallel=True,
                gap=PARTIAL_GAP,
            )
            prefixes = dict(zip(names, bl))
            cbuf = prefixes[CELLS]
        rd, cols = _sample_columns(cbuf, study._resolver, want_cols)
        n_rows = rd.n_rows
        if n_rows == 0:
            return _SampleData(
                sm.gsm, np.zeros(0, np.int64), {}, None, None, 0, skipped="count matrix has 0 cells"
            )
        sel = np.asarray(sel_fn(rd), dtype=np.int64)
        if len(sel) == 0:
            return _SampleData(sm.gsm, sel, {}, None, None, n_rows, skipped="0 cells called")
        if len(sel) and (
            sel[0] < 0 or sel[-1] >= n_rows or (len(sel) > 1 and not (np.diff(sel) > 0).all())
        ):
            raise V2Error(f"{sm.gsm}: column selection must be ascending and inside the matrix")
        cols = {
            k: (v[sel] if isinstance(v, np.ndarray) else np.asarray(v, dtype=object)[sel])
            for k, v in cols.items()
        }

        parts: Dict[str, Optional[_Part]] = {"exon": None, "intron": None}
        opened = []
        for tag, mref, name in (("exon", em, EXON), ("intron", im, INTRON)):
            if mref is None or mref.len == 0:
                continue
            if whole:
                mem = _open_member(src, mref, whole=bufs[name])
            else:
                mem = _open_member(src, mref, first=prefixes[name])
            opened.append((tag, mref, name, mem))
        # phase 2 (partial plans): the chunk runs that cover the wanted cells, all matrices at once
        run_bufs: Dict[str, Any] = {}
        plans: Dict[str, Any] = {}
        if not whole:
            want: List[Tuple[int, int]] = []
            owners: List[Tuple[str, int]] = []
            for tag, mref, name, mem in opened:
                first_col = mem.chunk_index()[:, 2].astype(np.int64)
                runs = _runs(sel, first_col)
                pl = [mem.plan(a, b) for a, b in runs]
                plans[name] = (runs, pl)
                for k, p in enumerate(pl):
                    if mem.has(
                        p["byte_off"], p["byte_len"]
                    ):  # already inside the prefix read (small members)
                        continue
                    want.append((mref.off + p["byte_off"], p["byte_len"]))
                    owners.append((name, k))
            got = src.read_ranges(want, parallel=True, gap=PARTIAL_GAP) if want else []
            for (name, k), buf in zip(owners, got):
                run_bufs[(name, k)] = buf
        for tag, mref, name, mem in opened:
            hdr = mem.header
            if hdr["n"] != n_rows:
                raise V2Error(
                    f"{sm.gsm}: {name} has {hdr['n']} columns, the cells table has {n_rows}"
                )
            if hdr["m"] == 0:
                continue
            if mref.digest and hdr["digest"] != str(mref.digest).lower():
                raise V2Error(
                    f"{sm.gsm}: {name} header digest {hdr['digest']} != manifest {mref.digest}"
                )
            gmap = _member_gene_map(study, mem, hdr, keep_map)
            if gmap is None:
                continue
            c_hi = int(sel[-1]) + 1
            if whole:
                parts[tag] = _decode_part(mem, 0, c_hi, sel, gmap, n_genes, threads)
            else:
                runs, pl = plans[name]
                pieces = []
                for k, ((a, b), p) in enumerate(zip(runs, pl)):
                    if (name, k) in run_bufs:
                        mem.add_range(p["byte_off"], run_bufs[(name, k)])
                    s_run = sel[(sel >= a) & (sel < b)]
                    pieces.append(_decode_part(mem, a, b, s_run, gmap, n_genes, threads))
                parts[tag] = _concat_parts(pieces) if pieces else None
            if verify:
                _verify_matrix(study, sm, name, mref, mem, hdr)
            del mem
        return _SampleData(sm.gsm, sel, cols, parts["exon"], parts["intron"], n_rows)
    finally:
        if own_src:
            src.close()


def _member_gene_map(
    study: Study, mem, hdr: dict, keep_map: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    kind = hdr["row_names_kind"]
    m = hdr["m"]
    if kind == "pack":
        pack = hdr["row_names_id"]
        gm = study._gene_map(pack, keep_map)
        if len(gm) != m:
            raise V2Error(f"vocabulary pack {pack} has {len(gm)} rows, the matrix has {m}")
        return gm
    if kind == "pack_subset":
        info = mem.row_names()
        idx = np.asarray(info["pack_index"], dtype=np.int64)
        return study._gene_map(
            info["pack_id"], keep_map, subset=idx, key_extra=hashlib.md5(idx.tobytes()).hexdigest()
        )
    # inline names: map by the gene id before the first underscore (the legacy rule)
    info = mem.row_names()
    names = info["names"]
    study_ids, _ = study.gene_table()
    gidx = {g: i for i, g in enumerate(study_ids.tolist())}
    gm = np.array([gidx.get(n.split("_")[0], -1) for n in names], dtype=np.int32)
    if keep_map is not None:
        gm = np.where(gm >= 0, keep_map[np.maximum(gm, 0)], -1).astype(np.int32)
    return gm


def _verify_matrix(study: Study, sm: SampleRef, name: str, mref: MemberRef, mem, hdr: dict) -> None:
    """Recompute the canonical digest of the full matrix and compare with the manifest (slow: full decode + name hashing)."""
    pz = _native()
    if hdr["row_names_kind"] != "pack" or hdr["col_names_kind"] not in ("cells", "pack"):
        return
    r = mem.decode(mode="canonical", threads=0, dtype="native")
    rows = study._resolver.get(hdr["row_names_id"]).names()
    rd = _open_cells(
        study.source_for(sm).read(sm.members[CELLS].off, sm.members[CELLS].len), study._resolver
    )
    cols = rd.column("barcode")
    d = pz.digest(
        r["indptr"],
        r["indices"],
        r["data"],
        hdr["m"],
        hdr["n"],
        value_type=hdr["value_type"],
        row_names=rows,
        col_names=cols,
        kv=mem.kv(),
    )
    want = str(mref.digest).lower() if mref.digest else hdr["digest"]
    if d.hex() != want:
        raise V2Error(f"{sm.gsm}: {name}: canonical digest {d.hex()} != {want}")


# ================================================================================================================= study -> AnnData
_OBS_STUDY_COLS = (
    "organism",
    "protocol",
    "protocol_name",
    "sample_source",
    "sample_characteristics",
    "qc_flag",
    "reference_build",
)


def _gene_keep(study: Study, genes: Optional[Sequence[str]]):
    """-> (keep_map int32 or None, n_genes, var_ids, var_names) for the requested genes (None = all)."""
    ids, names = study.gene_table()
    if genes is None:
        return None, len(ids), ids, names
    from singlet._loader import _gene_keep_mask

    keep = _gene_keep_mask([str(x) for x in ids], [str(x) for x in names], genes)
    if not keep.any():
        return np.full(len(ids), -1, dtype=np.int32), 0, ids[:0], names[:0]
    new = np.cumsum(keep).astype(np.int32) - 1
    kmap = np.where(keep, new, -1).astype(np.int32)
    return kmap, int(keep.sum()), ids[keep], names[keep]


def _sample_obs_value(meta: dict, key: str, default=""):
    v = meta.get(key, default)
    return default if v is None else v


def _select_samples(
    study: Study, samples: Optional[Sequence[str]], obs_filter: Optional[dict]
) -> List[str]:
    gsms = list(study.gsm_ids) if samples is None else [str(s).strip().upper() for s in samples]
    missing = [g for g in gsms if g not in study.samples]
    if missing:
        raise KeyError(
            f"sample(s) {missing[:5]} not in {study.gse} (has {len(study.samples)}: {study.gsm_ids[:5]}...)"
        )
    if obs_filter:
        meta = study.study_meta.get("gsm_meta", {})
        keep = []
        for g in gsms:
            gm = meta.get(g, {})
            ok = True
            for col, val in obs_filter.items():
                if col == "gsm_id":
                    cur = g
                elif col in _OBS_STUDY_COLS:
                    cur = gm.get(col, "")
                elif col == "n_cells_sample":
                    cur = gm.get("n_cells", None)
                else:
                    ok = True  # a per-cell column: decided after loading
                    continue
                ok = ok and ((cur in val) if isinstance(val, list) else (cur == val))
            if ok:
                keep.append(g)
        gsms = keep
    return gsms


def _cell_order(cols: Dict[str, Any], sel: np.ndarray, order: str) -> Optional[np.ndarray]:
    """Permutation of the selected cells for the requested ``order`` (None = keep matrix column order)."""
    if order == "column" or len(sel) < 2:
        return None
    bc = cols.get("barcode")
    if order == "barcode" or order == "legacy":
        if bc is None:
            return None
        if order == "legacy":
            if "cc_pos" in cols:
                return np.argsort(np.asarray(cols["cc_pos"]), kind="stable")
            if "cc_listed" in cols or "cc_pvalue" in cols:
                return None  # emptydrops writes ascending column order
        return np.argsort(np.asarray(bc, dtype=object), kind="stable")
    raise ValueError("cell_order must be 'legacy', 'barcode' or 'column'")


def _vstack_parts(mats: List[_Part], n_genes: int, perms: List[Optional[np.ndarray]]) -> "Any":
    """Stack per-sample CSR parts into one CSR (rows permuted per sample when asked), allocating the result once."""
    import scipy.sparse as sp

    nnz = sum(int(p.indptr[-1]) for p in mats)
    ncell = sum(p.n_cells for p in mats)
    indptr = np.empty(ncell + 1, dtype=np.int64 if nnz >= 2**31 else np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.int32)
    indptr[0] = 0
    c = z = 0
    for p, perm in zip(mats, perms):
        k = p.n_cells
        pn = int(p.indptr[-1])
        if perm is None:
            indptr[c + 1 : c + k + 1] = p.indptr[1:] + z
            indices[z : z + pn] = p.indices
            data[z : z + pn] = p.data
        else:
            lens = np.diff(p.indptr)[perm]
            indptr[c + 1 : c + k + 1] = np.cumsum(lens) + z
            # gather rows in the new order
            starts = p.indptr[:-1][perm]
            tot = int(lens.sum())
            if tot:
                row_of = np.repeat(np.arange(k), lens)
                off_in_row = np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens)
                src = starts[row_of] + off_in_row
                indices[z : z + tot] = p.indices[src]
                data[z : z + tot] = p.data[src]
        c += k
        z += pn
    m = sp.csr_matrix((data, indices, indptr), shape=(ncell, n_genes), copy=False)
    m.has_sorted_indices = True
    m.has_canonical_format = True
    return m


def _study_to_anndata(
    study: Study,
    samples: Optional[Sequence[str]] = None,
    *,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    layers: bool = True,
    cell_order: str = "legacy",
    concurrency: Optional[int] = None,
    threads: Optional[int] = None,
    verify: Optional[bool] = None,
    whole_max: int = WHOLE_MAX_BYTES,
    verbose: bool = False,
) -> "anndata.AnnData":
    import anndata as ad
    import pandas as pd

    gsms = _select_samples(study, samples, obs_filter)
    if not gsms:
        raise KeyError(f"no sample of {study.gse} matches the request")
    kmap, n_genes, var_ids, var_names = _gene_keep(study, genes)
    conc = max(1, min(concurrency or study.concurrency, len(gsms)))
    ncpu = os.cpu_count() or 4
    dec_threads = threads if threads is not None else max(1, ncpu // conc)
    do_verify = study.verify if verify is None else verify

    def work(g: str) -> _SampleData:
        sm = study.samples[g]
        return _read_sample(
            study,
            sm,
            sel_fn=lambda rd: np.flatnonzero(np.asarray(rd.column("called"), dtype=bool)),
            want_cols=("barcode", "called", "cc_pos", "cc_listed", "cc_pvalue"),
            keep_map=kmap,
            n_genes=n_genes,
            threads=dec_threads,
            parallel_parts=(conc == 1),
            whole_max=whole_max,
            verify=do_verify,
        )

    t0 = time.time()
    if conc == 1:
        results = [work(g) for g in gsms]
    else:
        with ThreadPoolExecutor(conc, thread_name_prefix="singlet-sample") as ex:
            results = list(ex.map(work, gsms))
    if verbose:
        print(
            f"  [singlet.v2] fetched+decoded {len(gsms)} sample(s) in {time.time() - t0:.2f}s ({study.stats.requests} requests, {study.stats.bytes:,} bytes)"
        )

    skipped = []
    ok: List[_SampleData] = []
    for r in results:
        if r.skipped or r.spliced is None and r.unspliced is None:
            reason = r.skipped or "count matrix has no usable features"
            skipped.append({"gsm_id": r.gsm, "reason": reason})
            warnings.warn(f"{r.gsm} skipped: {reason} (in {study.gse})", UserWarning, stacklevel=3)
        else:
            ok.append(r)
    if not ok:
        reasons = "; ".join(f"{s['gsm_id']}: {s['reason']}" for s in skipped[:5])
        raise NoUsableSamples(
            f"{study.gse} has no usable samples: none of its {len(skipped)} sample(s) has a non-empty count matrix with called cells "
            f"({reasons}). See https://singlet.bio/study/{study.gse} for the study's current status.",
            skipped,
        )

    perms = [_cell_order(r.cols, r.sel, cell_order) for r in ok]
    xs: List[_Part] = []
    spl: List[Optional[_Part]] = []
    uns_: List[Optional[_Part]] = []
    for r in ok:
        a, b = r.spliced, r.unspliced
        spl.append(a)
        uns_.append(b)
        if a is not None and b is not None:
            xm = a.csr() + b.csr()
            xs.append(_Part(xm.indptr, xm.indices, xm.data.astype(np.int32, copy=False), n_genes))
        else:
            xs.append(a if a is not None else b)  # type: ignore[arg-type]
    X = _vstack_parts(xs, n_genes, perms)
    xs.clear()

    gsm_meta = study.study_meta.get("gsm_meta", {})
    obs_frames = []
    for r, perm in zip(ok, perms):
        n = len(r.sel)
        bc = np.asarray(r.cols["barcode"], dtype=object)
        if perm is not None:
            bc = bc[perm]
        info = gsm_meta.get(r.gsm, {})
        idx = np.char.add(f"{r.gsm}_", bc.astype(str)) if n else np.array([], dtype=str)
        d = {"gsm_id": np.full(n, r.gsm, dtype=object)}
        for c in (
            "organism",
            "protocol",
            "protocol_name",
            "sample_source",
            "sample_characteristics",
            "qc_flag",
            "reference_build",
        ):
            d[c] = np.full(n, _sample_obs_value(info, c), dtype=object)
        d["n_cells_sample"] = np.full(
            n, info.get("n_cells", n) if info.get("n_cells", n) is not None else n, dtype=object
        )
        obs_frames.append(pd.DataFrame(d, index=pd.Index(idx.astype(object))))
    obs = pd.concat(obs_frames, axis=0) if len(obs_frames) > 1 else obs_frames[0]
    var = pd.DataFrame(
        {"gene_name": var_names.astype(object)},
        index=pd.Index(var_ids.astype(object), name="gene_id"),
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if layers:
        for lname, plist in (("spliced", spl), ("unspliced", uns_)):
            if all(p is not None for p in plist):
                adata.layers[lname] = _vstack_parts(plist, n_genes, perms)  # type: ignore[arg-type]
            elif verbose:
                print(f"  [singlet.v2] {lname} layer unavailable for some samples; skipped")
    spl.clear(), uns_.clear()
    _attach_uns(adata, study, [r.gsm for r in ok], skipped)
    if obs_filter:
        adata = _apply_obs_filter(adata, obs_filter)
    return adata


def _apply_obs_filter(adata, obs_filter: dict):
    mask = True
    for col, val in obs_filter.items():
        if col not in adata.obs.columns:
            raise KeyError(
                f"obs_filter column {col!r} not found. Available: {list(adata.obs.columns)}"
            )
        mask = mask & (
            adata.obs[col].isin(val) if isinstance(val, list) else (adata.obs[col] == val)
        )
    return adata[mask].copy()


def _attach_uns(adata, study: Study, gsms: List[str], skipped: list) -> None:
    import pandas as pd

    from singlet.bundle import _manifest_for_uns, _uns_safe

    adata.uns["study_meta"] = _uns_safe(study.study_meta)
    man = {
        "schema_version": "2.0",
        "gse_id": study.gse,
        "gsm_ids": list(gsms),
        "reference_build": study.reference_build,
        "n_gsms": len(gsms),
        "created_at": study.doc.get("created_at"),
        "singlet_version": (study.doc.get("pipeline") or {}).get("singlet_version")
        if isinstance(study.doc.get("pipeline"), dict)
        else None,
        "format": "singlet-v2",
        "rev": study.rev,
        "profile": "public",
    }
    adata.uns["manifest"] = _manifest_for_uns(man)
    adata.uns["singlet_bundle_path"] = study.transport.url(f"v2/m/{study.gse}.json")
    adata.uns["singlet_v2"] = {"gse": study.gse, "rev": study.rev, "base": study.transport.base}
    if skipped:
        adata.uns["skipped_samples"] = pd.DataFrame(skipped, columns=["gsm_id", "reason"])


# ================================================================================================================= cells()
def _eval_where(where, df):
    """Boolean mask over the rows of ``df`` from a pandas query string, a callable, or a dict {col: value | (lo, hi) | list}."""
    if where is None:
        return np.ones(len(df), dtype=bool)
    if callable(where):
        return np.asarray(where(df), dtype=bool)
    if isinstance(where, str):
        return np.asarray(df.eval(where), dtype=bool)
    if isinstance(where, dict):
        m = np.ones(len(df), dtype=bool)
        for col, cond in where.items():
            if col not in df.columns:
                raise KeyError(f"cells table has no column {col!r}; columns: {list(df.columns)}")
            v = df[col].to_numpy()
            if isinstance(cond, tuple) and len(cond) == 2:
                lo, hi = cond
                if lo is not None:
                    m &= v >= lo
                if hi is not None:
                    m &= v <= hi
            elif isinstance(cond, (list, set, frozenset)):
                m &= np.isin(v, list(cond))
            else:
                m &= v == cond
        return m
    raise TypeError("where must be a pandas query string, a callable(df) -> mask, or a dict")


def _study_cells(
    study: Study,
    where=None,
    *,
    samples: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
    genes: Optional[Sequence[str]] = None,
    layers: bool = False,
    max_cells: Optional[int] = None,
    concurrency: Optional[int] = None,
    threads: Optional[int] = None,
    verify: Optional[bool] = None,
    whole_max: int = WHOLE_MAX_BYTES,
    called_only: bool = False,
) -> "anndata.AnnData":
    """Cell-level slice: the cells of ``samples`` that satisfy ``where`` (evaluated on the cells table), as a cells x genes AnnData.

    Only the cells tables of the samples and the chunk runs of the matrices that contain the selected cells are fetched when the
    samples are large (small samples are fetched whole in one request). ``where`` is a pandas query over the cells-table columns
    (``called``, ``n_umi_exon``, ``n_umi_intron``, ``n_feat_exon``, ``n_feat_intron``, ...), a callable on that DataFrame, or a dict
    ``{column: value | (lo, hi) | [values]}``. ``max_cells`` keeps the first cells per sample in matrix-column order (largest first)."""
    import anndata as ad
    import pandas as pd

    gsms = _select_samples(study, samples, None)
    kmap, n_genes, var_ids, var_names = _gene_keep(study, genes)
    conc = max(1, min(concurrency or study.concurrency, len(gsms)))
    dec_threads = threads if threads is not None else max(1, (os.cpu_count() or 4) // conc)
    do_verify = study.verify if verify is None else verify

    def sel_fn(rd):
        df = _cells_frame(rd, columns=None)
        m = _eval_where(where, df)
        if called_only and "called" in df.columns:
            m &= df["called"].to_numpy(dtype=bool)
        idx = np.flatnonzero(m)
        if max_cells is not None:
            idx = idx[:max_cells]
        return idx

    def work(g: str) -> _SampleData:
        return _read_sample(
            study,
            study.samples[g],
            sel_fn=sel_fn,
            want_cols=_cells_wanted(columns),
            keep_map=kmap,
            n_genes=n_genes,
            threads=dec_threads,
            parallel_parts=(conc == 1),
            whole_max=whole_max,
            verify=do_verify,
        )

    if conc == 1:
        results = [work(g) for g in gsms]
    else:
        with ThreadPoolExecutor(conc, thread_name_prefix="singlet-sample") as ex:
            results = list(ex.map(work, gsms))
    ok = [r for r in results if r.spliced is not None or r.unspliced is not None]
    if not ok:
        return ad.AnnData(
            X=np.zeros((0, n_genes), dtype=np.int32),
            var=pd.DataFrame(
                {"gene_name": var_names.astype(object)},
                index=pd.Index(var_ids.astype(object), name="gene_id"),
            ),
        )
    xs, spl, uns_ = [], [], []
    for r in ok:
        a, b = r.spliced, r.unspliced
        spl.append(a)
        uns_.append(b)
        if a is not None and b is not None:
            xm = a.csr() + b.csr()
            xs.append(_Part(xm.indptr, xm.indices, xm.data.astype(np.int32, copy=False), n_genes))
        else:
            xs.append(a if a is not None else b)  # type: ignore[arg-type]
    X = _vstack_parts(xs, n_genes, [None] * len(xs))
    frames = []
    for r in ok:
        n = len(r.sel)
        bc = np.asarray(r.cols.get("barcode", np.array([""] * n, dtype=object)), dtype=object)
        d = {"gsm_id": np.full(n, r.gsm, dtype=object), "column": r.sel.astype(np.int64)}
        for k, v in r.cols.items():
            if k == "barcode" or (columns is not None and k not in columns):
                continue
            d[k] = np.asarray(v)
        frames.append(
            pd.DataFrame(d, index=pd.Index(np.char.add(f"{r.gsm}_", bc.astype(str)).astype(object)))
        )
    obs = pd.concat(frames, axis=0) if len(frames) > 1 else frames[0]
    var = pd.DataFrame(
        {"gene_name": var_names.astype(object)},
        index=pd.Index(var_ids.astype(object), name="gene_id"),
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if layers and all(p is not None for p in spl) and all(p is not None for p in uns_):
        adata.layers["spliced"] = _vstack_parts(spl, n_genes, [None] * len(spl))  # type: ignore[arg-type]
        adata.layers["unspliced"] = _vstack_parts(uns_, n_genes, [None] * len(uns_))  # type: ignore[arg-type]
    adata.uns["singlet_v2"] = {"gse": study.gse, "rev": study.rev, "base": study.transport.base}
    return adata


def _cells_wanted(columns: Optional[Sequence[str]]) -> Sequence[str]:
    base = ["barcode", "called", "n_umi_exon", "n_umi_intron", "n_feat_exon", "n_feat_intron"]
    return base if columns is None else list(dict.fromkeys(base[:1] + list(columns)))


def _cells_frame(rd, columns: Optional[Sequence[str]]):
    """Numeric / boolean columns of an opened cells table as a DataFrame (strings and barcodes excluded)."""
    import pandas as pd

    names = (
        [n for n in rd.names if n != "barcode"]
        if columns is None
        else [n for n in columns if n in rd.entries]
    )
    cols = {}
    for n in names:
        if rd.entries[n][1] in (8, 9):  # string / barcode
            continue
        cols[n] = np.asarray(rd.column(n))
    return pd.DataFrame(cols)


# ================================================================================================================= public functions
def load(
    source: str,
    *,
    genes: Optional[Sequence[str]] = None,
    obs_filter: Optional[dict] = None,
    samples: Optional[Sequence[str]] = None,
    layers: bool = True,
    force: bool = False,
    base: Optional[str] = None,
    **kw,
) -> "anndata.AnnData":
    """Load a study (``GSE...``), one sample (``GSM...``, resolved to its parent study), a downloaded ``.s2`` sample object, or a
    manifest ``.json`` path as an AnnData of the called cells at gene level (``X = spliced + unspliced``)."""
    s = str(source).strip()
    p = Path(s).expanduser()
    if p.suffix.lower() == ".s2" and p.exists():
        return load_s2(p, genes=genes, layers=layers, **kw)
    if p.suffix.lower() == ".json" and p.exists():
        root, gse = _manifest_root(p)
        st = open_study(gse, transport=T.DirTransport(root), force=True)
        return st.to_anndata(samples, genes=genes, obs_filter=obs_filter, layers=layers, **kw)
    from singlet._integrity import normalize_accession

    acc = normalize_accession(s)
    gsm_filter = None
    if acc.startswith("GSM"):
        from singlet._loader import _resolve_gsm_to_gse

        parent = _resolve_gsm_to_gse(acc)
        if parent is None:
            raise FileNotFoundError(
                f"Could not resolve parent GSE for sample {acc!r} via the public API. Load the parent series directly."
            )
        gsm_filter, acc = acc, parent
        samples = [gsm_filter] if samples is None else samples
    st = open_study(acc, base=base, force=force)
    return st.to_anndata(samples, genes=genes, obs_filter=obs_filter, layers=layers, **kw)


def cells(
    source: str,
    where=None,
    *,
    samples: Optional[Sequence[str]] = None,
    base: Optional[str] = None,
    force: bool = False,
    **kw,
):
    """Cell-level slice of a study (or of the parent study of a GSM): see :meth:`Study.cells`."""
    from singlet._integrity import normalize_accession

    acc = normalize_accession(str(source))
    if acc.startswith("GSM"):
        from singlet._loader import _resolve_gsm_to_gse

        parent = _resolve_gsm_to_gse(acc)
        if parent is None:
            raise FileNotFoundError(f"Could not resolve parent GSE for sample {acc!r}")
        samples = [acc] if samples is None else samples
        acc = parent
    return open_study(acc, base=base, force=force).cells(where, samples=samples, **kw)


def _manifest_root(p: Path) -> Tuple[str, str]:
    """``<root>/v2/m/<GSE>.json`` -> (root, GSE)."""
    p = p.resolve()
    if p.parent.name != "m" or p.parent.parent.name != "v2":
        raise V2Error(f"{p} is not under a v2/m/ directory")
    return str(p.parent.parent.parent), p.stem


def load_s2(
    path: Union[str, os.PathLike],
    *,
    genes: Optional[Sequence[str]] = None,
    layers: bool = True,
    threads: Optional[int] = None,
    verify: bool = False,
    **kw,
) -> "anndata.AnnData":
    """AnnData of the called cells of a local ``.s2`` sample object (memory mapped, no network, no copies of the file)."""
    import anndata as ad
    import pandas as pd

    from .container import S2Reader

    path = str(path)
    rd = S2Reader(path)
    members = {
        n: MemberRef(
            n, mi.kind, mi.off, mi.len, mi.digest.hex() if mi.digest else None, None, None, mi.meta
        )
        for n, mi in rd.members.items()
    }
    sm = SampleRef(rd.sample_id, "", rd.file_len, None, None, None, members)
    exon = members.get(EXON)
    if exon is None:
        raise V2Error(f"{path} has no {EXON}")
    t = T.DirTransport(os.path.dirname(os.path.abspath(path)) or ".")
    # a throwaway Study that only supplies packs and the gene axis
    pack_hex = None
    pz = _native()
    src = _MmapSource(path, rd.file_len)
    try:
        mem = pz.Member(src.read(exon.off, exon.len))
        h = mem.header
        pack_hex = h["row_names_id"]
        del mem
        doc = {
            "format": "singlet-v2",
            "manifest_version": 1,
            "gse": "GSE0",
            "rev": "",
            "packs": {"exon": pack_hex},
            "samples": [],
        }
        st = Study(doc, t, {"origin": "s2"}, concurrency=1, verify=verify)
        kmap, n_genes, var_ids, var_names = _gene_keep(st, genes)
        sd = _read_sample(
            st,
            sm,
            sel_fn=lambda r: np.flatnonzero(np.asarray(r.column("called"), dtype=bool)),
            want_cols=("barcode", "called", "cc_pos", "cc_listed", "cc_pvalue"),
            keep_map=kmap,
            n_genes=n_genes,
            threads=threads if threads is not None else 0,
            src=src,
            verify=verify,
        )
    finally:
        src.close()
    if sd.skipped:
        raise ValueError(f"{sm.gsm} has no usable cells: {sd.skipped}")
    perm = _cell_order(sd.cols, sd.sel, kw.get("cell_order", "legacy"))
    a, b = sd.spliced, sd.unspliced
    if a is not None and b is not None:
        xm = a.csr() + b.csr()
        xp = _Part(xm.indptr, xm.indices, xm.data.astype(np.int32, copy=False), n_genes)
    else:
        xp = a if a is not None else b
    X = _vstack_parts([xp], n_genes, [perm])  # type: ignore[list-item]
    bc = np.asarray(sd.cols["barcode"], dtype=object)
    if perm is not None:
        bc = bc[perm]
    n = len(sd.sel)
    obs = pd.DataFrame(
        {"gsm_id": np.full(n, sm.gsm, dtype=object)},
        index=pd.Index(np.char.add(f"{sm.gsm}_", bc.astype(str)).astype(object)),
    )
    var = pd.DataFrame(
        {"gene_name": var_names.astype(object)},
        index=pd.Index(var_ids.astype(object), name="gene_id"),
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if layers and a is not None and b is not None:
        adata.layers["spliced"] = _vstack_parts([a], n_genes, [perm])
        adata.layers["unspliced"] = _vstack_parts([b], n_genes, [perm])
    adata.uns["singlet_v2"] = {"sample_object": path}
    return adata


def download_study(
    gse: str,
    output_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    samples: Optional[Sequence[str]] = None,
    force: bool = False,
    base: Optional[str] = None,
    concurrency: int = 4,
    verify: bool = True,
) -> Path:
    """Mirror a study for offline use: the manifest, the sample objects (whole-object sha256 verified) and the packs they need.

    The layout is the server's (``<root>/v2/m/<GSE>.json``, ``<root>/v2/s/<GSM>/<d>.s2``); returns the manifest path. ``load``
    prefers the local copies in the default cache directory automatically."""
    st = open_study(gse, base=base, force=force)
    root = (Path(output_dir) if output_dir else cache_root().parent) / "v2"
    gsms = list(samples) if samples is not None else st.gsm_ids

    def one(g: str) -> None:
        sm = st.samples[g]
        dest = root / Path(*sm.key.split("/")[1:])
        if dest.exists() and dest.stat().st_size == sm.bytes and not force:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = _RemoteSource(st.transport, sm.key, sm.bytes, st._pool())
        tmp = dest.with_suffix(dest.suffix + f".part{os.getpid()}")
        h = hashlib.sha256()
        with open(tmp, "wb") as fh:
            pos, step = 0, 64 << 20
            while pos < sm.bytes:
                n = min(step, sm.bytes - pos)
                buf = src.read(pos, n, parallel=True)
                fh.write(buf)
                h.update(buf)
                pos += n
        if verify and sm.sha256 and h.hexdigest() != str(sm.sha256).lower():
            tmp.unlink(missing_ok=True)
            raise V2Error(f"{g}: sha256 of the downloaded object does not match the manifest")
        os.replace(tmp, dest)

    with ThreadPoolExecutor(max(1, min(concurrency, len(gsms)))) as ex:
        list(ex.map(one, gsms))
    mp = root / "m" / f"{st.gse}.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(st.doc), encoding="utf-8")
    for pid in {v for v in st.packs.values() if v}:
        try:
            st._resolver.get(pid)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"pack {pid} could not be fetched: {e}", RuntimeWarning)
    return mp
