# SPDX-License-Identifier: MIT
"""Stream cells from the Singlet atlas into PyTorch by HTTP range reads, without downloading the atlas.

:class:`AtlasCells` is an ``IterableDataset`` over the cells of any metadata or ontology filter. The atlas cell index
(``POST data.singlet.bio/s1/cidx/query``, :func:`singlet.s1.atlas.count_cells`) picks the samples that hold matching cells, or a
cohort manifest (``/api/manifest`` TSV, a list of accessions, a DataFrame) names them. Each sample's ``.s1`` object is then read
with byte ranges (only the cells table and the matrix chunks that hold the wanted cells), the cell-level filters are applied to
its cells table, and the cells are mapped onto one fixed gene axis.

Splits are held out by study: a study's split is a hash of ``(split_seed, GSE)``, so it is the same on every machine, worker and
epoch, and no study contributes cells to two splits. Shuffling works at two levels: the sample order is a fresh permutation each
epoch (``set_epoch``), and every batch draws cells from ``mix_samples`` samples held open at once (each in a random cell order),
so a batch mixes studies instead of being one sample. Samples are read ``prefetch`` at a time in background threads, and their
range requests are sent together, at most ``io_threads`` in flight for the dataset (``"auto"``: 8 over HTTP, 1 from a local
directory).

The labels in each batch are the Research Alpha per-cell labels (Cell Ontology path, chromosomal sex, genetic ancestry):
predictions of the Singlet labeller, whose own per-study accuracy is documented in :mod:`singlet.s1.labels`. They are atlas
metadata, not author annotations.

Example::

    from torch.utils.data import DataLoader
    from singlet.torch import AtlasCells

    ds = AtlasCells({"organism": "Homo sapiens", "tissue": "UBERON:0002048"}, split="train", batch_size=512)
    for batch in DataLoader(ds, batch_size=None, num_workers=2):
        x, y = batch["X"], batch["cell_type_l2"]
"""

from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:  # the module is importable without torch for planning and the split helpers
    from torch.utils.data import IterableDataset as _Base
except ImportError:  # pragma: no cover
    _Base = object  # type: ignore[assignment,misc]

SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
#: filter keys the cell index applies to samples (every cell of a selected sample satisfies them)
SAMPLE_KEYS = (
    "organism",
    "tissue",
    "disease",
    "development_stage",
    "donor_status",
    "perturbation",
    "sample_kind",
    "categories",
    "samples",
    "studies",
)
#: filter keys that are properties of the cell; applied again to each sample's cells table
CELL_KEYS = (
    "cell_type",
    "sex",
    "ancestry",
    "min_umi",
    "max_umi",
    "min_genes",
    "max_genes",
    "min_cell_type_score",
)
#: the cells-table columns a batch needs
_OBS = (
    "cell_type_path",
    "cell_type_deepest_score",
    "sex_chrom",
    "ancestry_hancestro",
    "n_umi_exon",
    "n_umi_intron",
    "n_feat_exon",
    "called",
)
_INDEX_GROUP_CAP = 10_000  # the Worker returns at most this many groups per count request
_SEX = {
    "female": "female",
    "male": "male",
    "PATO:0000383": "female",
    "PATO:0000384": "male",
    "ambiguous": "ambiguous",
}


# ================================================================================================================= splits
def study_split(gse: str, fractions: Sequence[float] = DEFAULT_FRACTIONS, seed: int = 0) -> str:
    """The split (``train`` / ``val`` / ``test``) of a study: a uniform hash of ``(seed, GSE)`` cut at the cumulative fractions.

    Deterministic across machines and Python versions (SHA-256, not ``hash``), so a held-out study stays held out."""
    f = np.asarray(fractions, dtype=np.float64)
    if f.shape != (3,) or (f < 0).any() or abs(f.sum() - 1.0) > 1e-9:
        raise ValueError(
            f"split fractions must be three non-negative numbers summing to 1, got {tuple(fractions)}"
        )
    h = hashlib.sha256(f"singlet-split/{int(seed)}/{gse.strip().upper()}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2.0**64
    return SPLITS[int(np.searchsorted(np.cumsum(f)[:2], u, side="right"))]


# ================================================================================================================= cell labels
def path_level(paths: Sequence[Optional[str]], level: int) -> list:
    """CL id at ``level`` (1 compartment, 2 lineage, 3 cell type, 4+ subtype) of each ``cell_type_path``; None where the call
    stops above that level or there is none."""
    out: list = []
    for p in paths:
        parts = p.split(">") if isinstance(p, str) and p else ()
        out.append(parts[level - 1] if len(parts) >= level else None)
    return out


_CL_ANC: Optional[dict] = None


def _cl_ancestors() -> dict:
    """CL id -> set of itself and its ``is_a`` ancestors, from the Cell Ontology release the labeller uses."""
    global _CL_ANC
    if _CL_ANC is None:
        import gzip
        import json

        p = Path(__file__).resolve().parent.parent / "celllabels" / "data" / "cl.json.gz"
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            terms = json.load(fh).get("terms") or {}
        memo: dict = {}

        def anc(t: str, seen=()) -> frozenset:
            if t in memo:
                return memo[t]
            out = {t}
            for parent in (terms.get(t) or {}).get("is_a") or ():
                if parent not in seen:
                    out |= anc(parent, seen + (t,))
            memo[t] = frozenset(out)
            return memo[t]

        for t in terms:
            anc(t)
        _CL_ANC = memo
    return _CL_ANC


def _cl_ids(values) -> set:
    """CL ids for ids or names (case-insensitive exact names of the shipped CL release)."""
    from singlet.s1.labels import cl_names

    names = {v.lower(): k for k, v in cl_names().items()}
    out = set()
    for v in [values] if isinstance(values, str) else values:
        v = str(v).strip()
        if v.upper().startswith("CL:"):
            out.add(v.upper())
        elif v.lower() in names:
            out.add(names[v.lower()])
        else:
            raise ValueError(f"unknown Cell Ontology term {v!r}: use a CL id or an exact CL name")
    return out


def cell_mask(obs, where: Mapping[str, Any]) -> np.ndarray:
    """Boolean mask of the called cells in a cells-table frame (``singlet.s1`` obs columns) that satisfy the cell-level filters
    of ``where`` (:data:`CELL_KEYS`), with the cell index's semantics: ``cell_type`` matches descendants (CL ``is_a``), ``n_umi`` is
    exon + intron, ``n_genes`` is exonic features."""
    n = len(obs)
    m = (
        obs["called"].to_numpy(dtype=bool, copy=True)
        if "called" in obs.columns
        else np.ones(n, bool)
    )
    for key, colname in (
        ("cell_type", "cell_type_path"),
        ("sex", "sex_chrom"),
        ("ancestry", "ancestry_hancestro"),
    ):
        if (
            where.get(key) is not None and colname not in obs.columns
        ):  # a sample without labels matches no label filter
            return np.zeros(n, bool)
    if where.get("cell_type") is not None:
        want = _cl_ids(where["cell_type"])
        anc = _cl_ancestors()
        deepest = [
            p.split(">")[-1] if isinstance(p, str) and p else None for p in obs["cell_type_path"]
        ]
        m &= np.fromiter((d is not None and bool(anc.get(d, {d}) & want) for d in deepest), bool, n)
    if where.get("sex") is not None:
        vals = [where["sex"]] if isinstance(where["sex"], str) else where["sex"]
        want = {_SEX.get(str(v), str(v).lower()) for v in vals}
        m &= obs["sex_chrom"].astype(object).isin(want).to_numpy()
    if where.get("ancestry") is not None:
        vals = [where["ancestry"]] if isinstance(where["ancestry"], str) else where["ancestry"]
        m &= (
            obs["ancestry_hancestro"].astype(object).isin({str(v).upper() for v in vals}).to_numpy()
        )
    umi = obs["n_umi_exon"].to_numpy(np.int64) + obs["n_umi_intron"].to_numpy(np.int64)
    if where.get("min_umi") is not None:
        m &= umi >= where["min_umi"]
    if where.get("max_umi") is not None:
        m &= umi <= where["max_umi"]
    genes = obs["n_feat_exon"].to_numpy(np.int64)
    if where.get("min_genes") is not None:
        m &= genes >= where["min_genes"]
    if where.get("max_genes") is not None:
        m &= genes <= where["max_genes"]
    if where.get("min_cell_type_score") is not None:
        s = obs["cell_type_deepest_score"].to_numpy(np.float64)
        m &= np.nan_to_num(s, nan=-1.0) >= where["min_cell_type_score"]
    return m


# ================================================================================================================= planning
def _check_where(where: Optional[Mapping[str, Any]]) -> dict:
    w = dict(where or {})
    bad = sorted(set(w) - set(SAMPLE_KEYS) - set(CELL_KEYS))
    if bad:
        raise ValueError(
            f"unknown filter keys {bad}; sample-level: {list(SAMPLE_KEYS)}, cell-level: {list(CELL_KEYS)}"
        )
    return w


def _cohort_frame(cohort):
    """``(gse, gsm or None)`` rows from a cohort: a DataFrame, a manifest TSV/CSV path (``/api/manifest``), or accessions."""
    import pandas as pd

    if isinstance(cohort, (str, Path)) and Path(cohort).exists():
        p = Path(cohort)
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
        sep = "\t" if lines and "\t" in lines[0] else ","
        from io import StringIO

        cohort = pd.read_csv(StringIO("\n".join(lines)), sep=sep, dtype=str)
    if isinstance(cohort, pd.DataFrame):
        cols = {c.lower(): c for c in cohort.columns}
        gse_c = next((cols[c] for c in ("gse", "gse_id", "study") if c in cols), None)
        gsm_c = next((cols[c] for c in ("gsm", "gsm_id", "sample") if c in cols), None)
        if gse_c is None and gsm_c is None:
            raise ValueError("a cohort table needs a gse/gse_id or gsm/gsm_id column")
        gse = cohort[gse_c].tolist() if gse_c else [None] * len(cohort)
        gsm = cohort[gsm_c].tolist() if gsm_c else [None] * len(cohort)
        return list(zip(gse, gsm))
    if isinstance(cohort, str):
        cohort = [cohort]
    out = []
    for a in cohort:
        if isinstance(a, (tuple, list)):
            out.append((a[0], a[1]))
        elif str(a).upper().startswith("GSM"):
            out.append((None, str(a).upper()))
        else:
            out.append((str(a).upper(), None))
    return out


def plan_samples(
    where: Optional[Mapping[str, Any]] = None,
    *,
    cohort=None,
    base: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """The samples to stream, as a DataFrame ``gse, gsm, n_cells`` (``n_cells`` = matching cells per the cell index).

    With ``where`` only, the atlas cell index lists the samples with matching cells (one count request, paged by study when the
    answer reaches the per-request group cap). With ``cohort``, the cohort names the studies and samples and the index (if
    ``where`` is given) narrows them; a cohort without ``where`` keeps every sample of its studies."""
    import pandas as pd

    from singlet.s1 import atlas as A

    w = _check_where(where)
    if cohort is not None:
        rows = _cohort_frame(cohort)
        studies = sorted({g for g, _ in rows if g})
        gsms = {s for _, s in rows if s}
        if gsms and not studies:
            raise ValueError(
                "a cohort of GSM accessions needs their GSE (a gse column or (gse, gsm) pairs)"
            )
        if not w:
            from singlet.s1.client import open_study

            out = []
            for g in studies:
                st = open_study(g, base=base)
                for gsm, sm in st.samples.items():
                    if (not gsms or gsm in gsms) and sm.n_called:
                        out.append((g, gsm, int(sm.n_called)))
            return pd.DataFrame(out, columns=["gse", "gsm", "n_cells"])
        w["studies"] = studies
        if gsms:
            w["samples"] = sorted(gsms)

    def retry(fn, *args, **kw):
        # the index answers in about a second; a dropped connection is retried, not fatal to a long plan
        for attempt in range(4):
            try:
                return fn(*args, **kw)
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2.0 * (attempt + 1))

    def count(ww):
        df = retry(A.count_cells, by=["study", "sample"], base=base, api_key=api_key, **ww)
        return df.rename(columns={"study": "gse", "sample": "gsm"})

    if "studies" in w:
        chunks = [w["studies"][i : i + 200] for i in range(0, len(w["studies"]), 200)]
        df = pd.concat([count({**w, "studies": c}) for c in chunks], ignore_index=True)
    else:
        df = count(w)
        if (
            len(df) >= _INDEX_GROUP_CAP
        ):  # capped: list the studies first, then their samples in chunks
            st = retry(A.count_cells, by=["study"], base=base, api_key=api_key, **w)[
                "study"
            ].tolist()
            chunks = [st[i : i + 200] for i in range(0, len(st), 200)]
            df = pd.concat([count({**w, "studies": c}) for c in chunks], ignore_index=True)
    df = df[df["n_cells"] > 0][["gse", "gsm", "n_cells"]]
    return df.sort_values(["gse", "gsm"], ignore_index=True)


# ================================================================================================================= gene axis
def _strip_version(ids) -> np.ndarray:
    return np.array([str(x).split(".")[0] for x in ids], dtype=object)


class GeneAxis:
    """A fixed gene vocabulary (Ensembl ids) and, per study, the map from the study's gene table onto it."""

    def __init__(self, gene_ids: Sequence[str], gene_names: Optional[Sequence[str]] = None):
        self.ids = _strip_version(gene_ids)
        self.names = np.asarray(gene_names if gene_names is not None else self.ids, dtype=object)
        self._pos = {g: i for i, g in enumerate(self.ids)}
        self._maps: dict = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.ids)

    def map_for(self, study) -> np.ndarray:
        """int32 map: study gene index -> axis index, -1 for genes not on the axis."""
        with self._lock:
            m = self._maps.get(study.gse)
            if m is None:
                ids, _ = study.gene_table()
                m = np.fromiter(
                    (self._pos.get(g, -1) for g in _strip_version(ids)), np.int32, len(ids)
                )
                self._maps[study.gse] = m
            return m


def remap_columns(X, gmap: np.ndarray, n_out: int):
    """CSR cells x study-genes -> CSR cells x axis-genes (columns off the axis dropped)."""
    import scipy.sparse as sp

    X = X.tocsr()
    new = gmap[X.indices]
    keep = new >= 0
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))[keep]
    indptr = np.zeros(X.shape[0] + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=X.shape[0]), out=indptr[1:])
    return sp.csr_matrix(
        (X.data[keep].astype(np.float32), new[keep], indptr), shape=(X.shape[0], n_out)
    )


def log1p_cp10k(X):
    """log1p of counts per 10,000 (per-cell totals over the axis genes), in place on a float32 CSR."""
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    X.data *= np.repeat((1e4 / tot).astype(np.float32), np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    return X


# ================================================================================================================= dataset
class _Loaded:
    """One sample's selected cells, already in a random order (and transformed), consumed from the front."""

    __slots__ = ("gse", "gsm", "X", "obs", "pos")

    def __init__(self, gse, gsm, X, obs):
        self.gse, self.gsm, self.X, self.obs, self.pos = gse, gsm, X, obs, 0

    @property
    def left(self) -> int:
        return self.X.shape[0] - self.pos


class AtlasCells(_Base):
    """Cells of the Singlet atlas matching ``where`` (or a ``cohort``), streamed by HTTP range reads as shuffled mini-batches.

    Parameters
    ----------
    where : dict, optional
        Cell-index filters, the keywords of :func:`singlet.s1.atlas.find_cells`: sample-level ``organism``, ``tissue``
        (UBERON), ``disease`` (MONDO / PATO:0000461), ``development_stage``, ``donor_status``, ``perturbation``, ``sample_kind``,
        ``studies``, ``samples``; cell-level ``cell_type`` (CL id or name, descendants included), ``sex``, ``ancestry``
        (HANCESTRO), ``min_umi`` / ``max_umi``, ``min_genes`` / ``max_genes``, ``min_cell_type_score``.
    cohort : optional
        Instead of (or narrowed by) ``where``: a ``/api/manifest`` TSV path, a DataFrame with ``gse`` / ``gsm`` columns, or a list
        of GSE accessions or ``(gse, gsm)`` pairs.
    split : ``"train"`` / ``"val"`` / ``"test"`` / None
        Keep only the studies of this split (:func:`study_split` with ``split_fractions`` and ``split_seed``); None keeps all.
    genes : sequence of Ensembl ids, optional
        The gene axis. Default: the full gene table of the first study (one organism per dataset).
    batch_size : int
        Cells per yielded batch (use ``DataLoader(ds, batch_size=None)``).
    mix_samples : int
        Samples held open at once; each batch draws its cells from all of them in proportion to what each has left.
    prefetch : int
        Samples read ahead in background threads.
    io_threads : int or ``"auto"``
        Range requests in flight at once, shared by the ``prefetch`` readers (each sample's cells table, matrix prefixes and chunk
        runs are independent requests): ``"auto"`` is 8 over HTTP and 1 from a local directory; 1 sends them one at a time. A
        process never has more than ``$SINGLET_MAX_CONNECTIONS`` (default 16) in flight; each DataLoader worker is a process.
    max_cells_per_sample : int, optional
        Keep at most this many matching cells of a sample (a seeded random subset), to bound the weight of very large samples.
    transform : ``"log1p"`` / None / callable
        ``"log1p"``: log1p of counts per 10k; None: raw counts (float32); a callable gets and returns the CSR batch.
    output : ``"dense"`` / ``"csr"`` / ``"scipy"``
        ``X`` as a dense float32 tensor, a torch sparse CSR tensor, or the scipy CSR matrix.
    levels : sequence of int
        Cell Ontology levels returned as ``cell_type_l<k>`` (CL ids, None where the call stops above the level).
    seed : int
        Seeds the sample order (with the epoch), the cell order and ``max_cells_per_sample``.

    Every batch is a dict: ``X``, ``gse``, ``gsm``, ``col`` (the cell's column in its sample object), ``cell_type_l<k>``,
    ``cell_type_score``, ``sex``, ``ancestry``. ``stats()`` reports cells, samples, requests, bytes and seconds of this process.
    Samples are divided between DataLoader workers and between distributed ranks (``torch.distributed`` when initialised).
    """

    def __init__(
        self,
        where: Optional[Mapping[str, Any]] = None,
        *,
        cohort=None,
        split: Optional[str] = None,
        split_fractions: Sequence[float] = DEFAULT_FRACTIONS,
        split_seed: int = 0,
        genes: Optional[Sequence[str]] = None,
        batch_size: int = 256,
        mix_samples: int = 16,
        prefetch: int = 4,
        io_threads: Union[int, str] = "auto",
        max_cells_per_sample: Optional[int] = None,
        transform: Union[None, str, Callable] = "log1p",
        output: str = "dense",
        levels: Sequence[int] = (1, 2, 3),
        seed: int = 0,
        drop_last: bool = False,
        base: Optional[str] = None,
        api_key: Optional[str] = None,
        plan=None,
    ):
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None")
        if output not in ("dense", "csr", "scipy"):
            raise ValueError("output must be 'dense', 'csr' or 'scipy'")
        self.where = _check_where(where)
        self.split, self.split_fractions, self.split_seed = (
            split,
            tuple(split_fractions),
            int(split_seed),
        )
        self.batch_size, self.mix_samples, self.prefetch = (
            int(batch_size),
            max(1, int(mix_samples)),
            max(1, int(prefetch)),
        )
        if io_threads != "auto" and int(io_threads) < 1:
            raise ValueError("io_threads must be 'auto' or at least 1")
        self.io_threads = io_threads
        self.max_cells_per_sample = max_cells_per_sample
        self.transform, self.output, self.levels = transform, output, tuple(int(k) for k in levels)
        self.seed, self.drop_last, self.base = int(seed), bool(drop_last), base
        self.epoch = 0
        df = (
            plan
            if plan is not None
            else plan_samples(self.where, cohort=cohort, base=base, api_key=api_key)
        )
        df = df.copy()
        df["split"] = [study_split(g, self.split_fractions, self.split_seed) for g in df["gse"]]
        self.all_samples = df
        self.samples = (
            df[df["split"] == split].reset_index(drop=True) if split else df.reset_index(drop=True)
        )
        self._genes = genes
        self._axis: Optional[GeneAxis] = None
        self._studies: dict = {}
        self._slock = threading.Lock()
        self._stats = {"cells": 0, "batches": 0, "samples": 0, "samples_skipped": 0, "seconds": 0.0}
        self._t0_req: Optional[tuple] = None

    # ------------------------------------------------------------------ metadata
    def __repr__(self) -> str:
        return (
            f"AtlasCells(split={self.split!r}, studies={self.samples['gse'].nunique()}, samples={len(self.samples)}, "
            f"cells~{int(self.samples['n_cells'].sum()):,}, genes={len(self.gene_axis) if self._axis else '?'})"
        )

    @property
    def n_cells(self) -> int:
        """Matching cells in this split per the cell index (before ``max_cells_per_sample``)."""
        return int(self.samples["n_cells"].sum())

    def __getstate__(self) -> dict:
        # DataLoader worker processes get a copy without the open studies and locks; each opens its own
        d = self.__dict__.copy()
        d["_studies"], d["_slock"], d["_t0_req"] = {}, None, None
        if d["_axis"] is not None:
            d["_genes"], d["_axis"] = list(d["_axis"].ids), None
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        self._slock = threading.Lock()

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle the sample order for this epoch (call before iterating, as with ``DistributedSampler``)."""
        self.epoch = int(epoch)

    def study(self, gse: str):
        from singlet.s1.client import open_study

        with self._slock:
            st = self._studies.get(gse)
        if st is None:
            st = open_study(gse, base=self.base)
            with self._slock:
                self._studies[gse] = st
        return st

    @property
    def gene_axis(self) -> GeneAxis:
        if self._axis is None:
            if self._genes is not None:
                self._axis = GeneAxis(list(self._genes))
            else:
                if not len(self.all_samples):
                    raise ValueError("no samples match: nothing to take a gene axis from")
                ids, names = self.study(self.all_samples["gse"].iloc[0]).gene_table()
                self._axis = GeneAxis(ids, names)
        return self._axis

    @staticmethod
    def _worker_id() -> int:
        try:
            from torch.utils.data import get_worker_info

            wi = get_worker_info()
            return 0 if wi is None else int(wi.id)
        except ImportError:  # pragma: no cover
            return 0

    def _io(self) -> tuple:
        r, b = self._transport_stats()
        r0, b0 = self._t0_req or (r, b)
        return r - r0, b - b0

    def _transport_stats(self):
        from singlet.s1.client import _transport_for

        s = _transport_for(self.base).stats
        return s.requests, s.bytes

    def stats(self) -> dict:
        """Cells, batches, samples, requests, bytes and wall seconds streamed by this process since the dataset was made."""
        r, b = self._io()
        out = dict(self._stats, requests=r, bytes=b)
        sec = out["seconds"] or float("nan")
        out.update(cells_per_s=out["cells"] / sec, mb_per_s=out["bytes"] / 1e6 / sec)
        return out

    # ------------------------------------------------------------------ reading
    def _my_samples(self) -> list:
        """This worker's and rank's share of this epoch's sample order."""
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(len(self.samples))
        rank, world = 0, 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank, world = dist.get_rank(), dist.get_world_size()
        except ImportError:  # pragma: no cover
            pass
        wid, nw = 0, 1
        try:
            from torch.utils.data import get_worker_info

            wi = get_worker_info()
            if wi is not None:
                wid, nw = wi.id, wi.num_workers
        except ImportError:  # pragma: no cover
            pass
        shard = rank * nw + wid
        mine = order[shard :: world * nw]
        return [(self.samples["gse"].iloc[i], self.samples["gsm"].iloc[i]) for i in mine]

    def _load(self, gse: str, gsm: str, threads: int, budget=None) -> Optional[_Loaded]:
        st = self.study(gse)
        axis = self.gene_axis
        where = self.where

        cap = self.max_cells_per_sample

        def sel(df):
            m = cell_mask(df, where)
            if cap is not None and m.sum() > cap:
                # a seeded subset (the seed, not the epoch), chosen before the read so large samples fetch fewer chunks
                idx = np.flatnonzero(m)
                keep = np.random.default_rng([self.seed, int(gsm[3:])]).permutation(len(idx))[:cap]
                m = np.zeros(len(m), bool)
                m[idx[keep]] = True
            return m

        a = st.cells(
            sel,
            samples=[gsm],
            modalities="counts",
            columns=list(_OBS),
            threads=threads,
            io_threads=budget if budget is not None else self.io_threads,
        )
        if a.n_obs == 0:
            return None
        X = remap_columns(a.X, axis.map_for(st), len(axis))

        def col(name, dtype=object, fill=None):
            if name not in a.obs.columns:  # a sample without labels
                return np.full(a.n_obs, fill, dtype=dtype)
            v = a.obs[name]
            return (
                v.to_numpy(dtype)
                if dtype is not object
                else v.astype(object).where(v.notna(), None).to_numpy()
            )

        obs = {
            "col": a.obs["column"].to_numpy(np.int64),
            "path": col("cell_type_path"),
            "score": col("cell_type_deepest_score", np.float32, np.nan),
            "sex": col("sex_chrom"),
            "ancestry": col("ancestry_hancestro"),
        }
        h = int.from_bytes(
            hashlib.sha256(f"{self.seed}/{self.epoch}/{gsm}".encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(h)
        order = rng.permutation(a.n_obs)
        # permute (and normalise) once here, in the reader thread, so a batch is contiguous slices
        X = X[order]
        if self.transform == "log1p":
            X = log1p_cp10k(X)
        return _Loaded(gse, gsm, X, {k: v[order] for k, v in obs.items()})

    def _batch(self, open_: list, n: int, rng) -> dict:
        import scipy.sparse as sp

        left = np.array([s.left for s in open_], np.int64)
        take = rng.multivariate_hypergeometric(left, min(n, int(left.sum())))
        mats, parts = (
            [],
            {"gse": [], "gsm": [], "col": [], "path": [], "score": [], "sex": [], "ancestry": []},
        )
        for s, k in zip(open_, take):
            if not k:
                continue
            a, b = s.pos, s.pos + int(k)
            s.pos = b
            mats.append(s.X[a:b])
            parts["gse"].extend([s.gse] * int(k))
            parts["gsm"].extend([s.gsm] * int(k))
            for key in ("col", "path", "score", "sex", "ancestry"):
                parts[key].append(s.obs[key][a:b])
        # cells stay grouped by sample inside the batch (the order inside a batch does not matter to a gradient step)
        X = sp.vstack(mats, format="csr", dtype=np.float32)
        cat = {k: np.concatenate(parts[k]) for k in ("col", "path", "score", "sex", "ancestry")}
        if callable(self.transform):
            X = self.transform(X)
        out: dict = {
            "X": self._tensor(X),
            "gse": parts["gse"],
            "gsm": parts["gsm"],
            "col": cat["col"],
            "cell_type_score": cat["score"],
            "sex": list(cat["sex"]),
            "ancestry": list(cat["ancestry"]),
            # this process's cumulative I/O, so a DataLoader with worker processes can add up requests and bytes
            "stream": dict(
                zip(("worker", "requests", "bytes"), (self._worker_id(), *self._io())),
                samples=self._stats["samples"],
                samples_skipped=self._stats["samples_skipped"],
            ),
        }
        for k in self.levels:
            out[f"cell_type_l{k}"] = path_level(cat["path"], k)
        return out

    def _tensor(self, X):
        if self.output == "scipy":
            return X
        import torch

        if self.output == "csr":
            return torch.sparse_csr_tensor(
                torch.from_numpy(X.indptr.astype(np.int64)),
                torch.from_numpy(X.indices.astype(np.int64)),
                torch.from_numpy(X.data),
                size=X.shape,
                check_invariants=False,
            )
        return torch.from_numpy(X.toarray())

    def __iter__(self) -> Iterator[dict]:
        if self._t0_req is None:
            self._t0_req = self._transport_stats()
        mine = self._my_samples()
        _ = self.gene_axis  # resolve once, before the reader threads
        threads = max(1, (os.cpu_count() or 4) // self.prefetch)
        from singlet.s1 import transport as T
        from singlet.s1.client import _transport_for

        # one request budget for all the readers of this iterator
        budget = T.IOBudget(T.io_threads(self.io_threads, _transport_for(self.base)))
        q: "queue.Queue" = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        DONE = object()

        def reader(items):
            for gse, gsm in items:
                if stop.is_set():
                    return
                try:
                    q.put(self._load(gse, gsm, threads, budget))
                except (
                    Exception
                ) as e:  # a sample that cannot be read is skipped and counted, not fatal
                    q.put(("error", gse, gsm, repr(e)))
            q.put(DONE)

        # ``prefetch`` reader threads, each taking every prefetch-th sample of this shard's order
        workers = [
            threading.Thread(
                target=reader,
                args=(mine[i :: self.prefetch],),
                daemon=True,
                name=f"singlet-stream-{i}",
            )
            for i in range(self.prefetch)
        ]
        for t in workers:
            t.start()
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        t0 = time.perf_counter()
        open_: list = []
        done = 0
        self.errors: list = getattr(self, "errors", [])
        try:
            while True:
                # fill the open set up to mix_samples; block only when the open samples cannot fill a batch
                while done < len(workers):
                    short = sum(s.left for s in open_) < self.batch_size
                    if not short and len(open_) >= self.mix_samples:
                        break
                    try:
                        item = q.get() if short else q.get_nowait()
                    except queue.Empty:
                        break
                    if item is DONE:
                        done += 1
                    elif isinstance(item, tuple):
                        self._stats["samples_skipped"] += 1
                        self.errors.append(item[1:])
                    elif item is not None:
                        open_.append(item)
                        self._stats["samples"] += 1
                avail = sum(s.left for s in open_)
                if avail == 0:  # the fill loop only gives up when every reader is done
                    break
                if avail < self.batch_size and self.drop_last:
                    break
                b = self._batch(open_, self.batch_size, rng)
                open_ = [s for s in open_ if s.left]
                n = len(b["gsm"])
                self._stats["cells"] += n
                self._stats["batches"] += 1
                self._stats["seconds"] += time.perf_counter() - t0
                yield b
                t0 = time.perf_counter()
        finally:
            stop.set()
            while True:  # unblock readers waiting on a full queue
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
