# SPDX-License-Identifier: MIT
"""Gene sets bundled with the feature job, and their mapping onto one sample's genes.

Each collection is a JSON file in ``data/`` written by ``scripts/features_build_data.py``::

    {"collection": "hallmark", "release": "2026.2", "licence": "CC-BY-4.0", "attribution": "...", "sources": {...},
     "sets": {"<name>": {"source_name": ..., "source_id": {"human": ..., "mouse": ...},
                         "genes": {"human": [[gene_id, symbol], ...], "mouse": [...]}}}}

Gene ids are the pipeline reference's Ensembl ids (GRCh38 / GRCm39). A sample's genes are matched by id when the reader gives
ids, otherwise by symbol (the sample's symbols come from the same reference). Null sets for each set are drawn once from the
per-organism abundance reference (``data/abundance_<organism>.parquet``) with a fixed seed, so they are constants of the method.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DATA = Path(__file__).with_name("data")
ORGANISMS = {"human": "Homo sapiens", "mouse": "Mus musculus"}


@lru_cache(maxsize=None)
def collection(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def abundance(organism: str):
    """(gene_ids, symbols, bins) of the per-organism abundance reference."""
    import pyarrow.parquet as pq

    t = pq.read_table(DATA / f"abundance_{organism}.parquet")
    return (
        np.array(t.column("gene_id").to_pylist(), dtype=object),
        np.array(t.column("symbol").to_pylist(), dtype=object),
        np.asarray(t.column("bin").to_numpy(), np.int64),
    )


def abundance_meta(organism: str) -> dict:
    import pyarrow.parquet as pq

    md = pq.read_schema(DATA / f"abundance_{organism}.parquet").metadata or {}
    return {k.decode(): v.decode() for k, v in md.items()}


@dataclass
class Compiled:
    """One collection mapped onto a sample's genes. ``index[j]`` = column indices of set ``j`` (unique); ``nulls[j]`` = index arrays
    of its null sets; ``coverage[j]`` = fraction of the set's reference genes found in the sample."""

    names: List[str]
    index: List[np.ndarray]
    nulls: List[List[np.ndarray]]
    coverage: np.ndarray
    n_listed: np.ndarray
    ok: np.ndarray  # set is scored in this sample
    dropped: Dict[str, str] = field(default_factory=dict)


class GeneIndex:
    """Lookup from gene id or symbol to the sample's column (first column for duplicated symbols)."""

    def __init__(self, gene_names: Sequence[str], gene_ids: Optional[Sequence[str]] = None):
        self.by_id: Dict[str, int] = {}
        if gene_ids is not None:
            for j, g in enumerate(gene_ids):
                self.by_id.setdefault(str(g).split(".")[0], j)
        self.by_sym: Dict[str, int] = {}
        for j, s in enumerate(gene_names):
            self.by_sym.setdefault(str(s).upper(), j)
        self.n = len(gene_names)

    def get(self, gene_id: str, symbol: str) -> Optional[int]:
        if self.by_id:
            j = self.by_id.get(gene_id)
            if j is not None:
                return j
        return self.by_sym.get(str(symbol).upper())

    def many(self, pairs) -> np.ndarray:
        out = [self.get(g, s) for g, s in pairs]
        return np.array(sorted({j for j in out if j is not None}), np.int64)


def _seed(collection_name: str, set_name: str, organism: str) -> int:
    return zlib.crc32(f"{collection_name}/{set_name}/{organism}".encode()) & 0x7FFFFFFF


@lru_cache(maxsize=None)
def null_members(collection_name: str, organism: str, k: int) -> Dict[str, List[List[tuple]]]:
    """Per set, ``k`` abundance-matched random sets from the organism's reference as (gene_id, symbol) pairs. Fixed by the seed."""
    from .rank import draw_null_sets

    ids, syms, bins = abundance(organism)
    sym_of = dict(zip(ids, syms))
    out = {}
    for name, d in collection(collection_name)["sets"].items():
        members = [g for g, _ in d["genes"].get(organism, [])]
        sets = draw_null_sets(members, ids, bins, k, _seed(collection_name, name, organism))
        out[name] = [[(g, sym_of[g]) for g in s] for s in sets]
    return out


def compile_collection(
    name: str,
    organism: str,
    genes: GeneIndex,
    *,
    k_null: int = 0,
    min_coverage: float = 0.5,
    min_genes: int = 5,
) -> Compiled:
    coll = collection(name)
    names, index, nulls, cov, nl, ok = [], [], [], [], [], []
    dropped: Dict[str, str] = {}
    nm = null_members(name, organism, k_null) if k_null else {}
    for sname, d in coll["sets"].items():
        pairs = [tuple(p) for p in d["genes"].get(organism, [])]
        idx = genes.many(pairs)
        c = len(idx) / len(pairs) if pairs else 0.0
        names.append(sname)
        index.append(idx)
        cov.append(c)
        nl.append(len(pairs))
        nulls.append([genes.many(p) for p in nm.get(sname, [])])
        good = c >= min_coverage and len(idx) >= min_genes
        ok.append(good)
        if not good:
            dropped[sname] = f"{len(idx)} of {len(pairs)} genes in the sample"
    return Compiled(names, index, nulls, np.array(cov), np.array(nl), np.array(ok, bool), dropped)
