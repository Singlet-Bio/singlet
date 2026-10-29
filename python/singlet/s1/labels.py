# SPDX-License-Identifier: MIT
"""Per-cell columns of a .s1 sample as AnnData ``obs`` columns: the Research Alpha labels, the cell calls and the QC columns.

Every sample object's ``cells`` table carries, next to the counts summaries, the labels written by the transcode
(docs/cell_annotation.md, ``singlet.s1.labels_step``) and the cell calls (docs/called-ed.md). The stored names are short and
partly internal (``cell_type_cl``, ``cell_type_p1``..``p6``, categorical codes for donor and cell-cycle columns); this module turns
them into the columns users see:

=================================== ============================================================================================
``cell_type``                        Cell Ontology name at the default level (see below); missing when the cell has no call there
``cell_type_ontology_term_id``       its CL id (CELLxGENE naming)
``cell_type_level``                  the level it was taken from (1 compartment, 2 lineage, 3 cell type, 4+ subtype; 0 none)
``cell_type_score``                  product of the per-level gating confidences down to that level
``cell_type_deepest`` / ``_ontology_term_id`` / ``_level`` / ``_score``
                                     the labeller's deepest call (levels 3 and deeper are previews, see the accuracy below)
``cell_type_path``                   the full CL path of the deepest call, compartment first, ids joined by ``>``
``cell_type_p1`` .. ``cell_type_p6`` gating confidence of each level of that path
``sex_chrom``, ``sex_score``         chromosomal sex (``female`` / ``male`` / ``ambiguous``) and its posterior
``ancestry_hancestro``, ``ancestry_prob``
                                     continental genetic ancestry of the cell's donor (HANCESTRO id) and its proportion
``called``                           the default cell call (the knee call the pipeline made)
``called_ed``, ``called_ed_fdr``     EmptyDrops call (additional, not the default) and its FDR
``n_umi_*``, ``n_feat_*``, ``cyc_*``, ``qc_*``, ``don_*``, ``mts_*``
                                     counts summaries, cell cycle, QC, genotype-free donor assignment, mtDNA summary
=================================== ============================================================================================

**Default cell type level.** ``cell_type`` is the deepest level ``k <= cell_type_level`` (default 2, the levels the atlas shows
by default) that the labeller reached, i.e. whose gating passed the guide's own per-level thresholds (minimum score and margin,
``singlet.celllabels.celltype``). ``cell_type_min_score`` (default 0: no extra threshold) additionally requires the path score
``p1 * ... * pk`` to reach a value. Per-study accuracy of the labeller against CELLxGENE author labels, measured on exactly these
calls, is 0.84 at level 1, 0.75 at level 2, 0.43 at level 3 and 0.32 at level 4 (docs/cell_annotation.md section 3), so levels 3
and deeper are previews: ask for them with ``cell_type_level=6`` or read the ``cell_type_deepest*`` columns. The score ranks
calls; it is optimistic above 0.9 and is not a calibrated probability.

**Sex** agrees with GEO sample metadata for 153 of 156 called samples (98.1%; the three disagreements are GEO errors).
**Ancestry** is computed only for human donors whose per-donor coverage tables exist (about 1% of cells); elsewhere it is missing.

**Hidden by default.** ``qc_doublet_score`` is an unvalidated layer (no benchmark supports it): it is left out of ``obs`` unless
``include_unvalidated=True`` or it is named in ``columns=``. ``where=`` filters can still use it.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

#: the default level of ``cell_type`` (docs/cell_annotation.md: levels 1-2 are shown by default, 3+ are previews)
CELL_TYPE_LEVEL = 2
#: an extra minimum path score (product of the gating confidences down to the level) for ``cell_type``; 0 keeps every level the
#: guide's gating reached, which is what the documented accuracy was measured on
CELL_TYPE_MIN_SCORE = 0.0
MAX_LEVELS = 6
LEVEL_COLUMNS = tuple(f"cell_type_p{k}" for k in range(1, MAX_LEVELS + 1))
#: stored label columns of the cells table (label schema 2)
STORED_LABEL_COLUMNS = (
    "cell_type_cl",
    "cell_type_label",
    "cell_type_score",
    "cell_type_path",
    *LEVEL_COLUMNS,
    "sex_chrom",
    "sex_score",
    "ancestry_hancestro",
    "ancestry_prob",
)
#: stored name -> obs name, where they differ
RENAMES = {
    "cell_type_cl": "cell_type_deepest_ontology_term_id",
    "cell_type_label": "cell_type_deepest",
    "cell_type_score": "cell_type_deepest_score",
}
#: cells-table columns that no benchmark supports: hidden from obs unless asked for
UNVALIDATED = ("qc_doublet_score",)
#: cells-table columns that only order or re-create legacy files (never shown)
INTERNAL = ("barcode", "cc_pos", "cc_listed", "cc_pvalue")
#: the columns in front of every obs, in this order (only those present)
LEAD = (
    "cell_type",
    "cell_type_ontology_term_id",
    "cell_type_level",
    "cell_type_score",
    "cell_type_deepest",
    "cell_type_deepest_ontology_term_id",
    "cell_type_deepest_level",
    "cell_type_deepest_score",
    "cell_type_path",
    *LEVEL_COLUMNS,
    "sex_chrom",
    "sex_score",
    "ancestry_hancestro",
    "ancestry_prob",
    "called",
    "called_ed",
    "called_ed_fdr",
)
_STRING_COLUMNS = {
    "cell_type",
    "cell_type_ontology_term_id",
    "cell_type_deepest",
    "cell_type_deepest_ontology_term_id",
    "cell_type_path",
    "sex_chrom",
    "ancestry_hancestro",
}


@lru_cache(maxsize=1)
def cl_names() -> Dict[str, str]:
    """CL id -> name, from the Cell Ontology release the labeller uses (shipped with the package)."""
    p = Path(__file__).resolve().parent.parent / "celllabels" / "data" / "cl.json.gz"
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            terms = json.load(fh).get("terms") or {}
        return {k: str(v.get("name") or k) for k, v in terms.items()}
    except OSError:  # pragma: no cover - the file ships in the wheel
        return {}


def categories(members: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Labels of the categorical cells-table columns (stored as codes), from the recipes of a sample's derived members.

    ``members`` maps member names to a manifest member dict or a :class:`singlet.s1.client.MemberRef` (its ``meta``)."""
    out: Dict[str, List[str]] = {}
    for m in members.values():
        meta = m if isinstance(m, dict) else (getattr(m, "meta", None) or {})
        params = meta.get("params") if isinstance(meta, dict) else None
        if not isinstance(params, dict):
            continue
        for c in params.get("c") or ():
            if not isinstance(c, dict) or c.get("s") != "c":
                continue
            f = c.get("f") or {}
            col = c.get("c")
            if f.get("k") == "cat" and col and isinstance(f.get("labels"), list):
                out.setdefault(str(col), [str(x) for x in f["labels"]])
    return out


def _strings(v) -> np.ndarray:
    """Object array with None for empty strings."""
    a = np.asarray(v, dtype=object)
    if a.size:
        a = a.copy()
        a[(a == "") | (a == None)] = None  # noqa: E711 - elementwise
    return a


def _decode_cat(v, labels: Sequence[str]) -> np.ndarray:
    codes = np.asarray(v)
    out = np.full(len(codes), None, dtype=object)
    if not len(codes) or not np.issubdtype(codes.dtype, np.integer):
        return _strings(v) if codes.dtype == object else out
    lab = np.asarray(list(labels), dtype=object)
    ok = (codes >= 0) & (codes < len(lab))
    out[ok] = lab[codes[ok].astype(np.int64)]
    return out


def cell_type_columns(
    cols: Mapping[str, Any],
    n: int,
    *,
    level: int = CELL_TYPE_LEVEL,
    min_score: float = CELL_TYPE_MIN_SCORE,
) -> Dict[str, np.ndarray]:
    """``cell_type``, ``cell_type_ontology_term_id``, ``cell_type_level``, ``cell_type_score`` and the ``cell_type_deepest*``
    columns from the stored path and per-level confidences (empty dict when the sample has no cell type labels)."""
    if "cell_type_path" not in cols:
        return {}
    names = cl_names()
    path = _strings(cols["cell_type_path"])
    deep_id = _strings(cols["cell_type_cl"]) if "cell_type_cl" in cols else np.full(n, None, object)
    deep_name = (
        _strings(cols["cell_type_label"]) if "cell_type_label" in cols else np.full(n, None, object)
    )
    local = {i: nm for i, nm in zip(deep_id, deep_name) if i is not None and nm is not None}
    # parse each distinct path once: ids[u, k] is level k+1 of path u
    has = np.array([p is not None for p in path], dtype=bool)
    uniq, inv = (
        np.unique(path[has].astype(str), return_inverse=True)
        if has.any()
        else (np.array([], dtype=str), np.zeros(0, np.int64))
    )
    ids = np.full((len(uniq) + 1, MAX_LEVELS), None, dtype=object)  # last row: no path
    depth_u = np.zeros(len(uniq) + 1, dtype=np.int8)
    for u, s in enumerate(uniq):
        parts = s.split(">")[:MAX_LEVELS]
        ids[u, : len(parts)] = parts
        depth_u[u] = len(parts)
    code = np.full(n, len(uniq), dtype=np.int64)
    code[np.flatnonzero(has)] = inv
    depth = depth_u[code]
    P = np.full((n, MAX_LEVELS), np.nan, dtype=np.float64)
    for k, c in enumerate(LEVEL_COLUMNS):
        if c in cols:
            P[:, k] = np.asarray(cols[c], dtype=np.float64)
    lv = max(0, min(int(level), MAX_LEVELS))
    # level k is usable when it is on the path, has a confidence, and the path score down to it passes
    on_path = np.arange(1, MAX_LEVELS + 1)[None, :] <= depth[:, None]
    score_k = np.cumprod(np.where(np.isnan(P), 0.0, P), axis=1)
    ok = on_path & ~np.isnan(P) & (score_k >= float(min_score))
    ok[:, lv:] = False
    ok = np.cumprod(ok, axis=1).astype(bool)  # a level counts only if every level above it does
    chosen = ok.sum(axis=1).astype(np.int8)
    rows = np.flatnonzero(chosen > 0)
    ct_id = np.full(n, None, dtype=object)
    ct_score = np.full(n, np.nan, dtype=np.float32)
    ct_id[rows] = ids[code[rows], chosen[rows] - 1]
    ct_score[rows] = score_k[rows, chosen[rows] - 1]

    def name_of(i):
        return None if i is None else names.get(i) or local.get(i) or i

    lookup = {i: name_of(i) for i in set(ct_id[rows].tolist())}
    ct_name = np.array([lookup.get(i) if i is not None else None for i in ct_id], dtype=object)
    deep_score = (
        np.asarray(cols["cell_type_score"], dtype=np.float32)
        if "cell_type_score" in cols
        else np.full(n, np.nan, np.float32)
    )
    return {
        "cell_type": ct_name,
        "cell_type_ontology_term_id": ct_id,
        "cell_type_level": chosen,
        "cell_type_score": ct_score,
        "cell_type_deepest": deep_name,
        "cell_type_deepest_ontology_term_id": deep_id,
        "cell_type_deepest_level": np.where(deep_id != None, depth, 0).astype(np.int8),  # noqa: E711
        "cell_type_deepest_score": deep_score,
        "cell_type_path": path,
    }


def obs_columns(
    cols: Mapping[str, Any],
    n: int,
    *,
    cats: Optional[Mapping[str, Sequence[str]]] = None,
    cell_type_level: int = CELL_TYPE_LEVEL,
    cell_type_min_score: float = CELL_TYPE_MIN_SCORE,
    include_unvalidated: bool = False,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """The obs columns (name -> array of length ``n``) for one sample's selected cells.

    ``cols`` holds the stored cells-table columns (already restricted to the selected cells), ``cats`` the labels of the
    categorical ones (:func:`categories`). ``columns`` keeps only those obs names (stored names are accepted too, e.g.
    ``cell_type_cl``); otherwise every column is returned except :data:`INTERNAL` and, without ``include_unvalidated``,
    :data:`UNVALIDATED`."""
    cats = cats or {}
    out: Dict[str, np.ndarray] = {}
    out.update(cell_type_columns(cols, n, level=cell_type_level, min_score=cell_type_min_score))
    for name, v in cols.items():
        if name in INTERNAL or name in RENAMES or name == "cell_type_path":
            continue
        if name in cats:
            out[name] = _decode_cat(v, cats[name])
        elif isinstance(v, np.ndarray) and v.dtype.kind not in "OUS":
            out[name] = v
        else:
            out[name] = _strings(v)
    want = None
    if columns is not None:
        want = {RENAMES.get(c, c) for c in columns}
    elif not include_unvalidated:
        for c in UNVALIDATED:
            out.pop(c, None)
    order = [c for c in LEAD if c in out] + [c for c in out if c not in LEAD]
    return {c: out[c] for c in order if want is None or c in want}


def to_frame(parts: List[Dict[str, np.ndarray]], index=None):
    """Concatenate per-sample obs column dicts into one DataFrame: string columns become categoricals (missing = NaN), a column
    a sample lacks is missing for its cells."""
    import pandas as pd

    names: List[str] = []
    for p in parts:
        for c in p:
            if c not in names:
                names.append(c)
    lens = [len(next(iter(p.values()))) if p else 0 for p in parts]
    data = {}
    for c in names:
        present = [np.asarray(p[c]) for p in parts if c in p]
        kinds = {x.dtype.kind for x in present}
        if c in _STRING_COLUMNS or kinds & {"O", "U", "S"}:
            arr = np.concatenate(
                [
                    np.asarray(p[c], dtype=object) if c in p else np.full(ln, None, dtype=object)
                    for p, ln in zip(parts, lens)
                ]
            )
            data[c] = pd.Categorical(arr)  # None -> NaN
            continue
        complete = len(present) == len(parts)
        if kinds == {"b"}:
            if complete:
                data[c] = np.concatenate(present)
            else:  # nullable boolean: missing for the samples without the column
                data[c] = pd.array(
                    np.concatenate(
                        [
                            np.asarray(p[c], dtype=object) if c in p else np.full(ln, None, object)
                            for p, ln in zip(parts, lens)
                        ]
                    ),
                    dtype="boolean",
                )
        elif complete:
            data[c] = np.concatenate(present)
        else:
            dt = np.result_type(np.float32, *[x.dtype for x in present])
            data[c] = np.concatenate(
                [
                    np.asarray(p[c], dtype=dt) if c in p else np.full(ln, np.nan, dtype=dt)
                    for p, ln in zip(parts, lens)
                ]
            )
    return pd.DataFrame(data, index=index)
