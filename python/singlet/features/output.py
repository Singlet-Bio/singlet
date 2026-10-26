# SPDX-License-Identifier: MIT
"""The per-sample feature sidecar (docs/CELL-ARCHITECTURE.md section 3) and the guard every table passes before it is written.

``<GSM>.features.<feature set>.parquet``, one row per called cell::

    gsm         string   sample accession (one value per file)
    col         int64    column of the cell in the sample's store ``cells`` table / matrix order (unique)
    <name> ...  one column per registered feature of the set, in registry order:
                float32 in [0, 1] (NaN = not computed) for kind 'score'; uint16 code into the feature's categories
                (65535 = not computed) for 'category'; bool (null = not computed) for 'flag'

Parquet schema metadata: ``singlet_features_schema = "1"``, ``feature_set``, ``method_version``. :func:`assert_public_safe` checks
the exact columns and types, value ranges, unique ``col``, and that no column name or metadata value is a gene symbol or
genotype-like string, so per-gene values cannot ride along.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from .registry import Feature

SCHEMA_VERSION = "1"
NA16 = 65535
_GSM = re.compile(r"^GSM\d+$")
_SET = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_VERSION = re.compile(r"^[A-Za-z0-9_.,;=:/+ \-()]{1,400}$")
_GENOTYPE_LIKE = re.compile(
    r"(\brs\d{3,}\b)|(\bchr[0-9XYMT]{1,2}[:_]\d+)|(\b\d{1,2}[:_]\d{4,}[:_][ACGT])|(\b[01.][/|][01.]\b)|(\b[ACGT]>[ACGT]\b)"
    r"|(\bGT:)|(\bAD[=:]\d)|(\bDP[=:]\d)",
    re.IGNORECASE,
)
_META_KEYS = {"singlet_features_schema", "feature_set", "method_version"}


class PublicOutputError(ValueError):
    """A table destined for public storage failed the schema / content check."""


def _arrow_type(f: Feature):
    import pyarrow as pa

    return {"score": pa.float32(), "category": pa.uint16(), "flag": pa.bool_()}[f.kind]


def arrow_schema(features: Sequence[Feature], method_version: str):
    import pyarrow as pa

    fset = features[0].feature_set
    return pa.schema(
        [("gsm", pa.string()), ("col", pa.int64())] + [(f.name, _arrow_type(f)) for f in features],
        metadata={
            b"singlet_features_schema": SCHEMA_VERSION.encode(),
            b"feature_set": fset.encode(),
            b"method_version": method_version.encode(),
        },
    )


def assert_public_safe(
    table,
    features: Sequence[Feature],
    *,
    method_version: Optional[str] = None,
    gene_symbols: Optional[Iterable[str]] = None,
) -> None:
    """Raise :class:`PublicOutputError` unless ``table`` (pyarrow Table, or pandas DataFrame) is a valid sidecar for ``features``
    (the registered features of one set, in registry order).

    Checked: exactly the columns ``gsm, col, <feature names>`` with their types; one GSM accession; ``col`` non-negative and
    unique; scores NaN or in [0, 1]; category codes inside the feature's categories or 65535; metadata keys and values (when the
    table carries them); no column name equal to a gene symbol (``gene_symbols``, case-insensitive) and no genotype-like string."""
    import pyarrow as pa

    if not features or len({f.feature_set for f in features}) != 1:
        raise PublicOutputError("give the registered features of exactly one feature set")
    fset = features[0].feature_set
    if not _SET.match(fset):
        raise PublicOutputError(f"feature set {fset!r} must match {_SET.pattern}")
    if isinstance(table, pa.Table):
        t = table
    else:
        t = pa.Table.from_pandas(table, preserve_index=False)
    names = ["gsm", "col"] + [f.name for f in features]
    if t.column_names != names:
        raise PublicOutputError(f"columns must be exactly {names}, got {t.column_names}")
    want = {"gsm": pa.string(), "col": pa.int64(), **{f.name: _arrow_type(f) for f in features}}
    for n in names:
        if not t.schema.field(n).type.equals(want[n]):
            raise PublicOutputError(f"{n} must be {want[n]}, got {t.schema.field(n).type}")
    if gene_symbols is not None:
        syms = {str(s).upper() for s in gene_symbols}
        bad = [f.name for f in features if f.name.upper() in syms]
        if bad:
            raise PublicOutputError(f"feature column {bad[0]!r} is a gene symbol")
    for f in features:
        if not _SET.match(f.name) or _GENOTYPE_LIKE.search(f.name):
            raise PublicOutputError(f"feature name {f.name!r} is not allowed")
    gsm = t.column("gsm")
    if gsm.null_count:
        raise PublicOutputError("gsm is required on every row")
    gv = set(gsm.unique().to_pylist())
    if len(gv) > 1 or not all(_GSM.match(x) for x in gv):
        raise PublicOutputError(f"gsm must be one sample accession, got {sorted(gv)[:3]}")
    col = t.column("col")
    if col.null_count:
        raise PublicOutputError("col is required on every row")
    c = np.asarray(col.to_numpy(zero_copy_only=False), np.int64)
    if len(c) and c.min() < 0:
        raise PublicOutputError("col must be non-negative")
    if len(np.unique(c)) != len(c):
        raise PublicOutputError("col must be unique within a sample")
    for f in features:
        a = t.column(f.name)
        if f.kind == "score":
            v = np.asarray(a.to_numpy(zero_copy_only=False), np.float64)
            fin = v[np.isfinite(v)]
            if np.isinf(v).any() or ((fin < 0) | (fin > 1)).any():
                raise PublicOutputError(f"{f.name} must be NaN or in [0, 1]")
        elif f.kind == "category":
            v = np.asarray(a.to_numpy(zero_copy_only=False), np.int64)
            if ((v >= len(f.categories)) & (v != NA16)).any():
                raise PublicOutputError(
                    f"{f.name} has a code outside its {len(f.categories)} categories"
                )
    md = {
        k.decode(): v.decode()
        for k, v in (t.schema.metadata or {}).items()
        if not k.startswith(b"pandas")
    }
    if md:
        if set(md) != _META_KEYS:
            raise PublicOutputError(
                f"metadata keys must be exactly {sorted(_META_KEYS)}, got {sorted(md)}"
            )
        if md["singlet_features_schema"] != SCHEMA_VERSION or md["feature_set"] != fset:
            raise PublicOutputError("metadata names another schema version or feature set")
    mv = method_version if method_version is not None else md.get("method_version")
    if mv is not None and (not _VERSION.match(mv) or _GENOTYPE_LIKE.search(mv)):
        raise PublicOutputError(f"method_version {mv!r} is not allowed")


def build_table(
    gsm: str, col: np.ndarray, values: dict, features: Sequence[Feature], method_version: str
):
    """Arrow table in sidecar layout: ``values[name]`` = per-cell array (scores float, categories int codes or -1 / NA16 for NA,
    flags bool or NaN)."""
    import pyarrow as pa

    n = len(col)
    cols = {"gsm": pa.array([gsm] * n, pa.string()), "col": pa.array(np.asarray(col, np.int64))}
    for f in features:
        v = values.get(f.name)
        if f.kind == "score":
            a = np.full(n, np.nan, np.float32) if v is None else np.asarray(v, np.float32)
            cols[f.name] = pa.array(a)
        elif f.kind == "category":
            a = np.full(n, NA16, np.int64) if v is None else np.asarray(v, np.int64)
            a = np.where((a < 0) | (a >= len(f.categories)), NA16, a).astype(np.uint16)
            cols[f.name] = pa.array(a)
        else:
            vals = (
                [None] * n if v is None else [None if (x is None or x != x) else bool(x) for x in v]
            )
            cols[f.name] = pa.array(vals, pa.bool_())
    return pa.table(cols, schema=arrow_schema(features, method_version))


def sidecar_name(gsm: str, feature_set: str) -> str:
    return f"{gsm}.features.{feature_set}.parquet"


def write_features(
    table,
    path: Union[str, Path],
    features: Sequence[Feature],
    *,
    gene_symbols: Optional[Iterable[str]] = None,
) -> Path:
    """Check ``table`` (from :func:`build_table`) with :func:`assert_public_safe` and write it as Parquet (zstd) atomically."""
    import pyarrow.parquet as pq

    assert_public_safe(table, features, gene_symbols=gene_symbols)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    pq.write_table(table, tmp, compression="zstd", write_statistics=True)
    tmp.replace(path)
    return path
