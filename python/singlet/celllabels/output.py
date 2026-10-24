# SPDX-License-Identifier: MIT
"""The per-sample label file the atlas cell index ingests, and the guard that keeps germline data out of it.

One Parquet file per sample, one row per called cell::

    gsm                 string   sample accession
    col                 int64    column index of the cell in the sample's v2 ``cells`` table / matrix column order
    cell_type_cl        string   Cell Ontology id (null: no call)
    cell_type_label     string   Cell Ontology name of that id
    cell_type_score     float32  confidence of the call in [0, 1] (guide: product of the gating confidences on its path)
    sex_chrom           string   'female' (PATO:0000383) | 'male' (PATO:0000384) | 'ambiguous'
    sex_score           float32  posterior of that label
    ancestry_hancestro  string   HANCESTRO genetic ancestry category of the cell's donor
    ancestry_prob       float32  proportion of that ancestry in the donor's estimate
    model_version       string   versions of every model that produced the row

Every column may be null where it was not computed. :func:`assert_public_safe` is called on every table before it is written
and rejects anything that is not exactly this schema or that carries genotype-like content.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import numpy as np

SCHEMA_VERSION = "1"
COLUMNS = (
    ("gsm", "string"),
    ("col", "int64"),
    ("cell_type_cl", "string"),
    ("cell_type_label", "string"),
    ("cell_type_score", "float32"),
    ("sex_chrom", "string"),
    ("sex_score", "float32"),
    ("ancestry_hancestro", "string"),
    ("ancestry_prob", "float32"),
    ("model_version", "string"),
)
SEX_LABELS = frozenset({"female", "male", "ambiguous"})
_GSM = re.compile(r"^GSM\d+$")
_CL = re.compile(r"^CL:\d{7}$")
_HANCESTRO = re.compile(r"^HANCESTRO:\d{4}$")
# Anything that looks like a variant, a genotype or an allele: rsIDs, chr:pos, VCF genotype fields, bare nucleotide strings.
_GENOTYPE_LIKE = re.compile(
    r"(\brs\d{3,}\b)|(\bchr[0-9XYMT]{1,2}[:_]\d+)|(\b\d{1,2}[:_]\d{4,}[:_][ACGT])|(\b[01.][/|][01.]\b)|(\b[ACGT]>[ACGT]\b)"
    r"|(^[ACGTN]{2,}$)|(\bGT:)|(\bAD[=:]\d)|(\bDP[=:]\d)",
    re.IGNORECASE,
)


class PublicOutputError(ValueError):
    """A table destined for public storage failed the privacy / schema check."""


def _arrow_schema():
    import pyarrow as pa

    m = {"string": pa.string(), "int64": pa.int64(), "float32": pa.float32()}
    return pa.schema(
        [(n, m[t]) for n, t in COLUMNS],
        metadata={b"singlet_cell_labels_schema": SCHEMA_VERSION.encode()},
    )


def assert_public_safe(table) -> None:
    """Raise :class:`PublicOutputError` unless ``table`` (pandas DataFrame or pyarrow Table) is a valid label table.

    Checked: the column set and dtypes are exactly :data:`COLUMNS` (so no depth, allele or per-site column can ride along); every
    value of every string column matches its vocabulary (GSM accession, CL id, HANCESTRO id, a sex label, an ontology name, a
    version string); no string anywhere looks like a variant id, a genotype or an allele; scores are probabilities."""
    import pandas as pd
    import pyarrow as pa

    df = table.to_pandas() if isinstance(table, pa.Table) else table
    if not isinstance(df, pd.DataFrame):
        raise PublicOutputError("expected a DataFrame or an Arrow table")
    names = [n for n, _ in COLUMNS]
    if list(df.columns) != names:
        raise PublicOutputError(f"columns must be exactly {names}, got {list(df.columns)}")
    for n, t in COLUMNS:
        s = df[n]
        if t == "int64" and not (pd.api.types.is_integer_dtype(s) and (s >= 0).all()):
            raise PublicOutputError(f"{n} must be non-negative integers")
        if t == "float32":
            v = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
            fin = v[np.isfinite(v)]
            if s.notna().sum() != len(fin) or ((fin < 0) | (fin > 1)).any():
                raise PublicOutputError(f"{n} must be null or in [0, 1]")
    checks = {
        "gsm": lambda x: bool(_GSM.match(x)),
        "cell_type_cl": lambda x: bool(_CL.match(x)),
        "sex_chrom": lambda x: x in SEX_LABELS,
        "ancestry_hancestro": lambda x: bool(_HANCESTRO.match(x)),
        "cell_type_label": lambda x: len(x) <= 200 and not _GENOTYPE_LIKE.search(x),
        "model_version": lambda x: (
            len(x) <= 400
            and bool(re.match(r"^[A-Za-z0-9_.;=:/+ \-()]+$", x))
            and not _GENOTYPE_LIKE.search(x)
        ),
    }
    for n, ok in checks.items():
        vals = df[n].dropna().astype(str).unique()
        bad = [v for v in vals if not ok(v)]
        if bad:
            raise PublicOutputError(
                f"{n} holds a value that is not allowed in public output: {bad[0][:60]!r}"
            )
    if bool(df["gsm"].isna().any()) or bool(df["col"].isna().any()):
        raise PublicOutputError("gsm and col are required on every row")
    if bool(df["col"].duplicated().any()):
        raise PublicOutputError("col must be unique within a sample")


def scan_bytes_for_genotypes(data: bytes) -> list:
    """Genotype-like strings found in text bytes (logs, JSON). Do not use it on compressed files: in the atlas run, random byte
    runs in zstd pages matched (``C>C``, ``0/1``) in 446 of 1,012 label files whose decoded content was clean; use
    :func:`scan_labels_file`."""
    txt = data.decode("utf-8", errors="ignore")
    return [m.group(0) for m in _GENOTYPE_LIKE.finditer(txt)][:20]


def scan_labels_file(path: Union[str, Path]) -> list:
    """Genotype-like strings in a written label file: every decoded string value, the column names and the file metadata."""
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    parts = list(t.column_names) + [f"{k!r}={v!r}" for k, v in (t.schema.metadata or {}).items()]
    for name in t.column_names:
        col = t.column(name)
        if col.type == "string" or str(col.type).startswith("dictionary"):
            parts += [str(x) for x in col.to_pylist() if x is not None]
    return scan_bytes_for_genotypes("\n".join(parts).encode())


def write_labels(df, path: Union[str, Path]) -> Path:
    """Check ``df`` with :func:`assert_public_safe` and write it as Parquet (zstd) atomically."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert_public_safe(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab = pa.Table.from_pandas(df, schema=_arrow_schema(), preserve_index=False)
    tmp = path.with_name(path.name + ".part")
    pq.write_table(tab, tmp, compression="zstd")
    tmp.replace(path)
    return path


def empty_table(gsm: str, cols: np.ndarray, model_version: str):
    """All-null label table for the given cells."""
    import pandas as pd

    n = len(cols)
    return pd.DataFrame(
        {
            "gsm": pd.array([gsm] * n, dtype="string"),
            "col": np.asarray(cols, np.int64),
            "cell_type_cl": pd.array([None] * n, dtype="string"),
            "cell_type_label": pd.array([None] * n, dtype="string"),
            "cell_type_score": np.full(n, np.nan, np.float32),
            "sex_chrom": pd.array([None] * n, dtype="string"),
            "sex_score": np.full(n, np.nan, np.float32),
            "ancestry_hancestro": pd.array([None] * n, dtype="string"),
            "ancestry_prob": np.full(n, np.nan, np.float32),
            "model_version": pd.array([model_version] * n, dtype="string"),
        }
    )
