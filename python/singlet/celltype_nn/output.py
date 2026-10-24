# SPDX-License-Identifier: MIT
"""The per-cell cell type probability sidecar and the guard every table passes before it is written.

One Parquet file per sample, ``<GSM>.celltype.parquet``, one row per called cell, joined to everything else on ``(gsm, col)``
(``docs/CELL-ARCHITECTURE.md``)::

    gsm               string   sample accession
    col               int64    column of the cell in the sample's v2 ``cells`` table / matrix column order
    cell_type_cl      string   the call: the most specific CL term with rolled-up probability >= 0.5 (null: only the root)
    cell_type_label   string   CL name of that term
    cell_type_prob    float32  rolled-up probability of the call
    argmax_cl         string   output term with the most probability mass
    argmax_prob       float32  its probability
    top1_cl .. top5_cl      string   the five output terms with the most mass, in order
    top1_prob .. top5_prob  float32  their probabilities (the rest of the mass is spread over the other terms)
    model_version     string   e.g. ``celltype-nn=0.1.0;census=2025-11-08;cl=2026-06-08``

``cell_type_cl`` / ``cell_type_label`` mean the same as in the label file (:mod:`singlet.celllabels.output`), so the cell
index can take the network's call in place of the guide's. Parquet metadata ``singlet_celltype_probs_schema = "1"``. The
probability of any ancestor term is the sum of the top-k probabilities of its descendants (a lower bound; exact from the model).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import numpy as np

SCHEMA_VERSION = "1"
TOP_K = 5
COLUMNS = (
    [
        ("gsm", "string"),
        ("col", "int64"),
        ("cell_type_cl", "string"),
        ("cell_type_label", "string"),
        ("cell_type_prob", "float32"),
        ("argmax_cl", "string"),
        ("argmax_prob", "float32"),
    ]
    + [(f"top{i}_cl", "string") for i in range(1, TOP_K + 1)]
    + [(f"top{i}_prob", "float32") for i in range(1, TOP_K + 1)]
    + [("model_version", "string")]
)
_GSM = re.compile(r"^GSM\d+$")
_CL = re.compile(r"^CL:\d{7}$")
_VERSION = re.compile(r"^[A-Za-z0-9_.:;=/+-]{1,200}$")
_LABEL = re.compile(r"^[^\t\n\r]{1,200}$")


class PublicOutputError(ValueError):
    """A table destined for public storage failed the schema check."""


def arrow_schema():
    import pyarrow as pa

    m = {"string": pa.string(), "int64": pa.int64(), "float32": pa.float32()}
    return pa.schema([(n, m[t]) for n, t in COLUMNS], metadata={b"singlet_celltype_probs_schema": SCHEMA_VERSION.encode()})


def assert_public_safe(table) -> None:
    """Raise :class:`PublicOutputError` unless ``table`` is exactly the sidecar schema with valid values: GSM accessions, unique
    non-negative ``col``, CL ids, CL names, probabilities in [0, 1] with top-k descending and summing to at most 1, a version
    string. Nothing else (genes, counts, barcodes, genotypes) can ride along because no other column is accepted."""
    import pandas as pd
    import pyarrow as pa

    df = table.to_pandas() if isinstance(table, pa.Table) else table
    if not isinstance(df, pd.DataFrame):
        raise PublicOutputError("expected a DataFrame or an Arrow table")
    names = [n for n, _ in COLUMNS]
    if list(df.columns) != names:
        raise PublicOutputError(f"columns {list(df.columns)} != {names}")
    if len(df) == 0:
        return
    if not df["gsm"].map(lambda x: isinstance(x, str) and bool(_GSM.match(x))).all() or df["gsm"].nunique() != 1:
        raise PublicOutputError("gsm must be one GSM accession")
    col = df["col"].to_numpy()
    if col.dtype.kind not in "iu" or (col < 0).any() or len(np.unique(col)) != len(col):
        raise PublicOutputError("col must be unique non-negative integers")
    cl_cols = ["cell_type_cl", "argmax_cl"] + [f"top{i}_cl" for i in range(1, TOP_K + 1)]
    for c in cl_cols:
        v = df[c]
        bad = v.notna() & ~v.map(lambda x: isinstance(x, str) and bool(_CL.match(x)))
        if bad.any():
            raise PublicOutputError(f"{c}: not a CL id: {v[bad].iloc[0]!r}")
    if df["argmax_cl"].isna().any() or df["top1_cl"].isna().any():
        raise PublicOutputError("argmax_cl and top1_cl are always set")
    lab = df["cell_type_label"]
    if (lab.notna() != df["cell_type_cl"].notna()).any():
        raise PublicOutputError("cell_type_label must be set exactly where cell_type_cl is")
    if not lab.dropna().map(lambda x: bool(_LABEL.match(str(x)))).all():
        raise PublicOutputError("cell_type_label: invalid name")
    probs = ["cell_type_prob", "argmax_prob"] + [f"top{i}_prob" for i in range(1, TOP_K + 1)]
    for c in probs:
        v = df[c].to_numpy(np.float64)
        v = v[~np.isnan(v)] if c == "cell_type_prob" else v
        if np.isnan(v).any() or (v < -1e-6).any() or (v > 1 + 1e-5).any():
            raise PublicOutputError(f"{c}: not a probability")
    T = df[[f"top{i}_prob" for i in range(1, TOP_K + 1)]].to_numpy(np.float64)
    if (np.diff(T, axis=1) > 1e-6).any() or (T.sum(1) > 1 + 1e-4).any():
        raise PublicOutputError("top-k probabilities must be descending and sum to at most 1")
    if not np.allclose(df["argmax_prob"].to_numpy(np.float64), T[:, 0], atol=1e-6):
        raise PublicOutputError("argmax_prob must equal top1_prob")
    if not df["model_version"].map(lambda x: isinstance(x, str) and bool(_VERSION.match(x))).all():
        raise PublicOutputError("model_version: invalid")


def write_sidecar(table, path: Union[str, Path]) -> Path:
    """Validate and write one sample's sidecar."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert_public_safe(table)
    t = pa.Table.from_pandas(table, schema=arrow_schema(), preserve_index=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, path, compression="zstd")
    return path
