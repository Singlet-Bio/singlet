# SPDX-License-Identifier: MIT
"""Per-cell labels for the atlas cell index: cell type (Cell Ontology), chromosomal sex, inferred genetic ancestry.

::

    from singlet.celllabels import read_store, annotate_sample, write_labels
    s = read_store("GSE145370", "GSM4317426")
    table, summary = annotate_sample(s)
    write_labels(table, "GSM4317426.labels.parquet")

Modules: :mod:`~singlet.celllabels.celltype` (marker guide + per-sample self-training), :mod:`~singlet.celllabels.sex`,
:mod:`~singlet.celllabels.ancestry` (from restricted donor SNP depths; only labels leave it), :mod:`~singlet.celllabels.output`
(the interchange schema and the public-output guard), :mod:`~singlet.celllabels.io` (store and legacy readers).
``docs/cell_annotation.md`` documents the methods and their validation.
"""

from .celltype import CellTypeResult, annotate_cell_types
from .io import SampleInput, read_legacy, read_store, read_v2  # noqa: F401  (read_v2: DUAL old name)
from .output import COLUMNS, PublicOutputError, assert_public_safe, write_labels
from .pipeline import ANNOTATE_VERSION, annotate_sample, model_version
from .sex import call_sex

__all__ = [
    "ANNOTATE_VERSION",
    "COLUMNS",
    "CellTypeResult",
    "PublicOutputError",
    "SampleInput",
    "annotate_cell_types",
    "annotate_sample",
    "assert_public_safe",
    "call_sex",
    "model_version",
    "read_legacy",
    "read_store",
    "write_labels",
]
