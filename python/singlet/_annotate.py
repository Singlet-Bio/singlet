# SPDX-License-Identifier: MIT
"""Retired: NMF gene-program annotation (``gene_programs``/``project``/``annotate``).

These functions downloaded pre-trained NMF gene-program models from
``https://models.singlet.bio``. That host was never brought into service, so
every call failed with a connection error. They now raise
:class:`NotImplementedError` straight away with a pointer to what works.

Local, model-free alternatives that ship with the package:
:func:`singlet.annotate_cell_types` (marker-gene scoring),
:func:`singlet.predict_cell_type` and :func:`singlet.transfer_labels`
(label transfer from a reference AnnData, e.g. one loaded with
:func:`singlet.load`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def _retired(name: str) -> str:
    return (
        f"singlet.{name}() is not available: it needed NMF gene-program models from "
        "https://models.singlet.bio, which is not in service. Load data with "
        "singlet.load('GSE…') (search with singlet.find('…')) and annotate locally with "
        "singlet.annotate_cell_types(), singlet.predict_cell_type() or "
        "singlet.transfer_labels()."
    )


def gene_programs(organism: str, k: int = 100) -> pd.DataFrame:
    """Retired — raises :class:`NotImplementedError` (model host not in service)."""
    raise NotImplementedError(_retired("gene_programs"))


def project(adata, *, organism: Optional[str] = None, k: int = 100) -> np.ndarray:
    """Retired — raises :class:`NotImplementedError` (model host not in service)."""
    raise NotImplementedError(_retired("project"))


def annotate(
    adata, *, organism: Optional[str] = None, k: int = 100, inplace: bool = False
) -> pd.DataFrame:
    """Retired — raises :class:`NotImplementedError` (model host not in service)."""
    raise NotImplementedError(_retired("annotate"))
