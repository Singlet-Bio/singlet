# SPDX-License-Identifier: MIT
"""Retired: cross-atlas ``query()`` and cell-level ``search()``.

Both streamed cells from ``https://api.singlet.bio/v1``, which was never
brought into service. They now raise :class:`NotImplementedError` pointing
at the working replacements: :func:`singlet.find` (natural-language search
over studies/samples) followed by :func:`singlet.load`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from singlet._auth import retired

if TYPE_CHECKING:
    from anndata import AnnData


def query(
    *,
    species: Optional[Union[str, list]] = None,
    tissue: Optional[Union[str, list]] = None,
    disease: Optional[Union[str, list]] = None,
    cell_type: Optional[Union[str, list]] = None,
    perturbation: Optional[str] = None,
    developmental_stage: Optional[str] = None,
    sex: Optional[str] = None,
    modality: Optional[str] = None,
    min_cells: int = 0,
    max_results: int = 100_000,
) -> AnnData:
    """Retired — raises :class:`NotImplementedError`.

    Use ``singlet.load(singlet.find("human lung 10x"))`` instead.
    """
    raise NotImplementedError(retired("singlet.query()"))


def search(text: str, max_results: int = 100_000) -> AnnData:
    """Retired — raises :class:`NotImplementedError`.

    Use :func:`singlet.find` (returns accessions) and :func:`singlet.load`.
    """
    raise NotImplementedError(retired("singlet.search()"))
