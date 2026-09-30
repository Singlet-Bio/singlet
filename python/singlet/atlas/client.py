# SPDX-License-Identifier: MIT
"""Retired skeleton of a Cloudflare-hosted atlas client (never implemented)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def _retired(name: str, use: str) -> str:
    return (
        f"singlet.atlas.{name}() is not available: it targeted r2.singlet.bio / "
        f"api.singlet.bio, which are not in service. Use {use} instead."
    )


def index() -> "pd.DataFrame":
    """Retired — raises :class:`NotImplementedError`. Use :func:`singlet.catalog`."""
    raise NotImplementedError(
        _retired("index", "singlet.catalog() (offline snapshot) or singlet.find('…')")
    )


def sample(gsm_id: str) -> dict[str, Any]:
    """Retired — raises :class:`NotImplementedError`. Use :func:`singlet.info`."""
    raise NotImplementedError(
        _retired("sample", f"singlet.info({gsm_id!r}) or singlet.load({gsm_id!r})")
    )


def search(**filters: Any) -> dict[str, Any]:
    """Retired — raises :class:`NotImplementedError`. Use :func:`singlet.find`."""
    raise NotImplementedError(_retired("search", "singlet.find('…') and singlet.load(...)"))
