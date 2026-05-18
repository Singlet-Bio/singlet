# SPDX-License-Identifier: MIT
"""Authentication for token-priced features."""

from __future__ import annotations

import os
from typing import Optional

_API_KEY: Optional[str] = None
_API_BASE = "https://api.singlet.bio/v1"


def login(api_key: Optional[str] = None) -> None:
    """Authenticate for token-priced features.

    Parameters
    ----------
    api_key : str, optional
        Your SingletDB API key. If not provided, reads from the
        ``SINGLET_API_KEY`` environment variable.

    Raises
    ------
    ValueError
        If no API key is provided or found in the environment.
    """
    global _API_KEY
    key = api_key or os.environ.get("SINGLET_API_KEY")
    if not key:
        raise ValueError(
            "No API key provided. Pass api_key= or set SINGLET_API_KEY "
            "environment variable. Get your key at https://singlet.bio/pricing"
        )
    _API_KEY = key


def _get_key() -> str:
    """Return the current API key, or raise."""
    key = _API_KEY or os.environ.get("SINGLET_API_KEY")
    if not key:
        raise RuntimeError(
            "Token-priced feature requires authentication. "
            "Call singlet.login('sk-...') or set SINGLET_API_KEY."
        )
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {_get_key()}", "User-Agent": "singlet-python"}


def _get_headers() -> dict:
    """Return auth headers for AWS streaming."""
    return _headers()
