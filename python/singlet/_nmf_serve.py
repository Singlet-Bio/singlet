# SPDX-License-Identifier: MIT
"""Retired: hosted NMF model serving (``transform``/``annotate``).

Both called ``https://api.singlet.bio/v1/nmf``, which was never brought into
service, and now raise :class:`NotImplementedError`. :func:`singlet.nmf`
factorises a matrix locally.
"""

from __future__ import annotations

from singlet._auth import retired


def transform(adata, *, model: str = "human_global_k100"):
    """Retired — raises :class:`NotImplementedError`. Use :func:`singlet.nmf`."""
    raise NotImplementedError(retired("NMF model serving (transform)"))


def annotate(adata, *, model: str = "human_global_k100") -> dict:
    """Retired — raises :class:`NotImplementedError`. Use :func:`singlet.nmf`."""
    raise NotImplementedError(retired("NMF model serving (annotate)"))
