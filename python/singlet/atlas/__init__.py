# SPDX-License-Identifier: MIT
"""
singlet.atlas — retired skeleton for a Cloudflare-hosted atlas client.

This module was a placeholder for a client of ``r2.singlet.bio`` (a Parquet
atlas index) and ``api.singlet.bio`` (sample/search Worker). Neither host was
brought into service, and every function here raises
:class:`NotImplementedError`. The module is kept only so old imports fail
with a helpful message.

Use instead:

    import singlet
    singlet.find("human lung 10x")        # search → GSE accessions (live)
    singlet.load("GSE138867")              # download + assemble → AnnData
    singlet.info("GSE138867")              # metadata (snapshot, then live API)
    singlet.catalog()                      # offline catalog snapshot
"""

from __future__ import annotations

from .client import index, sample, search

__all__ = ["index", "sample", "search"]
