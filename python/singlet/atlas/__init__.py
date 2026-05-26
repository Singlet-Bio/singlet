# SPDX-License-Identifier: MIT
"""
singlet.atlas — Cloudflare-native published-atlas client.

Per MVP_ROADMAP.md BLOCKER #8, the Singlet atlas catalog is published as
``r2://singlet-atlas/catalog/human_10x_atlas.parquet`` and served through a
Cloudflare Worker at ``api.singlet.bio``. There is no Supabase, no Postgres,
no auth — the catalog is public, read-only, and edge-cached.

Note: this module is distinct from the top-level ``singlet.catalog()``
function (which browses the user's local catalog cache). ``singlet.atlas``
specifically targets the published Cloudflare-hosted atlas.

This module exposes three thin clients:

    from singlet.atlas import index, sample, search

    df = index()                         # pandas.DataFrame from the Parquet
    info = sample("GSM3308814")           # dict from /api/sample/:gsm_id
    hits = search(tissue="lung",
                  protocol="10xv3")     # dict[str, list] from /api/search

Implementation notes:
- The Parquet is fetched via HTTP range requests with PyArrow, so loading
  the full catalog is bandwidth-cheap and trivially cacheable on disk.
- ``sample()`` and ``search()`` go through the Worker, which is in turn
  cached at the Cloudflare edge.
- All clients fall back to ``SINGLET_CDN_BASE`` / ``SINGLET_API_BASE``
  env vars for local development against staging endpoints.
"""
from __future__ import annotations

from .client import index, sample, search

__all__ = ["index", "sample", "search"]
