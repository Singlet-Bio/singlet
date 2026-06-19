#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Singlet MCP Server — Model Context Protocol server for the Singlet Atlas.

Exposes these tools to any MCP client (Claude Desktop, Cursor, VS Code Copilot):
  - singlet_stats:      Get corpus-wide statistics (samples, cells, species)
  - singlet_nl_search:  Natural-language search → interpreted filters + accessions
  - singlet_search:     Search processed single-cell datasets
  - singlet_qc:        Get QC metrics for a sample
  - singlet_load:      Get download/access info for a sample
  - singlet_browse:    Browse samples with filters (organism, protocol, tissue, status)
  - singlet_protocols: Get protocol distribution and success rates
  - singlet_quality:   Get quality tier breakdown (gold/silver/bronze)
  - singlet_tissues:   Get tissue distribution across samples
  - singlet_failures:  Get failure category breakdown for non-SUCCESS samples
  - singlet_cell_types: Get cell type distribution (PBMC, T cells, stem cells, etc.)
  - singlet_species:   Get species list with sample counts

Usage:
    # Start the server (stdio transport):
    python -m singlet.mcp

    # Or use the entry point:
    singlet

    # Configure in Claude Desktop's claude_desktop_config.json:
    {
      "mcpServers": {
        "singlet": {
          "command": "python",
          "args": ["-m", "singlet.mcp"]
        }
      }
    }

The live tools (search, qc, load, browse) call the public Singlet REST API at
https://singlet.bio/api (override with $SINGLET_API_BASE). The aggregate tools
(stats, protocols, quality, tissues, ...) read the bundled catalog parquet.

Requires: pip install mcp
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError:
    sys.exit(
        "MCP SDK not installed. Run: pip install mcp\n"
        "See: https://github.com/modelcontextprotocol/python-sdk"
    )


# ─── Configuration ───────────────────────────────────────────────────────────

API_BASE = os.environ.get("SINGLET_API_BASE", "https://singlet.bio/api").rstrip("/")
_USER_AGENT = "singlet-mcp/1"


def _api_get(path: str, params: dict | None = None) -> Any:
    """GET ``{API_BASE}{path}`` (with optional query params) → parsed JSON.

    ``path`` should start with ``/`` (e.g. ``/search``). Returns the decoded
    JSON payload, or raises RuntimeError on transport/HTTP failure.
    """
    url = f"{API_BASE}{path}"
    if params:
        clean = {k: v for k, v in params.items() if v is not None and v != ""}
        if clean:
            url = f"{url}?{urllib.parse.urlencode(clean)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API request failed: {url} → HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API request failed: {url} → {e.reason}") from e


# ─── MCP Server Setup ────────────────────────────────────────────────────────

app = Server("singlet")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="singlet_stats",
            description=(
                "Get corpus-wide statistics for the Singlet Atlas: total samples, "
                "total cells, success rate, species count, average QC metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="singlet_nl_search",
            description=(
                "Natural-language search over the Singlet Atlas. Give a plain-English "
                "description (e.g. 'human lung fibroblasts', 'exhausted T cells in "
                "melanoma') and get back the interpreted filters plus matching GEO "
                "accessions, ready to pass to singlet.load()."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Plain-English description of the data you want.",
                    },
                    "level": {
                        "type": "string",
                        "enum": ["gsm", "gse"],
                        "description": "Accession granularity: 'gsm' (samples, default) or 'gse' (series).",
                        "default": "gsm",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max accessions to return (default 50).",
                        "default": 50,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="singlet_search",
            description=(
                "Search for single-cell RNA-seq samples in the Singlet Atlas. "
                "Filter by organism, protocol (10xv3, 10xv2, dropseq, etc.), "
                "modality (scrna, cite, multiome, atac, visium), or free-text query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "organism": {
                        "type": "string",
                        "description": "Filter by organism (e.g. 'Homo sapiens', 'Mus musculus')",
                    },
                    "protocol": {
                        "type": "string",
                        "description": "Filter by protocol (e.g. '10xv3', '10xv2', 'dropseq')",
                    },
                    "modality": {
                        "type": "string",
                        "description": "Filter by modality (scrna, cite, multiome, atac, visium)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Free-text search across GSM ID, GSE ID, title, and tissue/source",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["SUCCESS", "SOFT_FAIL", "HARD_FAIL", "PENDING"],
                        "description": "Filter by processing status",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 20, max 100)",
                        "default": 20,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="singlet_qc",
            description=(
                "Get detailed QC metrics and metadata for a specific sample. "
                "Returns mapping rate, cells called, median genes/UMIs, "
                "mitochondrial %, doublet rate, processing time, and more."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "gsm_id": {
                        "type": "string",
                        "description": "GEO sample accession (e.g. 'GSM5238385')",
                    },
                },
                "required": ["gsm_id"],
            },
        ),
        Tool(
            name="singlet_load",
            description=(
                "Get access information for loading a processed sample. "
                "Returns the file path, size, and Python code to load it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "gsm_id": {
                        "type": "string",
                        "description": "GEO sample accession (e.g. 'GSM5238385')",
                    },
                },
                "required": ["gsm_id"],
            },
        ),
        Tool(
            name="singlet_browse",
            description=(
                "Browse samples with pagination. Use for discovering what's "
                "available in the atlas. Filter by organism, tissue, or status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Page number (0-indexed, default 0)",
                        "default": 0,
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Results per page (default 25, max 100)",
                        "default": 25,
                    },
                    "organism": {
                        "type": "string",
                        "description": "Filter by organism (e.g. 'Homo sapiens')",
                    },
                    "tissue": {
                        "type": "string",
                        "description": "Filter by tissue/source (substring match, e.g. 'brain', 'lung', 'PBMC')",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status (SUCCESS, SOFT_FAIL, HARD_FAIL)",
                        "enum": ["SUCCESS", "SOFT_FAIL", "HARD_FAIL"],
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["pipeline_date", "cells_called", "mapping_rate"],
                        "description": "Sort field (default: pipeline_date desc)",
                        "default": "pipeline_date",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="singlet_protocols",
            description=(
                "Get protocol distribution across the atlas. Shows which "
                "single-cell sequencing protocols (10xv3, 10xv2, dropseq, etc.) "
                "are represented and their sample counts and success rates."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="singlet_quality",
            description=(
                "Get quality tier breakdown of SUCCESS samples. Classifies into "
                "gold (MR>=0.7, genes>=500, cells>=500), silver (MR>=0.5, genes>=200, "
                "cells>=100), and bronze (all other SUCCESS)."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="singlet_tissues",
            description=(
                "Get tissue distribution across processed samples. Returns normalized "
                "tissue categories (e.g., brain, blood, lung, pbmc, tumor) extracted "
                "from GEO characteristics metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top tissues to return (default: 25)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="singlet_failures",
            description=(
                "Get failure category breakdown for non-SUCCESS samples. Shows why samples "
                "failed: download_fail, align_low_map, cells_below_threshold, pipeline_crash, "
                "align_oom. Useful for understanding pipeline health."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="singlet_cell_types",
            description=(
                "Get cell type distribution across SUCCESS samples. Returns normalized "
                "cell type categories (PBMC, T cells, stem cells, cell lines, organoids, "
                "fibroblasts, etc.) extracted from GEO source annotations. Covers ~34% "
                "of SUCCESS samples."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top cell types to return (default: 20)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="singlet_species",
            description=(
                "Get species list with sample counts. Returns all organisms with "
                "successfully processed data in the atlas, deduplicated from combo "
                "organisms (e.g. 'Homo sapiens;Mus musculus' split into separate species)."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to implementations."""
    try:
        if name == "singlet_stats":
            result = await _tool_stats()
        elif name == "singlet_nl_search":
            result = await _tool_nl_search(arguments)
        elif name == "singlet_search":
            result = await _tool_search(arguments)
        elif name == "singlet_qc":
            result = await _tool_qc(arguments)
        elif name == "singlet_load":
            result = await _tool_load(arguments)
        elif name == "singlet_browse":
            result = await _tool_browse(arguments)
        elif name == "singlet_protocols":
            result = await _tool_protocols()
        elif name == "singlet_quality":
            result = await _tool_quality()
        elif name == "singlet_tissues":
            result = await _tool_tissues(arguments)
        elif name == "singlet_failures":
            result = await _tool_failures()
        elif name == "singlet_cell_types":
            result = await _tool_cell_types(arguments)
        elif name == "singlet_species":
            result = await _tool_species()
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ─── Tool Implementations ────────────────────────────────────────────────────


async def _tool_stats() -> dict:
    """Get corpus-wide statistics from bundled parquet data."""
    import singlet

    df = singlet.samples()
    success = df[df["status"] == "SUCCESS"]

    # Species breakdown
    species_list = singlet.species()
    species_counts = []
    for sp in species_list:
        mask = success["organism"].str.contains(sp, na=False, regex=False)
        count = int(mask.sum())
        cells = int(success.loc[mask, "cells_called"].sum())
        species_counts.append({"organism": sp, "samples": count, "cells": cells})
    species_counts.sort(key=lambda x: x["samples"], reverse=True)

    return {
        "total_samples": len(df),
        "successful_samples": len(success),
        "total_cells": int(success["cells_called"].sum()),
        "species_count": len(species_list),
        "series_count": int(df["gse_id"].nunique()),
        "success_rate": round(len(success) / len(df), 4) if len(df) else 0,
        "avg_mapping_rate": round(float(success["mapping_rate"].mean()), 4) if len(success) else 0,
        "avg_median_genes": round(float(success["median_genes"].mean()), 1) if len(success) else 0,
        "top_species": species_counts[:10],
    }


def _lookup_gsm(gsm_id: str) -> dict | None:
    """Fetch a single GSM record from the public API, or None if not found.

    Uses ``/api/gsm?q=<GSM>`` (the ``/api/gsm/<id>`` detail endpoint is
    currently unreliable) and falls back to ``/api/search``.
    """
    try:
        payload = _api_get("/gsm", {"q": gsm_id, "limit": 5})
        for row in payload.get("data") or []:
            if row.get("gsm_id") == gsm_id:
                return row
    except RuntimeError:
        pass
    try:
        payload = _api_get("/search", {"q": gsm_id})
        for row in payload.get("gsm") or []:
            if row.get("gsm_id") == gsm_id:
                return row
    except RuntimeError:
        pass
    return None


async def _tool_nl_search(args: dict) -> dict:
    """Natural-language search via the public ``/nl-search`` endpoint.

    Returns the interpreted filters (if the endpoint reports them) plus the
    matching GEO accessions.
    """
    query = args.get("query")
    if not query or not str(query).strip():
        return {"error": "query is required"}
    level = args.get("level", "gsm")
    limit = min(int(args.get("limit", 50)), 500)

    payload = _api_get("/nl-search", {"q": query, "level": level, "limit": limit})

    accessions = payload.get("accessions")
    if accessions is None:
        accessions = payload.get("results") or []
    accessions = [str(a) for a in accessions][:limit]

    return {
        "query": query,
        "level": level,
        "filters": payload.get("filters", payload.get("interpreted", {})),
        "count": len(accessions),
        "accessions": accessions,
        "load_hint": "singlet.load(accessions)  # → one AnnData",
    }


async def _tool_search(args: dict) -> dict:
    """Search samples with filters via the public REST API."""
    limit = min(args.get("limit", 20), 100)

    # /api/search?q=<text> is the free-text endpoint; /api/gsm supports
    # structured filters. Prefer /api/gsm when structured filters are present.
    structured = any(args.get(k) for k in ("organism", "protocol", "status"))
    samples: list[dict] = []

    if args.get("query") and not structured:
        payload = _api_get("/search", {"q": args["query"]})
        rows = payload.get("gsm") or []
        samples = rows[:limit]
    else:
        payload = _api_get(
            "/gsm",
            {
                "q": args.get("query"),
                "organism": args.get("organism"),
                "protocol": args.get("protocol"),
                "status": args.get("status"),
                "limit": limit,
            },
        )
        samples = payload.get("data") or []

    # Optional client-side modality filter (not a first-class API param).
    if args.get("modality"):
        m = args["modality"].lower()
        samples = [s for s in samples if str(s.get("modality", "")).lower() == m]

    return {
        "count": len(samples),
        "samples": samples,
    }


async def _tool_qc(args: dict) -> dict:
    """Get QC metrics for a sample via the public REST API."""
    gsm_id = args["gsm_id"]
    sample = _lookup_gsm(gsm_id)
    if not sample:
        return {"error": f"Sample {gsm_id} not found in atlas"}

    return {
        "gsm_id": sample.get("gsm_id"),
        "gse_id": sample.get("gse_id"),
        "organism": sample.get("organism"),
        "protocol": sample.get("protocol"),
        "modality": sample.get("modality"),
        "status": sample.get("status"),
        "tissue": sample.get("tissue", ""),
        "cell_type": sample.get("cell_type", ""),
        "qc_metrics": {
            "mapping_rate": sample.get("mapping_rate"),
            "cells_called": sample.get("n_cells", sample.get("cells_called")),
            "median_genes": sample.get("median_genes"),
            "median_umis": sample.get("median_umis"),
            "mt_pct": sample.get("mt_pct"),
            "qc_flag": sample.get("qc_flag"),
        },
        "processing": {
            "singlet_version": sample.get("singlet_version"),
            "pipeline_date": sample.get("pipeline_date"),
        },
        "title": sample.get("title"),
        "source": sample.get("source"),
        "web_url": f"https://singlet.bio/sample/{gsm_id}",
    }


async def _tool_load(args: dict) -> dict:
    """Get load/access info for a sample, plus the public R2 bundle URL.

    The public data is hosted per-GSE on Cloudflare R2. This returns the parent
    GSE, the direct bundle URL, and a ready-to-run code snippet (which calls
    ``singlet.load()`` to fetch + assemble the data).
    """
    gsm_id = args["gsm_id"]
    sample = _lookup_gsm(gsm_id)
    if not sample:
        return {"error": f"Sample {gsm_id} not found in atlas"}

    gse_id = sample.get("gse_id")
    data_base = os.environ.get("SINGLET_DATA_BASE", "https://data.singlet.bio").rstrip("/")
    bundle_url = f"{data_base}/data/{gse_id}/{gse_id}.singlet" if gse_id else None

    return {
        "gsm_id": gsm_id,
        "gse_id": gse_id,
        "organism": sample.get("organism"),
        "protocol": sample.get("protocol"),
        "status": sample.get("status"),
        "title": sample.get("title", ""),
        "tissue": sample.get("tissue", ""),
        "cell_type": sample.get("cell_type", ""),
        "cells": sample.get("n_cells", sample.get("cells_called")),
        "format": ".singlet bundle (per-GSE ZIP; free, no auth)",
        "bundle_url": bundle_url,
        "python_code": (
            f'import singlet\n'
            f'# Loads just this sample (downloads the parent {gse_id} bundle, cached):\n'
            f'adata = singlet.load("{gsm_id}")\n'
            f'# ...or load the whole series:\n'
            f'adata = singlet.load("{gse_id}")'
        ),
        "web_url": f"https://singlet.bio/sample/{gsm_id}",
    }


async def _tool_browse(args: dict) -> dict:
    """Browse samples with pagination via the public REST API."""
    page = args.get("page", 0)
    page_size = min(args.get("page_size", 25), 100)

    payload = _api_get(
        "/gsm",
        {
            "page": page,
            "limit": page_size,
            "organism": args.get("organism"),
            "status": args.get("status"),
            "q": args.get("tissue"),
        },
    )
    samples = payload.get("data") or []

    return {
        "page": payload.get("page", page),
        "page_size": payload.get("page_size", page_size),
        "total": payload.get("total", 0),
        "samples": samples,
    }


async def _tool_protocols() -> dict:
    """Get protocol distribution across the atlas."""
    import singlet

    df = singlet.samples()
    # Normalize protocol column
    df = df.copy()
    df["protocol"] = df["protocol"].fillna("unknown").replace("", "unknown")

    # Aggregate by protocol
    protocols = []
    for protocol, group in df.groupby("protocol"):
        total = len(group)
        success = int((group["status"] == "SUCCESS").sum())
        protocols.append(
            {
                "protocol": protocol,
                "total_samples": total,
                "success_samples": success,
                "success_rate": round(success / total, 3) if total else 0,
            }
        )

    protocols.sort(key=lambda x: x["total_samples"], reverse=True)
    return {
        "total_protocols": len([p for p in protocols if p["protocol"] != "unknown"]),
        "protocols": protocols,
    }


async def _tool_quality() -> dict:
    """Get quality tier breakdown of SUCCESS samples."""
    import singlet

    tiers_df = singlet.quality_tiers()
    total = int(tiers_df["count"].sum())

    tiers = {}
    criteria_map = {
        "gold": "mapping_rate>=0.7, median_genes>=500, cells>=500",
        "silver": "mapping_rate>=0.5, median_genes>=200, cells>=100",
        "bronze": "all other SUCCESS samples",
    }
    for _, row in tiers_df.iterrows():
        tier_name = row["tier"]
        count = int(row["count"])
        tiers[tier_name] = {
            "count": count,
            "pct": round(count / total * 100, 1) if total else 0,
            "criteria": criteria_map.get(tier_name, ""),
        }

    return {
        "total_success": total,
        "tiers": tiers,
    }


async def _tool_tissues(args: dict) -> dict:
    """Get tissue distribution across processed samples."""
    import singlet

    top_n = args.get("top_n", 25)

    # Use bundled parquet data (properly normalized 37 categories)
    df = singlet.tissues()
    total_success = len(singlet.samples(status="SUCCESS"))
    total_with_tissue = int(df["count"].sum())

    top_tissues = df.head(top_n)

    return {
        "total_success_samples": total_success,
        "samples_with_tissue": total_with_tissue,
        "coverage_pct": round(total_with_tissue / total_success * 100, 1) if total_success else 0,
        "unique_tissues": len(df),
        "top_tissues": [
            {"tissue": t, "count": c} for t, c in zip(top_tissues["tissue"], top_tissues["count"])
        ],
    }


async def _tool_failures() -> dict:
    """Get failure category breakdown for non-SUCCESS samples."""
    import singlet

    df = singlet.samples()
    non_success = df[df["status"] != "SUCCESS"]
    total = len(non_success)

    # Count by failure_category
    cats = non_success["failure_category"].dropna().value_counts()
    sorted_cats = [(cat, int(count)) for cat, count in cats.items()]

    return {
        "total_failed": total,
        "categories": [
            {"category": cat, "count": c, "pct": round(c / total * 100, 1) if total else 0}
            for cat, c in sorted_cats
        ],
        "status_breakdown": {
            "HARD_FAIL": int((non_success["status"] == "HARD_FAIL").sum()),
            "SOFT_FAIL": int((non_success["status"] == "SOFT_FAIL").sum()),
        },
    }


async def _tool_cell_types(arguments: dict) -> dict:
    """Get cell type distribution from pre-computed annotations."""
    top_n = arguments.get("top_n", 20)

    # Use bundled parquet for fast, consistent results
    import singlet

    ct_df = singlet.cell_types()
    total_success = len(singlet.samples(status="SUCCESS"))
    total_annotated = int(ct_df["count"].sum())

    return {
        "total_success": total_success,
        "annotated_with_cell_type": total_annotated,
        "coverage_pct": round(total_annotated / total_success * 100, 1) if total_success else 0,
        "categories": len(ct_df),
        "cell_types": [
            {"cell_type": row["cell_type"], "count": int(row["count"])}
            for _, row in ct_df.head(top_n).iterrows()
        ],
    }


async def _tool_species() -> dict:
    """Get species list with sample counts."""
    import singlet

    species_list = singlet.species()
    # Get sample counts per species
    samples_df = singlet.samples(status="SUCCESS")
    species_counts = []
    for sp in species_list:
        count = len(samples_df[samples_df["organism"].str.contains(sp, na=False, regex=False)])
        cells = int(
            samples_df[samples_df["organism"].str.contains(sp, na=False, regex=False)][
                "cells_called"
            ].sum()
        )
        species_counts.append({"species": sp, "samples": count, "cells": cells})
    species_counts.sort(key=lambda x: x["samples"], reverse=True)

    return {
        "total_species": len(species_list),
        "species": species_counts,
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


async def main():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
