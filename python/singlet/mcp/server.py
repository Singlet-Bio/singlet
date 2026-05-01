#!/usr/bin/env python3
"""
Singlet MCP Server — Model Context Protocol server for the Singlet Atlas.

Exposes these tools to any MCP client (Claude Desktop, Cursor, VS Code Copilot):
  - singlet_search:  Search processed single-cell datasets
  - singlet_load:    Get download/access info for a sample
  - singlet_qc:     Get QC metrics for a sample
  - singlet_stats:  Get corpus-wide statistics
  - singlet_browse: Browse samples with filters

Usage:
    # Start the server (stdio transport):
    python -m singlet.mcp.server

    # Or use the entry point:
    singlet-mcp

    # Configure in Claude Desktop's claude_desktop_config.json:
    {
      "mcpServers": {
        "singlet": {
          "command": "python",
          "args": ["-m", "singlet.mcp.server"],
          "env": {
            "SUPABASE_URL": "https://vbswbitfyallghbgxkuw.supabase.co",
            "SUPABASE_ANON_KEY": "<your-anon-key>"
          }
        }
      }
    }

Requires: pip install mcp supabase
"""

import json
import os
import sys
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    sys.exit(
        "MCP SDK not installed. Run: pip install mcp\n"
        "See: https://github.com/modelcontextprotocol/python-sdk"
    )

try:
    from supabase import create_client, Client
except ImportError:
    sys.exit("Supabase client not installed. Run: pip install supabase")


# ─── Configuration ───────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://vbswbitfyallghbgxkuw.supabase.co"
)
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

if not SUPABASE_KEY:
    # Try service key as fallback
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

if not SUPABASE_KEY:
    print(
        "Warning: No SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY set. "
        "Tools will fail at runtime.",
        file=sys.stderr,
    )


def get_client() -> Client:
    """Create Supabase client (lazy, reused)."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


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
                "available in the atlas. Returns a page of samples with basic info."
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
                        "description": "Filter by organism",
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
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route tool calls to implementations."""
    try:
        if name == "singlet_stats":
            result = await _tool_stats()
        elif name == "singlet_search":
            result = await _tool_search(arguments)
        elif name == "singlet_qc":
            result = await _tool_qc(arguments)
        elif name == "singlet_load":
            result = await _tool_load(arguments)
        elif name == "singlet_browse":
            result = await _tool_browse(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ─── Tool Implementations ────────────────────────────────────────────────────


async def _tool_stats() -> dict:
    """Get corpus-wide statistics using the materialized view."""
    client = get_client()

    # Try pre-computed corpus_stats view (refreshed by ETL)
    try:
        resp = client.from_("corpus_stats").select("*").execute()
        if resp.data:
            row = resp.data[0]
            return {
                "total_samples": row.get("total_samples", 0),
                "successful_samples": row.get("success_samples", 0),
                "total_cells": row.get("total_cells", 0),
                "species_count": row.get("species_count", 0),
                "series_count": row.get("series_count", 0),
                "success_rate": row.get("success_rate", 0),
                "avg_mapping_rate": row.get("avg_mapping_rate", 0),
                "avg_median_genes": row.get("avg_median_genes", 0),
            }
    except Exception:
        pass  # Fall through to direct query

    # Fallback: query directly (paginate to avoid row limit)
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = client.table("samples").select(
            "status, cells_called, organism, gse_id, mapping_rate, median_genes"
        ).range(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    rows = all_rows
    success = [r for r in rows if r["status"] == "SUCCESS"]
    terminal = [r for r in rows if r["status"] in ("SUCCESS", "SOFT_FAIL", "HARD_FAIL")]

    def avg(vals):
        nums = [v for v in vals if v is not None]
        return round(sum(nums) / len(nums), 4) if nums else None

    return {
        "total_samples": len(rows),
        "successful_samples": len(success),
        "total_cells": sum(r.get("cells_called") or 0 for r in success),
        "species_count": len(set(r["organism"] for r in rows if r.get("organism"))),
        "series_count": len(set(r["gse_id"] for r in rows if r.get("gse_id"))),
        "success_rate": round(len(success) / len(terminal), 4) if terminal else None,
        "avg_mapping_rate": avg([r.get("mapping_rate") for r in success]),
        "avg_median_genes": avg([r.get("median_genes") for r in success]),
    }


async def _tool_search(args: dict) -> dict:
    """Search samples with filters."""
    client = get_client()
    limit = min(args.get("limit", 20), 100)

    query = client.table("samples").select(
        "gsm_id, gse_id, organism, protocol, modality, status, "
        "cells_called, mapping_rate, title, source"
    )

    if args.get("organism"):
        query = query.eq("organism", args["organism"])
    if args.get("protocol"):
        query = query.eq("protocol", args["protocol"])
    if args.get("modality"):
        query = query.eq("modality", args["modality"])
    if args.get("status"):
        query = query.eq("status", args["status"])
    if args.get("query"):
        q = args["query"]
        query = query.or_(
            f"gsm_id.ilike.%{q}%,gse_id.ilike.%{q}%,title.ilike.%{q}%,source.ilike.%{q}%"
        )

    query = query.order("pipeline_date", desc=True).limit(limit)
    resp = query.execute()

    return {
        "count": len(resp.data or []),
        "samples": resp.data or [],
    }


async def _tool_qc(args: dict) -> dict:
    """Get QC metrics for a sample."""
    gsm_id = args["gsm_id"]
    client = get_client()

    resp = client.table("samples").select("*").eq("gsm_id", gsm_id).execute()
    if not resp.data:
        return {"error": f"Sample {gsm_id} not found in atlas"}

    sample = resp.data[0]
    return {
        "gsm_id": sample["gsm_id"],
        "gse_id": sample["gse_id"],
        "organism": sample["organism"],
        "protocol": sample["protocol"],
        "modality": sample["modality"],
        "status": sample["status"],
        "qc_metrics": {
            "mapping_rate": sample.get("mapping_rate"),
            "cells_called": sample.get("cells_called"),
            "median_genes": sample.get("median_genes"),
            "median_umis": sample.get("median_umis"),
            "mt_pct": sample.get("mt_pct"),
            "doublet_rate": sample.get("doublet_rate"),
            "ambient_pct": sample.get("ambient_pct"),
            "saturation": sample.get("saturation"),
        },
        "processing": {
            "singlet_version": sample.get("singlet_version"),
            "wall_time_s": sample.get("wall_time_s"),
            "pipeline_date": sample.get("pipeline_date"),
        },
        "title": sample.get("title"),
        "source": sample.get("source"),
        "characteristics": sample.get("characteristics"),
        "web_url": f"https://singlet.bio/sample/{gsm_id}",
    }


async def _tool_load(args: dict) -> dict:
    """Get load/access info for a sample."""
    gsm_id = args["gsm_id"]
    client = get_client()

    resp = client.table("samples").select(
        "gsm_id, gse_id, status, pz_path, pz_size_bytes, cells_called"
    ).eq("gsm_id", gsm_id).execute()

    if not resp.data:
        return {"error": f"Sample {gsm_id} not found in atlas"}

    sample = resp.data[0]

    if sample["status"] != "SUCCESS":
        return {
            "gsm_id": gsm_id,
            "status": sample["status"],
            "error": f"Sample not successfully processed (status: {sample['status']})",
        }

    size_mb = round(sample["pz_size_bytes"] / 1e6, 1) if sample.get("pz_size_bytes") else None

    return {
        "gsm_id": gsm_id,
        "gse_id": sample["gse_id"],
        "cells": sample["cells_called"],
        "file_size_mb": size_mb,
        "format": ".1pz (SinglePress compressed sparse matrix)",
        "python_code": f'import singlet\nadata = singlet.load("{gsm_id}")\nprint(adata)',
        "r_code": f'library(singlepress)\nmat <- read_1pz(singlet_path("{gsm_id}"))',
        "web_url": f"https://singlet.bio/sample/{gsm_id}",
    }


async def _tool_browse(args: dict) -> dict:
    """Browse samples with pagination."""
    client = get_client()
    page = args.get("page", 0)
    page_size = min(args.get("page_size", 25), 100)
    sort_by = args.get("sort_by", "pipeline_date")

    query = client.table("samples").select(
        "gsm_id, gse_id, organism, protocol, status, cells_called, mapping_rate",
        count="exact",
    )

    if args.get("organism"):
        query = query.eq("organism", args["organism"])

    query = query.order(sort_by, desc=True)
    query = query.range(page * page_size, (page + 1) * page_size - 1)
    resp = query.execute()

    return {
        "page": page,
        "page_size": page_size,
        "total": resp.count or 0,
        "samples": resp.data or [],
    }


# ─── Entry point ─────────────────────────────────────────────────────────────


async def main():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
