#!/usr/bin/env python3
"""
Singlet MCP Server — Model Context Protocol server for the Singlet Atlas.

Exposes these tools to any MCP client (Claude Desktop, Cursor, VS Code Copilot):
  - singlet_stats:      Get corpus-wide statistics (samples, cells, species)
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
    """Get corpus-wide statistics using the materialized view."""
    client = get_client()

    # Try pre-computed corpus_stats view (refreshed by ETL)
    try:
        resp = client.from_("corpus_stats").select("*").execute()
        if resp.data:
            row = resp.data[0]
            stats = {
                "total_samples": row.get("total_samples", 0),
                "successful_samples": row.get("success_samples", 0),
                "total_cells": row.get("total_cells", 0),
                "species_count": row.get("species_count", 0),
                "series_count": row.get("series_count", 0),
                "success_rate": row.get("success_rate", 0),
                "avg_mapping_rate": row.get("avg_mapping_rate", 0),
                "avg_median_genes": row.get("avg_median_genes", 0),
            }

            # Add species breakdown
            try:
                sp = client.from_("species_stats").select("*").order(
                    "sample_count", desc=True
                ).limit(10).execute()
                if sp.data:
                    stats["top_species"] = [
                        {"organism": r["organism"], "samples": r["sample_count"],
                         "cells": r["total_cells"]}
                        for r in sp.data
                    ]
            except Exception:
                pass

            return stats
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
        "gsm_id, gse_id, organism, protocol, status, cells_called, mapping_rate, source",
        count="exact",
    )

    if args.get("organism"):
        query = query.eq("organism", args["organism"])
    if args.get("status"):
        query = query.eq("status", args["status"])
    if args.get("tissue"):
        query = query.ilike("source", f"%{args['tissue']}%")

    query = query.order(sort_by, desc=True)
    query = query.range(page * page_size, (page + 1) * page_size - 1)
    resp = query.execute()

    return {
        "page": page,
        "page_size": page_size,
        "total": resp.count or 0,
        "samples": resp.data or [],
    }


async def _tool_protocols() -> dict:
    """Get protocol distribution across the atlas."""
    client = get_client()

    # Query all samples with protocol info
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = client.table("samples").select(
            "protocol, status"
        ).range(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    # Aggregate
    from collections import Counter
    total_counts = Counter()
    success_counts = Counter()
    for row in all_rows:
        p = row.get("protocol") or "unknown"
        if not p.strip():
            p = "unknown"
        total_counts[p] += 1
        if row["status"] == "SUCCESS":
            success_counts[p] += 1

    protocols = []
    for protocol, total in total_counts.most_common():
        success = success_counts.get(protocol, 0)
        protocols.append({
            "protocol": protocol,
            "total_samples": total,
            "success_samples": success,
            "success_rate": round(success / total, 3) if total else 0,
        })

    return {
        "total_protocols": len([p for p in protocols if p["protocol"] != "unknown"]),
        "protocols": protocols,
    }


async def _tool_quality() -> dict:
    """Get quality tier breakdown of SUCCESS samples."""
    client = get_client()

    # Query SUCCESS samples with quality metrics
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = client.table("samples").select(
            "mapping_rate, median_genes, cells_called"
        ).eq("status", "SUCCESS").range(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    total = len(all_rows)
    gold = silver = bronze = 0
    for row in all_rows:
        mr = row.get("mapping_rate") or 0
        mg = row.get("median_genes") or 0
        cc = row.get("cells_called") or 0
        if mr >= 0.7 and mg >= 500 and cc >= 500:
            gold += 1
        elif mr >= 0.5 and mg >= 200 and cc >= 100:
            silver += 1
        else:
            bronze += 1

    return {
        "total_success": total,
        "tiers": {
            "gold": {"count": gold, "pct": round(gold / total * 100, 1) if total else 0,
                     "criteria": "mapping_rate>=0.7, median_genes>=500, cells>=500"},
            "silver": {"count": silver, "pct": round(silver / total * 100, 1) if total else 0,
                       "criteria": "mapping_rate>=0.5, median_genes>=200, cells>=100"},
            "bronze": {"count": bronze, "pct": round(bronze / total * 100, 1) if total else 0,
                       "criteria": "all other SUCCESS samples"},
        },
    }


async def _tool_tissues(args: dict) -> dict:
    """Get tissue distribution across processed samples."""
    client = get_client()
    top_n = args.get("top_n", 25)

    # Query SUCCESS samples with characteristics
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = client.table("samples").select(
            "characteristics"
        ).eq("status", "SUCCESS").not_.is_("characteristics", "null").range(
            offset, offset + page_size - 1
        ).execute()
        batch = resp.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    # Extract and normalize tissue
    tissue_counts: dict[str, int] = {}
    for row in all_rows:
        chars = row.get("characteristics")
        if isinstance(chars, dict) and "tissue" in chars:
            raw = chars["tissue"].strip().lower()
            # Basic normalization
            norm = _normalize_tissue(raw)
            tissue_counts[norm] = tissue_counts.get(norm, 0) + 1

    total_with_tissue = sum(tissue_counts.values())
    sorted_tissues = sorted(tissue_counts.items(), key=lambda x: -x[1])[:top_n]

    return {
        "total_success_samples": len(all_rows),
        "samples_with_tissue": total_with_tissue,
        "coverage_pct": round(total_with_tissue / len(all_rows) * 100, 1) if all_rows else 0,
        "unique_tissues": len(tissue_counts),
        "top_tissues": [{"tissue": t, "count": c} for t, c in sorted_tissues],
    }


def _normalize_tissue(raw: str) -> str:
    """Normalize a raw tissue string to a canonical category."""
    NORMALIZE = {
        "pbmc": "pbmc", "pbmcs": "pbmc", "peripheral blood": "blood",
        "whole blood": "blood", "blood": "blood",
        "brain": "brain", "dorsolateral prefrontal cortex": "brain",
        "leptomeninges": "brain", "hippocampus": "brain", "amygdala": "brain",
        "bone marrow": "bone marrow", "lung": "lung",
        "liver": "liver", "fetal liver": "liver",
        "pancreas": "pancreas", "spleen": "spleen", "kidney": "kidney",
        "breast": "breast", "skin": "skin", "tumor": "tumor",
        "thyroid": "thyroid", "retina": "retina", "tonsil": "tonsil",
        "esophagus": "esophagus", "villi": "placenta", "csf": "csf",
        "hpsc derived cardiomyocytes": "heart",
    }
    if raw in NORMALIZE:
        return NORMALIZE[raw]
    if "tumor" in raw or "tumour" in raw or "cancer" in raw:
        return "tumor"
    if "blood" in raw:
        return "blood"
    if "brain" in raw or "cortex" in raw:
        return "brain"
    if "liver" in raw:
        return "liver"
    if "lung" in raw:
        return "lung"
    if "kidney" in raw or "renal" in raw:
        return "kidney"
    if "skin" in raw or "dermis" in raw:
        return "skin"
    if "bone marrow" in raw:
        return "bone marrow"
    if "pbmc" in raw:
        return "pbmc"
    if "heart" in raw or "cardiac" in raw:
        return "heart"
    if "intestin" in raw or "colon" in raw or "gut" in raw:
        return "intestine"
    if "lymph" in raw:
        return "lymph node"
    if "muscle" in raw:
        return "muscle"
    if "spleen" in raw:
        return "spleen"
    if "pancrea" in raw:
        return "pancreas"
    return raw


async def _tool_failures() -> dict:
    """Get failure category breakdown for non-SUCCESS samples."""
    client = get_client()

    # Query non-SUCCESS samples
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        resp = client.table("samples").select(
            "status, mapping_rate, cells_called, failure_category"
        ).neq("status", "SUCCESS").range(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    total = len(all_rows)
    categories: dict[str, int] = {}
    for row in all_rows:
        cat = row.get("failure_category")
        if cat:
            categories[cat] = categories.get(cat, 0) + 1
        else:
            # Infer from metrics
            mr = row.get("mapping_rate") or 0
            cells = row.get("cells_called") or 0
            status = row.get("status", "")
            if mr == 0 and cells == 0:
                key = "download_fail" if status == "HARD_FAIL" else "pipeline_crash"
            elif mr > 0 and mr < 0.1:
                key = "align_low_map"
            elif mr >= 0.1 and cells < 50:
                key = "cells_below_threshold"
            else:
                key = "align_low_map"
            categories[key] = categories.get(key, 0) + 1

    sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
    return {
        "total_failed": total,
        "categories": [
            {"category": cat, "count": c, "pct": round(c / total * 100, 1) if total else 0}
            for cat, c in sorted_cats
        ],
        "status_breakdown": {
            "HARD_FAIL": sum(1 for r in all_rows if r["status"] == "HARD_FAIL"),
            "SOFT_FAIL": sum(1 for r in all_rows if r["status"] == "SOFT_FAIL"),
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
        count = len(samples_df[samples_df["organism"].str.contains(sp, na=False)])
        cells = int(samples_df[samples_df["organism"].str.contains(sp, na=False)]["cells_called"].sum())
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
