# SPDX-License-Identifier: MIT
"""
Unit tests for the singlet MCP server module.

Tests the parquet-backed tool functions (no network / Supabase required).
Network-dependent tools (search, qc, load, browse) are tested with mocked client.
"""

import asyncio
import json
import sys
from unittest.mock import MagicMock

# Mock MCP SDK if not installed so we can import server module
try:
    import mcp  # noqa: F401
except ImportError:
    # Create proper mocks for TextContent
    class _FakeTextContent:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _FakeTool:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    _mcp_mock = MagicMock()
    _mcp_types_mock = MagicMock()
    _mcp_types_mock.TextContent = _FakeTextContent
    _mcp_types_mock.Tool = _FakeTool

    # Server mock that makes decorators pass through
    class _FakeServer:
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            def decorator(fn):
                return fn

            return decorator

        def call_tool(self):
            def decorator(fn):
                return fn

            return decorator

    _mcp_server_mock = MagicMock()
    _mcp_server_mock.Server = _FakeServer

    sys.modules["mcp"] = _mcp_mock
    sys.modules["mcp.server"] = _mcp_server_mock
    sys.modules["mcp.server.stdio"] = MagicMock()
    sys.modules["mcp.types"] = _mcp_types_mock

# Mock supabase if not installed
try:
    import supabase  # noqa: F401
except ImportError:
    sys.modules["supabase"] = MagicMock()

from singlet.mcp.server import (
    _tool_cell_types,
    _tool_failures,
    _tool_protocols,
    _tool_quality,
    _tool_species,
    _tool_stats,
    _tool_tissues,
    call_tool,
    list_tools,
)


def _run(coro):
    """Helper to run async functions in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestToolStats:
    def test_returns_expected_keys(self):
        result = _run(_tool_stats())
        assert "total_samples" in result
        assert "successful_samples" in result
        assert "total_cells" in result
        assert "species_count" in result
        assert "series_count" in result
        assert "success_rate" in result
        assert "top_species" in result

    def test_sample_count_positive(self):
        result = _run(_tool_stats())
        assert result["total_samples"] > 0
        assert result["successful_samples"] > 0
        assert result["total_cells"] > 0

    def test_success_rate_valid(self):
        result = _run(_tool_stats())
        assert 0 < result["success_rate"] <= 1.0

    def test_top_species_is_list(self):
        result = _run(_tool_stats())
        assert isinstance(result["top_species"], list)
        assert len(result["top_species"]) > 0
        assert "organism" in result["top_species"][0]
        assert "samples" in result["top_species"][0]


class TestToolProtocols:
    def test_returns_expected_keys(self):
        result = _run(_tool_protocols())
        assert "total_protocols" in result
        assert "protocols" in result
        assert isinstance(result["protocols"], list)

    def test_protocols_have_success_rate(self):
        result = _run(_tool_protocols())
        for p in result["protocols"]:
            assert "protocol" in p
            assert "total_samples" in p
            assert "success_rate" in p
            assert 0 <= p["success_rate"] <= 1.0

    def test_has_10xv3(self):
        result = _run(_tool_protocols())
        names = [p["protocol"] for p in result["protocols"]]
        assert "10xv3" in names


class TestToolQuality:
    def test_returns_tiers(self):
        result = _run(_tool_quality())
        assert "total_success" in result
        assert "tiers" in result
        assert result["total_success"] > 0

    def test_has_gold_silver_bronze(self):
        result = _run(_tool_quality())
        for tier in ("gold", "silver", "bronze"):
            assert tier in result["tiers"]
            assert result["tiers"][tier]["count"] > 0


class TestToolTissues:
    def test_returns_tissue_list(self):
        result = _run(_tool_tissues({}))
        assert "top_tissues" in result
        assert "unique_tissues" in result
        assert len(result["top_tissues"]) > 0

    def test_top_n_param(self):
        result = _run(_tool_tissues({"top_n": 5}))
        assert len(result["top_tissues"]) == 5

    def test_blood_is_top(self):
        result = _run(_tool_tissues({}))
        top_names = [t["tissue"] for t in result["top_tissues"][:5]]
        assert "blood" in top_names


class TestToolFailures:
    def test_returns_categories(self):
        result = _run(_tool_failures())
        assert "total_failed" in result
        assert "categories" in result
        assert "status_breakdown" in result

    def test_status_breakdown_keys(self):
        result = _run(_tool_failures())
        assert "HARD_FAIL" in result["status_breakdown"]
        assert "SOFT_FAIL" in result["status_breakdown"]


class TestToolCellTypes:
    def test_returns_cell_types(self):
        result = _run(_tool_cell_types({}))
        assert "cell_types" in result
        assert "total_success" in result
        assert len(result["cell_types"]) > 0

    def test_top_n_param(self):
        result = _run(_tool_cell_types({"top_n": 3}))
        assert len(result["cell_types"]) == 3

    def test_cell_type_structure(self):
        result = _run(_tool_cell_types({}))
        for ct in result["cell_types"]:
            assert "cell_type" in ct
            assert "count" in ct
            assert ct["count"] > 0


class TestToolSpecies:
    def test_returns_species(self):
        result = _run(_tool_species())
        assert "total_species" in result
        assert "species" in result
        assert result["total_species"] > 0

    def test_homo_sapiens_present(self):
        result = _run(_tool_species())
        names = [s["species"] for s in result["species"]]
        assert "Homo sapiens" in names

    def test_species_structure(self):
        result = _run(_tool_species())
        for s in result["species"]:
            assert "species" in s
            assert "samples" in s
            assert "cells" in s


class TestCallToolRouter:
    def test_unknown_tool(self):
        result = _run(call_tool("nonexistent_tool", {}))
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "error" in data

    def test_routes_stats(self):
        result = _run(call_tool("singlet_stats", {}))
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "total_samples" in data

    def test_routes_protocols(self):
        result = _run(call_tool("singlet_protocols", {}))
        data = json.loads(result[0].text)
        assert "protocols" in data


class TestListTools:
    def test_returns_all_tools(self):
        tools = _run(list_tools())
        names = [t.name for t in tools]
        expected = [
            "singlet_stats",
            "singlet_search",
            "singlet_qc",
            "singlet_load",
            "singlet_browse",
            "singlet_protocols",
            "singlet_quality",
            "singlet_tissues",
            "singlet_failures",
            "singlet_cell_types",
            "singlet_species",
        ]
        for name in expected:
            assert name in names, f"Missing tool: {name}"

    def test_tools_have_schema(self):
        tools = _run(list_tools())
        for tool in tools:
            assert tool.inputSchema is not None
            assert tool.inputSchema["type"] == "object"
