#!/usr/bin/env python3
"""
Smoke test for the singlet MCP server tools.

Validates that all tools work correctly against Supabase.
Run after ETL has populated the database.

Usage:
    export SUPABASE_URL="https://vbswbitfyallghbgxkuw.supabase.co"
    export SUPABASE_ANON_KEY="<your-key>"
    python -m singlet.mcp.smoke_test
"""

import asyncio
import json
import os
import sys

# Ensure we can import from the local package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try importing from the server module; if MCP SDK is missing, mock it first
try:
    from singlet.mcp.server import (
        _tool_stats,
        _tool_search,
        _tool_qc,
        _tool_load,
        _tool_browse,
        _tool_protocols,
        _tool_quality,
        _tool_tissues,
        _tool_failures,
        _tool_cell_types,
        _tool_species,
    )
except SystemExit:
    # MCP SDK not installed — mock it so we can import tool functions
    from unittest.mock import MagicMock
    sys.modules["mcp"] = MagicMock()
    sys.modules["mcp.server"] = MagicMock()
    sys.modules["mcp.server.stdio"] = MagicMock()
    sys.modules["mcp.types"] = MagicMock()
    from singlet.mcp.server import (
        _tool_stats,
        _tool_search,
        _tool_qc,
        _tool_load,
        _tool_browse,
        _tool_protocols,
        _tool_quality,
        _tool_tissues,
        _tool_failures,
        _tool_cell_types,
        _tool_species,
    )


async def run_smoke_tests():
    print("=" * 60)
    print("Singlet MCP Server — Smoke Test")
    print("=" * 60)
    passed = 0
    failed = 0

    # Test 1: Stats
    print("\n[1/11] singlet_stats...")
    try:
        result = await _tool_stats()
        assert "total_samples" in result, "Missing total_samples"
        assert result["total_samples"] > 0, "No samples in database"
        print(f"  ✓ {result['total_samples']} samples, "
              f"{result['total_cells']:,} cells, "
              f"{result['species_count']} species")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 2: Search
    print("\n[2/11] singlet_search (organism=Homo sapiens, limit=5)...")
    try:
        result = await _tool_search({"organism": "Homo sapiens", "limit": 5})
        assert "samples" in result, "Missing samples key"
        print(f"  ✓ Found {result['count']} results")
        if result["samples"]:
            s = result["samples"][0]
            print(f"  ✓ First: {s['gsm_id']} ({s['protocol']}, {s.get('cells_called', '?')} cells)")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 3: QC
    print("\n[3/11] singlet_qc (first available sample)...")
    try:
        # Get a sample to query
        browse = await _tool_browse({"page": 0, "page_size": 1})
        if browse["samples"]:
            gsm_id = browse["samples"][0]["gsm_id"]
            result = await _tool_qc({"gsm_id": gsm_id})
            assert "qc_metrics" in result, "Missing qc_metrics"
            assert result["gsm_id"] == gsm_id
            print(f"  ✓ {gsm_id}: mapping_rate={result['qc_metrics']['mapping_rate']}, "
                  f"cells={result['qc_metrics']['cells_called']}")
            passed += 1
        else:
            print("  ⚠ No samples to test against")
            failed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 4: Load
    print("\n[4/11] singlet_load...")
    try:
        browse = await _tool_browse({"page": 0, "page_size": 1})
        if browse["samples"]:
            gsm_id = browse["samples"][0]["gsm_id"]
            result = await _tool_load({"gsm_id": gsm_id})
            assert "python_code" in result or "error" in result
            if "python_code" in result:
                print(f"  ✓ {gsm_id}: {result.get('file_size_mb', '?')} MB, "
                      f"{result.get('cells', '?')} cells")
            else:
                print(f"  ✓ {gsm_id}: {result.get('status', 'N/A')} (not loadable)")
            passed += 1
        else:
            print("  ⚠ No samples")
            failed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 5: Browse with pagination
    print("\n[5/11] singlet_browse (page 0, size 10)...")
    try:
        result = await _tool_browse({"page": 0, "page_size": 10})
        assert "total" in result, "Missing total"
        assert "samples" in result, "Missing samples"
        print(f"  ✓ Page 0: {len(result['samples'])} samples shown, "
              f"{result['total']} total in atlas")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 6: Protocols
    print("\n[6/11] singlet_protocols...")
    try:
        result = await _tool_protocols()
        assert "protocols" in result, "Missing protocols"
        print(f"  ✓ {len(result['protocols'])} protocols, "
              f"top: {result['protocols'][0]['protocol']} ({result['protocols'][0]['count']})")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 7: Quality tiers
    print("\n[7/11] singlet_quality...")
    try:
        result = await _tool_quality()
        assert "tiers" in result, "Missing tiers"
        print(f"  ✓ {len(result['tiers'])} tiers: " +
              ", ".join(f"{t['tier']}={t['count']}" for t in result['tiers']))
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 8: Tissues
    print("\n[8/11] singlet_tissues...")
    try:
        result = await _tool_tissues({})
        assert "tissues" in result, "Missing tissues"
        print(f"  ✓ {result.get('categories', '?')} tissue categories, "
              f"{result.get('coverage_pct', '?')}% coverage")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 9: Failures
    print("\n[9/11] singlet_failures...")
    try:
        result = await _tool_failures()
        assert "categories" in result or "failures" in result, "Missing data"
        key = "categories" if "categories" in result else "failures"
        print(f"  ✓ {len(result[key])} failure categories")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 10: Cell types
    print("\n[10/11] singlet_cell_types...")
    try:
        result = await _tool_cell_types({})
        assert "cell_types" in result, "Missing cell_types"
        print(f"  ✓ {result.get('categories', '?')} categories, "
              f"{result.get('coverage_pct', '?')}% coverage")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 11: Species
    print("\n[11/11] singlet_species...")
    try:
        result = await _tool_species()
        assert "species" in result, "Missing species"
        assert result["total_species"] > 0, "No species"
        print(f"  ✓ {result['total_species']} species: " +
              ", ".join(s['species'].split()[-1] for s in result['species'][:4]))
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n⚠ Some tests failed. Ensure:")
        print("  1. SUPABASE_URL and SUPABASE_ANON_KEY are set")
        print("  2. ETL has been run (scripts/etl/etl_sync.py)")
        print("  3. Database tables exist (check Supabase dashboard)")
        sys.exit(1)
    else:
        print("\n✓ All smoke tests passed! MCP server is ready.")
        print("\nTo start the server:")
        print("  python -m singlet.mcp.server")
        print("\nOr configure in Claude Desktop / VS Code:")
        print('  "command": "python", "args": ["-m", "singlet.mcp.server"]')


if __name__ == "__main__":
    asyncio.run(run_smoke_tests())
