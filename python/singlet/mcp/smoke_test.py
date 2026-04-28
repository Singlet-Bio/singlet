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

from singlet.mcp.server import (
    _tool_stats,
    _tool_search,
    _tool_qc,
    _tool_load,
    _tool_browse,
)


async def run_smoke_tests():
    print("=" * 60)
    print("Singlet MCP Server — Smoke Test")
    print("=" * 60)
    passed = 0
    failed = 0

    # Test 1: Stats
    print("\n[1/5] singlet_stats...")
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
    print("\n[2/5] singlet_search (organism=Homo sapiens, limit=5)...")
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
    print("\n[3/5] singlet_qc (first available sample)...")
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
    print("\n[4/5] singlet_load...")
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
    print("\n[5/5] singlet_browse (page 0, size 10)...")
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
