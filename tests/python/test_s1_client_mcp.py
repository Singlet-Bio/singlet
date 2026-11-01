# SPDX-License-Identifier: MIT
"""The local MCP server on the .s1 format: compact JSON, per-sample object URL / member ranges / pack ids in singlet_load, the safe filter
parser, and singlet_slice_cells, against the synthetic study of _s1_synth (no network, no mcp package: it is faked like in
test_mcp_server.py)."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from unittest.mock import MagicMock

import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
pytest.importorskip("zstandard")

try:
    import mcp  # noqa: F401
except ImportError:

    class _TextContent:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Tool:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Server:
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            return lambda fn: fn

    types = MagicMock()
    types.TextContent, types.Tool = _TextContent, _Tool
    srv = MagicMock()
    srv.Server = _Server
    sys.modules.setdefault("mcp", MagicMock())
    sys.modules.setdefault("mcp.server", srv)
    sys.modules.setdefault("mcp.server.stdio", MagicMock())
    sys.modules.setdefault("mcp.types", types)

import _s1_synth as synth  # noqa: E402
from singlet.mcp import server  # noqa: E402
from singlet.s1 import client  # noqa: E402

GSE, G1 = synth.GSE, "GSM9000001"


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("mcpstudy")
    try:
        synth.Synth(str(d)).build()
    except IndexError:  # pragma: no cover
        pytest.skip("vocabulary packs are not available")
    return d


@pytest.fixture()
def env(root, tmp_path, monkeypatch):
    monkeypatch.setenv("SINGLET_DATA_BASE", str(root))
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    client.clear_memory_cache()
    monkeypatch.setattr(
        server,
        "_lookup_gsm",
        lambda g: {
            "gsm_id": g,
            "gse_id": GSE,
            "organism": "Homo sapiens",
            "protocol": "10xv3",
            "status": "SUCCESS",
            "title": "t",
            "n_cells": 40,
        },
    )
    yield


def test_results_are_compact_json():
    res = _run(server.call_tool("nonexistent_tool", {}))
    text = res[0].text
    assert json.loads(text) == {"error": "Unknown tool: nonexistent_tool"}
    assert text == json.dumps(
        json.loads(text), separators=(",", ":")
    )  # no indentation, no spaces after separators


def test_singlet_load_gives_object_ranges_and_packs(env, root):
    out = _run(server._tool_load({"gsm_id": G1}))
    info = out["s1"]
    assert (
        out["load"] == f'singlet.load("{G1}")'
        and "bundle_url" not in out
        and "python_code" not in out
    )
    assert info["format"] == "singlet-s1" and len(info["rev"]) == 16
    assert info["object"].endswith(".s1") and "/s1/s/%s/" % G1 in info["object"].replace("\\", "/")
    assert set(info["members"]) >= {"cells", "exon_counts.1pz", "intron_counts.1pz", "summary.json"}
    assert set(info["packs"]) == {"exon", "intron"} and info["n_called"] == 40
    g = info["gene_level"]
    assert g["requests"] == 1 and g["ranges"][0][0] == info["members"]["cells"][0]
    assert g["ranges"][0][1] + 1 - g["ranges"][0][0] == g["bytes"] and g["bytes"] < info["bytes"]
    # the advertised range really is what the client reads
    st = client.open_study(GSE)
    sm = st.samples[G1]
    assert (
        g["bytes"]
        <= sum(sm.members[n].len for n in ("cells", "exon_counts.1pz", "intron_counts.1pz")) + 128
    )
    again = _run(server._tool_load({"gsm_id": G1, "if_rev": info["rev"]}))
    assert again == {"gsm_id": G1, "gse_id": GSE, "unchanged": True, "rev": info["rev"]}
    assert len(json.dumps(again)) < 100  # the "unchanged" answer is tiny


def test_singlet_load_falls_back_to_the_legacy_url_when_not_migrated(env, monkeypatch):
    monkeypatch.setattr(
        server, "_lookup_gsm", lambda g: {"gsm_id": g, "gse_id": "GSE1", "organism": "Homo sapiens"}
    )
    out = _run(server._tool_load({"gsm_id": "GSM1"}))
    assert (
        "s1" not in out
        and out["bundle_url"].endswith("/data/GSE1/GSE1.singlet")
        and "legacy=True" in out["load"]
    )


def test_unknown_sample_in_a_migrated_study(env, monkeypatch):
    monkeypatch.setattr(server, "_lookup_gsm", lambda g: {"gsm_id": g, "gse_id": GSE})
    out = _run(server._tool_load({"gsm_id": "GSM0000000"}))
    assert "is not a sample of" in out["s1"]["error"]


def test_where_parser_is_not_an_evaluator():
    import pandas as pd

    df = pd.DataFrame({"n_umi_exon": [10, 500, 9000], "called": [False, True, True]})
    m = server._parse_where("n_umi_exon >= 500 and called")
    assert m(df).tolist() == [False, True, True]
    assert server._parse_where("not called")(df).tolist() == [True, False, False]
    assert server._parse_where("n_umi_exon < 100")(df).tolist() == [True, False, False]
    assert server._parse_where("") is None
    for bad in (
        "__import__('os').system('x')",
        "n_umi_exon >= 5 or called",
        "n_umi_exon >=",
        "a.b > 3",
        "(called)",
    ):
        with pytest.raises(ValueError):
            server._parse_where(bad)
    with pytest.raises(ValueError, match="unknown cell-table column"):
        server._parse_where("nope > 1")(df)


def test_slice_cells_summary_and_save(env, root, tmp_path):
    out = _run(
        server._tool_slice_cells(
            {
                "gse_id": GSE,
                "where": "n_umi_exon >= 1 and called",
                "samples": [G1],
                "max_cells": 1000,
            }
        )
    )
    assert (
        out["n_cells"] == 40 and out["samples"] == {G1: 40} and out["requests"] == 0
    )  # a local directory is memory mapped: no requests
    assert out["median_n_umi_exon"] > 0 and out["load"].startswith("singlet.s1.cells(")
    small = _run(
        server._tool_slice_cells(
            {
                "gse_id": GSE,
                "where": "called",
                "samples": [G1],
                "max_cells": 7,
                "save_to": str(tmp_path / "s.h5ad"),
            }
        )
    )
    assert (
        small["n_cells"] == 7
        and (tmp_path / "s.h5ad").exists()
        and small["saved"].endswith("s.h5ad")
    )
    assert "error" in _run(server._tool_slice_cells({"gse_id": GSE, "where": "n_umi_exon >>> 3"}))
    assert "error" in _run(
        server._tool_slice_cells({"gse_id": GSE, "save_to": str(tmp_path / "x.txt")})
    )
    assert "error" in _run(server._tool_slice_cells({}))


def test_slice_cells_over_http_counts_requests(root, tmp_path, monkeypatch):
    import _s1_range_server as rs

    with rs.serve(str(root)) as (base, rec):
        monkeypatch.setenv("SINGLET_DATA_BASE", base)
        monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "c2"))
        client.clear_memory_cache()
        out = _run(
            server._tool_slice_cells(
                {"gse_id": GSE, "where": "called", "samples": [G1], "max_cells": 10}
            )
        )
        assert out["n_cells"] == 10 and out["requests"] >= 1 and out["bytes"] > 0
        assert out["requests"] == len([r for r in rec.log if "/s1/s/" in r[1]])


def test_tool_list_has_the_new_tool():
    tools = _run(server.list_tools())
    names = [t.name for t in tools]
    assert "singlet_slice_cells" in names and "singlet_load" in names
    sl = next(t for t in tools if t.name == "singlet_slice_cells")
    assert sl.inputSchema["type"] == "object" and "gse_id" in sl.inputSchema["properties"]
    assert "if_rev" in next(t for t in tools if t.name == "singlet_load").inputSchema["properties"]
