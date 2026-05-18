# SPDX-License-Identifier: MIT
"""Tests for singlet._query: query() and search() with mocked API."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    """Set fake API key to bypass auth guard."""
    monkeypatch.setenv("SINGLET_API_KEY", "sk-test-fake-key")


def _make_1pz_bytes():
    """Create a minimal valid .1pz file to return from mocked API."""
    import anndata as ad
    import pandas as pd
    from singlet._io import write_1pz

    mat = sp.random(3, 5, density=0.5, format="csr", dtype=np.float64)
    mat.data = np.round(mat.data * 10).astype(np.float64)
    adata = ad.AnnData(X=mat)
    adata.obs_names = pd.Index([f"C{i}" for i in range(3)])
    adata.var_names = pd.Index([f"G{j}" for j in range(5)])

    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mktemp(suffix=".1pz"))
    write_1pz(adata, tmp)
    data = tmp.read_bytes()
    tmp.unlink()
    return data


class TestQuery:
    """Test query() API call."""

    def test_query_builds_params(self):
        """Sends structured metadata as JSON."""
        from singlet._query import query

        pz_data = _make_1pz_bytes()
        mock_resp = MagicMock()
        mock_resp.content = pz_data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            adata = query(species="human", tissue="lung", min_cells=100)

        call_args = mock_post.call_args
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert json_body["species"] == "human"
        assert json_body["tissue"] == "lung"
        assert json_body["min_cells"] == "100"
        assert adata.shape == (3, 5)

    def test_query_list_params_joined(self):
        """List params are comma-joined."""
        from singlet._query import query

        pz_data = _make_1pz_bytes()
        mock_resp = MagicMock()
        mock_resp.content = pz_data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            query(species=["human", "mouse"], tissue=["lung", "liver"])

        json_body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert json_body["species"] == "human,mouse"
        assert json_body["tissue"] == "lung,liver"

    def test_query_omits_none_params(self):
        """None values are not included in params."""
        from singlet._query import query

        pz_data = _make_1pz_bytes()
        mock_resp = MagicMock()
        mock_resp.content = pz_data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            query(species="human")

        json_body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "tissue" not in json_body
        assert "disease" not in json_body


class TestSearch:
    """Test search() semantic query."""

    def test_search_sends_text(self):
        """Sends text query to search endpoint."""
        from singlet._query import search

        pz_data = _make_1pz_bytes()
        mock_resp = MagicMock()
        mock_resp.content = pz_data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            adata = search("exhausted T cells in lung cancer")

        call_args = mock_post.call_args
        assert "/search" in call_args[0][0]
        json_body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert json_body["query"] == "exhausted T cells in lung cancer"
        assert adata.shape == (3, 5)

    def test_search_max_results(self):
        """max_results is passed through."""
        from singlet._query import search

        pz_data = _make_1pz_bytes()
        mock_resp = MagicMock()
        mock_resp.content = pz_data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            search("neurons", max_results=500)

        json_body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert json_body["max_results"] == 500
