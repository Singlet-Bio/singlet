"""Tests for singlet._auth and singlet._query modules."""

from unittest.mock import patch

import pytest
import singlet._auth as auth

# ---------------------------------------------------------------------------
# _auth module
# ---------------------------------------------------------------------------


class TestLogin:
    def setup_method(self):
        """Reset global state."""
        auth._API_KEY = None

    def test_login_with_key(self):
        auth.login("sk-test-key-123")
        assert auth._API_KEY == "sk-test-key-123"

    def test_login_from_env(self, monkeypatch):
        monkeypatch.setenv("SINGLET_API_KEY", "sk-env-key-456")
        auth.login()
        assert auth._API_KEY == "sk-env-key-456"

    def test_login_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("SINGLET_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            auth.login()

    def test_get_key_after_login(self):
        auth.login("sk-hello")
        assert auth._get_key() == "sk-hello"

    def test_get_key_from_env(self, monkeypatch):
        monkeypatch.setenv("SINGLET_API_KEY", "sk-from-env")
        assert auth._get_key() == "sk-from-env"

    def test_get_key_raises_without_auth(self, monkeypatch):
        monkeypatch.delenv("SINGLET_API_KEY", raising=False)
        auth._API_KEY = None
        with pytest.raises(RuntimeError, match="requires authentication"):
            auth._get_key()

    def test_headers_format(self):
        auth.login("sk-test")
        headers = auth._headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert "User-Agent" in headers


# ---------------------------------------------------------------------------
# _query module (mocked network)
# ---------------------------------------------------------------------------


class TestQuery:
    def setup_method(self):
        auth._API_KEY = "sk-test-key"

    @patch("requests.post")
    def test_query_builds_params(self, mock_post):
        """Verify query() passes structured params to API."""
        from singlet._query import query

        mock_post.side_effect = Exception("Stop before network")

        with pytest.raises(Exception, match="Stop before network"):
            query(species="human", tissue="lung", min_cells=1000)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["species"] == "human"
        assert payload["tissue"] == "lung"
        assert payload["min_cells"] == "1000"

    @patch("requests.post")
    def test_query_joins_lists(self, mock_post):
        """List params are joined with commas."""
        from singlet._query import query

        mock_post.side_effect = Exception("Stop")

        with pytest.raises(Exception, match="Stop"):
            query(tissue=["lung", "liver", "brain"])

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["tissue"] == "lung,liver,brain"

    @patch("requests.post")
    def test_search_passes_text(self, mock_post):
        """Verify search() passes text query to API."""
        from singlet._query import search

        mock_post.side_effect = Exception("Stop")

        with pytest.raises(Exception, match="Stop"):
            search("exhausted T cells in lung cancer")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["query"] == "exhausted T cells in lung cancer"
        assert payload["max_results"] == 100_000

    def test_query_requires_auth(self, monkeypatch):
        """Without auth, query should raise RuntimeError."""
        auth._API_KEY = None
        monkeypatch.delenv("SINGLET_API_KEY", raising=False)

        from singlet._query import query

        with pytest.raises(RuntimeError, match="requires authentication"):
            query(species="human")

    def test_search_requires_auth(self, monkeypatch):
        """Without auth, search should raise RuntimeError."""
        auth._API_KEY = None
        monkeypatch.delenv("SINGLET_API_KEY", raising=False)

        from singlet._query import search

        with pytest.raises(RuntimeError, match="requires authentication"):
            search("test query")
