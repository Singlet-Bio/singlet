"""Tests for singlet.nmf (NMF model serving)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Set a fake API key so auth doesn't block tests."""
    monkeypatch.setenv("SINGLET_API_KEY", "sk-test-fake-key")


class TestNmfTransform:
    @patch("requests.post")
    def test_transform_success(self, mock_post):
        """transform adds obsm['X_nmf'] on success."""
        from singlet.nmf import transform

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"loadings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        adata = MagicMock()
        adata.var_names = ["GENE1", "GENE2", "GENE3"]
        adata.n_obs = 3
        adata.obsm = {}

        transform(adata, model="test_model")

        assert "X_nmf" in adata.obsm
        assert adata.obsm["X_nmf"].shape == (3, 2)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "test_model" in str(call_kwargs)

    @patch("requests.post")
    def test_transform_raises_on_http_error(self, mock_post):
        """transform raises if API returns error."""
        from singlet.nmf import transform

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_post.return_value = mock_resp

        adata = MagicMock()
        adata.var_names = ["G1"]
        adata.n_obs = 1

        with pytest.raises(Exception, match="401"):
            transform(adata)


class TestNmfAnnotate:
    @patch("requests.get")
    def test_annotate_success(self, mock_get):
        """annotate returns factor → annotation dict."""
        from singlet.nmf import annotate

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "annotations": {"0": "T cell activation", "1": "mitochondrial metabolism"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        adata = MagicMock()
        result = annotate(adata, model="human_global_k100")

        assert result["0"] == "T cell activation"
        assert result["1"] == "mitochondrial metabolism"
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_annotate_raises_on_http_error(self, mock_get):
        """annotate raises if API returns error."""
        from singlet.nmf import annotate

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_resp

        adata = MagicMock()
        with pytest.raises(Exception, match="404"):
            annotate(adata)
