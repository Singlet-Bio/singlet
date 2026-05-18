# SPDX-License-Identifier: MIT
"""Tests for singlet.enrichr() and singlet.enrichr_from_de()."""

import pandas as pd
import pytest
import scipy.sparse as sp
import singlet
from singlet._enrichr import enrichr, enrichr_from_de

# Mark tests that require network access
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _has_network():
    """Check if we can actually query Enrichr API."""
    import json
    import urllib.request

    try:
        genes_str = "TP53\nBRCA1"
        data = f"list={urllib.request.quote(genes_str)}&description=test"
        req = urllib.request.Request(
            "https://maayanlab.cloud/speedrichr/api/addList",
            data=data.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
        return "userListId" in result
    except Exception:
        return False


needs_network = pytest.mark.skipif(not _has_network(), reason="Enrichr API not reachable")


class TestEnrichr:
    def test_empty_raises(self):
        """Should raise ValueError for empty gene list."""
        with pytest.raises(ValueError, match="must not be empty"):
            enrichr([])

    def test_whitespace_only_raises(self):
        """Should raise ValueError for whitespace-only genes."""
        with pytest.raises(ValueError, match="no valid"):
            enrichr(["", "  ", "\t"])

    @needs_network
    def test_basic_query(self):
        """Should return DataFrame with expected columns."""
        genes = ["CD3D", "CD3E", "CD4", "IL7R", "LCK", "CD28", "PTPRC"]
        result = enrichr(genes, gene_sets="GO_Biological_Process_2023", top_n=5)
        assert isinstance(result, pd.DataFrame)
        assert "term" in result.columns
        assert "adjusted_p_value" in result.columns
        assert "genes" in result.columns
        assert len(result) <= 5

    @needs_network
    def test_returns_sorted(self):
        """Results should be sorted by adjusted_p_value."""
        genes = ["BRCA1", "BRCA2", "TP53", "ATM", "CHEK2", "RAD51"]
        result = enrichr(genes, top_n=10)
        if len(result) > 1:
            pvals = result["adjusted_p_value"].values
            assert all(pvals[i] <= pvals[i + 1] for i in range(len(pvals) - 1))

    @needs_network
    def test_kegg_library(self):
        """Should work with different gene set libraries."""
        genes = ["INS", "GCG", "SST", "PPY", "PDX1"]
        result = enrichr(genes, gene_sets="KEGG_2021_Human", top_n=3)
        assert isinstance(result, pd.DataFrame)

    def test_public_api(self):
        assert hasattr(singlet, "enrichr")
        assert callable(singlet.enrichr)
        assert hasattr(singlet, "enrichr_from_de")
        assert callable(singlet.enrichr_from_de)


class TestEnrichrFromDE:
    def _make_de_adata(self):
        """Create AnnData with rank_genes_groups results."""
        import anndata as ad

        adata = ad.AnnData(X=sp.random(100, 50, format="csr"))
        adata.var_names = [f"GENE{i}" for i in range(50)]
        adata.uns["rank_genes_groups"] = {
            "names": {
                "0": [f"GENE{i}" for i in range(20)],
                "1": [f"GENE{i}" for i in range(20, 40)],
            },
            "pvals_adj": {
                "0": [0.001] * 10 + [0.1] * 10,
                "1": [0.001] * 15 + [0.2] * 5,
            },
        }
        return adata

    def test_missing_de_raises(self):
        """Should raise KeyError when rank_genes_groups missing."""
        import anndata as ad

        adata = ad.AnnData(X=sp.random(50, 30, format="csr"))
        with pytest.raises(KeyError, match="rank_genes_groups"):
            enrichr_from_de(adata, "0")

    def test_missing_group_raises(self):
        """Should raise KeyError for invalid group."""
        adata = self._make_de_adata()
        with pytest.raises(KeyError, match="nonexistent"):
            enrichr_from_de(adata, "nonexistent")

    def test_pval_filtering(self):
        """Should filter genes by adjusted p-value cutoff."""
        adata = self._make_de_adata()
        # With strict cutoff, only 10 genes pass for group "0"
        # This tests the filtering logic (would fail at enrichr call without network)
        # Just test it doesn't crash with an empty list
        result = enrichr_from_de(adata, "0", pval_cutoff=0.0001)
        # All pvals are 0.001, none pass 0.0001
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
