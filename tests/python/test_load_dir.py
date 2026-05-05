"""Tests for singlet._loader.load_dir (singlify output loading)."""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def singlify_dir(tmp_path):
    """Create a fake singlify output directory with all expected files."""
    import anndata as ad
    from singlet._io import write_1pz

    n_cells, n_genes = 20, 10

    # Create count matrix
    mat = sp.random(n_cells, n_genes, density=0.4, format="csr", dtype=np.float32)
    mat.data = np.round(mat.data * 50).astype(np.float32)
    adata = ad.AnnData(X=mat)
    barcodes = [f"ACGT{i:04d}" for i in range(n_cells)]
    genes = [f"Gene{j}" for j in range(n_genes)]
    adata.obs_names = pd.Index(barcodes)
    adata.var_names = pd.Index(genes)

    write_1pz(adata, tmp_path / "gene_counts.1pz")

    # Gene expression TSV
    gene_df = pd.DataFrame(
        {"gene_id": [f"ENSG{j:05d}" for j in range(n_genes)], "gene_name": genes}
    )
    gene_df.to_csv(tmp_path / "gene_expression.tsv", sep="\t", index=False)

    # Barcodes
    pd.Series(barcodes).to_csv(tmp_path / "auto_barcodes.tsv", index=False, header=False)

    # QC metrics
    qc_df = pd.DataFrame(
        {
            "barcode": barcodes,
            "n_genes": np.random.randint(100, 5000, n_cells),
            "total_counts": np.random.randint(1000, 50000, n_cells),
        }
    )
    qc_df.to_csv(tmp_path / "cell_qc_metrics.tsv", sep="\t", index=False)

    # Doublet scores
    dub_df = pd.DataFrame({"barcode": barcodes, "doublet_score": np.random.rand(n_cells)})
    dub_df.to_csv(tmp_path / "doublet_scores.tsv", sep="\t", index=False)

    # Cell cycle
    cc_df = pd.DataFrame(
        {"barcode": barcodes, "phase": np.random.choice(["G1", "S", "G2M"], n_cells)}
    )
    cc_df.to_csv(tmp_path / "cell_cycle_scores.tsv", sep="\t", index=False)

    # Ancestry
    with open(tmp_path / "ancestry_call.json", "w") as f:
        json.dump({"ancestry": "European", "confidence": 0.95}, f)

    # Sex call
    with open(tmp_path / "sex_call.json", "w") as f:
        json.dump({"sex": "female", "xist_ratio": 0.89}, f)

    # Summary
    with open(tmp_path / "summary.json", "w") as f:
        json.dump({"mapping_rate": 0.87, "n_cells": n_cells}, f)

    # Saturation curve
    sat_df = pd.DataFrame({"reads": [1000, 5000, 10000], "genes_detected": [500, 2000, 4000]})
    sat_df.to_csv(tmp_path / "saturation_curve.tsv", sep="\t", index=False)

    return tmp_path


class TestLoadDir:
    def test_basic_load(self, singlify_dir):
        """Loads count matrix with correct shape."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert adata.shape == (20, 10)

    def test_gene_names(self, singlify_dir):
        """Gene names from gene_expression.tsv are attached."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "Gene0" in adata.var_names
        assert "gene_id" in adata.var.columns

    def test_barcodes(self, singlify_dir):
        """Barcodes from auto_barcodes.tsv are attached."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "ACGT0000" in adata.obs_names

    def test_qc_metrics(self, singlify_dir):
        """QC metrics merged into obs."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "n_genes" in adata.obs.columns
        assert "total_counts" in adata.obs.columns

    def test_doublet_scores(self, singlify_dir):
        """Doublet scores merged into obs."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "doublet_score" in adata.obs.columns

    def test_cell_cycle(self, singlify_dir):
        """Cell cycle phases merged into obs."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "phase" in adata.obs.columns

    def test_ancestry_uns(self, singlify_dir):
        """Ancestry JSON stored in uns."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "ancestry" in adata.uns
        assert adata.uns["ancestry"]["ancestry"] == "European"

    def test_sex_call_uns(self, singlify_dir):
        """Sex call JSON stored in uns."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "sex_call" in adata.uns

    def test_summary_uns(self, singlify_dir):
        """Summary JSON stored in uns."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "summary" in adata.uns
        assert adata.uns["summary"]["mapping_rate"] == 0.87

    def test_saturation_curve(self, singlify_dir):
        """Saturation curve stored as DataFrame in uns."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert "saturation_curve" in adata.uns
        assert len(adata.uns["saturation_curve"]) == 3

    def test_singlify_dir_in_uns(self, singlify_dir):
        """Source directory path stored in uns."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir)
        assert adata.uns["singlify_dir"] == str(singlify_dir)

    def test_without_qc(self, singlify_dir):
        """with_qc=False skips QC merge."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir, with_qc=False)
        assert "n_genes" not in adata.obs.columns

    def test_without_doublets(self, singlify_dir):
        """with_doublets=False skips doublet merge."""
        from singlet._loader import load_dir

        adata = load_dir(singlify_dir, with_doublets=False)
        assert "doublet_score" not in adata.obs.columns

    def test_not_a_directory_raises(self, tmp_path):
        """Raises FileNotFoundError if path is not a directory."""
        from singlet._loader import load_dir

        with pytest.raises(FileNotFoundError, match="Not a directory"):
            load_dir(tmp_path / "nonexistent")

    def test_missing_pz_raises(self, tmp_path):
        """Raises FileNotFoundError if gene_counts.1pz missing."""
        from singlet._loader import load_dir

        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Missing"):
            load_dir(tmp_path)
