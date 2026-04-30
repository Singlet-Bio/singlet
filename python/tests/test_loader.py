"""Tests for singlet._loader (load, load_dir)."""

import os
import pytest
import numpy as np
import pandas as pd

SAMPLE_DIR = '/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE125/GSE125416/GSM3573650'
HAS_SAMPLE = os.path.isdir(SAMPLE_DIR)


@pytest.mark.skipif(not HAS_SAMPLE, reason="Pipeline output not available")
class TestLoadDir:
    """Test load_dir against real pipeline output."""

    def test_basic_load(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR)
        assert adata.n_obs == 75420
        assert adata.n_vars == 38606

    def test_gene_names(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR, with_qc=False, with_doublets=False)
        assert 'TSPAN6' in adata.var_names
        assert 'gene_id' in adata.var.columns

    def test_barcodes(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR, with_qc=False, with_doublets=False)
        assert adata.obs_names[0] == 'GGATGTTTCCGCGCAA'

    def test_qc_metrics(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR)
        assert 'total_umis' in adata.obs.columns
        assert 'total_genes' in adata.obs.columns
        assert 'mt_pct' in adata.obs.columns
        assert adata.obs['total_umis'].median() > 0

    def test_doublet_scores(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR)
        assert 'is_doublet' in adata.obs.columns
        assert 'doublet_score' in adata.obs.columns
        assert adata.obs['is_doublet'].sum() > 0

    def test_no_qc(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR, with_qc=False, with_doublets=False)
        assert 'total_umis' not in adata.obs.columns
        assert 'is_doublet' not in adata.obs.columns

    def test_exon_layer(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR, layer='exon_counts', with_qc=False, with_doublets=False)
        assert adata.n_obs > 0

    def test_uns_metadata(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR, with_qc=False, with_doublets=False)
        assert adata.uns['singlify_dir'] == SAMPLE_DIR

    def test_missing_dir(self):
        from singlet._loader import load_dir

        with pytest.raises(FileNotFoundError):
            load_dir('/nonexistent/path')

    def test_missing_pz(self, tmp_path):
        from singlet._loader import load_dir

        with pytest.raises(FileNotFoundError, match="Missing"):
            load_dir(tmp_path)

    def test_cell_cycle(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR)
        assert 'phase' in adata.obs.columns
        assert 's_score' in adata.obs.columns
        assert 'g2m_score' in adata.obs.columns
        assert set(adata.obs['phase'].unique()) <= {'G1', 'S', 'G2M'}

    def test_ancestry(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR)
        assert 'ancestry' in adata.uns
        assert 'ancestry' in adata.uns['ancestry']
        assert 'confidence' in adata.uns['ancestry']

    def test_sex_call(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR)
        assert 'sex_call' in adata.uns
        assert 'sex' in adata.uns['sex_call']

    def test_summary(self):
        from singlet._loader import load_dir

        adata = load_dir(SAMPLE_DIR)
        assert 'summary' in adata.uns
        assert 'mapping_rate' in adata.uns['summary']
        assert 'protocol' in adata.uns['summary']
