# SPDX-License-Identifier: MIT
"""Tests for the multi-modality accessors on :class:`singlet.SingletBundle`.

A ``.singlet`` bundle carries far more than gene counts: splice junctions,
PSI, mitochondrial heteroplasmy, donor demultiplexing, non-host abundance,
V(D)J usage and a stack of per-cell annotation tables. These tests build a
synthetic bundle with both the flat and the nested sidecar layouts and check
that discovery, reading and the raw-counts assembly all behave.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from singlet.bundle import MODALITIES, SingletBundle

pytest.importorskip("anndata")

GSM = "GSM0000001"

# Two genes, each with one exon feature and one intron feature.
GENES = [
    {"gene_id": "ENSG01", "gene_name": "AAA"},
    {"gene_id": "ENSG02", "gene_name": "BBB"},
]
EXON_FEATURES = ["ENSG01_AAA_chr1:1-2", "ENSG02_BBB_chr1:5-6"]
INTRON_FEATURES = ["ENSG01_AAA_chr1:2-5", "ENSG02_BBB_chr1:6-9"]
BARCODES = ["AAACCCA", "AAACCCT", "AAACCCG"]
CALLED = BARCODES[:2]

EXON_DENSE = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int32)  # features x cells
INTRON_DENSE = np.array([[0, 4, 0], [5, 6, 0]], dtype=np.int32)


def _write_1pz(path, dense, rownames, colnames):
    """Write a features x cells .1pz from a dense features x cells array."""
    import anndata as ad
    import pandas as pd
    from scipy.sparse import csr_matrix

    from singlet._io import write_1pz

    adata = ad.AnnData(
        X=csr_matrix(dense.T),  # write_1pz expects cells x features
        obs=pd.DataFrame(index=pd.Index(colnames)),
        var=pd.DataFrame(index=pd.Index(rownames)),
    )
    write_1pz(adata, path)


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    pytest.importorskip("singlet._pz", reason="native .1pz codec not built")
    work = tmp_path_factory.mktemp("bundle")
    exon = work / "exon.1pz"
    intron = work / "intron.1pz"
    _write_1pz(exon, EXON_DENSE, EXON_FEATURES, BARCODES)
    _write_1pz(intron, INTRON_DENSE, INTRON_FEATURES, BARCODES)

    out = work / "GSE000001.singlet"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"gsm_ids": [GSM], "gse_id": "GSE000001"}))
        zf.writestr("study_meta.json", json.dumps({"gse_id": "GSE000001"}))
        zf.writestr(
            "feature_vocab.json",
            json.dumps({"genes": GENES, "reference_build": "GRCh38-2024-A"}),
        )
        p = f"samples/{GSM}/"
        zf.write(exon, p + "exon_counts.1pz")
        zf.write(intron, p + "intron_counts.1pz")
        zf.writestr(p + "cell_calls.tsv", "barcode\tn_umi\n" + "".join(f"{b}\t10\n" for b in CALLED))
        zf.writestr(p + "summary.json", json.dumps({"sample_id": GSM, "status": "DONE"}))
        # Nested layout (newer bundles).
        zf.writestr(p + "donor/donor_assignments.tsv", "barcode\tdonor\nAAACCCA\tdonor0\n")
        zf.writestr(
            p + "nonhost/nonhost_em_abundance.tsv", "taxon\treads\nEscherichia coli\t42\n"
        )
        # Flat layout (older bundles) for a different modality.
        zf.writestr(p + "mt_variants.tsv", "pos\tref\talt\taf\n3243\tA\tG\t0.12\n")
    return out


@pytest.fixture
def bundle(bundle_path):
    with SingletBundle.open(bundle_path) as b:
        yield b


class TestDiscovery:
    def test_registry_members_are_unique_per_modality(self):
        for name, mod in MODALITIES.items():
            assert mod.members, f"{name} declares no archive members"
            assert len(set(mod.members)) == len(mod.members)
            assert mod.kind in {"matrix", "table", "json", "text"}

    def test_list_files_is_relative_to_the_sample(self, bundle):
        files = set(bundle.list_files(GSM))
        assert "exon_counts.1pz" in files
        assert "donor/donor_assignments.tsv" in files
        assert not any(f.startswith("samples/") for f in files)

    def test_modalities_reports_what_is_present(self, bundle):
        found = bundle.modalities(GSM)
        assert {"exon_counts", "intron_counts", "cell_calls", "summary"} <= set(found)
        assert {"donor_assignments", "nonhost_species", "mt_variants"} <= set(found)
        assert "junctions" not in found  # no sj_counts.1pz in this fixture
        assert all(isinstance(v, str) and v for v in found.values())

    def test_has_resolves_both_layouts(self, bundle):
        assert bundle.has("donor_assignments", GSM)  # nested donor/ path
        assert bundle.has("mt_variants", GSM)  # flat path
        assert not bundle.has("junctions", GSM)

    def test_has_without_gsm_scans_all_samples(self, bundle):
        assert bundle.has("exon_counts")
        assert not bundle.has("junctions")


class TestRead:
    def test_missing_modality_raises_with_a_useful_message(self, bundle):
        with pytest.raises(KeyError, match="junctions"):
            bundle.read("junctions", GSM)

    def test_json_modality(self, bundle):
        assert bundle.read("summary", GSM)["sample_id"] == GSM

    def test_table_modality(self, bundle):
        df = bundle.read("donor_assignments", GSM)
        assert list(df.columns) == ["barcode", "donor"]
        assert df.iloc[0]["donor"] == "donor0"

    def test_nonhost_and_mt_convenience_accessors(self, bundle):
        assert bundle.nonhost(GSM).iloc[0]["taxon"] == "Escherichia coli"
        assert bundle.read("mt_variants", GSM).iloc[0]["pos"] == 3243

    def test_matrix_modality_is_transposed_to_cells_by_features(self, bundle):
        adata = bundle.read("exon_counts", GSM)
        assert adata.shape == (len(BARCODES), len(EXON_FEATURES))
        assert list(adata.obs_names) == BARCODES
        assert list(adata.var_names) == EXON_FEATURES
        np.testing.assert_array_equal(adata.X.toarray(), EXON_DENSE.T)

    def test_literal_member_paths_work(self, bundle):
        assert bundle.read("donor/donor_assignments.tsv", GSM).shape == (1, 2)


class TestRawCounts:
    def test_gene_level_sums_features_onto_the_gene_axis(self, bundle):
        adata = bundle.raw_counts(GSM)
        assert list(adata.var_names) == ["ENSG01", "ENSG02"]
        assert list(adata.obs_names) == CALLED
        # ENSG01 = exon 1,2 + intron 0,4 ; ENSG02 = exon 3,0 + intron 5,6
        np.testing.assert_array_equal(adata.X.toarray(), np.array([[1, 8], [6, 6]]))

    def test_layers_partition_x(self, bundle):
        adata = bundle.raw_counts(GSM)
        assert set(adata.layers) == {"spliced", "unspliced"}
        total = adata.layers["spliced"].sum() + adata.layers["unspliced"].sum()
        assert adata.X.sum() == total

    def test_feature_level_keeps_the_native_axis(self, bundle):
        adata = bundle.raw_counts(GSM, gene_level=False)
        assert list(adata.var_names) == EXON_FEATURES + INTRON_FEATURES
        assert list(adata.var["feature_kind"]) == ["exon", "exon", "intron", "intron"]
        assert list(adata.var["gene_id"]) == ["ENSG01", "ENSG02", "ENSG01", "ENSG02"]
        assert adata.X.sum() == adata.layers["spliced"].sum() + adata.layers["unspliced"].sum()

    def test_feature_and_gene_level_agree_on_totals(self, bundle):
        gene = bundle.raw_counts(GSM)
        feat = bundle.raw_counts(GSM, gene_level=False)
        assert sorted(gene.obs_names) == sorted(feat.obs_names)
        assert gene.X.sum() == feat.X.sum()

    def test_cells_all_keeps_empty_droplets(self, bundle):
        called = bundle.raw_counts(GSM, gene_level=False)
        every = bundle.raw_counts(GSM, gene_level=False, cells="all")
        assert called.n_obs == len(CALLED)
        assert every.n_obs == len(BARCODES)
        assert every.X.sum() >= called.X.sum()

    def test_bad_cells_argument_is_rejected(self, bundle):
        with pytest.raises(ValueError, match="called"):
            bundle.raw_counts(GSM, cells="some")

    def test_reference_build_is_carried_through(self, bundle):
        assert bundle.raw_counts(GSM).uns["reference_build"] == "GRCh38-2024-A"
        assert (bundle.raw_counts(GSM).obs["gsm_id"] == GSM).all()


class TestToAnnData:
    def test_layers_are_attached(self, bundle):
        adata = bundle.to_anndata(verbose=False)
        assert set(adata.layers) == {"spliced", "unspliced"}
        assert adata.X.sum() == adata.layers["spliced"].sum() + adata.layers["unspliced"].sum()

    def test_layers_can_be_switched_off(self, bundle):
        adata = bundle.to_anndata(verbose=False, layers=False)
        assert not adata.layers
