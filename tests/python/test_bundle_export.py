# SPDX-License-Identifier: MIT
"""Regression tests: a real-shaped bundle must export to .h5ad / .zarr.

Every published bundle's ``manifest.json`` has ``checksums`` keyed by archive
path (``samples/<GSM>/exon_counts.1pz``). Copying that dict into ``adata.uns``
made ``write_h5ad``/``write_zarr`` fail on every real study, because both
formats turn dict keys into group names. These tests build a synthetic bundle
with that manifest shape — plus a hollow (0x0) sample, a ``cb`` barcode column
and a text ``is_cell`` column — and round-trip it.
"""

from __future__ import annotations

import json
import struct
import zipfile

import numpy as np
import pytest
from singlet.bundle import SingletBundle

ad = pytest.importorskip("anndata")

GSE = "GSE000002"
GOOD, HOLLOW = "GSM0000021", "GSM0000022"

GENES = [
    {"gene_id": "ENSG01", "gene_name": "AAA"},
    {"gene_id": "ENSG02", "gene_name": "BBB"},
    {"gene_id": "ENSG03", "gene_name": "AAA"},  # duplicated symbol
]
EXON_FEATURES = ["ENSG01_AAA_chr1:1-2", "ENSG02_BBB_chr1:5-6", "ENSG03_AAA_chr2:1-2"]
INTRON_FEATURES = ["ENSG01_AAA_chr1:2-5", "ENSG02_BBB_chr1:6-9", "ENSG03_AAA_chr2:2-5"]
BARCODES = ["AAACCCA", "AAACCCT", "AAACCCG"]
EXON_DENSE = np.array([[1, 2, 0], [3, 0, 1], [0, 5, 0]], dtype=np.int32)  # features x cells
INTRON_DENSE = np.array([[0, 4, 0], [5, 6, 0], [1, 0, 0]], dtype=np.int32)
# is_cell written as text, and a "cb" barcode column: AAACCCA and AAACCCT are cells.
CELL_CALLS = "cb\tis_cell\tn_umi\nAAACCCA\tTrue\t10\nAAACCCT\tTrue\t9\nAAACCCG\tFalse\t1\n"


def _stub_1pz() -> bytes:
    """The 0x0 .1pz stub the pipeline writes for an output it failed to make."""
    header = struct.pack("<IHBBIIQ", 0x5A315054, 1, 0, 0, 0, 0, 0)
    return header + b"\0" * (96 - len(header) + 16)


def _write_1pz(path, dense, rownames, colnames):
    """Write a features x cells .1pz from a dense features x cells array."""
    import pandas as pd
    from scipy.sparse import csr_matrix

    from singlet._io import write_1pz

    adata = ad.AnnData(
        X=csr_matrix(dense.T),  # write_1pz expects cells x features
        obs=pd.DataFrame(index=pd.Index(colnames)),
        var=pd.DataFrame(index=pd.Index(rownames)),
    )
    write_1pz(adata, path)


def _write_bundle(path, members: dict, gsm_ids):
    manifest = {
        "schema_version": "1.0",
        "gse_id": GSE,
        "gsm_ids": list(gsm_ids),
        "n_gsms": len(gsm_ids),
        "singlet_version": "2.0.0",
        "included_files": {g: ["exon_counts.1pz", "cell_calls.tsv"] for g in gsm_ids},
        "excluded_samples": [],
        "checksums": {name: "0" * 64 for name in members},
    }
    study_meta = {
        "schema_version": "1.0",
        "gse_id": GSE,
        "series_title": "",
        "publications": [{"pmid": "1", "title": "A paper"}],
        "gsm_meta": {
            g: {"gsm_id": g, "organism": "Homo sapiens", "n_cells": 2, "mapping_rate": None}
            for g in gsm_ids
        },
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("study_meta.json", json.dumps(study_meta))
        zf.writestr("feature_vocab.json", json.dumps({"genes": GENES, "reference_build": "GRCh38"}))
        for name, data in members.items():
            zf.writestr(name, data)
    return path


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    pytest.importorskip("singlet._pz", reason="native .1pz codec not built")
    work = tmp_path_factory.mktemp("export")
    exon, intron = work / "exon.1pz", work / "intron.1pz"
    _write_1pz(exon, EXON_DENSE, EXON_FEATURES, BARCODES)
    _write_1pz(intron, INTRON_DENSE, INTRON_FEATURES, BARCODES)
    members = {
        f"samples/{GOOD}/exon_counts.1pz": exon.read_bytes(),
        f"samples/{GOOD}/intron_counts.1pz": intron.read_bytes(),
        f"samples/{GOOD}/cell_calls.tsv": CELL_CALLS,
        f"samples/{HOLLOW}/exon_counts.1pz": _stub_1pz(),
        f"samples/{HOLLOW}/intron_counts.1pz": _stub_1pz(),
        f"samples/{HOLLOW}/cell_calls.tsv": CELL_CALLS,
    }
    return _write_bundle(work / f"{GSE}.singlet", members, [GOOD, HOLLOW])


@pytest.fixture(scope="module")
def adata(bundle_path):
    with pytest.warns(UserWarning, match=f"{HOLLOW} skipped"):
        return SingletBundle.open(bundle_path).to_anndata(verbose=False)


class TestAssembly:
    def test_hollow_sample_is_skipped_and_recorded(self, adata):
        assert set(adata.obs["gsm_id"]) == {GOOD}
        skipped = adata.uns["skipped_samples"]
        assert list(skipped["gsm_id"]) == [HOLLOW]
        assert "0x0" in skipped["reason"].iloc[0]

    def test_is_cell_text_and_cb_column_select_called_cells(self, adata):
        assert list(adata.obs_names) == [f"{GOOD}_AAACCCA", f"{GOOD}_AAACCCT"]
        # gene = exon + intron for the two called barcodes (columns 0 and 1).
        expected = (EXON_DENSE + INTRON_DENSE)[:, :2].T
        np.testing.assert_array_equal(adata.X.toarray(), expected)
        assert adata.X.sum() == adata.layers["spliced"].sum() + adata.layers["unspliced"].sum()

    def test_checksums_are_stored_as_columns(self, adata):
        checksums = adata.uns["manifest"]["checksums"]
        assert f"samples/{GOOD}/exon_counts.1pz" in list(checksums["path"])
        assert len(checksums["path"]) == len(checksums["sha256"])
        assert not any("/" in k for k in adata.uns["manifest"])

    def test_study_meta_keeps_keys_whose_value_is_none(self, adata):
        gsm_meta = adata.uns["study_meta"]["gsm_meta"][GOOD]
        assert "mapping_rate" in gsm_meta
        assert gsm_meta["mapping_rate"] is None
        assert gsm_meta["organism"] == "Homo sapiens"

    def test_publications_are_json_text_that_decodes_to_the_records(self, adata):
        pubs = json.loads(adata.uns["study_meta"]["publications"])
        assert pubs == [{"pmid": "1", "title": "A paper"}]


class TestExport:
    def test_h5ad_round_trip(self, adata, tmp_path):
        path = tmp_path / "study.h5ad"
        adata.write_h5ad(path)
        back = ad.read_h5ad(path)
        assert back.shape == adata.shape
        np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())
        assert f"samples/{GOOD}/exon_counts.1pz" in list(back.uns["manifest"]["checksums"]["path"])
        assert list(back.uns["skipped_samples"]["gsm_id"]) == [HOLLOW]
        assert json.loads(back.uns["study_meta"]["publications"])[0]["pmid"] == "1"

    def test_zarr_round_trip(self, adata, tmp_path):
        pytest.importorskip("zarr")
        path = tmp_path / "study.zarr"
        adata.write_zarr(path)
        back = ad.read_zarr(path)
        assert back.shape == adata.shape
        assert list(back.uns["skipped_samples"]["gsm_id"]) == [HOLLOW]

    def test_bundle_to_h5ad(self, bundle_path, tmp_path):
        out = tmp_path / "direct.h5ad"
        with pytest.warns(UserWarning):
            SingletBundle.open(bundle_path).to_h5ad(out, verbose=False)
        assert ad.read_h5ad(out).n_obs == 2


class TestLoadGenes:
    def test_symbols_match_case_insensitively_and_keep_ensembl_ids(self, bundle_path):
        import singlet

        with pytest.warns(UserWarning):
            sub = singlet.load(bundle_path, genes=["bbb"])
        assert list(sub.var_names) == ["ENSG02"]

    def test_duplicate_symbol_keeps_every_id(self, bundle_path):
        import singlet

        with pytest.warns(UserWarning):
            sub = singlet.load(bundle_path, genes=["AAA", "ENSG02"])
        assert list(sub.var_names) == ["ENSG01", "ENSG02", "ENSG03"]


class TestHollowBundles:
    """No compiled codec needed: 0x0 stubs are recognised from their header."""

    def test_all_hollow_bundle_raises_with_study_link(self, tmp_path):
        path = _write_bundle(
            tmp_path / f"{GSE}.singlet",
            {f"samples/{HOLLOW}/exon_counts.1pz": _stub_1pz()},
            [HOLLOW],
        )
        with pytest.warns(UserWarning, match="0x0"):
            with pytest.raises(RuntimeError, match="hollow") as exc:
                SingletBundle.open(path).to_anndata(verbose=False)
        assert f"https://singlet.bio/study/{GSE}" in str(exc.value)

    def test_missing_matrix_reason(self, tmp_path):
        path = _write_bundle(
            tmp_path / f"{GSE}.singlet", {f"samples/{HOLLOW}/cell_calls.tsv": CELL_CALLS}, [HOLLOW]
        )
        with SingletBundle.open(path) as b:
            with pytest.raises(ValueError, match="exon_counts.1pz is missing"):
                b.raw_counts(HOLLOW)

    def test_gsm_filter_reports_skip_reason(self):
        import pandas as pd

        from singlet._loader import _skip_reason

        adata = ad.AnnData(np.zeros((1, 1)))
        adata.uns["skipped_samples"] = pd.DataFrame(
            {"gsm_id": [HOLLOW], "reason": ["count matrix is empty (0x0)"]}
        )
        assert _skip_reason(adata, HOLLOW) == "count matrix is empty (0x0)"
        assert _skip_reason(adata, GOOD) is None
        assert _skip_reason(ad.AnnData(np.zeros((1, 1))), GOOD) is None
