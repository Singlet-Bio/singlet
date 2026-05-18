# SPDX-License-Identifier: MIT
"""Tests for v1 → v2 transcoder."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.sparse import csc_matrix

singlepress = pytest.importorskip("singlepress")

from singlet.pz_v2 import read_pz_v2
from singlet.transcode import transcode_v1_to_v2


def _write_legacy(path, mat, rownames, colnames):
    singlepress.write_1pz(
        str(path), mat, rownames=rownames, colnames=colnames, num_threads=2
    )


def test_transcode_gene_counts_only(tmp_path):
    src = tmp_path / "v1"
    src.mkdir()
    mat = csc_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32))
    _write_legacy(src / "gene_counts.1pz", mat, ["g0", "g1"], ["c0", "c1", "c2"])

    dst = tmp_path / "v2"
    res = transcode_v1_to_v2(src, dst)

    assert res.blocks_written == ["gene_counts"]
    assert res.n_cells == 3
    assert (dst / "counts.1pz").exists()
    assert (dst / "summary.json").exists()

    with read_pz_v2(dst / "counts.1pz") as rd:
        assert rd.cell_barcodes == ["c0", "c1", "c2"]
        np.testing.assert_array_equal(
            rd.block("gene_counts").data().toarray(), mat.toarray()
        )

    summary = json.loads((dst / "summary.json").read_text())
    assert summary["transcoded_from"] == "v1"


def test_transcode_usa_triple_maps_to_v2_blocks(tmp_path):
    src = tmp_path / "v1"
    src.mkdir()
    a = csc_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
    b = csc_matrix(np.array([[0, 3], [4, 0]], dtype=np.int32))
    c = csc_matrix(np.array([[5, 0], [0, 0]], dtype=np.int32))
    _write_legacy(src / "spliced.1pz", a, ["g0", "g1"], ["A", "B"])
    _write_legacy(src / "unspliced.1pz", b, ["g0", "g1"], ["A", "B"])
    _write_legacy(src / "ambiguous.1pz", c, ["g0", "g1"], ["A", "B"])

    res = transcode_v1_to_v2(src, tmp_path / "v2")
    assert sorted(res.blocks_written) == ["exon_body", "intron_body", "junctions"]

    with read_pz_v2(tmp_path / "v2" / "counts.1pz") as rd:
        assert rd.cell_barcodes == ["A", "B"]
        np.testing.assert_array_equal(
            rd.block("exon_body").data().toarray(), a.toarray()
        )
        np.testing.assert_array_equal(
            rd.block("intron_body").data().toarray(), b.toarray()
        )
        np.testing.assert_array_equal(
            rd.block("junctions").data().toarray(), c.toarray()
        )


def test_transcode_passes_through_summary(tmp_path):
    src = tmp_path / "v1"
    src.mkdir()
    mat = csc_matrix(np.array([[1, 0]], dtype=np.int32))
    _write_legacy(src / "gene_counts.1pz", mat, ["g0"], ["c0", "c1"])
    (src / "summary.json").write_text(json.dumps({"sample_id": "X", "n_cells": 2}))

    res = transcode_v1_to_v2(src, tmp_path / "v2")
    summary = json.loads((tmp_path / "v2" / "summary.json").read_text())
    assert summary["sample_id"] == "X"
    assert "summary.json" in res.files_copied


def test_transcode_no_legacy_files(tmp_path):
    src = tmp_path / "v1"
    src.mkdir()
    with pytest.raises(FileNotFoundError):
        transcode_v1_to_v2(src, tmp_path / "v2")


def test_transcode_overwrite_protection(tmp_path):
    src = tmp_path / "v1"
    src.mkdir()
    mat = csc_matrix(np.array([[1]], dtype=np.int32))
    _write_legacy(src / "gene_counts.1pz", mat, ["g0"], ["c0"])
    dst = tmp_path / "v2"
    transcode_v1_to_v2(src, dst)
    with pytest.raises(FileExistsError):
        transcode_v1_to_v2(src, dst)
    # overwrite=True works
    transcode_v1_to_v2(src, dst, overwrite=True)
