# SPDX-License-Identifier: MIT
"""Tests for the .1pz v2 multi-block codec."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from singlet.pz_v2 import (
    MAGIC,
    BlockSpec,
    PzV2Error,
    read_pz_v2,
    write_pz_v2,
)


def _make_csc(rows, cols, dense_pattern, dtype=np.int32):
    arr = np.asarray(dense_pattern, dtype=dtype).reshape(rows, cols)
    return csc_matrix(arr)


def test_round_trip_single_block(tmp_path):
    X = _make_csc(3, 4, [
        1, 0, 2, 0,
        0, 3, 0, 0,
        4, 0, 0, 5,
    ])
    bc = ["AAA", "CCC", "GGG", "TTT"]
    out = tmp_path / "x.1pz"
    write_pz_v2(out, cell_barcodes=bc, blocks=[BlockSpec("exon_body", X)])
    with read_pz_v2(out) as rd:
        assert rd.cell_barcodes == bc
        assert rd.n_cells == 4
        assert rd.block_names == ["exon_body"]
        b = rd.block("exon_body")
        assert b.shape == (3, 4)
        assert b.n_data_layers == 1
        assert b.data_names == ["counts"]
        np.testing.assert_array_equal(b.data().toarray(), X.toarray())


def test_round_trip_multi_block(tmp_path):
    A = _make_csc(2, 3, [1, 0, 2, 0, 3, 0])
    B = _make_csc(5, 3, [
        0, 0, 0,
        1, 0, 0,
        0, 2, 0,
        0, 0, 3,
        0, 4, 0,
    ])
    out = tmp_path / "x.1pz"
    write_pz_v2(
        out,
        cell_barcodes=["c0", "c1", "c2"],
        blocks=[BlockSpec("exon_body", A), BlockSpec("junctions", B)],
    )
    with read_pz_v2(out) as rd:
        assert rd.block_names == ["exon_body", "junctions"]
        np.testing.assert_array_equal(rd.block("exon_body").data().toarray(), A.toarray())
        np.testing.assert_array_equal(rd.block("junctions").data().toarray(), B.toarray())


def test_two_layer_csc_ad_dp(tmp_path):
    ad_arr = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32)
    dp_arr = np.array([[4, 0, 5], [0, 6, 0]], dtype=np.int32)
    ad = csc_matrix(ad_arr)
    dp = csc_matrix(dp_arr)
    assert ad.nnz == dp.nnz
    out = tmp_path / "snp.1pz"
    write_pz_v2(
        out,
        cell_barcodes=["a", "b", "c"],
        blocks=[BlockSpec("snp", ad, data2=dp, data_names=["ad", "dp"])],
    )
    with read_pz_v2(out) as rd:
        b = rd.block("snp")
        assert b.n_data_layers == 2
        assert b.data_names == ["ad", "dp"]
        np.testing.assert_array_equal(b.data("ad").toarray(), ad_arr)
        np.testing.assert_array_equal(b.data("dp").toarray(), dp_arr)
        # access by index also works
        np.testing.assert_array_equal(b.data(0).toarray(), ad_arr)
        np.testing.assert_array_equal(b.data(1).toarray(), dp_arr)


def test_two_layer_pattern_mismatch_raises(tmp_path):
    ad = csc_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32))
    # dp has an extra nonzero → pattern mismatch
    dp = csc_matrix(np.array([[3, 4], [0, 5]], dtype=np.int32))
    with pytest.raises(PzV2Error, match="nnz mismatch"):
        write_pz_v2(
            tmp_path / "x.1pz",
            cell_barcodes=["a", "b"],
            blocks=[BlockSpec("snp", ad, data2=dp, data_names=["ad", "dp"])],
        )


def test_bad_magic_rejected(tmp_path):
    p = tmp_path / "bad.1pz"
    p.write_bytes(b"XXXXXXXX" + struct.pack("<I", 0))
    with pytest.raises(PzV2Error, match="bad magic"):
        read_pz_v2(p)


def test_unknown_block_raises(tmp_path):
    out = tmp_path / "x.1pz"
    X = _make_csc(2, 2, [1, 0, 0, 2])
    write_pz_v2(out, cell_barcodes=["a", "b"], blocks=[BlockSpec("exon_body", X)])
    with read_pz_v2(out) as rd:
        with pytest.raises(KeyError):
            rd.block("not_a_block")


def test_extra_header_preserved(tmp_path):
    X = _make_csc(2, 2, [1, 0, 0, 2])
    out = tmp_path / "x.1pz"
    write_pz_v2(
        out,
        cell_barcodes=["a", "b"],
        blocks=[BlockSpec("exon_body", X)],
        extra_header={"row_axis": "exon_interval", "build_id": "GRCh38_v44"},
    )
    with read_pz_v2(out) as rd:
        assert rd.extra_header == {"row_axis": "exon_interval", "build_id": "GRCh38_v44"}


def test_file_starts_with_magic(tmp_path):
    X = _make_csc(2, 2, [1, 0, 0, 2])
    out = tmp_path / "x.1pz"
    write_pz_v2(out, cell_barcodes=["a", "b"], blocks=[BlockSpec("exon_body", X)])
    raw = out.read_bytes()
    assert raw[:8] == MAGIC
    (header_len,) = struct.unpack("<I", raw[8:12])
    hdr = json.loads(raw[12 : 12 + header_len].decode("utf-8"))
    assert hdr["version"] == 2
    assert hdr["n_cells"] == 2


def test_cell_count_mismatch_raises(tmp_path):
    X = _make_csc(2, 3, [1, 0, 0, 0, 2, 0])  # 3 cols
    with pytest.raises(PzV2Error, match="cell axis"):
        write_pz_v2(
            tmp_path / "x.1pz",
            cell_barcodes=["a", "b"],  # 2 cells
            blocks=[BlockSpec("exon_body", X)],
        )


def test_empty_barcodes_raises(tmp_path):
    with pytest.raises(PzV2Error, match="non-empty"):
        write_pz_v2(tmp_path / "x.1pz", cell_barcodes=[], blocks=[])
