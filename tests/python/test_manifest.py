# SPDX-License-Identifier: MIT
"""Tests for singlet.manifest validator."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csc_matrix

from singlet.manifest import validate_sample
from singlet.pz_v2 import BlockSpec, write_pz_v2


def _build_minimal(sample):
    sample.mkdir(parents=True, exist_ok=True)
    bc = ["a", "b"]
    write_pz_v2(
        sample / "counts.1pz",
        cell_barcodes=bc,
        blocks=[BlockSpec("gene_counts", csc_matrix(np.array([[1, 0], [0, 2]], dtype=np.int32)))],
    )
    pq.write_table(pa.table({"barcode": bc, "n_umi": [1, 2]}), sample / "cell_meta.parquet")
    (sample / "summary.json").write_text(json.dumps({"sample_id": "s0"}))


def test_validate_passes_minimal_sample(tmp_path):
    s = tmp_path / "s0"
    _build_minimal(s)
    rep = validate_sample(s)
    assert rep.ok, rep.errors
    assert rep.n_cells == 2
    assert rep.block_names == ["gene_counts"]


def test_validate_missing_counts_fails(tmp_path):
    s = tmp_path / "s0"
    _build_minimal(s)
    (s / "counts.1pz").unlink()
    rep = validate_sample(s)
    assert not rep.ok
    assert any("counts.1pz" in e for e in rep.errors)


def test_validate_optional_snp_two_layer_warning(tmp_path):
    s = tmp_path / "s0"
    _build_minimal(s)
    # write a snp.1pz with only one data layer (wrong)
    write_pz_v2(
        s / "snp.1pz",
        cell_barcodes=["a", "b"],
        blocks=[BlockSpec("snp", csc_matrix(np.array([[1, 0]], dtype=np.int32)))],
    )
    rep = validate_sample(s)
    assert rep.ok  # warning, not error
    assert any("data layers" in w for w in rep.warnings)


def test_validate_not_a_dir(tmp_path):
    rep = validate_sample(tmp_path / "does_not_exist")
    assert not rep.ok


def test_validate_bad_summary_json(tmp_path):
    s = tmp_path / "s0"
    _build_minimal(s)
    (s / "summary.json").write_text("{ not valid json")
    rep = validate_sample(s)
    assert not rep.ok
