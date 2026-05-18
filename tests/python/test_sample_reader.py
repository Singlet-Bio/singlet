# SPDX-License-Identifier: MIT
"""Behavioural tests for the canonical v2 sample reader stubs.

These tests pin the public API surface of :mod:`singlet.io.sample` so
the Phase-6 implementation can land without breaking signatures.

Where reader bodies are still ``NotImplementedError`` stubs, we only
assert that the right call path raises that exception — never that it
silently no-ops.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from singlet.io import (
    SingletCounts,
    SingletMt,
    SingletNonhost,
    SingletSample,
    SingletSnp,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def sample_dir(tmp_path):
    """Build a fake canonical v2 sample directory with the minimum
    set of files SingletSample touches eagerly."""
    p = tmp_path / "GSMFAKE"
    p.mkdir()
    (p / "summary.json").write_text(
        json.dumps(
            {
                "n_cells": 12,
                "protocol": "10xv3",
                "mapping_rate": 0.91,
                "reference": {"build_id": "GRCh38-2024-A"},
            }
        )
    )
    # Touch the placeholder .1pz files; readers are lazy so contents
    # don't need to be valid until matrix methods are called.
    for name in ("counts.1pz", "snp.1pz", "mt.1pz"):
        (p / name).write_bytes(b"")
    # Minimal pyarrow parquet — write a tiny table.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "barcode": ["AAAA", "CCCC", "GGGG"],
            "n_umi": [100, 200, 300],
        }
    )
    pq.write_table(table, p / "cell_meta.parquet")
    return p


# --------------------------------------------------------------------------
# SingletSample
# --------------------------------------------------------------------------


class TestSingletSample:
    def test_summary(self, sample_dir):
        s = SingletSample(sample_dir)
        assert s.summary["n_cells"] == 12
        assert s.summary["protocol"] == "10xv3"

    def test_path_property(self, sample_dir):
        s = SingletSample(sample_dir)
        assert s.path == Path(sample_dir)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="summary.json"):
            SingletSample(tmp_path / "nope")

    def test_no_summary_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="summary.json"):
            SingletSample(empty)

    def test_subreader_types(self, sample_dir):
        s = SingletSample(sample_dir)
        assert isinstance(s.counts, SingletCounts)
        assert isinstance(s.snp, SingletSnp)
        assert isinstance(s.mt, SingletMt)

    def test_cell_meta_returns_table(self, sample_dir):
        import pyarrow as pa

        s = SingletSample(sample_dir)
        table = s.cell_meta
        assert isinstance(table, pa.Table)
        assert table.num_rows == 3
        assert "barcode" in table.column_names

    def test_nonhost_absent_returns_none(self, sample_dir):
        s = SingletSample(sample_dir)
        assert s.nonhost is None

    def test_nonhost_present(self, sample_dir):
        (sample_dir / "nonhost.json").write_text("{}")
        s = SingletSample(sample_dir)
        assert isinstance(s.nonhost, SingletNonhost)

    @pytest.mark.parametrize(
        "filename,attr",
        [
            ("guides.1pz", "guides_path"),
            ("antibodies.1pz", "antibodies_path"),
            ("vdj_gene_usage.1pz", "vdj_path"),
        ],
    )
    def test_optional_paths(self, sample_dir, filename, attr):
        s = SingletSample(sample_dir)
        assert getattr(s, attr) is None
        (sample_dir / filename).write_bytes(b"")
        assert getattr(s, attr) == sample_dir / filename


# --------------------------------------------------------------------------
# Sub-readers — error surface
# --------------------------------------------------------------------------


class TestSubReaderErrors:
    """Phase 6 made the sub-readers real; ensure that calling them
    against an empty/placeholder ``.1pz`` raises a clear error from the
    pz_v2 codec rather than silently no-oping.
    """

    @pytest.mark.parametrize("method", ["exon_body", "intron_body", "junctions"])
    def test_counts_methods_raise_on_empty(self, sample_dir, method):
        s = SingletSample(sample_dir)
        with pytest.raises(Exception):
            getattr(s.counts, method)()

    @pytest.mark.parametrize("method", ["ad", "dp"])
    def test_snp_methods_raise_on_empty(self, sample_dir, method):
        s = SingletSample(sample_dir)
        with pytest.raises(Exception):
            getattr(s.snp, method)()

    @pytest.mark.parametrize("method", ["ad", "dp"])
    def test_mt_methods_raise_on_empty(self, sample_dir, method):
        s = SingletSample(sample_dir)
        with pytest.raises(Exception):
            getattr(s.mt, method)()

    def test_nonhost_missing_json_returns_none(self, sample_dir):
        # Without nonhost.json present, SingletSample.nonhost is None.
        s = SingletSample(sample_dir)
        assert s.nonhost is None


# --------------------------------------------------------------------------
# Views — error surface
# --------------------------------------------------------------------------


class TestViewsErrors:
    """The views require a features bundle; calling without one raises TypeError."""

    def test_gene_counts_requires_features(self, sample_dir):
        from singlet.views import gene_counts

        with pytest.raises(TypeError):
            gene_counts(SingletSample(sample_dir))  # type: ignore[call-arg]

    def test_usa_requires_features(self, sample_dir):
        from singlet.views import usa

        with pytest.raises(TypeError):
            usa(SingletSample(sample_dir))  # type: ignore[call-arg]

    def test_psi_requires_features(self, sample_dir):
        from singlet.views import psi

        with pytest.raises(TypeError):
            psi(SingletSample(sample_dir))  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Public-surface re-exports
# --------------------------------------------------------------------------


class TestPublicSurface:
    def test_io_reexports(self):
        import singlet.io as io_pkg

        for name in (
            "SingletSample",
            "SingletCounts",
            "SingletSnp",
            "SingletMt",
            "SingletNonhost",
        ):
            assert hasattr(io_pkg, name), f"singlet.io missing {name}"

    def test_views_reexports(self):
        import singlet.views as v

        for name in ("gene_counts", "usa", "psi"):
            assert hasattr(v, name), f"singlet.views missing {name}"

    def test_pipeline_reexports(self):
        import singlet.pipeline as p

        for name in ("run", "Run", "PipelineError"):
            assert hasattr(p, name), f"singlet.pipeline missing {name}"
