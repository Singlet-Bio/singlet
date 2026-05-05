"""Tests for singlet.preprocessing._qc (QCMetrics, run_qc)."""

import gzip
import json

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp
from singlet.preprocessing._qc import QCMetrics, _read_mtx_shape, run_qc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_quant_dir(tmp_path):
    """Create a minimal simpleaf-style output directory."""
    alevin_dir = tmp_path / "af_quant" / "alevin"
    alevin_dir.mkdir(parents=True)

    # Create a sparse matrix (100 cells × 500 genes)
    rng = np.random.default_rng(42)
    mat = sp.random(100, 500, density=0.05, format="coo", random_state=rng)
    mat.data = (mat.data * 100).astype(np.int32)

    # Write as MTX
    mtx_path = alevin_dir / "quants_mat.mtx"
    scipy.io.mmwrite(str(mtx_path), mat)

    # Write map_info.json
    map_info = {"num_processed": 10000, "num_mapped": 8500}
    (alevin_dir / "map_info.json").write_text(json.dumps(map_info))

    return tmp_path


@pytest.fixture
def mock_quant_dir_gzipped(tmp_path):
    """Create directory with gzipped MTX."""
    alevin_dir = tmp_path / "af_quant" / "alevin"
    alevin_dir.mkdir(parents=True)

    mat = sp.random(50, 200, density=0.1, format="coo", random_state=7)
    mat.data = (mat.data * 50).astype(np.int32)

    mtx_path = alevin_dir / "quants_mat.mtx.gz"
    # Write MTX to gzip
    import io

    buf = io.BytesIO()
    scipy.io.mmwrite(buf, mat)
    with gzip.open(mtx_path, "wb") as f:
        f.write(buf.getvalue())

    return tmp_path


# ---------------------------------------------------------------------------
# QCMetrics dataclass
# ---------------------------------------------------------------------------


class TestQCMetrics:
    def test_defaults(self):
        qc = QCMetrics()
        assert qc.n_cells == 0
        assert qc.pass_qc is False
        assert qc.qc_status == "unknown"
        assert qc.fail_reasons == []

    def test_custom_values(self):
        qc = QCMetrics(n_cells=1000, pass_qc=True, qc_status="pass")
        assert qc.n_cells == 1000
        assert qc.pass_qc is True


# ---------------------------------------------------------------------------
# _read_mtx_shape
# ---------------------------------------------------------------------------


class TestReadMtxShape:
    def test_reads_shape(self, mock_quant_dir):
        mtx_path = mock_quant_dir / "af_quant" / "alevin" / "quants_mat.mtx"
        rows, cols, nnz = _read_mtx_shape(mtx_path)
        assert rows == 100
        assert cols == 500
        assert nnz > 0

    def test_reads_gzipped(self, mock_quant_dir_gzipped):
        mtx_path = mock_quant_dir_gzipped / "af_quant" / "alevin" / "quants_mat.mtx.gz"
        rows, cols, nnz = _read_mtx_shape(mtx_path)
        assert rows == 50
        assert cols == 200


# ---------------------------------------------------------------------------
# run_qc
# ---------------------------------------------------------------------------


class TestRunQC:
    def test_basic_pass(self, mock_quant_dir):
        """Good sample passes QC."""
        qc = run_qc(mock_quant_dir, min_cells=10, min_genes_per_cell=1)
        assert qc.n_cells == 100
        assert qc.n_genes == 500
        assert qc.mapping_rate == pytest.approx(0.85)
        assert qc.pass_qc is True
        assert qc.qc_status == "pass"

    def test_too_few_cells(self, mock_quant_dir):
        """Fail when min_cells threshold not met."""
        qc = run_qc(mock_quant_dir, min_cells=500)
        assert qc.pass_qc is False
        assert any("Too few cells" in r for r in qc.fail_reasons)

    def test_low_genes_per_cell(self, mock_quant_dir):
        """Warn/fail on low genes per cell."""
        qc = run_qc(mock_quant_dir, min_cells=10, min_genes_per_cell=5000)
        assert qc.pass_qc is False
        assert any("Low genes/cell" in r for r in qc.fail_reasons)

    def test_no_matrix_found(self, tmp_path):
        """Returns fail status when no matrix file exists."""
        empty = tmp_path / "empty_quant"
        empty.mkdir()
        qc = run_qc(empty)
        assert qc.qc_status == "fail"
        assert "No count matrix found" in qc.fail_reasons

    def test_gzipped_matrix(self, mock_quant_dir_gzipped):
        """Works with .mtx.gz files."""
        qc = run_qc(mock_quant_dir_gzipped, min_cells=10, min_genes_per_cell=1)
        assert qc.n_cells == 50
        assert qc.n_genes == 200

    def test_no_map_info(self, tmp_path):
        """Works when map_info.json is missing (mapping_rate = 0)."""
        alevin_dir = tmp_path / "af_quant" / "alevin"
        alevin_dir.mkdir(parents=True)
        mat = sp.random(20, 100, density=0.1, format="coo", random_state=3)
        mat.data = (mat.data * 10).astype(np.int32)
        scipy.io.mmwrite(str(alevin_dir / "quants_mat.mtx"), mat)

        qc = run_qc(tmp_path, min_cells=5, min_genes_per_cell=1, min_mapping_rate=0.0)
        assert qc.mapping_rate == 0.0
        assert qc.n_cells == 20

    def test_total_counts(self, mock_quant_dir):
        """total_counts is sum of all values."""
        qc = run_qc(mock_quant_dir, min_cells=1, min_genes_per_cell=1)
        assert qc.total_counts > 0
        assert qc.median_counts_per_cell > 0
