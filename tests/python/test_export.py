"""Tests for singlet.preprocessing._export (quantification export)."""

from unittest.mock import patch

import numpy as np
import scipy.io
import scipy.sparse as sp
from singlet.preprocessing._export import export_to_1pz, export_to_spz

# ---------------------------------------------------------------------------
# export_to_1pz
# ---------------------------------------------------------------------------


class TestExportTo1pz:
    def _create_quant_dir(self, tmp_path, shape=(100, 50)):
        """Create a fake simpleaf output directory with mtx file."""
        alevin = tmp_path / "af_quant" / "alevin"
        alevin.mkdir(parents=True)
        mat = sp.random(shape[0], shape[1], density=0.1, format="coo")
        scipy.io.mmwrite(alevin / "quants_mat.mtx", mat)
        return tmp_path

    @patch("singlepress.write_1pz")
    def test_success(self, mock_write, tmp_path):
        qdir = self._create_quant_dir(tmp_path / "quant")
        out_path = tmp_path / "output" / "sample.1pz"
        result = export_to_1pz(qdir, out_path, sample_id="GSM001")
        assert result is True
        mock_write.assert_called_once()
        # Check the output parent dir was created
        assert out_path.parent.exists()

    @patch("singlepress.write_1pz")
    def test_matrix_is_transposed(self, mock_write, tmp_path):
        """Export should write genes × cells (transposed from cells × genes)."""
        qdir = self._create_quant_dir(tmp_path / "quant", shape=(20, 10))
        out_path = tmp_path / "sample.1pz"
        export_to_1pz(qdir, out_path)
        # write_1pz called with (path, matrix) — matrix should be 10×20
        call_args = mock_write.call_args[0]
        assert call_args[1].shape == (10, 20)

    def test_no_matrix_returns_false(self, tmp_path):
        """Returns False if no count matrix found."""
        qdir = tmp_path / "empty"
        qdir.mkdir()
        result = export_to_1pz(qdir, tmp_path / "out.1pz")
        assert result is False

    @patch("singlepress.write_1pz")
    def test_gzipped_matrix(self, mock_write, tmp_path):
        """Finds .mtx.gz variant."""
        import gzip

        alevin = tmp_path / "quant" / "af_quant" / "alevin"
        alevin.mkdir(parents=True)

        mat = sp.random(5, 3, density=0.5, format="coo")
        # Write to buffer then gzip
        import io

        buf = io.BytesIO()
        scipy.io.mmwrite(buf, mat)
        with gzip.open(alevin / "quants_mat.mtx.gz", "wb") as f:
            f.write(buf.getvalue())

        out_path = tmp_path / "sample.1pz"
        result = export_to_1pz(tmp_path / "quant", out_path)
        assert result is True


# ---------------------------------------------------------------------------
# export_to_spz
# ---------------------------------------------------------------------------


class TestExportToSpz:
    def _create_quant_dir(self, tmp_path, n_cells=10, n_genes=5):
        """Create fake simpleaf output with gene names and barcodes."""
        alevin = tmp_path / "af_quant" / "alevin"
        alevin.mkdir(parents=True)
        mat = sp.random(n_cells, n_genes, density=0.3, format="coo", dtype=np.int32)
        scipy.io.mmwrite(alevin / "quants_mat.mtx", mat)

        # Write gene names
        with open(alevin / "quants_mat_cols.txt", "w") as f:
            for i in range(n_genes):
                f.write(f"GENE{i}\n")

        # Write barcodes
        with open(alevin / "quants_mat_rows.txt", "w") as f:
            for i in range(n_cells):
                f.write(f"BARCODE{i}\n")

        return tmp_path

    @patch("singlet._singlepress.sp_write_int")
    def test_integer_export(self, mock_write_int, tmp_path):
        """Integer matrix uses sp_write_int."""
        qdir = self._create_quant_dir(tmp_path / "quant")
        out_path = tmp_path / "sample.spz"
        result = export_to_spz(qdir, out_path, sample_id="test")
        assert result is True
        mock_write_int.assert_called_once()

    def test_no_matrix_returns_false(self, tmp_path):
        """Returns False if no count matrix found."""
        qdir = tmp_path / "empty"
        qdir.mkdir()
        result = export_to_spz(qdir, tmp_path / "out.spz")
        assert result is False
