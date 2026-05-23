# SPDX-License-Identifier: MIT
"""Tests for _io format detection, read_matrix dispatch, and read_kraken2."""

import struct
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


class TestDetectFormat:
    """Test _detect_format magic-byte identification."""

    def test_1pz_magic(self, tmp_path):
        """Detects .1pz from TP1Z magic."""
        from singlet._io import _detect_format

        path = tmp_path / "test.1pz"
        # TP1Z magic + 4 bytes padding
        path.write_bytes(b"\x54\x50\x31\x5a" + b"\x00" * 4)
        assert _detect_format(path) == "1pz"

    def test_spz_magic_no_longer_supported(self, tmp_path):
        """Legacy SPRZ/SPZ2 magic is rejected as unknown."""
        from singlet._io import _detect_format

        path = tmp_path / "test.spz"
        header = b"SPRZ" + b"\x00\x00" + struct.pack("<H", 256)
        path.write_bytes(header)
        with pytest.raises(ValueError, match="Unknown file format"):
            _detect_format(path)

    def test_unknown_magic_raises(self, tmp_path):
        """Raises ValueError for unrecognized magic."""
        from singlet._io import _detect_format

        path = tmp_path / "test.bin"
        path.write_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00")
        with pytest.raises(ValueError, match="Unknown file format"):
            _detect_format(path)

    def test_too_small_raises(self, tmp_path):
        """Raises ValueError if file < 8 bytes."""
        from singlet._io import _detect_format

        path = tmp_path / "tiny.bin"
        path.write_bytes(b"\x00\x00")
        with pytest.raises(ValueError, match="too small"):
            _detect_format(path)

    def test_empty_file_raises(self, tmp_path):
        """Empty file raises ValueError."""
        from singlet._io import _detect_format

        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="too small"):
            _detect_format(path)


class TestReadMatrix:
    """Test read_matrix format dispatch."""

    def test_nonexistent_file_raises(self):
        from singlet._io import read_matrix

        with pytest.raises((FileNotFoundError, OSError)):
            read_matrix("/nonexistent/path/file.1pz")

    def test_bad_format_raises(self, tmp_path):
        from singlet._io import read_matrix

        f = tmp_path / "garbage.1pz"
        f.write_bytes(b"XXXX" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Unknown file format"):
            read_matrix(f)


def _write_minimal_1pz(path: Path, taxa: list[str], n_cells: int, data_array) -> None:
    """Helper: write a small valid .1pz file using the native codec."""
    from singlet._pz import write_1pz as _native_write

    m = len(taxa)
    n = n_cells
    mat = sp.csc_matrix(data_array)
    _native_write(
        str(path),
        mat.indptr.astype(np.int32),
        mat.indices.astype(np.int32),
        mat.data.astype(np.uint32),
        m, n,
        rownames=taxa,
    )


class TestReadKraken2:
    """Test read_kraken2 using the in-tree codec (no singlepress mocking)."""

    def test_missing_file_raises(self, tmp_path):
        """Raises FileNotFoundError if kraken2.1pz missing."""
        from singlet._io import read_kraken2

        with pytest.raises(FileNotFoundError, match="No kraken2.1pz"):
            read_kraken2(tmp_path)

    def test_reads_taxa_and_cells(self, tmp_path):
        """read_kraken2 returns cells × taxa AnnData with var_names set."""
        from singlet._io import read_kraken2

        taxa = ["taxon_A", "taxon_B"]
        # genes×cells data: 2 taxa × 3 cells
        data = sp.csc_matrix(np.array([[1, 0, 2], [3, 4, 0]], dtype=np.uint32))
        k2_path = tmp_path / "kraken2.1pz"
        _write_minimal_1pz(k2_path, taxa, 3, data)

        adata = read_kraken2(tmp_path)
        # Should be 3 cells × 2 taxa
        assert adata.shape == (3, 2)
        assert list(adata.var_names) == ["taxon_A", "taxon_B"]

    def test_with_features_parquet(self, tmp_path):
        """Loads kraken2_features.parquet into var if present."""
        pytest.importorskip("pyarrow")
        import pandas as pd

        from singlet._io import read_kraken2

        taxa = ["tax1", "tax2"]
        data = sp.csc_matrix(np.array([[5, 0], [0, 3]], dtype=np.uint32))
        k2_path = tmp_path / "kraken2.1pz"
        _write_minimal_1pz(k2_path, taxa, 2, data)

        feat_df = pd.DataFrame(
            {"kingdom": ["Bacteria", "Viruses"], "abundance": [0.8, 0.2]},
            index=["tax1", "tax2"],
        )
        feat_df.to_parquet(str(tmp_path / "kraken2_features.parquet"))

        adata = read_kraken2(tmp_path)
        assert "kingdom" in adata.var.columns
        assert list(adata.var["kingdom"]) == ["Bacteria", "Viruses"]
