# SPDX-License-Identifier: MIT
"""Tests for _io format detection, read_matrix dispatch, and read_kraken2."""

import struct
import sys
from types import ModuleType
from unittest.mock import MagicMock

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


class TestReadKraken2:
    """Test read_kraken2 with mocked singlepress."""

    def test_missing_file_raises(self, tmp_path):
        """Raises FileNotFoundError if kraken2.1pz missing."""
        from singlet._io import read_kraken2

        with pytest.raises(FileNotFoundError, match="No kraken2.1pz"):
            read_kraken2(tmp_path)

    def test_with_singlepress_module(self, tmp_path, monkeypatch):
        """Uses singlepress.read_1pz when available."""
        from singlet._io import read_kraken2

        # Create dummy kraken2.1pz file
        k2_path = tmp_path / "kraken2.1pz"
        k2_path.write_bytes(b"dummy")

        # Mock singlepress module
        mock_sp = MagicMock()
        mat = sp.csc_matrix(np.array([[1, 0, 2], [3, 4, 0]], dtype=np.float64))
        mat.rownames = ["taxon_A", "taxon_B"]
        mat.uns = {"version": "1.0"}
        mock_sp.read_1pz.return_value = mat
        monkeypatch.setitem(sys.modules, "singlepress", mock_sp)

        adata = read_kraken2(tmp_path)
        # Transpose: 2 taxa × 3 cells → 3 cells × 2 taxa
        assert adata.shape == (3, 2)
        assert list(adata.var_names) == ["taxon_A", "taxon_B"]
        assert adata.uns["version"] == "1.0"

    def test_with_features_parquet(self, tmp_path, monkeypatch):
        """Loads kraken2_features.parquet into var if present."""
        import pandas as pd
        from singlet._io import read_kraken2

        k2_path = tmp_path / "kraken2.1pz"
        k2_path.write_bytes(b"dummy")

        # Mock singlepress
        mock_sp = MagicMock()
        mat = sp.csc_matrix(np.array([[5, 0], [0, 3]], dtype=np.float64))
        mat.rownames = ["tax1", "tax2"]
        mat.uns = {}
        mock_sp.read_1pz.return_value = mat
        monkeypatch.setitem(sys.modules, "singlepress", mock_sp)

        # Create features parquet
        feat_df = pd.DataFrame(
            {"kingdom": ["Bacteria", "Viruses"], "abundance": [0.8, 0.2]},
            index=["tax1", "tax2"],
        )
        feat_path = tmp_path / "kraken2_features.parquet"
        feat_df.to_parquet(feat_path)

        adata = read_kraken2(tmp_path)
        assert "kingdom" in adata.var.columns
        assert list(adata.var["kingdom"]) == ["Bacteria", "Viruses"]

    def test_fallback_to_pz_codec(self, tmp_path, monkeypatch):
        """Falls back to singlepress._pz_codec.pz_read on ImportError."""
        from singlet._io import read_kraken2

        k2_path = tmp_path / "kraken2.1pz"
        k2_path.write_bytes(b"dummy")

        # Create a mock singlepress that raises AttributeError on read_1pz
        mock_sp = ModuleType("singlepress")
        # Don't define read_1pz at all → AttributeError when accessed

        mock_codec = MagicMock()
        mock_codec.pz_read.return_value = {
            "values": np.array([1.0, 2.0, 3.0]),
            "indices": np.array([0, 1, 0]),
            "indptr": np.array([0, 2, 3]),
            "m": 2,
            "n": 2,
        }
        mock_sp._pz_codec = mock_codec
        monkeypatch.setitem(sys.modules, "singlepress", mock_sp)
        monkeypatch.setitem(sys.modules, "singlepress._pz_codec", mock_codec)

        adata = read_kraken2(tmp_path)
        # 2 taxa × 2 cells → 2 cells × 2 taxa
        assert adata.shape == (2, 2)
