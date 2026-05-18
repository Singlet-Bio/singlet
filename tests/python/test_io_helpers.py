# SPDX-License-Identifier: MIT
"""Tests for singlet._io public helpers and validation paths."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


class TestWrite1pzUnsFiltering:
    """Test write_1pz with non-scalar uns values."""

    def test_non_scalar_uns_excluded(self, tmp_path):
        """Non-scalar uns values (arrays, dicts) are excluded from metadata."""
        import anndata as ad
        from singlet._io import read_1pz, write_1pz

        mat = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        adata = ad.AnnData(X=mat)
        adata.uns["array_val"] = np.array([1, 2, 3])
        adata.uns["dict_val"] = {"nested": True}
        adata.uns["list_val"] = [1, 2, 3]

        path = tmp_path / "no_uns.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)
        assert loaded.shape == (2, 2)
        assert len(loaded.uns) == 0

    def test_mixed_uns_keeps_scalars(self, tmp_path):
        """Scalar uns values survive roundtrip, non-scalars are excluded."""
        import anndata as ad
        from singlet._io import read_1pz, write_1pz

        mat = sp.csr_matrix(np.array([[5, 6], [7, 8]], dtype=np.float32))
        adata = ad.AnnData(X=mat)
        adata.uns["version"] = "1.0"
        adata.uns["n_cells"] = 42
        adata.uns["big_array"] = np.zeros(1000)

        path = tmp_path / "mixed_uns.1pz"
        write_1pz(adata, path)
        loaded = read_1pz(path)
        assert "version" in loaded.uns
        assert loaded.uns["version"] == "1.0"


class TestRead1pzFallback:
    def test_read_1pz_fallback_pz_codec(self, tmp_path):
        """read_1pz falls back to pz_read when singlepress.read_1pz missing."""
        from types import ModuleType
        from unittest.mock import MagicMock, patch

        fake_sp = ModuleType("singlepress")
        fake_sp._pz_codec = MagicMock()
        fake_sp._pz_codec.pz_read.return_value = {
            "m": 3,
            "n": 4,
            "values": np.array([1.0, 2.0, 3.0]),
            "indices": np.array([0, 1, 2]),
            "indptr": np.array([0, 1, 2, 3, 3]),
        }

        fake_file = tmp_path / "fake.1pz"
        fake_file.write_bytes(b"\x00")

        import sys

        with patch.dict(
            sys.modules,
            {"singlepress": fake_sp, "singlepress._pz_codec": fake_sp._pz_codec},
        ):
            from singlet._io import read_1pz

            adata = read_1pz(str(fake_file))
            assert adata.shape == (4, 3)


class TestReadWrite1pzValidation:
    """Input validation for read_1pz / write_1pz."""

    def test_read_1pz_none_raises_typeerror(self):
        from singlet._io import read_1pz

        with pytest.raises(TypeError, match="got None"):
            read_1pz(None)

    def test_read_1pz_empty_raises_valueerror(self):
        from singlet._io import read_1pz

        with pytest.raises(ValueError, match="non-empty"):
            read_1pz("")

    def test_read_1pz_nonexistent_raises_filenotfounderror(self):
        from singlet._io import read_1pz

        with pytest.raises(FileNotFoundError, match="File not found"):
            read_1pz("/nonexistent/path/to/file.1pz")

    def test_write_1pz_none_raises_typeerror(self):
        from singlet._io import write_1pz

        with pytest.raises(TypeError, match="got None"):
            write_1pz(None, "/tmp/out.1pz")

    def test_write_1pz_non_adata_raises_typeerror(self):
        from singlet._io import write_1pz

        with pytest.raises(TypeError, match="write_1pz.*requires an AnnData"):
            write_1pz({"X": [1, 2]}, "/tmp/out.1pz")


def test_info_1pz_nonexistent_raises():
    """info_1pz raises FileNotFoundError for missing file."""
    from singlet._io import info_1pz

    with pytest.raises(FileNotFoundError, match="File not found"):
        info_1pz("/nonexistent/path/data.1pz")
