"""Tests for singlet._io format detection and read_matrix dispatch."""

import struct

import pytest
from singlet._io import _TP1_MAGIC, _detect_format, _detect_format_version

# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Test magic-byte format detection."""

    def test_1pz_format(self, tmp_path):
        """TP1Z magic → '1pz'."""
        f = tmp_path / "test.1pz"
        # Write valid 1pz header (magic + 4 more bytes)
        f.write_bytes(_TP1_MAGIC + b"\x00" * 4)
        assert _detect_format(f) == "1pz"

    def test_spz1_format(self, tmp_path):
        """SPRZ magic with 64-byte header → 'spz1'."""
        f = tmp_path / "test.spz"
        # SPRZ magic + 2 bytes padding + header_size=64 (little-endian u16 at offset 6)
        header = b"SPRZ" + b"\x00\x00" + struct.pack("<H", 64)
        f.write_bytes(header)
        assert _detect_format(f) == "spz1"

    def test_spz2_format(self, tmp_path):
        """SPZ2 magic with 128-byte header → 'spz2'."""
        f = tmp_path / "test.spz"
        # SPZ2 magic + 2 bytes + header_size=128
        header = b"SPZ2" + b"\x00\x00" + struct.pack("<H", 128)
        f.write_bytes(header)
        assert _detect_format(f) == "spz2"

    def test_unknown_magic_raises(self, tmp_path):
        """Unknown magic bytes raise ValueError."""
        f = tmp_path / "bad.bin"
        f.write_bytes(b"BAAD" + b"\x00" * 4)
        with pytest.raises(ValueError, match="Unknown file format"):
            _detect_format(f)

    def test_too_small_file_raises(self, tmp_path):
        """File smaller than 8 bytes raises ValueError."""
        f = tmp_path / "tiny.bin"
        f.write_bytes(b"\x00" * 3)
        with pytest.raises(ValueError, match="too small"):
            _detect_format(f)

    def test_empty_file_raises(self, tmp_path):
        """Empty file raises ValueError."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        with pytest.raises(ValueError, match="too small"):
            _detect_format(f)


# ---------------------------------------------------------------------------
# _detect_format_version
# ---------------------------------------------------------------------------


class TestDetectFormatVersion:
    def test_spz1_returns_1(self, tmp_path):
        f = tmp_path / "v1.spz"
        header = b"SPRZ" + b"\x00\x00" + struct.pack("<H", 64)
        f.write_bytes(header)
        assert _detect_format_version(f) == 1

    def test_spz2_returns_2(self, tmp_path):
        f = tmp_path / "v2.spz"
        header = b"SPZ2" + b"\x00\x00" + struct.pack("<H", 128)
        f.write_bytes(header)
        assert _detect_format_version(f) == 2

    def test_1pz_raises(self, tmp_path):
        f = tmp_path / "not_spz.1pz"
        f.write_bytes(_TP1_MAGIC + b"\x00" * 4)
        with pytest.raises(ValueError, match="Not a .spz file"):
            _detect_format_version(f)


# ---------------------------------------------------------------------------
# read_matrix dispatch (integration — requires singlepress)
# ---------------------------------------------------------------------------


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
