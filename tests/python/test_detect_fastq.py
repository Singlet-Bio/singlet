# SPDX-License-Identifier: MIT
"""Tests for singlet.preprocessing._detect FASTQ-based functions."""

import gzip
from pathlib import Path

import pytest
from singlet.preprocessing._detect import (
    _check_barcode_fraction,
    _detect_read_length,
    _load_barcode_whitelist,
    detect_protocol,
)

# ---------------------------------------------------------------------------
# Helper: create synthetic FASTQ
# ---------------------------------------------------------------------------


def _write_fastq(path: Path, read_length: int, n_reads: int = 50, compressed: bool = False):
    """Write a minimal synthetic FASTQ file."""
    lines = []
    for i in range(n_reads):
        seq = "A" * read_length
        qual = "I" * read_length
        lines.extend([f"@read_{i}", seq, "+", qual])

    content = "\n".join(lines) + "\n"
    if compressed:
        with gzip.open(path, "wt") as f:
            f.write(content)
    else:
        path.write_text(content)


# ---------------------------------------------------------------------------
# _detect_read_length
# ---------------------------------------------------------------------------


class TestDetectReadLength:
    def test_uncompressed(self, tmp_path):
        """Reads plain FASTQ and returns correct length."""
        fq = tmp_path / "reads.fastq"
        _write_fastq(fq, read_length=28, n_reads=20)
        assert _detect_read_length(fq, num_reads=10) == 28

    def test_gzipped(self, tmp_path):
        """Reads gzipped FASTQ."""
        fq = tmp_path / "reads.fastq.gz"
        _write_fastq(fq, read_length=91, n_reads=20, compressed=True)
        assert _detect_read_length(fq, num_reads=10) == 91

    def test_empty_file(self, tmp_path):
        """Empty file returns 0."""
        fq = tmp_path / "empty.fastq"
        fq.write_text("")
        assert _detect_read_length(fq) == 0

    def test_limited_reads(self, tmp_path):
        """Stops after num_reads."""
        fq = tmp_path / "many.fastq"
        _write_fastq(fq, read_length=50, n_reads=500)
        result = _detect_read_length(fq, num_reads=10)
        assert result == 50


# ---------------------------------------------------------------------------
# _check_barcode_fraction
# ---------------------------------------------------------------------------


class TestCheckBarcodeFraction:
    def test_all_match(self, tmp_path):
        """All reads match whitelist → fraction ~1.0."""
        fq = tmp_path / "reads.fastq"
        barcode = "ACGTACGTACGTACGT"  # 16bp
        lines = []
        for i in range(10):
            seq = barcode + "NNNN" * 10  # 16bp barcode + padding
            lines.extend([f"@read_{i}", seq, "+", "I" * len(seq)])
        fq.write_text("\n".join(lines) + "\n")

        whitelist = {barcode}
        frac = _check_barcode_fraction(fq, whitelist, n_reads=10, bc_len=16)
        assert frac == 1.0

    def test_no_match(self, tmp_path):
        """No reads match → fraction 0.0."""
        fq = tmp_path / "reads.fastq"
        _write_fastq(fq, read_length=28, n_reads=10)

        whitelist = {"CCCCCCCCCCCCCCCC"}
        frac = _check_barcode_fraction(fq, whitelist, n_reads=10, bc_len=16)
        assert frac == 0.0

    def test_empty_whitelist(self, tmp_path):
        """Empty whitelist returns 0.0."""
        fq = tmp_path / "reads.fastq"
        _write_fastq(fq, read_length=28, n_reads=5)
        frac = _check_barcode_fraction(fq, set(), n_reads=5)
        assert frac == 0.0

    def test_partial_match(self, tmp_path):
        """Some reads match → fraction between 0 and 1."""
        fq = tmp_path / "reads.fastq"
        barcode = "ACGTACGTACGTACGT"
        lines = []
        for i in range(10):
            # First 5 reads have matching barcode, last 5 don't
            if i < 5:
                seq = barcode + "N" * 12
            else:
                seq = "T" * 28
            lines.extend([f"@read_{i}", seq, "+", "I" * len(seq)])
        fq.write_text("\n".join(lines) + "\n")

        whitelist = {barcode}
        frac = _check_barcode_fraction(fq, whitelist, n_reads=10, bc_len=16)
        assert frac == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _load_barcode_whitelist
# ---------------------------------------------------------------------------


class TestLoadBarcodeWhitelist:
    def test_loads_from_alevin_home(self, tmp_path, monkeypatch):
        """Loads barcodes when ALEVIN_FRY_HOME is set."""
        monkeypatch.setenv("ALEVIN_FRY_HOME", str(tmp_path))
        bc_file = tmp_path / "3M-february-2018.txt"
        bc_file.write_text("AAACCCAAGAAACAAT\nAAAGTAGAGAAACAAT\n")

        result = _load_barcode_whitelist()
        assert "AAACCCAAGAAACAAT" in result
        assert len(result) == 2

    def test_empty_when_no_env(self, monkeypatch):
        """Returns empty set when ALEVIN_FRY_HOME not set or empty."""
        monkeypatch.setenv("ALEVIN_FRY_HOME", "")
        result = _load_barcode_whitelist()
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# detect_protocol (end-to-end with synthetic FASTQs)
# ---------------------------------------------------------------------------


class TestDetectProtocol:
    def test_10xv3_from_fastq(self, tmp_path):
        """28bp R1 + 91bp R2 → 10xv3."""
        r1 = tmp_path / "R1.fastq"
        r2 = tmp_path / "R2.fastq"
        _write_fastq(r1, read_length=28, n_reads=20)
        _write_fastq(r2, read_length=91, n_reads=20)

        result = detect_protocol(r1, r2)
        assert result.protocol == "10xv3"
        assert result.confidence == "high"

    def test_10xv2_from_fastq(self, tmp_path):
        """26bp R1 + 98bp R2 → 10xv2."""
        r1 = tmp_path / "R1.fastq"
        r2 = tmp_path / "R2.fastq"
        _write_fastq(r1, read_length=26, n_reads=20)
        _write_fastq(r2, read_length=98, n_reads=20)

        result = detect_protocol(r1, r2)
        assert result.protocol == "10xv2"

    def test_catalog_hint_overrides_fastq(self, tmp_path):
        """Catalog hint takes priority over read-length heuristics."""
        r1 = tmp_path / "R1.fastq"
        _write_fastq(r1, read_length=150, n_reads=10)

        result = detect_protocol(r1, catalog_hint="SmartSeq2 (smartseq)")
        assert result.protocol == "smartseq"

    def test_r1_only(self, tmp_path):
        """Single-end detection works (r2_len=0)."""
        r1 = tmp_path / "R1.fastq"
        _write_fastq(r1, read_length=28, n_reads=10)

        result = detect_protocol(r1)
        # 28bp with no R2 → unknown/ambiguous
        assert result.protocol in ("unknown", "ambiguous", "10xv3", "10xv2")

    def test_gzipped_fastq(self, tmp_path):
        """Works with .fastq.gz files."""
        r1 = tmp_path / "R1.fastq.gz"
        r2 = tmp_path / "R2.fastq.gz"
        _write_fastq(r1, read_length=28, n_reads=20, compressed=True)
        _write_fastq(r2, read_length=91, n_reads=20, compressed=True)

        result = detect_protocol(r1, r2)
        assert result.protocol == "10xv3"
