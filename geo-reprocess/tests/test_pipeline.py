"""Tests for pipeline module."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from scgeo.pipeline.api import SampleResult
from scgeo.pipeline.detect import (
    infer_protocol,
    ProtocolDetection,
    peek_fastq_read_length,
    peek_protocol,
    _PEEK_BARCODE_MAX_BP,
)
from scgeo.pipeline.download import _construct_ena_urls
from scgeo.pipeline.qc import QCMetrics, check_qc_thresholds
from scgeo.config.protocols import get_chemistry
from scgeo.config.defaults import DownloadConfig


def test_sample_result_structure():
    """Test SampleResult dataclass structure."""
    result = SampleResult(
        gsm_id="GSM3308545",
        gse_id="GSE115978",
        organism="Homo sapiens",
        status="success",
        error="",
        total_time_s=100.0,
    )
    
    assert result.gsm_id == "GSM3308545"
    assert result.status == "success"
    assert result.total_time_s == 100.0


def test_sample_result_to_dict():
    """Test SampleResult.to_dict() method."""
    result = SampleResult(
        gsm_id="GSM123",
        gse_id="GSE456",
        organism="Homo sapiens",
        status="failed",
        error="test error",
    )
    d = result.to_dict()
    assert d["gsm_id"] == "GSM123"
    assert d["status"] == "failed"


def test_qc_metrics():
    """Test QC metrics dataclass."""
    qc = QCMetrics(
        n_cells=1500,
        median_counts_per_cell=5000,
        median_genes_per_cell=2500,
        mapping_rate=0.85,
        pass_qc=True,
    )
    
    assert qc.n_cells == 1500
    assert qc.mapping_rate == 0.85
    assert qc.pass_qc is True


def test_infer_protocol_10xv3():
    """Test protocol inference for 10x Chromium v3."""
    detection = infer_protocol(r1_len=28, r2_len=91)
    assert detection.protocol == "10xv3"
    assert detection.mode == "droplet"


def test_infer_protocol_with_catalog_hint():
    """Test that catalog hint overrides heuristics."""
    # R1=25bp would be ambiguous, but hint resolves it
    detection = infer_protocol(r1_len=25, r2_len=91, catalog_hint="10xv3")
    assert detection.protocol == "10xv3"
    assert detection.confidence == "high"


def test_infer_protocol_dropseq():
    """Test protocol inference for Drop-seq."""
    detection = infer_protocol(r1_len=20, r2_len=60)
    assert detection.protocol == "dropseq"


def test_infer_protocol_bd_rhapsody():
    """Test protocol inference for BD Rhapsody."""
    detection = infer_protocol(r1_len=75, r2_len=75, catalog_hint="bd_rhapsody")
    assert detection.protocol == "bd_rhapsody"
    assert detection.confidence == "high"


# ── Protocol chemistry audit tests ──

def test_surecell_chemistry_corrected():
    """SureCell has 3×6bp barcodes with 15bp spacers + 3bp ACG + 8bp UMI."""
    from scgeo.config.protocols import get_chemistry
    chem = get_chemistry("surecell")
    assert chem == "1{b[6]x[15]b[6]x[15]b[6]x[3]u[8]x:}2{r:}"
    # Must NOT be the old celseq2-like value
    assert chem != "1{b[6]u[6]x:}2{r:}"
    assert chem != "1{b[8]x[6]b[8]u[6]x:}2{r:}"


def test_ddseq_chemistry_corrected():
    """ddSEQ has UMI(8) + 3×7bp barcodes with 10bp linkers (PB=0 geometry)."""
    from scgeo.config.protocols import get_chemistry
    chem = get_chemistry("ddseq")
    assert chem == "1{u[8]b[7]x[10]b[7]x[10]b[7]x:}2{r:}"


def test_scirna_chemistry_has_trailing_skip():
    """sci-RNA-seq3 FGDL must end with x: on R1 to handle variable-length reads."""
    from scgeo.config.protocols import get_chemistry
    chem = get_chemistry("scirna")
    assert "x:}" in chem  # trailing skip present
    assert chem == "1{b[10]x[6]u[8]b[10]x:}2{r:}"


def test_droplet_hint_single_end_falls_through():
    """Droplet catalog hint with R2=0 should NOT return a droplet detection."""
    detection = infer_protocol(r1_len=68, r2_len=0, catalog_hint="surecell")
    # Should fall through to heuristics and get classified as smartseq2 (single-end)
    assert detection.mode == "smartseq"
    assert detection.protocol == "smartseq2"


def test_10x_hint_single_end_falls_through():
    """10x catalog hint with R2=0 should fall to smartseq2."""
    detection = infer_protocol(r1_len=58, r2_len=0, catalog_hint="10xv3")
    assert detection.mode == "smartseq"
    assert detection.protocol == "smartseq2"


def test_10x_short_r2_still_detects():
    """10x with short R2 (< k-mer size) still detects as 10x but warns."""
    detection = infer_protocol(r1_len=28, r2_len=25, catalog_hint="10xv3")
    # Should still detect as 10xv3 (chemistry works, just mapping will be poor)
    assert detection.protocol == "10xv3"
    assert detection.mode == "droplet"


def test_bd_rhapsody_chemistry():
    """BD Rhapsody V1: 3×9bp barcodes with 12bp + 13bp linkers + 8bp UMI."""
    from scgeo.config.protocols import get_chemistry
    chem = get_chemistry("bd_rhapsody")
    assert chem == "1{b[9]x[12]b[9]x[13]b[9]u[8]x:}2{r:}"


def test_splitseq_chemistry():
    """SPLiT-seq: barcodes on R2, cDNA on R1."""
    from scgeo.config.protocols import get_chemistry
    chem = get_chemistry("splitseq")
    assert chem == "1{r:}2{u[10]b[8]x[30]b[8]x[30]b[8]x:}"
    assert chem.startswith("1{r:}")  # R1 is cDNA


def test_dropseq_seqwell_same_chemistry():
    """Drop-seq and Seq-Well share the same library structure."""
    from scgeo.config.protocols import get_chemistry
    assert get_chemistry("dropseq") == get_chemistry("seqwell")
    assert get_chemistry("dropseq") == "1{b[12]u[8]x:}2{r:}"


# ── FASTQ peek tests ──

def _make_gzip_fastq(reads: list[tuple[str, str]], compress: bool = True) -> bytes:
    """Build a gzipped FASTQ blob from a list of (header, sequence) tuples."""
    import gzip
    import io
    lines = []
    for i, (header, seq) in enumerate(reads):
        lines.append(f"@{header}")
        lines.append(seq)
        lines.append("+")
        lines.append("F" * len(seq))
    text = "\n".join(lines) + "\n"
    if compress:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(text.encode("ascii"))
        return buf.getvalue()
    return text.encode("ascii")


def _mock_urlopen(compressed_data):
    """Return a mock for urllib.request.urlopen that yields *compressed_data*."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = compressed_data
    mock_resp.status = 206
    return MagicMock(return_value=mock_resp)


class TestPeekFastqReadLength:
    """Unit tests for peek_fastq_read_length."""

    def test_smartseq_long_reads(self):
        reads = [("r1", "A" * 150)] * 10
        data = _make_gzip_fastq(reads)
        with patch("scgeo.pipeline.detect.urlopen", _mock_urlopen(data)):
            result = peek_fastq_read_length("ftp://example.com/test_1.fastq.gz")
        assert result == 150

    def test_droplet_short_reads(self):
        reads = [("r1", "A" * 28)] * 10
        data = _make_gzip_fastq(reads)
        with patch("scgeo.pipeline.detect.urlopen", _mock_urlopen(data)):
            result = peek_fastq_read_length("ftp://example.com/test_1.fastq.gz")
        assert result == 28

    def test_mixed_lengths_returns_median(self):
        reads = [("r1", "A" * 100)] * 5 + [("r2", "A" * 150)] * 5
        data = _make_gzip_fastq(reads)
        with patch("scgeo.pipeline.detect.urlopen", _mock_urlopen(data)):
            result = peek_fastq_read_length("ftp://example.com/test_1.fastq.gz")
        assert result in (100, 150)  # median of sorted list

    def test_network_failure_returns_none(self):
        with patch("scgeo.pipeline.detect.urlopen", side_effect=Exception("timeout")):
            result = peek_fastq_read_length("ftp://example.com/test_1.fastq.gz")
        assert result is None

    def test_ftp_to_http_conversion(self):
        reads = [("r1", "A" * 90)] * 5
        data = _make_gzip_fastq(reads)
        mock_open = _mock_urlopen(data)
        with patch("scgeo.pipeline.detect.urlopen", mock_open):
            peek_fastq_read_length("ftp://ftp.sra.ebi.ac.uk/vol1/test.fastq.gz")
        # Verify the Request URL was converted from ftp:// to http://
        call_args = mock_open.call_args
        req = call_args[0][0]
        assert req.full_url.startswith("http://")


class TestPeekProtocol:
    """Unit tests for peek_protocol (pre-download classification)."""

    def _patch_peek(self, r1_len, r2_len=None):
        """Patch peek_fastq_read_length to return fixed lengths."""
        def side_effect(url, **kwargs):
            if "_1.fastq" in url or url.endswith("R1.fastq.gz"):
                return r1_len
            elif "_2.fastq" in url or url.endswith("R2.fastq.gz"):
                return r2_len
            return r1_len  # default
        return patch("scgeo.pipeline.detect.peek_fastq_read_length", side_effect=side_effect)

    def test_both_reads_long_returns_smartseq(self):
        """R1=150bp, R2=150bp → smartseq → skip."""
        with self._patch_peek(150, 150):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                "ftp://example.com/SRR_2.fastq.gz",
            )
        assert det is not None
        assert det.mode == "smartseq"
        assert det.protocol == "smartseq2"
        assert det.confidence == "high"

    def test_short_r1_returns_none(self):
        """R1=28bp → likely droplet barcode → proceed (return None)."""
        with self._patch_peek(28, 91):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                "ftp://example.com/SRR_2.fastq.gz",
            )
        assert det is None

    def test_r1_long_r2_short_returns_none(self):
        """R1=150bp, R2=28bp → swapped droplet → proceed (return None)."""
        with self._patch_peek(150, 28):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                "ftp://example.com/SRR_2.fastq.gz",
            )
        assert det is None

    def test_single_end_long_returns_smartseq(self):
        """R1=150bp, no R2 → single-end smartseq → skip."""
        with self._patch_peek(150):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                None,
            )
        assert det is not None
        assert det.mode == "smartseq"

    def test_no_r1_url_returns_none(self):
        """No R1 URL → can't peek → proceed."""
        det = peek_protocol(None, None)
        assert det is None

    def test_r1_peek_failure_returns_none(self):
        """R1 peek fails → proceed (conservative)."""
        with patch("scgeo.pipeline.detect.peek_fastq_read_length", return_value=None):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                "ftp://example.com/SRR_2.fastq.gz",
            )
        assert det is None

    def test_r2_peek_failure_returns_none(self):
        """R1 long but R2 peek fails → can't confirm → proceed (conservative)."""
        def side_effect(url, **kwargs):
            if "_1.fastq" in url:
                return 150
            return None  # R2 fails
        with patch("scgeo.pipeline.detect.peek_fastq_read_length", side_effect=side_effect):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                "ftp://example.com/SRR_2.fastq.gz",
            )
        assert det is None

    def test_boundary_50bp_returns_none(self):
        """R1=50bp is at the boundary → should NOT classify as smartseq."""
        with self._patch_peek(50, 50):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                "ftp://example.com/SRR_2.fastq.gz",
            )
        assert det is None

    def test_boundary_51bp_returns_smartseq(self):
        """R1=51bp, R2=51bp → just above boundary → smartseq."""
        with self._patch_peek(51, 51):
            det = peek_protocol(
                "ftp://example.com/SRR_1.fastq.gz",
                "ftp://example.com/SRR_2.fastq.gz",
            )
        assert det is not None
        assert det.mode == "smartseq"


# --- Tests merged from test_v7_fixes.py ---


def test_ena_url_construction():
    """Test ENA URL construction for various SRR accession lengths."""
    # 9-char accession (no subdirectory)
    r1, r2 = _construct_ena_urls("SRR123456")
    assert r1 == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/SRR123456/SRR123456_1.fastq.gz"
    assert r2 == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/SRR123456/SRR123456_2.fastq.gz"

    # 10-char accession (1-digit suffix, zero-padded)
    r1, _ = _construct_ena_urls("SRR1234567")
    assert "/007/" in r1

    # 11-char accession (2-digit suffix, zero-padded)
    r1, _ = _construct_ena_urls("SRR12345678")
    assert "/078/" in r1

    # Known real URLs from batch CSV
    r1, _ = _construct_ena_urls("SRR14747538")
    assert r1 == "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR147/038/SRR14747538/SRR14747538_1.fastq.gz"
    r1, _ = _construct_ena_urls("SRR14747710")
    assert "/010/" in r1
    r1, _ = _construct_ena_urls("SRR14747886")
    assert "/086/" in r1


def test_smartseq_chemistry_detection():
    """Test that smartseq protocols are correctly detected for early skip."""
    assert get_chemistry("smartseq2") == "smartseq"
    assert get_chemistry("smartseq3") == "smartseq"
    assert get_chemistry("plate_based") == "smartseq"
    assert get_chemistry("10xv3") == "10xv3"
    assert get_chemistry("unknown_sc") is None
    assert get_chemistry("dropseq") is not None
    assert get_chemistry("seqwell") is not None


def test_download_segments_default():
    """Test that default segments reduced to 4."""
    assert DownloadConfig().segments == 4
