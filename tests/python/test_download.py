"""Tests for singlet.preprocessing._download (FASTQ download logic)."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from singlet.preprocessing._download import (
    DownloadResult,
    _convert_ftp_to_https,
    download_fastq,
    download_from_ena,
    download_from_sra,
)

# ---------------------------------------------------------------------------
# DownloadResult dataclass
# ---------------------------------------------------------------------------


class TestDownloadResult:
    def test_defaults(self):
        r = DownloadResult()
        assert r.success is False
        assert r.r1_paths == []
        assert r.r2_paths == []
        assert r.method == ""
        assert r.error == ""

    def test_error_state(self):
        r = DownloadResult(error="network timeout")
        assert not r.success
        assert "network" in r.error


# ---------------------------------------------------------------------------
# _convert_ftp_to_https
# ---------------------------------------------------------------------------


class TestConvertFtpToHttps:
    def test_ftp_prefix(self):
        url = "ftp://ftp.ebi.ac.uk/vol1/fastq/SRR123/SRR123456_1.fastq.gz"
        result = _convert_ftp_to_https(url)
        assert result.startswith("https://")
        assert "ftp.ebi.ac.uk" in result

    def test_https_unchanged(self):
        url = "https://ftp.ebi.ac.uk/vol1/fastq/SRR123/file.fastq.gz"
        assert _convert_ftp_to_https(url) == url

    def test_http_unchanged(self):
        url = "http://example.com/file.gz"
        assert _convert_ftp_to_https(url) == url

    def test_only_first_ftp_replaced(self):
        url = "ftp://ftp.sra.ebi.ac.uk/ftp/path"
        result = _convert_ftp_to_https(url)
        assert result == "https://ftp.sra.ebi.ac.uk/ftp/path"


# ---------------------------------------------------------------------------
# download_from_ena (mocked subprocess)
# ---------------------------------------------------------------------------


class TestDownloadFromEna:
    @patch("singlet.preprocessing._download._download_parallel_segments")
    def test_success_r1_only(self, mock_dl, tmp_path):
        mock_dl.return_value = (True, None)
        result = download_from_ena(
            "https://ftp.ebi.ac.uk/R1.fq.gz",
            None,
            tmp_path,
            "GSM123",
        )
        assert result.success
        assert result.method == "ena_direct"
        assert len(result.r1_paths) == 1

    @patch("singlet.preprocessing._download._download_parallel_segments")
    def test_success_r1_r2(self, mock_dl, tmp_path):
        mock_dl.return_value = (True, None)
        result = download_from_ena(
            "https://ftp.ebi.ac.uk/R1.fq.gz",
            "https://ftp.ebi.ac.uk/R2.fq.gz",
            tmp_path,
            "GSM123",
        )
        assert result.success
        assert len(result.r2_paths) == 1

    @patch("singlet.preprocessing._download._download_parallel_segments")
    def test_r1_failure(self, mock_dl, tmp_path):
        mock_dl.return_value = (False, "connection refused")
        result = download_from_ena(
            "https://ftp.ebi.ac.uk/R1.fq.gz",
            None,
            tmp_path,
            "GSM123",
        )
        assert not result.success
        assert "R1 download failed" in result.error

    @patch("singlet.preprocessing._download._download_parallel_segments")
    def test_r2_failure(self, mock_dl, tmp_path):
        """R1 succeeds but R2 fails."""
        mock_dl.side_effect = [(True, None), (False, "timeout")]
        result = download_from_ena(
            "https://ftp.ebi.ac.uk/R1.fq.gz",
            "ftp://ftp.ebi.ac.uk/R2.fq.gz",
            tmp_path,
            "GSM123",
        )
        assert not result.success
        assert "R2 download failed" in result.error

    def test_cache_hit(self, tmp_path):
        """If files already exist, returns immediately."""
        r1 = tmp_path / "GSM123_R1.fastq.gz"
        r1.write_bytes(b"fake_data")
        result = download_from_ena(
            "https://ftp.ebi.ac.uk/R1.fq.gz",
            None,
            tmp_path,
            "GSM123",
        )
        assert result.success
        assert result.method == "ena_cached"


# ---------------------------------------------------------------------------
# download_fastq (integration-level with mocked inner functions)
# ---------------------------------------------------------------------------


class TestDownloadFastq:
    @patch("singlet.preprocessing._download.download_from_ena")
    def test_ena_preferred(self, mock_ena, tmp_path):
        mock_ena.return_value = DownloadResult(
            success=True, r1_paths=[tmp_path / "R1.fq.gz"], method="ena_direct"
        )
        result = download_fastq(
            "GSM999",
            ena_r1_url="https://url/R1.fq.gz",
            output_dir=tmp_path,
        )
        assert result.success
        assert result.method == "ena_direct"
        mock_ena.assert_called_once()

    @patch("singlet.preprocessing._download.download_from_sra")
    @patch("singlet.preprocessing._download.download_from_ena")
    def test_fallback_to_sra(self, mock_ena, mock_sra, tmp_path):
        mock_ena.return_value = DownloadResult(error="ENA failed")
        mock_sra.return_value = DownloadResult(
            success=True, r1_paths=[tmp_path / "R1.fq.gz"], method="fasterq_dump"
        )
        result = download_fastq(
            "GSM999",
            ena_r1_url="https://url/R1.fq.gz",
            srr_accession="SRR1234",
            output_dir=tmp_path,
        )
        assert result.success
        assert result.method == "fasterq_dump"

    def test_no_urls_returns_error(self, tmp_path):
        result = download_fastq("GSM999", output_dir=tmp_path)
        assert not result.success
        assert "All download methods failed" in result.error

    @patch("singlet.preprocessing._download.download_from_ena")
    def test_prefer_ena_false(self, mock_ena, tmp_path):
        """When prefer_ena=False but no SRR, still falls back to ENA."""
        mock_ena.return_value = DownloadResult(
            success=True, r1_paths=[tmp_path / "R1.fq.gz"], method="ena_direct"
        )
        result = download_fastq(
            "GSM999",
            ena_r1_url="https://url/R1.fq.gz",
            output_dir=tmp_path,
            prefer_ena=False,
        )
        assert result.success


# ---------------------------------------------------------------------------
# download_from_sra (mocked subprocess)
# ---------------------------------------------------------------------------


class TestDownloadFromSra:
    @patch("singlet.preprocessing._download.subprocess.run")
    def test_success(self, mock_run, tmp_path):
        """Successful fasterq-dump + pigz compression."""

        # fasterq-dump "creates" fastq files
        def fake_run(cmd, **kwargs):
            if "fasterq-dump" in cmd:
                # Simulate output files
                (tmp_path / "SRR123_1.fastq").write_text("@read\nACGT\n+\nIIII\n")
                (tmp_path / "SRR123_2.fastq").write_text("@read\nACGT\n+\nIIII\n")
            elif "pigz" in cmd or "gzip" in cmd:
                # Simulate compression
                src = Path(cmd[-1])
                gz = src.with_suffix(".fastq.gz")
                gz.write_bytes(b"compressed")
                src.unlink()
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_run

        result = download_from_sra("SRR123", tmp_path, threads=2)
        assert result.success
        assert result.method == "fasterq_dump"

    @patch("singlet.preprocessing._download.subprocess.run")
    def test_fasterq_dump_failure(self, mock_run, tmp_path):
        """CalledProcessError returns error result."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "fasterq-dump")
        result = download_from_sra("SRR123", tmp_path)
        assert not result.success
        assert "fasterq-dump failed" in result.error

    @patch("singlet.preprocessing._download.subprocess.run")
    def test_fasterq_dump_not_found(self, mock_run, tmp_path):
        """FileNotFoundError returns error result."""
        mock_run.side_effect = FileNotFoundError("fasterq-dump not installed")
        result = download_from_sra("SRR123", tmp_path)
        assert not result.success
        assert "fasterq-dump failed" in result.error


# ---------------------------------------------------------------------------
# ENA cache with R2 file
# ---------------------------------------------------------------------------


class TestEnaCacheR2:
    def test_cache_hit_r1_and_r2(self, tmp_path):
        """Both R1 and R2 cached → returns both paths."""
        r1 = tmp_path / "GSM456_R1.fastq.gz"
        r2 = tmp_path / "GSM456_R2.fastq.gz"
        r1.write_bytes(b"r1_data")
        r2.write_bytes(b"r2_data")

        result = download_from_ena(
            "https://ftp.ebi.ac.uk/R1.fq.gz",
            "https://ftp.ebi.ac.uk/R2.fq.gz",
            tmp_path,
            "GSM456",
        )
        assert result.success
        assert result.method == "ena_cached"
        assert len(result.r1_paths) == 1
        assert len(result.r2_paths) == 1
