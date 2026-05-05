"""Unit tests for kraken2 lockstep R1/R2 reading."""
import gzip
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scgeo.pipeline.kraken2 import (
    _parse_barcode_umi_lens,
    _r1_sequence_reader,
    classify_nonhost,
    Kraken2Result,
)


# ---------------------------------------------------------------------------
# _parse_barcode_umi_lens
# ---------------------------------------------------------------------------

class TestParseBarcodeUmiLens:
    def test_10xv2(self):
        assert _parse_barcode_umi_lens("10xv2") == (16, 10)

    def test_10xv3(self):
        assert _parse_barcode_umi_lens("10xv3") == (16, 12)

    def test_custom_geometry(self):
        assert _parse_barcode_umi_lens("1{b[12]u[8]x:}2{r:}") == (12, 8)

    def test_multi_barcode(self):
        # SPLiT-seq style: three 8bp barcodes
        assert _parse_barcode_umi_lens("1{b[8]b[8]b[8]u[10]x:}2{r:}") == (24, 10)

    def test_fallback_default(self):
        bc, umi = _parse_barcode_umi_lens("unknown_chemistry")
        assert (bc, umi) == (16, 12)


# ---------------------------------------------------------------------------
# _r1_sequence_reader
# ---------------------------------------------------------------------------

def _write_fastq_gz(path, records):
    """Write a gzipped FASTQ file from list of (header, seq, qual) tuples."""
    with gzip.open(path, "wt") as f:
        for header, seq, qual in records:
            f.write(f"@{header}\n{seq}\n+\n{qual}\n")


def _write_fastq(path, records):
    """Write a plain FASTQ file from list of (header, seq, qual) tuples."""
    with open(path, "w") as f:
        for header, seq, qual in records:
            f.write(f"@{header}\n{seq}\n+\n{qual}\n")


class TestR1SequenceReader:
    def test_single_gz_file(self, tmp_path):
        records = [
            ("read1", "ACGTACGTACGTACGT" + "AACCGGTTAABB", "I" * 28),
            ("read2", "TTTTTTTTTTTTTTTTT" + "CCCCDDDDEEEE", "I" * 29),
        ]
        fq = tmp_path / "r1.fastq.gz"
        _write_fastq_gz(fq, records)

        seqs = list(_r1_sequence_reader([fq]))
        assert len(seqs) == 2
        assert seqs[0] == "ACGTACGTACGTACGTAACCGGTTAABB"
        assert seqs[1] == "TTTTTTTTTTTTTTTTTCCCCDDDDEEEE"

    def test_multiple_files(self, tmp_path):
        r1 = [("r1", "AAAA", "IIII")]
        r2 = [("r2", "CCCC", "IIII"), ("r3", "GGGG", "IIII")]
        f1 = tmp_path / "r1a.fastq.gz"
        f2 = tmp_path / "r1b.fastq.gz"
        _write_fastq_gz(f1, r1)
        _write_fastq_gz(f2, r2)

        seqs = list(_r1_sequence_reader([f1, f2]))
        assert seqs == ["AAAA", "CCCC", "GGGG"]

    def test_plain_fastq(self, tmp_path):
        records = [("read1", "ATCG", "IIII")]
        fq = tmp_path / "r1.fastq"
        _write_fastq(fq, records)

        seqs = list(_r1_sequence_reader([fq]))
        assert seqs == ["ATCG"]

    def test_empty_file(self, tmp_path):
        fq = tmp_path / "empty.fastq.gz"
        with gzip.open(fq, "wt") as f:
            pass
        assert list(_r1_sequence_reader([fq])) == []


# ---------------------------------------------------------------------------
# classify_nonhost — lockstep integration test with mocked kraken2
# ---------------------------------------------------------------------------

class TestClassifyNonhostLockstep:
    """Test the lockstep R1/R2 reading via mocked kraken2 subprocess."""

    def _make_config(self, tmp_path):
        """Create a minimal config mock."""
        cfg = MagicMock()
        cfg.kraken2.enabled = True
        cfg.kraken2.db = str(tmp_path / "db")
        (tmp_path / "db").mkdir()
        cfg.kraken2.threads = 1
        cfg.kraken2.confidence = 0.0
        cfg.kraken2.timeout = 60
        cfg.kraken2.min_nonhost_umis = 1
        return cfg

    def _make_quant_dir(self, tmp_path, barcodes):
        """Create mock alevin-fry output with barcodes."""
        quant = tmp_path / "quant"
        bc_dir = quant / "af_quant" / "alevin"
        bc_dir.mkdir(parents=True)
        (bc_dir / "quants_mat_rows.txt").write_text("\n".join(barcodes) + "\n")
        return quant

    def _make_fastq_pair(self, tmp_path, reads_r1, reads_r2, prefix=""):
        """Create paired R1/R2 FASTQ files.
        
        reads_r1: list of sequences
        reads_r2: list of sequences (same length)
        """
        r1_path = tmp_path / f"{prefix}r1.fastq.gz"
        r2_path = tmp_path / f"{prefix}r2.fastq.gz"

        r1_records = [(f"read{i}", seq, "I" * len(seq)) for i, seq in enumerate(reads_r1)]
        r2_records = [(f"read{i}", seq, "I" * len(seq)) for i, seq in enumerate(reads_r2)]

        _write_fastq_gz(r1_path, r1_records)
        _write_fastq_gz(r2_path, r2_records)
        return r1_path, r2_path

    @patch("scgeo.pipeline.kraken2.subprocess.Popen")
    @patch("scgeo.pipeline.kraken2._get_kraken2_version", return_value="2.1.3")
    def test_lockstep_basic(self, mock_ver, mock_popen, tmp_path):
        """Verify lockstep correctly associates R1 barcodes with kraken2 classifications."""
        # 10xv3: 16bp barcode + 12bp UMI = 28bp R1
        bc1 = "ACGTACGTACGTACGT"  # valid barcode
        bc2 = "TTTTTTTTTTTTTTTG"  # valid barcode
        bc3 = "GGGGGGGGGGGGGGGG"  # NOT in valid set
        umi1 = "AACCGGTTAABB"
        umi2 = "CCCCDDDDEEEE"
        umi3 = "FFFFHHHHIIII"

        r1_seqs = [bc1 + umi1, bc2 + umi2, bc3 + umi3, bc1 + "ZZZZZZZZZZZZ"]
        r2_seqs = ["ATCG" * 25] * 4  # cDNA content doesn't matter for this test

        r1_path, r2_path = self._make_fastq_pair(tmp_path, r1_seqs, r2_seqs)
        quant_dir = self._make_quant_dir(tmp_path, [bc1, bc2])
        config = self._make_config(tmp_path)
        output_dir = tmp_path / "kraken_out"

        # Simulate kraken2 stdout: 4 reads
        # Read 0: classified, non-host (taxon 1234)
        # Read 1: unclassified
        # Read 2: classified, non-host (taxon 5678) — but bc3 not valid
        # Read 3: classified, HOST (taxon 9606)
        kraken_stdout = [
            "C\tread0\t1234\t100\t1234:100\n",
            "U\tread1\t0\t100\t0:100\n",
            "C\tread2\t5678\t100\t5678:100\n",
            "C\tread3\t9606\t100\t9606:100\n",
        ]

        # Mock the subprocess
        mock_proc = MagicMock()
        mock_proc.stdout = iter(kraken_stdout)
        mock_proc.communicate.return_value = ("", "4 sequences classified")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        # Write a dummy report
        (tmp_path / "kraken_out").mkdir(parents=True, exist_ok=True)
        report = output_dir / "kraken2_report.txt"
        report.write_text(
            "50.00\t2\t1\tS\t1234\tEscherichia coli\n"
            "25.00\t1\t1\tS\t5678\tStaphylococcus aureus\n"
        )

        result = classify_nonhost(
            r1_paths=[r1_path],
            r2_paths=[r2_path],
            chemistry="10xv3",
            host_taxon_id=9606,
            quant_dir=quant_dir,
            output_dir=output_dir,
            config=config,
        )

        assert result.success
        assert result.total_reads == 4
        assert result.classified_reads == 3  # reads 0, 2, 3
        assert result.nonhost_reads == 2  # reads 0, 2 (read 3 is host)
        # Only bc1 should have a UMI (bc3 from read 2 is not valid)
        assert result.cells_with_nonhost == 1
        assert result.total_nonhost_umis == 1

        # Verify parquet was written
        parquet_path = output_dir / "kraken2_cell_taxa.parquet"
        assert parquet_path.exists()

        import pandas as pd
        df = pd.read_parquet(parquet_path)
        assert len(df) == 1
        assert df.iloc[0]["barcode"] == bc1
        assert df.iloc[0]["taxon_id"] == 1234
        assert df.iloc[0]["umi_count"] == 1

    @patch("scgeo.pipeline.kraken2.subprocess.Popen")
    @patch("scgeo.pipeline.kraken2._get_kraken2_version", return_value="2.1.3")
    def test_lockstep_no_nonhost(self, mock_ver, mock_popen, tmp_path):
        """All reads classified as host → empty results."""
        r1_seqs = ["ACGTACGTACGTACGT" + "AACCGGTTAABB"]
        r2_seqs = ["ATCG" * 25]
        r1_path, r2_path = self._make_fastq_pair(tmp_path, r1_seqs, r2_seqs)
        quant_dir = self._make_quant_dir(tmp_path, ["ACGTACGTACGTACGT"])
        config = self._make_config(tmp_path)
        output_dir = tmp_path / "kraken_out"

        kraken_stdout = ["C\tread0\t9606\t100\t9606:100\n"]
        mock_proc = MagicMock()
        mock_proc.stdout = iter(kraken_stdout)
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        result = classify_nonhost(
            r1_paths=[r1_path],
            r2_paths=[r2_path],
            chemistry="10xv3",
            host_taxon_id=9606,
            quant_dir=quant_dir,
            output_dir=output_dir,
            config=config,
        )

        assert result.success
        assert result.nonhost_reads == 0
        assert result.cells_with_nonhost == 0

    @patch("scgeo.pipeline.kraken2.subprocess.Popen")
    @patch("scgeo.pipeline.kraken2._get_kraken2_version", return_value="2.1.3")
    def test_lockstep_umi_dedup(self, mock_ver, mock_popen, tmp_path):
        """Same barcode + taxon + UMI should be deduplicated."""
        bc = "ACGTACGTACGTACGT"
        umi = "AACCGGTTAABB"

        # Two reads with identical barcode+UMI, both classified as same non-host taxon
        r1_seqs = [bc + umi, bc + umi]
        r2_seqs = ["ATCG" * 25] * 2
        r1_path, r2_path = self._make_fastq_pair(tmp_path, r1_seqs, r2_seqs)
        quant_dir = self._make_quant_dir(tmp_path, [bc])
        config = self._make_config(tmp_path)
        output_dir = tmp_path / "kraken_out"

        kraken_stdout = [
            "C\tread0\t1234\t100\t1234:100\n",
            "C\tread1\t1234\t100\t1234:100\n",
        ]
        mock_proc = MagicMock()
        mock_proc.stdout = iter(kraken_stdout)
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        (tmp_path / "kraken_out").mkdir(parents=True, exist_ok=True)
        (output_dir / "kraken2_report.txt").write_text("")

        result = classify_nonhost(
            r1_paths=[r1_path],
            r2_paths=[r2_path],
            chemistry="10xv3",
            host_taxon_id=9606,
            quant_dir=quant_dir,
            output_dir=output_dir,
            config=config,
        )

        assert result.success
        assert result.nonhost_reads == 2  # Both reads are non-host
        assert result.total_nonhost_umis == 1  # Deduplicated to 1 UMI

    @patch("scgeo.pipeline.kraken2.subprocess.Popen")
    @patch("scgeo.pipeline.kraken2._get_kraken2_version", return_value="2.1.3")
    def test_lockstep_multi_file(self, mock_ver, mock_popen, tmp_path):
        """Multiple R1/R2 file pairs processed correctly in lockstep order."""
        bc1 = "AAAAAAAAAAAAAAAA"
        bc2 = "CCCCCCCCCCCCCCCC"
        umi1 = "AACCGGTTAABB"
        umi2 = "TTTTTTTTTTTT"

        # File pair 1: 1 read
        r1a_path, r2a_path = self._make_fastq_pair(
            tmp_path, [bc1 + umi1], ["ATCG" * 25], prefix="a_"
        )
        # File pair 2: 1 read
        r1b_path, r2b_path = self._make_fastq_pair(
            tmp_path, [bc2 + umi2], ["ATCG" * 25], prefix="b_"
        )

        quant_dir = self._make_quant_dir(tmp_path, [bc1, bc2])
        config = self._make_config(tmp_path)
        output_dir = tmp_path / "kraken_out"

        # Kraken2 processes both files sequentially: read from file 1, then file 2
        kraken_stdout = [
            "C\tread0\t1234\t100\t1234:100\n",
            "C\tread1\t5678\t100\t5678:100\n",
        ]
        mock_proc = MagicMock()
        mock_proc.stdout = iter(kraken_stdout)
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        (tmp_path / "kraken_out").mkdir(parents=True, exist_ok=True)
        (output_dir / "kraken2_report.txt").write_text(
            "50.00\t1\t1\tS\t1234\tE coli\n50.00\t1\t1\tS\t5678\tS aureus\n"
        )

        result = classify_nonhost(
            r1_paths=[r1a_path, r1b_path],
            r2_paths=[r2a_path, r2b_path],
            chemistry="10xv3",
            host_taxon_id=9606,
            quant_dir=quant_dir,
            output_dir=output_dir,
            config=config,
        )

        assert result.success
        assert result.total_reads == 2
        assert result.nonhost_reads == 2
        assert result.cells_with_nonhost == 2

        import pandas as pd
        df = pd.read_parquet(output_dir / "kraken2_cell_taxa.parquet")
        assert set(df["barcode"]) == {bc1, bc2}
        assert set(df["taxon_id"]) == {1234, 5678}

    def test_disabled_kraken2(self, tmp_path):
        """Kraken2 disabled returns immediately."""
        cfg = MagicMock()
        cfg.kraken2.enabled = False
        result = classify_nonhost([], [], "10xv3", 9606, tmp_path, tmp_path, cfg)
        assert result.success
        assert result.time_s == 0
