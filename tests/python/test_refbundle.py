# SPDX-License-Identifier: MIT
"""Tests for the reference bundle format (``features.fbin``, ``snp_sites.fbin``).

Covers:

* Round-trip of the binary writers and readers.
* Sentinel handling (absent symbols → empty string).
* Chrom-table indirection.
* End-to-end builder CLIs operating on synthetic GTF / VCF inputs.
"""

from __future__ import annotations

import gzip
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from singlet.refbundle import (
    EXON_REC_SIZE,
    FEATURES_HEADER_SIZE,
    GENE_REC_SIZE,
    INTRON_REC_SIZE,
    JUNCTION_REC_SIZE,
    SNP_HEADER_SIZE,
    SNP_REC_SIZE,
    FeaturesBundle,
    SnpPanel,
    SnpSite,
    load_features,
    load_snp_panel,
    write_snp_panel,
)
from singlet.refbundle._features import _GeneIn, write_features


# --------------------------------------------------------------------------
# Format constants
# --------------------------------------------------------------------------


class TestFormatConstants:
    def test_record_sizes(self):
        assert GENE_REC_SIZE == 44
        assert EXON_REC_SIZE == 16
        assert INTRON_REC_SIZE == 24
        assert JUNCTION_REC_SIZE == 24
        assert SNP_REC_SIZE == 16
        assert FEATURES_HEADER_SIZE == 256
        assert SNP_HEADER_SIZE == 128


# --------------------------------------------------------------------------
# features.fbin round-trip
# --------------------------------------------------------------------------


def _two_gene_fixture():
    return [
        _GeneIn(
            name="ENSG00000111111",
            symbol="CD3D",
            chrom="chr1",
            strand=0,
            biotype=1,
            tx_start=1000,
            tx_end=2000,
            exons=[(1000, 1200), (1500, 2000)],
            introns=[(1201, 1499, 0, 1)],
            junctions=[(1200, 1500, 0, 1, 0, 0)],
        ),
        _GeneIn(
            name="ENSG00000222222",
            symbol="",  # symbol absent
            chrom="chrX",
            strand=1,
            biotype=2,
            tx_start=500,
            tx_end=900,
            exons=[(500, 900)],
            introns=[],
            junctions=[],
        ),
    ]


class TestFeaturesRoundTrip:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "features.fbin"
        write_features(
            path,
            build_id="GRCh38-2024-A",
            gtf_sha256="abc" * 10 + "ab",
            genes=_two_gene_fixture(),
        )

        with load_features(path) as fb:
            assert fb.build_id == "GRCh38-2024-A"
            assert fb.n_genes == 2
            assert fb.n_exons == 3
            assert fb.n_introns == 1
            assert fb.n_junctions == 1
            # Sorted by (chrom, tx_start, name): chr1 first.
            g0 = fb.gene(0)
            assert g0.name == "ENSG00000111111"
            assert g0.symbol == "CD3D"
            assert g0.chrom == "chr1"
            assert g0.strand == 0
            assert g0.biotype == 1
            assert g0.tx_start == 1000
            assert g0.tx_end == 2000
            assert g0.exon_hi - g0.exon_lo == 2

            g1 = fb.gene(1)
            assert g1.name == "ENSG00000222222"
            assert g1.symbol == ""  # absent sentinel preserved
            assert g1.chrom == "chrX"
            assert g1.strand == 1

            assert fb.gene_names() == [
                "ENSG00000111111",
                "ENSG00000222222",
            ]

    def test_exon_decoding(self, tmp_path):
        path = tmp_path / "features.fbin"
        write_features(
            path,
            build_id="X",
            gtf_sha256="0" * 64,
            genes=_two_gene_fixture(),
        )
        with load_features(path) as fb:
            g0 = fb.gene(0)
            exons = [fb.exon(i) for i in range(g0.exon_lo, g0.exon_hi)]
            # (gene_id, chrom_id, start, end)
            assert exons[0][0] == 0
            assert exons[0][2] == 1000 and exons[0][3] == 1200
            assert exons[1][2] == 1500 and exons[1][3] == 2000

    def test_junction_decoding(self, tmp_path):
        path = tmp_path / "features.fbin"
        write_features(
            path,
            build_id="X",
            gtf_sha256="0" * 64,
            genes=_two_gene_fixture(),
        )
        with load_features(path) as fb:
            gene_id, df, af, flags, motif, dp, ap = fb.junction(0)
            assert gene_id == 0
            assert dp == 1200 and ap == 1500

    def test_bad_magic(self, tmp_path):
        path = tmp_path / "bad.fbin"
        path.write_bytes(b"BADMAGIC" + b"\0" * 4096)
        with pytest.raises(ValueError, match="bad magic"):
            load_features(path)

    def test_context_manager_closes(self, tmp_path):
        path = tmp_path / "features.fbin"
        write_features(
            path,
            build_id="X",
            gtf_sha256="0" * 64,
            genes=_two_gene_fixture(),
        )
        with load_features(path) as fb:
            assert fb.n_genes == 2
        # mmap closed; further access would raise. Just ensure no exception.

    def test_chrom_overflow(self, tmp_path):
        path = tmp_path / "f.fbin"
        too_many = [
            _GeneIn(
                name=f"G{i}",
                symbol="",
                chrom=f"chr{i}",
                strand=0,
                biotype=0,
                tx_start=1,
                tx_end=2,
                exons=[(1, 2)],
                introns=[],
                junctions=[],
            )
            for i in range(257)
        ]
        with pytest.raises(ValueError, match="too many chroms"):
            write_features(path, build_id="X", gtf_sha256="0", genes=too_many)


# --------------------------------------------------------------------------
# snp_sites.fbin round-trip
# --------------------------------------------------------------------------


def _snp_fixture():
    return [
        SnpSite(chrom="chr1", pos=12345, ref="A", alt="G", af_pop=0.21, rsid=1234567),
        SnpSite(chrom="chr1", pos=67890, ref="C", alt="T", af_pop=0.49, rsid=0),
        SnpSite(chrom="chrX", pos=11111, ref="G", alt="A", af_pop=0.10, rsid=9999),
    ]


class TestSnpRoundTrip:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "snps.fbin"
        write_snp_panel(
            path,
            build_id="GRCh38-2024-A",
            panel_id="1KGP_AF5e2",
            sites=_snp_fixture(),
        )

        with load_snp_panel(path) as sp:
            assert sp.build_id == "GRCh38-2024-A"
            assert sp.panel_id == "1KGP_AF5e2"
            assert sp.n_sites == 3
            assert sp.n_chroms == 2
            assert sp.chroms == ["chr1", "chrX"]
            s0 = sp.site(0)
            assert (s0.chrom, s0.pos, s0.ref, s0.alt) == ("chr1", 12345, "A", "G")
            assert abs(s0.af_pop - 0.21) < 1e-6
            assert s0.rsid == 1234567
            s2 = sp.site(2)
            assert s2.chrom == "chrX" and s2.pos == 11111

    def test_len(self, tmp_path):
        path = tmp_path / "snps.fbin"
        write_snp_panel(
            path, build_id="X", panel_id="P", sites=_snp_fixture()
        )
        with load_snp_panel(path) as sp:
            assert len(sp) == 3
            assert list(sp.iter_sites())[1].pos == 67890

    def test_bad_magic(self, tmp_path):
        path = tmp_path / "bad.fbin"
        path.write_bytes(b"NOTSNP\0\0" + b"\0" * 4096)
        with pytest.raises(ValueError, match="bad magic"):
            load_snp_panel(path)


# --------------------------------------------------------------------------
# Builder CLI: features
# --------------------------------------------------------------------------


_SYNTHETIC_GTF = """\
##GTF
chr1\tHAVANA\tgene\t1000\t2000\t.\t+\t.\tgene_id "ENSG0001"; gene_name "AAA"; gene_biotype "protein_coding";
chr1\tHAVANA\texon\t1000\t1200\t.\t+\t.\tgene_id "ENSG0001"; transcript_id "T1";
chr1\tHAVANA\texon\t1500\t2000\t.\t+\t.\tgene_id "ENSG0001"; transcript_id "T1";
chrX\tHAVANA\tgene\t100\t900\t.\t-\t.\tgene_id "ENSG0002"; gene_name "BBB"; gene_biotype "lncRNA";
chrX\tHAVANA\texon\t100\t900\t.\t-\t.\tgene_id "ENSG0002"; transcript_id "T2";
"""

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


class TestFeaturesBuilderCLI:
    def test_end_to_end(self, tmp_path):
        gtf = tmp_path / "genes.gtf"
        gtf.write_text(_SYNTHETIC_GTF)
        out = tmp_path / "features.fbin"
        cp = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "build_features_fbin.py"),
                "--gtf",
                str(gtf),
                "--build-id",
                "TEST-1",
                "--out",
                str(out),
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr
        with load_features(out) as fb:
            assert fb.n_genes == 2
            names = sorted(fb.gene_names())
            assert names == ["ENSG0001", "ENSG0002"]
            # Gene with two exons → one intron + one junction.
            for g in fb.iter_genes():
                if g.name == "ENSG0001":
                    assert g.symbol == "AAA"
                    assert g.exon_hi - g.exon_lo == 2
                    assert g.intron_hi - g.intron_lo == 1
                    assert g.junction_hi - g.junction_lo == 1
                if g.name == "ENSG0002":
                    assert g.symbol == "BBB"
                    assert g.biotype == 2  # lncRNA
                    assert g.strand == 1
                    assert g.exon_hi - g.exon_lo == 1

    def test_gz_input(self, tmp_path):
        gtf = tmp_path / "genes.gtf.gz"
        with gzip.open(gtf, "wb") as f:
            f.write(_SYNTHETIC_GTF.encode("utf-8"))
        out = tmp_path / "features.fbin"
        cp = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "build_features_fbin.py"),
                "--gtf",
                str(gtf),
                "--build-id",
                "TEST-1",
                "--out",
                str(out),
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr
        assert out.is_file()


# --------------------------------------------------------------------------
# Builder CLI: SNPs
# --------------------------------------------------------------------------


_SYNTHETIC_VCF = """\
##fileformat=VCFv4.2
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t100\trs111\tA\tG\t.\tPASS\tAF=0.20
chr1\t200\t.\tC\tT\t.\tPASS\tAF=0.05
chr1\t300\trs333\tA\tAT\t.\tPASS\tAF=0.10
chr1\t400\trs444\tA\tG,C\t.\tPASS\tAF=0.30,0.10
chrX\t1000\trs555\tG\tA\t.\tPASS\tAF=0.40
"""


class TestSnpBuilderCLI:
    def test_end_to_end(self, tmp_path):
        vcf = tmp_path / "snps.vcf"
        vcf.write_text(_SYNTHETIC_VCF)
        out = tmp_path / "snp_sites.fbin"
        cp = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "build_snp_sites_fbin.py"),
                "--vcf",
                str(vcf),
                "--build-id",
                "TEST-1",
                "--panel-id",
                "MINI",
                "--out",
                str(out),
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr
        with load_snp_panel(out) as sp:
            assert sp.panel_id == "MINI"
            # 3 biallelic SNPs kept: 100 (A>G), 200 (C>T), 1000 (G>A).
            # Dropped: 300 (A>AT indel), 400 (multi-allelic A>G,C).
            assert sp.n_sites == 3
            chroms = [sp.site(i).chrom for i in range(sp.n_sites)]
            assert chroms.count("chr1") == 2
            assert chroms.count("chrX") == 1
            assert sp.site(0).rsid == 111

    def test_min_af_filter(self, tmp_path):
        vcf = tmp_path / "snps.vcf"
        vcf.write_text(_SYNTHETIC_VCF)
        out = tmp_path / "snp_sites.fbin"
        cp = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "build_snp_sites_fbin.py"),
                "--vcf",
                str(vcf),
                "--build-id",
                "X",
                "--panel-id",
                "M",
                "--min-af",
                "0.15",
                "--out",
                str(out),
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr
        with load_snp_panel(out) as sp:
            # AF >= 0.15 keeps 100 (0.20) and 1000 (0.40); drops 200 (0.05).
            assert sp.n_sites == 2
            assert all(sp.site(i).af_pop >= 0.15 for i in range(sp.n_sites))
