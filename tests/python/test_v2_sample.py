# SPDX-License-Identifier: MIT
"""End-to-end tests for canonical v2 sample readers + derived views.

Builds a synthetic sample directory + features.fbin, writes counts.1pz /
snp.1pz / mt.1pz / nonhost_species.1pz via the pz_v2 codec, then checks
that ``SingletSample`` and the ``views.*`` functions return the expected
matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.sparse import csc_matrix

from singlet.io import SingletSample
from singlet.pz_v2 import BlockSpec, write_pz_v2
from singlet.refbundle import GeneRecord, write_features
from singlet.refbundle._features import _GeneIn, biotype_code
from singlet.views import gene_counts, psi, usa


# --------------------------------------------------------------------------
# Synthetic fixture builders
# --------------------------------------------------------------------------


def _build_features(path: Path) -> None:
    """Write a features.fbin with two genes.

    Gene 0: exons 0..2 (3 intervals), introns 0..1 (2), junctions 0..1 (2).
    Gene 1: exons 2..3 (1 interval), introns 1..2 (1), junctions 2..3 (1).

    Junction flags & 0x03:
        0 → EE, 1 → EI, 2 → IE, 3 → II
    """
    genes = [
        _GeneIn(
            name="ENSG0",
            symbol="G0",
            chrom="chr1",
            strand=ord("+"),
            biotype=biotype_code("protein_coding"),
            tx_start=100,
            tx_end=1000,
            exons=[(100, 200), (300, 400), (500, 600)],
            introns=[(200, 300, 0, 1), (400, 500, 1, 2)],
            junctions=[
                # EE between exon0/exon1; flags=0
                (200, 300, 0, 1, 0, 0),
                # EI between exon1/intron1; flags=1
                (400, 500, 1, 2, 1, 0),
            ],
        ),
        _GeneIn(
            name="ENSG1",
            symbol="G1",
            chrom="chr1",
            strand=ord("+"),
            biotype=biotype_code("protein_coding"),
            tx_start=1000,
            tx_end=2000,
            exons=[(1000, 1100)],
            introns=[(1100, 1200, 2, 2)],
            junctions=[
                # II for gene1; flags=3
                (1100, 1200, 2, 2, 3, 0),
            ],
        ),
    ]
    write_features(path, build_id="test_v1", gtf_sha256="0" * 64, genes=genes)


def _build_sample(tmp_path: Path) -> Path:
    sample = tmp_path / "sample0"
    sample.mkdir()

    bc = ["AAA", "CCC", "GGG"]
    n_cells = len(bc)

    # 4 exon intervals total (3 + 1)
    exon = csc_matrix(np.array([
        [1, 0, 2],
        [0, 3, 0],
        [4, 0, 0],
        [0, 0, 5],
    ], dtype=np.int32))
    # 3 intron intervals total (2 + 1)
    intron = csc_matrix(np.array([
        [0, 1, 0],
        [2, 0, 0],
        [0, 0, 3],
    ], dtype=np.int32))
    # 3 junction rows total (2 + 1)
    jct = csc_matrix(np.array([
        [1, 0, 0],   # EE, gene0
        [0, 2, 0],   # EI, gene0
        [0, 0, 4],   # II, gene1
    ], dtype=np.int32))

    write_pz_v2(
        sample / "counts.1pz",
        cell_barcodes=bc,
        blocks=[
            BlockSpec("exon_body", exon),
            BlockSpec("intron_body", intron),
            BlockSpec("junctions", jct),
        ],
    )

    # SNP: 2 sites × 3 cells, two-layer
    ad_arr = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32)
    dp_arr = np.array([[4, 0, 5], [0, 6, 0]], dtype=np.int32)
    ad = csc_matrix(ad_arr)
    dp = csc_matrix(dp_arr)
    write_pz_v2(
        sample / "snp.1pz",
        cell_barcodes=bc,
        blocks=[BlockSpec("snp", ad, data2=dp, data_names=["ad", "dp"])],
    )
    write_pz_v2(
        sample / "mt.1pz",
        cell_barcodes=bc,
        blocks=[BlockSpec("mt", ad, data2=dp, data_names=["ad", "dp"])],
    )

    # cell_meta.parquet
    table = pa.table({"barcode": bc, "n_umi": [10, 5, 14]})
    pq.write_table(table, sample / "cell_meta.parquet")

    # summary.json
    (sample / "summary.json").write_text(json.dumps({"sample_id": "sample0", "n_cells": n_cells}))

    # nonhost
    (sample / "nonhost.json").write_text(json.dumps({
        "kraken_total_reads": 1234,
        "species": [
            {"row": 0, "taxid": 9606, "name": "Homo sapiens", "rank": "species",
             "kraken_reads": 100, "kraken_kmer_hits": 200,
             "bracken_reads": 90, "bracken_abundance": 0.5, "lineage": "..."},
        ],
    }))
    nonhost_mat = csc_matrix(np.array([[1, 2, 3]], dtype=np.int32))
    write_pz_v2(
        sample / "nonhost_species.1pz",
        cell_barcodes=bc,
        blocks=[BlockSpec("species", nonhost_mat)],
    )

    return sample


@pytest.fixture
def sample_dir(tmp_path):
    _build_features(tmp_path / "features.fbin")
    return _build_sample(tmp_path)


@pytest.fixture
def features_path(tmp_path, sample_dir):  # depends on sample_dir to ensure features built
    return tmp_path / "features.fbin"


# --------------------------------------------------------------------------
# SingletSample / sub-readers
# --------------------------------------------------------------------------


def test_sample_summary_and_meta(sample_dir):
    s = SingletSample(sample_dir)
    assert s.summary["sample_id"] == "sample0"
    meta = s.cell_meta
    assert meta.num_rows == 3
    assert meta.column_names == ["barcode", "n_umi"]


def test_sample_counts_three_blocks(sample_dir):
    s = SingletSample(sample_dir)
    c = s.counts
    assert c.n_cells == 3
    assert sorted(c.block_names) == sorted(["exon_body", "intron_body", "junctions"])
    assert c.exon_body().shape == (4, 3)
    assert c.intron_body().shape == (3, 3)
    assert c.junctions().shape == (3, 3)


def test_sample_snp_two_layer(sample_dir):
    s = SingletSample(sample_dir)
    ad = s.snp.ad()
    dp = s.snp.dp()
    vaf = s.snp.vaf()
    assert ad.shape == dp.shape == vaf.shape == (2, 3)
    # vaf = ad/dp where dp>0; row 0 col 0 → 1/4
    arr = vaf.toarray()
    assert pytest.approx(arr[0, 0], rel=1e-6) == 0.25
    assert pytest.approx(arr[0, 2], rel=1e-6) == 2.0 / 5.0
    assert pytest.approx(arr[1, 1], rel=1e-6) == 0.5


def test_sample_mt_two_layer(sample_dir):
    s = SingletSample(sample_dir)
    assert s.mt.ad().shape == (2, 3)


def test_sample_nonhost(sample_dir):
    s = SingletSample(sample_dir)
    nh = s.nonhost
    assert nh is not None
    summary = nh.summary()
    assert summary["kraken_total_reads"] == 1234
    df = nh.species_table()
    assert list(df["taxid"]) == [9606]
    pc = nh.per_cell()
    assert pc.shape == (1, 3)


def test_sample_rejects_missing_summary(tmp_path):
    with pytest.raises(FileNotFoundError):
        SingletSample(tmp_path)


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------


def test_gene_counts_view(sample_dir, features_path):
    s = SingletSample(sample_dir)
    g = gene_counts(s, features_path)
    # Gene 0: exons rows 0..2 + introns rows 0..1 + junctions rows 0..1
    # Gene 1: exons row 2..3 + introns row 2..2 + junctions row 2..2
    # Cell 0: gene0 = (1+0+4)+(0+2)+(1+0)=8; gene1 = 0 + 0 + 0 = 0
    # Cell 1: gene0 = (0+3+0)+(1+0)+(0+2)=6; gene1 = 0 + 0 + 0 = 0
    # Cell 2: gene0 = (2+0+0)+(0+0)+(0+0)=2; gene1 = 5 + 3 + 4 = 12
    expected = np.array([
        [8, 6, 2],
        [0, 0, 12],
    ], dtype=np.int32)
    np.testing.assert_array_equal(g.toarray(), expected)


def test_usa_partition_invariant(sample_dir, features_path):
    s = SingletSample(sample_dir)
    g = gene_counts(s, features_path).toarray()
    triplet = usa(s, features_path)
    total = (
        triplet.spliced.toarray()
        + triplet.unspliced.toarray()
        + triplet.ambiguous.toarray()
    )
    np.testing.assert_array_equal(total, g)


def test_usa_class_split(sample_dir, features_path):
    s = SingletSample(sample_dir)
    t = usa(s, features_path)
    # ambiguous = II junctions only → gene1 row, all cells from junction row 2
    amb = t.ambiguous.toarray()
    assert amb[0].sum() == 0
    np.testing.assert_array_equal(amb[1], np.array([0, 0, 4]))


def test_psi_view(sample_dir, features_path):
    s = SingletSample(sample_dir)
    p = psi(s, features_path)
    # 3 junctions × 3 cells; values in [0, 1].
    assert p.shape == (3, 3)
    arr = p.toarray()
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_sample_close_releases_files(sample_dir):
    s = SingletSample(sample_dir)
    c = s.counts
    _ = c.exon_body()
    c.close()
    # second close is a no-op
    c.close()
