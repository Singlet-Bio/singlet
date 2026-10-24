# SPDX-License-Identifier: MIT
"""The per-cell label files are public: nothing genotype-like may reach them.

These tests fail if the label schema grows a column, if any string column carries a variant id, a genotype, an allele or a
chromosome position, or if the end-to-end pipeline writes anything of the kind even when it is handed restricted SNP depths.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from singlet.annotate import output as O
from singlet.annotate import pipeline as P
from singlet.annotate.ancestry import POPS
from singlet.annotate.guide import DATA
from singlet.annotate.io import SampleInput


def _table(n=5, **over):
    df = O.empty_table("GSM123", np.arange(n), P.model_version())
    df["cell_type_cl"] = pd.array(["CL:0000084"] * n, dtype="string")
    df["cell_type_label"] = pd.array(["T cell"] * n, dtype="string")
    df["cell_type_score"] = np.full(n, 0.9, np.float32)
    df["sex_chrom"] = pd.array(["female"] * n, dtype="string")
    df["sex_score"] = np.full(n, 0.99, np.float32)
    df["ancestry_hancestro"] = pd.array(["HANCESTRO:0005"] * n, dtype="string")
    df["ancestry_prob"] = np.full(n, 0.95, np.float32)
    for k, v in over.items():
        df[k] = v
    return df


def test_valid_table_passes(tmp_path):
    df = _table()
    O.assert_public_safe(df)
    p = O.write_labels(df, tmp_path / "x.parquet")
    back = pd.read_parquet(p)
    assert list(back.columns) == [c for c, _ in O.COLUMNS]
    assert not O.scan_labels_file(p)


def test_all_null_table_passes():
    O.assert_public_safe(O.empty_table("GSM1", np.arange(3), P.model_version()))


@pytest.mark.parametrize(
    "column,value",
    [
        ("cell_type_label", "rs1426654"),
        ("cell_type_label", "chr15:48426484"),
        ("cell_type_label", "0/1"),
        ("cell_type_label", "1|1"),
        ("cell_type_label", "G>A"),
        ("cell_type_label", "15_48426484_G_A"),
        ("model_version", "GT:AF:AD:DP"),
        ("model_version", "AD=3,1"),
        ("ancestry_hancestro", "EUR"),
        ("ancestry_hancestro", "ACGT"),
        ("sex_chrom", "XY"),
        ("cell_type_cl", "T cell"),
        ("gsm", "donor0"),
    ],
)
def test_genotype_like_values_are_rejected(column, value):
    df = _table()
    df[column] = pd.array([value] * len(df), dtype="string")
    with pytest.raises(O.PublicOutputError):
        O.assert_public_safe(df)


def test_file_scan_reads_decoded_values(tmp_path):
    """The file-level scan looks at decoded strings, so a genotype smuggled into a written file is found."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = _table()
    df["cell_type_label"] = pd.array(["rs1426654"] * len(df), dtype="string")
    p = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p, compression="zstd")
    assert O.scan_labels_file(p)


@pytest.mark.parametrize("extra", ["snp_dp", "genotype", "barcode", "donor", "ad"])
def test_extra_columns_are_rejected(extra):
    df = _table()
    df[extra] = 1
    with pytest.raises(O.PublicOutputError):
        O.assert_public_safe(df)


def test_scores_must_be_probabilities():
    with pytest.raises(O.PublicOutputError):
        O.assert_public_safe(_table(ancestry_prob=np.full(5, 3.0, np.float32)))
    with pytest.raises(O.PublicOutputError):
        O.assert_public_safe(_table(col=np.array([0, 1, 1, 2, 3])))


def test_every_cell_ontology_name_passes_the_guard():
    """No legitimate label of the guide's ontology may trip the genotype filter (else real calls would be dropped)."""
    with gzip.open(DATA / "cl.json.gz", "rt", encoding="utf-8") as f:
        terms = json.load(f)["terms"]
    names = [v["name"] for v in terms.values()]
    df = O.empty_table("GSM1", np.arange(len(names)), P.model_version())
    df["cell_type_cl"] = pd.array(list(terms), dtype="string")
    df["cell_type_label"] = pd.array(names, dtype="string")
    O.assert_public_safe(df)


def test_pipeline_output_carries_no_genotype_from_restricted_input(tmp_path):
    """End to end: a sample with per-donor SNP depths (the restricted input) produces a label file with labels only."""
    rng = np.random.default_rng(1)
    genes = ["XIST", "RPS4Y1", "CD3E", "CD3D", "MS4A1", "CD19"] + [f"G{i}" for i in range(300)]
    n = 80
    X = rng.poisson(0.5, (n, len(genes))).astype(np.float32)
    X[:, 0] += rng.poisson(4, n)
    s = SampleInput(
        gsm="GSM9",
        gse="GSE9",
        X=sps.csr_matrix(X),
        gene_names=genes,
        col=np.arange(n),
        barcodes=[f"AC{i}" for i in range(n)],
        species="human",
        tissue="blood",
        disease="",
    )
    m = 2000
    af = pd.DataFrame(
        {"chrom": ["1"] * m, "pos": np.arange(m) + 10_000, "ref": ["A"] * m, "alt": ["G"] * m}
    )
    F = rng.uniform(0.05, 0.95, (m, 5))
    F[:, POPS.index("EAS")] = rng.uniform(0.0, 0.1, m)
    for k, p in enumerate(POPS):
        af[p] = F[:, k]
    f = F[:, POPS.index("EUR")]
    g = rng.binomial(2, f)
    dp = rng.integers(1, 6, m)
    ad = rng.binomial(dp, np.clip(g / 2, 0.02, 0.98))
    s.donor_sites = {
        "donor0": pd.DataFrame(
            {"chrom": "chr1", "pos": af["pos"], "ref": "A", "alt": "G", "dp": dp, "ad": ad}
        )
    }
    df, summary = P.annotate_sample(s, af_table=af, method="guide")
    assert df["ancestry_hancestro"].notna().all()
    out = O.write_labels(df, tmp_path / "GSM9.parquet")
    assert not O.scan_labels_file(out)
    back = pd.read_parquet(out)
    assert list(back.columns) == [c for c, _ in O.COLUMNS]
    # the summary (kept internal) must not hold per-site data either
    text = json.dumps(summary, default=str)
    assert not O.scan_bytes_for_genotypes(text.encode())
    assert "pos" not in json.dumps(summary["ancestry"])
