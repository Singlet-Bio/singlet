# SPDX-License-Identifier: MIT
"""bench/pipeline/analysis/loaders.load_singlet reads spliced / unspliced / ambiguous from the USA mtx layers when they exist
and derives them from exon / intron counts for older runs. Tiny synthetic run directories in both layouts must agree.
Synthetic data lives only in this unit test."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

pytest.importorskip("anndata")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bench_loaders", ROOT / "bench" / "pipeline" / "analysis" / "loaders.py")
LD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LD)

GENES = ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]
BCS = ["AAAAAAAAAAAAAAAA-1", "CCCCCCCCCCCCCCCC-1", "GGGGGGGGGGGGGGGG-1", "TTTTTTTTTTTTTTTT-1"]
# truth (genes x barcodes); exon_counts = S + A (Gene), intron_counts = GeneFull - Gene, which differs from U
S = np.array([[3, 0, 1, 2], [0, 4, 0, 1], [2, 2, 0, 0]])
U = np.array([[1, 2, 0, 0], [5, 0, 0, 3], [0, 1, 1, 0]])
A = np.array([[1, 0, 2, 0], [0, 1, 0, 0], [1, 0, 0, 1]])
INTRON_LEGACY = np.array([[0, 2, 0, 0], [3, 0, 0, 1], [0, 0, 1, 0]])  # deliberately not U


def _write(d: Path, stem: str, M, feats: list[str]) -> None:
    with gzip.open(d / f"{stem}.mtx.gz", "wb") as f:
        scipy.io.mmwrite(f, sp.coo_matrix(M))
    with gzip.open(d / f"{stem}_features.tsv.gz", "wt") as f:
        f.write("\n".join(feats) + "\n")
    with gzip.open(d / f"{stem}_barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(BCS) + "\n")


def _run(tmp: Path, usa: bool) -> Path:
    d = tmp / ("usa" if usa else "legacy")
    (d / "raw_feature_bc_matrix").mkdir(parents=True)
    with gzip.open(d / "raw_feature_bc_matrix" / "features.tsv.gz", "wt") as f:
        f.write("\n".join(f"{g}\t{g}\tGene Expression" for g in GENES) + "\n")
    gene_feats = [f"{g}_NAME{i}" for i, g in enumerate(GENES)]
    exon_feats = [f"{g}_NAME{i}_chr1:{100 + i}-{200 + i}" for i, g in enumerate(GENES)]
    intron_feats = [f"{g}_NAME{i}_intron_chr1:{300 + i}-{400 + i}" for i, g in enumerate(GENES)]
    _write(d, "exon_counts", S + A, exon_feats)
    _write(d, "ambiguous_counts", A, gene_feats)
    if usa:
        _write(d, "intron_counts", INTRON_LEGACY, intron_feats)
        _write(d, "spliced_counts", S, gene_feats)
        _write(d, "unspliced_counts", U, gene_feats)
    else:
        _write(d, "intron_counts", U, intron_feats)  # pre-1.3.5: intron_counts is the unspliced layer
    return d


def _dense(a, layer):
    order = [list(a.var_names).index(g) for g in GENES]
    return np.asarray(a.layers[layer].todense())[:, order].T


@pytest.mark.parametrize("usa", [True, False])
def test_layers_match_truth(tmp_path, usa):
    a = LD.load_singlet(_run(tmp_path, usa))
    np.testing.assert_array_equal(_dense(a, "spliced"), S)
    np.testing.assert_array_equal(_dense(a, "unspliced"), U)
    np.testing.assert_array_equal(_dense(a, "ambiguous"), A)


def test_both_layouts_agree(tmp_path):
    a, b = LD.load_singlet(_run(tmp_path, True)), LD.load_singlet(_run(tmp_path, False))
    for layer in ("spliced", "unspliced", "ambiguous"):
        np.testing.assert_array_equal(_dense(a, layer), _dense(b, layer))
