# SPDX-License-Identifier: MIT
"""Tests for bench/adt/hto_methods.py on a synthetic 4-hashtag dataset (tools that are missing are skipped)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[2] / "bench" / "adt" / "hto_methods.py"
_spec = importlib.util.spec_from_file_location("hto_methods", _PATH)
hm = importlib.util.module_from_spec(_spec)
sys.modules["hto_methods"] = hm
_spec.loader.exec_module(hm)

TAGS = ["HTO_A", "HTO_B", "HTO_C", "HTO_D"]
N_SINGLET, N_DOUBLET, N_EMPTY = 380, 20, 300


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = N_SINGLET + N_DOUBLET
    X = rng.poisson(3, size=(4, n))
    truth = []
    for j in range(N_SINGLET):
        k = j % 4
        X[k, j] += rng.poisson(200)
        truth.append(TAGS[k])
    for j in range(N_SINGLET, n):
        a, b = rng.choice(4, 2, replace=False)
        X[a, j] += rng.poisson(200)
        X[b, j] += rng.poisson(200)
        truth.append("Doublet")
    cells = [f"CELL{j:04d}-1" for j in range(n)]
    hto = pd.DataFrame(X, index=TAGS, columns=cells)
    empties = pd.DataFrame(
        rng.poisson(2, size=(4, N_EMPTY)),
        index=TAGS,
        columns=[f"EMPTY{j:04d}-1" for j in range(N_EMPTY)],
    )
    return hto, empties, pd.Series(truth, index=cells)


def _check_shape(res, hto):
    assert list(res.index) == list(hto.columns)
    assert set(res.columns) == {"call", "score"}
    assert res.attrs["seconds"] >= 0
    assert set(res["call"]) <= set(TAGS) | {"Doublet", "Negative"}


def _singlet_accuracy(res, truth):
    s = truth != "Doublet"
    return float((res["call"][s] == truth[s]).mean())


def test_singlet(data):
    hto, empties, truth = data
    res = hm.run_singlet(hto, empties)
    _check_shape(res, hto)
    assert _singlet_accuracy(res, truth) > 0.9
    d = truth == "Doublet"
    assert (res["call"][d] == "Doublet").mean() > 0.5


def test_singlet_without_empties(data):
    hto, _, truth = data
    res = hm.run_singlet(hto, None)
    _check_shape(res, hto)
    assert _singlet_accuracy(res, truth) > 0.9


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")
@pytest.mark.parametrize("name", ["htodemux", "multiseqdemux"])
def test_seurat_methods(data, name, tmp_path):
    import subprocess

    ok = (
        subprocess.run(
            ["Rscript", "-e", "quit(status = !requireNamespace('Seurat', quietly = TRUE))"],
            capture_output=True,
        ).returncode
        == 0
    )
    if not ok:
        pytest.skip("R package Seurat not installed")
    hto, empties, truth = data
    res = hm.METHODS[name](hto, empties, workdir=tmp_path)
    _check_shape(res, hto)
    assert _singlet_accuracy(res, truth) > 0.8


def test_hashsolo(data):
    pytest.importorskip("scanpy")
    pytest.importorskip("anndata")
    hto, empties, truth = data
    try:
        res = hm.run_hashsolo(hto, empties)
    except ImportError as e:  # scanpy.external needs optional extras
        pytest.skip(str(e))
    _check_shape(res, hto)
    assert _singlet_accuracy(res, truth) > 0.8


def test_demuxem(data):
    pytest.importorskip("demuxEM")
    pytest.importorskip("pegasusio")
    import anndata as ad
    import scipy.sparse as sp

    hto, empties, truth = data
    rng = np.random.default_rng(2)
    barcodes = list(hto.columns) + list(empties.columns)
    ng = 200
    X = np.vstack([rng.poisson(5, (hto.shape[1], ng)), rng.poisson(0.05, (empties.shape[1], ng))])
    rna = ad.AnnData(
        X=sp.csr_matrix(X.astype(np.float32)),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=[f"g{i}" for i in range(ng)]),
    )
    res = hm.run_demuxem(hto, empties, rna)
    _check_shape(res, hto)
    assert _singlet_accuracy(res, truth) > 0.8


def test_methods_registry():
    assert set(hm.METHODS) == {
        "singlet",
        "singlet_mixture",
        "htodemux",
        "multiseqdemux",
        "hashsolo",
        "demuxem",
    }
