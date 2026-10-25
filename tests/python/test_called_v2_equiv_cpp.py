# SPDX-License-Identifier: MIT
"""bench/called_v2/equiv_cpp.py: the regression check that a binary's cell_calls.tsv equals cellcall.call on that binary's
own stored counts, and that the inputs it handed the caller (per-barcode totals, knee set, filtered matrix) are the stored
ones. A synthetic run directory in the binary's output layout; the Monte Carlo call needs numba (skipped without it)."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

pytest.importorskip("numba")

_spec = importlib.util.spec_from_file_location(
    "equiv_cpp", ROOT / "bench" / "called_v2" / "equiv_cpp.py"
)
EQ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EQ)
cellcall = EQ.cellcall


def _write_mtx(path: Path, M: np.ndarray, features, barcodes):
    buf = io.BytesIO()
    scipy.io.mmwrite(buf, sp.coo_matrix(M.astype(np.int64)), field="integer")
    with gzip.open(str(path) + ".mtx.gz", "wb") as f:
        f.write(buf.getvalue())
    for suffix, names in (("_features.tsv.gz", features), ("_barcodes.tsv.gz", barcodes)):
        with gzip.open(str(path) + suffix, "wt") as f:
            f.write("".join(n + "\n" for n in names))


def _write_calls(run: Path, bc, umi_exon, umi_total, is_cell, cv2, fdr):
    with open(run / "cell_calls.tsv", "w", newline="\n") as f:
        f.write(
            "barcode\tis_cell\tcalled\tcalled_v2\tumi_exon\tumi_total\tfdr\tpvalue\tlog_prob\tcall_method\n"
        )
        for j, b in enumerate(bc):
            q = "NA" if np.isnan(fdr[j]) else "%.6g" % fdr[j]
            f.write(
                "%s\t%s\t%d\t%d\t%d\t%d\t%s\tNA\tNA\tcalled_v2\n"
                % (
                    b,
                    "TRUE" if is_cell[j] else "FALSE",
                    int(is_cell[j]),
                    int(cv2[j]),
                    umi_exon[j],
                    umi_total[j],
                    q,
                )
            )


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Ambient droplets, large cells above the knee and uncalled low-RNA cells; exon + intron features of 300 genes."""
    rng = np.random.default_rng(7)
    G, n_amb, n_big, n_low = 300, 3000, 200, 120
    amb = rng.dirichlet(np.full(G, 0.3))
    types = [rng.dirichlet(np.full(G, 0.1)) for _ in range(3)]
    cols = [rng.multinomial(rng.integers(1, 40), amb) for _ in range(n_amb)]
    cols += [
        rng.multinomial(rng.integers(3000, 20000), 0.9 * types[i % 3] + 0.1 * amb)
        for i in range(n_big)
    ]
    cols += [
        rng.multinomial(rng.integers(150, 600), 0.8 * types[i % 3] + 0.2 * amb)
        for i in range(n_low)
    ]
    M = np.array(cols).T.astype(np.int64)
    E = np.floor(M * 0.7).astype(np.int64)
    Intr = M - E
    n = M.shape[1]
    bc = ["BC%06d" % j for j in range(n)]
    genes = ["ENSG%05d" % g for g in range(G)]
    run = tmp_path_factory.mktemp("run")
    _write_mtx(run / "exon_counts", E, [g + "_exon1" for g in genes], bc)
    _write_mtx(run / "intron_counts", Intr, [g + "_intron1" for g in genes], bc)

    # what the binary does: knee on exon totals, then cellcall on exon + intron genes
    ex_tot = E.sum(axis=0)
    is_cell, umi_min = EQ.cellranger22(ex_tot)
    X = cellcall.gene_counts(
        EQ.load(str(run), "exon_counts"), EQ.load(str(run), "intron_counts"), bc
    )
    cv2, fdr, params = cellcall.call(X, is_cell, "10xv3")
    assert (
        cv2.sum() > is_cell.sum()
    )  # the call adds low-RNA cells, so the check below is not vacuous
    _write_calls(run, bc, ex_tot, M.sum(axis=0), is_cell, cv2, fdr.astype(np.float64))
    (run / "filtered_feature_bc_matrix").mkdir()
    with gzip.open(run / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "wt") as f:
        f.write(
            "".join(b + "-1\n" for b, c in zip(bc, cv2) if c)
        )  # 10x GEM-well suffix, as the binary writes
    stats = {
        "provenance": {
            "cell_call": {
                "knee": {
                    "method": "CellRanger2.2",
                    "umi_min": int(umi_min),
                    "n_called": int(is_cell.sum()),
                },
                "called_v2": {
                    "protocol": "10xv3",
                    "lower": params["lower"],
                    "alpha": params.get("alpha"),
                },
            }
        }
    }
    (run / "pileup_stats.json").write_text(json.dumps(stats))
    return run


def _copy(src: Path, dst: Path):
    import shutil

    shutil.copytree(src, dst)
    return dst


def test_identical_run_passes(run_dir):
    r = EQ.compare(str(run_dir))
    assert r["identical"] and r["inputs_ok"] and r["pass"], r
    assert r["inputs"] == {
        "umi_exon_equal": True,
        "umi_total_equal": True,
        "knee_equal": True,
        "knee_umi_min": r["inputs"]["knee_umi_min"],
        "filtered_is_called_v2": True,
    }
    assert r["protocol"] == "10xv3"


def test_wrong_called_v2_fails(run_dir, tmp_path):
    run = _copy(run_dir, tmp_path / "r")
    lines = (run / "cell_calls.tsv").read_text().splitlines()
    # turn every added (low-RNA) cell off: what a binary whose caller saw different inputs would write
    out = [lines[0]]
    for ln in lines[1:]:
        f = ln.split("\t")
        if f[1] == "FALSE" and f[3] == "1":
            f[3] = "0"
        out.append("\t".join(f))
    (run / "cell_calls.tsv").write_text("\n".join(out) + "\n")
    r = EQ.compare(str(run))
    assert not r["identical"] and not r["pass"]


def test_wrong_knee_set_fails(run_dir, tmp_path):
    run = _copy(run_dir, tmp_path / "r")
    lines = (run / "cell_calls.tsv").read_text().splitlines()
    f = lines[1].split("\t")  # first barcode is ambient: mark it as a knee cell
    f[1], f[2] = "TRUE", "1"
    lines[1] = "\t".join(f)
    (run / "cell_calls.tsv").write_text("\n".join(lines) + "\n")
    r = EQ.compare(str(run))
    assert r["inputs"]["knee_equal"] is False and not r["pass"]


def test_filtered_matrix_not_called_v2_fails(run_dir, tmp_path):
    run = _copy(run_dir, tmp_path / "r")
    with gzip.open(run / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "wt") as f:
        f.write("BC000000\n")
    r = EQ.compare(str(run))
    assert r["inputs"]["filtered_is_called_v2"] is False and not r["pass"]


def test_knee_rounds_half_away_from_zero():
    # nmax = 10865 -> nmin = 1086.5 -> 1087 (C++ std::round); Python's round() would give 1086
    tot = np.array([10865] * 31 + [1087, 1086, 5], dtype=np.int64)
    flags, umi_min = EQ.cellranger22(tot)
    assert umi_min == 1087
    assert flags.tolist() == [True] * 32 + [False, False]
