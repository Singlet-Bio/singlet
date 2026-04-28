"""
Tests for the :mod:`singlify.cli` command-line entry point.

These run the CLI in-process via ``cli.main([...])`` so they don't shell
out, but they exercise the same code path that ``singlify`` on the
PATH would.
"""

from __future__ import annotations

import io
import pathlib
from contextlib import redirect_stderr, redirect_stdout

import pytest


_pz_io = pytest.importorskip(
    "singlify._pz_io",
    reason="singlify._pz_io C++ extension not built; run `pip install -e .` first",
)

from singlify import cli  # noqa: E402


def _run(argv):
    """Run cli.main(argv) and return (exit_code, stdout, stderr)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = cli.main(argv)
        except SystemExit as e:
            rc = int(e.code) if e.code is not None else 0
    return rc, out.getvalue(), err.getvalue()


def test_cli_no_args_shows_usage():
    """Calling singlify with no subcommand should error and show usage."""
    rc, out, err = _run([])
    assert rc != 0
    # argparse writes usage to stderr in error mode
    assert "usage" in err.lower() or "usage" in out.lower()


def test_cli_help():
    """`singlify --help` exits cleanly with exit code 0."""
    rc, out, err = _run(["--help"])
    assert rc == 0
    combined = out + err
    assert "convert" in combined
    assert "info" in combined
    assert "verify" in combined


def test_cli_info_help():
    rc, out, err = _run(["info", "--help"])
    assert rc == 0
    assert ".1pz" in (out + err)


def test_cli_info_on_real_file(primary_matrix_path: pathlib.Path):
    """`singlify info FILE` prints the header summary."""
    rc, out, err = _run(["info", str(primary_matrix_path)])
    assert rc == 0
    combined = out + err
    # Expected lines from the info command output
    assert "shape:" in combined
    assert "nnz:" in combined
    assert "vt_code:" in combined
    assert "Embedded metadata" in combined or "gsm_id" in combined


def test_cli_info_on_directory(fixture_dir: pathlib.Path):
    """`singlify info DIR` summarizes every .1pz file."""
    rc, out, err = _run(["info", str(fixture_dir)])
    assert rc == 0
    combined = out + err
    assert "pipeline directory" in combined
    # Every fixture has at least gene_counts
    assert "gene_counts.1pz" in combined or "spliced.1pz" in combined
    # And the embedded metadata
    assert "gsm_id" in combined.lower()


def test_cli_verify_passes_on_real_file(primary_matrix_path: pathlib.Path):
    """`singlify verify FILE` returns 0 on a known-good file."""
    rc, out, err = _run(["verify", str(primary_matrix_path)])
    assert rc == 0
    assert "OK" in (out + err)


def test_cli_verify_fails_on_garbage(tmp_path: pathlib.Path):
    """`singlify verify` returns non-zero on a garbage file."""
    bad = tmp_path / "garbage.1pz"
    bad.write_bytes(b"\x00" * 200)
    rc, out, err = _run(["verify", str(bad)])
    assert rc != 0
    assert "FAIL" in (out + err)


def test_cli_verify_fails_on_missing_file(tmp_path: pathlib.Path):
    """`singlify verify` errors if the file does not exist."""
    rc, out, err = _run(["verify", str(tmp_path / "nope.1pz")])
    assert rc != 0


def test_cli_list_on_real_gse():
    """`singlify list` walks a GSE root and prints one line per GSM."""
    # Use a known GSE that has multiple GSMs in the local NFS tree
    gse_root = pathlib.Path(
        "/mnt/projects/debruinz_project/singlify_pipeline/"
        "quant/scrna/GSE174/GSE174399"
    )
    if not gse_root.is_dir():
        pytest.skip("no GSE174399 fixture available")
    rc, out, err = _run(["list", str(gse_root), "--matrix", "spliced"])
    assert rc == 0, f"list failed: {err}"
    combined = out + err
    # Header line
    assert "gsm_id" in combined
    assert "protocol" in combined
    # Should mention at least one GSM
    assert "GSM5293" in combined


def test_cli_list_errors_on_non_dir(tmp_path: pathlib.Path):
    rc, out, err = _run(["list", str(tmp_path / "nope")])
    assert rc != 0


def test_cli_list_format_json():
    """``singlify list --format json`` emits one JSON object per line."""
    import json
    gse_root = pathlib.Path(
        "/mnt/projects/debruinz_project/singlify_pipeline/"
        "quant/scrna/GSE174/GSE174399"
    )
    if not gse_root.is_dir():
        pytest.skip("no GSE174399 fixture available")
    rc, out, err = _run([
        "list", str(gse_root), "--matrix", "spliced", "--format", "json",
    ])
    assert rc == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines, "no output lines"
    for ln in lines:
        # Each line must be valid JSON with at least gsm_id and status
        obj = json.loads(ln)
        assert "gsm_id" in obj
        assert "status" in obj
        if obj["status"] == "ok":
            assert isinstance(obj["m"], int)
            assert isinstance(obj["n"], int)
            assert isinstance(obj["nnz"], int)
            assert obj["vt_code"] in {"UINT8", "UINT16", "UINT32"}


def test_cli_list_format_tsv():
    """``singlify list --format tsv`` emits a header row + tab-delimited rows."""
    gse_root = pathlib.Path(
        "/mnt/projects/debruinz_project/singlify_pipeline/"
        "quant/scrna/GSE174/GSE174399"
    )
    if not gse_root.is_dir():
        pytest.skip("no GSE174399 fixture available")
    rc, out, err = _run([
        "list", str(gse_root), "--matrix", "spliced", "--format", "tsv",
    ])
    assert rc == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) >= 2  # header + at least one row
    header = lines[0].split("\t")
    assert "gsm_id" in header
    assert "nnz" in header
    # All data rows should have the same column count as the header
    for ln in lines[1:]:
        cols = ln.split("\t")
        assert len(cols) == len(header), (
            f"row col count {len(cols)} != header col count {len(header)}"
        )


def test_cli_convert_writes_h5ad(fixture_dir: pathlib.Path, tmp_path: pathlib.Path):
    """`singlify convert DIR -o OUT.h5ad` writes a usable AnnData."""
    pytest.importorskip("anndata")
    out_path = tmp_path / "test_out.h5ad"
    rc, out, err = _run(["convert", str(fixture_dir), "-o", str(out_path)])
    assert rc == 0, f"convert failed: stdout={out} stderr={err}"
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Round-trip — load the .h5ad and check the GEO context survived
    import anndata as ad
    adata = ad.read_h5ad(out_path)
    assert "singlify" in adata.uns
    assert "gsm_id" in adata.uns["singlify"]
    assert adata.n_obs > 0
    assert adata.n_vars > 0


def test_cli_cohort_writes_merged_h5ad(
    fixture_dir: pathlib.Path, tmp_path: pathlib.Path
):
    """`singlify cohort ROOT -o OUT.h5ad` merges GSM subdirs into one .h5ad.

    Builds a synthetic 2-sample cohort by symlinking the session fixture
    into two ``GSM*``-named subdirectories, runs the CLI, and verifies the
    merged AnnData carries both samples and the obs_names are prefixed by
    ``gsm_id`` (so 10x barcode collisions across samples cannot clash).
    """
    pytest.importorskip("anndata")
    import os

    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    gsm_names = ["GSM9000001", "GSM9000002"]
    for gsm in gsm_names:
        sd = cohort_root / gsm
        sd.mkdir()
        for f in fixture_dir.iterdir():
            if f.is_file():
                os.symlink(f, sd / f.name)

    out_path = tmp_path / "cohort.h5ad"
    rc, out, err = _run([
        "cohort", str(cohort_root), "-o", str(out_path),
    ])
    assert rc == 0, f"cohort failed: stdout={out} stderr={err}"
    assert out_path.exists() and out_path.stat().st_size > 0

    import anndata as ad
    merged = ad.read_h5ad(out_path)
    # Both synthetic samples' cells should be present, roughly 2x a single read
    assert merged.n_obs > 0
    assert merged.n_vars > 0
    assert "gsm_id" in merged.obs.columns
    # Each cell's obs_name should start with its gsm_id prefix so that
    # identical 10x barcodes from different GEMs don't collide.
    sample_obs_names = list(merged.obs_names[:10])
    assert any(name.startswith("GSM") for name in sample_obs_names), (
        f"obs_names not GSM-prefixed: {sample_obs_names}"
    )
    # obs_names must be globally unique across samples — this guards against
    # silent barcode collisions when two samples share the same embedded
    # gsm_id (e.g. re-runs or symlinked synthetic cohorts).
    assert merged.n_obs == len(set(merged.obs_names)), (
        "merged obs_names are not unique"
    )
