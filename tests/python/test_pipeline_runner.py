# SPDX-License-Identifier: MIT
"""Unit tests for ``singlet.pipeline`` — the URL/accession → outputs runner.

These tests stub out the C++ binary with a tiny mock executable so the
wrapper logic (input resolution, argv construction, error handling) is
covered without needing the real pipeline.
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from singlet.pipeline import PipelineError, Run, run
from singlet.pipeline import _run as runner

# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("http://example.com/x", True),
            ("HTTPS://example.com/x", True),
            ("ftp://ftp.sra.ebi.ac.uk/x", True),
            ("s3://bucket/key", True),
            ("/local/path.1fq", False),
            ("SRR11537951", False),
        ],
    )
    def test_is_url(self, value, expected):
        assert runner._is_url(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("SRR11537951", True),
            ("ERR1234567", True),
            ("DRR9999999", True),
            ("srr11537951", False),  # case-sensitive
            ("SRR123", False),  # too short
            ("not_an_accession", False),
        ],
    )
    def test_looks_like_accession(self, value, expected):
        assert runner._looks_like_accession(value) is expected

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("SRR11537951", "SRR11537951"),
            (
                "https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR11537951/SRR11537951",
                "SRR11537951",
            ),
            ("/data/SRR12345678.1fq", "SRR12345678"),
            (["SRR99887766_1.fastq.gz", "SRR99887766_2.fastq.gz"], "SRR99887766"),
            ("nameless_file.1fq", "nameless_file"),
        ],
    )
    def test_infer_accession(self, source, expected):
        assert runner._infer_accession(source) == expected


# --------------------------------------------------------------------------
# Binary discovery
# --------------------------------------------------------------------------


def _make_fake_binary(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return path


class TestFindBinary:
    def test_explicit_path(self, tmp_path):
        b = _make_fake_binary(tmp_path / "singlet")
        assert runner._find_binary(b) == b.resolve()

    def test_env_var(self, tmp_path, monkeypatch):
        b = _make_fake_binary(tmp_path / "singlet")
        monkeypatch.setenv("SINGLET_BINARY", str(b))
        # Ensure $PATH lookup can't shadow the env var with a real binary.
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        assert runner._find_binary() == b.resolve()

    def test_missing_raises(self, tmp_path, monkeypatch):
        from singlet.pipeline import _binary as binary_mod

        monkeypatch.delenv("SINGLET_BINARY", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        # Force find_binary to only consider explicit + env + PATH by
        # neutralising the repo-default fallbacks.
        monkeypatch.setattr(binary_mod, "__file__", str(tmp_path / "fake_pkg" / "x.py"))
        with pytest.raises(PipelineError, match="locate the singlet binary"):
            runner._find_binary(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------
# Reference resolution
# --------------------------------------------------------------------------


class TestResolveReference:
    def test_explicit_ref_base(self, tmp_path):
        (tmp_path / "GRCh38-2024-A").mkdir()
        out = runner._resolve_reference("human", tmp_path)
        assert out == tmp_path / "GRCh38-2024-A"

    def test_missing_organism_dir_returns_base(self, tmp_path):
        # Base exists but no GRCh38-2024-A subdir; we fall back to base so
        # the binary can do its own discovery.
        out = runner._resolve_reference("human", tmp_path)
        assert out == tmp_path

    def test_missing_base_raises(self, tmp_path):
        with pytest.raises(PipelineError, match="does not exist"):
            runner._resolve_reference("human", tmp_path / "nope")

    def test_env_var(self, tmp_path, monkeypatch):
        (tmp_path / "GRCm39-2024-A").mkdir()
        monkeypatch.setenv("SINGLET_REF_BASE", str(tmp_path))
        out = runner._resolve_reference("mouse", None)
        assert out == tmp_path / "GRCm39-2024-A"


# --------------------------------------------------------------------------
# Argv construction
# --------------------------------------------------------------------------


class TestBuildProcessArgv:
    def _base_kwargs(self, tmp_path):
        ref_dir = tmp_path / "GRCh38-2024-A"
        (ref_dir / "genes").mkdir(parents=True)
        (ref_dir / "genes" / "genes.gtf").write_text("# fake gtf\n")
        return dict(
            binary=tmp_path / "singlet",
            archive=tmp_path / "in.1fq",
            output_dir=tmp_path / "out",
            ref_dir=ref_dir,
            organism="human",
            threads=8,
            enable_snps=False,
            enable_pipeline_extras=False,
            cascade="off",
            te_classify="off",
            nonhost=False,
            raw_matrix=False,
            metadata_json=None,
            extra_args=(),
        )

    def test_minimal(self, tmp_path):
        argv = runner._build_process_argv(**self._base_kwargs(tmp_path))
        assert argv[0] == str(tmp_path / "singlet")
        assert argv[1] == str(tmp_path / "in.1fq")
        assert "--exons" in argv
        assert "--out-prefix" in argv
        assert "--threads" in argv and "8" in argv
        # nonhost=False → screens disabled
        assert "--no-viral-screen" in argv
        assert "--no-microbial-screen" in argv

    def test_full(self, tmp_path):
        kw = self._base_kwargs(tmp_path)
        kw.update(
            enable_pipeline_extras=True,
            cascade="on",
            te_classify="on",
            nonhost=True,
            raw_matrix=True,
            metadata_json=tmp_path / "meta.json",
            extra_args=("--min-mapq", "30"),
        )
        argv = runner._build_process_argv(**kw)
        assert "--pipeline" in argv
        assert "--cascade" in argv and "on" in argv
        assert "--te-classify" in argv
        assert "--raw-matrix" in argv
        assert "--no-viral-screen" not in argv  # nonhost on
        assert "--metadata-json" in argv
        assert argv[-2:] == ["--min-mapq", "30"]


# --------------------------------------------------------------------------
# run() — end-to-end with mocked binary
# --------------------------------------------------------------------------


_MOCK_BINARY = """#!/usr/bin/env python3
import sys, os, json, pathlib

argv = sys.argv[1:]
# Sub-commands: download / encode write a .1fq file to -o
if argv and argv[0] in ("download", "encode"):
    out_idx = argv.index("-o") if "-o" in argv else -1
    if out_idx >= 0 and out_idx + 1 < len(argv):
        pathlib.Path(argv[out_idx + 1]).write_text("MOCK_1FQ")
    sys.exit(0)

# Otherwise treat as the process invocation.
out_prefix = None
for i, a in enumerate(argv):
    if a == "--out-prefix" and i + 1 < len(argv):
        out_prefix = argv[i + 1]

if out_prefix:
    p = pathlib.Path(out_prefix)
    p.mkdir(parents=True, exist_ok=True)
    (p / "summary.json").write_text(json.dumps({"n_cells": 1234, "argv": argv}))
sys.exit(0)
"""


@pytest.fixture
def mock_binary(tmp_path):
    p = tmp_path / "singlet"
    p.write_text(_MOCK_BINARY)
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return p


@pytest.fixture
def ref_base(tmp_path):
    base = tmp_path / "ref"
    (base / "GRCh38-2024-A" / "genes").mkdir(parents=True)
    (base / "GRCh38-2024-A" / "genes" / "genes.gtf").write_text("# gtf\n")
    return base


class TestRunEndToEnd:
    def test_accession_source(self, tmp_path, mock_binary, ref_base):
        out = tmp_path / "out"
        result = run(
            "SRR11537951",
            output_dir=out,
            binary=mock_binary,
            ref_base=ref_base,
            organism="human",
            enable_snps=False,
        )
        assert isinstance(result, Run)
        assert bool(result) is True
        assert result.success
        assert result.output_dir == out
        assert result.accession == "SRR11537951"
        assert (out / "summary.json").is_file()
        summary = json.loads((out / "summary.json").read_text())
        assert summary["n_cells"] == 1234

    def test_local_1fq(self, tmp_path, mock_binary, ref_base):
        archive = tmp_path / "in.1fq"
        archive.write_text("MOCK_1FQ")
        result = run(
            archive,
            output_dir=tmp_path / "out",
            binary=mock_binary,
            ref_base=ref_base,
            enable_snps=False,
        )
        assert result.success
        assert "in.1fq" in " ".join(result.command)

    def test_failure_raises(self, tmp_path, ref_base):
        # Binary that always fails
        bad = tmp_path / "bad"
        bad.write_text("#!/bin/sh\necho oops 1>&2\nexit 7\n")
        bad.chmod(bad.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        archive = tmp_path / "in.1fq"
        archive.write_text("x")
        with pytest.raises(PipelineError, match="status 7"):
            run(
                archive,
                output_dir=tmp_path / "out",
                binary=bad,
                ref_base=ref_base,
                enable_snps=False,
            )

    def test_keep_intermediate(self, tmp_path, mock_binary, ref_base):
        archive = tmp_path / "in.1fq"
        archive.write_text("MOCK_1FQ")
        work = tmp_path / "work"
        result = run(
            archive,
            output_dir=tmp_path / "out",
            binary=mock_binary,
            ref_base=ref_base,
            work_dir=work,
            keep_intermediate=True,
            enable_snps=False,
        )
        assert result.success
        # work_dir was created
        assert work.exists()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCLI:
    def test_help(self):
        cp = subprocess.run(
            [sys.executable, "-m", "singlet.pipeline", "--help"],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0
        assert "singlet-process" in cp.stdout
        assert "--output-dir" in cp.stdout

    def test_requires_source_or_reads(self):
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "singlet.pipeline",
                "--output-dir",
                "/tmp/whatever",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode != 0
        assert "source" in cp.stderr.lower() or "reads" in cp.stderr.lower()

    def test_rejects_both_source_and_reads(self):
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "singlet.pipeline",
                "SRR123456",
                "--reads",
                "a.fq",
                "b.fq",
                "-o",
                "/tmp/x",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode != 0

    def test_end_to_end(self, tmp_path, mock_binary, ref_base, monkeypatch):
        out = tmp_path / "out"
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "singlet.pipeline",
                "SRR11537951",
                "--output-dir",
                str(out),
                "--organism",
                "human",
                "--binary",
                str(mock_binary),
                "--ref-base",
                str(ref_base),
                "--no-snps",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr
        assert (out / "summary.json").is_file()

    def test_passthrough_after_dashdash(self, tmp_path, mock_binary, ref_base):
        out = tmp_path / "out"
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "singlet.pipeline",
                "SRR11537951",
                "--output-dir",
                str(out),
                "--binary",
                str(mock_binary),
                "--ref-base",
                str(ref_base),
                "--no-snps",
                "--quiet",
                "--",
                "--min-mapq",
                "30",
            ],
            capture_output=True,
            text=True,
        )
        assert cp.returncode == 0, cp.stderr
        summary = json.loads((out / "summary.json").read_text())
        assert "--min-mapq" in summary["argv"] and "30" in summary["argv"]
