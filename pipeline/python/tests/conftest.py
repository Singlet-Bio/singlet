"""
Shared pytest fixtures for the singlify wrapper tests.

Most tests need a real `.1pz` fixture — and the only fully-realistic way to
get one is to run the singlify pipeline on a tiny sample. Rather than
committing large fixtures to git, we resolve a fixture lazily at test time:

1. If the environment variable ``SINGLIFY_TEST_FIXTURE_DIR`` is set, use it.
2. Otherwise, look for a known-good pipeline output on the Clipper NFS
   tree. Skip tests that need a fixture if nothing is found.

This keeps the test suite runnable both in CI (where the fixture env-var
would be set to a downloaded tarball) and locally on Clipper (where real
pipeline outputs already exist on NFS).
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import pytest


# Canonical fallback search paths on Clipper — use the smallest / fastest
# known-good samples first so the suite doesn't spend tens of seconds
# reading a 12,925-cell matrix for every test.
_KNOWN_GOOD_SAMPLES = [
    # GSM7103327 — 10xv3, 205K reads, 19 cells (tiny — ideal for CI)
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE227/GSE227136/GSM7103327",
    # GSM3587956 — seqwell, 76 cells
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE116/GSE116256/GSM3587956",
    # GSM3693219 — drop-seq, 3512 cells
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE129/GSE129096/GSM3693219",
    # GSM5293863 — 10xv3, 12925 cells (the canonical large test)
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE174/GSE174399/GSM5293863",
]


def _resolve_fixture_dir() -> Optional[pathlib.Path]:
    env = os.environ.get("SINGLIFY_TEST_FIXTURE_DIR")
    if env:
        p = pathlib.Path(env)
        if p.is_dir():
            return p
    for candidate in _KNOWN_GOOD_SAMPLES:
        p = pathlib.Path(candidate)
        if not p.is_dir():
            continue
        if (p / "gene_counts.1pz").is_file() or (p / "spliced.1pz").is_file():
            return p
    return None


@pytest.fixture(scope="session")
def fixture_dir() -> pathlib.Path:
    """A directory containing a real singlify pipeline output."""
    d = _resolve_fixture_dir()
    if d is None:
        pytest.skip(
            "no singlify pipeline fixture available. Set "
            "SINGLIFY_TEST_FIXTURE_DIR to a directory with .1pz files."
        )
    return d


@pytest.fixture(scope="session")
def gene_counts_path(fixture_dir: pathlib.Path) -> pathlib.Path:
    """Path to the gene_counts.1pz of the session fixture (if present)."""
    p = fixture_dir / "gene_counts.1pz"
    if not p.is_file():
        pytest.skip(f"gene_counts.1pz not present in {fixture_dir}")
    return p


@pytest.fixture(scope="session")
def primary_matrix_path(fixture_dir: pathlib.Path) -> pathlib.Path:
    """Any usable per-gene .1pz from the session fixture.

    Prefers ``gene_counts.1pz`` on older pipeline outputs and falls back to
    ``spliced.1pz`` on Tier-1-dropped newer outputs. Skips if neither is
    present.
    """
    for name in ("gene_counts.1pz", "spliced.1pz"):
        p = fixture_dir / name
        if p.is_file():
            return p
    pytest.skip(f"no gene_counts.1pz or spliced.1pz in {fixture_dir}")


@pytest.fixture(scope="session")
def spliced_path(fixture_dir: pathlib.Path) -> pathlib.Path:
    p = fixture_dir / "spliced.1pz"
    if not p.is_file():
        pytest.skip(f"spliced.1pz not present in {fixture_dir}")
    return p


@pytest.fixture(scope="session")
def unspliced_path(fixture_dir: pathlib.Path) -> pathlib.Path:
    p = fixture_dir / "unspliced.1pz"
    if not p.is_file():
        pytest.skip(f"unspliced.1pz not present in {fixture_dir}")
    return p


@pytest.fixture(scope="session")
def ambiguous_path(fixture_dir: pathlib.Path) -> pathlib.Path:
    p = fixture_dir / "ambiguous.1pz"
    if not p.is_file():
        pytest.skip(f"ambiguous.1pz not present in {fixture_dir}")
    return p
