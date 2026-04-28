"""
Smoke-test the worked-example script under ``examples/``.

Running the example as a subprocess catches:

- regressions in the example itself (it should always run cleanly on a
  small fixture)
- accidental hard dependencies that creep into ``singlify.io`` /
  ``singlify.interop`` (the example uses the public surface only)
- breakage in the lazy-loaded interop submodules

The test skips if the example script doesn't exist (some checkouts may
prune ``examples/``) or if the fixture resolver can't find a sample.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest


def _subprocess_env() -> dict:
    """Build an env dict that lets the subprocess find ``singlify``.

    In CI the package is ``pip install -e .``-installed and ``singlify``
    is on sys.path automatically. In editable dev runs we may instead be
    relying on the cwd, so we inject the package root into PYTHONPATH
    explicitly. Either way, the subprocess sees the same module the
    parent does.
    """
    here = pathlib.Path(__file__).resolve().parent
    pkg_root = here.parent  # singlify/python/
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(pkg_root) + (os.pathsep + existing if existing else "")
    )
    return env


_pz_io = pytest.importorskip(
    "singlify._pz_io",
    reason="singlify._pz_io C++ extension not built; run `pip install -e .` first",
)


@pytest.fixture(scope="module")
def example_script() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    p = here.parent / "examples" / "end_to_end_workflow.py"
    if not p.is_file():
        pytest.skip(f"example script not found at {p}")
    return p


def test_example_help(example_script: pathlib.Path):
    """The script must accept ``--help`` and exit 0."""
    result = subprocess.run(
        [sys.executable, str(example_script), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, (
        f"--help failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "pipeline_dir" in (result.stdout + result.stderr).lower()


def test_python_dash_m_singlify_help():
    """``python -m singlify --help`` works as a fallback CLI entry point."""
    here = pathlib.Path(__file__).resolve().parent
    pkg_root = here.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pkg_root) + (
        os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [sys.executable, "-m", "singlify", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    assert result.returncode == 0, (
        f"python -m singlify --help failed: stderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "convert" in combined
    assert "info" in combined
    assert "verify" in combined


def test_python_dash_m_singlify_info_on_fixture(fixture_dir: pathlib.Path):
    """``python -m singlify info DIR`` runs cleanly on a real fixture."""
    here = pathlib.Path(__file__).resolve().parent
    pkg_root = here.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pkg_root) + (
        os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [sys.executable, "-m", "singlify", "info", str(fixture_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, (
        f"python -m singlify info failed: stderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "pipeline directory" in combined
    assert "gsm_id" in combined.lower()


def test_example_runs_on_real_fixture(
    example_script: pathlib.Path,
    fixture_dir: pathlib.Path,
):
    """The script must run end-to-end on a real pipeline fixture without
    raising. It may complete the QC step early on tiny fixtures (which
    is the expected behavior for the smallest sample), but it must
    print the final ``=== done ===`` marker."""
    pytest.importorskip("anndata")
    result = subprocess.run(
        [sys.executable, str(example_script), str(fixture_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        env=_subprocess_env(),
    )
    if result.returncode != 0:
        pytest.fail(
            f"example failed (rc={result.returncode}):\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )
    combined = result.stdout + result.stderr
    assert "=== done ===" in combined, (
        "example did not reach the done marker"
    )
    # The example must always print the GSM ID it loaded
    assert "gsm_id:" in combined.lower()
