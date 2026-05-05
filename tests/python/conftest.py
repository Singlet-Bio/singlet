"""
Shared pytest fixtures for the singlet-gpu Python wrapper test suite.

Session-scoped fixtures point at the canonical GSM4037629 test sample on
Clipper NFS.  All GPU tests are automatically skipped when:
  - cupy is not importable, OR
  - no CUDA device is enumerable at cupy.cuda.Device(0), OR
  - the singlet.gpu package is not available.

The environment variable ``SINGLET_GPU_TEST_FIXTURE_DIR`` may override the
default NFS path (for CI or off-cluster development).
"""
from __future__ import annotations

import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Canonical test sample: GSM4037629
# ---------------------------------------------------------------------------
_DEFAULT_GSM4037629_PATH = (
    "/mnt/projects/debruinz_project/singlify_pipeline/"
    "quant/scrna/GSE127/GSE127918/GSM4037629"
)

_EXON_FILENAME = "exon_counts.1pz"


def _resolve_gsm4037629_dir() -> pathlib.Path | None:
    env = os.environ.get("SINGLET_GPU_TEST_FIXTURE_DIR")
    candidates = [env, _DEFAULT_GSM4037629_PATH]
    for c in candidates:
        if c is None:
            continue
        p = pathlib.Path(c)
        if p.is_dir() and (p / _EXON_FILENAME).is_file():
            return p
    return None


@pytest.fixture(scope="session")
def gsm4037629_path() -> pathlib.Path:
    """Session-scoped fixture: absolute path to the GSM4037629 sample dir.

    Skips the test if the directory is not found on the local filesystem.
    Set ``SINGLET_GPU_TEST_FIXTURE_DIR`` to point at an alternative location.
    """
    d = _resolve_gsm4037629_dir()
    if d is None:
        pytest.skip(
            "GSM4037629 pipeline output not found. "
            "Set SINGLET_GPU_TEST_FIXTURE_DIR to a directory containing "
            f"{_EXON_FILENAME}."
        )
    return d


# ---------------------------------------------------------------------------
# GPU availability marker
# ---------------------------------------------------------------------------
def _gpu_available() -> bool:
    try:
        import cupy  # noqa: F401
        import cupy.cuda

        cupy.cuda.Device(0).use()
        return True
    except Exception:  # ImportError, cupy.cuda.runtime.CUDARuntimeError, etc.
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA device not available (cupy import failed or no GPU visible)",
)


# ---------------------------------------------------------------------------
# Lightweight AnnData fixture for torch/integration tests
# ---------------------------------------------------------------------------
@pytest.fixture
def small_anndata():
    """Create a small synthetic AnnData for testing."""
    ad = pytest.importorskip("anndata")
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(42)
    X = sp.random(20, 50, density=0.3, format="csr", random_state=rng,
                  dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(50)]
    adata.obs_names = [f"Cell_{i}" for i in range(20)]
    return adata
