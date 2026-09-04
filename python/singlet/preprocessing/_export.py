# SPDX-License-Identifier: MIT
"""Export quantification output to .1pz format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def export_to_1pz(
    quant_dir: str | Path,
    output_path: str | Path,
    *,
    sample_id: Optional[str] = None,
) -> bool:
    """Export simpleaf/alevin quantification to .1pz format.

    Reads the count matrix from ``af_quant/alevin/`` and writes a compressed
    .1pz file (VOCSC + zstd-3).

    Parameters
    ----------
    quant_dir : path
        simpleaf output directory (containing ``af_quant/``).
    output_path : path
        Output .1pz file path.
    sample_id : str, optional
        Sample identifier for logging.

    Returns
    -------
    bool
        True if export succeeded.
    """
    import scipy.io
    import scipy.sparse as sp

    qdir = Path(quant_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Find count matrix
    mtx_candidates = [
        qdir / "af_quant" / "alevin" / "quants_mat.mtx",
        qdir / "af_quant" / "alevin" / "quants_mat.mtx.gz",
    ]
    mtx_path = None
    for p in mtx_candidates:
        if p.exists():
            mtx_path = p
            break

    if mtx_path is None:
        logger.error("No count matrix found in %s", qdir)
        return False

    # Load cells × genes matrix
    mat = scipy.io.mmread(mtx_path)
    if sp.issparse(mat):
        mat = mat.tocsc()
    else:
        mat = sp.csc_matrix(mat)

    # Transpose to genes × cells for .1pz container
    mat_gc = mat.T.tocsc()

    import numpy as np

    from singlet._pz import write_1pz as _native_write

    data = mat_gc.data
    if np.issubdtype(data.dtype, np.floating):
        data = np.round(data).astype(np.int64)
    elif not np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.int64)
    max_val = int(data.max()) if len(data) > 0 else 0
    if max_val <= 255:
        data = data.astype(np.uint8)
    elif max_val <= 65535:
        data = data.astype(np.uint16)
    else:
        data = data.astype(np.uint32)

    _native_write(
        str(out),
        mat_gc.indptr.astype(np.int32),
        mat_gc.indices.astype(np.int32),
        data,
        mat_gc.shape[0],
        mat_gc.shape[1],
    )

    sid = sample_id or out.stem
    logger.info(
        "Exported %s: %d genes × %d cells → %s",
        sid,
        mat_gc.shape[0],
        mat_gc.shape[1],
        out,
    )
    return True


# Keep legacy name for backward compatibility
def export_to_spz(
    quant_dir: str | Path,
    output_path: str | Path,
    *,
    sample_id: Optional[str] = None,
    row_sort: bool = True,
) -> bool:
    """Legacy alias for :func:`export_to_1pz`.

    The ``.spz`` container and its ``singlet._singlepress`` extension were
    retired in 2.0; this writes the current ``.1pz`` container to
    ``output_path`` (whatever its extension) so old call sites keep working.
    ``row_sort`` is accepted for signature compatibility and ignored.
    """
    import warnings

    warnings.warn(
        "export_to_spz() is deprecated; use export_to_1pz() (writes the .1pz container).",
        DeprecationWarning,
        stacklevel=2,
    )
    del row_sort
    return export_to_1pz(quant_dir, output_path, sample_id=sample_id)
