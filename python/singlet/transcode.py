# SPDX-License-Identifier: MIT
"""singlet.transcode — bridge legacy v1 outputs to canonical v2 layout.

The v1 pipeline emits per-feature ``.1pz`` files (``gene_counts.1pz``,
``spliced.1pz`` / ``unspliced.1pz`` / ``ambiguous.1pz``) using the
legacy TP1Z single-block codec. The v2 pipeline emits one
``counts.1pz`` per sample with three named row blocks
(``exon_body``, ``intron_body``, ``junctions``).

This module makes it easy to migrate existing v1 sample directories to
v2 without re-running the pipeline:

.. code-block:: python

    from singlet.transcode import transcode_v1_to_v2
    info = transcode_v1_to_v2("/old/sample", "/new/sample")
    # creates /new/sample/counts.1pz, copies summary.json, ...

CLI::

    python -m singlet.transcode SRC_DIR DST_DIR
    python -m singlet.transcode --inplace SRC_DIR   # writes in place

When ``spliced/unspliced/ambiguous`` legacy files are present, they are
mapped to ``exon_body / intron_body / junctions`` respectively, since
that is the closest semantic equivalent under v2's partition. When only
``gene_counts.1pz`` is present, a single-block ``counts.1pz`` is written
with block name ``gene_counts``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.sparse import csc_matrix

from singlet.pz_v2 import BlockSpec, write_pz_v2

__all__ = ["TranscodeResult", "transcode_v1_to_v2", "main"]


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass
class TranscodeResult:
    """Summary of one v1 → v2 transcode operation."""

    src: Path
    dst: Path
    blocks_written: List[str] = field(default_factory=list)
    files_copied: List[str] = field(default_factory=list)
    n_cells: int = 0
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Legacy reader
# --------------------------------------------------------------------------


def _read_legacy_1pz(path: Path):
    """Read a legacy TP1Z .1pz file. Returns ``(matrix_genes_x_cells,
    var_names, obs_names)``.

    Tries the optional ``singlepress`` accelerator first; falls back to
    the in-tree Python codec.
    """
    try:  # pragma: no cover — optional fast path
        import singlepress

        mat = singlepress.read_1pz(str(path), num_threads=4)
        rownames = list(getattr(mat, "rownames", []) or [])
        colnames = list(getattr(mat, "colnames", []) or [])
        m = csc_matrix(mat) if not isinstance(mat, csc_matrix) else mat
        return m, rownames, colnames
    except (ImportError, AttributeError):
        from singlepress._pz_codec import pz_read

        result = pz_read(str(path), 4)
        m = csc_matrix(
            (result["values"], result["indices"], result["indptr"]),
            shape=(result["m"], result["n"]),
        )
        return m, list(result.get("rownames", []) or []), list(result.get("colnames", []) or [])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


_LEGACY_TO_V2_BLOCK = {
    "gene_counts": "gene_counts",
    "spliced": "exon_body",
    "unspliced": "intron_body",
    "ambiguous": "junctions",
}


_OPTIONAL_PASSTHROUGH = (
    "summary.json",
    "cell_meta.parquet",
    "saturation_curve.tsv",
    "star_Log.final.out",
    "nonhost.json",
    "donor_consensus.fa",
    "donor_variants.vcf.gz",
    "ambient_profile.npy",
    "splice_events.tsv",
)


def transcode_v1_to_v2(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    overwrite: bool = False,
) -> TranscodeResult:
    """Convert one v1 sample directory to canonical v2 layout.

    Parameters
    ----------
    src
        Source directory containing legacy ``.1pz`` files.
    dst
        Destination directory. Created if missing.
    overwrite
        If ``False`` (default) and ``dst/counts.1pz`` already exists,
        raises ``FileExistsError``. If ``True``, overwrites.

    Returns
    -------
    TranscodeResult
        Records what was migrated.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)

    out_counts = dst / "counts.1pz"
    if out_counts.exists() and not overwrite:
        raise FileExistsError(f"{out_counts} exists (pass overwrite=True)")

    result = TranscodeResult(src=src, dst=dst)

    # ---- gather legacy count matrices ----
    blocks: List[BlockSpec] = []
    cell_barcodes: Optional[List[str]] = None
    for legacy_name in ("spliced", "unspliced", "ambiguous", "gene_counts"):
        path = src / f"{legacy_name}.1pz"
        if not path.exists():
            continue
        mat, _gene_names, bcs = _read_legacy_1pz(path)
        if cell_barcodes is None and bcs:
            cell_barcodes = bcs
        elif bcs and bcs != cell_barcodes:
            result.notes.append(
                f"barcode mismatch between blocks; using {legacy_name} order"
            )
            cell_barcodes = bcs
        blocks.append(BlockSpec(_LEGACY_TO_V2_BLOCK[legacy_name], mat))

    if not blocks:
        raise FileNotFoundError(
            f"no legacy .1pz files found under {src} "
            f"(looked for gene_counts/spliced/unspliced/ambiguous)"
        )

    if cell_barcodes is None:
        # Synthesize positional barcodes (legacy file lacked metadata).
        n_cells = blocks[0].matrix.shape[1]
        cell_barcodes = [f"cell_{i}" for i in range(n_cells)]
        result.notes.append("cell barcodes were missing; synthesized positional ids")

    write_pz_v2(out_counts, cell_barcodes=cell_barcodes, blocks=blocks)
    result.blocks_written = [b.name for b in blocks]
    result.n_cells = len(cell_barcodes)

    # ---- pass-through sibling files ----
    for name in _OPTIONAL_PASSTHROUGH:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
            result.files_copied.append(name)

    # ---- minimal summary.json if missing ----
    summary_path = dst / "summary.json"
    if not summary_path.exists():
        summary_path.write_text(
            json.dumps(
                {
                    "sample_id": src.name,
                    "n_cells": result.n_cells,
                    "schema_version": "v2",
                    "transcoded_from": "v1",
                },
                indent=2,
            )
        )
        result.files_copied.append("summary.json (synthesized)")

    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m singlet.transcode",
        description="Transcode legacy v1 sample outputs to canonical v2.",
    )
    p.add_argument("src", help="Source directory (v1 outputs)")
    p.add_argument(
        "dst",
        nargs="?",
        help="Destination directory (v2 outputs). Defaults to SRC when --inplace.",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Write canonical files into SRC, alongside legacy ones.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing counts.1pz in the destination.",
    )
    args = p.parse_args(argv)

    src = Path(args.src)
    if args.inplace:
        dst = src
    elif args.dst is None:
        p.error("DST required unless --inplace")
        return 2  # pragma: no cover
    else:
        dst = Path(args.dst)

    res = transcode_v1_to_v2(src, dst, overwrite=args.overwrite or args.inplace)
    print(f"wrote {dst}/counts.1pz  blocks={res.blocks_written}  n_cells={res.n_cells}")
    if res.files_copied:
        print(f"copied: {', '.join(res.files_copied)}")
    if res.notes:
        for n in res.notes:
            print(f"note: {n}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
