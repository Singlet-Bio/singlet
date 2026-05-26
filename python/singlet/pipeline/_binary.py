# SPDX-License-Identifier: MIT
"""Binary + reference discovery for the singlet pipeline."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Union

from singlet.pipeline._errors import PipelineError


# --------------------------------------------------------------------------
# Binary discovery
# --------------------------------------------------------------------------


def find_binary(explicit: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the ``singlet`` C++ binary path.

    Resolution order:

    1. ``explicit`` argument, if provided.
    2. ``$SINGLET_BINARY`` env var.
    3. First ``singlet`` on ``$PATH``.
    4. Repository-local build outputs (legacy + v2).

    Raises :class:`PipelineError` if no candidate is executable.
    """
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if "SINGLET_BINARY" in os.environ:
        candidates.append(Path(os.environ["SINGLET_BINARY"]))
    on_path = shutil.which("singlet")
    if on_path:
        candidates.append(Path(on_path))
    pkg_root = Path(__file__).resolve().parents[3]  # …/singlet
    repo_root = pkg_root.parent  # …/Singlet-AI
    candidates.extend(
        [
            repo_root / "singlify" / "build" / "singlet",  # legacy: singlify repo (archived)
            pkg_root / "build" / "src" / "pipeline" / "singlet",
            pkg_root / "build" / "singlet",
        ]
    )
    for c in candidates:
        if c and c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    raise PipelineError(
        "Could not locate the singlet binary. Build it with "
        "`cmake --build singlet/build` or set $SINGLET_BINARY."
    )


# --------------------------------------------------------------------------
# Reference discovery
# --------------------------------------------------------------------------


_ORGANISM_TO_BUILD = {
    "human": "GRCh38-2024-A",
    "mouse": "GRCm39-2024-A",
}


def resolve_reference(organism: str, ref_base: Optional[Path]) -> Path:
    """Resolve the reference bundle directory for ``organism``.

    Uses ``ref_base`` if provided, then ``$SINGLET_REF_BASE``, then a
    default cluster location. Falls back to ``ref_base`` itself when the
    expected organism subdirectory is not present so the binary can do
    its own discovery.
    """
    if ref_base is not None:
        base = Path(ref_base)
    elif "SINGLET_REF_BASE" in os.environ:
        base = Path(os.environ["SINGLET_REF_BASE"])
    else:
        base = Path("/mnt/projects/debruinz_project/cellarium/reference")
    if not base.exists():
        raise PipelineError(
            f"Reference base {base} does not exist. Pass `ref_base=` or set "
            "$SINGLET_REF_BASE."
        )
    expected = _ORGANISM_TO_BUILD.get(organism.lower(), organism)
    candidate = base / expected
    return candidate if candidate.exists() else base


# --------------------------------------------------------------------------
# Argv construction
# --------------------------------------------------------------------------


def build_process_argv(
    *,
    binary: Path,
    archive: Path,
    output_dir: Path,
    ref_dir: Path,
    organism: str,
    threads: Optional[int],
    enable_snps: bool,
    enable_pipeline_extras: bool,
    cascade: str,
    te_classify: str,
    nonhost: bool,
    raw_matrix: bool,
    metadata_json: Optional[Path],
    extra_args: Sequence[str],
) -> List[str]:
    """Build the argv for invoking the ``singlet`` binary on a staged archive."""
    argv: List[str] = [str(binary), str(archive)]
    # If ref_dir points at a per-organism subdirectory, hand the binary
    # the parent so it can autodetect siblings; otherwise pass as-is.
    ref_base = ref_dir.parent if ref_dir.name in _ORGANISM_TO_BUILD.values() else ref_dir
    argv += ["--ref-base", str(ref_base)]
    gtf = ref_dir / "genes" / "genes.gtf"
    if gtf.exists():
        argv += ["--exons", str(gtf)]
    if enable_snps:
        snps = ref_dir / "snps" / "common.vcf.gz"
        if snps.exists():
            argv += ["--snps", str(snps)]
    argv += ["--out-prefix", str(output_dir)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    if enable_pipeline_extras:
        argv += ["--pipeline"]
    if cascade and cascade != "off":
        argv += ["--cascade", cascade]
    if te_classify and te_classify != "off":
        argv += ["--te-classify", te_classify]
    if not nonhost:
        argv += ["--no-viral-screen", "--no-microbial-screen"]
    if raw_matrix:
        argv += ["--raw-matrix"]
    if metadata_json is not None:
        argv += ["--metadata-json", str(metadata_json)]
    argv.extend(extra_args)
    return argv
