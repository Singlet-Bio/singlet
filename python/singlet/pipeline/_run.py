# SPDX-License-Identifier: MIT
"""End-to-end runner: accession/URL/local reads -> canonical singlet sample.

Resolves the input source to a local ``.1fq``/``.1pz`` archive (downloading
or encoding it via the compiled ``singlet`` binary if needed), builds the
argv for the main pipeline invocation, and runs it as a subprocess.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

from singlet.pipeline._binary import (
    build_process_argv as _build_process_argv,
)
from singlet.pipeline._binary import (
    find_binary as _find_binary,
)
from singlet.pipeline._binary import (
    resolve_reference as _resolve_reference,
)
from singlet.pipeline._errors import PipelineError

Source = Union[str, "Path", Sequence[Union[str, "Path"]]]

_URL_SCHEMES = ("http://", "https://", "ftp://", "s3://")
_ACCESSION_RE = re.compile(r"(SRR|ERR|DRR)\d{6,}")
_ACCESSION_FULL_RE = re.compile(r"^(SRR|ERR|DRR)\d{6,}$")


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def _is_url(value) -> bool:
    """True if ``value`` looks like a download URL (http/https/ftp/s3)."""
    return str(value).lower().startswith(_URL_SCHEMES)


def _looks_like_accession(value) -> bool:
    """True if ``value`` is exactly an SRA/ENA/DDBJ run accession."""
    return bool(_ACCESSION_FULL_RE.match(str(value)))


def _infer_accession(source: Source) -> str:
    """Best-effort accession/name for ``source``, used for default naming."""
    first = source[0] if isinstance(source, (list, tuple)) else source
    s = str(first)
    if _looks_like_accession(s):
        return s
    m = _ACCESSION_RE.search(s)
    if m:
        return m.group(0)
    stem = Path(s).name
    if stem.endswith(".gz"):
        stem = stem[: -len(".gz")]
    stem = Path(stem).stem
    stem = re.sub(r"_[12]$", "", stem)
    return stem


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class Run:
    """Outcome of a :func:`run` invocation."""

    success: bool
    output_dir: Path
    accession: str
    command: List[str] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""

    def __bool__(self) -> bool:
        return self.success


# --------------------------------------------------------------------------
# Subprocess helpers
# --------------------------------------------------------------------------


def _run_subprocess(cmd: Sequence[object], quiet: bool) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise PipelineError(
            f"`{' '.join(cmd)}` exited with status {proc.returncode}"
            + (f": {detail}" if detail else "")
        )
    if not quiet and proc.stdout:
        print(proc.stdout, end="")
    return proc


def _stage_archive(source: Source, binary_path: Path, stage_dir: Path, quiet: bool) -> Path:
    """Return a local archive path for ``source``, downloading/encoding first
    via the binary if ``source`` is a URL, a bare accession, or a list of raw
    read files."""
    if isinstance(source, (list, tuple)):
        dest = stage_dir / f"{_infer_accession(source)}.1fq"
        cmd = [binary_path, "encode", *[str(s) for s in source], "-o", dest]
        _run_subprocess(cmd, quiet)
        return dest

    s = str(source)
    if _is_url(s) or _looks_like_accession(s):
        dest = stage_dir / f"{_infer_accession(s)}.1fq"
        cmd = [binary_path, "download", s, "-o", dest]
        _run_subprocess(cmd, quiet)
        return dest

    path = Path(s)
    if not path.exists():
        raise PipelineError(
            f"Input source {s!r} is not a URL, a recognized SRA/ENA/DDBJ "
            "accession, or an existing local file."
        )
    return path


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------


def run(
    source: Source,
    *,
    output_dir: Union[str, Path],
    binary: Optional[Union[str, Path]] = None,
    ref_base: Optional[Union[str, Path]] = None,
    organism: str = "human",
    threads: Optional[int] = None,
    enable_snps: bool = True,
    enable_pipeline_extras: bool = False,
    cascade: str = "off",
    te_classify: str = "off",
    nonhost: bool = False,
    raw_matrix: bool = False,
    metadata_json: Optional[Union[str, Path]] = None,
    work_dir: Optional[Union[str, Path]] = None,
    keep_intermediate: bool = False,
    extra_args: Sequence[str] = (),
    quiet: bool = False,
) -> Run:
    """Process ``source`` (an accession, URL, local archive, or list of raw
    read files) into a canonical singlet sample under ``output_dir``.

    Runs the compiled ``singlet`` binary as a subprocess; raises
    :class:`PipelineError` if it cannot be located or exits non-zero.
    """
    output_dir = Path(output_dir)
    binary_path = _find_binary(Path(binary) if binary is not None else None)
    ref_dir = _resolve_reference(organism, Path(ref_base) if ref_base is not None else None)
    accession = _infer_accession(source)

    cleanup_dir = None
    if work_dir is not None:
        stage_dir = Path(work_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
    elif keep_intermediate:
        stage_dir = output_dir
        stage_dir.mkdir(parents=True, exist_ok=True)
    else:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="singlet-pipeline-")
        stage_dir = Path(cleanup_dir.name)

    try:
        archive = _stage_archive(source, binary_path, stage_dir, quiet)
        argv = _build_process_argv(
            binary=binary_path,
            archive=archive,
            output_dir=output_dir,
            ref_dir=ref_dir,
            organism=organism,
            threads=threads,
            enable_snps=enable_snps,
            enable_pipeline_extras=enable_pipeline_extras,
            cascade=cascade,
            te_classify=te_classify,
            nonhost=nonhost,
            raw_matrix=raw_matrix,
            metadata_json=Path(metadata_json) if metadata_json is not None else None,
            extra_args=tuple(extra_args),
        )
        proc = _run_subprocess(argv, quiet)
        return Run(
            success=True,
            output_dir=output_dir,
            accession=accession,
            command=[str(a) for a in argv],
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()
