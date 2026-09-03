# SPDX-License-Identifier: MIT
"""singlet.pipeline — process raw reads into a canonical singlet sample.

Wraps the compiled ``singlet`` C++ binary: resolves an accession, URL, or
local FASTQ/.1fq files to a local archive, builds the argv for the requested
organism/reference, and runs it as a subprocess.

    singlet.run_pipeline("SRR12345678", output_dir="results/")
    singlet.run_pipeline("https://.../SRR12345678", organism="human")

Also available as a CLI once installed: ``singlet-process`` (or
``python -m singlet.pipeline``).

Requires the ``singlet`` binary to be built separately
(``cmake --build build``) or discoverable via ``$SINGLET_BINARY`` / ``$PATH``
— importing this module does not require it; only calling :func:`run` does.
"""

from __future__ import annotations

from singlet.pipeline._errors import PipelineError
from singlet.pipeline._run import Run, run

__all__ = ["PipelineError", "Run", "run"]
