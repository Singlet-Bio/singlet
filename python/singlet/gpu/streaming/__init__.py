# SPDX-License-Identifier: MIT
"""
singlet.gpu.streaming — Streaming end-to-end pipeline wrapper.

Exposes:
    run_pipeline(input_paths, ...) → PipelineResult

Underlying C++ cycle: cycle 7 (streaming/streamed_pipeline.h —
``singlet::gpu::streaming::StreamedPipeline``).

This module provides a high-level interface for running the full
singlet-gpu processing pipeline over multiple .1pz files without
materialising all data simultaneously.  Since the output spans
multiple inputs and does not map cleanly to a single AnnData, the
result is a ``PipelineResult`` dataclass.
"""

from .pipeline import run_pipeline, PipelineResult

__all__ = ["run_pipeline", "PipelineResult"]
