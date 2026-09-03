# SPDX-License-Identifier: MIT
"""Exceptions for ``singlet.pipeline``."""

from __future__ import annotations


class PipelineError(Exception):
    """Raised when the singlet pipeline binary cannot be located, or fails."""
