"""
SLURM batch processing module.

Provides functions for submitting, monitoring, and managing batch jobs on
SLURM clusters for large-scale single-cell data processing.
"""

from .api import (
    submit_batch,
    monitor_job,
    cancel_job,
    get_all_jobs,
    list_jobs,
)

__all__ = [
    "submit_batch",
    "monitor_job",
    "cancel_job",
    "get_all_jobs",
    "list_jobs",
]
