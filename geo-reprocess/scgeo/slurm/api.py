"""
High-level API for SLURM batch processing.

Provides convenient functions for submitting and monitoring jobs.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .submit import JobSubmission, submit_batch as _submit_batch, cancel_job
from .monitor import monitor_job as _monitor_job, get_all_jobs, calculate_progress


__all__ = [
    "submit_batch",
    "monitor_job", 
    "cancel_job",
    "get_all_jobs",
    "list_jobs",
]


def submit_batch(
    catalog: Union[str, Path],
    job_name: str = "scgeo-batch",
    partition: str = "cpu",
    samples_per_batch: int = 50,
    cpus: int = 38,
    memory: str = "128G",
    time: str = "12:00:00",
    max_concurrent: int = 20,
    dry_run: bool = False,
) -> Optional[JobSubmission]:
    """
    Submit a batch processing job for a catalog.
    
    This is the main entry point for submitting large-scale processing jobs
    to a SLURM cluster. It automatically splits the catalog into batches,
    generates a SLURM array job script, and submits it.
    
    Args:
        catalog: Path to catalog CSV or parquet file
        job_name: Name for the job (default: "scgeo-batch")
        partition: SLURM partition (cpu, bigmem, gpu)
        samples_per_batch: Number of samples per batch (default: 50)
        cpus: CPUs per task (default: 38)
        memory: Memory per node, e.g., "128G" (default: "128G")
        time: Time limit in HH:MM:SS format (default: "12:00:00")
        max_concurrent: Maximum concurrent array tasks (default: 20)
        dry_run: If True, generate script without submitting (default: False)
        
    Returns:
        JobSubmission object with job details, or None if submission failed
        
    Example:
        >>> from scgeo.slurm import submit_batch
        >>> 
        >>> # Submit 10x human samples to CPU partition
        >>> job = submit_batch(
        ...     catalog="catalog_10x_human.csv",
        ...     job_name="10x-human",
        ...     partition="cpu",
        ... )
        >>> print(f"Job ID: {job.job_id}")
        >>> 
        >>> # Submit with custom resources
        >>> job = submit_batch(
        ...     catalog="catalog_large_memory.csv",
        ...     partition="bigmem",
        ...     memory="256G",
        ...     time="24:00:00",
        ... )
    """
    catalog_path = Path(catalog)
    
    return _submit_batch(
        catalog_file=catalog_path,
        job_name=job_name,
        partition=partition,
        samples_per_batch=samples_per_batch,
        cpus=cpus,
        memory=memory,
        time=time,
        max_concurrent=max_concurrent,
        dry_run=dry_run,
    )


def monitor_job(
    job_id: str,
    batch_dir: Union[str, Path],
    refresh_interval: int = 30,
) -> None:
    """
    Monitor a SLURM job with live updates.
    
    Displays a real-time dashboard showing:
    - Sample completion progress
    - Job status (running, pending, completed)
    - Success/failure rates
    - Estimated time remaining
    
    Args:
        job_id: SLURM job ID to monitor
        batch_dir: Directory containing batch files
        refresh_interval: Refresh interval in seconds (default: 30)
        
    Example:
        >>> from scgeo.slurm import monitor_job
        >>> 
        >>> # Monitor a running job
        >>> monitor_job(
        ...     job_id="12345",
        ...     batch_dir="pipeline/batches",
        ...     refresh_interval=60,
        ... )
        
        # Output:
        # ================================================================================
        # SLURM Job Monitor - Job 12345 (cpu)
        # ================================================================================
        # 
        # Samples:       1,250 / 5,000 (25.0%)
        #   ✓ Success:   1,180
        #   ✗ Failed:    70
        # 
        # Jobs:
        #   Running:     18
        #   Pending:     32
        #   Completed:   50
        # 
        # Success Rate:  94.4%
        # Elapsed:       2:15:30
        # Est. Remain:   6:45:00
        # 
        # Refreshing every 60s... (Ctrl+C to stop)
        # ================================================================================
    """
    batch_path = Path(batch_dir)
    
    _monitor_job(
        job_id=job_id,
        batch_dir=batch_path,
        start_time=datetime.now(),
        refresh_interval=refresh_interval,
    )


def list_jobs(user: Optional[str] = None) -> None:
    """
    List all SLURM jobs for the current user.
    
    Args:
        user: Username to filter by (default: current user)
        
    Example:
        >>> from scgeo.slurm import list_jobs
        >>> list_jobs()
        
        # Output:
        # Job ID    Name              Partition  State     Node    Time
        # --------  ----------------  ---------  --------  ------  --------
        # 12345     scgeo-batch       cpu        RUNNING   c005    02:15:30
        # 12345_0   scgeo-batch       cpu        RUNNING   c005    02:15:28
        # 12345_1   scgeo-batch       cpu        RUNNING   c006    02:15:25
        # 12345_2   scgeo-batch       cpu        PENDING   -       00:00:00
    """
    jobs = get_all_jobs(user)
    
    if not jobs:
        print("No jobs found.")
        return
    
    # Print header
    print(f"{'Job ID':<10} {'Name':<18} {'Partition':<11} {'State':<10} {'Node':<8} {'Time':<10}")
    print("-" * 80)
    
    # Print jobs
    for job in jobs:
        node = job.node or "-"
        print(
            f"{job.job_id:<10} {job.job_name:<18} {job.partition:<11} "
            f"{job.state:<10} {node:<8} {job.time_elapsed:<10}"
        )
    
    # Print summary
    print()
    print(f"Total jobs: {len(jobs)}")
    print(f"  Running:  {sum(1 for j in jobs if j.state == 'RUNNING')}")
    print(f"  Pending:  {sum(1 for j in jobs if j.state == 'PENDING')}")
