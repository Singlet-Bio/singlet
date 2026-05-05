"""
SLURM job monitoring and progress dashboard.

Provides real-time monitoring of batch jobs, result aggregation,
and progress reporting.
"""
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from scgeo.config import get_config


@dataclass
class JobInfo:
    """Information about a running SLURM job."""
    job_id: str
    job_name: str
    partition: str
    state: str
    node: Optional[str]
    time_elapsed: str
    array_task_id: Optional[int] = None


@dataclass
class BatchProgress:
    """Progress statistics for a batch job."""
    total_samples: int
    completed_samples: int
    failed_samples: int
    running_jobs: int
    pending_jobs: int
    completed_jobs: int
    failed_jobs: int
    success_rate: float
    elapsed_time: timedelta
    estimated_remaining: Optional[timedelta]


def get_job_status(job_id: str, user: Optional[str] = None) -> List[JobInfo]:
    """
    Query SLURM for job status.
    
    Args:
        job_id: Job ID to query
        user: Username filter (default: current user)
        
    Returns:
        List of JobInfo objects
    """
    cmd = [
        "squeue",
        "-u", user or subprocess.getoutput("whoami"),
        "-j", job_id,
        "--format=%A|%i|%P|%j|%T|%N|%M",
        "--noheader",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 7:
                continue
            
            job_array_id, task_id, partition, name, state, node, elapsed = parts
            
            # Parse array task ID if present
            array_task_id = None
            if "_" in job_array_id:
                array_task_id = int(task_id) if task_id.isdigit() else None
            
            jobs.append(JobInfo(
                job_id=job_array_id,
                job_name=name,
                partition=partition,
                state=state,
                node=node if node != "(None)" else None,
                time_elapsed=elapsed,
                array_task_id=array_task_id,
            ))
        
        return jobs
        
    except subprocess.CalledProcessError:
        return []


def get_all_jobs(user: Optional[str] = None) -> List[JobInfo]:
    """
    Get all SLURM jobs for a user.
    
    Args:
        user: Username filter (default: current user)
        
    Returns:
        List of JobInfo objects
    """
    cmd = [
        "squeue",
        "-u", user or subprocess.getoutput("whoami"),
        "--format=%A|%i|%P|%j|%T|%N|%M",
        "--noheader",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 7:
                continue
            
            job_array_id, task_id, partition, name, state, node, elapsed = parts
            
            # Parse array task ID if present
            array_task_id = None
            if "_" in task_id:
                array_task_id = int(task_id.split("_")[-1]) if task_id.split("_")[-1].isdigit() else None
            
            jobs.append(JobInfo(
                job_id=job_array_id,
                job_name=name,
                partition=partition,
                state=state,
                node=node if node != "(None)" else None,
                time_elapsed=elapsed,
                array_task_id=array_task_id,
            ))
        
        return jobs
        
    except subprocess.CalledProcessError:
        return []


def collect_results(
    batch_dir: Path,
    config=None,
) -> pd.DataFrame:
    """
    Collect results from completed batch jobs.
    
    Args:
        batch_dir: Directory containing batch result files
        config: Configuration object
        
    Returns:
        DataFrame with aggregated results
    """
    if config is None:
        config = get_config()
    
    result_files = list(batch_dir.glob("batch_*_results.csv"))
    
    if not result_files:
        return pd.DataFrame()
    
    dfs = []
    for f in result_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def calculate_progress(
    batch_dir: Path,
    job_id: str,
    start_time: datetime,
    config=None,
) -> BatchProgress:
    """
    Calculate progress statistics for a batch job.
    
    Args:
        batch_dir: Directory containing batch files
        job_id: Job ID to monitor
        start_time: Job start time
        config: Configuration object
        
    Returns:
        BatchProgress object
    """
    if config is None:
        config = get_config()
    
    # Count total samples
    batch_files = list(batch_dir.glob("batch_*.csv"))
    total_samples = 0
    for f in batch_files:
        if f.name.endswith("_results.csv"):
            continue
        try:
            df = pd.read_csv(f)
            total_samples += len(df)
        except:
            pass
    
    # Get job status
    jobs = get_job_status(job_id)
    
    running_jobs = sum(1 for j in jobs if j.state == "RUNNING")
    pending_jobs = sum(1 for j in jobs if j.state == "PENDING")
    
    # Count completed samples
    results_df = collect_results(batch_dir, config)
    completed_samples = len(results_df)
    failed_samples = len(results_df[results_df["status"] == "failed"]) if len(results_df) > 0 else 0
    
    # Calculate completion rate
    completed_jobs = len(list(batch_dir.glob("batch_*_results.csv")))
    failed_jobs = 0  # TODO: Track failed jobs separately
    
    success_rate = (
        (completed_samples - failed_samples) / completed_samples
        if completed_samples > 0
        else 0.0
    )
    
    # Estimate remaining time
    elapsed = datetime.now() - start_time
    estimated_remaining = None
    
    if completed_samples > 0 and total_samples > completed_samples:
        rate = completed_samples / elapsed.total_seconds()
        remaining_samples = total_samples - completed_samples
        estimated_remaining = timedelta(seconds=remaining_samples / rate)
    
    return BatchProgress(
        total_samples=total_samples,
        completed_samples=completed_samples,
        failed_samples=failed_samples,
        running_jobs=running_jobs,
        pending_jobs=pending_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        success_rate=success_rate,
        elapsed_time=elapsed,
        estimated_remaining=estimated_remaining,
    )


def display_dashboard(
    progress: BatchProgress,
    job_id: str,
    partition: str,
    refresh_interval: int = 30,
) -> None:
    """
    Display a progress dashboard.
    
    Args:
        progress: BatchProgress object
        job_id: Job ID
        partition: SLURM partition
        refresh_interval: Refresh interval in seconds
    """
    pct_complete = (
        progress.completed_samples / progress.total_samples * 100
        if progress.total_samples > 0
        else 0.0
    )
    
    print("\n" + "=" * 80)
    print(f"SLURM Job Monitor - Job {job_id} ({partition})")
    print("=" * 80)
    print()
    print(f"Samples:       {progress.completed_samples:,} / {progress.total_samples:,} ({pct_complete:.1f}%)")
    print(f"  ✓ Success:   {progress.completed_samples - progress.failed_samples:,}")
    print(f"  ✗ Failed:    {progress.failed_samples:,}")
    print()
    print(f"Jobs:")
    print(f"  Running:     {progress.running_jobs}")
    print(f"  Pending:     {progress.pending_jobs}")
    print(f"  Completed:   {progress.completed_jobs}")
    print()
    print(f"Success Rate:  {progress.success_rate * 100:.1f}%")
    print(f"Elapsed:       {str(progress.elapsed_time).split('.')[0]}")
    
    if progress.estimated_remaining:
        print(f"Est. Remain:   {str(progress.estimated_remaining).split('.')[0]}")
    
    print()
    print(f"Refreshing every {refresh_interval}s... (Ctrl+C to stop)")
    print("=" * 80)


def monitor_job(
    job_id: str,
    batch_dir: Path,
    start_time: Optional[datetime] = None,
    refresh_interval: int = 30,
    config=None,
) -> None:
    """
    Monitor a SLURM job with live updates.
    
    Args:
        job_id: Job ID to monitor
        batch_dir: Directory containing batch files
        start_time: Job start time (default: now)
        refresh_interval: Refresh interval in seconds
        config: Configuration object
    """
    if config is None:
        config = get_config()
    
    if start_time is None:
        start_time = datetime.now()
    
    # Get partition from first job
    jobs = get_job_status(job_id)
    partition = jobs[0].partition if jobs else "unknown"
    
    try:
        while True:
            # Calculate progress
            progress = calculate_progress(batch_dir, job_id, start_time, config)
            
            # Display dashboard
            display_dashboard(progress, job_id, partition, refresh_interval)
            
            # Check if job is complete
            jobs = get_job_status(job_id)
            if not jobs and progress.completed_jobs > 0:
                print("\n✓ Job complete!")
                break
            
            # Wait before next refresh
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def get_job_logs(
    job_id: str,
    log_dir: Path,
    array_task_id: Optional[int] = None,
    tail_lines: int = 50,
) -> Dict[str, str]:
    """
    Get log output for a job.
    
    Args:
        job_id: Job ID
        log_dir: Directory containing log files
        array_task_id: Array task ID (for array jobs)
        tail_lines: Number of lines to return from end of log
        
    Returns:
        Dict with 'stdout' and 'stderr' keys
    """
    logs = {"stdout": "", "stderr": ""}
    
    # Find log files
    if array_task_id is not None:
        pattern = f"*_{array_task_id}_{job_id}.out"
    else:
        pattern = f"*_{job_id}.out"
    
    out_files = list(log_dir.glob(pattern))
    
    if out_files:
        out_file = out_files[0]
        err_file = out_file.with_suffix(".err")
        
        # Read stdout
        if out_file.exists():
            lines = out_file.read_text().split("\n")
            logs["stdout"] = "\n".join(lines[-tail_lines:])
        
        # Read stderr
        if err_file.exists():
            lines = err_file.read_text().split("\n")
            logs["stderr"] = "\n".join(lines[-tail_lines:])
    
    return logs
