# SLURM Batch Submission & Monitoring

## Overview

The SLURM module handles array job submission, live monitoring, and result aggregation across HPC clusters.

## Submitting batch jobs

```python
import scgeo

job_id = scgeo.submit_batch(
    catalog=filtered_catalog,
    job_name="singlet_human",
    partition="general",
    cpus=36,
    memory="96G",
    time="4:00:00",
    max_concurrent=20,       # Max simultaneous tasks
    samples_per_batch=50,    # Samples per array element
)
```

## Monitoring

Real-time dashboard that refreshes every 30 seconds:

```python
scgeo.monitor_job(job_id, batch_dir="/path/to/batch")
```

Shows:
- Sample progress: X / total (percentage)
- Success/failure breakdown
- SLURM job status (running/pending/completed)
- Estimated time to completion

## Job management

```python
# List running jobs
scgeo.list_jobs(user="debruinz")

# Cancel a job
scgeo.cancel_job(job_id)
```

## CLI

```bash
sc-geo batch submit --catalog filtered.parquet --partition general --cpus 36
sc-geo batch monitor --job-id 12345
sc-geo batch list
sc-geo batch cancel --job-id 12345
```
