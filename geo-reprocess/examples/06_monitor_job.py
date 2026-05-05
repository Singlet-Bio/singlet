"""
Job monitoring example.

This script demonstrates how to monitor running batch jobs.
"""
import sys
from scgeo import monitor_job
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python 06_monitor_job.py <job_id>")
    print("\nExample:")
    print("  python 06_monitor_job.py 12345")
    sys.exit(1)

job_id = sys.argv[1]

print(f"Monitoring job {job_id}...")
print("Press Ctrl+C to stop monitoring\n")

# Monitor with 30-second refresh
monitor_job(
    job_id=job_id,
    batch_dir=Path("pipeline/batches"),
    refresh_interval=30,
)
