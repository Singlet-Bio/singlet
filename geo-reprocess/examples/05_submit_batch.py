"""
Batch job submission example.

This script demonstrates how to submit batch jobs to SLURM.
"""
from scgeo import submit_batch, list_jobs

# Submit job for filtered catalog
print("Submitting batch job...")
job = submit_batch(
    catalog="human_10x_filtered.csv",
    job_name="human-10x",
    partition="cpu",
    samples_per_batch=50,
    cpus=38,
    memory="128G",
    time="12:00:00",
    max_concurrent=20,
)

if job:
    print(f"\n✓ Job submitted successfully!")
    print(f"  Job ID: {job.job_id}")
    print(f"  Job name: {job.job_name}")
    print(f"  Partition: {job.partition}")
    if job.array_size:
        print(f"  Array size: {job.array_size}")
    print(f"  Script: {job.script_path}")
    
    print(f"\nMonitor with:")
    print(f"  sc-geo batch monitor {job.job_id}")
    
    print(f"\nCheck logs:")
    print(f"  tail -f pipeline/logs/{job.job_name}_*_{job.job_id}.out")
else:
    print("\n✗ Job submission failed!")

# Show all current jobs
print("\n" + "="*60)
print("Current SLURM jobs:")
print("="*60)
list_jobs()
