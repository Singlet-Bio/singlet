"""Tests for SLURM module."""
import pytest
from pathlib import Path
from scgeo.slurm import submit_batch
from scgeo.slurm.submit import generate_batch_script, JobSubmission
from scgeo.slurm.monitor import JobInfo, BatchProgress


def test_job_submission_dataclass():
    """Test JobSubmission structure."""
    job = JobSubmission(
        job_id="12345",
        job_name="test-job",
        partition="cpu",
        array_size=100,
        batch_file=Path("/tmp/batch.csv"),
        script_path=Path("/tmp/script.sh"),
    )
    
    assert job.job_id == "12345"
    assert job.partition == "cpu"
    assert job.array_size == 100


def test_job_info_dataclass():
    """Test JobInfo structure."""
    job = JobInfo(
        job_id="12345",
        job_name="test-job",
        partition="cpu",
        state="RUNNING",
        node="c005",
        time_elapsed="01:23:45",
        array_task_id=5,
    )
    
    assert job.job_id == "12345"
    assert job.state == "RUNNING"
    assert job.array_task_id == 5


def test_batch_progress_dataclass():
    """Test BatchProgress structure."""
    from datetime import timedelta
    
    progress = BatchProgress(
        total_samples=5000,
        completed_samples=1250,
        failed_samples=50,
        running_jobs=18,
        pending_jobs=32,
        completed_jobs=50,
        failed_jobs=2,
        success_rate=0.96,
        elapsed_time=timedelta(hours=2, minutes=15),
        estimated_remaining=timedelta(hours=6, minutes=45),
    )
    
    assert progress.total_samples == 5000
    assert progress.completed_samples == 1250
    assert progress.success_rate == 0.96


def test_generate_batch_script(test_config, tmp_path):
    """Test SLURM script generation."""
    batch_file = tmp_path / "batch_00000.csv"
    batch_file.write_text("gsm_id,gse_id,organism\nGSM123,GSE456,Homo sapiens\n")
    
    script_path = generate_batch_script(
        batch_file=batch_file,
        job_name="test-job",
        partition="cpu",
        cpus=4,
        memory="16G",
        time="1:00:00",
        array_size=10,
        max_concurrent=5,
        config=test_config,
    )
    
    assert script_path.exists()
    assert script_path.suffix == ".sh"
    
    # Check script content
    content = script_path.read_text()
    assert "#SBATCH --job-name=test-job" in content
    assert "#SBATCH --partition=cpu" in content
    assert "#SBATCH --cpus-per-task=4" in content
    assert "#SBATCH --mem=16G" in content
    assert "#SBATCH --array=0-9%5" in content


def test_submit_batch_dry_run(test_config, tmp_path, sample_catalog_data):
    """Test batch submission in dry-run mode."""
    import pandas as pd
    
    # Create test catalog
    catalog_file = tmp_path / "catalog.csv"
    df = pd.DataFrame(sample_catalog_data)
    df.to_csv(catalog_file, index=False)
    
    # Dry run should not actually submit
    job = submit_batch(
        catalog=catalog_file,
        job_name="test-batch",
        partition="cpu",
        samples_per_batch=1,
        dry_run=True,
    )
    
    # Dry run returns None
    assert job is None


# Note: Actual job submission tests require SLURM access
# These should be in separate integration test suite
