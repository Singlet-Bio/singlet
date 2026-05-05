"""
SLURM job submission wrapper.

Provides functions for submitting batch jobs with array job support,
dependency management, and resource configuration.
"""
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from scgeo.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class JobSubmission:
    """Result of a job submission."""
    job_id: str
    job_name: str
    partition: str
    array_size: Optional[int]
    batch_file: Optional[Path]
    script_path: Path


def generate_batch_script(
    batch_file: Path,
    job_name: str,
    partition: str = "cpu",
    cpus: int = 38,
    memory: str = "128G",
    time: str = "12:00:00",
    array_size: Optional[int] = None,
    max_concurrent: int = 20,
    config=None,
) -> Path:
    """
    Generate a SLURM batch script for processing samples.
    
    Args:
        batch_file: Path to batch CSV file
        job_name: SLURM job name
        partition: SLURM partition (cpu, bigmem, gpu)
        cpus: CPUs per task
        memory: Memory per node (e.g., "128G")
        time: Time limit (HH:MM:SS)
        array_size: Array size for array jobs (None for single job)
        max_concurrent: Maximum concurrent array tasks
        config: Configuration object
        
    Returns:
        Path to generated script
    """
    if config is None:
        config = get_config()
    
    log_dir = config.paths.pipeline_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = config.paths.pipeline_dir / f"slurm_{job_name}.sh"
    
    # Build SBATCH directives
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time}",
    ]
    
    # GPU partition requires --gres even when using it for CPU access
    if partition == "gpu":
        sbatch_lines.append("#SBATCH --gres=gpu:1")
    
    if array_size:
        sbatch_lines.append(f"#SBATCH --array=0-{array_size-1}%{max_concurrent}")
        sbatch_lines.append(f"#SBATCH --output={log_dir}/{job_name}_%a_%j.out")
        sbatch_lines.append(f"#SBATCH --error={log_dir}/{job_name}_%a_%j.err")
    else:
        sbatch_lines.append(f"#SBATCH --output={log_dir}/{job_name}_%j.out")
        sbatch_lines.append(f"#SBATCH --error={log_dir}/{job_name}_%j.err")
    
    # Build script body
    workspace_dir = config.paths.workspace
    
    script_body = [
        "",
        "set -euo pipefail",
        "",
        "# Source environment — module adds conda to PATH, then source profile.d for shell functions",
        "module load miniconda3/25.5.1",
        "source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh",
        "conda activate cellarium",
        f"export SCGEO_BASE=\"{config.paths.project_base}\"",
        f"export SCGEO_WORKSPACE=\"{workspace_dir}\"",
        f"export ALEVIN_FRY_HOME=\"{config.paths.project_base}/af_home\"",
        "ulimit -n 2048",
        "",
        "# Determine batch file",
    ]
    
    if array_size:
        script_body.extend([
            f'export BATCH_FILE="{batch_file.parent}/batch_$(printf %05d $SLURM_ARRAY_TASK_ID).csv"',
        ])
    else:
        script_body.extend([
            f'export BATCH_FILE="{batch_file}"',
        ])
    
    script_body.extend([
        "",
        "if [[ ! -f $BATCH_FILE ]]; then",
        '    echo "ERROR: Batch file not found: $BATCH_FILE"',
        "    exit 1",
        "fi",
        "",
        f'echo "════════════════════════════════════════════════════"',
        f'echo "Job: {job_name}"',
        'echo "Batch: $BATCH_FILE"',
        'echo "Node: $(hostname)"',
        'echo "CPUs: $SLURM_CPUS_PER_TASK"',
        'echo "Memory: $SLURM_MEM_PER_NODE MB"',
        'echo "Job ID: $SLURM_JOB_ID"',
        'echo "════════════════════════════════════════════════════"',
        "",
        "# Process batch with Python heredoc",
        f"cd {workspace_dir}",
        "python3 << 'PYEOF'",
        "import sys",
        "import os",
        "import logging",
        "import pandas as pd",
        "",
        "logging.basicConfig(",
        "    level=logging.INFO,",
        '    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",',
        ")",
        "",
        f'sys.path.insert(0, "{workspace_dir}")',
        "",
        "from scgeo.pipeline.api import process_samples",
        "",
        f'batch_file = os.environ.get("BATCH_FILE", "{batch_file}")',
        "if not os.path.isfile(batch_file):",
        '    # Reconstruct from SLURM vars',
        "    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))",
        f'    batch_file = f"{batch_file.parent}/batch_{{task_id:05d}}.csv"',
        "",
        "df = pd.read_csv(batch_file)",
        'print(f"Processing {{len(df)}} samples from {{batch_file}}")',
        "",
        "# Build sample dicts for process_samples",
        "samples = []",
        "for _, row in df.iterrows():",
        "    sample = {",
        "        'gsm_id': row['gsm_id'],",
        "        'gse_id': row['gse_id'],",
        "        'organism': row.get('organism', row.get('scientific_name', '')),",
        "    }",
        "    if 'ena_fastq_r1' in row and pd.notna(row.get('ena_fastq_r1')):",
        "        sample['ena_r1_url'] = row['ena_fastq_r1']",
        "    if 'ena_fastq_r2' in row and pd.notna(row.get('ena_fastq_r2')):",
        "        sample['ena_r2_url'] = row['ena_fastq_r2']",
        "    if 'srr_accessions' in row and pd.notna(row.get('srr_accessions')):",
        "        sample['srr_accession'] = str(row['srr_accessions'])",
        "    # Pass catalog protocol hint to override read-length heuristics",
        "    for col in ('protocol_inferred', 'protocol'):",
        "        if col in row.index and pd.notna(row.get(col)):",
        "            sample['protocol_hint'] = row[col]",
        "            break",
        "    # Pass per-sample chemistry if catalog provides it (overrides PROTOCOL_CHEMISTRY)",
        "    if 'simpleaf_chemistry' in row.index and pd.notna(row.get('simpleaf_chemistry')):",
        "        sample['chemistry_hint'] = row['simpleaf_chemistry']",
        "    samples.append(sample)",
        "",
        "results = process_samples(samples=samples)",
        "",
        "# Save results",
        "results_df = pd.DataFrame([r.to_dict() for r in results])",
        'output_file = batch_file.replace(".csv", "_results.csv")',
        "results_df.to_csv(output_file, index=False)",
        'print(f"Results saved to {output_file}")',
        "",
        "n_ok = sum(1 for r in results if r.status in ('success', 'qc_warn'))",
        "n_skip = sum(1 for r in results if r.status == 'skipped')",
        "n_fail = sum(1 for r in results if r.status == 'failed')",
        'print(f"Summary: {n_ok} success, {n_skip} skipped, {n_fail} failed")',
        "PYEOF",
        "",
        'echo "Job complete"',
    ])
    
    # Write script
    script_content = "\n".join(sbatch_lines + script_body)
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    logger.info(f"Generated SLURM script: {script_path}")
    return script_path


def generate_autograb_script(
    batch_dir: Path,
    job_name: str,
    partition: str = "cpu",
    cpus: int = 38,
    memory: str = "128G",
    time: str = "24:00:00",
    n_workers: int = 75,
    max_concurrent: int = 20,
    config=None,
) -> Path:
    """
    Generate a SLURM auto-grab batch script.

    Each worker loops: atomically lock the next available batch via ``mkdir``
    (POSIX-atomic), process it, then grab the next one until no batches remain.
    Workers exit gracefully when all batches are done.

    Args:
        batch_dir: Directory containing batch CSVs (batches_v9_autograb/)
        job_name: SLURM job name
        partition: SLURM partition
        cpus: CPUs per task
        memory: Memory per node
        time: Wall-clock limit per worker (should be generous; workers loop)
        n_workers: Number of SLURM array tasks to launch
        max_concurrent: Maximum concurrent array tasks (SLURM %N)
        config: Configuration object

    Returns:
        Path to generated SLURM script
    """
    if config is None:
        config = get_config()

    log_dir = config.paths.pipeline_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script_path = config.paths.pipeline_dir / f"slurm_{job_name}.sh"
    workspace_dir = config.paths.workspace

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --array=0-{n_workers - 1}%{max_concurrent}
#SBATCH --output={log_dir}/{job_name}_%a_%j.out
#SBATCH --error={log_dir}/{job_name}_%a_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────
module load miniconda3/25.5.1
source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
conda activate cellarium
export SCGEO_BASE="{config.paths.project_base}"
export SCGEO_WORKSPACE="{workspace_dir}"
export ALEVIN_FRY_HOME="{config.paths.project_base}/af_home"
ulimit -n 2048

BATCH_DIR="{batch_dir}"
BATCHES_PROCESSED=0

echo "════════════════════════════════════════════════════"
echo "Auto-grab worker $SLURM_ARRAY_TASK_ID starting"
echo "Node: $(hostname)"
echo "Batch dir: $BATCH_DIR"
echo "════════════════════════════════════════════════════"

# ── Auto-grab loop ───────────────────────────────────────────────────
while true; do
    BATCH_FILE=""

    # Scan for next available batch (sorted order = priority order)
    for f in $(ls -1 "$BATCH_DIR"/batch_*.csv 2>/dev/null | sort); do
        LOCK_DIR="${{f%.csv}}.lock"
        DONE_FILE="${{f%.csv}}_results.csv"

        # Skip completed batches (results CSV exists)
        [[ -f "$DONE_FILE" ]] && continue

        # Attempt atomic lock (mkdir is atomic on POSIX)
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            # Write ownership info for debugging / stale-lock cleanup
            echo "$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID@$(hostname) $(date -Iseconds)" > "$LOCK_DIR/owner"
            BATCH_FILE="$f"
            break
        fi
    done

    # No more batches — exit gracefully
    if [[ -z "$BATCH_FILE" ]]; then
        echo "No unlocked batches remaining — worker $SLURM_ARRAY_TASK_ID exiting after $BATCHES_PROCESSED batches"
        break
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Grabbed: $BATCH_FILE"
    echo "Worker: $SLURM_ARRAY_TASK_ID  |  Node: $(hostname)  |  Batch #$((BATCHES_PROCESSED + 1))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Process this batch ───────────────────────────────────────────
    cd {workspace_dir}
    python3 << PYEOF
import sys, os, logging, pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

sys.path.insert(0, "{workspace_dir}")
from scgeo.pipeline.api import process_samples

batch_file = "$BATCH_FILE"
df = pd.read_csv(batch_file)
print(f"Processing {{len(df)}} samples from {{batch_file}}")

samples = []
for _, row in df.iterrows():
    sample = {{
        'gsm_id': row['gsm_id'],
        'gse_id': row['gse_id'],
        'organism': row.get('organism', row.get('scientific_name', '')),
    }}
    if 'ena_fastq_r1' in row and pd.notna(row.get('ena_fastq_r1')):
        sample['ena_r1_url'] = row['ena_fastq_r1']
    if 'ena_fastq_r2' in row and pd.notna(row.get('ena_fastq_r2')):
        sample['ena_r2_url'] = row['ena_fastq_r2']
    if 'srr_accessions' in row and pd.notna(row.get('srr_accessions')):
        sample['srr_accession'] = str(row['srr_accessions'])
    for col in ('protocol_inferred', 'protocol'):
        if col in row.index and pd.notna(row.get(col)):
            sample['protocol_hint'] = row[col]
            break
    if 'simpleaf_chemistry' in row.index and pd.notna(row.get('simpleaf_chemistry')):
        sample['chemistry_hint'] = row['simpleaf_chemistry']
    samples.append(sample)

results = process_samples(samples=samples)

results_df = pd.DataFrame([r.to_dict() for r in results])
output_file = batch_file.replace(".csv", "_results.csv")
results_df.to_csv(output_file, index=False)
print(f"Results saved to {{output_file}}")

n_ok = sum(1 for r in results if r.status in ('success', 'qc_warn'))
n_skip = sum(1 for r in results if r.status == 'skipped')
n_fail = sum(1 for r in results if r.status == 'failed')
print(f"Summary: {{n_ok}} success, {{n_skip}} skipped, {{n_fail}} failed")
PYEOF

    BATCHES_PROCESSED=$((BATCHES_PROCESSED + 1))
    echo "Batch complete. Total batches processed by this worker: $BATCHES_PROCESSED"
done

echo "Worker $SLURM_ARRAY_TASK_ID finished. Processed $BATCHES_PROCESSED batches total."
"""

    script_path.write_text(script)
    script_path.chmod(0o755)

    logger.info(f"Generated auto-grab SLURM script: {script_path}")
    logger.info(f"  Workers: {n_workers}, max concurrent: {max_concurrent}")
    return script_path


def submit_job(
    script_path: Path,
    dependency: Optional[str] = None,
    dry_run: bool = False,
) -> Optional[JobSubmission]:
    """
    Submit a SLURM job.
    
    Args:
        script_path: Path to batch script
        dependency: Dependency specification (e.g., "afterok:12345")
        dry_run: If True, only print command without submitting
        
    Returns:
        JobSubmission object or None if dry_run
    """
    cmd = ["sbatch"]
    
    if dependency:
        cmd.extend(["--dependency", dependency])
    
    cmd.append(str(script_path))
    
    if dry_run:
        logger.info(f"Dry run: {' '.join(cmd)}")
        return None
    
    logger.info(f"Submitting: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Parse job ID from output: "Submitted batch job 12345"
        output = result.stdout.strip()
        job_id = output.split()[-1]
        
        logger.info(f"✓ Job submitted: {job_id}")
        
        # Parse script to get job details
        script_content = script_path.read_text()
        job_name = ""
        partition = ""
        array_size = None
        
        for line in script_content.split("\n"):
            if line.startswith("#SBATCH --job-name="):
                job_name = line.split("=")[1]
            elif line.startswith("#SBATCH --partition="):
                partition = line.split("=")[1]
            elif line.startswith("#SBATCH --array="):
                array_spec = line.split("=")[1]
                # Parse "0-99%20" format
                range_part = array_spec.split("%")[0]
                start, end = range_part.split("-")
                array_size = int(end) - int(start) + 1
        
        return JobSubmission(
            job_id=job_id,
            job_name=job_name,
            partition=partition,
            array_size=array_size,
            batch_file=None,
            script_path=script_path,
        )
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return None


def submit_batch(
    catalog_file: Path,
    job_name: str = "scgeo-batch",
    partition: str = "cpu",
    array_size: int = 100,
    samples_per_batch: int = 50,
    cpus: int = 38,
    memory: str = "128G",
    time: str = "12:00:00",
    max_concurrent: int = 20,
    dry_run: bool = False,
    config=None,
) -> Optional[JobSubmission]:
    """
    Submit a batch processing job for a catalog.
    
    Args:
        catalog_file: Path to catalog CSV/parquet file
        job_name: Job name
        partition: SLURM partition
        array_size: Number of array tasks
        samples_per_batch: Samples per batch
        cpus: CPUs per task
        memory: Memory per node
        time: Time limit
        max_concurrent: Max concurrent array tasks
        dry_run: If True, only generate script without submitting
        config: Configuration object
        
    Returns:
        JobSubmission object or None
    """
    if config is None:
        config = get_config()
    
    # Create batches directory
    batch_dir = config.paths.pipeline_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Load catalog and create batches
    import pandas as pd
    
    if catalog_file.suffix == ".parquet":
        df = pd.read_parquet(catalog_file)
    else:
        df = pd.read_csv(catalog_file)
    
    logger.info(f"Loaded catalog: {len(df):,} samples")
    
    # Normalize column names — catalog uses 'scientific_name', batch needs 'organism'
    if 'scientific_name' in df.columns and 'organism' not in df.columns:
        df['organism'] = df['scientific_name']
    
    # Ensure required columns exist
    required = ['gsm_id', 'gse_id', 'organism']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Catalog missing required columns: {missing}")
        logger.error(f"Available columns: {list(df.columns)}")
        return None
    
    # Determine which columns to keep in batch files
    batch_cols = ['gsm_id', 'gse_id', 'organism']
    optional_cols = ['fastq_r1_url', 'fastq_r2_url', 'ena_fastq_r1', 'ena_fastq_r2', 
                     'srr_accessions', 'protocol', 'protocol_inferred', 'matched_taxon_id']
    for col in optional_cols:
        if col in df.columns:
            batch_cols.append(col)
    
    # Rename ENA columns if needed for consistency
    if 'fastq_r1_url' in df.columns and 'ena_fastq_r1' not in df.columns:
        df['ena_fastq_r1'] = df['fastq_r1_url']
        if 'ena_fastq_r1' not in batch_cols:
            batch_cols.append('ena_fastq_r1')
    if 'fastq_r2_url' in df.columns and 'ena_fastq_r2' not in df.columns:
        df['ena_fastq_r2'] = df['fastq_r2_url']
        if 'ena_fastq_r2' not in batch_cols:
            batch_cols.append('ena_fastq_r2')
    
    # Deduplicate batch_cols  
    batch_cols = list(dict.fromkeys(batch_cols))
    
    # Split into batches
    n_batches = (len(df) + samples_per_batch - 1) // samples_per_batch
    logger.info(f"Creating {n_batches} batches ({samples_per_batch} samples each)")
    
    for i in range(n_batches):
        start_idx = i * samples_per_batch
        end_idx = min((i + 1) * samples_per_batch, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        batch_file = batch_dir / f"batch_{i:05d}.csv"
        available_cols = [c for c in batch_cols if c in batch_df.columns]
        batch_df[available_cols].to_csv(batch_file, index=False)
    
    logger.info(f"Batches created in {batch_dir}")
    
    # Generate batch script
    batch_file_template = batch_dir / "batch_00000.csv"
    script_path = generate_batch_script(
        batch_file=batch_file_template,
        job_name=job_name,
        partition=partition,
        cpus=cpus,
        memory=memory,
        time=time,
        array_size=n_batches,
        max_concurrent=max_concurrent,
        config=config,
    )
    
    # Submit job
    return submit_job(script_path, dry_run=dry_run)


def cancel_job(job_id: str) -> bool:
    """
    Cancel a SLURM job.
    
    Args:
        job_id: Job ID to cancel
        
    Returns:
        True if successful
    """
    try:
        subprocess.run(
            ["scancel", job_id],
            check=True,
            capture_output=True,
        )
        logger.info(f"✓ Cancelled job {job_id}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        return False
