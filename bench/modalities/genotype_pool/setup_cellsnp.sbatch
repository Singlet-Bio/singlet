#!/bin/bash
#SBATCH --job-name=gp-cellsnp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/gpool
export APPTAINER_CACHEDIR=$W/cache/apptainer; export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR; cd $E
apptainer pull cellsnp-lite.sif docker://quay.io/biocontainers/cellsnp-lite:1.2.3--ha0c3a46_6
apptainer exec cellsnp-lite.sif cellsnp-lite -h 2>&1 | head -5
$E/venv/bin/pip list 2>/dev/null | grep -i "mgatk\|scrublet\|pysam\|doublet\|scanpy\|vireo"
$E/venv/bin/mgatk --help 2>&1 | head -5
