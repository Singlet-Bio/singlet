#!/bin/bash
#SBATCH --job-name=gp-env
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Reference-tool environment for the genotype-pool benchmark: venv (vireoSNP, scrublet, pysam, mgatk, ...)
# and cellsnp-lite (container, else source build).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/gpool; export PATH=$W/envs/pbench/bin:$PATH
export APPTAINER_CACHEDIR=$W/cache/apptainer; export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR $APPTAINER_CACHEDIR $E
python -m venv $E/venv && $E/venv/bin/pip install -q --upgrade pip
$E/venv/bin/pip install -q numpy scipy pandas scikit-learn matplotlib h5py pysam vireoSNP scrublet DoubletDetection scanpy 2>&1 | tail -5
$E/venv/bin/pip install -q mgatk 2>&1 | tail -5
$E/venv/bin/python -c "import vireoSNP, scrublet, pysam; print('vireo', vireoSNP.__version__)"
$E/venv/bin/pip list 2>/dev/null | grep -i "vireo\|scrublet\|pysam\|mgatk\|doublet\|numpy\|scanpy\|snakemake"
cd $E
for img in docker://quay.io/biocontainers/cellsnp-lite:1.2.3--h6a0a6b2_3 docker://quay.io/biocontainers/cellsnp-lite:1.2.3--h1d76141_1 docker://hxj5/cellsnp-lite:latest; do
  apptainer pull cellsnp-lite.sif $img && break
done
ls -la $E
apptainer exec cellsnp-lite.sif cellsnp-lite -h 2>&1 | head -5
