#!/bin/bash
#SBATCH --job-name=w11-ing-meta
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only: does the .1fg carry the source .1fq metadata JSON? Encode 2 blocks of PBMC 1k and print header + embedded metadata.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; D=$W/jobs/w11/ingest; M8=$W/jobs/w11/m8; BIN=$D/bin3
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
ctr() { apptainer exec -B $W --env LD_LIBRARY_PATH=$M8/lib $IMG "$@"; }
ctr $BIN/enc_1fg --in $W/pipeline-benchmark/data/pbmc1k_v3/pbmc1k_v3.1fq --out $S/m.1fg --codec zstd --level 3 --threads 2 --max-blocks 2 --chunk-pairs 32768 | cut -c1-200
ctr $BIN/dec_1fg --fg $S/m.1fg --meta | head -c 2500
echo
