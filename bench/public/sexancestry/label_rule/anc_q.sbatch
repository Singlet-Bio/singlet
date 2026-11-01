#!/bin/bash
#SBATCH --job-name=gsota-ancq
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/bpub
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1 ANC_PY=$W/src/singlet-py/prod__campaign/python/singlet/celllabels/ancestry.py
R=$W/data/1kgp_grch38
ls $R/ALL.chr*.vcf.gz | xargs -P 16 -I{} sh -c '[ -e {}.tbi ] || bcftools index -t --threads 1 {}'
python -u $W/jobs/gsota-r12/anc_q.py $W/bench-public/artifacts/genotype-sota-anc
