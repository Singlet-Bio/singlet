#!/bin/bash
#SBATCH --job-name=rl-rpu
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Reads-per-UMI check: the engine's reads_per_umi_hist.tsv.gz (CPU reference of the frontier gate, REF) against Cell Ranger 3.0's
# pbmc_1k_v3 molecule_info.h5 (10x public sample). Needs BR (branch with bench/readlevel). Output: $W/jobs/readlevel/rpu/<job>/result.json
set -euo pipefail
: "${BR:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; BARE=$W/repos/singlet-py.git
REF=${REF:-$W/jobs/gpu-campaign/ref-654109ad/T18/pbmc1k_v3/out}; PB=$W/pipeline-benchmark/data/pbmc1k_v3
D=$W/data/readlevel/pbmc1k_v3; mkdir -p $D; O=$W/jobs/readlevel/rpu/$SLURM_JOB_ID; mkdir -p $O
MI=$D/pbmc_1k_v3_molecule_info.h5
[ -s $MI ] || curl -sSfL -o $MI.part https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_molecule_info.h5 && mv -f $MI.part $MI 2>/dev/null || true
ls -l $MI
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/rl/rpu-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
SCR=/tmp/$USER-rl-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/rl/rpu-$SLURM_JOB_ID bench/readlevel | tar -x -C $SCR
git -C $BARE rev-parse refs/rl/rpu-$SLURM_JOB_ID > $O/commit; echo "$REF" > $O/ref
$W/envs/pbench/bin/python -I $SCR/bench/readlevel/rpu_vs_molinfo.py --hist $REF/reads_per_umi_hist.tsv.gz --molinfo $MI \
  --cr-filtered $PB/cellranger_pbmc_1k_v3_filtered_feature_bc_matrix.h5 --calls $REF/cell_calls.tsv --out $O/result.json
