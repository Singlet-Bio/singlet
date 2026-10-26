#!/bin/bash
#SBATCH --job-name=bp-artifacts
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch artifacts.sbatch barnyard <cohort-id>
#   sbatch artifacts.sbatch kang <cohort-id> <truth tsv.gz> <lane filter, e.g. stim=ctrl or batch=C>
# Collect the tool outputs of one library into the compact per-cell artifacts (barnyard_artifacts.py / kang_artifacts.py) under
# $BP/artifacts/{ambient-rna,doublets}/.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; H=$W/jobs/bp/harness
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
KIND=$1; ID=$2
case $KIND in
  barnyard) python -I $H/barnyard_artifacts.py $ID $BP/runs/$ID/singlet $BP/work/$ID $BP/artifacts ;;
  kang) python -I $H/kang_artifacts.py $ID $BP/runs/$ID/singlet $BP/work/$ID $BP/artifacts $3 $4 ;;
esac
ls -la $BP/artifacts/ambient-rna $BP/artifacts/doublets || true
