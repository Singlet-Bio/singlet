#!/bin/bash
# Smoke tier of a pilot item, on a V100S or RTX 8000 instead of an H100: the item script itself, with SMOKE=1 (tp_gc.sbatch then runs two
# pbmc2m runs of each config the item names, one at a time) and the engine admitted below compute capability 8.0. It catches what ended
# earlier items after H100 hours: a refused binary, a script error, a config that fails. The binary must carry sm_70/sm_75 code
# (gc-build.sbatch builds "70;75;90" by default). Exit 0 and no POINT FAILED line: the item may go to the H100 queue.
#   sbatch --export=ALL,ITEM=$WORK/jobs/gpu-campaign/items/<item>.g1.sh smoke.sbatch            (RTX 8000)
#   sbatch --gres=gpu:tesla_v100s:1 -c 11 --export=ALL,ITEM=... smoke.sbatch                      (V100S: at most 11 cores per GPU)
#SBATCH --job-name=gc-smoke
#SBATCH --account=debruinz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
: "${ITEM:?ITEM: the pilot item script}"
name=$(basename $ITEM); name=${name%.g?.sh}
echo "smoke job=$SLURM_JOB_ID node=$(hostname -s) item=$ITEM $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -Is)"
export SMOKE=1 SINGLET_ENGINE_GPU_MIN_CC=70 PILOT=1 PILOT_JOB=$SLURM_JOB_ID PILOT_NODE=$(hostname -s) PILOT_GPU=0 PILOT_SHARED=0 \
  PINDEV=0 CUDA_VISIBLE_DEVICES=0 RUNID=smoke-$SLURM_JOB_ID-$name
bash $ITEM; rc=$?
echo "SMOKE $name: exit $rc, $(grep -c '^POINT FAILED' /mnt/projects/debruinz_project/debruinz/singlet/logs/gc-smoke-$SLURM_JOB_ID.out) failed points"
exit $rc
