#!/bin/bash
# Run one fastloop (or any pilot g1) item as it is, without the smoke reduction, on an idle RTX 8000 or V100S: the screening tier off the
# H100s. Same variables as the pilot sets for a g1 item that is alone on its node (FL_SOLE=1), with the engine admitted below compute
# capability 8.0. The binary must carry sm_70/sm_75 code (the -multi builds).
#   sbatch --export=ALL,ITEM=$WORK/jobs/gpu-fastloop/items/<item>.g1.sh run_on.sbatch                                      (RTX 8000)
#   sbatch --gres=gpu:tesla_v100s:1 -c 11 --mem=170G --export=ALL,ITEM=... run_on.sbatch                                       (V100S)
#SBATCH --job-name=fl-run
#SBATCH --account=debruinz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
: "${ITEM:?ITEM: the item script}"
name=$(basename $ITEM); name=${name%.g?.sh}
echo "fl-run job=$SLURM_JOB_ID node=$(hostname -s) item=$ITEM $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date -Is)"
export SINGLET_ENGINE_GPU_MIN_CC=70 PILOT=1 PILOT_JOB=$SLURM_JOB_ID PILOT_NODE=$(hostname -s) PILOT_GPU=0 PILOT_SHARED=0 FL_SOLE=1 \
  PINDEV=0 CUDA_VISIBLE_DEVICES=0 RUNID=$SLURM_JOB_ID-$name
bash $ITEM; rc=$?
echo "fl-run $name: exit $rc ($(date -Is))"
exit $rc
