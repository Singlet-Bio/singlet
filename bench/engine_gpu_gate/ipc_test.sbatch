#!/bin/bash
#SBATCH --job-name=s8-ipc
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: CUDA IPC between two processes of the same Slurm step (the bare binary, no container), as the resident-index server would use it. See ipc_test.cu.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; SCR=/tmp/$USER-ipc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT
BARE=$W/repos/singlet-py.git; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/${BRANCH:-perf/gpu-instrumentation}:refs/s8/ipc
SRC=$SCR/src; mkdir -p $SRC; git -C $W/repos/singlet-py.git archive refs/s8/${TAG:-ipc} bench/engine_gpu_gate/ipc_test.cu | tar -x -C $SRC
export PATH=/usr/local/cuda/bin:$PATH
nvcc -O2 -o $SCR/ipc_test $SRC/bench/engine_gpu_gate/ipc_test.cu 2>&1 | tail -5
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$SCR/ipc_test server $SCR/s.sock ${GB:-40} &
S=$!
$SCR/ipc_test client $SCR/s.sock; echo "client rc=$?"
wait $S; echo "server rc=$?"
