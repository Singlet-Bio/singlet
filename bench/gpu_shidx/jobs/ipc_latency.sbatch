#!/bin/bash
#SBATCH --job-name=gshidx-iplat
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Random-load speed of private, driver-virtual-memory, legacy IPC and VMM-imported device buffers (bench/engine_gpu_gate/ipc_latency_test.cu), 12 GB.
# BR: branch holding the source (default gpu/bold-shidx-fix). Compiled in the CUDA 12.4 devel image, run in it with --nv.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; BR=${BR:-gpu/bold-shidx-fix}; SCR=/tmp/$USER-iplat-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/gshidx/iplat ) 9>$BARE/.gn-fetch.lock
git -C $BARE rev-parse refs/gshidx/iplat
git -C $BARE archive refs/gshidx/iplat bench/engine_gpu_gate/ipc_latency_test.cu | tar -x -C $SCR/src
hostname; nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
apptainer exec --nv -B $SCR $IMG bash -c "cd $SCR && nvcc -O2 -o ipl src/bench/engine_gpu_gate/ipc_latency_test.cu 2>&1 | tail -5 && ./ipl 12"
