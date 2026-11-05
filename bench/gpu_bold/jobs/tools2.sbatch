#!/bin/bash
#SBATCH --job-name=gbold-tools2
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=g052
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Profiling tool details inside a job: GPM metric names for nvidia-smi dmon, nsys and ncu versions and GPU metric options, DCGM host engine.
echo "== node $(hostname) job $SLURM_JOB_ID $(date -Is)"
echo "== dmon gpm list"
nvidia-smi dmon -h 2>&1 | sed -n '/GPM metrics/,$p' | head -60
echo "== nsys"
/usr/local/cuda/bin/nsys --version
/usr/local/cuda/bin/nsys profile --gpu-metrics-devices=help 2>&1 | head -20
/usr/local/cuda/bin/nsys profile --help 2>&1 | grep -iE -A3 "gpu-metrics-set|gpu-metrics-frequency" | head -30
echo "== ncu"
/usr/local/cuda/bin/ncu --version 2>&1 | tail -3
/usr/local/cuda/bin/ncu --list-sets 2>&1 | head -30
echo "== dcgm"
dcgmi --version 2>&1 | head -3
( timeout 20 nv-hostengine -n --service-account debruinz >/tmp/he-$SLURM_JOB_ID.log 2>&1 & ) ; sleep 3
timeout 20 dcgmi discovery -l 2>&1 | head -10
timeout 20 dcgmi profile -l 2>&1 | head -30
timeout 30 dcgmi dmon -e 1002,1003,1004,1005,1009,1010 -c 2 2>&1 | head -10
cat /tmp/he-$SLURM_JOB_ID.log 2>&1 | head -10
pkill -u $USER nv-hostengine 2>/dev/null
echo "== mps"
nvidia-cuda-mps-control -h 2>&1 | head -5
echo done $(date -Is)
