#!/bin/bash
#SBATCH --job-name=gbold-tools
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=g052
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# What profiling tools exist inside a job on g050 or g051. Reads only. Nothing here runs a workload.
W=/mnt/projects/debruinz_project/debruinz/singlet
echo "== node $(hostname) job $SLURM_JOB_ID $(date -Is)"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,compute_mode,mig.mode.current,pstate,clocks.max.sm,clocks.max.mem,memory.total --format=csv
echo "== profiling permission (RmProfilingAdminOnly 1 means counters need admin)"
grep -i "RmProfilingAdminOnly\|ProfilingAdmin" /proc/driver/nvidia/params 2>&1
echo "== which"
for t in nsys ncu nsight-sys nsight-compute dcgmi nv-hostengine dcgmproftester nvidia-cuda-mps-control nvprof cuobjdump nvcc compute-sanitizer; do printf "%s: " $t; which $t 2>/dev/null || echo none; done
echo "== cuda dirs"
ls -d /usr/local/cuda* /opt/nvidia* /opt/cuda* 2>&1 | head
ls /usr/local/cuda/bin 2>/dev/null | head -40
echo "== cupti"
ls /usr/local/cuda*/extras/CUPTI/lib64 2>&1 | head
find /usr /opt -maxdepth 6 \( -name 'libcupti*' -o -name 'libnvperf*' -o -name 'libnvToolsExt*' -o -name 'libdcgm*' \) 2>/dev/null | head -20
echo "== modules"
source /etc/profile.d/lmod.sh 2>/dev/null
module avail 2>&1 | grep -iE "cuda|nsight|dcgm|nvhpc|cupti" | head -20
echo "== devel image"
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
ls -la $IMG
apptainer exec --nv $IMG bash -c 'ls /usr/local/cuda/bin | head -60; ls /usr/local/cuda/extras/CUPTI/lib64 2>&1 | head; which nsys ncu dcgmi 2>&1; ls /opt/nvidia 2>&1 | head; nvidia-smi -L' 2>&1 | head -90
echo "== nvidia-smi dmon capabilities"
nvidia-smi dmon -h 2>&1 | head -40
echo "== dmon 3 samples"
nvidia-smi dmon -s pucvmet -c 3 2>&1
echo "== nvidia-smi pmon / query fields"
nvidia-smi --help-query-gpu 2>&1 | grep -iE "utilization|memory|clocks|pcie|power" | head -30
echo "== other processes on the node"
squeue -w $(hostname -s) -o "%i %u %j %C %b" 2>&1
echo done $(date -Is)
