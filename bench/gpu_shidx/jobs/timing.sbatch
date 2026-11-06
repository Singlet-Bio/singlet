#!/bin/bash
#SBATCH --job-name=gshidx-timing
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodelist=g050
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Timing of the shared resident genome index (gpu/bold-shidx) on g050 (the timing node). One job, five arms, each the throughput harness
# (tp.sbatch, bench/gpu_shidx/jobs/tp.sbatch: a copy of bench/gpu_bold/jobs/tp.sbatch whose results directory lies under jobs/gpu-shidx) at k=8, T=7, r3 (24 runs):
#   A  b0 at 2 slots (every process holds its own index)
#   B  shared index, 2 slots          C  shared index, 3 slots (normal mode expected)
#   Bm B under MPS                    Cm C under MPS
# The holder is started before each shared arm (so the first run does not pay the upload) and stopped after it. MPS arms run under their own
# MPS daemon with pipe and log directories in the job's /tmp. Per arm: arm.txt with the mode lines (lean or normal), the device memory peak
# (nvidia-smi once a second), the sharing lines of the logs and the harness' results table.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference
SH=${SH:-$W/jobs/gpu-shidx/bin/shidx}; B0=${B0:-$W/jobs/gpu-bold/bin/b0}; TP=${TP:-$W/jobs/gpu-shidx/jobs/tp.sbatch}
RES=$W/jobs/gpu-shidx/results/overnight/timing-$SLURM_JOB_ID; mkdir -p $RES; S=$RES/summary.txt
HG=$REF/GRCh38-2024-A/star_2.7.11b
SH_COMMIT=$(cat $SH/commit-gpu); B0_COMMIT=$(cat $B0/commit-gpu)
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) new=$SH_COMMIT sha256=$(sha256sum $SH/singlet-gpu | cut -c1-64) b0=$B0_COMMIT $(date -Is)" | tee $S
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $S
squeue -w $(hostname -s) -o "%i %u %j %T" | tee -a $S
MPSD=/tmp/$USER-mps-$SLURM_JOB_ID
stop_holder() { $SH/singlet-gpu gpu-holder --stop --genome-dir $HG --device 0 2>/dev/null; sleep 8; }
stop_mps() { echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPSD/pipe CUDA_MPS_LOG_DIRECTORY=$MPSD/log nvidia-cuda-mps-control 2>/dev/null; sleep 3; }
trap 'kill $(jobs -p) 2>/dev/null; stop_holder; stop_mps' EXIT INT TERM

# arm NAME BINDIR COMMIT POINT SHARED(0|1) MPS(0|1)
arm() {
  local name=$1 bd=$2 cm=$3 pt=$4 shared=$5 mps=$6 R=$RES/$1
  mkdir -p $R
  echo "######## arm $name: bin $bd point $pt shared=$shared mps=$mps $(date -Is)" | tee -a $S
  ( unset SINGLET_GPU_SHARED_INDEX SINGLET_GPU_SHARED_INDEX_IDLE_S
    if [ $mps = 1 ]; then
      mkdir -p $MPSD/pipe $MPSD/log; export CUDA_MPS_PIPE_DIRECTORY=$MPSD/pipe CUDA_MPS_LOG_DIRECTORY=$MPSD/log
      nvidia-cuda-mps-control -d && echo "  MPS daemon started" | tee -a $S; sleep 3
    fi
    if [ $shared = 1 ]; then
      export SINGLET_GPU_SHARED_INDEX=1 SINGLET_GPU_SHARED_INDEX_IDLE_S=600
      $bd/singlet-gpu gpu-holder --genome-dir $HG --device 0 > $R/holder.log 2>&1 &
      for i in $(seq 1 120); do ls /tmp/singlet_gpu_holder.*.info >/dev/null 2>&1 && break; sleep 2; done
      echo "  holder: $(tail -1 $R/holder.log | cut -c1-250)" | tee -a $S
    fi
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > $R/memused.csv 2>/dev/null &
    NV=$!
    BINDIR=$bd EXPECT_COMMIT=$cm POINTS=$pt RES0=$R/tp bash $TP > $R/tp.log 2>&1
    kill $NV 2>/dev/null
    if [ $shared = 1 ]; then $bd/singlet-gpu gpu-holder --stop --genome-dir $HG --device 0; sleep 8; fi
    if [ $mps = 1 ]; then echo quit | nvidia-cuda-mps-control; sleep 3; cp $MPSD/log/*.log $R/ 2>/dev/null; fi
  )
  # per-arm record
  local A=$R/arm.txt; : > $A
  local logs; logs=$(ls $R/tp/*/c*/star_Log.out 2>/dev/null)
  echo "arm $name ($pt, shared=$shared, mps=$mps)" >> $A
  echo "mode lines (runs): $(cat $logs 2>/dev/null | grep -ao 'E17 budget[^:]*: mode [a-z]*' | grep -ao 'mode [a-z]*' | sort | uniq -c | tr '\n' ' ')" >> $A
  echo "budget line (first run): $(cat $logs 2>/dev/null | grep -a 'E17 budget' | head -1 | cut -c1-420)" >> $A
  echo "device memory peak (nvidia-smi, GB): $(awk '{if($1>m)m=$1} END{printf "%.1f", m/1024}' $R/memused.csv 2>/dev/null)" >> $A
  echo "shared-index lines: imported=$(cat $logs 2>/dev/null | grep -ac 'shared index: imported') private=$(cat $logs 2>/dev/null | grep -ac 'shared index:.*private upload')" >> $A
  cat $logs 2>/dev/null | grep -a 'shared index:.*private upload' | sort | uniq -c | head -3 | cut -c1-300 >> $A
  for f in $R/tp/*/point.txt; do echo "-- $f" >> $A; cut -c1-300 $f >> $A; done
  [ -f $R/tp/results.tsv ] && { echo "-- results.tsv"; column -t -s $'\t' $R/tp/results.tsv; } >> $A
  cat $A | tee -a $S
}
arm A  $B0 $B0_COMMIT k8t7s2r3 0 0
arm B  $SH $SH_COMMIT k8t7s2r3 1 0
arm C  $SH $SH_COMMIT k8t7s3r3 1 0
arm Bm $SH $SH_COMMIT k8t7s2r3 1 1
arm Cm $SH $SH_COMMIT k8t7s3r3 1 1
echo done $(date -Is) | tee -a $S
