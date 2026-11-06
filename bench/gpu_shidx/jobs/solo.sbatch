#!/bin/bash
#SBATCH --job-name=gshidx-solo
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Diagnosis of the slow shared index (gpu/bold-shidx): one process alone on one GPU (point k1t7s1: one run alone, then one more), arms
#   P  private index (SINGLET_GPU_SHARED_INDEX unset)
#   S  shared index (holder started first)
#   C  shared index, then a private device-to-device copy of the import (SINGLET_GPU_SHARED_INDEX_COPY=1): the control
# Needs SH (BINDIR of the build with the diagnostics and the control) and its commit in SH_COMMIT_X. ARMS selects and orders the arms.
set -uo pipefail
export ALLOW_NO_EMBED=1
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference
SH=${SH:?SH}; TP=${TP:-$W/jobs/gpu-shidx/jobs/tp.sbatch}; ARMS=${ARMS:-C P S}
RES=$W/jobs/gpu-shidx/results/solo-$SLURM_JOB_ID; mkdir -p $RES; S=$RES/summary.txt
HG=$REF/GRCh38-2024-A/star_2.7.11b
SH_COMMIT=$(cat $SH/commit-gpu)
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$SH commit=$SH_COMMIT sha256=$(sha256sum $SH/singlet-gpu | cut -c1-64) $(date -Is)" | tee $S
nvidia-smi --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader | tee -a $S
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a $S
stop_holder() { $SH/singlet-gpu gpu-holder --stop --genome-dir $HG --device 0 2>/dev/null; sleep 8; }
trap 'kill $(jobs -p) 2>/dev/null; stop_holder' EXIT INT TERM

arm() {   # NAME SHARED COPY
  local name=$1 shared=$2 copy=$3 R=$RES/$1
  mkdir -p $R
  echo "######## arm $name: shared=$shared copy=$copy $(date -Is)" | tee -a $S
  ( unset SINGLET_GPU_SHARED_INDEX SINGLET_GPU_SHARED_INDEX_IDLE_S SINGLET_GPU_SHARED_INDEX_COPY
    if [ $shared = 1 ]; then
      export SINGLET_GPU_SHARED_INDEX=1 SINGLET_GPU_SHARED_INDEX_IDLE_S=600
      [ $copy = 1 ] && export SINGLET_GPU_SHARED_INDEX_COPY=1
      $SH/singlet-gpu gpu-holder --genome-dir $HG --device 0 > $R/holder.log 2>&1 &
      for i in $(seq 1 120); do ls /tmp/singlet_gpu_holder.*.info >/dev/null 2>&1 && break; sleep 2; done
      echo "  holder: $(tail -1 $R/holder.log | cut -c1-250)" | tee -a $S
    fi
    BINDIR=$SH EXPECT_COMMIT=$SH_COMMIT POINTS=k1t7s1 RES0=$R/tp bash $TP > $R/tp.log 2>&1
    if [ $shared = 1 ]; then $SH/singlet-gpu gpu-holder --stop --genome-dir $HG --device 0; sleep 8; fi
  )
  local logs; logs=$(ls $R/tp/*/c*/star_Log.out 2>/dev/null)
  { echo "arm $name"
    grep -ah 'shared index' $logs | cut -c1-420 | head -6
    for f in $logs; do echo "-- $f"; grep -ao 'steps: .*' $f | cut -c1-400; done
    for f in $R/tp/*/point.txt; do echo "-- $f"; sed -n 2p $f | cut -c1-300; done
  } | tee -a $S
}
for a in $ARMS; do case $a in P) arm P 0 0 ;; S) arm S 1 0 ;; C) arm C 1 1 ;; esac; done
echo done $(date -Is) | tee -a $S
