#!/bin/bash
#SBATCH --job-name=w11-m8dual
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# M8, full node: two independent pipelines (one per H100), node-level CPU load, shared-FS rate with two consumers.
set -u
WORK=/mnt/projects/debruinz_project/debruinz/singlet
OUT=$WORK/jobs/w11/m8/results_dual_$SLURM_JOB_ID; mkdir -p $OUT
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
source $WORK/jobs/w11/m8/jobs/m8_common.sh
envinfo
# pinned bandwidth: each GPU alone, then both at once
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g m8 --bw --dev 0 --reps 30 | sed "s/\"bw_dev\":0/\"bw_dev\":0,\"gpu\":$g,\"mode\":\"alone\"/" >> $RUNS; done
( CUDA_VISIBLE_DEVICES=0 m8 --bw --dev 0 --reps 60 | sed 's/"bw_dev":0/"bw_dev":0,"gpu":0,"mode":"both"/' >> $RUNS ) &
( CUDA_VISIBLE_DEVICES=1 m8 --bw --dev 0 --reps 60 | sed 's/"bw_dev":0/"bw_dev":0,"gpu":1,"mode":"both"/' >> $RUNS ) &
wait
# staging with two concurrent consumers of the shared filesystem
stage() { node_wrap stage_two_consumers bash -c "
  (apptainer exec -B $WORK --env LD_LIBRARY_PATH=$M8/lib $IMG $BIN --copy $A_SHARED $SCRATCH/A.1fq --streams 1 >> $RUNS) &
  (apptainer exec -B $WORK --env LD_LIBRARY_PATH=$M8/lib $IMG $BIN --copy $C_SHARED $SCRATCH/C.1fq --streams 1 >> $RUNS) &
  wait"; }
stage
A=$SCRATCH/A.1fq; C=$SCRATCH/C.1fq
MP=100000000
# one GPU alone (reference for the same file and settings)
for S in 0 20; do node_wrap single_nvme_t3_s$S run single_nvme_A_t3_s$S --file $A --threads 3 --spin $S --max-pairs $MP; done
# two pipelines on NVMe copies: GPU0 reads A, GPU1 reads C, 100 M pairs each
dual() { # name fileA fileB threads spin
  local name=$1 fa=$2 fb=$3 n=$4 s=$5
  read b0 t0 < <(busy_jiffies); w0=$(date +%s.%N)
  runbg 0 ${name}_g0 --file $fa --threads $n --spin $s --max-pairs $MP
  runbg 1 ${name}_g1 --file $fb --threads $n --spin $s --max-pairs $MP
  wait
  read b1 t1 < <(busy_jiffies); w1=$(date +%s.%N)
  awk -v l="$name" -v b0=$b0 -v b1=$b1 -v w0=$w0 -v w1=$w1 'BEGIN{printf "{\"node_load\":\"%s\",\"wall_s\":%.2f,\"node_cores_busy\":%.2f}\n", l, w1-w0, (b1-b0)/100.0/(w1-w0)}' >> $RUNS
}
for N in 2 3 4 6; do dual dual_nvme_t${N}_s0 $A $C $N 0; done
for N in 3 4 6; do dual dual_nvme_t${N}_s20 $A $C $N 20; done
# two pipelines reading the shared filesystem directly
for N in 3 4 6; do dual dual_shared_t${N}_s0 $A_SHARED $C_SHARED $N 0; done
dual dual_shared_t4_s20 $A_SHARED $C_SHARED 4 20
cp $RUNS $OUT/ ; ls -l $OUT
