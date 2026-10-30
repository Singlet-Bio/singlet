#!/bin/bash
#SBATCH --job-name=w11-m8single
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# M8, one GPU, 28 cores: staging (copy) rates, node-local NVMe matrix, shared-filesystem matrix, ablations.
set -u
WORK=/mnt/projects/debruinz_project/debruinz/singlet
OUT=$WORK/jobs/w11/m8/results_single_$SLURM_JOB_ID; mkdir -p $OUT
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
source $WORK/jobs/w11/m8/jobs/m8_common.sh
envinfo
m8 --inspect $A_SHARED --selftest 100 || { echo "SELFTEST FAILED"; exit 1; }
m8 --inspect $B_SHARED --selftest 100 || { echo "SELFTEST FAILED"; exit 1; }
m8 --bw --dev 0 --reps 30 | tee -a $RUNS

# --- staging: timed copies from the shared filesystem to node-local NVMe (client page cache dropped first, residency reported)
m8 --copy $A_SHARED $SCRATCH/A.1fq --streams 1 | tee -a $RUNS
m8 --copy $B_SHARED $SCRATCH/B.1fq --streams 4 | tee -a $RUNS
m8 --copy $C_SHARED $SCRATCH/C.1fq --streams 2 | tee -a $RUNS
A=$SCRATCH/A.1fq; B=$SCRATCH/B.1fq

# --- node-local NVMe, cold (page cache dropped before each run): threads x spin
for F in A B; do
  FILE=$SCRATCH/$F.1fq
  for N in 1 2 3 4 6; do for S in 0 10 20; do run nvme_cold_${F}_t${N}_s${S} --file $FILE --threads $N --spin $S; done; done
done
run nvme_cold_A_t8_s0 --file $A --threads 8 --spin 0
# --- ablations on A (NVMe): warm page cache, no readahead hint, with CRC, no quality column (32 B/pair, what the files really hold)
cat $A > /dev/null
for N in 1 3 6; do run nvme_warm_A_t${N}_s0 --file $A --threads $N --spin 0 --dontneed 0; done
run nvme_cold_A_t3_s0_noprefetch --file $A --threads 3 --spin 0 --prefetch 0
run nvme_cold_A_t3_s0_crc --file $A --threads 3 --spin 0 --crc
for N in 3 6; do run nvme_cold_A_t${N}_s0_qualnone --file $A --threads $N --spin 0 --qual none; done
run nvme_cold_B_t3_s0_qualnone --file $B --threads 3 --spin 0 --qual none
run nvme_cold_A_t3_s10_verify --file $A --threads 3 --spin 10 --max-pairs 50000000 --verify

# --- shared filesystem read directly (client page cache dropped; server cache unknown)
for F in A B; do
  if [ $F = A ]; then FILE=$A_SHARED; else FILE=$B_SHARED; fi
  for N in 1 2 3 4 6; do run shared_${F}_t${N}_s0 --file $FILE --threads $N --spin 0; done
  for N in 3 6; do run shared_${F}_t${N}_s20 --file $FILE --threads $N --spin 20; done
done
run shared_A_t3_s0_repeat --file $A_SHARED --threads 3 --spin 0
run shared_A_t6_s0_qualnone --file $A_SHARED --threads 6 --spin 0 --qual none
m8 --resid $A_SHARED | tee -a $RUNS
cp $RUNS $OUT/ ; ls -l $OUT
