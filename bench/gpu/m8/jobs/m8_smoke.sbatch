#!/bin/bash
#SBATCH --job-name=w11-m8smoke
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# M8 smoke test: environment, header inspection, decoder self-test against the library, pinned bandwidth, one short pipeline run.
set -u
WORK=/mnt/projects/debruinz_project/debruinz/singlet
OUT=$WORK/jobs/w11/m8/results_smoke; mkdir -p $OUT
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
source $WORK/jobs/w11/m8/jobs/m8_common.sh
envinfo
for f in $A_SHARED $B_SHARED; do m8 --inspect $f --selftest 300; done
m8 --bw --dev 0 --reps 30
m8 --copy $PERF/GSM6470208/SRR21102187.1fq $SCRATCH/b1.1fq --streams 1
run smoke_nvme_t3_s10_verify --file $SCRATCH/b1.1fq --threads 3 --spin 10 --verify
run smoke_nvme_t1_s0 --file $SCRATCH/b1.1fq --threads 1 --spin 0
run smoke_nvme_t3_s0_qualnone --file $SCRATCH/b1.1fq --threads 3 --spin 0 --qual none
run smoke_nvme_t3_s20_crc --file $SCRATCH/b1.1fq --threads 3 --spin 20 --crc
run smoke_shared_t3_s0 --file $C_SHARED --threads 3 --spin 0 --max-pairs 30000000
cp $RUNS $OUT/ 2>/dev/null; ls -l $OUT
