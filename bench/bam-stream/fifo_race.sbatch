#!/bin/bash
#SBATCH --job-name=bs-fifo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,OLD=<bin>,NEW=<bin> fifo_race.sbatch      (~0.3 node-h)
# Reproduces the 1fq FIFO open race (fix/fifo-open-race): STAR's start is delayed 5 s with
# SINGLET_TEST_STAR_START_DELAY_MS, so the writer reaches its first flush before STAR opens the FIFOs.
# OLD (hook only) must give 0 reads; NEW (fix) must map every read, with and without the delay.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
O=$W/bam-stream/fifo_race; mkdir -p $O
SCR=/tmp/$USER-bsfifo-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF
cd $SCR
run() {  # name bin delay_ms
  local n=$1 b=$PB/bin/$2 d=$3
  SINGLET_TEST_STAR_START_DELAY_MS=$d $b $PB/data/pbmc2m/pbmc2m.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
    --exons $G/genes/genes.gtf --out-prefix $SCR/$n/ --threads 8 --pipeline --output-format 1pz --genome-shared off \
    > $O/$n.log 2>&1; local rc=$?
  printf "%-12s bin=%s delay_ms=%s exit=%s | %s | %s | %s\n" $n $2 $d $rc \
    "$(grep -m1 -E '1fq-fifo\] (ERROR|Pass 2)' $O/$n.log)" \
    "$(grep -m1 'Number of input reads' $SCR/$n/star_Log.final.out 2>/dev/null | tr -s ' \t' ' ')" \
    "$(grep -m1 -E 'status=|No reads processed' $O/$n.log)" | tee -a $O/summary.txt
  rm -rf $SCR/$n
}
: > $O/summary.txt
run old_delay ${OLD} 5000
run new_delay ${NEW} 5000
run new_nodelay ${NEW} 0
