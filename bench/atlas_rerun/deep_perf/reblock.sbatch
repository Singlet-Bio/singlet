#!/bin/bash
#SBATCH --job-name=dp-reblock
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch reblock.sbatch <GSM> <block_size>
# Control for read-order effects: the pooled sample.1fq re-encoded with another block size (same reads; the encoder
# sorts reads by barcode within each block, so only the order changes) -> sample_bs<N>.1fq.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/singlet-campaign
export SINGLET_WHITELIST_DIR=$PB/whitelists
D=$W/data/atlas-rerun/perf/$1; BS=$2
SCR=/tmp/$USER-rb-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
$BIN decode $D/sample.1fq --r1 $SCR/R1.fq --r2 $SCR/R2.fq --threads 4 > dec.log 2>&1
$BIN encode --reads $SCR/R1.fq $SCR/R2.fq -o $D/sample_bs$BS.1fq.tmp --no-dedup --no-trim --quality none \
  --block-size $BS > enc.log 2>&1 || { tail enc.log; exit 1; }
mv $D/sample_bs$BS.1fq.tmp $D/sample_bs$BS.1fq
$BIN decode $D/sample_bs$BS.1fq --header 2>&1 | head -5
