#!/bin/bash
#SBATCH --job-name=bp-ksinglet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch kang_singlet.sbatch <cohort-id> <path to the lane .1fq>
# Singlet 1.0.0 (bin/singlet-campaign, 02aa9b9) at pipeline defaults on one genotype-multiplexed Kang 2018 lane: GRCh38-2024-A,
# `--pipeline` (default 1000G SNP panel, donor count estimated by Singlet), plus --tagged-bam so that vireo reads the same
# alignments (CB/UB tags). Output: $BP/runs/<id>/singlet/.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; PB=$W/pipeline-benchmark
ID=$1; FQ=$2; BIN=$PB/bin/singlet-campaign
echo "$(sha256sum $BIN)"; export PATH=$W/envs/pbench/bin:$PATH
REFB=/mnt/projects/debruinz_project/cellarium/reference; REF=$REFB/GRCh38-2024-A
O=$BP/runs/$ID/singlet; rm -rf $O; mkdir -p $O; lscpu | sed -n 's/^Model name: *//p' > $O/node.txt
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cp $FQ $SCR/s.1fq; cd $SCR
[ -f $(dirname $FQ)/encode.time.txt ] && cp $(dirname $FQ)/encode.time.txt $O/encode.time.txt
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REFB --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf \
  --out-prefix $O/ --threads 16 --pipeline --tagged-bam $O/possorted.bam --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt
grep -E "Default SNP|\[demux|\[doublet\]|auto-K|K=|\[cell_calling\]" $O/run.log | head -30 || true
ls -la $O $O/donor $O/private 2>/dev/null | head -80 || true
