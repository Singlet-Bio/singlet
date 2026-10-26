#!/bin/bash
#SBATCH --job-name=bp-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_singlet_10x.sbatch <cohort-id> <gex fastq list> [reference dir name, default GRCh38-GRCm39-barnyard]
# Singlet 1.0.0 (bin/singlet-campaign, commit 02aa9b9) at pipeline defaults on one public 10x library: encode the FASTQs to
# .1fq (timed), then `singlet --pipeline --output-format mtx --raw-matrix`, 16 threads, as bench/pipeline/run_tool.sbatch.
# Output: $BP/runs/<id>/singlet/ (+ encode.time.txt, time.txt, node.txt).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; PB=$W/pipeline-benchmark
ID=$1; LIST=$2; REFN=${3:-GRCh38-GRCm39-barnyard}
BIN=$PB/bin/singlet-campaign
echo "$(sha256sum $BIN)"
export PATH=$W/envs/pbench/bin:$PATH
REFB=/mnt/projects/debruinz_project/cellarium/reference; REF=$REFB/$REFN
O=$BP/runs/$ID/singlet; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "[run] $ID node=$(hostname) $(date -Is)"; echo "$(lscpu | sed -n 's/^Model name: *//p')" > $O/node.txt
cat $(grep "_R1_" $LIST) > $SCR/R1.fastq.gz
cat $(grep "_R2_" $LIST) > $SCR/R2.fastq.gz
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$ID.1fq > $O/encode.log 2>&1
rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
export SINGLET_STALL_TIMEOUT=7200
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/$ID.1fq --ref-base $REFB --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf \
  --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1
grep -E "Elapsed \(wall|Maximum resident" $O/encode.time.txt $O/time.txt
grep -E "\[ambient\]|\[doublet\]|\[barnyard\]|\[cell_calling\]" $O/run.log
du -sh $O
