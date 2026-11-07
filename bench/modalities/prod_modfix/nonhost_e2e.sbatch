#!/bin/bash
#SBATCH --job-name=mf-nonhost
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Non-host end to end with bin/singlet-modfix on GSE166766 mock / dpi2 (docs/modality-benchmarks/junctions-nonhost.md),
# on a FRESH output directory (no pre-made nonhost/): the per-cell table must be written, with EM-confirmed species only.
# Then bench/modalities/sj_nonhost/nonhost_compare.py against the existing Kraken2 / minimap2 results.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; B=$W/modality-bench/sj_nonhost
NB=$W/modality-bench/sj_nonhost_modfix; mkdir -p $NB/singlet
for d in kraken oracle cells; do [ -e $NB/$d ] || ln -s $B/$d $NB/$d; done
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A; BIN=$PB/bin/singlet-modfix
LABS=(mock dpi2); L=${LABS[$SLURM_ARRAY_TASK_ID]}; D=$B/data/$L; O=$NB/singlet/$L; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$L; mkdir -p $SCR; trap "rm -rf $SCR" EXIT
$BIN encode --reads $D/R1.fastq.gz $D/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1
export SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference SINGLET_STALL_TIMEOUT=10800
cd $SCR; /usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $SINGLET_REF_BASE --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf \
  --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum" $O/time.txt; grep -E "nonhost" $O/run.log | head -30; ls -la $O/nonhost
head -5 $O/nonhost/nonhost_per_cell.tsv 2>/dev/null; cut -f3 $O/nonhost/nonhost_per_cell.tsv 2>/dev/null | sort | uniq -c | sort -rn | head
grep -A3 missing_reference_fasta $O/nonhost/nonhost_summary.json
if [ "$L" = dpi2 ]; then
  # wait for mock, then compare (PBMC row: the earlier run; it is not part of this check)
  for i in $(seq 1 120); do [ -f $NB/singlet/mock/summary.json ] && break; sleep 30; done
  PBMC_SINGLET=$B/singlet/pbmc1k_v3 $W/envs/pbench/bin/python -I -W ignore $B/scripts/nonhost_compare.py $NB $NB/nonhost_compare.json | tail -60
fi
