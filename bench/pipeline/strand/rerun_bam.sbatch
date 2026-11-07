#!/bin/bash
#SBATCH --job-name=pb-strand-rebam
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# The 8 kept GRCh38 campaign BAMs (6 are >90% wrong-strand in the atlas): orientation-free
# probe (strand_probe_bam.py), then re-pile with the strand-fixed engine. --bam-file always
# takes the run() path (indexed probe + safety net); run_parallel() is covered by the FASTQ runs.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__strand-detection/bench/pipeline/strand/common.sh
GSMS=(GSM7090360 GSM7090362 GSM7412962 GSM7596602 GSM7596603 GSM7983360 ${EXTRA_GSMS:-})
G=${1:-${GSMS[$SLURM_ARRAY_TASK_ID]:-}}; [ -n "$G" ] || { echo "no GSM for task $SLURM_ARRAY_TASK_ID"; exit 0; }
SRC=$CAMPAIGN/$G/out
SCR=/tmp/$USER-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
ln -s $SRC/star_Aligned.sortedByCoord.out.bam $SCR/in.bam
samtools index -@ 8 $SCR/in.bam
O=$RES_STRAND/bam/$G; mkdir -p $O
python -I $SRC_STRAND/bench/pipeline/analysis/strand_probe_bam.py $REF/GRCh38-2024-A/genes/genes.gtf \
   $O/probe.tsv $G=$SCR/in.bam
UMIL=${UMIL:-12}   # Visium / 10x v3
for MODE in file; do
  P=$PB/runs/bam_$G/singlet-strandfix-$MODE; rm -rf $P; mkdir -p $P; cd $SCR
  /usr/bin/time -v -o $P/time.txt $BIN_STRAND --bam-file $SCR/in.bam \
     --exons $REF/GRCh38-2024-A/genes/genes.gtf --barcodes $SRC/auto_barcodes.tsv \
     --umi-len $UMIL --pipeline --cell-calling --threads 8 --output-format mtx --raw-matrix \
     --out-prefix $P/ > $P/run.log 2>&1 || echo "[rebam] $MODE exit $?"
  grep -E "Strand|strand" $P/run.log | head -20
  python -I -c "
import json; s=json.load(open('$P/pileup_stats.json'))
print('$G $MODE', {k: s.get(k) for k in ('exon_hits','wrong_strand','intron_hits','strand_sense','strand_antisense','reverse_strand','strand_auto_flipped','strand_probed','strand_probe_antisense_frac','strand_retried','counted_antisense_frac')})" | tee -a $O/repile.txt
done
