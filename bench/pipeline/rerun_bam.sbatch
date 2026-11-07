#!/bin/bash
#SBATCH --job-name=pb-rebam
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --array=0-4
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Re-run the pileup on a kept campaign BAM (Singlet's own STAR output, CB/UB tagged) with the
# baseline engine (= atlas counting code) and the fixed engine, same barcodes, same flags.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
GSMS=(GSM7596594 GSM7412962 GSM7090362 GSM7983360 GSM9359399)
UMIL=(12 12 12 12 6)
G=${GSMS[$SLURM_ARRAY_TASK_ID]}; U=${UMIL[$SLURM_ARRAY_TASK_ID]}
SRC=/mnt/projects/debruinz_project/cellarium/full_campaign/results/$G/out
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
ln -s $SRC/star_Aligned.sortedByCoord.out.bam $SCR/in.bam    # read-only; index lives in scratch
samtools index -@ 8 $SCR/in.bam
for TOOL in singlet-baseline singlet-fixed; do
  O=$PB/runs/bam_$G/$TOOL; rm -rf $O; mkdir -p $O; cd $SCR
  /usr/bin/time -v -o $O/time.txt $PB/bin/$TOOL --bam-file $SCR/in.bam \
     --exons $REF/GRCh38-2024-A/genes/genes.gtf --barcodes $SRC/auto_barcodes.tsv \
     --umi-len $U --pipeline --cell-calling --threads 8 --output-format mtx --raw-matrix \
     --out-prefix $O/ > $O/run.log 2>&1 || echo "[rebam] $TOOL exit $?"
  grep -E "Elapsed \(wall|Maximum resident" $O/time.txt
done
grep -h -E "molecules_|umi_unique|exon_hits|intron_hits" $PB/runs/bam_$G/singlet-fixed/pileup_stats.json
