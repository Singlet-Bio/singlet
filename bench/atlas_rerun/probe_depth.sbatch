#!/bin/bash
#SBATCH --job-name=ar-depth
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#   sbatch probe_depth.sbatch <GSM>
# Depth probe: keep every K-th read pair (K = 1,2,4,8,16 by array index) from every run of the
# sample, so a cap samples all runs evenly, then run STARsolo GeneFull with EmptyDrops_CR.
# Index 5 ("h4") keeps the FIRST quarter of each run instead: the cheap way to cap, since only
# that part of each run has to be downloaded. h4 vs k4 tests whether head sampling is biased.
# Output: $D/k<K>/Solo.out/GeneFull/Summary.csv (cells, median UMI and genes per cell) + time.txt.
set -eu   # no pipefail: zcat | head closes pipes early on purpose
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
GSM=$1
KS=(1 2 4 8 16 h4); K=${KS[$SLURM_ARRAY_TASK_ID]}
D=$W/data/atlas-rerun/depth/$GSM; O=$D/k$K; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$K; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
# barcode read = the 26-28 bp file, cDNA read = the longest (fasterq-dump numbering varies)
first=$(ls $D/fq/*_1.fastq.gz | head -1); base=${first%_1.fastq.gz}
CB=""; CDNA=""; best=0
for f in ${base}_*.fastq.gz; do
  n=${f##*_}; n=${n%.fastq.gz}; len=$(zcat $f | head -2 | tail -1 | tr -d '
' | wc -c)
  if [ $len -ge 26 ] && [ $len -le 30 ]; then CB=$n; fi
  if [ $len -gt $best ]; then best=$len; CDNA=$n; fi
done
echo "barcode read _$CB, cDNA read _$CDNA ($best bp)"
for side in $CB $CDNA; do
  ls $D/fq/*_${side}.fastq.gz | while read f; do
    if [ $K = h4 ]; then n=$(zcat $f | wc -l); zcat $f | head -n $(( n / 16 * 4 ));
    else zcat $f | awk -v k=$K 'int((NR-1)/4) % k == 0'; fi
  done | pigz -p 4 > $SCR/r$side.fastq.gz
done
READS=$(( $(zcat $SCR/r$CB.fastq.gz | wc -l) / 4 ))
echo "$GSM K=$K reads=$READS" | tee $O/reads.txt
cd $O
/usr/bin/time -v -o time.txt STAR --runThreadN 16 --genomeDir $REF/star_2.7.11b \
  --readFilesIn $SCR/r$CDNA.fastq.gz $SCR/r$CB.fastq.gz --readFilesCommand zcat \
  --soloType CB_UMI_Simple --soloCBwhitelist $W/pipeline-benchmark/whitelists/3M-february-2018.txt \
  --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 --soloBarcodeReadLength 0 \
  --soloFeatures GeneFull --soloCellFilter EmptyDrops_CR \
  --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR \
  --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30 \
  --outSAMtype None --outTmpDir $SCR/startmp > run.log 2>&1
grep -E "Estimated Number of Cells|Median UMI per Cell|Median GeneFull per Cell|Sequencing Saturation|UMIs in Cells" Solo.out/GeneFull/Summary.csv
grep -E "Elapsed|Maximum resident" time.txt
