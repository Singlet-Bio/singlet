#!/bin/bash
#SBATCH --job-name=bpub-mage
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# One MAGE individual (row N of mage_subset.tsv): the first 8M read pairs of its LCL RNA-seq run, aligned once with STAR
# (Singlet's GRCh38-2024-A index), then thinned to 0.5M / 2M / 8M read pairs, the range of a single-cell donor's
# coverage. For each depth, the inputs every tool needs from the SAME alignments:
#   private/<id>/<L>/sites.parquet   alt/total read depth at the pipeline's SNP panel (bcftools mpileup, MAPQ 255, BQ 20):
#                                    what singlet.celllabels.ancestry reads. Genotype-like: stays here, never published.
#   somalier/<L>/<id>.somalier       somalier extract at its hg38 sites
#   peddy/<L>/<id>.vcf.gz            bcftools call at peddy's GRCh38 sites
#   genecounts/<id>.tsv              STAR GeneCounts (XIST and chrY genes feed the bulk sex comparison)
#   sbatch --array=1-150%12 mage.sbatch
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
BPE=$WORK/envs/bench-public; SOM=$WORK/tools/bench-public/somalier
IFS=$'\t' read -r RUN IND POP SUPER SEX NREADS BYTES FTP < <(tail -n +2 $SRC/bench/public/sexancestry/mage_subset.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
O=$BP/work/ancestry/mage; P=$O/private/$IND; mkdir -p $P $O/genecounts $O/timing
[ -e $P/.done ] && { echo "$IND done"; exit 0; }
SCR=/tmp/$USER-mage-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
G=$REF/GRCh38-2024-A; FA=$G/fasta/genome.fa; T=$SLURM_CPUS_PER_TASK
echo "[mage] $IND $POP $SUPER $SEX $RUN $(date -Is) $(hostname)"
t0=$(date +%s)
prefetch -q -O $SCR $RUN && fasterq-dump -q -e $T -t $SCR -O $SCR $SCR/$RUN
for i in 1 2; do head -n 32000000 $SCR/${RUN}_$i.fastq > $SCR/r$i.fq; rm $SCR/${RUN}_$i.fastq; done
t1=$(date +%s)
/usr/bin/time -v -o $O/timing/$IND.star.txt STAR --runThreadN $T --genomeDir $G/star_2.7.11b --readFilesIn $SCR/r1.fq $SCR/r2.fq \
  --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 30000000000 --quantMode GeneCounts \
  --outSAMattrRGline ID:$IND SM:$IND --outFileNamePrefix $SCR/ > $P/star.log 2>&1
cp $SCR/ReadsPerGene.out.tab $O/genecounts/$IND.tsv
$BPE/bin/samtools index -@ $T $SCR/Aligned.sortedByCoord.out.bam
t2=$(date +%s)
PEDDY_SITES=$($BPE/bin/python -c "import peddy, os; print(os.path.join(os.path.dirname(peddy.__file__), 'GRCH38.sites'))")
[ -s "$PEDDY_SITES" ] && $BPE/bin/python $SRC/bench/public/sexancestry/sites_to_targets.py $PEDDY_SITES $SCR/peddy_targets.tsv
for L in 8 2 0.5; do
  B=$SCR/L$L.bam
  if [ $L = 8 ]; then ln -sf $SCR/Aligned.sortedByCoord.out.bam $B; ln -sf $SCR/Aligned.sortedByCoord.out.bam.bai $B.bai
  else FR=$(awk -v l=$L 'BEGIN{printf "%.4f", l/8}'); $BPE/bin/samtools view -@ $T -b -s 7${FR#0} -o $B $SCR/Aligned.sortedByCoord.out.bam && $BPE/bin/samtools index $B; fi
  mkdir -p $P/$L $O/somalier/$L $O/peddy/$L
  ts=$(date +%s.%N)
  $BPE/bin/bcftools mpileup -f $FA -T $BP/work/ancestry/targets.chr.tsv -a AD -q 255 -Q 20 -d 10000 -I -Ou $B 2>/dev/null \
    | $BPE/bin/bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%AD]\n' > $SCR/pile.tsv
  $BPE/bin/python $SRC/bench/public/sexancestry/pileup_to_sites.py $SCR/pile.tsv $BP/work/ancestry/af_leaveout.parquet $P/$L/sites.parquet
  tp=$(date +%s.%N)
  $SOM/somalier extract -d $O/somalier/$L -s $SOM/sites.hg38.vcf.gz -f $FA $B > /dev/null 2>&1
  tq=$(date +%s.%N)
  if [ -s $SCR/peddy_targets.tsv ]; then
    $BPE/bin/bcftools mpileup -f $FA -T $SCR/peddy_targets.tsv -a AD,DP -q 255 -Q 20 -d 10000 -Ou $B 2>/dev/null \
      | $BPE/bin/bcftools call -m -Oz -o $O/peddy/$L/$IND.vcf.gz && $BPE/bin/bcftools index -t $O/peddy/$L/$IND.vcf.gz
  fi
  tr=$(date +%s.%N)
  # seconds for pileup, somalier extract, peddy calling
  echo "$IND $L $ts $tp $tq $tr" | awk -v OFS='\t' '{print $1, $2, $4-$3, $5-$4, $6-$5}' >> $O/timing/$IND.levels.tsv
done
mv $SCR/Aligned.sortedByCoord.out.bam $P/aligned.bam; mv $SCR/Aligned.sortedByCoord.out.bam.bai $P/aligned.bam.bai
echo -e "$IND\tdownload_s\t$((t1-t0))\tstar_s\t$((t2-t1))" > $O/timing/$IND.tsv
touch $P/.done; echo "[mage] $IND done $(date -Is)"
