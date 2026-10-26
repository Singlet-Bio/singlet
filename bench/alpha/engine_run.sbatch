#!/bin/bash
#SBATCH --job-name=alpha-eng
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Research Alpha benchmark B: the published atlas engine at the campaign's exact settings on the 7 public 10x datasets
# of the pipeline benchmark: campaign binary (8e1803ba), first 30M read pairs (campaign `download --max-reads 30000000`
# keeps the head of the run; here the head of the lane-sorted FASTQs), `--threads 4 --pipeline --output-format 1pz`,
# plus the human-only campaign flags (--snps, --cascade auto, --txome-index, --te-classify on, --te-db) for human data.
# Output: $WORK/pipeline-benchmark/runs/<dataset>/singlet-alpha30m/
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
PB=$WORK/pipeline-benchmark
DS=(pbmc1k_v3 pbmc1k_v2 pbmc5p_v1 pbmc1k_cite_v3 neuron1k_v3 nuclei2k_v2 hgmm1k_v3)
ID=${DS[$SLURM_ARRAY_TASK_ID]}
BIN=$WORK/alpha_bench/bin/singlet-alpha-8e1803b
export SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REFERENCE_BUILD=GRCh38-2024-A
WL_DIR=$PB/whitelists
REF=$SINGLET_REF_BASE
case $ID in
  neuron1k_v3|nuclei2k_v2) SPEC=GRCm39-2024-A; export SINGLET_REFERENCE_BUILD=GRCm39-2024-A ;;
  hgmm1k_v3) SPEC=GRCh38-GRCm39-barnyard ;;
  *) SPEC=GRCh38-2024-A ;;
esac
GENOME=$REF/$SPEC/star_2.7.11b; GTF=$REF/$SPEC/genes/genes.gtf
HUMAN_FLAGS=""
if [ $SPEC = GRCh38-2024-A ]; then
  HUMAN_FLAGS="--snps $REF/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz --cascade auto --txome-index $REF/GRCh38-2024-A/txome/tx.fa --te-classify on --te-db $REF/GRCh38-2024-A/te/human/te_sketch.bin"
fi
D=$PB/data/$ID; O=$PB/runs/$ID/singlet-alpha30m; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "[alpha] $ID spec=$SPEC node=$(hostname) $(date -Is)" | tee $O/node.txt
lscpu | sed -n "s/^Model name: *//p" >> $O/node.txt
# first 30M read pairs, lanes in sorted order (R1 and R2 stay paired: same files, same order, same line count)
NL=$((30000000 * 4))
zcat $(grep "_R1_" $D/gex_fastqs.txt | sort) | head -n $NL > $SCR/R1.fastq || true
zcat $(grep "_R2_" $D/gex_fastqs.txt | sort) | head -n $NL > $SCR/R2.fastq || true
echo "[alpha] R1 lines $(wc -l < $SCR/R1.fastq) R2 lines $(wc -l < $SCR/R2.fastq)"
pigz -p 4 -1 $SCR/R1.fastq $SCR/R2.fastq 2>/dev/null || gzip -1 $SCR/R1.fastq $SCR/R2.fastq
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$ID.1fq \
    --whitelist-dir $WL_DIR > $O/encode.log 2>&1
rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
cd $SCR
set +e
/usr/bin/time -v -o $O/time.txt $BIN $SCR/$ID.1fq --ref-base $REF --genome-dir $GENOME --exons $GTF --out-prefix $O/ \
    --threads 4 --output-format 1pz --pipeline $HUMAN_FLAGS > $O/run.log 2>&1
RC=$?
set -e
echo "[alpha] exit $RC"; tail -5 $O/run.log
grep -E "Elapsed \(wall|Maximum resident|Percent of CPU" $O/time.txt || true
rm -f $O/star_Aligned.sortedByCoord.out.bam $O/star_Unmapped.out.mate* $O/nonhost_reads.fq
du -sh $O
