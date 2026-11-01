#!/bin/bash
#SBATCH --job-name=s7-identity
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --nice=10000
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,BIN=singlet-s7-cr4[,N=3000000] identity.sbatch
# Production-path identity check for the Beta aligner profile: the first N spots of the real PBMC-E pbmc1 lane
# (ground-truth/raw/PBMC-E/ERR16108677) go through (a) the frozen bench STAR command (K5, improvements/pbmc1/K5/command.txt)
# on R2 and (b) singlet encode --no-trim --no-dedup + singlet --pipeline (profile on by default for 10x 3' v3).
# Read names carry the spot ordinal; compare_bam.py compares per read: strand, chrom, pos, CIGAR, NH, AS.
# Result: $W/jobs/fq-read-source/identity/<jobid>/summary.json (+ both Log.final.out). Nothing large is kept.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; N=${N:-3000000}
BIN=$W/jobs/fq-read-source/bin/$BIN
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
RAW=/mnt/projects/debruinz_project/debruinz/ground-truth/raw/PBMC-E
ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH
O=$W/jobs/fq-read-source/identity/$SLURM_JOB_ID; mkdir -p $O
SCR=/tmp/$USER-s7id-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT; cd $SCR
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF TMPDIR=$SCR SINGLET_KEEP_STAR_BAM=1
pigz -dc $RAW/ERR16108677_1.fastq.gz | head -n $((N*4)) > R1.fq || true
pigz -dc $RAW/ERR16108677_2.fastq.gz | head -n $((N*4)) | awk 'NR%4==1{print "@r" (NR-1)/4; next}{print}' > R2.fq || true
wc -l R1.fq R2.fq
sha256sum $BIN | tee $O/bin.txt; cat $BIN.commit | tee -a $O/bin.txt
echo "== bench STAR (K5)"
STAR --runMode alignReads --genomeDir $G/star_2.7.11b --runThreadN 16 --genomeLoad NoSharedMemory --readFilesIn R2.fq \
  --outSAMattributes NH HI AS nM NM --outSAMtype BAM Unsorted --outBAMcompression 1 --outSAMunmapped None --outFilterScoreMin 10 \
  --clipAdapterType CellRanger4 --alignSJoverhangMin 8 --outFilterIntronMotifs RemoveNoncanonicalUnannotated \
  --outFilterMismatchNoverLmax 0.05 --outFileNamePrefix $SCR/bench_ --outTmpDir $SCR/tmp_bench > /dev/null
cp bench_Log.final.out $O/bench_Log.final.out
echo "== encode"
$BIN encode --reads R1.fq R2.fq -o s.1fq --protocol 10x-3p-v3 --no-trim --no-dedup --quality none --whitelist-dir $PB/whitelists 2>&1 | tail -3
echo "== pipeline"
$BIN s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 16 \
  --pipeline --output-format mtx --bam-sort-mode disk --genome-shared off --read-source chunk --sorted-bam > run.log 2>&1 || { tail -30 run.log; exit 1; }
grep -E "CR4 probe|Beta aligner" run.log | tee $O/probe.txt
cp out/star_Log.final.out $O/pipeline_Log.final.out; cp out/cr4_probe.json $O/ 2>/dev/null || true
BAM=$(ls out/star_Aligned.sortedByCoord.out.bam)
python3 $W/jobs/fq-read-source/compare_bam.py bench_Aligned.out.bam $BAM samtools | tee $O/summary.json
