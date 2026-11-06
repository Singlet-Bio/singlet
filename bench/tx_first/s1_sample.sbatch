#!/bin/bash
#SBATCH --job-name=txf-sample
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:30:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Study 1 per sample (CPU): first N read pairs of lane 1, cDNA read (R2) only.
#  1. STAR 2.7.11b on the genome with the pipeline's index (the reference record; the pipeline's CPU engine is this STAR).
#  2. STAR on the transcriptome of the same GTF: end to end, at most 1 mismatch, no introns (the transcriptome-first pass).
#  3. tx_concordance.py: projection of the transcriptome alignments to the genome and comparison.
# Tasks: 0 pbmc1k_v3 (human 3'), 1 nuclei2k_v2 (mouse nuclei, intron rich), 2 neuron1k_v3 (mouse 3').
# Parameters: NREADS (default 6000000), RES (default $W/jobs/tx-first/results/study1).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; PB=$W/pipeline-benchmark/data
export PATH=$W/envs/pbench/bin:$PATH
NREADS=${NREADS:-6000000}; RES=${RES:-$W/jobs/tx-first/results/study1}; mkdir -p $RES
BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/tx-first}
SCR=/tmp/$USER-txf-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf "${SCR:?}"' EXIT
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:refs/txf/$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/txf/$SLURM_JOB_ID bench/tx_first | tar -x -C $SCR/repo; git -C $BARE update-ref -d refs/txf/$SLURM_JOB_ID
case ${SLURM_ARRAY_TASK_ID} in
  0) S=pbmc1k_v3;   G=GRCh38-2024-A; R2=$PB/pbmc1k_v3/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L001_R2_001.fastq.gz ;;
  1) S=nuclei2k_v2; G=GRCm39-2024-A; R2=$PB/nuclei2k_v2/nuclei_2k_fastqs/nuclei_2k_S1_L001_R2_001.fastq.gz ;;
  2) S=neuron1k_v3; G=GRCm39-2024-A; R2=$PB/neuron1k_v3/neuron_1k_v3_fastqs/neuron_1k_v3_S1_L001_R2_001.fastq.gz ;;
esac
TXI=$W/jobs/tx-first/txidx/$G; D=$SCR/$S; mkdir -p $D
echo "sample $S genome $G reads $NREADS $(date -Is)"
set +o pipefail; zcat $R2 | head -n $((NREADS * 4)) > $D/r2.fq; set -o pipefail
echo "reads extracted: $(( $(wc -l < $D/r2.fq) / 4 ))"
STAR --runThreadN 16 --genomeDir $REF/$G/star_2.7.11b --readFilesIn $D/r2.fq --outSAMtype BAM Unsorted --outSAMattributes NH HI AS nM \
     --outSAMunmapped Within --outFileNamePrefix $D/g_ > $D/g.stdout
echo "genome STAR done $(date -Is)"; tail -n +6 $D/g_Log.final.out | head -30
STAR --runThreadN 16 --genomeDir $TXI/star --readFilesIn $D/r2.fq --outSAMtype BAM Unsorted --outSAMattributes NH HI AS nM \
     --alignEndsType EndToEnd --outFilterMismatchNmax 1 --alignIntronMax 1 --outFilterMultimapNmax 50 \
     --outFilterScoreMinOverLread 0.9 --outFilterMatchNminOverLread 0.9 --outSAMunmapped None --outFileNamePrefix $D/t_ > $D/t.stdout
echo "transcriptome STAR done $(date -Is)"; grep -E "Uniquely mapped reads %|multiple loci %|Number of input" $D/t_Log.final.out
python3 $SCR/repo/bench/tx_first/tx_concordance.py $TXI/tx.tsv $D/t_Aligned.out.bam $D/g_Aligned.out.bam $S $RES/$S.json | cut -c1-600
cp $D/g_Log.final.out $RES/$S.genome.Log.final.out; cp $D/t_Log.final.out $RES/$S.tx.Log.final.out
echo done $(date -Is)
