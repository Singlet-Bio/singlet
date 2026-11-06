#!/bin/bash
#SBATCH --job-name=txf-prep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Study 1 prep (CPU): transcriptome FASTA and transcript table from the GTF of the STAR index, then a transcriptome STAR index.
# Task 0 = GRCh38-2024-A, task 1 = GRCm39-2024-A. Scripts come from branch bench/tx-first (fetched as bench/gpu-batch1's tp.sbatch does).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference
export PATH=$W/envs/pbench/bin:$PATH
BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/tx-first}
SCR=/tmp/$USER-txf-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf "${SCR:?}"' EXIT
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:refs/txf/$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/txf/$SLURM_JOB_ID bench/tx_first | tar -x -C $SCR/repo; git -C $BARE update-ref -d refs/txf/$SLURM_JOB_ID
echo "scripts $(git -C $BARE rev-parse --short $SBR 2>/dev/null || echo ?) task ${SLURM_ARRAY_TASK_ID}"
G=$( [ "${SLURM_ARRAY_TASK_ID}" = 0 ] && echo GRCh38-2024-A || echo GRCm39-2024-A )
OUT=$W/jobs/tx-first/txidx/$G; mkdir -p $OUT
if [ ! -f $OUT/tx.tsv ]; then python3 $SCR/repo/bench/tx_first/build_tx.py $REF/$G/fasta/genome.fa $REF/$G/genes/genes.gtf $OUT; fi
if [ ! -f $OUT/star/SAindex ]; then
  mkdir -p $OUT/star
  STAR --runMode genomeGenerate --runThreadN 8 --genomeDir $OUT/star --genomeFastaFiles $OUT/tx.fa \
       --genomeSAindexNbases 12 --genomeChrBinNbits 11 --outFileNamePrefix $SCR/gen_
fi
ls -la $OUT $OUT/star | head -30; echo done $(date -Is)
