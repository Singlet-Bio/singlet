#!/bin/bash
#SBATCH --job-name=d101-dl
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# D-101 validation data, Oetjen et al. 2018 human bone marrow (GSE120221 10x 3' v2, GSE120446 bulk RNA-seq of the same donors).
# Not in the NM corpus (sprint/intel/corpus-engine-touch.md). Row SLURM_ARRAY_TASK_ID of bench/d101/libs.tsv (1-based, header excluded).
#   KIND=sc   : `singlet download <sc_run>` -> $W/data/d101/sc/<sc_run>.1fq
#   KIND=bulk : fasterq-dump <bulk_run> (paired, 2 x 150) -> $W/data/d101/bulk/<bulk_run>_{1,2}.fastq.gz
# BIN: any singlet binary with `download` (default: the gpu/frontier build 654109ad). LIBS: path of libs.tsv on Clipper.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
KIND=${KIND:?sc or bulk}; LIBS=${LIBS:?}; BIN=${BIN:-$W/jobs/gpu-pr2/bin2/654109ad/singlet-gpu}
ROW=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS); [ -n "$ROW" ] || { echo "no row $SLURM_ARRAY_TASK_ID"; exit 0; }
read -r DONOR SC SCG BULK BULKG <<< "$ROW"
SCR=/tmp/$USER-d101dl-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
if [ $KIND = sc ]; then
  D=$W/data/d101/sc; mkdir -p $D; O=$D/$SC.1fq
  [ -s $O ] && { echo "have $O"; exit 0; }
  echo "donor $DONOR $SC ($SCG)"
  ( cd $SCR && $BIN download $SC --vdb-threads 4 -o $SCR/$SC.1fq ) 2>&1 | tail -25
  mv $SCR/$SC.1fq $O.part && mv $O.part $O
  ls -l $O
else
  [ "$BULK" = "-" ] && { echo "donor $DONOR has no bulk run"; exit 0; }
  D=$W/data/d101/bulk; mkdir -p $D
  [ -s $D/${BULK}_2.fastq.gz ] && { echo "have $BULK"; exit 0; }
  echo "donor $DONOR $BULK ($BULKG)"
  ( cd $SCR && prefetch -q -O $SCR $BULK && fasterq-dump -q -e 6 -t $SCR --split-files -O $SCR $SCR/$BULK/$BULK.sra )
  for m in 1 2; do gzip -1 -c $SCR/${BULK}_$m.fastq > $D/${BULK}_$m.fastq.gz.part && mv $D/${BULK}_$m.fastq.gz.part $D/${BULK}_$m.fastq.gz; done
  ls -l $D/${BULK}_*
fi
