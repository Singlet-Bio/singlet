#!/bin/bash
#SBATCH --job-name=dmx2-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:59:00
#SBATCH --array=0-3%4
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Download the four new single-donor GEX PBMC libraries the way the fq-read-source setup did:
# prefetch the SRA run, then `singlet download --quality none` writes the production .1fq.
# Temporary: the .1fq is deleted after the pipeline run.
set -euo pipefail
SAMPLES=(GSM7873657 GSM7873658 GSM5102900 GSM5102901)
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN=$W/jobs/fq-read-source/bin/singlet-fqrs-pgo
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome SINGLET_WHITELIST_DIR=$PB/whitelists
GSM=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
SRRS=$(in_runtime python -c "
import pandas as pd
s=pd.read_parquet('$W/data/atlas-rerun/samples.parquet',columns=['gsm_id','srr_accessions']).set_index('gsm_id')
print(' '.join(str(s.loc['$GSM','srr_accessions']).replace(',',' ').replace(';',' ').split()))")
echo "GSM=$GSM SRRS=$SRRS bin=$(sha256sum $BIN | cut -c1-16) $(cat $BIN.commit)"
[ -n "$SRRS" ]
D=$W/data/demux-truth2/$GSM; mkdir -p $D/none
SCR=/tmp/$USER-dmx2st-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
for R in $SRRS; do
  [ -s $D/none/$R.1fq ] && { echo "have $R"; continue; }
  prefetch --max-size u -O $SCR/sra $R > pf.log 2>&1 || { tail pf.log; exit 1; }
  SRA=$(ls $SCR/sra/$R/*.sra* | head -1); echo "$R $(basename $SRA) $(stat -c %s $SRA) bytes"
  $BIN download $SRA --vdb-threads 4 --no-profile --quality none -o $D/none/$R.1fq.tmp > dl.log 2>&1 || { tail -20 dl.log; exit 1; }
  tr '\r' '\n' < dl.log | grep -E "barcodes:|Done|Protocol" | grep -v "reads (" | head -4 || true
  mv $D/none/$R.1fq.tmp $D/none/$R.1fq; echo "size $R $(stat -c %s $D/none/$R.1fq)"
  rm -rf $SCR/sra/$R
done
ls -la $D/none
