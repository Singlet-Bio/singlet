#!/bin/bash
#SBATCH --job-name=fqrs-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=100
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#   sbatch --array=0-N%4 --export=ALL,BIN=singlet-fqrs-pgo stage.sbatch
# Stages the evidence samples under $W/data/fq-read-source/<GSM>/: every run downloaded twice with the new binary,
# --quality none (none/<SRR>.1fq: the production .1fq today) and the default ILLUMINA4 (ilmn4/<SRR>.1fq), and for a
# multi-run sample also the orchestrator's pool (decode each run, cat, encode --quality none: pooled.1fq), timed.
# Temporary: deleted when the evidence is done (no reads kept long-term).
set -euo pipefail
SAMPLES=(GSM4006843 GSM7901264 GSM6470208 ${EXTRA_SAMPLES:-})
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN=$W/jobs/fq-read-source/bin/$BIN
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome SINGLET_WHITELIST_DIR=$PB/whitelists
GSM=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
SRRS=$(in_runtime python -c "
import pandas as pd
s=pd.read_parquet('$W/data/atlas-rerun/samples.parquet',columns=['gsm_id','srr_accessions']).set_index('gsm_id')
print(' '.join(str(s.loc['$GSM','srr_accessions']).replace(',',' ').replace(';',' ').split()))")
echo "GSM=$GSM SRRS=$SRRS bin=$(sha256sum $BIN | cut -c1-16) $(cat $BIN.commit)"
D=$W/data/fq-read-source/$GSM; mkdir -p $D/none $D/ilmn4
SCR=/tmp/$USER-fqst-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
for R in $SRRS; do
  [ -s $D/ilmn4/$R.1fq ] && [ -s $D/none/$R.1fq ] && { echo "have $R"; continue; }
  prefetch --max-size u -O $SCR/sra $R > pf.log 2>&1 || { tail pf.log; exit 1; }
  SRA=$(ls $SCR/sra/$R/*.sra* | head -1); echo "$R $(basename $SRA) $(stat -c %s $SRA) bytes"
  for Q in none ilmn4; do
    QA=binned; [ $Q = none ] && QA=none
    /usr/bin/time -f "download $R $Q %e s %U user %S sys maxrss %M KB" $BIN download $SRA --vdb-threads 4 --no-profile \
      --quality $QA -o $D/$Q/$R.1fq.tmp > dl_$Q.log 2>&1 || { tail -20 dl_$Q.log; exit 1; }
    tr '\r' '\n' < dl_$Q.log | grep -E "barcodes:|Done|Protocol" | grep -v "reads (" | head -4 || true
    mv $D/$Q/$R.1fq.tmp $D/$Q/$R.1fq; echo "size $R $Q $(stat -c %s $D/$Q/$R.1fq)"
  done
  rm -rf $SCR/sra/$R
done
N=$(echo $SRRS | wc -w)
if [ $N -gt 1 ] && [ ! -s $D/pooled.1fq ]; then
  # the orchestrator's stages.pool_runs (jobs/deep-perf/fetch_smoke.sbatch): decode each run, cat, re-encode
  t0=$(date +%s.%N)
  for R in $SRRS; do
    /usr/bin/time -a -o pool.time -f "decode $R %e s %U user %S sys maxrss %M KB" $BIN decode $D/none/$R.1fq \
      --r1 $SCR/t_R1.fq --r2 $SCR/t_R2.fq --threads 1 > dec.log 2>&1 || { tail dec.log; exit 1; }
    cat $SCR/t_R1.fq >> $SCR/pool_R1.fq; cat $SCR/t_R2.fq >> $SCR/pool_R2.fq; rm $SCR/t_R1.fq $SCR/t_R2.fq
  done
  /usr/bin/time -a -o pool.time -f "encode %e s %U user %S sys maxrss %M KB" $BIN encode --reads $SCR/pool_R1.fq \
    $SCR/pool_R2.fq -o $D/pooled.1fq.tmp --no-dedup --no-trim --quality none > enc.log 2>&1 || { tail -20 enc.log; exit 1; }
  mv $D/pooled.1fq.tmp $D/pooled.1fq
  echo "pool total $(echo "$(date +%s.%N) - $t0" | bc) s"; cat pool.time
fi
ls -la $D/*
