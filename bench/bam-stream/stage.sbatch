#!/bin/bash
#SBATCH --job-name=bs-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#   sbatch --array=0-2%3 stage.sbatch    (index into GSMS below)
# Stages public samples as .1fq under $W/data/bam-stream/<GSM>/ (temporary: deleted when the experiment ends).
# Network-bound I/O job: ~2 core-h each. The .1fq is made with the pinned campaign binary.
set -euo pipefail
GSMS=(GSM4006843 GSM7901264 GSM5411066)   # 3' v3 human 67M, 5' human 100M, mouse 3' v3 70M
GSM=${GSMS[$SLURM_ARRAY_TASK_ID]}
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/singlet-campaign
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome SINGLET_WHITELIST_DIR=$PB/whitelists
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
SRRS=$(in_runtime python -c "
import pandas as pd
s=pd.read_parquet('$W/data/atlas-rerun/samples.parquet',columns=['gsm_id','srr_accessions']).set_index('gsm_id')
print(' '.join(str(s.loc['$GSM','srr_accessions']).replace(',',' ').replace(';',' ').split()))")
echo "GSM=$GSM SRRS=$SRRS"
D=$W/data/bam-stream/$GSM; mkdir -p $D
SCR=/tmp/$USER-bsst-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
[ $(echo $SRRS | wc -w) -eq 1 ] || { echo "multi-run sample, not handled here"; exit 2; }
R=$SRRS
prefetch --max-size u -O $SCR/sra $R > pf.log 2>&1 || { tail pf.log; exit 1; }
$BIN download $(ls $SCR/sra/$R/*.sra* | head -1) --vdb-threads 4 --no-profile -o $D/$R.1fq.tmp > dl.log 2>&1 || { tail -20 dl.log; exit 1; }
mv $D/$R.1fq.tmp $D/$R.1fq; ln -sf $R.1fq $D/sample.1fq
$BIN decode $D/$R.1fq --header 2>&1 | head -8
ls -l $D
