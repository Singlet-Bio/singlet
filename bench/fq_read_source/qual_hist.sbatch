#!/bin/bash
#SBATCH --job-name=fqrs-qhist
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-6%4
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#   sbatch qual_hist.sbatch
# Chooses the .1fq quality bins from data: R2 (cDNA) Phred histogram of the first 5 M spots of a run, and the .1fq
# size with --quality none / binned / full on the same spots. Reads stay in job /tmp; only the histogram and sizes
# come back ($W/jobs/fq-read-source/qhist/<run>.txt).
set -euo pipefail
RUNS=(GSM4006843 GSM5411066 SRR21102187 SRR23353990 SRR25376447 SRR25751252 GSM7901264)
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/singlet-campaign
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome SINGLET_WHITELIST_DIR=$PB/whitelists
R=${RUNS[$SLURM_ARRAY_TASK_ID]}
if [[ $R == GSM* ]]; then
  source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
  R=$(in_runtime python -c "
import pandas as pd
s=pd.read_parquet('$W/data/atlas-rerun/samples.parquet',columns=['gsm_id','srr_accessions']).set_index('gsm_id')
print(str(s.loc['$R','srr_accessions']).replace(',',' ').replace(';',' ').split()[0])")
fi
OUT=$W/jobs/fq-read-source/qhist; mkdir -p $OUT
SCR=/tmp/$USER-fqqh-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
prefetch --max-size u -O $SCR/sra $R > pf.log 2>&1 || { tail pf.log; exit 1; }
SRA=$(ls $SCR/sra/$R/*.sra* | head -1)
{
echo "run $R file $(basename $SRA) bytes $(stat -c %s $SRA)"
for q in none binned full; do
  $BIN download $SRA --vdb-threads 4 --no-profile --max-reads 5000000 --quality $q -o $SCR/$q.1fq > dl_$q.log 2>&1 || { tail -20 dl_$q.log; exit 1; }
  echo "size $q $(stat -c %s $SCR/$q.1fq)"
done
$BIN decode $SCR/full.1fq --header 2>&1 | head -14
$BIN decode $SCR/full.1fq --r2 $SCR/r2.fq --r1 /dev/null --threads 4 > dec.log 2>&1 || { tail dec.log; exit 1; }
echo "r2 length histogram (top 5):"
awk 'NR%4==2{print length($0)}' r2.fq | sort -n | uniq -c | sort -k1,1nr | head -5
echo "phred histogram (R2, all bases):"
awk 'BEGIN{for(i=33;i<127;i++) o[sprintf("%c",i)]=i-33} NR%4==0{n=length($0); for(i=1;i<=n;i++) c[o[substr($0,i,1)]]++} END{for(q in c) print q, c[q]}' r2.fq | sort -n
} > $OUT/$R.txt 2>&1
cat $OUT/$R.txt
