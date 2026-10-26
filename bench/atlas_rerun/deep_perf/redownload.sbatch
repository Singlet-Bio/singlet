#!/bin/bash
#SBATCH --job-name=dp-redl
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch redownload.sbatch <binary> <GSM> <run>
# prefetch + `singlet download <sra>` with another binary -> $W/data/atlas-rerun/perf/<GSM>/<binary>/<run>.1fq
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/$1; GSM=$2; R=$3
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome SINGLET_WHITELIST_DIR=$PB/whitelists
D=$W/data/atlas-rerun/perf/$GSM/$1; mkdir -p $D
SCR=/tmp/$USER-rdl-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
prefetch --max-size u -O $SCR/sra $R > pf.log 2>&1 || { tail pf.log; exit 1; }
/usr/bin/time -f "download %e s maxrss %M KB" $BIN download $(ls $SCR/sra/$R/*.sra* | head -1) --vdb-threads 4 --no-profile \
  -o $D/$R.1fq.tmp > dl.log 2>&1 || { tail -20 dl.log; exit 1; }
tr '\r' '\n' < dl.log | grep -E "protocol_detect|Protocol|R1 length|UMI|Done|download " | grep -v "reads ("
mv $D/$R.1fq.tmp $D/$R.1fq; ln -sf $R.1fq $D/sample.1fq
$BIN decode $D/$R.1fq --header 2>&1 | head -5
