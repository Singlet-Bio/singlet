#!/bin/bash
#SBATCH --job-name=dp-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:59:00
#SBATCH --array=0-6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Re-stage the Clipper smoke samples' runs as the orchestrator did (prefetch -> `singlet download <sra>`), into
# $W/data/atlas-rerun/perf/<GSM>/<SRR>.1fq, and pool multi-run samples into sample.1fq with the orchestrator's
# decode + encode (stages.pool_runs) so the smoke input is reproduced. Binary = the smoke binary (singlet-campaign).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN=$PB/bin/singlet-campaign
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome SINGLET_WHITELIST_DIR=$PB/whitelists
SAMPLES=(GSM7656149:SRR25376447 GSM7036403:SRR23353990 GSM6470208:SRR21102187,SRR21102188 GSM6250705:SRR25751252
         GSM6996272:SRR23246554,SRR23246555,SRR23246556,SRR23246557 GSM6996273:SRR23246550,SRR23246551,SRR23246552,SRR23246553
         GSM6996274:SRR23246546,SRR23246547,SRR23246548,SRR23246549)
IFS=: read GSM RUNS <<< "${SAMPLES[$SLURM_ARRAY_TASK_ID]}"
D=$W/data/atlas-rerun/perf/$GSM; mkdir -p $D
SCR=/tmp/$USER-dpf-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
IFS=, read -ra RR <<< "$RUNS"
for r in "${RR[@]}"; do
  [ -s $D/$r.1fq ] && { echo "have $r"; continue; }
  /usr/bin/time -f "prefetch $r %e s" prefetch --max-size u -O $SCR/sra $r > $SCR/pf.log 2>&1 || { tail $SCR/pf.log; exit 1; }
  SRA=$(ls $SCR/sra/$r/*.sra* | head -1)
  /usr/bin/time -f "download $r %e s maxrss %M KB" $BIN download $SRA --vdb-threads 4 --no-profile -o $D/$r.1fq.tmp > $SCR/dl.log 2>&1 \
    || { tail -20 $SCR/dl.log; exit 1; }
  grep -E "Protocol|R1 length|Total spots|Done" $SCR/dl.log | tr '\r' '\n' | grep -v "reads (" || true
  mv $D/$r.1fq.tmp $D/$r.1fq; rm -rf $SCR/sra/$r
done
if [ ${#RR[@]} -gt 1 ] && [ ! -s $D/sample.1fq ]; then
  rm -f $SCR/pool_R1.fq $SCR/pool_R2.fq
  t0=$(date +%s)
  for r in "${RR[@]}"; do
    $BIN decode $D/$r.1fq --r1 $SCR/t_R1.fq --r2 $SCR/t_R2.fq --threads 1 > $SCR/dec.log 2>&1 || { tail $SCR/dec.log; exit 1; }
    cat $SCR/t_R1.fq >> $SCR/pool_R1.fq; cat $SCR/t_R2.fq >> $SCR/pool_R2.fq; rm $SCR/t_R1.fq $SCR/t_R2.fq
  done
  /usr/bin/time -f "pool-encode %e s maxrss %M KB" $BIN encode --reads $SCR/pool_R1.fq $SCR/pool_R2.fq -o $D/sample.1fq.tmp \
    --no-dedup --no-trim --quality none > $SCR/enc.log 2>&1 || { tail -20 $SCR/enc.log; exit 1; }
  grep -E "Protocol|R1 length|Total spots" $SCR/enc.log || true
  mv $D/sample.1fq.tmp $D/sample.1fq
  echo "pool total $(( $(date +%s) - t0 )) s"
elif [ ${#RR[@]} -eq 1 ] && [ ! -e $D/sample.1fq ]; then
  ln -s ${RR[0]}.1fq $D/sample.1fq
fi
for f in $D/*.1fq; do echo "== $f"; $BIN decode $f --header 2>&1 | head -12; done
ls -la $D
