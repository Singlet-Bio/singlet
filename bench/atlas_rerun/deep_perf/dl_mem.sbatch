#!/bin/bash
#SBATCH --job-name=dp-dlmem
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch dl_mem.sbatch <binary> <run> [vdb_threads] [max_reads]
# `singlet download RUN --max-reads N` (remote VDB stream, as the orchestrator's head screen) with peak RSS and an RSS
# trace every 2 s next to the timestamped log, to find what grows with the run instead of with N.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN=$PB/bin/$1; RUN=$2; VT=${3:-2}; N=${4:-1000000}
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome SINGLET_WHITELIST_DIR=$PB/whitelists
SCR=/tmp/$USER-dlm-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
echo "[dlmem] $1 $RUN vdb_threads=$VT max_reads=$N node=$(hostname) $(date -Is)"
/usr/bin/time -v -o $SCR/time.txt $BIN download $RUN --max-reads $N --vdb-threads $VT --no-profile -o $SCR/head.1fq \
  2> >(while IFS= read -r l; do printf '%s %s\n' "$(date +%s)" "$l"; done > $SCR/dl.log) &
P=$!
sleep 1
while kill -0 $P 2>/dev/null; do
  C=$(pgrep -P $P -f "download" | head -1); [ -z "$C" ] && C=$(pgrep -f "$BIN download $RUN" | head -1)
  [ -n "$C" ] && echo "$(date +%s) rss_kb=$(awk '/VmRSS/{print $2}' /proc/$C/status 2>/dev/null) threads=$(ls /proc/$C/task 2>/dev/null | wc -l)" >> $SCR/rss.txt
  sleep 2
done
wait $P; echo "rc=$?"
tr '\r' '\n' < $SCR/dl.log | grep -v "reads (" | tail -30
awk 'NR%10==1' $SCR/rss.txt | tail -60
grep -E "Elapsed|Maximum resident" $SCR/time.txt
ls -la $SCR/head.1fq 2>/dev/null && $BIN decode $SCR/head.1fq --header 2>&1 | head -5
