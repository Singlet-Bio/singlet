#!/bin/bash
#SBATCH --job-name=cpuperf
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-engine profile, stream path, K5 + default rules. Env: SET (pbmc1k_v3|neuron1k_v3|GSM7036403), T (threads per run), PACK (concurrent runs
# on this node, default 1), BINV (singlet-cpu | singlet-cpu-prof), CM (commit dir under jobs/cpu-perf, default d9834b8a), TAG.
# Size with sbatch: --cpus-per-task=$((T*PACK)) --mem=<PACK*36+20>G. Per run: the log with wall-clock stamps, time -v (RSS), and for PACK=1 the job
# cgroup's CPU counter every second (scripts/cpu_perf/stage_profile.py). Every run must print STREAM PATH OK or the job exits 3.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
S=${SET:-pbmc1k_v3}; T=${T:-4}; PACK=${PACK:-1}; CM=${CM:-d9834b8a}; BINV=${BINV:-singlet-cpu}; LIMIT=${LIMIT:-12000}
BIN=$W/jobs/cpu-perf/$CM/bin/$BINV; TAG=${TAG:-$S-t$T-p$PACK-$BINV-$CM}; RES=$W/jobs/cpu-perf/results/$TAG-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-cpuperf-$SLURM_JOB_ID; mkdir -p $SCR
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 2; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
export SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF      # no SINGLET_WHITELIST_DIR: the whitelists must be found next to the binary
G=$REF/GRCh38-2024-A; [ "${MOUSE:-0}" = 1 ] && G=$REF/GRCm39-2024-A
FQ=$PB/data/$S/$S.1fq; [ -e $FQ ] || FQ=$W/data/atlas-rerun/perf/$S/sample.1fq
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID set=$S T=$T pack=$PACK bin=$BINV commit=$(cat $W/jobs/cpu-perf/$CM/commit) cpus=$SLURM_CPUS_PER_TASK $(lscpu | grep 'Model name' | sed 's/  */ /g') $(date -Is)"
[ -e $W/jobs/cpu-perf/$CM/whitelists ] || { echo "no whitelists next to the binary"; exit 2; }
if [ $PACK = 1 ]; then ( while true; do echo "$(date +%s.%N) $(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)"; sleep 1; done ) > $RES/cpu.txt & MON=$!; fi
if [ $PACK = 1 ]; then ( while true; do echo "@ $(date +%s)"; P=$(tree $$ | tr '
' ' '); ps -eLo pid=,lwp=,comm=,time= | awk -v P=" $P " 'index(P," "$1" ")'; sleep 10; done ) > $RES/threads.txt 2>/dev/null & MON2=$!; fi
if [ $PACK = 1 ]; then ( while true; do echo "@ $(date +%s)"; for p in $(tree $$) ; do [ -r /proc/$p/smaps_rollup ] && echo "$p $(cat /proc/$p/comm) $(awk '/^(Rss|Pss|Private_Clean|Private_Dirty|Shared_Clean|Shared_Dirty|Anonymous):/{printf "%s=%s ",$1,$2}' /proc/$p/smaps_rollup)"; done; sleep 15; done ) > $RES/mem.txt 2>/dev/null & MON3=$!; fi
GS=""; if [ "${SHARED:-0}" = 1 ]; then GS="--genome-shared on"; $BIN genome load $G/star_2.7.11b > $RES/genome_load.txt 2>&1; $BIN genome status $G/star_2.7.11b >> $RES/genome_load.txt 2>&1; fi   # SHARED=1: STAR genome in SysV shared memory, loaded once (S8's singlet genome load), runs attach
S0=$(date +%s.%N)
for i in $(seq 1 $PACK); do
  mkdir -p $SCR/r$i/out
  ( cd $SCR/r$i && timeout -k 30 $LIMIT /usr/bin/time -v -o $RES/time_$i.txt $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $SCR/r$i/out/ --threads $T --pipeline --output-format 1pz --engine cpu $GS 2>&1 \
      | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $RES/log_$i.txt; echo "exit ${PIPESTATUS[0]}" >> $RES/log_$i.txt
    cp out/star_Log.final.out $RES/final_$i.txt 2>/dev/null; cp out/star_Log.out $RES/starlog_$i.txt 2>/dev/null
    ( cd out && find . -maxdepth 2 -type f \( -name '*.mtx*' -o -name '*.1pz' -o -name '*.tsv' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum ) > $RES/hashes_$i.txt 2>/dev/null ) &
  PIDS="${PIDS:-} $!"
done
wait $PIDS; E0=$(date +%s.%N); [ -n "${MON:-}" ] && kill $MON $MON2 $MON3 2>/dev/null
[ "${SHARED:-0}" = 1 ] && $BIN genome unload $G/star_2.7.11b >> $RES/genome_load.txt 2>&1
echo "pack wall $(echo "$E0 - $S0" | bc) s" | tee $RES/pack_wall.txt
rc=0
for i in $(seq 1 $PACK); do
  L=$RES/log_$i.txt
  if grep -aq "BAM: unsorted STAR stream into the pileup" $L && grep -aq "Pileup: streaming from" $L && ! grep -aq "Pileup: parallel" $L \
     && ! grep -aqi "whitelist.*not found\|no whitelist" $L && grep -aq "^exit 0" $L; then echo "run $i: STREAM PATH OK" | tee -a $RES/stream_path.txt
  else echo "run $i: STREAM PATH FAILED" | tee -a $RES/stream_path.txt; rc=3; fi
  grep -aE "Elapsed \(wall|Maximum resident|User time|System time" $RES/time_$i.txt | tr -s '\t' ' ' | tr '\n' ';'; echo
  grep -aE "Number of input reads" $RES/final_$i.txt
done
exit $rc
