#!/bin/bash
#SBATCH --job-name=eng-gpu-prof
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU time per pipeline stage: runs one sample with each engine, stamps every engine log line with the wall clock and samples the job
# cgroup's CPU counter every second; stage_profile.py attributes the core-s between log markers to stages. Outputs stay in job /tmp.
set -uo pipefail
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:?set BIN=$W/builds/<commit8>/bin/<binary> (an immutable build from scripts/engine_gpu/build_clipper.sbatch)}; pin "$BIN"; RES=$W/jobs/w11/engine/results/profile-$SLURM_JOB_ID; mkdir -p $RES
S=${SET:-pbmc1k_v3}; T=${T:-9}; ENGINES=(${ENGINES:-gpu cpu}); LIMIT=${LIMIT:-2400}
SCR=/tmp/$USER-engprof-$SLURM_JOB_ID; mkdir -p $SCR
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 2; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF
G=$REF/GRCh38-2024-A
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
FQ=$PB/data/$S/$S.1fq; [ -e $FQ ] || FQ=$W/data/atlas-rerun/perf/$S/sample.1fq
echo "node=$(hostname) job=$SLURM_JOB_ID set=$S T=$T bin=$(sha256sum $BIN | cut -c1-16) $(date -Is)"
for E in "${ENGINES[@]}"; do
  mkdir -p $SCR/$E/out
  ( while true; do echo "$(date +%s.%N) $(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)"; sleep 1; done ) > $RES/cpu_$E.txt &
  MON=$!
  ( cd $SCR/$E && timeout -k 30 $LIMIT /usr/bin/time -v -o time.txt $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $SCR/$E/out/ --threads $T --pipeline --output-format 1pz --engine $E 2>&1 \
      | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $RES/log_$E.txt; echo "exit ${PIPESTATUS[0]}" >> $RES/log_$E.txt )
  kill $MON 2>/dev/null
  cp $SCR/$E/time.txt $RES/time_$E.txt 2>/dev/null
  cp $SCR/$E/out/star_Log.final.out $RES/star_final_$E.txt 2>/dev/null
  cp $SCR/$E/out/star_Log.out $RES/star_log_$E.txt 2>/dev/null
  tail -2 $RES/log_$E.txt
done
echo done $(date -Is)
