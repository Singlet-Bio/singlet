#!/bin/bash
#SBATCH --job-name=eng-gpu-node
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --cpus-per-task=54
#SBATCH --mem=700G
#SBATCH --exclusive
#SBATCH --time=03:59:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# A whole Clipper GPU node (2 x H100 NVL, 54 schedulable cores) with several samples in flight: wall (makespan), read pairs/s per node and
# core-s, for --engine gpu and then --engine cpu on the same node and samples. Outputs in job /tmp; summary to results/.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:-$W/jobs/w11/engine/bin/singlet-gpu}; RES=$W/jobs/w11/engine/results/node-$SLURM_JOB_ID; mkdir -p $RES
SETS=(${SETS:-pbmc1k_v3 GSM7036403 GSM7656149 GSM6470208})
T=${T:-13}; ENGINES=(${ENGINES:-gpu cpu})
SCR=/tmp/$USER-engnode-$SLURM_JOB_ID; mkdir -p $SCR
LIMIT=${LIMIT:-5400}   # per-sample wall limit (s): a hung sample is killed instead of holding the node
# kill-safe: on exit or cancel, stop every child (samples, monitors) before removing the scratch, so nothing outlives the job
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 5; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
G=$REF/GRCh38-2024-A
echo "node=$(hostname) job=$SLURM_JOB_ID sets=${SETS[*]} T=$T bin=$(sha256sum $BIN | cut -c1-16) $(date -Is)"; nvidia-smi -L; lscpu | grep "Model name"
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
for E in "${ENGINES[@]}"; do
  ( while true; do echo "$(date +%s) $(awk '$1=="usage_usec"{printf "%.0f", $2/1e6}' $CG/cpu.stat) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; sleep 5; done ) > $RES/node_${E}_monitor.txt &
  MON=$!
  C0=$(awk '$1=="usage_usec"{print $2/1e6}' $CG/cpu.stat); T0=$(date +%s.%N)
  for s in "${SETS[@]}"; do
    FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || FQ=$W/data/atlas-rerun/perf/$s/sample.1fq
    SCREEN=(); [ -f $(dirname $FQ)/screen.json ] && SCREEN=(--run-screen $(dirname $FQ)/screen.json)
    mkdir -p $SCR/$E/$s/out
    ( cd $SCR/$E/$s && timeout -k 60 $LIMIT /usr/bin/time -v -o time.txt $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $SCR/$E/$s/out/ --threads $T --pipeline --output-format 1pz --engine $E "${SCREEN[@]}" > run.log 2>&1; echo "exit $?" >> run.log ) &
  done
  wait $(jobs -p | grep -v "^$MON$") 2>/dev/null
  # wait for the sample processes only
  while [ $(jobs -rp | grep -vc "^$MON$") -gt 0 ]; do sleep 5; done
  T1=$(date +%s.%N); C1=$(awk '$1=="usage_usec"{print $2/1e6}' $CG/cpu.stat)
  kill $MON 2>/dev/null
  echo "=== engine $E: makespan $(echo "$T1 - $T0" | bc) s, node core-s $(echo "$C1 - $C0" | bc)" | tee -a $RES/node_summary.txt
  for s in "${SETS[@]}"; do
    echo "  $s: $(grep -E 'Elapsed' $SCR/$E/$s/time.txt | awk '{print $NF}') wall, $(tail -1 $SCR/$E/$s/run.log), $(grep -h 'ENGINE_GPU: cache filled\|ENGINE_GPU: not used\|engine_gpu:' $SCR/$E/$s/out/*Log.out 2>/dev/null | tr '\n' ' ' | cut -c1-300)" | tee -a $RES/node_summary.txt
  done
  for s in "${SETS[@]}"; do ( cd $SCR/$E/$s/out && find . -type f -printf '%s\t%p\n' | sort -k2 ) > $RES/node_${E}_${s}_files.tsv; done
done
# identity of the gpu engine's outputs with the cpu engine's, per sample (byte for byte; order-independent for files that differ)
for s in "${SETS[@]}"; do
  [ -d $SCR/gpu/$s/out ] && [ -d $SCR/cpu/$s/out ] || continue
  n=0; d=0; od=0
  for f in $(cd $SCR/cpu/$s/out && find . -type f | sort); do
    n=$((n+1)); cmp -s $SCR/cpu/$s/out/$f $SCR/gpu/$s/out/$f && continue
    d=$((d+1))
    case $f in *.fq|*mate1|*mate2) a=$(paste - - - - < $SCR/cpu/$s/out/$f | LC_ALL=C sort | md5sum); b=$(paste - - - - < $SCR/gpu/$s/out/$f | LC_ALL=C sort | md5sum);;
               *) a=$(LC_ALL=C sort $SCR/cpu/$s/out/$f | md5sum); b=$(LC_ALL=C sort $SCR/gpu/$s/out/$f | md5sum);; esac
    [ "$a" = "$b" ] || { od=$((od+1)); echo "    $s differs order-independently: $f" | tee -a $RES/node_summary.txt; }
  done
  echo "  identity $s: $n files, $d differ byte-wise, $od differ order-independently" | tee -a $RES/node_summary.txt
done
