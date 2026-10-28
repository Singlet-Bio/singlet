#!/bin/bash
#SBATCH --job-name=eng-gpu-perf
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Where STAR spends its CPU on --engine gpu once the cache is filled: perf samples STAR's threads (call graphs) for PERF_S seconds
# from the moment the GPU prefill logs "cache filled". The run stops after the sample (outputs are not needed). Data in job /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:-$W/jobs/w11/engine/bin/singlet-gpu}; RES=$W/jobs/w11/engine/results/perf-$SLURM_JOB_ID; mkdir -p $RES
S=${SET:-pbmc1k_v3}; T=${T:-9}; PERF_S=${PERF_S:-40}
SCR=/tmp/$USER-engperf-$SLURM_JOB_ID; mkdir -p $SCR/out
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 2; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF
G=$REF/GRCh38-2024-A
FQ=$PB/data/$S/$S.1fq; [ -e $FQ ] || FQ=$W/data/atlas-rerun/perf/$S/sample.1fq
echo "node=$(hostname) job=$SLURM_JOB_ID set=$S bin=$(sha256sum $BIN | cut -c1-16) paranoid=$(cat /proc/sys/kernel/perf_event_paranoid) $(date -Is)"
( cd $SCR && timeout -k 30 1500 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ \
    --threads $T --pipeline --output-format 1pz --engine ${ENGINE:-gpu} > $SCR/run.log 2>&1 ) &
RUN=$!
MARK=${MARK:-"ENGINE_GPU: cache filled"}
for i in $(seq 1 1200); do grep -q "$MARK" $SCR/out/star_Log.out 2>/dev/null && break; kill -0 $RUN 2>/dev/null || break; sleep 1; done
[ "${ENGINE:-gpu}" = cpu ] && sleep ${CPU_DELAY:-60}
SP=$(grep -o "STAR pid=[0-9]*" $SCR/run.log | head -1 | cut -d= -f2)
echo "STAR pid=$SP, marker seen at $(date -Is)"; grep -h "engine_gpu:\|ENGINE_GPU" $SCR/out/star_Log.out
timeout -k 10 $((PERF_S + 60)) perf record -F 499 -g -p $SP -o $SCR/perf.data -- sleep $PERF_S 2> $RES/perf_record.err
kill -TERM $RUN 2>/dev/null; pkill -TERM -P $RUN 2>/dev/null
perf report -i $SCR/perf.data --stdio --no-children --sort dso,symbol --percent-limit 0.3 2>/dev/null | head -150 > $RES/self.txt
perf report -i $SCR/perf.data --stdio --children --sort symbol --percent-limit 2 -g none 2>/dev/null | head -120 > $RES/children.txt
perf report -i $SCR/perf.data --stdio --no-children --sort comm 2>/dev/null | head -30 > $RES/comm.txt
head -60 $RES/self.txt
echo done $(date -Is)
