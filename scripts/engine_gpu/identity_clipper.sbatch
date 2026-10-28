#!/bin/bash
#SBATCH --job-name=eng-gpu-id
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# identity_test.sh on Clipper for one sample (default pbmc1k_v3), outputs in job /tmp; summary to $W/jobs/w11/engine/results/.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BARE=$W/repos/singlet-py.git; BIN=${BIN:-$W/jobs/w11/engine/bin/singlet-gpu}; SET=${SET:-pbmc1k_v3}
FQ=${FQ:-$PB/data/$SET/$SET.1fq}; [ -e "$FQ" ] || FQ=$W/data/atlas-rerun/perf/$SET/sample.1fq
RES=$W/jobs/w11/engine/results/identity-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-engid-$SLURM_JOB_ID; mkdir -p $SCR/repo
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 5; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
git -C $BARE archive refs/w11/engine scripts/engine_gpu bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
echo "set=$SET node=$(hostname) job=$SLURM_JOB_ID bin=$(sha256sum $BIN | cut -c1-16) commit=$(cat $(dirname $BIN)/commit) $(date -Is)"
nvidia-smi --query-gpu=name,pci.bus_id --format=csv,noheader
G=${GENOME:-GRCh38-2024-A}
RUN_CPU2=${RUN_CPU2:-1} THREADS=${THREADS:-24} REF_BASE=$REF GENOME_DIR=$REF/$G/star_2.7.11b GTF=$REF/$G/genes/genes.gtf \
  timeout -k 60 ${LIMIT:-10000} bash $SCR/repo/scripts/engine_gpu/identity_test.sh $BIN $FQ $SCR/t 2>&1 | tee $RES/identity_$SET.txt
for r in cpu cpu2 verify gpu; do [ -d $SCR/t/$r ] && grep -h "Elapsed\|ENGINE_GPU\|engine_gpu" $SCR/t/$r/run.log $SCR/t/$r/out/*Log.out 2>/dev/null | sed "s/^/$r: /"; done | tee -a $RES/identity_$SET.txt
cp $SCR/t/compare_*.txt $RES/ 2>/dev/null; for r in cpu cpu2 verify gpu; do [ -f $SCR/t/$r/run.log ] && cp $SCR/t/$r/run.log $RES/identity_${SET}_$r.run.log; done
