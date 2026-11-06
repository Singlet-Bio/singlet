#!/bin/bash
# Self-running chain of the shared resident genome index (gpu/bold-shidx). Runs on the Clipper login node: only sbatch and squeue.
#   TAIL=<job id> submit_chain.sh COMMIT      COMMIT: the branch tip that the build will record (8 or more hex digits)
# Jobs (all with --dependency; GPU jobs one at a time):
#   build (short-bigmem) -> preflight (debug-gpu: holder resident, one run must import) -> identity (g051/g052) -> corpus (refs on short-cpu, then the gpu-paths job, g051/g052) -> timing (g050, after the
#   last job queued there now, and after the corpus) -> summarizer (short-cpu, afterany on all).
# The jobs of bench/gpu_shidx/jobs are copied to $W/jobs/gpu-shidx/jobs first (clipper.py put). Corpus scripts are the ones of bench/gpu-paths.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/gpu-shidx/jobs; PJ=$W/jobs/gpu-paths/jobs
COMMIT=${1:?COMMIT}; BIND=$W/jobs/gpu-shidx/bin/shidx; cd $W/jobs/gpu-shidx
BUILD=$(sbatch --parsable --export=ALL,BR=gpu/bold-shidx,NAME=shidx $J/build.sbatch)
PRE=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$BUILD --exclude=g050 $J/preflight.sbatch)
IDENT=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$PRE $J/identity.sbatch)
REFS=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$BUILD --array=0-15%6 -p short-cpu --gres=none --mem=64G --time=03:00:00 --job-name=gshidx-refs \
       --export=ALL,BIN=$BIND,COMMIT=$COMMIT,MODES=cpu,ALLOW_NO_EMBED=1 $PJ/paths.sbatch)
PATHS=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$IDENT,afterany:$REFS --exclude=g050 --job-name=gshidx-paths \
        --export=ALL,BIN=$BIND,COMMIT=$COMMIT,MODES=gpu,ALLOW_NO_EMBED=1,SINGLET_GPU_SHARED_INDEX=1,SINGLET_GPU_SHARED_INDEX_IDLE_S=60 $PJ/paths.sbatch)
# TAIL: the last job queued on g050 at submission time (check with scontrol show job: ReqNodeList=g050), given by the caller
TAIL=${TAIL:-}
echo "tail of g050: ${TAIL:-none}"
TIMING=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$IDENT,afterany:$PATHS $J/timing.sbatch)   # exclusive, any H100 node
SUMM=$(sbatch --parsable --dependency=afterany:$BUILD:$PRE:$IDENT:$REFS:$PATHS:$TIMING \
       --export=ALL,BUILD_ID=$BUILD,IDENT_ID=$IDENT,PATHS_ID=$PATHS,TIMING_ID=$TIMING $J/summarize.sbatch)
echo "build=$BUILD preflight=$PRE identity=$IDENT refs=$REFS paths=$PATHS timing=$TIMING(after $PATHS,${TAIL:-none}) summary=$SUMM"
