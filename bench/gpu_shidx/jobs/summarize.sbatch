#!/bin/bash
#SBATCH --job-name=gshidx-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Writes results/overnight/summary.md of the shared-index chain from the logs and result directories of its jobs.
# Parameters: BUILD_ID IDENT_ID PATHS_ID TIMING_ID (job ids). Reads files only.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; O=$W/jobs/gpu-shidx/results/overnight; M=$O/summary.md
L=$W/logs
{
echo "# Shared resident genome index: overnight summary"
echo
echo "Written $(date -Is) by job $SLURM_JOB_ID. Branch gpu/bold-shidx."
echo
echo "## Build (job ${BUILD_ID:-?})"
echo '```'
grep -a "^COMMIT=\|^[0-9a-f]\{64\}  \|\[build\]\|error" $L/gshidx-build-${BUILD_ID:-0}.out 2>/dev/null | head -10
echo "sacct: $(sacct -j ${BUILD_ID:-0} -n -X -o State,ExitCode | head -1)"
echo '```'
echo
echo "## Identity (job ${IDENT_ID:-?}): pbmc1k_v3, two processes sharing one holder, with and without MPS, against the CPU engine and b0"
echo '```'
cat $O/identity-${IDENT_ID:-0}/verdict.txt 2>/dev/null | cut -c1-400 || echo "no verdict file"
echo '```'
echo
echo "## Corpus (job ${PATHS_ID:-?}): bench/gpu-paths, 16 inputs, feature on"
echo '```'
P=$W/jobs/gpu-paths/results/paths-${PATHS_ID:-0}
[ -f $P/verdicts.tsv ] && { column -t -s $'\t' $P/verdicts.tsv | cut -c1-250; echo; grep -ac "shared index: imported" $P/*/star_Log.out 2>/dev/null | awk -F: '{n+=$2} END{print "runs with an import line: " n}'; tail -5 $P/summary.txt | cut -c1-250; } || { echo "no verdicts.tsv"; tail -5 $P/summary.txt 2>/dev/null | cut -c1-250; }
echo '```'
echo
echo "## Timing (job ${TIMING_ID:-?}) on g050: k=8, T=7, r3. A b0 2 slots, B shared 2 slots, C shared 3 slots, Bm and Cm the same under MPS"
echo
for a in A B C Bm Cm; do
  F=$O/timing-${TIMING_ID:-0}/$a/arm.txt
  echo "### Arm $a"; echo '```'; [ -f $F ] && cut -c1-300 $F || echo "no arm record"; echo '```'
done
echo "## MPS and CUDA IPC"
echo '```'
grep -ah "shared index\|cudaIpc" $O/identity-${IDENT_ID:-0}/mps*/star_Log.out 2>/dev/null | cut -c1-300 | sort | uniq -c | head -8
grep -a "holder:" $O/timing-${TIMING_ID:-0}/summary.txt 2>/dev/null | cut -c1-250
echo '```'
} > $M
echo wrote $M; wc -c $M
