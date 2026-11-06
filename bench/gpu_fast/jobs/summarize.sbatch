#!/bin/bash
#SBATCH --job-name=gf-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Final job of a fast lane campaign. Runs bench/gpu_fast/summarize.py inside the Singlet runtime (in_runtime). CPU only. Nothing from the laptop.
# Parameters (sbatch --export=ALL,...): CAMPAIGN_MANIFEST (a campaign.json on Clipper), OUTDIR (under $W/jobs/gpu-fast/results), PIN_SCRIPTS (a full commit of bench/gpu-fast).
# It writes OUTDIR/summary.md, OUTDIR/load_fit.json and OUTDIR/steps/<id>/. Run it with afterany, so it summarizes whatever the steps left.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
: "${CAMPAIGN_MANIFEST:?}" "${OUTDIR:?}" "${PIN_SCRIPTS:?}"
case $OUTDIR in $W/jobs/gpu-fast/results/*) ;; *) echo "OUTDIR must lie under $W/jobs/gpu-fast/results"; exit 2 ;; esac
[[ $PIN_SCRIPTS =~ ^[0-9a-f]{40}$ ]] || { echo "PIN_SCRIPTS must be a full commit"; exit 2; }
[ -f "$CAMPAIGN_MANIFEST" ] || { echo "no manifest $CAMPAIGN_MANIFEST"; exit 2; }
mkdir -p $OUTDIR
BARE=$W/repos/singlet-py.git; SBR=bench/gpu-fast; SREF=refs/gf/summary-$SLURM_JOB_ID
trap 'git -C $BARE update-ref -d $SREF 2>/dev/null' EXIT
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.gn-fetch.lock
git -C $BARE cat-file -e "$PIN_SCRIPTS^{commit}" 2>/dev/null || { echo "commit $PIN_SCRIPTS is not on $SBR"; exit 3; }
rm -rf $OUTDIR/code; mkdir -p $OUTDIR/code
git -C $BARE archive "$PIN_SCRIPTS" bench/gpu_fast | tar -x -C $OUTDIR/code
echo "SUMMARY job=$SLURM_JOB_ID scripts=${PIN_SCRIPTS:0:8} manifest=$CAMPAIGN_MANIFEST $(date -Is)"
IR=$(ls $W/src/singlet-agents/main/hpc/clipper/in_runtime.sh $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh 2>/dev/null | head -1)
[ -n "$IR" ] || { echo "no in_runtime.sh found"; exit 4; }
# shellcheck disable=SC1090
source "$IR"
in_runtime python3 -I $OUTDIR/code/bench/gpu_fast/summarize.py "$CAMPAIGN_MANIFEST" "$OUTDIR" || { echo "summarize.py failed"; exit 5; }
echo done $(date -Is)
