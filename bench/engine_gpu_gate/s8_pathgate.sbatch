#!/bin/bash
#SBATCH --job-name=s8-pathgate
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8: stream vs sorted pileup on the same binary (BINDIR), and what changed against the previous stream output (BINDIR_OLD).
# Per sample: new stream, new sorted (--sorted-bam), old stream; 8 threads, CPU engine, .1fg2 input.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BINDIR_OLD=${BINDIR_OLD:?}; SAMPLES=(${SAMPLES:-SRR19421372 SRR19573735 SRR30477995}); s=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
for b in $BINDIR $BINDIR_OLD; do [ -e $b/../whitelists ] || ln -s $PB/whitelists $b/../whitelists; done
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
G=$REF/GRCh38-2024-A; IN=$W/jobs/s9/fixtures/$s.1fg2
RES=$W/jobs/s8/results/pathgate-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $RES; S=$RES/summary.txt
SCR=/tmp/$USER-s8pg-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf $SCR' EXIT
git -C $W/repos/singlet-py.git archive refs/s8/campaign bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
echo "sample=$s new=$(cat $BINDIR/commit | cut -c1-8) old=$(cat $BINDIR_OLD/commit | cut -c1-8)" | tee $S
run() {  # tag bindir extra
  mkdir -p $SCR/$1/out
  ( cd $SCR/$1 && $2/singlet-gpu $IN --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/$1/out/ --threads 8 --pipeline --output-format 1pz $3 > $RES/$1.log 2>&1 )
  python3 - <<PY | tee -a $S
import json
d=json.load(open("$SCR/$1/out/pileup_stats.json")); a=json.load(open("$SCR/$1/out/ancestry_call.json")) if __import__("os").path.exists("$SCR/$1/out/ancestry_call.json") else {}
print("$1", {k:d.get(k) for k in ("pileup_path","n_barcodes","exon_hits","mt_numt_rescued","n_called_cells")}, "ancestry", a.get("ancestry") or a.get("call"), a.get("confidence"))
PY
  grep -a -E "barcode universe|Discovered|CB-tag barcode refresh|Done:" $RES/$1.log | cut -c1-260 | tee -a $S
}
run new_stream $BINDIR ""
run new_sorted $BINDIR "--sorted-bam"
run old_stream $BINDIR_OLD ""
for pair in "new_stream new_sorted" "old_stream new_stream"; do set -- $pair
  echo "=== $1 vs $2" | tee -a $S
  python3 $SCR/repo/bench/star_accel/out_equiv.py $SCR/$1/out $SCR/$2/out > $RES/equiv-$1-$2.txt 2>&1
  grep -a -E '^  DIFF|^files:|^RESULT|SAME RECORDS' $RES/equiv-$1-$2.txt | cut -c1-230 | head -40 | tee -a $S
done
echo done | tee -a $S
