#!/bin/bash
#SBATCH --job-name=s8-thrgate
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8: is the output independent of the thread count? Per sample the CPU engine, same binary, same .1fg2 input: stream path (default) and sorted-BAM parallel pileup
# (--sorted-bam), each at 8 and 16 threads. Gate: the files of the two thread counts are identical (out_equiv.py; accepted: read order of Unmapped/nonhost,
# read_stats.tsv timing rows, star_Log.std.out, provenance paths/threads/timings). Env: BINDIR (needs singlet-gpu with a commit file).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; SAMPLES=(${SAMPLES:-SRR19421372 SRR19573735 SRR30477995}); s=${SAMPLES[$SLURM_ARRAY_TASK_ID]}; THR=(${THR:-8 16}); PATHS=(${PATHS:-stream sorted})
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
G=$REF/GRCh38-2024-A; IN=$W/jobs/s9/fixtures/$s.1fg2
RES=$W/jobs/s8/results/thrgate-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $RES; S=$RES/summary.txt
SCR=/tmp/$USER-s8tg-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf $SCR' EXIT
git -C $W/repos/singlet-py.git archive refs/s8/campaign bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
echo "sample=$s commit=$(cat $BINDIR/commit) $(date -Is)" | tee $S
for p in "${PATHS[@]}"; do for t in "${THR[@]}"; do
  m=${p}_$t; mkdir -p $SCR/$m/out; X=""; [ $p = sorted ] && X="--sorted-bam"
  ( cd $SCR/$m && $BINDIR/singlet-gpu $IN --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/$m/out/ --threads $t --pipeline --output-format 1pz $X > $RES/$m.log 2>&1 )
  python3 - <<PY | tee -a $S
import json
d=json.load(open("$SCR/$m/out/pileup_stats.json"))
print("$s $m", {k:d.get(k) for k in ("exon_hits","mt_numt_rescued","molecules_ambiguous","n_called_cells","pileup_threads")})
PY
  grep -a -E "parallel-cross-worker|Pileup: streaming|\[parallel\] Worker" $RES/$m.log | head -3 | cut -c1-160 | tee -a $S
done; done
for p in "${PATHS[@]}"; do
  a=$SCR/${p}_${THR[0]}/out; b=$SCR/${p}_${THR[1]}/out
  echo "=== $s $p ${THR[0]} vs ${THR[1]}" | tee -a $S
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv-$p.txt 2>&1
  grep -a '^  DIFF\|^RESULT\|SAME RECORDS' $RES/equiv-$p.txt | cut -c1-220 | head -30 | tee -a $S
done
echo done | tee -a $S
