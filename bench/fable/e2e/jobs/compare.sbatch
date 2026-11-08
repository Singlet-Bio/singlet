#!/bin/bash
#SBATCH --job-name=e8-cmp
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E8: out_equiv.py on kept outputs of gate runs in different jobs (gate.sbatch keeps <mode>/out without sequence-bearing files).
# Usage: PAIRS="nameA=dirA,nameB=dirB ..." sbatch compare.sbatch   (one pair per word; the report goes to $RES/equiv-nameA-nameB.txt)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BARE=$W/repos/singlet-py.git
RES=$W/jobs/fable/e2e/results/cmp-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e8c-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf $SCR' EXIT INT TERM
git -C $BARE archive refs/fable/e2e bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
export PATH=$W/envs/pbench/bin:$PATH
for p in ${PAIRS:?set PAIRS}; do
  A=${p%%,*}; B=${p#*,}; na=${A%%=*}; da=${A#*=}; nb=${B%%=*}; db=${B#*=}
  echo "=== $na vs $nb ($da vs $db)" | tee -a $RES/summary.txt
  python3 $SCR/repo/bench/star_accel/out_equiv.py $da $db > $RES/equiv-$na-$nb.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $RES/equiv-$na-$nb.txt | cut -c1-200 | head -30 | tee -a $RES/summary.txt
done
echo done $(date -Is) | tee -a $RES/summary.txt
