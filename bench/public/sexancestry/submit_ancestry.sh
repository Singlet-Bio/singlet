#!/bin/bash
# Genetic-ancestry benchmark on Clipper: leave-out frequencies -> 150 MAGE individuals (align, thin, per-tool inputs) ->
# scoring (Singlet, somalier, peddy) and the contract document. Login-node safe (sbatch only).
#   [PART=short-cpu] bash bench/public/sexancestry/submit_ancestry.sh      (default partition: scavenger)
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
H=$SRC/bench/public/sexancestry; PART=${PART:-scavenger}
mkdir -p $BP/work/ancestry
F=$(sbatch --parsable --partition=$PART --requeue $H/ancestry_af.sbatch)
M=$(sbatch --parsable --partition=$PART --dependency=afterok:$F --array=1-150%12 $H/mage.sbatch)
S=$(sbatch --parsable --partition=$PART --dependency=afterany:$M $H/ancestry_score.sbatch)
echo "af $F  mage $M  score $S" | tee -a $BP/work/submissions.log
