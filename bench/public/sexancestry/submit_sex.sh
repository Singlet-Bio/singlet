#!/bin/bash
# Chromosomal-sex benchmark on Clipper: extract (one task per study) -> cellXY -> collect. Login-node safe (sbatch only).
#   bash bench/public/sexancestry/submit_sex.sh
# Studies: every GSE of the annotation validation run (out/annotate_validation/run1/), whose samples have GEO sex labels
# in data/annotate/sex_labels.json (docs/cell_annotation.md section 4).
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
H=$SRC/bench/public/sexancestry; W=$BP/work/sex; mkdir -p $W
PART=${PART:-scavenger}   # short-* partitions often hit the per-user node limit; scavenger jobs requeue on preemption
ls -d $WORK/out/annotate_validation/run1/GSE* | xargs -n1 basename > $W/studies.txt
N=$(wc -l < $W/studies.txt); echo "$N studies"
E=$(sbatch --parsable --partition=$PART --requeue --array=1-$N%8 $H/sex.sbatch extract)
X=$(sbatch --parsable --partition=$PART --requeue --dependency=afterany:$E --cpus-per-task=4 --mem=96G $H/sex.sbatch cellxy)
K=$(sbatch --parsable --partition=$PART --requeue --dependency=afterany:$X --cpus-per-task=4 --mem=32G $H/sex.sbatch collect)
echo "extract $E  cellxy $X  collect $K" | tee -a $BP/work/submissions.log
