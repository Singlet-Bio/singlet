# Strand-detection validation (fix/strand-detection) on top of the pipeline benchmark.
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__strand-detection/bench/pipeline/common.sh
SRC_STRAND=$WORK/src/singlet-py/fix__strand-detection
BIN_STRAND=$PB/bin/singlet-strandfix
RES_STRAND=$PB/results/strand
CAMPAIGN=/mnt/projects/debruinz_project/cellarium/full_campaign/results
mkdir -p $RES_STRAND
