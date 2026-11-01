#!/bin/bash
# Login node: THE single run of the new BCR held-out set (PREREG.md Addendum 3/4), frozen code pinned by SHA (sbatch only).
#   SHA=<frozen commit> bash submit_bcrho.sh
set -e
[ -n "$SHA" ] || { echo "SHA required"; exit 1; }
W=/mnt/projects/debruinz_project/debruinz/singlet
cd $W/jobs/vdj2
H="pbmc22_b:BCR:cr3+cd19_b:BCR:cr3+b1k_b:BCR:cr5"
S=$(sbatch --parsable -A debruinz --job-name=vdj2-bho-singlet --array=0-2%3 --export=ALL,SHA=$SHA,LIBSPEC=$H,METHODS=singlet,TAG= run.sbatch)
T=$(sbatch --parsable -A debruinz --job-name=vdj2-bho-trust4 --dependency=afterany:$S --partition=bigmem --time=12:00:00 --array=0-2%3 --export=ALL,SHA=$SHA,LIBSPEC=$H,METHODS=trust4 run.sbatch)
C=$(sbatch --parsable -A debruinz --dependency=afterany:$T --export=ALL,SHA=$SHA,SET=bcrho,TAGS=_ score.sbatch)
echo "singlet $S trust4 $T score $C"
