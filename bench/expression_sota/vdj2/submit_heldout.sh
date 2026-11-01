#!/bin/bash
# Login node: THE single held-out run (PREREG.md), frozen code pinned by SHA (sbatch only).
#   SHA=<frozen commit> bash submit_heldout.sh
set -e
[ -n "$SHA" ] || { echo "SHA required"; exit 1; }
W=/mnt/projects/debruinz_project/debruinz/singlet
cd $W/jobs/vdj2
H="gse145370_S134A:TCR:cr3+gse145370_S134B:TCR:cr3+gse145370_S159A:TCR:cr3+gse145370_S159B:TCR:cr3+pbmc3_t:TCR:cr3+ngpbmc3_t:TCR:cr3+pbmc1k_t:TCR:cr5+pbmc3_b:BCR:cr3+ngpbmc3_b:BCR:cr3+pbmc1k_b:BCR:cr5+melanoma_b:BCR:cr3"
S=$(sbatch --parsable -A debruinz --job-name=vdj2-ho-singlet --array=0-10%3 --export=ALL,SHA=$SHA,LIBSPEC=$H,METHODS=singlet,TAG= run.sbatch)
T=$(sbatch --parsable -A debruinz --job-name=vdj2-ho-trust4 --dependency=afterany:$S --partition=bigmem --time=12:00:00 --array=0-10%3 --export=ALL,SHA=$SHA,LIBSPEC=$H,METHODS=trust4 run.sbatch)
sbatch -A debruinz --dependency=afterany:$T --export=ALL,SHA=$SHA,SET=heldout,TAGS=_ score.sbatch
echo "singlet $S trust4 $T"
