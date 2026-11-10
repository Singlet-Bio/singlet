#!/bin/bash
# Login node (sbatch only): BCR study development runs (PREREG.md Addendum 3) after the bcrdev download (DEP=afterok:<id>).
#   HEAD=<sha> DEP=afterok:<download job> bash submit_bcrdev.sh
# _frz  = frozen defaults (HEAD), _old = the defaults before the freeze (7e05e681: ambient_rescue none, v_open_frac 0,
# references without the AIRR-C genes), _js = frozen + j_slack 99 (the read vote alone names J), and TRUST4.
set -e
[ -n "$HEAD" ] || { echo "HEAD required"; exit 1; }
OLD=7e05e681070cee7daf12b0edaf6e200aa59558a3
cd /mnt/projects/debruinz_project/debruinz/singlet/jobs/vdj2
N="melv11_b:BCR:cr3+nsclc22_b:BCR:cr3+chx10k_b:BCR:cr5"
E="pbmc2_b:BCR:cr3+pbmc10k_b:BCR:cr5"
F=$(sbatch --parsable -A debruinz --job-name=vdj2-bcr-frz ${DEP:+--dependency=$DEP} --array=0-2%3 --export=ALL,SHA=$HEAD,LIBSPEC=$N,METHODS=singlet,TAG=_frz run.sbatch)
O=$(sbatch --parsable -A debruinz --job-name=vdj2-bcr-old --dependency=afterany:$F --array=0-2%3 --export=ALL,SHA=$OLD,LIBSPEC=$N,METHODS=singlet,TAG=_old run.sbatch)
J=$(sbatch --parsable -A debruinz --job-name=vdj2-bcr-js --dependency=afterany:$O --array=0-4%3 --export=ALL,SHA=$HEAD,LIBSPEC=$N+$E,METHODS=singlet,TAG=_js,PARAMS="--param j_slack=99" run.sbatch)
T=$(sbatch --parsable -A debruinz --job-name=vdj2-bcr-t4 --dependency=afterany:$J --partition=bigmem --time=12:00:00 --array=0-2%3 --export=ALL,SHA=$HEAD,LIBSPEC=$N,METHODS=trust4 run.sbatch)
echo "frz $F old $O js $J trust4 $T"
