#!/bin/bash
# Login node: full re-assembly of the development libraries with candidate settings (sbatch only).
#   TAG=_cand PARAMS="--param ambient_rescue=clone --param v_open_frac=0.5" DEP=afterany:794913 bash submit_cand.sh
W=/mnt/projects/debruinz_project/debruinz/singlet
cd $W/jobs/vdj2
ALL="pbmc2_t:TCR:cr3+pbmc2_b:BCR:cr3+pbmc10k_t:TCR:cr5+pbmc10k_b:BCR:cr5+gse145370_S133A:TCR:cr3+gse145370_S158A:TCR:cr3+gse145370_S149B:TCR:cr3+gse145370_S150B:TCR:cr3+gse145370_S135B:TCR:cr3"
sbatch -A debruinz --array=0-8%3 ${DEP:+--dependency=$DEP} --export=ALL,LIBSPEC=$ALL,METHODS=singlet,TAG=$TAG,PARAMS="$PARAMS" run.sbatch
