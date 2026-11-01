#!/bin/bash
# Login node: re-assembly variants of the development libraries (sbatch only).
#   _asm: AIRR-C-extended references + v_open_frac=0.5 (all 9 libraries); _air: AIRR-C references only (BCR libraries)
W=/mnt/projects/debruinz_project/debruinz/singlet
cd $W/jobs/vdj2
R=$W/expression-sota/vdj2/ref
ALL="pbmc2_b:BCR:cr3+pbmc10k_b:BCR:cr5+pbmc2_t:TCR:cr3+pbmc10k_t:TCR:cr5+gse145370_S133A:TCR:cr3+gse145370_S158A:TCR:cr3+gse145370_S149B:TCR:cr3+gse145370_S150B:TCR:cr3+gse145370_S135B:TCR:cr3"
sbatch -A debruinz --array=0-8%3 ${DEP:+--dependency=$DEP} --export=ALL,LIBSPEC=$ALL,METHODS=singlet,TAG=_asm,REFDIR=$R,PARAMS="--param v_open_frac=0.5" run.sbatch
sbatch -A debruinz --array=0-1%2 ${DEP:+--dependency=$DEP} --export=ALL,LIBSPEC=$ALL,METHODS=singlet,TAG=_air,REFDIR=$R run.sbatch
