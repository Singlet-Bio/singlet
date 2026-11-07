#!/bin/bash
# Login node: link the earlier development TRUST4 runs and GEX files, then submit the baseline Singlet runs (sbatch only).
W=/mnt/projects/debruinz_project/debruinz/singlet
V=$W/expression-sota/vdj2/runs
for l in pbmc2_t pbmc2_b pbmc10k_t pbmc10k_b gse145370_S133A gse145370_S158A; do
  mkdir -p $V/$l
  for f in trust4 gex_cells.txt gex_lineage.csv; do [ -e $W/state/vdjc/runs/$l/$f ] && ln -sfn $W/state/vdjc/runs/$l/$f $V/$l/$f; done
done
cd $W/jobs/vdj2
OLD="pbmc2_t:TCR:cr3+pbmc2_b:BCR:cr3+pbmc10k_t:TCR:cr5+pbmc10k_b:BCR:cr5+gse145370_S133A:TCR:cr3+gse145370_S158A:TCR:cr3"
NEW="gse145370_S149B:TCR:cr3+gse145370_S150B:TCR:cr3+gse145370_S135B:TCR:cr3"
sbatch -A debruinz --array=0-5%3 --export=ALL,LIBSPEC=$OLD,METHODS=singlet,TAG=_base run.sbatch
sbatch -A debruinz --array=0-2%3 --dependency=afterok:${DEP:-794910} --export=ALL,LIBSPEC=$NEW,METHODS="singlet trust4",TAG=_base run.sbatch
