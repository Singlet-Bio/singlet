#!/bin/bash
# Login node: filter variants of the development baseline runs without re-assembly (sbatch only).
#   VARIANTS="_rf0: _cell:--param~ambient_rescue=cell _pair:--param~ambient_rescue=pair" bash submit_refilter.sh
W=/mnt/projects/debruinz_project/debruinz/singlet
V=$W/expression-sota/vdj2/runs
VARIANTS=${VARIANTS:-"_rf0: _cell:--param~ambient_rescue=cell _pair:--param~ambient_rescue=pair"}
A=""
for l in pbmc2_t pbmc2_b pbmc10k_t pbmc10k_b gse145370_S133A gse145370_S158A gse145370_S149B gse145370_S150B gse145370_S135B; do
  for v in $VARIANTS; do
    tag=${v%%:*}; args=${v#*:}; args=${args//\~/ }
    [ -n "$NOCHECK" ] || [ -d $V/$l/singlet${IN:-_base} ] || continue
    A="$A$V/$l/singlet${IN:-_base} $V/$l/singlet$tag $args;"
  done
done
cd $W/jobs/vdj2
sbatch -A debruinz --job-name=vdj2-refilter ${DEP:+--dependency=$DEP} --export=ALL,SCRIPT=bench/expression_sota/vdj2/refilter.py,PYARGS="$A" py.sbatch
