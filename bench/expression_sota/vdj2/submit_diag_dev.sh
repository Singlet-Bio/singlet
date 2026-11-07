#!/bin/bash
# Login node: submit diag2.py over the existing development runs (sbatch only).
W=/mnt/projects/debruinz_project/debruinz/singlet
R=$W/state/vdjc
C="--runs $R/runs --data $W/data/vdjc"
cd $W/jobs/vdj2
sbatch -A debruinz --export=ALL,SCRIPT=bench/expression_sota/vdj2/diag2.py,PYARGS="\
$C --lib gse145370_S133A:TCR --out diag_S133A.json;\
$C --lib gse145370_S158A:TCR --out diag_S158A.json;\
$C --lib pbmc2_t:TCR --out diag_pbmc2_t.json;\
$C --lib pbmc10k_t:TCR --out diag_pbmc10k_t.json;\
$C --lib pbmc10k_b:BCR --sdir singlet_fix --vdj-ref $W/state/vdjh/vdj_human_fix.tsv.gz --out diag_pbmc10k_b_fix.json;\
$C --lib pbmc2_b:BCR --vdj-ref $R/ref/vdj_human.tsv.gz --out diag_pbmc2_b.json" py.sbatch
