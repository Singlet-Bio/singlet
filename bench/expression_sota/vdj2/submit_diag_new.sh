#!/bin/bash
# Login node: diagnostics on the development runs of this study (sbatch only).  TAG=_base bash submit_diag_new.sh
W=/mnt/projects/debruinz_project/debruinz/singlet
V=$W/expression-sota/vdj2/runs
T=${TAG:-_base}
C="--runs $V --data $W/data/vdjc --sdir singlet$T"
cd $W/jobs/vdj2
A=""
for l in gse145370_S149B gse145370_S150B gse145370_S135B gse145370_S133A gse145370_S158A pbmc2_t pbmc10k_t; do
  A="$A$C --lib $l:TCR --out diag2_$l$T.json;"
done
sbatch -A debruinz --job-name=vdj2-diag --export=ALL,SCRIPT=bench/expression_sota/vdj2/diag2.py,PYARGS="$A" py.sbatch
B=""
for l in gse145370_S149B gse145370_S150B gse145370_S135B; do
  B="$B--fastq-dir $W/data/vdjc/$l/fastqs --ref $W/data/vdjc/$l/filtered_contig_annotations.csv --contigs $V/$l/singlet$T/contigs.csv --run $V/$l/singlet$T/run.json --chains TCR --n 400 --out reads_$l$T.json;"
done
sbatch -A debruinz --job-name=vdj2-reads --time=03:00:00 --export=ALL,SCRIPT=bench/expression_sota/vdj2/diag_reads.py,PYARGS="$B" py.sbatch
