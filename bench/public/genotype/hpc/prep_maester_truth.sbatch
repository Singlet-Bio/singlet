#!/bin/bash
#SBATCH --job-name=bpg-mtruth
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Truth tables of the MAESTER paper (Miller et al. 2022, GSE182685), as already staged under data/maester/truth by the
# mtDNA-lineage work: cell-line mix cell labels (expression and mt) and the BPDCN712 clone-defining variants / positive cells.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/mtl/bin:$PATH
O=$W/bench-public/data/maester_truth; mkdir -p $O
Rscript -e 'x <- readRDS("'$W'/data/maester/truth/TenX_CellLineMix_Cells.rds");
 d <- data.frame(barcode=sub("-1$","",x$cell), line_rna=as.character(x$CellType_RNA), line_mt=as.character(x$CellType_MT), mean_cov=x$Mean_Cov);
 write.table(d, "'$O'/celllinemix_cells.tsv", sep="\t", quote=FALSE, row.names=FALSE)'
cp $W/data/maester/truth/4.4_positive_cells.txt $O/bpdcn712_positive_cells.tsv
cp $W/data/maester/truth/TenX_CellLineMix_Variants1.txt $O/celllinemix_variants.tsv
head -3 $O/celllinemix_cells.tsv; wc -l $O/*
