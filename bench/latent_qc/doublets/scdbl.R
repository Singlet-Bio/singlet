# scDblFinder (default settings, random doublets, cluster-free) on $SCORES/<name>/counts.mtx -> scdblfinder.npy-compatible TSV
# Usage: Rscript scdbl.R SCORES_DIR NAME
suppressPackageStartupMessages({library(Matrix); library(SingleCellExperiment); library(scDblFinder); library(BiocParallel)})
a <- commandArgs(TRUE); d <- file.path(a[1], a[2])
m <- as(readMM(file.path(d, "counts.mtx")), "CsparseMatrix")
rownames(m) <- paste0("g", seq_len(nrow(m))); colnames(m) <- paste0("c", seq_len(ncol(m)))
set.seed(1)
t0 <- Sys.time()
sce <- scDblFinder(SingleCellExperiment(list(counts = m)), BPPARAM = SerialParam(RNGseed = 1))
s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(data.frame(score = sce$scDblFinder.score, call = as.integer(sce$scDblFinder.class == "doublet")),
            file.path(d, "scdblfinder.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
writeLines(as.character(s), file.path(d, "scdblfinder.seconds"))
cat(a[2], "scDblFinder", s, "s\n")
