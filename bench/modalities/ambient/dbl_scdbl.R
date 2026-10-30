suppressMessages({library(Matrix); library(scDblFinder); library(SingleCellExperiment)})
W <- "/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/ambient/work/dbl"
m <- as(readMM(file.path(W, "gene_synth.mtx")), "CsparseMatrix"); set.seed(1)
s <- scDblFinder(SingleCellExperiment(assays = list(counts = m)), verbose = FALSE)
write.table(data.frame(col = seq_len(ncol(m)) - 1, score = s$scDblFinder.score, is_doublet = as.integer(s$scDblFinder.class == "doublet")),
            file.path(W, "scdblfinder.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
