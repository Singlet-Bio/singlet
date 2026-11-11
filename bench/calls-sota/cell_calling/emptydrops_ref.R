# EmptyDrops comparator: DropletUtils::emptyDrops (GPL-3, called as an external program) at the standard settings.
args <- commandArgs(TRUE); out <- args[1]
.libPaths(c("/mnt/projects/debruinz_project/debruinz/singlet/rlib/bioc-3.20", .libPaths()))
suppressMessages({library(DropletUtils); library(Matrix)})
d <- scan(file.path(out, "dims.txt"), quiet = TRUE)
rd <- function(f) readBin(file.path(out, f), "integer", n = d[3], size = 4)
m <- sparseMatrix(i = rd("i.bin") + 1L, j = rd("j.bin") + 1L, x = as.numeric(rd("x.bin")), dims = d[1:2])
set.seed(100)
t0 <- proc.time()[["elapsed"]]
e <- emptyDrops(m, lower = 100, niters = 10000)
cat("emptyDrops seconds", proc.time()[["elapsed"]] - t0, " DropletUtils", as.character(packageVersion("DropletUtils")), "\n")
call <- as.integer(!is.na(e$FDR) & e$FDR <= 0.001)
cat("called", sum(call), " tested", sum(!is.na(e$FDR)), "\n")
write.table(data.frame(idx = seq_len(ncol(m)), total = e$Total, FDR = e$FDR, edcr = call), file.path(out, "emptydrops.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
