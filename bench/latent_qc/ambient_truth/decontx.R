# DecontX (celda) on dumped inputs. Usage: Rscript decontx.R DIR   (DIR from dump_inputs.py)
# Writes DIR/decontx_{default,background}.csv: rho, removed UMIs from mouse / human genes, wall seconds.
suppressMessages({library(Matrix); library(celda)})
args <- commandArgs(TRUE); d <- args[1]
rd <- function(name) {
  dim <- scan(file.path(d, paste0(name, ".dim")), quiet = TRUE)
  rb <- function(s, n, what = integer()) readBin(file.path(d, paste0(name, ".", s)), "integer", n = n)
  p <- rb("p", dim[2] + 1)
  i <- rb("i", p[length(p)]); x <- rb("x", p[length(p)])
  sparseMatrix(i = i + 1L, p = p, x = as.numeric(x), dims = dim)
}
toc <- rd("cells"); bg <- rd("bg")
mouse <- as.logical(as.integer(strsplit(readLines(file.path(d, "gene_mouse.txt")), ",")[[1]]))
rownames(toc) <- rownames(bg) <- paste0("g", seq_len(nrow(toc))); colnames(toc) <- paste0("c", seq_len(ncol(toc)))
colnames(bg) <- paste0("b", seq_len(ncol(bg)))
for (v in c("default", "background")) {
  set.seed(1); t0 <- Sys.time()
  res <- if (v == "default") decontX(toc, verbose = FALSE) else decontX(toc, background = bg, verbose = FALSE)
  sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  dc <- res$decontXcounts
  rem <- toc - dc
  out <- data.frame(rho = res$contamination, rem_m = colSums(rem[mouse, , drop = FALSE]), rem_h = colSums(rem[!mouse, , drop = FALSE]), seconds = sec)
  write.csv(out, file.path(d, paste0("decontx_", v, ".csv")), row.names = FALSE)
  cat("DecontX", v, "mean rho", mean(res$contamination), "seconds", sec, "\n")
}
