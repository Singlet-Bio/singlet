# BPCells minibatch timing (called by s1bench/minibatch.py).
#   Rscript minibatch_bpcells.R <matrix_dir> <workload.json> <threads> <warmup_batches> [<url of a tar of matrix_dir>]
# The workload is a list of minibatches, each a list of [start, end) cell blocks (0-based). Each minibatch is read as a
# dgCMatrix of the selected cells (genes x cells). BPCells reads local directories only, so a remote tier downloads first.
# Prints one JSON line with the same fields as the Python loaders (conversion to a torch tensor is not attempted).
suppressPackageStartupMessages({ library(BPCells); library(Matrix); library(jsonlite) })
a <- commandArgs(trailingOnly = TRUE)
path <- a[1]; wl <- fromJSON(a[2], simplifyVector = FALSE); threads <- as.integer(a[3]); warm <- as.integer(a[4])
t0 <- Sys.time()
rec <- list(access = "local")
if (length(a) >= 5) {
  # The object store holds the directory's files under <key>/<file>; fetch them all, then read locally.
  tmp <- tempfile("bpc"); dir.create(tmp)
  for (f in list.files(path)) download.file(paste0(a[5], "/", f), file.path(tmp, f), quiet = TRUE, mode = "wb")
  path <- tmp
  rec$access <- "download"
  rec$download_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
} else {
  system2("python3", c("-m", "s1bench.evict", shQuote(path)))
}
t1 <- Sys.time()
m <- open_matrix_dir(path)
t2 <- Sys.time()
cols <- function(b) unlist(lapply(b, function(r) seq.int(r[[1]] + 1, r[[2]])))
x <- as(m[, cols(wl[[1]])], "dgCMatrix")
t3 <- Sys.time()
s <- function(a, b) as.numeric(difftime(b, a, units = "secs"))
rec$open_s <- s(t1, t2) + ifelse(is.null(rec$download_s), 0, rec$download_s)
rec$first_read_s <- s(t2, t3)
rec$to_tensor_s <- 0
rec$first_batch_s <- s(t0, t3)
rec$batch_rows <- ncol(x)
for (b in wl[2:(1 + warm)]) invisible(as(m[, cols(b)], "dgCMatrix"))
t5 <- Sys.time(); n <- 0
rest <- wl[(2 + warm):length(wl)]
for (b in rest) n <- n + ncol(as(m[, cols(b)], "dgCMatrix"))
rec$sustained_cells_per_s <- n / s(t5, Sys.time())
rec$sustained_batches <- length(rest)
cat(toJSON(rec, auto_unbox = TRUE, digits = NA), "\n")
