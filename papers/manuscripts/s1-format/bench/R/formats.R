# R side of the format benchmark (called by s1bench/formats/rfmt.py).
#   Rscript formats.R write            <fmt> <exchange_dir> <out_path> <threads>
#   Rscript formats.R write_from_h5ad  <fmt> <h5ad> <out_path> <threads>       (BPCells, matrices above 2^31 non-zeros)
#   Rscript formats.R read             <fmt> <path> <exchange_dir> <threads>
#   Rscript formats.R time             <fmt> <path> <threads> <repeats> <warmup> <cold 0|1>   -> JSON seconds on the last line
# Formats: rds_gz, rds_xz (saveRDS of a dgCMatrix), bpcells (BPCells bit-packed directory, uint32 counts).
# Exchange directory: p.f64, i.i32, x.i32, genes.txt, barcodes.txt (see rfmt.py).

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
cmd <- args[1]
fmt <- args[2]

read_exchange <- function(d) {
  g <- readLines(file.path(d, "genes.txt"))
  b <- readLines(file.path(d, "barcodes.txt"))
  p <- readBin(file.path(d, "p.f64"), "double", n = length(b) + 1, size = 8)
  nnz <- p[length(p)]
  i <- readBin(file.path(d, "i.i32"), "integer", n = nnz, size = 4)
  x <- readBin(file.path(d, "x.i32"), "integer", n = nnz, size = 4)
  new("dgCMatrix", i = i, p = as.integer(p), x = as.numeric(x), Dim = c(length(g), length(b)),
      Dimnames = list(g, b))
}

write_exchange <- function(m, d) {
  m <- as(m, "CsparseMatrix")
  m <- as(m, "generalMatrix")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  writeLines(rownames(m), file.path(d, "genes.txt"))
  writeLines(colnames(m), file.path(d, "barcodes.txt"))
  writeBin(as.double(m@p), file.path(d, "p.f64"), size = 8)
  writeBin(as.integer(m@i), file.path(d, "i.i32"), size = 4)
  writeBin(as.integer(round(m@x)), file.path(d, "x.i32"), size = 4)
}

evict <- function(path) {
  # Same eviction as the Python stages: posix_fadvise(DONTNEED) on every file under path.
  system2("python3", c("-m", "s1bench.evict", shQuote(path)))
}

write_fmt <- function(m, path, threads) {
  if (fmt == "rds_gz") {
    saveRDS(m, path, compress = "gzip")
  } else if (fmt == "rds_xz") {
    saveRDS(m, path, compress = "xz")
  } else if (fmt == "bpcells") {
    suppressPackageStartupMessages(library(BPCells))
    unlink(path, recursive = TRUE)
    mm <- convert_matrix_type(as(m, "IterableMatrix"), "uint32_t")
    write_matrix_dir(mm, path)
  } else stop("unknown format ", fmt)
}

read_fmt <- function(path, threads) {
  if (fmt %in% c("rds_gz", "rds_xz")) {
    readRDS(path)
  } else if (fmt == "bpcells") {
    suppressPackageStartupMessages(library(BPCells))
    as(open_matrix_dir(path), "dgCMatrix")
  } else stop("unknown format ", fmt)
}

if (cmd == "write") {
  m <- read_exchange(args[3])
  write_fmt(m, args[4], as.integer(args[5]))
} else if (cmd == "write_from_h5ad") {
  stopifnot(fmt == "bpcells")
  suppressPackageStartupMessages(library(BPCells))
  unlink(args[4], recursive = TRUE)
  # AnnData stores cells x genes; BPCells opens it as genes x cells.
  mm <- convert_matrix_type(open_matrix_anndata_hdf5(args[3]), "uint32_t")
  write_matrix_dir(mm, args[4])
} else if (cmd == "read") {
  write_exchange(read_fmt(args[3], as.integer(args[5])), args[4])
} else if (cmd == "time") {
  path <- args[3]; threads <- as.integer(args[4]); reps <- as.integer(args[5]); warm <- as.integer(args[6])
  cold <- args[7] == "1"
  out <- c()
  for (k in seq_len(warm + reps)) {
    if (cold) evict(path)
    t0 <- Sys.time()
    m <- read_fmt(path, threads)
    dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (k > warm) out <- c(out, dt)
    rm(m); invisible(gc())
  }
  cat(toJSON(out, digits = NA), "\n")
} else {
  stop("unknown command ", cmd)
}
