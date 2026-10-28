# Ecosystem timings in R (called by s1bench/eco.py). Prints one JSON line per measurement.
#   Rscript eco.R <dataset_dir> <threads> <repeats>
# dataset_dir holds s1.s1, h5ad_gzip.h5ad and tenx_h5.h5 from the write stage.
# Each call is the one shown in Fig. 4; the object it returns is checked against the 10x HDF5 reference.
suppressPackageStartupMessages({ library(Matrix); library(jsonlite) })
a <- commandArgs(trailingOnly = TRUE)
d <- a[1]; threads <- as.integer(a[2]); reps <- as.integer(a[3])
s1 <- file.path(d, "s1.s1"); h5ad <- file.path(d, "h5ad_gzip.h5ad"); h5 <- file.path(d, "tenx_h5.h5")
evict <- function(p) system2("python3", c("-m", "s1bench.evict", shQuote(p)))
counts_of <- function(x) {
  if (inherits(x, "SingleCellExperiment")) return(SummarizedExperiment::assay(x, "counts"))
  if (inherits(x, "Seurat")) return(SeuratObject::LayerData(x, assay = "RNA", layer = "counts"))
  x
}
ref <- NULL
time_it <- function(name, file, expr_fn) {
  for (k in seq_len(reps + 1)) {
    evict(file); gc()
    t0 <- Sys.time(); obj <- expr_fn(); dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (k > 1) {
      m <- counts_of(obj)
      same <- if (is.null(ref)) NA else isTRUE(all.equal(unname(as(m, "CsparseMatrix")), unname(ref)))
      cat(toJSON(list(lang = "R", call = name, seconds = dt, rep = k - 1, equal_to_reference = same), auto_unbox = TRUE), "\n")
    }
  }
}
ref <- Seurat::Read10X_h5(h5)
time_it("Seurat::Read10X_h5", h5, function() Seurat::Read10X_h5(h5))
time_it("zellkonverter::readH5AD", h5ad, function() zellkonverter::readH5AD(h5ad, reader = "R"))
time_it("singlet::load(as='sce')", s1, function() singlet::load(s1, as = "sce"))
time_it("singlet::load(as='seurat')", s1, function() singlet::load(s1, as = "seurat"))
