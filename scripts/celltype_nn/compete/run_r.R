# SPDX-License-Identifier: MIT
# R competitors in the cell type comparison, run as published (separate process; this script only calls their APIs).
#
#   seurat   Seurat 5 reference mapping (the label-transfer method behind Azimuth): reference = the stratified Census corpus
#            subsample (export.py reference), NormalizeData + FindVariableFeatures + ScaleData + RunPCA(30);
#            per query sample FindTransferAnchors(reference.reduction = "pca") + TransferData.
#            Output: predicted.id and prediction.score.max.
#   singler  SingleR (Bioconductor) with the same reference, log1p CP10k, aggr.ref = TRUE, de.method = "classic".
#            Output: labels (score = delta.next), and pruned.labels (NA = SingleR's own abstention).
#
#   Rscript scripts/celltype_nn/compete/run_r.R <method> <compete dir> [studies, comma-separated]
#
# Writes <dir>/calls/<method>/<GSE>/<GSM>.csv (barcode, pred_cl, score, prob, pruned) and timings.jsonl.

suppressPackageStartupMessages({ library(Matrix); library(Seurat); library(SingleR); library(BiocParallel) })
args <- commandArgs(trailingOnly = TRUE)
method <- args[1]; dir <- args[2]; studies <- if (length(args) >= 3 && nzchar(args[3])) strsplit(args[3], ",")[[1]] else NULL
threads <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8"))

read_set <- function(stem) {
  X <- as(readMM(paste0(stem, ".mtx")), "CsparseMatrix")
  obs <- read.csv(paste0(stem, ".cells.csv"), stringsAsFactors = FALSE)
  genes <- read.csv(file.path(dirname(stem), "genes.csv"), stringsAsFactors = FALSE)
  rownames(X) <- make.unique(genes$gene_id); colnames(X) <- paste0("c", seq_len(ncol(X)))
  list(X = X, obs = obs)
}
lognorm <- function(X) { s <- colSums(X); s[s == 0] <- 1; log1p(t(t(X) / s) * 1e4) }

samples <- read.csv(file.path(dir, "query", "samples.csv"), stringsAsFactors = FALSE)
if (!is.null(studies)) samples <- samples[samples$gse %in% studies, ]
refs <- list()
get_ref <- function(sp) {
  if (!is.null(refs[[sp]])) return(refs[[sp]])
  r <- read_set(file.path(dir, "reference", paste0("ref_", sp)))
  if (method == "seurat") {
    o <- CreateSeuratObject(r$X, meta.data = data.frame(cl = r$obs$cl, row.names = colnames(r$X)))
    o <- NormalizeData(o, verbose = FALSE); o <- FindVariableFeatures(o, nfeatures = 2000, verbose = FALSE)
    o <- ScaleData(o, verbose = FALSE); o <- RunPCA(o, npcs = 30, verbose = FALSE)
    refs[[sp]] <<- o
  } else {
    refs[[sp]] <<- list(L = lognorm(r$X), cl = r$obs$cl)
  }
  refs[[sp]]
}

outdir <- file.path(dir, "calls", method); dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
for (i in seq_len(nrow(samples))) {
  s <- samples[i, ]
  q <- read_set(file.path(dir, "query", s$gse, s$gsm))
  t0 <- Sys.time(); ref <- get_ref(s$species); t0 <- Sys.time()  # reference preparation is not charged to the sample
  pred <- rep(NA_character_, ncol(q$X)); score <- rep(NA_real_, ncol(q$X)); pruned <- pred
  res <- tryCatch({
    if (method == "seurat") {
      qo <- CreateSeuratObject(q$X); qo <- NormalizeData(qo, verbose = FALSE)
      k <- min(30, ncol(q$X) - 1)
      a <- FindTransferAnchors(reference = ref, query = qo, dims = 1:30, reference.reduction = "pca", k.filter = NA, verbose = FALSE)
      p <- TransferData(anchorset = a, refdata = ref$cl, dims = 1:30, k.weight = min(50, k), verbose = FALSE)
      pred <- p$predicted.id; score <- p$prediction.score.max; pruned <- pred
    } else {
      p <- SingleR(test = lognorm(q$X), ref = ref$L, labels = ref$cl, aggr.ref = TRUE, de.method = "classic",
                   BPPARAM = MulticoreParam(threads))
      pred <- p$labels; score <- p$delta.next; pruned <- p$pruned.labels
    }
    TRUE
  }, error = function(e) { message(s$gse, "/", s$gsm, ": ", conditionMessage(e)); FALSE })
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  od <- file.path(outdir, s$gse); dir.create(od, showWarnings = FALSE)
  write.csv(data.frame(barcode = q$obs$barcode, pred_cl = pred, score = score, prob = if (method == "seurat") score else NA,
                       pruned = pruned), file.path(od, paste0(s$gsm, ".csv")), row.names = FALSE)
  cat(sprintf('{"gse": "%s", "gsm": "%s", "n_cells": %d, "seconds": %.2f, "threads": %d, "device": "cpu", "ok": %s}\n',
              s$gse, s$gsm, ncol(q$X), secs, if (method == "singler") threads else 1L, tolower(res)),
      file = file.path(outdir, "timings.jsonl"), append = TRUE)
  message(method, " ", s$gse, "/", s$gsm, ": ", ncol(q$X), " cells ", round(secs, 1), "s")
}
