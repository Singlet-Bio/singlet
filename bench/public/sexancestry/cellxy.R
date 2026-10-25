#!/usr/bin/env Rscript
# Chromosomal-sex benchmark, step 2: cellXY::classifySex on the same counts Singlet used (sex_extract.py wrote them).
#   Rscript cellxy.R <work>/sex          (every <work>/sex/mtx/<GSM>/ without a <work>/sex/cellxy/<GSM>.tsv)
# Writes cellxy/<GSM>.tsv: col, pred_qc (classifySex default, qc = TRUE), pred_noqc (qc = FALSE), plus seconds per sample.
# cellXY: phipsonlab/cellXY (GitHub; it is not in Bioconductor), installed by setup_r.sbatch.
suppressPackageStartupMessages({ library(Matrix); library(cellXY) })
args <- commandArgs(trailingOnly = TRUE)
work <- args[1]
dir.create(file.path(work, "cellxy"), showWarnings = FALSE)
for (d in list.dirs(file.path(work, "mtx"), recursive = FALSE)) {
  gsm <- basename(d)
  out <- file.path(work, "cellxy", paste0(gsm, ".tsv"))
  if (file.exists(out)) next
  sp <- readLines(file.path(d, "species.txt"))
  genome <- if (identical(sp, "mouse")) "Mm" else "Hs"
  m <- readMM(gzfile(file.path(d, "matrix.mtx.gz")))
  genes <- readLines(file.path(d, "genes.tsv")); cols <- readLines(file.path(d, "cells.tsv"))
  # duplicated symbols (rare) are summed so rownames are unique, as classifySex needs symbols as rownames
  if (anyDuplicated(genes)) { f <- factor(genes, levels = unique(genes)); m <- fac2sparse(f) %*% m; genes <- levels(f) }
  m <- as(m, "CsparseMatrix"); rownames(m) <- genes; colnames(m) <- paste0("c", cols)
  t0 <- proc.time()[["elapsed"]]
  # classifySex densifies the whole matrix (as.matrix); process in blocks of cells to bound memory. Its QC (scuttle
  # perCellQCFilters, 3-MAD outliers) depends on the block, so blocks are random subsets of >= 5000 cells.
  n <- ncol(m); set.seed(1); idx <- sample.int(n); nb <- max(1, floor(n / 5000)); blocks <- split(idx, rep_len(seq_len(nb), n))
  pq <- pn <- setNames(rep(NA_character_, n), colnames(m))
  for (b in blocks) {
    sub <- m[, b, drop = FALSE]
    r1 <- classifySex(sub, genome = genome, qc = TRUE); pq[rownames(r1)] <- r1$prediction
    r2 <- classifySex(sub, genome = genome, qc = FALSE); pn[rownames(r2)] <- r2$prediction
  }
  secs <- proc.time()[["elapsed"]] - t0
  write.table(data.frame(col = cols, pred_qc = pq, pred_noqc = pn), out, sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf('{"gsm": "%s", "n_cells": %d, "cellxy_s": %.2f}\n', gsm, n, secs),
      file = file.path(work, "cellxy_timing.jsonl"), append = TRUE)
  message(gsm, ": ", n, " cells, ", round(secs, 1), " s")
}
