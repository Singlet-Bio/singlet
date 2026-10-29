# SPDX-License-Identifier: MIT
# Research Alpha QC check 7 (R): load studies end to end with the R client from data.singlet.bio, timed.
#   Rscript qc_clients.R OUT.json GSE [GSE ...]
args <- commandArgs(trailingOnly = TRUE)
out <- args[[1]]; gses <- args[-1]
suppressPackageStartupMessages({ library(singlet); library(SingleCellExperiment); library(jsonlite) })
res <- list()
for (g in gses) {
  r <- list(gse = g)
  ok <- tryCatch({
    t <- system.time(st <- singlet_study(g, force = TRUE))[["elapsed"]]
    r$open_s <- t; r$n_samples <- length(st$gsm_ids)
    r$n_called_manifest <- sum(vapply(st$samples, function(s) as.numeric(if (is.null(s$n_called)) 0 else s$n_called), 0))
    t <- system.time(sce <- singlet::singlet_load(g))[["elapsed"]]
    r$load_s <- t; r$dim <- dim(sce); r$assays <- SummarizedExperiment::assayNames(sce)
    r$ncol_equals_called <- ncol(sce) == r$n_called_manifest
    cd <- colnames(SummarizedExperiment::colData(sce))
    r$label_cols <- cd[grepl("^(cell_type|sex_|ancestry)", cd)]
    if (all(c("spliced", "unspliced", "counts") %in% r$assays)) {
      d <- SummarizedExperiment::assay(sce, "counts") - SummarizedExperiment::assay(sce, "spliced") - SummarizedExperiment::assay(sce, "unspliced")
      r$counts_equals_spliced_plus_unspliced <- sum(abs(d)) == 0
    }
    r$total_counts <- sum(SummarizedExperiment::assay(sce, "counts"))
    big <- st$gsm_ids[[which.max(vapply(st$samples, function(s) as.numeric(if (is.null(s$n_called)) 0 else s$n_called), 0))]]
    t <- system.time(part <- singlet_cells(st, n_umi_exon > 500, samples = big, max_cells = 200, layers = TRUE))[["elapsed"]]
    r$partial <- list(gsm = big, dim = dim(part), s = t, assays = SummarizedExperiment::assayNames(part))
    t <- system.time(cl <- singlet_cells(st, called, columns = NULL))[["elapsed"]]
    r$cells <- list(dim = dim(cl), s = t)
    TRUE
  }, error = function(e) { r$error <<- conditionMessage(e); FALSE })
  cat(toJSON(r, auto_unbox = TRUE), "\n")
  res[[g]] <- r
}
writeLines(toJSON(res, auto_unbox = TRUE, pretty = TRUE), out)
