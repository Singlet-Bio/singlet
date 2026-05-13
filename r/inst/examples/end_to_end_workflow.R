# End-to-end singlet pipeline-output workflow example (R version).
#
# Demonstrates the typical user journey from a singlet pipeline output
# directory to a SingleCellExperiment ready for Bioconductor downstream
# analysis. This script is meant to be readable as documentation as much
# as it is to be executed.
#
# Usage:
#     Rscript end_to_end_workflow.R [pipeline_output_dir]
#
# If no path is given, the script picks one of the canonical small
# fixtures on the Clipper NFS tree (GSM7103327, ~410 cells).
#
# Requirements: singlet, SingleCellExperiment, scater, scran (optional)

suppressPackageStartupMessages({
    library(singlet)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1L) {
    sample_dir <- args[1]
} else {
    sample_dir <- file.path(
        "/mnt/projects/debruinz_project/singlet_pipeline",
        "quant/scrna/GSE227/GSE227136/GSM7103327"
    )
}

if (!dir.exists(sample_dir)) {
    stop("not a directory: ", sample_dir)
}

cat("=== singlet R end-to-end workflow ===\n")
cat("sample:", sample_dir, "\n\n")

# ----------------------------------------------------------------------
# Step 1: Read the pipeline directory into a list of dgCMatrix
# ----------------------------------------------------------------------
cat("Step 1: reading pipeline directory\n")
t0 <- Sys.time()
dd <- read_singlet_dir(sample_dir)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  loaded %d matrices in %.2fs\n", length(dd), elapsed))
cat("  matrices:", paste(names(dd), collapse = ", "), "\n")
kv <- attr(dd, "user_kv")
cat("  gsm_id:    ", kv[["gsm_id"]], "\n")
cat("  gse_id:    ", kv[["gse_id"]], "\n")
cat("  organism:  ", kv[["organism"]], "\n")
cat("  protocol:  ", kv[["protocol"]], "\n")
cat("  pipeline:  ", kv[["singlet_version"]], "(", kv[["pipeline_date"]], ")\n\n")

# ----------------------------------------------------------------------
# Step 2: Convert to SingleCellExperiment
# ----------------------------------------------------------------------
if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    cat("SingleCellExperiment not installed — stopping at Step 1.\n")
    cat("Install with: BiocManager::install('SingleCellExperiment')\n")
    quit(status = 0)
}

cat("Step 2: building SingleCellExperiment\n")
sce <- as_sce(sample_dir)
cat("  class:     ", paste(class(sce), collapse = "/"), "\n")
cat("  dim:       ", paste(dim(sce), collapse = " x "), "\n")
cat("  assays:    ", paste(SummarizedExperiment::assayNames(sce), collapse = ", "), "\n")
cat("  altExps:   ", paste(SingleCellExperiment::altExpNames(sce), collapse = ", "), "\n")
cat("  colData cols:", ncol(SummarizedExperiment::colData(sce)), "(from auto-loaded sidecars)\n")
md <- SummarizedExperiment::metadata(sce)
cat("  metadata$singlet$gsm_id:", md$singlet[["gsm_id"]], "\n")
derived <- md$singlet_derived_assays
if (!is.null(derived) && length(derived) > 0) {
    cat("  *** derived assays from per-feature matrices:", paste(derived, collapse = ", "), "\n")
}
cat("\n")

# ----------------------------------------------------------------------
# Step 3: Build a Seurat object from the same data
# ----------------------------------------------------------------------
if (!requireNamespace("Seurat", quietly = TRUE)) {
    cat("Seurat not installed — stopping at Step 2.\n")
    cat("Install with: install.packages('Seurat')\n")
    quit(status = 0)
}

cat("Step 3: building Seurat\n")
obj <- as_seurat(sample_dir)
cat("  class:    ", paste(class(obj), collapse = "/"), "\n")
cat("  cells:    ", ncol(obj), "\n")
cat("  features: ", nrow(obj), "\n")
cat("  assays:   ", paste(Seurat::Assays(obj), collapse = ", "), "\n")
cat("  meta.data cols:", ncol(obj@meta.data), "\n")
cat("  misc$singlet$gsm_id:", obj@misc$singlet[["gsm_id"]], "\n")
cat("\n")

# ----------------------------------------------------------------------
# Step 4: Show the Tier 2 drop reconstruction in action
# ----------------------------------------------------------------------
cat("Step 4: aggregator demonstration\n")
ex <- read_1pz(file.path(sample_dir, "exon_counts.1pz"))
cat("  exon_counts:", paste(dim(ex), collapse = " x "), "nnz=", length(ex@x), "\n")
agg <- aggregate_features_to_gene(ex)
cat("  aggregated: ", paste(dim(agg), collapse = " x "), "nnz=", length(agg@x), "\n")
cat("  total sum match: ", sum(ex) == sum(agg), "\n")
cat("\n=== done ===\n")
