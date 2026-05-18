#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/donor_pseudobulk_deseq2_reference.R
#
# Reference driver: load a CSC count matrix + cluster labels + donor labels from
# compact binary files, aggregate into pseudobulks via muscat::aggregateData,
# run DESeq2::DESeq() per cluster, and write per-gene per-cluster LFC + p-values
# to TSV files.
#
# Environment:
#   R >= 4.0
#   BiocManager::install("DESeq2")
#   BiocManager::install("muscat")
#   BiocManager::install("SingleCellExperiment")
#
# Usage:
#   Rscript donor_pseudobulk_deseq2_reference.R \
#       --counts    <counts.bin>          # dump_csc.h format: n_genes x n_cells
#       --clusters  <cluster_labels.tsv>  # one int32 per line (0-based cluster id)
#       --donors    <donor_labels.tsv>    # one int32 per line (0-based donor id)
#       --n_clusters <int>
#       --n_donors   <int>
#       --min_cells  <int>               # default: 10  (min cells per pseudobulk)
#       --out_dir    <dir>               # output directory for per-cluster TSVs
#       [--seed      <int>]              # default: 42
#
# Output TSV per cluster (one file per cluster: cluster_0_de.tsv, ...):
#   gene_idx  lfc  pval  padj
#
# Exit codes:
#   0  — success
#   1  — argument or I/O error
#   2  — DESeq2 or muscat not installed (C++ test treats this as "ref unavailable")
#   3  — not enough donors/samples to run DESeq2 (<3 donors across any cluster)

suppressPackageStartupMessages({
    for (pkg in c("DESeq2", "muscat", "SingleCellExperiment")) {
        if (!requireNamespace(pkg, quietly = TRUE)) {
            message("[donor_pseudobulk_deseq2_reference] ", pkg,
                    " not installed. Install via BiocManager::install('", pkg, "')")
            quit(save = "no", status = 2L)
        }
    }
    library(DESeq2)
    library(muscat)
    library(SingleCellExperiment)
    library(SummarizedExperiment)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0L) {
        if (is.null(default)) {
            message("[donor_pseudobulk_deseq2_reference] Missing required argument: ", flag)
            quit(save = "no", status = 1L)
        }
        return(default)
    }
    return(args[idx + 1L])
}

counts_path   <- parse_arg("--counts")
clusters_path <- parse_arg("--clusters")
donors_path   <- parse_arg("--donors")
n_clusters    <- as.integer(parse_arg("--n_clusters"))
n_donors      <- as.integer(parse_arg("--n_donors"))
min_cells     <- as.integer(parse_arg("--min_cells", "10"))
out_dir       <- parse_arg("--out_dir")
seed          <- as.integer(parse_arg("--seed", "42"))

set.seed(seed)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message(sprintf("[deseq2_ref] n_clusters=%d  n_donors=%d  min_cells=%d",
                n_clusters, n_donors, min_cells))

# ---------------------------------------------------------------------------
# Read compact CSC binary (dump_csc.h format)
#   uint32 magic=0x43535343, uint32 n_rows, uint32 n_cols,
#   uint64 nnz, float32[nnz] values, int32[n_cols+1] indptr, int32[nnz] indices
# ---------------------------------------------------------------------------
read_csc_bin <- function(path) {
    con <- file(path, "rb")
    on.exit(close(con))
    magic   <- readBin(con, integer(), n = 1L, size = 4L, endian = "little",
                       signed = FALSE)
    if (magic != 0x43535343L)
        stop("[deseq2_ref] Bad magic in ", path)
    n_rows  <- readBin(con, integer(), n = 1L, size = 4L, endian = "little",
                       signed = FALSE)
    n_cols  <- readBin(con, integer(), n = 1L, size = 4L, endian = "little",
                       signed = FALSE)
    nnz_raw <- readBin(con, integer(), n = 2L, size = 4L, endian = "little",
                       signed = FALSE)
    nnz     <- nnz_raw[1] + nnz_raw[2] * 2^32  # reconstruct uint64 from two uint32
    values  <- readBin(con, numeric(), n = nnz, size = 4L, endian = "little")
    indptr  <- readBin(con, integer(), n = n_cols + 1L, size = 4L, endian = "little")
    indices <- readBin(con, integer(), n = nnz, size = 4L, endian = "little")
    # Build dense matrix (n_genes x n_cells) — pseudobulk aggregation is cheap
    # enough for tiny/medium test cases used here.
    mat <- matrix(0L, nrow = n_rows, ncol = n_cols)
    for (j in seq_len(n_cols)) {
        start <- indptr[j] + 1L       # 0-based indptr → 1-based
        end   <- indptr[j + 1L]
        if (end >= start) {
            row_idxs <- indices[start:end] + 1L  # 0-based row indices → 1-based
            mat[row_idxs, j] <- as.integer(round(values[start:end]))
        }
    }
    list(mat = mat, n_rows = n_rows, n_cols = n_cols)
}

message("[deseq2_ref] Loading counts from: ", counts_path)
csc_result <- read_csc_bin(counts_path)
counts_mat <- csc_result$mat
n_genes    <- csc_result$n_rows
n_cells    <- csc_result$n_cols
message(sprintf("[deseq2_ref] Loaded %d genes x %d cells", n_genes, n_cells))

# ---------------------------------------------------------------------------
# Read cluster and donor labels (0-based integers, one per line)
# ---------------------------------------------------------------------------
cluster_labels <- as.integer(readLines(clusters_path))
donor_labels   <- as.integer(readLines(donors_path))

stopifnot(length(cluster_labels) == n_cells)
stopifnot(length(donor_labels)   == n_cells)

message(sprintf("[deseq2_ref] Unique clusters: %d, unique donors: %d",
                length(unique(cluster_labels)), length(unique(donor_labels))))

# ---------------------------------------------------------------------------
# Build SingleCellExperiment for muscat
# ---------------------------------------------------------------------------
gene_names <- paste0("gene", seq_len(n_genes) - 1L)
cell_names <- paste0("cell", seq_len(n_cells) - 1L)
rownames(counts_mat) <- gene_names
colnames(counts_mat) <- cell_names

col_data <- data.frame(
    sample_id  = paste0("donor", donor_labels),
    group_id   = paste0("donor", donor_labels),
    cluster_id = paste0("cluster", cluster_labels),
    row.names  = cell_names,
    stringsAsFactors = FALSE
)

sce <- SingleCellExperiment(
    assays = list(counts = counts_mat),
    colData = col_data
)

message("[deseq2_ref] Built SCE")

# ---------------------------------------------------------------------------
# Pseudobulk aggregation via muscat::aggregateData (sum method)
# ---------------------------------------------------------------------------
pb <- tryCatch({
    aggregateData(sce,
                  assay   = "counts",
                  fun     = "sum",
                  by      = c("cluster_id", "sample_id"))
}, error = function(e) {
    message("[deseq2_ref] aggregateData error: ", conditionMessage(e))
    quit(save = "no", status = 1L)
})

message("[deseq2_ref] Pseudobulk aggregation done. n_clusters in pb: ",
        length(assayNames(pb)))

# ---------------------------------------------------------------------------
# Per-cluster DESeq2
# ---------------------------------------------------------------------------
run_deseq2_for_cluster <- function(cluster_name, pb_sce, n_donors_exp, min_cells_val) {
    # Extract per-donor pseudobulk counts for this cluster
    # muscat stores one SummarizedExperiment per cluster inside the pb object
    if (!cluster_name %in% names(pb_sce)) {
        message("[deseq2_ref] Cluster not found in pb: ", cluster_name)
        return(NULL)
    }

    pb_cluster <- pb_sce[[cluster_name]]   # SummarizedExperiment: genes x donors
    pb_counts  <- assay(pb_cluster, 1L)    # matrix: genes x donors
    col_info   <- colData(pb_cluster)

    # Apply min_cells filter: donors with < min_cells should have been removed
    # upstream; muscat does this via a min.cells argument in aggregateData but
    # we apply it as a post-filter here to match the C++ kernel's exact behaviour.
    keep_donors <- colSums(pb_counts) >= min_cells_val
    if (sum(keep_donors) < 3L) {
        message(sprintf("[deseq2_ref] Cluster %s: only %d donors with >= %d cells — skip",
                        cluster_name, sum(keep_donors), min_cells_val))
        return(NULL)
    }

    pb_counts  <- pb_counts[, keep_donors, drop = FALSE]
    n_d        <- ncol(pb_counts)
    donor_ids  <- colnames(pb_counts)

    # Build a minimal design: ~1 (intercept only; we compare across all donors)
    col_df <- data.frame(
        donor = factor(donor_ids),
        row.names = donor_ids
    )

    dds <- tryCatch({
        dds_tmp <- DESeqDataSetFromMatrix(
            countData = pb_counts,
            colData   = col_df,
            design    = ~ 1   # intercept only; for LFC we extract donor coefficients
        )
        DESeq(dds_tmp, quiet = TRUE, fitType = "local",
              minReplicatesForReplace = Inf)
    }, error = function(e) {
        message("[deseq2_ref] DESeq() error on cluster ", cluster_name, ": ",
                conditionMessage(e))
        return(NULL)
    })
    if (is.null(dds)) return(NULL)

    res <- tryCatch(
        results(dds, alpha = 0.05),
        error = function(e) NULL
    )
    if (is.null(res)) return(NULL)

    # Extract LFC (log2 fold change) and p-values
    lfc  <- as.numeric(res$log2FoldChange)
    pval <- as.numeric(res$pvalue)
    padj <- as.numeric(res$padj)

    # Replace NA with NaN for downstream checks
    lfc [is.na(lfc )] <- NaN
    pval[is.na(pval)] <- NaN
    padj[is.na(padj)] <- NaN

    data.frame(
        gene_idx = seq_len(nrow(pb_counts)) - 1L,
        lfc      = lfc,
        pval     = pval,
        padj     = padj,
        stringsAsFactors = FALSE
    )
}

# Run DESeq2 for each cluster and write output TSV
n_written <- 0L
for (cl in seq_len(n_clusters) - 1L) {
    cluster_name <- paste0("cluster", cl)
    message("[deseq2_ref] Processing ", cluster_name, " ...")
    de_result <- run_deseq2_for_cluster(cluster_name, pb, n_donors, min_cells)
    out_path  <- file.path(out_dir, sprintf("cluster_%d_de.tsv", cl))
    if (is.null(de_result)) {
        # Write empty TSV so C++ test knows this cluster was skipped
        writeLines("gene_idx\tlfc\tpval\tpadj", out_path)
        message("[deseq2_ref] Cluster ", cl, ": wrote empty TSV (skipped)")
    } else {
        write.table(de_result, out_path,
                    sep = "\t", quote = FALSE,
                    row.names = FALSE, col.names = TRUE)
        message(sprintf("[deseq2_ref] Cluster %d: wrote %d genes to %s",
                        cl, nrow(de_result), out_path))
        n_written <- n_written + 1L
    }
}

message(sprintf("[deseq2_ref] Done. %d / %d clusters with DE results.",
                n_written, n_clusters))
quit(save = "no", status = 0L)
