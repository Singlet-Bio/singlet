#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/fgsea_r_reference.R
#
# Reference driver: run fgsea::fgsea on a preranked stats vector + gene sets,
# then write results to a TSV file.
#
# Environment:
#   R >= 4.0
#   BiocManager::install("fgsea")
#   install.packages(c("jsonlite"))
#
# Usage:
#   Rscript fgsea_r_reference.R \
#       --stats   <stats.tsv>        # two-col: gene_idx<TAB>stat
#       --genesets <genesets.tsv>    # three-col: set_name<TAB>gene_idx<TAB>weight
#       --output  <results.tsv>      # output: one row per pathway
#       --minSize 15
#       --maxSize 500
#       --nperm   10000
#       --seed    42
#
# Output TSV columns (header row included):
#   pathway  es  nes  pval  padj  size  n_leading_edge
#
# The C++ harness writes the input files and reads the output TSV.

suppressPackageStartupMessages({
    if (!requireNamespace("fgsea", quietly = TRUE)) {
        stop("[fgsea_r_reference] fgsea not installed. ",
             "Run: BiocManager::install('fgsea')")
    }
    library(fgsea)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0) {
        if (is.null(default)) stop("Missing required argument: ", flag)
        return(default)
    }
    return(args[idx + 1L])
}

stats_path    <- parse_arg("--stats")
genesets_path <- parse_arg("--genesets")
output_path   <- parse_arg("--output")
min_size      <- as.integer(parse_arg("--minSize", "15"))
max_size      <- as.integer(parse_arg("--maxSize", "500"))
nperm         <- as.integer(parse_arg("--nperm",   "10000"))
seed          <- as.integer(parse_arg("--seed",    "42"))

set.seed(seed)

# ---------------------------------------------------------------------------
# Load stats (gene_idx -> stat)
# ---------------------------------------------------------------------------
stats_df <- read.table(stats_path, header = FALSE, sep = "\t",
                       col.names = c("gene_idx", "stat"),
                       colClasses = c("integer", "numeric"))
if (nrow(stats_df) == 0) stop("[fgsea_r_reference] Empty stats file: ", stats_path)

stats_vec <- stats_df$stat
names(stats_vec) <- as.character(stats_df$gene_idx)

message(sprintf("[fgsea_r_reference] Loaded %d stats.", length(stats_vec)))

# ---------------------------------------------------------------------------
# Load gene sets (set_name, gene_idx, weight) -> named list of integer vectors
# The weight column is ignored by fgsea (preranked uses only membership).
# ---------------------------------------------------------------------------
gs_df <- read.table(genesets_path, header = FALSE, sep = "\t",
                    col.names = c("set_name", "gene_idx", "weight"),
                    colClasses = c("character", "integer", "numeric"))
if (nrow(gs_df) == 0) stop("[fgsea_r_reference] Empty gene sets file: ", genesets_path)

gene_sets <- lapply(split(as.character(gs_df$gene_idx), gs_df$set_name),
                    function(x) x)

message(sprintf("[fgsea_r_reference] Loaded %d gene sets.", length(gene_sets)))

# ---------------------------------------------------------------------------
# Run fgsea
# ---------------------------------------------------------------------------
set.seed(seed)
res <- fgsea::fgsea(
    pathways  = gene_sets,
    stats     = stats_vec,
    minSize   = min_size,
    maxSize   = max_size,
    nPermSimple = nperm
)

if (is.null(res) || nrow(res) == 0) {
    stop("[fgsea_r_reference] fgsea returned empty results.")
}

# ---------------------------------------------------------------------------
# Write output TSV
# ---------------------------------------------------------------------------
out_df <- data.frame(
    pathway       = res$pathway,
    es            = as.numeric(res$ES),
    nes           = as.numeric(res$NES),
    pval          = as.numeric(res$pval),
    padj          = as.numeric(res$padj),
    size          = as.integer(res$size),
    n_leading_edge = sapply(res$leadingEdge, length),
    stringsAsFactors = FALSE
)

write.table(out_df, file = output_path, sep = "\t",
            row.names = FALSE, quote = FALSE)

message(sprintf("[fgsea_r_reference] Wrote %d pathway results to: %s",
                nrow(out_df), output_path))
