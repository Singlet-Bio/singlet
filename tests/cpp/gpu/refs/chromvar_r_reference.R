#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/chromvar_r_reference.R
#
# Reference driver: run chromVAR::computeDeviations() +
#                   chromVAR::computeVariability() on a
#                   peak × cell accessibility matrix + motif × peak binary
#                   annotation, then write results to TSV files.
#
# Environment:
#   R >= 4.0
#   BiocManager::install("chromVAR")   # also pulls SummarizedExperiment, BiocParallel
#   install.packages(c("Matrix"))
#
# Usage:
#   Rscript chromvar_r_reference.R \
#       --counts    <counts.tsv>          # peak × cell integer matrix (tab-sep, no header)
#       --motif_in_peak <motif_peak.tsv>  # motif × peak 0/1 matrix (tab-sep, no header)
#       --n_cells   <int>                 # number of cells (columns)
#       --n_peaks   <int>                 # number of peaks (rows)
#       --n_motifs  <int>                 # number of motifs (rows of motif_in_peak)
#       --output_deviations <file>        # TSV: cell × motif bias-corrected deviations
#       --output_variability <file>       # TSV: motif variability scores (one per line)
#       --n_bg      <int>                 # background peaks per peak (default 50)
#       --seed      <int>                 # RNG seed (default 42)
#
# Output formats:
#   deviations TSV  — n_cells rows × n_motifs columns, numeric, no row/col names.
#   variability TSV — n_motifs rows × 1 column: variability score, no header.
#
# Exit code 2 if chromVAR is not installed (allows C++ harness to GTEST_SKIP
# rather than GTEST_FAIL).

suppressPackageStartupMessages({
    if (!requireNamespace("chromVAR", quietly = TRUE)) {
        message("[chromvar_r_reference] chromVAR not installed. ",
                "Run: BiocManager::install('chromVAR')")
        quit(status = 2L)
    }
    if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
        message("[chromvar_r_reference] SummarizedExperiment not installed.")
        quit(status = 2L)
    }
    if (!requireNamespace("Matrix", quietly = TRUE)) {
        message("[chromvar_r_reference] Matrix not installed.")
        quit(status = 2L)
    }
    library(chromVAR)
    library(SummarizedExperiment)
    library(Matrix)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0L) {
        if (is.null(default)) stop("Missing required argument: ", flag)
        return(default)
    }
    return(args[idx + 1L])
}

counts_path     <- parse_arg("--counts")
motif_path      <- parse_arg("--motif_in_peak")
n_cells         <- as.integer(parse_arg("--n_cells"))
n_peaks         <- as.integer(parse_arg("--n_peaks"))
n_motifs        <- as.integer(parse_arg("--n_motifs"))
out_dev_path    <- parse_arg("--output_deviations")
out_var_path    <- parse_arg("--output_variability")
n_bg            <- as.integer(parse_arg("--n_bg", "50"))
seed            <- as.integer(parse_arg("--seed", "42"))

set.seed(seed)

# ---------------------------------------------------------------------------
# Load peak × cell count matrix
# ---------------------------------------------------------------------------
message(sprintf("[chromvar_r_reference] Reading counts (%d peaks × %d cells)...",
                n_peaks, n_cells))
counts_raw <- as.matrix(read.table(counts_path, header = FALSE, sep = "\t"))
if (nrow(counts_raw) != n_peaks || ncol(counts_raw) != n_cells) {
    stop(sprintf("[chromvar_r_reference] counts shape mismatch: got %d × %d, expected %d × %d",
                 nrow(counts_raw), ncol(counts_raw), n_peaks, n_cells))
}
counts_sparse <- Matrix::Matrix(counts_raw, sparse = TRUE)

# ---------------------------------------------------------------------------
# Load motif × peak binary annotation
# ---------------------------------------------------------------------------
message(sprintf("[chromvar_r_reference] Reading motif_in_peak (%d motifs × %d peaks)...",
                n_motifs, n_peaks))
motif_raw <- as.matrix(read.table(motif_path, header = FALSE, sep = "\t"))
if (nrow(motif_raw) != n_motifs || ncol(motif_raw) != n_peaks) {
    stop(sprintf("[chromvar_r_reference] motif_in_peak shape mismatch: got %d × %d, expected %d × %d",
                 nrow(motif_raw), ncol(motif_raw), n_motifs, n_peaks))
}

# Convert to list-of-integer-vectors format expected by chromVAR::computeDeviations.
# Each element is a logical vector of length n_peaks indicating presence.
peak_indices_list <- lapply(seq_len(n_motifs), function(m) {
    which(motif_raw[m, ] != 0L)
})
names(peak_indices_list) <- paste0("motif_", seq_len(n_motifs))

# ---------------------------------------------------------------------------
# Build a minimal RangedSummarizedExperiment (chromVAR requires it)
# chromVAR expects:
#   - assay named "counts": peak × cell matrix
#   - rowData with a "GC" column (GC content per peak); we derive from peak index.
#   - No real genomic ranges needed — use dummy GRanges.
# ---------------------------------------------------------------------------
peak_names <- paste0("peak_", seq_len(n_peaks))
cell_names <- paste0("cell_", seq_len(n_cells))
rownames(counts_sparse) <- peak_names
colnames(counts_sparse) <- cell_names
rownames(motif_raw) <- names(peak_indices_list)

# Dummy GRanges (chromVAR only uses them for background matching via GC content
# and accessibility — we supply a fake GC column instead).
if (!requireNamespace("GenomicRanges", quietly = TRUE)) {
    stop("[chromvar_r_reference] GenomicRanges not installed; required by chromVAR.")
}
gr <- GenomicRanges::GRanges(
    seqnames = "chr1",
    ranges   = IRanges::IRanges(
        start = (seq_len(n_peaks) - 1L) * 500L + 1L,
        width = 500L
    )
)
names(gr) <- peak_names

# Build SE
se <- SummarizedExperiment::SummarizedExperiment(
    assays   = list(counts = counts_sparse),
    rowRanges = gr
)

# Add GC content (fake but deterministic — chromVAR uses this for background
# peak matching; for synthetic data any reasonable value works).
SummarizedExperiment::rowData(se)$GC <- 0.4 + 0.2 * sin(seq_len(n_peaks) * 0.1)

# ---------------------------------------------------------------------------
# chromVAR pipeline
# ---------------------------------------------------------------------------
message("[chromvar_r_reference] Adding GC bias...")
se <- chromVAR::addGCBias(se, genome = NULL)   # genome=NULL uses the rowData GC column

message("[chromvar_r_reference] Getting background peaks...")
set.seed(seed)
bg_peaks <- chromVAR::getBackgroundPeaks(se, niterations = n_bg)

message("[chromvar_r_reference] Computing deviations...")
set.seed(seed)
dev <- chromVAR::computeDeviations(
    object           = se,
    annotations      = peak_indices_list,
    background_peaks = bg_peaks
)

# dev is a SummarizedExperiment with assays "deviations" (z-score) and
# "deviationScores" (bias-corrected deviation).  The design doc uses the
# bias-corrected deviation (deviationScores) for the Spearman correlation.
dev_matrix <- SummarizedExperiment::assay(dev, "deviationScores")  # motif × cell
# Transpose to cell × motif for output (matches C++ layout).
dev_matrix <- t(dev_matrix)  # n_cells × n_motifs

message("[chromvar_r_reference] Computing variability...")
variability <- chromVAR::computeVariability(dev)
# variability is a data.frame with columns: name, variability, bootstrap_lower,
# bootstrap_upper, p_value_change.  We want the "variability" column in motif order.
var_scores <- variability$variability   # length n_motifs

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
message(sprintf("[chromvar_r_reference] Writing deviations (%d × %d) to: %s",
                nrow(dev_matrix), ncol(dev_matrix), out_dev_path))
write.table(dev_matrix, file = out_dev_path,
            sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)

message(sprintf("[chromvar_r_reference] Writing variability (%d entries) to: %s",
                length(var_scores), out_var_path))
write.table(data.frame(variability = var_scores), file = out_var_path,
            sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)

message("[chromvar_r_reference] Done.")
