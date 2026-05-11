#' Convert a singlet pipeline directory to a SingleCellExperiment.
#'
#' Builds a \code{\link[SingleCellExperiment:SingleCellExperiment]{SingleCellExperiment}}
#' from a singlet pipeline output directory. The primary per-gene matrix
#' goes into ``assays$counts``; the velocity trio (spliced / unspliced /
#' ambiguous) become additional assays when present; per-cell TSV sidecars
#' (``cell_qc_metrics.tsv``, ``cell_cycle_scores.tsv``, etc.) are loaded
#' into ``colData``; the embedded GEO metadata (``user_kv``) becomes
#' ``metadata(sce)$singlet``.
#'
#' Per-feature matrices with a different axis than the per-gene matrices
#' (``exon_counts``, ``intron_counts``, ``sj_counts``, ``splice_psi``,
#' ``mt_heteroplasmy``, ``vdj_gene_usage``) are attached to
#' ``altExp(sce, name)`` so they stay query-addressable without polluting
#' the main assay axis.
#'
#' @param path Directory containing the pipeline's ``.1pz`` + TSV files.
#' @param primary_assay Matrix name to use for ``assays$counts``. Tried in
#'   order: user value, ``"spliced"``, ``"gene_counts"``, ``"gene_counts_em"``.
#' @return A \code{SingleCellExperiment}.
#'
#' @examples
#' \dontrun{
#' sce <- as_sce("quant/scrna/GSE174/GSE174399/GSM5293863")
#' sce
#' metadata(sce)$singlet[["gsm_id"]]
#' }
#'
#' @export
as_sce <- function(path, primary_assay = "spliced") {
    if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
        stop("as_sce requires the SingleCellExperiment package. ",
             "Install with `BiocManager::install('SingleCellExperiment')`.")
    }
    if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
        stop("as_sce requires the SummarizedExperiment package.")
    }

    dd <- read_singlet_dir(path)

    # Pick the primary per-gene assay
    per_gene <- c(primary_assay, "spliced", "gene_counts", "gene_counts_em")
    chosen <- NULL
    for (name in per_gene) {
        if (name %in% names(dd)) { chosen <- name; break }
    }
    if (is.null(chosen)) {
        stop("no per-gene count matrix found in ", path)
    }

    X <- dd[[chosen]]
    assays_list <- list(counts = X)
    derived_assays <- character(0)

    # Additional per-gene assays (same feature axis).
    # If a per-gene velocity matrix is missing but its per-feature
    # counterpart is present, reconstruct it via aggregate_features_to_gene.
    # This makes the SCE complete even when the pipeline drops
    # spliced.1pz / unspliced.1pz to save NFS storage.
    derive_from <- list(spliced = "exon_counts", unspliced = "intron_counts")
    for (nm in c("spliced", "unspliced", "ambiguous", "gene_counts_em")) {
        if (nm == chosen) next
        if (nm %in% names(dd) && all(dim(dd[[nm]]) == dim(X))) {
            assays_list[[nm]] <- dd[[nm]]
        } else if (nm %in% names(derive_from)) {
            src <- derive_from[[nm]]
            if (src %in% names(dd)) {
                agg <- aggregate_features_to_gene(dd[[src]])
                if (all(dim(agg) == dim(X))) {
                    assays_list[[nm]] <- agg
                    derived_assays <- c(derived_assays,
                                        sprintf("%s (from %s)", nm, src))
                }
            }
        }
    }

    # Column data — per-cell sidecar TSVs
    coldata <- .load_cell_sidecars(attr(dd, "path"), colnames(X))

    sce <- SingleCellExperiment::SingleCellExperiment(
        assays = assays_list,
        colData = coldata
    )

    # Alternative experiments — per-feature matrices with a different axis
    for (nm in c("exon_counts", "intron_counts", "sj_counts",
                 "splice_psi", "mt_heteroplasmy", "vdj_gene_usage")) {
        if (nm %in% names(dd)) {
            mat <- dd[[nm]]
            if (ncol(mat) == ncol(X)) {
                alt <- SingleCellExperiment::SingleCellExperiment(
                    assays = list(counts = mat)
                )
                SingleCellExperiment::altExp(sce, nm) <- alt
            }
        }
    }

    # Embedded GEO metadata
    user_kv <- attr(dd, "user_kv")
    if (!is.null(user_kv)) {
        SummarizedExperiment::metadata(sce)$singlet <- as.list(user_kv)
    }
    SummarizedExperiment::metadata(sce)$singlet_primary_assay <- chosen
    SummarizedExperiment::metadata(sce)$singlet_source_path <- attr(dd, "path")
    if (length(derived_assays) > 0L) {
        SummarizedExperiment::metadata(sce)$singlet_derived_assays <- derived_assays
    }

    sce
}


# Internal — load cell_qc_metrics.tsv and friends and align to barcode order.
.load_cell_sidecars <- function(dir_path, barcodes) {
    sidecars <- c(
        "cell_qc_metrics.tsv",
        "cell_cycle_scores.tsv",
        "doublet_scores.tsv",
        "read_stats.tsv",
        "ambient_contamination.tsv"
    )
    coldata <- data.frame(row.names = barcodes)
    for (sc in sidecars) {
        p <- file.path(dir_path, sc)
        if (!file.exists(p)) next
        df <- tryCatch(
            utils::read.table(p, sep = "\t", header = TRUE,
                              stringsAsFactors = FALSE, check.names = FALSE),
            error = function(e) NULL
        )
        if (is.null(df) || nrow(df) == 0L) next

        # Detect the barcode column
        bc_col <- NULL
        for (c in c("barcode", "cell_barcode", "cell", "cell_id", "CB")) {
            if (c %in% colnames(df)) { bc_col <- c; break }
        }
        if (is.null(bc_col)) {
            if (is.character(df[[1]])) bc_col <- colnames(df)[1]
            else next
        }

        rownames(df) <- df[[bc_col]]
        df[[bc_col]] <- NULL
        df <- df[match(barcodes, rownames(df)), , drop = FALSE]

        prefix <- sub("\\.tsv$", "", sc)
        colnames(df) <- paste0(prefix, "__", colnames(df))
        coldata <- cbind(coldata, df)
    }
    coldata
}
