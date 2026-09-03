# SPDX-License-Identifier: MIT
# Reader for the .singlet bundle format (per-GSE ZIP64 archive).
#
# A .singlet file bundles ALL processed samples (GSMs) for one GEO Series
# (GSE) into a single ZIP64 archive. It is the primary distribution unit
# for the Singlet atlas and the only on-disk format that end users are
# expected to handle directly. The lower-level per-sample matrix codec
# (.1pz) is an implementation detail of the bundle.
#
# Bundle layout (mirrors the Python `singlet.bundle.SingletBundle`):
#
#   <GSE>.singlet                       (ZIP64 archive)
#   |-- manifest.json                   schema, gse_id, gsm_ids, checksums
#   |-- study_meta.json                 series title/summary + per-GSM obs
#   |-- feature_vocab.json              canonical gene_id <-> gene_name axis
#   `-- samples/
#       `-- <GSM>/
#           |-- exon_counts.1pz         per-feature spliced counts (STORED)
#           |-- intron_counts.1pz       per-feature unspliced counts (STORED)
#           |-- ... other .1pz matrices
#           |-- cell_calls.tsv          barcode, is_cell, ... (DEFLATED)
#           |-- summary.json            (DEFLATED)
#           `-- ... other JSON sidecars (DEFLATED)


# ---------------------------------------------------------------------------
# Internal: read a JSON member from an already-extracted bundle directory.
# ---------------------------------------------------------------------------
.bundle_read_json <- function(extract_dir, member) {
    p <- file.path(extract_dir, member)
    if (!file.exists(p)) {
        return(NULL)
    }
    jsonlite::fromJSON(p, simplifyVector = FALSE)
}


# ---------------------------------------------------------------------------
# Internal: aggregate a per-feature .1pz matrix to the canonical gene axis.
#
# Pipeline feature rownames look like
#   ENSG00000000003_TSPAN6_chrX:100627107-100629986
# where the Ensembl gene id is everything before the first underscore. We
# project every feature onto its gene and then re-order / pad the rows to
# match `gene_ids` (the canonical feature_vocab axis), so all GSMs in a
# bundle share an identical gene axis and can be combined column-wise.
#
# Returns a (n_genes x n_cells) dgCMatrix, or NULL if `mat` is NULL.
# ---------------------------------------------------------------------------
.bundle_aggregate_to_vocab <- function(mat, gene_ids) {
    if (is.null(mat)) {
        return(NULL)
    }
    feature_rownames <- rownames(mat)
    if (is.null(feature_rownames)) {
        stop("bundle .1pz matrix has no rownames; cannot map features to genes")
    }
    gene_of_feature <- sub("_.*$", "", feature_rownames)
    gene_index <- match(gene_of_feature, gene_ids)
    keep <- !is.na(gene_index)
    n_genes <- length(gene_ids)
    n_cells <- ncol(mat)
    if (!any(keep)) {
        out <- Matrix::sparseMatrix(i = integer(0), j = integer(0),
                                    x = numeric(0),
                                    dims = c(n_genes, n_cells))
        return(methods::as(out, "CsparseMatrix"))
    }
    # Projection matrix P (n_genes x n_features): P[g, f] = 1 if feature f
    # belongs to gene g. aggregated = P %*% mat.
    proj <- Matrix::sparseMatrix(
        i = gene_index[keep],
        j = which(keep),
        x = rep(1.0, sum(keep)),
        dims = c(n_genes, nrow(mat))
    )
    agg <- proj %*% mat
    methods::as(agg, "CsparseMatrix")
}


# ---------------------------------------------------------------------------
# Internal: load gene-level counts for one GSM from an extracted bundle.
#
# Sums exon + intron feature counts to gene level, restricts to called
# cells (cell_calls.tsv where is_cell is TRUE), and returns a list with the
# gene x cell matrix and the matched barcodes. Mirrors the Python reader's
# `_load_gsm_gene_counts`.
# ---------------------------------------------------------------------------
.bundle_load_gsm <- function(extract_dir, gsm, gene_ids) {
    sample_dir <- file.path(extract_dir, "samples", gsm)

    read_member <- function(fname) {
        p <- file.path(sample_dir, fname)
        if (!file.exists(p)) {
            return(NULL)
        }
        read_1pz(p)
    }

    exon <- read_member("exon_counts.1pz")
    intron <- read_member("intron_counts.1pz")

    if (is.null(exon) && is.null(intron)) {
        return(NULL)
    }

    exon_g <- .bundle_aggregate_to_vocab(exon, gene_ids)
    intron_g <- .bundle_aggregate_to_vocab(intron, gene_ids)

    # Barcodes: prefer the union order from whichever matrix exists. Both
    # .1pz files for one GSM share the same barcode (column) ordering.
    bc_source <- if (!is.null(exon)) exon else intron
    all_bcs <- colnames(bc_source)

    if (is.null(exon_g)) {
        gene_mat <- intron_g
    } else if (is.null(intron_g)) {
        gene_mat <- exon_g
    } else {
        gene_mat <- exon_g + intron_g
    }
    colnames(gene_mat) <- all_bcs
    rownames(gene_mat) <- gene_ids

    # Restrict to called cells if cell_calls.tsv is present and non-empty.
    cc_path <- file.path(sample_dir, "cell_calls.tsv")
    called <- NULL
    if (file.exists(cc_path)) {
        cc <- tryCatch(
            utils::read.table(cc_path, sep = "\t", header = TRUE,
                              stringsAsFactors = FALSE, check.names = FALSE),
            error = function(e) NULL
        )
        if (!is.null(cc) && nrow(cc) > 0L && "barcode" %in% colnames(cc)) {
            if ("is_cell" %in% colnames(cc)) {
                is_cell <- as.logical(cc$is_cell)
                is_cell[is.na(is_cell)] <- FALSE
                called <- cc$barcode[is_cell]
            } else {
                called <- cc$barcode
            }
        }
    }

    if (!is.null(called) && length(called) > 0L) {
        keep <- intersect(called, all_bcs)
        if (length(keep) == 0L) {
            return(NULL)
        }
        gene_mat <- gene_mat[, keep, drop = FALSE]
        all_bcs <- keep
    }

    if (ncol(gene_mat) == 0L) {
        return(NULL)
    }

    gene_mat <- methods::as(gene_mat, "CsparseMatrix")
    list(matrix = gene_mat, barcodes = all_bcs)
}


# ---------------------------------------------------------------------------
# read_singlet — read one local .singlet bundle into a SingleCellExperiment.
# ---------------------------------------------------------------------------

#' Read a `.singlet` bundle into a SingleCellExperiment
#'
#' Reads a single local `.singlet` bundle (the per-Series distribution unit
#' of the Singlet atlas) and assembles all of its samples into one combined
#' \code{\link[SingleCellExperiment:SingleCellExperiment]{SingleCellExperiment}}.
#' Gene-level counts are formed by summing spliced and unspliced features for
#' each gene onto the bundle's canonical gene axis, restricted to called
#' cells. Per-sample study metadata (series title, tissue, cell type,
#' disease, protocol, and any enriched fields) is attached to
#' \code{colData(sce)}.
#'
#' This is the file-path workhorse used by \code{\link{load}}. Most users
#' should call \code{\link{load}} instead, which also accepts GEO accessions
#' and downloads bundles on demand.
#'
#' @param path Path to a local `.singlet` file.
#' @return A \code{SingleCellExperiment} with one column per called cell
#'   (named \code{<GSM>_<barcode>}) and one row per gene. \code{colData}
#'   carries per-sample metadata; \code{metadata(sce)} carries the bundle's
#'   parsed \code{manifest} and \code{study_meta}.
#'
#' @examples
#' \dontrun{
#' sce <- read_singlet("GSE149298.singlet")
#' sce
#' table(sce$gsm_id)
#' }
#'
#' @seealso \code{\link{load}}, \code{\link{find}}
#' @export
read_singlet <- function(path) {
    path <- path.expand(as.character(path))
            if (!file.exists(path)) {
                            stop(sprintf("no such .singlet file: %s", path))
                        }
            if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
                            stop("read_singlet requires the SingleCellExperiment package. ",
                                                  "Install with `BiocManager::install('SingleCellExperiment')`.")
                        }
            if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
                            stop("read_singlet requires the SummarizedExperiment package.")
                        }

    extract_dir <- tempfile("singlet_bundle_")
    dir.create(extract_dir)
    on.exit(unlink(extract_dir, recursive = TRUE, force = TRUE), add = TRUE)
    utils::unzip(path, exdir = extract_dir)

    manifest <- .bundle_read_json(extract_dir, "manifest.json")
    study_meta <- .bundle_read_json(extract_dir, "study_meta.json")
    feature_vocab <- .bundle_read_json(extract_dir, "feature_vocab.json")
    if (is.null(feature_vocab)) {
        stop(sprintf("malformed bundle (no feature_vocab.json): %s", path))
    }

    gene_ids <- vapply(feature_vocab$genes, function(g) g$gene_id, character(1))
    gene_names <- vapply(feature_vocab$genes, function(g) g$gene_name, character(1))

    gsm_ids <- if (!is.null(manifest$gsm_ids)) {
        unlist(manifest$gsm_ids, use.names = FALSE)
    } else {
        list.dirs(file.path(extract_dir, "samples"),
                  full.names = FALSE, recursive = FALSE)
    }

    gsm_meta_map <- if (!is.null(study_meta)) study_meta$gsm_meta else NULL

    mats <- list()
    coldata_rows <- list()
    for (gsm in gsm_ids) {
        loaded <- .bundle_load_gsm(extract_dir, gsm, gene_ids)
        if (is.null(loaded)) next
        mat <- loaded$matrix
        bcs <- loaded$barcodes
        colnames(mat) <- paste0(gsm, "_", bcs)
        mats[[gsm]] <- mat

        info <- if (!is.null(gsm_meta_map) && gsm %in% names(gsm_meta_map)) {
            gsm_meta_map[[gsm]]
        } else {
            list()
        }
        coldata_rows[[gsm]] <- .bundle_obs_frame(gsm, info, ncol(mat))
    }

    if (length(mats) == 0L) {
        stop(sprintf("bundle %s contained no loadable cells", path))
    }

    X <- do.call(cbind, unname(mats))
    X <- methods::as(X, "CsparseMatrix")
    coldata <- do.call(rbind, unname(coldata_rows))
    rownames(coldata) <- colnames(X)

    rowdata <- S4Vectors::DataFrame(gene_name = gene_names)
    rownames(rowdata) <- gene_ids

    sce <- SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = X),
        colData = S4Vectors::DataFrame(coldata, check.names = FALSE),
        rowData = rowdata
    )

    if (!is.null(manifest)) {
        S4Vectors::metadata(sce)$manifest <- manifest
    }
    if (!is.null(study_meta)) {
        S4Vectors::metadata(sce)$study_meta <- study_meta
    }
    S4Vectors::metadata(sce)$singlet_bundle_path <- path

    sce
}


# ---------------------------------------------------------------------------
# Internal: build a per-GSM obs data.frame from study_meta fields.
# ---------------------------------------------------------------------------
.bundle_obs_frame <- function(gsm, info, n_cells) {
    get_field <- function(key, default = NA_character_) {
        v <- info[[key]]
        if (is.null(v) || length(v) == 0L) {
            return(default)
        }
        as.character(v[[1]])
    }
    enriched <- info$enriched
    get_enriched <- function(key) {
        if (is.null(enriched)) {
            return(NA_character_)
        }
        v <- enriched[[key]]
        if (is.null(v) || length(v) == 0L) NA_character_ else as.character(v[[1]])
    }

    df <- data.frame(
        gsm_id = rep(gsm, n_cells),
        organism = get_field("organism"),
        protocol = get_field("protocol"),
        protocol_name = get_field("protocol_name"),
        sample_source = get_field("sample_source"),
        sample_characteristics = get_field("sample_characteristics"),
        qc_flag = get_field("qc_flag"),
        reference_build = get_field("reference_build"),
        tissue = get_enriched("meta_tissue"),
        cell_type = get_enriched("meta_cell_type"),
        disease = get_enriched("meta_disease"),
        sex = get_enriched("meta_sex"),
        age = get_enriched("meta_age"),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    df
}
