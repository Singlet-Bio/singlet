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
#
# A sample with no usable cells returns a one-line reason (a character
# string) instead, so the caller can skip it and say why rather than fail
# the whole study. That covers the hollow samples some pipeline runs left
# behind: a 0 x 0 stub in place of a count matrix that was never written.
# ---------------------------------------------------------------------------
.bundle_load_gsm <- function(extract_dir, gsm, gene_ids) {
    sample_dir <- file.path(extract_dir, "samples", gsm)

    # A matrix with no rows or no columns carries no cells: treat it
    # exactly like an absent file.
    read_member <- function(fname) {
        p <- file.path(sample_dir, fname)
        if (!file.exists(p)) {
            return(NULL)
        }
        m <- read_1pz(p)
        if (nrow(m) == 0L || ncol(m) == 0L) {
            return(NULL)
        }
        m
    }

    exon <- read_member("exon_counts.1pz")
    intron <- read_member("intron_counts.1pz")

    if (is.null(exon) && is.null(intron)) {
        return("no count matrix (missing or empty)")
    }

    # Both .1pz files for one GSM come from one pileup and share the same
    # barcode (column) ordering. If they ever disagree, realign rather than
    # add mismatched columns.
    if (!is.null(exon) && !is.null(intron) &&
        !identical(colnames(exon), colnames(intron))) {
        if (is.null(colnames(exon)) || is.null(colnames(intron)) ||
            !setequal(colnames(exon), colnames(intron))) {
            return("exon and intron matrices have different barcodes")
        }
        intron <- intron[, colnames(exon), drop = FALSE]
    }

    bc_source <- if (!is.null(exon)) exon else intron
    all_bcs <- colnames(bc_source)
    if (is.null(all_bcs)) {
        return("count matrix has no barcodes")
    }

    exon_g <- .bundle_aggregate_to_vocab(exon, gene_ids)
    intron_g <- .bundle_aggregate_to_vocab(intron, gene_ids)

    if (is.null(exon_g)) {
        gene_mat <- intron_g
    } else if (is.null(intron_g)) {
        gene_mat <- exon_g
    } else {
        gene_mat <- exon_g + intron_g
    }
    colnames(gene_mat) <- all_bcs
    rownames(gene_mat) <- gene_ids

    zero <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                 dims = c(length(gene_ids), length(all_bcs)))
    spliced_g <- if (is.null(exon_g)) zero else exon_g
    unspliced_g <- if (is.null(intron_g)) zero else intron_g
    dimnames(spliced_g) <- dimnames(unspliced_g) <- list(gene_ids, all_bcs)

    # Restrict to called cells when cell_calls.tsv says which they are. An
    # empty call set means the pipeline looked and found no cells, so the
    # sample is skipped (as the Python reader does) rather than loaded with
    # every barcode, empty droplets included.
    called <- .called_barcodes_from_table(
        .read_cell_calls(file.path(sample_dir, "cell_calls.tsv")))
    if (!is.null(called)) {
        keep <- intersect(called, all_bcs)
        if (length(keep) == 0L) {
            return(if (length(called) == 0L) {
                "no called cells"
            } else {
                "called cells not found in the count matrix"
            })
        }
        gene_mat <- gene_mat[, keep, drop = FALSE]
        spliced_g <- spliced_g[, keep, drop = FALSE]
        unspliced_g <- unspliced_g[, keep, drop = FALSE]
        all_bcs <- keep
    }

    if (ncol(gene_mat) == 0L) {
        return("no cells")
    }

    gene_mat <- methods::as(gene_mat, "CsparseMatrix")
    list(
        matrix = gene_mat,
        spliced = methods::as(spliced_g, "CsparseMatrix"),
        unspliced = methods::as(unspliced_g, "CsparseMatrix"),
        barcodes = all_bcs
    )
}


# ---------------------------------------------------------------------------
# Internal: one line per skipped sample, capped so a study with hundreds of
# hollow samples does not produce a wall of text.
# ---------------------------------------------------------------------------
.describe_skipped <- function(skipped, max_listed = 10L) {
    items <- sprintf("%s (%s)", names(skipped), skipped)
    if (length(items) > max_listed) {
        items <- c(items[seq_len(max_listed)],
                   sprintf("and %d more", length(items) - max_listed))
    }
    paste(items, collapse = "; ")
}

# A string field of one feature_vocab gene entry, NA when missing or null.
.vocab_string <- function(gene, key) {
    v <- gene[[key]]
    if (is.null(v) || length(v) == 0L) NA_character_ else as.character(v[[1L]])
}


# ---------------------------------------------------------------------------
# Internal: the bundle reader behind read_singlet() and load().
#
# `gsms` restricts the result to those samples (load() passes it for GSM
# accessions); NULL reads every sample in the manifest. Samples with no
# usable cells are skipped with a single warning and recorded in
# metadata(sce)$skipped_samples, a data frame with columns gsm_id and
# reason (zero rows when nothing was skipped).
# ---------------------------------------------------------------------------
.read_singlet_bundle <- function(path, gsms = NULL) {
    path <- path.expand(as.character(path))
    if (!file.exists(path)) {
        stop(sprintf("no such .singlet file: %s", path))
    }
    .require_sce("read_singlet()")

    extract_dir <- tempfile("singlet_bundle_")
    dir.create(extract_dir)
    on.exit(unlink(extract_dir, recursive = TRUE, force = TRUE), add = TRUE)

    # Extract only what gene-level assembly reads: the top-level JSON
    # documents and, per sample, the two count matrices and the cell calls.
    # Bundles also carry splicing, mitochondrial, donor and other outputs
    # that can dwarf the counts; singlet_read() reaches those on demand.
    members <- .bundle_members(path)
    top <- intersect(c("manifest.json", "study_meta.json", "feature_vocab.json"),
                     members)
    if (length(top) > 0L) {
        utils::unzip(path, files = top, exdir = extract_dir)
    }

    manifest <- .bundle_read_json(extract_dir, "manifest.json")
    study_meta <- .bundle_read_json(extract_dir, "study_meta.json")
    feature_vocab <- .bundle_read_json(extract_dir, "feature_vocab.json")
    if (is.null(feature_vocab)) {
        stop(sprintf("malformed bundle (no feature_vocab.json): %s", path))
    }

    gene_ids <- vapply(feature_vocab$genes, .vocab_string, character(1),
                       key = "gene_id")
    gene_names <- vapply(feature_vocab$genes, .vocab_string, character(1),
                         key = "gene_name")

    manifest_gsms <- .json_field(manifest, "gsm_ids")
    all_gsms <- if (!is.null(manifest_gsms)) {
        as.character(unlist(manifest_gsms, use.names = FALSE))
    } else {
        .bundle_gsm_ids(path)
    }

    skipped <- character(0)  # named: GSM -> reason
    if (is.null(gsms)) {
        targets <- unique(all_gsms)
    } else {
        gsms <- unique(as.character(gsms))
        absent <- setdiff(gsms, all_gsms)
        skipped[absent] <- "not in this bundle; excluded when the study was packed"
        targets <- gsms[gsms %in% all_gsms]
    }
    n_asked <- length(targets) + length(skipped)

    wanted <- unlist(lapply(targets, function(g) {
        paste0("samples/", g, "/",
               c("exon_counts.1pz", "intron_counts.1pz", "cell_calls.tsv"))
    }), use.names = FALSE)
    wanted <- intersect(wanted, members)
    if (length(wanted) > 0L) {
        utils::unzip(path, files = wanted, exdir = extract_dir)
    }

    gsm_meta_map <- if (!is.null(study_meta)) study_meta$gsm_meta else NULL

    mats <- list()
    spliced_mats <- list()
    unspliced_mats <- list()
    coldata_rows <- list()
    for (gsm in targets) {
        # One unreadable sample must not take the rest of the study with it.
        loaded <- tryCatch(
            .bundle_load_gsm(extract_dir, gsm, gene_ids),
            error = function(e) paste("unreadable:", conditionMessage(e))
        )
        if (is.character(loaded)) {
            skipped[[gsm]] <- loaded
            next
        }
        mat <- loaded$matrix
        bcs <- loaded$barcodes
        cell_names <- paste0(gsm, "_", bcs)
        colnames(mat) <- cell_names
        mats[[gsm]] <- mat
        colnames(loaded$spliced) <- cell_names
        colnames(loaded$unspliced) <- cell_names
        spliced_mats[[gsm]] <- loaded$spliced
        unspliced_mats[[gsm]] <- loaded$unspliced

        info <- if (!is.null(gsm_meta_map) && gsm %in% names(gsm_meta_map)) {
            gsm_meta_map[[gsm]]
        } else {
            list()
        }
        coldata_rows[[gsm]] <- .bundle_obs_frame(gsm, info, ncol(mat))
    }

    if (length(mats) == 0L) {
        detail <- if (length(skipped) > 0L) {
            paste0(": ", .describe_skipped(skipped))
        } else {
            ""
        }
        stop(sprintf("bundle %s contained no loadable cells%s", path, detail),
             call. = FALSE)
    }
    if (length(skipped) > 0L) {
        warning(sprintf(
            "%s: skipped %d of %d sample(s) with no usable cells: %s",
            basename(path), length(skipped), n_asked, .describe_skipped(skipped)
        ), call. = FALSE)
    }

    X <- do.call(cbind, unname(mats))
    X <- methods::as(X, "CsparseMatrix")
    spliced <- methods::as(do.call(cbind, unname(spliced_mats)), "CsparseMatrix")
    unspliced <- methods::as(do.call(cbind, unname(unspliced_mats)), "CsparseMatrix")
    coldata <- do.call(rbind, unname(coldata_rows))
    rownames(coldata) <- colnames(X)

    rowdata <- S4Vectors::DataFrame(gene_name = gene_names)
    rownames(rowdata) <- gene_ids

    sce <- SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = X, spliced = spliced, unspliced = unspliced),
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
    S4Vectors::metadata(sce)$skipped_samples <- data.frame(
        gsm_id = as.character(names(skipped)),
        reason = unname(skipped),
        stringsAsFactors = FALSE
    )

    sce
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
#' Samples with no usable cells (a missing or empty count matrix, or no
#' called cells) are skipped with a warning instead of failing the whole
#' study; they are listed in \code{metadata(sce)$skipped_samples}. The
#' barcode column of \code{cell_calls.tsv} may be named \code{barcode},
#' \code{cb}, \code{cell_barcode} or \code{CB}.
#'
#' @param path Path to a local `.singlet` file.
#' @return A \code{SingleCellExperiment} with one column per called cell
#'   (named \code{<GSM>_<barcode>}) and one row per gene. Assays are
#'   \code{counts} (exonic + intronic), \code{spliced} (exonic) and
#'   \code{unspliced} (intronic). \code{colData}
#'   carries per-sample metadata; \code{metadata(sce)} carries the bundle's
#'   parsed \code{manifest} and \code{study_meta}, and
#'   \code{skipped_samples}, a data frame (\code{gsm_id}, \code{reason}) of
#'   samples that had no usable cells.
#'
#' @examples
#' \dontrun{
#' path <- singlet_download("GSE138867")
#' sce <- read_singlet(path)
#' sce
#' table(sce$gsm_id)
#' S4Vectors::metadata(sce)$skipped_samples
#' }
#'
#' @seealso \code{\link{load}}, \code{\link{find}},
#'   \code{\link{singlet_modalities}}, \code{\link{singlet_read}}
#' @export
read_singlet <- function(path) {
    .read_singlet_bundle(path)
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
