# SPDX-License-Identifier: MIT
# Multi-modality access for .singlet bundles.
#
# A .singlet bundle carries far more than gene counts. Alongside the exon
# and intron count matrices, the pipeline emits splice junctions and PSI,
# mitochondrial heteroplasmy and variant calls, genotype-free donor
# demultiplexing, non-host (microbial/viral) abundance, V(D)J segment
# usage, and a stack of per-cell annotation tables.
#
# `read_singlet()` collapses all of that into gene-level counts because
# that is what most analyses need. The functions here let you reach the
# rest. They mirror the Python API on `singlet.SingletBundle`:
#
#   singlet_files()       <->  bundle.list_files()
#   singlet_modalities()  <->  bundle.modalities()
#   singlet_has()         <->  bundle.has()
#   singlet_read()        <->  bundle.read()
#   singlet_raw_counts()  <->  bundle.raw_counts()


# ---------------------------------------------------------------------------
# Modality registry
# ---------------------------------------------------------------------------

.modality <- function(kind, members, description) {
    list(kind = kind, members = members, description = description)
}

#' Modalities a `.singlet` bundle can contain
#'
#' The registry of addressable per-sample outputs, keyed by the short name
#' accepted by \code{\link{singlet_read}}. Each entry records the reader
#' \code{kind} (\code{"matrix"}, \code{"table"}, \code{"json"} or
#' \code{"text"}), the candidate archive paths inside
#' \code{samples/<GSM>/} in priority order (bundle layouts changed across
#' pipeline versions), and a one-line description.
#'
#' Not every bundle has every modality. Older bundles carry the counts,
#' splicing, heteroplasmy and V(D)J matrices only; the donor, non-host and
#' per-cell annotation outputs appear in newer ones. Always check with
#' \code{\link{singlet_modalities}} before reading.
#'
#' @format A named list of lists.
#' @seealso \code{\link{singlet_modalities}}, \code{\link{singlet_read}}
#' @export
SINGLET_MODALITIES <- list(
    # -- Counts ------------------------------------------------------------
    exon_counts = .modality(
        "matrix", "exon_counts.1pz",
        "Per-exon-feature UMI counts (the spliced half of the raw matrix)."),
    intron_counts = .modality(
        "matrix", "intron_counts.1pz",
        "Per-intron-feature UMI counts (the unspliced half of the raw matrix)."),
    cell_calls = .modality(
        "table", "cell_calls.tsv",
        "Barcode, is_cell and the cell-calling statistics per barcode."),
    cell_qc = .modality(
        "table", "cell_qc_metrics.tsv",
        "Per-cell QC metrics (UMIs, genes, mitochondrial fraction, ...)."),
    # -- Splicing ----------------------------------------------------------
    junctions = .modality(
        "matrix", "sj_counts.1pz",
        "Per-cell splice-junction counts."),
    splice_psi = .modality(
        "matrix", "splice_psi.1pz",
        "Per-cell percent-spliced-in (PSI) per splice event."),
    splice_events = .modality(
        "table", "splice_events.tsv",
        "Splice-event annotation for the rows of splice_psi."),
    # -- Mitochondrial genome ----------------------------------------------
    mt_heteroplasmy = .modality(
        "matrix", "mt_heteroplasmy.1pz",
        "Per-cell mitochondrial heteroplasmy (VAF) per chrM variant site."),
    mt_variants = .modality(
        "table", c("mt_variants.tsv", "mt/mt_summary.tsv"),
        "Called chrM variants with depth, allele counts and annotation."),
    mt_events = .modality(
        "matrix", "mt/mt_events.1pz",
        "Per-cell chrM allele-support matrix used for lineage tracing."),
    # -- Donor / genotype --------------------------------------------------
    donor_assignments = .modality(
        "table", c("donor/donor_assignments.tsv", "donor_assignments.tsv"),
        "Genotype-free donor demultiplexing: barcode to donor, with doublet calls."),
    donor_snp_ad = .modality(
        "matrix", "donor/snp_ad.1pz",
        "Per-cell alternate-allele depth over the SNP panel."),
    donor_snp_dp = .modality(
        "matrix", "donor/snp_dp.1pz",
        "Per-cell total read depth over the SNP panel (denominator for snp_ad)."),
    ase_counts = .modality(
        "table", "ase_counts.tsv",
        "Allele-specific expression counts per gene."),
    ancestry_call = .modality(
        "json", "ancestry_call.json",
        "Continental ancestry estimate from the SNP panel."),
    sex_call = .modality(
        "json", "sex_call.json",
        "Genetic sex call from chrX/chrY expression and coverage."),
    # -- Non-host ----------------------------------------------------------
    nonhost_species = .modality(
        "table", c("nonhost/nonhost_em_abundance.tsv", "nonhost_em_abundance.tsv"),
        "Per-taxon non-host (microbial/viral) abundance after EM re-assignment."),
    nonhost_summary = .modality(
        "json", c("nonhost/nonhost_summary.json", "nonhost_summary.json"),
        "Non-host classification summary: reads classified, top taxa, database."),
    # -- Immune repertoire -------------------------------------------------
    vdj_gene_usage = .modality(
        "matrix", "vdj_gene_usage.1pz",
        "Per-cell V(D)J segment usage counts."),
    # -- Per-cell annotations ----------------------------------------------
    doublet_scores = .modality(
        "table", "doublet_scores.tsv",
        "Per-cell doublet score and call."),
    cell_cycle_scores = .modality(
        "table", "cell_cycle_scores.tsv",
        "Per-cell S/G2M scores and phase assignment."),
    ambient_contamination = .modality(
        "table", "ambient_contamination.tsv",
        "Per-cell ambient-RNA contamination fraction."),
    ambient_profile = .modality(
        "table", "ambient_profile.tsv",
        "Ambient-RNA expression profile estimated from empty droplets."),
    # -- Sample-level QC ---------------------------------------------------
    summary = .modality(
        "json", "summary.json",
        "Sample-level metrics: cells called, mapping rate, medians, reference build."),
    pileup_stats = .modality(
        "json", "pileup_stats.json",
        "Alignment/pileup statistics for the sample."),
    provenance = .modality(
        "json", "provenance.json",
        "Pipeline version, command line, reference checksums."),
    saturation_curve = .modality(
        "table", "saturation_curve.tsv",
        "Sequencing-saturation curve (downsampled read depth vs genes detected)."),
    star_log = .modality(
        "text", "star_Log.final.out",
        "STAR final alignment log for the sample.")
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.bundle_path <- function(path) {
    path <- path.expand(as.character(path))
    if (!file.exists(path)) {
        stop(sprintf("no such .singlet file: %s", path))
    }
    path
}

# Every member path in the archive, cached per file+mtime so repeated calls
# on the same bundle do not re-read the central directory.
.bundle_members_cache <- new.env(parent = emptyenv())

.bundle_members <- function(path) {
    key <- paste0(path, "@", as.numeric(file.info(path)$mtime))
    hit <- .bundle_members_cache[[key]]
    if (!is.null(hit)) {
        return(hit)
    }
    names <- utils::unzip(path, list = TRUE)$Name
    assign(key, names, envir = .bundle_members_cache)
    names
}

# Reader kind for a literal member that is not in the registry.
.bundle_infer_kind <- function(member) {
    if (grepl("\\.1pz$", member)) return("matrix")
    if (grepl("\\.json$", member)) return("json")
    if (grepl("\\.(tsv|csv)$", member)) return("table")
    "text"
}

# Map a modality name (or a literal member path) to an archive member.
# Returns NULL when the sample does not have it.
.bundle_resolve <- function(path, gsm, name) {
    available <- singlet_files(path, gsm)
    candidates <- if (!is.null(SINGLET_MODALITIES[[name]])) {
        SINGLET_MODALITIES[[name]]$members
    } else {
        name
    }
    for (cand in candidates) {
        if (cand %in% available) {
            return(paste0("samples/", gsm, "/", cand))
        }
    }
    NULL
}

# Extract a single member to a temporary directory and return its path.
.bundle_extract <- function(path, member, exdir) {
    utils::unzip(path, files = member, exdir = exdir, junkpaths = FALSE)
    file.path(exdir, member)
}

.bundle_gsm_ids <- function(path) {
    members <- .bundle_members(path)
    sample_members <- grep("^samples/[^/]+/", members, value = TRUE)
    unique(sub("^samples/([^/]+)/.*$", "\\1", sample_members))
}

.bundle_default_gsm <- function(path, gsm) {
    if (!is.null(gsm)) {
        return(as.character(gsm))
    }
    ids <- .bundle_gsm_ids(path)
    if (length(ids) == 0L) {
        stop(sprintf("bundle %s contains no samples", basename(path)))
    }
    ids[[1L]]
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#' List the files a `.singlet` bundle holds for a sample
#'
#' @param path Path to a local `.singlet` file.
#' @param gsm GEO sample accession. When \code{NULL} (default), every
#'   member path in the archive is returned; otherwise paths are returned
#'   relative to \code{samples/<gsm>/} (e.g. \code{"exon_counts.1pz"},
#'   \code{"donor/donor_assignments.tsv"}).
#' @return A character vector of member paths.
#'
#' @examples
#' \dontrun{
#' singlet_files("GSE149298.singlet", "GSM4495571")
#' }
#'
#' @seealso \code{\link{singlet_modalities}}, \code{\link{singlet_read}}
#' @export
singlet_files <- function(path, gsm = NULL) {
    path <- .bundle_path(path)
    members <- .bundle_members(path)
    if (is.null(gsm)) {
        return(members)
    }
    prefix <- paste0("samples/", as.character(gsm), "/")
    hits <- members[startsWith(members, prefix) & !endsWith(members, "/")]
    substring(hits, nchar(prefix) + 1L)
}


#' Which modalities are available for a sample
#'
#' @param path Path to a local `.singlet` file.
#' @param gsm GEO sample accession. Defaults to the first sample in the
#'   bundle, which is representative because every sample of a study is
#'   processed with the same pipeline invocation.
#' @return A named character vector: names are modality keys accepted by
#'   \code{\link{singlet_read}}, values are their descriptions.
#'
#' @examples
#' \dontrun{
#' singlet_modalities("GSE149298.singlet")
#' }
#'
#' @seealso \code{\link{SINGLET_MODALITIES}}, \code{\link{singlet_read}}
#' @export
singlet_modalities <- function(path, gsm = NULL) {
    path <- .bundle_path(path)
    target <- .bundle_default_gsm(path, gsm)
    present <- vapply(
        names(SINGLET_MODALITIES),
        function(nm) !is.null(.bundle_resolve(path, target, nm)),
        logical(1)
    )
    found <- names(SINGLET_MODALITIES)[present]
    out <- vapply(found, function(nm) SINGLET_MODALITIES[[nm]]$description, character(1))
    names(out) <- found
    out
}


#' Is a modality present for a sample?
#'
#' @param path Path to a local `.singlet` file.
#' @param name A modality key (see \code{\link{SINGLET_MODALITIES}}) or a
#'   literal file path inside \code{samples/<gsm>/}.
#' @param gsm GEO sample accession. When \code{NULL} (default), the check
#'   passes if \emph{any} sample in the bundle has the modality.
#' @return \code{TRUE} or \code{FALSE}.
#'
#' @examples
#' \dontrun{
#' singlet_has("GSE149298.singlet", "donor_assignments")
#' }
#'
#' @export
singlet_has <- function(path, name, gsm = NULL) {
    path <- .bundle_path(path)
    targets <- if (is.null(gsm)) .bundle_gsm_ids(path) else as.character(gsm)
    for (g in targets) {
        if (!is.null(.bundle_resolve(path, g, name))) {
            return(TRUE)
        }
    }
    FALSE
}


#' Read one modality for one sample
#'
#' Reads any single output the pipeline produced for a sample: a sparse
#' matrix, a table, a JSON sidecar or a log file. Use
#' \code{\link{singlet_modalities}} first to see what a given bundle has.
#'
#' @param path Path to a local `.singlet` file.
#' @param gsm GEO sample accession.
#' @param name A modality key (see \code{\link{SINGLET_MODALITIES}}) or a
#'   literal file path inside \code{samples/<gsm>/}.
#' @return Depends on the modality kind:
#'   \describe{
#'     \item{matrix}{A features x cells \code{dgCMatrix}, with feature ids
#'       as rownames and cell barcodes as colnames. Note this is the
#'       Bioconductor orientation, the transpose of what the Python
#'       package returns.}
#'     \item{table}{A \code{data.frame}.}
#'     \item{json}{A named list.}
#'     \item{text}{A single character string.}
#'   }
#'
#' @examples
#' \dontrun{
#' het <- singlet_read("GSE149298.singlet", "GSM4495571", "mt_heteroplasmy")
#' donors <- singlet_read("GSE149298.singlet", "GSM4495571", "donor_assignments")
#' bugs <- singlet_read("GSE149298.singlet", "GSM4495571", "nonhost_species")
#' }
#'
#' @seealso \code{\link{singlet_modalities}}, \code{\link{singlet_raw_counts}}
#' @export
singlet_read <- function(path, gsm, name) {
    path <- .bundle_path(path)
    gsm <- as.character(gsm)
    member <- .bundle_resolve(path, gsm, name)
    if (is.null(member)) {
        stop(sprintf(
            "'%s' is not present for %s in %s. Available: %s",
            name, gsm, basename(path),
            paste(names(singlet_modalities(path, gsm)), collapse = ", ")
        ))
    }
    kind <- if (!is.null(SINGLET_MODALITIES[[name]])) {
        SINGLET_MODALITIES[[name]]$kind
    } else {
        .bundle_infer_kind(member)
    }

    exdir <- tempfile("singlet_member_")
    dir.create(exdir)
    on.exit(unlink(exdir, recursive = TRUE, force = TRUE), add = TRUE)
    local_path <- .bundle_extract(path, member, exdir)

    switch(
        kind,
        matrix = read_1pz(local_path),
        table = utils::read.table(local_path, sep = "\t", header = TRUE,
                                  stringsAsFactors = FALSE, check.names = FALSE),
        json = jsonlite::fromJSON(local_path, simplifyVector = FALSE),
        text = paste(readLines(local_path, warn = FALSE), collapse = "\n"),
        stop(sprintf("unknown modality kind '%s' for '%s'", kind, name))
    )
}


#' Raw counts for one sample, with spliced and unspliced assays
#'
#' The matrix a conventional (non-USA) pipeline would hand you: the
#' \code{counts} assay is exonic + intronic UMIs, and the two halves are
#' kept as the \code{spliced} and \code{unspliced} assays so nothing is
#' lost. This is the single-sample counterpart to
#' \code{\link{read_singlet}}, which does the same for a whole study.
#'
#' @param path Path to a local `.singlet` file.
#' @param gsm GEO sample accession.
#' @param gene_level \code{TRUE} (default) sums the per-feature counts onto
#'   the bundle's canonical gene axis, so the matrix is comparable across
#'   samples and studies. \code{FALSE} keeps the pipeline's native
#'   per-feature rows (individual exons and introns), which is what you
#'   want for feature-resolved work; \code{rowData} then carries
#'   \code{feature_kind} and \code{gene_id}.
#' @param cells \code{"called"} (default) keeps only barcodes the pipeline
#'   called as cells. \code{"all"} keeps every barcode including empty
#'   droplets, the equivalent of a \code{raw_feature_bc_matrix}. Only
#'   meaningful with \code{gene_level = FALSE}; the gene-level path is
#'   always restricted to called cells.
#' @return A \code{SingleCellExperiment} with \code{counts},
#'   \code{spliced} and \code{unspliced} assays.
#'
#' @examples
#' \dontrun{
#' sce <- singlet_raw_counts("GSE149298.singlet", "GSM4495571")
#' assayNames(sce)
#' identical(sum(assay(sce, "counts")),
#'           sum(assay(sce, "spliced")) + sum(assay(sce, "unspliced")))
#'
#' # keep the native exon/intron feature axis
#' feat <- singlet_raw_counts("GSE149298.singlet", "GSM4495571", gene_level = FALSE)
#' table(rowData(feat)$feature_kind)
#' }
#'
#' @seealso \code{\link{read_singlet}}, \code{\link{singlet_read}}
#' @export
singlet_raw_counts <- function(path, gsm, gene_level = TRUE, cells = c("called", "all")) {
    if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
        stop("singlet_raw_counts requires the SingleCellExperiment package. ",
             "Install with `BiocManager::install('SingleCellExperiment')`.")
    }
    cells <- match.arg(cells)
    path <- .bundle_path(path)
    gsm <- as.character(gsm)

    exon <- if (singlet_has(path, "exon_counts", gsm)) singlet_read(path, gsm, "exon_counts") else NULL
    intron <- if (singlet_has(path, "intron_counts", gsm)) singlet_read(path, gsm, "intron_counts") else NULL
    if (is.null(exon) && is.null(intron)) {
        stop(sprintf("%s has no count matrices in %s", gsm, basename(path)))
    }

    called <- .bundle_called_barcodes(path, gsm)

    if (gene_level) {
        vocab <- .bundle_feature_vocab(path)
        gene_ids <- vapply(vocab$genes, function(g) g$gene_id, character(1))
        gene_names <- vapply(vocab$genes, function(g) g$gene_name, character(1))

        spliced <- .bundle_aggregate_to_vocab(exon, gene_ids)
        unspliced <- .bundle_aggregate_to_vocab(intron, gene_ids)
        barcodes <- colnames(if (!is.null(exon)) exon else intron)
        spliced <- .bundle_zero_like(spliced, length(gene_ids), barcodes)
        unspliced <- .bundle_zero_like(unspliced, length(gene_ids), barcodes)
        rownames(spliced) <- rownames(unspliced) <- gene_ids
        colnames(spliced) <- colnames(unspliced) <- barcodes

        keep <- .bundle_keep_cells(barcodes, called, "called", gsm, path)
        spliced <- spliced[, keep, drop = FALSE]
        unspliced <- unspliced[, keep, drop = FALSE]
        rowdata <- S4Vectors::DataFrame(gene_name = gene_names, row.names = gene_ids)
    } else {
        parts <- list(exon = exon, intron = intron)
        parts <- parts[!vapply(parts, is.null, logical(1))]
        barcodes <- colnames(parts[[1L]])
        for (p in parts) {
            barcodes <- intersect(barcodes, colnames(p))
        }
        if (length(barcodes) == 0L) {
            stop("exon and intron matrices share no barcodes")
        }
        feature_names <- unlist(lapply(parts, rownames), use.names = FALSE)
        feature_kind <- rep(names(parts), vapply(parts, nrow, integer(1)))

        # Each block occupies a contiguous row range, so an assay is that
        # block padded with all-zero rows.
        block <- function(which) {
            blocks <- lapply(names(parts), function(k) {
                m <- parts[[k]][, barcodes, drop = FALSE]
                if (k == which) m else Matrix::sparseMatrix(
                    i = integer(0), j = integer(0), x = numeric(0),
                    dims = dim(m))
            })
            out <- if (length(blocks) > 1L) do.call(rbind, blocks) else blocks[[1L]]
            out <- methods::as(out, "CsparseMatrix")
            rownames(out) <- feature_names
            colnames(out) <- barcodes
            out
        }
        spliced <- block("exon")
        unspliced <- block("intron")

        keep <- .bundle_keep_cells(barcodes, called, cells, gsm, path)
        spliced <- spliced[, keep, drop = FALSE]
        unspliced <- unspliced[, keep, drop = FALSE]
        rowdata <- S4Vectors::DataFrame(
            feature_kind = feature_kind,
            gene_id = sub("_.*$", "", feature_names),
            row.names = feature_names
        )
    }

    counts <- methods::as(spliced + unspliced, "CsparseMatrix")
    sce <- SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = counts, spliced = spliced, unspliced = unspliced),
        colData = S4Vectors::DataFrame(
            gsm_id = rep(gsm, ncol(counts)),
            row.names = colnames(counts)
        ),
        rowData = rowdata
    )
    vocab <- .bundle_feature_vocab(path)
    S4Vectors::metadata(sce)$reference_build <-
        if (is.null(vocab$reference_build)) NA_character_ else vocab$reference_build
    S4Vectors::metadata(sce)$singlet_bundle_path <- path
    sce
}


# Barcodes the pipeline called as cells, or NULL when not recorded.
.bundle_called_barcodes <- function(path, gsm) {
    if (!singlet_has(path, "cell_calls", gsm)) {
        return(NULL)
    }
    cc <- tryCatch(singlet_read(path, gsm, "cell_calls"), error = function(e) NULL)
    if (is.null(cc) || nrow(cc) == 0L) {
        return(NULL)
    }
    col <- intersect(c("barcode", "cb", "cell_barcode"), colnames(cc))
    if (length(col) == 0L) {
        return(NULL)
    }
    bc <- as.character(cc[[col[[1L]]]])
    if ("is_cell" %in% colnames(cc)) {
        is_cell <- as.logical(cc$is_cell)
        is_cell[is.na(is_cell)] <- FALSE
        bc <- bc[is_cell]
    }
    bc
}

.bundle_keep_cells <- function(barcodes, called, cells, gsm, path) {
    if (cells == "all" || is.null(called) || length(called) == 0L) {
        return(barcodes)
    }
    keep <- intersect(barcodes, called)
    if (length(keep) == 0L) {
        stop(sprintf("%s has no called cells in %s", gsm, basename(path)))
    }
    keep
}

.bundle_zero_like <- function(mat, n_rows, barcodes) {
    if (!is.null(mat)) {
        return(mat)
    }
    Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                         dims = c(n_rows, length(barcodes)))
}

.bundle_feature_vocab <- function(path) {
    exdir <- tempfile("singlet_vocab_")
    dir.create(exdir)
    on.exit(unlink(exdir, recursive = TRUE, force = TRUE), add = TRUE)
    local_path <- .bundle_extract(path, "feature_vocab.json", exdir)
    if (!file.exists(local_path)) {
        stop(sprintf("malformed bundle (no feature_vocab.json): %s", path))
    }
    jsonlite::fromJSON(local_path, simplifyVector = FALSE)
}
