# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/load_pz.R
#
# read_1pz_sce() — load a singlet .1pz count matrix into a SingleCellExperiment.

#' Load a singlet .1pz directory into a SingleCellExperiment
#'
#' Reads a singlet `.1pz` count matrix from disk, decompresses it on the CPU,
#' uploads it to the GPU, copies the result back to host as a
#' \code{Matrix::dgCMatrix}, and wraps it in a
#' \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#'
#' @details
#' **Host-copy cost**: R does not have a native GPU sparse type, so every call
#' incurs two O(nnz) host copies (host→device for potential preprocessing,
#' device→host for the returned SCE).  For a typical 10k-cell sample this is
#' approximately 100–200 ms on a PCIe 16x Gen3 system.  Users who require full
#' GPU residency between multiple kernel calls should use the Python wrapper.
#'
#' The embedded GEO metadata (gsm_id, gse_id, organism, protocol, …) is stored
#' in \code{metadata(sce)$singlet} as a named list.
#'
#' @param pz_dir   Character scalar. Path to the sample output directory
#'   containing one or more `.1pz` files (e.g.
#'   \code{"/path/to/quant/scrna/GSE0/GSE123/GSM456/"}).
#' @param modality Character scalar. Which count matrix to load:
#'   \describe{
#'     \item{\code{"exon"}}{Spliced (exon) UMI count matrix — primary output.}
#'     \item{\code{"intron"}}{Unspliced (intron) count matrix — RNA velocity.}
#'     \item{\code{"gene"}}{GeneFull = exon + intron (STARsolo-compatible).}
#'     \item{\code{"adt"}}{Antibody-derived tag counts (CITE-seq).}
#'     \item{\code{"atac"}}{ATAC fragment counts (Multiome / standalone ATAC).}
#'   }
#'   Default: \code{"exon"}.
#' @param keep_host_pinned Logical. If \code{TRUE}, the loader retains pinned
#'   host copies of the CSC buffers inside the returned C++ object, which
#'   speeds up subsequent NMF calls (factornet reads host pointers directly).
#'   Doubles memory use.  Default: \code{FALSE}.
#'
#' @return A \code{\link[SingleCellExperiment]{SingleCellExperiment}} with:
#'   \describe{
#'     \item{\code{assays(sce)$counts}}{The count matrix as a
#'       \code{Matrix::dgCMatrix} (genes × cells).}
#'     \item{\code{metadata(sce)$singlet}}{Named list of GEO metadata fields:
#'       \code{gsm_id}, \code{gse_id}, \code{organism}, \code{taxon_id},
#'       \code{protocol}, \code{modality}, \code{srr_ids}, \code{read_count},
#'       \code{geo_title}, \code{geo_source_name}, \code{singlet_version},
#'       \code{pipeline_date}.}
#'   }
#'   Row names are gene names (from the .1pz metadata TLV); column names are
#'   barcode strings.
#'
#' @examples
#' \dontrun{
#' sce <- read_1pz_sce("/path/to/GSM4037629/", modality = "exon")
#' dim(sce)
#' metadata(sce)$singlet$gsm_id
#' }
#'
#' @seealso \code{\link{lognorm}}, \code{\link{hvg}}, \code{\link{run_pca}}
#' @export
read_1pz_sce <- function(pz_dir,
                          modality         = c("exon", "intron", "gene",
                                               "adt", "atac"),
                          keep_host_pinned = FALSE) {

    modality <- match.arg(modality)

    if (!is.character(pz_dir) || length(pz_dir) != 1L) {
        stop("read_1pz_sce: `pz_dir` must be a single character string.")
    }
    pz_dir <- normalizePath(pz_dir, mustWork = FALSE)

    if (!dir.exists(pz_dir)) {
        stop(sprintf("read_1pz_sce: directory does not exist: %s", pz_dir))
    }

    # Delegate to Rcpp binding
    raw <- load_pz_to_sce_cpp(pz_dir,
                               modality,
                               isTRUE(keep_host_pinned))

    counts  <- raw[["counts"]]   # Matrix::dgCMatrix
    meta    <- raw[["metadata"]] # named list
    cells   <- raw[["cells"]]    # CharacterVector (may be empty)
    genes   <- raw[["genes"]]    # CharacterVector (may be empty)

    # Set dimnames on the count matrix
    if (length(genes) == nrow(counts)) {
        rownames(counts) <- genes
    }
    if (length(cells) == ncol(counts)) {
        colnames(counts) <- cells
    }

    # Build SingleCellExperiment
    sce <- SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = counts)
    )

    # Embed singlet GEO metadata
    SummarizedExperiment::metadata(sce)$singlet <- meta

    sce
}
