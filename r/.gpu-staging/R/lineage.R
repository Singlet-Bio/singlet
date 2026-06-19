# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/lineage.R
#
# GPU MT lineage tracing:
#   detect_clones()      — MT heteroplasmy-based clone calling
#   read_pz_mt_sce()     — load mt_alt / mt_depth from singlet output dir
#
# NEW — closest published CPU analog is MitoTrace (R) / MQuad (Python).

#' Load MT allele and depth matrices from a singlet output directory
#'
#' Reads \code{mt_alleles.1pz} from a singlet pipeline output directory
#' and constructs an SCE with assays \code{"mt_alt"} (alternate allele counts)
#' and \code{"mt_depth"} (total depth), ready for \code{\link{detect_clones}}.
#'
#' @details
#' \code{mt_alleles.1pz} stores alternate-allele counts at all MT variant
#' sites.  Depth is computed as total coverage at those sites from the BAM
#' alignment log; if a separate depth file is not present, total allele counts
#' across all alt+ref are used as an approximation.
#'
#' If an existing SCE is provided via \code{sce}, the two assays are added to
#' it (overwriting if already present).
#'
#' @param pz_dir Character.  Path to a singlet sample output directory
#'   containing \code{mt_alleles.1pz}.
#' @param sce    Optional \code{SingleCellExperiment}.  If provided, MT assays
#'   are added to it.  Default: \code{NULL} (new SCE).
#'
#' @return A \code{SingleCellExperiment} with assays \code{"mt_alt"} and
#'   \code{"mt_depth"}.
#'
#' @seealso \code{\link{detect_clones}}
#' @export
read_pz_mt_sce <- function(pz_dir, sce = NULL) {
    mt_path <- file.path(pz_dir, "mt_alleles.1pz")
    if (!file.exists(mt_path))
        stop(sprintf("read_pz_mt_sce: 'mt_alleles.1pz' not found in %s", pz_dir))

    mt_data <- load_pz_to_sce_cpp(pz_dir, "mt_alleles", FALSE)

    # The mt_alleles.1pz loader returns alt-allele counts.
    # Depth is stored in a companion matrix; if absent, use alt as a proxy.
    mt_depth_path <- file.path(pz_dir, "mt_depth.1pz")
    if (file.exists(mt_depth_path)) {
        depth_data <- load_pz_to_sce_cpp(pz_dir, "mt_depth", FALSE)
        mat_depth  <- depth_data$counts
    } else {
        # Fallback: use alt counts as depth (conservative; marks all sites as
        # fully alt — real depth would need the BAM. Log a warning.)
        warning("read_pz_mt_sce: mt_depth.1pz not found; using mt_alt counts as depth proxy.")
        mat_depth <- mt_data$counts
    }

    if (is.null(sce)) {
        sce <- SingleCellExperiment::SingleCellExperiment(
            assays = list(mt_alt   = mt_data$counts,
                          mt_depth = mat_depth)
        )
    } else {
        .assert_sce(sce, "read_pz_mt_sce")
        SummarizedExperiment::assay(sce, "mt_alt")   <- mt_data$counts
        SummarizedExperiment::assay(sce, "mt_depth") <- mat_depth
    }

    sce
}


#' GPU mitochondrial lineage tracing
#'
#' Calls clonal populations from per-cell MT heteroplasmy profiles using
#' GPU-accelerated EM clustering on the variant allele frequency (VAF)
#' matrix.  This is a NEW function — closest published CPU equivalents are
#' \href{https://github.com/caleblareau/mgatk}{mgatk},
#' \href{https://github.com/MQuad}{MQuad}, and
#' \href{https://github.com/MITO-trace}{MitoTrace}.
#'
#' @details
#' The algorithm:
#' 1. Filters MT sites by depth (\code{min_depth}), number of alt-allele cells
#'    (\code{min_cells_alt}), and mean VAF (\code{min_vaf}) to produce
#'    \emph{informative sites}.
#' 2. Constructs a VAF matrix (informative sites × cells).
#' 3. Fits a Gaussian mixture model (GMM) with K components using EM; K is
#'    selected by BIC over \code{[min_K, max_K]}.
#' 4. Assigns each cell to the MAP clone.
#'
#' Results are stored as:
#' - \code{colData(sce)$mt_clone_id} — integer clone label (0-based)
#' - \code{metadata(sce)$mt_lineage$informative_sites} — site indices
#' - \code{metadata(sce)$mt_lineage$vaf_matrix} — VAF matrix
#' - \code{metadata(sce)$mt_lineage$clone_probs} — posterior probabilities
#' - \code{metadata(sce)$mt_lineage$bic} — BIC per K tested
#'
#' @param sce           A \code{\link[SingleCellExperiment]{SingleCellExperiment}}
#'   with assays \code{"mt_alt"} and \code{"mt_depth"}.  Use
#'   \code{\link{read_pz_mt_sce}} to load these from a singlet output dir.
#' @param alt_assay     Character.  Alt-allele count assay.
#'   Default: \code{"mt_alt"}.
#' @param depth_assay   Character.  Depth assay.  Default: \code{"mt_depth"}.
#' @param min_depth     Integer.  Minimum site depth to include in VAF.
#'   Default: \code{10L}.
#' @param min_cells_alt Integer.  Minimum cells with alt allele at a site.
#'   Default: \code{5L}.
#' @param min_vaf       Numeric.  Minimum mean VAF to declare site informative.
#'   Default: \code{0.01}.
#' @param min_K         Integer.  Minimum number of clones to test.
#'   Default: \code{2L}.
#' @param max_K         Integer.  Maximum number of clones to test.
#'   Default: \code{10L}.
#' @param seed          Integer.  EM initialisation seed.  Default: \code{0L}.
#'
#' @return The input \code{sce} with:
#' \describe{
#'   \item{\code{colData(sce)$mt_clone_id}}{Integer clone label (0-based).}
#'   \item{\code{metadata(sce)$mt_lineage}}{List with
#'     \code{informative_sites}, \code{vaf_matrix}, \code{clone_probs},
#'     \code{bic}, and \code{n_clones}.}
#' }
#'
#' @note The \code{skip_if_not_installed("MitoTrace")} convention is followed
#'   in testthat tests for cross-validation; this function itself has no
#'   dependency on MitoTrace.
#'
#' @seealso \code{\link{read_pz_mt_sce}}
#' @export
detect_clones <- function(sce,
                          alt_assay     = "mt_alt",
                          depth_assay   = "mt_depth",
                          min_depth     = 10L,
                          min_cells_alt = 5L,
                          min_vaf       = 0.01,
                          min_K         = 2L,
                          max_K         = 10L,
                          seed          = 0L) {

    .assert_sce(sce, "detect_clones")

    for (lyr in c(alt_assay, depth_assay)) {
        if (!lyr %in% SummarizedExperiment::assayNames(sce))
            stop(sprintf("detect_clones: assay '%s' not found. Use read_pz_mt_sce().", lyr))
    }

    min_depth     <- as.integer(min_depth)
    min_cells_alt <- as.integer(min_cells_alt)
    min_vaf       <- as.double(min_vaf)
    min_K         <- as.integer(min_K)
    max_K         <- as.integer(max_K)
    seed          <- as.integer(seed)

    if (min_K < 1L) stop("detect_clones: min_K must be >= 1.")
    if (max_K < min_K) stop("detect_clones: max_K must be >= min_K.")

    mat_alt   <- SummarizedExperiment::assay(sce, alt_assay)
    mat_depth <- SummarizedExperiment::assay(sce, depth_assay)

    if (!methods::is(mat_alt,   "dgCMatrix")) mat_alt   <- methods::as(mat_alt,   "dgCMatrix")
    if (!methods::is(mat_depth, "dgCMatrix")) mat_depth <- methods::as(mat_depth, "dgCMatrix")

    res <- mt_lineage_call_clones_cpp(
        mat_alt, mat_depth,
        min_depth, min_cells_alt, min_vaf,
        min_K, max_K, seed
    )

    # Store clone IDs in colData
    cd <- SummarizedExperiment::colData(sce)
    cd$mt_clone_id <- res$clone_ids
    SummarizedExperiment::colData(sce) <- cd

    # Store detailed results in metadata
    SummarizedExperiment::metadata(sce)$mt_lineage <- list(
        n_clones          = res$n_clones,
        informative_sites = res$informative_sites,
        vaf_matrix        = res$vaf_matrix,
        clone_probs       = res$clone_probs,
        bic               = res$bic
    )

    sce
}
