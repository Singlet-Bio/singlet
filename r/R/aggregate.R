#' Aggregate a per-feature matrix to per-gene by summing exons of each gene.
#'
#' The singlet pipeline writes per-feature rownames in the format
#' \code{ENSG00000000003_TSPAN6_chrX:100627107-100629986} where the leading
#' Ensembl gene ID is everything before the first underscore. This helper
#' parses the gene ID, builds a gene-projection matrix, and returns a
#' per-gene aggregated \code{dgCMatrix}.
#'
#' Verified empirically: on a real pipeline output, summing all exons of
#' each gene in \code{exon_counts.1pz} reproduces \code{spliced.1pz}
#' bit-exactly (same shape, same nnz, same per-gene per-cell values).
#'
#' Use this to reconstruct \code{spliced} from \code{exon_counts}, or
#' \code{unspliced} from \code{intron_counts}, when the pipeline drops
#' those redundant matrices to save NFS storage.
#'
#' @param feature_mat A \code{Matrix::dgCMatrix} from \code{read_1pz()} on
#'   a per-feature file (\code{exon_counts.1pz} or
#'   \code{intron_counts.1pz}). Rows are features (exons or introns),
#'   columns are cells.
#' @return A \code{dgCMatrix} with rows aggregated to per-gene level. The
#'   row names are Ensembl gene IDs in the order they first appear in
#'   the input.
#'
#' @examples
#' \dontrun{
#' ex <- read_1pz("exon_counts.1pz")
#' spliced <- aggregate_features_to_gene(ex)
#' identical(dim(spliced), dim(read_1pz("spliced.1pz")))
#' }
#'
#' @export
aggregate_features_to_gene <- function(feature_mat) {
    if (!inherits(feature_mat, "dgCMatrix")) {
        stop("feature_mat must be a Matrix::dgCMatrix")
    }
    feature_rownames <- rownames(feature_mat)
    if (is.null(feature_rownames)) {
        stop("feature_mat must have rownames in the format ",
             "'ENSG..._GENE_chr:start-end'")
    }
    if (length(feature_rownames) != nrow(feature_mat)) {
        stop("rowname count does not match nrow(feature_mat)")
    }

    # Extract the gene ID prefix: everything before the first underscore.
    # Ensembl IDs (ENSG..., ENSMUSG..., etc.) never contain underscores.
    gene_ids <- sub("_.*$", "", feature_rownames)

    # Build the per-gene factor preserving order of first appearance.
    gene_factor <- factor(gene_ids, levels = unique(gene_ids))
    n_genes <- nlevels(gene_factor)
    n_features <- nrow(feature_mat)

    # Build a (n_genes × n_features) projection matrix where each entry
    # P[g, f] = 1 if feature f belongs to gene g. Then aggregated = P %*% mat.
    proj <- Matrix::sparseMatrix(
        i = as.integer(gene_factor),
        j = seq_len(n_features),
        x = rep(1.0, n_features),
        dims = c(n_genes, n_features)
    )
    aggregated <- proj %*% feature_mat

    # The result of dgCMatrix %*% dgCMatrix is dgCMatrix; ensure it.
    aggregated <- methods::as(aggregated, "CsparseMatrix")
    rownames(aggregated) <- levels(gene_factor)
    colnames(aggregated) <- colnames(feature_mat)

    # Preserve user_kv if it was attached to the input
    if (!is.null(attr(feature_mat, "user_kv"))) {
        attr(aggregated, "user_kv") <- attr(feature_mat, "user_kv")
    }
    aggregated
}
