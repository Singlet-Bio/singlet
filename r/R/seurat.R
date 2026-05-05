#' Convert a singlify pipeline directory to a Seurat object.
#'
#' Builds a \code{\link[Seurat:Seurat]{Seurat}} object from a singlify
#' pipeline output directory. The primary per-gene matrix becomes the
#' default ``RNA`` assay; the velocity trio
#' (``spliced`` / ``unspliced`` / ``ambiguous``) become additional assays
#' when present, named matching the scvelo convention so velocity
#' analyses work without extra renaming.
#'
#' Per-cell TSV sidecars are loaded into ``obj@meta.data``. The embedded
#' GEO metadata goes into ``obj@misc$singlify``.
#'
#' @param path Directory containing the pipeline's ``.1pz`` + TSV files.
#' @param primary_assay Matrix name for the default ``RNA`` assay. Tried
#'   in order: user value, ``"spliced"``, ``"gene_counts"``, ``"gene_counts_em"``.
#' @param project Project name attached to the Seurat object. Defaults to
#'   the ``gsm_id`` from the embedded pipeline metadata.
#' @return A \code{Seurat} object.
#'
#' @examples
#' \dontrun{
#' obj <- as_seurat("quant/scrna/GSE174/GSE174399/GSM5293863")
#' obj
#' obj@misc$singlify[["gsm_id"]]
#' }
#'
#' @export
as_seurat <- function(path, primary_assay = "spliced", project = NULL) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
        stop("as_seurat requires the Seurat package. ",
             "Install with `install.packages('Seurat')`.")
    }

    dd <- read_singlify_dir(path)

    per_gene <- c(primary_assay, "spliced", "gene_counts", "gene_counts_em")
    chosen <- NULL
    for (name in per_gene) {
        if (name %in% names(dd)) { chosen <- name; break }
    }
    if (is.null(chosen)) {
        stop("no per-gene count matrix found in ", path)
    }

    X <- dd[[chosen]]
    user_kv <- attr(dd, "user_kv")
    if (is.null(project)) {
        project <- if (!is.null(user_kv) && "gsm_id" %in% names(user_kv))
                       user_kv[["gsm_id"]] else "singlify"
    }

    obj <- Seurat::CreateSeuratObject(
        counts = X,
        project = project,
        assay = "RNA",
        min.cells = 0,
        min.features = 0
    )
    derived_assays <- character(0)

    # Velocity assays (scvelo-compatible). If the per-gene velocity matrix
    # is missing but the per-feature counterpart is present, reconstruct
    # it via aggregate_features_to_gene.
    derive_from <- list(spliced = "exon_counts", unspliced = "intron_counts")
    for (nm in c("spliced", "unspliced", "ambiguous")) {
        if (nm == chosen) next
        if (nm %in% names(dd) && all(dim(dd[[nm]]) == dim(X))) {
            obj[[nm]] <- Seurat::CreateAssayObject(counts = dd[[nm]])
        } else if (nm %in% names(derive_from)) {
            src <- derive_from[[nm]]
            if (src %in% names(dd)) {
                agg <- aggregate_features_to_gene(dd[[src]])
                if (all(dim(agg) == dim(X))) {
                    obj[[nm]] <- Seurat::CreateAssayObject(counts = agg)
                    derived_assays <- c(derived_assays,
                                        sprintf("%s (from %s)", nm, src))
                }
            }
        }
    }

    # Per-cell sidecars → meta.data
    sidecar_df <- .load_cell_sidecars(attr(dd, "path"), colnames(X))
    if (ncol(sidecar_df) > 0L) {
        obj <- Seurat::AddMetaData(obj, metadata = sidecar_df)
    }

    # GEO context
    obj@misc$singlify <- as.list(user_kv)
    obj@misc$singlify_primary_assay <- chosen
    obj@misc$singlify_source_path <- attr(dd, "path")
    if (length(derived_assays) > 0L) {
        obj@misc$singlify_derived_assays <- derived_assays
    }

    obj
}
