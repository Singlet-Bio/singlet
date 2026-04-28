#' @useDynLib singlify, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

.onAttach <- function(libname, pkgname) {
    v <- utils::packageVersion(pkgname)
    packageStartupMessage(
        sprintf("singlify %s — reader for singlify pipeline outputs.\n",
                v),
        "  read_1pz(path)           — single .1pz file\n",
        "  read_singlify_dir(path)  — full pipeline directory\n",
        "  as_sce(path)             — SingleCellExperiment adapter\n",
        "  as_seurat(path)          — Seurat adapter"
    )
}
