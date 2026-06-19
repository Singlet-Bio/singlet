# SPDX-License-Identifier: MIT
# Package startup — native library registration.

#' @useDynLib singlet, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

.onUnload <- function(libpath) {  # nolint: object_name_linter
    library.dynam.unload("singlet", libpath)
}
