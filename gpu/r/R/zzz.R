# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/zzz.R
#
# Package startup — dynlib registration + CUDA availability check.

#' @useDynLib singletGpu, .registration = TRUE
#' @importFrom Rcpp evalCpp
NULL

.onLoad <- function(libname, pkgname) {  # nolint: object_name_linter
    # Load the compiled shared library (singletGpu.so / singletGpu.dll).
    # The .registration = TRUE in useDynLib (NAMESPACE) handles the actual
    # library binding; this hook runs additional startup checks.

    # Warn users if CUDA device count returns 0 (no GPU or CUDA unavailable).
    # We do this as a message, not stop(), so that non-GPU machines can still
    # inspect the package metadata (e.g. for documentation tooling).
    tryCatch({
        n <- .pkg_version()  # cheap no-op CUDA call wrapped in C++
        rm(n)
    }, error = function(e) {
        packageStartupMessage(
            "[singletGpu] CUDA not available: ", conditionMessage(e), "\n",
            "  GPU functions will fail. Install CUDA Toolkit >= 12.0 and ensure\n",
            "  the package was compiled with a CUDA-aware toolchain."
        )
    })

    invisible(NULL)
}

.onUnload <- function(libpath) {  # nolint: object_name_linter
    library.dynam.unload("singletGpu", libpath)
}
