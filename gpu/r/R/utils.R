# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/utils.R
#
# Internal utility functions for singletGpu.
# These are not exported (names start with ".").

# ---------------------------------------------------------------------------
# .assert_sce — validate that `x` is a SingleCellExperiment
# ---------------------------------------------------------------------------
.assert_sce <- function(x, caller) {
    if (!methods::is(x, "SingleCellExperiment")) {
        stop(sprintf(
            "%s: `sce` must be a SingleCellExperiment, got %s.",
            caller, class(x)[1L]))
    }
    invisible(x)
}

# ---------------------------------------------------------------------------
# .gpu_available — return TRUE if at least one CUDA device is visible
#
# WHY a simple integer check: we do NOT call into C++ here because this
# function is used in error messages before the package dynlib may be loaded.
# Instead we query the system environment for CUDA_VISIBLE_DEVICES or try
# a cheap nvidia-smi probe.
# ---------------------------------------------------------------------------
.gpu_available <- function() {
    # Fast path: if CUDA_VISIBLE_DEVICES is set to empty string, no GPU.
    cvd <- Sys.getenv("CUDA_VISIBLE_DEVICES", unset = NA_character_)
    if (!is.na(cvd) && nchar(cvd) == 0L) return(FALSE)

    # Try nvidia-smi (non-blocking, suppress output)
    result <- tryCatch(
        system("nvidia-smi -L", intern = TRUE, ignore.stderr = TRUE),
        warning = function(w) character(0),
        error   = function(e) character(0)
    )
    length(result) > 0L
}

# ---------------------------------------------------------------------------
# .check_reticulate — stop with an informative message if reticulate is absent
# ---------------------------------------------------------------------------
.check_reticulate <- function() {
    if (!requireNamespace("reticulate", quietly = TRUE)) {
        stop(
            "The `reticulate` package is required for this function.\n",
            "  Install it with: install.packages('reticulate')\n",
            "  Then configure a Python environment with the singlet_gpu wheel:\n",
            "    pip install git+https://github.com/zdebruine/Singlet-AI",
            "#subdirectory=singlet-gpu/python",
            call. = FALSE
        )
    }
    invisible(TRUE)
}

# ---------------------------------------------------------------------------
# .import_singlet_gpu_python — import the singlet_gpu Python module via reticulate
# ---------------------------------------------------------------------------
.import_singlet_gpu_python <- function() {
    # Cache the module object in the package environment to avoid re-importing
    # on every call.
    if (!exists(".rsc_module", envir = .singletGpu_env)) {
        tryCatch({
            rsc <- reticulate::import("rapids_singlecell",
                                      delay_load = FALSE,
                                      convert    = FALSE)
            assign(".rsc_module", rsc, envir = .singletGpu_env)
        }, error = function(e) {
            stop(
                "Could not import `rapids_singlecell` Python module.\n",
                "  Install it into your active Python environment:\n",
                "    pip install rapids-singlecell\n",
                "  Then also install the singlet_gpu wheel:\n",
                "    pip install git+https://github.com/zdebruine/Singlet-AI",
                "#subdirectory=singlet-gpu/python\n",
                "  Error: ", conditionMessage(e),
                call. = FALSE
            )
        })
    }
    get(".rsc_module", envir = .singletGpu_env)
}

# Package-level environment for caching imported modules
.singletGpu_env <- new.env(parent = emptyenv())

# ---------------------------------------------------------------------------
# .sce_to_anndata — marshal a SingleCellExperiment to anndata.AnnData
#
# Uses the "counts" or "logcounts" assay + all reducedDims.
# Only called from reticulate_bridge.R — must have reticulate loaded.
# ---------------------------------------------------------------------------
.sce_to_anndata <- function(sce) {
    ad <- reticulate::import("anndata", convert = FALSE)
    scipy_sparse <- reticulate::import("scipy.sparse", convert = FALSE)

    # Primary assay: prefer logcounts, fall back to counts
    assay_name <- if ("logcounts" %in% SummarizedExperiment::assayNames(sce)) {
        "logcounts"
    } else {
        "counts"
    }
    mat <- SummarizedExperiment::assay(sce, assay_name)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    # Transpose to cells × genes for AnnData convention
    mat_t  <- Matrix::t(mat)
    X_csr  <- scipy_sparse$csr_matrix(
        reticulate::tuple(
            reticulate::np_array(mat_t@x,  dtype = "float32"),
            reticulate::np_array(mat_t@i,  dtype = "int32"),
            reticulate::np_array(mat_t@p,  dtype = "int32")
        ),
        shape = reticulate::tuple(ncol(sce), nrow(sce))
    )

    # obs (cells) and var (genes) data frames
    obs_df <- as.data.frame(SummarizedExperiment::colData(sce))
    var_df <- as.data.frame(SummarizedExperiment::rowData(sce))

    adata <- ad$AnnData(X   = X_csr,
                        obs = reticulate::r_to_py(obs_df),
                        var = reticulate::r_to_py(var_df))

    # Transfer reducedDims to obsm
    for (nm in SingleCellExperiment::reducedDimNames(sce)) {
        emb   <- SingleCellExperiment::reducedDim(sce, nm)
        py_nm <- if (nm == "PCA") "X_pca" else nm
        adata$obsm[py_nm] <- reticulate::np_array(emb, dtype = "float32")
    }

    adata
}
