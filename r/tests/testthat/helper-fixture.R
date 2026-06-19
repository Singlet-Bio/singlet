# SPDX-License-Identifier: MIT
# Shared fixture resolver for testthat tests.
#
# Tests need a real singlet pipeline output because the whole point of
# the package is to decode the wire format faithfully. Rather than
# shipping large fixtures in the package, we look up a known-good sample
# at test time:
#
#   1. Environment variable SINGLET_TEST_FIXTURE_DIR takes precedence.
#   2. Otherwise search the canonical fallback paths on the Clipper NFS
#      tree (smallest known-good sample first).
#
# Tests that need a fixture call skip_if_no_fixture() first. This keeps
# R CMD check clean in environments without pipeline outputs.

fixture_dir <- function() {
    env <- Sys.getenv("SINGLET_TEST_FIXTURE_DIR")
    if (nzchar(env) && dir.exists(env)) return(env)

    candidates <- c(
        # GSM7103327 — tiny, 410 cells
        "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/GSE227/GSE227136/GSM7103327",
        # GSM3587956 — seqwell, 76 cells
        "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/GSE116/GSE116256/GSM3587956",
        # GSM3693219 — drop-seq, 3512 cells
        "/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/GSE129/GSE129096/GSM3693219"
    )
    for (p in candidates) {
        if (!dir.exists(p)) next
        if (file.exists(file.path(p, "gene_counts.1pz")) ||
            file.exists(file.path(p, "spliced.1pz"))) {
            return(p)
        }
    }
    NA_character_
}

skip_if_no_fixture <- function() {
    d <- fixture_dir()
    if (is.na(d)) {
        testthat::skip(paste0(
            "no singlet pipeline fixture available. Set ",
            "SINGLET_TEST_FIXTURE_DIR to a directory with .1pz files."
        ))
    }
    d
}
