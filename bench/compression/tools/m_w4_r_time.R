# m_w4: R read path, legacy bundle vs v2, one sample and the whole study (local files; no network).
#
#   Rscript tools/m_w4_r_time.R <library with the built package> <v2 root> <legacy .singlet bundle> [GSM]
#
# Reports min-of-3 elapsed seconds, R's own peak memory (gc max used, MB; this is R heap, not process RSS) and the size of the result.
args <- commandArgs(TRUE)
.libPaths(c(args[[1]], .libPaths()))
suppressMessages({library(singlet); library(SingleCellExperiment)})
v2root <- args[[2]]; bundle <- args[[3]]; gsm <- if (length(args) >= 4) args[[4]] else "GSM4317426"
Sys.setenv(SINGLET_CACHE_DIR = tempfile("m_w4_r_cache_"))

measure <- function(label, f, reps = 3) {
    best <- Inf; peak <- NA; sz <- NA; nc <- NA
    for (i in seq_len(reps)) {
        gc(reset = TRUE)
        t <- system.time(x <- f())[["elapsed"]]
        g <- gc()
        mx <- sum(g[, ncol(g)])  # "max used (Mb)" column
        if (t < best) { best <- t; peak <- mx; sz <- as.numeric(object.size(x)) / 2^20; nc <- ncol(x) }
        rm(x)
    }
    cat(sprintf("%-34s min %7.2f s   R heap peak %7.0f MB   result %6.0f MB   cells %d\n", label, best, peak, sz, nc))
}

invisible(gc(reset = TRUE)); g0 <- gc(); cat(sprintf("R heap before any load: %.0f MB (packages attached)
", sum(g0[, ncol(g0)])))
st <- singlet_study("GSE145370", base = v2root)
cat(sprintf("R %s, singlet isa: %s\n", getRversion(), singlet:::pz2_isa_report_()))
measure("legacy bundle, one sample", function() suppressWarnings(singlet:::.read_singlet_bundle(bundle, gsms = gsm)), reps = 3)
measure("v2 local, one sample", function() singlet:::.v2_read_study(st, gsm))
measure("legacy bundle, 14-sample study", function() suppressWarnings(singlet:::.read_singlet_bundle(bundle)), reps = 1)
measure("v2 local, 14-sample study", function() singlet:::.v2_read_study(st, NULL))
