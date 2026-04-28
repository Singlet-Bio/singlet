#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/hdwgcna_r_reference.R
#
# Reference driver for network/hdwgcna correctness harness (cycle 46).
#
# Primary path: WGCNA Bioconductor package (Langfelder & Horvath 2008).
#   hdWGCNA R package tried first; falls back to vanilla WGCNA if absent.
# Pure-R fallback: when WGCNA is absent, computes TOM + tree via base R stats::hclust.
#
# Usage:
#   Rscript hdwgcna_r_reference.R \
#       --data_dir    <dir>  # contains expression.bin, soft_power.bin
#       --out_dir     <dir>  # output: py_module_labels.bin, py_tom.bin, ...
#       --task        <str>  # task label (logging only)
#       --height_cut  <num>  # dendrogram height threshold (default 0.9)
#       --min_module_size <int>  # minimum genes per module (default 5)
#
# Exit codes:
#   0  — WGCNA (or hdWGCNA) ran successfully; output written.
#   2  — WGCNA not installed; pure-R fallback used; output written.
#   1  — error.
#
# Output binary files (same layout as wgcna_py_reference.py):
#   py_module_labels.bin  — [uint32_t n][int32_t × n]
#   py_tom.bin            — [uint32_t n][float32 × n × n]  row-major
#   py_eigengenes.bin     — [uint32_t n_cells][uint32_t n_mods][float32 × n_cells × n_mods]
#   py_hub_kme.bin        — [uint32_t n_entries][int32_t mod][int32_t gene][float32 kme] × n

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0L) {
        if (is.null(default)) stop("Missing required argument: ", flag)
        return(default)
    }
    return(args[idx + 1L])
}

data_dir         <- parse_arg("--data_dir")
out_dir          <- parse_arg("--out_dir")
task_label       <- parse_arg("--task",            "unknown")
height_cut       <- as.numeric(parse_arg("--height_cut",       "0.9"))
min_module_size  <- as.integer(parse_arg("--min_module_size",  "5"))

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Binary I/O helpers
# ---------------------------------------------------------------------------

read_expression <- function(data_dir) {
    path <- file.path(data_dir, "expression.bin")
    con  <- file(path, "rb")
    dims <- readBin(con, what = "integer", n = 2L, size = 4L, signed = FALSE,
                    endian = "little")
    n_cells <- dims[1L]; n_genes <- dims[2L]
    data    <- readBin(con, what = "numeric", n = n_cells * n_genes, size = 4L,
                       endian = "little")
    close(con)
    # R is column-major; data is row-major (C) → need to transpose
    m <- matrix(data, nrow = n_genes, ncol = n_cells, byrow = FALSE)
    # After byrow=FALSE with n_cells*n_genes floats stored C-order:
    # Correct reading: read as n_genes rows × n_cells cols (FORTRAN layout of the
    # transposed matrix), which gives us genes × cells — then t() to get cells × genes.
    m <- t(matrix(data, nrow = n_cells, ncol = n_genes, byrow = TRUE))  # genes × cells
    return(m)  # genes × cells
}

read_soft_power <- function(data_dir) {
    path <- file.path(data_dir, "soft_power.bin")
    con  <- file(path, "rb")
    beta <- readBin(con, what = "integer", n = 1L, size = 4L, signed = FALSE,
                    endian = "little")
    close(con)
    return(as.integer(beta))
}

write_int32_vec <- function(path, vec) {
    con <- file(path, "wb")
    writeBin(as.integer(length(vec)), con, size = 4L, endian = "little")
    writeBin(as.integer(vec), con, size = 4L, endian = "little")
    close(con)
}

write_float32_mat <- function(path, mat) {
    # mat: matrix in R (column-major); write n × n as row-major
    n   <- nrow(mat)
    con <- file(path, "wb")
    writeBin(as.integer(n), con, size = 4L, endian = "little")
    # t(mat) makes it row-major when serialised column-by-column
    writeBin(as.numeric(t(mat)), con, size = 4L, endian = "little")
    close(con)
}

write_eigengenes <- function(path, eigen_mat) {
    # eigen_mat: cells × modules (row-major output)
    n_cells <- nrow(eigen_mat)
    n_mods  <- ncol(eigen_mat)
    con     <- file(path, "wb")
    writeBin(as.integer(c(n_cells, n_mods)), con, size = 4L, endian = "little")
    writeBin(as.numeric(t(eigen_mat)), con, size = 4L, endian = "little")
    close(con)
}

write_hub_kme <- function(path, hub_df) {
    # hub_df: data.frame with columns module (int), gene (int), kme (float)
    n   <- nrow(hub_df)
    con <- file(path, "wb")
    writeBin(as.integer(n), con, size = 4L, endian = "little")
    for (i in seq_len(n)) {
        writeBin(as.integer(hub_df$module[i]), con, size = 4L, endian = "little")
        writeBin(as.integer(hub_df$gene[i]),   con, size = 4L, endian = "little")
        writeBin(as.numeric(hub_df$kme[i]),    con, size = 4L, endian = "little")
    }
    close(con)
}

# ---------------------------------------------------------------------------
# Pure-R WGCNA-style fallback (no Bioconductor dependency)
# ---------------------------------------------------------------------------

compute_tom_r <- function(adj) {
    # adj: genes × genes soft power adjacency
    n   <- nrow(adj)
    k   <- rowSums(adj) - diag(adj)  # degree (exclude self)
    L   <- adj %*% adj               # n×n numerator inner sum
    num <- L + adj
    ki  <- matrix(k, nrow = n, ncol = n, byrow = FALSE)
    kj  <- matrix(k, nrow = n, ncol = n, byrow = TRUE)
    denom <- pmin(ki, kj) + 1.0 - adj
    denom[denom < 1e-10] <- 1e-10
    tom   <- num / denom
    tom   <- pmin(pmax(tom, 0.0), 1.0)
    diag(tom) <- 1.0
    return(tom)
}

compute_eigengene_r <- function(expr_sub) {
    # expr_sub: cells × genes
    expr_sub <- sweep(expr_sub, 2L, colMeans(expr_sub), "-")
    if (ncol(expr_sub) == 1L) {
        v <- expr_sub[, 1L]
        nr <- sqrt(sum(v^2))
        return(if (nr < 1e-10) v else v / nr)
    }
    sv <- tryCatch(svd(expr_sub, nu = 1L, nv = 0L), error = function(e) NULL)
    if (is.null(sv)) return(rep(0.0, nrow(expr_sub)))
    pc1 <- sv$u[, 1L]
    if (mean(pc1) < 0) pc1 <- -pc1
    return(pc1)
}

run_wgcna_fallback <- function(expr_mat, beta, height_cut, min_module_size) {
    # expr_mat: genes × cells
    n_genes <- nrow(expr_mat)
    n_cells <- ncol(expr_mat)
    expr_t  <- t(expr_mat)                   # cells × genes

    # 1. Pearson correlation matrix
    corr <- cor(expr_t, method = "pearson")  # genes × genes
    corr[is.na(corr)] <- 0.0

    # 2. Soft power adjacency
    adj  <- abs(corr)^beta

    # 3. TOM
    tom  <- compute_tom_r(adj)               # genes × genes

    # 4. Hierarchical clustering
    dist_vec <- as.dist(1.0 - tom)
    hc       <- hclust(dist_vec, method = "average")

    # 5. Dynamic tree cut (height threshold)
    raw_labels <- cutree(hc, h = height_cut)  # 1-based labels

    # 6. Relabel small modules to 0 (grey)
    tab       <- table(raw_labels)
    label_map <- setNames(integer(length(tab)), names(tab))
    mod_id    <- 1L
    for (lab in names(tab)) {
        if (tab[lab] >= min_module_size) {
            label_map[lab] <- mod_id
            mod_id <- mod_id + 1L
        } else {
            label_map[lab] <- 0L
        }
    }
    labels <- label_map[as.character(raw_labels)]
    names(labels) <- NULL
    labels <- as.integer(labels)

    # 7. Eigengenes
    unique_mods  <- sort(unique(labels[labels > 0L]))
    n_mods       <- length(unique_mods)
    eigen_mat    <- matrix(0.0, nrow = n_cells, ncol = n_mods)
    for (col_i in seq_along(unique_mods)) {
        mod       <- unique_mods[col_i]
        gene_mask <- which(labels == mod)
        sub       <- expr_t[, gene_mask, drop = FALSE]  # cells × mod_genes
        eigen_mat[, col_i] <- compute_eigengene_r(sub)
    }

    # 8. Hub kME
    top_n    <- 10L
    hub_rows <- vector("list", n_mods)
    for (col_i in seq_along(unique_mods)) {
        mod       <- unique_mods[col_i]
        gene_idxs <- which(labels == mod) - 1L  # 0-based for C++
        eigen     <- eigen_mat[, col_i]
        kmes      <- vapply(gene_idxs, function(g) {
            v <- expr_t[, g + 1L]
            if (sd(v) < 1e-10 || sd(eigen) < 1e-10) return(0.0)
            cor(v, eigen, method = "pearson")
        }, numeric(1L))
        ord     <- order(kmes, decreasing = TRUE)
        sel_n   <- min(top_n, length(ord))
        hub_rows[[col_i]] <- data.frame(
            module = as.integer(mod),
            gene   = as.integer(gene_idxs[ord[seq_len(sel_n)]]),
            kme    = as.numeric(kmes[ord[seq_len(sel_n)]])
        )
    }
    hub_df <- if (n_mods > 0L) do.call(rbind, hub_rows) else
        data.frame(module = integer(0), gene = integer(0), kme = numeric(0))

    list(labels = labels, tom = tom, eigengenes = eigen_mat, hub_df = hub_df)
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

message("[hdwgcna_r_reference] task=", task_label,
        " height_cut=", height_cut,
        " min_module_size=", min_module_size)

expr_mat <- read_expression(data_dir)   # genes × cells
beta     <- read_soft_power(data_dir)

# Try WGCNA Bioconductor package first
wgcna_ok <- requireNamespace("WGCNA", quietly = TRUE)

if (wgcna_ok) {
    message("[hdwgcna_r_reference] WGCNA package found — using WGCNA path.")
    suppressPackageStartupMessages(library(WGCNA))

    n_genes <- nrow(expr_mat)
    n_cells <- ncol(expr_mat)
    expr_t  <- t(expr_mat)  # cells × genes

    # Pearson correlation + soft power
    corr <- WGCNA::cor(expr_t, method = "pearson", use = "p")
    corr[is.na(corr)] <- 0.0
    adj  <- abs(corr)^beta

    # TOM via WGCNA
    tom_wgcna <- WGCNA::TOMsimilarity(adj, TOMType = "unsigned", verbose = 0L)

    # Hierarchical clustering
    dist_vec <- as.dist(1.0 - tom_wgcna)
    hc       <- hclust(dist_vec, method = "average")
    raw_labels <- cutree(hc, h = height_cut)

    # Relabel small modules
    tab       <- table(raw_labels)
    label_map <- setNames(integer(length(tab)), names(tab))
    mod_id    <- 1L
    for (lab in names(tab)) {
        if (tab[lab] >= min_module_size) {
            label_map[lab] <- mod_id
            mod_id <- mod_id + 1L
        } else {
            label_map[lab] <- 0L
        }
    }
    labels <- as.integer(label_map[as.character(raw_labels)])

    # Eigengenes via WGCNA::moduleEigengenes
    ME_res   <- WGCNA::moduleEigengenes(expr_t, colors = paste0("M", labels),
                                        verbose = 0L)
    eigen_df <- ME_res$eigengenes  # data.frame: cells × n_mods+grey
    # Remove grey module column if present
    grey_col <- which(grepl("grey", colnames(eigen_df), ignore.case = TRUE))
    if (length(grey_col) > 0L) eigen_df <- eigen_df[, -grey_col, drop = FALSE]
    eigen_mat <- as.matrix(eigen_df)  # cells × n_mods

    # kME hub genes
    unique_mods <- sort(unique(labels[labels > 0L]))
    n_mods      <- length(unique_mods)
    top_n       <- 10L
    hub_rows    <- vector("list", n_mods)
    for (col_i in seq_along(unique_mods)) {
        mod       <- unique_mods[col_i]
        gene_idxs <- which(labels == mod) - 1L  # 0-based for C++
        if (col_i <= ncol(eigen_mat)) {
            eigen <- eigen_mat[, col_i]
        } else {
            eigen <- rep(0.0, nrow(eigen_mat))
        }
        kmes <- vapply(gene_idxs, function(g) {
            v <- expr_t[, g + 1L]
            if (sd(v) < 1e-10 || sd(eigen) < 1e-10) return(0.0)
            cor(v, eigen, method = "pearson")
        }, numeric(1L))
        ord   <- order(kmes, decreasing = TRUE)
        sel_n <- min(top_n, length(ord))
        hub_rows[[col_i]] <- data.frame(
            module = as.integer(mod),
            gene   = as.integer(gene_idxs[ord[seq_len(sel_n)]]),
            kme    = as.numeric(kmes[ord[seq_len(sel_n)]])
        )
    }
    hub_df <- if (n_mods > 0L) do.call(rbind, hub_rows) else
        data.frame(module = integer(0), gene = integer(0), kme = numeric(0))

    write_int32_vec(file.path(out_dir, "py_module_labels.bin"), labels)
    write_float32_mat(file.path(out_dir, "py_tom.bin"), tom_wgcna)
    write_eigengenes(file.path(out_dir, "py_eigengenes.bin"), eigen_mat)
    write_hub_kme(file.path(out_dir, "py_hub_kme.bin"), hub_df)

    n_mods_detected <- length(unique(labels[labels > 0L]))
    message("[hdwgcna_r_reference] WGCNA done. n_genes=", nrow(expr_mat),
            " n_modules=", n_mods_detected, " beta=", beta)
    quit(status = 0L)

} else {
    message("[hdwgcna_r_reference] WGCNA not installed — using pure-R fallback.")
    res <- run_wgcna_fallback(expr_mat, beta, height_cut, min_module_size)

    write_int32_vec(file.path(out_dir, "py_module_labels.bin"), res$labels)
    write_float32_mat(file.path(out_dir, "py_tom.bin"), res$tom)
    write_eigengenes(file.path(out_dir, "py_eigengenes.bin"), res$eigengenes)
    write_hub_kme(file.path(out_dir, "py_hub_kme.bin"), res$hub_df)

    n_mods_detected <- length(unique(res$labels[res$labels > 0L]))
    message("[hdwgcna_r_reference] Pure-R fallback done. n_genes=", nrow(expr_mat),
            " n_modules=", n_mods_detected, " beta=", beta)
    quit(status = 2L)  # 2 = fallback used (valid output)
}
