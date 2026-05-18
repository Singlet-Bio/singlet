#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/daesc_r_reference.R
#
# Reference implementation for Daesc_TinySynthetic_VsR.
#
# Method A (primary): DAESC R package (Hu et al. 2023).
#   library(daesc); runs daesc() per SNP on the synthetic data.
#
# Method B (fallback): pure-R beta-binomial MLE via VGAM::vglm with
#   betabinomial family.  Always available if VGAM is installed.
#   Fits an intercept-only beta-binomial GLM per SNP.
#
# Exit codes:
#   0  — success using DAESC R
#   2  — success using VGAM fallback (DAESC absent)
#   1  — runtime error
#   3  — neither DAESC nor VGAM available (test will GTEST_SKIP)
#
# Usage:
#   Rscript daesc_r_reference.R \
#     --snp_ad    /tmp/.../syn_snp_ad.bin  \
#     --snp_dp    /tmp/.../syn_snp_dp.bin  \
#     --cell_type /tmp/.../syn_cell_type.tsv \
#     --out_beta  /tmp/.../r_beta.tsv \
#     --out_pval  /tmp/.../r_pval.tsv \
#     --n_snps    100 \
#     --n_cells   200 \
#     --min_depth 5

suppressMessages({ library(methods) })

# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) {
    if (!is.null(default)) return(default)
    stop(paste("Missing argument:", flag))
  }
  args[idx + 1]
}

snp_ad_path  <- get_arg("--snp_ad")
snp_dp_path  <- get_arg("--snp_dp")
ct_path      <- get_arg("--cell_type")
out_beta     <- get_arg("--out_beta")
out_pval     <- get_arg("--out_pval")
n_snps       <- as.integer(get_arg("--n_snps"))
n_cells      <- as.integer(get_arg("--n_cells"))
min_depth    <- as.integer(get_arg("--min_depth", "5"))

# --------------------------------------------------------------------------- #
# Binary CSC reader (CSCC format written by C++ test fixture)                 #
# Magic 0x43535343, then uint32 n_rows, uint32 n_cols, uint64 nnz,            #
# float[nnz] values, int32[n_cols+1] indptr, int32[nnz] indices.              #
# --------------------------------------------------------------------------- #
read_csc_bin <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))

  magic  <- readBin(con, integer(), n = 1, size = 4, endian = "little")
  stopifnot(magic == 0x43535343L)

  n_rows <- readBin(con, integer(), n = 1, size = 4, endian = "little")
  n_cols <- readBin(con, integer(), n = 1, size = 4, endian = "little")
  nnz    <- readBin(con, integer(), n = 1, size = 8, endian = "little")

  values  <- readBin(con, numeric(), n = nnz,          size = 4, endian = "little")
  indptr  <- readBin(con, integer(), n = n_cols + 1,   size = 4, endian = "little")
  indices <- readBin(con, integer(), n = nnz,          size = 4, endian = "little")

  list(n_rows = n_rows, n_cols = n_cols, nnz = nnz,
       values = values, indptr = indptr, indices = indices)
}

# Convert CSC list → dense matrix (n_rows × n_cols).
csc_to_dense <- function(csc) {
  mat <- matrix(0.0, nrow = csc$n_rows, ncol = csc$n_cols)
  for (j in seq_len(csc$n_cols)) {
    start <- csc$indptr[j] + 1     # 0-based → 1-based
    end   <- csc$indptr[j + 1]
    if (end >= start) {
      rows <- csc$indices[start:end] + 1
      vals <- csc$values [start:end]
      mat[rows, j] <- vals
    }
  }
  mat
}

# --------------------------------------------------------------------------- #
# Load inputs                                                                  #
# --------------------------------------------------------------------------- #
snp_ad_csc  <- read_csc_bin(snp_ad_path)   # n_snps × n_cells
snp_dp_csc  <- read_csc_bin(snp_dp_path)   # n_snps × n_cells
cell_type   <- as.integer(readLines(ct_path)) + 1L   # 0-based → 1-based

snp_ad_mat  <- csc_to_dense(snp_ad_csc)    # n_snps × n_cells
snp_dp_mat  <- csc_to_dense(snp_dp_csc)    # n_snps × n_cells

stopifnot(nrow(snp_ad_mat) == n_snps, ncol(snp_ad_mat) == n_cells,
          nrow(snp_dp_mat) == n_snps, ncol(snp_dp_mat) == n_cells,
          length(cell_type) == n_cells)

# --------------------------------------------------------------------------- #
# Method A: DAESC R (gold standard)                                           #
# --------------------------------------------------------------------------- #
has_daesc <- requireNamespace("daesc", quietly = TRUE)

if (has_daesc) {
  message("[daesc_r_reference] Using DAESC R package")

  beta_vec <- numeric(n_snps)
  pval_vec <- numeric(n_snps)

  for (s in seq_len(n_snps)) {
    ad_s  <- as.integer(snp_ad_mat[s, ])
    dp_s  <- as.integer(snp_dp_mat[s, ])
    # Filter cells with dp >= min_depth.
    keep  <- dp_s >= min_depth
    if (sum(keep) < 5) {
      beta_vec[s] <- 0.0
      pval_vec[s] <- 1.0
      next
    }
    ad_k  <- ad_s[keep]
    dp_k  <- dp_s[keep]
    ct_k  <- cell_type[keep]

    fit <- tryCatch(
      daesc::daesc(
        Y        = ad_k,
        N        = dp_k,
        label    = ct_k,
        covariate = NULL,
        model    = "daesc-bb"
      ),
      error = function(e) NULL
    )

    if (!is.null(fit) && "beta" %in% names(fit)) {
      beta_vec[s] <- fit$beta
      pval_vec[s] <- if ("pvalue" %in% names(fit)) fit$pvalue else 1.0
    } else {
      beta_vec[s] <- 0.0
      pval_vec[s] <- 1.0
    }
  }

  write.table(data.frame(beta = beta_vec), out_beta,
              sep = "\t", row.names = FALSE, col.names = FALSE)
  write.table(data.frame(pval = pval_vec), out_pval,
              sep = "\t", row.names = FALSE, col.names = FALSE)
  message("[daesc_r_reference] DAESC R complete")
  quit(status = 0)
}

# --------------------------------------------------------------------------- #
# Method B: VGAM betabinomial fallback                                        #
#                                                                              #
# Per-SNP intercept-only beta-binomial MLE via VGAM::vglm with               #
# betabinomial(lmu="logitlink") family.                                        #
# This fits logit(mu) and log(rho) (overdispersion).                          #
# The intercept logit(mu) = beta corresponds to the allelic bias log-odds.    #
# p-value from Wald test: z = beta / se(beta), p = 2*pnorm(-|z|).            #
# --------------------------------------------------------------------------- #
has_vgam <- requireNamespace("VGAM", quietly = TRUE)

if (!has_vgam) {
  message("[daesc_r_reference] Neither DAESC nor VGAM available — cannot produce reference")
  quit(status = 3)
}

message("[daesc_r_reference] DAESC not available — using VGAM betabinomial fallback")

suppressMessages(library(VGAM))

beta_vec <- numeric(n_snps)
pval_vec <- numeric(n_snps)

for (s in seq_len(n_snps)) {
  ad_s  <- as.integer(snp_ad_mat[s, ])
  dp_s  <- as.integer(snp_dp_mat[s, ])
  keep  <- dp_s >= min_depth

  if (sum(keep) < 5) {
    beta_vec[s] <- 0.0
    pval_vec[s] <- 1.0
    next
  }

  ad_k <- ad_s[keep]
  dp_k <- dp_s[keep]

  # VGAM betabinomial: response is cbind(successes, failures).
  resp <- cbind(ad_k, dp_k - ad_k)

  fit <- tryCatch(
    VGAM::vglm(resp ~ 1,
               family = VGAM::betabinomial(lmu = "logitlink",
                                           lrho = "logitlink"),
               control = VGAM::vglm.control(maxit = 50, epsilon = 1e-5)),
    error   = function(e) NULL,
    warning = function(w) suppressWarnings(
      VGAM::vglm(resp ~ 1,
                 family = VGAM::betabinomial(lmu = "logitlink",
                                             lrho = "logitlink"),
                 control = VGAM::vglm.control(maxit = 50, epsilon = 1e-5)))
  )

  if (is.null(fit)) {
    beta_vec[s] <- 0.0
    pval_vec[s] <- 1.0
    next
  }

  # Extract intercept (mu link) coefficient and its SE.
  co  <- tryCatch(VGAM::coef(fit), error = function(e) NULL)
  cse <- tryCatch(
    sqrt(diag(VGAM::vcov(fit))),
    error = function(e) rep(NA_real_, length(co))
  )

  if (is.null(co) || length(co) < 1 || is.na(cse[1])) {
    beta_vec[s] <- 0.0
    pval_vec[s] <- 1.0
    next
  }

  # co[1] = logit(mu) intercept = beta (allelic bias log-odds).
  b  <- co[1]
  se <- cse[1]

  # Wald test vs beta = 0 (balanced null).
  z    <- b / max(se, 1e-10)
  pval <- 2.0 * pnorm(-abs(z))

  beta_vec[s] <- b
  pval_vec[s] <- max(1e-300, min(1.0, pval))
}

write.table(data.frame(beta = beta_vec), out_beta,
            sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(data.frame(pval = pval_vec), out_pval,
            sep = "\t", row.names = FALSE, col.names = FALSE)
message("[daesc_r_reference] VGAM betabinomial fallback complete")
quit(status = 2)
