#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/nebula_r_reference.R
#
# Reference implementation for Nebula_TinySynthetic_VsR.
#
# Tries NEBULA R first (library(nebula)).  If that package is absent, falls
# back to a pure-R per-(SNP, gene) negative-binomial GLM via IRLS — an
# approximation that has no donor random effect but produces comparable beta
# and p-value rankings for the planted synthetic data.
#
# Exit codes:
#   0  — success (outputs written)
#   2  — both NEBULA and the IRLS fallback are unavailable
#   1  — runtime error
#
# Usage:
#   Rscript nebula_r_reference.R \
#     --counts  /tmp/.../counts.bin  \
#     --snp_ad  /tmp/.../snp_ad.bin  \
#     --snp_dp  /tmp/.../snp_dp.bin  \
#     --donor_ids /tmp/.../donor_ids.tsv \
#     --out_beta  /tmp/.../r_beta.tsv \
#     --out_pval  /tmp/.../r_pval.tsv \
#     --n_genes 50 \
#     --n_snps  100

suppressMessages({
  library(methods)
})

# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  idx <- which(args == flag)
  if (length(idx) == 0) stop(paste("Missing argument:", flag))
  args[idx + 1]
}

counts_path  <- get_arg("--counts")
snp_ad_path  <- get_arg("--snp_ad")
snp_dp_path  <- get_arg("--snp_dp")
donor_path   <- get_arg("--donor_ids")
out_beta     <- get_arg("--out_beta")
out_pval     <- get_arg("--out_pval")
n_genes      <- as.integer(get_arg("--n_genes"))
n_snps       <- as.integer(get_arg("--n_snps"))

# --------------------------------------------------------------------------- #
# Binary CSC reader (CSCC format written by C++ test fixture)                 #
# Magic 0x43535343, then uint32 n_rows, uint32 n_cols, uint64 nnz,            #
# float[nnz] values, int32[n_cols+1] indptr, int32[nnz] indices.              #
# --------------------------------------------------------------------------- #
read_csc_bin <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))

  magic <- readBin(con, integer(), n = 1, size = 4, endian = "little")
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

# Convert CSC list to a dense R matrix (rows × cols).
csc_to_dense <- function(csc) {
  mat <- matrix(0.0, nrow = csc$n_rows, ncol = csc$n_cols)
  for (j in seq_len(csc$n_cols)) {
    start <- csc$indptr[j] + 1   # 0-based → 1-based
    end   <- csc$indptr[j + 1]
    if (end >= start) {
      rows <- csc$indices[start:end] + 1  # 0-based → 1-based
      vals <- csc$values [start:end]
      mat[rows, j] <- vals
    }
  }
  mat
}

# --------------------------------------------------------------------------- #
# Load inputs                                                                  #
# --------------------------------------------------------------------------- #
counts_csc  <- read_csc_bin(counts_path)  # genes × cells (CSC)
snp_ad_csc  <- read_csc_bin(snp_ad_path)  # snps  × cells (CSC)
snp_dp_csc  <- read_csc_bin(snp_dp_path)  # snps  × cells (CSC)

counts_mat  <- csc_to_dense(counts_csc)   # n_genes × n_cells
snp_ad_mat  <- csc_to_dense(snp_ad_csc)   # n_snps  × n_cells
snp_dp_mat  <- csc_to_dense(snp_dp_csc)   # n_snps  × n_cells

donor_ids   <- as.integer(readLines(donor_path))
n_cells     <- length(donor_ids)
n_donors    <- length(unique(donor_ids))

stopifnot(ncol(counts_mat) == n_cells,
          ncol(snp_ad_mat) == n_cells,
          ncol(snp_dp_mat) == n_cells,
          nrow(counts_mat) == n_genes,
          nrow(snp_ad_mat) == n_snps,
          nrow(snp_dp_mat) == n_snps)

# --------------------------------------------------------------------------- #
# Compute per-donor genotype dosage (mean alt-allele frequency × 2)            #
# dosage[donor, snp] ∈ [0, 2]                                                  #
# --------------------------------------------------------------------------- #
compute_dosage <- function(ad_mat, dp_mat, donor_ids) {
  # ad_mat and dp_mat: n_snps × n_cells
  n_snps   <- nrow(ad_mat)
  n_donors <- length(unique(donor_ids))
  donors   <- sort(unique(donor_ids))

  dosage <- matrix(0.0, nrow = n_donors, ncol = n_snps)
  for (di in seq_along(donors)) {
    d <- donors[di]
    cells <- which(donor_ids == d)
    ad_sum <- rowSums(ad_mat[, cells, drop = FALSE])
    dp_sum <- rowSums(dp_mat[, cells, drop = FALSE])
    # dosage = 2 × (total alt reads / total depth), clamped to [0, 2]
    dosage[di, ] <- pmin(2.0, 2.0 * ad_sum / pmax(dp_sum, 1))
  }
  dosage  # n_donors × n_snps
}

donor_dosage <- compute_dosage(snp_ad_mat, snp_dp_mat, donor_ids)  # n_donors × n_snps

# --------------------------------------------------------------------------- #
# Per-donor pseudo-bulk counts for each gene                                  #
# --------------------------------------------------------------------------- #
donors <- sort(unique(donor_ids))
pb_counts <- matrix(0.0, nrow = n_donors, ncol = n_genes)  # n_donors × n_genes
for (di in seq_along(donors)) {
  d <- donors[di]
  cells <- which(donor_ids == d)
  pb_counts[di, ] <- colSums(t(counts_mat[, cells, drop = FALSE]))
  # counts_mat is genes × cells; colSums(t(...)) sums cells per gene
}

# --------------------------------------------------------------------------- #
# Method A: NEBULA R (mixed-model NB, gold standard)                          #
# --------------------------------------------------------------------------- #
has_nebula <- requireNamespace("nebula", quietly = TRUE)

if (has_nebula) {
  message("[nebula_r_reference] Using NEBULA R package")

  beta_mat <- matrix(NA_real_, nrow = n_snps, ncol = n_genes)
  pval_mat <- matrix(NA_real_, nrow = n_snps, ncol = n_genes)

  # NEBULA expects: count matrix (genes × cells), donor_ids, covariate matrix.
  # We loop over SNPs (each SNP is one covariate).
  for (s in seq_len(n_snps)) {
    # Per-cell genotype dosage: look up donor_id → dosage[donor, snp].
    geno_per_cell <- donor_dosage[donor_ids + 1, s]  # donor_ids is 0-based

    # Suppress NEBULA convergence messages.
    suppressMessages({
      fit <- tryCatch(
        nebula::nebula(
          count  = counts_mat,        # genes × cells
          id     = donor_ids + 1L,    # NEBULA expects 1-based integers
          pred   = data.frame(geno = geno_per_cell),
          model  = "NBLMM"
        ),
        error = function(e) NULL
      )
    })

    if (!is.null(fit) && "summary" %in% names(fit)) {
      # fit$summary is a data.frame with one row per gene;
      # columns: logFC_geno, p_geno (others vary by version).
      summ <- fit$summary
      beta_col <- grep("logFC|beta|coef", colnames(summ), ignore.case = TRUE, value = TRUE)
      pval_col <- grep("^p_", colnames(summ), ignore.case = TRUE, value = TRUE)
      if (length(beta_col) >= 1 && length(pval_col) >= 1) {
        beta_mat[s, ] <- summ[[beta_col[1]]]
        pval_mat[s, ] <- summ[[pval_col[1]]]
      }
    }
  }

  # Replace NAs with 0 / 1 for failed pairs.
  beta_mat[is.na(beta_mat)] <- 0.0
  pval_mat[is.na(pval_mat)] <- 1.0

  write.table(beta_mat, out_beta, sep = "\t", row.names = FALSE, col.names = FALSE)
  write.table(pval_mat, out_pval, sep = "\t", row.names = FALSE, col.names = FALSE)
  quit(status = 0)
}

# --------------------------------------------------------------------------- #
# Method B: Pure-R IRLS fallback                                              #
#                                                                              #
# Per-(SNP, gene) NB GLM: y ~ offset(log(lib_size)) + genotype               #
# Fit via MASS::glm.nb if available, else Poisson GLM as approximation.       #
# No donor random effect — this is an approximation, sufficient for rank      #
# correlation validation of the GPU kernel.                                    #
# --------------------------------------------------------------------------- #
has_mass <- requireNamespace("MASS", quietly = TRUE)

# Poisson IRLS fallback when MASS is absent.
poisson_nb_fit <- function(y, x, offset_vec) {
  # Fit Poisson GLM (log link) via glm().
  fit <- tryCatch(
    glm(y ~ x + offset(offset_vec),
        family = poisson(link = "log"),
        control = glm.control(maxit = 50, epsilon = 1e-6)),
    error = function(e) NULL,
    warning = function(w) suppressWarnings(
      glm(y ~ x + offset(offset_vec),
          family = poisson(link = "log"),
          control = glm.control(maxit = 50, epsilon = 1e-6)))
  )
  if (is.null(fit)) return(c(beta = 0.0, pval = 1.0))
  s <- tryCatch(summary(fit)$coefficients, error = function(e) NULL)
  if (is.null(s) || nrow(s) < 2) return(c(beta = 0.0, pval = 1.0))
  c(beta = s[2, 1], pval = s[2, 4])
}

nb_fit <- function(y, x, offset_vec) {
  if (has_mass) {
    fit <- tryCatch(
      MASS::glm.nb(y ~ x + offset(offset_vec),
                   control = glm.control(maxit = 50, epsilon = 1e-6)),
      error   = function(e) NULL,
      warning = function(w) suppressWarnings(
        MASS::glm.nb(y ~ x + offset(offset_vec),
                     control = glm.control(maxit = 50, epsilon = 1e-6)))
    )
    if (!is.null(fit)) {
      s <- tryCatch(summary(fit)$coefficients, error = function(e) NULL)
      if (!is.null(s) && nrow(s) >= 2)
        return(c(beta = s[2, 1], pval = s[2, 4]))
    }
  }
  # Fallback: Poisson.
  poisson_nb_fit(y, x, offset_vec)
}

message("[nebula_r_reference] NEBULA not available — using pure-R NB/Poisson IRLS fallback")

# Library size per cell (for offset term).
lib_size      <- colSums(counts_mat)
log_lib_size  <- log(pmax(lib_size, 1))

# Per-cell genotype dosage for each SNP (same approach as NEBULA branch above).
# donor_ids is 0-based.
beta_mat <- matrix(NA_real_, nrow = n_snps, ncol = n_genes)
pval_mat <- matrix(NA_real_, nrow = n_snps, ncol = n_genes)

for (s in seq_len(n_snps)) {
  geno <- donor_dosage[donor_ids + 1, s]  # n_cells vector, 0-based donor_ids

  for (g in seq_len(n_genes)) {
    y <- as.numeric(counts_mat[g, ])
    res <- nb_fit(y, geno, log_lib_size)
    beta_mat[s, g] <- res["beta"]
    pval_mat[s, g] <- res["pval"]
  }
}

beta_mat[is.na(beta_mat)] <- 0.0
pval_mat[is.na(pval_mat)] <- 1.0

write.table(beta_mat, out_beta, sep = "\t", row.names = FALSE, col.names = FALSE)
write.table(pval_mat, out_pval, sep = "\t", row.names = FALSE, col.names = FALSE)
message("[nebula_r_reference] IRLS fallback complete")
quit(status = 0)
