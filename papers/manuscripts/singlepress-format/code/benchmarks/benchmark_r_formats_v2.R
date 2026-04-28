#!/usr/bin/env Rscript
# Benchmark BPCells, dgCMatrix (RDS), and HDF5Array read/write performance
# Usage: ssh <node> "module load r/4.5.2 && cd <dir> && Rscript benchmark_r_formats_v2.R"

suppressMessages({
  library(Matrix)
  library(BPCells)
  library(HDF5Array)
  library(reticulate)
  library(methods)
})

QUANT_DIR <- "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR <- "/mnt/home/debruinz/Singlet-AI/papers/manuscripts/singlepress-format"

cat("Loading singlepress via reticulate...\n")
use_python("/usr/bin/python3", required = TRUE)
sp <- import("singlepress")
scipy_sparse <- import("scipy.sparse")

# Read survey to pick datasets
survey <- read.csv(file.path(SCRIPT_DIR, "all_datasets_survey.csv"), stringsAsFactors = FALSE)
# Filter: 1M < nnz < 100M, to keep memory reasonable
survey <- survey[survey$nnz > 1e6 & survey$nnz < 100e6, ]
survey <- survey[order(survey$nnz), ]

# Stratified sample of 20
n_sample <- min(20L, nrow(survey))
indices <- round(seq(1, nrow(survey), length.out = n_sample))
survey <- survey[indices, ]
cat(sprintf("Selected %d datasets (nnz range: %s – %s)\n",
            n_sample,
            format(min(survey$nnz), big.mark = ","),
            format(max(survey$nnz), big.mark = ",")))

results <- data.frame()

for (i in seq_len(nrow(survey))) {
  gse_id <- survey$gse_id[i]
  pz_path <- file.path(QUANT_DIR, gse_id, "counts.1pz")
  nnz <- survey$nnz[i]

  cat(sprintf("\n[%d/%d] %s (%s nnz)\n", i, n_sample, gse_id, format(nnz, big.mark = ",")))

  tryCatch({
    # Read .1pz via Python singlepress using py_run_string
    t0 <- proc.time()
    py_run_string(sprintf("
import singlepress as sp
import numpy as np
mat = sp.read_1pz('%s')
csc = mat.tocsc()
csc.sort_indices()
_data = csc.data.astype(np.float64)
_indices = csc.indices.astype(np.int32)
_indptr = csc.indptr.astype(np.int32)
_nrows = csc.shape[0]
_ncols = csc.shape[1]
_nnz = csc.nnz
", pz_path))
    t_read_py <- (proc.time() - t0)[3]
    cat(sprintf("  .1pz read: %.2fs\n", t_read_py))

    # Extract from Python namespace and force R conversion
    data_arr <- as.vector(as.numeric(py$`_data`))
    indices_arr <- as.vector(as.integer(py$`_indices`))
    indptr_arr <- as.vector(as.integer(py$`_indptr`))
    nrows <- as.integer(py$`_nrows`)
    ncols <- as.integer(py$`_ncols`)
    nnz_actual <- as.integer(py$`_nnz`)

    mat <- new("dgCMatrix",
               i = indices_arr, p = indptr_arr, x = data_arr,
               Dim = c(nrows, ncols))

    # Convert to integer for BPCells (it's count data)
    mat_int <- as(mat, "dgCMatrix")
    mat_int@x <- as.double(as.integer(mat_int@x))

    raw_int32_bytes <- length(indptr_arr) * 4 + nnz_actual * 4 + nnz_actual * 4
    rm(data_arr, indices_arr, indptr_arr)

    cat(sprintf("  Shape: %d x %d, raw_bytes: %s\n",
                nrows, ncols, format(raw_int32_bytes, big.mark = ",")))

    tmpdir <- tempdir()

    # ══════════ RDS dgCMatrix ══════════
    rds_path <- file.path(tmpdir, paste0(gse_id, ".rds"))
    saveRDS(mat, rds_path, compress = "gzip")
    rds_size <- file.size(rds_path)

    invisible(gc(FALSE)); invisible(readRDS(rds_path))  # warmup
    rds_times <- numeric(3)
    for (t in 1:3) {
      invisible(gc(FALSE))
      t0 <- proc.time()
      invisible(readRDS(rds_path))
      rds_times[t] <- (proc.time() - t0)[3]
    }
    rds_read_s <- median(rds_times)
    cat(sprintf("  RDS: %.1f MB, ratio=%.1fx, read=%.0fms\n",
                rds_size / 1e6, raw_int32_bytes / rds_size, rds_read_s * 1000))
    file.remove(rds_path)

    # ══════════ BPCells ══════════
    bp_dir <- file.path(tmpdir, paste0(gse_id, "_bp"))
    if (dir.exists(bp_dir)) unlink(bp_dir, recursive = TRUE)
    # BPCells needs integer types for good compression
    bp_input <- convert_matrix_type(mat_int, type = "uint32_t")
    write_matrix_dir(bp_input, bp_dir, compress = TRUE)
    bp_size <- sum(file.size(list.files(bp_dir, full.names = TRUE, recursive = TRUE)))

    bp_times <- numeric(3)
    for (t in 1:3) {
      invisible(gc(FALSE))
      t0 <- proc.time()
      bp_mat2 <- open_matrix_dir(bp_dir)
      full <- as(bp_mat2, "dgCMatrix")
      bp_times[t] <- (proc.time() - t0)[3]
      rm(full, bp_mat2); invisible(gc(FALSE))
    }
    bp_read_s <- median(bp_times)
    cat(sprintf("  BPCells: %.1f MB, ratio=%.1fx, read=%.0fms\n",
                bp_size / 1e6, raw_int32_bytes / bp_size, bp_read_s * 1000))
    unlink(bp_dir, recursive = TRUE)

    # ══════════ HDF5Array ══════════
    h5_path <- file.path(tmpdir, paste0(gse_id, ".h5"))
    h5_size <- NA; h5_read_s <- NA
    tryCatch({
      if (file.exists(h5_path)) file.remove(h5_path)
      writeTENxMatrix(mat, h5_path, group = "matrix")
      h5_size <- file.size(h5_path)

      h5_times <- numeric(3)
      for (t in 1:3) {
        invisible(gc(FALSE))
        t0 <- proc.time()
        h5_mat <- TENxMatrix(h5_path, group = "matrix")
        full <- as(h5_mat, "dgCMatrix")
        h5_times[t] <- (proc.time() - t0)[3]
        rm(full, h5_mat); invisible(gc(FALSE))
      }
      h5_read_s <- median(h5_times)
      cat(sprintf("  HDF5Array: %.1f MB, ratio=%.1fx, read=%.0fms\n",
                  h5_size / 1e6, raw_int32_bytes / h5_size, h5_read_s * 1000))
      file.remove(h5_path)
    }, error = function(e) {
      cat(sprintf("  HDF5Array FAILED: %s\n", substr(conditionMessage(e), 1, 60)))
    })

    results <- rbind(results, data.frame(
      gse_id = gse_id,
      nrows = nrows, ncols = ncols, nnz = nnz,
      raw_int32_bytes = raw_int32_bytes,
      pz_bytes = as.numeric(survey$pz_bytes[i]),
      pz_ratio = round(raw_int32_bytes / survey$pz_bytes[i], 3),
      rds_bytes = rds_size,
      rds_ratio = round(raw_int32_bytes / rds_size, 3),
      rds_read_s = round(rds_read_s, 4),
      rds_read_mbps = round(raw_int32_bytes / rds_read_s / 1e6, 1),
      bp_bytes = bp_size,
      bp_ratio = round(raw_int32_bytes / bp_size, 3),
      bp_read_s = round(bp_read_s, 4),
      bp_read_mbps = round(raw_int32_bytes / bp_read_s / 1e6, 1),
      h5_bytes = ifelse(is.na(h5_size), NA, h5_size),
      h5_ratio = ifelse(is.na(h5_size), NA, round(raw_int32_bytes / h5_size, 3)),
      h5_read_s = ifelse(is.na(h5_read_s), NA, round(h5_read_s, 4)),
      h5_read_mbps = ifelse(is.na(h5_read_s), NA, round(raw_int32_bytes / h5_read_s / 1e6, 1)),
      stringsAsFactors = FALSE
    ))

    rm(mat); invisible(gc(FALSE))
  }, error = function(e) {
    cat(sprintf("  CRASHED: %s\n", substr(conditionMessage(e), 1, 80)))
  })
}

out_path <- file.path(SCRIPT_DIR, "r_format_benchmarks.csv")
write.csv(results, out_path, row.names = FALSE)
cat(sprintf("\nSaved %d results to %s\n", nrow(results), out_path))
