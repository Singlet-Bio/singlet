#!/usr/bin/env Rscript
# Investigate BPCells compression properly — test different modes and check actual sizes
# Usage: ssh <node> "module load r/4.5.2 && Rscript benchmark_bpcells_v3.R"

suppressMessages({
  library(Matrix)
  library(BPCells)
  library(reticulate)
})

QUANT_DIR <- "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
SCRIPT_DIR <- "/mnt/home/debruinz/Singlet-AI/papers/manuscripts/singlepress-format"

use_python("/usr/bin/python3", required = TRUE)

survey <- read.csv(file.path(SCRIPT_DIR, "all_datasets_survey.csv"), stringsAsFactors = FALSE)
survey <- survey[survey$nnz > 1e6 & survey$nnz < 200e6, ]
survey <- survey[order(survey$nnz), ]

# 20 stratified datasets
n_sample <- min(20L, nrow(survey))
indices <- round(seq(1, nrow(survey), length.out = n_sample))
survey <- survey[indices, ]
cat(sprintf("Selected %d datasets (nnz: %s – %s)\n\n",
            n_sample,
            format(min(survey$nnz), big.mark = ","),
            format(max(survey$nnz), big.mark = ",")))

results <- data.frame()

for (i in seq_len(nrow(survey))) {
  gse_id <- survey$gse_id[i]
  pz_path <- file.path(QUANT_DIR, gse_id, "counts.1pz")
  nnz <- survey$nnz[i]

  cat(sprintf("[%d/%d] %s (%s nnz)\n", i, n_sample, gse_id, format(nnz, big.mark = ",")))

  tryCatch({
    # Read via singlepress
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

    data_arr <- as.vector(as.numeric(py$`_data`))
    indices_arr <- as.vector(as.integer(py$`_indices`))
    indptr_arr <- as.vector(as.integer(py$`_indptr`))
    nrows <- as.integer(py$`_nrows`)
    ncols <- as.integer(py$`_ncols`)
    nnz_actual <- as.integer(py$`_nnz`)

    mat <- new("dgCMatrix",
               i = indices_arr, p = indptr_arr, x = data_arr,
               Dim = c(nrows, ncols))

    raw_int32_bytes <- length(indptr_arr) * 4 + nnz_actual * 4 + nnz_actual * 4
    rm(data_arr, indices_arr, indptr_arr)

    tmpdir <- tempdir()

    # === BPCells: uncompressed (compress=FALSE) ===
    bp_dir_unc <- file.path(tmpdir, paste0(gse_id, "_bp_unc"))
    if (dir.exists(bp_dir_unc)) unlink(bp_dir_unc, recursive = TRUE)
    bp_in <- convert_matrix_type(mat, type = "uint32_t")
    write_matrix_dir(bp_in, bp_dir_unc, compress = FALSE)
    bp_unc_size <- sum(file.size(list.files(bp_dir_unc, full.names = TRUE, recursive = TRUE)))

    # List the files BPCells creates
    bp_files <- list.files(bp_dir_unc, recursive = TRUE)
    bp_file_sizes <- file.size(list.files(bp_dir_unc, full.names = TRUE, recursive = TRUE))
    cat(sprintf("  BPCells uncompressed files:\n"))
    for (j in seq_along(bp_files)) {
      cat(sprintf("    %s: %s bytes\n", bp_files[j], format(bp_file_sizes[j], big.mark = ",")))
    }
    unlink(bp_dir_unc, recursive = TRUE)

    # === BPCells: compressed (compress=TRUE) ===
    bp_dir_comp <- file.path(tmpdir, paste0(gse_id, "_bp_comp"))
    if (dir.exists(bp_dir_comp)) unlink(bp_dir_comp, recursive = TRUE)
    write_matrix_dir(bp_in, bp_dir_comp, compress = TRUE)
    bp_comp_size <- sum(file.size(list.files(bp_dir_comp, full.names = TRUE, recursive = TRUE)))

    bp_files2 <- list.files(bp_dir_comp, recursive = TRUE)
    bp_file_sizes2 <- file.size(list.files(bp_dir_comp, full.names = TRUE, recursive = TRUE))
    cat(sprintf("  BPCells compressed files:\n"))
    for (j in seq_along(bp_files2)) {
      cat(sprintf("    %s: %s bytes\n", bp_files2[j], format(bp_file_sizes2[j], big.mark = ",")))
    }

    # Read benchmark for compressed
    bp_times <- numeric(3)
    for (t in 1:3) {
      invisible(gc(FALSE))
      t0 <- proc.time()
      bp_mat2 <- open_matrix_dir(bp_dir_comp)
      full <- as(bp_mat2, "dgCMatrix")
      bp_times[t] <- (proc.time() - t0)[3]
      rm(full, bp_mat2); invisible(gc(FALSE))
    }
    bp_comp_read_s <- median(bp_times)
    unlink(bp_dir_comp, recursive = TRUE)

    # === BPCells: HDF5 backend ===
    bp_h5_path <- file.path(tmpdir, paste0(gse_id, "_bp.h5"))
    if (file.exists(bp_h5_path)) file.remove(bp_h5_path)
    tryCatch({
      write_matrix_hdf5(bp_in, bp_h5_path, compress = TRUE)
      bp_h5_size <- file.size(bp_h5_path)
      cat(sprintf("  BPCells HDF5: %s bytes (ratio=%.2fx)\n",
                  format(bp_h5_size, big.mark = ","),
                  raw_int32_bytes / bp_h5_size))
      file.remove(bp_h5_path)
    }, error = function(e) {
      cat(sprintf("  BPCells HDF5 FAILED: %s\n", conditionMessage(e)))
      bp_h5_size <<- NA
    })

    pz_bytes <- as.numeric(survey$pz_bytes[i])

    cat(sprintf("  SUMMARY: raw=%s, pz=%s (%.1fx), bp_unc=%s (%.2fx), bp_comp=%s (%.2fx)\n",
                format(raw_int32_bytes, big.mark = ","),
                format(pz_bytes, big.mark = ","), raw_int32_bytes / pz_bytes,
                format(bp_unc_size, big.mark = ","), raw_int32_bytes / bp_unc_size,
                format(bp_comp_size, big.mark = ","), raw_int32_bytes / bp_comp_size))

    results <- rbind(results, data.frame(
      gse_id = gse_id, nrows = nrows, ncols = ncols, nnz = nnz_actual,
      raw_int32_bytes = raw_int32_bytes,
      pz_bytes = pz_bytes,
      pz_ratio = round(raw_int32_bytes / pz_bytes, 3),
      bp_unc_bytes = bp_unc_size,
      bp_unc_ratio = round(raw_int32_bytes / bp_unc_size, 3),
      bp_comp_bytes = bp_comp_size,
      bp_comp_ratio = round(raw_int32_bytes / bp_comp_size, 3),
      bp_comp_read_s = round(bp_comp_read_s, 4),
      bp_comp_read_mbps = round(raw_int32_bytes / bp_comp_read_s / 1e6, 1),
      bp_h5_bytes = ifelse(exists("bp_h5_size") && !is.na(bp_h5_size), bp_h5_size, NA),
      bp_h5_ratio = ifelse(exists("bp_h5_size") && !is.na(bp_h5_size),
                           round(raw_int32_bytes / bp_h5_size, 3), NA),
      stringsAsFactors = FALSE
    ))

    rm(mat, bp_in); invisible(gc(FALSE))
  }, error = function(e) {
    cat(sprintf("  CRASHED: %s\n", conditionMessage(e)))
  })
}

out_path <- file.path(SCRIPT_DIR, "bpcells_compression_bench.csv")
write.csv(results, out_path, row.names = FALSE)
cat(sprintf("\nSaved %d results to %s\n", nrow(results), out_path))
