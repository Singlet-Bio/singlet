#!/usr/bin/env Rscript
## Benchmark R-native read_1pz() across the 20 datasets in r_format_benchmarks.csv
## Appends pz_read_s and pz_read_mbps columns to the CSV.
##
## Usage: ssh b003 "module load r/4.5.2 && cd .../singlepress-format && Rscript benchmark_r_read_1pz.R"

library(singlepress)

QUANT_DIR <- "/mnt/projects/debruinz_project/cellarium/pipeline/quant"
N_WARMUP  <- 1
N_REPS    <- 5

bench <- read.csv("r_format_benchmarks.csv", stringsAsFactors = FALSE)
cat(sprintf("Benchmarking R read_1pz() on %d datasets\n", nrow(bench)))

pz_read_s   <- numeric(nrow(bench))
pz_read_mbps <- numeric(nrow(bench))

for (i in seq_len(nrow(bench))) {
  gse <- bench$gse_id[i]
  pz_path <- file.path(QUANT_DIR, gse, "counts.1pz")

  if (!file.exists(pz_path)) {
    cat(sprintf("  [%d/%d] %s — MISSING %s\n", i, nrow(bench), gse, pz_path))
    pz_read_s[i]   <- NA
    pz_read_mbps[i] <- NA
    next
  }

  # Warmup
  for (w in seq_len(N_WARMUP)) {
    invisible(read_1pz(pz_path, num_threads = 4L))
    gc(verbose = FALSE)
  }

  # Timed runs
  times <- numeric(N_REPS)
  for (r in seq_len(N_REPS)) {
    gc(verbose = FALSE)
    t0 <- proc.time()["elapsed"]
    mat <- read_1pz(pz_path, num_threads = 4L)
    t1 <- proc.time()["elapsed"]
    times[r] <- t1 - t0
    rm(mat)
    gc(verbose = FALSE)
  }

  med_s <- median(times)
  raw_bytes <- bench$raw_int32_bytes[i]
  mbps <- raw_bytes / med_s / 1e6

  pz_read_s[i]   <- med_s
  pz_read_mbps[i] <- mbps
  cat(sprintf("  [%d/%d] %s — %.3f s (%.0f MB/s)  nnz=%.1fM\n",
              i, nrow(bench), gse, med_s, mbps, bench$nnz[i] / 1e6))
}

bench$pz_read_s    <- pz_read_s
bench$pz_read_mbps <- pz_read_mbps

write.csv(bench, "r_format_benchmarks.csv", row.names = FALSE)
cat(sprintf("\nDone. Median R read_1pz: %.3f s (%.0f MB/s)\n",
            median(pz_read_s, na.rm = TRUE),
            median(pz_read_mbps, na.rm = TRUE)))
