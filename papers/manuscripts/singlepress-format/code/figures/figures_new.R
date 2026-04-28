#!/usr/bin/env Rscript
## SinglePress manuscript — new benchmark figures
## Generates: fig5_ecosystem.pdf (scATAC + Census + DataLoader comparison)
##
## Usage: cd code/figures && Rscript figures_new.R
##
## Depends on data files:
##   ../data/scatac_bench.csv         (scATAC + scRNA comparison)
##   ../data/census_bench.csv         (Census tissue slice benchmarks)
##   ../data/bpcells_compression_bench.csv  (existing BPCells comparison)
##   ../data/r_format_benchmarks.csv  (existing R format comparison with speeds)

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# ── Data directory ───────────────────────────────────────────────
DATA_DIR <- "../data"

# ── Brand palette ────────────────────────────────────────────────
PAL_1PZ     <- "#009688"
PAL_H5AD    <- "#90a4ae"
PAL_BPCELLS <- "#b0bec5"
PAL_10X     <- "#78909c"
PAL_TILEDB  <- "#7986cb"
PAL_ATAC    <- "#ef6c00"
PAL_RNA     <- "#009688"

format_pal <- c(
  ".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD, "BPCells" = PAL_BPCELLS,
  "10x H5" = PAL_10X, "TileDB-SOMA" = PAL_TILEDB
)

dtype_pal <- c(
  "scATAC" = PAL_ATAC, "scRNA" = PAL_RNA
)

# ── Theme ────────────────────────────────────────────────────────
theme_sp <- function(base_size = 10) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey92", linewidth = 0.25),
      panel.background   = element_blank(),
      plot.background    = element_blank(),
      axis.line          = element_line(color = "grey40", linewidth = 0.25),
      axis.ticks         = element_line(color = "grey40", linewidth = 0.2),
      axis.ticks.length  = unit(0.06, "cm"),
      strip.text         = element_text(face = "bold", size = rel(1)),
      plot.title         = element_text(face = "bold", size = rel(0.85), hjust = 0),
      legend.key.size    = unit(0.22, "cm"),
      legend.background  = element_blank(),
      legend.text        = element_text(size = 7),
      plot.margin        = margin(2, 4, 2, 2),
      aspect.ratio       = 1
    )
}


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

# scATAC benchmark
scatac <- tryCatch(
  read.csv(file.path(DATA_DIR, "scatac_bench.csv"), stringsAsFactors = FALSE),
  error = function(e) { cat("Warning: scatac_bench.csv not found\n"); NULL }
)

# Census benchmark
census <- tryCatch(
  read.csv(file.path(DATA_DIR, "census_bench.csv"), stringsAsFactors = FALSE),
  error = function(e) { cat("Warning: census_bench.csv not found\n"); NULL }
)

# BPCells compression (existing)
bpcells <- tryCatch(
  read.csv(file.path(DATA_DIR, "bpcells_compression_bench.csv"), stringsAsFactors = FALSE),
  error = function(e) { cat("Warning: bpcells_compression_bench.csv not found\n"); NULL }
)

# R format benchmarks (existing, has BPCells read speeds)
r_bench <- tryCatch(
  read.csv(file.path(DATA_DIR, "r_format_benchmarks.csv"), stringsAsFactors = FALSE),
  error = function(e) { cat("Warning: r_format_benchmarks.csv not found\n"); NULL }
)


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Ecosystem benchmarks (3×2 = 6 panels)
# ═══════════════════════════════════════════════════════════════════

panels <- list()

# ── (a) scATAC vs scRNA: compression ratio ──────────────────────
if (!is.null(scatac)) {
  sc <- scatac %>%
    mutate(dtype = case_when(
      grepl("ATAC|atac", data_type) ~ "scATAC",
      TRUE ~ "scRNA"
    ))

  panels$a <- ggplot(sc, aes(x = nnz / 1e6, y = pz_ratio, color = dtype)) +
    geom_point(size = 2.5, alpha = 0.8) +
    scale_color_manual(values = dtype_pal, name = "Data type") +
    scale_x_log10(labels = label_comma()) +
    labs(x = "Nonzeros (millions)", y = "Compression ratio",
         title = "scATAC vs scRNA compression") +
    theme_sp() +
    theme(legend.position = c(0.7, 0.25),
          panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

  cat(sprintf("Panel a: scATAC median ratio = %.1f×, scRNA median ratio = %.1f×\n",
              median(sc$pz_ratio[sc$dtype == "scATAC"]),
              median(sc$pz_ratio[sc$dtype == "scRNA"])))
}

# ── (b) scATAC vs scRNA: bytes per nonzero ──────────────────────
if (!is.null(scatac)) {
  panels$b <- ggplot(sc, aes(x = entropy_bits, y = pz_bytes_per_nnz, color = dtype)) +
    geom_point(size = 2.5, alpha = 0.8) +
    scale_color_manual(values = dtype_pal, name = "Data type") +
    labs(x = "Value entropy (bits)", y = "Bytes per nonzero",
         title = "Compression efficiency") +
    theme_sp() +
    theme(legend.position = "none",
          panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))
}

# ── (c) BPCells head-to-head: compression + speed ──────────────
# Use bpcells_compression_bench.csv (has proper uint32+compress=TRUE BPCells values)
if (!is.null(bpcells)) {
  bp_cmp <- bpcells %>%
    select(gse_id, nnz, pz_bytes, pz_ratio, bp_comp_bytes, bp_comp_ratio,
           bp_comp_read_s, bp_comp_read_mbps) %>%
    filter(!is.na(bp_comp_ratio), !is.na(pz_ratio))

  bp_long <- bp_cmp %>%
    pivot_longer(
      cols = c(pz_ratio, bp_comp_ratio),
      names_to = "format",
      values_to = "ratio"
    ) %>%
    mutate(format = ifelse(format == "pz_ratio", ".1pz", "BPCells"))

  panels$c <- ggplot(bp_long, aes(x = nnz / 1e6, y = ratio, color = format)) +
    geom_line(aes(group = gse_id), color = "grey80", linewidth = 0.3) +
    geom_point(size = 1.5, alpha = 0.8) +
    scale_color_manual(values = c(".1pz" = PAL_1PZ, "BPCells" = PAL_BPCELLS),
                       name = "Format") +
    scale_x_log10(labels = label_comma()) +
    labs(x = "Nonzeros (millions)", y = "Compression ratio",
         title = ".1pz vs BPCells (20 datasets)") +
    theme_sp() +
    theme(legend.position = c(0.7, 0.25),
          panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

  cat(sprintf("Panel c: .1pz median = %.1f×, BPCells median = %.1f×\n",
              median(bp_cmp$pz_ratio), median(bp_cmp$bp_comp_ratio)))
}

# ── (d) BPCells head-to-head: read throughput ───────────────────
if (!is.null(bpcells)) {
  bp_speed <- bp_cmp %>%
    filter(!is.na(bp_comp_read_mbps)) %>%
    select(gse_id, nnz, pz_read_mbps = pz_ratio, bp_read_mbps = bp_comp_read_mbps)

  # Compute .1pz read MB/s from r_format_benchmarks if available
  if (!is.null(r_bench)) {
    pz_read <- r_bench %>% select(gse_id, pz_read_mbps)
    bp_speed2 <- bpcells %>%
      filter(!is.na(bp_comp_read_mbps)) %>%
      select(gse_id, nnz, bp_comp_read_mbps) %>%
      left_join(pz_read, by = "gse_id") %>%
      filter(!is.na(pz_read_mbps))

    bp_speed_long <- bp_speed2 %>%
      pivot_longer(
        cols = c(pz_read_mbps, bp_comp_read_mbps),
        names_to = "format",
        values_to = "read_mbps"
      ) %>%
      mutate(format = ifelse(format == "pz_read_mbps", ".1pz", "BPCells"))

    panels$d <- ggplot(bp_speed_long, aes(x = nnz / 1e6, y = read_mbps, color = format)) +
      geom_line(aes(group = gse_id), color = "grey80", linewidth = 0.3) +
      geom_point(size = 1.5, alpha = 0.8) +
      scale_color_manual(values = c(".1pz" = PAL_1PZ, "BPCells" = PAL_BPCELLS),
                         name = "Format") +
      scale_x_log10(labels = label_comma()) +
      labs(x = "Nonzeros (millions)", y = "Read throughput (MB/s)",
           title = "Read speed: .1pz vs BPCells (R)") +
      theme_sp() +
      theme(legend.position = c(0.7, 0.75),
            panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

    cat(sprintf("Panel d: .1pz median read = %.0f MB/s, BPCells median read = %.0f MB/s\n",
                median(bp_speed2$pz_read_mbps, na.rm = TRUE),
                median(bp_speed2$bp_comp_read_mbps, na.rm = TRUE)))
  }
}

# ── (e) Census: storage comparison ──────────────────────────────
if (!is.null(census)) {
  census_storage <- census %>%
    select(name, pz_bytes, h5ad_bytes, raw_bytes, pz_ratio, h5ad_ratio) %>%
    pivot_longer(
      cols = c(pz_bytes, h5ad_bytes),
      names_to = "format",
      values_to = "bytes"
    ) %>%
    mutate(format = ifelse(format == "pz_bytes", ".1pz", "H5AD"),
           mb = bytes / 1e6)

  panels$e <- ggplot(census_storage, aes(x = name, y = mb, fill = format)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD),
                      name = "Format") +
    labs(x = NULL, y = "File size (MB)",
         title = "Census tissue slices") +
    theme_sp() +
    theme(legend.position = c(0.3, 0.8),
          axis.text.x = element_text(angle = 30, hjust = 1))
}

# ── (f) Census: read speed comparison ───────────────────────────
if (!is.null(census)) {
  census_read <- census %>%
    select(name, pz_read_s, h5ad_read_s, tiledb_read_s) %>%
    pivot_longer(
      cols = c(pz_read_s, h5ad_read_s, tiledb_read_s),
      names_to = "format",
      values_to = "read_s"
    ) %>%
    mutate(format = case_when(
      format == "pz_read_s" ~ ".1pz",
      format == "h5ad_read_s" ~ "H5AD",
      format == "tiledb_read_s" ~ "TileDB-SOMA"
    ))

  panels$f <- ggplot(census_read, aes(x = name, y = read_s, fill = format)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD,
                                  "TileDB-SOMA" = PAL_TILEDB),
                      name = "Format") +
    scale_y_log10() +
    labs(x = NULL, y = "Read time (seconds, log scale)",
         title = "Census read speed") +
    theme_sp() +
    theme(legend.position = c(0.35, 0.8),
          axis.text.x = element_text(angle = 30, hjust = 1))
}

# ── Assemble Figure 5 ───────────────────────────────────────────
if (length(panels) >= 2) {
  # Fill missing panels with empty plots
  for (nm in c("a", "b", "c", "d", "e", "f")) {
    if (is.null(panels[[nm]])) {
      panels[[nm]] <- ggplot() + theme_void() +
        annotate("text", x = 0.5, y = 0.5, label = paste("Panel", nm, "\n(data pending)"),
                 size = 4, color = "grey50")
    }
  }

  fig5 <- (panels$a | panels$b | panels$c) /
           (panels$d | panels$e | panels$f) +
    plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 12))

  ggsave("fig5_ecosystem.pdf", fig5, width = 7.2, height = 5.2, device = cairo_pdf)
  cat("Wrote fig5_ecosystem.pdf\n")
} else {
  cat("Not enough data to generate fig5; need at least scatac_bench.csv\n")
}


# ═══════════════════════════════════════════════════════════════════
# Print summary statistics for manuscript
# ═══════════════════════════════════════════════════════════════════
cat("\n══ Manuscript statistics ══\n")

if (!is.null(scatac)) {
  atac <- scatac %>% filter(grepl("ATAC|atac", data_type))
  rna  <- scatac %>% filter(data_type == "scRNA")
  cat(sprintf("\nscATAC (%d datasets):\n", nrow(atac)))
  cat(sprintf("  Compression: %.1f× median (range %.1f–%.1f×)\n",
              median(atac$pz_ratio), min(atac$pz_ratio), max(atac$pz_ratio)))
  cat(sprintf("  Bytes/nnz: %.3f median\n", median(atac$pz_bytes_per_nnz)))
  cat(sprintf("  Entropy: %.2f bits median\n", median(atac$entropy_bits)))
  cat(sprintf("  Frac(val=1): %.1f%% median\n", median(atac$frac_val_1) * 100))

  cat(sprintf("\nscRNA (%d datasets):\n", nrow(rna)))
  cat(sprintf("  Compression: %.1f× median (range %.1f–%.1f×)\n",
              median(rna$pz_ratio), min(rna$pz_ratio), max(rna$pz_ratio)))
  cat(sprintf("  Bytes/nnz: %.3f median\n", median(rna$pz_bytes_per_nnz)))
}

if (!is.null(bpcells)) {
  bp_ok <- bpcells %>% filter(!is.na(bp_comp_ratio))
  cat(sprintf("\nBPCells head-to-head (%d datasets):\n", nrow(bp_ok)))
  cat(sprintf("  .1pz compression: %.1f× median\n", median(bp_ok$pz_ratio)))
  cat(sprintf("  BPCells compression: %.1f× median\n", median(bp_ok$bp_comp_ratio)))
  cat(sprintf("  .1pz advantage: %.1f×\n", median(bp_ok$pz_ratio) / median(bp_ok$bp_comp_ratio)))
  cat(sprintf("  BPCells read: %.0f MB/s median\n", median(bp_ok$bp_comp_read_mbps, na.rm = TRUE)))
  if (!is.null(r_bench)) {
    cat(sprintf("  .1pz read (R): %.0f MB/s median\n", median(r_bench$pz_read_mbps, na.rm = TRUE)))
  }
}

if (!is.null(census)) {
  cat(sprintf("\nCensus benchmarks (%d slices):\n", nrow(census)))
  for (i in seq_len(nrow(census))) {
    r <- census[i, ]
    cat(sprintf("  %s: .1pz=%.1fMB (%.1f×) H5AD=%.1fMB (%.1f×)\n",
                r$name, r$pz_bytes / 1e6, r$pz_ratio,
                r$h5ad_bytes / 1e6, r$h5ad_ratio))
    cat(sprintf("    Read: .1pz=%.3fs H5AD=%.3fs TileDB=%.1fs (TileDB %.0f× slower)\n",
                r$pz_read_s, r$h5ad_read_s, r$tiledb_read_s,
                r$tiledb_read_s / r$pz_read_s))
  }
}

cat("\nDone.\n")
