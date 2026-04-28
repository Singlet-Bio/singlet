#!/usr/bin/env Rscript
## SinglePress compression frontier figures
##
## Generates: fig4_frontier.pdf
## Usage: cd code/figures && Rscript figures_frontier.R
##
## Fig 4: "Compression frontier"  (7.2 × 5.2, 3×2 = 6 panels)
##   a: .1pz vs entropy bound by dataset (bar chart)
##   b: zstd level vs compression ratio (line + ribbon)
##   c: zstd level vs write throughput (line + ribbon)
##   d: zstd level vs read throughput (line + ribbon)
##   e: alternative codecs Pareto (compression vs decode speed)
##   f: VOCSC encoding advantage (ratio with vs without preprocessing)
##
## Uses: compression_frontier.csv

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)

# ── Data directory ───────────────────────────────────────────────
DATA_DIR <- "../data"

# ── Brand palette ────────────────────────────────────────────────
PAL_1PZ     <- "#009688"
PAL_DEFAULT <- "#009688"
PAL_THEORY  <- "#e53935"

# Codec palette
codec_pal <- c(
  "zstd"   = "#009688",
  "lz4"    = "#78909c",
  "gzip"   = "#90a4ae",
  "brotli" = "#a1887f",
  "bz2"    = "#bcaaa4",
  "lzma"   = "#b0bec5"
)

# ── Theme ────────────────────────────────────────────────────────
theme_sp <- function(base_size = 10) {
  theme_bw(base_size = base_size) %+replace%
    theme(
      aspect.ratio = 1,
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey96"),
      plot.title = element_text(size = base_size, face = "bold", hjust = 0),
      plot.tag = element_text(face = "bold", size = 12),
      legend.position = "bottom",
      legend.key.size = unit(0.35, "cm"),
      legend.margin = margin(0, 0, 0, 0)
    )
}

# ── Load data ────────────────────────────────────────────────────
cf <- read.csv(file.path(DATA_DIR, "compression_frontier.csv"), stringsAsFactors = FALSE)
cat("Loaded:", nrow(cf), "rows from compression_frontier.csv\n")

# Separate experiment types
entropy_rows <- cf %>% filter(experiment == "entropy", codec == "theory")
zstd_rows    <- cf %>% filter(experiment == "zstd_level")
alt_rows     <- cf %>% filter(experiment == "alt_codec_raw")

n_datasets <- length(unique(cf$gse_id))
cat("Datasets:", n_datasets, "\n")
cat("Zstd rows:", nrow(zstd_rows), "Alt codec rows:", nrow(alt_rows), "\n")

# ── Panel A: .1pz vs entropy bound ──────────────────────────────
# For each dataset, compare actual .1pz size to theory minimum
entropy_compare <- entropy_rows %>%
  mutate(
    theory_mb = file_bytes / 1e6,
    actual_mb = pz_default_bytes / 1e6,
    raw_mb = raw_int32_bytes / 1e6,
    overhead_ratio = pz_default_bytes / file_bytes,
    nnz_label = ifelse(nnz >= 1e9, paste0(round(nnz/1e9, 1), "B"),
                ifelse(nnz >= 1e6, paste0(round(nnz/1e6, 0), "M"),
                       paste0(round(nnz/1e3, 0), "K")))
  ) %>%
  arrange(nnz)

# Long format for grouped bar chart
entropy_long <- entropy_compare %>%
  select(gse_id, nnz, nnz_label, theory_mb, actual_mb) %>%
  pivot_longer(cols = c(theory_mb, actual_mb),
               names_to = "type", values_to = "size_mb") %>%
  mutate(type = factor(type, levels = c("actual_mb", "theory_mb"),
                       labels = c(".1pz (actual)", "Entropy bound")))

# Order by nnz
entropy_long$gse_id <- factor(entropy_long$gse_id,
                               levels = entropy_compare$gse_id)

p_a <- ggplot(entropy_compare,
              aes(x = reorder(gse_id, nnz))) +
  geom_col(aes(y = actual_mb), fill = PAL_1PZ, alpha = 0.8, width = 0.6) +
  geom_point(aes(y = theory_mb), color = PAL_THEORY, size = 2, shape = 18) +
  scale_y_log10(labels = label_number(suffix = "")) +
  labs(title = ".1pz size vs. entropy bound",
       x = NULL, y = "File size (MB, log)") +
  theme_sp() +
  theme(axis.text.x = element_text(angle = 50, hjust = 1, size = 6),
        aspect.ratio = 0.7) +
  annotate("text", x = n_datasets * 0.7, y = max(entropy_compare$actual_mb) * 0.5,
           label = "◆ = entropy bound", color = PAL_THEORY, size = 3, hjust = 0)

# ── Panel B: zstd level vs compression ratio ────────────────────
zstd_summary <- zstd_rows %>%
  mutate(level = as.numeric(gsub("zstd-", "", codec))) %>%
  group_by(level) %>%
  summarise(
    med_ratio = median(ratio_vs_int32),
    q25_ratio = quantile(ratio_vs_int32, 0.25),
    q75_ratio = quantile(ratio_vs_int32, 0.75),
    .groups = "drop"
  )

p_b <- ggplot(zstd_summary, aes(x = level)) +
  geom_ribbon(aes(ymin = q25_ratio, ymax = q75_ratio), fill = PAL_1PZ, alpha = 0.2) +
  geom_line(aes(y = med_ratio), color = PAL_1PZ, linewidth = 0.9) +
  geom_point(aes(y = med_ratio), color = PAL_1PZ, size = 2) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  annotate("text", x = 3.5, y = min(zstd_summary$q25_ratio),
           label = "default (3)", size = 3, hjust = 0, color = "grey40") +
  scale_x_continuous(breaks = c(1, 3, 5, 9, 12, 15, 19, 22)) +
  labs(title = "Compression vs. zstd level",
       x = "zstd level", y = "Compression ratio (×)") +
  theme_sp()

# ── Panel C: zstd level vs write throughput ──────────────────────
zstd_write <- zstd_rows %>%
  mutate(level = as.numeric(gsub("zstd-", "", codec))) %>%
  group_by(level) %>%
  summarise(
    med_write = median(write_mbps),
    q25_write = quantile(write_mbps, 0.25),
    q75_write = quantile(write_mbps, 0.75),
    .groups = "drop"
  )

p_c <- ggplot(zstd_write, aes(x = level)) +
  geom_ribbon(aes(ymin = q25_write, ymax = q75_write), fill = PAL_1PZ, alpha = 0.2) +
  geom_line(aes(y = med_write), color = PAL_1PZ, linewidth = 0.9) +
  geom_point(aes(y = med_write), color = PAL_1PZ, size = 2) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  scale_x_continuous(breaks = c(1, 3, 5, 9, 12, 15, 19, 22)) +
  labs(title = "Write throughput vs. zstd level",
       x = "zstd level", y = "Write throughput (MB/s)") +
  theme_sp()

# ── Panel D: zstd level vs read throughput ───────────────────────
zstd_read <- zstd_rows %>%
  mutate(level = as.numeric(gsub("zstd-", "", codec))) %>%
  group_by(level) %>%
  summarise(
    med_read = median(read_mbps),
    q25_read = quantile(read_mbps, 0.25),
    q75_read = quantile(read_mbps, 0.75),
    .groups = "drop"
  )

p_d <- ggplot(zstd_read, aes(x = level)) +
  geom_ribbon(aes(ymin = q25_read, ymax = q75_read), fill = PAL_1PZ, alpha = 0.2) +
  geom_line(aes(y = med_read), color = PAL_1PZ, linewidth = 0.9) +
  geom_point(aes(y = med_read), color = PAL_1PZ, size = 2) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  scale_x_continuous(breaks = c(1, 3, 5, 9, 12, 15, 19, 22)) +
  labs(title = "Read throughput vs. zstd level",
       x = "zstd level", y = "Read throughput (MB/s)") +
  theme_sp()

# ── Panel E: Alternative codecs Pareto ───────────────────────────
if (nrow(alt_rows) > 0) {
  # Extract base codec name and level
  alt_summary <- alt_rows %>%
    mutate(
      base_codec = gsub("-[0-9]+$", "", codec),
      clevel = as.numeric(gsub(".*-", "", codec))
    ) %>%
    group_by(base_codec, clevel, codec) %>%
    summarise(
      med_ratio = median(ratio_vs_int32),
      med_decode = median(read_mbps),
      .groups = "drop"
    )
  
  # Add SinglePress zstd-3 point for comparison
  sp_point <- zstd_rows %>%
    filter(codec == "zstd-3") %>%
    summarise(
      base_codec = "VOCSC+zstd",
      clevel = 3,
      codec = "VOCSC+zstd-3",
      med_ratio = median(ratio_vs_int32),
      med_decode = median(read_mbps)
    )
  
  alt_plot <- bind_rows(alt_summary, sp_point)
  
  p_e <- ggplot(alt_plot, aes(x = med_decode, y = med_ratio, 
                               color = base_codec)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_text(aes(label = clevel), size = 2.5, vjust = -0.8, show.legend = FALSE) +
    # Highlight the SinglePress point
    geom_point(data = filter(alt_plot, base_codec == "VOCSC+zstd"),
               size = 5, shape = 18, show.legend = FALSE) +
    scale_x_log10(labels = label_comma()) +
    scale_color_manual(values = c(codec_pal, "VOCSC+zstd" = PAL_1PZ),
                       name = "Backend codec") +
    labs(title = "Compression–decode Pareto frontier",
         x = "Decode throughput (MB/s, log)", y = "Compression ratio (×)") +
    theme_sp() +
    theme(legend.position = "right")
} else {
  p_e <- ggplot() + annotate("text", x = 0.5, y = 0.5, label = "Awaiting data") + theme_void()
}

# ── Panel F: VOCSC encoding advantage ────────────────────────────
# Compare: VOCSC+zstd-3 compression ratio vs best raw codec per dataset
vocsc_advantage <- zstd_rows %>%
  filter(codec == "zstd-3") %>%
  select(gse_id, nnz, vocsc_ratio = ratio_vs_int32, vocsc_bytes = file_bytes)

if (nrow(alt_rows) > 0) {
  # Best raw codec per dataset (highest compression ratio)
  best_raw <- alt_rows %>%
    group_by(gse_id) %>%
    slice_max(ratio_vs_int32, n = 1) %>%
    ungroup() %>%
    select(gse_id, raw_ratio = ratio_vs_int32, raw_codec = codec, raw_bytes = file_bytes)
  
  advantage <- inner_join(vocsc_advantage, best_raw, by = "gse_id") %>%
    mutate(
      vocsc_factor = vocsc_ratio / raw_ratio,
      nnz_M = nnz / 1e6
    )
  
  p_f <- ggplot(advantage, aes(x = nnz_M, y = vocsc_factor)) +
    geom_point(color = PAL_1PZ, size = 2.5, alpha = 0.7) +
    geom_smooth(method = "loess", color = PAL_1PZ, fill = PAL_1PZ, 
                alpha = 0.15, linewidth = 0.7, se = TRUE) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey60") +
    scale_x_log10(labels = label_comma()) +
    labs(title = "VOCSC encoding advantage",
         x = "Nonzeros (millions, log)", 
         y = "VOCSC+zstd / best raw codec ratio") +
    theme_sp()
} else {
  p_f <- ggplot() + annotate("text", x = 0.5, y = 0.5, label = "Awaiting data") + theme_void()
}

# ── Compose: 3×2 ────────────────────────────────────────────────
fig <- (p_a | p_b) / (p_c | p_d) / (p_e | p_f) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave("fig4_frontier.pdf", fig, width = 7.2, height = 7.8, device = cairo_pdf)
cat("Wrote fig4_frontier.pdf\n")

# ── Print summary stats for manuscript ───────────────────────────
cat("\n══ Compression Frontier Summary ══\n")

# Entropy analysis
if (nrow(entropy_rows) > 0) {
  cat("Entropy bound comparisons:\n")
  for (i in seq_len(nrow(entropy_compare))) {
    row <- entropy_compare[i, ]
    cat(sprintf("  %s: .1pz=%.1fMB, theory=%.1fMB (%.2f× theory)\n",
                row$gse_id, row$actual_mb, row$theory_mb, row$overhead_ratio))
  }
  med_overhead <- median(entropy_compare$overhead_ratio)
  cat(sprintf("  Median .1pz/theory ratio: %.2f×\n", med_overhead))
  below <- sum(entropy_compare$overhead_ratio < 1.0)
  cat(sprintf("  Datasets beating entropy bound: %d/%d\n", below, nrow(entropy_compare)))
}

# Zstd level impact
if (nrow(zstd_summary) > 0) {
  base <- zstd_summary$med_ratio[zstd_summary$level == 1]
  best <- max(zstd_summary$med_ratio)
  cat(sprintf("\nZstd level 1→22: ratio %.1f× → %.1f× (%.1f%% improvement)\n",
              base, best, (best/base - 1) * 100))
  base_w <- zstd_write$med_write[zstd_write$level == 1]
  slow_w <- min(zstd_write$med_write)
  cat(sprintf("Write throughput: %.0f → %.0f MB/s (%.1f× decrease)\n",
              base_w, slow_w, base_w/slow_w))
  med_r_3 <- zstd_read$med_read[zstd_read$level == 3]
  cat(sprintf("Read throughput at level 3: %.0f MB/s (constant across levels)\n", med_r_3))
}

cat("\nAll figures written.\n")
