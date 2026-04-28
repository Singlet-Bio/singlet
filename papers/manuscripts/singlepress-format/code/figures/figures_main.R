#!/usr/bin/env Rscript
## SinglePress manuscript figures
## Generates: fig1_compression.pdf, fig2_performance.pdf, fig3_io.pdf, figS1_structure.pdf
##
## Usage: cd code/figures && Rscript figures_main.R
##
## Design:
##   - 1:1 aspect ratio for all panels
##   - Descriptive panel titles
##   - Bold lowercase panel tags (a, b, c, ...)
##   - Consistent format palette throughout
##   - 10pt base text
##
## Fig 1: "File format comparison" (7.2 × 5.2, 3×2 = 6 panels)
## Fig 2: "Performance"            (7.2 × 5.2, 3×2 = 6 panels)
## Fig 3: "I/O throughput"         (3.5 × 4.0, 2×2 = 4 panels)
## Fig S1: "Statistical structure"  (7.2 × 5.2, 3×2 = 6 panels)

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
PAL_NPZ     <- "#a1887f"
PAL_RDS     <- "#bcaaa4"

format_pal <- c(
  ".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD, "BPCells" = PAL_BPCELLS,
  "10x H5" = PAL_10X, "npz" = PAL_NPZ, "RDS" = PAL_RDS
)

species_pal <- c(
  "Human" = "#546e7a", "Mouse" = "#78909c", "Zebrafish" = "#80cbc4",
  "Rat" = "#a1887f", "Fly" = "#b0bec5", "Macaque" = "#90a4ae",
  "Chicken" = "#bcaaa4", "Frog" = "#80deea", "Other" = "#e0e0e0"
)

# ── Helpers ──────────────────────────────────────────────────────
species_recode <- function(x) {
  case_when(
    grepl("Homo|sapiens", x, ignore.case = TRUE)            ~ "Human",
    grepl("Mus|musculus", x, ignore.case = TRUE)             ~ "Mouse",
    grepl("Danio|rerio", x, ignore.case = TRUE)              ~ "Zebrafish",
    grepl("Rattus|norvegicus", x, ignore.case = TRUE)        ~ "Rat",
    grepl("Drosophila|melanogaster", x, ignore.case = TRUE)  ~ "Fly",
    grepl("Macaca", x, ignore.case = TRUE)                   ~ "Macaque",
    grepl("Gallus|gallus", x, ignore.case = TRUE)            ~ "Chicken",
    grepl("Xenopus|laevis", x, ignore.case = TRUE)           ~ "Frog",
    TRUE ~ "Other"
  )
}

protocol_recode <- function(x) {
  case_when(
    grepl("10xv3_5prime|10x.*5", x, ignore.case = TRUE)   ~ "10x 5'",
    grepl("10xv4", x, ignore.case = TRUE)                  ~ "10x v4",
    grepl("10xv3", x, ignore.case = TRUE)                  ~ "10x v3",
    grepl("10xv2", x, ignore.case = TRUE)                  ~ "10x v2",
    grepl("10x_suspect", x, ignore.case = TRUE)            ~ "10x inf.",
    grepl("dropseq", x, ignore.case = TRUE)                ~ "Drop-seq",
    grepl("indrop", x, ignore.case = TRUE)                 ~ "inDrop",
    grepl("seqwell", x, ignore.case = TRUE)                ~ "Seq-Well",
    grepl("dnbelab", x, ignore.case = TRUE)                ~ "DNBelab",
    grepl("rhapsody", x, ignore.case = TRUE)               ~ "Rhapsody",
    grepl("10x_multiome", x, ignore.case = TRUE)           ~ "Multiome",
    is.na(x) | x == "" | grepl("unknown|empty", x, ignore.case = TRUE) ~ "Unknown",
    TRUE ~ x
  )
}

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
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════

survey <- read.csv(file.path(DATA_DIR, "all_datasets_survey.csv"), stringsAsFactors = FALSE) %>%
  mutate(sp = species_recode(species), proto = protocol_recode(protocol),
         bits_per_nz = (pz_bytes * 8) / nnz, bytes_per_nz = pz_bytes / nnz) %>%
  filter(nnz > 0, ratio > 0)
survey_main <- survey %>% filter(nnz > 1e6)

io <- read.csv(file.path(DATA_DIR, "io_benchmarks.csv"), stringsAsFactors = FALSE) %>% filter(nnz > 1e6)
r_bench <- read.csv(file.path(DATA_DIR, "r_format_benchmarks.csv"), stringsAsFactors = FALSE)
bpcells <- read.csv(file.path(DATA_DIR, "bpcells_compression_bench.csv"), stringsAsFactors = FALSE)
read_tp <- read.csv(file.path(DATA_DIR, "read_throughput.csv"), stringsAsFactors = FALSE) %>%
  mutate(sp = species_recode(species)) %>% filter(nnz > 1e6)
write_bench <- read.csv(file.path(DATA_DIR, "write_benchmarks.csv"), stringsAsFactors = FALSE)
r_write_bench <- tryCatch(
  read.csv(file.path(DATA_DIR, "r_write_benchmarks.csv"), stringsAsFactors = FALSE),
  error = function(e) NULL
)
vdist <- read.csv(file.path(DATA_DIR, "value_distributions.csv"), stringsAsFactors = FALSE)
zinb <- read.csv(file.path(DATA_DIR, "zinb_data.csv"), stringsAsFactors = FALSE) %>%
  mutate(sp = species_recode(species))
ops <- read.csv(file.path(DATA_DIR, "operations_benchmark.csv"), stringsAsFactors = FALSE)

# Threading: prefer v2 (100 reps) if available
threading <- tryCatch(
  read.csv(file.path(DATA_DIR, "threading_benchmark_v2.csv"), stringsAsFactors = FALSE),
  error = function(e) read.csv(file.path(DATA_DIR, "threading_benchmarks.csv"), stringsAsFactors = FALSE)
)

# GPU: prefer v2 (expanded H100) if available
gpu_bench <- tryCatch({
  gb <- read.csv(file.path(DATA_DIR, "gpu_benchmark_v2.csv"), stringsAsFactors = FALSE)
  if (nrow(gb) >= 10) gb else stop("too few rows")
}, error = function(e) {
  read.csv(file.path(DATA_DIR, "format_pytorch_bench.csv"), stringsAsFactors = FALSE)
})

cat(sprintf("Loaded: survey=%d (main=%d), io=%d, threading=%d, gpu=%d, ops=%d\n",
    nrow(survey), nrow(survey_main), nrow(io), nrow(threading), nrow(gpu_bench), nrow(ops)))


# ═══════════════════════════════════════════════════════════════════
# Build tidy multi-format data
# ═══════════════════════════════════════════════════════════════════

fmt_tidy <- bind_rows(
  io %>% transmute(gse_id, format = ".1pz", nnz,
                   ratio = raw_int32_bytes / pz_bytes,
                   read_mbps = read_1pz_mbps),
  io %>% filter(!is.na(h5ad_bytes), h5ad_bytes > 0) %>%
    transmute(gse_id, format = "H5AD", nnz,
              ratio = raw_int32_bytes / h5ad_bytes,
              read_mbps = read_h5ad_mbps),
  io %>% filter(!is.na(npz_bytes), npz_bytes > 0) %>%
    transmute(gse_id, format = "npz", nnz,
              ratio = raw_int32_bytes / npz_bytes,
              read_mbps = read_npz_mbps),
  io %>% filter(!is.na(h5_bytes), h5_bytes > 0) %>%
    transmute(gse_id, format = "10x H5", nnz,
              ratio = raw_int32_bytes / h5_bytes,
              read_mbps = read_h5_mbps),
  r_bench %>% filter(nnz > 1e6) %>%
    transmute(gse_id, format = "RDS", nnz,
              ratio = rds_ratio,
              read_mbps = rds_read_mbps),
  bpcells %>% filter(nnz > 1e6) %>%
    transmute(gse_id, format = "BPCells", nnz,
              ratio = bp_comp_ratio,
              read_mbps = bp_comp_read_mbps)
) %>% distinct(gse_id, format, .keep_all = TRUE)

# Sort formats by increasing median compression ratio (.1pz last = rightmost)
fmt_order <- fmt_tidy %>%
  filter(!is.na(ratio), ratio > 0) %>%
  group_by(format) %>%
  summarise(med_ratio = median(ratio), .groups = "drop") %>%
  arrange(med_ratio)
fmt_tidy$format <- factor(fmt_tidy$format, levels = fmt_order$format)

# Prediction model
fit <- lm(pz_bytes ~ nnz, data = survey_main)
survey_main$pred <- predict(fit)
r2 <- summary(fit)$r.squared
bpnz <- coef(fit)["nnz"]

# Value distribution global summary
vdist_main <- vdist %>%
  filter(value <= 30, value >= 1) %>%
  group_by(gse_id) %>%
  mutate(frac_norm = count / sum(count)) %>%
  ungroup()

vdist_global <- vdist_main %>%
  group_by(value) %>%
  summarise(mean_frac = mean(frac_norm, na.rm = TRUE),
            sd_frac = sd(frac_norm, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(lo = pmax(mean_frac - sd_frac, 1e-5),
         hi = mean_frac + sd_frac)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: FILE FORMAT COMPARISON  (3x2, 7.2 x 5.2)
#
# Row 1: (a) compression ratio  (b) read throughput  (c) .1pz vs H5AD size
# Row 2: (d) by species         (e) by protocol      (f) value distribution
# ═══════════════════════════════════════════════════════════════════

# --- (a) Compression ratio by format ---
p1a <- ggplot(fmt_tidy %>% filter(!is.na(ratio), ratio > 0),
              aes(x = format, y = ratio, fill = format)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, color = "grey50",
               linewidth = 0.25, alpha = 0.7) +
  geom_jitter(aes(color = format), width = 0.15, size = 0.3, alpha = 0.25,
              stroke = 0, show.legend = FALSE) +
  scale_fill_manual(values = format_pal, guide = "none") +
  scale_color_manual(values = format_pal, guide = "none") +
  labs(x = NULL, y = "Compression ratio", title = "Compression ratio") +
  theme_sp() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 8))

# --- (b) Read throughput by format ---
p1b <- ggplot(fmt_tidy %>% filter(!is.na(read_mbps), read_mbps > 0),
              aes(x = format, y = read_mbps / 1000, fill = format)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, color = "grey50",
               linewidth = 0.25, alpha = 0.7) +
  geom_jitter(aes(color = format), width = 0.15, size = 0.3, alpha = 0.25,
              stroke = 0, show.legend = FALSE) +
  scale_fill_manual(values = format_pal, guide = "none") +
  scale_color_manual(values = format_pal, guide = "none") +
  labs(x = NULL, y = "Read (GB/s)", title = "Read throughput") +
  theme_sp() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 8))

# --- (c) .1pz vs H5AD file sizes ---
file_sizes <- io %>%
  filter(!is.na(h5ad_bytes), h5ad_bytes > 0) %>%
  transmute(pz_mb = pz_bytes / 1e6, h5ad_mb = h5ad_bytes / 1e6)

p1c <- ggplot(file_sizes, aes(x = h5ad_mb, y = pz_mb)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted",
              color = "grey60", linewidth = 0.25) +
  geom_point(size = 0.8, alpha = 0.5, color = PAL_1PZ) +
  geom_smooth(method = "lm", se = FALSE, color = PAL_1PZ,
              linewidth = 0.5, linetype = "dashed") +
  labs(x = "H5AD size (MB)", y = ".1pz size (MB)",
       title = ".1pz vs H5AD size") +
  theme_sp() +
  theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (d) Compression by species ---
sp_n <- survey_main %>% count(sp) %>% filter(n >= 5)
survey_sp <- survey_main %>% filter(sp %in% sp_n$sp)
sp_order <- survey_sp %>% group_by(sp) %>%
  summarise(med = median(ratio)) %>% arrange(med)
survey_sp$sp <- factor(survey_sp$sp, levels = sp_order$sp)

p1d <- ggplot(survey_sp, aes(x = sp, y = ratio)) +
  geom_boxplot(outlier.shape = NA, fill = "grey96", width = 0.55,
               color = "grey50", linewidth = 0.25) +
  geom_jitter(width = 0.12, size = 0.15, alpha = 0.12,
              stroke = 0, color = "black") +
  geom_hline(yintercept = median(survey_main$ratio),
             linetype = "dashed", color = PAL_1PZ, linewidth = 0.3, alpha = 0.5) +
  labs(x = NULL, y = "Compression ratio", title = "By species") +
  coord_flip() +
  theme_sp() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (e) Compression by protocol ---
proto_n <- survey_main %>% count(proto) %>% filter(n >= 10)
survey_pr <- survey_main %>% filter(proto %in% proto_n$proto)
pr_order <- survey_pr %>% group_by(proto) %>%
  summarise(med = median(ratio)) %>% arrange(med)
survey_pr$proto <- factor(survey_pr$proto, levels = pr_order$proto)

p1e <- ggplot(survey_pr, aes(x = proto, y = ratio)) +
  geom_boxplot(outlier.shape = NA, fill = "grey96", width = 0.55,
               color = "grey50", linewidth = 0.25) +
  geom_jitter(width = 0.12, size = 0.15, alpha = 0.12,
              stroke = 0, color = "black") +
  geom_hline(yintercept = median(survey_main$ratio),
             linetype = "dashed", color = PAL_1PZ, linewidth = 0.3, alpha = 0.5) +
  labs(x = NULL, y = "Compression ratio", title = "By protocol") +
  coord_flip() +
  theme_sp() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (f) Value distribution (mean + ribbon) ---
p1f <- ggplot(vdist_global, aes(x = value)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = PAL_1PZ, alpha = 0.2) +
  geom_line(aes(y = mean_frac), color = PAL_1PZ, linewidth = 0.8) +
  scale_y_log10(labels = label_percent(accuracy = 0.1),
                breaks = c(0.001, 0.01, 0.1, 0.5)) +
  scale_x_continuous(breaks = c(1, 5, 10, 20, 30)) +
  labs(x = "Count value", y = "Frac. of nonzeros",
       title = "Value distribution") +
  theme_sp() +
  theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- Assemble Figure 1 ---
fig1 <- (p1a | p1b | p1c) / (p1d | p1e | p1f) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave("fig1_compression.pdf", fig1, width = 7.2, height = 5.2, device = cairo_pdf)
cat("Wrote fig1_compression.pdf\n")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: PERFORMANCE  (3x2, 7.2 x 5.2)
#
# Row 1: (a) read throughput vs nnz  (b) thread scaling  (c) contiguous subsample
# Row 2: (d) random subsample        (e) GPU compute %   (f) GPU I/O vs compute
# ═══════════════════════════════════════════════════════════════════

# --- (a) Read throughput vs nnz ---
p2a <- ggplot(read_tp, aes(x = nnz / 1e6, y = read_mbps / 1000)) +
  geom_point(size = 0.5, alpha = 0.3, stroke = 0, color = PAL_1PZ) +
  geom_smooth(method = "loess", se = FALSE, color = PAL_1PZ, linewidth = 0.5) +
  labs(x = "Nonzeros (millions)", y = "Read (GB/s)",
       title = "Read throughput vs. size") +
  theme_sp() +
  theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (b) Thread scaling ---
base_speed <- threading %>% filter(n_threads == 1) %>%
  select(gse_id, base_gbps = read_gbps)
thread_norm <- threading %>%
  left_join(base_speed, by = "gse_id") %>%
  mutate(speedup = read_gbps / base_gbps)

p2b <- ggplot(thread_norm, aes(x = factor(n_threads), y = speedup)) +
  geom_boxplot(outlier.shape = NA, fill = "grey96", width = 0.45,
               color = "grey50", linewidth = 0.25) +
  geom_jitter(width = 0.08, size = 1.0, alpha = 0.5, color = PAL_1PZ, stroke = 0) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey60", linewidth = 0.25) +
  labs(x = "Threads", y = "Speedup vs. 1 thread",
       title = "Multi-threaded scaling") +
  theme_sp()

# --- (c) Contiguous 10% subsample ---
ops_pal <- c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD,
             "10x H5" = PAL_10X, "npz" = PAL_NPZ)
ops$format <- factor(ops$format, levels = c(".1pz", "H5AD", "10x H5", "npz"))

ops_contig <- ops %>% filter(operation == "contiguous_10pct")
contig_wide <- ops_contig %>%
  select(gse_id, format, time_s) %>%
  pivot_wider(names_from = format, values_from = time_s)
contig_speedup <- if ("H5AD" %in% names(contig_wide) && ".1pz" %in% names(contig_wide))
  median(contig_wide[["H5AD"]] / contig_wide[[".1pz"]], na.rm = TRUE) else NA

p2c <- ggplot(ops_contig, aes(x = nnz / 1e6, y = time_s * 1000,
                               color = format, shape = format)) +
  geom_point(size = 1.2, alpha = 0.8) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.4, span = 1.2) +
  scale_color_manual(values = ops_pal, name = NULL) +
  scale_shape_manual(values = c(".1pz" = 16, "H5AD" = 17,
                                 "10x H5" = 15, "npz" = 18), name = NULL) +
  scale_x_log10(labels = label_number(suffix = "M"), breaks = c(5, 20, 50, 100)) +
  scale_y_log10(labels = label_number(suffix = "ms"), breaks = c(10, 100, 1000)) +
  {if (!is.na(contig_speedup))
    annotate("text", x = 7, y = 1500,
             label = sprintf("%.0f\u00d7", contig_speedup),
             color = PAL_1PZ, size = 2.5, fontface = "bold", hjust = 0)
  } +
  labs(x = "Nonzeros (M)", y = "Time",
       title = "Contiguous 10% subsample") +
  theme_sp() +
  theme(legend.position = c(0.70, 0.22),
        legend.key.size = unit(0.15, "cm"),
        legend.text = element_text(size = 6),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (d) Random 10% subsample ---
ops_rand <- ops %>% filter(operation == "random_10pct")

p2d <- ggplot(ops_rand, aes(x = nnz / 1e6, y = time_s * 1000,
                             color = format, shape = format)) +
  geom_point(size = 1.2, alpha = 0.8) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.4, span = 1.2) +
  scale_color_manual(values = ops_pal, name = NULL) +
  scale_shape_manual(values = c(".1pz" = 16, "H5AD" = 17,
                                 "10x H5" = 15, "npz" = 18), name = NULL) +
  scale_x_log10(labels = label_number(suffix = "M"), breaks = c(5, 20, 50, 100)) +
  scale_y_log10(labels = label_number(suffix = "ms"), breaks = c(100, 500, 2000, 5000)) +
  labs(x = "Nonzeros (M)", y = "Time",
       title = "Random 10% subsample") +
  theme_sp() +
  theme(legend.position = c(0.70, 0.22),
        legend.key.size = unit(0.15, "cm"),
        legend.text = element_text(size = 6),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (e) GPU compute fraction ---
if ("n_cells" %in% names(gpu_bench)) {
  fb <- gpu_bench %>%
    mutate(fmt = ifelse(format == "1pz", ".1pz", "H5AD"),
           fmt = factor(fmt, levels = c(".1pz", "H5AD")))
} else {
  fb <- gpu_bench %>%
    mutate(n_cells = total_cells / n_epochs,
           fmt = ifelse(format == "1pz", ".1pz", "H5AD"),
           fmt = factor(fmt, levels = c(".1pz", "H5AD")))
}

p2e <- ggplot(fb, aes(x = n_cells / 1e3, y = compute_pct,
                       color = fmt, shape = fmt)) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "grey65", linewidth = 0.25) +
  geom_point(size = 1.2, alpha = 0.7) +
  scale_color_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD), name = NULL) +
  scale_shape_manual(values = c(".1pz" = 16, "H5AD" = 17), name = NULL) +
  scale_x_continuous(labels = label_number(suffix = "K")) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25),
                     labels = label_number(suffix = "%")) +
  annotate("text", x = max(fb$n_cells/1e3) * 0.45, y = 70,
           label = "compute-bound", color = "grey50", size = 2.2, fontface = "italic") +
  annotate("text", x = max(fb$n_cells/1e3) * 0.45, y = 30,
           label = "I/O-bound", color = "grey50", size = 2.2, fontface = "italic") +
  labs(x = "Cells (thousands)", y = "GPU compute %",
       title = "GPU compute fraction") +
  theme_sp() +
  theme(legend.position = c(0.75, 0.15),
        legend.key.size = unit(0.15, "cm"),
        legend.text = element_text(size = 6),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (f) GPU I/O vs compute density ---
if (nrow(fb) >= 10) {
  p2f <- ggplot(fb, aes(x = compute_pct, color = fmt, fill = fmt)) +
    geom_density(alpha = 0.15, linewidth = 0.6, adjust = 1.5) +
    geom_vline(xintercept = 50, linetype = "dashed", color = "grey65", linewidth = 0.25) +
    scale_color_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD), name = NULL) +
    scale_fill_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD), name = NULL) +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 25),
                       labels = label_number(suffix = "%")) +
    labs(x = "GPU compute %", y = "Density",
         title = "I/O vs. compute time") +
    theme_sp() +
    theme(legend.position = c(0.25, 0.85),
          legend.key.size = unit(0.15, "cm"),
          legend.text = element_text(size = 6))
} else {
  # Fallback: load vs compute scatter
  p2f <- ggplot(fb, aes(x = load_s, y = compute_s, color = fmt, shape = fmt)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "grey65", linewidth = 0.25) +
    geom_point(size = 1.5, alpha = 0.8) +
    scale_color_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD), name = NULL) +
    scale_shape_manual(values = c(".1pz" = 16, "H5AD" = 17), name = NULL) +
    labs(x = "Load time (s)", y = "Compute time (s)",
         title = "I/O vs. compute time") +
    theme_sp() +
    theme(legend.position = c(0.75, 0.15),
          legend.key.size = unit(0.15, "cm"),
          legend.text = element_text(size = 6),
          panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))
}

# --- Assemble Figure 2 ---
fig2 <- (p2a | p2b | p2c) / (p2d | p2e | p2f) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave("fig2_performance.pdf", fig2, width = 7.2, height = 5.2, device = cairo_pdf)
cat("Wrote fig2_performance.pdf\n")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: I/O THROUGHPUT  (2x2, 3.5 x 4.0)
#
# Row 1: (a) Python write  (b) Python read
# Row 2: (c) R write       (d) R read
# ═══════════════════════════════════════════════════════════════════

# --- Python write data ---
py_write_tidy <- bind_rows(
  write_bench %>% filter(!is.na(pz_write_s), pz_write_s > 0, nnz > 1e5) %>%
    transmute(nnz, format = ".1pz",
              write_mbps = raw_int32_bytes / pz_write_s / 1e6),
  write_bench %>% filter(!is.na(h5ad_write_s), h5ad_write_s > 0) %>%
    transmute(nnz, format = "H5AD",
              write_mbps = raw_int32_bytes / h5ad_write_s / 1e6)
) %>% mutate(format = factor(format, levels = c(".1pz", "H5AD")))

py_paired <- write_bench %>%
  filter(!is.na(pz_write_s), !is.na(h5ad_write_s), pz_write_s > 0, h5ad_write_s > 0)
py_write_speedup <- median(py_paired$h5ad_write_s / py_paired$pz_write_s)

p3a <- ggplot(py_write_tidy, aes(x = nnz / 1e6, y = write_mbps, color = format)) +
  geom_point(size = 0.9, alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.5) +
  scale_color_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD), name = NULL) +
  scale_x_log10(labels = label_number(suffix = "M"), breaks = c(0.1, 1, 10, 100)) +
  scale_y_log10() +
  annotate("text", x = 0.5, y = max(py_write_tidy$write_mbps) * 0.5,
           label = sprintf("%.1f\u00d7", py_write_speedup),
           color = PAL_1PZ, size = 2.8, fontface = "bold", hjust = 0) +
  labs(x = "Nonzeros (M)", y = "Write (MB/s)", title = "Python write") +
  theme_sp() +
  theme(legend.position = c(0.70, 0.20),
        legend.key.size = unit(0.15, "cm"),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- Python read data ---
py_read_tidy <- bind_rows(
  io %>% filter(!is.na(read_1pz_mbps), read_1pz_mbps > 0) %>%
    transmute(nnz, format = ".1pz", read_mbps = read_1pz_mbps),
  io %>% filter(!is.na(read_h5ad_mbps), read_h5ad_mbps > 0) %>%
    transmute(nnz, format = "H5AD", read_mbps = read_h5ad_mbps)
) %>% mutate(format = factor(format, levels = c(".1pz", "H5AD")))

py_read_paired <- io %>%
  filter(!is.na(read_1pz_mbps), !is.na(read_h5ad_mbps),
         read_1pz_mbps > 0, read_h5ad_mbps > 0)
py_read_speedup <- median(py_read_paired$read_1pz_mbps / py_read_paired$read_h5ad_mbps)

p3b <- ggplot(py_read_tidy, aes(x = nnz / 1e6, y = read_mbps, color = format)) +
  geom_point(size = 0.9, alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.5) +
  scale_color_manual(values = c(".1pz" = PAL_1PZ, "H5AD" = PAL_H5AD), name = NULL) +
  scale_x_log10(labels = label_number(suffix = "M"), breaks = c(1, 10, 100)) +
  scale_y_log10() +
  annotate("text", x = 2, y = max(py_read_tidy$read_mbps, na.rm = TRUE) * 0.5,
           label = sprintf("%.1f\u00d7", py_read_speedup),
           color = PAL_1PZ, size = 2.8, fontface = "bold", hjust = 0) +
  labs(x = "Nonzeros (M)", y = "Read (MB/s)", title = "Python read") +
  theme_sp() +
  theme(legend.position = c(0.70, 0.20),
        legend.key.size = unit(0.15, "cm"),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- R write data ---
if (!is.null(r_write_bench) && nrow(r_write_bench) > 0) {
  r_write_tidy <- bind_rows(
    write_bench %>%
      filter(!is.na(pz_write_s), pz_write_s > 0,
             gse_id %in% r_write_bench$gse_id) %>%
      transmute(nnz, format = ".1pz",
                write_mbps = raw_int32_bytes / pz_write_s / 1e6),
    r_write_bench %>%
      transmute(nnz, format = "RDS",
                write_mbps = rds_write_mbps)
  ) %>% mutate(format = factor(format, levels = c(".1pz", "RDS")))

  r_paired_w <- r_write_bench %>%
    inner_join(write_bench %>% select(gse_id, pz_write_s, raw_int32_bytes),
               by = "gse_id") %>%
    filter(!is.na(pz_write_s), pz_write_s > 0)
  r_write_speedup <- median(r_paired_w$rds_write_s / r_paired_w$pz_write_s)

  p3c <- ggplot(r_write_tidy, aes(x = nnz / 1e6, y = write_mbps, color = format)) +
    geom_point(size = 0.9, alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 0.5) +
    scale_color_manual(values = c(".1pz" = PAL_1PZ, "RDS" = PAL_RDS), name = NULL) +
    scale_x_log10(labels = label_number(suffix = "M"), breaks = c(0.1, 1, 10, 100)) +
    scale_y_log10() +
    annotate("text", x = 0.5, y = max(r_write_tidy$write_mbps) * 0.5,
             label = sprintf("%.0f\u00d7", r_write_speedup),
             color = PAL_1PZ, size = 2.8, fontface = "bold", hjust = 0) +
    labs(x = "Nonzeros (M)", y = "Write (MB/s)", title = "R write") +
    theme_sp() +
    theme(legend.position = c(0.70, 0.20),
          legend.key.size = unit(0.15, "cm"),
          panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))
} else {
  p3c <- ggplot() + theme_void() + labs(title = "R write (no data)")
  r_write_speedup <- NA
}

# --- R read data ---
r_read_tidy <- bind_rows(
  io %>% filter(!is.na(read_1pz_mbps), read_1pz_mbps > 0,
                gse_id %in% r_bench$gse_id) %>%
    transmute(nnz, format = ".1pz", read_mbps = read_1pz_mbps),
  r_bench %>% filter(!is.na(rds_read_mbps), rds_read_mbps > 0) %>%
    transmute(nnz, format = "RDS", read_mbps = rds_read_mbps)
) %>% mutate(format = factor(format, levels = c(".1pz", "RDS")))

r_read_paired <- r_bench %>%
  inner_join(io %>% select(gse_id, read_1pz_mbps), by = "gse_id") %>%
  filter(!is.na(read_1pz_mbps), !is.na(rds_read_mbps),
         read_1pz_mbps > 0, rds_read_mbps > 0)
r_read_speedup <- if (nrow(r_read_paired) > 0)
  median(r_read_paired$read_1pz_mbps / r_read_paired$rds_read_mbps) else NA

p3d <- ggplot(r_read_tidy, aes(x = nnz / 1e6, y = read_mbps, color = format)) +
  geom_point(size = 0.9, alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.5) +
  scale_color_manual(values = c(".1pz" = PAL_1PZ, "RDS" = PAL_RDS), name = NULL) +
  scale_x_log10(labels = label_number(suffix = "M"), breaks = c(1, 10, 100)) +
  scale_y_log10() +
  {if (!is.na(r_read_speedup))
    annotate("text", x = 2, y = max(r_read_tidy$read_mbps, na.rm = TRUE) * 0.5,
             label = sprintf("%.1f\u00d7", r_read_speedup),
             color = PAL_1PZ, size = 2.8, fontface = "bold", hjust = 0)
  } +
  labs(x = "Nonzeros (M)", y = "Read (MB/s)", title = "R read") +
  theme_sp() +
  theme(legend.position = c(0.70, 0.20),
        legend.key.size = unit(0.15, "cm"),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- Assemble Figure 3 ---
fig3 <- (p3a | p3b) / (p3c | p3d) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave("fig3_io.pdf", fig3, width = 3.5, height = 4.0, device = cairo_pdf)
cat(sprintf("Wrote fig3_io.pdf (Py write: %.1fx, Py read: %.1fx, R write: %.0fx, R read: %.1fx)\n",
            py_write_speedup, py_read_speedup, r_write_speedup, r_read_speedup))


# ═══════════════════════════════════════════════════════════════════
# FIGURE S1: STATISTICAL STRUCTURE  (3x2, 7.2 x 5.2)
#
# Row 1: (a) spaghetti  (b) entropy vs frac_one  (c) bits/nz vs ratio
# Row 2: (d) ratio vs nnz  (e) R ecosystem formats  (f) predicted vs observed
# ═══════════════════════════════════════════════════════════════════

# --- (a) Value distributions spaghetti ---
ps1a <- ggplot(vdist_main, aes(x = value, y = frac_norm, group = gse_id)) +
  geom_line(alpha = 0.08, color = "grey50", linewidth = 0.15) +
  stat_summary(aes(group = 1), fun = median, geom = "line",
               color = PAL_1PZ, linewidth = 0.6) +
  scale_y_log10(labels = label_percent(accuracy = 0.1),
                breaks = c(0.001, 0.01, 0.1, 0.5)) +
  scale_x_continuous(breaks = c(1, 10, 20, 30)) +
  labs(x = "Count value", y = "Frac. of nonzeros",
       title = "Per-dataset value distributions") +
  theme_sp() +
  theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (b) Entropy vs frac_one ---
ps1b <- ggplot(zinb %>% filter(nnz > 1e6),
               aes(x = frac_one, y = entropy_nz, color = sp)) +
  geom_point(size = 0.8, alpha = 0.5) +
  scale_color_manual(values = species_pal, name = NULL) +
  scale_x_continuous(labels = label_percent(), limits = c(0.4, 1)) +
  labs(x = "Frac. nnz = 1", y = "Entropy (bits/nz)",
       title = "Entropy vs. fraction of ones") +
  theme_sp() +
  theme(legend.position = c(0.28, 0.28),
        legend.key.size = unit(0.12, "cm"),
        legend.text = element_text(size = 5),
        legend.spacing.y = unit(0, "pt"),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (c) Bits/nz vs compression ratio ---
ps1c <- ggplot(survey_main, aes(x = ratio, y = bits_per_nz)) +
  geom_point(aes(color = sp), size = 0.15, alpha = 0.1, stroke = 0) +
  scale_color_manual(values = species_pal, guide = "none") +
  stat_function(fun = function(r) 32 / r, color = "grey30", linewidth = 0.35) +
  geom_smooth(method = "loess", se = FALSE, color = PAL_1PZ, linewidth = 0.4,
              linetype = "dashed") +
  labs(x = "Compression ratio", y = "Bits per nonzero",
       title = "Encoding efficiency") +
  theme_sp() +
  theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (d) Ratio vs nnz (size invariance) ---
ps1d <- ggplot(survey_main, aes(x = nnz / 1e6, y = ratio)) +
  geom_point(aes(color = sp), size = 0.15, alpha = 0.1, stroke = 0) +
  scale_color_manual(values = species_pal, guide = "none") +
  geom_hline(yintercept = median(survey_main$ratio), linetype = "dashed",
             color = PAL_1PZ, linewidth = 0.3) +
  scale_x_log10(labels = label_number(suffix = "M"), breaks = c(1, 10, 100)) +
  labs(x = "Nonzeros (M)", y = "Compression ratio",
       title = "Size invariance") +
  theme_sp() +
  theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- (e) R ecosystem format comparison ---
r_fmt_tidy <- bind_rows(
  io %>% filter(gse_id %in% r_bench$gse_id) %>%
    transmute(gse_id, format = ".1pz", ratio = raw_int32_bytes / pz_bytes),
  r_bench %>% transmute(gse_id, format = "RDS", ratio = rds_ratio),
  bpcells %>% transmute(gse_id, format = "BPCells", ratio = bp_comp_ratio)
) %>% mutate(format = factor(format, levels = c("RDS", "BPCells", ".1pz")))

ps1e <- ggplot(r_fmt_tidy, aes(x = format, y = ratio, fill = format)) +
  geom_boxplot(outlier.shape = NA, width = 0.55, color = "grey50",
               linewidth = 0.25, alpha = 0.7) +
  geom_jitter(aes(color = format), width = 0.12, size = 0.8, alpha = 0.4,
              stroke = 0, show.legend = FALSE) +
  scale_fill_manual(values = c(".1pz" = PAL_1PZ, "BPCells" = PAL_BPCELLS,
                                "RDS" = PAL_RDS), guide = "none") +
  scale_color_manual(values = c(".1pz" = PAL_1PZ, "BPCells" = PAL_BPCELLS,
                                 "RDS" = PAL_RDS), guide = "none") +
  labs(x = NULL, y = "Compression ratio",
       title = "R ecosystem formats") +
  theme_sp()

# --- (f) Predicted vs observed ---
ps1f <- ggplot(survey_main, aes(x = pred / 1e6, y = pz_bytes / 1e6)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              color = "grey60", linewidth = 0.25) +
  geom_point(aes(color = sp), size = 0.2, alpha = 0.15, stroke = 0) +
  scale_color_manual(values = species_pal, guide = "none") +
  labs(x = "Predicted (MB)", y = "Observed (MB)",
       title = sprintf("Pred. vs. obs. (R\u00b2 = %.3f)", r2)) +
  coord_equal() +
  theme_sp() +
  theme(panel.grid.major.x = element_line(color = "grey92", linewidth = 0.25))

# --- Assemble Figure S1 ---
figS1 <- (ps1a | ps1b | ps1c) / (ps1d | ps1e | ps1f) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12))

ggsave("figS1_structure.pdf", figS1, width = 7.2, height = 5.2, device = cairo_pdf)
cat("Wrote figS1_structure.pdf\n")


# ═══════════════════════════════════════════════════════════════════
# Summary stats
# ═══════════════════════════════════════════════════════════════════
cat("\n\u2550\u2550 Summary \u2550\u2550\n")
cat(sprintf("Datasets: %d total, %d with nnz>1M\n", nrow(survey), nrow(survey_main)))
cat(sprintf("Median ratio: %.1f\u00d7  range: %.1f\u2013%.1f\u00d7\n",
    median(survey_main$ratio), min(survey_main$ratio), max(survey_main$ratio)))
cat(sprintf("R\u00b2 = %.4f, %.3f bytes/nnz\n", r2, bpnz))
cat(sprintf("Python write: %.1f\u00d7, Python read: %.1f\u00d7\n",
    py_write_speedup, py_read_speedup))
if (!is.na(r_write_speedup))
  cat(sprintf("R write: %.0f\u00d7, R read: %.1f\u00d7\n",
      r_write_speedup, r_read_speedup))
cat("\nAll figures written.\n")
