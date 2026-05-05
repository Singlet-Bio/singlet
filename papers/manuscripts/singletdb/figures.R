#!/usr/bin/env Rscript
# singletdb manuscript figures — all panels generated from catalog parquet files
# Requires: arrow, ggplot2, dplyr, tidyr, patchwork, scales, forcats

suppressPackageStartupMessages({
  library(arrow)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(forcats)
})

out_dir <- "/mnt/home/debruinz/Singlet-AI/papers/manuscripts/singletdb"

# ── Load data ────────────────────────────────────────────────────────────────
cat("Loading catalogs...\n")
full <- read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet")
filt <- read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/filtered_catalog.parquet")
remv <- read_parquet("/mnt/projects/debruinz_project/cellarium/catalog/removed_catalog.parquet")

cat(sprintf("Full catalog: %s samples, %s series\n", format(nrow(full), big.mark=","), format(n_distinct(full$gse_id), big.mark=",")))
cat(sprintf("Filtered: %s samples (%0.1f%%)\n", format(nrow(filt), big.mark=","), 100*nrow(filt)/nrow(full)))

# ── Theme ────────────────────────────────────────────────────────────────────
theme_singlet <- theme_minimal(base_size = 10, base_family = "sans") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.text = element_text(face = "bold", size = 9),
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 9, color = "grey40"),
    legend.position = "bottom",
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8)
  )
theme_set(theme_singlet)

# Color palette for assays
assay_colors <- c(
  "scrna" = "#2166AC", "plate_scrna" = "#92C5DE",
  "atac" = "#D6604D", "multiome" = "#F4A582",
  "spatial" = "#4DAF4A", "cite" = "#984EA3",
  "ambiguous" = "#999999", "non_rna" = "#E0E0E0",
  "vdj" = "#FF7F00"
)

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Catalog overview
# ════════════════════════════════════════════════════════════════════════════════
cat("Figure 1: Catalog overview...\n")

# Panel A: Assay type distribution (full catalog)
assay_df <- full %>%
  count(target_assay) %>%
  mutate(pct = n / sum(n) * 100,
         target_assay = fct_reorder(target_assay, n))

fig1a <- ggplot(assay_df, aes(x = n, y = target_assay, fill = target_assay)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("%s (%0.1f%%)", format(n, big.mark=","), pct)),
            hjust = -0.05, size = 2.8) +
  scale_x_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.35))) +
  scale_fill_manual(values = assay_colors) +
  labs(x = "Number of GEO samples", y = NULL,
       title = "a  Assay classification",
       subtitle = sprintf("n = %s GEO samples from %s series",
                          format(nrow(full), big.mark=","),
                          format(n_distinct(full$gse_id), big.mark=",")))

# Panel B: Submission year timeline
year_df <- full %>%
  mutate(year = as.integer(substr(submission_date, 1, 4))) %>%
  filter(!is.na(year), year >= 2015, year <= 2026) %>%
  count(year, target_assay) %>%
  mutate(target_assay = factor(target_assay,
    levels = c("plate_scrna", "scrna", "atac", "ambiguous", "multiome",
               "spatial", "cite", "non_rna", "vdj")))

fig1b <- ggplot(year_df, aes(x = year, y = n, fill = target_assay)) +
  geom_area(alpha = 0.85) +
  scale_fill_manual(values = assay_colors, name = "Assay") +
  scale_x_continuous(breaks = 2015:2026) +
  scale_y_continuous(labels = label_comma()) +
  labs(x = "Submission year", y = "Samples",
       title = "b  Growth of single-cell GEO submissions",
       subtitle = "By inferred assay type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Panel C: Processing status funnel
status_summary <- data.frame(
  stage = factor(c("GEO discovered", "Droplet-compatible", "Processable\n(all screens pass)",
                    "Processing\nattempted", "QC passed"),
                 levels = c("GEO discovered", "Droplet-compatible", "Processable\n(all screens pass)",
                            "Processing\nattempted", "QC passed")),
  n = c(nrow(full),
        sum(full$target_assay %in% c("scrna","cite","atac","spatial","multiome","ambiguous")),
        nrow(filt),
        sum(full$processing_status %in% c("done","done_qc_warn","fail_qc_low_genes",
          "fail_qc_other","fail_qc_few_cells","fail_simpleaf_permit","fail_simpleaf_map",
          "fail_simpleaf_timeout","fail_simpleaf_other","fail_download","fail_protocol_detect",
          "fail_no_r2","fail_low_mapping")),
        sum(full$processing_status %in% c("done","done_qc_warn")))
)

fig1c <- ggplot(status_summary, aes(x = stage, y = n)) +
  geom_col(fill = "#2166AC", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = format(n, big.mark = ",")), vjust = -0.3, size = 3) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = "Samples",
       title = "c  Processing funnel",
       subtitle = "From GEO discovery to QC-passed output")

fig1 <- fig1a / (fig1b | fig1c) +
  plot_annotation(title = "Figure 1. singletDB catalog overview",
                  theme = theme(plot.title = element_text(face = "bold", size = 13)))

ggsave(file.path(out_dir, "fig1_catalog_overview.pdf"), fig1, width = 10, height = 9)
ggsave(file.path(out_dir, "fig1_catalog_overview.png"), fig1, width = 10, height = 9, dpi = 300)
cat("  Saved fig1\n")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Species and organism diversity
# ════════════════════════════════════════════════════════════════════════════════
cat("Figure 2: Species diversity...\n")

# Panel A: Top 20 organisms (filtered catalog)
org_top <- filt %>%
  count(organism, sort = TRUE) %>%
  slice_head(n = 20) %>%
  mutate(organism = fct_reorder(organism, n))

fig2a <- ggplot(org_top, aes(x = n, y = organism)) +
  geom_col(fill = "#2166AC", alpha = 0.8) +
  geom_text(aes(label = format(n, big.mark=",")), hjust = -0.05, size = 2.5) +
  scale_x_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.25))) +
  labs(x = "Processable samples", y = NULL,
       title = "a  Organism distribution (processable catalog)",
       subtitle = sprintf("Top 20 of %s unique organisms", format(n_distinct(filt$organism), big.mark=",")))

# Panel B: Protocol by top species heatmap (filtered)
top_species <- filt %>%
  mutate(species_clean = case_when(
    grepl("Homo sapiens", organism, ignore.case=TRUE) & !grepl(";", organism) ~ "Human",
    grepl("Mus musculus", organism, ignore.case=TRUE) & !grepl(";", organism) ~ "Mouse",
    grepl("Drosophila", organism, ignore.case=TRUE) ~ "Drosophila",
    grepl("Macaca mulatta", organism, ignore.case=TRUE) & !grepl(";", organism) ~ "Rhesus macaque",
    grepl("Macaca fascicularis", organism, ignore.case=TRUE) & !grepl(";", organism) ~ "Cynomolgus",
    grepl("Danio rerio", organism, ignore.case=TRUE) ~ "Zebrafish",
    grepl("Ovis aries", organism, ignore.case=TRUE) ~ "Sheep",
    grepl("Rattus", organism, ignore.case=TRUE) ~ "Rat",
    grepl("Saccharomyces", organism, ignore.case=TRUE) ~ "Yeast",
    grepl("Arabidopsis", organism, ignore.case=TRUE) ~ "Arabidopsis",
    grepl("Sus scrofa", organism, ignore.case=TRUE) ~ "Pig",
    grepl("Gallus", organism, ignore.case=TRUE) ~ "Chicken",
    grepl("Bos taurus", organism, ignore.case=TRUE) ~ "Cow",
    TRUE ~ "Other"
  )) %>%
  filter(species_clean != "Other") %>%
  mutate(proto_clean = case_when(
    grepl("10xv3", protocol_inferred) ~ "10x v3",
    grepl("10xv2", protocol_inferred) ~ "10x v2",
    grepl("10xv4", protocol_inferred) ~ "10x v4",
    grepl("10x_suspect", protocol_inferred) ~ "10x (unspec.)",
    grepl("10x_multiome", protocol_inferred) ~ "Multiome",
    grepl("scATAC|10x_atac", protocol_inferred) ~ "scATAC",
    grepl("indrop", protocol_inferred) ~ "inDrop",
    grepl("scirna", protocol_inferred) ~ "sci-RNA-seq",
    grepl("dropseq", protocol_inferred) ~ "Drop-seq",
    grepl("bd_rhapsody", protocol_inferred) ~ "BD Rhapsody",
    grepl("parse", protocol_inferred) ~ "Parse",
    grepl("unknown", protocol_inferred) ~ "Unknown",
    TRUE ~ protocol_inferred
  ))

heat_df <- top_species %>%
  count(species_clean, proto_clean) %>%
  group_by(species_clean) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup()

fig2b <- ggplot(heat_df, aes(x = proto_clean, y = species_clean, fill = log10(n + 1))) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(n >= 50, format(n, big.mark=","), "")),
            size = 2.2, color = "white") +
  scale_fill_viridis_c(option = "D", name = expression(log[10](samples))) +
  labs(x = NULL, y = NULL,
       title = "b  Protocol × species composition",
       subtitle = "Processable samples by inferred protocol and species") +
  theme(axis.text.x = element_text(angle = 50, hjust = 1, size = 7),
        axis.text.y = element_text(size = 8))

fig2 <- fig2a / fig2b +
  plot_layout(heights = c(1.2, 1)) +
  plot_annotation(title = "Figure 2. Species and protocol diversity",
                  theme = theme(plot.title = element_text(face = "bold", size = 13)))

ggsave(file.path(out_dir, "fig2_species_protocol.pdf"), fig2, width = 10, height = 11)
ggsave(file.path(out_dir, "fig2_species_protocol.png"), fig2, width = 10, height = 11, dpi = 300)
cat("  Saved fig2\n")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Protocol inference and confidence
# ════════════════════════════════════════════════════════════════════════════════
cat("Figure 3: Protocol inference...\n")

# Panel A: Confidence level by assay (filtered)
conf_df <- filt %>%
  count(target_assay, protocol_confidence) %>%
  mutate(protocol_confidence = factor(protocol_confidence, levels = c("low", "medium", "high")))

fig3a <- ggplot(conf_df, aes(x = target_assay, y = n, fill = protocol_confidence)) +
  geom_col(position = "fill", alpha = 0.9) +
  scale_fill_manual(values = c("low" = "#D6604D", "medium" = "#F4A582", "high" = "#2166AC"),
                    name = "Confidence") +
  scale_y_continuous(labels = percent_format()) +
  labs(x = "Assay type", y = "Proportion of samples",
       title = "a  Classification confidence by assay",
       subtitle = "Metadata-based protocol inference confidence levels")

# Panel B: Protocol inference sources
proto_conf_df <- filt %>%
  filter(!grepl("atac|ATAC", protocol_inferred)) %>%
  count(protocol_inferred, protocol_confidence, sort = TRUE) %>%
  group_by(protocol_inferred) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>%
  filter(total >= 200) %>%
  mutate(protocol_inferred = fct_reorder(protocol_inferred, total),
         protocol_confidence = factor(protocol_confidence, levels = c("low", "medium", "high")))

fig3b <- ggplot(proto_conf_df, aes(x = n, y = protocol_inferred, fill = protocol_confidence)) +
  geom_col(alpha = 0.9) +
  geom_text(aes(label = ifelse(n >= 500, format(n, big.mark=","), ""), x = n),
            position = position_stack(vjust = 0.5), size = 2.2, color = "white") +
  scale_fill_manual(values = c("low" = "#D6604D", "medium" = "#F4A582", "high" = "#2166AC"),
                    name = "Confidence") +
  scale_x_continuous(labels = label_comma()) +
  labs(x = "Number of samples", y = NULL,
       title = "b  Inferred protocol distribution",
       subtitle = "Non-ATAC protocols with ≥200 samples, colored by confidence")

fig3 <- fig3a | fig3b
fig3 <- fig3 + plot_layout(widths = c(1, 1.5)) +
  plot_annotation(title = "Figure 3. Metadata-based protocol inference",
                  theme = theme(plot.title = element_text(face = "bold", size = 13)))

ggsave(file.path(out_dir, "fig3_protocol_inference.pdf"), fig3, width = 12, height = 5.5)
ggsave(file.path(out_dir, "fig3_protocol_inference.png"), fig3, width = 12, height = 5.5, dpi = 300)
cat("  Saved fig3\n")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Processing results and QC
# ════════════════════════════════════════════════════════════════════════════════
cat("Figure 4: Processing results...\n")

# Use catalog columns for samples that have been processed
proc <- full %>%
  filter(processing_status %in% c("done", "done_qc_warn",
    "fail_qc_low_genes", "fail_qc_other", "fail_qc_few_cells",
    "fail_low_mapping"))

# Panel A: Processing status breakdown (processed samples)
proc_status <- full %>%
  filter(!processing_status %in% c("pending", "skip_plate_based", "skip_likely_plate",
    "skip_non_rna", "skip_atac", "skip_vdj", "skip_reclassified", "skip_spatial",
    "retry_skipped")) %>%
  count(processing_status, sort = TRUE) %>%
  mutate(processing_status = fct_reorder(processing_status, n),
         category = case_when(
           grepl("^done", processing_status) ~ "Success",
           grepl("^fail_qc", processing_status) ~ "QC failure",
           grepl("^fail_simpleaf", processing_status) ~ "Pipeline failure",
           grepl("^fail_download", processing_status) ~ "Download failure",
           grepl("^fail_protocol", processing_status) ~ "Detection failure",
           TRUE ~ "Other failure"
         ))

cat_colors <- c("Success" = "#2166AC", "QC failure" = "#D6604D",
                "Pipeline failure" = "#F4A582", "Download failure" = "#FDDBC7",
                "Detection failure" = "#999999", "Other failure" = "#E0E0E0")

fig4a <- ggplot(proc_status, aes(x = n, y = processing_status, fill = category)) +
  geom_col(alpha = 0.9, show.legend = TRUE) +
  geom_text(aes(label = format(n, big.mark = ",")), hjust = -0.05, size = 2.5) +
  scale_fill_manual(values = cat_colors, name = "Category") +
  scale_x_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.25))) +
  labs(x = "Samples", y = NULL,
       title = "a  Processing outcomes",
       subtitle = sprintf("n = %s samples attempted", format(sum(proc_status$n), big.mark=",")))

# Panel B: Mapping rate distribution (from catalog migration data)
map_df <- full %>%
  mutate(mapping_rate_num = as.numeric(as.character(pipeline_mapping_rate))) %>%
  filter(!is.na(mapping_rate_num), mapping_rate_num > 0) %>%
  select(mapping_rate_num, processing_status) %>%
  mutate(qc_outcome = ifelse(grepl("^done", processing_status), "Pass", "Fail"))

if (nrow(map_df) > 100) {
  fig4b <- ggplot(map_df, aes(x = mapping_rate_num, fill = qc_outcome)) +
    geom_histogram(bins = 50, alpha = 0.8, position = "identity") +
    geom_vline(xintercept = 0.10, linetype = "dashed", color = "red", linewidth = 0.5) +
    annotate("text", x = 0.12, y = Inf, label = "Min threshold (10%)",
             hjust = 0, vjust = 1.5, size = 2.8, color = "red") +
    scale_fill_manual(values = c("Pass" = "#2166AC", "Fail" = "#D6604D"), name = "QC") +
    scale_x_continuous(labels = percent_format()) +
    labs(x = "Mapping rate", y = "Samples",
         title = "b  Mapping rate distribution",
         subtitle = sprintf("n = %s samples with mapping data", format(nrow(map_df), big.mark=",")))
} else {
  fig4b <- ggplot() + annotate("text", x=0.5, y=0.5, label="Insufficient mapping data") +
    theme_void() + labs(title = "b  Mapping rate distribution")
}

# Panel C: Cells per sample distribution (from migration data)
cell_df <- full %>%
  mutate(n_cells_num = as.numeric(as.character(migration_n_cells))) %>%
  filter(!is.na(n_cells_num), n_cells_num > 0, n_cells_num < 1e6)

if (nrow(cell_df) > 50) {
  fig4c <- ggplot(cell_df, aes(x = n_cells_num)) +
    geom_histogram(bins = 60, fill = "#2166AC", alpha = 0.8) +
    scale_x_log10(labels = label_comma()) +
    labs(x = "Cells per sample (log scale)", y = "Count",
         title = "c  Cells per sample",
         subtitle = sprintf("Median = %s cells (n = %s)",
                            format(median(cell_df$n_cells_num), big.mark=","),
                            format(nrow(cell_df), big.mark=","))) +
    annotation_logticks(sides = "b")
} else {
  fig4c <- ggplot() + annotate("text", x=0.5, y=0.5, label="Insufficient cell count data") +
    theme_void() + labs(title = "c  Cells per sample")
}

fig4 <- fig4a / (fig4b | fig4c) +
  plot_annotation(title = "Figure 4. Processing outcomes and quality metrics",
                  theme = theme(plot.title = element_text(face = "bold", size = 13)))

ggsave(file.path(out_dir, "fig4_processing_qc.pdf"), fig4, width = 11, height = 9)
ggsave(file.path(out_dir, "fig4_processing_qc.png"), fig4, width = 11, height = 9, dpi = 300)
cat("  Saved fig4\n")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Screen flags and data quality gates
# ════════════════════════════════════════════════════════════════════════════════
cat("Figure 5: Screen flags...\n")

screen_cols <- grep("^screen_", names(full), value = TRUE)
screen_cols <- screen_cols[screen_cols != "screen_any_flag"]

screen_df <- full %>%
  select(all_of(screen_cols)) %>%
  summarise(across(everything(), ~sum(. == TRUE | . == "True", na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "flag", values_to = "count") %>%
  mutate(pct = count / nrow(full) * 100,
         flag = gsub("screen_", "", flag),
         flag = fct_reorder(flag, count))

fig5a <- ggplot(screen_df, aes(x = count, y = flag)) +
  geom_col(fill = "#D6604D", alpha = 0.8) +
  geom_text(aes(label = sprintf("%s (%0.1f%%)", format(count, big.mark=","), pct)),
            hjust = -0.05, size = 2.8) +
  scale_x_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.35))) +
  labs(x = "Samples flagged", y = NULL,
       title = "a  Pre-processing screen flags",
       subtitle = "Flags that prevent sample processing")

# Panel B: Removal reasons pie/bar
removal_reasons <- data.frame(
  reason = c("Plate-based scRNA", "Non-RNA assay", "No FASTQ available",
             "Plate-based (screen)", "QC failures", "Unsupported organism",
             "VDJ", "Non-Illumina", "Other"),
  count = c(1083262, 114757, 53214, 19175, 6000, 1399, 2143, 1046, 451)
) %>%
  mutate(pct = count / sum(count) * 100,
         reason = fct_reorder(reason, count))

fig5b <- ggplot(removal_reasons, aes(x = count, y = reason)) +
  geom_col(fill = "#92C5DE", alpha = 0.9) +
  geom_text(aes(label = sprintf("%s (%0.1f%%)", format(count, big.mark=","), pct)),
            hjust = -0.02, size = 2.8) +
  scale_x_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.35))) +
  labs(x = "Samples removed", y = NULL,
       title = "b  Reasons for exclusion",
       subtitle = sprintf("Total removed: %s", format(nrow(remv), big.mark=",")))

fig5 <- fig5a | fig5b
fig5 <- fig5 + plot_annotation(
  title = "Figure 5. Quality screening and sample exclusion",
  theme = theme(plot.title = element_text(face = "bold", size = 13)))

ggsave(file.path(out_dir, "fig5_screening.pdf"), fig5, width = 12, height = 5)
ggsave(file.path(out_dir, "fig5_screening.png"), fig5, width = 12, height = 5, dpi = 300)
cat("  Saved fig5\n")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Metadata richness and standardization
# ════════════════════════════════════════════════════════════════════════════════
cat("Figure 6: Metadata coverage...\n")

# Column completeness
col_complete <- full %>%
  summarise(across(everything(), ~sum(!is.na(.) & . != "") / n() * 100)) %>%
  pivot_longer(everything(), names_to = "field", values_to = "completeness") %>%
  filter(completeness > 0, completeness < 100) %>%
  mutate(category = case_when(
    grepl("^screen_|^pipeline_|^migration_|^failure_|^production|^actionable|^v12", field) ~ "Derived",
    grepl("protocol|target_assay|species_ref|species_ann", field) ~ "Inferred",
    TRUE ~ "GEO metadata"
  )) %>%
  mutate(field = fct_reorder(field, completeness))

fig6 <- ggplot(col_complete %>% filter(category == "GEO metadata"),
               aes(x = completeness, y = field)) +
  geom_col(fill = "#2166AC", alpha = 0.8) +
  geom_text(aes(label = sprintf("%0.0f%%", completeness)), hjust = -0.1, size = 2.5) +
  scale_x_continuous(limits = c(0, 110), breaks = seq(0, 100, 25)) +
  labs(x = "Completeness (%)", y = NULL,
       title = "Figure 6. GEO metadata field completeness",
       subtitle = sprintf("Across %s samples in the full catalog", format(nrow(full), big.mark=","))) +
  theme(plot.title = element_text(face = "bold", size = 13))

ggsave(file.path(out_dir, "fig6_metadata.pdf"), fig6, width = 8, height = 7)
ggsave(file.path(out_dir, "fig6_metadata.png"), fig6, width = 8, height = 7, dpi = 300)
cat("  Saved fig6\n")

cat("All figures generated.\n")
