# figures/render.R — Render all figures for the genomics-intelligence manuscript
# Run: Rscript render.R (on compute node, not login node)

library(ggplot2)
library(dplyr)

# Singlet AI ggplot2 theme
theme_singletai <- theme_minimal(base_size = 8) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 9, face = "bold"),
    strip.text = element_text(size = 8, face = "italic"),
    legend.position = "bottom",
    legend.key.size = unit(0.3, "cm")
  )

# Column widths for the two-column template:
#   Single column: 3.4 inches
#   Full width:    7.1 inches

# ── Figure 1: CPM Architecture Diagram ──────────────────────────
# This is a schematic — create manually or programmatically with tikz.
# Placeholder: architecture diagram showing metadata → encoder → h → W → x

# ── Figure 2: Speckled Holdout CV for Rank Selection ────────────
# TODO: Generate from actual NMF CV results
# Placeholder showing reconstruction error vs rank k
# k_values <- c(256, 512, 1024, 2048, 4096)
# errors <- c(0.45, 0.32, 0.22, 0.19, 0.18)
# holdout_errors <- c(0.48, 0.35, 0.24, 0.23, 0.28)
#
# df <- data.frame(
#   k = rep(k_values, 2),
#   error = c(errors, holdout_errors),
#   type = rep(c("Training", "Holdout"), each = length(k_values))
# )
#
# p2 <- ggplot(df, aes(x = k, y = error, color = type, linetype = type)) +
#   geom_line(linewidth = 0.8) +
#   geom_point(size = 2) +
#   scale_x_log10() +
#   labs(x = "Number of programs (k)", y = "Reconstruction error",
#        title = "Rank selection via speckled holdout CV") +
#   scale_color_manual(values = c("Training" = "#2166ac", "Holdout" = "#b2182b")) +
#   theme_singletai +
#   theme(legend.title = element_blank())
#
# ggsave("figure2.pdf", p2, width = 3.4, height = 2.5, units = "in")

cat("Figure rendering complete.\n")
cat("Note: Uncomment and supply real data to generate figures.\n")
