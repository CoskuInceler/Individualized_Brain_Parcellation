# ============================================================
# Stage 2 Unleaked - Script 3B: KRR Visualizations
# ============================================================
# g SCORE ONLY — behavioral domain plots skipped.
# To add them back, uncomment the behavioral figures and load
# krr_behavioral_results.csv.
#
# Produces:
#   FigKRR_gScore_BarChart.png
# ============================================================

library(tidyverse)

BASE    <- "C:/Thesis_Main/Analysis"
KRR_DIR <- file.path(BASE, "Outputs/Stage_2_Unleaked/KernelRegression")
OUT_DIR <- file.path(KRR_DIR, "Figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

METHOD_ORDER  <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C")
METHOD_LABELS <- c("Schaefer\n(M0)", "gMSHBM\n(M1)", "AGP\n(M2)",
                   "SLIC_F\n(M3)", "SLIC_C\n(M4)")
names(METHOD_LABELS) <- METHOD_ORDER

.vtheme <- theme_bw(base_size = 12) +
  theme(plot.title       = element_text(face = "bold", size = 13),
        plot.subtitle    = element_text(size = 10, color = "grey30"),
        axis.title       = element_text(size = 11),
        axis.text        = element_text(size = 10),
        panel.grid.minor = element_blank(),
        legend.title     = element_text(size = 10),
        legend.text      = element_text(size = 9))

cat("============================================================\n")
cat("Stage 2 Unleaked - Script 3B: KRR Visualizations\n")
cat("  (g score only — behavioral plots skipped)\n")
cat("============================================================\n\n")

# ============================================================
# LOAD DATA
# ============================================================

g_results <- read.csv(file.path(KRR_DIR, "krr_g_results.csv"),
                      stringsAsFactors = FALSE)

cat("Loaded:\n")
cat("  krr_g_results.csv:", nrow(g_results), "rows\n\n")

# ============================================================
# FIGURE 1: g Score Prediction Bar Chart (method × kernel)
# ============================================================

cat("--- Figure 1: g Score Prediction Bar Chart ---\n")

g_plot <- g_results %>%
  filter(!is.na(r)) %>%
  mutate(method       = factor(method, levels = METHOD_ORDER),
         kernel_label = recode(kernel,
                               "F3_ParcelSize" = "F3: Parcel Size",
                               "F4B_FC"        = "F4B: FC Similarity"),
         label = sprintf("%.2f%s", r, ifelse(significant, "*", "")))

png(file.path(OUT_DIR, "FigKRR_gScore_BarChart.png"),
    width = 2400, height = 1500, res = 220)
print(
  ggplot(g_plot, aes(x = method, y = r, fill = kernel_label)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_text(aes(label = label),
              position = position_dodge(width = 0.7),
              vjust = -0.5, size = 3.5, fontface = "bold") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    scale_fill_manual(values = c("F3: Parcel Size"    = "#e41a1c",
                                 "F4B: FC Similarity" = "#377eb8"),
                      name = "Kernel") +
    scale_x_discrete(labels = METHOD_LABELS) +
    coord_cartesian(ylim = c(min(g_plot$r, 0) - 0.05,
                             max(g_plot$r, 0) + 0.10)) +
    labs(title    = "KRR Prediction of Brain-Free g Score",
         subtitle = "Single g score from pure bifactor model (no brain contamination). * p < .05. Covariate-regressed.",
         x = "Parcellation Method (M0 → M4)", y = "Pearson r") +
    .vtheme + theme(legend.position = "bottom")
)
dev.off()
cat("  Saved: FigKRR_gScore_BarChart.png\n")

# ============================================================
# DONE
# ============================================================

cat("\n============================================================\n")
cat("Script 3B complete. Figures saved to:\n", OUT_DIR, "\n\n")
cat("  FigKRR_gScore_BarChart.png\n")
cat("\nNOTE: Behavioral domain plots skipped.\n")
cat("============================================================\n")