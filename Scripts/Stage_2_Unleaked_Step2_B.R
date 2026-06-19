# ============================================================
# Stage 2 Unleaked - Script 2b: SEM Visualizations
# ============================================================
# Requires: Script 2 fit objects in memory (for path diagrams)
#           Script 2 CSVs on disk (for all other plots)
#
# Adapted for the fixed-loading pipeline:
#   • Factor loadings are fixed — shown once from the pure model
#   • Regression paths are the primary focus
#   • g score is a single brain-free score (one distribution)
#
# Produces:
#   Path diagrams (up to 39)   : FigSEM_<method>_<model>_PathDiagram.png
#   Pure model loadings        : FigSEM_PureModel_Loadings.png
#   Fit index heatmap          : FigSEM_FitIndex_Heatmap.png
#   Regression path heatmap    : FigSEM_RegressionPaths_Heatmap.png
#   g score distribution       : FigSEM_gScore_Distribution.png
# ============================================================

library(semPlot)
library(tidyverse)

BASE     <- "C:/Thesis_Main/Analysis"
SEM_DIR  <- file.path(BASE, "Outputs/Stage_2_Unleaked/SEM")
OUT_FIGS <- file.path(SEM_DIR, "Figures")
dir.create(OUT_FIGS, recursive = TRUE, showWarnings = FALSE)

METHOD_ORDER  <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C")
METHOD_LABELS <- c("Schaefer\n(M0)", "gMSHBM\n(M1)", "AGP\n(M2)",
                   "SLIC_F\n(M3)", "SLIC_C\n(M4)")
names(METHOD_LABELS) <- METHOD_ORDER

MODEL_ORDER <- paste0("SEM", 1:9)

.vtheme <- theme_bw(base_size = 12) +
  theme(plot.title       = element_text(face = "bold", size = 13),
        plot.subtitle    = element_text(size = 10, color = "grey30"),
        axis.title       = element_text(size = 11),
        axis.text        = element_text(size = 10),
        panel.grid.minor = element_blank(),
        legend.title     = element_text(size = 10),
        legend.text      = element_text(size = 9))

cat("============================================================\n")
cat("Stage 2 Unleaked - Script 2b: SEM Visualizations\n")
cat("============================================================\n\n")

# ============================================================
# LOAD CSV DATA
# ============================================================

fit_indices_all <- read.csv(file.path(SEM_DIR, "sem_fit_indices.csv"),
                            stringsAsFactors = FALSE)
params_all      <- read.csv(file.path(SEM_DIR, "sem_parameters.csv"),
                            stringsAsFactors = FALSE)
g_scores        <- read.csv(file.path(SEM_DIR, "g_scores.csv"),
                            stringsAsFactors = FALSE)
pure_loadings   <- read.csv(file.path(SEM_DIR, "pure_model_loadings.csv"),
                            stringsAsFactors = FALSE)

cat("Loaded:\n")
cat("  sem_fit_indices.csv:", nrow(fit_indices_all), "rows\n")
cat("  sem_parameters.csv: ", nrow(params_all),      "rows\n")
cat("  g_scores.csv:        ", nrow(g_scores),        "subjects\n")
cat("  pure_model_loadings.csv:", nrow(pure_loadings), "loadings\n\n")

# ============================================================
# SECTION 1: PATH DIAGRAMS — all converged models
# ============================================================

cat("--- Section 1: Path Diagrams (all converged models) ---\n")

NODE_LABELS <- c(
  "g"                   = "g",
  "spd"                 = "Speed",
  "cry"                 = "Crystal.",
  "mem"                 = "Memory",
  "CardSort_Unadj"      = "CardSort",
  "Flanker_Unadj"       = "Flanker",
  "ProcSpeed_Unadj"     = "ProcSpeed",
  "PicVocab_Unadj"      = "PicVocab",
  "ReadEng_Unadj"       = "ReadEng",
  "PMAT24_A_CR"         = "PMAT",
  "VSPLOT_TC"           = "VSPLOT",
  "IWRD_TOT"            = "IWRD",
  "PicSeq_Unadj"        = "PicSeq",
  "GE_raw"              = "GE\n(raw)",
  "GE_top10"            = "GE\n(top10)",
  "GE_r05"              = "GE\n(r>0.5)",
  "ASPL_raw_r"          = "ASPL\n(raw)",
  "ASPL_top10_r"        = "ASPL\n(top10)",
  "ASPL_r05_r"          = "ASPL\n(r>0.5)",
  "Parcel_Size_CV"      = "PS-CV",
  "Parcel_Size_MeanSim" = "PS-Sim",
  "FC_MeanSim"          = "FC-Sim"
)

all_converged <- fit_indices_all %>%
  filter(status %in% c("ok", "ok_heywood")) %>%
  mutate(method = factor(method, levels = METHOD_ORDER),
         model  = factor(model,  levels = MODEL_ORDER)) %>%
  arrange(method, model) %>%
  select(method, model, CFI, RMSEA, status)

cat("Generating", nrow(all_converged), "path diagrams...\n\n")

diagrams_drawn <- 0
for (i in seq_len(nrow(all_converged))) {
  meth     <- as.character(all_converged$method[i])
  mod      <- as.character(all_converged$model[i])
  cfi      <- all_converged$CFI[i]
  rmsea    <- all_converged$RMSEA[i]
  status_i <- all_converged$status[i]
  obj_name <- paste0("fit_", meth, "_", mod)
  
  if (!exists(obj_name) || is.null(get(obj_name))) {
    cat(sprintf("  [SKIP] %s — fit object not in memory\n", obj_name))
    next
  }
  
  fit_obj <- get(obj_name)
  
  p <- semPaths(
    fit_obj,
    what           = "std.all",
    whatLabels     = "std.all",
    layout         = "tree2",
    rotation       = 2,
    style          = "ram",
    edge.color     = "black",
    edge.label.cex = 0.75,
    label.cex      = 0.80,
    residuals      = FALSE,
    intercepts     = FALSE,
    mar            = c(6, 5, 6, 5),
    fade           = FALSE,
    weighted       = FALSE,
    asize          = 2.5,
    DoNotPlot      = TRUE
  )
  
  full_names <- semPlotModel(fit_obj)@Vars$name
  extended_labels <- NODE_LABELS
  for (j in seq_along(full_names)) {
    fn <- full_names[j]
    if (fn %in% names(NODE_LABELS)) {
      for (ml in 3:6) {
        abbr <- abbreviate(fn, minlength = ml, strict = FALSE)
        extended_labels[abbr] <- NODE_LABELS[fn]
      }
    }
  }
  
  internal_names <- p$graphAttributes$Nodes$labels
  node_labs <- ifelse(internal_names %in% names(extended_labels),
                      extended_labels[internal_names],
                      internal_names)
  p$graphAttributes$Nodes$labels <- node_labs
  
  filename <- sprintf("FigSEM_%s_%s_PathDiagram.png", meth, mod)
  png(file.path(OUT_FIGS, filename), width = 1800, height = 1100, res = 130)
  plot(p)
  title(
    main = sprintf("%s — %s: Fixed-Loading Bifactor SEM", meth, mod),
    sub  = sprintf("CFI = %.3f | RMSEA = %.3f | Loadings fixed from pure model%s",
                   cfi, rmsea,
                   ifelse(status_i == "ok_heywood", "  [Heywood]", "")),
    cex.main = 1.05, cex.sub = 0.80, line = 1
  )
  dev.off()
  
  cat(sprintf("  [%d] Saved: %s\n", diagrams_drawn + 1, filename))
  diagrams_drawn <- diagrams_drawn + 1
}

cat(sprintf("\n%d path diagrams saved.\n", diagrams_drawn))
if (diagrams_drawn < nrow(all_converged)) {
  cat("NOTE: Some fit objects were not in memory.\n")
  cat("Re-run Script 2 and then immediately run this script.\n")
}

# ============================================================
# SECTION 2: PURE MODEL LOADINGS — bar chart
# ============================================================

cat("\n--- Section 2: Pure Model Loadings ---\n")

test_labels <- c(
  "CardSort_Unadj"  = "CardSort",  "Flanker_Unadj"   = "Flanker",
  "ProcSpeed_Unadj" = "ProcSpeed", "PicVocab_Unadj"  = "PicVocab",
  "ReadEng_Unadj"   = "ReadEng",   "PMAT24_A_CR"     = "PMAT",
  "VSPLOT_TC"       = "VSPLOT",    "IWRD_TOT"        = "IWRD",
  "PicSeq_Unadj"    = "PicSeq"
)

load_plot <- pure_loadings %>%
  mutate(
    test   = recode(rhs, !!!test_labels),
    factor = factor(lhs, levels = c("g", "spd", "cry", "mem")),
    test   = factor(test, levels = rev(test_labels))
  )

png(file.path(OUT_FIGS, "FigSEM_PureModel_Loadings.png"),
    width = 2200, height = 1600, res = 200)
print(
  ggplot(load_plot, aes(x = test, y = est, fill = factor)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_text(aes(label = sprintf("%.2f", est)),
              position = position_dodge(width = 0.7),
              hjust = -0.15, size = 3) +
    coord_flip() +
    scale_fill_brewer(palette = "Set2", name = "Factor") +
    labs(title    = "Pure Bifactor Model: Factor Loadings (Unstandardized)",
         subtitle = "Estimated from cognitive tests only (no brain features). These values are FIXED in all subsequent SEMs.",
         x = "Cognitive Test", y = "Unstandardized Loading") +
    .vtheme + theme(legend.position = "bottom")
)
dev.off()
cat("  Saved: FigSEM_PureModel_Loadings.png\n")

# ============================================================
# SECTION 3: FIT INDEX HEATMAP
# ============================================================

cat("--- Section 3: Fit Index Heatmap ---\n")

fit_heatmap_data <- fit_indices_all %>%
  filter(status %in% c("ok", "ok_heywood")) %>%
  mutate(method = factor(method, levels = METHOD_ORDER),
         model  = factor(model,  levels = MODEL_ORDER),
         label  = sprintf("CFI=%.2f\nRMSEA=%.2f", CFI, RMSEA))

png(file.path(OUT_FIGS, "FigSEM_FitIndex_Heatmap.png"),
    width = 2400, height = 1400, res = 200)
print(
  ggplot(fit_heatmap_data, aes(x = model, y = method, fill = CFI)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = label), size = 2.5, lineheight = 0.9) +
    scale_fill_gradient2(low = "#d73027", mid = "#ffffbf", high = "#1a9850",
                         midpoint = 0.95, limits = c(0.80, 1.00),
                         name = "CFI") +
    scale_y_discrete(labels = METHOD_LABELS) +
    labs(title    = "Fixed-Loading SEM Fit Indices across Methods and Model Variants",
         subtitle = "Loadings fixed from pure model. Green = good fit (CFI ≥ 0.95, RMSEA ≤ 0.06).",
         x = "Model Variant", y = "Parcellation Method") +
    .vtheme
)
dev.off()
cat("  Saved: FigSEM_FitIndex_Heatmap.png\n")

# ============================================================
# SECTION 4: REGRESSION PATH HEATMAP
# ============================================================

cat("--- Section 4: Regression Paths Heatmap ---\n")

reg_labels <- c(
  "GE_raw"              = "GE (raw)",     "GE_top10"            = "GE (top10)",
  "GE_r05"              = "GE (r>0.5)",   "ASPL_raw_r"          = "ASPL (raw)",
  "ASPL_top10_r"        = "ASPL (top10)", "ASPL_r05_r"          = "ASPL (r>0.5)",
  "Parcel_Size_CV"      = "PS-CV",         "Parcel_Size_MeanSim" = "PS-Sim",
  "FC_MeanSim"          = "FC-Sim"
)

reg_avg <- params_all %>%
  filter(status %in% c("ok", "ok_heywood")) %>%
  mutate(method    = factor(method, levels = METHOD_ORDER),
         predictor = recode(rhs, !!!reg_labels)) %>%
  group_by(method, predictor) %>%
  summarise(mean_est = round(mean(std_est), 3), .groups = "drop") %>%
  mutate(method = factor(method, levels = METHOD_ORDER))

png(file.path(OUT_FIGS, "FigSEM_RegressionPaths_Heatmap.png"),
    width = 2400, height = 1400, res = 200)
print(
  ggplot(reg_avg, aes(x = predictor, y = method, fill = mean_est)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", mean_est)), size = 3) +
    scale_fill_gradient2(low = "#d73027", mid = "white", high = "#1a9850",
                         midpoint = 0, name = "Std. β") +
    scale_y_discrete(labels = METHOD_LABELS) +
    labs(title    = "Regression Paths: Brain Features → Latent g (std.all)",
         subtitle = "Loadings fixed from pure model. Mean β across converged variants. Green = positive, Red = negative.",
         x = "Brain Feature", y = "Parcellation Method") +
    .vtheme +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
)
dev.off()
cat("  Saved: FigSEM_RegressionPaths_Heatmap.png\n")

# ============================================================
# SECTION 5: G SCORE DISTRIBUTION — single brain-free g score
# ============================================================

cat("--- Section 5: g Score Distribution ---\n")

png(file.path(OUT_FIGS, "FigSEM_gScore_Distribution.png"),
    width = 2000, height = 1400, res = 200)
print(
  ggplot(g_scores, aes(x = g_score)) +
    geom_histogram(aes(y = after_stat(density)),
                   bins = 20, fill = "#4daf4a", alpha = 0.7,
                   color = "white") +
    geom_density(linewidth = 1.0, color = "#1a9850") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    labs(title    = "Latent g Score Distribution (Brain-Free)",
         subtitle = paste0("Single g score from pure bifactor SEM (cognitive tests only, N = ",
                           nrow(g_scores), "). No brain feature contamination."),
         x = "g Score", y = "Density") +
    .vtheme
)
dev.off()
cat("  Saved: FigSEM_gScore_Distribution.png\n")

# ============================================================
# DONE
# ============================================================

cat("\n============================================================\n")
cat("Script 2b complete.\n")
cat("Saved to:", OUT_FIGS, "\n\n")
cat("Path diagrams       : FigSEM_<method>_<model>_PathDiagram.png\n")
cat("Pure model loadings : FigSEM_PureModel_Loadings.png\n")
cat("Fit heatmap         : FigSEM_FitIndex_Heatmap.png\n")
cat("Regression heatmap  : FigSEM_RegressionPaths_Heatmap.png\n")
cat("g score distribution: FigSEM_gScore_Distribution.png\n")
cat("============================================================\n")