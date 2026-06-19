# ============================================================
# Stage 2 Unleaked - Script 1: Data Loading, Feature Computation,
#                               Descriptive Statistics & Violin Plots
# ============================================================
# Identical to the original Stage 2 Script 1 — the input data
# and brain features are not affected by the leakage fix.
# Only the output folder changes to Stage_2_Unleaked.
# ============================================================

library(tidyverse)

# ============================================================
# PATHS
# ============================================================

BASE     <- "C:/Thesis_Main/Analysis"
BEH_DIR  <- file.path(BASE, "Inputs/Behavior")
HCP_FILE <- file.path(BEH_DIR, "HCP_all_data.csv")
FEAT_DIR <- file.path(BASE, "Outputs/Features")

F1F2_DIR <- file.path(FEAT_DIR, "F1_F2_Graph_Metrics")
F3_DIR   <- file.path(FEAT_DIR, "F3_Parcel_Sizes")
F4A_DIR  <- file.path(FEAT_DIR, "F4A_FC_Matrices")
F4B_DIR  <- file.path(FEAT_DIR, "F4B_FC_Similarity")

OUT_DATA <- file.path(BASE, "Outputs/Stage_2_Unleaked/Input_Data")
OUT_FIGS <- file.path(BASE, "Outputs/Stage_2_Unleaked/Figures")
dir.create(OUT_DATA, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT_FIGS, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# CONSTANTS
# ============================================================

COG_COLS <- c(
  "CardSort_Unadj", "Flanker_Unadj", "ProcSpeed_Unadj",
  "PicVocab_Unadj", "ReadEng_Unadj", "PMAT24_A_CR",
  "VSPLOT_TC", "IWRD_TOT", "PicSeq_Unadj"
)

DOMAINS <- c(
  "Episodic_memory", "Executive_function", "Fluid_Intelligence",
  "Language", "Processing_Speed", "Self_regulation",
  "Spatial_Orientation", "Sustained_visual_attention",
  "Verbal_episodic_memory", "Working_Memory"
)

METHOD_ORDER  <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C")
METHOD_LABELS <- c("Schaefer\n(M0)", "gMSHBM\n(M1)", "AGP\n(M2)",
                   "SLIC_F\n(M3)", "SLIC_C\n(M4)")

MISSING_SUBJECTS <- c("117122", "151526", "156637", "298051", "366446")
N_PARCELS <- 200

# ============================================================
# HELPER
# ============================================================

mean_off_diagonal <- function(mat) {
  n        <- nrow(mat)
  row_sums <- rowSums(mat) - diag(mat)
  row_sums / (n - 1)
}

# ============================================================
# STEP 1: Subject list
# ============================================================

cat("============================================================\n")
cat("Stage 2 Unleaked - Script 1: Data Loading\n")
cat("============================================================\n\n")

all_subjects <- trimws(readLines(file.path(BEH_DIR, "list_subjects.txt")))
all_subjects <- all_subjects[nchar(all_subjects) > 0]
subjects     <- all_subjects[!all_subjects %in% MISSING_SUBJECTS]
n            <- length(subjects)

cat("Total subjects in list   :", length(all_subjects), "\n")
cat("Excluded (no gMSHBM)     :", paste(MISSING_SUBJECTS, collapse = ", "), "\n")
cat("Subjects used in analysis:", n, "\n\n")
stopifnot(n == 95)

# ============================================================
# STEP 2: Cognitive test data
# ============================================================

cat("Loading HCP cognitive data...\n")
HCP_Data <- read.csv(HCP_FILE)

cog_data <- HCP_Data %>%
  filter(Subject %in% as.integer(subjects)) %>%
  select(Subject, all_of(COG_COLS)) %>%
  mutate(subject_id = as.character(Subject)) %>%
  select(-Subject) %>%
  arrange(match(subject_id, subjects)) %>%
  select(subject_id, everything())

stopifnot(nrow(cog_data) == 95)
stopifnot(all(cog_data$subject_id == subjects))
stopifnot(sum(is.na(cog_data)) == 0)

write.csv(cog_data, file.path(OUT_DATA, "cog_data.csv"), row.names = FALSE)
cat("  Cognitive data: 95 subjects x", length(COG_COLS), "tests — saved\n\n")

# ============================================================
# STEP 3: Brain features per method
# ============================================================

load_method <- function(method) {
  cat(strrep("-", 50), "\n")
  cat("Processing:", method, "\n")
  
  f1f2  <- read.csv(file.path(F1F2_DIR, paste0("graph_metrics_", method, ".csv")))
  f3    <- read.csv(file.path(F3_DIR,   paste0("parcel_sizes_",  method, ".csv")))
  sizes <- as.matrix(f3[, paste0("parcel_", 1:N_PARCELS)])
  f3_sim  <- cor(t(sizes))
  f4b_sim <- as.matrix(read.csv(
    file.path(F4B_DIR, paste0("fc_similarity_", method, ".csv")),
    row.names = 1, check.names = FALSE))
  
  df <- data.frame(
    subject_id          = subjects,
    GE_raw              = f1f2$global_efficiency_raw,
    GE_top10            = f1f2$global_efficiency_top10,
    GE_r05              = f1f2$global_efficiency_r05,
    ASPL_raw            = f1f2$avg_shortest_path_raw,
    ASPL_top10          = f1f2$avg_shortest_path_top10,
    ASPL_r05            = f1f2$avg_shortest_path_r05,
    ASPL_raw_r          = -f1f2$avg_shortest_path_raw,
    ASPL_top10_r        = -f1f2$avg_shortest_path_top10,
    ASPL_r05_r          = -f1f2$avg_shortest_path_r05,
    Parcel_Size_CV      = apply(sizes, 1, function(x) sd(x) / mean(x)),
    Parcel_Size_MeanSim = mean_off_diagonal(f3_sim),
    FC_MeanSim          = mean_off_diagonal(f4b_sim),
    stringsAsFactors    = FALSE)
  
  cat("  df:", nrow(df), "rows x", ncol(df), "columns\n")
  cat("  GE_raw    : mean =", round(mean(df$GE_raw),    4),
      " sd =", round(sd(df$GE_raw),    4), "\n")
  cat("  ASPL_raw  : mean =", round(mean(df$ASPL_raw),  4),
      " sd =", round(sd(df$ASPL_raw),  4), "\n")
  cat("  FC_MeanSim: mean =", round(mean(df$FC_MeanSim),4),
      " sd =", round(sd(df$FC_MeanSim),4), "\n\n")
  
  write.csv(df,
            file.path(OUT_DATA, paste0("brain_features_", method, ".csv")),
            row.names = FALSE)
  write.csv(as.data.frame(f3_sim),
            file.path(OUT_DATA, paste0("f3_similarity_", method, ".csv")),
            row.names = TRUE)
  write.csv(as.data.frame(f4b_sim),
            file.path(OUT_DATA, paste0("fc_similarity_", method, ".csv")),
            row.names = TRUE)
  
  list(df = df, f3_sim = f3_sim, fc_sim = f4b_sim)
}

r_Schaefer <- load_method("Schaefer")
r_gMSHBM   <- load_method("gMSHBM")
r_AGP      <- load_method("AGP")
r_SLIC_F   <- load_method("SLIC_F")
r_SLIC_C   <- load_method("SLIC_C")

df_Schaefer <- r_Schaefer$df ; df_gMSHBM <- r_gMSHBM$df
df_AGP      <- r_AGP$df      ; df_SLIC_F <- r_SLIC_F$df
df_SLIC_C   <- r_SLIC_C$df

f3_sim_Schaefer <- r_Schaefer$f3_sim ; f3_sim_gMSHBM <- r_gMSHBM$f3_sim
f3_sim_AGP      <- r_AGP$f3_sim      ; f3_sim_SLIC_F <- r_SLIC_F$f3_sim
f3_sim_SLIC_C   <- r_SLIC_C$f3_sim

fc_sim_Schaefer <- r_Schaefer$fc_sim ; fc_sim_gMSHBM <- r_gMSHBM$fc_sim
fc_sim_AGP      <- r_AGP$fc_sim      ; fc_sim_SLIC_F <- r_SLIC_F$fc_sim
fc_sim_SLIC_C   <- r_SLIC_C$fc_sim

# ============================================================
# STEP 4: Descriptive statistics
# ============================================================

cat("Computing descriptive statistics...\n")

brain_long <- bind_rows(
  df_Schaefer %>% mutate(method = "Schaefer"),
  df_gMSHBM   %>% mutate(method = "gMSHBM"),
  df_AGP       %>% mutate(method = "AGP"),
  df_SLIC_F    %>% mutate(method = "SLIC_F"),
  df_SLIC_C    %>% mutate(method = "SLIC_C")
) %>% mutate(method = factor(method, levels = METHOD_ORDER))

desc_stats <- brain_long %>%
  group_by(method) %>%
  summarise(
    GE_raw_mean              = round(mean(GE_raw),              4),
    GE_raw_sd                = round(sd(GE_raw),                4),
    GE_top10_mean            = round(mean(GE_top10),            4),
    GE_top10_sd              = round(sd(GE_top10),              4),
    GE_r05_mean              = round(mean(GE_r05),              4),
    GE_r05_sd                = round(sd(GE_r05),                4),
    ASPL_raw_mean            = round(mean(ASPL_raw),            4),
    ASPL_raw_sd              = round(sd(ASPL_raw),              4),
    ASPL_top10_mean          = round(mean(ASPL_top10),          4),
    ASPL_top10_sd            = round(sd(ASPL_top10),            4),
    ASPL_r05_mean            = round(mean(ASPL_r05),            4),
    ASPL_r05_sd              = round(sd(ASPL_r05),              4),
    Parcel_Size_CV_mean      = round(mean(Parcel_Size_CV),      4),
    Parcel_Size_CV_sd        = round(sd(Parcel_Size_CV),        4),
    Parcel_Size_MeanSim_mean = round(mean(Parcel_Size_MeanSim), 4),
    Parcel_Size_MeanSim_sd   = round(sd(Parcel_Size_MeanSim),   4),
    FC_MeanSim_mean          = round(mean(FC_MeanSim),          4),
    FC_MeanSim_sd            = round(sd(FC_MeanSim),            4),
    .groups = "drop")

write.csv(desc_stats, file.path(OUT_DATA, "descriptive_stats.csv"), row.names = FALSE)
cat("  Saved: descriptive_stats.csv\n\n")

cat("DESCRIPTIVE STATISTICS SUMMARY\n")
cat(strrep("-", 60), "\n")
desc_stats %>%
  select(method,
         `GE_raw M`  = GE_raw_mean,    `GE_raw SD`  = GE_raw_sd,
         `ASPL_raw M`= ASPL_raw_mean,  `ASPL_raw SD`= ASPL_raw_sd,
         `FC_Sim M`  = FC_MeanSim_mean,`FC_Sim SD`  = FC_MeanSim_sd) %>%
  as.data.frame() %>% print(row.names = FALSE)

# ============================================================
# STEP 5: Violin plots
# ============================================================

cat("\nGenerating violin plots...\n")

.vtheme <- theme_bw(base_size = 13) +
  theme(plot.title       = element_text(face = "bold", size = 14),
        plot.subtitle    = element_text(size = 11, color = "grey30"),
        axis.title       = element_text(size = 12),
        axis.text        = element_text(size = 11),
        panel.grid.minor = element_blank())

.pd <- brain_long
.pd$method <- factor(.pd$method, levels = METHOD_ORDER)

# -- FigS1a: GE raw --
png(file.path(OUT_FIGS, "FigS1a_violin_GE_raw.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = GE_raw, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        labs(title    = "Global Efficiency across Parcellation Methods",
             subtitle = "Raw weighted graph variant. Higher = more efficient network.",
             x = "Parcellation Method (M0 → M4)", y = "Global Efficiency") + .vtheme)
dev.off()
cat("  Saved: FigS1a_violin_GE_raw.png\n")

# -- FigS1b: GE top10 --
png(file.path(OUT_FIGS, "FigS1b_violin_GE_top10.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = GE_top10, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        labs(title    = "Global Efficiency (top 10%) across Parcellation Methods",
             subtitle = "Top 10% connection threshold. Higher = more efficient network.",
             x = "Parcellation Method (M0 → M4)", y = "Global Efficiency (top 10%)") + .vtheme)
dev.off()
cat("  Saved: FigS1b_violin_GE_top10.png\n")

# -- FigS1c: GE r05 --
png(file.path(OUT_FIGS, "FigS1c_violin_GE_r05.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = GE_r05, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        labs(title    = "Global Efficiency (r > 0.5) across Parcellation Methods",
             subtitle = "r > 0.5 threshold. Higher = more efficient network.",
             x = "Parcellation Method (M0 → M4)", y = "Global Efficiency (r > 0.5)") + .vtheme)
dev.off()
cat("  Saved: FigS1c_violin_GE_r05.png\n")

# -- FigS2a: ASPL raw --
png(file.path(OUT_FIGS, "FigS2a_violin_ASPL_raw.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = ASPL_raw, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        labs(title    = "Average Shortest Path Length across Parcellation Methods",
             subtitle = "Raw weighted graph variant. Lower = more efficient network.",
             x = "Parcellation Method (M0 → M4)", y = "Average Shortest Path Length") + .vtheme)
dev.off()
cat("  Saved: FigS2a_violin_ASPL_raw.png\n")

# -- FigS2b: ASPL top10 --
png(file.path(OUT_FIGS, "FigS2b_violin_ASPL_top10.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = ASPL_top10, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        labs(title    = "Average Shortest Path Length (top 10%) across Parcellation Methods",
             subtitle = "Top 10% connection threshold. Lower = more efficient network.",
             x = "Parcellation Method (M0 → M4)", y = "ASPL (top 10%)") + .vtheme)
dev.off()
cat("  Saved: FigS2b_violin_ASPL_top10.png\n")

# -- FigS2c: ASPL r05 --
png(file.path(OUT_FIGS, "FigS2c_violin_ASPL_r05.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = ASPL_r05, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        labs(title    = "Average Shortest Path Length (r > 0.5) across Parcellation Methods",
             subtitle = "r > 0.5 threshold. Lower = more efficient network.",
             x = "Parcellation Method (M0 → M4)", y = "ASPL (r > 0.5)") + .vtheme)
dev.off()
cat("  Saved: FigS2c_violin_ASPL_r05.png\n")

# -- FigS3: FC Mean Similarity --
png(file.path(OUT_FIGS, "FigS3_violin_FC_MeanSim.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = FC_MeanSim, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        coord_cartesian(ylim = c(0, NA)) +
        labs(title    = "FC Mean Similarity across Parcellation Methods",
             subtitle = "Mean off-diagonal of 95x95 FC similarity matrix per subject.",
             x = "Parcellation Method (M0 → M4)", y = "Mean FC Similarity") + .vtheme)
dev.off()
cat("  Saved: FigS3_violin_FC_MeanSim.png\n")

# -- FigS4: Parcel Size CV --
png(file.path(OUT_FIGS, "FigS4_violin_Parcel_Size_CV.png"), width = 2400, height = 1500, res = 300)
print(ggplot(.pd, aes(x = method, y = Parcel_Size_CV, fill = method)) +
        geom_violin(trim = TRUE, alpha = 0.75) +
        geom_boxplot(width = 0.10, fill = "white", outlier.size = 1.5, outlier.alpha = 0.6) +
        scale_x_discrete(labels = METHOD_LABELS) +
        scale_fill_brewer(palette = "Set2", guide = "none") +
        coord_cartesian(ylim = c(0, NA)) +
        labs(title    = "Parcel Size CV across Parcellation Methods",
             subtitle = "Higher = more variable parcel sizes. Schaefer near-zero by construction.",
             x = "Parcellation Method (M0 → M4)", y = "Parcel Size CV (SD/Mean)") + .vtheme)
dev.off()
cat("  Saved: FigS4_violin_Parcel_Size_CV.png\n")

# ============================================================
# STEP 6: Done
# ============================================================

cat("\n============================================================\n")
cat("Script 1 complete.\n")
cat("============================================================\n")
cat("Saved to:", OUT_DATA, "\n")
cat("  cog_data.csv\n")
cat("  brain_features_<method>.csv  (5 files)\n")
cat("  fc_similarity_<method>.csv   (5 files)\n")
cat("  f3_similarity_<method>.csv   (5 files)\n")
cat("  descriptive_stats.csv\n\n")
cat("Saved to:", OUT_FIGS, "\n")
cat("  FigS1a-c  GE violin plots (3 variants)\n")
cat("  FigS2a-c  ASPL violin plots (3 variants)\n")
cat("  FigS3     FC Mean Similarity\n")
cat("  FigS4     Parcel Size CV\n\n")
cat("Objects in memory:\n")
cat("  subjects, cog_data, COG_COLS, DOMAINS\n")
cat("  df_Schaefer, df_gMSHBM, df_AGP, df_SLIC_F, df_SLIC_C\n")
cat("  fc_sim_*, f3_sim_*, brain_long\n")
cat("============================================================\n")