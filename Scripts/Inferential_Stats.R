# =============================================================================
#  Inferential statistics for Stage 1 (parcellation validation) and the
#  feature-extraction comparison (Results sections 3.1 and 3.2).
#
#  Regenerates every RM-ANOVA, Friedman test, and effect size in the Results,
#  directly from the per-subject data, so the numbers can be reproduced and
#  verified independently.
#
#  Conventions (standard for repeated-measures designs in psychology):
#    * RM-ANOVA  -> afex::aov_ez (Greenhouse-Geisser correction;
#                   generalized eta-squared, ges = eta^2_G).
#    * Friedman  -> base R friedman.test (used where a method has zero variance).
#    * Pairwise significance -> paired t-tests with Holm correction.
#    * Effect-size magnitude -> pooled Cohen's d_av = mean(diff) / mean(SD1, SD2).
#
#  Analysis sample: the 95 subjects present in all five methods.
#  Required packages: afex, dplyr, tidyr
#    install once with: install.packages(c("afex","dplyr","tidyr"))
# =============================================================================

suppressMessages({library(afex); library(dplyr); library(tidyr)})

## ---- 0. EDIT THESE TWO PATHS ------------------------------------------------
# Use forward slashes (R accepts them on Windows), or double backslashes.
comp_csv <- "C:/Thesis_Main/Thesis_Stage1_Results/Master_Comparison/Comparison_Raw_All.csv"
tar_path <- "C:/Thesis_Main/Daniel_Package.tar.gz"
methods  <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C")

## ---- 0b. Pull only the 10 needed files out of the .tar.gz (no full extract) --
# The archive is large only because of the FC matrices, which are not used here.
need <- c(paste0("Daniel_Package/F1_F2_Graph_Metrics/graph_metrics_", methods, ".csv"),
          paste0("Daniel_Package/F4B_FC_Similarity/fc_similarity_", methods, ".csv"))
exdir <- tempfile("dpkg_"); dir.create(exdir)
untar(tar_path, files = need, exdir = exdir)   # extracts just these 10 small files
gdir <- file.path(exdir, "Daniel_Package", "F1_F2_Graph_Metrics")
fdir <- file.path(exdir, "Daniel_Package", "F4B_FC_Similarity")

## ---- helpers ----------------------------------------------------------------
dav <- function(a, b) { (mean(a) - mean(b)) / ((sd(a) + sd(b)) / 2) }  # pooled Cohen's d_av

rm_anova <- function(data, dv, id, label, thesis) {
  a  <- suppressMessages(aov_ez(id = id, dv = dv, data = data, within = "Method"))
  at <- a$anova_table
  cat(sprintf("  %-26s F = %9.1f   eta2_G = %.3f   p = %s   [thesis: %s]\n",
              label, at$F[1], at$ges[1], format.pval(at$`Pr(>F)`[1], digits = 3), thesis))
  invisible(a)
}

posthoc <- function(wide_df, cols) {
  pm  <- combn(cols, 2, simplify = FALSE)
  res <- lapply(pm, function(p) {
    a <- wide_df[[p[1]]]; b <- wide_df[[p[2]]]
    data.frame(pair = paste(p[1], "vs", p[2]),
               d_av = round(dav(a, b), 2),
               p_raw = t.test(a, b, paired = TRUE)$p.value)
  })
  res <- do.call(rbind, res)
  res$p_holm <- format.pval(p.adjust(res$p_raw, method = "holm"), digits = 3)
  res$p_raw  <- NULL
  print(res, row.names = FALSE)
}

## ---- 1. Validation metrics: load + restrict to the 95 common subjects -------
comp   <- read.csv(comp_csv)
common <- Reduce(intersect, lapply(methods, function(m) comp$Subject_ID[comp$Method == m]))
cat("Analysis sample: N =", length(common), "subjects present in all five methods\n\n")

comp <- comp[comp$Subject_ID %in% common, ]
comp$Method     <- factor(comp$Method, levels = methods)
comp$Subject_ID <- factor(comp$Subject_ID)

homog_w <- comp %>% select(Subject_ID, Method, Homogeneity_Mean) %>%
  pivot_wider(names_from = Method, values_from = Homogeneity_Mean)
trt_w   <- comp %>% select(Subject_ID, Method, TestRetest_R) %>%
  pivot_wider(names_from = Method, values_from = TestRetest_R)

## ---- 3.1.1 Functional homogeneity -------------------------------------------
cat("=== 3.1.1 Functional homogeneity ===\n")
print(round(tapply(comp$Homogeneity_Mean, comp$Method, mean), 3))
rm_anova(comp, "Homogeneity_Mean", "Subject_ID", "RM-ANOVA", "F(4,376)=2062.5, eta2G=0.20")
cat("  Key effect sizes (pooled d_av):\n")
cat(sprintf("    SLIC_C vs Schaefer: %.2f   [report ~1.30]\n", abs(dav(homog_w$SLIC_C, homog_w$Schaefer))))
cat(sprintf("    SLIC_C vs SLIC_F:   %.2f   [report ~1.46]\n", abs(dav(homog_w$SLIC_C, homog_w$SLIC_F))))
cat("  Holm-corrected post-hoc (all pairs):\n"); posthoc(homog_w, methods); cat("\n")

## ---- 3.1.2 Test-retest reliability ------------------------------------------
cat("=== 3.1.2 Test-retest reliability ===\n")
print(round(tapply(comp$TestRetest_R, comp$Method, mean), 3))
rm_anova(comp, "TestRetest_R", "Subject_ID", "RM-ANOVA", "F(4,376)=406.5, eta2G=0.16")
cat("  Holm-corrected post-hoc (all pairs); d_av ~0.05-0.19 among the first four,\n")
cat("  and ~0.9-1.1 for SLIC_C vs each of the others:\n")
posthoc(trt_w, methods); cat("\n")

## ---- 3.1.3 Spatial contiguity (Friedman: Schaefer & AGP are constant) --------
cat("=== 3.1.3 Spatial contiguity ===\n")
print(round(tapply(comp$Contiguity_Pct_Combined, comp$Method, mean), 3))
ft <- friedman.test(Contiguity_Pct_Combined ~ Method | Subject_ID, data = comp)
cat(sprintf("  Friedman: chi2(%d) = %.1f, p = %s   [thesis: chi2(4)=363.7, p<.001]\n\n",
            ft$parameter, ft$statistic, format.pval(ft$p.value, digits = 3)))

## ---- 3.1.4 Inter-subject Dice (descriptive only; one value per method) -------
cat("=== 3.1.4 Inter-subject Dice (means; no inferential test) ===\n")
print(round(tapply(comp$InterSubject_Dice_Mean, comp$Method, mean), 3)); cat("\n")

## ---- 2. Feature extraction: graph metrics (GE, ASPL) ------------------------
gm <- do.call(rbind, lapply(methods, function(m) {
  x <- read.csv(file.path(gdir, paste0("graph_metrics_", m, ".csv"))); x$Method <- m; x
}))
gm <- gm[gm$subject_id %in% common, ]
gm$Method     <- factor(gm$Method, levels = methods)
gm$subject_id <- factor(gm$subject_id)

cat("=== 3.2.1 Global efficiency ===\n")
print(round(sapply(c("global_efficiency_raw","global_efficiency_top10","global_efficiency_r05"),
                   function(v) tapply(gm[[v]], gm$Method, mean)), 3))
rm_anova(gm, "global_efficiency_raw",  "subject_id", "GE raw",   "F=435.7, eta2G=0.07")
rm_anova(gm, "global_efficiency_top10","subject_id", "GE top10", "F=12.5,  eta2G=0.01")
rm_anova(gm, "global_efficiency_r05",  "subject_id", "GE r>0.5", "F=422.1, eta2G=0.11")
cat("\n")

cat("=== 3.2.2 Average shortest path length ===\n")
print(round(sapply(c("avg_shortest_path_raw","avg_shortest_path_top10","avg_shortest_path_r05"),
                   function(v) tapply(gm[[v]], gm$Method, mean)), 3))
rm_anova(gm, "avg_shortest_path_raw",  "subject_id", "ASPL raw",   "F=361.2, eta2G=0.08")
rm_anova(gm, "avg_shortest_path_top10","subject_id", "ASPL top10", "F=112.0, eta2G=0.29")
rm_anova(gm, "avg_shortest_path_r05",  "subject_id", "ASPL r>0.5", "F=20.3,  eta2G=0.01")
cat("\n")

## ---- 3.2.3 Parcel-size CV (Friedman: Schaefer is constant) ------------------
cat("=== 3.2.3 Parcel-size coefficient of variation ===\n")
print(round(tapply(comp$Parcel_Size_CV, comp$Method, mean), 3))
ftcv <- friedman.test(Parcel_Size_CV ~ Method | Subject_ID, data = comp)
cat(sprintf("  Friedman: chi2(%d) = %.1f, p = %s   [thesis: chi2(4)=333.1, p<.001]\n\n",
            ftcv$parameter, ftcv$statistic, format.pval(ftcv$p.value, digits = 3)))

## ---- 3.2.4 FC mean similarity ------------------------------------------------
cat("=== 3.2.4 FC mean similarity ===\n")
fc <- do.call(rbind, lapply(methods, function(m) {
  M <- as.matrix(read.csv(file.path(fdir, paste0("fc_similarity_", m, ".csv")),
                          row.names = 1, check.names = FALSE))
  diag(M) <- NA
  data.frame(subject_id = rownames(M), fcsim = rowMeans(M, na.rm = TRUE), Method = m)
}))
fc <- fc[fc$subject_id %in% common, ]
fc$Method     <- factor(fc$Method, levels = methods)
fc$subject_id <- factor(fc$subject_id)
print(round(tapply(fc$fcsim, fc$Method, mean), 3))
rm_anova(fc, "fcsim", "subject_id", "RM-ANOVA", "F=11591.4, eta2G=0.95")
cat("\n----------------------------------------------------------------------\n")
cat("Done. Each printed value should match the corresponding number in Results.\n")