# ============================================================
# STAGE 1: INFERENTIAL STATISTICS
# ============================================================
# Tests whether the parcellation methods differ on the validation
# metrics, and by how much.
#
# Every subject is measured under every method, so the comparisons are
# repeated-measures. Two tests are used:
#
#   RM-ANOVA    where all methods vary. Greenhouse-Geisser corrected,
#               with generalised eta squared as the effect size.
#   Friedman    where a method has zero variance, as Schaefer does for
#               contiguity and inter-subject Dice, which breaks the
#               assumptions an ANOVA rests on.
#
# Pairwise differences follow, as paired t tests with Holm correction
# and Cohen's d_av.
#
# At this sample size almost any difference reaches significance, so
# the effect sizes carry the interpretation, not the p values.
#
# Outputs, to Outputs/<variant>/Stage1_Stats/:
#   omnibus_tests.csv    one row per metric
#   pairwise_tests.csv   one row per method pair per metric
# ============================================================

.libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(afex)
  library(dplyr)
  library(tidyr)
})

# Project root: set IP_BASE, or run this script from the Scripts folder
BASE    <- Sys.getenv("IP_BASE", normalizePath(".."))
VARIANT <- Sys.getenv("IP_VARIANT", "none")
OUT_ROOT <- file.path(BASE, "Outputs", VARIANT)
OUT_DIR <- file.path(OUT_ROOT, "Stage1_Stats")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Gordon enters with the 200-parcel label set, so that every method is
# compared at the same resolution.
METHODS <- c(
  Schaefer = "Method_0_Schaefer/validation_all_subjects.csv",
  gMSHBM   = "Method_1_gMSHBM/validation_all_subjects.csv",
  AGP      = "Method_2_AGP/validation_all_subjects.csv",
  SLIC_F   = "Method_3_SLIC_F/validation_all_subjects.csv",
  SLIC_C   = "Method_4_SLIC_C/validation_all_subjects.csv",
  Gordon   = "Method_5_Gordon/validation_all_subjects_200.csv"
)

METRICS <- c("homogeneity_ALL", "fc_test_retest",
             "pct_contiguous", "intersubject_dice")

cat("Stage 1 inferential statistics\n")
cat("Variant:", VARIANT, "\n\n")

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

load_method <- function(label, rel_path) {
  df <- read.csv(file.path(OUT_ROOT, rel_path),
                 colClasses = c(subject = "character"))
  df$method <- label

  # Schaefer is the same atlas in every subject, so its inter-subject
  # Dice is 1 by construction and is never computed; it is filled in
  # here so the method can enter the comparison.
  if (!"intersubject_dice" %in% names(df)) df$intersubject_dice <- 1.0

  keep <- c("subject", "method", intersect(METRICS, names(df)))
  df[, keep]
}

long <- bind_rows(lapply(names(METHODS),
                         function(m) load_method(m, METHODS[[m]])))
long$method <- factor(long$method, levels = names(METHODS))

# only subjects present under every method
complete_subjects <- long %>%
  count(subject) %>%
  filter(n == length(METHODS)) %>%
  pull(subject)

long <- long[long$subject %in% complete_subjects, ]
long$subject <- factor(long$subject)

cat("Subjects in all methods:", length(complete_subjects), "\n")
cat("Methods:", paste(names(METHODS), collapse = ", "), "\n\n")

# ------------------------------------------------------------
# Omnibus tests
# ------------------------------------------------------------

omnibus <- list()

for (metric in METRICS) {
  if (!metric %in% names(long)) next

  wide <- long %>%
    select(subject, method, all_of(metric)) %>%
    pivot_wider(names_from = method, values_from = all_of(metric))

  variances <- sapply(wide[, -1], var, na.rm = TRUE)
  degenerate <- names(variances)[variances < 1e-12]

  cat(strrep("-", 58), "\n")
  cat(metric, "\n")

  if (length(degenerate)) {
    # a constant column makes the sphericity assumption meaningless,
    # so a rank-based test is used instead
    mat <- as.matrix(wide[, -1])
    ft <- friedman.test(mat)

    # Kendall's W, the effect size that goes with Friedman
    k <- ncol(mat); n <- nrow(mat)
    W <- as.numeric(ft$statistic) / (n * (k - 1))

    cat("  Friedman (zero variance in:", paste(degenerate, collapse = ", "), ")\n")
    cat(sprintf("    chi2(%d) = %.1f, p = %s, Kendall W = %.3f\n",
                ft$parameter, ft$statistic,
                format.pval(ft$p.value, digits = 3), W))

    omnibus[[length(omnibus) + 1]] <- data.frame(
      variant = VARIANT, metric = metric, test = "Friedman",
      statistic = round(as.numeric(ft$statistic), 2),
      df = as.numeric(ft$parameter),
      p = ft$p.value,
      effect_size = round(W, 4),
      effect_size_name = "Kendall W",
      note = paste("zero variance:", paste(degenerate, collapse = "/")),
      row.names = NULL)

  } else {
    a <- suppressMessages(
      aov_ez(id = "subject", dv = metric, data = long, within = "method"))
    at <- a$anova_table

    cat("  RM-ANOVA (Greenhouse-Geisser)\n")
    cat(sprintf("    F(%.1f, %.1f) = %.1f, p = %s, eta2_G = %.3f\n",
                at$`num Df`[1], at$`den Df`[1], at$F[1],
                format.pval(at$`Pr(>F)`[1], digits = 3), at$ges[1]))

    omnibus[[length(omnibus) + 1]] <- data.frame(
      variant = VARIANT, metric = metric, test = "RM-ANOVA",
      statistic = round(at$F[1], 2),
      df = round(at$`num Df`[1], 2),
      p = at$`Pr(>F)`[1],
      effect_size = round(at$ges[1], 4),
      effect_size_name = "generalised eta squared",
      note = "",
      row.names = NULL)
  }
}

write.csv(do.call(rbind, omnibus),
          file.path(OUT_DIR, "omnibus_tests.csv"), row.names = FALSE)

# ------------------------------------------------------------
# Pairwise comparisons
# ------------------------------------------------------------

cohens_d_av <- function(a, b) {
  (mean(a) - mean(b)) / ((sd(a) + sd(b)) / 2)
}

pairwise <- list()

for (metric in METRICS) {
  if (!metric %in% names(long)) next

  wide <- long %>%
    select(subject, method, all_of(metric)) %>%
    pivot_wider(names_from = method, values_from = all_of(metric))

  pairs <- combn(names(METHODS), 2, simplify = FALSE)
  rows <- lapply(pairs, function(p) {
    a <- wide[[p[1]]]
    b <- wide[[p[2]]]

    # a pair where both sides are constant has nothing to test
    if (sd(a) < 1e-12 && sd(b) < 1e-12) {
      return(data.frame(metric = metric, method_1 = p[1], method_2 = p[2],
                        mean_1 = mean(a), mean_2 = mean(b),
                        d_av = NA, p = NA, row.names = NULL))
    }

    tt <- t.test(a, b, paired = TRUE)
    data.frame(metric = metric, method_1 = p[1], method_2 = p[2],
               mean_1 = round(mean(a), 4), mean_2 = round(mean(b), 4),
               d_av = round(cohens_d_av(a, b), 3),
               p = tt$p.value, row.names = NULL)
  })

  df <- do.call(rbind, rows)
  df$p_holm <- p.adjust(df$p, method = "holm")
  pairwise[[length(pairwise) + 1]] <- df
}

pw <- do.call(rbind, pairwise)
pw$variant <- VARIANT
pw <- pw[, c("variant", setdiff(names(pw), "variant"))]
write.csv(pw, file.path(OUT_DIR, "pairwise_tests.csv"), row.names = FALSE)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

cat("\n")
cat(strrep("=", 58), "\n")
cat("Largest pairwise effects per metric\n\n")

for (metric in unique(pw$metric)) {
  sub <- pw[pw$metric == metric & !is.na(pw$d_av), ]
  sub <- sub[order(-abs(sub$d_av)), ]
  cat(metric, "\n")
  top <- head(sub, 3)
  for (i in seq_len(nrow(top))) {
    cat(sprintf("  %-10s vs %-10s  d = %6.2f  p_holm = %s\n",
                top$method_1[i], top$method_2[i], top$d_av[i],
                format.pval(top$p_holm[i], digits = 2)))
  }
  cat("\n")
}

cat("Saved to", OUT_DIR, "\n")

