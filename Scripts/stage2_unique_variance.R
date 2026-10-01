# ============================================================
# STAGE 2C: UNIQUE VARIANCE DECOMPOSITION
# ============================================================
# Asks how much each brain feature contributes on its own.
#
# A standardised regression path says how much g changes per unit of a
# predictor, holding the others constant. When predictors are correlated
# that can be misleading: a feature may carry real information and still
# show a small path, because a correlated feature absorbed it.
#
# This script asks a different question. It fits the full model, then
# refits it with one predictor removed, and takes the drop in explained
# variance. What is left is the part of g that only that feature could
# account for.
#
#   unique_i = R2(full) - R2(without predictor i)
#
# Loadings stay fixed to the Stage 2A values throughout, as in the main
# SEM, so the measurement model never moves between fits.
#
# Only the raw graph variant is decomposed. The two binary variants
# carry the same predictors with a different threshold, so decomposing
# all three would triple the output without adding a question.
#
# Output: Outputs/<variant>/Stage2/sem_unique_variance.csv
# ============================================================

library(lavaan)

# Project root: set IP_BASE, or run this script from the Scripts folder
BASE    <- Sys.getenv("IP_BASE", normalizePath(".."))
VARIANT <- Sys.getenv("IP_VARIANT", "none")
OUT_DIR <- file.path(BASE, "Outputs", VARIANT, "Stage2")
FEAT    <- file.path(BASE, "Outputs", VARIANT, "Features")
IN_DIR  <- file.path(BASE, "Inputs")

COG_COLS <- c(
  "CardSort_Unadj", "Flanker_Unadj", "ProcSpeed_Unadj",
  "PicVocab_Unadj", "ReadEng_Unadj", "PMAT24_A_CR",
  "VSPLOT_TC", "IWRD_TOT", "PicSeq_Unadj"
)

METHODS <- c("Method_0_Schaefer", "Method_1_gMSHBM", "Method_2_AGP",
             "Method_3_SLIC_F", "Method_4_SLIC_C", "Method_5_Gordon")
LABELS  <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C", "Gordon")

# The full predictor set. Schaefer has no parcel-size variance, since
# every subject carries the same atlas, so it is decomposed without it.
FULL_SET <- c("global_efficiency_raw", "parcel_size_cv", "fc_similarity")
SCHAEFER_SET <- c("global_efficiency_raw", "fc_similarity")

cat("Stage 2C: unique variance decomposition\n")
cat("Variant:", VARIANT, "\n\n")

# ------------------------------------------------------------
# Data and model syntax
# ------------------------------------------------------------

g_scores <- read.csv(file.path(OUT_DIR, "g_scores.csv"),
                     colClasses = c(subject_id = "character"))
loadings <- read.csv(file.path(OUT_DIR, "bifactor_loadings.csv"))

hcp <- read.csv(file.path(IN_DIR, "HCP_all_data.csv"))
cog <- hcp[match(as.integer(g_scores$subject_id), hcp$Subject),
           c("Subject", COG_COLS)]
cog$subject_id <- as.character(cog$Subject)
cog[COG_COLS] <- lapply(cog[COG_COLS], function(x) as.numeric(scale(x)))
cog$family_id <- g_scores$family_id

fixed_factor <- function(factor_name) {
  rows <- loadings[loadings$lhs == factor_name, ]
  terms <- paste0(round(rows$est, 8), "*", rows$rhs)
  paste("  ", factor_name, "=~", paste(terms, collapse = " + "))
}

build_syntax <- function(predictors) {
  paste(
    fixed_factor("g"), fixed_factor("spd"),
    fixed_factor("cry"), fixed_factor("mem"),
    "  g   ~~ 0*spd", "  g   ~~ 0*cry", "  g   ~~ 0*mem",
    "  spd ~~ 0*cry", "  spd ~~ 0*mem", "  cry ~~ 0*mem",
    "  spd ~~ 1*spd", "  cry ~~ 1*cry", "  mem ~~ 1*mem",
    paste("  g ~", paste(predictors, collapse = " + ")),
    sep = "\n")
}

# Proportion of g explained by the predictors. lavaan reports the
# residual variance of g; the rest of its total variance is what the
# brain features account for.
r_squared <- function(fit) {
  as.numeric(inspect(fit, "r2")["g"])
}

fit_model <- function(dat, predictors) {
  d <- dat
  d[predictors] <- lapply(d[predictors], function(x) as.numeric(scale(x)))
  d <- d[complete.cases(d[, c(COG_COLS, predictors)]), ]

  fit <- try(sem(build_syntax(predictors), data = d,
                 estimator = "MLR", cluster = "family_id"), silent = TRUE)

  if (inherits(fit, "try-error") || !lavInspect(fit, "converged")) return(NA)
  r_squared(fit)
}

# ------------------------------------------------------------
# Decomposition
# ------------------------------------------------------------

rows <- list()

for (i in seq_along(METHODS)) {
  method <- METHODS[i]
  label  <- LABELS[i]

  feat_file <- file.path(FEAT, method, "features_all_subjects.csv")
  if (!file.exists(feat_file)) next

  brain <- read.csv(feat_file, colClasses = c(subject = "character"))
  names(brain)[names(brain) == "subject"] <- "subject_id"
  dat <- merge(cog, brain, by = "subject_id")

  predictors <- if (label == "Schaefer") SCHAEFER_SET else FULL_SET
  predictors <- intersect(predictors, names(dat))

  r2_full <- fit_model(dat, predictors)
  if (is.na(r2_full)) {
    cat(label, ": full model did not converge\n")
    next
  }

  cat(sprintf("%-10s R2 full = %.4f\n", label, r2_full))

  for (p in predictors) {
    reduced <- setdiff(predictors, p)
    r2_reduced <- fit_model(dat, reduced)

    unique_var <- if (is.na(r2_reduced)) NA else r2_full - r2_reduced

    rows[[length(rows) + 1]] <- data.frame(
      variant = VARIANT, method = label, predictor = p,
      r2_full = round(r2_full, 5),
      r2_reduced = if (is.na(r2_reduced)) NA else round(r2_reduced, 5),
      unique_variance = if (is.na(unique_var)) NA else round(unique_var, 5),
      row.names = NULL)

    if (!is.na(unique_var)) {
      cat(sprintf("             %-24s unique = %.4f\n", p, unique_var))
    }
  }
  cat("\n")
}

out <- do.call(rbind, rows)
write.csv(out, file.path(OUT_DIR, "sem_unique_variance.csv"), row.names = FALSE)
cat("Saved to", file.path(OUT_DIR, "sem_unique_variance.csv"), "\n")

