# ============================================================
# STAGE 2B: FIXED-LOADING SEM WITH BRAIN FEATURES
# ============================================================
# Asks which properties of a parcellation predict general ability.
#
# The factor loadings are fixed to the values estimated in Stage 2A,
# from the cognitive tests alone. Only the brain-to-g regression paths
# are free. Estimating both together would let the brain features shape
# what g is, which would make any association between them partly
# circular.
#
# Nine predictor sets are fitted per method, crossing three graph
# variants with three choices of parcel-size predictor. In the two
# binary graph variants the network fragments, so the average shortest
# path is undefined; the number of components stands in its place,
# measuring the same thing the truncated path length would have
# measured, under its own name. Both are reversed so that a higher
# value always means a better-connected network.
#
# Standard errors are clustered by family throughout.
#
# Outputs, to Outputs/<variant>/Stage2/:
#   sem_fit_indices.csv   fit of every model
#   sem_paths.csv         standardised regression paths
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

LABELS <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C", "Gordon")

# Three graph variants. In the raw weighted graph the network stays
# connected, so the average shortest path carries information that
# global efficiency does not. In the two binary variants the network
# fragments, and global efficiency already reflects that: unreachable
# pairs contribute nothing to it. Adding a separate fragmentation term
# there would be redundant, and it correlates with global efficiency at
# around -.9, which flips the signs of both.
GRAPHS <- list(
  raw   = c("global_efficiency_raw"),
  top10 = c("global_efficiency_top10"),
  r05   = c("global_efficiency_r05")
)

# three choices for the parcel-size term
SIZE_TERMS <- list(
  cv      = "parcel_size_cv",
  meansim = "parcel_size_similarity",
  none    = character(0)
)

cat("Stage 2B: fixed-loading SEM\n")
cat("Variant:", VARIANT, "\n\n")

# ------------------------------------------------------------
# Data
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

cat("Subjects:", nrow(cog), " Families:", length(unique(cog$family_id)), "\n\n")

# ------------------------------------------------------------
# Model syntax with loadings fixed to the Stage 2A values
# ------------------------------------------------------------

fixed_factor <- function(factor_name) {
  rows <- loadings[loadings$lhs == factor_name, ]
  terms <- paste0(round(rows$est, 8), "*", rows$rhs)
  paste("  ", factor_name, "=~", paste(terms, collapse = " + "))
}

build_syntax <- function(predictors) {
  paste(
    fixed_factor("g"),
    fixed_factor("spd"),
    fixed_factor("cry"),
    fixed_factor("mem"),
    "  g   ~~ 0*spd",
    "  g   ~~ 0*cry",
    "  g   ~~ 0*mem",
    "  spd ~~ 0*cry",
    "  spd ~~ 0*mem",
    "  cry ~~ 0*mem",
    "  spd ~~ 1*spd",
    "  cry ~~ 1*cry",
    "  mem ~~ 1*mem",
    paste("  g ~", paste(predictors, collapse = " + ")),
    sep = "\n"
  )
}

# ------------------------------------------------------------
# Fitting
# ------------------------------------------------------------

fit_indices <- list()
paths <- list()

for (i in seq_along(METHODS)) {
  method <- METHODS[i]
  label  <- LABELS[i]

  feat_file <- file.path(FEAT, method, "features_all_subjects.csv")
  if (!file.exists(feat_file)) {
    cat("  ", label, ": features missing, skipped\n")
    next
  }

  brain <- read.csv(feat_file, colClasses = c(subject = "character"))
  names(brain)[names(brain) == "subject"] <- "subject_id"

  dat <- merge(cog, brain, by = "subject_id")
  cat(strrep("-", 46), "\n")
  cat(label, ":", nrow(dat), "subjects\n")

  for (gname in names(GRAPHS)) {
    for (sname in names(SIZE_TERMS)) {

      predictors <- c(GRAPHS[[gname]], SIZE_TERMS[[sname]], "fc_similarity")
      model_id <- paste(gname, sname, sep = "_")

      missing <- setdiff(predictors, names(dat))
      if (length(missing)) {
        cat("    ", model_id, ": missing", paste(missing, collapse = ", "), "\n")
        next
      }

      d <- dat
      d[predictors] <- lapply(d[predictors], function(x) as.numeric(scale(x)))
      d <- d[complete.cases(d[, c(COG_COLS, predictors)]), ]

      fit <- try(sem(build_syntax(predictors), data = d,
                     estimator = "MLR", cluster = "family_id"),
                 silent = TRUE)

      if (inherits(fit, "try-error") || !lavInspect(fit, "converged")) {
        cat("    ", model_id, ": did not converge\n")
        next
      }

      fi <- fitMeasures(fit, c("cfi.robust", "tli.robust", "rmsea.robust",
                               "srmr", "aic", "bic"))
      fit_indices[[length(fit_indices) + 1]] <- data.frame(
        variant = VARIANT, method = label, model = model_id,
        n = nrow(d),
        CFI = round(fi["cfi.robust"], 3),
        TLI = round(fi["tli.robust"], 3),
        RMSEA = round(fi["rmsea.robust"], 3),
        SRMR = round(fi["srmr"], 3),
        AIC = round(fi["aic"], 1),
        BIC = round(fi["bic"], 1),
        row.names = NULL)

      p <- standardizedSolution(fit)
      p <- p[p$op == "~", ]
      paths[[length(paths) + 1]] <- data.frame(
        variant = VARIANT, method = label, model = model_id,
        predictor = p$rhs,
        beta = round(p$est.std, 4),
        se = round(p$se, 4),
        z = round(p$z, 3),
        p = round(p$pvalue, 5),
        row.names = NULL)

      cat("    ", model_id, ": CFI", round(fi["cfi.robust"], 3), "\n")
    }
  }
}

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

fit_df  <- do.call(rbind, fit_indices)
path_df <- do.call(rbind, paths)

write.csv(fit_df, file.path(OUT_DIR, "sem_fit_indices.csv"), row.names = FALSE)

# Many models are fitted per variant, so the raw p values are corrected
# across all regression paths within a variant. Without this, a handful
# of paths would be expected to reach .05 by chance alone.
path_df$p_fdr <- p.adjust(path_df$p, method = "fdr")
write.csv(path_df, file.path(OUT_DIR, "sem_paths.csv"), row.names = FALSE)

cat("\n")
cat(strrep("=", 60), "\n")
cat("Models fitted:", nrow(fit_df), "\n")
cat("Regression paths:", nrow(path_df), "\n\n")

sig <- path_df[path_df$p_fdr < 0.05, ]
if (nrow(sig)) {
  cat("Paths surviving FDR correction:\n")
  out <- sig[order(-abs(sig$beta)),
             c("method", "model", "predictor", "beta", "p", "p_fdr")]
  print(head(out, 30), row.names = FALSE)
} else {
  cat("No path survives FDR correction.\n")
}

cat("\nSaved to", OUT_DIR, "\n")

