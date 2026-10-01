# ============================================================
# STAGE 2A: PURE BIFACTOR MEASUREMENT MODEL
# ============================================================
# Estimates g from the cognitive tests alone, with no brain data
# involved. Fitting the loadings and the brain-to-g paths together
# would let the brain features shape how g is defined, so the two are
# kept apart: this script fixes what g is, and the next one asks what
# predicts it.
#
# Nine HCP tests, following Dubois et al. (2018), with three group
# factors alongside g. PMAT and VSPLOT load on g alone and so serve as
# pure indicators of general ability. Equal loadings are imposed within
# the crystallised and memory factors, which have two indicators each
# and would otherwise be underidentified.
#
# Standard errors are clustered by family: the sample contains siblings
# and twins, so the subjects are not independent observations.
#
# Outputs, to Outputs/<variant>/Stage2/:
#   g_scores.csv              one g score per subject
#   bifactor_loadings.csv     unstandardised loadings, to be fixed later
#   bifactor_fit.txt          full lavaan summary
# ============================================================

library(lavaan)

# Project root: set IP_BASE, or run this script from the Scripts folder
BASE    <- Sys.getenv("IP_BASE", normalizePath(".."))
VARIANT <- Sys.getenv("IP_VARIANT", "none")
IN_DIR  <- file.path(BASE, "Inputs")
OUT_DIR <- file.path(BASE, "Outputs", VARIANT, "Stage2")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

COG_COLS <- c(
  "CardSort_Unadj", "Flanker_Unadj", "ProcSpeed_Unadj",
  "PicVocab_Unadj", "ReadEng_Unadj", "PMAT24_A_CR",
  "VSPLOT_TC", "IWRD_TOT", "PicSeq_Unadj"
)

cat("Stage 2A: pure bifactor measurement model\n")
cat("Variant:", VARIANT, "\n\n")

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

subjects <- trimws(readLines(file.path(IN_DIR, "subjects_final.txt")))
subjects <- subjects[nchar(subjects) > 0]

hcp <- read.csv(file.path(IN_DIR, "HCP_all_data.csv"))
dat <- hcp[match(as.integer(subjects), hcp$Subject),
           c("Subject", "Family_ID", COG_COLS)]
dat$subject_id <- as.character(dat$Subject)

cat("Subjects           :", nrow(dat), "\n")
cat("Families           :", length(unique(dat$Family_ID)), "\n")
cat("Rows with any NA   :", sum(!complete.cases(dat[, COG_COLS])), "\n")

# listwise deletion; the few subjects with a missing test cannot
# contribute a g score
keep <- complete.cases(dat[, COG_COLS])
dat  <- dat[keep, ]
cat("Subjects retained  :", nrow(dat), "\n\n")

dat[COG_COLS] <- lapply(dat[COG_COLS], function(x) as.numeric(scale(x)))

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

SYNTAX <- '
  g   =~ CardSort_Unadj + Flanker_Unadj + ProcSpeed_Unadj +
         PicVocab_Unadj + ReadEng_Unadj + PMAT24_A_CR +
         VSPLOT_TC + IWRD_TOT + PicSeq_Unadj
  spd =~ CardSort_Unadj + Flanker_Unadj + ProcSpeed_Unadj
  cry =~ a*PicVocab_Unadj + a*ReadEng_Unadj
  mem =~ c*IWRD_TOT       + c*PicSeq_Unadj
  g  ~~ 0*spd
  g  ~~ 0*cry
  g  ~~ 0*mem
  spd ~~ 0*cry
  spd ~~ 0*mem
  cry ~~ 0*mem
'

fit <- sem(
  model     = SYNTAX,
  data      = dat,
  std.lv    = TRUE,
  estimator = "MLR",
  cluster   = "Family_ID"
)

stopifnot(lavInspect(fit, "converged"))
cat("Model converged.\n\n")

# ------------------------------------------------------------
# Fit
# ------------------------------------------------------------

fi <- fitMeasures(fit, c("cfi.robust", "tli.robust", "rmsea.robust",
                         "rmsea.ci.lower.robust", "rmsea.ci.upper.robust",
                         "srmr", "aic", "bic"))

cat("Fit indices\n")
cat(sprintf("  CFI   = %.3f\n", fi["cfi.robust"]))
cat(sprintf("  TLI   = %.3f\n", fi["tli.robust"]))
cat(sprintf("  RMSEA = %.3f [%.3f, %.3f]\n",
            fi["rmsea.robust"], fi["rmsea.ci.lower.robust"],
            fi["rmsea.ci.upper.robust"]))
cat(sprintf("  SRMR  = %.3f\n", fi["srmr"]))
cat(sprintf("  AIC   = %.1f\n", fi["aic"]))
cat(sprintf("  BIC   = %.1f\n\n", fi["bic"]))

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

g_scores <- data.frame(
  subject_id = dat$subject_id,
  family_id  = dat$Family_ID,
  g_score    = as.numeric(lavPredict(fit, type = "lv")[, "g"]),
  stringsAsFactors = FALSE
)
write.csv(g_scores, file.path(OUT_DIR, "g_scores.csv"), row.names = FALSE)

cat("g score: mean =", round(mean(g_scores$g_score), 4),
    " sd =", round(sd(g_scores$g_score), 4),
    " range =", round(min(g_scores$g_score), 3), "to",
    round(max(g_scores$g_score), 3), "\n\n")

loadings <- parameterEstimates(fit)
loadings <- loadings[loadings$op == "=~",
                     c("lhs", "rhs", "est", "se", "z", "pvalue")]
write.csv(loadings, file.path(OUT_DIR, "bifactor_loadings.csv"),
          row.names = FALSE)

cat("Standardised loadings\n")
std <- standardizedSolution(fit)
std <- std[std$op == "=~", c("lhs", "rhs", "est.std", "se", "pvalue")]
std[, c("est.std", "se")] <- round(std[, c("est.std", "se")], 3)
std$pvalue <- round(std$pvalue, 4)
print(std, row.names = FALSE)

sink(file.path(OUT_DIR, "bifactor_fit.txt"))
summary(fit, fit.measures = TRUE, standardized = TRUE)
sink()

cat("\nSaved to", OUT_DIR, "\n")

