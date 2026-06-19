# ============================================================
# Stage 2 Unleaked - Script 2: Pure Bifactor SEM +
#                               Fixed-Loading SEM with Brain Features
# ============================================================
# DATA LEAKAGE FIX
# ----------------
# The original pipeline estimated g loadings and brain → g
# regression paths jointly, letting brain features influence
# how g is defined.  This script eliminates that leakage:
#
#   STAGE A — Pure measurement model (cognitive tests only).
#     • Fits the bifactor model with NO brain features.
#     • Extracts ONE g score per subject (brain-free).
#     • Extracts the raw (unstandardized) factor loadings.
#
#   STAGE B — Fixed-loading SEM (per method × model variant).
#     • Fixes ALL factor loadings to the Stage A values.
#     • Only the brain → g regression paths are freely estimated.
#     • Brain features cannot back-door influence g's definition.
#
# The single g score from Stage A is used in the KRR (Script 3).
# The fixed-loading SEMs from Stage B provide the brain → g
# regression path estimates.
#
# Outputs saved to Stage_2_Unleaked/SEM/:
#   pure_model_summary.txt       — full lavaan summary
#   pure_model_loadings.csv      — unstandardized loadings (Stage A)
#   g_scores.csv                 — single g score per subject
#   sem_fit_indices.csv          — fit indices (Stage B, 45 models)
#   sem_parameters.csv           — regression path estimates
#   sem_fixed_loadings_used.csv  — the fixed loading values
# ============================================================

library(lavaan)
library(tidyverse)

BASE    <- "C:/Thesis_Main/Analysis"
IN_DATA <- file.path(BASE, "Outputs/Stage_2_Unleaked/Input_Data")
OUT_DIR <- file.path(BASE, "Outputs/Stage_2_Unleaked/SEM")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

METHOD_ORDER <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C")

COG_COLS <- c(
  "CardSort_Unadj", "Flanker_Unadj", "ProcSpeed_Unadj",
  "PicVocab_Unadj", "ReadEng_Unadj", "PMAT24_A_CR",
  "VSPLOT_TC", "IWRD_TOT", "PicSeq_Unadj"
)

cat("============================================================\n")
cat("Stage 2 Unleaked - Script 2: Pure + Fixed-Loading SEM\n")
cat("============================================================\n\n")
cat("Loading data from Script 1 outputs...\n")

cog_data <- read.csv(file.path(IN_DATA, "cog_data.csv"),
                     colClasses = c(subject_id = "character"))

read_features <- function(method) {
  read.csv(file.path(IN_DATA, paste0("brain_features_", method, ".csv")),
           colClasses = c(subject_id = "character"))
}

df_Schaefer <- read_features("Schaefer")
df_gMSHBM   <- read_features("gMSHBM")
df_AGP      <- read_features("AGP")
df_SLIC_F   <- read_features("SLIC_F")
df_SLIC_C   <- read_features("SLIC_C")

cat("  cog_data     :", nrow(cog_data), "subjects x", ncol(cog_data) - 1, "tests\n\n")

# ============================================================
# STAGE A: PURE BIFACTOR MEASUREMENT MODEL (cognitive only)
# ============================================================
# No brain features, no regression paths.
# This defines g purely from cognitive tests.
# ============================================================

cat("============================================================\n")
cat("STAGE A: Pure Bifactor Measurement Model\n")
cat("============================================================\n\n")

# Z-score cognitive data once here. cog_scaled is used in both Stage A and prepare_data().
cog_scaled <- cog_data
cog_scaled[COG_COLS] <- lapply(cog_scaled[COG_COLS],
                               function(x) as.numeric(scale(x)))

SYNTAX_PURE <- '
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

fit_pure <- sem(
  model     = SYNTAX_PURE,
  data      = cog_scaled[, COG_COLS, drop = FALSE],
  std.lv    = TRUE,
  estimator = "MLR"
)

stopifnot(lavInspect(fit_pure, "converged"))

cat("Pure measurement model converged.\n\n")

# --- Extract g scores ---
g_scores <- data.frame(
  subject_id = cog_scaled$subject_id,
  g_score    = as.numeric(lavPredict(fit_pure, type = "lv")[, "g"]),
  stringsAsFactors = FALSE
)

cat("g score summary:\n")
cat("  mean =", round(mean(g_scores$g_score), 4), "\n")
cat("  sd   =", round(sd(g_scores$g_score),   4), "\n")
cat("  min  =", round(min(g_scores$g_score),   4), "\n")
cat("  max  =", round(max(g_scores$g_score),   4), "\n\n")

# --- Extract raw (unstandardized) loadings ---
raw_params   <- parameterEstimates(fit_pure)
raw_loadings <- raw_params %>% filter(op == "=~")

cat("Raw (unstandardized) factor loadings from pure model:\n")
raw_loadings %>%
  select(lhs, op, rhs, est, se, z, pvalue) %>%
  mutate(across(c(est, se, z), ~ round(.x, 4)),
         pvalue = round(pvalue, 5)) %>%
  as.data.frame() %>%
  print(row.names = FALSE)
cat("\n")

# Also extract standardized loadings for reference
std_loadings <- standardizedSolution(fit_pure) %>%
  filter(op == "=~") %>%
  select(lhs, op, rhs, est.std, se, z, pvalue) %>%
  mutate(across(c(est.std, se, z), ~ round(.x, 4)),
         pvalue = round(pvalue, 5))

cat("Standardized factor loadings (for reference):\n")
as.data.frame(std_loadings) %>% print(row.names = FALSE)
cat("\n")

# --- Save Stage A outputs ---
write.csv(g_scores, file.path(OUT_DIR, "g_scores.csv"), row.names = FALSE)

write.csv(raw_loadings %>% select(lhs, rhs, est, se, z, pvalue),
          file.path(OUT_DIR, "pure_model_loadings.csv"),
          row.names = FALSE)

sink(file.path(OUT_DIR, "pure_model_summary.txt"))
summary(fit_pure, fit.measures = TRUE, standardized = TRUE)
sink()

# Extract fit indices for the pure model
fi_pure <- fitMeasures(fit_pure, c(
  "cfi.robust", "tli.robust",
  "rmsea.robust", "rmsea.ci.lower.robust", "rmsea.ci.upper.robust",
  "srmr", "aic", "bic"))

cat("Pure model fit indices:\n")
cat(sprintf("  CFI   = %.3f\n", fi_pure["cfi.robust"]))
cat(sprintf("  TLI   = %.3f\n", fi_pure["tli.robust"]))
cat(sprintf("  RMSEA = %.3f [%.3f, %.3f]\n",
            fi_pure["rmsea.robust"],
            fi_pure["rmsea.ci.lower.robust"],
            fi_pure["rmsea.ci.upper.robust"]))
cat(sprintf("  SRMR  = %.3f\n", fi_pure["srmr"]))
cat(sprintf("  AIC   = %.1f\n", fi_pure["aic"]))
cat(sprintf("  BIC   = %.1f\n\n", fi_pure["bic"]))

cat("Stage A complete. g scores and loadings saved.\n\n")

# ============================================================
# STAGE B: FIXED-LOADING SEM WITH BRAIN FEATURES
# ============================================================
# All factor loadings are fixed to Stage A values.
# Only brain → g regression paths are freely estimated.
# Latent variances for spd, cry, mem fixed to 1 (matching
# the std.lv = TRUE parameterisation of the pure model).
# g's residual variance is freely estimated.
# ============================================================

cat("============================================================\n")
cat("STAGE B: Fixed-Loading SEM with Brain Features\n")
cat("============================================================\n\n")

# --- Build the loading lookup ---
loading_lookup <- raw_loadings %>%
  select(factor = lhs, indicator = rhs, loading = est)

cat("Loading values to be fixed:\n")
as.data.frame(loading_lookup) %>%
  mutate(loading = round(loading, 6)) %>%
  print(row.names = FALSE)
cat("\n")

# Save the fixed loadings for documentation
write.csv(loading_lookup,
          file.path(OUT_DIR, "sem_fixed_loadings_used.csv"),
          row.names = FALSE)

# --- Syntax builder ---
build_fixed_syntax <- function(loading_lookup, brain_predictors) {
  # g =~ line with fixed loadings
  g_loads <- loading_lookup %>% filter(factor == "g")
  g_terms <- sprintf("%.6f*%s", g_loads$loading, g_loads$indicator)
  g_line  <- paste("  g =~", paste(g_terms, collapse = " + "))
  
  # spd =~ line
  spd_loads <- loading_lookup %>% filter(factor == "spd")
  spd_terms <- sprintf("%.6f*%s", spd_loads$loading, spd_loads$indicator)
  spd_line  <- paste("  spd =~", paste(spd_terms, collapse = " + "))
  
  # cry =~ line
  cry_loads <- loading_lookup %>% filter(factor == "cry")
  cry_terms <- sprintf("%.6f*%s", cry_loads$loading, cry_loads$indicator)
  cry_line  <- paste("  cry =~", paste(cry_terms, collapse = " + "))
  
  # mem =~ line
  mem_loads <- loading_lookup %>% filter(factor == "mem")
  mem_terms <- sprintf("%.6f*%s", mem_loads$loading, mem_loads$indicator)
  mem_line  <- paste("  mem =~", paste(mem_terms, collapse = " + "))
  
  # Orthogonality constraints + latent variance constraints
  constraints <- "
  g   ~~ 0*spd
  g   ~~ 0*cry
  g   ~~ 0*mem
  spd ~~ 0*cry
  spd ~~ 0*mem
  cry ~~ 0*mem
  spd ~~ 1*spd
  cry ~~ 1*cry
  mem ~~ 1*mem"
  
  # Regression: brain → g
  reg_line <- paste("  g ~", paste(brain_predictors, collapse = " + "))
  
  paste(g_line, spd_line, cry_line, mem_line, constraints, reg_line, sep = "\n")
}

# --- Define the 9 SEM variants (same brain predictor combos as original) ---
BRAIN_VARS <- list(
  SEM1 = c("GE_raw",   "ASPL_raw_r",   "Parcel_Size_CV",      "FC_MeanSim"),
  SEM2 = c("GE_top10", "ASPL_top10_r", "Parcel_Size_CV",      "FC_MeanSim"),
  SEM3 = c("GE_r05",   "ASPL_r05_r",   "Parcel_Size_CV",      "FC_MeanSim"),
  SEM4 = c("GE_raw",   "ASPL_raw_r",   "Parcel_Size_MeanSim", "FC_MeanSim"),
  SEM5 = c("GE_top10", "ASPL_top10_r", "Parcel_Size_MeanSim", "FC_MeanSim"),
  SEM6 = c("GE_r05",   "ASPL_r05_r",   "Parcel_Size_MeanSim", "FC_MeanSim"),
  SEM7 = c("GE_raw",   "ASPL_raw_r",   "FC_MeanSim"),
  SEM8 = c("GE_top10", "ASPL_top10_r", "FC_MeanSim"),
  SEM9 = c("GE_r05",   "ASPL_r05_r",   "FC_MeanSim")
)

# Pre-build syntax for each variant
ALL_SYNTAX <- lapply(BRAIN_VARS, function(bv) {
  build_fixed_syntax(loading_lookup, bv)
})

# Print one example syntax for verification
cat("Example syntax (SEM1):\n")
cat(ALL_SYNTAX[["SEM1"]], "\n\n")

# --- Helper: prepare data (merge brain features with pre-scaled cog data + z-score brain vars) ---
prepare_data <- function(df_brain, brain_vars) {
  df <- df_brain %>% inner_join(cog_scaled, by = "subject_id")
  cols_to_scale <- brain_vars
  df[cols_to_scale] <- lapply(df[cols_to_scale],
                              function(x) as.numeric(scale(x)))
  df
}

# --- Helper: extract fit indices ---
extract_fit <- function(fit, method, model, status, note = "") {
  fi <- fitMeasures(fit, c(
    "cfi.robust", "tli.robust",
    "rmsea.robust", "rmsea.ci.lower.robust", "rmsea.ci.upper.robust",
    "srmr", "aic", "bic"))
  data.frame(
    method        = method, model = model,
    status        = status, note  = note,
    CFI           = round(fi["cfi.robust"],            3),
    TLI           = round(fi["tli.robust"],            3),
    RMSEA         = round(fi["rmsea.robust"],          3),
    RMSEA_CI_low  = round(fi["rmsea.ci.lower.robust"], 3),
    RMSEA_CI_high = round(fi["rmsea.ci.upper.robust"], 3),
    SRMR          = round(fi["srmr"],                  3),
    AIC           = round(fi["aic"],                   1),
    BIC           = round(fi["bic"],                   1),
    stringsAsFactors = FALSE, row.names = NULL)
}

# --- Helper: extract regression parameters only ---
# (Loadings are fixed — no inference to report for them)
extract_params <- function(fit, method, model, status) {
  standardizedSolution(fit, type = "std.all") %>%
    filter(op == "~") %>%
    mutate(
      method     = method, model = model, status = status,
      param_type = "regression",
      std_est    = round(est.std, 3),
      std_error  = round(se,      3),
      z_value    = round(z,       3),
      p_value    = round(pvalue,  4)) %>%
    select(method, model, status, param_type,
           lhs, op, rhs, std_est, std_error, z_value, p_value)
}

# ============================================================
# FIT ALL 45 FIXED-LOADING MODELS
# ============================================================

cat("Fitting 45 fixed-loading models (5 methods x 9 variants)...\n\n")

all_fit_indices <- list()
all_parameters  <- list()

for (method in METHOD_ORDER) {
  
  cat(strrep("-", 50), "\n")
  cat("Method:", method, "\n")
  
  df_brain <- switch(method,
                     "Schaefer" = df_Schaefer, "gMSHBM" = df_gMSHBM,
                     "AGP"      = df_AGP,      "SLIC_F" = df_SLIC_F,
                     "SLIC_C"   = df_SLIC_C)
  
  for (sem_name in names(ALL_SYNTAX)) {
    
    brain_vars <- BRAIN_VARS[[sem_name]]
    obj_name   <- paste0("fit_", method, "_", sem_name)
    
    # Check for zero-variance brain features
    zero_var <- brain_vars[sapply(brain_vars,
                                  function(v) sd(df_brain[[v]], na.rm = TRUE) < 1e-10)]
    
    if (length(zero_var) > 0) {
      cat(sprintf("  [SKIP] %s | %s — zero variance in: %s\n",
                  method, sem_name, paste(zero_var, collapse = ", ")))
      assign(obj_name, NULL, envir = .GlobalEnv)
      all_fit_indices[[paste(method, sem_name)]] <- data.frame(
        method = method, model = sem_name,
        status = "skipped_zero_variance",
        note   = paste("Zero variance:", paste(zero_var, collapse = ", ")),
        CFI = NA, TLI = NA, RMSEA = NA,
        RMSEA_CI_low = NA, RMSEA_CI_high = NA,
        SRMR = NA, AIC = NA, BIC = NA,
        stringsAsFactors = FALSE)
      next
    }
    
    df_fit <- prepare_data(df_brain, brain_vars)
    
    fit <- tryCatch(
      sem(model      = ALL_SYNTAX[[sem_name]],
          data       = df_fit,
          estimator  = "MLR",
          check.post = FALSE),
      error = function(e) e)
    
    if (inherits(fit, "error")) {
      cat(sprintf("  [FAIL] %s | %s — %s\n",
                  method, sem_name, conditionMessage(fit)))
      assign(obj_name, NULL, envir = .GlobalEnv)
      all_fit_indices[[paste(method, sem_name)]] <- data.frame(
        method = method, model = sem_name,
        status = "failed", note = conditionMessage(fit),
        CFI = NA, TLI = NA, RMSEA = NA,
        RMSEA_CI_low = NA, RMSEA_CI_high = NA,
        SRMR = NA, AIC = NA, BIC = NA,
        stringsAsFactors = FALSE)
      next
    }
    
    if (!lavInspect(fit, "converged")) {
      cat(sprintf("  [NO CONVERGE] %s | %s\n", method, sem_name))
      assign(obj_name, NULL, envir = .GlobalEnv)
      all_fit_indices[[paste(method, sem_name)]] <- data.frame(
        method = method, model = sem_name,
        status = "not_converged", note = "lavaan did not converge",
        CFI = NA, TLI = NA, RMSEA = NA,
        RMSEA_CI_low = NA, RMSEA_CI_high = NA,
        SRMR = NA, AIC = NA, BIC = NA,
        stringsAsFactors = FALSE)
      next
    }
    
    resid_vars <- diag(lavInspect(fit, "est")$theta)
    heywood    <- any(resid_vars < 0, na.rm = TRUE)
    status     <- ifelse(heywood, "ok_heywood", "ok")
    note       <- ifelse(heywood, "Heywood: negative residual variance", "")
    
    assign(obj_name, fit, envir = .GlobalEnv)
    
    fi_row    <- extract_fit(fit, method, sem_name, status, note)
    param_row <- extract_params(fit, method, sem_name, status)
    
    all_fit_indices[[paste(method, sem_name)]] <- fi_row
    all_parameters[[paste(method, sem_name)]]  <- param_row
    
    cat(sprintf(
      "  [%s] %s | %s — CFI=%.3f  TLI=%.3f  RMSEA=%.3f  SRMR=%.3f%s\n",
      toupper(status), method, sem_name,
      fi_row$CFI, fi_row$TLI, fi_row$RMSEA, fi_row$SRMR,
      ifelse(heywood, "  *** HEYWOOD", "")))
  }
}

# ============================================================
# COMPILE AND SAVE
# ============================================================

fit_indices_all <- bind_rows(all_fit_indices)
rownames(fit_indices_all) <- NULL

params_all <- bind_rows(all_parameters)
rownames(params_all) <- NULL

write.csv(fit_indices_all,
          file.path(OUT_DIR, "sem_fit_indices.csv"), row.names = FALSE)
write.csv(params_all,
          file.path(OUT_DIR, "sem_parameters.csv"),  row.names = FALSE)

cat("\nSaved to:", OUT_DIR, "\n")
cat("  pure_model_summary.txt\n")
cat("  pure_model_loadings.csv\n")
cat("  sem_fixed_loadings_used.csv\n")
cat("  g_scores.csv\n")
cat("  sem_fit_indices.csv\n")
cat("  sem_parameters.csv\n\n")

# ============================================================
# CONSOLE SUMMARIES
# ============================================================

cat("============================================================\n")
cat("FIT INDEX SUMMARY — all converged fixed-loading models\n")
cat("============================================================\n")
fit_indices_all %>%
  filter(status %in% c("ok", "ok_heywood")) %>%
  select(method, model, status, CFI, TLI, RMSEA, SRMR) %>%
  mutate(across(c(CFI, TLI, RMSEA, SRMR), \(x) round(x, 3))) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

cat("\n============================================================\n")
cat("REGRESSION PATHS — brain features → g (std.all)\n")
cat("(Factor loadings FIXED from pure model — not re-estimated)\n")
cat("============================================================\n")
params_all %>%
  filter(status %in% c("ok", "ok_heywood")) %>%
  mutate(sig = case_when(
    p_value < 0.001 ~ "***",
    p_value < 0.01  ~ "**",
    p_value < 0.05  ~ "*",
    TRUE            ~ "")) %>%
  select(method, model, predictor = rhs,
         std_est, std_error, z_value, p_value, sig) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

cat("\n============================================================\n")
cat("G SCORE SUMMARY — single brain-free g score (from pure model)\n")
cat("============================================================\n")
cat("  N     =", nrow(g_scores), "\n")
cat("  mean  =", round(mean(g_scores$g_score), 4), "\n")
cat("  sd    =", round(sd(g_scores$g_score),   4), "\n")
cat("  min   =", round(min(g_scores$g_score),  4), "\n")
cat("  max   =", round(max(g_scores$g_score),  4), "\n")

cat("\n============================================================\n")
cat("SKIPPED AND FAILED MODELS\n")
cat("============================================================\n")
fit_indices_all %>%
  filter(!status %in% c("ok", "ok_heywood")) %>%
  select(method, model, status, note) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

cat("\n============================================================\n")
cat("NAMED FIT OBJECTS IN MEMORY\n")
cat("============================================================\n")
fit_objects <- sort(ls(envir = .GlobalEnv, pattern = "^fit_"))
fit_objects <- fit_objects[grepl("_SEM[0-9]$", fit_objects)]
cat(paste(fit_objects, collapse = "\n"), "\n")
cat("\nAlso in memory: fit_pure (pure measurement model)\n")
cat("  summary(fit_pure, fit.measures = TRUE, standardized = TRUE)\n")
cat("  summary(fit_gMSHBM_SEM1, fit.measures = TRUE, standardized = TRUE)\n")