# ============================================================
# Stage 2 Unleaked — Script 2c: Proper Unique Variance Decomposition
# ============================================================
# Adapted for the fixed-loading pipeline.
#
# For each method, refits the fixed-loading SEM four times
# (once full, then dropping one brain predictor at a time)
# and computes:
#   R^2(full), R^2(reduced_i), unique_i = R^2(full) - R^2(reduced_i)
#
# All factor loadings are fixed to the pure model values.
# Only structural (regression) paths are freely estimated.
#
# Schaefer uses SEM7 base (GE + ASPL_r + FC_MeanSim) = 3 predictors.
# The four individualised methods use SEM1 base
# (GE + ASPL_r + Parcel_Size_CV + FC_MeanSim) = 4 predictors.
#
# Output: Stage_2_Unleaked/SEM/sem_unique_variance.csv
#         Stage_2_Unleaked/SEM/sem_unique_variance_wide.csv
# ============================================================

library(lavaan)
library(tidyverse)

BASE    <- "C:/Thesis_Main/Analysis"
IN_DATA <- file.path(BASE, "Outputs/Stage_2_Unleaked/Input_Data")
OUT_DIR <- file.path(BASE, "Outputs/Stage_2_Unleaked/SEM")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

COG_COLS <- c(
  "CardSort_Unadj", "Flanker_Unadj", "ProcSpeed_Unadj",
  "PicVocab_Unadj", "ReadEng_Unadj", "PMAT24_A_CR",
  "VSPLOT_TC", "IWRD_TOT", "PicSeq_Unadj"
)

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

# ------------------------------------------------------------
# Load fixed loadings from pure model (Script 2 output)
# ------------------------------------------------------------
loading_lookup <- read.csv(file.path(OUT_DIR, "sem_fixed_loadings_used.csv"),
                           stringsAsFactors = FALSE)

cat("============================================================\n")
cat("Stage 2 Unleaked - Script 2c: Unique Variance Decomposition\n")
cat("============================================================\n\n")
cat("Using fixed loadings from pure model:\n")
as.data.frame(loading_lookup) %>%
  mutate(loading = round(loading, 6)) %>%
  print(row.names = FALSE)
cat("\n")

# ------------------------------------------------------------
# Syntax builder — same as Script 2 but brain predictors vary
# ------------------------------------------------------------
build_fixed_syntax <- function(loading_lookup, predictors) {
  # g =~ line
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
  
  if (length(predictors) == 0) {
    # Null model — no regression paths
    return(paste(g_line, spd_line, cry_line, mem_line, constraints, sep = "\n"))
  }
  
  reg_line <- paste("  g ~", paste(predictors, collapse = " + "))
  paste(g_line, spd_line, cry_line, mem_line, constraints, reg_line, sep = "\n")
}

# Helper: fit and return R^2 for g
fit_and_r2 <- function(syntax, df) {
  df_fit <- inner_join(df, cog_data, by = "subject_id")
  brain_cols <- setdiff(colnames(df), "subject_id")
  cols_to_scale <- c(brain_cols, COG_COLS)
  # Only scale columns that actually exist in the data
  cols_to_scale <- intersect(cols_to_scale, colnames(df_fit))
  df_fit[cols_to_scale] <- lapply(df_fit[cols_to_scale],
                                  function(x) as.numeric(scale(x)))
  
  fit <- tryCatch(
    sem(syntax, data = df_fit, estimator = "MLR", check.post = FALSE),
    error = function(e) NULL)
  if (is.null(fit) || !lavInspect(fit, "converged")) return(NA_real_)
  
  std_all <- standardizedSolution(fit)
  g_resid <- std_all %>%
    filter(op == "~~", lhs == "g", rhs == "g") %>%
    pull(est.std)
  1 - g_resid
}

# ------------------------------------------------------------
# Predictor sets per method
# ------------------------------------------------------------
PREDICTORS <- list(
  Schaefer = c("GE_raw", "ASPL_raw_r", "FC_MeanSim"),
  gMSHBM   = c("GE_raw", "ASPL_raw_r", "Parcel_Size_CV", "FC_MeanSim"),
  AGP      = c("GE_raw", "ASPL_raw_r", "Parcel_Size_CV", "FC_MeanSim"),
  SLIC_F   = c("GE_raw", "ASPL_raw_r", "Parcel_Size_CV", "FC_MeanSim"),
  SLIC_C   = c("GE_raw", "ASPL_raw_r", "Parcel_Size_CV", "FC_MeanSim")
)

DFS <- list(Schaefer = df_Schaefer, gMSHBM = df_gMSHBM,
            AGP = df_AGP, SLIC_F = df_SLIC_F, SLIC_C = df_SLIC_C)

# ------------------------------------------------------------
# Compute unique variance for each predictor in each method
# ------------------------------------------------------------
results <- list()

for (method in names(PREDICTORS)) {
  cat("\n", strrep("-", 50), "\n", sep = "")
  cat("Method:", method, "\n")
  
  preds <- PREDICTORS[[method]]
  df    <- DFS[[method]]
  
  # Full model
  r2_full <- fit_and_r2(build_fixed_syntax(loading_lookup, preds), df)
  cat(sprintf("  R^2 full  = %.4f\n", r2_full))
  
  # Leave-one-out models
  for (p in preds) {
    reduced <- setdiff(preds, p)
    r2_red  <- fit_and_r2(build_fixed_syntax(loading_lookup, reduced), df)
    unique_r2 <- r2_full - r2_red
    cat(sprintf("  Drop %-18s R^2 = %.4f  unique = %+.4f\n",
                p, r2_red, unique_r2))
    results[[paste(method, p)]] <- data.frame(
      method     = method,
      predictor  = p,
      r2_full    = r2_full,
      r2_reduced = r2_red,
      unique_r2  = unique_r2,
      stringsAsFactors = FALSE
    )
  }
}

unique_df <- bind_rows(results)
write.csv(unique_df,
          file.path(OUT_DIR, "sem_unique_variance.csv"),
          row.names = FALSE)

cat("\nSaved: sem_unique_variance.csv\n\n")

# ------------------------------------------------------------
# Summary wide table
# ------------------------------------------------------------
wide <- unique_df %>%
  select(method, predictor, unique_r2) %>%
  pivot_wider(names_from = predictor, values_from = unique_r2) %>%
  left_join(unique_df %>% select(method, r2_full) %>% distinct(), by = "method")

cat("Unique R^2 per predictor (semipartial, fixed loadings):\n")
print(as.data.frame(wide), row.names = FALSE)

write.csv(wide, file.path(OUT_DIR, "sem_unique_variance_wide.csv"),
          row.names = FALSE)
cat("\nSaved: sem_unique_variance_wide.csv\n")