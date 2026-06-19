# ============================================================
# Stage 2 Unleaked - Script 3: Kernel Ridge Regression (LOOCV)
# ============================================================
# g SCORE PREDICTIONS ONLY — behavioral domains skipped for speed.
# To add behavioral domains back, uncomment the PART A section
# and the behavioral save/summary blocks.
#
# DATA LEAKAGE FIXES (Daniel's feedback, May 2026)
# -------------------------------------------------
# Two leakage points from the previous version are fixed:
#
# FIX 1 — Covariate regression now happens INSIDE each fold.
#   Previously: lm(g ~ Age + Gender) fit on all 95 subjects,
#     residuals extracted for all 95, then used in LOOCV.
#     Problem: test subject i's residual was contaminated by
#     its own data in the covariate model.
#   Now: For each fold i, lm is fit on the 94 training subjects
#     only. That fitted model is then used to de-mean subject
#     i's score to produce a clean test residual.
#
# FIX 2 — Kernel centering now happens INSIDE each fold.
#   Previously: center_kernel() applied to full 95x95 matrix
#     before LOOCV. Problem: centering used all subjects
#     including the test subject.
#   Now: For each fold i, the (n-1)x(n-1) training kernel is
#     centered. The centering parameters (column means and grand
#     mean of the training kernel) are then applied to the
#     1x(n-1) test row, following Kong et al. 2021.
#
# Implements KRR following Kong et al. 2021 with:
#   1. Covariate regression within each fold (FIX 1)
#   2. Kernel centering within each fold (FIX 2)
#   3. Lambda tuned via inner LOOCV (10^seq(-5,5), 11 values)
#   4. LOOCV prediction: y_hat_i = k_test_c' * (K_tr_c + lI)^{-1} * y_train_resid
#
# Note: Schaefer F3 is degenerate (zero variance) — skipped.
#
# Outputs saved to Stage_2_Unleaked/KernelRegression/:
#   krr_g_results.csv
#   krr_g_summary.csv
# ============================================================

library(tidyverse)

BASE     <- "C:/Thesis_Main/Analysis"
IN_DATA  <- file.path(BASE, "Outputs/Stage_2_Unleaked/Input_Data")
SEM_DIR  <- file.path(BASE, "Outputs/Stage_2_Unleaked/SEM")
OUT_DIR  <- file.path(BASE, "Outputs/Stage_2_Unleaked/KernelRegression")
HCP_FILE <- file.path(BASE, "Inputs/Behavior/HCP_all_data.csv")
BEH_DIR  <- file.path(BASE, "Inputs/Behavior")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

METHOD_ORDER <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C")

MISSING_SUBJECTS <- c("117122", "151526", "156637", "298051", "366446")

# ============================================================
# SECTION 0: LOAD ALL DATA
# ============================================================

cat("============================================================\n")
cat("Stage 2 Unleaked - Script 3: Kernel Ridge Regression\n")
cat("  (g score only — behavioral domains skipped)\n")
cat("  (leakage-free: covariate regression + kernel centering\n")
cat("   both performed within each LOOCV fold)\n")
cat("============================================================\n\n")
cat("Loading data...\n")

# subject list
all_subjects <- trimws(readLines(file.path(BEH_DIR, "list_subjects.txt")))
all_subjects <- all_subjects[nchar(all_subjects) > 0]
valid_mask   <- !all_subjects %in% MISSING_SUBJECTS
subjects     <- all_subjects[valid_mask]
n            <- length(subjects)
stopifnot(n == 95)

# covariates
HCP_Data   <- read.csv(HCP_FILE)
covariates <- HCP_Data %>%
  filter(Subject %in% as.integer(subjects)) %>%
  select(Subject, Age, Gender) %>%
  mutate(subject_id = as.character(Subject)) %>%
  arrange(match(subject_id, subjects)) %>%
  select(subject_id, Age, Gender)

stopifnot(nrow(covariates) == 95)
stopifnot(all(covariates$subject_id == subjects))
cat("  Covariates: Age levels =",
    paste(sort(unique(covariates$Age)), collapse = ", "), "\n")
cat("  Gender:", table(covariates$Gender)["F"], "F,",
    table(covariates$Gender)["M"], "M\n\n")

# FC similarity matrices (F4B)
read_sim <- function(method) {
  as.matrix(read.csv(
    file.path(IN_DATA, paste0("fc_similarity_", method, ".csv")),
    row.names = 1, check.names = FALSE))
}
fc_sim_Schaefer <- read_sim("Schaefer")
fc_sim_gMSHBM   <- read_sim("gMSHBM")
fc_sim_AGP      <- read_sim("AGP")
fc_sim_SLIC_F   <- read_sim("SLIC_F")
fc_sim_SLIC_C   <- read_sim("SLIC_C")

# Parcel size similarity matrices (F3) — Schaefer degenerate
read_f3 <- function(method) {
  as.matrix(read.csv(
    file.path(IN_DATA, paste0("f3_similarity_", method, ".csv")),
    row.names = 1, check.names = FALSE))
}
f3_sim_gMSHBM <- read_f3("gMSHBM")
f3_sim_AGP    <- read_f3("AGP")
f3_sim_SLIC_F <- read_f3("SLIC_F")
f3_sim_SLIC_C <- read_f3("SLIC_C")

# Kernel lookup lists
FC_KERNELS <- list(
  Schaefer = fc_sim_Schaefer,
  gMSHBM   = fc_sim_gMSHBM,
  AGP      = fc_sim_AGP,
  SLIC_F   = fc_sim_SLIC_F,
  SLIC_C   = fc_sim_SLIC_C
)

F3_KERNELS <- list(
  gMSHBM = f3_sim_gMSHBM,
  AGP    = f3_sim_AGP,
  SLIC_F = f3_sim_SLIC_F,
  SLIC_C = f3_sim_SLIC_C
  # Schaefer excluded — degenerate
)

# g scores — SINGLE brain-free g score from pure measurement model
g_scores <- read.csv(
  file.path(SEM_DIR, "g_scores.csv"),
  stringsAsFactors = FALSE,
  colClasses = c(subject_id = "character"))

g_scores <- g_scores %>%
  arrange(match(subject_id, subjects))

stopifnot(all(g_scores$subject_id == subjects))
cat("  g scores: single brain-free score, N =", nrow(g_scores), "\n")
cat("    mean =", round(mean(g_scores$g_score), 4),
    " sd =", round(sd(g_scores$g_score), 4), "\n\n")

# Raw g score vector (NOT pre-residualised — residualisation happens inside each fold)
g_raw <- g_scores$g_score

# ============================================================
# SECTION 1: KRR CORE FUNCTIONS (leakage-free)
# ============================================================

LAMBDAS <- 10^seq(-5, 5, by = 1)

# --- FIX 1: Covariate regression within fold ---
# Fit lm on training subjects [-i], apply to test subject [i].
# Returns: list(y_train_resid, y_test_resid)
regress_covariates_fold <- function(y, covariates, test_idx) {
  y_train  <- y[-test_idx]
  cov_tr   <- covariates[-test_idx, ]
  cov_te   <- covariates[test_idx,  ]
  
  df_train <- data.frame(
    y      = y_train,
    Age    = factor(cov_tr$Age),
    Gender = factor(cov_tr$Gender)
  )
  fit <- lm(y ~ Age + Gender, data = df_train)
  
  # Build test data frame — ensure factor levels match training
  df_test <- data.frame(
    Age    = factor(cov_te$Age,    levels = levels(df_train$Age)),
    Gender = factor(cov_te$Gender, levels = levels(df_train$Gender))
  )
  
  y_hat_test <- tryCatch(
    predict(fit, newdata = df_test),
    error   = function(e) mean(y_train),  # fallback: use training mean
    warning = function(w) {               # unseen level: use training mean
      suppressWarnings(predict(fit, newdata = df_test))
    }
  )
  
  # If prediction is NA (completely unseen level), fall back to training mean
  if (is.na(y_hat_test)) y_hat_test <- mean(y_train)
  
  list(
    y_train_resid = as.numeric(residuals(fit)),
    y_test_resid  = as.numeric(cov_te$y_raw - y_hat_test)  # placeholder; see below
  )
}

# Cleaner implementation used in the main loop (avoids the y_raw issue above):
# Returns y_train_resid (n-1 vector) and y_test_resid (scalar)
residualise_fold <- function(y_all, covariates, test_idx) {
  y_train  <- y_all[-test_idx]
  y_test   <- y_all[test_idx]
  cov_tr   <- covariates[-test_idx, ]
  cov_te   <- covariates[test_idx,  , drop = FALSE]
  
  df_train <- data.frame(
    y      = y_train,
    Age    = factor(cov_tr$Age),
    Gender = factor(cov_tr$Gender)
  )
  fit_cov <- lm(y ~ Age + Gender, data = df_train)
  
  df_test <- data.frame(
    Age    = factor(cov_te$Age,    levels = levels(df_train$Age)),
    Gender = factor(cov_te$Gender, levels = levels(df_train$Gender))
  )
  
  y_hat_te <- tryCatch({
    p <- predict(fit_cov, newdata = df_test)
    if (is.na(p)) mean(y_train) else p
  }, error = function(e) mean(y_train))
  
  list(
    y_train_resid = as.numeric(residuals(fit_cov)),
    y_test_resid  = as.numeric(y_test - y_hat_te)
  )
}

# --- FIX 2: Kernel centering within fold (Kong et al. 2021) ---
# Centers (n-1)x(n-1) training kernel and applies same transform to test row.
# Returns: list(K_tr_c, k_te_c)
center_kernel_fold <- function(K_raw, test_idx) {
  K_tr     <- K_raw[-test_idx, -test_idx]   # (n-1) x (n-1) training kernel
  k_te     <- K_raw[test_idx,  -test_idx]   # 1 x (n-1) test row
  
  # Training centering parameters
  col_means_tr  <- colMeans(K_tr)            # mean of each column in training kernel
  grand_mean_tr <- mean(K_tr)                # grand mean of training kernel
  
  # Center training kernel: K_c[j,k] = K[j,k] - col_means[j] - col_means[k] + grand_mean
  K_tr_c <- K_tr -
    outer(col_means_tr, rep(1, nrow(K_tr))) -
    outer(rep(1, nrow(K_tr)), col_means_tr) +
    grand_mean_tr
  
  # Center test row using training parameters (Kong et al. 2021):
  # k_te_c[j] = k_te[j] - mean(k_te) - col_means_tr[j] + grand_mean_tr
  mean_k_te <- mean(k_te)
  k_te_c    <- k_te - mean_k_te - col_means_tr + grand_mean_tr
  
  list(K_tr_c = K_tr_c, k_te_c = as.numeric(k_te_c))
}

# --- Lambda selection (unchanged, operates on training fold only) ---
select_lambda <- function(K_train, y_train) {
  m      <- length(y_train)
  errors <- numeric(length(LAMBDAS))
  for (l in seq_along(LAMBDAS)) {
    lambda <- LAMBDAS[l]
    y_hat  <- numeric(m)
    for (j in 1:m) {
      alpha    <- solve(K_train[-j,-j] + lambda * diag(m-1), y_train[-j])
      y_hat[j] <- sum(K_train[j,-j] * alpha)
    }
    errors[l] <- mean((y_train - y_hat)^2)
  }
  LAMBDAS[which.min(errors)]
}

# --- Main LOOCV loop: leakage-free ---
loocv_krr_clean <- function(K_raw, y_all, covariates) {
  n     <- nrow(K_raw)
  y_hat <- numeric(n)
  y_res <- numeric(n)   # collect test residuals for correlation
  
  for (i in 1:n) {
    
    # FIX 1: residualise within fold
    res        <- residualise_fold(y_all, covariates, i)
    y_tr_resid <- res$y_train_resid
    y_te_resid <- res$y_test_resid
    
    # FIX 2: center kernel within fold
    kc         <- center_kernel_fold(K_raw, i)
    K_tr_c     <- kc$K_tr_c
    k_te_c     <- kc$k_te_c
    
    # Tune lambda on training fold
    lambda     <- select_lambda(K_tr_c, y_tr_resid)
    
    # Predict
    alpha      <- solve(K_tr_c + lambda * diag(n - 1), y_tr_resid)
    y_hat[i]   <- sum(k_te_c * alpha)
    y_res[i]   <- y_te_resid
  }
  
  list(y_hat = y_hat, y_res = y_res)
}

# --- Wrapper: run KRR and return r, p, significant ---
run_krr_eval <- function(K_raw, y_all, covariates) {
  result <- loocv_krr_clean(K_raw, y_all, covariates)
  test   <- cor.test(result$y_res, result$y_hat, method = "pearson")
  data.frame(
    r           = as.numeric(test$estimate),
    p_value     = test$p.value,
    significant = test$p.value < 0.05
  )
}

# ============================================================
# SECTION 2: PART A — BEHAVIORAL DOMAINS (SKIPPED)
# ============================================================
# Uncomment and add back the behavioral domain section from
# the full version of this script when ready to run.
# ============================================================

cat("============================================================\n")
cat("PART A: Behavioral Domains — SKIPPED (for speed)\n")
cat("============================================================\n\n")

# ============================================================
# SECTION 3: PART B — SINGLE G SCORE (both kernels)
# ============================================================

cat("============================================================\n")
cat("PART B: Single Brain-Free g Score — both F3 and F4B kernels\n")
cat("  (leakage-free: covariate regression + centering in fold)\n")
cat("============================================================\n\n")

g_results <- list()

for (method in METHOD_ORDER) {
  
  cat(sprintf("--- %s ---\n", method))
  
  # --- F4B FC kernel ---
  ev_f4b <- run_krr_eval(FC_KERNELS[[method]], g_raw, covariates)
  cat(sprintf("  [%s | F4B_FC]         r = %+.3f  p = %.4f%s\n",
              method, ev_f4b$r, ev_f4b$p_value,
              ifelse(ev_f4b$significant, " *", "")))
  g_results[[paste(method, "F4B")]] <- data.frame(
    method       = method,
    kernel       = "F4B_FC",
    outcome_type = "g_score",
    outcome      = "g",
    r            = ev_f4b$r,
    p_value      = ev_f4b$p_value,
    significant  = ev_f4b$significant,
    stringsAsFactors = FALSE)
  
  # --- F3 parcel size kernel (Schaefer excluded) ---
  if (method %in% names(F3_KERNELS)) {
    ev_f3 <- run_krr_eval(F3_KERNELS[[method]], g_raw, covariates)
    cat(sprintf("  [%s | F3_ParcelSize]  r = %+.3f  p = %.4f%s\n",
                method, ev_f3$r, ev_f3$p_value,
                ifelse(ev_f3$significant, " *", "")))
    g_results[[paste(method, "F3")]] <- data.frame(
      method       = method,
      kernel       = "F3_ParcelSize",
      outcome_type = "g_score",
      outcome      = "g",
      r            = ev_f3$r,
      p_value      = ev_f3$p_value,
      significant  = ev_f3$significant,
      stringsAsFactors = FALSE)
  } else {
    cat(sprintf("  [%s | F3_ParcelSize]  SKIPPED (degenerate)\n", method))
    g_results[[paste(method, "F3")]] <- data.frame(
      method       = method,
      kernel       = "F3_ParcelSize",
      outcome_type = "g_score",
      outcome      = "g",
      r            = NA_real_,
      p_value      = NA_real_,
      significant  = NA,
      stringsAsFactors = FALSE)
  }
  
  cat("\n")
}

krr_g <- bind_rows(g_results)
rownames(krr_g) <- NULL

# ============================================================
# SECTION 4: SAVE
# ============================================================

krr_g_summary <- krr_g %>%
  filter(!is.na(r)) %>%
  group_by(method, kernel, outcome_type) %>%
  summarise(r           = round(mean(r), 3),
            p_value     = round(mean(p_value), 4),
            significant = all(significant),
            .groups = "drop")

write.csv(krr_g,
          file.path(OUT_DIR, "krr_g_results.csv"),   row.names = FALSE)
write.csv(krr_g_summary,
          file.path(OUT_DIR, "krr_g_summary.csv"),   row.names = FALSE)

cat("\nSaved to:", OUT_DIR, "\n")
cat("  krr_g_results.csv\n")
cat("  krr_g_summary.csv\n\n")

# ============================================================
# SECTION 5: CONSOLE SUMMARIES
# ============================================================

cat("============================================================\n")
cat("SUMMARY — Single g Score: r per method x kernel\n")
cat("(Leakage-free LOOCV: covariate regression + kernel centering\n")
cat(" both performed within each fold)\n")
cat("============================================================\n")
krr_g %>%
  filter(!is.na(r)) %>%
  mutate(method = factor(method, levels = METHOD_ORDER),
         sig    = ifelse(significant, "*", "")) %>%
  arrange(kernel, method) %>%
  select(method, kernel, r, p_value, sig) %>%
  mutate(r = round(r, 3), p_value = round(p_value, 4)) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

cat("\nNOTE: Behavioral domain predictions were skipped.\n")
cat("Re-run the full version of this script to include them.\n")