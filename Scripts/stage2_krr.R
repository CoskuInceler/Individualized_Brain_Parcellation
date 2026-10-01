# ============================================================
# STAGE 2D: KERNEL RIDGE REGRESSION
# ============================================================
# Asks how well g can be predicted from a parcellation, rather than
# which of its features carry a significant path. The SEM answers the
# second question; this answers the first, and the two can disagree.
#
# Two kernels per method, each a subject-by-subject similarity matrix:
#   sizes   similarity of parcel-size profiles
#   fc      similarity of connectivity matrices
#
# Following Kong et al. (2021), with the safeguards the thesis version
# already had and two the sample now requires:
#
#   Covariates and kernel centering are handled inside each fold, so a
#   test subject never contributes to the quantities used to predict it.
#
#   Folds are split by family. The sample holds 419 families across 947
#   subjects, so leaving out one subject at a time would leave its
#   siblings in the training set and inflate the accuracy.
#
#   Lambda is chosen by the closed-form leave-one-out error, which one
#   eigendecomposition of the training kernel gives exactly. The nested
#   loop the thesis used would need about ten million matrix solves at
#   this sample size.
#
#   Significance comes from a permutation test. The leave-one-out
#   predictions are not independent, so the t test on their correlation
#   with the outcome does not hold.
#
# Outputs, to Outputs/<variant>/Stage2/:
#   krr_results.csv        accuracy per method and kernel
#   krr_predictions.csv    predicted and observed g per subject
# ============================================================

.libPaths(c("~/R/library", .libPaths()))

# Project root: set IP_BASE, or run this script from the Scripts folder
BASE    <- Sys.getenv("IP_BASE", normalizePath(".."))
VARIANT <- Sys.getenv("IP_VARIANT", "none")
OUT_DIR <- file.path(BASE, "Outputs", VARIANT, "Stage2")
FEAT    <- file.path(BASE, "Outputs", VARIANT, "Features")
IN_DIR  <- file.path(BASE, "Inputs")

METHODS <- c("Method_0_Schaefer", "Method_1_gMSHBM", "Method_2_AGP",
             "Method_3_SLIC_F", "Method_4_SLIC_C", "Method_5_Gordon")
LABELS  <- c("Schaefer", "gMSHBM", "AGP", "SLIC_F", "SLIC_C", "Gordon")

KERNELS <- c("sizes", "fc")
LAMBDAS <- 10^seq(-5, 5, by = 0.5)
N_FOLDS <- 10
N_PERM  <- 1000
SEED    <- 42

cat("Stage 2D: kernel ridge regression\n")
cat("Variant:", VARIANT, "\n\n")

# ------------------------------------------------------------
# Core functions
# ------------------------------------------------------------

center_kernel_fold <- function(K, train, test) {
  # Centering parameters come from the training block alone, then the
  # same transform is applied to the test rows.
  K_tr <- K[train, train, drop = FALSE]
  K_te <- K[test,  train, drop = FALSE]

  col_means <- colMeans(K_tr)
  grand     <- mean(K_tr)

  K_tr_c <- sweep(sweep(K_tr, 2, col_means), 1, col_means) + grand
  K_te_c <- sweep(sweep(K_te, 2, col_means), 1, rowMeans(K_te)) + grand

  list(train = K_tr_c, test = K_te_c)
}

select_lambda <- function(K, y) {
  # Leave-one-out error for kernel ridge has a closed form: with
  # H = K (K + lambda I)^-1, the held-out residual for subject i is
  # the fitted residual divided by 1 - H[i,i]. One eigendecomposition
  # serves every lambda, which is what makes this affordable.
  eig <- eigen(K, symmetric = TRUE)
  V <- eig$vectors
  d <- eig$values
  Vty <- crossprod(V, y)

  errors <- sapply(LAMBDAS, function(lambda) {
    shrink <- d / (d + lambda)
    y_hat  <- V %*% (shrink * Vty)
    h_diag <- rowSums(sweep(V^2, 2, shrink, "*"))
    resid  <- (y - y_hat) / pmax(1 - h_diag, 1e-10)
    mean(resid^2)
  })

  LAMBDAS[which.min(errors)]
}

residualise_fold <- function(y, covariates, train, test) {
  # Age and sex are removed with a model fitted on the training
  # subjects only, then applied to the held-out ones.
  df_tr <- data.frame(y = y[train],
                      Age = factor(covariates$Age[train]),
                      Gender = factor(covariates$Gender[train]))
  fit <- lm(y ~ Age + Gender, data = df_tr)

  df_te <- data.frame(
    Age = factor(covariates$Age[test], levels = levels(df_tr$Age)),
    Gender = factor(covariates$Gender[test], levels = levels(df_tr$Gender)))

  pred <- suppressWarnings(predict(fit, newdata = df_te))
  pred[is.na(pred)] <- mean(y[train])

  list(train = as.numeric(residuals(fit)),
       test  = as.numeric(y[test] - pred))
}

run_krr <- function(K, y, covariates, folds) {
  y_pred <- rep(NA_real_, length(y))
  y_obs  <- rep(NA_real_, length(y))

  for (f in sort(unique(folds))) {
    test  <- which(folds == f)
    train <- which(folds != f)

    r <- residualise_fold(y, covariates, train, test)
    kc <- center_kernel_fold(K, train, test)

    lambda <- select_lambda(kc$train, r$train)
    alpha  <- solve(kc$train + lambda * diag(length(train)), r$train)

    y_pred[test] <- as.numeric(kc$test %*% alpha)
    y_obs[test]  <- r$test
  }

  list(pred = y_pred, obs = y_obs, r = cor(y_pred, y_obs))
}

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

g_scores <- read.csv(file.path(OUT_DIR, "g_scores.csv"),
                     colClasses = c(subject_id = "character"))

hcp <- read.csv(file.path(IN_DIR, "HCP_all_data.csv"))
cov <- hcp[match(as.integer(g_scores$subject_id), hcp$Subject),
           c("Age", "Gender")]

subjects <- g_scores$subject_id
y <- g_scores$g_score
families <- g_scores$family_id

# Families are assigned to folds, not subjects, so siblings always end
# up on the same side of the split.
set.seed(SEED)
fam_ids <- unique(families)
fam_fold <- sample(rep_len(seq_len(N_FOLDS), length(fam_ids)))
names(fam_fold) <- fam_ids
folds <- as.integer(fam_fold[families])

cat("Subjects:", length(subjects), " Families:", length(fam_ids),
    " Folds:", N_FOLDS, "\n")
cat("Fold sizes:", paste(table(folds), collapse = ", "), "\n\n")

# ------------------------------------------------------------
# Fitting
# ------------------------------------------------------------

results <- list()
predictions <- list()

for (i in seq_along(METHODS)) {
  method <- METHODS[i]
  label  <- LABELS[i]

  for (kern in KERNELS) {
    path <- file.path(FEAT, method, paste0("kernel_", kern, ".csv"))
    if (!file.exists(path)) next

    K <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))

    if (any(is.na(K))) {
      cat(sprintf("%-10s %-6s degenerate, skipped\n", label, kern))
      next
    }

    K <- K[subjects, subjects]

    fit <- run_krr(K, y, cov, folds)

    # The held-out predictions share folds and are therefore not
    # independent, so significance comes from permuting the outcome
    # rather than from a test that assumes they are.
    set.seed(SEED)
    null_r <- replicate(N_PERM, {
      perm <- sample(seq_along(y))
      cor(fit$pred, fit$obs[perm])
    })
    p_perm <- (sum(abs(null_r) >= abs(fit$r)) + 1) / (N_PERM + 1)

    cat(sprintf("%-10s %-6s r = %.3f   p_perm = %.4f\n",
                label, kern, fit$r, p_perm))

    results[[length(results) + 1]] <- data.frame(
      variant = VARIANT, method = label, kernel = kern,
      r = round(fit$r, 4),
      r_squared = round(fit$r^2, 4),
      p_perm = p_perm,
      null_mean = round(mean(null_r), 4),
      null_sd = round(sd(null_r), 4),
      row.names = NULL)

    predictions[[length(predictions) + 1]] <- data.frame(
      variant = VARIANT, method = label, kernel = kern,
      subject = subjects, predicted = round(fit$pred, 5),
      observed = round(fit$obs, 5), row.names = NULL)
  }
}

res <- do.call(rbind, results)
res$p_fdr <- p.adjust(res$p_perm, method = "fdr")
write.csv(res, file.path(OUT_DIR, "krr_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, predictions),
          file.path(OUT_DIR, "krr_predictions.csv"), row.names = FALSE)

cat("\n")
cat(strrep("=", 52), "\n")
print(res[order(-res$r), c("method", "kernel", "r", "p_perm", "p_fdr")],
      row.names = FALSE)
cat("\nSaved to", OUT_DIR, "\n")
