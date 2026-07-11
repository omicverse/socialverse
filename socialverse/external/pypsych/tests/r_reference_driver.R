# psych reference driver — canonical fixture psych::bfi first 5 items (A1..A5),
# complete cases. Runs psych::alpha (raw/std/G6/average_r), psych::fa(nfactors=1,
# fm="pa") loadings+communalities+uniquenesses, and McDonald omega_total from the
# same one-factor PA solution.  Dumps the raw item matrix so Python reads the SAME
# inputs.  Written via jsonlite::toJSON(auto_unbox=TRUE, digits=15).
suppressMessages({library(psych); library(jsonlite)})
data(bfi)
X <- bfi[, 1:5]
X <- X[complete.cases(X), ]
X <- as.matrix(X)                    # 2709 x 5 integer item responses
storage.mode(X) <- "double"

out <- list()
out$data <- list(
  items = unname(as.matrix(X)),      # row-major 2709x5 -> nested JSON arrays
  colnames = colnames(X),
  n = nrow(X), p = ncol(X))

# --- Cronbach alpha (no key reversal: check.keys stays FALSE) ---
a <- suppressWarnings(alpha(X))
out$alpha <- list(
  raw_alpha = as.numeric(a$total$raw_alpha),
  std_alpha = as.numeric(a$total$std.alpha),
  G6        = as.numeric(a$total$`G6(smc)`),
  average_r = as.numeric(a$total$average_r))

# --- principal-axis factor analysis, 1 factor (from the correlation matrix) ---
R <- cor(X)
f <- suppressWarnings(fa(R, nfactors = 1, fm = "pa", n.obs = nrow(X), rotate = "none"))
out$fa <- list(
  loadings     = as.numeric(f$loadings),
  communality  = as.numeric(f$communality),
  uniqueness   = as.numeric(f$uniquenesses))

# --- McDonald omega_total from the 1-factor PA loadings (closed form) ---
# omega_tot = 1 - sum(1 - h^2) / sum(R)   with h^2 the PA communalities.
h2 <- as.numeric(f$communality)
omega_tot_paform <- (sum(R) - sum(1 - h2)) / sum(R)

# psych::omega() automatically REVERSES negatively-keyed items (here A1) before
# factoring, which is why its omega.tot differs sharply from the raw-fixture form.
# Reproduce that key reversal (item scale 1..6 -> 7-x) so the two are comparable:
Xr <- X
neg <- which(sign(f$loadings) < 0)          # items with negative g-loading (A1)
for (j in neg) Xr[, j] <- 7 - Xr[, j]
Rr <- cor(Xr)
fr <- suppressWarnings(fa(Rr, nfactors = 1, fm = "pa", n.obs = nrow(Xr), rotate = "none"))
omega_tot_keyrev <- (sum(Rr) - sum(1 - as.numeric(fr$communality))) / sum(Rr)

out$omega <- list(
  omega_tot_paform = omega_tot_paform,       # our definition, gated 1e-6 vs Python
  omega_tot_keyrev = omega_tot_keyrev,       # after A1 reversal, gated 1e-6 vs Python
  keyrev_items     = as.integer(neg),        # 1-based indices reversed
  psych_omega_tot  = as.numeric(suppressWarnings(
    omega(X, nfactors = 1, plot = FALSE))$omega.tot))  # reference-only (psych pipeline)

# --- ICC: intraclass correlations on a fixed subjects x raters matrix ---
# Canonical Shrout & Fleiss (1979) style 6 subjects x 4 raters fixture.
ratings <- matrix(c(9,2,5,8, 6,1,3,2, 8,4,6,8,
                    7,1,2,6, 10,5,6,9, 6,2,4,7),
                  ncol = 4, byrow = TRUE)
storage.mode(ratings) <- "double"
ic <- suppressWarnings(ICC(ratings, lmer = FALSE))
icr <- ic$results
out$icc <- list(
  ratings = unname(ratings),
  n = nrow(ratings), k = ncol(ratings),
  type  = as.character(icr[, "type"]),
  ICC   = as.numeric(icr[, "ICC"]),
  F     = as.numeric(icr[, "F"]),
  df1   = as.numeric(icr[, "df1"]),
  df2   = as.numeric(icr[, "df2"]),
  p     = as.numeric(icr[, "p"]),
  lower = as.numeric(icr[, "lower bound"]),
  upper = as.numeric(icr[, "upper bound"]),
  MSW   = as.numeric(ic$MSW))

# --- corr.test: correlation matrix + pairwise n + raw p on a fixed matrix ---
# Use the same 5-item complete-case fixture (A1..A5) so inputs are already dumped.
ct <- suppressWarnings(corr.test(X, adjust = "none", ci = FALSE))
out$corr_test <- list(
  r = unname(ct$r),                    # p x p correlation matrix
  n = as.numeric(ct$n),                # constant here (complete cases)
  t = unname(ct$t),                    # p x p Hotelling t
  p = unname(ct$p),                    # p x p RAW two-sided p (adjust="none")
  se = unname(ct$se))                  # p x p standard errors

write(toJSON(out, auto_unbox = TRUE, digits = 15), "pypsych/tests/reference.json")
cat("psych", as.character(packageVersion("psych")),
    "-> reference.json (raw_alpha=", round(out$alpha$raw_alpha, 6),
    ", fa L1=", round(out$fa$loadings[1], 6),
    ", omega_paform=", round(omega_tot_paform, 6), ")\n")
