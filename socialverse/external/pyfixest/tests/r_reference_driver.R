# fixest reference driver — a FIXED small panel (id, time, y, x) →
# feols within estimators (one-way & two-way FE) with clustered SE → JSON spec.
# The fixture INPUT data is dumped into reference.json so Python reads the SAME
# inputs.  Reproducible via set.seed.
suppressMessages({library(fixest); library(jsonlite)})

set.seed(1)
id   <- rep(1:5, each = 4)
time <- rep(1:4, 5)
x    <- rnorm(20)
y    <- 2 * x + rep(rnorm(5), each = 4) + rnorm(20)
d    <- data.frame(id = id, time = time, x = x, y = y)

out <- list(data = list(id = id, time = time, x = x, y = y))

# --- one-way FE (id), cluster on id (nested) ---
m1 <- feols(y ~ x | id, data = d, cluster = ~id)
out$oneway_id <- list(
  coef      = as.numeric(coef(m1)),
  se        = as.numeric(se(m1)),
  within_r2 = as.numeric(fitstat(m1, "wr2")$wr2),
  nobs      = m1$nobs,
  nparams   = m1$nparams)

# --- one-way FE (id), cluster on time (NON-nested) ---
m2 <- feols(y ~ x | id, data = d, cluster = ~time)
out$oneway_id_clustertime <- list(
  coef      = as.numeric(coef(m2)),
  se        = as.numeric(se(m2)),
  within_r2 = as.numeric(fitstat(m2, "wr2")$wr2))

# --- two-way FE (id + time), cluster on id ---
m3 <- feols(y ~ x | id + time, data = d, cluster = ~id)
out$twoway <- list(
  coef      = as.numeric(coef(m3)),
  se        = as.numeric(se(m3)),
  within_r2 = as.numeric(fitstat(m3, "wr2")$wr2),
  nparams   = m3$nparams)

# ==========================================================================
# fepois — Poisson PML with HD fixed effects, clustered SE
# Fixed count panel (id, x, y) with NO all-zero id group (else fixest drops
# obs and parity would compare different samples).  Inputs dumped to JSON.
# ==========================================================================
set.seed(7)
pn_id <- 6; pn_t <- 6
pid  <- rep(1:pn_id, each = pn_t)
ptime <- rep(1:pn_t, pn_id)
px   <- round(rnorm(pn_id * pn_t), 4)
peta <- 0.4 * px + rep(c(0.3, -0.2, 0.5, -0.4, 0.1, 0.2), each = pn_t)
set.seed(99)
py   <- rpois(pn_id * pn_t, exp(peta))
pd   <- data.frame(id = pid, x = px, y = py)

mp <- fepois(y ~ x | id, data = pd, cluster = ~id)
out$fepois_data <- list(id = pid, x = px, y = py)
out$fepois <- list(
  coef     = as.numeric(coef(mp)),
  se       = as.numeric(se(mp)),
  deviance = as.numeric(deviance(mp)),
  nobs     = mp$nobs)

# ==========================================================================
# newey_west — HAC vcov on a time-series feols (intercept + 2 regressors)
# ==========================================================================
set.seed(123)
nn <- 40
nt  <- 1:nn
nx1 <- round(rnorm(nn), 4)
nx2 <- round(rnorm(nn), 4)
ne  <- as.numeric(arima.sim(list(ar = 0.5), nn))
ny  <- round(1.0 + 0.7 * nx1 - 0.3 * nx2 + ne, 4)
nd  <- data.frame(t = nt, y = ny, x1 = nx1, x2 = nx2)

mnw   <- feols(y ~ x1 + x2, data = nd)
out$nw_data <- list(t = nt, y = ny, x1 = nx1, x2 = nx2)
mnw3  <- summary(mnw, vcov = NW(lag = 3) ~ t)
out$nw_lag3 <- list(
  coef = as.numeric(coef(mnw3)),
  se   = as.numeric(se(mnw3)))
mnw2  <- summary(mnw, vcov = NW(lag = 2) ~ t)
out$nw_lag2 <- list(
  coef = as.numeric(coef(mnw2)),
  se   = as.numeric(se(mnw2)))

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pyfixest/tests/reference.json")
cat("fixest", as.character(packageVersion("fixest")),
    "-> reference.json (oneway coef=", round(coef(m1), 6),
    "se=", round(se(m1), 6),
    "| fepois coef=", round(coef(mp), 6), "se=", round(se(mp), 6),
    "| NW3 se=", round(se(mnw3), 6), ")\n")
