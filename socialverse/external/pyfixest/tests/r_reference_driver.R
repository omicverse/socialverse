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

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pyfixest/tests/reference.json")
cat("fixest", as.character(packageVersion("fixest")),
    "-> reference.json (oneway coef=", round(coef(m1), 6),
    "se=", round(se(m1), 6), ")\n")
