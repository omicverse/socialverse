# MatchIt reference driver — canonical lalonde fixture.
# matchit(treat~age+educ+re74+re75, method="nearest", distance="glm"):
#   - propensity score via binomial GLM (logit)
#   - 1:1 greedy nearest-neighbour matching w/o replacement, m.order="largest"
#   - standardized mean differences (Std. Mean Diff.) before/after from summary()
# Dumps the fixture INPUTS so Python reads the SAME data.
suppressMessages({library(MatchIt); library(jsonlite)})
data(lalonde)
rn <- rownames(lalonde)
covs <- c("age", "educ", "re74", "re75")

m <- matchit(treat ~ age + educ + re74 + re75, data = lalonde,
             method = "nearest", distance = "glm")

# PS logistic-regression coefficients (Intercept, age, educ, re74, re75)
ps_coef <- as.numeric(coef(m$model))
names(ps_coef) <- names(coef(m$model))

# match matrix: for each treated (rowname) the matched control rowname
mm <- m$match.matrix[, 1]
names(mm) <- rownames(m$match.matrix)

# summary() standardized mean differences
s <- summary(m)
smd_before <- s$sum.all[, "Std. Mean Diff."]
smd_after  <- s$sum.matched[, "Std. Mean Diff."]
smd_vars   <- rownames(s$sum.all)   # c("distance", covs...)

# --- extensions: WeightIt::get_w_from_ps + MatchIt Mahalanobis + summary() ---
suppressMessages(library(WeightIt))

ps  <- as.numeric(m$distance)          # fitted propensity score (P(treat=1))
trt <- as.integer(lalonde$treat)

# (1) get_w_from_ps -> balancing weights for ATE / ATT / ATC
w_ate <- as.numeric(get_w_from_ps(ps, trt, estimand = "ATE"))
w_att <- as.numeric(get_w_from_ps(ps, trt, estimand = "ATT"))
w_atc <- as.numeric(get_w_from_ps(ps, trt, estimand = "ATC"))

# (2) mahalanobis_dist: n1 x n0 pairwise (scaled) Mahalanobis distances.
#     Use the treat-supplied path via a Mahalanobis matchit and its distance
#     helper. We call the internal directly on the covariate formula.
maha <- getFromNamespace("mahalanobis_dist", "MatchIt")
Dm <- maha(treat ~ age + educ + re74 + re75, data = lalonde)
# Dm rows = treated rownames, cols = control rownames
maha_rows <- rownames(Dm)
maha_cols <- colnames(Dm)
# flatten row-major so Python can rebuild the same (n1 x n0) matrix
Dm_flat <- as.numeric(t(Dm))

# (3) summary() balance table with propensity-score weights (ATT weights),
#     giving the WEIGHTED (after) balance columns; the all-ones gives before.
#     Build a WeightIt-weighted matchit-style balance via summary.matchit on a
#     weighting object is complex; instead report the raw covariate balance
#     columns MatchIt computes for the nearest-match object above (before/after
#     matching) plus the covariate-only eCDF/VarRatio for the parity of the
#     balance_table() port.
bt_vars   <- rownames(s$sum.all)
bt_before <- list(
  std_mean_diff = as.numeric(s$sum.all[, "Std. Mean Diff."]),
  var_ratio     = as.numeric(s$sum.all[, "Var. Ratio"]),
  ecdf_mean     = as.numeric(s$sum.all[, "eCDF Mean"]),
  ecdf_max      = as.numeric(s$sum.all[, "eCDF Max"])
)
bt_after <- list(
  std_mean_diff = as.numeric(s$sum.matched[, "Std. Mean Diff."]),
  var_ratio     = as.numeric(s$sum.matched[, "Var. Ratio"]),
  ecdf_mean     = as.numeric(s$sum.matched[, "eCDF Mean"]),
  ecdf_max      = as.numeric(s$sum.matched[, "eCDF Max"])
)
# matching weights R used for the "after" columns
match_w <- as.numeric(m$weights)

out <- list(
  data = list(
    rownames = rn,
    treat = as.integer(lalonde$treat),
    age   = as.numeric(lalonde$age),
    educ  = as.numeric(lalonde$educ),
    re74  = as.numeric(lalonde$re74),
    re75  = as.numeric(lalonde$re75)
  ),
  ps_coef = list(names = names(ps_coef), value = ps_coef),
  distance = as.numeric(m$distance),                 # fitted PS (probabilities)
  match = list(treated = names(mm), control = as.character(mm)),
  smd = list(vars = smd_vars,
             before = as.numeric(smd_before),
             after  = as.numeric(smd_after)),
  get_w_from_ps = list(ate = w_ate, att = w_att, atc = w_atc),
  mahalanobis = list(rows = maha_rows, cols = maha_cols,
                     n1 = nrow(Dm), n0 = ncol(Dm),
                     flat = Dm_flat),
  balance_table = list(vars = bt_vars,
                       match_weights = match_w,
                       before = bt_before,
                       after = bt_after)
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pymatchit/tests/reference.json")
cat("MatchIt", as.character(packageVersion("MatchIt")),
    "-> reference.json (n=", nrow(lalonde),
    ", n.matched.treated=", length(mm),
    ", distance.SMD.before=", round(smd_before[1], 6), ")\n")
