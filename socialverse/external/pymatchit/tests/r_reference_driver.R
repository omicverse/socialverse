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
             after  = as.numeric(smd_after))
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pymatchit/tests/reference.json")
cat("MatchIt", as.character(packageVersion("MatchIt")),
    "-> reference.json (n=", nrow(lalonde),
    ", n.matched.treated=", length(mm),
    ", distance.SMD.before=", round(smd_before[1], 6), ")\n")
