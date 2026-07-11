# demography reference driver.
#
# Life table: replicate demography:::lt (the numerical core of demography::lifetable)
# on a FIXED single-year age-specific mortality schedule. lt() outputs the full
# actuarial chain ax, qx, lx, dx, Lx, Tx, ex and life expectancy e0 = ex[age 0].
# We run it for sex = female / male / total (the a0/a1 separation-factor branch).
#
# Kitagawa and Oaxaca-Blinder are NOT in the demography package, so their
# reference values are computed here directly from the standard closed-form
# formulas on a FIXED 2-group rate/covariate table.
suppressMessages({library(demography); library(jsonlite)})
out <- list()

# ---- FIXED single-year mortality schedule (ages 0..9, nx = 1) ----
mx <- c(0.020, 0.001, 0.002, 0.005, 0.010, 0.020, 0.040, 0.080, 0.160, 0.300)
ages <- 0:9

lt_for <- function(sex) {
  d <- demography:::lt(mx, startage = 0, agegroup = 1, sex = sex)
  list(ax = as.numeric(d$ax), mx = as.numeric(d$mx), qx = as.numeric(d$qx),
       lx = as.numeric(d$lx), dx = as.numeric(d$dx), Lx = as.numeric(d$Lx),
       Tx = as.numeric(d$Tx), ex = as.numeric(d$ex), nx = as.numeric(d$nx),
       e0 = as.numeric(d$ex[1]))
}

out$lifetable <- list(
  input = list(mx = mx, ages = ages, agegroup = 1),
  female = lt_for("female"),
  male   = lt_for("male"),
  total  = lt_for("total")
)

# ---- Kitagawa rate decomposition ----
# Two populations, each stratified into K groups. A crude rate for pop p is
#   R_p = sum_i c_{p,i} * r_{p,i}   where c = compositional share (sum to 1),
#                                          r = group-specific rate.
# Kitagawa splits R2 - R1 into:
#   rate effect        = sum_i [(r2_i - r1_i) * (c1_i + c2_i)/2]
#   composition effect = sum_i [(c2_i - c1_i) * (r1_i + r2_i)/2]
c1 <- c(0.40, 0.35, 0.15, 0.10)          # population 1 composition
c2 <- c(0.25, 0.30, 0.25, 0.20)          # population 2 composition
r1 <- c(0.005, 0.010, 0.030, 0.090)      # population 1 group rates
r2 <- c(0.004, 0.009, 0.028, 0.085)      # population 2 group rates
R1 <- sum(c1 * r1); R2 <- sum(c2 * r2)
rate_effect <- sum((r2 - r1) * (c1 + c2) / 2)
comp_effect <- sum((c2 - c1) * (r1 + r2) / 2)
out$kitagawa <- list(
  input = list(c1 = c1, c2 = c2, r1 = r1, r2 = r2),
  R1 = R1, R2 = R2, total = R2 - R1,
  rate_effect = rate_effect, composition_effect = comp_effect
)

# ---- Oaxaca-Blinder decomposition (twofold, reference = group B coefficients) ----
# Groups A and B, outcome regressed on covariates X (with intercept).
# Mean gap  Ybar_A - Ybar_B  =  (Xbar_A - Xbar_B) %*% beta_B          [explained / endowments]
#                             +  Xbar_A %*% (beta_A - beta_B)          [unexplained / coefficients]
set.seed(42)
nA <- 200; nB <- 220
xA1 <- rnorm(nA, 5, 2); xA2 <- rnorm(nA, 3, 1)
xB1 <- rnorm(nB, 4, 2); xB2 <- rnorm(nB, 3.5, 1)
yA <- 2.0 + 1.5 * xA1 + 0.8 * xA2 + rnorm(nA, 0, 1)
yB <- 1.0 + 1.2 * xB1 + 1.0 * xB2 + rnorm(nB, 0, 1)
fitA <- lm(yA ~ xA1 + xA2); fitB <- lm(yB ~ xB1 + xB2)
bA <- as.numeric(coef(fitA)); bB <- as.numeric(coef(fitB))
XbarA <- c(1, mean(xA1), mean(xA2)); XbarB <- c(1, mean(xB1), mean(xB2))
gap <- mean(yA) - mean(yB)
explained   <- sum((XbarA - XbarB) * bB)
unexplained <- sum(XbarA * (bA - bB))
out$oaxaca <- list(
  input = list(
    yA = yA, xA = cbind(xA1, xA2),
    yB = yB, xB = cbind(xB1, xB2)
  ),
  betaA = bA, betaB = bB,
  meanYA = mean(yA), meanYB = mean(yB), gap = gap,
  explained = explained, unexplained = unexplained
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pydemography/tests/reference.json")
cat("demography", as.character(packageVersion("demography")),
    "-> reference.json (e0 total =", round(out$lifetable$total$e0, 6),
    ", kitagawa total =", round(out$kitagawa$total, 6),
    ", oaxaca gap =", round(out$oaxaca$gap, 6), ")\n")
