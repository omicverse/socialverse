# lavaan reference driver — canonical HolzingerSwineford1939 3-factor CFA by ML.
# Dumps the fixture INPUT data (x1..x9) plus parameterEstimates() and the key
# fitMeasures() so the Python port reads the SAME inputs and matches outputs.
suppressMessages({library(lavaan); library(jsonlite)})

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"

d <- HolzingerSwineford1939
vars <- c("x1","x2","x3","x4","x5","x6","x7","x8","x9")

fit <- cfa(model, data = d)

pe <- parameterEstimates(fit, standardized = TRUE)
# keep only structural rows (loadings + variances/covariances)
pe <- pe[pe$op %in% c("=~", "~~"), ]

fm <- fitMeasures(fit, c("chisq","df","cfi","tli","rmsea","srmr"))

# full fit-index battery beyond the original chisq/cfi/tli/rmsea/srmr
fmk <- c("npar","fmin","logl","unrestricted.logl","aic","bic","bic2",
         "chisq","df","pvalue","baseline.chisq","baseline.df",
         "cfi","tli","nfi","gfi","agfi","rmsea","rmsea.ci.lower",
         "rmsea.ci.upper","rmsea.pvalue","srmr")
fmall <- fitMeasures(fit, fmk)

# modification indices (univariate score test + expected parameter change),
# sorted by mi descending; gate the top entries.
mi <- modindices(fit)
mi <- mi[order(-mi$mi), ]
mi_top <- head(mi, 15)

# lavaan's free-parameter estimates at its own optimizer stopping point.
# lavaan's nlminb converges to a finite tolerance (its gradient norm at the
# solution is ~5e-7), so its reported modification indices are evaluated at
# THESE estimates.  We store them so the parity test can verify the MI / EPC
# *formula* reproduces lavaan bit-for-bit given identical inputs, independent
# of any residual optimizer slack between the two ML solvers.
ptf <- parTable(fit)
ptf <- ptf[ptf$free > 0, ]

out <- list(
  data = as.list(d[, vars]),
  params = list(
    lhs = pe$lhs, op = pe$op, rhs = pe$rhs,
    est = as.numeric(pe$est), se = as.numeric(pe$se),
    std_lv = as.numeric(pe$std.lv), std_all = as.numeric(pe$std.all)
  ),
  fit = list(
    chisq = as.numeric(fm["chisq"]), df = as.numeric(fm["df"]),
    cfi = as.numeric(fm["cfi"]), tli = as.numeric(fm["tli"]),
    rmsea = as.numeric(fm["rmsea"]), srmr = as.numeric(fm["srmr"])
  ),
  fitmeasures = list(
    npar = as.numeric(fmall["npar"]), fmin = as.numeric(fmall["fmin"]),
    logl = as.numeric(fmall["logl"]),
    unrestricted_logl = as.numeric(fmall["unrestricted.logl"]),
    aic = as.numeric(fmall["aic"]), bic = as.numeric(fmall["bic"]),
    bic2 = as.numeric(fmall["bic2"]),
    chisq = as.numeric(fmall["chisq"]), df = as.numeric(fmall["df"]),
    pvalue = as.numeric(fmall["pvalue"]),
    baseline_chisq = as.numeric(fmall["baseline.chisq"]),
    baseline_df = as.numeric(fmall["baseline.df"]),
    cfi = as.numeric(fmall["cfi"]), tli = as.numeric(fmall["tli"]),
    nfi = as.numeric(fmall["nfi"]), gfi = as.numeric(fmall["gfi"]),
    agfi = as.numeric(fmall["agfi"]), rmsea = as.numeric(fmall["rmsea"]),
    rmsea_ci_lower = as.numeric(fmall["rmsea.ci.lower"]),
    rmsea_ci_upper = as.numeric(fmall["rmsea.ci.upper"]),
    rmsea_pvalue = as.numeric(fmall["rmsea.pvalue"]),
    srmr = as.numeric(fmall["srmr"])
  ),
  modindices = list(
    lhs = as.character(mi_top$lhs), op = as.character(mi_top$op),
    rhs = as.character(mi_top$rhs),
    mi = as.numeric(mi_top$mi), epc = as.numeric(mi_top$epc)
  ),
  coef = list(
    lhs = as.character(ptf$lhs), op = as.character(ptf$op),
    rhs = as.character(ptf$rhs), est = as.numeric(ptf$est)
  )
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pylavaan/tests/reference.json")
cat("lavaan", as.character(packageVersion("lavaan")),
    "-> reference.json (chisq=", round(fm["chisq"],4),
    ", cfi=", round(fm["cfi"],4), ", srmr=", round(fm["srmr"],4), ")\n")
