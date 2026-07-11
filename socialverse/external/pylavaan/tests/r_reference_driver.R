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
  )
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pylavaan/tests/reference.json")
cat("lavaan", as.character(packageVersion("lavaan")),
    "-> reference.json (chisq=", round(fm["chisq"],4),
    ", cfi=", round(fm["cfi"],4), ", srmr=", round(fm["srmr"],4), ")\n")
