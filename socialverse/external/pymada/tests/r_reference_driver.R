# mada reference driver -- Reitsma bivariate REML on the canonical AuditC
# fixture.  Dumps the raw TP/FN/FP/TN inputs AND the fitted pooled sens/spec,
# their SEs, and the bivariate variance components so Python reads the SAME
# inputs and matches element-wise.
suppressMessages({library(mada); library(jsonlite)})
data(AuditC)

fit <- reitsma(AuditC)
s   <- summary(fit)

coef  <- as.numeric(fit$coefficients)          # logit(sens), logit(fpr)
vcov  <- fit$vcov
Psi   <- fit$Psi
se    <- sqrt(diag(vcov))
sens  <- plogis(coef[1])
fpr   <- plogis(coef[2])

out <- list(
  data = list(TP = AuditC$TP, FN = AuditC$FN, FP = AuditC$FP, TN = AuditC$TN),
  reitsma = list(
    coef            = coef,
    se              = as.numeric(se),
    vcov            = as.numeric(as.vector(vcov)),   # column-major 2x2
    Psi             = as.numeric(as.vector(Psi)),    # column-major 2x2
    sensitivity     = as.numeric(sens),
    false_pos_rate  = as.numeric(fpr),
    logLik          = as.numeric(fit$logLik),
    par             = as.numeric(fit$par)
  )
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pymada/tests/reference.json")
cat("mada", as.character(packageVersion("mada")),
    "-> reference.json (sens=", round(sens, 6),
    ", fpr=", round(fpr, 6), ")\n")
