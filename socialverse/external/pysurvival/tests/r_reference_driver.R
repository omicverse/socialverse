# survival reference driver — Kaplan-Meier + Cox PH (Efron & Breslow ties) → JSON.
suppressMessages({library(survival); library(jsonlite)})
lung2 <- na.omit(lung[, c("time","status","age","sex","ph.ecog")])
out <- list()

# --- Kaplan-Meier (status: 1=censored, 2=event in `lung`) ---
km <- survfit(Surv(time, status) ~ 1, data=lung2)
out$km <- list(time=km$time, n.risk=km$n.risk, n.event=km$n.event,
               surv=km$surv, std.err=km$std.err,
               lower=km$lower, upper=km$upper,
               median=as.numeric(summary(km)$table["median"]))
out$km_data <- list(time=lung2$time, status=lung2$status)

# --- Cox PH: Efron ties (default) ---
cox_e <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data=lung2, ties="efron")
out$cox_efron <- list(coef=as.numeric(coef(cox_e)),
                      se=as.numeric(sqrt(diag(vcov(cox_e)))),
                      z=as.numeric(coef(cox_e)/sqrt(diag(vcov(cox_e)))),
                      loglik=cox_e$loglik,               # (null, fitted)
                      concordance=as.numeric(cox_e$concordance["concordance"]))
# --- Cox PH: Breslow ties ---
cox_b <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data=lung2, ties="breslow")
out$cox_breslow <- list(coef=as.numeric(coef(cox_b)),
                        se=as.numeric(sqrt(diag(vcov(cox_b)))),
                        loglik=cox_b$loglik)
out$cox_data <- list(time=lung2$time, status=lung2$status,
                     age=lung2$age, sex=lung2$sex, ph.ecog=lung2$ph.ecog)

write(toJSON(out, auto_unbox=TRUE, digits=15, pretty=TRUE), "pysurvival/tests/reference.json")
cat("survival", as.character(packageVersion("survival")),
    "-> reference.json (n=", nrow(lung2), ", cox efron coef=", round(coef(cox_e),5), ")\n")
