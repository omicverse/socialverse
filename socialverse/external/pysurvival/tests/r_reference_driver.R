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

# --- clogit: conditional logistic on matched case-control (infert) ---
data(infert)
cl <- clogit(case ~ spontaneous + induced + strata(stratum), data=infert)
out$clogit <- list(coef=as.numeric(coef(cl)),
                   se=as.numeric(sqrt(diag(vcov(cl)))),
                   loglik=cl$loglik)                 # (null, fitted)
out$clogit_data <- list(case=infert$case, stratum=infert$stratum,
                        spontaneous=infert$spontaneous, induced=infert$induced)

# --- survreg: parametric AFT (Weibull / exponential / lognormal) on lung ---
lung3 <- na.omit(lung[, c("time","status","age","sex")])
srw <- survreg(Surv(time, status) ~ age + sex, data=lung3, dist="weibull")
out$survreg_weibull <- list(coef=as.numeric(coef(srw)),   # (Intercept, age, sex)
                            scale=as.numeric(srw$scale),
                            se=as.numeric(sqrt(diag(vcov(srw)))),  # [coef..., Log(scale)]
                            loglik=as.numeric(srw$loglik[2]))
sre <- survreg(Surv(time, status) ~ age + sex, data=lung3, dist="exponential")
out$survreg_exp <- list(coef=as.numeric(coef(sre)),
                        scale=as.numeric(sre$scale),
                        se=as.numeric(sqrt(diag(vcov(sre)))),      # [coef...] only
                        loglik=as.numeric(sre$loglik[2]))
srl <- survreg(Surv(time, status) ~ age + sex, data=lung3, dist="lognormal")
out$survreg_lognormal <- list(coef=as.numeric(coef(srl)),
                              scale=as.numeric(srl$scale),
                              se=as.numeric(sqrt(diag(vcov(srl)))),
                              loglik=as.numeric(srl$loglik[2]))
out$survreg_data <- list(time=lung3$time, status=lung3$status,
                         age=lung3$age, sex=lung3$sex)

write(toJSON(out, auto_unbox=TRUE, digits=15, pretty=TRUE), "pysurvival/tests/reference.json")
cat("survival", as.character(packageVersion("survival")),
    "-> reference.json (n=", nrow(lung2), ", cox efron coef=", round(coef(cox_e),5),
    ", clogit coef=", round(coef(cl),5),
    ", survreg.wb scale=", round(srw$scale,5), ")\n")
