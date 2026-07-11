# metafor reference driver — canonical dat.bcg fixture → JSON spec.
suppressMessages({library(metafor); library(jsonlite)})
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
yi <- as.numeric(dat$yi); vi <- as.numeric(dat$vi)
out <- list(fixture=list(yi=yi, vi=vi, ablat=dat$ablat, year=dat$year))

grab <- function(m) list(
  beta=as.numeric(m$beta), se=as.numeric(m$se), zval=as.numeric(m$zval),
  pval=as.numeric(m$pval), ci.lb=as.numeric(m$ci.lb), ci.ub=as.numeric(m$ci.ub),
  tau2=m$tau2, se.tau2=m$se.tau2, I2=m$I2, H2=m$H2, QE=m$QE, QEp=m$QEp,
  k=m$k, p=m$p, test=m$test)

# 1. Random-effects REML (default)
out$rma_reml <- grab(rma(yi, vi, method="REML"))
# 2. Random-effects DL
out$rma_dl   <- grab(rma(yi, vi, method="DL"))
# 3. Fixed-effect (common)
out$rma_fe   <- grab(rma(yi, vi, method="EE"))
# 4. Knapp-Hartung
out$rma_hk   <- grab(rma(yi, vi, method="REML", test="knha"))
# 5. Meta-regression (moderators: ablat + year), REML
mr <- rma(yi, vi, mods=~ablat+year, method="REML", data=dat)
out$rma_mods <- c(grab(mr), list(QM=mr$QM, QMp=mr$QMp))
# prediction interval
pr <- predict(rma(yi, vi, method="REML"))
out$pred_reml <- list(pi.lb=as.numeric(pr$pi.lb), pi.ub=as.numeric(pr$pi.ub))

write(toJSON(out, auto_unbox=TRUE, digits=15, pretty=TRUE), "pymetafor/tests/reference.json")
cat("metafor", as.character(packageVersion("metafor")), "-> reference.json (k=", length(yi), ")\n")
# 6. Meta-regression with CENTERED moderators (well-conditioned) — proves the
#    port is exact; the uncentered case above is ill-conditioned in metafor too.
datc <- dat; datc$ablat_c <- datc$ablat - mean(datc$ablat); datc$year_c <- datc$year - mean(datc$year)
mrc <- rma(yi, vi, mods=~ablat_c+year_c, method="REML", data=datc)
outc <- list(
  ablat_c=as.numeric(datc$ablat_c), year_c=as.numeric(datc$year_c),
  beta=as.numeric(mrc$beta), se=as.numeric(mrc$se), zval=as.numeric(mrc$zval),
  pval=as.numeric(mrc$pval), ci.lb=as.numeric(mrc$ci.lb), ci.ub=as.numeric(mrc$ci.ub),
  tau2=mrc$tau2, I2=mrc$I2, H2=mrc$H2, QE=mrc$QE, QM=mrc$QM, QMp=mrc$QMp)
ref <- jsonlite::fromJSON("pymetafor/tests/reference.json", simplifyVector=TRUE)
ref$rma_mods_centered <- outc
write(jsonlite::toJSON(ref, auto_unbox=TRUE, digits=15, pretty=TRUE), "pymetafor/tests/reference.json")
cat("added centered-mods case; cond well-behaved\n")

# 7. BLUP (best linear unbiased predictors) — empirical-Bayes shrinkage per study
#    on the default REML random-effects fit (intercept-only).
mb <- rma(yi, vi, method="REML")
bl <- blup(mb)
ref <- jsonlite::fromJSON("pymetafor/tests/reference.json", simplifyVector=TRUE)
ref$blup_reml <- list(
  pred=as.numeric(bl$pred), se=as.numeric(bl$se),
  pi.lb=as.numeric(bl$pi.lb), pi.ub=as.numeric(bl$pi.ub))
write(jsonlite::toJSON(ref, auto_unbox=TRUE, digits=15, pretty=TRUE), "pymetafor/tests/reference.json")
cat("added blup_reml (k=", length(bl$pred), ")\n")
