# robumeta reference driver — RVE for dependent effect sizes.
# CORR (correlated effects) on corrdat + HIER (hierarchical effects) on hierdat,
# both with Tipton CR2 small-sample correction + Satterthwaite df.
# Dumps the fixture INPUT columns AND the robu() outputs (b, SE, df, CI, prob, tau.sq)
# so Python reads exactly the same inputs.
suppressMessages({library(robumeta); library(jsonlite)})
out <- list()

# ---- CORR model: corrdat, rho=0.8, small=TRUE ----
data(corrdat)
mc <- robu(effectsize ~ males + college + binge, data = corrdat,
           studynum = studyid, var.eff.size = var,
           modelweights = "CORR", rho = 0.8, small = TRUE)
rt <- mc$reg_table
out$corr <- list(
  data = list(effectsize = corrdat$effectsize, var = corrdat$var,
              studyid = corrdat$studyid, males = corrdat$males,
              college = corrdat$college, binge = corrdat$binge),
  rho = 0.8,
  labels = as.character(rt$labels),
  b = as.numeric(rt$b.r), SE = as.numeric(rt$SE), t = as.numeric(rt$t),
  dfs = as.numeric(rt$dfs), prob = as.numeric(rt$prob),
  CI.L = as.numeric(rt$CI.L), CI.U = as.numeric(rt$CI.U),
  tau.sq = as.numeric(mc$mod_info$tau.sq), I.2 = as.numeric(mc$mod_info$I.2),
  N = mc$N, M = mc$M, p = mc$p)

# ---- HIER model: hierdat, small=TRUE ----
data(hierdat)
mh <- robu(effectsize ~ binge + sreport + males + age + followup, data = hierdat,
           studynum = studyid, var.eff.size = var,
           modelweights = "HIER", small = TRUE)
rt2 <- mh$reg_table
out$hier <- list(
  data = list(effectsize = hierdat$effectsize, var = hierdat$var,
              studyid = hierdat$studyid, binge = hierdat$binge,
              sreport = hierdat$sreport, males = hierdat$males,
              age = hierdat$age, followup = hierdat$followup),
  labels = as.character(rt2$labels),
  b = as.numeric(rt2$b.r), SE = as.numeric(rt2$SE), t = as.numeric(rt2$t),
  dfs = as.numeric(rt2$dfs), prob = as.numeric(rt2$prob),
  CI.L = as.numeric(rt2$CI.L), CI.U = as.numeric(rt2$CI.U),
  tau.sq = as.numeric(mh$mod_info$tau.sq), omega.sq = as.numeric(mh$mod_info$omega.sq),
  N = mh$N, M = mh$M, p = mh$p)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pyrobumeta/tests/reference.json")
cat("robumeta", as.character(packageVersion("robumeta")),
    "-> reference.json (corr SE[1]=", round(out$corr$SE[1], 6),
    ", hier SE[1]=", round(out$hier$SE[1], 6), ")\n")
