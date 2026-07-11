# survey reference driver — canonical apistrat (stratified) + apiclus1 (one-stage
# cluster) designs → Taylor-linearization estimates/SE/df → JSON spec.
suppressMessages({library(survey); library(jsonlite)})
data(api)
out <- list()

# --- stratified design (strata=stype, weights=pw, fpc=fpc) ---
ds <- svydesign(id=~1, strata=~stype, weights=~pw, fpc=~fpc, data=apistrat)
m  <- svymean(~api00, ds); t <- svytotal(~api00, ds)
g  <- svyglm(api00 ~ ell + meals, design=ds)
# --- domain (subpopulation) means per stype via svyby ---
by <- svyby(~api00, ~stype, ds, svymean)
by <- by[order(as.character(by$stype)), ]       # sort by level for stable order
# --- ratio api.stu / enroll (Taylor SE) ---
rr <- svyratio(~api.stu, ~enroll, ds)
# --- logit CI for a proportion: P(api00 > 700) ---
cp <- svyciprop(~I(api00>700), ds, method="logit", level=0.95)
out$apistrat <- list(
  data=list(api00=apistrat$api00, ell=apistrat$ell, meals=apistrat$meals,
            stype=as.character(apistrat$stype), pw=apistrat$pw, fpc=apistrat$fpc,
            api.stu=apistrat$api.stu, enroll=apistrat$enroll),
  svymean=list(est=as.numeric(coef(m)), se=as.numeric(SE(m)), df=degf(ds)),
  svytotal=list(est=as.numeric(coef(t)), se=as.numeric(SE(t))),
  svyglm=list(coef=as.numeric(coef(g)), se=as.numeric(SE(g)), df=df.residual(g)),
  svyby=list(levels=as.character(by$stype), est=as.numeric(by$api00),
             se=as.numeric(SE(by)), df=degf(ds)),
  svyratio=list(est=as.numeric(rr$ratio), se=as.numeric(SE(rr)), df=degf(ds)),
  svyciprop=list(est=as.numeric(cp), var=as.numeric(attr(cp,"var")),
                 ci=as.numeric(attr(cp,"ci")), df=degf(ds)))

# --- one-stage cluster design (id=dnum, weights=pw, fpc=fpc) ---
dc <- svydesign(id=~dnum, weights=~pw, fpc=~fpc, data=apiclus1)
mc <- svymean(~api00, dc); gc <- svyglm(api00 ~ ell, design=dc)
out$apiclus1 <- list(
  data=list(api00=apiclus1$api00, ell=apiclus1$ell, dnum=apiclus1$dnum,
            pw=apiclus1$pw, fpc=apiclus1$fpc),
  svymean=list(est=as.numeric(coef(mc)), se=as.numeric(SE(mc)), df=degf(dc)),
  svyglm=list(coef=as.numeric(coef(gc)), se=as.numeric(SE(gc)), df=df.residual(gc)))

write(toJSON(out, auto_unbox=TRUE, digits=15, pretty=TRUE), "pysurvey/tests/reference.json")
cat("survey", as.character(packageVersion("survey")),
    "-> reference.json (strat mean SE=", round(SE(m),6), ", clus mean SE=", round(SE(mc),6), ")\n")
