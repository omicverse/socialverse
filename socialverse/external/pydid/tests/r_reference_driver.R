# did reference driver -- Callaway & Sant'Anna staggered DID on the canonical
# mpdta min-wage county panel. att_gt(never-treated control, reg outcome method)
# + aggte(simple, dynamic). Dumps the fixture INPUT panel AND the R point
# estimates + bootstrap SE into reference.json so Python reads identical inputs.
suppressMessages({library(did); library(jsonlite)})
data(mpdta)

res <- att_gt(yname="lemp", tname="year", idname="countyreal", gname="first.treat",
              control_group="nevertreated", est_method="reg", data=mpdta,
              bstrap=TRUE, cband=FALSE, biters=1000)
simple  <- aggte(res, type="simple",  na.rm=TRUE)
dynamic <- aggte(res, type="dynamic", na.rm=TRUE)

out <- list(
  data = list(
    year        = as.integer(mpdta$year),
    countyreal  = as.numeric(mpdta$countyreal),
    lemp        = as.numeric(mpdta$lemp),
    first.treat = as.numeric(mpdta$first.treat)
  ),
  att_gt = list(
    group = as.numeric(res$group),
    t     = as.numeric(res$t),
    att   = as.numeric(res$att),
    se    = as.numeric(res$se)          # bootstrap SE -- stochastic, documented
  ),
  simple = list(
    overall.att = as.numeric(simple$overall.att),
    overall.se  = as.numeric(simple$overall.se)
  ),
  dynamic = list(
    egt         = as.numeric(dynamic$egt),
    att.egt     = as.numeric(dynamic$att.egt),
    se.egt      = as.numeric(dynamic$se.egt),
    overall.att = as.numeric(dynamic$overall.att),
    overall.se  = as.numeric(dynamic$overall.se)
  )
)

write(toJSON(out, auto_unbox=TRUE, digits=15, pretty=TRUE), "pydid/tests/reference.json")
cat("did", as.character(packageVersion("did")),
    "-> reference.json (n gt =", length(res$att),
    ", simple ATT =", round(simple$overall.att, 6),
    ", dynamic ATT =", round(dynamic$overall.att, 6), ")\n")
