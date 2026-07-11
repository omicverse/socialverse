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
group    <- aggte(res, type="group",    na.rm=TRUE)
calendar <- aggte(res, type="calendar", na.rm=TRUE)

# not-yet-treated control group (point estimates deterministic)
res_nyt <- att_gt(yname="lemp", tname="year", idname="countyreal", gname="first.treat",
                  control_group="notyettreated", est_method="reg", data=mpdta,
                  bstrap=TRUE, cband=FALSE, biters=1000)

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
  ),
  group = list(
    egt         = as.numeric(group$egt),
    att.egt     = as.numeric(group$att.egt),
    overall.att = as.numeric(group$overall.att),
    overall.se  = as.numeric(group$overall.se)
  ),
  calendar = list(
    egt         = as.numeric(calendar$egt),
    att.egt     = as.numeric(calendar$att.egt),
    overall.att = as.numeric(calendar$overall.att),
    overall.se  = as.numeric(calendar$overall.se)
  ),
  att_gt_notyettreated = list(
    group = as.numeric(res_nyt$group),
    t     = as.numeric(res_nyt$t),
    att   = as.numeric(res_nyt$att),
    se    = as.numeric(res_nyt$se)          # bootstrap SE -- stochastic, documented
  )
)

write(toJSON(out, auto_unbox=TRUE, digits=15, pretty=TRUE), "pydid/tests/reference.json")
cat("did", as.character(packageVersion("did")),
    "-> reference.json (n gt =", length(res$att),
    ", simple ATT =", round(simple$overall.att, 6),
    ", dynamic ATT =", round(dynamic$overall.att, 6),
    ", group ATT =", round(group$overall.att, 6),
    ", calendar ATT =", round(calendar$overall.att, 6),
    ", nyt n gt =", length(res_nyt$att), ")\n")
