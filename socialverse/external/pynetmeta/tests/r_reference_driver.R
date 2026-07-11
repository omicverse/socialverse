# netmeta reference driver -- canonical Senn2013 pairwise fixture ->
# graph-theoretical NMA (fixed + DL random) -> pooled TE/seTE matrices, Q.
# Dumps the fixture INPUT so Python reads the SAME data.
suppressMessages({library(netmeta); library(jsonlite)})
data(Senn2013)
d <- Senn2013

net <- netmeta(TE, seTE, treat1, treat2, studlab, data = d,
               reference.group = "plac")

trts <- net$trts  # sorted treatment names (matrix row/col order)

out <- list(
  data = list(
    TE      = as.numeric(d$TE),
    seTE    = as.numeric(d$seTE),
    treat1  = as.character(d$treat1),
    treat2  = as.character(d$treat2),
    studlab = as.character(d$studlab)
  ),
  trts = as.character(trts),
  # full pooled matrices (row/col ordered by trts)
  TE_fixed    = matrix(as.numeric(net$TE.fixed),    nrow = length(trts)),
  seTE_fixed  = matrix(as.numeric(net$seTE.fixed),  nrow = length(trts)),
  TE_random   = matrix(as.numeric(net$TE.random),   nrow = length(trts)),
  seTE_random = matrix(as.numeric(net$seTE.random), nrow = length(trts)),
  # heterogeneity
  Q      = as.numeric(net$Q),
  df_Q   = as.numeric(net$df.Q),
  pval_Q = as.numeric(net$pval.Q),
  tau2   = as.numeric(net$tau2),
  tau    = as.numeric(net$tau)
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pynetmeta/tests/reference.json")
cat("netmeta", as.character(packageVersion("netmeta")),
    "-> reference.json (Q=", round(net$Q, 6),
    ", tau2=", round(net$tau2, 6),
    ", metf-vs-plac fixed TE=", round(net$TE.fixed["metf", "plac"], 6), ")\n")
