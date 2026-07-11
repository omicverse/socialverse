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

# --- netmeasures: per-comparison network measures (Krahn design model) ---
# proportion of direct evidence, mean path length, minimal parallelism.
# Gate BOTH the common/fixed (random=FALSE, tau=0, fully deterministic) and
# the random-effects (random=TRUE, tau = fitted DL tau) variants.
nmF <- netmeasures(net, random = FALSE)
nmR <- netmeasures(net, random = TRUE)
out$netmeasures <- list(
  labels = names(nmF$proportion),  # comparison labels (rownames of H)
  fixed = list(
    proportion   = as.numeric(nmF$proportion),
    meanpath     = as.numeric(nmF$meanpath),
    minpar       = as.numeric(nmF$minpar),
    minpar_study = as.numeric(nmF$minpar.study)
  ),
  random = list(
    proportion   = as.numeric(nmR$proportion),
    meanpath     = as.numeric(nmR$meanpath),
    minpar       = as.numeric(nmR$minpar),
    minpar_study = as.numeric(nmR$minpar.study)
  )
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pynetmeta/tests/reference.json")
cat("netmeta", as.character(packageVersion("netmeta")),
    "-> reference.json (Q=", round(net$Q, 6),
    ", tau2=", round(net$tau2, 6),
    ", metf-vs-plac fixed TE=", round(net$TE.fixed["metf", "plac"], 6),
    ", netmeasures ncomp=", length(nmF$proportion), ")\n")
