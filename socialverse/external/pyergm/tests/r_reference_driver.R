# ergm reference driver — canonical Padgett Florentine marriage network
# (ergm::flomarriage, undirected, 16 nodes) → dyad-INDEPENDENT MPLE model
#   ergm(flomarriage ~ edges + nodecov("wealth"), estimate="MPLE")
# MPLE = logistic regression on dyads (change statistics as predictors).
# Dump MPLE coef + SE AND the fixture inputs (adjacency matrix + wealth
# covariate) so the Python parity test reconstructs the SAME model matrix.
#
# ALSO (extension): observed network sufficient statistics via
#   summary(net ~ <terms>)  [ergm]  on flomarriage (undirected) and a fixed
#   directed 5-node network, plus the Holland-Leinhardt 16-type directed
#   triad census via sna::triad.census on that directed network.
suppressMessages({library(ergm); library(sna); library(jsonlite)})
data(florentine)
f <- flomarriage

adj    <- as.matrix.network(f, matrix.type = "adjacency")  # 16x16 0/1 symmetric
wealth <- as.numeric(f %v% "wealth")
priorates <- as.numeric(f %v% "priorates")
n      <- network.size(f)

m  <- ergm(flomarriage ~ edges + nodecov("wealth"), estimate = "MPLE")
co <- as.numeric(coef(m))
se <- as.numeric(sqrt(diag(vcov(m))))

# --- observed sufficient statistics (undirected: flomarriage) ------------- #
s_undir <- summary(
  f ~ edges + triangle + degree(0:6) + kstar(2) +
    nodecov("wealth") + nodematch("priorates")
)
undir_terms <- names(s_undir)
undir_stats <- as.numeric(s_undir)

# --- fixed directed 5-node network ---------------------------------------- #
Ad <- matrix(c(
  0, 1, 0, 1, 0,
  0, 0, 1, 0, 1,
  1, 0, 0, 1, 0,
  0, 0, 1, 0, 1,
  1, 0, 0, 0, 0), 5, 5, byrow = TRUE)
dnet <- network(Ad, directed = TRUE)

s_dir <- summary(
  dnet ~ edges + mutual + istar(2) + ostar(2) +
    idegree(1:2) + odegree(1:2)
)
dir_terms <- names(s_dir)
dir_stats <- as.numeric(s_dir)

# --- Holland-Leinhardt directed triad census (sna) ------------------------ #
tc <- sna::triad.census(dnet)          # 1x16, columns 003..300
tc_labels <- colnames(tc)
tc_counts <- as.numeric(tc[1, ])

out <- list(
  data = list(
    adjacency = adj,          # row-major 16x16 (jsonlite -> list of rows)
    wealth    = wealth,
    priorates = priorates,
    n         = n,
    directed  = is.directed(f),
    dir_adjacency = Ad        # 5x5 directed 0/1 fixture
  ),
  mple = list(
    terms = c("edges", "nodecov.wealth"),
    coef  = co,
    se    = se
  ),
  summary_undirected = list(
    terms = undir_terms,
    stats = undir_stats
  ),
  summary_directed = list(
    terms = dir_terms,
    stats = dir_stats
  ),
  triad_census = list(
    labels = tc_labels,
    counts = tc_counts
  )
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pyergm/tests/reference.json")
cat("ergm", as.character(packageVersion("ergm")),
    "-> reference.json (MPLE coef =", round(co, 6),
    ", se =", round(se, 6), ")\n")
