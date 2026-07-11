# ergm reference driver — canonical Padgett Florentine marriage network
# (ergm::flomarriage, undirected, 16 nodes) → dyad-INDEPENDENT MPLE model
#   ergm(flomarriage ~ edges + nodecov("wealth"), estimate="MPLE")
# MPLE = logistic regression on dyads (change statistics as predictors).
# Dump MPLE coef + SE AND the fixture inputs (adjacency matrix + wealth
# covariate) so the Python parity test reconstructs the SAME model matrix.
suppressMessages({library(ergm); library(jsonlite)})
data(florentine)
f <- flomarriage

adj    <- as.matrix.network(f, matrix.type = "adjacency")  # 16x16 0/1 symmetric
wealth <- as.numeric(f %v% "wealth")
n      <- network.size(f)

m  <- ergm(flomarriage ~ edges + nodecov("wealth"), estimate = "MPLE")
co <- as.numeric(coef(m))
se <- as.numeric(sqrt(diag(vcov(m))))

out <- list(
  data = list(
    adjacency = adj,          # row-major 16x16 (jsonlite -> list of rows)
    wealth    = wealth,
    n         = n,
    directed  = is.directed(f)
  ),
  mple = list(
    terms = c("edges", "nodecov.wealth"),
    coef  = co,
    se    = se
  )
)

write(toJSON(out, auto_unbox = TRUE, digits = 15, pretty = TRUE),
      "pyergm/tests/reference.json")
cat("ergm", as.character(packageVersion("ergm")),
    "-> reference.json (MPLE coef =", round(co, 6),
    ", se =", round(se, 6), ")\n")
