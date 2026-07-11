# QCA reference driver — canonical Lipset (LF) fuzzy-set fixture.
#   truthTable(SURV ~ DEV,URB,LIT,IND,STB, incl.cut=0.8) -> per-row OUT/n/incl/PRI
#   minimize(tt)  (conservative solution, remainders excluded) -> prime implicants
#   parameters of fit for each term + overall solution (inclS/PRI/covS/covU)
# Also dumps the raw LF fixture so Python reads the SAME inputs. -> tests/reference.json
suppressMessages({library(QCA); library(jsonlite)})
data(LF)
conds <- c("DEV","URB","LIT","IND","STB")
out <- list()

# --- raw fixture inputs (18 cases x 5 conditions + outcome) ---
out$data <- list(
  cases = rownames(LF),
  conditions = conds,
  DEV = LF$DEV, URB = LF$URB, LIT = LF$LIT, IND = LF$IND, STB = LF$STB,
  SURV = LF$SURV,
  incl.cut = 0.8)

# --- truth table ---
tt <- truthTable(LF, outcome="SURV", conditions=paste(conds, collapse=","), incl.cut=0.8)
ttdf <- tt$tt
# emit only observed rows (OUT is 0/1, not "?"); keep row id, condition bits, OUT, n, incl, PRI
obs <- ttdf[ttdf$OUT %in% c(0,1), , drop=FALSE]
out$truthTable <- list(
  rownames = as.integer(rownames(obs)),
  DEV = as.integer(obs$DEV), URB = as.integer(obs$URB), LIT = as.integer(obs$LIT),
  IND = as.integer(obs$IND), STB = as.integer(obs$STB),
  OUT = as.integer(obs$OUT),
  n   = as.integer(obs$n),
  incl = as.numeric(obs$incl),
  PRI  = as.numeric(obs$PRI))

# --- conservative minimization (remainders excluded) ---
mc <- minimize(tt)
terms <- mc$solution[[1]]
ind <- mc$IC$incl.cov      # per-term data.frame: inclS, PRI, covS, covU (rownamed by term)
out$minimize <- list(
  terms = terms,
  inclS = as.numeric(ind$inclS),
  PRI   = as.numeric(ind$PRI),
  covS  = as.numeric(ind$covS),
  covU  = as.numeric(ind$covU))

ov <- mc$IC$sol.incl.cov   # solution-level pof row (inclS, PRI, covS)
out$overall <- list(
  inclS = as.numeric(ov$inclS),
  PRI   = as.numeric(ov$PRI),
  covS  = as.numeric(ov$covS))

# --- calibrate: fuzzy direct (3-anchor logistic) on a canonical numeric vector ---
calx <- c(1,2,3,4,5,6,7,8,9,10)
calth <- c(3, 5.5, 8)   # exclusion, crossover, inclusion anchors
calfs <- calibrate(calx, type="fuzzy", method="direct",
                   thresholds=calth, logistic=TRUE, idm=0.95)
# crisp calibration (findInterval) on the same vector with the same cut-points
calcrisp <- calibrate(calx, type="crisp", thresholds=calth)
out$calibrate <- list(
  x = calx,
  thresholds = calth,
  idm = 0.95,
  fuzzy = as.numeric(calfs),
  crisp = as.integer(calcrisp))

# --- superSubset: necessity superset search on LF ---
ss <- superSubset(LF, outcome="SURV", incl.cut=0.9, cov.cut=0.6)
ssic <- ss$incl.cov
out$superSubset <- list(
  incl.cut = 0.9,
  cov.cut  = 0.6,
  terms = rownames(ssic),
  inclN = as.numeric(ssic$inclN),
  RoN   = as.numeric(ssic$RoN),
  covN  = as.numeric(ssic$covN))

write(toJSON(out, auto_unbox=TRUE, digits=15, pretty=TRUE), "pyqca/tests/reference.json")
cat("QCA", as.character(packageVersion("QCA")), "-> reference.json (terms:",
    paste(terms, collapse=" + "), ")\n")
