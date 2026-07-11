"""pynetmeta -- pure-numpy/scipy reconstruction of R netmeta (frequentist
graph-theoretical / GLS network meta-analysis, Ruecker 2012).

Faithful port of netmeta::netmeta -> internal prepare() + nma_ruecker():

  * B      edge-vertex incidence matrix (row per pairwise comparison, +1/-1)
  * W      block-diagonal weight matrix; multi-arm studies get the exact
           Ruecker weight adjustment (netmeta::multiarm) so correlated arms
           are handled; two-arm rows are simply 1/seTE^2 (or 1/(seTE^2+tau^2)).
  * L      = B' W B         (weighted Laplacian)
  * Lplus  = invmat(L) = solve(L - J/n) + J/n   (Laplacian pseudo-inverse)
  * H      = B Lplus B' W   (hat matrix); v = H TE  (fitted comparisons)
  * Q      = (TE-v)' W (TE-v)   ;  df = 2*sum(1/narms) - (n-1)
  * R[i,j] = Lplus[i,i]+Lplus[j,j]-2 Lplus[i,j] ; seTE.pooled = sqrt(R)
  * pooled TE matrix built by consistency propagation from fitted edges v.

Fixed ("common") effect: tau = 0.  Random effect (DerSimonian-Laird, default
method.tau="DL"): tau2 estimated inside the common pass, then weights rebuilt
as 1/(seTE^2 + tau^2) with multi-arm re-adjustment and the whole machinery
re-run.  This mirrors netmeta exactly.
"""
import numpy as np
from scipy.stats import norm, chi2

_EPS = np.finfo(float).eps


def _is_zero(x, n=10):
    return np.abs(x) < n * _EPS


def _createB_full(ncol):
    """Full incidence matrix over all choose(ncol,2) comparisons (i<j)."""
    rows = []
    for i in range(ncol - 1):
        for j in range(i + 1, ncol):
            r = np.zeros(ncol)
            r[i] = 1.0
            r[j] = -1.0
            rows.append(r)
    return np.array(rows) if rows else np.zeros((0, ncol))


def _createB(pos1, pos2, ncol):
    """Incidence matrix, one row per observed comparison (0-based positions)."""
    nrow = len(pos1)
    B = np.zeros((nrow, ncol))
    for i in range(nrow):
        B[i, pos1[i]] = 1.0
        B[i, pos2[i]] = -1.0
    return B


def _invmat(X):
    """netmeta::invmat -- solve(X - J/n) + J/n (pseudo-inverse of a Laplacian)."""
    n = X.shape[0]
    J = np.ones((n, n))
    return np.linalg.solve(X - J / n, np.eye(n)) + J / n


def _multiarm(r, func_inverse=_invmat):
    """netmeta::multiarm -- returns the adjusted variances v for the m rows of
    a multi-arm study given the m raw variances r (m = choose(k,2))."""
    r = np.asarray(r, float)
    m = len(r)
    k = int(round((1 + np.sqrt(8 * m + 1)) / 2))
    B = _createB_full(k)
    Dr = np.diag(r)
    BtDrB = B.T @ Dr @ B
    R = np.diag(np.diag(BtDrB)) - BtDrB
    BtB = B.T @ B
    Lt = -0.5 * BtB @ R @ BtB / (k ** 2)
    L = func_inverse(Lt)
    W = np.diag(np.diag(L)) - L
    # zero-out tiny negative off-diagonals (netmeta guard)
    lower = W[np.tril_indices(k, -1)]
    denom = np.sum(np.abs(lower))
    if denom > 0:
        mask = (W < 0) & (np.abs(W) / denom < 0.001)
        W = np.where(mask, 0.0, W)
    with np.errstate(divide="ignore"):
        V = 1.0 / W
    v = np.zeros(m)
    edge = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            v[edge] = V[i, j]
            edge += 1
    return v


def _covar_study(v, func_inverse=_invmat):
    """netmeta::covar_study for correlated=FALSE (default): returns the block
    weight matrix W (diagonal) and adjusted variances for one study."""
    v = np.asarray(v, float)
    m = len(v)
    n = int(round((1 + np.sqrt(8 * m + 1)) / 2))  # number of arms
    if m > 1:
        v = _multiarm(v, func_inverse)
    W = np.diag(1.0 / v)
    return v, n, W


def _prepare(TE, seTE, treat1, treat2, studlab, tau=0.0, func_inverse=_invmat):
    """netmeta::prepare -- order rows, map treatments to positions, build the
    block-diagonal weight matrix W (with multi-arm adjustment) and adjusted
    per-row variances.  Returns ordered arrays + W + positions."""
    TE = np.asarray(TE, float)
    seTE = np.asarray(seTE, float)
    treat1 = np.asarray(treat1, dtype=object)
    treat2 = np.asarray(treat2, dtype=object)
    studlab = np.asarray(studlab, dtype=object)
    m = len(TE)
    if tau is None or np.isnan(tau):
        tau = 0.0
    weights = 1.0 / (seTE ** 2 + tau ** 2)

    # order by (studlab, treat1, treat2) -- lexicographic, stable
    keys = list(zip(studlab, treat1, treat2))
    o = sorted(range(m), key=lambda i: keys[i])
    o = np.array(o)
    TE, seTE = TE[o], seTE[o]
    treat1, treat2, studlab = treat1[o], treat2[o], studlab[o]
    weights = weights[o]

    names_treat = sorted(set(list(treat1) + list(treat2)))
    pos = {t: k for k, t in enumerate(names_treat)}
    treat1_pos = np.array([pos[t] for t in treat1])
    treat2_pos = np.array([pos[t] for t in treat2])

    # per-study block weight matrices
    sl = list(dict.fromkeys(studlab))  # unique preserving order
    W = np.zeros((m, m))
    narms = np.zeros(m)
    weights_adj = weights.copy().astype(float)
    for s in sl:
        sel = np.where(studlab == s)[0]
        v_s = 1.0 / weights[sel]  # raw variances (already include tau^2)
        v_adj, n_arms, W_s = _covar_study(v_s, func_inverse)
        W[np.ix_(sel, sel)] = W_s
        narms[sel] = n_arms
        weights_adj[sel] = np.diag(W_s)

    return {
        "TE": TE, "seTE": seTE, "treat1": treat1, "treat2": treat2,
        "studlab": studlab, "treat1_pos": treat1_pos, "treat2_pos": treat2_pos,
        "weights": weights_adj, "narms": narms, "W": W,
        "names_treat": names_treat, "n": len(names_treat), "order": o,
    }


def _nma_ruecker(TE, W, seTE, treat1_pos, treat2_pos, narms, studlab, n,
                 names_treat, level=0.95):
    """netmeta::nma_ruecker core -- pooled treatment estimates, SEs, Q, tau2 (DL).
    Returns matrices indexed by names_treat plus Q/df/tau2 scalars."""
    TE = np.asarray(TE, float)
    seTE = np.asarray(seTE, float)
    m = len(TE)
    w_pooled = 1.0 / seTE ** 2
    df1 = 2.0 * np.sum(1.0 / narms)

    B = _createB(treat1_pos, treat2_pos, n)
    B_full = _createB_full(n)
    M = B.T @ B
    D = np.diag(np.diag(M))
    A = D - M
    L = B.T @ W @ B
    Lplus = _invmat(L)
    Lplus[_is_zero(Lplus)] = 0.0

    # R[i,j] = Lplus[i,i] + Lplus[j,j] - 2 Lplus[i,j]
    dL = np.diag(Lplus)
    R = dL[:, None] + dL[None, :] - 2.0 * Lplus

    V = np.array([R[treat1_pos[i], treat2_pos[i]] for i in range(m)])

    G = B @ Lplus @ B.T
    H = G @ W
    H[_is_zero(H)] = 0.0

    v = (H @ TE).ravel()

    # consistency propagation to fill the full treatment x treatment matrix
    all_mat = np.full((n, n), np.nan)
    for i in range(m):
        all_mat[treat1_pos[i], treat2_pos[i]] = v[i]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if not np.isnan(all_mat[i, k]) and not np.isnan(all_mat[j, k]):
                    all_mat[i, j] = all_mat[i, k] - all_mat[j, k]
                    all_mat[j, i] = all_mat[j, k] - all_mat[i, k]
                if not np.isnan(all_mat[i, j]) and not np.isnan(all_mat[k, j]):
                    all_mat[i, k] = all_mat[i, j] - all_mat[k, j]
                    all_mat[k, i] = all_mat[k, j] - all_mat[i, j]
                if not np.isnan(all_mat[i, k]) and not np.isnan(all_mat[i, j]):
                    all_mat[j, k] = all_mat[i, k] - all_mat[i, j]
                    all_mat[k, j] = all_mat[i, j] - all_mat[i, k]

    resid = TE - v
    Q = float(resid @ W @ resid)
    df = df1 - (n - 1)
    pval_Q = np.nan if df == 0 else float(chi2.sf(Q, df))

    # DL tau2
    I = np.eye(m)
    E = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            E[i, j] = 1.0 if studlab[i] == studlab[j] else 0.0
    if df == 0:
        tau2 = np.nan
        tau = np.nan
    else:
        denom = np.trace((I - H) @ (B @ B.T * E / 2.0) @ W)
        tau2 = max(0.0, (Q - df) / denom)
        tau = np.sqrt(tau2)

    seTE_pooled = np.sqrt(R)
    TE_pooled = all_mat

    return {
        "TE_pooled": TE_pooled, "seTE_pooled": seTE_pooled,
        "Q": Q, "df": df, "pval_Q": pval_Q, "tau2": tau2, "tau": tau,
        "Lplus": Lplus, "H": H, "v": v, "A": A, "names_treat": names_treat,
    }


class NetMeta:
    """Result container mirroring the netmeta object (subset)."""

    def __init__(self, d):
        self.__dict__.update(d)

    def _mat(self, M, name):
        idx = {t: k for k, t in enumerate(self.trts)}
        return M

    def comparison(self, treat, reference, random=False):
        """Return (TE, seTE) of treat vs reference for fixed or random effect."""
        i = self.trts.index(treat)
        j = self.trts.index(reference)
        TE = self.TE_random if random else self.TE_fixed
        seTE = self.seTE_random if random else self.seTE_fixed
        return TE[i, j], seTE[i, j]


def netmeta(TE, seTE, treat1, treat2, studlab, reference_group=None,
            level=0.95, method_tau="DL"):
    """Frequentist graph-theoretical network meta-analysis (netmeta::netmeta).

    Parameters mirror R: TE/seTE per pairwise comparison, treat1/treat2 labels,
    studlab study labels.  Default DerSimonian-Laird random-effects tau.

    Returns a NetMeta object exposing (indexed by sorted treatment names in
    ``.trts``): TE_fixed/seTE_fixed and TE_random/seTE_random treatment-vs-
    treatment matrices, plus Q/df/pval_Q/tau2/tau heterogeneity statistics.
    Column ``reference_group`` gives estimates vs the reference treatment.
    """
    if method_tau != "DL":
        raise NotImplementedError("only DL (DerSimonian-Laird) is ported")

    # --- common (fixed) pass: tau = 0 ---
    p0 = _prepare(TE, seTE, treat1, treat2, studlab, tau=0.0)
    res_c = _nma_ruecker(
        p0["TE"], p0["W"], np.sqrt(1.0 / p0["weights"]),
        p0["treat1_pos"], p0["treat2_pos"], p0["narms"], p0["studlab"],
        p0["n"], p0["names_treat"], level=level,
    )

    # --- DL tau from common pass, rebuild weights for random pass ---
    tau = res_c["tau"]
    p1 = _prepare(TE, seTE, treat1, treat2, studlab, tau=tau)
    res_r = _nma_ruecker(
        p1["TE"], p1["W"], np.sqrt(1.0 / p1["weights"]),
        p1["treat1_pos"], p1["treat2_pos"], p1["narms"], p1["studlab"],
        p1["n"], p1["names_treat"], level=level,
    )

    trts = p0["names_treat"]
    z = norm.ppf(0.5 + level / 2.0)
    out = {
        "trts": trts, "n": p0["n"],
        "TE_fixed": res_c["TE_pooled"], "seTE_fixed": res_c["seTE_pooled"],
        "TE_random": res_r["TE_pooled"], "seTE_random": res_r["seTE_pooled"],
        "Q": res_c["Q"], "df_Q": res_c["df"], "pval_Q": res_c["pval_Q"],
        "tau2": res_c["tau2"], "tau": res_c["tau"],
        "reference_group": reference_group, "level": level, "z": z,
    }
    return NetMeta(out)
