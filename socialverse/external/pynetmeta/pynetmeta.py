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


# ---------------------------------------------------------------------------
# Krahn (2013) design-based decomposition (netmeta:::nma_krahn) + network
# measures (netmeta::netmeasures): proportion of direct evidence, mean path
# length, minimal parallelism.
# ---------------------------------------------------------------------------

def _iv_common(TE, seTE):
    """Inverse-variance fixed ("common") effect combination (metagen DL common)."""
    TE = np.asarray(TE, float)
    seTE = np.asarray(seTE, float)
    w = 1.0 / seTE ** 2
    te = float(np.sum(w * TE) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return te, se


def _fX(n, trts, sep=":"):
    """netmeta::fX -- full design matrix over all choose(n,2) comparisons
    trts[i]:trts[j] (i<j), columns = basis contrasts trts[0]:trts[k] (k=1..n-1).

    Row for trts[a]:trts[b] (a<b): +1 at column (a-1) when a>=1, -1 at column
    (b-1).  Row/col names mirror R for label-based indexing."""
    possK = n * (n - 1) // 2
    X = np.zeros((possK, n - 1))
    rn = []
    r = 0
    for a in range(n - 1):
        for b in range(a + 1, n):
            if a >= 1:
                X[r, a - 1] = 1.0
            X[r, b - 1] = -1.0
            rn.append(f"{trts[a]}{sep}{trts[b]}")
            r += 1
    cn = [f"{trts[0]}{sep}{trts[k]}" for k in range(1, n)]
    return X, rn, cn


def _orient(t1, t2, te, order, sep=":"):
    """Orient a comparison to canonical trts-index order (smaller index first),
    flipping TE if swapped.  ``order`` maps treatment label -> trts index.
    Because the reference is trts[0], this subsumes the reference-first flip
    that netmeta::nma_krahn performs (any plac comparison => plac first)."""
    if order[t1] > order[t2]:
        t1, t2, te = t2, t1, -te
    return t1, t2, te, f"{t1}{sep}{t2}"


def _nma_krahn(net, sep=":", tau_preset=0.0):
    """Port of netmeta:::nma_krahn for reference.group == "" (design/Krahn
    decomposition of a fitted NetMeta).  Handles two-arm and multi-arm studies.

    ``tau_preset`` inflates every seTE to sqrt(seTE^2 + tau_preset^2) (the
    random-effects branch used by netmeasures(random=TRUE)).

    Returns dict with H, H_studies (+ their row/col labels), direct (per
    observed comparison IV-common TE/seTE) and network (design GLS TE/seTE)."""
    trts = list(net.trts)
    n = len(trts)
    order = {t: i for i, t in enumerate(trts)}  # trts index (plac == 0)

    # --- per-row study data, oriented reference-first, ordered by studlab ---
    studlab = list(net._studlab)
    stt1 = list(net._treat1)
    stt2 = list(net._treat2)
    stTE = [-t for t in net._TE]        # nma_krahn uses TE = -x$TE
    tp = 0.0 if tau_preset is None else float(tau_preset)
    stse = [float(np.sqrt(s ** 2 + tp ** 2)) for s in net._seTE]
    narms_map = {s: net._narms_by_study[s] for s in net._narms_by_study}

    rows = []
    for i in range(len(studlab)):
        t1, t2, te, comp = _orient(stt1[i], stt2[i], stTE[i], order, sep)
        rows.append({"studlab": studlab[i], "treat1": t1, "treat2": t2,
                     "TE": te, "seTE": stse[i], "comparison": comp,
                     "narms": narms_map[studlab[i]]})
    rows.sort(key=lambda r: r["studlab"])

    selmulti = {i for i, r in enumerate(rows) if r["narms"] > 2}
    twoarm_rows = [r for i, r in enumerate(rows) if i not in selmulti]
    multi_rows = [r for i, r in enumerate(rows) if i in selmulti]

    # --- direct evidence per comparison (all studies + 2-arm-only) ---
    comps_all = sorted({r["comparison"] for r in rows})
    direct = {}
    for c in comps_all:
        te_i = [r["TE"] for r in rows if r["comparison"] == c]
        se_i = [r["seTE"] for r in rows if r["comparison"] == c]
        te, se = _iv_common(te_i, se_i)
        # 2-arm-only sub-combination (TE.2arm/seTE.2arm/n.2arm)
        te2 = [r["TE"] for r in twoarm_rows if r["comparison"] == c]
        se2 = [r["seTE"] for r in twoarm_rows if r["comparison"] == c]
        if te2:
            t2, s2 = _iv_common(te2, se2)
            n2 = len(te2)
        else:
            t2 = s2 = n2 = None
        direct[c] = {"TE": te, "seTE": se, "TE.2arm": t2,
                     "seTE.2arm": s2, "n.2arm": n2}

    # comparisons that have >=1 two-arm study (direct2 in R)
    direct2_comps = [c for c in comps_all if direct[c]["n.2arm"] is not None]

    # --- multi-arm designs: covariance block + aggregation across studies ---
    multi_by_study = {}
    for r in multi_rows:
        multi_by_study.setdefault(r["studlab"], []).append(r)

    # per multi-arm study: choose base treat = most frequent treat1, then the
    # (k-1) basis comparisons base:other ordered by other; build (k-1)x(k-1)
    # covariance matrix m.
    def _study_basis(recs):
        # base = treat1 that appears most often as treat1
        from collections import Counter
        cnt = Counter(rr["treat1"] for rr in recs)
        base = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
        basis = [rr for rr in recs if rr["treat1"] == base]
        basis.sort(key=lambda rr: rr["treat2"])
        others = [rr for rr in recs if rr["treat1"] != base]
        k = recs[0]["narms"]
        design = f"{base}" + "".join(sep + rr["treat2"] for rr in basis)
        return base, basis, others, k, design

    multi_designs = {}   # design string -> list of (studlab, basis recs, cov m)
    for s, recs in multi_by_study.items():
        base, basis, others, k, design = _study_basis(recs)
        # arm ordering for covariance: basis rows (base:other) then remaining
        # cross comparisons (others), matching R's multistudies2 layout.
        # cov m: (k-1)x(k-1); diag = seTE[basis]^2 ; off-diag(i,j) =
        # (se_i^2 + se_j^2 - se_cross^2)/2 where se_cross is the other:other row
        ordered = basis + others
        se = [rr["seTE"] for rr in ordered]
        m = np.full((k - 1, k - 1), np.nan)
        idx = 0
        for i in range(k - 2):
            for j in range(i + 1, k - 1):
                val = (se[i] ** 2 + se[j] ** 2 - se[k - 1 + idx] ** 2) / 2.0
                m[i, j] = val
                m[j, i] = val
                idx += 1
        for i in range(k - 1):
            m[i, i] = se[i] ** 2
        multi_designs.setdefault(design, []).append((s, basis, m))

    # V.design (two-arm designs) + aggregated multi-arm blocks (V3.agg)
    V_blocks = []
    V_names = []
    TE_dir = []
    for c in direct2_comps:
        V_blocks.append(np.array([[direct[c]["seTE.2arm"] ** 2]]))
        V_names.append(c)
        TE_dir.append(direct[c]["TE.2arm"])

    multicomp = sorted(multi_designs.keys())
    # aggregate each multi design across its studies (inverse-variance on the
    # (k-1)-dim basis vector); single study => passthrough.
    agg_basis_comps = {}   # design -> list of basis comparison names
    for design in multicomp:
        entries = multi_designs[design]
        dim = entries[0][1].__len__()
        # basis comparison labels (base:other) from first study
        basis_names = [rr["comparison"] for rr in entries[0][1]]
        agg_basis_comps[design] = basis_names
        # sum of precision matrices and precision-weighted TE
        Psum = np.zeros((dim, dim))
        b = np.zeros(dim)
        for (s, basis, m) in entries:
            P = np.linalg.solve(m, np.eye(dim))
            te_vec = np.array([rr["TE"] for rr in basis])
            Psum += P
            b += P @ te_vec
        covs3 = np.linalg.solve(Psum, np.eye(dim))
        te_agg = covs3 @ b
        V_blocks.append(covs3)
        V_names.extend([design] * dim)
        TE_dir.extend(list(te_agg))

    # assemble block-diagonal V and its comparison-row labels
    total = sum(bl.shape[0] for bl in V_blocks)
    V = np.zeros((total, total))
    off = 0
    for bl in V_blocks:
        s = bl.shape[0]
        V[off:off + s, off:off + s] = bl
        off += s
    TE_dir = np.array(TE_dir, float)

    # X.obs rows: two-arm comparison labels then multi-arm basis comparisons.
    # obs_row_labels -> which X.full row (basis comparison); H_col_names -> the
    # column label R gives H (design name for the multi-arm basis rows, so all
    # basis columns of one design share a group when aggregating H.tilde).
    X_full, xf_rn, xf_cn = _fX(n, trts, sep)
    rn_index = {name: i for i, name in enumerate(xf_rn)}
    obs_row_labels = list(direct2_comps)
    H_col_names = list(direct2_comps)
    for design in multicomp:
        obs_row_labels.extend(agg_basis_comps[design])
        H_col_names.extend([design] * len(agg_basis_comps[design]))
    X_obs = np.array([X_full[rn_index[c]] for c in obs_row_labels])

    Vinv = np.linalg.solve(V, np.eye(total))
    core = np.linalg.solve(X_obs.T @ Vinv @ X_obs, np.eye(n - 1))
    H = X_full @ core @ X_obs.T @ Vinv
    TE_net = H @ TE_dir

    # network SEs: covTE.net.base = core; diag then derived cross terms.
    covbase = core
    dcb = np.diag(covbase)
    covTE = list(dcb)
    for i in range(n - 2):
        for j in range(1, n - 1):
            if i < j:
                covTE.append(dcb[i] + dcb[j] - 2.0 * covbase[i, j])
    covTE = np.array(covTE)
    network = {xf_rn[i]: {"TE": float(TE_net[i]), "seTE": float(np.sqrt(covTE[i]))}
               for i in range(len(xf_rn))}

    # --- H.studies: per-study rows, V.studies per-study block-diagonal ---
    Vs_blocks = []
    studlabs_col = []
    comps_studies = []
    for r in twoarm_rows:
        Vs_blocks.append(np.array([[r["seTE"] ** 2]]))
        studlabs_col.append(r["studlab"])
        comps_studies.append(r["comparison"])
    for s, recs in multi_by_study.items():
        base, basis, others, k, design = _study_basis(recs)
        ordered = basis + others
        se = [rr["seTE"] for rr in ordered]
        m = np.full((k - 1, k - 1), np.nan)
        idx = 0
        for i in range(k - 2):
            for j in range(i + 1, k - 1):
                val = (se[i] ** 2 + se[j] ** 2 - se[k - 1 + idx] ** 2) / 2.0
                m[i, j] = val
                m[j, i] = val
                idx += 1
        for i in range(k - 1):
            m[i, i] = se[i] ** 2
        Vs_blocks.append(m)
        for rr in basis:
            studlabs_col.append(s)
            comps_studies.append(rr["comparison"])

    tots = sum(bl.shape[0] for bl in Vs_blocks)
    Vs = np.zeros((tots, tots))
    off = 0
    for bl in Vs_blocks:
        s = bl.shape[0]
        Vs[off:off + s, off:off + s] = bl
        off += s
    X_obs_studies = np.array([X_full[rn_index[c]] for c in comps_studies])
    Vsinv = np.linalg.solve(Vs, np.eye(tots))
    cores = np.linalg.solve(X_obs_studies.T @ Vsinv @ X_obs_studies, np.eye(n - 1))
    H_studies = X_full @ cores @ X_obs_studies.T @ Vsinv

    return {
        "trts": trts, "n": n,
        "H": H, "H_rn": xf_rn, "H_cn": H_col_names,
        "H_studies": H_studies, "Hs_cn": studlabs_col,
        "direct": direct, "network": network,
        "comparisons": comps_all,
    }


def netmeasures(net, random=False, tau_preset=None, sep=":"):
    """netmeta::netmeasures -- frequentist graph-model network measures.

    Per-comparison measures derived from the Krahn hat matrix H:

      * ``proportion`` -- proportion of direct evidence for each observed
        comparison, network$seTE^2 / direct$seTE^2 (0 for indirect-only).
      * ``meanpath``   -- mean path length = row sums of H.tilde
        (H.tilde[r,d] = 0.5*(|sum H_rd| + sum|H_rd|) aggregated per design d).
      * ``minpar``     -- minimal parallelism = 1 / max_d |H.tilde[r,d]|.
      * ``minpar_study`` -- same, computed from H aggregated per study.

    ``random=False`` (default) reproduces the common/fixed-effect measures.
    ``random=True`` (or an explicit ``tau_preset``) adds tau^2 to every seTE,
    matching netmeasures(random=TRUE); when random and no tau_preset is given,
    ``net.tau`` (the fitted DL tau) is used, exactly like netmeta.

    Returns a dict of {comparison_label: value}, keyed/ordered by the full
    comparison list (rownames of H)."""
    if tau_preset is None:
        tau = float(getattr(net, "tau", 0.0)) if random else 0.0
    else:
        tau = float(tau_preset)
        random = True
    k = _nma_krahn(net, sep, tau_preset=tau)
    H, rn, cn = k["H"], k["H_rn"], k["H_cn"]
    Hs, cns = k["H_studies"], k["Hs_cn"]
    direct, network = k["direct"], k["network"]
    comps = k["comparisons"]

    # proportion of direct evidence
    proportion = {c: 0.0 for c in rn}
    for c in comps:
        if c in network and c in direct:
            proportion[c] = network[c]["seTE"] ** 2 / direct[c]["seTE"] ** 2

    def _htilde(Hmat, colnames):
        # group columns by name, per group g compute for each row:
        # 0.5*(|sum_over_g|+sum_over_g|.|); return {rowlabel: {group: val}}
        groups = {}
        for j, g in enumerate(colnames):
            groups.setdefault(g, []).append(j)
        out = np.zeros((Hmat.shape[0], len(groups)))
        gnames = list(dict.fromkeys(colnames))
        for gi, g in enumerate(gnames):
            sub = Hmat[:, groups[g]]
            out[:, gi] = 0.5 * (np.abs(sub.sum(axis=1)) + np.abs(sub).sum(axis=1))
        return out

    Ht = _htilde(H, cn)
    meanpath = {rn[i]: float(Ht[i].sum()) for i in range(len(rn))}
    minpar = {rn[i]: float(1.0 / np.max(np.abs(Ht[i]))) for i in range(len(rn))}

    Hts = _htilde(Hs, cns)
    minpar_study = {rn[i]: float(1.0 / np.max(np.abs(Hts[i]))) for i in range(len(rn))}

    return {"proportion": proportion, "meanpath": meanpath,
            "minpar": minpar, "minpar_study": minpar_study}


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

    # raw (unordered) inputs + per-study arm counts, for the Krahn decomposition
    _studlab = list(np.asarray(studlab, dtype=object))
    _sl_arr = np.asarray(studlab, dtype=object)
    _narms_by_study = {}
    for s in dict.fromkeys(_studlab):
        m_rows = int(np.sum(_sl_arr == s))          # #pairwise rows for study
        _narms_by_study[s] = int(round((1 + np.sqrt(1 + 8 * m_rows)) / 2))

    out = {
        "trts": trts, "n": p0["n"],
        "_studlab": _studlab,
        "_treat1": list(np.asarray(treat1, dtype=object)),
        "_treat2": list(np.asarray(treat2, dtype=object)),
        "_TE": [float(t) for t in np.asarray(TE, float)],
        "_seTE": [float(t) for t in np.asarray(seTE, float)],
        "_narms_by_study": _narms_by_study,
        "TE_fixed": res_c["TE_pooled"], "seTE_fixed": res_c["seTE_pooled"],
        "TE_random": res_r["TE_pooled"], "seTE_random": res_r["seTE_pooled"],
        "Q": res_c["Q"], "df_Q": res_c["df"], "pval_Q": res_c["pval_Q"],
        "tau2": res_c["tau2"], "tau": res_c["tau"],
        "reference_group": reference_group, "level": level, "z": z,
    }
    return NetMeta(out)
