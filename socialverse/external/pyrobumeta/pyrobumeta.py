"""pyrobumeta — pure-numpy/scipy reconstruction of R ``robumeta::robu``.

Robust variance estimation (RVE) for dependent effect sizes, supporting the
correlated-effects ("CORR") and hierarchical-effects ("HIER") working models,
with the Tipton (2015) CR2 small-sample bias correction and Satterthwaite
degrees of freedom.

The algorithm mirrors ``robumeta`` 2.1's ``robu()`` element-for-element:

  * study index = 1..N obtained by sorting the *unique* original study ids and
    mapping each row to its rank (R's ``as.numeric(as.factor(studynum))``);
    rows are then grouped (stably) by that index.
  * CORR weights  w = 1 / (k * avg_var);  HIER weights  w = 1 / var.
  * WLS fit b = (X'WX)^-1 X'Wy on the first-stage weights, residuals e.
  * between-study variance component(s) via robumeta's method-of-moments
    (CORR: tau.sq; HIER: tau.sq + omega.sq), truncated at 0.
  * refit with r.weights (the variance-component-adjusted weights) -> b.r.
  * CR2-corrected robust covariance VR.r using the per-study adjustment
    matrices A.MBB, then Satterthwaite dfs from the g_i quadratic forms.

Only ``small=TRUE`` (the default, and the only interesting path) is ported for
the two non-user-weighted working models.
"""
from __future__ import annotations

import numpy as np


def _study_index(studynum):
    """Replicate R: as.numeric(as.factor(studynum)) -> 1..N by sorted unique."""
    studynum = np.asarray(studynum)
    uniq = np.unique(studynum)  # np.unique sorts, matching R factor level order
    # map each value to its 1-based rank among sorted uniques
    idx = np.searchsorted(uniq, studynum) + 1
    return idx.astype(int), len(uniq)


def _design_matrix(covariates):
    """Intercept + numeric covariates (all fixture predictors are numeric)."""
    n = None
    cols = [None]  # placeholder for intercept
    for c in covariates:
        c = np.asarray(c, float)
        n = c.shape[0]
        cols.append(c)
    cols[0] = np.ones(n)
    return np.column_stack(cols)


def robu(effect_size, var_eff_size, studynum, covariates,
         modelweights="CORR", rho=0.8, small=True):
    """Robust variance estimation for dependent effect sizes.

    Parameters
    ----------
    effect_size : (M,) array of effect sizes.
    var_eff_size : (M,) array of effect-size sampling variances.
    studynum : (M,) array of study (cluster) identifiers.
    covariates : sequence of (M,) arrays — the model predictors, in order.
        An intercept column is prepended automatically (matching
        ``effectsize ~ x1 + x2 + ...``).
    modelweights : {"CORR", "HIER"}.
    rho : within-study effect correlation assumption (CORR only), in [0, 1].
    small : use the CR2 small-sample correction + Satterthwaite df (default).

    Returns
    -------
    dict with keys: b, SE, t, dfs, prob, CI_L, CI_U (each length p+1),
    plus tau_sq (and omega_sq for HIER, I2 for CORR), N, M, p.
    """
    from scipy import stats

    modelweights = modelweights.upper()
    if modelweights == "CORR" and (rho > 1 or rho < 0):
        raise ValueError("Rho must be a value between 0 and 1.")

    y_all = np.asarray(effect_size, float)
    v_all = np.asarray(var_eff_size, float)
    study, N = _study_index(studynum)
    Xreg = _design_matrix(covariates)  # (M, p+1)

    # stable sort by study index (R: dframe[order(study),]) --------------------
    order = np.argsort(study, kind="stable")
    study = study[order]
    y_all = y_all[order]
    v_all = v_all[order]
    Xreg = Xreg[order]
    M = y_all.shape[0]
    p = Xreg.shape[1] - 1  # number of covariates excluding intercept

    # per-study bookkeeping ----------------------------------------------------
    uniq_study = np.unique(study)
    groups = [np.where(study == s)[0] for s in uniq_study]
    k_vec = np.array([len(g) for g in groups])          # study sizes
    k_row = np.array([len(groups[np.searchsorted(uniq_study, s)]) for s in study])

    # avg.var.eff.size = ave(var, study)  -> per-row mean of var within study
    avg_var_row = np.empty(M)
    for g in groups:
        avg_var_row[g] = v_all[g].mean()

    # first-stage weights ------------------------------------------------------
    if modelweights == "HIER":
        w_all = 1.0 / v_all
    else:  # CORR
        w_all = 1.0 / (k_row * avg_var_row)

    # WLS: b = (X'WX)^-1 X'Wy  (W diagonal) ------------------------------------
    Xw = Xreg * w_all[:, None]
    sumXWX = Xreg.T @ Xw
    sumXWy = Xreg.T @ (w_all * y_all)
    V_i = np.linalg.solve(sumXWX, np.eye(p + 1))
    b = V_i @ sumXWy
    e_all = y_all - Xreg @ b

    # ---- between-study variance component(s) ---------------------------------
    if modelweights == "HIER":
        tau_sq, omega_sq, I2 = _hier_components(
            Xreg, w_all, e_all, v_all, groups, V_i, p, M)
        r_weights = 1.0 / (v_all + tau_sq + omega_sq)
    else:  # CORR
        tau_sq, I2 = _corr_components(
            Xreg, w_all, e_all, groups, V_i, N, rho)
        omega_sq = None
        r_weights = 1.0 / (k_row * (avg_var_row + tau_sq))

    # ---- refit with r.weights ------------------------------------------------
    Xrw = Xreg * r_weights[:, None]
    sumXWX_r = Xreg.T @ Xrw
    Q = np.linalg.solve(sumXWX_r, np.eye(p + 1))  # (X'WrX)^-1
    b_r = Q @ (Xreg.T @ (r_weights * y_all))
    e_r = y_all - Xreg @ b_r

    if not small:
        # HC0-style with N/(N-p-1) scaling
        meat = np.zeros((p + 1, p + 1))
        for g in groups:
            Xj = Xreg[g]
            Wj = np.diag(r_weights[g])
            Sj = np.outer(e_r[g], e_r[g])
            meat += Xj.T @ Wj @ Sj @ Wj @ Xj
        VR = Q @ meat @ Q
        dfs = np.full(p + 1, N - (p + 1), float)
        SE = np.sqrt(np.diag(VR)) * np.sqrt(N / (N - (p + 1)))
    else:
        VR, dfs = _cr2(Xreg, r_weights, e_r, groups, k_vec, Q, N, M, p,
                       modelweights)
        SE = np.sqrt(np.diag(VR))

    t = b_r / SE
    prob = 2 * (1 - stats.t.cdf(np.abs(t), dfs))
    crit = stats.t.ppf(0.975, dfs)
    CI_L = b_r - crit * SE
    CI_U = b_r + crit * SE

    res = dict(b=b_r, SE=SE, t=t, dfs=dfs, prob=prob, CI_L=CI_L, CI_U=CI_U,
               tau_sq=float(tau_sq), N=int(N), M=int(M), p=int(p))
    if modelweights == "HIER":
        res["omega_sq"] = float(omega_sq)
    else:
        res["I2"] = float(I2)
    return res


# --------------------------------------------------------------------------- #
#  Variance components                                                        #
# --------------------------------------------------------------------------- #
def _corr_components(Xreg, w_all, e_all, groups, V_i, N, rho):
    """CORR method-of-moments tau.sq and I^2 (robumeta)."""
    W = w_all
    sumW = W.sum()
    Qe = float(e_all @ (W * e_all))

    # sumXWJWX = sum_j (X'W J W X)_j ; J = ones(k_j) block
    sumXWJWX = np.zeros_like(V_i)
    Matrx_WKXX = np.zeros_like(V_i)
    Matrx_wk_XJX_XX = np.zeros_like(V_i)
    for g in groups:
        Xj = Xreg[g]
        wj = W[g]
        k = len(g)
        # X'W J W X = (X'w)(w'X) with J=ones  -> (Xj^T wj)(wj^T Xj) outer
        Xw = Xj.T @ wj                     # (p+1,)
        sumXWJWX += np.outer(Xw, Xw)
        # (W/k) X'X = X' diag(w/k) X
        Matrx_WKXX += Xj.T @ ((wj / k)[:, None] * Xj)
        # (w/k)[1,1] * (X'J X - X'X)
        XJX = np.outer(Xj.sum(0), Xj.sum(0))   # X' J X = (sum X)(sum X)'
        XX = Xj.T @ Xj
        Matrx_wk_XJX_XX += (wj[0] / k) * (XJX - XX)

    denom = sumW - np.trace(V_i @ sumXWJWX)
    termA = np.trace(V_i @ Matrx_WKXX)
    termB = np.trace(V_i @ Matrx_wk_XJX_XX)
    term1 = (Qe - N + termA) / denom
    term2 = termB / denom
    tau_sq1 = term1 + rho * term2
    tau_sq = tau_sq1 if tau_sq1 >= 0 else 0.0
    df = N - termA - rho * termB
    I2_1 = ((Qe - df) / Qe) * 100
    I2 = I2_1 if I2_1 >= 0 else 0.0
    return tau_sq, I2


def _hier_components(Xreg, w_all, e_all, v_all, groups, V_i, p, M):
    """HIER method-of-moments tau.sq and omega.sq (robumeta)."""
    W = w_all  # = 1/var here
    sumV = float(v_all.sum())
    sumW = float(W.sum())
    Qe = float(e_all @ (W * e_all))

    sumEJE = 0.0
    tr_sumJJ = 0.0
    sumXJX = np.zeros_like(V_i)
    sumXWJJX = np.zeros_like(V_i)
    sumXJJWX = np.zeros_like(V_i)
    sumXWWX = np.zeros_like(V_i)
    sumXJWX = np.zeros_like(V_i)
    sumXWJX = np.zeros_like(V_i)
    sumXWJWX = np.zeros_like(V_i)

    for g in groups:
        Xj = Xreg[g]
        wj = W[g]
        ej = e_all[g]
        k = len(g)
        ones = np.ones(k)
        sumEJE += float(ej.sum() ** 2)          # e'J e = (sum e)^2
        tr_sumJJ += float(k * k)                # tr(J J) = k^2
        sX = Xj.sum(0)                          # column sums = X'ones
        sXw = Xj.T @ wj                         # X'W ones (W diag) = sum w_i X_i
        sumXJX += np.outer(sX, sX)              # X'J X
        # X'W J J X : W J J X = w_i * k * (sum_l X_l) rowwise... build explicitly
        # J J = k * J, so X'W(JJ)X = k * X'W J X
        XWJX = np.outer(sXw, sX)                # X'W J X = (X'W ones)(ones'X)
        XJWX = np.outer(sX, sXw)                # X'J W X = (X'ones)(ones'W X)
        sumXWJJX += k * XWJX
        sumXJJWX += k * XJWX
        sumXWWX += Xj.T @ ((wj * wj)[:, None] * Xj)  # X'W W X
        sumXJWX += XJWX
        sumXWJX += XWJX
        sumXWJWX += np.outer(sXw, sXw)          # X'W J W X

    A1 = (tr_sumJJ - np.trace(V_i @ sumXJJWX) - np.trace(V_i @ sumXWJJX)
          + np.trace(V_i @ sumXJX @ V_i @ sumXWJWX))
    B1 = (M - np.trace(V_i @ sumXWJX) - np.trace(V_i @ sumXJWX)
          + np.trace(V_i @ sumXJX @ V_i @ sumXWWX))
    C1 = sumV - np.trace(V_i @ sumXJX)
    A2 = sumW - np.trace(V_i @ sumXWJWX)
    B2 = sumW - np.trace(V_i @ sumXWWX)
    C2 = M - (p + 1)
    Qa = sumEJE

    omega_sq1 = ((Qa - C1) * A2 - (Qe - C2) * A1) / (B1 * A2 - B2 * A1)
    omega_sq = omega_sq1 if omega_sq1 >= 0 else 0.0
    tau_sq1 = ((Qe - C2) / A2) - omega_sq * (B2 / A2)
    tau_sq = tau_sq1 if tau_sq1 >= 0 else 0.0
    I2 = 0.0
    return tau_sq, omega_sq, I2


# --------------------------------------------------------------------------- #
#  CR2 small-sample correction + Satterthwaite df                             #
# --------------------------------------------------------------------------- #
def _cr2(Xreg, r_weights, e_r, groups, k_vec, Q, N, M, p, modelweights):
    """Tipton CR2 adjustment matrices A.MBB, corrected VR.r, and df."""
    W_r_big = r_weights  # diagonal
    # H = Xreg Q Xreg' W  (M x M);  I - H
    H = Xreg @ Q @ Xreg.T * W_r_big[None, :]
    ImH = np.eye(M) - H

    A_list = []
    for j, g in enumerate(groups):
        Xj = Xreg[g]
        k = k_vec[j]
        # ImHii_j = I_k - Xj Q Xj' Wj   (per-study diagonal block of (I-H) form)
        Wj = np.diag(r_weights[g])
        ImHjj = np.eye(k) - Xj @ Q @ Xj.T @ Wj
        if modelweights == "HIER":
            # Working_Matrx_E_j = diag(1/r.weights_j)
            WEj = np.diag(1.0 / r_weights[g])
            sqrtWEj = np.diag(np.sqrt(1.0 / r_weights[g]))
            WEj_15 = np.diag((1.0 / r_weights[g]) ** 1.5)
            inside = sqrtWEj @ ImHjj @ WEj_15
            evals, evecs = np.linalg.eig(inside)
            evals = np.real(evals)
            evecs = np.real(evecs)
            inv_sqrt = np.where(evals < 1e-10, 0.0, 1.0 / np.sqrt(evals))
            A = sqrtWEj @ evecs @ np.diag(inv_sqrt) @ evecs.T @ sqrtWEj
        else:  # CORR
            evals, evecs = np.linalg.eig(ImHjj)
            evals = np.real(evals)
            evecs = np.real(evecs)
            inv_sqrt = np.where(evals < 1e-10, 0.0, 1.0 / np.sqrt(evals))
            A = evecs @ np.diag(inv_sqrt) @ evecs.T
        A_list.append(A)

    # meat: sum_j X'W A (ee') A W X
    meat = np.zeros((p + 1, p + 1))
    for j, g in enumerate(groups):
        Xj = Xreg[g]
        Wj = np.diag(r_weights[g])
        A = A_list[j]
        S = np.outer(e_r[g], e_r[g])
        meat += Xj.T @ Wj @ A @ S @ A @ Wj @ Xj
    VR = Q @ meat @ Q

    # --- Satterthwaite df via g_i matrices ---
    # In robumeta, ImHj is the (M x k_j) column block of (I-H) for study j, so
    # each per-study block  ImHj A Wj Xj Q  is (M x (p+1)); rbind over N studies
    # gives giTemp of shape (N*M, p+1).  Column i is then reshaped column-major
    # into (M, N) as gi_matrix[i], scaled by W.mat, and B = tcrossprod(B_half).
    # W.mat = matrix(rep(1/sqrt(r.weights), times=N), nrow=M) -> (M, N)
    W_mat = np.tile((1.0 / np.sqrt(r_weights))[:, None], (1, N))

    dfs = np.empty(p + 1)
    full_blocks = []
    for j, g in enumerate(groups):
        # R: ImHj = the k_j ROWS of (I-H) for study j (all M cols); giTemp block
        # = t(ImHj) %*% A %*% W %*% X %*% Q  -> t(ImHj) is (M, k_j) = ImH[g,:].T
        ImHj_T = ImH[g, :].T          # (M, k_j)
        Xj = Xreg[g]
        Wj = np.diag(r_weights[g])
        A = A_list[j]
        block = ImHj_T @ A @ Wj @ Xj @ Q   # (M, p+1)
        full_blocks.append(block)
    giTemp_full = np.vstack(full_blocks)  # (N*M, p+1)

    for i in range(p + 1):
        col = giTemp_full[:, i]                 # length N*M
        gi_mat = col.reshape(N, M).T            # matrix(col, nrow=M) col-major
        B_half = W_mat * gi_mat                 # (M, N)
        B = B_half @ B_half.T                   # tcrossprod -> (M, M)
        tr_sq = np.trace(B) ** 2
        sq_tr = np.sum(B * B)
        dfs[i] = tr_sq / sq_tr

    return VR, dfs
