"""pydid -- pure-numpy reconstruction of R ``did`` (Callaway & Sant'Anna 2021).

Staggered difference-in-differences with a never-treated control group and the
outcome-regression 2x2 estimator (``est_method='reg'``), plus the ``simple`` and
``dynamic`` (event-study) aggregations of the group-time ATT(g,t).

This mirrors ``did::att_gt`` / ``did::aggte`` faithfully for the canonical
configuration used by socialverse's causal module:

  * panel data, balanced or unbalanced by unit;
  * ``control_group='nevertreated'``;
  * ``est_method='reg'`` with NO covariates (intercept-only outcome regression);
  * ``base_period='varying'`` (the package default), ``anticipation=0``.

Algorithm (matching ``did`` internals ``compute.att_gt`` +
``DRDID::reg_did_panel`` + ``compute.aggte``):

ATT(g,t): for each treated group ``g`` (first-treatment period) and each time
``t`` we form a 2x2 DiD.  With ``base_period='varying'`` the pre period is the
period immediately before ``t`` for pre-treatment comparisons, and is fixed at
``g-1`` (the last period strictly before ``g``) for post-treatment comparisons.
The treated units are those with first-treat ``== g``; controls are the
never-treated (``g == 0``).  With an intercept-only outcome regression the
control model just predicts the mean control ``deltaY``, so

    ATT(g,t) = mean_treated(Ypost - Ypre) - mean_control(Ypost - Ypre).

Aggregations: with unit weights all 1, ``pg(g)`` is the fraction of units first
treated at ``g``.  ``simple`` averages the post-treatment ATT(g,t) weighted by
``pg``; ``dynamic`` groups cells by event time ``e = t - g``, averages within
each ``e`` weighted by ``pg``, and reports the plain mean over ``e >= 0`` as the
overall dynamic ATT.

Point estimates are deterministic and matched to R at < 1e-6.  The standard
errors R reports are the multiplier-bootstrap SEs (``bstrap=TRUE``); they are
stochastic and are NOT gated for element-wise parity -- see the test module.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# core 2x2 outcome-regression DiD (intercept-only == DiD in means)
# --------------------------------------------------------------------------- #
def _reg_did_2x2(ypost, ypre, D):
    """Outcome-regression 2x2 ATT with no covariates.

    Replicates ``DRDID::reg_did_panel`` with an intercept-only design and unit
    weights: control regression of ``deltaY`` on a constant yields the control
    mean, giving ATT = mean(treated deltaY) - mean(control deltaY).
    """
    ypost = np.asarray(ypost, float)
    ypre = np.asarray(ypre, float)
    D = np.asarray(D, float)
    delta = ypost - ypre
    treat = D == 1
    cont = D == 0
    eta_treat = delta[treat].mean()
    eta_cont = delta[cont].mean()          # intercept-only OLS fitted value
    return float(eta_treat - eta_cont)


# --------------------------------------------------------------------------- #
# att_gt
# --------------------------------------------------------------------------- #
class ATTgtResult:
    """Container mirroring the relevant fields of a ``did`` MP object."""

    def __init__(self, group, t, att, glist, tlist, pg, gvar_unit):
        self.group = np.asarray(group, float)
        self.t = np.asarray(t, float)
        self.att = np.asarray(att, float)
        self.glist = np.asarray(glist, float)     # treated groups, sorted
        self.tlist = np.asarray(tlist, float)      # time periods, sorted
        self.pg = np.asarray(pg, float)            # pg per treated group
        self.gvar_unit = np.asarray(gvar_unit, float)


def att_gt(data, yname, tname, idname, gname,
           control_group="nevertreated", est_method="reg",
           anticipation=0, base_period="varying"):
    """Group-time average treatment effects (Callaway & Sant'Anna).

    Parameters mirror ``did::att_gt``.  ``data`` is a mapping of column-name ->
    sequence (e.g. columns of a DataFrame, or a dict of lists).  Supported
    configuration: ``control_group in {'nevertreated', 'notyettreated'}``,
    ``est_method='reg'``, no covariates, panel data.

    With ``control_group='notyettreated'`` the comparison group for a given
    (g, t) cell is the never-treated units PLUS the *not-yet-treated* units --
    those first treated strictly after ``time_threshold = tlist[max(t, pret)]``
    (plus ``anticipation``) and not equal to the current group.  This mirrors
    ``did:::compute.att_gt``'s ``.C`` construction:

        .C = (g == 0) | ((g > time_threshold) & (g != current_g))

    Returns an :class:`ATTgtResult`.
    """
    if control_group not in ("nevertreated", "notyettreated"):
        raise NotImplementedError(
            "only control_group in {'nevertreated','notyettreated'} is ported")
    if est_method != "reg":
        raise NotImplementedError("only est_method='reg' (no covariates) is ported")
    if anticipation != 0:
        raise NotImplementedError("only anticipation=0 is ported")
    if base_period != "varying":
        raise NotImplementedError("only base_period='varying' is ported")

    year = np.asarray(data[tname])
    cid = np.asarray(data[idname])
    y = np.asarray(data[yname], float)
    g = np.asarray(data[gname], float)

    # never-treated groups are coded 0 (Inf also maps to 0 in did); here mpdta
    # uses 0 for never treated.
    g = np.where(np.isinf(g), 0.0, g)

    tlist = np.sort(np.unique(year))
    glist = np.sort(np.unique(g[g > 0]))
    nT = len(tlist)

    # Build a unit x period outcome matrix keyed on the first period's unit order.
    units = np.unique(cid)
    unit_index = {u: i for i, u in enumerate(units)}
    n_units = len(units)
    # outcome[unit, period_idx]
    Y = np.full((n_units, nT), np.nan)
    period_index = {tt: j for j, tt in enumerate(tlist)}
    gvar_unit = np.full(n_units, np.nan)
    for k in range(len(cid)):
        ui = unit_index[cid[k]]
        pj = period_index[year[k]]
        Y[ui, pj] = y[k]
        gvar_unit[ui] = g[k]

    tfac = 1  # base_period != 'universal'
    tlist_length = nT - 1
    nevertreated = control_group == "nevertreated"

    groups, times, atts = [], [], []
    for gi in range(len(glist)):
        current_g = glist[gi]
        # pret_g = last time index (0-based) with (tlist + anticipation) < g
        idx_g = np.where((tlist + anticipation) < current_g)[0]
        pret_g = idx_g[-1] if len(idx_g) else None

        G_full = (gvar_unit == current_g).astype(float)
        if nevertreated:
            C_full = (gvar_unit == 0).astype(float)
            kept = np.where((G_full == 1) | (C_full == 1))[0]

        for t in range(tlist_length):          # t is 0-based, maps to R's 1:tlist.length
            pret = t
            if current_g <= tlist[t + tfac]:
                pret = pret_g
            if pret is None:
                # no pre-treatment period for this group -> drop remaining
                break

            if not nevertreated:
                # not-yet-treated controls: never-treated OR first treated
                # strictly after time_threshold = tlist[max(t, pret) + tfac].
                time_threshold = tlist[max(t, pret) + tfac] + anticipation
                C_full = (
                    (gvar_unit == 0)
                    | ((gvar_unit > time_threshold) & (gvar_unit != current_g))
                ).astype(float)
                kept = np.where((G_full == 1) | (C_full == 1))[0]

            Ypost = Y[kept, t + tfac]
            Ypre = Y[kept, pret]
            Gk = G_full[kept]

            # drop rows with missing outcomes in either period (unbalanced safety)
            ok = ~(np.isnan(Ypost) | np.isnan(Ypre))
            att = _reg_did_2x2(Ypost[ok], Ypre[ok], Gk[ok])

            groups.append(current_g)
            times.append(tlist[t + tfac])
            atts.append(att)

    # pg per treated group: fraction of units in each group (unit weights = 1)
    pg = np.array([np.mean(gvar_unit == gg) for gg in glist])

    return ATTgtResult(groups, times, atts, glist, tlist, pg, gvar_unit)


# --------------------------------------------------------------------------- #
# aggte
# --------------------------------------------------------------------------- #
class AGGTEResult:
    def __init__(self, type, overall_att, egt=None, att_egt=None):
        self.type = type
        self.overall_att = float(overall_att)
        self.egt = None if egt is None else np.asarray(egt, float)
        self.att_egt = None if att_egt is None else np.asarray(att_egt, float)


def aggte(res: ATTgtResult, type="simple", max_e=np.inf, min_e=-np.inf, na_rm=False):
    """Aggregate group-time ATTs (``did::aggte``).

    Supports:

      * ``type='simple'``   -- pg-weighted mean of post-treatment ATT(g,t).
      * ``type='dynamic'``  -- event-study: one ATT per event time ``e=t-g``.
      * ``type='group'``    -- one ATT per treatment cohort (unweighted mean of
        that cohort's post-treatment cells), overall weighted by cohort size.
      * ``type='calendar'`` -- one ATT per calendar period ``t`` (pg-weighted
        over cohorts already treated by ``t``), overall the plain mean over
        periods.

    ``pg`` is taken from the :class:`ATTgtResult`.  ``na_rm`` drops NA ATT(g,t)
    cells before aggregating.
    """
    group = res.group.copy()
    t = res.t.copy()
    att = res.att.copy()
    glist = res.glist
    pg_by_group = res.pg

    if na_rm:
        notna = ~np.isnan(att)
        group, t, att = group[notna], t[notna], att[notna]

    # pg aligned to each (g,t) cell
    pg_lookup = {float(gg): float(p) for gg, p in zip(glist, pg_by_group)}
    pg = np.array([pg_lookup[float(gg)] for gg in group])

    # keepers: post-treatment cells within the event window
    keepers = np.where((group <= t) & (t <= (group + max_e)))[0]

    if type == "simple":
        overall = np.sum(att[keepers] * pg[keepers]) / np.sum(pg[keepers])
        return AGGTEResult("simple", overall)

    if type == "dynamic":
        e_all = t - group
        eseq = np.unique(e_all)
        eseq = eseq[(eseq >= min_e) & (eseq <= max_e)]
        att_egt = []
        for e in eseq:
            whiche = np.where(e_all == e)[0]
            pge = pg[whiche] / np.sum(pg[whiche])
            att_egt.append(np.sum(att[whiche] * pge))
        att_egt = np.array(att_egt)
        epos = eseq >= 0
        overall = float(np.mean(att_egt[epos]))
        return AGGTEResult("dynamic", overall, egt=eseq, att_egt=att_egt)

    if type == "group":
        # one ATT per cohort g: unweighted mean over that cohort's post cells
        # within [g, g+max_e]; overall weighted by cohort size pgg.
        glist_here = np.sort(np.unique(group))
        att_g = []
        for g in glist_here:
            whichg = np.where((group == g) & (g <= t) & (t <= (g + max_e)))[0]
            att_g.append(np.mean(att[whichg]))
        att_g = np.array(att_g)
        pgg = np.array([pg_lookup[float(g)] for g in glist_here])
        overall = float(np.sum(att_g * pgg) / np.sum(pgg))
        return AGGTEResult("group", overall, egt=glist_here, att_egt=att_g)

    if type == "calendar":
        # one ATT per calendar period t >= min(group), pg-weighted over cohorts
        # already treated by t; overall the plain mean over periods.
        minG = np.min(group)
        tlist_here = np.sort(np.unique(t))
        cal_tlist = tlist_here[tlist_here >= minG]
        # keep only periods with a post-treatment cell
        cal_tlist = np.array(
            [t1 for t1 in cal_tlist if np.any((t == t1) & (group <= t))])
        att_t = []
        for t1 in cal_tlist:
            whicht = np.where((t == t1) & (group <= t))[0]
            pgt = pg[whicht] / np.sum(pg[whicht])
            att_t.append(np.sum(pgt * att[whicht]))
        att_t = np.array(att_t)
        overall = float(np.mean(att_t))
        return AGGTEResult("calendar", overall, egt=cal_tlist, att_egt=att_t)

    raise NotImplementedError(f"aggregation type '{type}' not ported")
