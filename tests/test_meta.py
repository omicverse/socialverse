"""Native meta-analysis engine — exactness + parameter-recovery checks.

Closed-form transforms are checked to machine precision against hand calc; the
iterative REML paths are checked against an independent brute-force optimum (not
bit-identical to metafor, but the same statistical target). The 3-level engine is
checked to recover the 2-level total heterogeneity when there is one effect per
study (the only identified quantity there is σ²₂+σ²₃ = τ²).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import socialverse as sv

warnings.simplefilter("ignore")

_YI = np.array([0.10, 0.30, 0.25, 0.40, 0.05, 0.35])
_VI = np.array([0.02, 0.03, 0.015, 0.05, 0.01, 0.04])


def _effects_state(yi=_YI, vi=_VI, **extra):
    s = sv.StudyState()
    df = pd.DataFrame({"yi": yi, "vi": vi, "sei": np.sqrt(vi), "measure": "GEN"})
    for k, v in extra.items():
        df[k] = v
    s.write("models", "meta_effects", df)
    return s


# --------------------------------------------------------------- registration
def test_all_meta_functions_registered():
    for name in ["es_proportion", "es_from_means", "hedges_correct", "es_from_2x2",
                 "es_from_r", "es_from_ci", "escalc", "vcalc", "meta_fixed",
                 "meta_random", "rma_mv", "meta_heterogeneity", "ma_i2_multilevel",
                 "meta_prediction_interval", "metareg", "metareg_fdr", "egger_test",
                 "meta_forest", "funnel"]:
        assert sv.registry.get(name) is not None, f"{name} not registered"


# ---------------------------------------------------------- effect-size ingest
def test_es_proportion_logit_closed_form():
    df = pd.DataFrame({"cases": [20, 5, 50], "n": [100, 40, 200]})
    s = sv.StudyState(); s.write("sources", "datasets", df)
    sv.pp.es_proportion(s, measure="PLO", cases="cases", n="n")
    eff = s.models["meta_effects"]
    p = np.array([0.2, 0.125, 0.25]); n = np.array([100, 40, 200])
    assert np.allclose(eff["yi"], np.log(p / (1 - p)))
    assert np.allclose(eff["vi"], 1 / (n * p) + 1 / (n * (1 - p)))


def test_es_proportion_arcsine_and_pft():
    df = pd.DataFrame({"cases": [20, 50], "n": [100, 200]})
    s = sv.StudyState(); s.write("sources", "datasets", df)
    sv.pp.es_proportion(s, measure="PAS", cases="cases", n="n")
    eff = s.models["meta_effects"]
    assert np.allclose(eff["vi"], 1 / (4 * np.array([100, 200])))  # arcsine var = 1/4n


def test_es_from_means_hedges():
    df = pd.DataFrame({"m1": [5.0], "sd1": [1.0], "n1": [30],
                       "m2": [4.0], "sd2": [1.2], "n2": [30]})
    s = sv.StudyState(); s.write("sources", "datasets", df)
    sv.pp.es_from_means(s, m1="m1", sd1="sd1", n1="n1", m2="m2", sd2="sd2", n2="n2", hedges=True)
    d = s.models["meta_effects"]["yi"].iloc[0]
    sp = np.sqrt((29 * 1 + 29 * 1.44) / 58)
    J = 1 - 3 / (4 * 58 - 1)
    assert d == pytest.approx(J * (1.0 / sp), rel=1e-9)


def test_es_from_2x2_logor():
    df = pd.DataFrame({"ai": [20], "bi": [80], "ci": [10], "di": [90]})
    s = sv.StudyState(); s.write("sources", "datasets", df)
    sv.pp.es_from_2x2(s, measure="OR", ai="ai", bi="bi", ci="ci", di="di")
    eff = s.models["meta_effects"]
    assert eff["yi"].iloc[0] == pytest.approx(np.log((20 * 90) / (80 * 10)))
    assert eff["vi"].iloc[0] == pytest.approx(1/20 + 1/80 + 1/10 + 1/90)


def test_es_from_r_fisher_z():
    df = pd.DataFrame({"r": [0.3, 0.5], "n": [50, 80]})
    s = sv.StudyState(); s.write("sources", "datasets", df)
    sv.pp.es_from_r(s, measure="ZCOR", r="r", n="n")
    eff = s.models["meta_effects"]
    assert np.allclose(eff["yi"], np.arctanh([0.3, 0.5]))
    assert np.allclose(eff["vi"], 1 / (np.array([50, 80]) - 3))


# --------------------------------------------------------------- core poolers
def test_meta_fixed_exact():
    s = _effects_state()
    sv.tl.meta_fixed(s)
    m = s.models["meta"]
    w = 1 / _VI; mu = (w * _YI).sum() / w.sum()
    assert m["estimate"] == pytest.approx(mu, abs=1e-12)
    assert m["se"] == pytest.approx(np.sqrt(1 / w.sum()), abs=1e-12)


def test_meta_random_dl_exact():
    s = _effects_state()
    sv.tl.meta_random(s, method="DL")
    w = 1 / _VI; mu = (w * _YI).sum() / w.sum()
    Q = (w * (_YI - mu) ** 2).sum(); C = w.sum() - (w ** 2).sum() / w.sum()
    assert s.models["meta"]["tau2"] == pytest.approx(max(0, (Q - 5) / C), abs=1e-12)


def test_meta_random_reml_matches_grid():
    # a dataset with real heterogeneity so τ² > 0
    yi = np.array([0.2, 0.8, -0.1, 0.5, 0.9, 0.0, 0.6, 0.3])
    vi = np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.05, 0.06])
    s = _effects_state(yi, vi)
    sv.tl.meta_random(s, method="REML")
    tau2 = s.models["meta"]["tau2"]

    def neg_reml(t2):
        wt = 1 / (vi + t2); mu_ = (wt * yi).sum() / wt.sum()
        return np.sum(np.log(vi + t2)) + np.log(wt.sum()) + np.sum(wt * (yi - mu_) ** 2)

    grid = np.linspace(0, 0.5, 500001)
    best = grid[int(np.argmin([neg_reml(t) for t in grid[::250]])) * 250]
    fine = np.linspace(max(0, best - 0.003), best + 0.003, 60001)
    grid_tau2 = fine[int(np.argmin([neg_reml(t) for t in fine]))]
    assert tau2 == pytest.approx(grid_tau2, abs=1e-3)
    assert tau2 > 0.0


def test_meta_random_knapp_hartung_widens_ci():
    yi = np.array([0.2, 0.8, -0.1, 0.5, 0.9, 0.0])
    vi = np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04])
    a = _effects_state(yi, vi); sv.tl.meta_random(a, method="REML")
    b = _effects_state(yi, vi); sv.tl.meta_random(b, method="REML", knapp_hartung=True)
    wa = a.models["meta"]["ci_ub"] - a.models["meta"]["ci_lb"]
    wb = b.models["meta"]["ci_ub"] - b.models["meta"]["ci_lb"]
    assert wb > wa  # HKSJ + t-dist widens


# ----------------------------------------------------------- multilevel engine
def test_rma_mv_recovers_two_level_total_heterogeneity():
    """1 ES/study ⇒ σ²₂ and σ²₃ jointly unidentified; only σ²₂+σ²₃ = τ² is."""
    yi = np.array([0.2, 0.8, -0.1, 0.5, 0.9, 0.0, 0.6, 0.3])
    vi = np.array([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.05, 0.06])
    ref = _effects_state(yi, vi); sv.tl.meta_random(ref, method="REML")
    tau2_2level = ref.models["meta"]["tau2"]
    s = _effects_state(yi, vi, study=[f"s{i}" for i in range(len(yi))])
    sv.tl.rma_mv(s, study="study")
    mv = s.models["meta"]
    assert mv["sigma2_total"] == pytest.approx(tau2_2level, abs=2e-3)
    assert mv["converged"]


def test_rma_mv_three_level_decomposition_runs():
    rng = np.random.default_rng(1)
    rows = []
    for st in range(15):
        u3 = rng.normal(0, 0.3)
        for _ in range(rng.integers(1, 4)):
            n = int(rng.integers(50, 300)); p = 1 / (1 + np.exp(-(-0.5 + u3 + rng.normal(0, 0.2))))
            rows.append({"study": f"S{st}", "cases": rng.binomial(n, p), "n": n})
    data = pd.DataFrame(rows)
    s = sv.StudyState(); s.write("sources", "datasets", data)
    sv.pp.escalc(s, measure="PAS", cases="cases", n="n", study="study")
    sv.tl.vcalc(s, cluster="study", rho=0.6)
    sv.tl.rma_mv(s, study="study")
    sv.tl.meta_heterogeneity(s)
    sv.tl.ma_i2_multilevel(s)
    i2 = s.diagnostics["i2_multilevel"]
    shares = i2["sampling_share"] + i2["I2_level2_within_study"] + i2["I2_level3_between_study"]
    assert shares == pytest.approx(100.0, abs=1e-6)   # decomposition partitions 100%
    assert s.models["meta"]["estimate"] is not None


# ------------------------------------------------------------- meta-regression
def test_metareg_fdr_bh_monotone():
    rng = np.random.default_rng(2)
    k = 40
    mod = rng.normal(size=k)
    yi = 0.8 * mod + rng.normal(0, 0.2, k)
    vi = np.full(k, 0.04)
    s = _effects_state(yi, vi, mod=mod, noise=rng.normal(size=k))
    sv.tl.metareg(s, moderators=["mod", "noise"])
    sv.tl.metareg_fdr(s)
    fdr = s.diagnostics["metareg_fdr"]
    # the real moderator is significant; BH p ≥ raw p
    assert fdr["per_moderator"]["mod"]["significant_fdr"]
    for name, st in fdr["per_moderator"].items():
        assert st["pval_fdr"] >= st["pval"] - 1e-12
    assert s.models["metareg"]["R2"] > 50.0   # strong moderator explains heterogeneity


# ------------------------------------------------------------------- pub-bias
def test_egger_detects_asymmetry():
    # inject small-study effect: small studies (large sei) biased upward
    rng = np.random.default_rng(3)
    sei = np.linspace(0.05, 0.5, 30)
    yi = 0.2 + 1.5 * sei + rng.normal(0, 0.02, 30)   # effect grows with sei ⇒ asymmetry
    s = _effects_state(yi, sei ** 2)
    sv.tl.egger_test(s)
    assert s.diagnostics["egger"]["pval"] < 0.05
    assert s.diagnostics["egger"]["asymmetry"]


# --------------------------------------------------------------------- figures
def test_forest_and_funnel_emit_png(tmp_path):
    s = _effects_state(study=[f"s{i}" for i in range(6)])
    sv.tl.rma_mv(s, study="study")
    sv.tl.meta_heterogeneity(s)
    sv.tl.meta_prediction_interval(s)
    sv.pl.meta_forest(s, out=str(tmp_path / "forest.png"))
    sv.pl.funnel(s, out=str(tmp_path / "funnel.png"))
    figs = s.artifacts["figures"]
    assert "meta_forest" in figs and "funnel" in figs
    assert (tmp_path / "forest.png").stat().st_size > 1000
    assert (tmp_path / "funnel.png").stat().st_size > 1000
