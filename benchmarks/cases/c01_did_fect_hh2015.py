"""Staggered-adoption DiD + counterfactual imputation on HH2015 (Liu-Wang-Xu 2024)."""
import io

import pandas as pd

import socialverse as sv
from benchmarks._harness import Case, approx, fetch

URL = "https://dataverse.harvard.edu/api/access/datafile/6165649"


def _run():
    raw = pd.read_csv(fetch(URL, "hh2015.tab"), sep="\t", na_values=[""])
    df = raw.dropna(subset=["nat_rate_ord"]).copy()
    df["treat_post"] = df["indirect"].astype(int)
    ft = df[df.treat_post == 1].groupby("bfs")["year"].min()
    df["first_treated"] = df["bfs"].map(ft).fillna(0).astype(int)

    st = sv.StudyState()
    st.write("variables", "outcome", "nat_rate_ord")
    st.write("estimand", "target", "ATT")
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="bfs", time="year",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.parallel_trends(st)
    sv.tl.did(st)
    sv.tl.fect(st, r=0, nboots=200, placebo=True, seed=42)
    return {
        "twfe_att": st.models["did"]["att"],
        "fect_att": st.models["fect"]["att"],
        "fect_hc1_se": next(s["se"] for s in st.diagnostics["robustness"]["specs"]
                            if s["spec"] == "HC1_robust"),
        "placebo_p": st.models["fect"]["placebo"]["placebo_p"],
    }


def _check(m):
    return [
        (f"TWFE ATT = {m['twfe_att']:+.3f} (published +1.339)", approx(m["twfe_att"], 1.339, 0.02)),
        (f"FEct ATT = {m['fect_att']:+.3f} (heterogeneity-robust ~+1.50)", approx(m["fect_att"], 1.50, 0.08)),
        (f"HC1 SE = {m['fect_hc1_se']:.3f} (published 0.161)", approx(m["fect_hc1_se"], 0.161, 0.01)),
        (f"placebo p = {m['placebo_p']:.3f} (not significant)", m["placebo_p"] > 0.05),
    ]


CASE = Case(
    id="did_fect_hh2015",
    capability="交错采纳 DiD + 反事实插补(异质稳健)",
    agent="social_science_econometrician",
    skill="causal-identification + modern-did",
    prompt="用这份瑞士市镇面板,估计切换到间接归化程序对归化率的效应:声明面板设计、检验平行趋势、"
           "跑 TWFE 双重差分,再用反事实插补估计量做异质稳健对照,并做 placebo 检验。",
    data="HH2015 (Liu-Wang-Xu 2024, AJPS) · Harvard Dataverse · 1211 市镇 × 1991–2009",
    run=_run, check=_check, offline=False, tags=["did", "fect"],
)
