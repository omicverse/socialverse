"""Double ML + causal forest on the 401(k) textbook case (Chernozhukov et al. 2018)."""
import pandas as pd

import socialverse as sv
from benchmarks._harness import Case, approx, fetch

URL = "https://github.com/VC2015/DMLonGitHub/raw/master/sipp1991.dta"
CONF = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]


def _run():
    df = pd.read_stata(fetch(URL, "sipp1991.dta"))
    naive = df.groupby("e401")["net_tfa"].mean()
    st = sv.StudyState()
    st.write("design", "treatment", "e401")
    st.write("variables", "outcome", "net_tfa")
    sv.pp.ingest(st, data=df)
    sv.tl.dml(st, treatment="e401", outcome="net_tfa", hetero=["inc"],
              controls=[c for c in CONF if c != "inc"], discrete_treatment=True, folds=5, seed=0)
    sv.tl.causal_forest(st, treatment="e401", outcome="net_tfa", hetero=CONF,
                        discrete_treatment=True, folds=5, nboots=20, seed=0)
    f = st.models["causal_forest"]
    top = max(f["feature_importance"], key=f["feature_importance"].get)
    return {
        "naive_diff": float(naive[1] - naive[0]),
        "dml_ate": st.models["dml"]["ate"],
        "cate_spread": f["cate_summary"]["p90"] - f["cate_summary"]["p10"],
        "top_modifier": top,
    }


def _check(m):
    return [
        (f"naive diff = ${m['naive_diff']:,.0f} (confounded, ~$19.5k)", approx(m["naive_diff"], 19559, 500)),
        (f"DML ATE = ${m['dml_ate']:,.0f} (published ~$9,000)", 7500 <= m["dml_ate"] <= 11500),
        (f"CATE spread = ${m['cate_spread']:,.0f} (strong heterogeneity)", m["cate_spread"] > 8000),
        (f"top effect modifier = {m['top_modifier']} (income)", m["top_modifier"] == "inc"),
    ]


CASE = Case(
    id="dml_cate_401k",
    capability="双重机器学习 + 因果森林(异质处理效应 CATE)",
    agent="causal_data_scientist",
    skill="causal-dag + causal-machine-learning",
    prompt="估计 401(k) 资格对净金融资产的效应:用双重机器学习去混杂估平均效应,"
           "再用因果森林看效应随收入怎么变、谁获益最多。",
    data="SIPP 1991 (Chernozhukov et al. 2018) · 9915 户 · DMLonGitHub",
    run=_run, check=_check, offline=False, tags=["dml", "cate"],
)
