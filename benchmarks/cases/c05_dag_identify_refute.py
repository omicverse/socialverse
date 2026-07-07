"""Causal-graph identification + four-step refutation on a confounded DGP (DoWhy-style)."""
import numpy as np
import pandas as pd

import socialverse as sv
from benchmarks._harness import Case, approx


def _run():
    rng = np.random.default_rng(0)
    n = 3000
    Z = rng.normal(0, 1, n)                         # confounder
    T = 0.7 * Z + rng.normal(0, 1, n)               # treatment
    Y = 1.5 * T + 2.0 * Z + rng.normal(0, 1, n)     # true ATE = 1.5, backdoor = {Z}
    df = pd.DataFrame({"Z": Z, "T": T, "Y": Y})

    st = sv.StudyState()
    st.write("design", "treatment", "T")
    st.write("variables", "outcome", "Y")
    sv.pp.ingest(st, data=df)
    sv.tl.dag_identify(st, graph="Z->T; Z->Y; T->Y", treatment="T", outcome="Y")
    sv.tl.dag_refute(st, seed=1)
    est = st.identification["estimand"]
    placebo = next(c["new_estimate"] for c in st.diagnostics["refutation"]["checks"]
                   if c["refuter"] == "placebo_treatment")
    naive = float(np.polyfit(df["T"], df["Y"], 1)[0])
    return {
        "strategy": est["strategy"], "adjustment": est["adjustment_set"],
        "ate": st.models["dag"]["ate"], "naive": naive,
        "placebo": placebo, "verdict": st.diagnostics["refutation"]["verdict"],
    }


def _check(m):
    return [
        (f"identification = {m['strategy']} · adjustment set = {m['adjustment']}",
         m["strategy"] == "backdoor" and m["adjustment"] == ["Z"]),
        (f"backdoor ATE = {m['ate']:.3f} (true 1.5; naive {m['naive']:.3f} confounded)",
         approx(m["ate"], 1.5, 0.15) and m["naive"] > 2.0),
        (f"placebo effect = {m['placebo']:+.3f} (≈ 0)", abs(m["placebo"]) < 0.15),
        (f"refutation verdict = {m['verdict']}", m["verdict"] == "robust"),
    ]


CASE = Case(
    id="dag_identify_refute",
    capability="因果图识别 + 四步反驳(Pearl/DoWhy)",
    agent="causal_data_scientist",
    skill="causal-dag",
    prompt="把'混杂同时影响处理和结果'的假设画成 DAG,用 d-分离找最小充分调整集,"
           "估效应后做安慰剂、随机共因、子样本和隐藏混杂敏感性反驳。",
    data="合成混杂 DGP(真 ATE=1.5,后门集 {Z})",
    run=_run, check=_check, offline=True, tags=["dag"],
)
