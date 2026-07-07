"""Mediation on Rogers et al. (2023, JPSP Study 5): the "hero's journey" indirect effect."""
import pandas as pd

import socialverse as sv
from benchmarks._harness import Case, fetch

URL = "https://osf.io/download/3qcyb/"


def _run():
    raw = pd.read_csv(fetch(URL, "rogers_s5.csv"))
    d = raw[raw["baddata"] != 1].copy()
    d["condition01"] = (d["condition"] == "manip").astype(int)
    d = d[["condition01", "HJS", "MEANING", "MEANINGT1"]].apply(pd.to_numeric, errors="coerce").dropna()

    st = sv.StudyState()
    sv.pp.ingest(st, data=d)
    st.write("variables", "outcome", "MEANING")
    st.write("design", "treatment", "condition01")
    sv.tl.mediation(st, treatment="condition01", mediator="HJS", outcome="MEANING", boot=5000)
    med = st.models["mediation"]
    return {"acme": med["acme"], "ci_lo": med["ci_acme"][0], "ci_hi": med["ci_acme"][1], "n": len(d)}


def _check(m):
    return [
        (f"N = {m['n']} (published 381)", m["n"] == 381),
        (f"ACME = {m['acme']:.3f} (published .31)", abs(m["acme"] - 0.31) <= 0.06),
        (f"95% CI = [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}] (published [.08,.53], excludes 0)", m["ci_lo"] > 0),
    ]


CASE = Case(
    id="mediation_jpsp2023",
    capability="中介效应(bootstrap 间接效应)",
    agent="social_science_econometrician",
    skill="causal-identification",
    prompt="检验'英雄之旅重述 → 意义感'是否经由英雄之旅感知(HJS)中介,给 5000 次 bootstrap 的间接效应。",
    data="Rogers et al. (2023, JPSP Study 5) · OSF · N=381",
    run=_run, check=_check, offline=False, tags=["mediation"],
)
