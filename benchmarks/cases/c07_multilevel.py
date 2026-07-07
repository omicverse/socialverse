"""Multilevel / hierarchical linear model (random intercept + slope, ICC)."""
import socialverse as sv
from socialverse import datasets as D
from benchmarks._harness import Case, approx


def _run():
    df = D.load_multilevel()
    st = sv.StudyState()
    st.write("variables", "outcome", "y")
    sv.pp.ingest(st, data=df)
    sv.tl.multilevel(st, groups="school", outcome="y", predictors=["x"])
    vc = st.diagnostics["variance_components"]
    return {"icc": vc["icc"], "slope": vc.get("primary_slope"),
            "n_groups": st.models["mixedlm"]["n_groups"]}


def _check(m):
    return [
        (f"ICC = {m['icc']:.3f} (substantial group clustering, ~0.5)", 0.3 < m["icc"] < 0.7),
        (f"slope on x = {m['slope']:.3f} (true ~2.0)", approx(m["slope"], 2.0, 0.3)),
        (f"{m['n_groups']} groups estimated", m["n_groups"] > 5),
    ]


CASE = Case(
    id="multilevel_hlm",
    capability="多层/分层线性模型(随机截距+斜率、ICC)",
    agent="social_science_econometrician",
    skill="multilevel-modeling",
    prompt="这是学生嵌套在学校里的数据,普通回归会违反独立性。跑分层线性模型:"
           "随机截距 + 随机斜率,给方差成分和 ICC。",
    data="socialverse 玩具嵌套数据(学生∈学校)",
    run=_run, check=_check, offline=True, tags=["multilevel"],
)
