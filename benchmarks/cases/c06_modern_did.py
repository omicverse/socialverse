"""Modern heterogeneity-robust DiD (Sun-Abraham / did2s / LP-DiD) on a staggered panel."""
import socialverse as sv
from socialverse import datasets as D
from benchmarks._harness import Case, approx


def _run():
    df = D.load_did_staggered(n_units=150, n_periods=12, att=2.0, seed=1)
    true_att = df.attrs["true_att"]
    st = sv.StudyState()
    st.write("variables", "outcome", "y")
    st.write("estimand", "target", "ATT")
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="unit", time="period",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.parallel_trends(st)
    sv.tl.did(st)
    sv.tl.sun_abraham(st)
    sv.tl.did2s(st, nboots=100)
    sv.tl.local_projection(st, max_horizon=4)
    sa = st.models["sun_abraham"]["coefs"]
    return {
        "true_att": true_att,
        "twfe": st.models["did"]["att"],
        "did2s": st.models["did2s"]["att"],
        "sa_pre": sa.get("-2", (0, 0))[0],
        "sa_h0": sa["0"][0], "sa_h2": sa["2"][0],
        "lp_h2": st.models["local_projection"]["coefs"]["2"][0],
    }


def _check(m):
    return [
        (f"did2s ATT = {m['did2s']:.3f} (true {m['true_att']:.3f})", approx(m["did2s"], m["true_att"], 0.2)),
        (f"TWFE = {m['twfe']:.3f} (biased below true {m['true_att']:.3f})", m["twfe"] < m["true_att"] - 0.3),
        (f"Sun-Abraham pre-trend(-2) = {m['sa_pre']:+.3f} (≈ 0)", abs(m["sa_pre"]) < 0.3),
        (f"dynamic effect grows: h0={m['sa_h0']:.2f} < h2={m['sa_h2']:.2f}", m["sa_h0"] < m["sa_h2"]),
        (f"LP-DiD h2 = {m['lp_h2']:.2f} (impulse response > 1)", m["lp_h2"] > 1.0),
    ]


CASE = Case(
    id="modern_did",
    capability="现代异质稳健 DiD:Sun-Abraham / did2s / 局部投影",
    agent="causal_data_scientist",
    skill="modern-did",
    prompt="这是交错采纳面板、效应随时间增长。别用经典 TWFE——用 Sun-Abraham 交互加权、"
           "Gardner 两步、局部投影三种异质稳健估计量各跑一遍,给动态效应路径。",
    data="socialverse 玩具交错采纳面板(动态真效应,TWFE 偏)",
    run=_run, check=_check, offline=True, tags=["did", "modern-did"],
)
