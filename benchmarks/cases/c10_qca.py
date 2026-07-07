"""Qualitative Comparative Analysis (fsQCA: truth table + Boolean minimization)."""
import socialverse as sv
from socialverse import datasets as D
from benchmarks._harness import Case


def _run():
    df = D.load_qca()
    st = sv.StudyState()
    st.write("variables", "outcome", "Y")
    sv.pp.ingest(st, data=df)
    sv.tl.qca(st, outcome="Y", conditions=["A", "B", "C"])
    m = st.models["qca"]
    return {"solution": m["solution"], "consistency": m["solution_consistency"],
            "coverage": m["solution_coverage"]}


def _check(m):
    sol = m["solution"].replace(" ", "")
    return [
        (f"solution = '{m['solution']}' (recovers C + A*B)", "C" in sol and "A*B" in sol),
        (f"solution consistency = {m['consistency']:.3f} (> 0.9)", m["consistency"] > 0.9),
        (f"solution coverage = {m['coverage']:.3f} (> 0.9)", m["coverage"] > 0.9),
    ]


CASE = Case(
    id="qca_fsqca",
    capability="定性比较分析 QCA(集合论:必要/充分、真值表最小化)",
    agent="qualitative_researcher",
    skill="qca",
    prompt="用模糊集 QCA 找导致结果的条件组态:校准、建真值表、布尔最小化,给一致性和覆盖度。",
    data="socialverse 玩具 QCA 数据(真解 C + A*B)",
    run=_run, check=_check, offline=True, tags=["qca"],
)
