"""Demographic analysis: period life table (life expectancy) + Kitagawa rate decomposition."""
import socialverse as sv
from socialverse import datasets as D
from benchmarks._harness import Case


def _run():
    df = D.load_demography()
    st = sv.StudyState()
    sv.pp.ingest(st, data=df)
    sv.tl.life_table(st, age="age_group", rate="mx_A", population="pop_A")
    sv.tl.decomposition(st, group="grp", rate_a="mx_A", rate_b="mx_B",
                        pop_a="pop_A", pop_b="pop_B")
    lt = st.models["life_table"]
    dec = st.models["decomposition"]
    return {"e0": lt["e0"],
            "total_diff": dec["total_diff"],
            "rate_effect": dec["rate_effect"],
            "composition_effect": dec["composition_effect"]}


def _check(m):
    add_up = m["rate_effect"] + m["composition_effect"]
    return [
        (f"life expectancy e0 = {m['e0']:.2f} (finite, positive)", 0 < m["e0"] < 120),
        (f"Kitagawa: rate + composition = {add_up:.5f} = total diff {m['total_diff']:.5f} (exact)",
         abs(add_up - m["total_diff"]) < 1e-6),
    ]


CASE = Case(
    id="demography_kitagawa",
    capability="人口学:生命表(预期寿命)+ Kitagawa 率分解",
    agent="social_science_econometrician",
    skill="demographic-analysis",
    prompt="构建周期生命表给预期寿命,再用 Kitagawa 分解把两组的率差拆成'构成'和'率'两部分。",
    data="socialverse 玩具年龄别死亡率(两组 A/B)",
    run=_run, check=_check, offline=True, tags=["demography"],
)
