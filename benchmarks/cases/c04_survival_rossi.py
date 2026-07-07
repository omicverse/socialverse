"""Cox proportional-hazards survival on the Rossi recidivism experiment (Allison 2014)."""
import pandas as pd

import socialverse as sv
from benchmarks._harness import Case, approx, fetch

URL = "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Rossi.csv"


def _run():
    df = pd.read_csv(fetch(URL, "rossi.csv"))
    df["fin"] = (df["fin"] == "yes").astype(int)
    df["race"] = (df["race"] == "black").astype(int)
    df["wexp"] = (df["wexp"] == "yes").astype(int)
    df["mar"] = (df["mar"] == "married").astype(int)
    df["paro"] = (df["paro"] == "yes").astype(int)

    st = sv.StudyState()
    st.write("variables", "outcome", "arrest")
    sv.pp.ingest(st, data=df)
    sv.tl.survival(st, time="week", event="arrest",
                   covariates=["fin", "age", "race", "wexp", "mar", "paro", "prio"])
    log_hr = st.models["cox"]["log_hr"]
    return {"fin": log_hr["fin"][0], "prio": log_hr["prio"][0], "age": log_hr["age"][0]}


def _check(m):
    # Allison (2014) published Cox coefficients
    return [
        (f"fin logHR = {m['fin']:+.3f} (Allison -0.379)", approx(m["fin"], -0.379, 0.02)),
        (f"prio logHR = {m['prio']:+.3f} (Allison +0.091)", approx(m["prio"], 0.091, 0.02)),
        (f"age logHR = {m['age']:+.3f} (Allison -0.057)", approx(m["age"], -0.057, 0.02)),
    ]


CASE = Case(
    id="survival_rossi",
    capability="生存分析:Cox 比例风险",
    agent="social_science_econometrician",
    skill="survival-analysis",
    prompt="做累犯的 Cox 比例风险模型:经济资助、年龄、前科等对再次被捕风险的效应,并查比例风险假设。",
    data="Rossi 累犯随机实验 (Allison 2014) · Rdatasets · N=432",
    run=_run, check=_check, offline=False, tags=["survival"],
)
