"""Spatial autocorrelation (Moran's I) + spatial-lag regression (SAR)."""
import socialverse as sv
from socialverse import datasets as D
from benchmarks._harness import Case, approx


def _run():
    df, W = D.load_spatial()
    st = sv.StudyState()
    st.write("variables", "outcome", "y")
    sv.pp.ingest(st, data=df)
    sv.tl.spatial_autocorr(st, value="y", w=W)
    sv.tl.spatial_regression(st, outcome="y", predictors=["x"], w=W)
    moran = st.diagnostics["moran"]
    return {"moran_I": moran["I"], "moran_p": moran["p_perm"], "sar_rho": st.models["sar"]["rho"]}


def _check(m):
    return [
        (f"Moran's I = {m['moran_I']:.3f} (positive spatial clustering)", m["moran_I"] > 0.15),
        (f"Moran p = {m['moran_p']:.3f} (significant)", m["moran_p"] < 0.05),
        (f"SAR spatial-lag rho = {m['sar_rho']:.3f} (true ~0.5)", approx(m["sar_rho"], 0.5, 0.12)),
    ]


CASE = Case(
    id="spatial_moran_sar",
    capability="空间分析:Moran's I 自相关 + SAR 空间回归",
    agent="social_science_econometrician",
    skill="spatial-analysis",
    prompt="检验这个指标在空间上是否聚集:算全局 Moran's I 和局部 LISA,再跑空间滞后回归 SAR。",
    data="socialverse 玩具空间网格 + 权重矩阵(真 rho=0.5)",
    run=_run, check=_check, offline=True, tags=["spatial"],
)
