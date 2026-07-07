"""Social-network construction + exponential random graph model (ERGM)."""
import socialverse as sv
from socialverse import datasets as D
from benchmarks._harness import Case


def _run():
    df = D.load_network()
    st = sv.StudyState()
    sv.pp.ingest(st, data=df)
    sv.tl.build_network(st, source="source", target="target")
    sv.tl.ergm(st, terms=["edges"])
    net = st.models["network"]
    ergm = st.models["ergm"]
    return {"n_nodes": net["n_nodes"], "n_edges": net["n_edges"], "density": net["density"],
            "ergm_terms": list(ergm.get("coef", {}).keys()), "ergm_coef": ergm.get("coef", {})}


def _check(m):
    return [
        (f"network built: {m['n_nodes']} nodes, {m['n_edges']} edges", m["n_nodes"] > 0 and m["n_edges"] > 0),
        (f"density = {m['density']:.3f} (valid 0–1)", 0 < m["density"] < 1),
        (f"ERGM estimated terms = {m['ergm_terms']}", "edges" in m["ergm_coef"]),
    ]


CASE = Case(
    id="network_ergm",
    capability="社会网络分析 + ERGM(结构生成模型)",
    agent="causal_data_scientist",
    skill="network-analysis",
    prompt="从这份边列表构建社会网络,给中心性和密度,再用指数随机图模型 ERGM 检验结构项。",
    data="socialverse 玩具边列表(有向/无向网络)",
    run=_run, check=_check, offline=True, tags=["network", "ergm"],
)
