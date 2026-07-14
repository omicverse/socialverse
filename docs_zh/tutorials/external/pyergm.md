# pyergm — ergm in Python

> 指数随机图模型 (ERGM) 用于社会网络，可从 Python 调用，与 R `ergm` 保持 1e-6 精度，无需 R 运行时。

## `ergm` 的功能

`ergm` (属于 `statnet` 套件的一部分) 是拟合指数随机图模型的标准 R 工具包 — 网络形成的统计模型，其中观测图的概率正比于 `exp(θ · g(y))`，其中 `g(y)` 是充分统计量向量 (边数、三角形、度分布、同质性项、互惠性等)。社会科学研究者使用它来检验假设，如"财富同质性是否比随机机制更好地解释这个婚姻网络"或"互惠性是否是这个建议网络的显著结构驱动因子"，同时正确考虑普通逻辑回归忽视的关系之间的依赖性。`ergm` 还通过 `summary(net ~ terms)` 暴露网络的*观测*充分统计量 (无需估计)，以及通过伴随包 `sna` 的 Holland–Leinhardt 有向三角形普查 — 两者都常用作网络的描述性总结或拟合优度目标。

## 端口

由 `socialverse.external.pyergm` 暴露的公开函数 (见 `__init__.py` 的 `__all__`):

- `dyads(n, directed=False)` — 列举 n 节点网络的所有二元体索引对 (无向的上三角，有向的所有有序 `i != j`)。
- `change_stats_edges(pairs)` — `edges` 项的变化统计量 (每个二元体为常数 1)。
- `change_stats_nodecov(pairs, attr)` — `nodecov(attr)` 的变化统计量，`attr[i] + attr[j]`。
- `change_stats_nodematch(pairs, attr)` — `nodematch(attr)` 的变化统计量，`1{attr[i] == attr[j]}`。
- `build_design(adjacency, terms, directed=False)` — 为一组二元体独立项组装二元设计矩阵 `X`、关系指示器响应 `y` 和列标签。
- `ergm_mple(adjacency, terms, directed=False, max_iter=100, tol=1e-12)` — 通过最大伪似然 (对变化统计量的 IRLS 逻辑回归) 拟合二元体独立 ERGM；返回 `MPLEResult`。
- `MPLEResult` — 数据类，持有 `terms`、`coef`、`se`、`vcov`、`n_iter`、`loglik`，配有 `.summary()` 漂亮打印器。
- `summary_formula(adjacency, terms, directed=False, attr_name=None)` — 请求项集的观测充分统计量 `g(y)` (`edges`、`triangle`、`degree`/`idegree`/`odegree`、`kstar`/`istar`/`ostar`、`mutual`、`nodecov`、`nodematch`)；精确计数，无估计。
- `triad_census(adjacency)` — Holland–Leinhardt 16 型有向三角形普查 (`sna::triad.census`)，列顺序由 `TRIAD_CENSUS_LABELS` 给出。
- `TRIAD_CENSUS_LABELS` — 16 个 MAN 码标签 (`"003"`、`"012"`、…、`"300"`) 按 `triad_census` 返回的顺序。

该端口是纯 `numpy`/`scipy` (IRLS 通过 `scipy.linalg.solve`/`inv`) — 无 R 运行时，无 `rpy2`。它限于**二元体独立**项 (`edges`、`nodecov`、`nodematch`)：对于这些项，ERGM 伪似然精确分解为对变化统计量的逻辑回归，因此拟合是确定的和凸的。二元体*依赖*项 (`triangle`、`gwesp`、k-stars 作为估计目标、MCMC-MLE 通常) 超出 `ergm_mple` 的范围 — 它们在真实 `ergm` 中需要随机 MCMC-MLE，此处不复现。

它接入 socialverse 作为 `socialverse/tl/_network2.py` 中的两个注册函数：

- `sv.tl.ergm` — 在有向边列表上通过 MPLE 拟合 ERGM。每当请求的 `terms` 是 `{edges, nodecov, nodematch}` 的子集时 (端口的精确范围)，它委托给 `ergm_mple`；对于 `mutual`/`transitive` 项，它回退到模块内的变化统计量 + 逻辑回归引擎，因为这些是二元体依赖的且超出 `ergm_mple` 的保证。
- `sv.tl.network_statistics` — 直接委托给 `summary_formula` (观测充分统计量) 和 `triad_census` (16 型 MAN 普查) 用于有向边列表；当存在可用的边表时始终使用端口。

:::{admonition} 精度门控
:class: note

该端口在 8 个确定性精度测试中被固定在 R `ergm`/`sna` 的 `max_abs_err < 1e-6` (`socialverse/external/pyergm/tests/test_parity.py`)，针对规范的 Padgett 佛罗伦萨婚姻网络运行 (`ergm::flomarriage`，16 节点，无向) 加一个用于仅有向项的小 5 节点有向夹具。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pyergm import (
    ergm_mple,
    summary_formula,
    triad_census,
    TRIAD_CENSUS_LABELS,
)

# --- a tiny undirected network with a numeric vertex covariate ("wealth") ---
adjacency = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
], dtype=float)
wealth = np.array([10.0, 25.0, 40.0, 15.0, 5.0])

# 1) MPLE fit: edges + nodecov(wealth) -- dyad-independent, deterministic
fit = ergm_mple(adjacency, ["edges", ("nodecov", wealth)], directed=False)
print(fit.summary())        # per-term coefficient + model-based SE
print("log pseudo-likelihood:", fit.loglik)

# 2) observed sufficient statistics -- summary(net ~ edges + triangle + degree(0:3))
stats, labels = summary_formula(
    adjacency,
    terms=["edges", "triangle", ("degree", [0, 1, 2, 3])],
    directed=False,
)
for label, value in zip(labels, stats):
    print(f"{label}: {value:.0f}")

# 3) directed triad census on a small directed network
dir_adjacency = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
], dtype=float)
census = triad_census(dir_adjacency)
print(dict(zip(TRIAD_CENSUS_LABELS, census.astype(int))))

# --- equivalently, via the registered socialverse functions on a StudyState ---
# sv.tl.ergm(state, edges=edge_df, terms=["edges", "nodecov"], wealth=wealth_map)
# sv.tl.network_statistics(state, edges=edge_df, terms=["edges", "mutual"])
```

## R ↔ Python 对照表

| R (`ergm`) | socialverse | 备注 |
|---|---|---|
| `ergm(net ~ edges + nodecov("wealth"), estimate="MPLE")` | `ergm_mple(adjacency, ["edges", ("nodecov", wealth)])` 或 `sv.tl.ergm(state, edges=..., terms=["edges","nodecov"])` | 仅二元体独立；MPLE = 对变化统计量的逻辑回归 |
| `ergm(net ~ edges + nodematch("attr"))` | `ergm_mple(adjacency, ["edges", ("nodematch", attr)])` | 同质性项，二元体独立 |
| `summary(net ~ edges + triangle + degree(0:6) + kstar(2))` | `summary_formula(adjacency, ["edges", "triangle", ("degree", list(range(7))), ("kstar", 2)])` | 精确观测充分统计量，无估计 |
| `summary(net ~ mutual + istar(2) + ostar(2))` (有向) | `summary_formula(adjacency, ["mutual", ("istar", 2), ("ostar", 2)], directed=True)` | 有向二元体独立 + 简单二元体依赖计数 |
| `sna::triad.census(net)` | `triad_census(adjacency)` | Holland–Leinhardt 16 型 MAN 普查；标签在 `TRIAD_CENSUS_LABELS` 中 |
| `ergm(net ~ edges + mutual + gwesp(...), estimate="MCMC-MLE")` | 未复现 (随机；在 `sv.tl.ergm` 内回退到 socialverse 的内部变化统计量逻辑引擎) | 二元体依赖项需要难以处理的归一化常数 MCMC-MLE |

## 精度证据

8 个确定性精度测试在 `socialverse/external/pyergm/tests/test_parity.py` 中，针对 `reference.json` 中的 R `ergm`/`sna` 输出进行门控：

- `test_mple_coef`、`test_mple_se` — `flomarriage` 上 `edges + nodecov(wealth)` 的 MPLE 系数和基于模型的标准误，在 `max_abs_err < 1e-6` 处门控。
- `test_design_dimensions` — 二元设计形状 (16 节点无向网络的 120 个二元体 × 2 个预测因子) 和响应/标签健全性检查。
- `test_summary_undirected_stats`、`test_summary_undirected_labels` — `flomarriage` 上的观测充分统计量 (`edges`、`triangle`、`degree0`..`degree6`、`kstar2`、`nodecov`、`nodematch`)，精确相等 (0 容差 — 这些是整数/实数计数，而非估计)。
- `test_summary_directed_stats` — 5 节点有向夹具上的观测统计量 (`edges`、`mutual`、`istar2`、`ostar2`、`idegree1/2`、`odegree1/2`)，精确相等。
- `test_triad_census_counts`、`test_triad_census_labels_and_total` — 同一有向夹具上的 16 类有向三角形普查，精确相等加总计数健全性检查 (`sum == C(n,3)`)。

:::{admonition} 随机项超出范围
:class: warning

`ergm_mple` 仅涵盖**二元体独立**项 (`edges`、`nodecov`、`nodematch`)，伪似然精确分解并 MPLE 是凸逻辑回归 — 真正确定的，因此精度门控在 1e-6。二元体依赖项 (`triangle`、`gwesp`、k-star 估计、通用 MCMC-MLE) 和动态 SAOM 模型 (RSiena) 在 R 中是随机的且此端口**不**复现它们；`sv.tl.ergm` 对这些项回退到自己的近似变化统计量逻辑引擎，而不是声称 1e-6 精度。
:::

在本地复现门控：

```bash
Rscript socialverse/external/pyergm/tests/r_reference_driver.R
pytest socialverse/external/pyergm/tests/
```

## 在 socialverse 工作流中

日常中，调用 `sv.tl.ergm` 在有向边列表上拟合 ERGM (对于 `edges`/`nodecov`/`nodematch` 项，它静默地优先采用精度门控的 `ergm_mple` 路径，否则回退)，或调用 `sv.tl.network_statistics` 用于观测充分统计量加三角形普查。两者都在 `net` 类别中注册，`requires={"sources": ["datasets"]}` — 注册表执行该约定，`registry_lookup("ergm")` / `sv.list_functions()` 将在围绕它构建工作流之前确认活跃签名和 `requires`/`produces`。