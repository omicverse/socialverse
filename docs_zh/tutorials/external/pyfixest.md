# pyfixest — Python版`fixest`

> 高维固定效应OLS（`feols`）、带固定效应的Poisson伪极大似然（`fepois`）和Newey-West HAC OLS——社会科学家在面板/引力回归中常用的`fixest`估计量——无需R运行时可直接从Python调用，与R `fixest` 0.14.2对齐，精度`max_abs_err < 1e-6`。

## `fixest`的工作原理

`fixest`是让高维固定效应估计变得快速的R程序包：它通过迭代均值消除而非构建虚拟变量设计矩阵来吸收一个或多个固定效应维度（单位、时间、单位×时间等），并配合`fixest`特有的聚类稳健和多向聚类标准误。`feols()`是大部分现代经济学和政治学中面板/DID/TWFE规范的OLS主力；`fepois()`将相同的均值消除机制扩展到Poisson伪极大似然（PPML），这是引力模型、贸易流以及其他非负计数/乘法类结果（Santos Silva & Tenreyro）的标准估计量。社会科学家倾向使用它，原因是一旦拥有数千个固定效应水平，封闭式虚拟变量OLS就变得不可行，而且其小样本聚类修正已成为人们期望从"真正"的`fixest`回归表中看到的事实标准。

## 移植版本

- `feols(y, X, fe, cluster)` — 一维或二维固定效应OLS，通过within估计量（迭代组均值消除）实现，配合`fixest`兼容的聚类稳健vcov（`ssc`：`adj=TRUE`、`cluster.adj=TRUE`、`cluster.df="conventional"`、`fixef.K="nested"`）和within-R²。
- `fepois(y, X, fe, cluster=None, tol=1e-10, maxit=1000)` — Poisson伪极大似然（PPML），通过迭代重加权最小二乘法（IRLS）在Poisson对数链路上实现高维固定效应，固定效应通过*加权*交替within均值消除被浓缩，采用与`feols`相同的`fixest`嵌套聚类小样本修正。
- `newey_west(y, X, lag, add_intercept=True, order=None)` — OLS配合Newey-West（Bartlett核）HAC vcov，匹配`feols(y ~ X)`在`fixest`默认`ssc(adj=TRUE)`下以`vcov = NW(lag) ~ t`重新总结的结果。
- `demean(M, fe_codes, tol=1e-12, maxit=100000)` — 底层偏离出机制（跨一个或多个固定效应维度的交替投影）；主要导出以供重用/测试而非直接终端用户使用。

全部四个均为纯`numpy`（无R运行时、无`rpy2`、无编译扩展）。该移植版本通过`socialverse/tl/_econ.py`中两个注册的流程函数连接到socialverse：

- `sv.tl.replicate` — 端到端AER风格复现流程（平衡表→基准TWFE→稳健性矩阵→发表表）。其基准TWFE步骤首先尝试`pyfixest`移植版本（`_twfe_pyfixest_port`，调用`feols`），如果已安装真实`pyfixest`程序包则回退，最后回退到`statsmodels` OLS路径——将结果写入`models['twfe']`。当有时间索引时，它还额外附加一个Newey-West HAC配应（`_newey_west_hac`，调用`newey_west`）到`diagnostics['newey_west']`。
- `sv.tl.poisson_fe` — 带固定效应的PPML，通过`_ppml_pyfixest_port`调用移植版本的`fepois`，写入`models['fepois']`。

:::{admonition} 奇偶校验门
:class: note

该移植版本与R `fixest` 0.14.2对齐，在6个确定性奇偶校验测试（`socialverse/external/pyfixest/tests/test_parity.py`）上`max_abs_err < 1e-6`。
:::

## 快速开始

```python
import numpy as np

from socialverse.external.pyfixest import feols, fepois, newey_west

# --- 一个微小合成面板：5个单位 x 4个时期 -------------------------
rng = np.random.default_rng(0)
n_id, n_t = 5, 4
idv = np.repeat(np.arange(1, n_id + 1), n_t)          # 单位id（一维FE）
timev = np.tile(np.arange(1, n_t + 1), n_id)           # 时间周期
x = rng.normal(size=n_id * n_t)
y = 2.0 * x + idv * 0.3 + rng.normal(scale=0.5, size=n_id * n_t)

# --- 一维FE OLS，按单位聚类 -------------------------------------
r1 = feols(y, x, fe=idv, cluster=idv)
print("coef:", r1["coef"], "se:", r1["se"], "within R2:", r1["within_r2"])

# --- 二维FE OLS（单位 + 时间），按单位聚类 ------------------------
r2 = feols(y, x, fe=[idv, timev], cluster=idv)
print("twfe coef:", r2["coef"], "n_clusters:", r2["n_clusters"])

# --- PPML带固定效应（计数结果） --------------------------------
counts = rng.poisson(lam=np.exp(0.4 * x)).astype(float)
r3 = fepois(counts, x, fe=idv, cluster=idv)
print("ppml coef:", r3["coef"], "deviance:", r3["deviance"], "n_iter:", r3["n_iter"])

# --- OLS配合Newey-West（Bartlett核）HAC SE，按时间排序 ---------
X2 = np.column_stack([x, rng.normal(size=n_id * n_t)])
r4 = newey_west(y, X2, lag=2, order=timev)
print("nw coef:", r4["coef"], "nw se:", r4["se"])
```

等效地，通过注册的socialverse流程（在幕后使用相同的`feols`/`fepois`移植版本，具有自动schema解析和回退）：

```python
import socialverse as sv

state = sv.StudyState()
state = sv.tl.replicate(state, data=df, outcome="y", treatment="x",
                         unit="id", time="time", cluster="id")
twfe = state.models["twfe"]           # backend == "pyfixest"（当使用移植版本路径时）
```

## R ↔ Python对照表

| R（`fixest`） | socialverse | 注释 |
|---|---|---|
| `feols(y ~ x \| id, cluster = ~id)` | `feols(y, x, fe=id, cluster=id)` | 一维FE，无截距列（由FE吸收） |
| `feols(y ~ x \| id + time, cluster = ~id)` | `feols(y, x, fe=[id, time], cluster=id)` | 二维FE，`fe=`为分组向量列表 |
| `fepois(y ~ x \| id, cluster = ~id)` | `fepois(y, x, fe=id, cluster=id)` | PPML；`cluster=None`默认使用第一个FE维度，匹配fixest的默认行为 |
| `feols(y ~ x1 + x2, vcov = NW(lag) ~ t)` | `newey_west(y, X, lag=lag, order=t)` | Newey-West HAC OLS；`add_intercept=True`添加fixest隐含的常数 |
| `summary(fit)$coeftable` | `result["coef"]`、`result["se"]` | 按斜率系数/SE数组，按`X`列顺序 |
| `fit$sigma2` / within-R² | `result["within_r2"]` | 仅限`feols` |
| `fit$deviance` | `result["deviance"]` | 仅限`fepois` |
| 高层驱动函数 | `sv.tl.replicate(...)`、`sv.tl.poisson_fe(...)` | schema解析，若不满足前置条件则优雅回退 |

## 奇偶校验证据

`socialverse/external/pyfixest/tests/test_parity.py`中6个奇偶校验测试，每个都断言对`socialverse/external/pyfixest/tests/reference.json`的`max_abs_err < 1e-6`（由`r_reference_driver.R`针对`fixest` 0.14.2生成）：

- `test_oneway_id` — 一维FE `feols`：系数、聚类稳健SE、within-R²、`nobs`、`nparams`（嵌套小样本规则下的参数计数）。
- `test_oneway_id_clustertime` — 相同设计，但按时间维度而非单位FE进行聚类，以测试嵌套与非嵌套小样本修正路径。
- `test_twoway` — 二维（单位 + 时间）FE `feols`：系数、SE、within-R²、`nparams`。
- `test_fepois_oneway_id` — 一维FE `fepois`（PPML/IRLS）：系数、SE、收敛时的偏差、`nobs`。
- `test_newey_west_lag3` / `test_newey_west_lag2` — 两个截断滞后处的Newey-West HAC OLS：系数向量和HAC SE向量（由于滞后仅影响vcov，两个测试均确认系数与滞后无关）。

全部六个量都是确定性的封闭式（或IRLS收敛）数值——此移植版本中任何地方均不涉及自举或MCMC步骤，因此没有随机容差注意事项需要在此文档中说明。

再现：

```bash
# 从真实R程序包重新生成reference.json（需要安装R + fixest）
Rscript socialverse/external/pyfixest/tests/r_reference_driver.R

# 针对已提交的reference.json运行奇偶校验测试（无需R）
pytest socialverse/external/pyfixest/tests/test_parity.py -v
```

## 在socialverse工作流中

日常使用中，调用`sv.tl.replicate(state, ...)`进行完整TWFE复现流程（平衡表、基准估计、稳健性矩阵、Newey-West配应、发表表）或直接调用`sv.tl.poisson_fe(state, ...)`进行PPML固定效应拟合——两者都注册在`econ`类别中，当满足其前置条件（至少一个FE维度加一个聚类变量）时自动通过`feols`/`fepois`移植版本路由。注册表强制执行每个函数的`requires`/`produces`契约；在将其连接到更大流程前，使用`sv.list_functions()`或`registry_lookup("replicate")` / `registry_lookup("poisson_fe")`确认实时签名和I/O契约。
