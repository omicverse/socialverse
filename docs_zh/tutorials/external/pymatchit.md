# pymatchit — Python 中的 R `MatchIt`

> 因果推断的倾向性评分预处理——逻辑回归倾向性评分、贪心最近邻匹配、`WeightIt` 风格的平衡权重、马氏距离和完整的 `summary.matchit` 平衡表——现在可从 Python 调用，与 R `MatchIt` 实现 1e-6 级精度对等，无需 R 运行时。

## `MatchIt` 的作用

R 的 **MatchIt**（Ho, Imai, King, Stuart）是观察性因果推断的标准预处理工具：它通过选择或重新加权比较单元使得处理组和对照组在拟合结果模型之前在观察到的协变量上看起来相似，遵循"匹配作为非参数预处理"的方法论。其 `nearest`/`glm` 默认方法通过逻辑回归估计倾向性评分，并将每个处理单元贪心地与该评分上最近的可用对照配对；其 `summary()` 报告标准化平均差异、方差比及 eCDF 统计以检查匹配是否真正平衡了协变量。社会科学研究者在需要可防御、可审计的协变量平衡步骤之前采用差分均值或结果回归时经常使用它（通常与 `WeightIt` 一起用于 IPW 权重）。

## 实现

`socialverse.external.pymatchit` 暴露以下接口：

- `glm_logit_ps(X, y, max_iter=25, tol=1e-8)` — 通过 IRLS 拟合二项 logit GLM，完全按照 R 的 `glm.fit` 方式（截距优先设计矩阵、R 的 `mu=(y+0.5)/2` 初值、基于离差度的收敛）；返回 `(coef, fitted_ps)`。
- `nearest_match(distance, treat)` — 贪心 1:1 最近邻匹配无放回，处理单元按 `m.order="largest"`（降序倾向性评分）顺序处理；返回 `dict {treated_index: matched_control_index}`。
- `smd(x, treat, weights=None)` — 标准化平均差异，使用完整样本处理组标准差作为分母（MatchIt 的惯例），匹配后可选加权。
- `matchit(X, treat, covariates=None)` — 完整的 `nearest`/`glm` 管道：拟合倾向性评分、执行匹配并报告匹配前后 SMD；返回 `MatchItResult`。
- `MatchItResult` — 类数据类容器，包含 `ps_coef`、`distance`（拟合倾向性评分）、`pairs`、`weights`、`smd_before`、`smd_after`、`smd_vars`。
- `get_w_from_ps(ps, treat, estimand="ATE", treated=1)` — `WeightIt::get_w_from_ps` 实现，将二元倾向性评分转换为 ATE/ATT/ATC 平衡权重。
- `mahalanobis_dist(X, treat)` — `MatchIt:::mahalanobis_dist` 实现：缩放协变量、计算池化组内协方差（使用 MatchIt 的小样本修正），返回处理行与对照行之间的 `n1 x n0` 成对马氏距离矩阵。
- `balance_table(X, treat, weights=None, covariates=None)` — `summary.matchit(standardize=TRUE)` 的平衡列（`Std. Mean Diff.`、`Var. Ratio`、`eCDF Mean`、`eCDF Max`），针对每个协变量的调整前后数据。

该实现是纯 `numpy`/`scipy`，无需 R 运行时。它连接到 socialverse 的已注册因果推断函数 `sv.tl.psm`（`socialverse/tl/_matching.py`）：`psm` 优先选择 `pymatchit.glm_logit_ps` 进行倾向性评分步骤（仅当该实现抛出异常时才回退到 `statsmodels.Logit` 或纯 numpy IRLS），并且在执行自己的放回最近邻匹配 / IPW ATT 加权后，调用 `pymatchit.balance_table` 获取额外的 `Var. Ratio`/`eCDF` 平衡列和 `pymatchit.get_w_from_ps`（estimand `"ATT"`）以总结有效对照样本大小。

:::{admonition} 精度门槛
:class: note

该实现固定到 R `MatchIt` 4.7.2（`__matchit_reference_version__`）以在 12 个确定性精度测试中达到 `max_abs_err < 1e-6`（`socialverse/external/pymatchit/tests/test_parity.py`），针对经典 Lalonde NSW/CPS 观察性基准进行测试。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pymatchit import (
    matchit, get_w_from_ps, mahalanobis_dist, balance_table,
)

# A tiny synthetic covariate set: age, education, and two lagged earnings
# columns, plus a 0/1 treatment indicator (same layout as the Lalonde fixture
# the parity tests use: age, educ, re74, re75).
rng = np.random.default_rng(0)
n = 20
age  = rng.integers(20, 50, n).astype(float)
educ = rng.integers(8, 16, n).astype(float)
re74 = rng.normal(5000, 2000, n).clip(min=0)
re75 = rng.normal(5000, 2000, n).clip(min=0)
treat = np.array([1] * 8 + [0] * 12)

X = np.column_stack([age, educ, re74, re75])
covariates = ["age", "educ", "re74", "re75"]

# Full nearest/glm pipeline: PS-logit fit + greedy 1:1 matching + before/after SMD.
res = matchit(X, treat, covariates=covariates)
print("PS coefficients (intercept, age, educ, re74, re75):", res.ps_coef)
print("fitted propensity scores:", res.distance)
print("matched pairs (treated_idx -> control_idx):", res.pairs)
print("SMD before:", dict(zip(res.smd_vars, res.smd_before)))
print("SMD after: ", dict(zip(res.smd_vars, res.smd_after)))

# WeightIt-style ATT balancing weights from the same fitted propensity score.
w_att = get_w_from_ps(res.distance, treat, estimand="ATT", treated=1)
print("ATT weights (treated=1, control=ps/(1-ps)):", w_att)

# Mahalanobis distance matrix (treated x control) on the raw covariates.
D = mahalanobis_dist(X, treat)
print("Mahalanobis distance shape:", D.shape)  # (n_treated, n_control)

# Full summary.matchit(standardize=TRUE) balance table, before vs. matched.
bt_before = balance_table(X, treat, weights=None, covariates=covariates)
bt_after = balance_table(X, treat, weights=res.weights, covariates=covariates)
print("Var. Ratio before:", dict(zip(bt_before["vars"], bt_before["var_ratio"])))
print("Var. Ratio after: ", dict(zip(bt_after["vars"], bt_after["var_ratio"])))
```

## R ↔ Python 对照表

| R (`MatchIt`) | socialverse | 注释 |
|---|---|---|
| `glm(treat ~ ., family=binomial(), data=df)` | `glm_logit_ps(X, y)` → `(coef, fitted_ps)` | 截距自动前置；IRLS 逐元素匹配 R 的实现 |
| `matchit(treat ~ age + educ + re74 + re75, data=df, method="nearest", distance="glm")` | `matchit(X, treat, covariates=["age","educ","re74","re75"])` → `MatchItResult` | `m.order="largest"`，无放回，匹配 MatchIt 的默认值 |
| `match.data(m)$weights` | `MatchItResult.weights` | 匹配的处理/对照组为 1，否则为 0 |
| `summary(m)$sum.all` / `summary(m)$sum.matched` | `smd(x, treat, weights)` / `balance_table(X, treat, weights)` | `matchit()` 仅报告 SMD；`balance_table` 额外提供 Var. Ratio + eCDF |
| `WeightIt::get_w_from_ps(ps, treat, estimand="ATT")` | `get_w_from_ps(ps, treat, estimand="ATT", treated=1)` | 同样支持 `"ATE"` / `"ATC"` |
| `MatchIt:::mahalanobis_dist(formula, data)` | `mahalanobis_dist(X, treat)` | 返回 `n1 x n0` 处理行-对照行距离矩阵 |
| 工作流入口 | `sv.tl.psm(state, method="nn", ...)` | `socialverse/tl/_matching.py` 中已注册的 `StudyState` 函数；将倾向性评分步骤委派给 `glm_logit_ps`，将扩展平衡列委派给 `balance_table`/`get_w_from_ps` |

## 精度验证

12 个确定性精度测试位于 `socialverse/external/pymatchit/tests/test_parity.py`，门槛为 `max_abs_err < 1e-6`，对标由 R 驱动 `r_reference_driver.R` 在 Lalonde 基准上生成的参考 JSON（`reference.json`）。门槛内的数量包括：

- 倾向性评分逻辑回归系数（`glm_logit_ps` 通过 `matchit` 调用）；
- 拟合倾向性评分（`distance` 向量）；
- 匹配前**后**的标准化平均差异（`smd_before`/`smd_after`）；
- 全部三个 estimand 的 `WeightIt::get_w_from_ps` 平衡权重（ATE、ATT、ATC）；
- 完整的 `n1 x n0` `mahalanobis_dist` 成对距离矩阵；
- 匹配前后的 `summary.matchit(standardize=TRUE)` 平衡表（`Std. Mean Diff.`、`Var. Ratio`、`eCDF Mean`、`eCDF Max`）。

:::{admonition} 已文档化的并列打破限制
:class: warning

倾向性评分位相同的对照的精确**配对**未实现位级复现——它遵循 R MatchIt 的内部 C++ 扫描顺序而非该实现的。在 Lalonde 基准上，177/185 匹配对与 R 完全复现；该实现断言剩余 8 个分歧中的每一个都涉及两个对照，其倾向性评分间隙为 `0.0`（匹配对照倾向性评分的**多重集**与 R 相同），这就是为什么匹配后 SMD 及每个其他距离基平衡统计仍在 1e-6 级复现。这是对单个对身份的参考容限限制，而非门槛检查的任何数值量。
:::

在本地复现：

```bash
Rscript socialverse/external/pymatchit/tests/r_reference_driver.R
pytest socialverse/external/pymatchit/tests/
```

## 在 socialverse 工作流中

日常使用中，调用已注册的 `sv.tl.psm`（倾向性评分匹配 / IPW ATT 估计，含匹配前后协变量平衡）——它优先选择该实现进行倾向性评分拟合和扩展的 `Var. Ratio`/`eCDF` 平衡列，若该实现导入失败则优雅降级。注册表强制 `psm` 的 `requires`/`produces` 契约（`sources['datasets']`、`design['treatment']`、`variables['outcome']` → `models['psm']`、`diagnostics['balance']`）；使用 `registry_lookup` 或 `sv.list_functions()` 在将其纳入管道前确认活实例的签名。