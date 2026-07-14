# pysurvival — R 的 `survival` 包在 Python 中的实现

> Kaplan-Meier、Cox 比例风险模型、条件（精确）logistic 回归和参数化 AFT 模型 — R `survival` 包的核心功能，可从 Python 调用，对等精度达 1e-6，无需 R 运行时。

## `survival` 做什么

`survival`（Therneau）是 R 中事件历史分析的参考实现，也是方法论论文在"我们在 R 中拟合了 Cox 模型"时引用的估计器。社会科学研究者用它进行任何生存时间到事件结果的持续时间/风险分析 — 工作分离、抗争发生、婚姻解体、政策采纳、再犯 — 处理右删失数据。其 `coxph`（Efron 并列处理偏然然似然）和 `survfit`（Kaplan-Meier 及 Greenwood 方差）是事实上的标准，而 `clogit`/`survreg` 将同样的机制扩展到匹配的病例对照设计和完全参数化的加速失效时间（AFT）模型。

## 移植内容

- `km(time, event, conf_level=0.95)` — Kaplan-Meier 估计器，使用 Greenwood 累积风险标准误和 log 变换置信带；对应 `survfit(Surv(time,status)~1)`。
- `coxph(time, event, X, ties="efron", maxiter=30, eps=1e-9)` — Cox 比例风险回归，通过 Newton-Raphson 法求偏然然似然，支持 **Efron**（R 默认）和 **Breslow** 两种并列处理方法；同时返回 Harrell 的一致性指数。
- `clogit(y, strata, X, maxiter=30, eps=1e-9)` — 条件（固定效应）logistic 回归，用于分层匹配数据，通过精确条件似然实现（等价于带 `method="exact"` 的分层 Cox 模型）。
- `survreg(time, status, X, dist="weibull", maxiter=50, eps=1e-10)` — 参数化加速失效时间最大似然估计，支持 Weibull、指数或对数正态误差分布，在 log 时间尺度上进行。

每个函数返回一个小的 `dataclass`（`KMResult`、`CoxResult`、`ClogitResult`、`SurvregResult`），包含系数、标准误、z/p 值、方差协方差矩阵和对数似然。实现完全基于 `numpy`/`scipy` — 无需 R 运行时、无需 `rpy2`、无需子进程。

它被集成到 socialverse 中，作为 `socialverse/tl/_longitudinal.py` 中三个已注册函数的数值后端：

- `sv.tl.survival` — 委派给 `pysurvival.coxph`（Efron 并列处理），用于支撑 `models["cox"]` 输出的非时变 Cox 拟合（时变/Andersen-Gill 设计改由 `statsmodels.PHReg` 后退，因为该移植不覆盖左截断）。
- `sv.tl.conditional_logit` — 完全委派给 `pysurvival.clogit`。
- `sv.tl.aft_survreg` — 完全委派给 `pysurvival.survreg`。

:::{admonition} 对等性门控
:class: note

本移植针对 R `survival` 3.8.3 的对等精度门控为 `max_abs_err < 1e-6`，在 **9** 个确定性对等性测试中验证（`socialverse/external/pysurvival/tests/test_parity.py`）。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pysurvival import km, coxph, clogit, survreg

# --- toy right-censored duration data (10 units) ---
rng = np.random.default_rng(0)
time = np.array([5, 6, 6, 8, 9, 10, 12, 14, 14, 20], dtype=float)
event = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1])       # 1 = event, 0 = censored
age = np.array([50, 62, 40, 58, 47, 65, 39, 71, 44, 55], dtype=float)
sex = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

# 1) Kaplan-Meier — matches survfit(Surv(time, status) ~ 1)
km_fit = km(time, event)
print(km_fit.time)      # unique event/censoring times
print(km_fit.surv)      # KM survival estimate S(t)
print(km_fit.median)    # median survival time

# 2) Cox PH — Efron ties (R default)
X = np.column_stack([age, sex])
cox_fit = coxph(time, event, X, ties="efron")
print(cox_fit.coef, cox_fit.se, cox_fit.pval)   # log-HR, SE, p-value per covariate
print(np.exp(cox_fit.coef))                      # hazard ratios
print(cox_fit.concordance)                        # Harrell's C

# 3) Conditional logistic regression — matched sets
y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
strata = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
Xc = np.column_stack([age, sex])
clogit_fit = clogit(y, strata, Xc)
print(clogit_fit.coef, np.exp(clogit_fit.coef))   # conditional log-OR, OR

# 4) Parametric AFT — Weibull, with an explicit intercept column
Xs = np.column_stack([np.ones_like(time), age, sex])
aft_fit = survreg(time, event, Xs, dist="weibull")
print(aft_fit.coef, aft_fit.scale, aft_fit.loglik)

# Equivalent day-to-day call inside a socialverse workflow:
# state = sv.tl.survival(state, time="time", event="event", covariates=["age", "sex"])
```

## R ↔ Python 对照表

| R（`survival`） | socialverse | 说明 |
|---|---|---|
| `survfit(Surv(time, status) ~ 1)` | `pysurvival.km(time, event)` | 在每个观测到的唯一时间点给出行；`std_err` 是累积风险的 Greenwood 标准误，遵循 R 的惯例 |
| `coxph(Surv(time,status) ~ x, ties="efron")` | `pysurvival.coxph(time, event, X, ties="efron")` / `sv.tl.survival(...)` | Newton-Raphson 偏然然似然；`sv.tl.survival` 对非时变设计使用此方法 |
| `coxph(..., ties="breslow")` | `pysurvival.coxph(time, event, X, ties="breslow")` | Breslow 并列处理 |
| `clogit(case ~ x + strata(id))` | `pysurvival.clogit(y, strata, X)` / `sv.tl.conditional_logit(...)` | 匹配集合上的精确条件似然 |
| `survreg(Surv(time,status) ~ x, dist="weibull")` | `pysurvival.survreg(time, status, X, dist="weibull")` / `sv.tl.aft_survreg(...)` | `X` 必须包含截距列，与 R 的 `~ x` 设计矩阵相匹配 |
| `survreg(..., dist="exponential")` | `pysurvival.survreg(..., dist="exponential")` | 尺度参数固定为 1 |
| `survreg(..., dist="lognormal")` | `pysurvival.survreg(..., dist="lognormal")` | log 时间上的高斯误差 |
| `summary(km)$table["median"]` | `KMResult.median` | 首次达到 `surv <= 0.5` 的时间 |
| `cox$concordance["concordance"]` | `CoxResult.concordance` | Harrell 的 C 指数 |

## 对等性验证

`socialverse/external/pysurvival/tests/test_parity.py` 中的 **9** 个对等性测试，每个都对照 R `survival` 3.8.3 的输出（由 `tests/r_reference_driver.R` 在规范数据集 `lung` 和 `infert` 上生成的 `reference.json`）断言 `max_abs_err < 1e-6`：

- **KM**：`time`、`n.risk`、`n.event`、`surv`、`std.err` 在 1e-6 精度，以及报告的中位数的精确匹配；log 变换 CI 边界（`lower`/`upper`）单独检查在 1e-6 精度。
- **Cox (Efron)**：系数、标准误、z 统计量和（零、拟合）对数似然对，均在 1e-6 精度。
- **Cox (Breslow)**：系数和对数似然在 1e-6 精度。
- **Cox 一致性**：在较宽松的 1e-3 容差下检查（见下方警告）。
- **clogit**：`infert` 匹配病例对照数据上的系数、标准误和（零、拟合）条件对数似然，在 1e-6 精度。
- **survreg**（Weibull、指数、对数正态）：`lung` 数据集上的系数、尺度参数、标准误（按 R 的 `vcov` 顺序 `[coef..., Log(scale)]`）和对数似然，均在 1e-6 精度。

:::{admonition} 一个宽松容差的量
:class: warning

`cox.concordance`（Harrell 的 C）在 **1e-3** 精度门控，而非 1e-6 — R 的 `coxph` 一致性使用内部并列处理/稳健方差算法，该移植用直接配对比较近似；点估计接近但非按位相同。所有系数/SE/对数似然量保持在完整的 1e-6 门控。
:::

本地复现：

```bash
Rscript socialverse/external/pysurvival/tests/r_reference_driver.R
pytest socialverse/external/pysurvival/tests/
```

## 在 socialverse 工作流中

调用 `sv.tl.survival` 进行 Cox + KM，调用 `sv.tl.conditional_logit` 进行匹配集合固定效应 logit，调用 `sv.tl.aft_survreg` 进行参数化 AFT — 日常使用中这些是主要入口点，不是直接调用 `pysurvival` 函数。每个都用显式的 `requires`/`produces` 契约注册，注册表在调用时强制执行；运行 `sv.list_functions()` 或 `registry_lookup("survival")`（以及 `"conditional_logit"` / `"aft_survreg"`）确认实时签名和默认 kwargs，然后再针对它们编写脚本。
