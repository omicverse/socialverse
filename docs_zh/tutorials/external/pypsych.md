# pypsych — Python 中的 psych

> 经典心理测量学 — Cronbach's α、McDonald's ω、ICC 相间者信度、`corr.test` 和主轴因子分析 — 对 R 的 `psych` 进行奇偶门控，精度为 1e-6，无需 R 运行时。

## `psych` 的功能

`psych`（由 William Revelle 开发）是社会与行为科学中用于经典测试理论和探索性因子分析的重量级 R 包。研究人员使用它来评分多项目量表（`alpha`、`omega`），在信任复合得分前进行校验；量化编码/观察数据的相间者一致性（`ICC`）；一次性调用得到具有配对显著性的完整相关矩阵（`corr.test`）；以及运行主轴因子提取（`fa(fm="pa")`）作为主成分分析的传统心理测量替代方案。它是心理学、教育学和调查研究中心理测量学方法部分所引用的实际标准。

## 移植版本

- `smc(R)` — 每个变量对其余变量的平方多重相关，在相关矩阵上计算为 `1 - 1/diag(solve(R))`（在内部用于因子分析共同度的初始化）。
- `cronbach_alpha(X)` — 从原始项目矩阵的 `psych::alpha` 总行中获取：`raw_alpha`、`std_alpha`、`G6`（Guttman's lambda-6）和 `average_r`（平均项目间相关）。不进行自动项目反向，与 `check.keys=FALSE` 一致。
- `fa_pa(R, nfactors=1, min_err=1e-3, max_iter=50)` — 主轴因子分析（`fa(fm="pa")`）：迭代主特征向量载荷，使对角线重置为模型共同度，直至共同度和的变化低于 `min_err`。返回 `loadings`、`communality`、`uniqueness`。
- `omega_total(R, communality=None, nfactors=1)` — 从单因子 PA 解计算的 McDonald's ω_total：`1 - Σ(1 - h²)/sum(R)`。
- `ICC(ratings, alpha=0.05)` — 从被试×评分者矩阵经过双因素方差分析得到的六个 Shrout & Fleiss 组内相关系数（`ICC1`、`ICC2`、`ICC3`、`ICC1k`、`ICC2k`、`ICC3k`），各项包括 F 比值、自由度、p 值和置信区间。
- `corr_test(x)` — Pearson 相关矩阵加上配对样本量、t 值、原始双尾 p 和标准误，复制 `psych::corr.test(method="pearson", normal=TRUE)`（未调整的 p — Holm 调整和 Fisher-z 置信区间不在此移植版本中）。

该移植版本采用纯 numpy/scipy 实现 — 不依赖 rpy2、无需 R 安装、运行时不调用 Rscript 子进程。它被集成到 socialverse 中作为四个注册函数的数值计算后端：

- `sv.tl.reliability` — 在内部调用 `cronbach_alpha` 和 `omega_total` 以构建完整的内部一致性报告。
- `sv.tl.icc` — 直接在解析后的被试×评分者矩阵上调用 `pypsych.ICC`。
- `sv.tl.correlation_test` — 直接在解析后的数值变量上调用 `pypsych.corr_test`。
- `sv.tl.efa` 带上 `method="pa"`（或 `fm="pa"`）— 调用 `pypsych.fa_pa` 进行 R 精确的主轴提取（不带此参数的 `efa` 默认值是主成分提取，不同于此移植版本）。

:::{admonition} 奇偶门控
:class: note

pypsych 与 R `psych` 2.6.5 固定在 `max_abs_err < 1e-6` 精度，通过 8 个确定性奇偶测试（`socialverse/external/pypsych/tests/test_parity.py`），在 `psych::bfi` 前 5 项目测试数据（2709 个完整个案被试）加上固定的 6 被试×4 评分者 ICC 测试数据上验证。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pypsych import cronbach_alpha, fa_pa, omega_total, ICC, corr_test

# --- a tiny 5-item, 8-subject congeneric scale (toy data, not the bfi fixture) ---
rng = np.random.default_rng(0)
true_trait = rng.normal(size=8)
items = np.column_stack([
    true_trait * 0.9 + rng.normal(scale=0.4, size=8) for _ in range(5)
])

alpha = cronbach_alpha(items)
print("Cronbach's alpha:", alpha)
# {'raw_alpha': ..., 'std_alpha': ..., 'G6': ..., 'average_r': ...}

# fa_pa works on a correlation matrix, not the raw item matrix
R = np.corrcoef(items, rowvar=False)
fa = fa_pa(R, nfactors=1)
print("PA loadings:", fa["loadings"])
print("PA communality:", fa["communality"])

omega = omega_total(R, communality=fa["communality"])
print("McDonald's omega_total:", omega)

# --- inter-rater reliability: 6 subjects rated by 4 judges ---
ratings = rng.integers(1, 8, size=(6, 4)).astype(float)
icc = ICC(ratings)
print("ICC types:", icc["type"])
print("ICC point estimates:", icc["ICC"])

# --- correlation matrix with pairwise significance ---
ct = corr_test(items)
print("r matrix:\n", ct["r"])
print("raw two-sided p matrix:\n", ct["p"])

# --- equivalently, via the registered sv.tl functions on a StudyState ---
import socialverse as sv
import pandas as pd

df = pd.DataFrame(items, columns=[f"item{i}" for i in range(1, 6)])
state = sv.StudyState()
state.write("sources", "datasets", df)

state = sv.tl.reliability(state)
print(state.diagnostics["reliability"]["cronbach_alpha"])

state = sv.tl.efa(state, method="pa", n_factors=1)
print(state.models["efa"]["estimator"])  # "pypsych.fa_pa (R psych::fa fm='pa', principal-axis)"
```

## R ↔ Python 对应表

| R (`psych`) | socialverse | 说明 |
|---|---|---|
| `alpha(items)$total` | `pypsych.cronbach_alpha(items)` / `sv.tl.reliability(state)` | 返回 `raw_alpha`/`std_alpha`/`G6`/`average_r`；不进行项目反向（`check.keys=FALSE`） |
| `omega(items)$omega.tot` | `pypsych.omega_total(R, communality=...)` / `sv.tl.reliability(state)` | 该移植版本的 ω_tot 是**公开的** `fa(fm="pa")` 解的闭式形式，不同于 psych 的内部自动项目反向 + GPArotation minres 管道 — 见奇偶证据 |
| `fa(items, nfactors=1, fm="pa")` | `pypsych.fa_pa(R, nfactors=1)` / `sv.tl.efa(state, method="pa")` | 接收相关矩阵 `R`，而非原始项目矩阵；返回 `loadings`、`communality`、`uniqueness` |
| `ICC(ratings, lmer=FALSE)` | `pypsych.ICC(ratings, alpha=0.05)` / `sv.tl.icc(state)` | `ratings` 为被试×评分者；返回全部六种 ICC1/2/3(+k) 类型，含 F/df/p/CI |
| `corr.test(x, method="pearson")` | `pypsych.corr_test(x)` / `sv.tl.correlation_test(state)` | 返回原始（未调整的）双尾 p；psych 的 Holm 调整和 Fisher-z 置信区间未在此移植版本中 |
| `smc(R)` | `pypsych.smc(R)` | 内部辅助函数，也可单独使用 |

## 奇偶证据

该移植版本通过 `socialverse/external/pypsych/tests/test_parity.py` 中的 8 个确定性奇偶测试与 R `psych` 2.6.5 进行检验，由 `r_reference_driver.R` 驱动并与 `reference.json` 对比：

- **`cronbach_alpha`**：在 `psych::bfi` 前 5 项目完整个案测试数据（n = 2709）上的 `raw_alpha`、`std_alpha`、`G6`、`average_r` — 精度门控为 1e-6。
- **`fa_pa`（`fa(fm="pa")`，nfactors=1）**：同一测试数据的相关矩阵上的 `loadings`、`communality`、`uniqueness` — 精度门控为 1e-6。
- **`omega_total`**：两种变体均在 1e-6 精度下与 R 驱动的匹配闭式计算进行门控 — 一种在原始测试数据的 PA 共同度上，另一种在应用与 `psych::omega()` 内部相同的项目反向后。
- **`ICC`**：在固定的 6 被试×4 评分者矩阵上的全部六个点估计（`ICC1/2/3`、`ICC1k/2k/3k`）加上 F、df1、df2、p 和 `MSW` — 精度门控为 1e-6；置信区间（`lower`/`upper`、闭式 qf/Satterthwaite）在单独的测试中精度门控为 1e-6。
- **`corr_test`**：5 项目完整个案测试数据上的 Pearson r 矩阵、原始双尾 p 和标准误 — 精度门控为 1e-6；t 矩阵的有限项精度门控为 1e-6，完全匹配哪些单元格为 `+Inf`（对角线）。

:::{admonition} 文档化的参考容差限
:class: warning

`psych::omega()` 的 `omega.tot` **不**能从公开的 `fa(fm='pa')` 解按元素复现：psych 的 `omega()` 运行单独的内部管道（自动负值项目反向，然后是 GPArotation minres 提取），而不是此移植版本所暴露的单因子 PA 解。套件中的一个测试（`test_omega_close_to_psych_reference`）检验该移植版本的 ω_tot 在应用**相同的**项目反向后，落在 `psych::omega()` 报告的 `omega.tot` 的较松弛的 **1e-3** 容差范围内 — 确认它是同一个量，而不是 1e-6 奇偶声明。原始（未反向）测试数据的 ω 合理地与 `psych::omega()` 相差约 0.13，这是意图上不进行门控的。
:::

要复现：

```bash
Rscript socialverse/external/pypsych/tests/r_reference_driver.R
pytest socialverse/external/pypsych/tests/test_parity.py -v
```

## 在 socialverse 工作流中

日常使用中，在 `StudyState` 上调用 `sv.tl.reliability`、`sv.tl.icc`、`sv.tl.correlation_test` 或 `sv.tl.efa(method="pa")`，而不是直接导入 pypsych — 注册表强制执行每个函数的 `requires`/`produces` 合约（例如 `reliability` 需要 `sources['datasets']` 并产生 `diagnostics['reliability']`），这样解析器就无法在实际运行前引用信度系数。在针对其进行脚本编写前，使用 `sv.list_functions()` 或 `registry_lookup` 确认实时签名和类别（`psychometrics`）。