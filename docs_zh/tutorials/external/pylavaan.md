# pylavaan — Python 中的 lavaan

> 利用标记变量最大似然法进行确认因子分析，配备 lavaan 的完整适配指数组和评分检验修正指数，与 R `lavaan` 在元素级 1e-6 误差范围内对齐 — 无需 R 运行时环境。

## `lavaan` 的功能

`lavaan` 是用于潜变量建模的标准 R 包 — 包括确认因子分析（CFA）和结构方程模型（SEM），通过最大似然法拟合，模型采用紧凑的公式式语法指定（`f =~ x1 + x2 + x3`）。社会科学研究者使用它来检验假设的测量结构（例如调查项目加载到潜在构造）是否符合观察到的协方差结构，获得用于构念效度的标准化因子负荷，以及读取审稿人在心理测量或 SEM 论文中期望的完整适配指数组（χ²、CFI、TLI、RMSEA、SRMR、AIC/BIC）。`lavaan` 的 `modindices()` 进一步通过单变量评分（拉格朗日乘数）检验与预期参数变化（EPC）标记出规范错误的路径 — 模型遗漏但数据需要的交叉加载或残差协方差。

## 移植版本

`socialverse.external.pylavaan` 提供：

- **`parse_model(model)`** — 解析 lavaan `=~` 测量模型语法为有序的 `{factor: [indicators]}` 映射（因子协方差自动添加，匹配 `cfa()` 的默认行为）。
- **`cfa(model, data, meanstructure=False)`** — 拟合标记变量识别的 ML CFA（仅协方差结构）并返回 `CFAResult`。
- **`CFAResult`** — 拟合模型对象；提供 `.parameter_estimates()`、`.standard_errors()`、`.fit_measures()` 和 `.modification_indices()` 方法。
- **`fit_measures(fitted_cfa)`** — 完整的 `lavaan::fitMeasures()` 组合，作为字典返回（点号替换为下划线：`rmsea_ci_lower`、`bic2`、`logl`、`gfi` 等）。
- **`modification_indices(fitted_cfa, sort=True, minimum_value=0.0)`** — 单变量评分检验修正指数 + 每个当前固定参数的 EPC（交叉加载和残差协方差），镜像 `lavaan::modindices()`。

它是纯 `numpy`/`scipy` 实现 — 正态理论 ML 不一致性通过 L-BFGS-B 最小化，通过 Fisher 评分牛顿精化、期望信息标准误和一个 R `uniroot` 式的 Brent 根查找器用于 RMSEA 置信区间 — **无需 R 运行时依赖**。

它已接入 socialverse，作为已注册 `sv.tl.cfa` 函数（`socialverse/tl/_psychometrics.py`）的主要后端：`cfa()` 拟合原始数据，`fit_measures()` 填充 `models['cfa']['fit_measures']` 和平面 `diagnostics/fit_indices` 键（`CFI`、`RMSEA`、`SRMR`、`AIC`、`BIC` 等），`modification_indices()` 填充 `models['cfa']['modification_indices']`。若移植版本导入或拟合因任何原因失败，`sv.tl.cfa` 将回退到 `semopy` 或内部基于 statsmodels 的分块近似。

:::{admonition} 配对度控制门
:class: note

该移植版本与 R `lavaan` 在 8 个确定性配对度测试中固定为 `max_abs_err < 1e-6`。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pylavaan import cfa, fit_measures, modification_indices

# Holzinger-Swineford 式 3 因子 / 9 指标测量模型，
# lavaan CFA 的标准示例。内联构建小型综合数据集。
rng = np.random.default_rng(0)
n = 300
visual = rng.normal(size=n)
textual = rng.normal(size=n)
speed = rng.normal(size=n)

def item(latent, loading=0.8, noise=0.6):
    return loading * latent + noise * rng.normal(size=n)

data = {
    "x1": item(visual), "x2": item(visual), "x3": item(visual),
    "x4": item(textual), "x5": item(textual), "x6": item(textual),
    "x7": item(speed), "x8": item(speed), "x9": item(speed),
}

model = """visual  =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
speed   =~ x7 + x8 + x9"""

# 通过 ML 拟合（标记变量识别、仅协方差、N 除数）
res = cfa(model, data)

# 非标准化 + 标准化（std.lv、std.all）加载，按 lavaan 的
# parameterEstimates() 顺序排列
for row in res.parameter_estimates():
    if row["op"] == "=~":
        print(row["lhs"], row["rhs"], row["est"], row["std_all"])

# 完整的 lavaan fitMeasures() 组合：chisq、df、cfi、tli、rmsea（+ CI）、
# srmr、aic、bic、bic2、logl、gfi、agfi、nfi 等
fm = fit_measures(res)
print(fm["cfi"], fm["rmsea"], fm["rmsea_ci_lower"], fm["rmsea_ci_upper"], fm["srmr"])

# 每个固定参数的评分检验修正指数 + EPC
#（交叉加载和残差协方差），按 MI 降序排列
mi = modification_indices(res, sort=True)
print(mi[0])  # {'lhs': ..., 'op': '=~'/'~~', 'rhs': ..., 'mi': ..., 'epc': ...}
```

或者，通过已注册的 socialverse 函数调用：

```python
import socialverse as sv

state = sv.tl.cfa(state, model_spec={
    "visual": ["x1", "x2", "x3"],
    "textual": ["x4", "x5", "x6"],
    "speed": ["x7", "x8", "x9"],
})
cfa_model = state.read("models", "cfa")
print(cfa_model["backend"])            # 当移植版本成功拟合时为 "pylavaan"
print(cfa_model["fit_measures"])       # 完整组合，lavaan 风格的键
print(cfa_model["modification_indices"])
```

## R ↔ Python 词汇表

| R（`lavaan`） | socialverse | 说明 |
|---|---|---|
| `cfa(model, data)` | `pylavaan.cfa(model, data)` | 相同的 `=~` 语法；标记变量识别、ML、仅协方差（无均值结构）、N（有偏）除数 |
| `parameterEstimates(fit)` | `res.parameter_estimates()` | 行包含 `est`、`se`、`std_lv`、`std_all`，按顺序排列：加载 → 残差方差 → 因子方差 → 因子协方差 |
| `fitMeasures(fit)` | `pylavaan.fit_measures(res)` / `res.fit_measures()` | 字典，按 lavaan 风格键（点号 → 下划线）：`chisq`、`df`、`cfi`、`tli`、`rmsea`、`rmsea_ci_lower/upper`、`rmsea_pvalue`、`srmr`、`aic`、`bic`、`bic2`、`logl`、`gfi`、`agfi`、`nfi` |
| `modindices(fit)` | `pylavaan.modification_indices(res)` / `res.modification_indices()` | `{lhs, op, rhs, mi, epc}` 的列表，可排序，`minimum_value` 过滤器 |
| `lavParseModelString(model)` | `pylavaan.parse_model(model)` | 仅 `=~`（被测量者）运算符，`cfa()` 足够 |
| —（由 `cfa()` 内部调用） | `sv.tl.cfa(state, model_spec=...)` | 已注册的 socialverse 函数；包装 `pylavaan.cfa` + `fit_measures` + `modification_indices`，写入 `models['cfa']` / `diagnostics/fit_indices` |

## 配对度证据

8 个确定性配对度测试在经典 3 因子 / 9 指标（visual/textual/speed）测量模型上将移植版本与 R `lavaan` 在 `max_abs_err < 1e-6` 处进行对齐：

- 非标准化加载和方差（`est`）
- 标准化加载，包括 `std.lv` 和 `std.all`
- 紧凑适配指数集（`chisq`、`df`、`cfi`、`tli`、`rmsea`、`srmr`）
- 完整 `fitMeasures()` 组合元素级（`fmin`、`logl`、`unrestricted_logl`、`aic`、`bic`、`bic2`、`chisq`、`pvalue`、`baseline_chisq`、`cfi`、`tli`、`nfi`、`gfi`、`agfi`、`rmsea`、`rmsea_ci_lower`、`rmsea_ci_upper`、`rmsea_pvalue`、`srmr`），加上 `npar`、`df`、`baseline_df` 上的精确整数检查
- 修正指数**公式**，通过将 lavaan 本身的拟合系数代入移植版本的评分检验机制中进行确定性验证，并将 MI/EPC 与 lavaan 直接比较（将数学与任何优化器差异隔离）
- 修正指数**排名**，检查前 MI 建议（`visual =~ x9`）是否与 lavaan 的前排入相匹配

:::{admonition} 端到端修正指数容差
:class: warning

R `lavaan` 的 `nlminb` 优化器在有限梯度容差（~5e-7）处停止，而移植版本的牛顿精化 ML 解驱动梯度至 ~1e-13。该优化器松弛放大器，由 N 放大，传播到从 lavaan *本身*端到端拟合计算的任何量 — 包括从移植版本独立优化估计计算的 MI/EPC。因此，端到端修正指数测试使用记录的 `5e-4` 容差而非 1e-6；公式本身上的 1e-6 门（使用 lavaan 存储的系数作为输入）是实际固定数学的位置。
:::

重现方法：

```bash
Rscript socialverse/external/pylavaan/tests/r_reference_driver.R
pytest socialverse/external/pylavaan/tests/test_parity.py -v
```

## 在 socialverse 工作流中

调用 `sv.tl.cfa(state, model_spec={...})` 进行日常 CFA 拟合 — 它在可用时驱动 pylavaan 移植版本，并将 `models['cfa']`（加载、`fit_measures`、`modification_indices`）和 `diagnostics/fit_indices` 写入 `StudyState`。注册表对 `cfa` 强制执行 `requires`/`produces` 合同；在脚本编写前使用 `sv.registry_lookup("cfa")` 或 `sv.list_functions()` 确认实时签名和 I/O 合同。
