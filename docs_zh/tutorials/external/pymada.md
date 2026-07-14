# pymada — 用 Python 实现 `mada`

> 二元诊断试验准确性 Meta 分析（Reitsma 模型、SROC/AUC 和 HSROC 系数）— 可从 Python 调用，与 R 的 `mada` 包在精度达 1e-6 级别完全相同，无需 R 运行时。

## `mada` 的功能

`mada`（Meta-Analysis of Diagnostic Accuracy，诊断准确性荟萃分析）是用于汇总以 2×2 表（TP/FN/FP/TN）形式报告的诊断试验研究的标准 R 包。与单独汇总敏感性和特异性的方法不同（该方法会忽略阈值效应在研究间引起的负相关），`mada::reitsma` 采用 Reitsma 等人（2005）提出的二元随机效应模型：对 Logit 变换后的敏感性和假阳性率建立联合线性混合模型，具有非结构化的研究间协方差矩阵，通过（分析）受限最大似然估计（`mvmeta` 实现）拟合。从拟合的模型中，`mada` 还推导出 Rutter–Gatsonis HSROC 参数化和 SROC 曲线下面积（AUC）— 这些是社会学家和流行病学家在筛查工具或诊断试验荟萃分析中报告的指标（例如汇总抑郁症筛查工具在多个验证研究中的敏感性/特异性）。

## 端口实现

`socialverse.external.pymada` 提供了以下接口：

- **`reitsma(data=None, TP=None, FN=None, FP=None, TN=None, correction=0.5, correction_control="all")`** — 在 Logit 尺度上拟合二元随机效应模型（分析 REML、非结构化 2×2 `Psi`、采用 R 自有 `vmmin`/BFGS 优化器和解析梯度）。返回包含 `coefficients`（汇总 Logit 敏感性、Logit FPR）、`vcov`、`Psi`、`se`、`sensitivity`、`false_pos_rate`、`logLik` 和 `par` 的字典。
- **`calc_hsroc_coef(fit)`** — 将 `reitsma` 拟合结果转换为 Rutter–Gatsonis HSROC 系数（`Theta`、`Lambda`、`beta`、`sigma2theta`、`sigma2alpha`），适用于无协变量情况。
- **`AUC(fit, fpr=None, sroc_type="ruttergatsonis")`** — Rutter–Gatsonis SROC 曲线下面积（`AUC`，默认在 `fpr = 1:99/100` 上积分）和部分 AUC（`pAUC`，在拟合的原始单元计数确定的观测 FPR 范围内积分）。

它是纯 `numpy`/`scipy` 实现 — REML 分析似然函数、其解析梯度、IGLS 起始值，甚至 R 的 `vmmin` BFGS 线搜索都直接用 Python 重新实现，运行时无需启动 R 进程。

在 socialverse 内部，它集成到 `sv.tl.dta_bivariate`（`socialverse/tl/_meta_dta.py`）中，后者调用 `external.pymada.reitsma` 和 `external.pymada.AUC` 作为首选后端，通过线性映射 `fpr = 1 - spec` 将端口的（Logit 敏感性、Logit FPR）基础转换为 socialverse 自有的（Logit 敏感性、Logit 特异性）基础。如果端口导入或拟合失败，`dta_bivariate` 会无声地回退到原生 Nelder-Mead 二元正态近似（输出中 `backend: "native"` 对应 `backend: "pymada"`）。`sv.tl.dta_glmm` 依次委托给 `dta_bivariate`。

:::{admonition} 完全性把关
:class: note

该端口在 8 个确定性完全性测试中与 R `mada` 对标，最大绝对误差 < 1e-6。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pymada import reitsma, AUC, calc_hsroc_coef

# 一个小的诊断准确性数据集：每个研究的 2×2 单元计数
# (TP/FN/FP/TN)，类似于 mada 内置的 AuditC 样本
data = {
    "TP": [17, 18, 18, 6, 34, 19, 19, 34, 6, 32],
    "FN": [3, 3, 2, 1, 5, 3, 1, 4, 1, 4],
    "FP": [1, 4, 2, 1, 12, 6, 3, 12, 1, 9],
    "TN": [10, 12, 10, 5, 33, 27, 12, 33, 5, 30],
}

# 拟合 Reitsma 二元随机效应模型（分析 REML）
fit = reitsma(data=data)

print("pooled sensitivity:", fit["sensitivity"])
print("pooled FPR:        ", fit["false_pos_rate"])
print("pooled specificity:", 1.0 - fit["false_pos_rate"])
print("coefficients (logit sens, logit fpr):", fit["coefficients"])
print("between-study covariance Psi:\n", fit["Psi"])
print("fixed-effects se:", fit["se"])

# 从同一拟合得出 Rutter-Gatsonis HSROC 系数
hs = calc_hsroc_coef(fit)
print("HSROC Theta/Lambda/beta:", hs["Theta"], hs["Lambda"], hs["beta"])

# 汇总 ROC 曲线下面积（及部分面积）
auc = AUC(fit)
print("AUC / pAUC:", auc["AUC"], auc["pAUC"])

# --- 通过已注册的 socialverse 函数的等价路径 ---
import pandas as pd
import socialverse as sv

df = pd.DataFrame(data)
state = sv.StudyState()
state.write("sources", "datasets", df)                 # 导入 2×2 表
state = sv.tl.dta_descriptives(state, tp="TP", fp="FP", fn="FN", tn="TN")
state = sv.tl.dta_bivariate(state, tp="TP", fp="FP", fn="FN", tn="TN")
summary = state.models["dta_bivariate"]
print(summary["backend"], summary["sensitivity"], summary["specificity"])
```

## R ↔ Python 对照表

| R (`mada`) | socialverse | 说明 |
|---|---|---|
| `reitsma(data)` | `socialverse.external.pymada.reitsma(data=...)` / `sv.tl.dta_bivariate(state, ...)` | 端口直接接收 `TP/FN/FP/TN` 或包含这些列的 `data=` 映射/DataFrame；`sv.tl.dta_bivariate` 包装它并重新映射 `fpr -> spec` |
| `fit$coefficients` | `fit["coefficients"]` | 端口中为 logit(sens)、logit(fpr)；`sv.tl.dta_bivariate` 将其写为 `mu_logit`（logit sens、logit spec） |
| `fit$vcov`, `fit$Psi` | `fit["vcov"]`, `fit["Psi"]` | R 的 JSON 转储中为列主序展平；端口返回 2×2 `numpy` 数组 |
| `summary(fit)`（汇总敏感性/特异性）| `fit["sensitivity"]`, `1 - fit["false_pos_rate"]` | `sv.tl.dta_bivariate` 也写出 `sens_ci`/`spec_ci`（Wald 置信区间，Logit 尺度上 1.96·SE） |
| `mada:::calc_hsroc_coef(fit)` | `calc_hsroc_coef(fit)` | 相同的键：`Theta`、`Lambda`、`beta`、`sigma2theta`、`sigma2alpha` |
| `AUC(fit)` | `AUC(fit)` | 返回 `{"AUC": ..., "pAUC": ...}`；`sv.tl.dta_bivariate` 在可用时将其存储在 `summary["auc"]` 下 |
| `data(AuditC); reitsma(AuditC)` | `sv.tl.dta_descriptives(state, ...)` 然后 `sv.tl.dta_bivariate(state, ...)` | `dta_descriptives` 用 0 单元连续性修正计算每个研究的敏感性/特异性/DOR/LR+/LR-；`dta_bivariate` 需要其 `models["dta"]` 输出 |

## 完全性证据

8 个确定性完全性测试（`socialverse/external/pymada/tests/test_parity.py`）在规范 `AuditC` 样本（14 项研究）上将该端口与 R `mada::reitsma` 对标，并在以下方面断言 `max_abs_err < 1e-6`：

- 固定效应 `coefficients`（汇总 Logit 敏感性、Logit FPR）
- 其标准误差（`se`）和完整 `vcov`（列主序展平）
- 研究间协方差 `Psi`（列主序展平）
- 推导得出的汇总 `sensitivity` 和 `false_pos_rate`
- Rutter–Gatsonis HSROC 系数（`Theta`、`Lambda`、`beta`、`sigma2theta`、`sigma2alpha`）
- SROC `AUC` 和部分 `pAUC`

参考值由 `tests/r_reference_driver.R` 通过 R `mada` + `jsonlite` 一次生成到 `tests/reference.json`，Python 测试加载并对比 — 测试时无 R 进程运行。

:::{admonition} 确定性，非随机性
:class: warning

这里的所有把关内容都是确定性的 REML/求积输出 — `reitsma`/`AUC`/`calc_hsroc_coef` 中没有自举或 MCMC 步骤，因此 1e-6 把关是严格的数值一致性界限，而非更宽松的随机收敛容差。端口重新实现 R 的 `vmmin` BFGS 线搜索（而非 `scipy.optimize`）的原因正是为了落在 R 到达的同一个驻点，而不是更尖锐或不同的局部最优。
:::

要重现：

```bash
Rscript socialverse/external/pymada/tests/r_reference_driver.R
pytest socialverse/external/pymada/tests/
```

## 在 socialverse 工作流中

日常使用时，调用 `sv.tl.dta_descriptives` 获取带连续性修正的每个研究的敏感性/特异性/DOR/LR±，然后调用 `sv.tl.dta_bivariate`（或其实验性别名 `sv.tl.dta_glmm`）获取 Reitsma 汇总点估计和 SROC/AUC — `dta_bivariate` 优先采用 `pymada` 后端，仅当端口抛出异常时才回退到原生近似。注册表执行 `requires={"models": ["dta"]}` / `produces={"models": ["dta_bivariate"]}` 合约，这两个步骤之间的必需链；在编写管道前使用 `sv.list_functions()` 或 `registry_lookup("dta_bivariate")` 确认活跃签名和前置要求链。
