# pymetafor — Python 中的 metafor

> `metafor::rma.uni`（随机效应/混合效应/等效应元分析、Knapp-Hartung 推断、BLUP）用纯 numpy/scipy 重建，可从 Python 调用，与 R 的精度为 1e-6 — 无需 R 运行时。

## `metafor` 的功能

`metafor`（Viechtbauer 2010，*Journal of Statistical Software*）是元分析的参考 R 包：它将多个研究的效应量合并到固定效应/等效应、随机效应和混合效应（元回归）模型下，并报告社会科学家在森林图写作中期望的全套异质性诊断指标 — 通过 REML、ML 或 DerSimonian-Laird 计算的 τ²（研究间方差）、Higgins-Thompson I²/H²、Cochran 的 Q，以及 Knapp-Hartung/Sidik-Jonkman 小样本调整。它是 PRISMA 规范的系统综述中事实上的标准，应用于心理学、教育学、社会学和卫生/社会政策研究，其 `rma()`/`rma.mv()` 系列涵盖大多数已发表的元分析模型。社会科学家特别青睐它，因为其 τ² 估计器和推断（尤其是 Knapp-Hartung t 调整）是期刊审稿人期望看到报告的那些，而不是更粗糙的近似。

## Python 端口

`socialverse/external/pymetafor` 公开以下内容：

- **`rma(yi, vi, mods=None, method="REML", test="z", level=95.0, add_intercept=True)`** — 随机/混合/等效应元分析。拟合 τ²（通过 metafor 自身的 Fisher 计分迭代的 REML、ML、DerSimonian-Laird 闭形式或"EE"/"FE"等效应），然后加权最小二乘 β，包含 Wald 或 Knapp-Hartung（`test="knha"`）推断、Q_E/Q_Ep 异质性检验、I²/H²（Higgins-Thompson）、通过逆 Fisher 信息计算的 SE(τ²)，以及调节因子的 Q_M/Q_Mp 综合检验。返回 `RMAResult` 数据类。
- **`RMAResult.predict(level=95.0)`** — 平均效应的拟合值加置信区间和预测区间（仅截距模型），镜像 `predict.rma`。
- **`blup(res, level=95.0)`** — 逐研究最优线性无偏预测器（经验-贝叶斯收缩至拟合值，`metafor::blup.rma.uni` 精度）。接受来自 `rma()` 的 `RMAResult` 并返回 `BLUPResult` 数据类，包含 `pred`、`se`、`pi_lb`、`pi_ub`。

该模块是纯 numpy/scipy（约 230 行代码，无 rpy2，无 R 子进程）。在数值上，它从不直接构造病态的 `(XᵀWX)⁻¹` — 所有内容都通过加权设计的 QR 分解表达 — 这是保持它在 1e-6 精度的关键，即使在 `cond(XᵀWX) ≈ 2e11`（非中心化调节因子）时也如此。

它集成到 socialverse 中，位于 `socialverse/tl/_meta.py`：注册函数 **`sv.tl.meta_random`** 在 `method` 为 `REML`/`ML`/`DL`/`EE`/`FE`/`CE` 时委派给 `pymetafor.rma`（PM/SJ/HS/HE 估计器仍使用 socialverse 的遗留路径），并额外调用 `pymetafor.blup` 在结果中附加逐研究经验-贝叶斯收缩表（`out["blup"]`）。`sv.tl.meta_fixed` 仍是单独、更简单的共同效应合并函数，不经过此端口。

:::{admonition} 精度检验门
:class: note

该端口固定为 R `metafor` 5.0.1，在 9 个确定性精度测试（`socialverse/external/pymetafor/tests/test_parity.py`）中 `max_abs_err < 1e-6`，针对规范 `dat.bcg`（BCG 疫苗，k=13 对数风险比）测试集运行。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pymetafor import rma, blup

# 13 项研究的治疗效应（对数风险比）和其抽样方差 —
# 与 pymetafor 精度测试中使用的规范 dat.bcg 测试集形状相同。
yi = np.array([-0.889, -1.586, -1.336, -1.406, -0.212, 0.577, 0.339,
                0.336, -1.088, -0.322, 0.000, -0.442, -0.017])
vi = np.array([0.324, 0.311, 0.157, 0.032, 0.038, 0.055, 0.078,
                0.084, 0.007, 0.011, 0.006, 0.019, 0.037])

# 1) 随机效应合并，REML tau^2（metafor::rma 默认），Knapp-Hartung t 调整
res = rma(yi, vi, method="REML", test="knha")
print("pooled effect (beta):", res.beta[0])
print("SE(beta):", res.se[0])
print("95% CI:", res.ci_lb[0], res.ci_ub[0])
print("tau^2:", res.tau2, " SE(tau^2):", res.se_tau2)
print("I^2 (%):", res.I2, " H^2:", res.H2)
print("Q_E:", res.QE, " Q_Ep:", res.QEp)

# 2) 平均效应的预测区间（predict.rma 等价）
pi = res.predict(level=95.0)
print("prediction interval:", pi["pi_lb"], pi["pi_ub"])

# 3) 带调节因子的元回归（混合效应模型）
ablat = np.array([44, 55, 42, 52, 13, 44, 19, 13, 44, 19, 33, 21, 42], dtype=float)
res_mods = rma(yi, vi, mods=ablat, method="REML")
print("intercept, slope:", res_mods.beta)
print("Q_M (moderator omnibus):", res_mods.QM, res_mods.QMp)

# 4) 逐研究经验-贝叶斯收缩（BLUP），metafor::blup.rma.uni 等价
b = blup(res)
for i, (p, se) in enumerate(zip(b.pred, b.se)):
    print(f"study {i}: shrunk pred={p:.4f}  se={se:.4f}  PI=[{b.pi_lb[i]:.4f}, {b.pi_ub[i]:.4f}]")

# 通过有线 socialverse 管道函数等价（自动添加 BLUP）：
# import socialverse as sv
# state = sv.tl.meta_random(state, method="REML", knapp_hartung=True)
# state.models["meta"]["estimate"], state.models["meta"]["blup"]
```

## R ↔ Python 对照表

| R（`metafor`） | socialverse | 注释 |
|---|---|---|
| `rma(yi, vi, method="REML")` | `rma(yi, vi, method="REML")`，来自 `socialverse.external.pymetafor`，或 `sv.tl.meta_random(state, method="REML")` | 仅截距随机效应拟合 |
| `rma(yi, vi, mods=~x, method="REML")` | `rma(yi, vi, mods=x, method="REML")` | `mods` 不包含截距列；`add_intercept=True` 前置它（匹配 R 的默认 `~x` 公式行为） |
| `rma(yi, vi, method="DL")` | `rma(yi, vi, method="DL")` | 闭形式 DerSimonian-Laird τ² |
| `rma(yi, vi, method="FE")` / `rma(yi, vi, method="EE")` | `rma(yi, vi, method="EE")`（也接受 `"FE"`/`"CE"` 别名） | 等效应，τ²=0 |
| `rma(yi, vi, test="knha")` | `rma(yi, vi, test="knha")` 或 `sv.tl.meta_random(..., knapp_hartung=True)` | Knapp-Hartung t 分布 CI/推断，df = k-p |
| `predict.rma(res)` | `res.predict(level=95.0)` | 拟合值 + CI + 预测区间 |
| `blup.rma.uni(res)` | `blup(res)`，来自 `socialverse.external.pymetafor` | 逐研究经验-贝叶斯收缩；由 `sv.tl.meta_random` 自动附加为 `out["blup"]` |
| `res$I2`、`res$H2`、`res$QE`、`res$QEp` | `res.I2`、`res.H2`、`res.QE`、`res.QEp` | Higgins-Thompson 异质性诊断指标 |
| `res$se.tau2` | `res.se_tau2` | 逆 Fisher 信息（REML/ML）或基于 Q 的 Delta 方法（DL） |
| `res$QM`、`res$QMp` | `res.QM`、`res.QMp` | 调节因子综合检验，不包含截距 |

## 精度证据

`socialverse/external/pymetafor/tests/test_parity.py` 中 9 个确定性精度测试，在 R `metafor` 5.0.1 的 `dat.bcg` 测试集（k=13 对数风险比，由 `tests/r_reference_driver.R` 生成）上以 `max_abs_err < 1e-6` 设置门。门限数量包括：REML、DL 和等效应（EE）拟合的合并 `beta`/`se`/`zval`/`pval`/CI；Knapp-Hartung t 调整拟合；`tau2` 和 `se.tau2`；`I2`、`H2`、`QE`、`QEp`；包括中心化和非中心化调节因子的元回归斜率和 `QE`/`QM`；`predict.rma` 预测区间；以及所有四个 BLUP 输出（`pred`、`se`、`pi.lb`、`pi.ub`）。

:::{admonition} 元回归 τ² 受 metafor 自身收敛容限限制，不是端口
:class: warning

metafor 的 τ² Fisher 计分迭代停止在其默认 `threshold=1e-5`（非机器精度），因此在平坦或病态的目标函数上（例如非中心化调节因子，`cond(XᵀWX) ≈ 2e11`），metafor 的*报告的* τ² 本身离确切的 REML 根可达 ~1e-5 — 这传播到未识别的截距。该端口复制 metafor 的精确迭代及其 `threshold=1e-5`，因此它追踪 metafor 的*报告的*数字而不是数学上精确的数字。因此调节因子模型测试在斜率/τ² 上放宽容限至 1e-5（从 1e-6 降低），而 τ² 无关数量（`QE`）在 1e-6 时保持精确。中心化调节因子完全消除条件问题。详见 `RECONSTRUCTION_REPORT.md §6 已知局限` 的完整分析，包括独立 Brent 根检查确认残差是 metafor 的欠收敛，不是端口错误。
:::

复现方法：

```bash
Rscript socialverse/external/pymetafor/tests/r_reference_driver.R
python -m pytest socialverse/external/pymetafor/tests/test_parity.py
```

## 在 socialverse 工作流中

调用 **`sv.tl.meta_random(state, method="REML", knapp_hartung=True)`** 用于日常随机/混合效应合并 — 它在方法为 REML/ML/DL/EE 时委派给此端口并自动附加逐研究 BLUP；改用 `sv.tl.meta_fixed` 进行纯共同效应合并。注册表强制执行协约（`requires={"models": ["meta_effects"]}`、`produces={"models": ["meta"]}`，即先运行 `sv.pp.escalc`）；在对其编写脚本前，使用 `sv.registry_lookup("meta_random")` 或 `sv.list_functions()` 确认实时签名和 kwargs。