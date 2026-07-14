# pynetmeta — netmeta 的 Python 实现

> 频率学图论网络（混合治疗比较）荟萃分析，从 R 的 `netmeta` 移植到纯 numpy/scipy，精度达 1e-6 级别——无需 R 运行时。

## `netmeta` 的功能

`netmeta` 是频率学网络荟萃分析（NMA）的标准 R 包：它使用 Rücker (2012) 提出的图论/电网络方法汇集直接和间接证据，跨越三种或以上治疗的网络，在代数上等价于多元加权最小二乘法。社会科学家和卫生政策研究者在文献中存在多个成对比较（药物、项目、政策等竞争性干预）但从未在同一研究中全部进行直接对比的情况下使用它，他们需要一个单一的一致排序以及借用整个比较图强度的一致性效应估计。除了汇集估计外，它还提供异质性（Q、τ²）、P 分数/SUCRA 风格的排名以及直接和间接证据间不一致性的诊断——节点分割和净热/基于设计的分解。

## 移植

`socialverse.external.pynetmeta` 提供：

- `netmeta(TE, seTE, treat1, treat2, studlab, reference_group=None, level=0.95, method_tau="DL")` — 通过 `prepare()` 加上 Rücker Laplacian-伪逆机制（`invmat`、`multiarm`、`nma_ruecker`）拟合网络模型，同时进行常效应（固定、τ=0）和随机效应（DerSimonian-Laird τ²）模型拟合。返回一个 `NetMeta` 对象。
- `NetMeta` — 结果容器，暴露 `.trts`（排序后的治疗名称）、`.TE_fixed`/`.seTE_fixed` 和 `.TE_random`/`.seTE_random`（治疗×治疗汇集效应和标准误矩阵）、`.Q`/`.df_Q`/`.pval_Q`、`.tau2`/`.tau`，以及便捷方法 `.comparison(treat, reference, random=False)` 返回一个治疗对的 `(TE, seTE)`。
- `netmeasures(net, random=False, tau_preset=None, sep=":")` — 移植 `netmeta::netmeasures`：基于 Krahn (2013) 设计帽矩阵的每个比较的网络指标——`proportion`（直接证据比例）、`meanpath`（平均路径长度）、`minpar` 和 `minpar_study`（最小平行性，网络级和研究级）。

这是纯 numpy/scipy 实现——无 `rpy2`、无 R 安装、调用路径中任何地方都没有 MCMC 采样器。它在 socialverse 中作为注册的第 3 层函数 **`sv.tl.netmeta`**（模块 `socialverse/tl/_meta_nma.py`）被接入，内部调用 `external.pynetmeta.netmeta` 来计算汇集的治疗对照参考的 β/协方差契约和联赛表，调用 `external.pynetmeta.netmeasures` 将每个比较的网络指标附加到结果中。`sv.tl.netmeta` 需要 `models["nma_contrasts"]`（由上游 `sv.pp.nma_pairwise` 产生）并生成 `models["nma"]`。

:::{admonition} 精度对标
:class: note

此移植针对 R `netmeta`（参考版本 3.6-1）固定为 `max_abs_err < 1e-6`，跨越 12 个确定性对标测试，覆盖固定效应和随机效应汇集 TE/SE 矩阵、异质性（Q、df、p 值、τ²、τ）、参考治疗列以及固定和随机两种模式下的全部四个 `netmeasures` 输出。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pynetmeta import netmeta, netmeasures

# A small 4-treatment network of two-arm studies, connected with one loop.
# TE/seTE are log-odds-ratio contrasts (treat1 vs treat2) per study.
TE       = [-0.30, -0.10,  0.20, -0.22, -0.15]
seTE     = [ 0.20,  0.18,  0.25,  0.22,  0.19]
treat1   = ["metf", "rosi", "sulf", "metf", "metf"]
treat2   = ["plac", "plac", "plac", "rosi", "sulf"]
studlab  = ["S1",    "S2",   "S3",   "S4",   "S5"]
# S4 (metf vs rosi) closes a loop, so the design carries both direct and
# indirect evidence — that is what drives the Q inconsistency test below.

net = netmeta(TE, seTE, treat1, treat2, studlab, reference_group="plac")

print(net.trts)                       # sorted treatment names: ['metf', 'plac', 'rosi', 'sulf']
print(net.TE_fixed, net.seTE_fixed)   # common-effect pooled TE/SE matrices (treatment x treatment)
print(net.TE_random, net.seTE_random) # random-effects (DerSimonian-Laird) pooled matrices
print(net.Q, net.df_Q, net.pval_Q)    # global heterogeneity/inconsistency test
print(net.tau2, net.tau)              # DL between-study variance

te, se = net.comparison("metf", "plac", random=True)  # one pairwise contrast
print("metf vs plac (random):", te, se)

# Per-comparison network measures: proportion of direct evidence, mean path
# length, and minimal parallelism (network- and study-level).
nm = netmeasures(net, random=False)
print(nm["proportion"])   # e.g. {'metf:plac': 0.7..., 'rosi:plac': 1.0, ...}
```

## R ↔ Python 对照表

| R (`netmeta`) | socialverse | 说明 |
|---|---|---|
| `netmeta(TE, seTE, treat1, treat2, studlab, reference.group=...)` | `socialverse.external.pynetmeta.netmeta(TE, seTE, treat1, treat2, studlab, reference_group=...)` | 同 DerSimonian-Laird 默认值（`method_tau="DL"` 是仅有的支持方法）；返回 `NetMeta` 对象而非 R 列表 |
| `net$TE.fixed` / `net$seTE.fixed` | `net.TE_fixed` / `net.seTE_fixed` | 治疗×治疗矩阵，由 `net.trts` 索引 |
| `net$TE.random` / `net$seTE.random` | `net.TE_random` / `net.seTE_random` | DerSimonian-Laird 随机效应矩阵 |
| `net$Q`, `net$df.Q`, `net$pval.Q`, `net$tau2`, `net$tau` | `net.Q`, `net.df_Q`, `net.pval_Q`, `net.tau2`, `net.tau` | 全局异质性/不一致性统计量 |
| `netmeasures(net, random=TRUE)` | `socialverse.external.pynetmeta.netmeasures(net, random=True)` | 返回每个指标的 `{comparison_label: value}` 字典而非 R 数据框 |
| 高层次分析员调用 | `sv.tl.netmeta(state, reference=..., comb="random"\|"fixed")` | 注册的 socialverse 函数；消耗 `models["nma_contrasts"]`（来自 `sv.pp.nma_pairwise`），写入 `models["nma"]` 包含联赛表、Q/τ² 和 `netmeasures` |

## 精度对标证据

12 个确定性对标测试（`socialverse/external/pynetmeta/tests/test_parity.py`）针对 R `netmeta` 参考拟合（`reference.json`，由 `r_reference_driver.R` 生成）在 `max_abs_err < 1e-6` 阈值下对移植进行把关。把关的量：治疗排序（`trts`）、完整的固定效应和随机效应 TE/SE 矩阵、全局异质性三元组（Q、df、p 值）、DerSimonian-Laird τ²/τ、一个对参考治疗的指定成对对比，以及固定和随机两种模式下的全部四个 `netmeasures` 输出（`proportion`、`meanpath`、`minpar`、`minpar_study`），加上 `proportion` 在 [0, 1] 范围内的理智性界限检查，并对仅具有直接双臂证据的比较等于 1。

:::{admonition} 此移植中无随机松弛
:class: warning

此处把关的每个量——Laplacian 伪逆、DerSimonian-Laird τ² 和 Krahn 帽矩阵网络指标——都是无采样或迭代优化器的闭式线性代数计算，因此 1e-6 容差在此移植中任何地方都不会放松（不像包装自举标准误或基于 MCMC 的估计器的移植）。
:::

本地再现：

```bash
Rscript socialverse/external/pynetmeta/tests/r_reference_driver.R
pytest socialverse/external/pynetmeta/tests/
```

## 在 socialverse 工作流中

日常操作中，在用 `sv.pp.nma_pairwise` 构建对比表后调用 `sv.tl.netmeta` ——它是注册的入口点，包装此移植并写入 `models["nma"]`（联赛表、异质性、`netmeasures`）供下游排名（`sv.tl.netrank`）、秩图/SUCRA（`sv.tl.nma_rankogram`）和不一致性检查（`sv.tl.nma_inconsistency`、`sv.tl.netsplit`）使用。在编脚本前使用 `registry_lookup("netmeta")` 或 `sv.list_functions()` 确认实时签名、`requires`/`produces` 契约和层级，因为注册表——而非此页面——是已部署 API 的真实来源。
