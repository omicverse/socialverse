# R 包移植版

社会科学研究者常用的方法大多用 **R** 实现。socialverse 将应用最广泛的包的数值计算核心用纯 `numpy`/`scipy` 重新实现,使你可以从 Python 运行它们,无需 R 运行时——并且在确定性核心上将每个实现与其 R 参考版本钉死在 `max_abs_err < 1e-6` 以内。

每个移植版都存放在 `socialverse.external.<port>` 下,并接入高级 `sv.tl.*` / `sv.pp.*` API,所以在日常使用中,你调用已注册的函数,而对等性门控引擎在后台运行。

## rebuildr 协议

每个移植版使用 [omicverse-rebuildr](https://github.com/omicverse/omicverse-rebuildr) 协议构建:

1. **R 源码是可执行规范** ——不是文字说明,而是实际的包。
2. 在编写移植版之前,**对等性门限已被提交** (`1e-6` 用于确定性的 1 类量)。
3. 门限**绝不会放宽来隐藏错误的数值**。如果某个量确实是随机的(自助法标准误、MCMC-MLE 估计)或参考实现本身只收敛到更宽松的容限,那么该限值会被记录下来,而不是被掩盖。

具体来说,每个移植版都包含:

```
socialverse/external/<port>/
├── <port>.py                     # 纯 numpy/scipy 移植版
├── __init__.py
└── tests/
    ├── r_reference_driver.R       # 运行真实 R 包 → reference.json
    ├── reference.json             # 已提交参考值(无需 R 即可运行测试)
    └── test_parity.py             # 断言移植版 == 参考值,精度 1e-6
```

已提交的 `reference.json` 意味着 **你可以在不使用 R 的情况下运行对等性检验测试**;R 仅在从头重新生成参考值时才需要。

## 14 个移植版

| 移植版 | R 包 | 功能 | 领域 | 对等性检验测试 |
|---|---|---|---|---|
| [pymetafor](pymetafor.md) | `metafor::rma` | 随机效应/混合效应荟萃分析 (REML/ML/DL/EE、Knapp–Hartung、I²/H²、荟萃回归、BLUP) | 荟萃分析 | 9 |
| [pynetmeta](pynetmeta.md) | `netmeta` | 频率学派网络荟萃分析 (图论、SUCRA、净热) | 荟萃分析 | 12 |
| [pyrobumeta](pyrobumeta.md) | `robumeta` | 稳健方差荟萃回归 (RVE、CR2、Satterthwaite 自由度) | 荟萃分析 | 4 |
| [pymada](pymada.md) | `mada` | 二元诊断准确性荟萃分析 (Reitsma、SROC/AUC) | 荟萃分析 | 8 |
| [pysurvey](pysurvey.md) | `survey` | 基于设计的复杂样本调查估计 (svydesign/svymean/svytotal/svyglm、svyby/ratio/ciprop) | 样本与因果 | 8 |
| [pyfixest](pyfixest.md) | `fixest` | 高维固定效应回归 + 聚类标准误 + Poisson PMLE | 样本与因果 | 6 |
| [pydid](pydid.md) | `did` | Callaway–Sant'Anna 交错式倍差法 (att_gt / aggte) | 样本与因果 | 7 |
| [pymatchit](pymatchit.md) | `MatchIt` | 倾向得分匹配 + 平衡诊断 | 样本与因果 | 12 |
| [pysurvival](pysurvival.md) | `survival` | Kaplan–Meier、Cox PH (Efron/Breslow)、条件 logit、参数型 AFT | 生存分析 | 9 |
| [pypsych](pypsych.md) | `psych` | 信度 (α/ω)、ICC、相关性检验、因子分析 | 心理测量 | 8 |
| [pylavaan](pylavaan.md) | `lavaan` | 验证性因子分析 / SEM (ML、完整拟合指数序列、修正指数) | 心理测量 | 8 |
| [pyqca](pyqca.md) | `QCA` | 定性比较分析 (校准、真值表、Quine–McCluskey 最小化) | 组态分析 | 10 |
| [pyergm](pyergm.md) | `ergm` | 指数随机图模型 (充分统计量、二元独立 MPLE、三元普查) | 网络分析 | 8 |
| [pydemography](pydemography.md) | `demography` | 生命表 + Kitagawa / Oaxaca 分解 | 人口统计 | 6 |

**总计 115 个对等性检验测试,所有测试在每个包的确定性核心上都以 `max_abs_err < 1e-6` 通过**。

从侧栏选择一个包。每个教程涵盖:R 包是什么、移植版及其如何接入 socialverse、可运行的 Python 示例、R↔Python 函数对照表和对等性检验证据。
