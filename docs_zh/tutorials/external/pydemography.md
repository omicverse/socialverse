# pydemography — R `demography` 生命表在 Python 中的实现

> 周期生命表构造（`nMx → nqx → lx / Lx / Tx / ex`）及 Kitagawa
> 和 Oaxaca-Blinder 率差分解，可从 Python 调用，
> 精度与 R 引擎达 1e-6 级——无需 R 运行时。

## `demography` 的功能

R `demography` 包（及相关的 `DemoTools` 生态系统）是人口统计学家用来
将原始年龄特异性死亡率转换为生命表精算机制的标准工具包：生存率（`lx`）、
每个区间内的人年数（`Lx`）、存活率加权的剩余人年数（`Tx`）和各年龄的
期望寿命（`ex`，包括出生时的 `e0`）。其数值引擎 `demography:::lt` 实现了
经典的 Chiang/Keyfitz 分离因子法处理婴幼儿区间（`a0`、`a1`）及标准 `qx`
闭包公式，这是为什么人口统计学家、精算师和研究死亡率、人口老龄化或跨国期望
寿命差异的社会科学家选择它而不是手工推导闭包。Kitagawa（1955）率分解和
Oaxaca-Blinder（1973）平均值差分解并非 `demography` 本身的功能，但它们是
社会科学家与生命表自然配对的工具——将两个群体之间的粗率差或结果差分解为
"组成/禀赋"分量和"率/系数"分量。

## 端口

- `life_table(mx, sex="total", startage=0, agegroup=1)` — `demography:::lt` 的
  忠实端口。从年龄特异性死亡率计划 `mx` 构建完整的精算列集
  （`ax`、`mx`、`qx`、`lx`、`dx`、`Lx`、`Tx`、`ex`、`nx`，各为 numpy 数组）
  加标量 `e0`。
- `life_expectancy(mx, sex="total", startage=0, agegroup=1, age=0)` —
  便捷包装函数，运行 `life_table` 并返回单个请求年龄的 `ex`（默认 `e0`）。
- `kitagawa(c1, r1, c2, r2)` — Kitagawa（1955）粗率差 `R2 - R1` 的分解，
  分解为 `rate_effect` 和 `composition_effect`，精确求和为 `total`。
- `oaxaca(yA, xA, yB, xB)` — 平均结果差 `meanYA - meanYB` 的二元 Oaxaca-Blinder
  分解，分解为 `explained`（禀赋，使用 B 组 OLS 系数作为参考结构）和
  `unexplained`（系数）。

该端口为纯 `numpy`（`_ols` 使用 `numpy.linalg.lstsq` 进行 Oaxaca 回归）
——无 `scipy`、无 R 运行时、无子进程调用 R。它通过 `socialverse/tl/_demography.py`
连接到 socialverse 的注册表：

- 当提供的区间宽度匹配单年（`agegroup=1`）或标准 `1, 4, 5, 5, …` 五年
  （`agegroup=5`）计划且 `mx` 无缺失值时，`sv.tl.life_table(...)` 委派
  给 `external.pydemography.life_table`；返回的 `lx`/`dx`/`Lx`/`Tx` 从
  端口的 radix-1 惯例重新缩放至请求的 `radix`。否则回退到内置生命表构建器，
  并在 `models.life_table["backend"]` 中报告运行的后端。
- `sv.tl.decomposition(...)` 将标量 Kitagawa 分解委派给
  `external.pydemography.kitagawa`，调用形式为
  `kitagawa(cA, mA, cB, mB)`，使得 `rate_effect + composition_effect`
  精确再现 `crude_B - crude_A`；它同样在 `models.decomposition["backend"]`
  中报告使用的后端（`"pydemography"` vs `"builtin"`）。

:::{admonition} 平衡关口
:class: note

该端口在 R `demography` 引擎（生命表）和已发表的 Kitagawa / Oaxaca-Blinder
闭式（在 R 驱动中独立计算）上精确到 `max_abs_err < 1e-6`，跨 6 个确定性
平衡测试。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pydemography import (
    life_table, life_expectancy, kitagawa, oaxaca,
)

# --- 1. 从单年死亡率计划计算周期生命表 --------
mx = [0.02, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.30]

lt = life_table(mx, sex="female", startage=0, agegroup=1)
print(lt["e0"])          # 出生时期望寿命
print(lt["lx"][:3])      # 年龄 0, 1, 2 处的生存率
print(lt["ex"][:3])      # 年龄 0, 1, 2 处的剩余期望寿命

# life_expectancy() 是 life_table()["ex"][age] 的单行包装函数
e0_male = life_expectancy(mx, sex="male", startage=0, agegroup=1, age=0)
print(e0_male)

# --- 2. 粗率差的 Kitagawa 分解 --------------
# c1/c2: 年龄构成份额（各求和为 1）；r1/r2: 群体率
c1 = [0.40, 0.35, 0.15, 0.10]
c2 = [0.25, 0.30, 0.25, 0.20]
r1 = [0.005, 0.010, 0.030, 0.090]
r2 = [0.004, 0.009, 0.028, 0.085]

ki = kitagawa(c1, r1, c2, r2)
print(ki["total"], ki["rate_effect"], ki["composition_effect"])
assert np.isclose(ki["rate_effect"] + ki["composition_effect"], ki["total"])

# --- 3. 平均结果差的 Oaxaca-Blinder 分解 --------------
rng = np.random.default_rng(0)
nA, nB = 50, 50
xA = rng.normal(5, 1, size=(nA, 2))
xB = rng.normal(4, 1, size=(nB, 2))
yA = 2.0 + 1.5 * xA[:, 0] + 0.9 * xA[:, 1] + rng.normal(0, 0.5, nA)
yB = 1.3 + 1.2 * xB[:, 0] + 1.0 * xB[:, 1] + rng.normal(0, 0.5, nB)

ox = oaxaca(yA, xA, yB, xB)
print(ox["gap"], ox["explained"], ox["unexplained"])
assert np.isclose(ox["explained"] + ox["unexplained"], ox["gap"])
```

## R ↔ Python 对照

| R (`demography`) | socialverse | 注记 |
|---|---|---|
| `demography:::lt(mx, sex, startage, agegroup)` | `sv.tl.life_table(...)` / `pydemography.life_table(mx, sex, startage, agegroup)` | `sv.tl.life_table` 从 `state.datasets` 读取 `mx`/age/width 列，按 `radix` 重新缩放 `lx`/`dx`/`Lx`/`Tx`，并在返回的模型中报告 `backend` |
| `lifetable(mx)$ex[age]` | `pydemography.life_expectancy(mx, sex, startage, agegroup, age)` | 单行 `ex` 查找，无直接 `sv.*` 包装函数 |
| R 驱动中的自定义 Kitagawa（1955）闭式 | `sv.tl.decomposition(...)` / `pydemography.kitagawa(c1, r1, c2, r2)` | `sv.tl.decomposition` 调用 `kitagawa(cA, mA, cB, mB)`；`rate_effect + composition_effect == crude_B - crude_A` |
| R 驱动中的自定义二元 Oaxaca-Blinder 闭式 | `pydemography.oaxaca(yA, xA, yB, xB)` | 无直接 `sv.tl.*` 包装函数；`sv.tl.decomposition` 运行其自己的年龄指标回归配套函数（`_oaxaca_blinder`）而非调用本函数 |

## 平衡证据

`socialverse/external/pydemography/tests/test_parity.py` 中的 6 个确定性 pytest
用例，各均断言 `max_abs_err < 1e-6`（对比 `reference.json`，由
`r_reference_driver.R` 生成）：

- `test_lifetable_female` / `test_lifetable_male` / `test_lifetable_total` —
  每个生命表列（`ax`、`mx`、`qx`、`lx`、`dx`、`Lx`、`Tx`、`ex`、`nx`）
  和标量 `e0`，覆盖婴幼儿分离因子的所有三个性别分支，跨 10 区间单年计划。
- `test_life_expectancy_e0` — `life_expectancy(..., age=0)` 包装函数
  为所有三个性别分支从完整生命表中再现 `e0`。
- `test_kitagawa` — `R1`、`R2`、`total`、`rate_effect`、`composition_effect`，
  加上精确求和恒等式 `rate_effect + composition_effect == total`。
- `test_oaxaca` — `betaA`、`betaB`（OLS 系数向量）、`meanYA`、`meanYB`、`gap`、
  `explained`、`unexplained`，加上精确求和恒等式 `explained + unexplained == gap`。

不涉及随机或容差较松的数量——生命表是闭式递推，两个分解都是闭式代数分割，
因此所有 6 个测试均按相同的 1e-6 关口执行。

如要重现结果：

```bash
Rscript socialverse/external/pydemography/tests/r_reference_driver.R
pytest socialverse/external/pydemography/tests/test_parity.py -v
```

## 在 socialverse 工作流中

日常工作中，调用 `sv.tl.life_table(...)` 获取周期生命表，调用
`sv.tl.decomposition(...)` 获取 Kitagawa（及配套 Oaxaca-Blinder）率分割
——两者都透明地优先选择 `pydemography` 后端，并报告实际运行的后端
（`models.life_table["backend"]` / `models.decomposition["backend"]`）。
注册表强制每个函数的声明 `requires`/`produces` 合约；使用
`registry_lookup("life_table")` 或 `sv.list_functions()` 在脚本编制前
确认实时签名和默认值。
