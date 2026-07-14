# pyqca — R `QCA` 的 Python 实现

> 定性比较分析 (QCA) — 真值表构建、Quine–McCluskey 布尔最小化、拟合参数、直接校准和必要超集搜索 — 现在可从 Python 调用，与 R `QCA` 的奇偶校验精度达 1e-6，无需 R 运行时。

## `QCA` 的工作原理

R 的 **QCA** 包（Adrian Duşa 编写）是 Charles Ragin 集合论方法的参考实现，用于将条件的*配置*映射到结果 — 是比较社会科学中脆集、多值集和模糊集 QCA (csQCA/mvQCA/fsQCA) 的标准工具。回归分析问的是"平均而言每个变量重要性如何"，而 QCA 问的是"条件的哪些*组合*足以（或必要）导致结果"，在 2^k 个配置角上构建真值表，用布尔（Quine–McCluskey）逻辑将其最小化为最小积和式解，并报告每条路径的一致性（包含度）和覆盖度。社会科学家在小/中等 N 值的比较研究（福利国家类型学、民主化、政策配置）中采用它，因为因果复杂性、等效性和联结因果关系是研究重点，而不是要平均处理掉的干扰因素。

## 端口实现

`socialverse.external.pyqca` 公开了以下接口：

- `truth_table(data, outcome, conditions, incl_cut=0.8)` — 模糊集真值表（R 中的 `truthTable`）：构建观测到的（非余项）角，按行的案例计数 `n`、充分包含度 `incl`、PRI 和通过 `incl_cut` 切点分配的 `OUT` 列。返回一个 `TruthTable`。
- `minimize(tt, include=None)` — 保守的（复杂的）Quine–McCluskey 布尔最小化，将 `TruthTable` 的 `OUT=1` 角最小化为质基元项和冗余无关的基本质因子覆盖，每项具有 `inclS`/`PRI`/`covS`/`covU`，解级别具有 `overall` 拟合度。仅支持保守解（`include=None`，无余项行简化假设）；简约解会引发 `NotImplementedError`。
- `pof(terms, data, outcome, conditions)` — 拟合参数（每项的 `inclS`/`PRI`/`covS`/`covU`，加上解级别的 `overall`），用于显式项列表，可以是基元项元组或项字符串，如 `"DEV*~URB*LIT"`。
- `calibrate(x, type="fuzzy", method="direct", thresholds=None, logistic=True, idm=0.95)` — 将原始数值数据直接校准为集合隶属度评分。`type="fuzzy"` 是 3 锚点逻辑直接法（排斥/交叉/包含阈值，隶属度 `idm`）；`type="crisp"` 是对排序切点的 `findInterval`。仅支持 `method="direct"` 和 `logistic=True`。
- `superSubset(data, outcome, conditions=None, incl_cut=1.0, cov_cut=0.0, ron_cut=0.0, depth=None)` — 必要超集搜索（R 中的 `superSubset`，`relation="necessity"`）：枚举条件的合取（模糊 `min`）和极小析取（模糊 `max`），报告通过 `incl_cut`/`cov_cut`/`ron_cut` 的项，附加 `inclN`/`RoN`/`covN`。
- `TruthTable` — `truth_table` 的类似数据类的结果，携带 `conditions`、`rownames`、`rows`、`OUT`、`n`、`incl`、`PRI`。

该端口采用纯 `numpy`/`scipy` 实现（无 rpy2，无 R 运行时）。它连接到 socialverse 在 `socialverse/tl/_setmethods.py` 中的注册集方法函数：`sv.tl.qca` 在模块内构建自己的真值表和 `OUT` 编码，然后将 Quine–McCluskey 最小化和按项/解拟合委托给 `pyqca.minimize`（仅当 `pyqca` 调用引发异常时才回退到模块内 Quine–McCluskey 实现），也将必要超集搜索委托给 `pyqca.superSubset` 以处理 `models['qca']['necessity']` 块。`sv.tl.calibrate` 直接委托给 `pyqca.calibrate` 处理模糊直接和脆集路径。`sv.tl.necessity_analysis` 直接委托给 `pyqca.superSubset` 以获取独立必要性报告（`models['necessity']` / `diagnostics['necessity']`）。

:::{admonition} 奇偶校验门
:class: note

该端口固定在 R `QCA` 3.25，以在 10 个确定性奇偶校验测试上达到 `max_abs_err < 1e-6`（`socialverse/external/pyqca/tests/test_parity.py`），由 18 个案例的比较数据集驱动（Ragin 风格的发展/城市化/识字率/工业化/稳定性条件相对于生存结果）。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pyqca import truth_table, minimize, pof, calibrate, superSubset

# 微小的模糊集数据集：6 个案例，两个条件（DEV、URB），一个结果（SURV）。
data = {
    "DEV":  [0.81, 0.99, 0.58, 0.16, 0.07, 0.98],
    "URB":  [0.12, 0.89, 0.98, 0.07, 0.16, 0.99],
    "SURV": [0.05, 0.95, 0.89, 0.12, 0.42, 0.95],
}

# 1. 真值表：仅观测角，有 n / incl（一致性）/ PRI / OUT。
tt = truth_table(data, outcome="SURV", conditions=["DEV", "URB"], incl_cut=0.8)
print("rownames:", tt.rownames)   # 基于 1 的 R 风格行 ID
print("OUT:", tt.OUT, "incl:", tt.incl, "PRI:", tt.PRI)

# 2. OUT=1 角的 Quine-McCluskey 最小化（保守解）。
sol = minimize(tt)
print("terms:", sol["terms"])                 # 如 ["DEV*URB", ...]
print("term inclS/covS:", sol["inclS"], sol["covS"])
print("solution overall:", sol["overall"])     # {"inclS":..., "PRI":..., "covS":...}

# 3. 明确项集的拟合参数（如假设的路径）。
fit = pof(["DEV*URB"], data, outcome="SURV", conditions=["DEV", "URB"])
print("pof inclS/PRI/covS:", fit["inclS"], fit["PRI"], fit["covS"])

# 4. 直接校准：原始尺度 -> 模糊隶属（3 锚点逻辑）。
raw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
fs = calibrate(raw, type="fuzzy", method="direct", thresholds=[3, 5.5, 8], idm=0.95)
print("fuzzy scores:", fs)                     # 交叉点处为 0.5（x=5.5）

# 脆集校准：每个值等于/超过的切点计数。
crisp = calibrate(raw, type="crisp", thresholds=[3, 5.5, 8])
print("crisp set values:", crisp)               # 0..3 中的整数

# 5. 必要超集搜索：哪些条件/组合对 SURV 是必要的？
ss = superSubset(data, outcome="SURV", conditions=["DEV", "URB"],
                  incl_cut=0.9, cov_cut=0.6)
print("necessary terms:", ss["terms"])
print("inclN/RoN/covN:", ss["incl_cov"]["inclN"], ss["incl_cov"]["RoN"], ss["incl_cov"]["covN"])
```

## R ↔ Python 词典

| R (`QCA`) | socialverse | 说明 |
|---|---|---|
| `truthTable(data, outcome, conditions, incl.cut = 0.8)` | `pyqca.truth_table(data, outcome, conditions, incl_cut=0.8)` | 仅观测行（余项行（`n==0`）被删除，匹配 R 的默认表）。 |
| `minimize(tt, include = "")` (conservative/complex, no `?` in `include`) | `pyqca.minimize(tt, include=None)` | 仅移植保守解；简约/`include="?"` 引发 `NotImplementedError`。 |
| `pof(expression, data)` | `pyqca.pof(terms, data, outcome, conditions)` | `terms` 可以是基元项元组或 `"A*~B"` 风格的字符串。 |
| `calibrate(x, type = "fuzzy", method = "direct", thresholds, logistic = TRUE, idm = 0.95)` | `pyqca.calibrate(x, type="fuzzy", method="direct", thresholds=..., logistic=True, idm=0.95)` | 间接/TFR 校准和非逻辑直接路径超出范围。 |
| `calibrate(x, type = "crisp", thresholds)` | `pyqca.calibrate(x, type="crisp", thresholds=...)` | `findInterval(x, sort(thresholds))`。 |
| `superSubset(data, outcome, conditions, relation = "necessity", incl.cut, cov.cut)` | `pyqca.superSubset(data, outcome, conditions, incl_cut=1.0, cov_cut=0.0, ron_cut=0.0, depth=None)` | 仅必要关系；合取 + 极小析取，R 的 `sqrt(.Machine$double.eps)` 截断容限被复现。 |
| (全 fsQCA 管道，脚本) | `sv.tl.qca(state, conditions=..., outcome=..., threshold=0.8, ...)` | 模块内真值表/编码 + `pyqca.minimize`/`pyqca.superSubset` 委托；在任何 `pyqca` 异常时回退到模块内 Quine–McCluskey。 |
| 对数据框列的 `calibrate(...)` | `sv.tl.calibrate(state, column=..., thresholds=..., type=...)` | 直接委托给 `pyqca.calibrate`；可选择将校准列写回 `sources['datasets']`。 |
| `superSubset(..., relation = "necessity")` | `sv.tl.necessity_analysis(state, outcome=..., conditions=..., incl_cut=0.9, cov_cut=0.6)` | 直接委托给 `pyqca.superSubset`。 |

## 奇偶校验证据

`socialverse/external/pyqca/tests/test_parity.py` 对 `tests/reference.json` 运行 10 个测试（由 `tests/r_reference_driver.R` 对 R `QCA` 3.25 生成），在 `max_abs_err < 1e-6` 时（离散列的精确整数相等性，`tol=0`）。门控数量如下：

- **真值表**：观测行 ID（`rownames`）、每行的条件位模式、`OUT` 编码、每行案例计数 `n`（全精确），以及每行充分包含度 `incl` 和 PRI（浮点，1e-6）。
- **最小化**：质基元项字符串（精确字符串匹配）和按项的 `inclS`、`PRI`、`covS`、`covU`（浮点，1e-6），加上解级别的 `overall` `inclS`/`PRI`/`covS`。
- **校准**：模糊直接逻辑评分（浮点，1e-6）和脆集 `findInterval` 值（精确整数），用于固定的 10 点原始尺度与 3 个锚点。
- **superSubset**：必要项列表（精确字符串匹配，按 R 报告顺序的合取后跟极小析取）和按项的 `inclN`、`RoN`、`covN`（浮点，1e-6）。

:::{admonition} 范围限制，诚实说明
:class: warning

该端口仅实现 `QCA` 的确定性组合精确核心。带有余项简化假设的 `minimize(..., include=...)` (简约/中间解) 被明确不支持并引发 `NotImplementedError`。`calibrate` 仅实现模糊直接逻辑路径和脆集 `findInterval`；间接/TFR 校准方法在文档字符串中被记为此门的超出范围。该端口中没有任何随机成分（不同于，比如自举或基于 MCMC 的重构）— 所有 10 个奇偶校验测试比较精确确定性输出。
:::

要复现：

```bash
Rscript socialverse/external/pyqca/tests/r_reference_driver.R
pytest socialverse/external/pyqca/tests/
```

## 在 socialverse 工作流中

日常工作中，调用 `sv.tl.qca` 以获得完整 fsQCA 管道（真值表到最小化解和必要性），调用 `sv.tl.calibrate` 在运行 `qca` 前将原始列转换为校准集合隶属度评分，调用 `sv.tl.necessity_analysis` 以获得独立的必要超集报告。注册表强制执行每个函数的 `requires`/`produces` 契约（例如 `sv.tl.necessity_analysis` 需要 `sources['datasets']` 和 `variables['outcome']` 并产生 `models['necessity']`/`diagnostics['necessity']`）— 在编写脚本前使用 `registry_lookup` 或 `sv.list_functions()` 确认实时签名和契约。
