# pysurvey — Python 中的 R `survey`

> 基于设计的（Taylor 线性化）复杂概率样本估计——svydesign、svymean/svytotal、svyglm、svyby、svyratio、svyciprop——现在可从 Python 直接调用，与 R `survey` 实现 1e-6 级精度对等，无需 R 运行时。

## survey 做什么

R 的 **survey** 包（Thomas Lumley）是分析复杂抽样设计收集的数据的标准工具——分层、多阶段聚类（PSU）抽样、不等概率权重和有限总体修正（fpc）。在调查数据上直接运行 `lm`/`glm` 会忽略设计，产生错误的标准误；`survey` 会计算**基于设计的**点估计和 Taylor 线性化（"终极聚类"）方差，正确传播分层和 PSU 结构到每个统计量。社会科学家在分析国家或跨国调查数据（如 GSS、ANES、NHANES、DHS、World Values Survey 等）并需要可发表的标准误而非虚假的 SRS 式紧密标准误时，都会用到它。

## 移植版本

`socialverse.external.pysurvey` 暴露以下接口：

- `svydesign(data, weights, ids=None, strata=None, fpc=None)` — 构建 `SurveyDesign`（权重 + PSU id + 分层 + fpc），返回 `SurveyDesign` 数据类，具有 `.degf` 属性（设计自由度 = #PSU − #分层）。
- `svymean(y, design, level=95.0)` — 列的基于设计的均值，具有线性化 SE、基于 t 的 CI 和 df。
- `svytotal(y, design, level=95.0)` — 基于设计的总值，机制同 `svymean`。
- `svyglm(y, X, design, level=95.0, add_intercept=True)` — 基于设计的高斯 GLM（加权最小二乘法）及调查三明治方差；返回系数、SE、t 值、p 值和 CI。
- `svyby(y, by, design, stat="svymean", level=95.0)` — 按层级域（子总体）`svymean`/`svytotal`，方差取遍**完整**设计（非重新声明的子集设计）——匹配 R 的 `svyby` 语义。
- `svyratio(num, den, design, level=95.0)` — 两个总值比的基于设计的估计，通过 Taylor 线性化影响函数实现。
- `svyciprop(y, design, level=95.0)` — logit 方法的调查比例 CI（与 R 的 `svyciprop(..., method="logit")` 相同，该包的默认值）。
- `SurveyDesign` — `svydesign` 返回的数据类。

该移植版本是纯 `numpy`/`scipy`（无 rpy2，无 R 运行时）实现，使用 `survey` 内部使用的相同终极聚类 Taylor 线性化估计器。它被接入 socialverse 在 `socialverse/tl/_survey.py` 中注册的调查域函数：`sv.tl.survey_estimate`（基于设计的加权回归，委派给 `svydesign`/`svyglm`/`svymean`/`svytotal`）、`sv.tl.survey_by`（委派给 `svyby`）和 `sv.tl.survey_ratio`（委派给 `svyratio`）、以及 `sv.tl.survey_ciprop`（委派给 `svyciprop`）。`sv.pp.design_survey`（或其别名）在 `StudyState` 上声明这些 `tl.*` 函数读取的设计槽位（`weights`/`strata`/`psu`/`fpc`）。

:::{admonition} 精度对等门槛
:class: note

本移植版本与 R `survey` 4.5 固定为 `max_abs_err < 1e-6`，通过 8 个确定性对等测试（`socialverse/external/pysurvey/tests/test_parity.py`）验证，覆盖分层设计（`apistrat`）和 `survey` 随附规范 `api` 数据集的单阶段聚类设计（`apiclus1`）。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pysurvey import svydesign, svymean, svyglm, svyby, svyratio, svyciprop

# 一个小型分层样本：2 个分层，不等权重，带有有限
# 总体修正 (fpc) — 与 R 的 `apistrat` 设计形状相同。
data = {
    "api00":  [693, 762, 811, 528, 601, 742, 895, 615],
    "ell":    [42.0, 10.0, 5.0, 71.0, 55.0, 18.0, 3.0, 46.0],
    "meals":  [67.0, 24.0, 11.0, 85.0, 78.0, 33.0, 6.0, 61.0],
    "stype":  ["E", "E", "E", "E", "H", "H", "H", "H"],
}
weights = [33.7, 33.7, 33.7, 33.7, 22.1, 22.1, 22.1, 22.1]
fpc     = [140, 140, 140, 140, 90, 90, 90, 90]  # 分层内总体 PSU 数

# 元素抽样 (ids=None -> 每行一个 PSU)，分层，带 fpc。
design = svydesign(data, weights=weights, ids=None, strata=data["stype"], fpc=fpc)
print("design df:", design.degf)  # (# PSU) - (# 分层)

# api00 的基于设计的均值，含线性化 SE + t CI。
m = svymean("api00", design)
print("mean:", m["estimate"], "se:", m["se"], "ci:", (m["ci_lb"], m["ci_ub"]))

# 基于设计的 GLM：api00 ~ ell + meals (调查三明治方差)。
X = np.column_stack([data["ell"], data["meals"]])
g = svyglm("api00", X, design)
print("coef:", g["coef"], "se:", g["se"], "pval:", g["pval"])

# 按分层层级的域（子总体）均值（方差来自完整设计）。
by = svyby("api00", "stype", design, stat="svymean")
print("levels:", by["levels"], "domain means:", by["estimate"])

# 基于设计的比值（需要在设计上添加两列）。
data["api_stu"] = [450, 480, 520, 300, 350, 470, 560, 380]
data["enroll"]  = [520, 500, 540, 400, 420, 500, 580, 450]
design2 = svydesign(data, weights=weights, ids=None, strata=data["stype"], fpc=fpc)
r = svyratio("api_stu", "enroll", design2)
print("ratio:", r["estimate"], "se:", r["se"])

# 比例的 logit 方法 CI：P(api00 > 700)。
prop_indicator = (np.asarray(data["api00"], float) > 700).astype(float)
cp = svyciprop(prop_indicator, design)
print("proportion:", cp["estimate"], "95% CI:", (cp["ci_lb"], cp["ci_ub"]))
```

## R ↔ Python 对照表

| R (`survey`) | socialverse | 注释 |
|---|---|---|
| `svydesign(id=~1, strata=~stype, weights=~pw, fpc=~fpc, data=df)` | `svydesign(data, weights=pw, ids=None, strata=stype, fpc=fpc)` | `ids=None`（或 `~1`）表示元素抽样——每行一个 PSU |
| `svydesign(id=~dnum, weights=~pw, fpc=~fpc, data=df)` | `svydesign(data, weights=pw, ids=dnum, fpc=fpc)` | 单阶段聚类设计；若无分层则 `strata=None` |
| `svymean(~api00, ds)` / `coef()`/`SE()` | `svymean("api00", design)` → `{"estimate", "se", "df", "ci_lb", "ci_ub"}` | |
| `svytotal(~api00, ds)` | `svytotal("api00", design)` | 返回形状同 `svymean` |
| `svyglm(api00 ~ ell + meals, design=ds)` | `svyglm("api00", X, design)` 其中 `X = np.column_stack([ell, meals])` | 截距自动添加，除非 `add_intercept=False` |
| `svyby(~api00, ~stype, ds, svymean)` | `svyby("api00", "stype", design, stat="svymean")` | 域方差来自完整设计，非子集重新声明 |
| `svyratio(~api.stu, ~enroll, ds)` | `svyratio("api_stu", "enroll", design)` | Taylor 线性化比值 SE |
| `svyciprop(~I(api00>700), ds, method="logit")` | `svyciprop(indicator_array, design)` | logit 方法是唯一（且 R 默认）实现的方法 |
| `degf(ds)` | `design.degf` | `SurveyDesign` 上的属性 |
| `df.residual(g)` | `svyglm(...)["df"]` | `= design.degf - (p - 1)` |
| 工作流入口点 | `sv.tl.survey_estimate` / `sv.tl.survey_by` / `sv.tl.survey_ratio` / `sv.tl.survey_ciprop` | 在 `socialverse/tl/_survey.py` 中注册的 `StudyState` 函数，委派给移植版本 |

## 一致性证据

`socialverse/external/pysurvey/tests/test_parity.py` 中的 8 个确定性对等测试，相对于参考 JSON（`reference.json`）（由 R 驱动程序 `r_reference_driver.R` 生成）以 `max_abs_err < 1e-6` 门槛验证。涵盖的量为：

- 分层设计：`svymean` 估计 + SE、`svytotal` 估计 + SE、`api00 ~ ell + meals` 的 `svyglm` 系数 + SE（及残差 df）、每分层域均值 + SE 的 `svyby`、`api.stu / enroll` 的 `svyratio` 估计 + SE、`P(api00 > 700)` 的 `svyciprop` 估计 + 方差 + logit-CI 界。
- 单阶段聚类设计（`apiclus1`）：`svymean` 估计 + SE、`api00 ~ ell` 的 `svyglm` 系数 + SE。
- 设计自由度（`degf`）在每个测试中都被精确断言（整数相等）。

:::{admonition} 无随机性注意
:class: warning

不同于包装 MCMC 或 bootstrap 程序的移植版本，`pysurvey` 的估计器是闭形式的（Taylor 线性化），所以所有 8 个测试都以严格的 1e-6 容差门槛验证，不放宽参考容差，除上述 `1e-4` 总值幅度异常情况外。
:::

在本地重现：

```bash
Rscript socialverse/external/pysurvey/tests/r_reference_driver.R
pytest socialverse/external/pysurvey/tests/
```

## 在 socialverse 工作流中

日常使用时，调用注册的 `sv.tl.survey_estimate`（基于设计的加权回归）、`sv.tl.survey_by`（域估计）、`sv.tl.survey_ratio` 或 `sv.tl.survey_ciprop`——这些读取 `sv.pp.design_survey` 风格声明写入 `StudyState` 的设计槽位（`weights`/`strata`/`psu`/`fpc`），然后内部委派给本移植版本。注册表强制每个函数的 `requires`/`produces` 契约；在接线流水线前，使用 `registry_lookup` 或 `sv.list_functions()` 确认实时签名和槽位名称。