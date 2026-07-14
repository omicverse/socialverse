# pydid — `did` (Callaway & Sant'Anna) 的 Python 实现

> 错位差分-差分设计与按群组-时期的平均处理效应 `att_gt` 及其 `aggte` 聚合，可从 Python 调用，与 R 的 `did` 精度相差 1e-6 级——无需 R 运行时。

## `did` 的功能

`did` (Brantly Callaway & Pedro H.C. Sant'Anna) 是错位采纳差分-差分设计（staggered-adoption difference-in-differences）的参考 R 实现，是处理不同单位在不同日历时期接受处理（政策、最低工资变化、平台推出）的工作马式识别策略。与单一双向固定效应系数不同——后者混合了"合理"的2x2比较与"禁忌"的已处理队列间比较，在处理效应异质性下可能严重有偏——`did` 从分解的按群组-时期平均处理效应 `ATT(g,t)` 构建估计量：指在时期 `g` 首次接受处理的队列在时期 `t` 测得的效应。当社会科学家拥有具有可变政策采纳日期的面板数据，并希望获得对异质性与动态处理效应稳健的因果估计，以及审稿人期望的标准事件研究/队列/日历时间汇总时，他们就会采用它。

## 这个 port

- `att_gt(data, yname, tname, idname, gname, control_group="nevertreated", est_method="reg", anticipation=0, base_period="varying")` — 对每个处理队列 `g` 和时期 `t` 计算按群组-时期的 ATT(g,t)，使用结果回归2x2估计量（仅截距项，即差分-均值）对未处理或未来处理控制组进行对比。返回 `ATTgtResult`。
- `aggte(res, type="simple", max_e=np.inf, min_e=-np.inf, na_rm=False)` — 将 `ATTgtResult` 聚合为 `type='simple'`（pg 加权的整体 ATT）、`type='dynamic'`（按相对事件时间的事件研究 ATT(e) 及整体）、`type='group'`（每个处理队列一个 ATT 加队列大小加权的整体）或 `type='calendar'`（每个日历期间一个 ATT 加简单平均的整体）。返回 `AGGTEResult`。
- `ATTgtResult` — 容器，镜像 R 的 `did` MP 对象的相关字段（`group`、`t`、`att`、`glist`、`tlist`、`pg`、`gvar_unit`）。
- `AGGTEResult` — 聚合的容器（`type`、`overall_att`、`egt`、`att_egt`）。

这个 port 使用纯 numpy——无 rpy2、无 R 子进程。它再现了 `did::att_gt` + `did:::compute.att_gt` + `DRDID::reg_did_panel`（仅截距的结果回归，无协变量）和 `did::aggte`，用于面板数据且 `control_group` 取值于 `{'nevertreated', 'notyettreated'}`、`est_method='reg'`、`anticipation=0`、`base_period='varying'`（R 默认值）。通过 `socialverse/tl/_causal.py` 中的因果模块将其接入 socialverse：注册函数 `sv.tl.did` 和 `sv.tl.event_study` 都调用内部辅助函数 `_cs_estimate`（其本身调用来自 `socialverse.external.pydid` 的 `att_gt`/`aggte`）以将临时 TWFE 点估计替换为已验证的 Callaway–Sant'Anna ATT——`did` 为整体 ATT 交换 `simple`/`group`/`calendar` 聚合，`event_study` 为相对时期系数交换 `dynamic` 聚合。当设计缺少 pydid 需要的列或 port 抛出异常时，两个函数都回退到既有的 TWFE 估计量（结果中 `backend="twfe"`）。

:::{admonition} 精度验证门控
:class: note

该 port 相对于 R `did`（参考版本2.5.1）在7个确定性精度测试上被固定为 `max_abs_err < 1e-6`。
:::

## 快速开始

```python
import numpy as np
from socialverse.external.pydid import att_gt, aggte

# 小型合成错位采纳面板：6个单位观测4个时期。
# first.treat = 0 标记未处理单位（did 约定中的"无 G"）。
n_periods = [1, 2, 3, 4]
data = {
    "id":    np.repeat([1, 2, 3, 4, 5, 6], 4),
    "t":     np.tile(n_periods, 6),
    # 队列 2（单位 1-2）从 t=2 处理，队列 3（单位 3-4）从 t=3 处理，
    # 单位 5-6 未处理（对照组）。
    "g":     np.repeat([2, 2, 3, 3, 0, 0], 4).astype(float),
    "y":     np.array([
        1.0, 1.1, 2.6, 3.4,   # 单位 1 (g=2): t=2 处跳跃
        1.2, 1.3, 2.9, 3.6,   # 单位 2 (g=2)
        0.9, 1.0, 1.1, 2.8,   # 单位 3 (g=3): t=3 处跳跃
        1.1, 1.2, 1.3, 3.0,   # 单位 4 (g=3)
        1.0, 1.15, 1.3, 1.45, # 单位 5（未处理）：平行趋势
        0.8, 0.95, 1.1, 1.25, # 单位 6（未处理）
    ]),
}

res = att_gt(
    data, yname="y", tname="t", idname="id", gname="g",
    control_group="nevertreated", est_method="reg",
)
for g, t, att in zip(res.group, res.t, res.att):
    print(f"ATT(g={g:.0f}, t={t:.0f}) = {att:.4f}")

# 聚合为单一整体 ATT（处理后单元格的 pg 加权平均）。
simple = aggte(res, type="simple", na_rm=True)
print("overall ATT (simple):", simple.overall_att)

# 事件研究聚合：每个相对事件时间 e = t - g 一个 ATT。
dyn = aggte(res, type="dynamic", na_rm=True)
for e, att_e in zip(dyn.egt, dyn.att_egt):
    print(f"ATT(e={e:.0f}) = {att_e:.4f}")
print("overall dynamic ATT (average over e >= 0):", dyn.overall_att)
```

## R ↔ Python 对照表

| R (`did`) | socialverse | 备注 |
|---|---|---|
| `att_gt(yname=, tname=, idname=, gname=, control_group=, est_method=, data=)` | `socialverse.external.pydid.att_gt(data, yname, tname, idname, gname, control_group, est_method, anticipation, base_period)` | port 接受列名 -> 数组映射而非 `data.frame`；仅移植了 `est_method='reg'`、`anticipation=0`、`base_period='varying'` |
| `aggte(res, type="simple", na.rm=)` | `aggte(res, type="simple", na_rm=)` | pg 加权平均处理后 ATT(g,t) |
| `aggte(res, type="dynamic", na.rm=)` | `aggte(res, type="dynamic", na_rm=)` | 返回 `egt`（事件时间）/ `att_egt` / `overall_att` |
| `aggte(res, type="group", na.rm=)` | `aggte(res, type="group", na_rm=)` | 每个队列一个 ATT，队列大小加权的整体 |
| `aggte(res, type="calendar", na.rm=)` | `aggte(res, type="calendar", na_rm=)` | 每个日历时期一个 ATT，简单平均的整体 |
| `res$att`、`res$se`（bootstrap SE） | `res.att`（仅点估计——无 bootstrap SE） | 见[精度证据](#精度证据)中的 SE 说明 |
| —（从分析代码调用） | `sv.tl.did(state, control_group="nevertreated")` | 整体/队列/日历时期 ATT 通过 `_cs_estimate`、`models.did.backend == "pydid"` 处于活跃时 |
| —（从分析代码调用） | `sv.tl.event_study(state)` | 动态 ATT(e) 通过 `_cs_estimate`、`models.event_study.backend == "pydid"` 处于活跃时 |

## 精度证据

7个精度测试门控该 port 对 R `did` 2.5.1 在规范 `mpdta` 县级最低工资面板上的精度，在 `max_abs_err < 1e-6` 下，覆盖：

- `att_gt` 点估计（`group`、`t`、`att`）对每个群组-时期单元格，同时包括 `control_group='nevertreated'` 和 `control_group='notyettreated'`；
- `aggte(type='simple')` 整体 ATT；
- `aggte(type='dynamic')` — 每个事件时间 `att.egt` 值加上整体动态 ATT；
- `aggte(type='group')` — 按队列的 `att.egt` 加上队列大小加权的整体 ATT；
- `aggte(type='calendar')` — 按时期的 `att.egt` 加上简单平均的整体 ATT。

:::{admonition} Bootstrap 标准误差未经精度门控
:class: warning

R 的 `did` 报告乘法-bootstrap 标准误差（`bstrap=TRUE`，Mammen/Rademacher 权重每次运行重新抽取），这在本质上是随机的，即使在 R 自身也每次运行都不同。该 port 不再现这个 RNG 且仅返回点估计。测试套件仅对 R 报告的 SE 有限且为正进行清健检查（`test_se_documented`）——它不能，也不会，断言逐元素的 SE 精度。下游，`sv.tl.did` / `sv.tl.event_study` 保留既有的 TWFE 聚类-稳健 SE 并在 pydid 点估计上重新居中置信区间。
:::

要复现：

```bash
Rscript socialverse/external/pydid/tests/r_reference_driver.R
pytest socialverse/external/pydid/tests/
```

## 在 socialverse 工作流中

调用 `sv.tl.did` 获得整体/队列/日历时期的 Callaway–Sant'Anna ATT，或调用 `sv.tl.event_study` 获得动态事件-时间路径——当设计缺少可用的 `first_treated`/面板结构时，两者都回退到 TWFE 估计量（结果中 `backend="twfe"`），所以始终检查 `models.did.backend` / `models.event_study.backend` 以确认 pydid 确实运行了。注册表对每个函数的 `requires`/`produces` 契约进行强制；在针对它进行脚本编写之前，使用 `registry_lookup` 或 `sv.list_functions()` 确认实时签名。
