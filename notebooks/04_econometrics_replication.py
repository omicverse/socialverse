# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 实证复现:AER 8 步管线 + 可复现脚本
#
# **这条分析链讲什么。** 一份「AER 风格的复现包」(replication package)不是一个数
# 字,而是一条**可审计的证据链**:平衡表 → 基线 TWFE → 稳健性矩阵 → 机制,并**吐出
# 一份能真正跑的 `.R`/`.do` 脚本**,让别人在自己的机器上一键重跑。本 notebook 用
# `socialverse` 走完这条链,并展示注册表(registry)是如何把「每一步依赖什么、产出
# 什么」写成机器可读契约的 —— 这正是 `socialverse` 相对普通脚本的差异:**不是猜 API,
# 而是查注册表**。
#
# **涉及的函数(按执行顺序)。**
#
# | 步骤 | 函数 | 相位 | 干什么 |
# |---|---|---|---|
# | 1 | `sv.pp.ingest` | prepare | 把面板 DataFrame 登记进 `sources.datasets` |
# | 2 | `sv.pp.declare_design` | prepare | 声明 panel_id / time / treatment / first_treated |
# | 3 | `sv.tl.parallel_trends` | analyze | event-study 前导期联合 Wald 检验平行趋势 |
# | 4 | `sv.tl.did` | analyze | 双向固定效应 DID,估 ATT + 聚类稳健 SE |
# | 5 | `sv.tl.replicate` | analyze | **AER 8 步**:平衡表→基线→稳健性矩阵→emit R/Stata 脚本 |
# | 6 | `sv.pl.forest` | plot | 把 ATT 画成森林图(点估计 ± 95% CI) |
# | 7 | `sv.pl.manuscript_docx` | plot | 保守排版稿件 + 结构质检 |
#
# **`StudyState` 会被填哪些槽。** 这条链会依次点亮 12 个槽里的这些:
# `sources`(数据)、`design`(设计列)、`variables`(结果/控制变量)、
# `estimand`(ATT)、`identification`(DiD + 平行趋势结论)、`models`(did/twfe)、
# `diagnostics`(pretrend/balance/robustness/coverage)、`artifacts`(scripts/tables/figures/docx)。
# 每一步都会往 append-only 的 `provenance` 账本里写一条记录 —— 这就是复现的「证据脊柱」。
#
# **对标的现实工具。** 这条链对标 R 的 **`fixest`**(`feols` 做 TWFE + 聚类 SE)、
# **`modelsummary`/`etable`**(出版级回归表)、Stata 的 **`reghdfe`/`esttab`**,以及一整套
# 「AER data & code appendix」的复现工作流。`socialverse` 的差异有两点:(1) 依赖用
# **注册表 grounding**(requires→produces 契约,查而非猜);(2) 每一步留下**证据链**
# (provenance),让复现包自带审计轨迹。

# %% [markdown]
# ## 步骤 0:先查注册表,别猜 API
#
# **为什么先查。** `socialverse` 的设计论点是:让分析 agent 可靠的不是统一数据容器,而是
# 一张**可查询的、标注了依赖的函数注册表**。所以正确的第一步永远是「查」而不是「写」。
# 我们先打印目录总览,再针对「replication」查具体函数的契约。这也是 OmicOS kernel 在
# `humanities_social` 域里会替 agent 跑的那一句 `print(sv.utils.registry_lookup(query))`。

# %%
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless:图只存盘不弹窗

import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

import socialverse as sv
from socialverse import datasets as ds

# 让产物(图/稿件)始终落在本 notebook 同目录,无论从哪个工作目录运行 ——
# 这样 markdown 里的 ![](fig_xxx.png) 引用永远解析得到。
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Jupyter 里没有 __file__,退回当前目录
    _HERE = os.getcwd()

def here(fname):
    return os.path.join(_HERE, fname)

# 一个能在 notebook 与纯脚本两种运行方式下都工作的 display 兜底
try:
    display  # noqa: F821  (Jupyter 里内置)
except NameError:
    def display(x):  # 纯脚本运行时退化为 print
        print(x)

print(sv.utils.registry_summary())

# %% [markdown]
# 目录里我们要走的是 `[econ] replicate` 这条。现在查它的契约 —— 注意 `registry_lookup`
# 直接把 **Requires / Produces / Must-run-first / Tier** 摆出来,agent 读到的就是这一段。

# %%
print(sv.utils.registry_lookup("replication", max_results=3))

# %% [markdown]
# ## 步骤 1:注册表 grounding —— 未满足契约时,它会**拒绝**你
#
# **为什么这步重要(契约演示)。** 注册表不是文档,是**活契约**:`@register` 包了每个函
# 数,调用时先拿 `state` 核对 `requires`,缺了就抛 `sv.RegistryError`。这不是 bug,是
# **特性** —— 它把「数据没准备好就别报因果结论」变成机器可强制的规则。我们故意在一个
# 空 `StudyState` 上调 `replicate`,看它怎么拒绝、并给出**如何补齐**的提示。

# %%
empty = sv.StudyState()
try:
    sv.tl.replicate(empty)
except sv.RegistryError as err:
    print("RegistryError(这是设计出来的护栏,不是崩溃):\n")
    print(err)

# %% [markdown]
# 拒绝信息里每个缺口都标了「谁能生产它」:`sources.datasets` 由 `ingest` 生产、
# `design.treatment` 由 `declare_design` 生产,而 `estimand.target` / `identification.strategy`
# 是 **user-supplied**(研究问题给定,没有函数能替你产出)。
#
# 与其手动拼,不如让注册表**排一条计划**。`resolve_plan` 走依赖图,把到达 `replicate`
# 需要的函数按拓扑序排好,并列出还需人工提供的输入(`needs_input`)与需要人确认的
# 自动插入步骤(`escalations`,因为这些函数 `auto_fix='escalate'`)。

# %%
import json

plan = sv.registry.resolve_plan("replicate")
print("有序计划(plan):")
for i, fn in enumerate(plan["plan"], 1):
    print(f"  {i}. {fn}")

print("\n还需你亲自提供的输入(needs_input,注册表无法代产):")
for item in plan["needs_input"]:
    print(f"  - {item['slot']}.{item['key']}  (for {item['for'].split('.')[-1]})")

print(f"\n需人工确认的自动插入步骤(escalations):{len(plan['escalations'])} 处(auto_fix=escalate)")

# %% [markdown]
# 计划已经把「对标 fixest 复现工作流」的顺序显式写出来了:
# `ingest → declare_design → parallel_trends → did → replicate`。下面就照这个计划,一步步
# 填满 `StudyState`,每一步都先看它的 requires→produces 契约。

# %% [markdown]
# ## 步骤 2:研究问题给定的输入 —— estimand / outcome / identification
#
# **为什么先写这三个。** `needs_input` 告诉我们:目标量(ATT)、结果变量(y)、识别策略
# (DiD)是研究者给定的,不是数据算出来的。把它们写进 state 是**声明研究意图**,后续
# `parallel_trends` / `did` / `replicate` 的 `requires` 会来读这些槽。

# %%
st = sv.StudyState()
st.write("estimand", "target", "ATT")            # 目标量:平均处理效应
st.write("variables", "outcome", "y")            # 结果变量
st.write("identification", "strategy", "DiD")    # 识别策略:双重差分

print("已写入的用户给定槽:")
print("  estimand.target       =", st.estimand.get("target"))
print("  variables.outcome     =", st.variables.get("outcome"))
print("  identification.strategy =", st.identification.get("strategy"))

# %% [markdown]
# ## 步骤 3:`ingest` —— 把数据登记进 `sources.datasets`
#
# **契约。** `ingest`:requires `{}` → produces `sources['datasets']`。这是每个研究的社区
# 版入口。我们用内置的 `load_did_panel(att=-0.8)`:一个有**真平行前趋势**、处理组在
# 第 5 期(2015 年)被处理、真实处理效应 `att=-0.8` 的面板。列:
# `firm_id, year, treat, post, treat_post, first_treated, y, x1`。

# %%
df = ds.load_did_panel(att=-0.8)
print("面板形状:", df.shape, "  (units × periods =", df.firm_id.nunique(), "×", df.year.nunique(), ")")
display(df.head(10))

sv.pp.ingest(st, data=df)
print("\ningest 后 sources 元数据:", st.sources.get("dataset_meta"))

# %% [markdown]
# ## 步骤 4:`declare_design` —— 声明设计列进 `design`
#
# **契约。** `declare_design`:requires `sources['datasets']` → produces `design` 的
# `panel_id / time / treatment / first_treated / ...`。它只写**列名字符串**(元数据),并
# 会拿登记好的数据核对列是否存在(不存在只告警不报错)。这一步把「DiD 的设计」翻译成
# 下游估计器读得懂的词汇。

# %%
sv.pp.declare_design(
    st,
    panel_id="firm_id",
    time="year",
    treatment="treat_post",
    first_treated="first_treated",
)
print("design 槽现在持有:")
for k in ("panel_id", "time", "treatment", "first_treated"):
    print(f"  design.{k:14s} = {st.design.get(k)!r}")
print("列名校验告警:", st.design.get("warnings") or "(无 —— 所有列都在数据里)")

# %% [markdown]
# ## 步骤 5:`parallel_trends` —— 先验证识别假设,再谈因果
#
# **为什么先做这步(而不是直接估 DID)。** DiD 的因果解读全押在**平行趋势**上。注册表把
# 这条纪律硬编码进了契约:`did` 的 requires 里含 `identification['parallel_trends']`,而只有
# `parallel_trends` 能生产它 —— 所以**没跑前趋势检验,`did` 就没法作为因果结论报出来**。
#
# **契约。** `parallel_trends`:requires `design[panel_id,time,treatment,first_treated]`
# + `variables['outcome']` + `estimand['target']` → produces `diagnostics['pretrend']`
# 和 `identification['parallel_trends']`。做法:估完整 event-study,对**前导期**系数做联合
# Wald 检验;`p>0.05`(未拒绝)记为 `"pass"`。

# %%
sv.tl.parallel_trends(st)
pretrend = st.diagnostics.get("pretrend")
print("平行趋势结论 identification.parallel_trends =", st.identification.get("parallel_trends"))
print(f"联合 Wald:  F = {pretrend['joint_F']:.4f},  p = {pretrend['p_value']:.4f}")
print("判定说明 :", pretrend["note"])
print("\n各前导期系数 (relative period → (coef, se)):")
for k, (coef, se) in pretrend["pre_coefs"].items():
    print(f"  t={k:>3}:  {coef:+.4f}  (se {se:.4f})")

# %% [markdown]
# `p = 0.755`,四个前导期系数都不显著异于 0 —— **未拒绝平行趋势**。注册表现在允许把下一步
# 的 DID 当因果解读。

# %% [markdown]
# ## 步骤 6:`did` —— 双向固定效应,估 ATT + 聚类稳健 SE
#
# **契约。** `did`:requires `design[panel_id,time,treatment]` + `variables['outcome']`
# + **`identification['parallel_trends']`**(上一步刚生产)→ produces `models['did','twfe']`
# 和 `diagnostics['robustness']`。它估 `y ~ treat_post + 单位FE + 时间FE`,SE 聚类到
# `firm_id`;若前趋势 `fail`,估计照样报但会被标注为「关联非因果」。

# %%
sv.tl.did(st)
m = st.models.get("did")
print("DID / TWFE 估计结果 (models.did):")
print(f"  ATT (att)      = {m['att']:+.4f}")
print(f"  标准误 (se)     = {m['se']:.4f}   [聚类到 {m['n_clusters']} 个 firm]")
print(f"  95% CI         = [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]")
print(f"  p 值           = {m['p']:.2e}")
print(f"  N              = {m['n']}   估计量 = {m['estimator']}")
print(f"  因果判定        = {m['note']}")
print(f"\n真实处理效应(数据生成时设定) att = -0.8;估出来 ATT ≈ {m['att']:+.3f} —— 恢复得很准。")

# %% [markdown]
# ## 步骤 7:`replicate` —— AER 8 步复现包,一次跑完
#
# **为什么用它(而不是手写)。** 前面几步已经把 ATT 估出来了;`replicate` 把一份「审稿人
# 想看的完整复现包」一次生成:**平衡表**(处理/控制组协变量均值 + Imbens–Rubin 规范化差)、
# **基线 TWFE**、**稳健性矩阵**(控制变量×SE 聚类的规格网格)、**出版级回归表**,并
# **emit 可运行的 `main.R`(feols)+ `main.do`(reghdfe)**。这正是对标 `fixest` + 一篇论文
# 复现工作流的核心产物。
#
# **契约。** `replicate`:requires `sources['datasets']` + `design['treatment']` +
# `estimand['target']` + `identification['strategy']`,且 prerequisites 要求先跑过 `did`
# → produces `variables['controls']`、`models['twfe']`、`diagnostics['robustness','balance']`、
# `artifacts['scripts','tables']`。我们前面正好按计划把这些前置都满足了。

# %%
sv.tl.replicate(st)

print("=== ① 平衡表 diagnostics.balance(处理 vs 控制,norm_diff>0.25 触发 flag)===")
display(st.diagnostics.get("balance"))

# %%
print("=== ② 稳健性矩阵 diagnostics.robustness(点估计在各规格下是否稳定)===")
display(st.diagnostics.get("robustness"))

# %% [markdown]
# 读这张矩阵:规格 (2)–(5) 都是 TWFE,ATT 稳稳落在 **−0.73 附近**,只有 (1)「无固定效应、
# 无控制」偏到 −0.44 —— 说明固定效应确实在吸收混淆。所有规格 `***`(p<0.01),点估计对
# 「加不加控制变量、怎么聚类 SE」不敏感。**这就是审稿人要看的稳健性证据。**

# %%
print("=== ③ 出版级回归表 artifacts.tables['regression'](SE 在括号内,带显著性星)===")
display(st.artifacts.get("tables")["regression"])

# %% [markdown]
# ## 步骤 8:emit 的可复现脚本 —— 别人一键重跑
#
# **为什么这是复现包的灵魂。** 一份数字表格不可复现,一段**能跑的代码**才可复现。
# `replicate` 生成的不是占位符,而是**按解析出的变量名拼好的、语法正确的** `feols` /
# `reghdfe` 脚本。把它和 `data.csv` 一起交出去,任何人都能在 R/Stata 里复现同一套估计。

# %%
scripts = st.artifacts.get("scripts")
print("emit 的脚本键:", list(scripts.keys()))
print("\n" + "=" * 72)
print("main.R  (fixest / feols —— 对标 R 复现工作流)")
print("=" * 72)
print(scripts["main.R"])

# %% [markdown]
# 注意一个诚实的细节:自动模式把 `treat` / `post` / `x1` 都当成了数值控制变量塞进
# `feols(y ~ treat_post + treat + post + x1 | firm_id + year, ...)`。这是「无 codebook 时按
# 数据类型自动挑控制变量」的默认行为 —— **看到 emit 的脚本你才能一眼发现并按需删改**,这
# 恰恰是「emit 可读脚本」比「藏在函数里的黑箱回归」更适合复现的原因。下面看 Stata 版:

# %%
print("=" * 72)
print("main.do  (reghdfe / esttab —— Stata 复现伴侣)")
print("=" * 72)
print(scripts["main.do"])

# %% [markdown]
# ## 步骤 9:`forest` —— 把 ATT 画成森林图
#
# **契约。** `forest`:requires `models['did']` → produces `artifacts['figures']`。注册表让
# 出图也诚实:**你没估过的系数画不出来**。它读 `models.did` 的点估计与 CI,存成真实 PNG,
# 路径回写进 `artifacts.figures`。

# %%
sv.pl.forest(st, out=here("fig_forest_att.png"), title="Replication ATT · forest (95% CI)")
fig_info = st.artifacts.get("figures")["forest"]
print("森林图已保存:", fig_info["path"], f"(dpi={fig_info['dpi']})")
print("说明:", fig_info["note"])

# %% [markdown]
# ![Replication ATT forest plot](fig_forest_att.png)
#
# 一根点在 −0.73、置信须完全落在 0 线左侧的森林图 —— 处理效应显著为负且稳健。

# %% [markdown]
# ## 步骤 10:`manuscript_docx` —— 保守排版稿件 + 结构质检
#
# **契约。** `manuscript_docx`:requires `sources['datasets']` → produces
# `artifacts['docx','pdf']` 和 `diagnostics['coverage']`。它「保守」到从不改写你的正文,
# 只做结构化排版,并生成一张**结构覆盖质检清单**(章节/图/表计数、公式安全标记)。装了
# `python-docx` 就写真 `.docx`,否则降级为 `.md` 且不丢内容。

# %%
manuscript = (
    "# Replication: the effect of the policy on firm outcomes\n\n"
    "## 方法\n\n"
    "We estimate a two-way fixed-effects difference-in-differences model, "
    "clustering standard errors at the firm level. Parallel pre-trends are not rejected.\n\n"
    "## 结果\n\n"
    "The ATT is negative (about -0.73) and stable across the robustness matrix; "
    "every specification is significant at the 1% level.\n\n"
    "## 讨论\n\n"
    "The estimated effect is robust to the choice of controls and SE clustering."
)

sv.pl.manuscript_docx(st, manuscript=manuscript, out=here("replication_manuscript.docx"))
cov = st.diagnostics.get("coverage")
print("稿件已生成:", st.artifacts.get("docx"))
print("渲染器      :", cov["renderer"], "(fallback:", cov["fallback"], ")")
print("章节数      :", cov["n_sections"], " 标题:", cov["headings"])
print("必备章节覆盖:", cov["present_required"], " 缺失:", cov["missing_required"])
print("公式安全    :", cov["math_note"])
print("结构 OK     :", cov["structure_ok"])

# %% [markdown]
# ## 结尾:`st.summary()` —— provenance 证据链
#
# 整条链跑完,`StudyState` 现在既是**结果容器**,又是**审计轨迹**。`summary()` 列出每个被
# 点亮的槽,以及 append-only 的 provenance 记了多少步 —— 这条「证据脊柱」让复现包自带
# 出处。下面既打印 summary,也把 provenance 账本逐条展开,看每一步的 requires→produces。

# %%
print(st.summary())

print("\n" + "=" * 72)
print("Provenance 账本(每一步的契约,append-only)")
print("=" * 72)
for rec in st.provenance:
    fn = rec["function"].split(".")[-1]
    req = ", ".join(f"{s}[{','.join(ks)}]" for s, ks in rec["requires"].items()) or "∅"
    pro = ", ".join(f"{s}[{','.join(ks)}]" for s, ks in rec["produces"].items()) or "∅"
    print(f"  step {rec['step']}: {fn}")
    print(f"           requires: {req}")
    print(f"           produces: {pro}")

# %% [markdown]
# ## 这条链对标的现实工具 + `socialverse` 的差异
#
# **对标。** 这条 `ingest → declare_design → parallel_trends → did → replicate → forest →
# manuscript_docx` 复现链,对标的是 R 的 **`fixest`**(`feols`:TWFE + 聚类 SE + `etable`
# 出版表)、Stata 的 **`reghdfe` + `esttab`**,外加一整套「AER data & code appendix」式的论文
# 复现工作流:平衡表 → 基线 → 稳健性矩阵 → emit 可跑脚本。
#
# **差异(为什么不只是又一个回归封装)。**
# 1. **注册表 grounding**:每个函数带机器可读的 `requires→produces` 契约。空 state 上调
#    `replicate` 会被**拒绝**并告诉你缺什么、谁能补(步骤 1);`resolve_plan` 能直接把整条
#    链**排序**出来 —— agent **查注册表**,而不是猜 API、也不会把没跑前趋势的 DID 当因果报。
# 2. **证据链 provenance**:每一步 append 一条出处,复现包自带审计轨迹(步骤 10 的
#    provenance 账本)。fixest 给你估计,`socialverse` 给你**估计 + 它是怎么被合法地得到的**。

# %%
print("Done. socialverse AER 复现链跑通:注册表 grounding(查而非猜) + provenance 证据链(自带审计)。")
