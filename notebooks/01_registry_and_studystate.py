# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 快速上手:注册表脊柱与 StudyState
#
# > **对标现实工具**:omicverse `ov.registry` / `ov.utils.registry_lookup`。
# > 这是 socialverse 所有 notebook 的**地基**——先把「脊柱」讲清楚,后面每条
# > 分析链(因果 / 复杂抽样 / 质性编码 / 文献核验 / 校勘学)都是在这套机制上跑。
#
# ## 这条「链」讲什么
#
# omicverse 让 AI agent 能**真的**规划一次生信分析而不瞎编 API,靠的**不是**统一的数据容器
# (AnnData),而是一张**带依赖契约的函数注册表** `ov.registry`:每个函数登记时都带机器可读的
# `requires`(要哪些前置态)/ `produces`(产出哪些态)/ `prerequisites`(先跑哪些函数)/
# `auto_fix`(依赖未满足时的策略)。于是 agent **查表**(`registry_lookup`)而非**猜**——
# 这就是 grounding。AnnData 只是这套契约说话用的**词汇表**。
#
# 社科数据天生不可通约(一份问卷 ≠ 一个语料库 ≠ 一张网络),不存在也永远不会有「社科版 AnnData」。
# 所以 **socialverse 保留注册表这根脊柱,丢掉容器**:用一个轻量的 **12 槽 `StudyState`** 当词汇表
# ——它**不是**数据矩阵,只是让依赖可被检查的共享槽位。
#
# ## 本 notebook 涉及的函数 / 接口
#
# - `sv.registry` 单例:`len()` · `.categories()` · `.list_functions()`
# - `sv.registry.find(q)` —— 模糊搜索(中 / 英 / 缩写 / 后端名)
# - `sv.registry.get_prerequisites(fn)` —— 单个函数的完整契约(机器可读)
# - `sv.registry.resolve_plan(target)` —— 把「达到某目标」排成有序计划
# - `sv.StudyState` + `sv.SLOTS` —— 12 槽词汇表
# - `sv.utils.registry_lookup(q, n)` / `sv.utils.registry_summary()` —— **OmicOS 面向 agent 的查询面**
# - 契约**执法**:在未准备好的 state 上调 `sv.tl.did` 会抛 `sv.RegistryError`(这是特性,不是 bug)
#
# ## StudyState 会被填哪些槽
#
# 本 notebook 以入门为主,末尾会跑一条最小因果链演示 provenance;运行后 state 里会出现:
# `sources`(登记的数据集)· `design`(panel_id/time/treatment)· `variables`(outcome)·
# `estimand`(用户给的 target)· `identification`(平行趋势结论)· `models`(twfe/did)·
# `diagnostics`(pretrend/robustness),外加一条 4 步的 `provenance` 证据链。
#
# ## 对标的现实 Py/R 生态
#
# | socialverse | 对标 |
# |---|---|
# | `sv.registry`(依赖注册表) | omicverse `ov.registry` |
# | `sv.utils.registry_lookup` | omicverse `ov.utils.registry_lookup`(agent 直接 `print` 它) |
# | `StudyState`(12 槽词汇表) | AnnData 的 `obs/var/obsm/uns`(但**不统一数据**,只统一词汇) |
# | `resolve_plan`(排链) | omicverse 的 `leiden → neighbors → pca` 依赖解析 |

# %% [markdown]
# ## 0. 导入并打印目录
#
# **为什么这步**:任何一次分析的第一件事,是让 agent(或你)先看到「注册表里有什么」——
# 有多少函数、分几类。这对应 omicverse 里 agent 开局先 `print(ov.utils.registry_summary())`。
# 导入 `socialverse` 时,各子模块(`pp/tl/pl/gov/lit`)的 `@register` 装饰器会**副作用式**
# 地把函数登进那个进程级单例 `sv.registry`。

# %%
import json

import socialverse as sv

print(len(sv.registry), "registered functions")
print("categories:", sv.registry.categories())

# %% [markdown]
# ### 按类别列出所有函数
#
# `list_functions()` 返回 `{category: [full_name, ...]}`。这就是注册表的「地图」——
# 因果(did/event_study/parallel_trends)、复杂抽样(design_survey/survey_estimate)、
# 质性(code_themes/trace_quotes/…)、文献(search_free/verify_citations/…)、
# 治理(ethics_check/redact_pii/…)等,和社科真实工作流一一对应。

# %%
for cat, fns in sv.registry.list_functions().items():
    print(f"[{cat:11s}] ->", [f.split(".")[-1] for f in fns])

# %% [markdown]
# ## 1. `find` —— 模糊搜索(中文 / 英文 / 缩写 / 后端名)
#
# **为什么这步**:agent 拿到一个研究问题(「我要做双重差分」),第一步不是猜函数名,而是
# **查表**。`find` 支持中文、英文、缩写(DID)、甚至后端工具名(statsmodels)。它返回的每个
# 结果都自带**契约**:`requires`(要什么前置态)→ `produces`(产出什么态)。
#
# **契约(requires → produces)**:这就是 grounding 的核心——你不是记住 `did` 怎么调,
# 而是从注册表读到「`did` 要 `design[panel_id,time,treatment]` + `variables[outcome]` +
# `identification[parallel_trends]`,产出 `models[did,twfe]` + `diagnostics[robustness]`」。

# %%
for r in sv.registry.find("双重差分"):
    print(r["full_name"])
    print("   requires:", r["requires"])
    print("   produces:", r["produces"])
    print("   tier:", r["tier"], " backend:", r["key_tools"])

# %% [markdown]
# ## 2. `get_prerequisites` —— 单个函数的机器可读契约
#
# **为什么这步**:`find` 给概览,`get_prerequisites` 给**可被程序消费**的完整契约。它的返回
# 形状**刻意对齐** omicverse 的 `get_prerequisites`,所以 OmicOS 的 `registry_lookup` 工具
# 无需改动就能吃 socialverse 的注册表。
#
# 注意 `satisfied_by`:它告诉你**每个未满足的 slot 由哪个函数产出**——例如
# `design.panel_id` 由 `declare_design` 产出,`identification.parallel_trends` 由
# `parallel_trends` 产出;而 `variables.outcome` 的产出者为空 = **需要用户提供的输入**。
# 这正是「查而非猜」:未满足时不硬编答案,而是明确告诉你缺口和补法。

# %%
prereq = sv.registry.get_prerequisites("did")
print(json.dumps(prereq, ensure_ascii=False, indent=1))

# %% [markdown]
# ## 3. `resolve_plan` —— 把「达到某目标」排成有序计划
#
# **为什么这步**:知道 `did` 的契约还不够——它的前置(`declare_design`、`parallel_trends`)
# 也各有前置。`resolve_plan` 递归走依赖图,把到达目标所需的函数**拓扑排序**成一条可执行链,
# 并分出两类特别信息:
#
# - `needs_input`:图里没有任何函数产出、state 也没有的 slot —— **必须由用户/研究问题给定**
#   (例如 `estimand.target` = 你要估的 ATT,`variables.outcome` = 结果变量名)。
# - `escalations`:被自动插入、但下游 `auto_fix='escalate'` 的步骤 —— **人应确认后再跑**
#   (社科里因果假设不能默默自动补,要研究者拍板)。
#
# 这就是 omicverse 的 `leiden → neighbors → pca` 解析,搬到社科。

# %%
plan = sv.registry.resolve_plan("did")
print("plan:      ", [p.split(".")[-1] for p in plan["plan"]])
print("needs_input:")
for ni in plan["needs_input"]:
    print("   -", ni["slot"] + "." + ni["key"], "  (for", ni["for"].split(".")[-1] + ")")
print(f"escalations: {len(plan['escalations'])} step(s) a human should confirm")
for esc in plan["escalations"][:3]:
    print("   -", esc["for"].split(".")[-1], "needs", esc["needs"],
          "-> auto_insert", esc["auto_insert"])

# %% [markdown]
# ### 把计划画出来
#
# 一图胜千言:下面把 `resolve_plan('did')` 的有序链 + 每一步「补上了哪个 slot」画成一条
# 流水线,并标出唯一需要用户给定的输入(`estimand.target`)。这不是装饰——它就是注册表
# **自动**从 `requires ↔ produces` 推导出的执行顺序。

# %%
import matplotlib

matplotlib.use("Agg")  # 无头环境:必须用 Agg 后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 让图里的中文正常显示:挑一个系统里存在的 CJK 字体(找不到就退回默认)
_have = {f.name for f in font_manager.fontManager.ttflist}
for _cjk in ("Heiti TC", "Songti SC", "Arial Unicode MS", "Hiragino Sans GB",
             "PingFang SC", "Noto Sans CJK SC", "STHeiti"):
    if _cjk in _have:
        plt.rcParams["font.sans-serif"] = [_cjk]
        plt.rcParams["axes.unicode_minus"] = False
        break

steps = [p.split(".")[-1] for p in plan["plan"]]
# 每一步「解锁/产出」的关键 slot(教学标注)
produced = {
    "ingest": "sources[datasets]",
    "declare_design": "design[panel_id,time,treatment]",
    "parallel_trends": "identification[parallel_trends]",
    "did": "models[did,twfe] + diagnostics[robustness]",
}

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, len(steps) * 3.2)
ax.set_ylim(0, 3)
ax.axis("off")

for i, s in enumerate(steps):
    x = i * 3.2 + 0.2
    box = FancyBboxPatch((x, 1.1), 2.4, 0.9, boxstyle="round,pad=0.08",
                         linewidth=1.6, edgecolor="#2b6cb0", facecolor="#ebf4ff")
    ax.add_patch(box)
    ax.text(x + 1.2, 1.72, f"{i+1}. {s}", ha="center", va="center",
            fontsize=11, weight="bold", color="#1a365d")
    ax.text(x + 1.2, 1.34, produced[s], ha="center", va="center",
            fontsize=7.5, color="#2c5282")
    if i < len(steps) - 1:
        arr = FancyArrowPatch((x + 2.4, 1.55), (x + 3.2 + 0.2, 1.55),
                              arrowstyle="-|>", mutation_scale=16, color="#4a5568")
        ax.add_patch(arr)

ax.text(0.2, 2.55, "resolve_plan('did')  —  从 requires ↔ produces 自动推导的执行链",
        fontsize=12, weight="bold", color="#1a202c")
ax.text(0.2, 0.55, "唯一需用户给定的输入(needs_input):  estimand['target'] = 'ATT'  ·  "
                   "variables['outcome'] = 'y'",
        fontsize=9, color="#c05621", style="italic")

plt.tight_layout()
fig.savefig("fig_resolve_plan_did.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved -> fig_resolve_plan_did.png")

# %% [markdown]
# ![](fig_resolve_plan_did.png)

# %% [markdown]
# ## 4. 契约执法:在未准备好的 state 上调 `did` 会被**拒绝**
#
# **为什么这步**:注册表不只是元数据——契约是**活的**。每个被 `@register(enforce=True)`
# 包裹的函数在调用时会拿 `requires` 去核对传入的 `StudyState`;不满足就抛 `sv.RegistryError`,
# 并**明确告诉你缺哪个 slot、由谁产出**。这就是 omicverse 里「跑 `leiden` 前必须先 `neighbors`」
# 的守卫,搬到社科:**grounding(查而非猜),而不是让你在错误的态上得到一个看似合理的假结果**。

# %%
st = sv.StudyState()
try:
    sv.tl.did(st)  # 空 state:必然被拒
except sv.RegistryError as e:
    print(e)

# %% [markdown]
# ## 5. `StudyState` 的 12 槽词汇表
#
# **为什么这步**:上面所有 `requires`/`produces` 都在用同一套槽名说话。这 12 个槽就是社科版
# 「AnnData 词汇表」——**不统一数据,只统一词汇**,好让依赖可被机器检查。
#
# | 槽 | 装什么 |
# |---|---|
# | `sources` | 原始输入:数据集 / 语料 / 手稿 / .bib / 扫描件 |
# | `design` | 研究设计:抽样框 / 权重 / 分层 / 聚类 / panel_id / time / 处理时点 |
# | `variables` | codebook:变量定义 / 类型 / 测量层次 / 量表题项 |
# | `corpus` | 文本即数据:文档 / 分词 / dfm / OCR文本 / TEI |
# | `codes` | 质性编码:codebook / 编码片段 / 主题 / 主题地图 |
# | `estimand` | 目标量:ATT / 患病率 / 关联 + 目标总体(**通常由用户给定**) |
# | `identification` | 识别假设:DAG / 平行趋势 / IV 有效性 / 排他性 / 正值性 |
# | `models` | 拟合结果:DID/FE / event-study / 加权估计 / 主题模型 / 网络 |
# | `diagnostics` | 检验:pretrend / 平衡性 / 稳健性 / 信度α / 敏感性 |
# | `evidence` | 证据链:claim→引语/引文 / quote-trace 索引 / 已核验 .bib |
# | `governance` | 伦理合规:IRB / 知情同意 / PII 去标识 / 数据许可 / AI 披露 |
# | `artifacts` | 交付物:图 / docx-pdf 稿 / 表 / TEI-XML / apparatus |

# %%
print(list(sv.SLOTS))
print()
# 槽名 + 一行含义(SLOTS 是 {slot: (含义, 典型键)})
for name, (meaning, keys) in sv.SLOTS.items():
    print(f"  {name:15s} {meaning}")

# %% [markdown]
# ## 6. OmicOS 面向 agent 的查询面
#
# **为什么这步**:前面用的都是「程序化」接口。真正跑在 OmicOS 内核里的 agent 看到的是
# `sv.utils.registry_lookup(query)` **打印出来的字符串**——布局(`Found N … Requires …
# Produces … Example`)和它在生信域从 `ov.utils.registry_lookup` 见到的**完全一致**。
# 所以一个域感知的内核只需把 `humanities_social` 域指向 `sv.registry`,agent 不用学任何新东西。
#
# 先看单点查询 `registry_lookup('survey', 3)`——注意它把契约渲染成人类/agent 都能读的样子,
# 还标了 `Tier`(community/plus)和 `auto_fix`。

# %%
print(sv.utils.registry_lookup("survey", 3))

# %% [markdown]
# ### 全局目录 `registry_summary()`
#
# 这是 agent 开局打印一次的「域地图」:函数分类 + 从 `requires ↔ produces` 自动导出的**典型链**
# (因果 / 复杂抽样 / 质性 / 校勘 / 文献 / 治理)。后续每个 notebook 各自展开其中一条链。

# %%
print(sv.utils.registry_summary())

# %% [markdown]
# ## 7. 收尾:跑一条最小因果链,展示 provenance 证据链
#
# **为什么这步**:光讲机制不够——最后把「查表 → 补输入 → 按序执行」真的跑一遍,看 `StudyState`
# 如何被逐槽填满,以及**每一步自动记进 `provenance`**。这条 append-only 的 provenance ledger
# 就是社科最看重的**可复现 / 可审计的「证据脊柱」**:一次完成的分析自带它自己的审计轨迹。
#
# 契约链:`ingest`(载入面板)→ `declare_design`(声明 panel/time/treatment)→ 用户给
# `variables[outcome]` → `parallel_trends`(识别假设检验)→ `did`(TWFE ATT + cluster-robust SE)。

# %%
from socialverse import datasets as ds

st = sv.StudyState()
st.write("estimand", "target", "ATT")          # 唯一由用户给定的目标量
df = ds.load_did_panel(att=-0.8)               # 玩具面板:真值 ATT = -0.8
print("panel columns:", list(df.columns), " shape:", df.shape)

sv.pp.ingest(st, data=df)
sv.pp.declare_design(st, panel_id="firm_id", time="year",
                     treatment="treat_post", first_treated="first_treated")
st.write("variables", "outcome", "y")          # 补上 needs_input 里的 outcome
sv.tl.parallel_trends(st)                       # 识别假设:平行趋势必须先过
sv.tl.did(st)                                   # 现在契约满足,DID 可跑

did = st.models["did"]
print("\nDID 估计结果:")
print(f"  ATT = {did['att']:.4f}   (真值 -0.8)")
print(f"  SE  = {did['se']:.4f}   95% CI = [{did['ci'][0]:.3f}, {did['ci'][1]:.3f}]")
print(f"  p   = {did['p']:.2e}   n = {did['n']}  clusters = {did['n_clusters']}")
print(f"  parallel_trends: {did['parallel_trends']}  ·  estimator: {did['estimator']}")

# %% [markdown]
# ### state 被填了哪些槽

# %%
print("populated slots ->")
for slot, keys in st.populated().items():
    print(f"  {slot:15s} {keys}")

# %% [markdown]
# ### provenance 证据链(append-only,可复现脊柱)
#
# 每一步都记下:第几步、跑了哪个函数、参数、消费了哪些 slot(`requires`)、产出了哪些 slot
# (`produces`)。这就是「evidence spine」——把一次分析变成可被第三方审计和复现的轨迹。

# %%
for rec in st.provenance:
    req = ", ".join(f"{s}{ks}" for s, ks in rec["requires"].items()) or "∅"
    pro = ", ".join(f"{s}{ks}" for s, ks in rec["produces"].items()) or "∅"
    print(f"  step {rec['step']}: {rec['function'].split('.')[-1]}")
    print(f"          requires: {req}")
    print(f"          produces: {pro}")

# %% [markdown]
# ### `st.summary()` —— 一眼看全:槽 + provenance 步数

# %%
print(st.summary())

# %% [markdown]
# ---
# ## 小结:这条链对标什么,socialverse 差在哪
#
# **对标现实工具**:这整套「注册表 + 查询面」就是 omicverse 的 `ov.registry` /
# `ov.utils.registry_lookup`——同样的 `find / get_prerequisites / resolve_plan` 三件套,
# 同样的 agent 打印布局。
#
# **socialverse 的差异/价值**:
#
# 1. **注册表 grounding(查而非猜)**:agent 不背 API,而是 `registry_lookup` → `resolve_plan`
#    读出契约与执行顺序;`requires` 未满足直接抛 `RegistryError` 并指明缺口与产出者——
#    杜绝「在错误的态上算出一个看似合理的假结果」。
# 2. **丢容器、留脊柱**:社科数据不可通约,所以用轻量 12 槽 `StudyState` 当**词汇表**而非
#    统一数据矩阵——只让依赖可检查,不强行把问卷/语料/网络塞进同一个容器。
# 3. **证据链内建**:每次注册调用自动写 `provenance`,一次完成的分析自带可复现/可审计轨迹——
#    这在社科(而非生信)里是第一等的关切。
# 4. **治理是一等公民**:`ethics_check / redact_pii / data_use_check / ai_use_disclosure`
#    是带契约的注册函数,不是事后补丁。
#
# 后续 notebook 会各展开一条链(因果 DID、复杂抽样、质性编码、文献核验、校勘学),
# 但机制都是这一根脊柱。
