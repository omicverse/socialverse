# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 理论透镜与网络:福柯 / 布迪厄 / 韦伯 + 社会网络分析
#
# **对标现实工具:** 前三步(理论透镜)对标的是**一整套没有软件的学术传统**——
# 你在任何 R/Python 包里都找不到 `foucault()`;学者靠**读书、笔记、手工编码**做话语分析、
# 场域分析、理想类型比较。最后一步(社会网络)对标 `networkx` / `igraph` / `Gephi`。
#
# 这条链要证明的一件事:socialverse 不是又一个统计库,而是一张**把「解释性主张 → 证据锚点」
# 也纳入契约的注册表**。三种社会理论透镜——福柯的**话语/权力-知识**、布迪厄的**场域/资本**、
# 韦伯的**理想类型**——被移植到同一条 `StudyState` / `registry` 脊柱上:
#
# - **透镜不是估计量,而是一套阅读协议(reading protocol)。** 它的产出刻意是**结构化的**
#   (协议、位置、评分、claim→evidence 骨架),**从不是一句占位字符串**;每一条解释都挂着
#   `support_units` / `capital_indicators` / `deviation` 之类的**证据指针**,让解释**可被定位、
#   可被反驳**。这正是「无软件的学术传统」最缺、而注册表能补上的东西:**主张的可追溯性**。
# - **网络分析则是真算**:`networkx` 建图,度 / 介数 / 特征向量三种中心性 + greedy-modularity
#   社群,全部在图上计算,不是桩。
#
# ## 涉及的 socialverse 函数(全部带注册契约)
#
# | 函数 | 域 | requires → produces | 对标现实工具 |
# |---|---|---|---|
# | `sv.pp.build_corpus` | prep | `sources['corpora']` → `corpus['units','documents','manifest']` | 手工分句/分段(NFC 规范化) |
# | `sv.tl.code_themes` | qual | `corpus['units']` → `codes['themes','segments',…]` | NVivo / ATLAS.ti / Braun&Clarke |
# | `sv.tl.foucault_discourse` | lens | `corpus['units']` → `evidence['claim_evidence']` | 无软件(archaeology/genealogy 阅读) |
# | `sv.tl.bourdieu_field` | lens | `codes['themes']`+`variables['constructs']` → `models['field_map']`, `evidence['claim_evidence']` | `prince`(MCA)/ 手工对应分析 |
# | `sv.tl.weber_ideal_type` | lens | `sources['datasets']` → `models['ideal_type']`, `diagnostics['coverage']`, `governance['ethics']`, `evidence['claim_evidence']` | 无软件(Idealtypus / Verstehen 比较) |
# | `sv.tl.build_network` | net | `sources['datasets']`(边表)→ `models['network']`, `diagnostics['coverage']` | `networkx` / `igraph` / `Gephi` |
#
# ## StudyState 会被填哪些槽(12 槽词汇表里的子集)
#
# - **`sources`** — 登记的原始输入:`corpora`(语料)、`datasets`(理想类型的案例表 / 网络的边表)。
# - **`corpus`** — 文本即数据态:`units`(带字符偏移的可编码单元)、`documents`、`manifest`。
# - **`codes`** — 质性编码态:`themes` / `segments` / `codebook` / `theme_map`(布迪厄透镜会读 `themes`)。
# - **`variables`** — 变量/建构:`constructs`(布迪厄的资本维度名)。
# - **`models`** — 拟合结果:`field_map`(场域位置空间)、`ideal_type`(理想类型标尺+打分)、`network`(网络结构)。
# - **`diagnostics`** — 诊断:`coverage`(理想类型的维度覆盖 / 网络的连通分量)。
# - **`governance`** — 治理:`ethics`(韦伯透镜写入的**价值中立 Wertfreiheit** 声明)。
# - **`evidence`** — 证据链:`claim_evidence`(每种透镜的 claim→evidence 骨架)、`provenance`。
#
# 此外,每个被 `@register` 包裹的函数**在成功调用后会自动向 `st.provenance`(只读追加账本)记一笔**,
# 带上它声明的 `requires`/`produces`——这就是社科/人文最看重的**可复现证据脊柱**,`st.summary()`
# 结尾会把它数出来。

# %% [markdown]
# ## 0. 环境与注册表:先查,不猜
#
# socialverse 的设计命题(见 README):让分析 agent 可靠的不是一个统一数据容器(AnnData),
# 而是一张**带依赖契约的可查询函数注册表**。所以我们做任何事之前先**查注册表**——这正是
# grounding 的全部意义:`registry_lookup` 让 agent「查而非猜」API。
#
# 下面先打印这四个理论/网络函数在注册表里的样子,确认它们的 `requires → produces` 契约,以及
# 它们对标的现实后端(注意:三种透镜的 `Backend` 都写着方法论名词而非某个 pip 包——因为现实里
# **本就没有这样的软件**)。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接存文件

import json
import os
import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import font_manager

import socialverse as sv
from socialverse import datasets as ds

# 让图里的中文标题正常渲染:挑一个装了的 CJK 字体(缺则回落,不报错)
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _cjk in ("Arial Unicode MS", "Songti SC", "STHeiti", "Hiragino Sans GB",
             "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei"):
    if _cjk in _installed:
        plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

try:
    from IPython.display import display  # 在 notebook 里用 display() 渲染 DataFrame
except Exception:  # 当普通脚本跑时回落到 print
    def display(x):
        print(x)


def show(title, obj):
    """把一个 dict/list 结果连同小标题打印出来(教学用)。"""
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


# 图存到 .py 同目录(而非运行时 cwd),这样下方 markdown 的 ![](fig.png) 才指得对。
# 当作 notebook 跑时 __file__ 可能不存在,回落到当前目录。
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pragma: no cover - notebook 内核里没有 __file__
    _HERE = os.getcwd()


def fig_path(name):
    """图文件的绝对保存路径(始终落在 notebook 同目录)。"""
    return os.path.join(_HERE, name)


print(f"socialverse v{sv.__version__} — 注册表里有 {len(sv.registry)} 个函数\n")

# 查询面:registry_lookup —— OmicOS kernel 打给 agent 看的同一套格式。
# 逐个透镜查(注册表按名/别名/描述做子串+模糊匹配),看它们各自的 requires→produces 契约。
for _q in ("福柯 foucault", "布迪厄 bourdieu", "韦伯 理想类型"):
    print(sv.utils.registry_lookup(_q, max_results=1))

# %% [markdown]
# 注意上面每条的 **Requires / Produces**——这不是文档字符串,而是**机器可读的契约**。
# 三种透镜共享同一条脊柱,但各自 requires 不同的前置槽:福柯要 `corpus['units']`、
# 布迪厄要 `codes['themes']` + `variables['constructs']`、韦伯只要 `sources['datasets']`。
# 注册表还能反过来把「要得到某个透镜的产出,该按什么顺序跑」排成计划:`resolve_plan` 走依赖图,
# 把缺的槽用其**生产者**补齐,并把真正需要人给的输入列进 `needs_input`。

# %%
plan_f = sv.registry.resolve_plan("foucault_discourse")
show("resolve_plan('foucault_discourse') — 到话语透镜的最短合法链", plan_f)

plan_b = sv.registry.resolve_plan("bourdieu_field")
show("resolve_plan('bourdieu_field') — 布迪厄要两条前置腿(themes + constructs)", plan_b)

# %% [markdown]
# 读一下上面两个计划:
#
# - **福柯**:`build_corpus → code_themes → foucault_discourse`。有意思的是 `code_themes` 出现在
#   计划里,但福柯的 `requires` 只写了 `corpus['units']`——`code_themes` 是它声明的
#   **prerequisites.functions**(「话语分析前该先把材料编码过一遍」),注册表把这条也排进了链。
# - **布迪厄**:计划里出现了 `design_survey`——因为 `variables['constructs']` 这个槽的**生产者**
#   恰好是它;而 `needs_input` 明确告诉你哪些是**必须由人提供**的原始输入(语料、estimand)。
#   `escalations` 里那几条 `auto_fix=escalate` 是在说:这些自动补进来的步骤**该由人确认再跑**——
#   透镜是解释性的,不该被静默自动化。
#
# 这就是「查而非猜」:agent 不必背 API,注册表把**该按什么顺序、缺什么、什么要人拍板**都算好了。

# %% [markdown]
# ## 1. 契约是「活」的:未满足 requires 会抛 RegistryError
#
# 契约不是摆设。`foucault_discourse` 声明 `requires={'corpus': ['units']}`;若在一个**空 StudyState**
# 上直接调它,注册表包装器会在调用前检查、抛出 `sv.RegistryError`——**而且错误信息会告诉你哪个函数
# 能生产缺失的槽**。这就是 grounding 的运行时形态:系统不会让你在缺前置的情况下静默产出「看起来
# 像话语分析、其实没有任何材料支撑」的垃圾解释。对解释性方法来说,这条护栏尤其重要。
#
# **契约(requires→produces):** requires `corpus['units']` → (未满足,拒绝执行)。

# %%
st_empty = sv.StudyState()
try:
    sv.tl.foucault_discourse(st_empty)  # 缺 corpus['units']
    print("(未预期:没有抛错)")
except sv.RegistryError as err:
    print("如预期抛出 RegistryError —— 这是特性,不是 bug:\n")
    print(textwrap.indent(str(err), "    "))

# 布迪厄要两条腿,错误信息会把两个缺口都列出来,并各自指向生产者
st_empty2 = sv.StudyState()
cap_demo = pd.DataFrame({"economic": [3, 1], "cultural": [1, 3]}, index=["a", "b"])
try:
    sv.tl.bourdieu_field(st_empty2, capital_table=cap_demo)  # 缺 codes.themes + variables.constructs
    print("(未预期:没有抛错)")
except sv.RegistryError as err:
    print("\n布迪厄在空 state 上同样被拦下,且两个缺口都被点名:\n")
    print(textwrap.indent(str(err), "    "))

# %% [markdown]
# ## 2. 福柯话语透镜:结构化的追问协议 + 证据锚点
#
# 先把材料准备到位——这是链的前两步,和其它质性 notebook 一样:`build_corpus` 把语料切成带
# 字符偏移的**可编码单元**,`code_themes` 做一遍反身主题编码。有了 `corpus['units']`,福柯透镜
# 才有合法输入。
#
# **契约(requires→produces):** requires `corpus['units']`(prereq 函数 `code_themes`)→
# produces `evidence['claim_evidence']`。
#
# 福柯透镜**刻意不做统计**——它是**考古学/系谱学**的一套固定追问网格:话语构成、可能性条件、
# 纳入/排除、规范化、权力-知识、主体化。对每一根轴,函数用保守的关键词把命中的 unit 挂成
# `support_units`,作为该轴解读的**证据锚点**;`claim` 字段留白,等研究者填入具体主张——但主张
# **必须**落在这些 unit 上。这就是把「无软件的阅读」变成**可追溯的 claim→evidence**。

# %%
st = sv.StudyState()
st.write("sources", "corpora", ds.load_corpus())   # 3 段访谈片段
sv.pp.build_corpus(st)                              # → corpus['units', ...]
sv.tl.code_themes(st)                               # → codes['themes', ...](满足福柯的 prereq)

sv.tl.foucault_discourse(st)                        # → evidence['claim_evidence']
ce_f = st.evidence["claim_evidence"]

print("foucault claim_evidence 顶层键:", list(ce_f.keys()))
print("lens:", ce_f["lens"], "· 覆盖 units 数:", ce_f["n_units"])
print("stance:", ce_f["stance"])

# 把六轴追问协议 + 各轴挂到的证据锚点排成一张表(教学最直观)
rows_f = [
    {
        "axis(轴)": r["axis"],
        "method": r["method"],
        "n_support": r["n_support"],
        "support_units(证据锚点)": ", ".join(r["support_units"][:3]) + ("…" if len(r["support_units"]) > 3 else ""),
        "question(福柯之问)": r["question"],
    }
    for r in ce_f["readings"]
]
df_f = pd.DataFrame(rows_f)
print("\n福柯话语追问协议(每轴一行,claim 待研究者填,但已锚定到具体 units):")
display(df_f)

# %% [markdown]
# 上表每一行 = 福柯分析的一根轴,`support_units` 是**触发该范畴的具体 unit_id**。换句话说,当你
# 之后写下「这段话语通过『专家资格』实现了权力-知识」这样的解释,系统能反查你到底在**哪几段文本**
# 上这么读——解释因此**可被定位、可被同行反驳**。这正是无软件传统里最容易丢失、而注册表强制保留的东西。
# 下面单看 `power_knowledge`(权力-知识)这根轴的完整读法条目:

# %%
pk = next(r for r in ce_f["readings"] if r["axis"] == "power_knowledge")
show("readings · power_knowledge(权力-知识)整条", pk)

# %% [markdown]
# ## 3. 布迪厄场域:actor × 资本 → 位置空间 + 位置-立场同源性
#
# 布迪厄的**场域(field)**是一个**位置空间**:行动者按其**资本**(经济/文化/社会/符号)的
# **总量与结构**分布其中。这一步有真数值——函数把 actor × capital 表投影到二维(装了 `prince`
# 就走**MCA 对应分析**,否则回落到从零实现的**中心化 SVD-PCA**),得到每个 actor 的坐标。
#
# 但布迪厄透镜的**契约**要求先有两条腿:`codes['themes']`(立场空间/position-taking 的来源)+
# `variables['constructs']`(资本维度的名字)。第 2 步的 `code_themes` 已经把 `themes` 填好了;
# 我们再声明 `constructs`,才满足 requires。
#
# **契约(requires→produces):** requires `codes['themes']` + `variables['constructs']` →
# produces `models['field_map']`, `evidence['claim_evidence']`。
#
# 关键点是**同源性(homology)**:每个 actor 的坐标都被**溯源回它自己那几个独立资本指标**——
# 位置不是凭空落点,而是可解释为「因为经济资本高、文化资本低,所以落在这里」。

# %%
# 一张小的 actor × 资本表:四位行动者在「经济资本 / 文化资本」两维上的量
capital = pd.DataFrame(
    {"economic": [9, 2, 5, 3],
     "cultural": [2, 9, 5, 7]},
    index=["实业家", "学者", "中间派", "青年艺术家"],
)
print("actor × capital 输入表(行=行动者,列=资本维度):")
display(capital)

st.write("variables", "constructs", ["economic", "cultural"])  # 满足 requires 的第二条腿
sv.tl.bourdieu_field(st, capital_table=capital)                # 读 codes['themes'] + 本表
fm = st.models["field_map"]

print("\nfield_map 方法:", fm["method"], "· 资本维度:", fm["capital_dims"])
print("两主轴解释方差:", [round(v, 3) for v in fm["explained_variance"]])
print("axes 语义:", fm["axes"])
print("\n位置空间坐标(每个行动者一个二维位置):")
display(pd.DataFrame(fm["positions"], index=["axis_1", "axis_2"]).T.round(3))

# %% [markdown]
# 现在看**同源性骨架**:`evidence['claim_evidence']['homology']` 里,每个 actor 都带着它的
# `capital_indicators`(把坐标溯源回原始资本值)和一个待填的 `claim`(它的立场与资本结构的
# 同源性主张)。`themes_available=True` 表示第 2 步的主题编码在场——布迪厄意义上,你可以进一步
# 去读**位置空间(资本)↔ 立场空间(themes)的同源性**。

# %%
homology = ce_b = st.evidence["claim_evidence"]
print("bourdieu claim_evidence 顶层键:", list(ce_b.keys()))
print("themes_available(立场空间是否在场):", ce_b["themes_available"])
show("homology · 第一位行动者(坐标已溯源到其独立资本指标)", ce_b["homology"][0])

# %% [markdown]
# ### 画出场域:位置空间散点图
#
# 布迪厄的场域天然是**要画出来的**——横轴 ≈ 总资本量/结构主轴,纵轴 ≈ 资本构成(经济↔文化)。
# 我们把 `field_map['positions']` 直接画成散点,标注每个行动者。这不是新调用,只是把注册表产出的
# 坐标可视化。

# %%
fig, ax = plt.subplots(figsize=(6.4, 5.2))
pos = fm["positions"]
xs = [pos[a][0] for a in pos]
ys = [pos[a][1] for a in pos]
ax.scatter(xs, ys, s=220, c="#4C72B0", edgecolors="white", linewidths=1.5, zorder=3)
for a in pos:
    ax.annotate(a, (pos[a][0], pos[a][1]), textcoords="offset points",
                xytext=(8, 8), fontsize=11, fontweight="bold")
ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)
ev = fm["explained_variance"]
ax.set_xlabel(f"axis 1 · 总资本量/结构主轴  (解释方差 {ev[0]:.0%})")
ax.set_ylabel(f"axis 2 · 资本构成 经济↔文化  (解释方差 {ev[1]:.0%})")
ax.set_title("布迪厄场域:行动者的位置空间(actor × capital 投影)")
fig.tight_layout()
fig.savefig(fig_path("fig_bourdieu_field.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("已保存 fig_bourdieu_field.png ->", fig_path("fig_bourdieu_field.png"))

# %% [markdown]
# ![](fig_bourdieu_field.png)
#
# 「实业家」(经济高、文化低)与「学者」(文化高、经济低)落在主轴两端——这正是布迪厄
# 《区分》里**统治阶级内部**沿资本构成对立的经典图景;「中间派」居中,「青年艺术家」偏文化端。
# 每个点的位置都能通过 `capital_indicators` 溯源,不是随手一放。

# %% [markdown]
# ## 4. 韦伯理想类型:纯粹类型标尺 → 逐案打分 → Verstehen 解释偏离
#
# 韦伯的**理想类型(Idealtypus)**是一个**刻意一侧强调的分析建构**——没有任何真实案例会与它
# 完全吻合,而**案例偏离纯粹类型的程度**才承载解释重量(这就是 Verstehen:解释性理解)。
# 这一步也有真数值:`schema={维度: 纯粹极描述}` 定义纯粹极,每个案例在每一维上被算成一个
# **到纯粹极的距离 ∈ [0,1]**(1=贴近纯粹极),偏离 = 1 − 贴近度。
#
# **契约(requires→produces):** requires `sources['datasets']` → produces `models['ideal_type']`,
# `diagnostics['coverage']`, `governance['ethics']`, `evidence['claim_evidence']`。
#
# ### 4a. 先照 skeleton 直调一次:一个「跑通但空心」的教训
#
# skeleton 里给的 schema 是 `{individualism, hierarchy}`,但喂进去的**案例表是 survey 的 item 列**,
# 里面根本没有这两列——于是每个维度打分为 NaN、`coverage.overall = 0`、Verstehen 的
# `deviations` 是**空的**。函数**不报错**(它优雅产出空骨架),但 `diagnostics['coverage']` 会
# **如实告诉你覆盖率是 0**。这正是诊断槽的价值:**契约让「跑通」和「有意义」是两件事,并把
# 后者量化出来**。

# %%
stw0 = sv.StudyState()
stw0.write("sources", "datasets", ds.load_survey())
sv.tl.weber_ideal_type(
    stw0,
    schema={"individualism": "collectivist..individualist", "hierarchy": "flat..hierarchical"},
    cases=ds.load_survey().head(6),   # 案例=survey items,不含 schema 里的维度列
)
print("ideal_type dims:", stw0.models["ideal_type"]["dimensions"])
print("coverage.overall(维度覆盖率):", stw0.diagnostics["coverage"]["overall"])
print("Verstehen deviations 条目数:", len(stw0.evidence["claim_evidence"]["deviations"]),
      "  ← 覆盖率 0 → 无偏离可解释(跑通 ≠ 有意义)")

# %% [markdown]
# ### 4b. 做对:让 schema 的维度就是案例表的数值列
#
# 现在给一个**真正韦伯式**的例子:沿韦伯**法理型支配(legal-rational / bureaucracy)**的三根轴
# ——**依规程度 rule_bound、非人格化 impersonal、科层等级 hierarchy**——为若干组织给分,纯粹极
# 就是「完全官僚制」。这些维度**正是案例表的数值列**,于是覆盖率满、逐案偏离全部算得出来。

# %%
cases = pd.DataFrame(
    {"rule_bound": [9, 7, 2, 5, 8],   # 依成文规程的程度
     "impersonal": [8, 6, 1, 4, 9],   # 对事不对人(非人格化)
     "hierarchy":  [9, 5, 2, 6, 7]},  # 科层等级严格度
    index=["普鲁士文官系统", "现代企业", "先知运动", "创业公司", "天主教会"],
)
schema = {
    "rule_bound": "凭好恶裁量 .. 严格依成文规则(法理型纯粹极)",
    "impersonal": "对人不对事 .. 完全非人格化(对事不对人)",
    "hierarchy":  "扁平 .. 严格科层等级",
}
print("案例 × 维度输入表(纯粹极 = 完全官僚制):")
display(cases)

stw = sv.StudyState()
stw.write("sources", "datasets", cases)
sv.tl.weber_ideal_type(stw, schema=schema, cases=cases)
it = stw.models["ideal_type"]

print("\n维度:", it["dimensions"], "· 案例数:", it["n_cases"])
print("coverage.overall:", stw.diagnostics["coverage"]["overall"], "(=1.0,全维度可打分)")
print("\n贴近纯粹极的评分 score∈[0,1](1=贴近『完全官僚制』极):")
display(it["scores"])
print("每个案例到纯粹类型的平均偏离(case_mean_deviation,越大越『非官僚』):")
display(pd.Series(it["case_mean_deviation"], name="mean_deviation").round(3).sort_values(ascending=False))

# %% [markdown]
# **偏离 → Verstehen**:`evidence['claim_evidence']['deviations']` 按偏离从大到小排好,每条是
# 「某案例在某一维上最偏离纯粹类型」,并挂着原始证据值 `value` 和一个待填的 `claim`。最偏离的是
# **先知运动**——它在依规、非人格化、科层三维上全部贴近**反面极**,正是韦伯用来和法理型对照的
# **卡里斯玛型支配(charismatic authority)**。理想类型的意义**恰恰在偏离处显影**。

# %%
ce_w = stw.evidence["claim_evidence"]
show("Verstehen · 偏离最大的三条(每条待研究者做解释性理解)", ce_w["deviations"][:3])

# %% [markdown]
# ### 韦伯透镜自带的治理护栏:价值中立(Wertfreiheit)
#
# 韦伯坚持**价值中立**:理想类型的评分是**分析性的「与纯粹类型的距离」,不是对案例价值/优劣的
# 评判**。这条透镜把这一点写进了 `governance['ethics']` 槽——不是注释,而是**结构化的治理记录**,
# 会随证据链一起留痕。这是 socialverse 相对「无软件传统」的又一处补强:**方法论的伦理承诺被
# 显式登记、可审计**。

# %%
show("governance['ethics'] · 韦伯透镜写入的价值中立声明", stw.governance["ethics"])

# %% [markdown]
# ## 5. 社会网络分析:三种中心性讲三个不同的故事 + 社群
#
# 前四步是解释性透镜;最后一步换成**真算**。`build_network` 把一张**边表**变成图(`networkx`),
# 然后读它的结构:三种**中心性**(度 / 介数 / 特征向量)、**密度**、**greedy-modularity 社群**。
#
# **契约(requires→produces):** requires `sources['datasets']`(边表)→
# produces `models['network']`, `diagnostics['coverage']`(连通分量结构)。注意它的
# `auto_fix='none'`——边表必须由人给,注册表不会替你伪造网络。
#
# 我们造一张**求助/请教网络**:七位研究者,谁向谁请教(带权重=请教频次)。它有意被设计成
# **两个松散簇 + 桥接者**,好让三种中心性讲出**不同**的故事。

# %%
edges = pd.DataFrame({
    "source": ["Ada", "Ben", "Cai", "Ada", "Cai", "Dev", "Eve", "Fei", "Dev", "Fei", "Cai", "Gao", "Eve", "Fei"],
    "target": ["Ben", "Cai", "Ada", "Cai", "Dev", "Eve", "Fei", "Dev", "Fei", "Eve", "Dev", "Fei", "Gao", "Gao"],
    "weight": [3, 2, 1, 2, 4, 1, 3, 2, 1, 2, 1, 1, 2, 1],
})
print("求助网络边表(source 向 target 请教,weight=频次):")
display(edges)

stn = sv.StudyState()
stn.write("sources", "datasets", edges)
sv.tl.build_network(stn, edges=edges, source="source", target="target",
                    weight="weight", directed=False, top_k=8)
net = stn.models["network"]

print("\n网络规模:", net["n_nodes"], "节点 /", net["n_edges"], "边 · 密度 =", round(net["density"], 3),
      "· 平均度 =", round(net["avg_degree"], 3), "· 加权 =", net["weighted"])
show("diagnostics['coverage'] · 连通分量结构", stn.diagnostics["coverage"])

# %% [markdown]
# 把三种中心性并排成一张表——**这是本步的教学核心**:它们量的是**不同的「重要」**。

# %%
cent = net["centrality"]
nodes = sorted(set(list(cent["degree"]) + list(cent["betweenness"]) + list(cent["eigenvector"])))
df_c = pd.DataFrame({
    "degree(度:直接联系数)": pd.Series(cent["degree"]),
    "betweenness(介数:桥接/掮客)": pd.Series(cent["betweenness"]),
    "eigenvector(特征向量:嵌入核心)": pd.Series(cent["eigenvector"]),
}).reindex(nodes).round(3)
print("三种中心性对照(同一批节点,三种『重要』):")
display(df_c)

show("communities · greedy-modularity 社群划分", net["communities"])

# %% [markdown]
# 读这张表——三种中心性**明确指向不同的人**:
#
# - **度中心性**:Cai / Dev / Eve / Fei 并列最高——直接联系最多。
# - **介数中心性**:**Dev(0.63)、Cai(0.53)** 遥遥领先——它们卡在两个簇之间,是**掮客(broker)**,
#   信息/请教要经它们中转;这正是 Burt 的**结构洞**位置。
# - **特征向量中心性**:反而是 **Ada / Ben(≈0.59)** 最高——它们身处 Ada-Ben-Cai 这个**紧密三角**里,
#   「和重要的人相连才算重要」,而非联系数最多。
#
# 同一张图,「谁最重要」的答案随你问的是**联系数、桥接力、还是核心嵌入**而不同——这是网络分析
# 最经典的一课。greedy-modularity 把七人**干净地切成两个社群(Q ≈ 0.43)**,与我们造图时的两簇吻合。

# %% [markdown]
# ### 画出网络:节点大小=介数,颜色=社群
#
# 最后把网络画出来:**节点大小按介数中心性**(掮客更大)、**颜色按 greedy-modularity 社群**、
# 边宽按请教频次。这样上面三段文字的结论能**一眼看见**。图的坐标由 `networkx` 的
# spring 布局给定(固定随机种子以可复现);中心性/社群数值全部来自注册表产出的 `models['network']`。

# %%
# 用同一张边表在 networkx 里重建图,仅用于布局与画图(数值仍取自 net)
G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="weight")
btw = cent["betweenness"]

# 社群 → 每个节点一个颜色
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
# 从 greedy-modularity 重算一次分区仅为着色(与注册表报告的 n_communities 一致)
from networkx.algorithms.community import greedy_modularity_communities
parts = list(greedy_modularity_communities(G, weight="weight"))
node_color = {}
for ci, part in enumerate(parts):
    for n in part:
        node_color[n] = palette[ci % len(palette)]

pos = nx.spring_layout(G, seed=1, weight="weight")
sizes = [300 + 4200 * btw.get(n, 0.0) for n in G.nodes()]      # 介数越高节点越大
colors = [node_color.get(n, "#999999") for n in G.nodes()]
widths = [0.6 + 0.9 * G[u][v]["weight"] for u, v in G.edges()]

fig, ax = plt.subplots(figsize=(7.2, 5.6))
nx.draw_networkx_edges(G, pos, width=widths, edge_color="#c9c9c9", ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                       edgecolors="white", linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
q = net["communities"].get("modularity")
ax.set_title(f"求助网络:节点大小=介数中心性(掮客更大),颜色=社群(Q≈{q:.2f})")
ax.axis("off")
fig.tight_layout()
fig.savefig(fig_path("fig_advice_network.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("已保存 fig_advice_network.png ->", fig_path("fig_advice_network.png"))

# %% [markdown]
# ![](fig_advice_network.png)
#
# 图里 **Dev 和 Cai 明显更大**(高介数,坐在两色社群的接缝上,是掮客);两种颜色对应
# greedy-modularity 切出的两个社群。眼睛看到的和上表的数字一致。

# %% [markdown]
# ## 6. 证据脊柱:一条链跑完,provenance 自证其身
#
# 这条 notebook 里,我们在**同一个 `st`** 上依次跑了 `build_corpus → code_themes →
# foucault_discourse → bourdieu_field`(网络与韦伯用了各自独立的 state 演示,不共享槽)。
# 每个 `@register` 函数成功后都自动向 `st.provenance` 追加了一笔带 `requires/produces` 的记录——
# 这就是**可复现的证据脊柱**。`st.summary()` 把「填了哪些槽 + 走了几步」一并数出来。

# %%
print("=== 主链 st(福柯+布迪厄共享)===")
print(st.summary())

print("\n=== provenance 逐步(function · requires → produces)===")
for rec in st.provenance:
    req = "+".join(f"{k}:{','.join(v)}" for k, v in rec["requires"].items()) or "∅"
    pro = "+".join(f"{k}:{','.join(v)}" for k, v in rec["produces"].items()) or "∅"
    print(f"  step {rec['step']}: {rec['function']}")
    print(f"           requires[{req}]  →  produces[{pro}]")

print("\n=== 韦伯 state 与 网络 state 各自的脊柱 ===")
print(stw.summary())
print()
print(stn.summary())

# %% [markdown]
# ## 小结:这条链对标什么,socialverse 补了什么
#
# **对标的现实工具:** 前三种透镜对标的是**没有软件的学术传统**——福柯的考古学/系谱学、布迪厄的
# 场域/资本分析、韦伯的理想类型比较,历来靠**读、记、手工编码**完成;社会网络这一步对标
# `networkx` / `igraph` / `Gephi`。
#
# **socialverse 的差异(注册表 grounding + 证据链):**
#
# 1. **把「解释」也纳入契约。** 透镜的产出**从不是一句占位字符串**,而是结构化的 claim→evidence
#    骨架:福柯的每根轴挂 `support_units`、布迪厄的每个位置溯源到 `capital_indicators`、韦伯的每条
#    偏离挂原始 `value`。解释因此**可被定位、可被反驳**——这是无软件传统最缺、注册表强制保留的东西。
# 2. **查而非猜。** `resolve_plan` 会把「福柯要先 code_themes、布迪厄要两条前置腿、哪些必须人给
#    (needs_input)、哪些该人确认(escalations)」全部算好;`RegistryError` 在缺前置时**拒绝
#    静默产出垃圾解释**。
# 3. **跑通 ≠ 有意义,且这件事被量化。** 韦伯 4a 那个 coverage=0 的例子说明:契约让诊断槽把
#    「函数没报错」和「结果有覆盖」分开,`diagnostics['coverage']` 把后者变成数字。
# 4. **方法论承诺被登记。** 韦伯的**价值中立**写进 `governance['ethics']`,随证据链可审计。
# 5. **证据脊柱自证。** 每步自动记 provenance,一条链跑完即得**可复现的审计轨迹**——
#    这正是社科/人文最看重、而普通脚本不会自带的东西。
