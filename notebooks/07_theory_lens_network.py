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
# # 用理论透镜读文本、用网络分析读关系
#
# 社会科学有两种很不一样的分析传统。一种是**解释性**的:研究者带着一套理论视角去读材料——福柯问「这段话语如何生产权力与知识」,布迪厄问「行动者在场域里占据什么位置、这位置由什么资本决定」,韦伯用「理想类型」这把刻意夸张的标尺去量一个真实案例偏离它多远。这一路没有现成软件,`foucault()` 这样的函数在任何 R/Python 包里都不存在,学者靠读书、做笔记、手工编码把主张一条条立起来。另一种是**计算性**的:社会网络分析把「谁和谁有关系」变成一张图,再用度中心性、介数中心性、特征向量中心性、社群划分这些量化指标,把「谁重要、谁是掮客、谁抱团」算出来——这一路有 `networkx` / `igraph` / `Gephi`。
#
# 这本教程把两条传统放进同一条工作流里走一遍:先用三种理论透镜读一批访谈与案例,再对一张求助网络做中心性与社群分析。三种透镜看似虚,但落到分析上都有可操作的骨架——福柯是一张固定的追问网格,每根轴都要指回具体的文本片段;布迪厄要把 actor × capital 表投影成一张位置空间的散点图;韦伯要给每个案例逐维打分、算出它离纯粹类型有多远。网络分析这一步则是实打实的图计算。贯穿全程的一个方法论要求是:**任何一条解释都要能被定位、被反驳**——福柯的每根轴挂着触发它的 `support_units`,布迪厄的每个坐标溯源到原始资本值,韦伯的每条偏离带着原始打分。这正是无软件传统最容易丢、也最该守住的东西。
#
# 我们用 `socialverse` 完成全流程,它是一套面向社会科学的分析库,把这些理论透镜和网络分析都做成了带证据锚点的函数。数据全部内置合成:一小批访谈片段供福柯/布迪厄透镜使用,一张 actor × capital 表给布迪厄的场域,一张组织 × 韦伯官僚制三维的案例表给理想类型,以及一张七人求助网络的边表。方法学背景可参考 Foucault《知识考古学》、Bourdieu《区分》、Weber《经济与社会》,以及网络分析的 Wasserman & Faust *Social Network Analysis*。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import os

import matplotlib.pyplot as plt
import networkx as nx
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

# 图存到 .py 同目录(而非运行时 cwd),这样下方 markdown 的 ![](fig.png) 才指得对
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # notebook 内核里没有 __file__
    _HERE = os.getcwd()


def fig_path(name):
    return os.path.join(_HERE, name)


# %% [markdown]
# ## 载入材料
#
# 福柯与布迪厄两种透镜都从文本出发,所以第一步是把语料准备成可编码的形态。内置的 `load_corpus` 给出三段访谈片段;`build_corpus` 把它们切成带字符偏移的**可编码单元**(每个单元有形如 `int01:0-184` 的 id,记录它来自哪份文档的哪一段),`code_themes` 再对这些单元做一遍反身主题编码。这两步和其它质性分析教程完全一样,是文本分析的通用起点。
#
# 之所以先做主题编码,是因为福柯透镜要读「已经被编码过一遍」的材料,而布迪厄的场域后面还要用到这批主题作为立场空间。下面把语料装进研究状态,跑完前两步,看一眼产出了哪些单元。

# %%
st = sv.StudyState()
st.write("sources", "corpora", ds.load_corpus())  # 3 段访谈片段

sv.pp.build_corpus(st)   # 切成带字符偏移的可编码单元 → corpus['units', ...]
sv.tl.code_themes(st)    # 反身主题编码 → codes['themes', ...]

units = st.corpus["units"]
print("语料单元数:", len(units))
print("主题数:", len(st.codes["themes"]))

# %% [markdown]
# ## 福柯:话语的追问协议
#
# 福柯的话语分析不是统计,而是一套**考古学/系谱学**的固定追问网格。`foucault_discourse` 把这张网格落成六根轴——话语构成、可能性条件、纳入/排除、规范化、权力-知识、主体化——对每一根轴,它用保守的关键词在语料单元里找命中的片段,把它们挂成这根轴的 `support_units`(证据锚点)。`claim` 字段刻意留白:具体的解释性主张由研究者填写,但这条主张**必须**落在这些被点名的单元上。这样一来,原本靠读书笔记完成的话语分析,就变成了一份可追溯的 claim→evidence 骨架。
#
# 跑完之后,产出落在 `evidence['claim_evidence']` 里。我们先看它覆盖了几个单元、整体是什么立场。

# %%
sv.tl.foucault_discourse(st)  # → evidence['claim_evidence']
ce_f = st.evidence["claim_evidence"]

print("透镜:", ce_f["lens"], "· 覆盖单元数:", ce_f["n_units"])
print("立场:", ce_f["stance"])

# %% [markdown]
# 把六根轴排成一张表最直观:每一行是福柯分析的一根轴,`method` 标出它属于考古学还是系谱学的追问,`n_support` 是这根轴锚定到的单元数,`support_units` 列出具体是哪几段文本触发了它。表里的 `question` 就是福柯之问的中文表述——你在写解释时正是在回答这些问题,而答案的证据必须来自对应那一行的单元。

# %%
df_f = pd.DataFrame([
    {
        "轴": r["axis"],
        "method": r["method"],
        "n_support": r["n_support"],
        "support_units": ", ".join(r["support_units"][:3]) + ("…" if len(r["support_units"]) > 3 else ""),
        "福柯之问": r["question"],
    }
    for r in ce_f["readings"]
])
df_f

# %% [markdown]
# 单看**权力-知识(power_knowledge)**这一根轴的完整条目:它是系谱学式的追问,`claim` 待研究者填,而三段访谈单元被点名为证据锚点。当你之后写下「这段话语通过某种专家资格实现了权力-知识」时,系统能反查你到底在哪几段文本上这么读——解释因此可被定位、可被同行反驳。

# %%
pk = next(r for r in ce_f["readings"] if r["axis"] == "power_knowledge")
print("轴   :", pk["axis"])
print("method:", pk["method"])
print("问   :", pk["question"])
print("claim:", pk["claim"])
print("证据锚点 support_units:", pk["support_units"])

# %% [markdown]
# ## 布迪厄:把行动者投进场域
#
# 布迪厄的**场域(field)**是一个位置空间:行动者按各自**资本**(经济、文化、社会、符号)的总量与结构分布其中。这一步有真数值——`bourdieu_field` 把一张 actor × capital 表投影到二维(装了 `prince` 就走 MCA 对应分析,否则回落到从零实现的中心化 SVD-PCA),给每个行动者一个坐标。关键概念是**同源性(homology)**:每个坐标都被溯源回该行动者自己的那几个资本指标,所以位置不是凭空落点,而是可解释为「因为经济资本高、文化资本低,所以落在这里」。
#
# 我们造一张小表:四位行动者在「经济资本 / 文化资本」两维上的量。实业家经济高文化低,学者反之,另有一个中间派和一个偏文化的青年艺术家。这张表就是场域分析的输入。

# %%
capital = pd.DataFrame(
    {"economic": [9, 2, 5, 3],
     "cultural": [2, 9, 5, 7]},
    index=["实业家", "学者", "中间派", "青年艺术家"],
)
capital

# %% [markdown]
# 布迪厄透镜除了读上面这张资本表,还要用到两样东西:主题编码(`code_themes` 已经填好了,作为立场空间)和一份资本维度的名字。我们把维度名声明进研究状态,再调用透镜。产出落在 `models['field_map']`,里面有投影用的方法、两条主轴各自解释的方差,以及每个行动者的二维坐标。

# %%
st.write("variables", "constructs", ["economic", "cultural"])  # 资本维度的名字
sv.tl.bourdieu_field(st, capital_table=capital)                # 读主题 + 本表 → 位置空间
fm = st.models["field_map"]

print("投影方法:", fm["method"])
print("两主轴解释方差:", [round(v, 3) for v in fm["explained_variance"]])
print("axis 语义:", fm["axes"])

# %% [markdown]
# 每个行动者的位置坐标。第一主轴解释了绝大部分方差(约 99%),对应「总资本量/结构主轴」;第二主轴对应「资本构成:经济↔文化」。实业家落在主轴一端、学者落在另一端,正是《区分》里统治阶级内部沿资本构成对立的经典图景。

# %%
pd.DataFrame(fm["positions"], index=["axis_1", "axis_2"]).T.round(3)

# %% [markdown]
# 同源性骨架存在 `evidence['claim_evidence']['homology']` 里,每个行动者一条,带着把坐标溯源回原始资本值的 `capital_indicators` 和一个待填的 `claim`。看第一位「实业家」:它的坐标 (−5.66, −0.25) 被明确关联到 economic=9、cultural=2 这两个原始指标上。`themes_available=True` 表示立场空间在场,意味着你可以进一步去读「资本位置 ↔ 立场主题」的同源性。

# %%
ce_b = st.evidence["claim_evidence"]
print("themes_available(立场空间是否在场):", ce_b["themes_available"])

h0 = ce_b["homology"][0]
print("\n行动者   :", h0["actor"])
print("位置坐标 :", [round(x, 3) for x in h0["position"]])
print("资本指标 :", h0["capital_indicators"])
print("待填 claim:", h0["claim"])

# %% [markdown]
# 布迪厄的场域天然是要画出来的。我们把 `field_map['positions']` 直接画成散点,横轴是总资本量/结构主轴、纵轴是资本构成(经济↔文化),标注每个行动者。这不是新调用,只是把上面产出的坐标可视化。

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
ax.set_title("布迪厄场域:行动者的位置空间")
fig.tight_layout()
fig.savefig(fig_path("fig_bourdieu_field.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("已保存 fig_bourdieu_field.png")

# %% [markdown]
# ![布迪厄场域散点图](fig_bourdieu_field.png)
#
# 「实业家」(经济高、文化低)与「学者」(文化高、经济低)落在主轴两端,「中间派」居中,「青年艺术家」偏文化端。每个点的位置都能通过 `capital_indicators` 溯源,不是随手一放。

# %% [markdown]
# ## 韦伯:理想类型这把标尺
#
# 韦伯的**理想类型(Idealtypus)**是一个刻意向一侧夸张的分析建构——没有任何真实案例会与它完全吻合,而**案例偏离纯粹类型的程度**才承载解释重量。这就是 Verstehen(解释性理解):意义恰恰在偏离处显影。`weber_ideal_type` 把这套逻辑做成真数值:你给一个 `schema`(每个维度配一句「从反面极到纯粹极」的描述)定义标尺,再给一张案例 × 维度的打分表,函数就为每个案例在每一维上算出一个到纯粹极的贴近度(∈[0,1],1=贴近纯粹极),偏离 = 1 − 贴近度。
#
# 这里有一个容易踩的坑,值得先演示一遍。理想类型只有当 **schema 的维度就是案例表里真实存在的数值列**时才有意义;若两者对不上,函数不会报错,但会算出一堆 NaN、覆盖率为 0、没有任何偏离可解释。下面先故意喂一份对不上的输入,看诊断怎么如实告诉你「跑通了,但空心」。

# %%
stw0 = sv.StudyState()
stw0.write("sources", "datasets", ds.load_survey())
sv.tl.weber_ideal_type(
    stw0,
    schema={"individualism": "collectivist..individualist", "hierarchy": "flat..hierarchical"},
    cases=ds.load_survey().head(6),   # 案例=survey 的 item 列,不含 schema 里的维度
)
print("声明的维度:", stw0.models["ideal_type"]["dimensions"])
print("coverage.overall(维度覆盖率):", stw0.diagnostics["coverage"]["overall"])
print("可解释的偏离条目数:", len(stw0.evidence["claim_evidence"]["deviations"]),
      "  ← 覆盖率 0,跑通但无意义")

# %% [markdown]
# 覆盖率是 0,偏离列表是空的——函数没报错,但 `diagnostics['coverage']` 把「有没有意义」量化成了一个数字。现在把它做对:沿韦伯**法理型支配(bureaucracy)**的三根轴——依规程度 rule_bound、非人格化 impersonal、科层等级 hierarchy——给五个组织打分,纯粹极就是「完全官僚制」。这三个维度正是案例表的数值列,于是覆盖率会满、每个案例的偏离都算得出来。

# %%
cases = pd.DataFrame(
    {"rule_bound": [9, 7, 2, 5, 8],   # 依成文规程的程度
     "impersonal": [8, 6, 1, 4, 9],   # 对事不对人(非人格化)
     "hierarchy":  [9, 5, 2, 6, 7]},  # 科层等级严格度
    index=["普鲁士文官系统", "现代企业", "先知运动", "创业公司", "天主教会"],
)
cases

# %% [markdown]
# 给每个维度配一句「反面极 .. 纯粹极」的标尺描述,纯粹极即完全官僚制,然后调用透镜。产出的 `models['ideal_type']` 里有逐案逐维的贴近度评分,以及每个案例的平均偏离。

# %%
schema = {
    "rule_bound": "凭好恶裁量 .. 严格依成文规则(法理型纯粹极)",
    "impersonal": "对人不对事 .. 完全非人格化(对事不对人)",
    "hierarchy":  "扁平 .. 严格科层等级",
}
stw = sv.StudyState()
stw.write("sources", "datasets", cases)
sv.tl.weber_ideal_type(stw, schema=schema, cases=cases)
it = stw.models["ideal_type"]

print("维度:", it["dimensions"], "· 案例数:", it["n_cases"])
print("coverage.overall:", stw.diagnostics["coverage"]["overall"], "(=1.0,全维度可打分)")

# %% [markdown]
# 逐案逐维的贴近度评分,1 表示贴近「完全官僚制」这个纯粹极。普鲁士文官系统在三维上都接近 1,是最典型的官僚制;先知运动三维都很低。

# %%
it["scores"]

# %% [markdown]
# 把每个案例到纯粹类型的平均偏离从大到小排出来:偏离越大越「非官僚」。先知运动偏离最大,现代企业、创业公司居中,普鲁士文官系统最贴近纯粹类型。

# %%
pd.Series(it["case_mean_deviation"], name="mean_deviation").round(3).sort_values(ascending=False)

# %% [markdown]
# 偏离最大的三条被排在 `evidence['claim_evidence']['deviations']` 最前面,每条是「某案例在某一维上最偏离纯粹类型」,挂着原始打分 `value` 和一个待填的 `claim`。最偏离的是**先知运动**——它在依规、非人格化、科层三维上全部贴近反面极,正是韦伯用来和法理型对照的**卡里斯玛型支配(charismatic authority)**。理想类型的意义,恰恰在这些偏离处显影。

# %%
ce_w = stw.evidence["claim_evidence"]
for d in ce_w["deviations"][:3]:
    print(f"{d['case']} · {d['dimension']}  偏离={d['deviation']}  原始值={d['value']}")

# %% [markdown]
# 韦伯坚持**价值中立(Wertfreiheit)**:理想类型的评分是分析性的「与纯粹类型的距离」,不是对案例价值或优劣的评判。这条透镜把这一承诺写进了 `governance['ethics']` 槽——不是注释,而是随分析一起留痕的结构化记录。

# %%
eth = stw.governance["ethics"]
print("原则:", eth["principle"])
print("声明:", eth["statement"])

# %% [markdown]
# ## 网络:三种中心性讲三个故事
#
# 前面三步是解释性透镜,最后一步换成实打实的图计算。`build_network` 把一张**边表**变成 `networkx` 图,再读它的结构:三种中心性(度、介数、特征向量)、密度、以及 greedy-modularity 社群划分。边表必须由人提供,函数不会替你伪造网络。
#
# 我们造一张求助/请教网络:七位研究者,谁向谁请教(权重=请教频次)。它被有意设计成两个松散簇加一个桥接结构,好让三种中心性讲出不同的故事——度中心性量「直接联系多不多」,介数中心性量「是不是卡在别人之间的掮客」,特征向量中心性量「有没有和重要的人相连」。

# %%
edges = pd.DataFrame({
    "source": ["Ada", "Ben", "Cai", "Ada", "Cai", "Dev", "Eve", "Fei", "Dev", "Fei", "Cai", "Gao", "Eve", "Fei"],
    "target": ["Ben", "Cai", "Ada", "Cai", "Dev", "Eve", "Fei", "Dev", "Fei", "Eve", "Dev", "Fei", "Gao", "Gao"],
    "weight": [3, 2, 1, 2, 4, 1, 3, 2, 1, 2, 1, 1, 2, 1],
})
edges

# %% [markdown]
# 把边表装进一个新的研究状态,建成无向加权图。产出的 `models['network']` 报告了网络规模、密度、平均度,`diagnostics['coverage']` 记录了连通分量结构——这里七个节点全部落在同一个连通分量里。

# %%
stn = sv.StudyState()
stn.write("sources", "datasets", edges)
sv.tl.build_network(stn, edges=edges, source="source", target="target",
                    weight="weight", directed=False, top_k=8)
net = stn.models["network"]

print("节点数:", net["n_nodes"], "· 边数:", net["n_edges"],
      "· 密度:", round(net["density"], 3), "· 平均度:", round(net["avg_degree"], 3))
cov = stn.diagnostics["coverage"]
print("连通分量:", cov["n_components"], "个 · 最大分量含", cov["largest_cc_size"], "个节点")

# %% [markdown]
# 把三种中心性并排成一张表,这是本步的教学核心:同一张图上,「谁最重要」的答案随你问的是联系数、桥接力、还是核心嵌入而不同。

# %%
cent = net["centrality"]
nodes = sorted(set(cent["degree"]) | set(cent["betweenness"]) | set(cent["eigenvector"]))
df_c = pd.DataFrame({
    "degree(度)": pd.Series(cent["degree"]),
    "betweenness(介数)": pd.Series(cent["betweenness"]),
    "eigenvector(特征向量)": pd.Series(cent["eigenvector"]),
}).reindex(nodes).round(3)
df_c

# %% [markdown]
# 三种中心性明确指向不同的人。**度中心性**里 Cai / Dev / Eve / Fei 并列最高,直接联系最多。**介数中心性**里 Dev(0.63)和 Cai(0.53)遥遥领先——它们卡在两个簇之间,是掮客(broker),请教要经它们中转,这正是 Burt 的**结构洞**位置。**特征向量中心性**反而是 Ada / Ben(≈0.59)最高——它们身处 Ada-Ben-Cai 这个紧密三角里,「和重要的人相连才算重要」,而不是联系数最多。
#
# 社群划分则把这七个人切成结构上的抱团。greedy-modularity 给出两个社群,模块度 Q≈0.43,和我们造图时的两簇吻合。

# %%
comm = net["communities"]
print("社群数:", comm["n_communities"], "· 模块度 Q:", round(comm["modularity"], 3),
      "· 各社群规模:", comm["sizes"])

# %% [markdown]
# 最后把网络画出来,让上面几段结论一眼看见:节点大小按介数中心性(掮客更大)、颜色按 greedy-modularity 社群、边宽按请教频次。坐标由 `networkx` 的 spring 布局给定(固定随机种子以可复现),中心性和社群的数值全部来自上面产出的 `models['network']`,这里只用同一张边表重建图做布局与着色。

# %%
from networkx.algorithms.community import greedy_modularity_communities

G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="weight")
btw = cent["betweenness"]

palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
parts = list(greedy_modularity_communities(G, weight="weight"))
node_color = {n: palette[ci % len(palette)] for ci, part in enumerate(parts) for n in part}

pos = nx.spring_layout(G, seed=1, weight="weight")
sizes = [300 + 4200 * btw.get(n, 0.0) for n in G.nodes()]   # 介数越高节点越大
colors = [node_color.get(n, "#999999") for n in G.nodes()]
widths = [0.6 + 0.9 * G[u][v]["weight"] for u, v in G.edges()]

fig, ax = plt.subplots(figsize=(7.2, 5.6))
nx.draw_networkx_edges(G, pos, width=widths, edge_color="#c9c9c9", ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                       edgecolors="white", linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
q = net["communities"].get("modularity")
ax.set_title(f"求助网络:节点大小=介数中心性,颜色=社群(Q≈{q:.2f})")
ax.axis("off")
fig.tight_layout()
fig.savefig(fig_path("fig_advice_network.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("已保存 fig_advice_network.png")

# %% [markdown]
# ![求助网络图](fig_advice_network.png)
#
# 图里 Dev 和 Cai 明显更大(高介数,坐在两色社群的接缝上,是掮客),两种颜色对应 greedy-modularity 切出的两个社群。眼睛看到的和上表的数字完全一致。

# %% [markdown]
# ## 可复现的证据链
#
# 这本教程一路走下来,和普通分析脚本有一处不易察觉的差别:每一步在成功后都会自动向研究状态里记一笔账,写清它用了哪个函数、消费了什么、产出了什么。福柯和布迪厄共享同一个 `st`(所以它累积了 build_corpus → code_themes → foucault → bourdieu 四步),韦伯和网络各用独立的状态。`st.summary()` 把「填了哪些槽、走了几步」一并数出来——在社会科学里,「结论从哪一步、哪份数据来」往往和结论本身同等重要,这份证据链正是为此而留。

# %%
print("=== 主链 st(福柯 + 布迪厄共享)===")
print(st.summary())

# %% [markdown]
# 逐步展开这份账本,能看到整条链的 requires → produces 依赖是怎么串起来的:build_corpus 从原始语料造出可编码单元,code_themes 消费单元产出主题,福柯透镜消费单元产出证据,布迪厄透镜消费主题与资本维度产出场域。

# %%
for rec in st.provenance:
    pro = "+".join(f"{k}:{','.join(v)}" for k, v in rec["produces"].items()) or "∅"
    print(f"step {rec['step']}: {rec['function'].split('.')[-1]}  →  produces[{pro}]")

# %% [markdown]
# ## 小结
#
# 我们用同一套工作流走了两条社会科学传统:三种理论透镜(福柯的话语追问、布迪厄的场域投影、韦伯的理想类型打分)读文本与案例,再对一张求助网络做中心性与社群分析。网络这一步对标的是 `networkx` / `igraph` / `Gephi`;而三种透镜对标的其实是**没有软件的学术传统**——历来靠读、记、手工编码完成。
#
# 相比纯粹的网络工具,这里多给了一样贯穿全程的东西:把解释也纳入证据链。福柯的每根轴挂着 `support_units`、布迪厄的每个位置溯源到 `capital_indicators`、韦伯的每条偏离带着原始打分,加上那个 coverage=0 的例子提醒你「跑通不等于有意义」——解释因此可被定位、可被反驳,这是无软件传统最容易丢、也最该守住的。下一本教程 [08_governance_gates](08_governance_gates.ipynb) 转向研究治理:伦理、偏见与合规如何被做成会真的拦住你的关卡。
