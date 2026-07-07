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
# # 快速上手:用一个最小分析认识 StudyState 与函数注册表
#
# 做一次实证研究,真正花时间的往往不是跑模型,而是把一堆散落的东西对齐:数据里哪一列是处理、哪一列是结果,用哪种设计来识别因果,做这种设计前得先检验什么假设,最后估出来的数字又是从哪一步、哪份数据来的。`socialverse` 是一套面向社会科学的分析库,它的做法是把这些「对齐工作」显式化:一份叫 `StudyState` 的研究状态收纳分析对象的方方面面(数据、设计、假设、模型、诊断……),一张函数注册表则记下每种方法「需要什么前置、产出什么」,让你先查清楚再动手,而不是凭记忆猜函数名和调用顺序。
#
# 这本教程不堆术语,而是带你走一遍最小的真实分析——用双重差分(DID)评估一项政策的效应——在这个过程中顺手认识两样东西:`StudyState` 是社科版的统一分析对象(类比生信里的 AnnData,但它统一的是**词汇**而非数据),注册表则像一本会帮你排流程、拦错误的方法手册。走完你会明白:为什么把一次分析组织成「状态 + 注册表」,能让它更难出错、也更容易被别人复现。
#
# 我们用的是一个内置的合成面板数据:40 家企业、8 年,其中一半在中途被某政策覆盖,真实效应设定为 −0.8。它足够小,能把每一步都看清楚。方法学背景可参考双重差分的标准文献(如 `fixest` / `did` 等 R 包);这里的重点是工具的组织方式。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ## 认识注册表:方法都登记在哪
#
# 导入 `socialverse` 时,各分析模块里的函数会自动登记进一个进程级的注册表 `sv.registry`。你可以把它当成这套库的目录:一共有多少方法、分成哪几类。分类跟着社科的真实工作流走——因果推断、复杂抽样、心理测量、文本与质性、网络、人口学、治理合规等等。开始分析前先扫一眼目录,是个好习惯。

# %%
print(len(sv.registry), "个已登记的方法")
print("类别:", sv.registry.categories())

# %% [markdown]
# ## 按名字找方法:`find`
#
# 假设你手上的研究问题是「我想做双重差分」,但记不清函数叫什么。不用猜——`find` 支持中文、英文、缩写甚至后端工具名。更有用的是,它返回的每个结果都带着这个方法的**契约**:`requires`(需要哪些前置)和 `produces`(会产出哪些东西)。看一眼契约,你立刻知道 `did` 不是拿来就能跑的,它需要先声明好设计、先过平行趋势检验。

# %%
for r in sv.registry.find("双重差分"):
    print(r["full_name"].split(".")[-1])
    print("   需要 (requires):", r["requires"])
    print("   产出 (produces):", r["produces"])
    print("   后端:", r["key_tools"])

# %% [markdown]
# ## 把整条流程排出来:`resolve_plan`
#
# `did` 的前置(声明设计、平行趋势检验)本身也各有前置。与其自己在脑子里理这张依赖图,不如让注册表来排:`resolve_plan("did")` 会递归地把「要跑到 `did`」所需的所有步骤,按正确顺序拓扑排序成一条可执行的链。
#
# 它还额外分出两类你需要留意的信息。`needs_input` 是注册表里没有任何函数能自动产出、必须由你(研究者)提供的东西——比如「你到底想估什么量」(`estimand.target`)、「结果变量是哪一列」(`variables.outcome`)。`escalations` 则是那些涉及因果假设、不该被工具默默补上、需要人来确认的步骤。这正是社科分析的分寸:能自动的自动,该拍板的留给人。

# %%
plan = sv.registry.resolve_plan("did")
print("执行顺序:", [p.split(".")[-1] for p in plan["plan"]])
print("需要你提供的输入 (needs_input):")
for ni in plan["needs_input"]:
    print("   -", ni["slot"] + "." + ni["key"])
print(f"需人工确认的步骤 (escalations): {len(plan['escalations'])} 处")

# %% [markdown]
# 把这条链画出来更直观。下面这张图不是装饰:每个方框是 `resolve_plan` 排出的一步,方框下方标着它「补上了状态里的哪一格」,底部标注唯一需要你手动给定的输入。整条顺序完全是注册表从各方法的 `requires ↔ produces` 自动推出来的。

# %%
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
produced = {   # 每一步补上的关键状态格(教学标注)
    "ingest": "载入数据",
    "declare_design": "声明设计\npanel_id/time/treatment",
    "parallel_trends": "识别假设\n平行趋势",
    "did": "估计 ATT\nmodels + diagnostics",
}

fig, ax = plt.subplots(figsize=(11, 3.0))
ax.set_xlim(0, len(steps) * 3.2)
ax.set_ylim(0, 3)
ax.axis("off")

for i, s in enumerate(steps):
    x = i * 3.2 + 0.2
    box = FancyBboxPatch((x, 1.05), 2.4, 0.95, boxstyle="round,pad=0.08",
                         linewidth=1.6, edgecolor="#2b6cb0", facecolor="#ebf4ff")
    ax.add_patch(box)
    ax.text(x + 1.2, 1.72, f"{i+1}. {s}", ha="center", va="center",
            fontsize=11, weight="bold", color="#1a365d")
    ax.text(x + 1.2, 1.28, produced[s], ha="center", va="center",
            fontsize=7.5, color="#2c5282")
    if i < len(steps) - 1:
        arr = FancyArrowPatch((x + 2.4, 1.55), (x + 3.2 + 0.2, 1.55),
                              arrowstyle="-|>", mutation_scale=16, color="#4a5568")
        ax.add_patch(arr)

ax.text(0.2, 2.6, "resolve_plan('did') 自动排出的执行链",
        fontsize=12, weight="bold", color="#1a202c")
ax.text(0.2, 0.5, "唯一需你给定的输入:  estimand['target'] = 'ATT'   ·   variables['outcome'] = 'y'",
        fontsize=9, color="#c05621", style="italic")

plt.tight_layout()
fig.savefig("fig_resolve_plan_did.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved -> fig_resolve_plan_did.png")

# %% [markdown]
# ![DID 执行链](fig_resolve_plan_did.png)

# %% [markdown]
# ## 认识 StudyState:一次分析的所有东西都装这里
#
# 上面所有 `requires` / `produces` 都在用同一套「格子名」说话:`design`、`variables`、`identification`、`models`…… 这些格子就来自 `StudyState`——它是这本库的统一分析对象,一次研究从原始数据到最终交付物,分门别类地都放进它的槽位里。生信里 AnnData 用固定的字段承载一份表达矩阵;这里不一样,社科的数据天生不可通约(一份问卷、一个语料库、一张社会网络无法塞进同一个矩阵),所以 `StudyState` 统一的不是数据本身,而是**词汇**:让「设计」「假设」「模型」这些概念有共同的名字,依赖关系才能被机器检查。
#
# 下面列出这 12 个槽和各自装什么。这本教程的 DID 分析只会用到其中几个(`sources` / `design` / `variables` / `identification` / `models` / `diagnostics`),后续教程会用到质性编码、文献证据、治理合规等其余槽位。

# %%
for name, (meaning, keys) in sv.SLOTS.items():
    print(f"  {name:15s} {meaning}")

# %% [markdown]
# ## 载入数据
#
# 现在开始真正的分析。用内置加载器取那个合成面板,看看它长什么样。数据是长格式(每行是一个「企业 × 年份」):`firm_id` 是单位、`year` 是时间、`treat_post` 标记「该观测是否已受处理」、`first_treated` 是每家企业首次受处理的年份,`y` 是结果变量。

# %%
df = ds.load_did_panel(att=-0.8)   # 真实 ATT 设为 -0.8,便于对照
print("面板维度:", df.shape)
df.head()

# %% [markdown]
# ## 把数据放进状态,并声明设计
#
# 分析的第一步不是跑回归,而是告诉工具「哪一列扮演什么角色」。`ingest` 把数据登记进状态的 `sources` 槽;`declare_design` 把面板 id、时间、处理指示、处理起始时点写进 `design` 槽。声明一次,后续所有因果函数都从这里读取,不必反复传参。别忘了还要写入 `estimand`(我们要估的是平均处理效应 ATT,而非单纯相关)和 `variables.outcome`(结果变量是 `y`)——这两样正是刚才 `resolve_plan` 提示「需要你提供」的输入。

# %%
st = sv.StudyState()
st.write("estimand", "target", "ATT")   # 你想估的量:平均处理效应
st.write("variables", "outcome", "y")   # 结果变量

sv.pp.ingest(st, data=df)
sv.pp.declare_design(
    st,
    panel_id="firm_id",
    time="year",
    treatment="treat_post",
    first_treated="first_treated",
)
st.design

# %% [markdown]
# ## 检验前提:平行趋势
#
# DID 能不能解读为因果,取决于一个关键前提——**平行趋势**:如果没有政策,处理组和对照组本会沿着平行的轨迹演化。这个前提无法直接检验,但可以用处理前若干期的「前趋势」间接考察。`parallel_trends` 估一个事件研究,对所有处理前的相对期系数做联合检验:原假设是「处理前各期系数全为 0」。若 `p > 0.05`,我们不拒绝平行趋势,前提站得住;若 `p` 很小,前趋势已经发散,后面即便算得出系数也不该称之为因果。

# %%
sv.tl.parallel_trends(st)

pt = st.diagnostics["pretrend"]
print("平行趋势判定:", st.identification["parallel_trends"])
print(f"联合 F = {pt['joint_F']:.2f}   p = {pt['p_value']:.3f}   (前导期数 = {pt['n_pre']})")

# %% [markdown]
# `p` 值明显大于 0.05——处理前各期系数联合不显著,平行趋势成立,可以进入估计。这一判定被写进了状态的 `identification` 槽,成为下一步 `did` 的前置条件。

# %% [markdown]
# ## 估计 ATT
#
# 前提通过,现在可以估计了。`did` 拟合 `y ~ treat_post + 单位固定效应 + 时间固定效应`,并按 `firm_id` 聚类计算稳健标准误(处理效应的推断通常要在单位层面聚类)。它还会把上一步的平行趋势判定读进结论:通过则标注为「因果 ATT」,未通过则自动降级为「关联,非因果」——这一步不会替你美化结果。

# %%
sv.tl.did(st)

m = st.models["did"]
print(f"ATT   = {m['att']:.3f}")
print(f"95%CI = [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
print(f"SE    = {m['se']:.3f}   (聚类于 {m['n_clusters']} 家企业)")
print(f"p     = {m['p']:.2e}")
print("平行趋势:", m["parallel_trends"], " · 估计量:", m["estimator"])

# %% [markdown]
# 估计的 ATT ≈ −0.73,95% 置信区间 [−0.93, −0.53] 不含 0,且覆盖了真实值 −0.8。政策使结果变量显著下降。到这里,一条最小但完整的因果分析链就跑完了。

# %% [markdown]
# ## 状态被填成了什么样
#
# 分析跑完,回头看看 `StudyState` 现在装了什么。`populated()` 列出所有被填过的槽和键——从原始数据(`sources`),到设计(`design`)、目标量(`estimand`)、结果变量(`variables`)、识别假设(`identification`),再到模型(`models`)和诊断(`diagnostics`)。这就是「统一分析对象」的价值:一次研究的全貌都在一个地方,谁想接着做、想复查,打开它就一目了然。

# %%
for slot, keys in st.populated().items():
    print(f"  {slot:15s} {keys}")

# %% [markdown]
# ## 两个让分析更稳的机制
#
# 前面自然地用到了注册表,这里再点出它带来的两个实际好处——它们正是把分析组织成「状态 + 注册表」相比一段普通脚本多出来的东西。
#
# 第一,**前置检查会真的拦住你**。如果在一个还没准备好的空状态上直接调 `did`,它不会给你一个看似合理的假结果,而是抛出错误,并明确告诉你缺哪一格、该由哪个函数补上。这就避免了「在错误的状态上算出一个能骗过自己的数字」。

# %%
empty = sv.StudyState()
try:
    sv.tl.did(empty)   # 空状态:必然被拒
except sv.RegistryError as e:
    print(e)

# %% [markdown]
# 第二,**每一步都被记进一份可复现的证据链**。分析跑下来,状态里自动积累了一份 `provenance` 账本:第几步、用了哪个函数、消费了什么、产出了什么。在社会科学里,「结论从哪一步、哪份数据来」往往和结论本身同等重要——这份账本让一次分析自带审计轨迹,别人不必重问你就能复现。`st.summary()` 把状态全貌加上这份账本的步数一眼呈现出来。

# %%
for rec in st.provenance:
    req = ", ".join(f"{s}{ks}" for s, ks in rec["requires"].items()) or "(无)"
    pro = ", ".join(f"{s}{ks}" for s, ks in rec["produces"].items()) or "(无)"
    print(f"  第 {rec['step']} 步: {rec['function'].split('.')[-1]}")
    print(f"          消费: {req}")
    print(f"          产出: {pro}")

print()
print(st.summary())

# %% [markdown]
# ## 小结
#
# 我们用一次最小的 DID 分析,把 `socialverse` 的两根支柱认了个遍:`StudyState` 是社科版的统一分析对象,一次研究的数据、设计、假设、模型、诊断都归拢在它的槽位里;注册表则像一本会帮你排流程(`resolve_plan`)、查方法(`find`)、拦错误(前置检查)的方法手册。整套「注册表 + 查询」的思路对标生信里的 `ov.registry`,而 `StudyState` 相比 AnnData 的不同在于——它统一的是**词汇**而不是数据,因为社科的问卷、语料、网络本就无法塞进同一个矩阵。相比一段普通的估计脚本,这里多出来的是一道会真的拦住你的前置门槛,和一份贯穿始终、可复现的证据链。
#
# 下一本教程 [02_causal_did](02_causal_did.ipynb) 会把这条因果链展开讲透:平行趋势的联合检验、动态效应的事件研究、标准误的稳健性,以及出版级图表。
