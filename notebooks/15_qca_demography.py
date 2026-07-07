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
# # 组态与分解:定性比较分析(QCA)与形式人口学
#
# 两种「配方」思维:哪些条件的**组合**导致了结果?一群人比另一群人多活的岁数,**拆开来**各由什么贡献?这本教程把两种在质性与人口学传统里很成熟、但长期缺一个顺手 Python 工具的分析方法各走一遍。它们看起来不相关,却共享同一种朴素直觉——不去问「某个变量平均而言有多大作用」,而去问「什么样的**组合**导致了结果」以及「两群人的差距**拆开来**各占多少」。前半是定性比较分析(QCA),后半是形式人口学(生命表与分解)。
#
# **定性比较分析(fuzzy-set QCA)** 出自 Charles Ragin,骨子里是集合论而非回归。回归假设各自变量的效应可加、可互换;QCA 则认为社会结果往往是「多重并发因果」——同一个结果可能由几条**不同的条件组合**分别触发,而单个条件既非必要也非充分。它把每个案例对每个条件的隶属看作 `[0,1]` 的模糊集,枚举所有条件组态,逐个算「这条组态在多大程度上足以导致结果」(一致性),再用布尔代数(Quine–McCluskey)把冗余的组态化简成最简的「充分路径之和」。这里的关键前提是:结果得是集合意义上「可被条件组合充分解释」的,阈值(一致性 cut)怎么定会直接影响哪些组态算数——定太高会漏、定太低会让含噪组态混进来。
#
# **形式人口学** 关心的是「活多久」和「为什么两群人的死亡率不一样」。生命表把一组**年龄别死亡率** `mx` 逐列推成 `qx→lx→ndx→nLx→Tx→ex`,最后给出各年龄的预期寿命,其中出生时的 `e0` 就是我们常说的「人均预期寿命」。它是纯确定性的列运算,难点只在婴儿区间与开区间的 `ax` 约定要处理对。Kitagawa 分解则回答对比性的问题:B 群的粗死亡率比 A 群高,这个差**有多少来自各年龄真的死得更凶(率效应)、有多少只是因为 B 群人更老(年龄构成效应)**?两项相加正好等于总差,这是分解的定义性质。
#
# 数据都用合成的、生成机制已知的小样本,这样我们能验证方法本身:QCA 数据里结果真的由 `(A 且 B) 或 C` 生成,看算法能不能把它复原;人口学数据里 B 群被设计成各年龄死亡率更高、人口结构也更老,看分解能不能把两种来源分清。全程用 `socialverse` 这套面向社会科学的分析库来跑,它对标的是 R 里的 `QCA`(Adrian Duşa)与 `demography` / `DemoDecomp`——这两条链在 Python 生态里此前基本是空白。

# %%
# 让本 worktree 的 socialverse 优先于任何已安装版本被导入,并把工作目录切到本 notebook
# 所在目录,使生成的 fig_*.png 与 notebook 同目录(教学产物,可安全删除)。
import os
import sys

_NB_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PKG_ROOT = os.path.dirname(_NB_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.chdir(_NB_DIR)

import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件
import matplotlib.pyplot as plt
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ---
# # 第一部分 · 定性比较分析(fsQCA)
#
# ## 载入数据
#
# `ds.load_qca()` 给了 40 个案例,四列都是 `[0,1]` 之间的**模糊集隶属度**:三个条件 `A / B / C` 与一个结果 `Y`。这里的「隶属度」不是有无(0/1),而是「在多大程度上属于这个集合」——比如 `A=0.64` 表示这个案例较大程度上具备条件 A。数据在生成时,`Y` 取 `max(min(A,B), C)` 再叠一点噪声,也就是背后藏着 `(A 且 B) 或 C` 这条真实的集合关系。QCA 的任务就是**不看这条生成公式**,只从数据把它复原出来。

# %%
qca_df = ds.load_qca()
print("案例数 × 列数:", qca_df.shape)
qca_df.head(6)

# %% [markdown]
# ## 声明结果与数据,然后做最小化
#
# QCA 只需要两样东西:哪一列是结果、数据挂在哪里。我们把结果变量名写进研究状态的 `variables.outcome`,把数据挂进 `sources.datasets`,`qca` 就能读到它需要的一切。`threshold` 是一致性阈值——一条组态要被认定为「充分」,它的一致性得达到这个门槛;这里设成较低的 `0.5`,好让含一点噪声的充分组态也进得来,再看能否复原原关系。
#
# `qca` 内部依次做:把每列校准为 `[0,1]` 模糊集 → 枚举 `2^3=8` 个条件组态角、逐个算一致性(充分性 `Σmin(X,Y)/ΣX`)与案例数 → 一致性达标的角标为「充分路径」→ 用 Quine–McCluskey 把这些角化简成最简的「路径之和」→ 在模糊数据上重算整解的一致性与覆盖率。

# %%
st = sv.StudyState()
st.write("variables", "outcome", "Y")          # 哪一列是结果
st.write("sources", "datasets", ds.load_qca())  # 数据挂进来

sv.tl.qca(st, conditions=["A", "B", "C"], outcome="Y", threshold=0.5)

qca_model = st.models["qca"]
print("解表达式 solution   :", qca_model["solution"])
print("解一致性 consistency:", qca_model["solution_consistency"])
print("解覆盖率 coverage   :", qca_model["solution_coverage"])
print("解类型              :", qca_model["solution_type"])

# %% [markdown]
# 解表达式是 `C + A*B`——正是 `(A 且 B) 或 C` 的等价写法。算法在完全不知道生成公式的前提下,把这条集合关系端到端复原了出来。**解一致性 0.97** 说明这两条路径合起来几乎总是「充分」导致 `Y`;**解覆盖率 0.98** 说明它们几乎覆盖了所有出现 `Y` 的案例。这就是 QCA 想要的结论形态:不是「A 的系数是多少」,而是「满足 `C`,或者同时满足 `A` 和 `B`,就足以带来结果」。

# %% [markdown]
# ## 读每条路径,再看整张真值表
#
# 解由两条路径组成,各自有 raw 一致性(这条路径本身有多充分)与 raw 覆盖率(它单独能解释多少结果案例)。`C` 覆盖面更广、`A*B` 补上 `C` 不成立时的那部分案例。下面的**真值表**把 8 个组态角全列出来,按一致性从高到低排:`consistency` 是充分性,`pri` 是更严格的 PRI 一致性(用来防止一条组态对结果与其反面都「显得充分」),`outcome=1` 的行就是被选进解的充分角。

# %%
print("=== 各充分路径(solution paths) ===")
for p in qca_model["paths"]:
    print(f"  {p['term']:>4} | raw 一致性 = {p['raw_consistency']:.3f}"
          f" | raw 覆盖率 = {p['raw_coverage']:.3f}")

# %%
tt = pd.DataFrame(st.diagnostics["consistency_coverage"]["truth_table"])
tt[["configuration", "n", "consistency", "pri", "outcome"]]

# %% [markdown]
# 前五行(三条含 `C` 的角、加上 `A*B*C` 与 `A*B*~C`)一致性都在 0.97 以上、被编码为充分角;而 `~A*~B*~C`、`A*~B*~C` 这些既没 `C` 也不满足 `A且B` 的角,一致性掉到 0.8 出头、PRI 更低,被排除在解之外。化简后正好归并成 `C + A*B`——这就是布尔最小化在做的事:把一堆具体组态角压成最精炼的充分性陈述。

# %% [markdown]
# ## 可视化:组态一致性 vs 案例数
#
# 每个点是一个组态角:横轴是落入该组态的案例数、纵轴是它的充分性一致性。红点是被编码为充分路径的角,灰点是被排除的。红色虚线是一致性阈值 `0.5`。可以直观看到:入选的充分角都稳稳落在高一致性区,和被排除的角拉开了明显的纵向间隔。

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#d62728" if r == 1 else "#7f7f7f" for r in tt["outcome"]]  # 红=充分角,灰=排除
ax.scatter(tt["n"], tt["consistency"], c=colors, s=90, edgecolor="black",
           linewidth=0.6, zorder=3)
for _, r in tt.iterrows():
    ax.annotate(r["configuration"], (r["n"], r["consistency"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0.5, color="#d62728", ls="--", lw=1, label="consistency threshold = 0.5")
ax.set_xlabel("cases n (membership > 0.5 in this configuration)")
ax.set_ylabel("sufficiency consistency = min-sum(X,Y) / sum(X)")
ax.set_title("fsQCA truth table: consistency vs. case count per configuration")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("fig_qca_truthtable.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved fig_qca_truthtable.png")

# %% [markdown]
# ![fsQCA 真值表](fig_qca_truthtable.png)

# %% [markdown]
# ---
# # 第二部分 · 形式人口学:生命表与分解
#
# ## 载入数据
#
# `ds.load_demography()` 给两个人群 A、B 的**年龄别死亡率** `mx` 与**人口暴露** `pop`,分成 9 个宽窄不等的年龄组(`n_years` 是每组的年数)。这份数据在设计上让 **B 群各年龄死亡率都更高、且人口结构更老**,正好用来分别演示两条链:生命表回答「A 群能活多久」,分解回答「B 与 A 的死亡率差距从何而来」。

# %%
demo_df = ds.load_demography()
demo_df

# %% [markdown]
# ## 构造生命表,读出预期寿命
#
# `life_table` 只需要三列:年龄组、年龄别死亡率、区间宽度。它按 Preston–Heuveline–Guillot 的标准列算法,把死亡率逐列推成 `mx → ax → qx → lx → ndx → nLx → Tx → ex`;婴儿区间用 `a0≈0.1`、开区间用 `a=1/m` 的常规约定。最右一列 `ex` 就是「活到该年龄的人,平均还能再活多少年」,第一行的 `ex` 即出生时预期寿命 `e0`。我们先对**人群 A** 建表。

# %%
st_lt = sv.StudyState()
st_lt.write("sources", "datasets", ds.load_demography())

sv.tl.life_table(st_lt, age="age_group", mx="mx_A", width="n_years")

lt = st_lt.models["life_table"]
print("人群 A 出生时预期寿命 e0 =", round(lt["e0"], 2), "岁")
lt["table"].round(4)

# %% [markdown]
# A 群 `e0 ≈ 75.06` 岁。表里 `lx` 从 10 万起步(life-table radix),随年龄递减记录「还活着的人数」;`ex` 一列则从 75 岁一路降到最后开区间的 5.6 年。这张表是生命表分析的完整产物,后面画的各年龄预期寿命曲线就取自它。

# %% [markdown]
# 接着对**人群 B**(死亡率整体更高)建同样的表,直接对比两群的 `e0`。

# %%
st_lt.write("sources", "datasets", ds.load_demography())
sv.tl.life_table(st_lt, age="age_group", mx="mx_B", width="n_years")

lt_B = st_lt.models["life_table"]
print("人群 B 出生时预期寿命 e0 =", round(lt_B["e0"], 2), "岁")
print("A − B 的 e0 差         =", round(lt["e0"] - lt_B["e0"], 2), "岁(A 更长寿)")

# %% [markdown]
# B 群 `e0 ≈ 72.25` 岁,比 A 群少约 2.8 岁——和「B 各年龄死亡率更高」的设定一致。

# %% [markdown]
# ## 可视化:两群各年龄预期寿命 `ex`
#
# 把两群的 `ex` 随年龄组画出来。两条线都单调下降(年纪越大、剩余寿命越短),而 B 群整条线都压在 A 群之下,差距在中老年段拉得最开。

# %%
tb_A = lt["table"]
ex_B = list(lt_B["ex"].values())

fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = range(len(tb_A))
ax.plot(xpos, tb_A["ex"], "-o", color="#1f77b4", label=f"population A (e0={lt['e0']:.1f})")
ax.plot(xpos, ex_B, "-s", color="#d62728", label=f"population B (e0={lt_B['e0']:.1f})")
ax.set_xticks(list(xpos))
ax.set_xticklabels(tb_A["age"], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("age group")
ax.set_ylabel("life expectancy ex (years)")
ax.set_title("Period life table: remaining life expectancy ex (A vs. B)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_lifetable_ex.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved fig_lifetable_ex.png")

# %% [markdown]
# ![生命表 ex](fig_lifetable_ex.png)

# %% [markdown]
# ## Kitagawa 分解:差距从率还是从年龄结构来?
#
# 两群的**粗死亡率**(全人口平均死亡率)不同,可能有两个原因:各年龄真的死得更凶,或者只是人口更老、老人占比高。Kitagawa 分解把 `crude_B − crude_A` 加法拆成两项(记 `c` 为年龄构成份额 `pop/Σpop`):**率效应** `Σ(mB−mA)·(cA+cB)/2` 归给死亡率本身的差异,**构成效应** `Σ(cB−cA)·(mA+mB)/2` 归给年龄结构的差异。两项精确相加等于总差,残差应当约等于 0——这是分解成立的检验。
#
# `decomposition` 默认就用数据里的 `mx_A/mx_B` 与 `pop_A/pop_B` 两组列,不用额外传参。

# %%
st_dec = sv.StudyState()
st_dec.write("sources", "datasets", ds.load_demography())

sv.tl.decomposition(st_dec)

dec = st_dec.models["decomposition"]
print("方法              :", dec["method"])
print("crude_A (粗死亡率):", round(dec["crude_A"], 5))
print("crude_B (粗死亡率):", round(dec["crude_B"], 5))
print("总差 total_diff   :", round(dec["total_diff"], 5))
print("  率效应   rate_effect        :", round(dec["rate_effect"], 5))
print("  构成效应 composition_effect :", round(dec["composition_effect"], 5))
print("adding-up 残差(应 ≈ 0)      :", round(dec["adding_up_residual"], 12))

# %% [markdown]
# 总差约 `0.0112`,拆开来看:**构成效应 `0.0089` 远大于率效应 `0.0023`**。也就是说,B 群粗死亡率之所以高出这么多,主要不是因为每个年龄段死得更凶,而是因为 B 群人口结构更老、高死亡率的老年组占比更大。残差为 0,两项精确相加回总差。这正是分解方法的价值:一个笼统的「B 死亡率更高」被拆成了可解释、可归因的两块。

# %% [markdown]
# 分解还带一个 **Oaxaca–Blinder** 回归分解伴侣,把差异拆成 endowments(禀赋差)与 coefficients(系数差)两部分,是社会科学(尤其劳动经济学的工资差距研究)里对同一逻辑的另一种常见表述。

# %%
if "oaxaca" in dec:
    ox = dec["oaxaca"]
    print(f"Oaxaca–Blinder: endowments = {ox['endowments']:.5f}, "
          f"coefficients = {ox['coefficients']:.5f}")

# %% [markdown]
# ## 可视化:总差 = 率效应 + 构成效应
#
# 一张条形图把「率效应 + 构成效应 = 总差」的 adding-up 性质摆出来。构成效应那一条明显更高,和上面的数字一致。

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.2))
labels = ["rate\neffect", "composition\neffect", "total\ndiff"]
vals = [dec["rate_effect"], dec["composition_effect"], dec["total_diff"]]
bars = ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black")
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("contribution to crude-rate difference")
ax.set_title("Kitagawa: crude_B - crude_A = rate effect + composition effect")
fig.tight_layout()
fig.savefig("fig_kitagawa.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved fig_kitagawa.png")

# %% [markdown]
# ![Kitagawa 分解](fig_kitagawa.png)

# %% [markdown]
# ## 逐年龄贡献:差异集中在哪一段?
#
# 分解不止给总量,还留下每个年龄组的率贡献与构成贡献,便于定位差距来自哪一段年龄。看 `composition_contribution` 一列:65 岁以上几个老年组的构成贡献特别大(75-84 组高达 `0.0025`),这正解释了为什么整体是「构成效应主导」——B 群多出来的死亡率,大头来自老年组占比更高。

# %%
comp = st_dec.diagnostics["components"]
comp.round(6)

# %% [markdown]
# ---
# ## 可复现的证据链
#
# 上面两部分各用了独立的 `StudyState`。除了跑出结果,`socialverse` 还在每个状态里记了一份 provenance——哪一步用了什么函数、往哪个槽位写了什么产物。对社会科学而言,「这个结论是从哪份数据、经哪几步得来的」常常和结论本身一样重要,这份账本让整条分析链可追溯、可复现。下面各打印一次 `summary()`。

# %%
print("=== QCA 链 ===")
print(st.summary())
print("\n=== 生命表链 ===")
print(st_lt.summary())
print("\n=== 分解链 ===")
print(st_dec.summary())

# %% [markdown]
# ## 小结
#
# 这本教程走了两条 Python 生态里长期缺工具的社会科学方法链。**fsQCA** 对标 R 的 `QCA` / `SetMethods`:它不是回归,而是集合论的组态分析,回答「哪些条件的组合足以导致结果」——我们让它在不看生成公式的前提下,从模糊隶属度里把 `(A 且 B) 或 C` 完整复原了出来。**生命表与 Kitagawa 分解** 对标 R 的 `demography` / `DemoDecomp` 与 `oaxaca`:生命表给出各年龄预期寿命与 `e0`,分解把两群的粗死亡率差精确拆成率效应与构成效应,并定位到具体年龄段。
#
# 与零散的脚本相比,`socialverse` 多给的两样是:把这些原本只在 R 里成熟的方法落到统一的 `StudyState` 上、共用同一套数据与绘图接口;以及一份贯穿全链的证据链,让每一步的输入产出都在案、可复现。下一本教程 [16_networks_stylometry](16_networks_stylometry.ipynb) 转向网络分析与文体计量学。
