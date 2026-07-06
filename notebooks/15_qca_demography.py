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
# # fsQCA 组态分析 + 人口学(生命表 / 分解)
#
# 这本 notebook 走两条**社会科学里 Python 长期空白**的方法链,共用同一根
# `StudyState` 脊柱(state-centric,契约 `requires → produces`):
#
# 1. **模糊集定性比较分析(fsQCA)** — Ragin 的集合论方法:把「条件的**组态**」
#    映射到结果。我们让 `sv.tl.qca` 从模糊隶属度里**复原**一个已知的集合关系
#    `Y ⇐ (A AND B) OR C`,全程走真值表 → 一致性/覆盖率 → Quine–McCluskey
#    布尔最小化,拿回最简充分性解。
# 2. **形式人口学** — `sv.tl.life_table` 把年龄别死亡率 `mx` 逐列构造成周期
#    生命表(`qx→lx→ndx→nLx→Tx→ex`),给出各年龄预期寿命(含出生时 `e0`);
#    `sv.tl.decomposition` 用 **Kitagawa(1955)** 把两人群的**粗死亡率之差**
#    加法拆成「率效应 + 年龄构成效应」,并附一个 Oaxaca–Blinder 回归分解伴侣。
#
# ## 涉及函数
# - `sv.tl.qca(state, conditions=, outcome=, threshold=, ...)`
#   — 真值表 + 一致性/覆盖率 + Quine–McCluskey 最小化(conservative/complex 解)
# - `sv.tl.life_table(state, age=, mx=, width=)` — 周期生命表 → `e0` 与各年龄 `ex`
# - `sv.tl.decomposition(state, ...)` — Kitagawa 粗率差分解(+ Oaxaca–Blinder)
#
# ## 对标现实冠军包
# | 方法 | R / Python 冠军 | Python 现状 |
# |---|---|---|
# | fsQCA | R `QCA`(Adrian Duşa)、`SetMethods` | **空白**(`fuzzy-qca`/`pyqca` 单薄失维) |
# | 生命表 / 分解 | R `demography`、`DemoDecomp`;`oaxaca` | **空白**(散在脚本,无 registry) |
#
# socialverse 的差异不在「又实现了一遍算法」,而在**注册表 grounding**:每个函数
# 声明 `requires`(缺就抛 `RegistryError`,不猜)与 `produces`(写回哪个 slot),
# 于是整条分析链自带可追溯的 provenance。下面每步都先讲**为什么 + 契约**,再跑真实输出。

# %%
# 让本 worktree 的 socialverse 包优先于任何已安装版本被导入,并把工作目录切到
# 本 notebook 所在目录,使 fig_*.png 与 notebook 同目录(教学用,可安全删除)。
import os
import sys

_NB_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PKG_ROOT = os.path.dirname(_NB_DIR)  # gap-methods worktree 根
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.chdir(_NB_DIR)

import socialverse as sv
from socialverse import datasets as ds

print("socialverse from:", os.path.dirname(sv.__file__))
print("socialverse version:", getattr(sv, "__version__", "(dev)"))

# %% [markdown]
# ---
# ## 第一部分 · fsQCA:从模糊隶属度复原 `Y ⇐ (A AND B) OR C`
#
# ### 数据
# `ds.load_qca()` 造了 40 个案例,四列都是 `[0,1]` 的**模糊集隶属度**:
# 条件 `A, B, C` 和结果 `Y`,其中 `Y = max(min(A,B), C)` 再加一点噪声 —
# 也就是**真实的集合关系** `(A 且 B) 或 C`。QCA 的活儿就是**不看生成公式**、
# 只从数据把这个最简充分性解找回来。

# %%
qca_df = ds.load_qca()
print("shape:", qca_df.shape)
print(qca_df.head(6).to_string(index=False))

# %% [markdown]
# ### 契约:先看 `qca` 要什么(requires)
#
# `sv.tl.qca` 注册时声明了 `requires={"sources": ["datasets"], "variables": ["outcome"]}`。
# 这是**注册表 grounding** 的关键:契约不满足就**抛 `RegistryError`,而不是
# 猜一个默认值**。我们故意先在一个**空 state** 上调用,把这条护栏演示出来。

# %%
st_bad = sv.StudyState()  # 什么都没写,requires 两项都缺
try:
    sv.tl.qca(st_bad, conditions=["A", "B", "C"], outcome="Y", threshold=0.5)
except sv.RegistryError as e:
    print("RegistryError(符合预期):")
    print(e)

# %% [markdown]
# ### 满足契约后再跑
#
# 把结果变量名写进 `variables.outcome`,把数据集挂进 `sources.datasets`,
# 契约就满足了。`qca` 会:
#
# 1. 校准 → 每列作为 `[0,1]` 模糊集;
# 2. 枚举 `2^k` 个组态角(corner),每个角算**一致性**(充分性
#    `Σmin(X,Y)/ΣX`)、PRI 一致性、案例数;
# 3. 一致性 ≥ `threshold` 且 PRI 达标的角编码为「充分路径」;
# 4. **Quine–McCluskey** 把这些角约简成素蕴涵,贪心取本质覆盖 → 最简 sum-of-products;
# 5. 在模糊数据上重算**解一致性 / 解覆盖率**。
#
# 这里把 `threshold` 设为 `0.5`(阈值偏低,让含噪的充分角进得来),看能否复原原关系。

# %%
st = sv.StudyState()
st.write("variables", "outcome", "Y")
st.write("sources", "datasets", ds.load_qca())

sv.tl.qca(st, conditions=["A", "B", "C"], outcome="Y", threshold=0.5)

qca_model = st.models["qca"]
print("解表达式 solution :", qca_model["solution"])
print("解一致性 consistency:", qca_model["solution_consistency"])
print("解覆盖率 coverage   :", qca_model["solution_coverage"])
print("解类型            :", qca_model["solution_type"])
print("估计器            :", qca_model["estimator"])

# %% [markdown]
# **读出**:解表达式应当把 `(A 且 B) 或 C` 复原出来(等价写法,如 `C + A*B`)。
# 这就是「复原了已知参数/关系」这条 socialverse 主张的实证 —— 不是套公式,
# 是从模糊数据端到端把生成机制找回来了。
#
# 下面看每条**路径**的 raw 一致性 / raw 覆盖率,以及整张**真值表**(按一致性排序)。

# %%
print("=== 各充分路径(solution paths) ===")
for p in qca_model["paths"]:
    print(f"  {p['term']:>10} | raw consistency={p['raw_consistency']:.3f}"
          f" | raw coverage={p['raw_coverage']:.3f}")

import pandas as pd

tt = pd.DataFrame(st.diagnostics["consistency_coverage"]["truth_table"])
print("\n=== 真值表(前 8 行,按一致性降序;outcome=1 即入选充分角) ===")
print(tt.head(8).to_string(index=False))

# %% [markdown]
# ### 画图:组态一致性 vs 案例数
#
# 每个点是一个组态角:横轴案例数、纵轴充分性一致性,红点是被编码为
# `outcome=1` 的充分路径。红色虚线是 `threshold` 一致性阈值。

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ["#d62728" if r == 1 else "#7f7f7f" for r in tt["outcome"]]
ax.scatter(tt["n"], tt["consistency"], c=colors, s=90, edgecolor="black",
           linewidth=0.6, zorder=3)
for _, r in tt.iterrows():
    ax.annotate(r["configuration"], (r["n"], r["consistency"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(0.5, color="#d62728", ls="--", lw=1, label="一致性阈值 threshold=0.5")
ax.set_xlabel("案例数 n(> 0.5 隶属该组态)")
ax.set_ylabel("充分性一致性 consistency = Σmin(X,Y)/ΣX")
ax.set_title("fsQCA 真值表:各组态角的一致性与案例数")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("fig_qca_truthtable.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved fig_qca_truthtable.png")

# %% [markdown]
# ![fsQCA 真值表](fig_qca_truthtable.png)

# %% [markdown]
# ---
# ## 第二部分 · 形式人口学:生命表 `e0` + Kitagawa 分解
#
# ### 数据
# `ds.load_demography()` 给两个人群 A、B 的**年龄别死亡率** `mx` 和**人口暴露**
# `pop`(9 个年龄组,含不等宽区间 `n_years`)。设计上 **B 各年龄死亡率更高、
# 且人口更老**;两条链要分别回答:
#
# - 生命表:A 群的**出生时预期寿命 `e0`** 是多少?各年龄 `ex` 如何递减?
# - 分解:B 与 A 的**粗死亡率之差**,有多少来自「死得更凶(率效应)」、
#   多少来自「人更老(年龄构成效应)」?

# %%
demo_df = ds.load_demography()
print(demo_df.to_string(index=False))

# %% [markdown]
# ### 生命表:`life_table`
#
# 契约:`requires={"sources": ["datasets"]}` → `produces={"models": ["life_table"]}`。
# 内部按 Preston-Heuveline-Guillot 的标准列算法逐列构造:
# `mx → ax → qx → px → lx → ndx → nLx → Tx → ex`,婴儿区间 `a0≈0.1`,
# 开区间 `a=1/m`。我们先对**人群 A** 建表。

# %%
st_lt = sv.StudyState()
st_lt.write("sources", "datasets", ds.load_demography())

sv.tl.life_table(st_lt, age="age_group", mx="mx_A", width="n_years")

lt = st_lt.models["life_table"]
print("出生时预期寿命 e0 (人群 A):", round(lt["e0"], 2), "岁")
print("\n=== 生命表(人群 A)===")
print(lt["table"].round(4).to_string(index=False))

# %% [markdown]
# 再对**人群 B**(死亡率整体更高)建表,直接对比两群的各年龄 `ex`,
# 预期 B 的 `e0` 更低。

# %%
st_lt.write("sources", "datasets", ds.load_demography())
sv.tl.life_table(st_lt, age="age_group", mx="mx_B", width="n_years")
lt_B = st_lt.models["life_table"]
print("出生时预期寿命 e0 (人群 B):", round(lt_B["e0"], 2), "岁")
print("A - B 的 e0 差 :", round(lt["e0"] - lt_B["e0"], 2), "岁(A 更长寿)")

# %% [markdown]
# ### 画图:两群各年龄预期寿命 `ex`

# %%
tb_A = lt["table"]
ex_B = list(lt_B["ex"].values())

fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = range(len(tb_A))
ax.plot(xpos, tb_A["ex"], "-o", color="#1f77b4", label=f"人群 A (e0={lt['e0']:.1f})")
ax.plot(xpos, ex_B, "-s", color="#d62728", label=f"人群 B (e0={lt_B['e0']:.1f})")
ax.set_xticks(list(xpos))
ax.set_xticklabels(tb_A["age"], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("年龄组 age group")
ax.set_ylabel("预期寿命 ex(年)")
ax.set_title("周期生命表:各年龄预期寿命 ex(A vs B)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_lifetable_ex.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved fig_lifetable_ex.png")

# %% [markdown]
# ![生命表 ex](fig_lifetable_ex.png)

# %% [markdown]
# ### Kitagawa 分解:`decomposition`
#
# 契约:`requires={"sources": ["datasets"]}` →
# `produces={"models": ["decomposition"], "diagnostics": ["components"]}`。
#
# 把 `crude_B − crude_A` 加法拆成两项(记 `c` 为年龄构成份额 `pop/Σpop`):
#
# - **率效应** `Σ (mB − mA)·(cA+cB)/2` —— 各年龄死亡率不同带来的差;
# - **构成效应** `Σ (cB − cA)·(mA+mB)/2` —— 年龄结构不同带来的差。
#
# 两项**精确相加**等于总差(adding-up 残差 ≈ 0),这是 Kitagawa 分解的定义性质。

# %%
st_dec = sv.StudyState()
st_dec.write("sources", "datasets", ds.load_demography())

sv.tl.decomposition(st_dec)

dec = st_dec.models["decomposition"]
print("方法             :", dec["method"])
print("crude_A (粗死亡率):", round(dec["crude_A"], 5))
print("crude_B (粗死亡率):", round(dec["crude_B"], 5))
print("总差 total_diff   :", round(dec["total_diff"], 5))
print("  率效应   rate_effect        :", round(dec["rate_effect"], 5))
print("  构成效应 composition_effect :", round(dec["composition_effect"], 5))
print("adding-up 残差(应≈0)        :", round(dec["adding_up_residual"], 12))
if "oaxaca" in dec:
    ox = dec["oaxaca"]
    print("Oaxaca-Blinder 伴侣: endowments=%.5f, coefficients=%.5f"
          % (ox["endowments"], ox["coefficients"]))

# %% [markdown]
# ### 画图:总差 = 率效应 + 构成效应
#
# 一张瀑布式条形:总差被拆成两块,验证「两效应相加 = 总差」的 adding-up 性质。

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.2))
labels = ["率效应\nrate", "构成效应\ncomposition", "总差\ntotal"]
vals = [dec["rate_effect"], dec["composition_effect"], dec["total_diff"]]
bars = ax.bar(labels, vals,
              color=["#1f77b4", "#ff7f0e", "#2ca02c"], edgecolor="black")
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("对粗死亡率差的贡献")
ax.set_title("Kitagawa 分解:crude_B − crude_A = 率效应 + 构成效应")
fig.tight_layout()
fig.savefig("fig_kitagawa.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved fig_kitagawa.png")

# %% [markdown]
# ![Kitagawa 分解](fig_kitagawa.png)

# %% [markdown]
# ### 逐年龄贡献(diagnostics.components)
#
# 分解不只给总量,还留下**每个年龄组**的率贡献与构成贡献,便于定位差异来自哪段年龄。

# %%
comp = st_dec.diagnostics["components"]
print(comp.round(6).to_string(index=False))

# %% [markdown]
# ---
# ## Provenance:整条链的可追溯记录
#
# 两段分析用了各自独立的 `StudyState`。下面打印 QCA 链与人口学链的 `summary()`,
# 每个 slot 里挂了什么、跑了几步都在案 —— 这就是**注册表 grounding** 的产物:
# 契约驱动的、可复现的分析轨迹。

# %%
print("=== QCA 链 state ===")
print(st.summary())
print("\n=== 生命表链 state ===")
print(st_lt.summary())
print("\n=== 分解链 state ===")
print(st_dec.summary())

# %% [markdown]
# ### socialverse 的差异,一句话
#
# 这两条方法链(fsQCA 组态分析、形式人口学生命表/分解)在 Python 生态里长期
# **只有 R 冠军包**(`QCA`/`SetMethods`、`demography`/`DemoDecomp`)。socialverse
# 把它们落到同一根 `StudyState` 脊柱上,靠**注册表 grounding**(`requires` 缺就
# 抛 `RegistryError`「查而非猜」、`produces` 写回明确 slot)保证每步契约明确、
# 全链可追溯;而且用**已知生成机制的合成数据**验证了方法本身 —— fsQCA 从模糊隶属度
# **复原了 `(A AND B) OR C`**,Kitagawa 分解满足**两效应精确相加**的定义性质。
