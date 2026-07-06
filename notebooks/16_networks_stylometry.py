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
# # 网络推断 + 网络—行为共演化 + 文体计量
# ## ERGM(指数随机图) · SAOM(随机行动者导向模型) · Burrows's Delta 作者归属
#
# **对标现实工具:**
#
# | 本章方法 | 世界冠军包 | socialverse 的定位 |
# |---|---|---|
# | **ERGM** 指数随机图 | R `ergm` / `statnet`(MCMC-MLE) | Python **原生空白**——`statnet` 是 R 的天下,Python 没有一等公民实现。这里用 **MPLE(极大伪似然)** 补缺,并**诚实标注它是对 MCMC-MLE 的近似**。 |
# | **SAOM** 网络行为共演化 | R `RSiena`(基于模拟的矩量法) | 同样 Python 空白。这里做**描述性/简化层**:两波之间的 Jaccard 稳定性、生成/消失率、Hamming 距离、(可选)行为交叉滞后,**不冒充**完整的 SIENA 估计。 |
# | **文体计量** 作者归属 | R `stylo`(Maciej Eder 等) | Burrows's Delta 全程真算(MFW → z-score → L1 距离 → 层次聚类 → 最近邻归属)。 |
#
# 这条链要讲的**一件事**:三个「Python 里没有好家的方法」被移植到**同一条 `StudyState` /
# `registry` 脊柱**上。它们不是三个孤立脚本,而是**说同一套 12 槽词汇表、按同一份 `@register`
# 契约**登记的方法。每个函数:
#
# 1. 声明 `requires`(要读哪些槽)→ `produces`(会写哪些槽),**未满足就抛 `RegistryError`**
#    (本章会**故意触发一次**给你看);
# 2. 成功后**自动向 `st.provenance` 只读追加一笔**,于是最后 `st.summary()` 就是一条**可追溯的证据链**;
# 3. 对「近似 vs. 冠军包精确解」**在结果里逐条写清楚**(`approximation` 字段),这正是 grounding:
#    **查契约、标近似,而不是猜**。
#
# ## 涉及的 socialverse 函数(全部带注册契约)
#
# | 函数 | 域 | requires → produces | 对标现实工具 |
# |---|---|---|---|
# | `ds.load_network` | data | —(造有互惠+传递闭合的有向边表) | 合成 DGP |
# | `sv.tl.ergm` | net | `sources['datasets']` → `models['ergm']`, `diagnostics['gof']` | R `ergm`/`statnet`(MPLE≈MCMC-MLE) |
# | `sv.tl.saom` | net | `sources['datasets']` → `models['saom']`, `diagnostics['coevolution']` | R `RSiena`(描述性简化层) |
# | `ds.load_stylometry` | data | —(3 作者 × 3 文档,函数词习惯不同) | 合成语料 |
# | `sv.tl.stylometry` | stylometry | `corpus['documents']` → `models['stylometry']`, `artifacts['figures']` | R `stylo`(Burrows's Delta) |
# | `sv.pl.dendrogram` | plot | `models['stylometry']`(linkage) → `artifacts['figures']` | `stylo` 树状图 / `scipy` |
#
# ## StudyState 会被填哪些槽(12 槽词汇表的子集)
#
# - **`sources`** — 登记的原始输入:`datasets`(网络的有向边表)。
# - **`corpus`** — 文本即数据态:`documents`(文体计量的 `{doc_id: text}` 语料)。
# - **`models`** — 拟合结果:`ergm`(边/互惠/传递系数)、`saom`(共演化描述量)、`stylometry`(Delta + 归属)。
# - **`diagnostics`** — 诊断:`gof`(ERGM 观测 vs 模型统计量)、`coevolution`(SAOM 两波结构变化)。
# - **`artifacts`** — 产物:`figures`(文体树状图 PNG 路径)。

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

import matplotlib
matplotlib.use("Agg")  # 无头后端:图直接存盘,不弹窗

import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

print("socialverse from:", os.path.dirname(sv.__file__))
print("socialverse version:", getattr(sv, "__version__", "(dev)"))
print("12 槽词汇表:", ", ".join(sv.SLOTS))

# %% [markdown]
# ---
# ## 第 1 步 · 载入有向社交网络(合成 DGP)
#
# **为什么:** ERGM 要证明的是「它能从一张网里**认出结构性倾向**」——互惠(你关注我 → 我更可能
# 关注你)与传递闭合(我的朋友的朋友 → 更可能成为我的朋友)。所以我们用一个**故意植入这两种倾向**
# 的生成过程造网:`ds.load_network` 先按潜在同质性放基础边,再注入互惠、再注入 `i→j→k ⇒ i→k`
# 的三角闭合。**已知答案 = mutual/transitive 系数应为正**,后面用它检验 MPLE 有没有复原出来。
#
# **契约:** 这一步只是**登记数据**,把边表写进 `sources['datasets']`——正是 `sv.tl.ergm` 的
# `requires`。

# %%
edges = ds.load_network(n=25, seed=0)   # columns: source, target
print("边表形状:", edges.shape, "| 列:", list(edges.columns))
print(edges.head(8).to_string(index=False))

n_nodes = len(set(edges["source"]) | set(edges["target"]))
print(f"\n节点数 = {n_nodes}, 有向边数 = {len(edges)}")

st = sv.StudyState()
st.write("sources", "datasets", edges)   # 满足 ergm/saom 的 requires: sources['datasets']
print("\n登记后已填槽:", st.populated())

# %% [markdown]
# ### 契约演示:未满足 `requires` 会怎样?
#
# 在正式拟合前,先**故意**用一个空的 `StudyState` 调 `sv.tl.ergm`,展示注册表**不是靠猜、而是
# 靠契约**:`ergm` 声明 `requires={'sources': ['datasets']}`,空 state 里没有 `datasets`,于是
# `@register` 包装器在**函数体执行之前**就抛 `RegistryError`。这就是 grounding——**先查契约,查不到就报错**。

# %%
empty = sv.StudyState()
try:
    sv.tl.ergm(empty)
except sv.RegistryError as e:
    print("如预期抛出 RegistryError:\n")
    print(e)

# %% [markdown]
# ---
# ## 第 2 步 · ERGM:MPLE 伪似然拟合边 / 互惠 / 传递闭合
#
# **为什么 + 方法:** 完整 ERGM 的 MLE 需要对**难解的归一化常数**做 MCMC(这正是 `statnet::ergm`
# 干的事)。**MPLE(极大伪似然)** 绕开它:把每个有序 dyad `(i,j)` 当作一次伯努利观测,响应是
# 观测到的连边 `A[i,j]`,预测子是**加上这条边所带来的网络统计量变化(change statistics)**——
# `edges`(密度/截距)、`mutual`(反向边 `A[j,i]` 是否存在)、`transitive`(闭合了多少条 `i→k→j`
# 两路)。于是 `tie ~ change-stats` 的 **logistic 回归就 = 伪似然**,回归系数就是 ERGM 参数。
#
# **契约 requires → produces:** `sources['datasets']` → `models['ergm']` + `diagnostics['gof']`。

# %%
st = sv.tl.ergm(st, terms=["edges", "mutual", "transitive"], seed=0)

ergm = st.models["ergm"]
print("方法:", ergm["method"])
print("后端:", ergm["backend"])
print("近似声明:", ergm["approximation"])
print(f"\n节点 {ergm['n_nodes']} · 边 {ergm['n_edges']} · dyad 观测 {ergm['n_dyads']}")

print("\n--- ERGM 系数 (log-odds 贡献) ---")
coef, se, z = ergm["coef"], ergm["se"], ergm["z"]
print(f"{'term':<12}{'coef':>10}{'se':>10}{'z':>10}")
for t in ergm["terms"]:
    print(f"{t:<12}{coef[t]:>10.4f}{se.get(t, float('nan')):>10.4f}{z.get(t, float('nan')):>10.2f}")

# %% [markdown]
# **读结果:** `mutual` 与 `transitive` 的系数应为**正**——MPLE 复原了我们植入 DGP 的互惠与
# 传递闭合倾向(即「同一个人被你关注 → 反向连边更可能出现」「共同邻居越多 → 越可能连边」)。
# `edges`(截距)为负,反映网络的稀疏基线密度。这就是本章的第一个「**已知参数被复原**」。

# %%
recovered = []
if coef.get("mutual", 0) > 0:
    recovered.append(f"mutual = {coef['mutual']:+.3f} > 0  → 复原了互惠倾向")
if coef.get("transitive", 0) > 0:
    recovered.append(f"transitive = {coef['transitive']:+.3f} > 0  → 复原了传递闭合倾向")
print("参数复原检查:")
for r in recovered:
    print("  ✓", r)
assert coef.get("mutual", 0) > 0, "MPLE 应复原正的 mutual 系数"
assert coef.get("transitive", 0) > 0, "MPLE 应复原正的 transitive 系数"
print("\n断言通过:mutual 与 transitive 系数均为正。")

# %% [markdown]
# ### ERGM 拟合优度(GoF):观测 vs 模型期望
#
# `diagnostics['gof']` 用拟合出的 MPLE 系数做**序贯条件模拟**(每条 dyad 用当前已生成的图算
# mutual/transitive),对比**观测网络**与**模型期望**的全局统计量。这是**诚实标注的简化 GoF**——
# 不是 `ergm::gof` 的完整 MCMC 诊断,但足以看出 MPLE 对 edges/density 拟合得好、对高阶三角项
# (MPLE 已知会低估)有多大偏差。

# %%
gof = st.diagnostics["gof"]
print("GoF 方法:", gof["method"])
obs, exp, sd = gof["observed"], gof["model_expected"], gof["model_sd"]
print(f"\n{'statistic':<20}{'observed':>12}{'model_exp':>12}{'model_sd':>10}")
for k in obs:
    print(f"{k:<20}{obs[k]:>12.3f}{exp.get(k, float('nan')):>12.3f}{sd.get(k, float('nan')):>10.3f}")
print("\n注:", gof["note"])

# %% [markdown]
# ### 手绘一张 ERGM 拟合图:系数 + GoF 观测vs期望
#
# socialverse 的 `pl` 里没有专门的 ERGM 图,所以这里**直接用 matplotlib 画**,把上面的两张表
# 可视化:左边是系数森林图(误差棒 = ±1.96·SE),右边是 GoF 的观测 vs 模型期望条形对比。

# %%
import matplotlib.pyplot as plt

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

# 左:系数森林图
terms = ergm["terms"]
vals = [coef[t] for t in terms]
errs = [1.96 * se.get(t, 0.0) for t in terms]
ypos = np.arange(len(terms))[::-1]
axL.errorbar(vals, ypos, xerr=errs, fmt="o", color="#2b6cb0", capsize=4)
axL.axvline(0, color="grey", lw=1, ls="--")
axL.set_yticks(ypos)
axL.set_yticklabels(terms)
axL.set_xlabel("MPLE 系数 (log-odds, ±1.96·SE)")
axL.set_title("ERGM 系数(MPLE):互惠 / 传递闭合 > 0")

# 右:GoF 观测 vs 期望
keys = list(obs.keys())
x = np.arange(len(keys))
w = 0.38
axR.bar(x - w / 2, [obs[k] for k in keys], w, label="观测", color="#2b6cb0")
axR.bar(x + w / 2, [exp.get(k, 0) for k in keys], w, label="模型期望", color="#ed8936")
axR.set_xticks(x)
axR.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
axR.set_title("ERGM 拟合优度:观测 vs 模型期望")
axR.legend()

fig.tight_layout()
fig.savefig("fig_ergm.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 fig_ergm.png")

# %% [markdown]
# ![ERGM 系数与拟合优度](fig_ergm.png)

# %% [markdown]
# ---
# ## 第 3 步 · SAOM:网络—行为两波共演化(描述性层)
#
# **为什么:** `RSiena` 的核心问题是**动态**:同一批人在两个时点(wave1 → wave2)之间,连边怎么
# 生、怎么灭?行为(比如某种态度打分)和网络位置**谁先谁后**(影响 influence vs 选择 selection)?
# 完整 SAOM 靠**基于模拟的矩量法**估计 rate/selection/influence 参数——Python 里没有这个家。
#
# **socialverse 的诚实做法:** 只做**描述性/简化层**——`sv.tl.saom` 计算两波之间的
# **Jaccard 稳定性**(SIENA 的关键数据质量指标,过低=两波离得太远不宜建模)、
# **生成/消失/维持** 计数与比率、**Hamming 距离**;若给了行为向量,再算两条**交叉滞后代理**
# (影响 ≈ 入度→行为变化,选择 ≈ 行为→度变化)。结果里明确标注**这不是 RSiena 的模拟估计**。
#
# **造第二波:** 对 wave1 换个 seed 重采样得到 wave2(`load_network(seed=1)`),模拟「时间推移后
# 网络重连」;再给每个 actor 编一个和入度相关的行为向量,演示行为共演化那一支。
#
# **契约 requires → produces:** `sources['datasets']`(wave1 回退到此) → `models['saom']` +
# `diagnostics['coevolution']`。

# %%
wave1 = edges                        # 第一波 = 已登记的边表
wave2 = ds.load_network(n=25, seed=1)  # 第二波 = 同规模换 seed 重采样

# 造 actor 行为向量:让入度高的人行为分更高(制造可检出的 influence 代理)
nodes = sorted(set(wave1["source"]) | set(wave1["target"])
               | set(wave2["source"]) | set(wave2["target"]), key=str)
indeg1 = pd.Series(0, index=nodes)
indeg1.update(wave1["target"].value_counts())
rng = np.random.default_rng(7)
behavior1 = np.array([float(indeg1[v]) for v in nodes])
behavior1 = (behavior1 - behavior1.mean()) / (behavior1.std() + 1e-9)
# wave2 行为 = wave1 行为 + 一点由 wave1 入度驱动的漂移(influence 信号) + 噪声
behavior2 = behavior1 + 0.6 * behavior1 + rng.normal(0, 0.3, len(nodes))

st_saom = sv.StudyState()
st_saom.write("sources", "datasets", wave1)
st_saom = sv.tl.saom(
    st_saom,
    wave1=wave1, wave2=wave2,
    behavior1=behavior1, behavior2=behavior2,
)

saom = st_saom.models["saom"]
coevo = st_saom.diagnostics["coevolution"]
print("方法:", saom["method"], "| 后端:", saom["backend"])
print("近似声明:", coevo["approximation"])
print(f"\n节点数 = {saom['n_nodes']}")
print(f"wave1 边 = {coevo['wave1_ties']}, wave2 边 = {coevo['wave2_ties']}")
print(f"Jaccard 稳定性 = {saom['jaccard']:.4f}   ({coevo['note']})")
print(f"Hamming 距离   = {saom['hamming_distance']}")
print(f"生成 / 消失 / 维持 = {saom['ties_created']} / {saom['ties_dropped']} / {saom['ties_maintained']}")
print(f"生成率 = {coevo['creation_rate']:.4f}, 消失率 = {coevo['dissipation_rate']:.4f}")

# %% [markdown]
# **读结果:** Jaccard 是 SIENA 会第一眼看的数——它告诉你两波之间连边的**重叠比例**;太低意味着
# 「网络几乎全换了」,任何演化模型都不可信。我们造的两波是独立重采样,所以 Jaccard 偏低是**符合
# 预期**的(演示指标本身,而非追求高稳定性)。生成/消失率把 Hamming 距离拆成方向。

# %%
beh = coevo.get("behavior")
if beh:
    print("--- 行为共演化交叉滞后代理 ---")
    print(f"influence 代理 (wave1 入度 → 行为变化) = {beh['influence_proxy']}")
    print(f"selection 代理 (wave1 行为 → 度变化)   = {beh['selection_proxy']}")
    print(f"行为变化均值 = {beh['behavior_change_mean']:.4f}")
    print("注:", beh["note"])
    print("\n解读:influence 代理为正 → 我们植入的「入度驱动行为漂移」被检出;"
          "这是 SAOM influence/selection 分解的**描述性前身**,不是结构估计。")

# %% [markdown]
# ### 画 SAOM 两波结构变化
#
# 同样没有现成 `pl` 图,直接用 matplotlib:左边是连边命运的分解(维持/生成/消失),右边把
# wave1 行为对 wave2 行为画散点,叠上入度着色,让「入度高的点行为涨得多」这条 influence 信号肉眼可见。

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

cats = ["维持\nmaintained", "生成\ncreated", "消失\ndropped"]
counts = [saom["ties_maintained"], saom["ties_created"], saom["ties_dropped"]]
axL.bar(cats, counts, color=["#38a169", "#2b6cb0", "#e53e3e"])
axL.set_ylabel("有向连边数")
axL.set_title(f"SAOM 两波连边命运  (Jaccard={saom['jaccard']:.2f})")
for i, c in enumerate(counts):
    axL.text(i, c, str(c), ha="center", va="bottom")

sc = axR.scatter(behavior1, behavior2, c=behavior1, cmap="viridis", s=45,
                 edgecolor="k", linewidth=0.4)
lims = [min(behavior1.min(), behavior2.min()), max(behavior1.max(), behavior2.max())]
axR.plot(lims, lims, "--", color="grey", lw=1, label="y = x (无变化)")
axR.set_xlabel("wave1 行为 (入度标准化)")
axR.set_ylabel("wave2 行为")
axR.set_title("行为共演化:入度高者行为涨得多")
axR.legend()
fig.colorbar(sc, ax=axR, label="wave1 行为")

fig.tight_layout()
fig.savefig("fig_saom.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 fig_saom.png")

# %% [markdown]
# ![SAOM 两波共演化](fig_saom.png)

# %% [markdown]
# ---
# ## 第 4 步 · 文体计量:Burrows's Delta 作者归属
#
# **为什么:** 文体计量的经典任务——**只凭函数词的使用习惯**(the/and/of/to…这些没有主题内容的
# 词)把匿名文本归到作者。R 的 `stylo` 是这个领域的世界冠军;这里全程真算,无任何对 R 的依赖。
#
# **方法(逐步真算,`numpy`/`scipy`):** ①分词(小写词元)→ ②**MFW** 选最高频的一批词做特征、
# 建每文档相对频率矩阵 → ③**z-score** 按语料对每个词标准化(Burrows 缩放)→ ④**Burrows's Delta**
# = 标准化频率上的平均绝对差(= 归一化 L1/Manhattan 距离)→ ⑤**平均连接层次聚类** → ⑥**最近邻
# 归属** + 留一准确率。
#
# **已知答案:** 语料是 3 作者 × 3 文档,每作者一套不同的函数词分布,所以同作者文档应**聚在一起**、
# 归属准确率应**高(接近 100%)**。这是本章第二个「**已知结构被复原**」。
#
# **契约 requires → produces:** `corpus['documents']` → `models['stylometry']` + `artifacts['figures']`。

# %%
corpus = ds.load_stylometry(seed=0)   # {doc_id: text}, 9 篇文档
print("语料:", len(corpus), "篇文档 ·", ", ".join(list(corpus)))
first_id = list(corpus)[0]
print(f"\n示例 [{first_id}] 前 120 字符:\n{corpus[first_id][:120]} ...")

st_sty = sv.StudyState()
st_sty.write("corpus", "documents", corpus)   # 满足 stylometry 的 requires
st_sty = sv.tl.stylometry(st_sty, n_mfw=20)

sty = st_sty.models["stylometry"]
print("\n方法:", sty["method"])
print("MFW 特征数:", sty["n_mfw"], "| 层次聚类后端:", sty["linkage_backend"])
print(f"留一归属准确率 = {sty['accuracy']:.0%}  ({sty['n_correct']}/{sty['n_documents']} 正确)")

# %% [markdown]
# **逐文档归属:** 每篇文档被归到「Delta 距离最近的另一篇文档」的作者。下表把每篇的真实作者、
# 最近邻、预测作者、Delta 距离摆出来——同作者文档互为最近邻即归对。

# %%
print(f"{'doc':<12}{'true':<10}{'nearest':<12}{'pred':<10}{'delta':>8}  ok")
for did, r in sty["attribution"].items():
    ok = "✓" if r["predicted_author"] == r["true_author"] else "✗"
    print(f"{did:<12}{r['true_author']:<10}{r['nearest']:<12}"
          f"{r['predicted_author']:<10}{r['delta']:>8.3f}  {ok}")

assert sty["accuracy"] >= 0.6, "同作者应彼此最近,归属准确率应偏高"
print(f"\n断言通过:归属准确率 {sty['accuracy']:.0%} ≥ 60%。")

# %% [markdown]
# ### 树状图:`sv.pl.dendrogram`(读 `models['stylometry']` 的 linkage)
#
# 归属背后是一棵**平均连接聚类树**:同作者文档在**低处**融合(Delta 小),不同作者在**高处**融合。
# `sv.pl.dendrogram` 读 `sv.tl.stylometry` 写下的 scipy 格式 `linkage` 矩阵来画,并把 PNG 路径
# 登记回 `artifacts['figures']`——**图也走契约、也进 provenance**。

# %%
st_sty = sv.pl.dendrogram(st_sty, out="fig_dendro.png")
figrec = st_sty.artifacts["figures"]["dendrogram"]
# artifacts['figures'][k] 是一条产物记录 {path, dpi, note};取出路径
figpath = figrec["path"] if isinstance(figrec, dict) else figrec
print("树状图已保存:", figpath, "|", figrec.get("note") if isinstance(figrec, dict) else "")

# %% [markdown]
# ![文体计量层次聚类树状图](fig_dendro.png)

# %% [markdown]
# ---
# ## 收尾 · Provenance 证据链
#
# 前面每一个 `@register` 函数(`ergm` / `saom` / `stylometry` / `dendrogram`)成功调用后都
# **自动向 `st.provenance` 追加了一笔**。`st.summary()` 把「哪些槽被填了、走了几步」一次性
# 摊开——这就是 socialverse 相对普通脚本的差异:**证据链不是靠你回忆,而是注册表边跑边记的。**

# %%
print("=== ERGM/SAOM 主链 StudyState ===")
print(st.summary())            # ergm 链(含 gof)
print("\n=== SAOM 共演化 StudyState ===")
print(st_saom.summary())
print("\n=== 文体计量 StudyState ===")
print(st_sty.summary())

# %% [markdown]
# ### provenance 逐步账本(以文体计量链为例)
#
# 每一步都留下 `function / requires / produces`——谁读了哪个槽、写出了哪个槽,一目了然。

# %%
for rec in st_sty.provenance:
    print(f"step {rec['step']}: {rec['function']}")
    print(f"    requires: {rec['requires']}")
    print(f"    produces: {rec['produces']}")

# %% [markdown]
# ---
# ## socialverse 的差异(一句话)
#
# 三个 **Python 原生空白**的方法——ERGM、SAOM、文体计量——被移植到**同一条注册表脊柱**上:
# 它们**查契约而非猜**(未满足 `requires` 立刻 `RegistryError`)、**边跑边记 provenance**
# (`summary()` 就是可追溯证据链)、并且**对每一处「近似 vs 冠军包精确解」逐条写清**
# (`approximation` 字段:MPLE ≈ statnet 的 MCMC-MLE、SAOM 描述层 ≠ RSiena 模拟估计)。
# 与此同时它**真的复原了已知参数**:MPLE 的 mutual/transitive 系数为正、Burrows's Delta 把
# 9 篇文档按 3 位作者聚对——**grounding(查而非猜)+ 已知参数复原**,正是它区别于「又一个统计脚本」的地方。
