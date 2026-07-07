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
# # 从一张社交网络里读出结构,再从文字里认出作者
#
# 这本教程把两类看起来不相干、实则都属于「关系与文本」的社会科学方法放到一起走一遍:**网络推断**(一张社交网络里,连边不是随机长出来的,背后有哪些结构性倾向?)和**文体计量**(只凭用词习惯,能不能把一篇匿名文本归到它真正的作者?)。它们共同的底色是——观测到的关系或文字里,藏着可以被统计模型识别出来的规律。
#
# 网络这一支我们做两件事。第一是 **ERGM(指数随机图模型)**:它回答「这张网里,互惠(你关注我、我就更可能回关你)和传递闭合(朋友的朋友更容易成为朋友)这两种倾向到底有多强」。它对标的是社会网络分析的世界冠军包——R 的 `ergm`/`statnet`。但这里有个绕不开的现实:完整 ERGM 的极大似然要对一个无法解析计算的归一化常数做 MCMC,Python 生态里一直没有一等公民实现。所以我们用 **MPLE(极大伪似然)** 来补这个缺口,并且诚实地把它标注成「对 MCMC-MLE 的近似」——这一点后面会反复强调,因为知道自己在用近似、近似差在哪,本身就是方法素养的一部分。第二件事是 **SAOM(随机行动者导向模型)** 那一类问题:同一批人在两个时点之间,网络怎么重连、行为怎么随之漂移?它对标 R 的 `RSiena`。完整 SAOM 靠基于模拟的矩量法估计,Python 同样空白,所以我们只做**描述性的简化层**——两波之间的 Jaccard 稳定性、连边的生成/消失、行为的交叉滞后代理——而不去冒充一个真正的 SIENA 估计。
#
# 文体计量这一支则可以全程真算,不欠任何近似。经典任务是**作者归属**:抛开主题词,只看 the/and/of/to 这类没有内容的**函数词**的使用频率,同一个作者会留下稳定的「指纹」。我们用 **Burrows's Delta** 走完整条链——挑最高频词(MFW)、按语料做 z-score 标准化、算 Delta 距离(标准化频率上的平均绝对差)、层次聚类、再按最近邻归属。它对标 R 的 `stylo`。
#
# 数据全部是内置的合成语料,好处是**答案已知**:网络的生成过程里我们**故意植入了互惠和传递闭合**,所以正确的模型应该把这两个系数估成正的;文体语料是 3 位作者各 3 篇,每人一套不同的函数词习惯,所以同作者的文档应该聚在一起、归属准确率应该很高。用「已知答案」检验「方法能不能复原它」,是判断一条分析链是否可信的最直接办法。整条链用 `socialverse` 串起来,它是一套面向社会科学的分析库,会把每一步的输入输出登记进一份可复现的证据链——你会在结尾看到它。

# %%
import os
import sys

# 让本 worktree 的 socialverse 优先于任何已安装版本被导入,并把工作目录切到本 notebook 所在处,
# 使 fig_*.png 与 notebook 同目录(教学用,可安全删除)。
_NB_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PKG_ROOT = os.path.dirname(_NB_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.chdir(_NB_DIR)

import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件
import matplotlib.pyplot as plt

# 本 notebook 有几张图是手绘的(socialverse 没有现成的 ERGM/SAOM 图),标题含中文;
# 这里选一个系统里装了的中文字体,避免图上出现「豆腐块」。这只影响绘图,不改任何分析。
for _f in ("Arial Unicode MS", "STHeiti", "Songti SC", "Hiragino Sans GB", "Heiti SC"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ## 载入一张有向社交网络
#
# 我们先造一张 25 个节点的有向网络。生成过程分三层:先按潜在同质性放一批基础边,再注入互惠(有 `i→j` 时更可能出现 `j→i`),最后注入传递闭合(有 `i→j→k` 时更可能补上 `i→k`)。这两种倾向就是 ERGM 待会要去识别的目标——因为我们亲手放进去了,所以**正确的模型应该把它们的系数估成正数**。
#
# 边表是长格式,每行一条有向边:`source` 关注 `target`。这就是后面 ERGM 和 SAOM 的原始输入。

# %%
edges = ds.load_network(n=25, seed=0)   # 返回列:source, target
n_nodes = len(set(edges["source"]) | set(edges["target"]))
print(f"节点数 = {n_nodes} · 有向边数 = {len(edges)} · 列 = {list(edges.columns)}")
edges.head(8)

# %% [markdown]
# 把边表登记进研究状态。`socialverse` 用一个 `StudyState` 对象贯穿全流程,它像一块有固定槽位的白板;这里我们把边表写进 `sources` 槽,后面的网络函数都从这里读数据,不必反复传参。

# %%
st = sv.StudyState()
st.write("sources", "datasets", edges)
print("已填槽:", st.populated())

# %% [markdown]
# ## ERGM:把互惠和传递闭合估出来
#
# ERGM 的核心思路是:一张网的每一条可能的连边,出现或不出现,取决于它会给整张网带来哪些**结构统计量**的变化。完整的极大似然要对所有可能的图求和(那个无法解析的归一化常数),`statnet` 用 MCMC 逼近它。我们改用 **MPLE**:把每一对有序节点 `(i,j)` 当成一次伯努利试验,响应是「这条边在不在」,预测子是「加上这条边会改变多少个结构统计量」——`edges`(密度/截距)、`mutual`(反向边 `j→i` 是否已存在)、`transitive`(能闭合多少条 `i→k→j` 的两跳路径)。这样一来,「连边 ~ change statistics」就是一个普通的 logistic 回归,回归系数就是 ERGM 参数。MPLE 的代价是它对高阶依赖(尤其三角项)会有偏,所以我们把它诚实标注为对 MCMC-MLE 的近似。
#
# 下面这一步拟合三项 `edges + mutual + transitive`。系数是 log-odds 贡献:某一项为正,意味着「多满足这项结构一分,这条边存在的对数几率就上升」。

# %%
st = sv.tl.ergm(st, terms=["edges", "mutual", "transitive"], seed=0)

ergm = st.models["ergm"]
print("方法:", ergm["method"], "| 后端:", ergm["backend"])
print("近似声明:", ergm["approximation"])
print(f"节点 {ergm['n_nodes']} · 边 {ergm['n_edges']} · dyad 观测 {ergm['n_dyads']}")

# %% [markdown]
# 把三个系数摆成一张表来读:每一项的点估计、标准误、z 值。判据很简单——`mutual` 和 `transitive` 应该显著为正(我们植入的倾向被复原),`edges` 应该为负(反映网络稀疏的基线密度)。

# %%
coef, se, z = ergm["coef"], ergm["se"], ergm["z"]
pd.DataFrame(
    [{"term": t, "coef": round(coef[t], 4), "se": round(se[t], 4), "z": round(z[t], 2)}
     for t in ergm["terms"]]
)

# %% [markdown]
# 三个系数都符合预期:`mutual ≈ +2.37`、`transitive ≈ +0.91`,都远离零线且 z 值很大——MPLE 把我们埋进生成过程的互惠与传递闭合都认了出来;`edges ≈ −3.34` 为负,对应稀疏基线。这是本章的第一个「已知参数被复原」。

# %%
assert coef["mutual"] > 0 and coef["transitive"] > 0, "MPLE 应复原正的 mutual/transitive 系数"
print(f"mutual = {coef['mutual']:+.3f}  → 互惠倾向被复原")
print(f"transitive = {coef['transitive']:+.3f}  → 传递闭合倾向被复原")

# %% [markdown]
# ## 检查拟合优度:观测网络 vs 模型期望
#
# 系数为正只说明「方向对了」,还要看这套系数**生成出来的网**长得像不像真实网。`diagnostics['gof']` 用拟合出的 MPLE 系数做序贯条件模拟,反复生成网络,再把观测网的几个全局统计量(边数、互惠对数、传递三元组数、密度)和模型期望比一比。这是一份诚实标注过的**简化版** GoF——不是 `ergm::gof` 那套完整 MCMC 诊断,但足以看清 MPLE 在哪拟合得好、在哪偏得多。

# %%
gof = st.diagnostics["gof"]
obs, exp, sd = gof["observed"], gof["model_expected"], gof["model_sd"]
pd.DataFrame(
    [{"statistic": k, "observed": round(obs[k], 3),
      "model_exp": round(exp[k], 3), "model_sd": round(sd[k], 3)} for k in obs]
)

# %% [markdown]
# 读法:密度这类一阶量,观测和模型期望大致同一量级;而 `transitive_triads` 这类高阶三角项,观测(227)远高于模型期望(约 1)。这个巨大缺口正是 **MPLE 的已知短板**——伪似然把每条边当独立观测,系统性低估了三角闭合这种强依赖结构。把它摊在明面上,而不是假装 GoF 全绿,才是对近似方法负责任的用法。

# %% [markdown]
# ## 画一张 ERGM 结果图
#
# `socialverse` 的绘图模块里没有专门的 ERGM 图,所以这里直接用 matplotlib 画两联:左边是系数森林图(误差棒 = ±1.96·SE,虚线为零效应),右边是 GoF 的观测 vs 模型期望条形对比。

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

terms = ergm["terms"]
vals = [coef[t] for t in terms]
errs = [1.96 * se[t] for t in terms]
ypos = np.arange(len(terms))[::-1]
axL.errorbar(vals, ypos, xerr=errs, fmt="o", color="#2b6cb0", capsize=4)
axL.axvline(0, color="grey", lw=1, ls="--")
axL.set_yticks(ypos); axL.set_yticklabels(terms)
axL.set_xlabel("MPLE 系数 (log-odds, ±1.96·SE)")
axL.set_title("ERGM 系数:互惠 / 传递闭合 > 0")

keys = list(obs.keys())
x = np.arange(len(keys)); w = 0.38
axR.bar(x - w / 2, [obs[k] for k in keys], w, label="观测", color="#2b6cb0")
axR.bar(x + w / 2, [exp[k] for k in keys], w, label="模型期望", color="#ed8936")
axR.set_xticks(x); axR.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
axR.set_title("拟合优度:观测 vs 模型期望"); axR.legend()

fig.tight_layout()
fig.savefig("fig_ergm.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存 fig_ergm.png")

# %% [markdown]
# ![ERGM 系数与拟合优度](fig_ergm.png)

# %% [markdown]
# ## SAOM:两个时点之间,网络与行为怎么共同演化
#
# ERGM 看的是一张静态网。`RSiena` 关心的是**动态**:同一批人从 wave1 到 wave2,连边是怎么生、怎么灭的?而且网络和个体行为往往互相牵引——是「你身边的人影响了你的行为」(influence),还是「你按自己的行为去选择连谁」(selection)?完整 SAOM 用基于模拟的矩量法把 rate/selection/influence 这些参数估出来,Python 里没有这个实现。我们不冒充它,只做一层**描述性的诊断**:两波之间的 Jaccard 稳定性(SIENA 上手第一眼就看的数据质量指标,太低意味着两波离得太远、任何演化模型都不可信)、连边的生成/消失/维持计数与比率、Hamming 距离,以及给了行为向量时的两条交叉滞后代理。
#
# 为了演示,我们把 wave1 就取已登记的边表,换一个 seed 重采样得到 wave2(模拟时间推移后网络重连),再给每个节点编一个和入度相关的行为分,并让它在 wave2 沿入度方向漂移一点——这样就埋进了一条「入度高的人行为涨得多」的 influence 信号,看描述层能不能把它检出来。

# %%
wave1 = edges                              # 第一波 = 已登记的边表
wave2 = ds.load_network(n=25, seed=1)      # 第二波 = 同规模换 seed 重采样

# 造行为向量:入度越高、行为分越高(制造可检出的 influence 代理)
nodes = sorted(set(wave1["source"]) | set(wave1["target"])
               | set(wave2["source"]) | set(wave2["target"]), key=str)
indeg1 = pd.Series(0, index=nodes)
indeg1.update(wave1["target"].value_counts())
rng = np.random.default_rng(7)
behavior1 = np.array([float(indeg1[v]) for v in nodes])
behavior1 = (behavior1 - behavior1.mean()) / (behavior1.std() + 1e-9)
behavior2 = behavior1 + 0.6 * behavior1 + rng.normal(0, 0.3, len(nodes))  # 入度驱动的漂移 + 噪声

st_saom = sv.StudyState()
st_saom.write("sources", "datasets", wave1)
st_saom = sv.tl.saom(st_saom, wave1=wave1, wave2=wave2,
                     behavior1=behavior1, behavior2=behavior2)

saom = st_saom.models["saom"]
coevo = st_saom.diagnostics["coevolution"]
print("方法:", saom["method"], "| 后端:", saom["backend"])
print("近似声明:", coevo["approximation"])

# %% [markdown]
# 先看网络这一支。Jaccard 稳定性是两波连边的重叠比例;我们的两波是独立重采样,所以它偏低(约 0.10)是**符合预期**的——这里的目的是演示指标本身,而不是追求高稳定性。Hamming 距离是两波邻接矩阵的差异总量,生成率/消失率把它拆成方向。

# %%
print(f"节点数 = {saom['n_nodes']}")
print(f"wave1 边 = {coevo['wave1_ties']} · wave2 边 = {coevo['wave2_ties']}")
print(f"Jaccard 稳定性 = {saom['jaccard']:.4f}")
print(f"Hamming 距离 = {saom['hamming_distance']}")
print(f"生成 / 消失 / 维持 = {saom['ties_created']} / {saom['ties_dropped']} / {saom['ties_maintained']}")
print(f"生成率 = {coevo['creation_rate']:.4f} · 消失率 = {coevo['dissipation_rate']:.4f}")

# %% [markdown]
# 再看行为这一支。两条交叉滞后代理粗略地对应 SAOM 想区分的两股力:influence 代理看「wave1 的入度能不能预测行为的变化」,selection 代理看「wave1 的行为能不能预测度的变化」。我们只植入了前者,所以 influence 代理应为正、且明显大于 selection 代理。

# %%
beh = coevo["behavior"]
print(f"influence 代理 (入度 → 行为变化) = {beh['influence_proxy']:.4f}")
print(f"selection 代理 (行为 → 度变化)   = {beh['selection_proxy']:.4f}")
print(f"行为变化均值 = {beh['behavior_change_mean']:.4f}")

# %% [markdown]
# influence 代理约 +0.94,清楚为正——我们埋进去的「入度驱动行为漂移」被检了出来。这是 SAOM influence/selection 分解的**描述性前身**,可以当作建模前的数据探查,但它不是结构估计,别把它当成 `RSiena` 的参数来解读。

# %% [markdown]
# ### 画 SAOM 两波结构变化
#
# 同样手绘两联:左边把连边的命运分解成维持/生成/消失,右边把 wave1 行为对 wave2 行为画散点、按入度着色,让「入度高的点行为涨得多」这条 influence 信号肉眼可见(点落在 `y=x` 上方越多、说明行为整体在涨)。

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

cats = ["维持", "生成", "消失"]
counts = [saom["ties_maintained"], saom["ties_created"], saom["ties_dropped"]]
axL.bar(cats, counts, color=["#38a169", "#2b6cb0", "#e53e3e"])
axL.set_ylabel("有向连边数")
axL.set_title(f"两波连边命运  (Jaccard={saom['jaccard']:.2f})")
for i, c in enumerate(counts):
    axL.text(i, c, str(c), ha="center", va="bottom")

sc = axR.scatter(behavior1, behavior2, c=behavior1, cmap="viridis",
                 s=45, edgecolor="k", linewidth=0.4)
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
# ## 文体计量:只凭用词习惯认出作者
#
# 换一支话题:作者归属。核心直觉是,一个作者写什么主题会变,但 the/and/of/to 这类**函数词**的使用频率相当稳定,几乎不受内容影响——它们构成了一枚可测量的「文体指纹」。Burrows's Delta 就是把这枚指纹量化:先挑出语料里最高频的一批词(MFW),对每篇文档算这些词的相对频率;再按整个语料把每个词标准化成 z-score(消掉「有些词天生就更常用」的影响);两篇文档的 **Delta 距离** 就是它们在这些标准化频率上的**平均绝对差**(等价于归一化的 L1/曼哈顿距离);最后按 Delta 做层次聚类,并把每篇归到「距离它最近的另一篇」的作者。
#
# 数据是 3 位作者(austen / dickens / melville)各 3 篇,每人一套不同的函数词分布。答案已知:同作者文档应彼此最近、聚在一起,留一归属准确率应该接近 100%。这是本章第二个「已知结构被复原」。

# %%
corpus = ds.load_stylometry(seed=0)   # {doc_id: text},9 篇文档
print("语料:", len(corpus), "篇 ·", ", ".join(corpus))
first_id = list(corpus)[0]
print(f"\n示例 [{first_id}] 前 100 字符:\n{corpus[first_id][:100]} ...")

# %% [markdown]
# 把语料登记进 `corpus` 槽,然后跑 Delta。`n_mfw=20` 表示用最高频的 20 个词做特征——函数词几乎必然落在这个高频区,所以哪怕只取 20 个词,归属信号就已经很强。

# %%
st_sty = sv.StudyState()
st_sty.write("corpus", "documents", corpus)
st_sty = sv.tl.stylometry(st_sty, n_mfw=20)

sty = st_sty.models["stylometry"]
print("方法:", sty["method"])
print("MFW 特征数:", sty["n_mfw"], "| 聚类后端:", sty["linkage_backend"])
print(f"留一归属准确率 = {sty['accuracy']:.0%}  ({sty['n_correct']}/{sty['n_documents']} 正确)")

# %% [markdown]
# 逐文档看归属:每篇文档的真实作者、它的最近邻文档、预测作者、以及到最近邻的 Delta 距离。同作者的两篇互为最近邻,就算归对。

# %%
pd.DataFrame(
    [{"doc": did, "true": r["true_author"], "nearest": r["nearest"],
      "pred": r["predicted_author"], "delta": round(r["delta"], 3),
      "ok": "✓" if r["predicted_author"] == r["true_author"] else "✗"}
     for did, r in sty["attribution"].items()]
)

# %% [markdown]
# 9 篇全部归对,准确率 100%:每篇文档的最近邻都落在同一作者名下,而且跨作者的 Delta 距离明显大于同作者内部。函数词指纹把三位作者干净地分了开。

# %%
assert sty["accuracy"] >= 0.6, "同作者应彼此最近,归属准确率应偏高"
print(f"归属准确率 {sty['accuracy']:.0%} ≥ 60%,断言通过。")

# %% [markdown]
# ### 树状图:同作者在低处融合
#
# 归属背后是一棵平均连接聚类树。同作者的文档 Delta 小、在**低处**先融合成簇;不同作者的簇 Delta 大、要到**高处**才合并。`sv.pl.dendrogram` 直接读 `sv.tl.stylometry` 写下的 scipy 格式 linkage 矩阵来画,把图存成 PNG。

# %%
st_sty = sv.pl.dendrogram(st_sty, out="fig_dendro.png")
figrec = st_sty.artifacts["figures"]["dendrogram"]
figpath = figrec["path"] if isinstance(figrec, dict) else figrec
print("树状图已保存:", figpath)

# %% [markdown]
# ![文体计量层次聚类树状图](fig_dendro.png)
#
# 三个作者形成三束低处融合的子树,束与束之间在高处才连到一起——聚类结构和上面的归属表完全一致。

# %% [markdown]
# ## 可复现的证据链
#
# 前面几个分析函数(`ergm` / `saom` / `stylometry` / `dendrogram`)每成功跑一次,`socialverse` 就自动往对应 `StudyState` 的证据链里记一笔:这一步用了哪个函数、读了哪些槽、写了哪些槽。`st.summary()` 把「哪些槽被填了、走了几步」一次性摊开——分析走到哪、结论从哪份数据来,不用回忆,注册表边跑边记着。

# %%
print("=== ERGM 主链 ===")
print(st.summary())
print("\n=== SAOM 共演化 ===")
print(st_saom.summary())
print("\n=== 文体计量 ===")
print(st_sty.summary())

# %% [markdown]
# 以文体计量这条链为例,把逐步账本展开——每一步的 `requires`(读了哪个槽)和 `produces`(写出了哪个槽)一目了然,这就是让分析可追溯、可复现的那份底账。

# %%
for rec in st_sty.provenance:
    print(f"step {rec['step']}: {rec['function']}")
    print(f"    requires: {rec['requires']}")
    print(f"    produces: {rec['produces']}")

# %% [markdown]
# ## 小结
#
# 我们在同一本教程里走了两条链:ERGM + SAOM 把一张社交网络里的**互惠、传递闭合、两波演化**读了出来,对标 R 的 `ergm`/`statnet` 与 `RSiena`;Burrows's Delta 只凭函数词就把 9 篇文档按 3 位作者干净归类,对标 R 的 `stylo`。三种方法在 Python 生态里要么原生空白、要么零散,这里把它们收进同一套工作流,并且都通过了「已知答案能否被复原」这道检验。
#
# 相比零散的脚本,`socialverse` 多给了两样东西:一是对「精确解 vs 近似」的**诚实标注**——ERGM 用 MPLE 近似 MCMC-MLE、SAOM 只做描述层而非 SIENA 估计,近似差在哪(比如 GoF 里三角项的巨大缺口)都摊在明面上;二是一份边跑边记的**证据链**,让每一步的来龙去脉可追溯。下一本教程 [17_text_scaling](17_text_scaling.ipynb) 继续文本这条线,转向从文本中估计潜在的意识形态位置。
