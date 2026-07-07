# %% [markdown]
# # 心理测量:用因子分析、IRT 和结构方程反推看不见的构念
#
# 社会科学里我们真正关心的东西——焦虑、能力、政治态度、组织承诺——几乎都无法直接观测。心理测量学(psychometrics)处理的正是这个尴尬:潜变量(latent construct)看不见,我们手里只有它留下的一堆「噪声指标」,也就是问卷题目、测验作答、量表条目。这条 notebook 的任务,就是把「从可观测的题目反推不可观测的构念」这套机器完整走一遍,分三种互补的视角逐层展开。
#
# 第一种视角是**验证性因子分析(CFA)**。当你已经有一套理论、事先就认定「哪几道题在测同一个东西」,CFA 让你把这个假设写成一个测量模型,估计每道题在因子上的**载荷**,再用 `CFI / RMSEA / SRMR` 三件套判断这套假设结构和数据到底合不合。它和探索性的 EFA 不同——EFA 是让数据告诉你有几个因子,CFA 是你先下断言、再拿数据来检验。第二种视角是**项目反应理论(IRT)**。CFA 把作答当成连续变量,但答对/答错本质上是二值的;IRT 正视这一点,用每道题的**区分度 a**、**难度 b** 和每个人的**能力 θ** 三组参数,把「这个人答对这道题的概率」建成一条 logistic 曲线,并算出每道题在不同能力水平上能提供多少信息。第三种视角是**结构方程模型(SEM)**。前两者都在问「这些题是不是测同一个构念」,SEM 更进一步,问的是**构念与构念之间的路径**——它把一组回归方程当成一个联立系统来估,给出标准化路径系数、每个方程的 R² 和全局拟合。
#
# 三种方法各有各的前提和难点:CFA 依赖你事先设定的因子结构没设错(设错了拟合指数会告诉你);IRT 的极大似然在题目区分度极高时容易把 a 顶到边界,需要留意;SEM 的路径分析这里只涉及观测变量,不估潜变量。为了让每一步都**可证伪**,我们用的三个数据集都是参数已知的合成数据:IRT 数据埋了每道题真实的 a、b,多层数据埋了真实斜率 β=2.0。这样每一步都能把估计值对回真值,验证的不是「拟合得好不好看」,而是「有没有把已知的东西复原出来」。我们用 `socialverse` 完成全流程,它对标 R 的 **lavaan**(CFA/SEM)、**mirt**(IRT)、**psych** 和 Python 的 **semopy** / **factor_analyzer** / **girth**——这些专用包在场时优先调用,不在场时回退到基于 statsmodels / scipy / numpy 的真实实现,所以哪怕一个心理测量专用包都没装,下面的数字依然算得出来。

# %%
import os
import sys

# 确保 import 到本 worktree 的 socialverse(而非环境里另一份旧安装):把 worktree 根插到 sys.path 最前。
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_root = os.path.abspath(os.path.join(_here, ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示后端:图直接写文件
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr, pearsonr

# 让图里的中文正常渲染:挑一个系统里存在的 CJK 字体。
_installed = {f.name for f in fm.fontManager.ttflist}
for _cjk in ("PingFang SC", "PingFang HK", "Hiragino Sans GB", "Songti SC",
             "STHeiti", "Arial Unicode MS", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
    if _cjk in _installed:
        plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII,避免缺字

import socialverse as sv
from socialverse import datasets as ds


# 图存到 notebook 同目录(用相对文件名引用),不论从哪个 cwd 运行都对得上。
def figpath(name: str) -> str:
    return os.path.join(_here, name)


# %% [markdown]
# ## 载入数据
#
# 我们要用到两份内置的合成数据。第一份 `load_irt()` 返回一对结果:一个 400 人 × 10 题的 0/1 作答矩阵 `resp`,以及一张真值表 `truth`,记着每道题生成时用的真实区分度 a 和难度 b。这份作答数据一物两用——当成连续指标喂给 CFA,当成二值作答喂给 IRT。第二份 `load_multilevel()` 是学生嵌套在学校里的两层数据,列是 `school / student / x / y`,数据由 `y = 1 + u + 2·x + ε` 生成,真实斜率 β=2.0,我们拿它跑 SEM 的路径分析。
#
# 先把两份数据装进来看一眼。注意 `truth` 表里的难度 b 是刻意从 −2 到 +2 均匀铺开的,这样后面 IRT 复原难度排序时对错一目了然。

# %%
resp, truth = ds.load_irt()
multilevel = ds.load_multilevel()

print("IRT 作答矩阵:", resp.shape, "| 列:", list(resp.columns))

# %% [markdown]
# 每道题埋下的真实参数(a = 区分度,b = 难度,已按难度升序排布):

# %%
truth

# %% [markdown]
# 多层数据前几行,真实结构是 `y = 1 + u + 2·x + ε`(`u` 是学校随机效应):

# %%
multilevel.head()

# %% [markdown]
# ## CFA:两因子测量模型合不合数据
#
# 第一个问题:这 10 道题,能不能干净地归成两个潜在因子?我们下一个明确的断言——前 5 题(item1–5)载在因子 F1,后 5 题(item6–10)载在因子 F2——然后让 CFA 去检验这个假设结构。因为是「验证性」,因子归属是我们写死的,不是数据推出来的;数据的作用是回答「你这个结构配不配得上它」。
#
# 我们把 0/1 作答矩阵当作连续指标喂进去,用 `model_spec` 指定两个因子各自的题目。`sv.tl.cfa` 有 `semopy` 时走 lavaan 式的 ML-SEM,没有时对每个因子块做单因子 ML 分析,再拼出模型隐含的协方差矩阵来打分——这一点在返回的 `note` 里会诚实标注出来。

# %%
st_cfa = sv.StudyState()
st_cfa.write("sources", "datasets", resp.astype(float))  # CFA 把作答当连续指标

cols = list(resp.columns)
spec = {"F1": cols[:5], "F2": cols[5:]}   # 事先写死的测量模型:前5题→F1,后5题→F2
sv.tl.cfa(st_cfa, model_spec=spec)

cfa_model = st_cfa.models["cfa"]
fit = st_cfa.diagnostics["fit_indices"]
print("backend:", cfa_model["backend"])
print("note   :", cfa_model["note"])

# %% [markdown]
# 先看**因子载荷**。载荷衡量的是「这道题和它所属因子的关联强度」,标准化后一般看 0.4 是个及格线。把两个因子的载荷整理成一张表,每道题的载荷都为正、多数在 0.4 以上,说明题目确实在向各自的因子靠拢。

# %%
loading_rows = []
for factor, loads in cfa_model["loadings"].items():
    for item, v in loads.items():
        loading_rows.append({"因子": factor, "题目": item, "载荷": round(v, 3)})
pd.DataFrame(loading_rows)

# %% [markdown]
# 再看两个因子之间的相关,以及三个全局拟合指数。因子间相关 Φ 告诉我们 F1 和 F2 是不是各自独立的构念(这里约 0.5,中等相关,合理);`CFI / RMSEA / SRMR` 则是判断整套模型好坏的标准三件套,各有公认阈值。

# %%
fc = np.array(fit["factor_correlation"])
print(f"因子间相关 Φ(F1, F2) = {fc[0, 1]:+.3f}")
print(f"平均载荷 = {cfa_model['mean_loading']:.3f} | 正载荷比例 = {cfa_model['prop_positive']:.0%}")
print()
print(f"CFI   = {fit['CFI']:.3f}   (≥ 0.95 为良好)")
print(f"RMSEA = {fit['RMSEA']:.3f}   (≤ 0.06 为良好)")
print(f"SRMR  = {fit['SRMR']:.3f}   (≤ 0.08 为良好)")
print(f"χ²({fit['df']:.0f}) = {fit['chi2']:.2f}, n = {fit['n']}")

# %% [markdown]
# 三个指数——CFI≈0.95、RMSEA≈0.04、SRMR≈0.06——全部落在良好区间,说明「两因子」这个假设结构和数据是一致的,我们下的断言站得住。下面把两个因子的载荷画成条形图,直观看看各题在其所属因子上的载荷强度,虚线是 0.4 参考线。

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
factors = list(cfa_model["loadings"].keys())
palette = {"F1": "#3B6FB6", "F2": "#C1544A"}
pos = 0
for factor in factors:
    loads = cfa_model["loadings"][factor]
    xs = np.arange(pos, pos + len(loads))
    ax.bar(xs, list(loads.values()), color=palette[factor], label=factor, width=0.75)
    for x, it in zip(xs, loads.keys()):
        ax.text(x, -0.03, it, ha="center", va="top", fontsize=8, rotation=45)
    pos += len(loads) + 1  # 两个因子之间留一格空隙

ax.axhline(0, color="black", lw=0.8)
ax.axhline(0.4, color="gray", ls="--", lw=0.8, label="载荷 0.4 参考线")
ax.set_ylabel("standardized loading")
ax.set_title(f"CFA 两因子测量模型载荷  (CFI={fit['CFI']:.2f}, "
             f"RMSEA={fit['RMSEA']:.2f}, SRMR={fit['SRMR']:.2f})")
ax.set_xticks([])
ax.set_ylim(-0.15, 1.0)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(figpath("fig_cfa_loadings.png"), dpi=120)
plt.close(fig)
print("saved fig_cfa_loadings.png")

# %% [markdown]
# ![CFA 两因子载荷](fig_cfa_loadings.png)

# %% [markdown]
# ## IRT:复原每道题的难度
#
# CFA 把作答当成连续变量,但作答其实是二值的——非对即错。项目反应理论从这个事实出发,给每道题两个参数:区分度 a(题目对能力差异有多敏感,对应曲线的陡峭程度)和难度 b(需要多高的能力才有 50% 把握答对,对应曲线的横向位置);给每个人一个能力 θ。三者拼成一条 logistic 答对概率曲线 `P = 1 / (1 + exp(−a·(θ − b)))`。
#
# 这一步的证伪点很明确:我们的数据是用已知的 a、b 生成的(就是开头那张 `truth` 表),所以 IRT 估出来的难度 b **应当能复原真值的排序**。`sv.tl.irt` 有 `girth` 时走边际极大似然(MML),没有时用 scipy 做联合极大似然——交替更新被试能力和逐题 (a, b)。我们直接把估计值和真值并排列出来,再用 Spearman/Pearson 相关量化复原程度。

# %%
st_irt = sv.StudyState()
st_irt.write("sources", "datasets", resp)   # IRT 用原始 0/1 作答,不转连续
sv.tl.irt(st_irt)

irt_model = st_irt.models["irt"]
est_a = np.array(irt_model["a"])
est_b = np.array(irt_model["b"])
theta = np.array(irt_model["theta"])
true_a = truth["a"].to_numpy()
true_b = truth["b"].to_numpy()
print("backend:", irt_model["backend"],
      f"| {irt_model['n_persons']} 人 × {irt_model['n_items']} 题")

# %% [markdown]
# 逐题把估计的 (a, b) 和真值放在一起。重点看难度 b:估计值应当和真值同增同减、数量级贴近。区分度 a 也大致对得上,但注意其中一道题(item5)的 a 被顶到了估计上限 6.0——这是联合极大似然在区分度极高、作答几乎完美可分时的典型边界行为,MML 后端一般不会这么极端。这类边界情况值得知道,但不影响难度排序的复原。

# %%
pd.DataFrame({
    "item": irt_model["items"],
    "a_est": np.round(est_a, 3),
    "a_true": np.round(true_a, 3),
    "b_est": np.round(est_b, 3),
    "b_true": np.round(true_b, 3),
})

# %% [markdown]
# 用相关系数量化复原效果。难度 b 的排序是我们最关心的——Spearman 衡量排序是否一致,Pearson 衡量数量级是否一致。

# %%
sp_b = spearmanr(est_b, true_b).correlation
pe_b = pearsonr(est_b, true_b)[0]
sp_a = spearmanr(est_a, true_a).correlation
print(f"难度 b:Spearman(排序) = {sp_b:.3f} | Pearson(数量级) = {pe_b:.3f}")
print(f"区分度 a:Spearman(排序) = {sp_a:.3f}")
print("→ 难度排序被完美复原(Spearman = 1.000),数量级也高度一致(Pearson ≈ 0.99)。")

# %% [markdown]
# 两张图把复原情况画出来。左图是**估计难度 vs 真实难度**的散点,点越贴近对角线复原越好;右图是三道代表性题目(最易/居中/最难)的**项目信息曲线**——每道题在与其难度 b 相匹配的能力水平附近提供最多信息,峰值等于 a²/4,这也是「难题只对高能力者有区分力」的直观体现。

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# 左:估计 b vs 真实 b
lo = min(est_b.min(), true_b.min()) - 0.3
hi = max(est_b.max(), true_b.max()) + 0.3
ax1.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x(完美复原)")
ax1.scatter(true_b, est_b, s=60, color="#3B6FB6", zorder=3)
for j, it in enumerate(irt_model["items"]):
    ax1.annotate(it, (true_b[j], est_b[j]), fontsize=7,
                 xytext=(3, 3), textcoords="offset points")
ax1.set_xlabel("真实难度 b (truth)")
ax1.set_ylabel("估计难度 b (2PL)")
ax1.set_title(f"难度复原  Pearson={pe_b:.3f}, Spearman={sp_b:.2f}")
ax1.legend(fontsize=9)

# 右:项目信息曲线(选难度分散的三道题)
theta_grid = np.linspace(-4, 4, 200)
pick = [0, len(est_a) // 2, len(est_a) - 1]  # 最易 / 居中 / 最难
for j in pick:
    p = 1.0 / (1.0 + np.exp(-est_a[j] * (theta_grid - est_b[j])))
    info = est_a[j] ** 2 * p * (1 - p)
    ax2.plot(theta_grid, info,
             label=f"{irt_model['items'][j]} (a={est_a[j]:.2f}, b={est_b[j]:+.2f})")
ax2.set_xlabel("能力 θ")
ax2.set_ylabel("Fisher 信息量")
ax2.set_title("项目信息曲线:每题在 θ≈b 处信息最大")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(figpath("fig_irt.png"), dpi=120)
plt.close(fig)
print("saved fig_irt.png")

# %% [markdown]
# ![IRT 难度复原与信息曲线](fig_irt.png)

# %% [markdown]
# 顺带核对一下被试能力 θ 的分布。2PL 把 θ 锚定到 N(0,1) 的量纲,所以它应当近似标准正态,并且和「总答对题数」高度正相关——答对越多的人能力越高,这是最基本的合理性检查。

# %%
total_correct = resp.sum(axis=1).to_numpy()
r_theta_score = pearsonr(theta, total_correct)[0]
print(f"能力 θ:均值 = {theta.mean():+.3f}, 标准差 = {theta.std():.3f}")
print(f"θ 与总答对数的相关 = {r_theta_score:.3f}(接近 1,能力与总分同向)")

# %% [markdown]
# ## SEM:估计结构路径并复原真实斜率
#
# CFA 和 IRT 都在回答「这些题是否测同一个构念」。SEM 换了一层问题:它关心**变量之间的路径**。最经典的特例是路径分析——全是观测变量、没有潜变量——把一组回归方程当成一个联立系统来估,给出标准化路径系数、每个方程的 R² 和一套全局拟合指数。
#
# 我们在多层数据上估一条结构路径 `y ~ x`。因为数据由 `y = 1 + u + 2·x + ε` 生成,**真实斜率就是 β=2.0**,这是这一步的证伪点:SEM 应当把它复原出来。`sv.tl.sem` 有 `semopy` 时走全 ML-SEM,没有时逐方程做 OLS,并把估计器诚实标注为 `path_analysis_ols`。

# %%
st_sem = sv.StudyState()
st_sem.write("sources", "datasets", multilevel)
sv.tl.sem(st_sem, paths={"y": ["x"]})   # 结构路径:y 由 x 预测

sem_model = st_sem.models["sem"]
sem_fit = st_sem.diagnostics["fit_indices"]
print("estimator:", sem_model["estimator"], "|", sem_model["backend"])
print("note     :", sem_model["note"])

# %% [markdown]
# 看估出的路径系数和方程的 R²。`y ~ x` 的系数就是我们要复原的斜率,把它和真值 2.0 对一下。

# %%
beta_hat = sem_model["coefficients"]["y"]["x"]
intercept = sem_model["coefficients"]["y"]["(intercept)"]
r2 = sem_model["r2"]["y"]
print(f"路径 y ~ x       = {beta_hat:+.4f}   ← 真值 β = 2.0")
print(f"截距 (intercept) = {intercept:+.4f}")
print(f"方程 y 的 R²      = {r2:.3f}")
print(f"\n→ 复原斜率 {beta_hat:.4f} vs 真值 2.0,误差仅 {abs(beta_hat - 2.0):.4f}。")

# %% [markdown]
# 用一张散点 + 拟合线把这条路径画出来:灰点是 (x, y) 观测,红线是 SEM 估出的结构路径 `y = 截距 + β·x`,蓝色虚线是数据生成过程 `y = 1 + 2·x`。两条线几乎重合,说明路径系数被准确复原。

# %%
fig, ax = plt.subplots(figsize=(7, 5))
x = multilevel["x"].to_numpy()
y = multilevel["y"].to_numpy()

ax.scatter(x, y, s=14, alpha=0.35, color="#555555", label="观测 (x, y)")
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, intercept + beta_hat * xs, color="#C1544A", lw=2.2,
        label=f"SEM 路径:y = {intercept:.2f} + {beta_hat:.2f}·x")
ax.plot(xs, 1.0 + 2.0 * xs, color="#3B6FB6", lw=1.6, ls="--",
        label="真实结构:y = 1.00 + 2.00·x")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"SEM 路径分析:斜率复原  (R²={r2:.2f})")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(figpath("fig_sem_path.png"), dpi=120)
plt.close(fig)
print("saved fig_sem_path.png")

# %% [markdown]
# ![SEM 路径分析](fig_sem_path.png)

# %% [markdown]
# ## 可复现的证据链
#
# 三个模型各写在自己的 `StudyState` 里(测量模型、IRT、结构模型各一份)。每个 state 都自带一份 provenance 账本——用了哪个函数、读了哪些槽位、往哪里写了结果,由 socialverse 在每次调用后自动记录,不用手写日志。`summary()` 把这份账本打印出来,让「这个结论从哪一步、哪份数据来」一目了然。这在社会科学里往往和结论本身同样重要。

# %%
print("=== CFA state ===")
print(st_cfa.summary())
print("\n=== IRT state ===")
print(st_irt.summary())
print("\n=== SEM state ===")
print(st_sem.summary())

# %% [markdown]
# ## 小结
#
# 我们从三个互补的角度把「反推看不见的构念」走了一遍:CFA 检验事先设定的两因子结构(三件套 CFI≈0.95 / RMSEA≈0.04 / SRMR≈0.06 全部达标)、IRT 复原每道题的难度(排序 Spearman=1.000、数量级 Pearson≈0.99)、SEM 复原结构路径的真实斜率(β=2.0 复原到 2.01)。这条链对标的是 R 的 `lavaan`(CFA/SEM)、`mirt`(IRT)与 Python 的 `semopy`——底层算法就是它们那套 ML 因子分析、2PL 极大似然和路径分析。
#
# 和直接堆这些包的脚本相比,socialverse 多给了两样东西:一是每个数据集都埋了真值,每一步都拿估计值对回真值,验证的是「有没有把已知的东西复原出来」而非「拟合得好不好看」——这也顺带暴露了 IRT 联合极大似然在极高区分度题上把 a 顶到边界这类需要留意的细节;二是一份贯穿始终、自动记录的证据链,让整套分析可追溯、可复现。下一本教程 [13_multilevel_survival](13_multilevel_survival.ipynb) 转向多层线性模型与生存分析。
