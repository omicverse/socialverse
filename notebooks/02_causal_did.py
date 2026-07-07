# %% [markdown]
# # 用双重差分评估一项政策的因果效应
#
# 双重差分(Difference-in-Differences, DID)是社会科学里最常用的准实验设计之一。它的想法很朴素:一项政策在某个时点只对一部分单位(处理组)生效,那么把「处理组前后的变化」减去「对照组同期的变化」,就能剥离掉共同的时间趋势,得到政策的**平均处理效应 ATT**。
#
# DID 能不能被解读为因果,取决于一个关键前提——**平行趋势**:如果没有政策,处理组和对照组的结果本会沿着平行的轨迹演化。这个前提无法直接检验,但可以用处理前若干期的「前趋势」来间接考察。本教程就沿着一条完整的识别链走一遍:载入面板 → 声明设计 → 检验平行趋势 → 估计 ATT → 展开动态效应 → 稳健性 → 出图。
#
# 我们用 `socialverse` 完成全流程。它是一套面向社会科学的分析库,把每种方法都登记进一张函数注册表,并在运行时校验「这一步需要的前置是否已经就绪」,同时把每一步记进一份可复现的证据链——你会在最后看到它。方法学背景可参考 Roth et al. (2023) *What's Trending in Difference-in-Differences?* 与 `fixest` / `did` 等 R 包的文档。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ## 载入面板数据
#
# 我们用一个内置的合成面板:40 家企业 × 8 年(2010–2017)。其中一半企业在 2015 年被某政策覆盖,真实的 ATT 约为 −0.8。数据在设计上带有干净的平行前趋势,方便我们把识别链的每一步看清楚。
#
# 面板是长格式(每行一个「企业 × 年份」):`firm_id` 是单位、`year` 是时间、`treat_post` 标记「已受处理」的观测、`first_treated` 是每家企业首次受处理的年份,`y` 是结果变量。

# %%
df = ds.load_did_panel(att=-0.8)
print("面板维度:", df.shape)
df.head()

# %% [markdown]
# ## 声明研究设计
#
# 分析的第一步不是跑回归,而是把「哪一列扮演什么角色」讲清楚。`declare_design` 把面板 id、时间、处理指示、处理起始时点登记进研究状态,后续所有因果函数都从这里读取设计,不必反复传参。

# %%
st = sv.StudyState()
st.write("estimand", "target", "ATT")   # 我们要估的是平均处理效应,而非单纯的相关
st.write("variables", "outcome", "y")   # 结果变量

sv.pp.ingest(st, data=df, name="policy_panel")
sv.pp.declare_design(
    st,
    panel_id="firm_id",
    time="year",
    treatment="treat_post",
    first_treated="first_treated",
)
st.design

# %% [markdown]
# ## 检验平行趋势
#
# 这是整条链的门槛。`parallel_trends` 估一个完整的事件研究(单位固定效应 + 时间固定效应),然后对所有**处理前**的相对期系数做一次联合检验。原假设是「处理前各期系数全为 0」,也就是两组在处理前的趋势平行。
#
# 若 `p > 0.05`,我们不拒绝平行趋势,识别前提站得住;若 `p` 很小,前趋势已经发散,后面即便算得出系数,也不该称之为「因果」。

# %%
sv.tl.parallel_trends(st)

pt = st.diagnostics["pretrend"]
print("平行趋势判定:", st.identification["parallel_trends"])
print(f"联合 F = {pt['joint_F']:.2f}   p = {pt['p_value']:.3f}   (前导期数 = {pt['n_pre']})")

# %% [markdown]
# `p` 值明显大于 0.05——处理前各期系数联合不显著,平行趋势成立,可以进入估计。这一判定被写入了研究状态,成为下一步 `did` 的前置条件。

# %% [markdown]
# ## 估计 ATT
#
# 现在可以估计了。`did` 拟合 `y ~ treat_post + 单位固定效应 + 时间固定效应`,并按 `firm_id` 聚类计算稳健标准误(处理效应的推断通常要在单位层面聚类)。它同时把上一步的平行趋势判定读进结论:通过则标注为「因果 ATT」,未通过则降级为「关联,非因果」。

# %%
sv.tl.did(st)

m = st.models["did"]
print(f"ATT   = {m['att']:.3f}")
print(f"95%CI = [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]")
print(f"SE    = {m['se']:.3f}   (聚类于 {m['n_clusters']} 家企业)")
print(f"p     = {m['p']:.2e}")
print("结论  :", m["note"])

# %% [markdown]
# 估计的 ATT ≈ −0.73,95% 置信区间不含 0,且覆盖了真实值 −0.8。政策使结果变量显著下降。

# %% [markdown]
# ## 展开动态效应:事件研究
#
# ATT 是一个「平均」。事件研究把它按相对处理时点展开:以处理前一期(−1)为基准,给出每个相对期的系数。处理前的系数(< 0)再一次让我们检视前趋势是否贴着零线;处理后的系数(≥ 0)刻画政策生效后效应如何随时间演化。

# %%
import pandas as pd

sv.tl.event_study(st)
es = st.models["event_study"]
pd.DataFrame(
    [{"相对时点": int(k), "系数": round(v[0], 3), "SE": round(v[1], 3)} for k, v in es["coefs"].items()]
)

# %% [markdown]
# ## 稳健性:标准误对设定的敏感度
#
# 点估计确定后,一个常规检查是看标准误在不同方差设定下是否稳定。`did` 产出的稳健性矩阵对比了三种设定——经典(同方差)、异方差稳健(HC1)、按企业聚类——的标准误。聚类 SE 通常最大,也最可信,因为它容许同一企业内部跨年的相关。

# %%
print(st.diagnostics["robustness"])

# %% [markdown]
# ## 出版级图表
#
# 结果就绪后,`socialverse` 的绘图函数直接从研究状态里读数据出图,不必手工整理。这里画两张:ATT 的森林图,以及动态效应的事件研究图。

# %%
sv.pl.forest(st, out="fig_forest.png", title="政策 ATT · 点估计 ± 95% CI")
sv.pl.event_study_plot(st, out="fig_eventstudy.png")
print("图已保存:fig_forest.png, fig_eventstudy.png")

# %% [markdown]
# **森林图**——单个 ATT 系数与 95% 置信区间,虚线为零效应参考:
#
# ![森林图](fig_forest.png)
#
# **事件研究图**——前导期贴着零线(平行趋势),处理时点 0 之后跳到约 −0.8 并维持:
#
# ![事件研究图](fig_eventstudy.png)

# %% [markdown]
# ## 可复现的证据链
#
# 最后看一眼 `socialverse` 与普通估计脚本的关键差别。整条链跑下来,研究状态里自动积累了一份 **provenance 账本**:每一步用了哪个函数、消费了什么、产出了什么。这份账本让分析可追溯、可复现——在社会科学里,「结论从哪一步、哪份数据来」往往和结论本身同等重要。

# %%
print(st.summary())

# %% [markdown]
# ## 小结
#
# 我们走完了一条标准的 DID 识别链:声明设计 → 检验平行趋势 → 估计 ATT → 展开动态效应 → 稳健性 → 出图。它对标 `pyfixest` / R 的 `fixest`(高维固定效应 + 聚类稳健 SE)与 `did`(Callaway–Sant Anna)的工作流。
#
# 与纯估计工具相比,这里多了两样东西:平行趋势是一道**会真的拦住你**的门槛(未通过时结论自动降级,而非默默给你一个系数),以及一份贯穿始终的证据链。下一本教程 [03_complex_survey](03_complex_survey.ipynb) 转向复杂抽样调查的设计加权推断。
