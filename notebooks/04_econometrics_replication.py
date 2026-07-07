# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 把一篇实证论文打包成可复现的复现件
#
# 一篇应用微观经济学论文的核心结论,往往只是一个系数——某项政策让企业结果下降了约 0.7。但期刊(以及任何认真的读者)想看到的不是这一个数字,而是它周围的一整套东西:处理组和对照组在处理前是否可比(平衡表)、基线模型怎么设的(双向固定效应)、这个系数换一套控制变量或换一种标准误还稳不稳(稳健性矩阵),以及最关键的——一份别人拿到数据就能在自己机器上一键重跑出同样结果的脚本。这套「平衡表 → 基线 → 稳健性 → 可运行脚本」的产物,就是所谓的 replication package(复现件),也是本教程要走的这条链。
#
# 这条链的方法学骨架是**双向固定效应(TWFE)**下的双重差分:用 `y ~ 处理 + 单位固定效应 + 时间固定效应` 吸收掉「不随时间变的单位差异」和「所有单位共享的时间冲击」,把政策效应从混淆中剥出来,标准误按单位聚类(容许同一企业跨年的相关)。它成立的前提仍然是平行趋势——处理组若无政策本会和对照组平行演化——所以复现件里第一件正经事是检验前趋势,前趋势不过关,后面的系数就只是关联而非因果。这套工作流对标的是 R 的 `fixest`(`feols` 做高维固定效应 + 聚类 SE + `etable` 出表)和 Stata 的 `reghdfe` + `esttab`,以及 AER 「data & code appendix」那一整套复现规范。
#
# 数据用一个内置的合成面板:40 家企业 × 8 年(2010–2017),其中一半企业在 2015 年被某政策覆盖,数据生成时设定的真实处理效应是 −0.8,并且带有干净的平行前趋势——这样我们既能把复现件的每一步看清楚,又能拿最后估出来的系数和「真值」对一下账。我们用 `socialverse` 走全流程:它是一套面向社会科学的分析库,这里就把它当作顺手的工具用,链路是 载入面板 → 声明设计 → 检验前提 → 基线估计 → 生成复现件 → 出图 → 出稿,最后再看一眼它顺带留下的证据链。

# %%
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

import socialverse as sv
from socialverse import datasets as ds

# 让产物(图/稿件)始终落在本 notebook 同目录,无论从哪个工作目录运行 ——
# 这样 markdown 里的 ![](fig_xxx.png) 引用永远解析得到。
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Jupyter 里没有 __file__,退回当前目录
    _HERE = os.getcwd()

def here(fname):
    return os.path.join(_HERE, fname)

# %% [markdown]
# ## 载入面板数据
#
# 我们用内置的合成面板 `load_did_panel(att=-0.8)`。它是长格式(每行一个「企业 × 年份」):`firm_id` 是单位、`year` 是时间、`treat_post` 标记「已受处理」的观测、`first_treated` 是每家企业首次受处理的年份,`y` 是结果变量,`x1` 是一个协变量。总共 40 家企业 × 8 年 = 320 行。

# %%
df = ds.load_did_panel(att=-0.8)
print("面板维度:", df.shape, "  (units × periods =", df.firm_id.nunique(), "×", df.year.nunique(), ")")
df.head(10)

# %% [markdown]
# ## 声明研究意图与设计
#
# 复现件的第一步不是跑回归,而是把「研究者想估什么、哪一列扮演什么角色」讲清楚。有些东西是数据算不出来、必须由研究问题给定的:目标量是**平均处理效应 ATT**(而非单纯相关)、结果变量是 `y`、识别策略是**双重差分**。把它们写进研究状态,是在声明研究意图。

# %%
st = sv.StudyState()
st.write("estimand", "target", "ATT")          # 目标量:平均处理效应,不是相关
st.write("variables", "outcome", "y")          # 结果变量
st.write("identification", "strategy", "DiD")  # 识别策略:双重差分

# %% [markdown]
# 接着把数据登记进来,并声明设计列——哪一列是单位、哪一列是时间、哪一列是处理指示、哪一列是首次处理时点。`declare_design` 只写列名字符串,并顺手拿数据核对这些列是否存在;后续每个估计函数都从这里读设计,不必反复传参。

# %%
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
# 这是整个复现件的门槛。DiD 的因果解读全押在平行趋势上:如果没有政策,处理组和对照组的结果本会沿平行轨迹演化。这个前提无法直接检验,但可以用处理前若干期的「前趋势」间接考察。`parallel_trends` 估一个完整的事件研究(单位固定效应 + 时间固定效应),再对所有**处理前**的相对期系数做一次联合 Wald 检验,原假设是「处理前各期系数全为 0」。若 `p > 0.05`,不拒绝平行趋势,识别前提站得住;若 `p` 很小,前趋势已发散,后面即便算得出系数也不该称之为因果。

# %%
sv.tl.parallel_trends(st)

pt = st.diagnostics["pretrend"]
print("平行趋势判定:", st.identification["parallel_trends"])
print(f"联合 Wald:  F = {pt['joint_F']:.3f}   p = {pt['p_value']:.3f}")
print("判定说明:", pt["note"])
print("\n各前导期系数 (相对时点 → coef, se):")
for k, (coef, se) in pt["pre_coefs"].items():
    print(f"  t={k:>3}:  {coef:+.4f}  (se {se:.4f})")

# %% [markdown]
# `p = 0.755`,四个前导期系数都不显著异于 0——**未拒绝平行趋势**。识别前提站得住,可以把下一步的 DID 当因果解读。

# %% [markdown]
# ## 估计基线 ATT
#
# 现在跑基线模型。`did` 拟合 `y ~ treat_post + 单位固定效应 + 时间固定效应`,标准误聚类到 `firm_id`(处理效应的推断通常要在单位层面聚类)。它同时把上一步的平行趋势判定读进结论:通过则标注为「因果 ATT」,未通过则降级为「关联,非因果」。

# %%
sv.tl.did(st)

m = st.models["did"]
print(f"ATT   = {m['att']:+.4f}")
print(f"95%CI = [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]")
print(f"SE    = {m['se']:.4f}   (聚类于 {m['n_clusters']} 家企业)")
print(f"p     = {m['p']:.2e}   N = {m['n']}   估计量 = {m['estimator']}")
print("结论  :", m["note"])

# %% [markdown]
# 基线 ATT ≈ **−0.731**,95% 置信区间 [−0.931, −0.531] 完全落在 0 的左侧,而且覆盖了数据生成时设定的真值 −0.8——恢复得很准。政策让结果变量显著下降。

# %% [markdown]
# ## 生成复现件:平衡表、稳健性矩阵、可运行脚本
#
# 基线系数只是复现件的中心,`replicate` 把审稿人想看的其余部分一次生成:处理组 vs 对照组的**平衡表**、点估计在一组规格下是否稳定的**稳健性矩阵**、一张出版级回归表,并**吐出能真正跑的 `main.R`(feols)和 `main.do`(reghdfe)脚本**。它读的是我们前面已经填好的设计与数据,一次调用把这几样产物全挂进研究状态。

# %%
sv.tl.replicate(st)
print("已生成产物:", "平衡表、稳健性矩阵、出版级回归表、main.R、main.do")

# %% [markdown]
# ### 平衡表:处理组和对照组可比吗
#
# 平衡表对比处理组与对照组在各协变量上的均值,并给出 Imbens–Rubin **规范化差**(`norm_diff`),绝对值 > 0.25 会被 `flag` 标红。这里 `treat` 和 `post` 的规范化差很大是**符合预期**的——它们本就是处理的构成成分,处理组当然全为 1;真正要看的协变量是 `x1`,`norm_diff ≈ 0.10` 远低于阈值,两组在它上面高度可比。

# %%
st.diagnostics["balance"]

# %% [markdown]
# ### 稳健性矩阵:换一套设定,系数还稳吗
#
# 稳健性矩阵把处理效应在一组规格下各估一遍:无固定效应、加固定效应但不加控制、半控制、全控制、以及换用异方差稳健 SE。审稿人要看的就是点估计在这张网格里是否稳定。

# %%
st.diagnostics["robustness"][["spec", "coef", "se", "stars", "n", "se_kind", "backend"]]

# %% [markdown]
# 读这张矩阵:规格 (2)–(5) 都带双向固定效应,ATT 稳稳落在 **−0.73 附近**,只有 (1)「无固定效应、无控制」偏到 −0.44——说明固定效应确实在吸收混淆。所有规格都是 `***`(p < 0.01),点估计对「加不加控制、怎么聚类 SE」都不敏感。这正是审稿人要看的稳健性证据。

# %% [markdown]
# ### 出版级回归表
#
# 同一批规格整理成一张出版级表:系数带显著性星,标准误在括号内,并标注每列用的是哪种 SE。这就是可以直接贴进论文的那张表。

# %%
st.artifacts["tables"]["regression"]

# %% [markdown]
# ## 导出可复现脚本
#
# 复现件的灵魂不是一张表格,而是一段能跑的代码。`replicate` 生成的不是占位符,而是按解析出的变量名拼好的、语法正确的 `feols` / `reghdfe` 脚本;把它和 `data.csv` 一起交出去,任何人都能在 R 或 Stata 里复现同一套估计。先看 R 版:

# %%
scripts = st.artifacts["scripts"]
print(scripts["main.R"])

# %% [markdown]
# 有个诚实的细节值得一提:自动模式在没有 codebook 时,按数据类型把 `treat` / `post` / `x1` 都当成了数值控制变量塞进 `feols(y ~ treat_post + treat + post + x1 | firm_id + year, ...)`。这是「无 codebook 时按类型自动挑控制变量」的默认行为——正因为脚本是**可读的、导出来的**,你才能一眼看到并按需删改,这比藏在函数里的黑箱回归更适合复现。下面是 Stata 版:

# %%
print(scripts["main.do"])

# %% [markdown]
# ## 可视化:ATT 森林图
#
# 结果就绪后,`forest` 直接从研究状态里读 `models.did` 的点估计与置信区间出图,存成真实 PNG,路径回写进研究状态。

# %%
sv.pl.forest(st, out=here("fig_forest_att.png"), title="Replication ATT · forest (95% CI)")
fig_info = st.artifacts["figures"]["forest"]
print("森林图已保存:", os.path.basename(fig_info["path"]), f"(dpi={fig_info['dpi']})")

# %% [markdown]
# 一根点落在 −0.73、置信须完全落在 0 线左侧的森林图——处理效应显著为负且稳健:
#
# ![Replication ATT forest plot](fig_forest_att.png)

# %% [markdown]
# ## 排版稿件与结构质检
#
# 复现件通常还配一份稿件。`manuscript_docx` 只做结构化排版、从不改写你的正文,并生成一张**结构覆盖质检清单**(章节/图/表计数、公式安全标记)。装了 `python-docx` 就写真 `.docx`,否则降级为 `.md` 且不丢内容。

# %%
manuscript = (
    "# Replication: the effect of the policy on firm outcomes\n\n"
    "## 方法\n\n"
    "We estimate a two-way fixed-effects difference-in-differences model, "
    "clustering standard errors at the firm level. Parallel pre-trends are not rejected.\n\n"
    "## 结果\n\n"
    "The ATT is negative (about -0.73) and stable across the robustness matrix; "
    "every specification is significant at the 1% level.\n\n"
    "## 讨论\n\n"
    "The estimated effect is robust to the choice of controls and SE clustering."
)

sv.pl.manuscript_docx(st, manuscript=manuscript, out=here("replication_manuscript.docx"))
cov = st.diagnostics["coverage"]
print("稿件已生成:", os.path.basename(st.artifacts["docx"]), " 渲染器:", cov["renderer"])
print("必备章节覆盖:", cov["present_required"], " 缺失:", cov["missing_required"])
print("公式安全:", cov["math_note"], " 结构 OK:", cov["structure_ok"])

# %% [markdown]
# 质检清单诚实地报出:正文有「方法/结果/讨论」三个章节,缺一个「引言」——这类结构提示正是交稿前想要的。

# %% [markdown]
# ## 可复现的证据链
#
# 最后看一眼 `socialverse` 与普通复现脚本的关键差别。整条链跑下来,研究状态里自动积累了一份 **provenance 账本**:每一步用了哪个函数、消费了什么、产出了什么。对复现件而言,「结论从哪一步、哪份数据来」往往和结论本身同等重要——这份账本让复现件自带审计轨迹。

# %%
print(st.summary())

# %% [markdown]
# ## 小结
#
# 我们把一篇实证论文打包成了完整复现件:声明设计 → 检验平行趋势 → 基线 TWFE → 平衡表 + 稳健性矩阵 + 出版级表 → 导出 R/Stata 脚本 → 森林图 → 排版稿件。它对标 R 的 `fixest`(`feols`:高维固定效应 + 聚类 SE + `etable`)与 Stata 的 `reghdfe` + `esttab`,外加一整套 AER 式的论文复现规范。
#
# 与纯估计工具相比,这里多了两样东西:平行趋势是一道**会真的拦住你**的门槛(未通过时基线结论自动降级,而非默默给你一个系数),以及一份贯穿始终、随复现件一起交付的证据链。下一本教程 [03_complex_survey](03_complex_survey.ipynb) 转向复杂抽样调查的设计加权推断。
