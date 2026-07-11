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
# # 系统综述闭环:PRISMA、偏倚风险、GRADE 与端到端复现
#
# 前六本 notebook 全在讲**估计**——怎么把一堆研究合成一个可辩护的数字。但一篇系统综述在
# 报「显著」之前,必须先交代**治理(governance)**:证据是**怎么找到、怎么筛、怎么评**的?
# 少了这一层,再漂亮的合并估计也只是一个悬空的数——读者无法判断它是不是被检索漏掉、被主观
# 筛掉、被高偏倚研究拖偏的产物。这本 notebook 把治理补齐,并在最后走一条完整的 **ECR 式患病率
# 系统综述**——从记录计数一路到 GRADE 证据确定性,把全套 96 函数串成一条**可复核的证据链**。
#
# 治理层要回答五个问题,每个都对应一个标准工具:
#
# 1. **流程透明**:检索到多少、去重后多少、筛掉多少、最终纳入多少?——**PRISMA 2020 流程**
#    (含算术自洽校验);
# 2. **报告完整**:该报告的 27 项内容都报了吗?——**PRISMA 27 项清单**;
# 3. **筛选可靠**:两位评审独立筛选,一致性够高吗?——**κ + Gwet AC1**(应对流行率悖论);
# 4. **研究可信**:每篇纳入研究的偏倚风险有多高?——**RoB2 / ROBINS-I / JBI** + 交通灯图;
# 5. **证据强度**:综合起来,这条证据的确定性是高还是低?——**GRADE**。
#
# 关键前提:治理层是**记账 + 算术**,不替评审下判断。RoB 的 domain 评级、GRADE 的最终确定性
# 都由评审录入;工具只做自洽校验、一致性计算,并从证据槽(I²、Egger)**建议**降级旗标供评审
# 确认。这正是可复核的意义——每一步既留痕,又不越权。
#
# > **对标**:PRISMA 2020 / RoB2 / robvis / GRADEpro / Cochrane Handbook。

# %%
import os
import sys

# 确保用的是本 worktree 里的 socialverse(而不是环境里 editable 安装指向的其它 checkout)
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 在 Jupyter cell 里没有 __file__,退回当前工作目录
    _HERE = os.path.abspath(os.getcwd())
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "notebooks" else _HERE
if os.path.isdir(os.path.join(_ROOT, "socialverse")) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
import pandas as pd
from IPython.display import Image

import socialverse as sv
from socialverse import datasets as ds

# 让本 notebook 自绘的图也能显示中文标签
_CJK = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "STHeiti",
        "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei"]
_have = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [c for c in _CJK if c in _have] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("socialverse", sv.__version__)

# %% [markdown]
# ## 1. 载体数据:一份患病率证据库
#
# 全程用 `ds.load_meta_prevalence()` 作端到端载体——它模拟一次抑郁症状患病率的系统综述:每行是
# 某项研究(`study`)在某个测量工具(`instrument`)下报告的**病例数 `cases` / 总样本 `n`**,还带上
# 发表年份 `year`、女性占比 `female_pct` 两个调节变量。注意**同一项研究会贡献多行**(S00 用了
# 三种量表)——这正是后面要用**三层 `rma_mv`** 而不是普通两层随机效应的原因:同一研究内的多个
# 效应量彼此相依,不能当成独立观测。

# %%
prev = ds.load_meta_prevalence()
study = sv.StudyState()
study.write("sources", "datasets", prev)
print(f"{len(prev)} 个效应量,来自 {prev['study'].nunique()} 项研究,{prev['instrument'].nunique()} 种量表")
prev.head(8)

# %% [markdown]
# ## 2. PRISMA 流程:记录计数 + 算术自洽
#
# **解决什么问题**:系统综述的第一张图永远是 PRISMA 流程图——它把「从数据库检索到几千条,最后
# 只纳入几十项」这条漏斗**透明地**摊开,让读者能复核每一步排除了多少、为什么。
#
# **关键前提**:各阶段计数必须**算术自洽**——`筛查数 − 标题摘要排除 = 全文评估数`,
# `全文评估数 − 全文排除 = 纳入数`。`prisma_flow` 不会偷偷改数,它只**校验并标记**不一致(投稿时
# 审稿人第一件事就是核这个)。
#
# **哪几步**:把七个阶段计数传给 `sv.gov.prisma_flow`,它自动推导去重后条数(`after_dedup`)、
# 校验两处等式,把结果连同 `consistent` / `warnings` 存进 `governance['prisma']`。

# %%
sv.gov.prisma_flow(
    study,
    identified=1284, duplicates=311,
    screened=973, excluded_screen=847,
    full_text=126, excluded_fulltext=98,
    included=28,
)
pf = study.governance["prisma"]
print(f"检索识别 {pf['identified']} → 去重后 {pf['after_dedup']} → 筛查 {pf['screened']}"
      f" → 全文 {pf['full_text']} → 纳入 {pf['included']}")
print(f"算术自洽:{pf['consistent']}   警告:{pf['warnings'] or '无'}")

# %% [markdown]
# `consistent=True` 说明两处等式都成立(973−847=126,126−98=28)——这份流程可以放心画图。若某处
# 对不上,`warnings` 会精确指出是哪条等式违反,而不是把数字悄悄改圆。

# %% [markdown]
# ## 3. PRISMA 流程图
#
# 把上一步登记的计数画成标准的四阶段箱-箭头图。`sv.pl.prisma_diagram` 直接读
# `governance['prisma']`,无需再传数字——这就是「登记一次、处处复用」的价值。

# %%
sv.pl.prisma_diagram(study, out="fig28_prisma.png", title="患病率系统综述 · PRISMA 2020 流程")
Image("fig28_prisma.png")

# %% [markdown]
# ## 4. PRISMA 27 项报告清单
#
# **解决什么问题**:流程透明只是其一;PRISMA 2020 还规定了一张 **27 项报告清单**(标题、摘要、
# 检索式、纳入标准、偏倚评估方法……),投稿时须逐项交代「报了没 / 在第几页」。
#
# **哪几步**:把已完成的条目(可传 `True` 或位置字符串如 `'p.3'`)组成字典传给
# `sv.gov.prisma_checklist`,它算出完成度百分比,存进 `governance['prisma_checklist']`,可直接导出成
# 投稿附件。这里示例填了 24/27 项。

# %%
done = {i: True for i in range(1, 25)}      # 1–24 已完成
done[3] = "p.3"; done[7] = "p.5 检索式附录"  # 部分带上位置说明
sv.gov.prisma_checklist(study, items=done)
ck = study.governance["prisma_checklist"]
print(f"报告清单完成度:{ck['n_addressed']}/{ck['n_total']}  ({ck['completeness']}%)")
print(f"第 3 项位置:{ck['items']['3']['location']!r}   第 25 项已报告:{ck['items']['25']['addressed']}")

# %% [markdown]
# ## 5. 双人筛选一致性:κ + Gwet AC1 应对流行率悖论
#
# **解决什么问题**:纳入决定不能靠一个人拍板——两位评审**独立**筛,再核对一致性。但常用的
# **Cohen's κ** 有个陷阱:当「纳入率极低」(绝大多数文献都该排除)时,即便两人几乎全一致,κ 也会
# 被算得很低,这就是**流行率悖论(prevalence paradox)**。
#
# **关键前提**:此时要同时看 **Gwet AC1** 和 **PABAK**——它们对极端基率稳健,更能反映真实一致性。
#
# **哪几步**:先在 `sources` 里放一张**两评审筛选表**(`rater1`/`rater2` 两列,include/exclude)。
# 这里用 `np.random.default_rng(28)` 合成 200 条,**真实一致率设为 96%**,且纳入率故意压到很低
# (~8%)以触发悖论。`sv.gov.screen_agreement` 一次给出 κ、AC1、PABAK 和冲突清单。

# %%
rng = np.random.default_rng(28)
n_rec = 200
# 真实标签:低纳入率 ~8%(触发流行率悖论)
truth = rng.random(n_rec) < 0.08
r1 = np.where(truth, "include", "exclude").astype(object)
r2 = r1.copy()
# 注入 ~4% 的评审分歧(→ 真实一致率 ~96%)
flip = rng.random(n_rec) < 0.04
r2[flip] = np.where(r2[flip] == "include", "exclude", "include")
screen = pd.DataFrame({"record": [f"R{i:03d}" for i in range(n_rec)],
                       "rater1": r1, "rater2": r2})
study.write("sources", "datasets", screen)   # screen_agreement 从 sources 读双评审表

sv.gov.screen_agreement(study, rater1="rater1", rater2="rater2")
sa = study.governance["screen_agreement"]
print(f"n = {sa['n']}   原始一致率 = {sa['percent_agreement']}%   冲突 {sa['n_conflicts']} 条")
print(f"Cohen's κ = {sa['cohen_kappa']:.3f}   ← 被低纳入率压低(流行率悖论)")
print(f"Gwet AC1  = {sa['gwet_ac1']:.3f}   PABAK = {sa['pabak']:.3f}   ← 对极端基率稳健,反映真实高一致")

# %% [markdown]
# 一致率高达 96%,但 κ 却明显偏低——正是流行率悖论。AC1/PABAK 贴近真实一致水平,提醒我们
# **别只看 κ**。分歧的 `conflict_rows` 就是需要第三位评审仲裁的记录。核完治理层,把数据还原成
# 患病率证据库,进入证据评估与合成。

# %%
study.write("sources", "datasets", prev)   # 还原:后续用回患病率数据

# %% [markdown]
# ## 6. 偏倚风险(RoB)评级 + 交通灯图
#
# **解决什么问题**:纳入的每项研究**可信度不同**——随机化不清、失访严重、结局测量有主观性,都会
# 让效应估计带偏。系统综述必须对每项研究**逐 domain 评级**(RCT 用 RoB2、观察性用 ROBINS-I、
# 患病率研究常用 JBI)。
#
# **关键前提**:`risk_of_bias` 只做**架构校验 + 汇总**——它检查你给的 domain 名是否属于该工具、把
# 每项研究的 domain 判断汇成一个「最差即整体」的总评。domain 评级本身仍是**评审录入**的判断。
#
# **哪几步**:选 `tool='ROB2'`,给若干示例研究逐 domain 填 low/some/high,函数写入
# `governance['risk_of_bias']`,再用 `sv.pl.rob_traffic_light` 画成研究×domain 的交通灯矩阵。

# %%
rob_studies = {
    "S00": {"randomization": "low",  "deviations": "low",  "missing_data": "low",
            "measurement": "some", "selection_reporting": "low"},
    "S01": {"randomization": "some", "deviations": "low",  "missing_data": "high",
            "measurement": "low",  "selection_reporting": "low"},
    "S02": {"randomization": "high", "deviations": "some", "missing_data": "low",
            "measurement": "high", "selection_reporting": "some"},
    "S03": {"randomization": "low",  "deviations": "low",  "missing_data": "some",
            "measurement": "low",  "selection_reporting": "low"},
    "S04": {"randomization": "low",  "deviations": "high", "missing_data": "low",
            "measurement": "some", "selection_reporting": "high"},
}
sv.gov.risk_of_bias(study, tool="ROB2", studies=rob_studies)
rob = study.governance["risk_of_bias"]
print(f"工具 = {rob['tool']}   domain = {rob['domains']}")
for s, o in rob["overall"].items():
    print(f"  {s}: 整体 = {o}")

# %%
sv.pl.rob_traffic_light(study, out="fig28_rob.png", title="纳入研究偏倚风险(RoB2)")
Image("fig28_rob.png")

# %% [markdown]
# 「最差即整体」的逻辑保守而透明:任一 domain 判为 high,整体就是 high——只要有一处高偏倚,整项
# 研究就不能算低偏倚。交通灯图让审稿人一眼看清偏倚集中在哪个 domain(这里 `measurement` 和
# `selection_reporting` 是薄弱环节)。

# %% [markdown]
# ## 7. 端到端合成(一):效应量 = logit 患病率
#
# 治理层核完,进入合成。**解决什么问题**:患病率是 0–1 的比例,直接合并会撞上边界、方差不稳。
# 标准做法是先做**方差稳定变换**——默认用 **logit(PLO)**(比 Freeman-Tukey 双反正弦更稳,
# Schwarzer 2019)。
#
# **哪几步**:`sv.pp.es_proportion` 传 `cases=` / `n=`,把每行比例转成 `yi=logit(p)` + 抽样方差 `vi`,
# 并把 `study`(作簇 id)和调节变量 `year`/`female_pct` 带进效应量表 `models['meta_effects']`。

# %%
sv.pp.es_proportion(
    study, measure="PLO",
    cases="cases", n="n",
    study="study", cluster="study", slab="instrument",
    moderators=["year", "female_pct"],
)
eff = study.models["meta_effects"]
print(f"{len(eff)} 个效应量(logit 尺度),measure = {eff['measure'].iloc[0]}")
eff[["study", "slab", "yi", "vi", "sei"]].head(6).round(4)

# %% [markdown]
# ## 8. 端到端合成(二):抽样协方差 V + 三层 rma_mv
#
# **解决什么问题**:同一研究内的多个效应量(S00 的三种量表)**共享受试者、彼此相依**。若当成独立
# 观测直接两层合并,会**低估不确定性**。正确做法是三层:研究间(σ²₃)+ 研究内/效应间(σ²₂)。
#
# **关键前提**:三层模型需要一个**已知的抽样协方差矩阵 V**——同簇效应量的抽样误差按相关 ρ 相关。
# `sv.tl.vcalc` 按 `cluster='study'`、假定 `rho=0.6` 构造块对角 V(存进 `models['meta_V']`),这是
# `rma_mv` 的必需前置。
#
# **哪几步**:先 `vcalc` 造 V,再 `sv.tl.rma_mv`(读 V、按 `study` 分组)拟合三层 REML,得到合并
# logit 患病率 + 两个方差分量。

# %%
sv.tl.vcalc(study, cluster="study", rho=0.6)
V = study.models["meta_V"]
print(f"抽样协方差 V:{V.shape},块对角(同研究效应量间有 ρ=0.6 相关)")

sv.tl.rma_mv(study, study="study", method="REML")
mv = study.models["meta"]
print(f"三层合并 logit = {mv['estimate']:.4f}  95% CI [{mv['ci_lb']:.3f}, {mv['ci_ub']:.3f}]")
print(f"σ²₂(研究内/效应间) = {mv['sigma2_2']:.4f}   σ²₃(研究间) = {mv['sigma2_3']:.4f}")
print(f"纳入 {mv['n_studies']} 项研究、{mv['k']} 个效应量;收敛 = {mv['converged']}")

# %% [markdown]
# ## 9. 端到端合成(三):异质性 + 分层 I²
#
# **解决什么问题**:合并点估计**从不单独汇报**——必须同时量化研究之间差多少。三层模型还要进一步
# 拆解:异质到底来自**研究之间**(level-3)还是**同研究不同量表之间**(level-2)?
#
# **哪几步**:`sv.tl.meta_heterogeneity` 给总体 Q / I² / τ²;`sv.tl.ma_i2_multilevel`(Cheung 2014)
# 把 I² 拆成 level-2 / level-3 / 抽样误差三份占比——它需要**先有三层 `rma_mv` 拟合**(读 σ²₂/σ²₃)。

# %%
sv.tl.meta_heterogeneity(study)
het = study.diagnostics["heterogeneity"]
print(f"Q = {het['Q']:.1f}  (df = {het['df']}, p = {het['Q_pval']:.2e})")
print(f"总体 I² = {het['I2']:.1f}%   τ² = {het['tau2']:.4f}")

sv.tl.ma_i2_multilevel(study)
i2m = study.diagnostics["i2_multilevel"]
print()
print(f"分层 I²(Cheung 2014):")
print(f"  抽样误差占比           = {i2m['sampling_share']:.1f}%")
print(f"  level-2 研究内/量表间   = {i2m['I2_level2_within_study']:.1f}%")
print(f"  level-3 研究之间        = {i2m['I2_level3_between_study']:.1f}%")

# %% [markdown]
# 分层 I² 回答了「异质从哪来」——这直接影响后续策略:若主要在 level-3(研究间),该找研究层面的
# 调节变量;若在 level-2(量表间),说明不同工具测出的患病率本就不同。下面用元回归验证。

# %% [markdown]
# ## 10. 端到端合成(四):元回归 + FDR
#
# **解决什么问题**:异质性不是终点,而是线索——**能不能用调节变量解释它**?年份越近患病率越高吗?
# 女性占比越高患病率越高吗?这就是**混合效应元回归**。
#
# **关键前提**:一次考察多个调节变量,就有**多重比较**问题;`metareg_fdr` 在综合检验 QM 之外,对各
# 调节变量的 p 值做 **Benjamini-Hochberg FDR** 校正,避免「测得多、假阳性多」。
#
# **哪几步**:`sv.tl.metareg` 用 `year`、`female_pct` 拟合,给系数 + 伪 R²(解释掉多少 τ²);再
# `sv.tl.metareg_fdr` 读 `metareg` 结果,输出 QM 综合检验 + 逐调节变量 FDR。

# %%
sv.tl.metareg(study, moderators=["year", "female_pct"])
mr = study.models["metareg"]
print(f"元回归(混合效应),伪 R² = {mr['R2']:.1f}%(调节变量解释掉的 τ² 比例)")
for term, c in mr["coefs"].items():
    print(f"  {term:14s}  β = {c['estimate']:+.4f}  SE = {c['se']:.4f}  p = {c['pval']:.3f}")

sv.tl.metareg_fdr(study, alpha=0.05)
fdr = study.diagnostics["metareg_fdr"]
print()
print(f"综合检验 QM = {fdr['QM']:.2f}  (df = {fdr['QM_df']}, p = {fdr['QM_pval']:.3f})")
for term, r in fdr["per_moderator"].items():
    print(f"  {term:14s}  p = {r['pval']:.3f}  →  FDR = {r['pval_fdr']:.3f}"
          f"  {'✓显著' if r['significant_fdr'] else '✗不显著'}")

# %% [markdown]
# ## 11. 端到端合成(五):发表偏倚(Egger)
#
# **解决什么问题**:如果「阳性/高患病率的小研究更易发表」,合并估计会被系统性拉偏。**Egger 回归
# 检验**把标准化效应对精度回归,检验截距是否偏离 0——偏离即提示小研究效应/漏斗不对称。
#
# **哪几步**:`sv.tl.egger_test` 读效应量表,给截距 + p 值。这个结果稍后会**喂给 GRADE**——高不对称
# 会建议「发表偏倚」降级。

# %%
sv.tl.egger_test(study)
eg = study.diagnostics["egger"]
print(f"Egger 截距 = {eg['intercept']:.3f}  (p = {eg['pval']:.3f})")
print(f"漏斗不对称:{eg['asymmetry']}  →  {'提示可能存在小研究效应/发表偏倚' if eg['asymmetry'] else '无强证据表明发表偏倚'}")

# %% [markdown]
# ## 12. 端到端合成(六):森林图 + 回变换成患病率
#
# **解决什么问题**:合成结果要**画出来**、并**回到可读尺度**。森林图把每个效应量、合并菱形放进
# 一张图;但纵轴现在是 logit,读者看不懂——必须把合并 logit + CI **回变换成 0–1 患病率**。
#
# **哪几步**:`sv.pl.meta_forest` 画森林图(读 `meta_effects` + `meta`);`sv.tl.backtransform_proportion`
# 按 `measure='PLO'` 把合并 logit 逆变换成百分数患病率。

# %%
sv.pl.meta_forest(study, out="fig28_forest.png",
                  title="抑郁症状患病率 · 三层随机效应 meta(logit 尺度)")
Image("fig28_forest.png")

# %%
sv.tl.backtransform_proportion(study, measure="PLO")
pp = study.diagnostics["pooled_proportion"]
print(f"合并患病率 = {pp['proportion']*100:.1f}%  95% CI [{pp['ci_lb']*100:.1f}%, {pp['ci_ub']*100:.1f}%]")
print(f"(由 logit {mv['estimate']:.3f} 回变换;这是可直接写进摘要的数)")

# %% [markdown]
# ## 13. 端到端合成(收束):GRADE 证据确定性
#
# **解决什么问题**:所有证据摆齐后,读者最想要一句话——**这条结论有多可信?** GRADE 把它规范成
# 四档:High / Moderate / Low / Very low。观察性研究(患病率综述属此类)**起点为 Low**,按五个域
# **降级**、按大效应等**升级**。
#
# **关键前提**:GRADE 是**代数 + 建议**,不替评审拍板。`grade` 从证据槽自动**建议**两个域:
# 不一致性(由前面的 I²)、发表偏倚(由 Egger p 值);其余降级(偏倚风险、间接性、不精确)由
# **评审录入**。这正好把前面每一步的产出收束成一个判断。
#
# **哪几步**:`design='observational'`,评审录入 `risk_of_bias`/`imprecision` 等旗标,函数读
# `diagnostics['heterogeneity']` 和 `diagnostics['egger']` 给出建议,算出最终确定性。

# %%
sv.gov.grade(
    study,
    design="observational",          # 患病率综述 → 起点 Low
    risk_of_bias=1,                  # 评审判断:部分研究偏倚风险高 → 降 1
    indirectness=0,
    imprecision=0,
    # inconsistency / publication_bias 留空 → 由 I²、Egger 自动建议
)
gr = study.governance["grade"]
print(f"起点({gr['start']}) → 最终证据确定性:{gr['certainty']}(level {gr['level']}/4)")
print(f"降级:{gr['downgrades']}")
print(f"升级:{gr['upgrades']}")
print(f"自动建议 · 不一致性(来自 I²)= {gr['suggested_inconsistency']}"
      f"   发表偏倚(来自 Egger)= {gr['suggested_publication_bias']}")
print(f"说明:{gr['note']}")

# %% [markdown]
# ## 小结:一条从记录计数到 GRADE 的可复核证据链
#
# 这本 notebook 把「治理」补进了 meta 分析,并走通一条**端到端**的患病率系统综述——每一步都落在
# 同一个 `StudyState` 上,每个数字都能追到是哪个 `sv.*` 函数、从哪份数据算出来的:
#
# ```
# 治理层(报显著之前必须交代):
#   sv.gov.prisma_flow ─────▶ 记录计数 + 算术自洽(1284→28,consistent)
#   sv.pl.prisma_diagram ───▶ PRISMA 流程图
#   sv.gov.prisma_checklist ▶ 27 项报告完整度
#   sv.gov.screen_agreement ▶ κ 低但 AC1/PABAK 高(识破流行率悖论)
#   sv.gov.risk_of_bias / sv.pl.rob_traffic_light ▶ 逐研究×domain 偏倚交通灯
#
# 合成层(ECR 式三层患病率 meta):
#   sv.pp.es_proportion(PLO) ──▶ logit 患病率效应量(yi, vi)
#   sv.tl.vcalc(rho=0.6) ──────▶ 块对角抽样协方差 V
#   sv.tl.rma_mv ─────────────▶ 三层合并 + σ²₂/σ²₃ 分量
#   sv.tl.meta_heterogeneity / ma_i2_multilevel ▶ 总体 I² + 分层 I²
#   sv.tl.metareg / metareg_fdr ▶ 调节变量解释异质 + FDR 校正
#   sv.tl.egger_test ─────────▶ 发表偏倚(喂给 GRADE)
#   sv.pl.meta_forest ────────▶ 森林图
#   sv.tl.backtransform_proportion ▶ 回变换成可读患病率
#   sv.gov.grade ─────────────▶ 证据确定性(I²/Egger 自动建议 + 评审录入)
# ```
#
# **要点**:一篇系统综述的可信度不止于合并那个数——它取决于**检索透明(PRISMA)、筛选可靠
# (κ/AC1)、研究可信(RoB)、异质诚实(分层 I²)、偏倚可查(Egger)、强度分级(GRADE)**这条完整
# 链条。治理层做记账与算术、把建议留给评审确认,让整条链既留痕又不越权——这就是**可复核**。至此
# meta 分析系列(22–28)闭环:从最基础的两层合并,一路到带治理与三层结构的端到端复现。
