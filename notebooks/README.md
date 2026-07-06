# socialverse 教程 · 端到端 Notebook

10 本可端到端运行、**带真实输出**的教学 notebook,每本 = **一条真实分析链**(不是函数罗列),从「拿到数据」走到「可复核结论 + 证据链」。全部用内置玩具数据([`socialverse.datasets`](../socialverse/datasets/)),用 omics env(numpy/pandas/statsmodels/matplotlib/networkx)执行过、图表齐全。

> **贯穿主线**:每本都展示 socialverse 独有的三件事 —— ① 用 `sv.registry.resolve_plan` **查注册表排链**(而非猜 API);② 契约在调用时**强制校验 requires**(缺前置直接拦截);③ 结尾 `st.summary()` 打印 **provenance 证据链**。这是与「暴露方法」的 rmcp、「造模拟人」的 AgentSociety 的根本切割:**演示的是「注册表如何决定下一步的合法性」**。

## 怎么运行

```bash
pip install -e ".[full]"          # 装 statsmodels/networkx/matplotlib
jupyter lab notebooks/            # 逐格运行;或直接看已执行的 .ipynb 输出
```
每本都配套一个 jupytext `.py`(percent 格式,干净可 diff)与一个已执行的 `.ipynb`(含输出与图)。

---

## Part 1 — 核心机制(地基)

| # | Notebook | 讲什么 | 演示函数 |
|---|---|---|---|
| **01** | [注册表脊柱与 StudyState](01_registry_and_studystate.ipynb) | `find` / `get_prerequisites` / `resolve_plan` / 调用拦截(grounding)/ 12 槽位 / `sv.utils.registry_lookup` | 核心 registry API |

## Part 2 — 定量分析链

| # | Notebook | 分析链 | 对标现实工具 |
|---|---|---|---|
| **02** | [因果计量:面板 FE + 双重差分](02_causal_did.ipynb) | ingest → declare_design → **平行趋势检验** → DiD → event-study → 森林图 | pyfixest / R `fixest`·`did` |
| **03** | [复杂抽样:设计加权推断](03_complex_survey.ipynb) | 声明 svydesign(权重/分层/PSU)→ 信度 → 加权回归 | samplics / R `survey` |
| **04** | [实证复现:AERS 8 步 + 可复现脚本](04_econometrics_replication.ipynb) | 平衡表 → 基线 → 稳健性矩阵 → emit R 脚本 → docx | R `fixest` + Quarto/targets |
| **07** | [理论透镜与网络](07_theory_lens_network.ipynb) | 福柯话语 / 布迪厄场域(MCA)/ 韦伯理想类型 / 社会网络中心性·社群 | networkx·igraph / R `ergm`·`FactoMineR` |

## Part 3 — 质性与人文链

| # | Notebook | 分析链 | 对标现实工具 |
|---|---|---|---|
| **05** | [质性:去标识 → 反身主题编码 → 引语溯源](05_qualitative_coding.ipynb) | build_corpus → redact_pii → 六阶段编码 → quote-trace → 主题地图 | CAQDAS(NVivo/QualCoder)+ Braun&Clarke |
| **06** | [数字人文:OCR→TEI + 校勘与谱系](06_text_philology.ipynb) | ocr_tei → 多见证本对勘 → apparatus → stemma → TEI-P5 | Tesseract/Kraken + CollateX + TEI |

## Part 4 — 研究闭环与治理

| # | Notebook | 分析链 | 对标现实工具 |
|---|---|---|---|
| **08** | [研究治理:伦理 + 数据合规 + AI 披露](08_governance_gates.ipynb) | k-匿名伦理闸门 / 五桶许可分诊 / ICMJE 披露声明 | sdcMicro / 期刊政策(socialverse 特有一等公民轴) |
| **09** | [文献引证:检索 → 三库核验(揪幻觉引用) → 稿件审计](09_literature_citation.ipynb) | search → 核验(标 chimeric/suspicious)→ 风格化 → claim-evidence 审计 | Zotero + CrossRef/OpenAlex |
| **10** | [研究闭环:可复核小研究 + 证据链导出](10_full_study_evidence_chain.ipynb) | 治理闸门 → 注册表排链 → 因果 → provenance 账本 → registry manifest 导出 | (socialverse 的差异化收束) |

## Part 5 — 补齐的定量方法(曾经的缺口,现全部真实实现)

每本用真 DGP 玩具数据端到端跑通并**复原已知参数**;近似处诚实标注(ERGM=MPLE、SAOM=描述性)。

| # | Notebook | 方法 | 复原 | 对标现实工具 |
|---|---|---|---|---|
| **11** | [准实验:RDD + 合成控制](11_quasi_experiment.ipynb) | 断点回归(局部线性)· 合成控制(SLSQP 权重) | RDD 跳 ≈3.0 | rdrobust / gsynth·augsynth |
| **12** | [心理测量:CFA → SEM → IRT](12_psychometrics.ipynb) | 验证性因子 · 结构方程 · 2PL 项目反应 | CFI 0.95/RMSEA 0.04 | lavaan / mirt / semopy |
| **13** | [多层与生存:HLM + Cox](13_multilevel_survival.ipynb) | 混合效应(随机截距/斜率)· Cox PH + KM | 斜率≈2.0,log-HR≈0.8 | lme4·brms / survival |
| **14** | [空间分析:Moran/LISA + SAR](14_spatial_analysis.ipynb) | 全局/局部空间自相关 · 空间滞后回归 + 效应分解 | ρ≈0.5 | PySAL / spdep·spatialreg |
| **15** | [fsQCA + 人口学](15_qca_demography.ipynb) | 模糊集真值表最小化 · 生命表 · Kitagawa 分解 | 解="C+A*B" 精确 | QCA·SetMethods / demography |
| **16** | [网络推断 + 文体计量](16_networks_stylometry.ipynb) | ERGM(MPLE)· SAOM 共演化 · Burrows's Delta | mutual 系数>0,按作者聚类 | ergm·statnet / RSiena / stylo |

---

## 覆盖与延伸

- 这 **16 本** notebook 覆盖 registry **全部 54 个函数**(见 [../docs/CONTRACT_CARDS.md](../docs/CONTRACT_CARDS.md) 逐一契约卡)。
- 曾经的缺口(SEM/CFA · IRT · RDD · 合成控制 · 多层 HLM · 事件史 · 空间 · ERGM/SAOM · QCA · 人口学 · 文体计量)**现已全部真实实现**(Part 5);诚实标注的近似:ERGM=MPLE(非 MCMC-MLE)、SAOM=描述性简化、SEM latent 不可用时退化 path analysis。
- 设计依据:社科生态「点状繁荣、彼此不互通」,socialverse 用**显式注册表契约层**补 R 的「隐式契约(formula+S3/S4+broom)」在 Python 侧的缺失。详见 [LANDSCAPE.md](../docs/LANDSCAPE.md)。
